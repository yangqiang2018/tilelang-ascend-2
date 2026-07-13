// clang-format off
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
// clang-format on

#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

// AscendC SoftmaxFlashV2 high-level API (lib/activation). Included explicitly so
// the softmax_flash_v2 template below resolves regardless of whether the catlass
// umbrella already pulls it in. The header is include-guarded and its
// begin_pipe(V)/end_pipe pair is balanced, so it is inert for any kernel that
// does not call softmax_flash_v2 (no effect on existing operators).
#include "lib/activation/softmaxflashv2.h"

#if defined(__has_include)
#if __has_include("version/cann_version.h")
#include "version/cann_version.h"
#endif
#endif

#include "shmem.h"

#define CUDART_INF_F 1.0f / 0.0f

typedef AscendC::int4b_t int4b_t;

namespace tl::ascend {
using namespace Catlass;
using namespace tla;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Gemm::Block;
using namespace AscendC;

using ArchTag = Arch::AtlasA2;
using LayoutGM = layout::RowMajor;

using LayoutL0A = layout::zZ;
using LayoutL0B = layout::nZ;
using LayoutL1 = layout::zN;
using LayoutL1T = layout::nZ;

constexpr int64_t UB_HALF_SIZE = 64;

template <typename T>
constexpr bool IsDuplicateSupported_v =
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, float>;

CATLASS_DEVICE void disable_dma_atomic_compat() {
#if defined(CANN_MAJOR) && CANN_MAJOR >= 9
  AscendC::DisableDmaAtomic();
#else
  AscendC::SetAtomicNone();
#endif
}

template <typename T, uint32_t dstM, uint32_t dstN>
CATLASS_DEVICE void copy_gm_to_l1(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor,
                                  uint32_t realSrcN = 1, uint32_t realTailM = 0,
                                  uint32_t realTailN = 0) {
  uint32_t tailM = realTailM == 0 ? dstM : realTailM;
  uint32_t tailN = realTailN == 0 ? dstN : realTailN;
  if (tailM != dstM || tailN != dstN) {
    AscendC::InitConstValue(
        dstTensor,
        {1, static_cast<uint16_t>(dstM * dstN * sizeof(T) / 32), 0, 0});
    AscendC::PipeBarrier<PIPE_MTE2>();
  }
  auto layout = MakeLayoutFromTag(LayoutGM{tailM, realSrcN});
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(tailM, tailN));
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                             AscendC::TPosition::GM>(srcTensor, src_LAYOUT);

  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto layoutInL1 = tla::MakeLayout<T, LayoutL1_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutInL1),
                             AscendC::TPosition::A1>(dstTensor, layoutInL1);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

// Paged-attention KV load: gather `copyRowNum` rows of a paged KV cache from GM
// straight into L1 (NZ), resolving each page through `blockTableGm`. This is a
// faithful port of the reference op's DataCopyPA (PA_ND branch,
// sparse_attn_sharedkv_common.h): it walks the window one *page* at a time --
// look up the page id in the block table, Nd2Nz-copy the contiguous run of rows
// that fall inside that page, advance, repeat -- so a <=128-row window that sits
// in one page is a single DataCopy (vs a per-row gather). Runs on the cube,
// L1-direct, so it needs no UB staging and no GM "workspace_kv" round-trip.
// PA_ND layout only (headNum/n2Idx/dIdx kept for parity with the reference
// signature; SWA passes n2Idx=dIdx=0, headNum=1).
template <typename T>
CATLASS_DEVICE void
copy_pa(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor,
        GlobalTensor<int32_t> blockTableGm, uint32_t blockSize,
        uint32_t headNum, uint32_t headDim, uint32_t kvStride,
        uint32_t maxblockNumPerBatch, uint32_t actHeadDim, uint32_t copyRowNum,
        uint32_t copyRowNumAlign, uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx,
        uint32_t dIdx) {
  (void)headNum; // PA_ND offset uses n2Idx/headDim/blockSize; kept for parity
  uint32_t copyFinishRowCnt = 0;
  uint64_t blockTableBaseOffset = (uint64_t)bIdx * maxblockNumPerBatch;
  uint32_t curS2Idx = s2Idx;
  uint32_t blockElementCnt = 32 / sizeof(T);
  while (copyFinishRowCnt < copyRowNum) {
    uint64_t blockIdOffset = curS2Idx / blockSize;
    uint64_t reaminRowCnt = curS2Idx % blockSize;
    uint64_t idInBlockTable =
        blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
    uint32_t copyRowCnt = blockSize - reaminRowCnt; // one block at a time
    if (copyFinishRowCnt + copyRowCnt > copyRowNum) {
      copyRowCnt = copyRowNum - copyFinishRowCnt;
    }
    uint64_t offset = (uint64_t)idInBlockTable * kvStride +
                      (uint64_t)n2Idx * headDim * blockSize +
                      reaminRowCnt * headDim + dIdx;
    AscendC::Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = copyRowCnt;
    nd2nzPara.dValue = actHeadDim;
    nd2nzPara.srcDValue = headDim;
    nd2nzPara.dstNzC0Stride = copyRowNumAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    AscendC::DataCopy(dstTensor[copyFinishRowCnt * blockElementCnt],
                      srcTensor[offset], nd2nzPara);
    copyFinishRowCnt += copyRowCnt;
    curS2Idx += copyRowCnt;
  }
}

template <typename T, uint32_t srcM, uint32_t srcN, bool transpose = false>
CATLASS_DEVICE void copy_l1_to_l0a(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor, uint32_t dstM,
                                   uint32_t dstN) {
  using LayoutL1_ =
      std::conditional_t<transpose,
                         Catlass::detail::TagToLayout_t<T, LayoutL1T>,
                         Catlass::detail::TagToLayout_t<T, LayoutL1>>;
  constexpr auto layout = transpose ? tla::MakeLayout<T, LayoutL1_>(srcN, srcM)
                                    : tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(dstM, dstN));
  auto src = MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                        AscendC::TPosition::A1>(srcTensor, src_LAYOUT);

  using LayoutL0A_ = Catlass::detail::TagToLayout_t<T, LayoutL0A>;
  auto layoutAInL0 = tla::MakeLayout<T, LayoutL0A_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutAInL0),
                             AscendC::TPosition::A2>(dstTensor, layoutAInL0);
  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T, uint32_t srcM, uint32_t srcN, bool transpose = false>
CATLASS_DEVICE void copy_l1_to_l0b(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor, uint32_t dstM,
                                   uint32_t dstN) {
  using LayoutL1_ =
      std::conditional_t<transpose,
                         Catlass::detail::TagToLayout_t<T, LayoutL1T>,
                         Catlass::detail::TagToLayout_t<T, LayoutL1>>;
  constexpr auto layout = transpose ? tla::MakeLayout<T, LayoutL1_>(srcN, srcM)
                                    : tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(dstM, dstN));
  auto src = MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                        AscendC::TPosition::A1>(srcTensor, src_LAYOUT);

  using LayoutL0B_ = Catlass::detail::TagToLayout_t<T, LayoutL0B>;
  auto layoutBInL0 = tla::MakeLayout<T, LayoutL0B_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutBInL0),
                             AscendC::TPosition::B2>(dstTensor, layoutBInL0);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T1, typename T2, uint32_t M, uint32_t N>
CATLASS_DEVICE void mma(LocalTensor<T1> const A, LocalTensor<T1> const B,
                        LocalTensor<T2> const C, bool init, uint32_t K,
                        uint32_t n_actual = N, uint8_t unitFlag = 0) {
  // n_actual: runtime number of output columns to compute (<= N). Defaults to
  // the compile-time N, so existing callers are unchanged. Used for variable-N
  // (e.g. QK over the actual window length), mirroring how K is a runtime arg.
  MmadParams mmadParams;
  mmadParams.m = M;
  mmadParams.n = n_actual;
  mmadParams.k = K;
  mmadParams.cmatrixInitVal = init;
  // cmatrixSource = false: faithful to the reference (block_cube.h:576 sets it
  // explicitly on EVERY Mmad). MmadParams does not default-initialise this field,
  // and the hardware reads it whenever cmatrixInitVal==false (an accumulate mma,
  // C sourced from L0C). The single-mma callers never hit it -- gemm_v0 (unitFlag
  // off) is insensitive, and PV's gemm_v0_fixp uses cmatrixInitVal==true (single K
  // tile, no accumulate) -- but QK's multi-K accumulate (cmatrixInitVal==false on
  // tiles 1..3) + unitFlag read the uninitialised field and hung the cube. Setting
  // it false is the faithful fix and byte-compatible (every caller's accumulate
  // semantics were already 'source from L0C').
  mmadParams.cmatrixSource = false;
  // unitFlag drives the hardware mma->fixpipe pipeline (0b10 accumulate / 0b11
  // flush), faithful to the Ascend C reference's cL0TensorPingPong overlap
  // (block_cube.h ComputeMm2:910). Defaults to 0 (off) so every pre-existing
  // caller is byte-for-byte unchanged; only gemm_v0_fixp opts into 0b10/0b11.
  mmadParams.unitFlag = unitFlag;

  Mmad(C, A, B, mmadParams);

  constexpr uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
  // if constexpr ((M / C0_NUM_PER_FRACTAL) * (N / C0_NUM_PER_FRACTAL) <
  //               PIPE_M_BARRIER_THRESHOLD) {
  //   PipeBarrier<PIPE_M>();
  // }
}

template <typename T1, typename T2, typename LayoutGM, uint32_t srcM,
          uint32_t srcN, bool enRelu = false>
CATLASS_DEVICE void
copy_l0c_to_gm(GlobalTensor<T2> dstTensor, LocalTensor<T1> srcTensor,
               uint32_t realDstN = 1, uint32_t realTailM = 0,
               uint32_t realTailN = 0, uint8_t unitFlag = 0) {
  uint32_t tailM = realTailM == 0 ? srcM : realTailM;
  uint32_t tailN = realTailN == 0 ? srcN : realTailN;
  auto layoutInL0C = tla::MakeLayoutL0C(srcM, srcN);
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(layoutInL0C),
                             AscendC::TPosition::CO1>(srcTensor, layoutInL0C);
  LayoutGM gm{tailM, realDstN};
  auto layout = MakeLayoutFromTag(gm);
  auto dTensor = MakeTensor(dstTensor, layout, Arch::PositionGM{});
  auto layout_ = dTensor.layout();
  auto dst_LAYOUT = MakeLayoutTile(layout_, tla::MakeShape(tailM, tailN));
  auto dst = MakeTensor<decltype(dstTensor), decltype(dst_LAYOUT),
                        AscendC::TPosition::GM>(dstTensor, dst_LAYOUT);

  CopyL0CToGmTla<ArchTag, decltype(src), decltype(dst),
                 ScaleGranularity::NO_QUANT, enRelu>
      tileCopier;
  // unitFlag (default 0) pairs with the Mmad unitFlag for the cL0 ping-pong
  // fixpipe||mma overlap; CopyL0CToGmTla already plumbs it into FixpipeParams.
  tileCopier(dst, src, unitFlag);
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t block_M,
          uint32_t block_N, uint32_t SwizzleOffset = 1,
          uint32_t SwizzleDirection = 0>
CATLASS_DEVICE auto thread_block_swizzle(uint64_t pid) {
  GemmCoord problem_shape = GemmCoord(M, N, K);
  MatrixCoord tile_shape = MatrixCoord(block_M, block_N);

  GemmIdentityBlockSwizzle swizzle =
      GemmIdentityBlockSwizzle<SwizzleOffset, SwizzleDirection>(problem_shape,
                                                                tile_shape);

  auto cols = swizzle.loopsMN.column();

  auto coord = swizzle.GetBlockCoord(pid);

  // return coord;
  return coord.m() * cols + coord.n();
}

template <typename T, uint32_t dstN, uint32_t dstM = 1>
CATLASS_DEVICE void
copy_gm_to_ub(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor,
              uint32_t realSrcN = 1, uint32_t maskShapeM = dstM,
              uint32_t maskShapeN = dstN, T padValue = T(0)) {

  bool isPad = true;
  uint32_t rightPadding = 1;
  if (maskShapeN == dstN || (maskShapeN * sizeof(T)) % 32 == 0) {
    isPad = false;
    rightPadding = 0;
  }
  if (maskShapeM != dstM || maskShapeN != dstN) {
    if constexpr (IsDuplicateSupported_v<T>) {
      SetFlag<HardEvent::MTE2_V>(0);
      WaitFlag<HardEvent::MTE2_V>(0);
      SetFlag<HardEvent::MTE3_V>(0);
      WaitFlag<HardEvent::MTE3_V>(0);
      AscendC::Duplicate<T>(dstTensor, padValue, dstM * dstN);
      SetFlag<HardEvent::V_MTE2>(0);
      WaitFlag<HardEvent::V_MTE2>(0);
    }
  }
  AscendC::DataCopyExtParams dataCopyParams(
      maskShapeM, maskShapeN * sizeof(T), (realSrcN - maskShapeN) * sizeof(T),
      (dstN - maskShapeN) * sizeof(T) / 32, 0);
  AscendC::DataCopyPadExtParams<T> padParams(isPad, 0, rightPadding, padValue);
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
}

template <typename T, uint32_t srcN, uint32_t srcM = 1>
CATLASS_DEVICE void
copy_ub_to_gm(GlobalTensor<T> dstTensor, LocalTensor<T> srcTensor,
              uint32_t realdstN = 1, uint32_t maskShapeM = srcM,
              uint32_t maskShapeN = srcN) {
  AscendC::DataCopyExtParams dataCopyParams(
      maskShapeM, maskShapeN * sizeof(T), (srcN - maskShapeN) * sizeof(T) / 32,
      (realdstN - maskShapeN) * sizeof(T), 0);
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
}

// Faithful Ascend C CopyInKv (scfa V0 sparse-block gather): one DataCopyPad that
// gathers `blockCount` (1 or 2) GM blocks straight into a packed UB merge buffer.
// Unlike copy_gm_to_ub (which derives blockCount from a slice's row extent and
// srcStride from the source buffer's innermost shape), every DataCopyExtParams
// field is a RUNTIME arg supplied by the kernel: blockLenBytes / srcStrideBytes
// are the per-block byte length and the byte gap between the two topk-selected
// (non-adjacent) GM blocks; dstStride==0 packs them in UB. No Duplicate pad --
// byte-identical to the reference's `DataCopyPad(dst, src, intriParams, padParams)`
// with a default-constructed padParams. Emitted by tl.ascend_copy_gather.
template <typename T>
CATLASS_DEVICE void
copy_gm_to_ub_gather(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor,
                     uint16_t blockCount, uint32_t blockLenBytes,
                     uint32_t srcStrideBytes, uint32_t dstStride = 0) {
  AscendC::DataCopyExtParams dataCopyParams(blockCount, blockLenBytes,
                                            srcStrideBytes, dstStride, 0);
  AscendC::DataCopyPadExtParams<T> padParams;
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
}

template <typename T, uint32_t srcN, uint32_t srcM = 1>
CATLASS_DEVICE void
atomic_add_ub_to_gm(GlobalTensor<T> dstTensor, LocalTensor<T> srcTensor,
                    uint32_t realdstN = 1, uint32_t maskShapeM = srcM,
                    uint32_t maskShapeN = srcN) {
  AscendC::SetAtomicAdd<T>();
  copy_ub_to_gm<T, srcN, srcM>(dstTensor, srcTensor, realdstN, maskShapeM,
                               maskShapeN);
  disable_dma_atomic_compat();
}

template <typename T1, typename T2, typename LayoutGM, uint32_t srcM,
          uint32_t srcN, bool enRelu = false>
CATLASS_DEVICE void
atomic_add_l0c_to_gm(GlobalTensor<T2> dstTensor, LocalTensor<T1> srcTensor,
                     uint32_t realDstN = 1, uint32_t realTailM = 0,
                     uint32_t realTailN = 0) {
  AscendC::SetAtomicAdd<T2>();
  copy_l0c_to_gm<T1, T2, LayoutGM, srcM, srcN, enRelu>(
      dstTensor, srcTensor, realDstN, realTailM, realTailN);
  disable_dma_atomic_compat();
}

template <typename T1, typename T2, uint32_t len>
CATLASS_DEVICE void copy_ub_to_ub(LocalTensor<T1> dstTensor,
                                  LocalTensor<T2> srcTensor) {
  if constexpr (std::is_same_v<T1, T2>) {
    AscendC::DataCopy(dstTensor, srcTensor, len);
  } else {
    if constexpr ((std::is_same_v<T1, float> && std::is_same_v<T2, half>) ||
                  (std::is_same_v<T1, float> &&
                   std::is_same_v<T2, bfloat16_t>) ||
                  (std::is_same_v<T1, float> && std::is_same_v<T2, int16_t>) ||
                  (std::is_same_v<T1, half> && std::is_same_v<T2, int8_t>) ||
                  (std::is_same_v<T1, int16_t> &&
                   std::is_same_v<T2, int32_t>)) {
      AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_NONE, len);
    } else {
      AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_RINT, len);
    }
  }
}

template <typename T1, typename T2, uint32_t len>
CATLASS_DEVICE void
copy_ub_to_ub(LocalTensor<T1> dstTensor, LocalTensor<T2> srcTensor,
              uint32_t src_rows, uint32_t src_cols, uint32_t src_stride,
              uint32_t dst_rows, uint32_t dst_cols, uint32_t dst_stride) {
  if (src_cols == src_stride && dst_cols == dst_stride) {
    copy_ub_to_ub<T1, T2, len>(dstTensor, srcTensor);
  } else {
    for (uint32_t i = 0; i < src_rows; i++) {
      if constexpr (std::is_same_v<T1, T2>) {
        AscendC::DataCopy(dstTensor[i * dst_stride], srcTensor[i * src_stride],
                          src_cols);
      } else {
        if constexpr ((std::is_same_v<T1, float> && std::is_same_v<T2, half>) ||
                      (std::is_same_v<T1, float> &&
                       std::is_same_v<T2, bfloat16_t>) ||
                      (std::is_same_v<T1, float> &&
                       std::is_same_v<T2, int16_t>) ||
                      (std::is_same_v<T1, half> &&
                       std::is_same_v<T2, int8_t>) ||
                      (std::is_same_v<T1, int16_t> &&
                       std::is_same_v<T2, int32_t>)) {
          AscendC::Cast(dstTensor[i * dst_stride], srcTensor[i * src_stride],
                        AscendC::RoundMode::CAST_NONE, src_cols);
        } else {
          AscendC::Cast(dstTensor[i * dst_stride], srcTensor[i * src_stride],
                        AscendC::RoundMode::CAST_RINT, src_cols);
        }
      }
    }
  }
}

template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void copy_ub_to_l1(LocalTensor<T> dstTensor,
                                  LocalTensor<T> srcTensor) {
  static_assert(std::is_same_v<T, half>, "only support half");
  static_assert(M % 16 == 0, "M must be the multiple of 16");

  AscendC::DataCopyExtParams dataCopyParams(M, N * sizeof(T), 0, 0, 0);

  AscendC::Nd2NzParams nd2nzParams;
  nd2nzParams.ndNum = 1;
  nd2nzParams.nValue = M;
  nd2nzParams.dValue = N;
  nd2nzParams.srcNdMatrixStride = 0;
  nd2nzParams.srcDValue = N;
  nd2nzParams.dstNzC0Stride = M;
  nd2nzParams.dstNzNStride = 1;
  nd2nzParams.dstNzMatrixStride = 0;

  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, nd2nzParams);
}

template <typename T, uint32_t Len>
CATLASS_DEVICE void tile_add(LocalTensor<T> const &ubIn0,
                             LocalTensor<T> const &ubIn1,
                             LocalTensor<T> const &ubOut) {
  AscendC::Add(ubOut, ubIn0, ubIn1, Len);
}

template <typename T, uint32_t Len, uint32_t op>
CATLASS_DEVICE void elementwise_binary(LocalTensor<T> const &ubIn0,
                                       LocalTensor<T> const &ubIn1,
                                       LocalTensor<T> const &ubOut) {
  // AscendC::Elementwise(ubOut, ubIn0, ubIn1, op, Len);
  if constexpr (op == 0) {
    AscendC::Add(ubOut, ubIn0, ubIn1, Len);
  } else if constexpr (op == 1) {
    AscendC::Sub(ubOut, ubIn0, ubIn1, Len);
  } else if constexpr (op == 2) {
    AscendC::Mul(ubOut, ubIn0, ubIn1, Len);
  } else if constexpr (op == 3) {
    AscendC::Div(ubOut, ubIn0, ubIn1, Len);
  }
}

template <typename T>
CATLASS_DEVICE void shmem_put_nbi(const GlobalTensor<T> &output,
                                  const GlobalTensor<T> &input, size_t nelems,
                                  size_t newPe) {
  AscendC::TPipe pipe;
  uint32_t ub_size = UB_HALF_SIZE * 2 + 64;
  AscendC::TBuf<AscendC::TPosition::VECIN> ub_buf;
  pipe.InitBuffer(ub_buf, ub_size);
  auto ub_tensor = ub_buf.Get<T>();
  pipe.Destroy();
  __gm__ T *outputPtr = const_cast<__gm__ T *>(output.GetPhyAddr());
  __gm__ T *inputPtr = const_cast<__gm__ T *>(input.GetPhyAddr());
  __ubuf__ T *buf = reinterpret_cast<__ubuf__ T *>(ub_tensor.GetPhyAddr());
  aclshmemx_mte_put_nbi(outputPtr, inputPtr, buf, ub_size, nelems, newPe,
                        EVENT_ID0);
}

template <typename T>
CATLASS_DEVICE void shmem_ub_put_nbi(const LocalTensor<T> &ubTensor,
                                     const GlobalTensor<T> &output,
                                     size_t nelems, int newPe, int strelem) {
  aclshmemx_mte_put_nbi(const_cast<__gm__ T *>(output.GetPhyAddr() + strelem),
                        reinterpret_cast<__ubuf__ T *>(ubTensor.GetPhyAddr()),
                        nelems, newPe, EVENT_ID0);
}

template <typename T>
CATLASS_DEVICE void shmem_get_nbi(const GlobalTensor<T> &output,
                                  const GlobalTensor<T> &input, size_t nelems,
                                  size_t newPe) {
  AscendC::TPipe pipe;
  uint32_t ub_size = UB_HALF_SIZE * 2 + 64;
  AscendC::TBuf<AscendC::TPosition::VECIN> ub_buf;
  pipe.InitBuffer(ub_buf, ub_size);
  auto ub_tensor = ub_buf.Get<T>();
  pipe.Destroy();
  __gm__ T *outputPtr = const_cast<__gm__ T *>(output.GetPhyAddr());
  __gm__ T *inputPtr = const_cast<__gm__ T *>(input.GetPhyAddr());
  __ubuf__ T *buf = reinterpret_cast<__ubuf__ T *>(ub_tensor.GetPhyAddr());
  aclshmemx_mte_get_nbi(outputPtr, inputPtr, buf, ub_size, nelems, newPe,
                        EVENT_ID0);
}

template <typename T>
CATLASS_DEVICE void shmem_ub_get_nbi(const LocalTensor<T> &output,
                                     const GlobalTensor<T> &input,
                                     size_t nelems, size_t newPe) {
  aclshmemx_mte_get_nbi(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                        const_cast<__gm__ T *>(input.GetPhyAddr()), nelems,
                        newPe, EVENT_ID0);
}

template <typename T, uint32_t Len, uint32_t op>
CATLASS_DEVICE void elementwise_unary(LocalTensor<T> const &ubIn,
                                      LocalTensor<T> const &ubOut) {
  // AscendC::Elementwise(ubOut, ubIn0, ubIn1, op, Len);
  if constexpr (op == 0) {
    // TODO: Check layout, Len only has bug.
    AscendC::Exp(ubOut, ubIn, Len);
  }
}

template <typename dst, typename src, const char round_mode[], uint32_t Len>
CATLASS_DEVICE void cast(LocalTensor<dst> const &ubOut,
                         LocalTensor<src> const &ubIn) {
  AscendC::Cast(ubOut, ubIn, round_mode, Len);
}

// template <typename T, uint32_t Len>
// CATLASS_DEVICE void fill(LocalTensor<T> const &ubOut, T value) {
//   AscendC::Duplicate(ubOut, value, Len);
// }

template <typename T>
CATLASS_DEVICE void
reduce_sum_half(LocalTensor<T> const &dstTensor,
                LocalTensor<T> const &srcTensor, const int32_t mask,
                const int32_t repeatTime, const int32_t srcRepStride) {
  AscendC::WholeReduceSum<T>(dstTensor, srcTensor, mask, repeatTime, 1, 1,
                             srcRepStride);
}

template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void
reduce_sum(LocalTensor<T> const &dstTensor, LocalTensor<T> const &srcTensor,
           LocalTensor<uint8_t> const &sharedTmpBuffer, bool clear = true) {
  uint32_t shape[] = {M, N};
  if (clear) {
    if constexpr (dim == -1) {
      AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    } else {
      AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    }
    return;
  }

  constexpr uint32_t kReduceResultLen = dim == -1 ? M : N;
  // ReduceSum appears to use scratch in a way that can interfere with a local
  // UB backup on real_shape/slice paths, so keep the old dst in scalar locals
  // before forcing clear=true and merging manually.
  T dstBackup[kReduceResultLen];
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    dstBackup[i] = dstTensor.GetValue(i);
  }

  if constexpr (dim == -1) {
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  } else {
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  }

  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    T reducedValue = dstTensor.GetValue(i);
    dstTensor.SetValue(i, static_cast<T>(reducedValue + dstBackup[i]));
  }
}

template <typename T>
CATLASS_DEVICE T reduce_scalar_max_safe(T lhsValue, T rhsValue) {
  // Bisheng/AICore does not allow scalar half/bfloat16 comparisons inside
  // device code, so the clear=false fallback compares through float.
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
    return static_cast<float>(lhsValue) > static_cast<float>(rhsValue)
               ? lhsValue
               : rhsValue;
  } else {
    return lhsValue > rhsValue ? lhsValue : rhsValue;
  }
}

template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void
reduce_max(LocalTensor<T> const &dstTensor, LocalTensor<T> const &srcTensor,
           LocalTensor<uint8_t> const &sharedTmpBuffer, bool clear = true) {
  uint32_t shape[] = {M, N};
  if (clear) {
    if constexpr (dim == -1) {
      AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    } else {
      AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    }
    return;
  }

  // AscendC::ReduceMax(..., clear=false) does not reliably preserve the
  // upstream "merge old dst with reduced value" contract on real_shape/slice
  // paths, so we make the merge explicit here.
  constexpr uint32_t kReduceResultLen = dim == -1 ? M : N;
  T dstBackup[kReduceResultLen];
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    dstBackup[i] = dstTensor.GetValue(i);
  }

  if constexpr (dim == -1) {
    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  } else {
    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  }

  // Keep the merge explicit instead of relying on an in-place vector max,
  // because aliasing dst with one input can produce unstable results here.
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    T reducedValue = dstTensor.GetValue(i);
    T backupValue = dstBackup[i];
    dstTensor.SetValue(i, reduce_scalar_max_safe(reducedValue, backupValue));
  }
}

template <typename T>
CATLASS_DEVICE T reduce_scalar_min_safe(T lhsValue, T rhsValue) {
  // Bisheng/AICore does not allow scalar half/bfloat16 comparisons inside
  // device code, so the clear=false fallback compares through float.
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
    return static_cast<float>(lhsValue) < static_cast<float>(rhsValue)
               ? lhsValue
               : rhsValue;
  } else {
    return lhsValue < rhsValue ? lhsValue : rhsValue;
  }
}

template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void
reduce_min(LocalTensor<T> const &dstTensor, LocalTensor<T> const &srcTensor,
           LocalTensor<uint8_t> const &sharedTmpBuffer, bool clear = true) {
  uint32_t shape[] = {M, N};
  if (clear) {
    if constexpr (dim == -1) {
      AscendC::ReduceMin<T, AscendC::Pattern::Reduce::AR>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    } else {
      AscendC::ReduceMin<T, AscendC::Pattern::Reduce::RA>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    }
    return;
  }

  // AscendC::ReduceMin(..., clear=false) does not reliably preserve the
  // upstream "merge old dst with reduced value" contract on real_shape/slice
  // paths, so we make the merge explicit here.
  constexpr uint32_t kReduceResultLen = dim == -1 ? M : N;
  T dstBackup[kReduceResultLen];
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    dstBackup[i] = dstTensor.GetValue(i);
  }

  if constexpr (dim == -1) {
    AscendC::ReduceMin<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  } else {
    AscendC::ReduceMin<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  }

  // Keep the merge explicit instead of relying on an in-place vector min,
  // because aliasing dst with one input can produce unstable results here.
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    T reducedValue = dstTensor.GetValue(i);
    T backupValue = dstBackup[i];
    dstTensor.SetValue(i, reduce_scalar_min_safe(reducedValue, backupValue));
  }
}

static constexpr uint32_t L0AB_EVENT = 0;
// Dedicated event ids for the L0AB M_MTE1/MTE1_M ping-pong when a caller hoists
// the prime/drain out of gemm_v0_fixp (prime_drain=false): the two flags are then
// held SET across the whole cube loop, so they must NOT share an id with the
// per-call MTE2_MTE1/MTE1_MTE2 self-pair fences (which stay on L0AB_EVENT=0) --
// otherwise the next call's SetFlag<MTE2_MTE1>(0) collides with the held
// M_MTE1(0) on the same physical flag register. Faithful to the reference, which
// puts M_MTE1 on its own EVENT_ID3/4 disjoint from the L1 flags. Ids {4,5} are
// free in the SWA kernel (KV flags use {2,3}, the self-pair fences use {0,1}).
// The default (prime_drain=true) keeps M_MTE1/MTE1_M on L0AB_EVENT so every
// existing caller is byte-for-byte unchanged.
static constexpr uint32_t L0AB_MM_EVENT = 4;

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool transpose_A = false, bool transpose_B = false>
CATLASS_DEVICE void
gemm_v0(LocalTensor<T1> const &A, LocalTensor<T1> const &B,
        LocalTensor<T2> const &C, // this must be located in l0c
        AscendC::TBuf<AscendC::TPosition::A2> &l0a_,
        AscendC::TBuf<AscendC::TPosition::B2> &l0b_, bool clear,
        uint32_t n_actual = N) {
  // n_actual: runtime output-column count (<= N), only honoured on the
  // transpose_B (single N-tile) path -- e.g. QK computing just the actual
  // window length instead of the padded BI. Defaults to N, so every existing
  // caller is byte-for-byte unchanged. The non-transpose N-tiling path ignores
  // it (each tile keeps its compile-time nTile). Mirrors the runtime K already
  // threaded through copy_l1_to_l0* / mma.
  auto l0a = l0a_.Get<T1>();
  auto l0b = l0b_.Get<T1>();
  constexpr uint32_t kL0Size = 128;
  uint32_t kL0split = (K + kL0Size - 1) / kL0Size;
  uint32_t kL0Tail = K - (kL0split - 1) * kL0Size;
  bool initflag = false;

  // ---- N tiling -----------------------------------------------------------
  // The B operand tile loaded into L0B is (kL0Size x nTile); L0B holds 64KB,
  // and with the kL0 ping-pong the per-slot budget is 32KB. So a single mma
  // over the whole N (l0b slot = N*kL0Size) overflows L0B once N is large
  // (e.g. the PV matmul's N = headDim = 512 -> 512*128*2 = 128KB). Tile N into
  // nTile columns just like the Ascend C reference (N_SPLIT_SIZE = 128): each
  // tile loads its own (kL0Size x nTile) B sub-block and writes its own column
  // band of the L0C accumulator. Only the non-transpose-B path is tiled; the
  // transpose-B callers (e.g. QK with N = block_I <= 128) already fit, so they
  // keep nTile == N (a single pass, byte-for-byte the original behaviour) and
  // need no L1 column-offset formula. Compatibility: any existing caller with
  // N <= nMaxByL0B (transpose or not) sees nL0split == 1 and identical codegen.
  //
  // Sub-tile offsets come straight from the catlass tla fractal layouts:
  //   L0C column n0  ->  n0 * roundUp16(M)   (tla::MakeLayoutL0C N1 stride)
  //   L1 zN B col n0 ->  n0 * roundUp16(K)   (tla::MakeLayout<zN>  C1 stride)
  // both of which are consistent with the original K-offset B[kL0Idx*16*kL0Size]
  // (zN K-row stride) already used below.
  constexpr uint32_t nMaxByL0B = (32u * 1024u) / (kL0Size * sizeof(T1));
  constexpr uint32_t nTile = (transpose_B || N <= nMaxByL0B) ? N : nMaxByL0B;
  static_assert(transpose_B || (N % nTile == 0),
                "gemm_v0 N-tiling requires N divisible by the N tile size");
  constexpr uint32_t nL0split = N / nTile;
  constexpr uint32_t mRound = ((M + 15u) / 16u) * 16u;
  constexpr uint32_t kRound = ((K + 15u) / 16u) * 16u;

  // ---- Pipelined main loop. Prime/drain the L0A/L0B ping-pong buffers ONCE
  // and let the ping-pong run continuously across the WHOLE (N-tile, K-tile)
  // sequence (flattened by tileIdx). Each tile's L1->L0 load goes into the
  // free buffer while the previous tile's mma runs, so N-tiles overlap with K
  // exactly like the Ascend C matmul pipeline -- a per-N-tile drain (the first
  // N-tiling version) instead serialised the tiles. For nL0split == 1 (every
  // pre-existing caller, and the QK transpose-B path) tileIdx == kL0Idx, so
  // this is byte-for-byte the original K ping-pong.
  SetFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);
  WaitFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);
  SetFlag<HardEvent::FIX_M>(L0AB_EVENT);
  WaitFlag<HardEvent::FIX_M>(L0AB_EVENT);

  SetFlag<HardEvent::M_MTE1>(L0AB_EVENT);
  SetFlag<HardEvent::M_MTE1>(L0AB_EVENT + 1);

  uint32_t tileIdx = 0;
  for (uint32_t nL0Idx = 0; nL0Idx < nL0split; nL0Idx++) {
    uint32_t bNOffset = transpose_B ? 0u : (nL0Idx * nTile * kRound);
    uint32_t cNOffset = nL0Idx * nTile * mRound;

    for (uint32_t kL0Idx = 0; kL0Idx < kL0split; kL0Idx++) {
      // clear THIS N-tile's C column band on its first K-tile (each band is an
      // independent accumulation over K).
      initflag = (clear && (kL0Idx == 0));
      uint32_t kSize = (kL0Idx == kL0split - 1) ? kL0Tail : kL0Size;
      uint32_t pp = (tileIdx & 1);

      uint32_t l0a_base = pp * (M * kL0Size);
      uint32_t l0b_base = pp * (nTile * kL0Size);

      WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT + pp);
      if constexpr (!transpose_A) {
        tl::ascend::copy_l1_to_l0a<T1, M, K>(l0a[l0a_base],
                                             A[kL0Idx * M * kL0Size], M, kSize);
      } else {
        tl::ascend::copy_l1_to_l0a<T1, K, M, true>(
            l0a[l0a_base], A[kL0Idx * 16 * kL0Size], M, kSize);
      }
      if constexpr (!transpose_B) {
        tl::ascend::copy_l1_to_l0b<T1, K, N>(
            l0b[l0b_base], B[bNOffset + kL0Idx * 16 * kL0Size], kSize, nTile);
      } else {
        // transpose_B (QK): load only the n_actual real columns (window rows of
        // K^T); the [n_actual:N] columns stay unloaded (masked downstream).
        tl::ascend::copy_l1_to_l0b<T1, N, K, true>(
            l0b[l0b_base], B[kL0Idx * N * kL0Size], kSize, n_actual);
      }
      SetFlag<HardEvent::MTE1_M>(L0AB_EVENT + pp);
      WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT + pp);
      PipeBarrier<PIPE_M>();
      // transpose_B computes n_actual columns; the N-tiling path keeps nTile.
      tl::ascend::mma<T1, T2, M, nTile>(l0a[l0a_base], l0b[l0b_base],
                                        C[cNOffset], initflag, kSize,
                                        transpose_B ? n_actual : nTile);
      SetFlag<HardEvent::M_MTE1>(L0AB_EVENT + pp);
      tileIdx++;
    }
  }
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT);
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT + 1);

  SetFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);
  WaitFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);
  SetFlag<HardEvent::M_FIX>(L0AB_EVENT);
  WaitFlag<HardEvent::M_FIX>(L0AB_EVENT);
}

// gemm_v0 with the per-N-tile fixpipe fused in (faithful to the Ascend C
// reference's ComputeMm2: each [M, nTile] tile is Fixpipe'd to GM as soon as
// its K accumulation finishes, so L0C only holds the live ping-pong tiles).
// Unlike gemm_v0 -- which keeps the whole [M, N] result resident in L0C until
// the caller copies it out (N=512 PV => 128KB = the entire L0C) -- this writes
// each N-tile straight to the GM destination. C is a 2-slot [2, M, nTile] L0C
// ping-pong (e.g. [2,64,128]=64KB): consecutive N-tiles alternate slots so the
// fixpipe(tile i) overlaps the mma(tile i+1), carried by the hardware unitFlag
// (Mmad 0b11 + Fixpipe 0b11) -- faithful to the reference cL0TensorPingPong,
// NOT a software M_FIX/FIX_M handshake. Same N/K tiling and L0A/L0B ping-pong
// as gemm_v0. (No PV K-splitting needed here: K=block_I<=128 fits one kL0 tile.)
template <typename T1, typename T2, typename LayoutGM, uint32_t M, uint32_t N,
          uint32_t K, bool transpose_A = false, bool transpose_B = false>
CATLASS_DEVICE void
gemm_v0_fixp(LocalTensor<T1> const &A, LocalTensor<T1> const &B,
             LocalTensor<T2> const &C, // 2-slot [2, M, nTile] L0C ping-pong
             GlobalTensor<T2> dst,     // GM destination [M, N], row major
             AscendC::TBuf<AscendC::TPosition::A2> &l0a_,
             AscendC::TBuf<AscendC::TPosition::B2> &l0b_, bool clear,
             uint32_t k_actual, uint32_t n_actual = N, uint32_t cl0_base = 0,
             bool prime_drain = true, bool flush_last = true,
             bool do_fixpipe = true) {
  auto l0a = l0a_.Get<T1>();
  auto l0b = l0b_.Get<T1>();
  constexpr uint32_t kL0Size = 128;
  // k_actual: the runtime contraction length (<= K), split across ceil(K/128)
  // kL0 tiles that K-accumulate into ONE cL0 slot. PV passes k_actual=window
  // (<=128, single tile, the original 008 behaviour); QK passes k_actual=K=512
  // (head dim, 4 tiles) so QK joins this same fused-fixpipe path (= ComputeMm1).
  // Contracting only k_actual keeps the PV from summing the unwritten pad rows
  // of a paged KV tile (0 (masked P) * NaN (garbage V) -> NaN).
  //
  // n_actual: runtime output-column count (<= N), honoured on the transpose_B
  // (single N-tile) path = QK's window width; defaults to N so the non-transpose
  // PV path is byte-for-byte unchanged.
  //
  // cl0_base: the starting cL0 ping-pong slot for this call, so QK and PV share
  // ONE persistent cL0TensorPingPong rotation (= the reference advancing one
  // cL0BufIter across ComputeMm1 then ComputeMm2). Default 0 = standalone.
  //
  // prime_drain: when true (default) this call self-primes and self-drains its
  // two M_MTE1 L0AB ping-pong flags (the original self-contained behaviour --
  // every existing caller is byte-for-byte unchanged). When false the caller is
  // responsible for priming the flags ONCE before the cube loop and draining
  // them ONCE after (faithful to the reference's AllocEventID/FreeEventID:
  // block_cube.h:225-226/239-240 SetFlag/WaitFlag<M_MTE1>(L0AB_EVENT0/1)), so
  // back-to-back QK/PV calls no longer re-prime+drain the L0AB ring at every
  // call boundary -- ComputeMm1/Mm2 never touch the L0AB flag lifecycle, only
  // the per-slot Wait/Set inside the tile loop (block_cube.h:563/583). The
  // per-slot Wait/Set still protect L0A/L0B buffer reuse; the SWA cadence makes
  // each call start at abL0 parity 0 (every call advances abL0BufIter by an even
  // 4), so the local tileIdx reset is equivalent to a persistent abL0BufIter.
  //
  // flush_last / do_fixpipe: per-K-chunk accumulation. The reference loads the K
  // (or D) dimension as several GM->L1 chunks (ComputeMm1 splits headDim into 2x
  // 256 kL1 halves, block_cube.h:341-450), each into its own L1 ring slot, and
  // the cube accumulates them into ONE cL0 slot before a single Fixpipe. To drive
  // that from the kernel, call gemm_v0_fixp once per chunk into the SAME cl0_base
  // slot: chunk 0 with clear=true, flush_last=false, do_fixpipe=false (its last
  // kL0 stays 0b10, no flush, no copy-out); the final chunk with clear=false,
  // flush_last=true, do_fixpipe=true (its last kL0 is 0b11 and the Fixpipe flushes
  // the fully-accumulated cL0). Both default true = the old single-call behaviour
  // (every existing caller byte-for-byte unchanged).
  //
  // kL0split is over the RUNTIME k_actual (this chunk's contraction length), not
  // the compile-time K: a 256-wide D-chunk runs 2 kL0 tiles, the whole 512 runs 4,
  // PV's window<=128 runs 1. For the existing callers k_actual==K so kL0split is
  // unchanged (byte-compatible); a chunk passes k_actual=256 (= its own K=256).
  uint32_t kL0split = (k_actual + kL0Size - 1) / kL0Size;
  bool initflag = false;
  // L0AB M_MTE1/MTE1_M ping-pong event base. prime_drain=true keeps it on
  // L0AB_EVENT (byte-identical to all existing callers); prime_drain=false moves
  // it to the dedicated L0AB_MM_EVENT so the held-across-the-loop M_MTE1 flags do
  // not collide with the per-call MTE2_MTE1/MTE1_MTE2 self-pair fences (which stay
  // on L0AB_EVENT). The caller primes/drains M_MTE1(L0AB_MM_EVENT, +1) once.
  const uint32_t mmEv = prime_drain ? L0AB_EVENT : L0AB_MM_EVENT;

  // N tiling: identical to gemm_v0 (B tile (kL0Size x nTile) must fit the 32KB
  // L0B ping-pong slot). C is a single [M, nTile] slot reused per N-tile.
  constexpr uint32_t nMaxByL0B = (32u * 1024u) / (kL0Size * sizeof(T1));
  constexpr uint32_t nTile = (transpose_B || N <= nMaxByL0B) ? N : nMaxByL0B;
  static_assert(transpose_B || (N % nTile == 0),
                "gemm_v0_fixp N-tiling requires N divisible by the N tile size");
  constexpr uint32_t nL0split = N / nTile;
  constexpr uint32_t kRound = ((K + 15u) / 16u) * 16u;

  SetFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);
  WaitFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);

  // L0AB ring prime: self-contained (prime_drain=true) re-arms both M_MTE1 slots
  // every call; shared mode (prime_drain=false) relies on the caller's once-per-
  // cube-loop AllocEventID prime instead (the faithful structure).
  if (prime_drain) {
    SetFlag<HardEvent::M_MTE1>(mmEv);
    SetFlag<HardEvent::M_MTE1>(mmEv + 1);
  }

  // cL0 ping-pong over N-tiles, faithful to the Ascend C reference ComputeMm2
  // (block_cube.h:833-940): two L0C slots picked by cL0BufIter%2; the
  // fixpipe(N-tile i) || mma(N-tile i+1) overlap is carried by the hardware
  // unitFlag (Mmad 0b11 on the last K + Fixpipe 0b11), NOT by software
  // M_FIX/FIX_M flags. The caller allocates C as a [2, M, nTile] L0C tensor.
  uint32_t cL0BufIter = 0;
  uint32_t tileIdx = 0;
  for (uint32_t nL0Idx = 0; nL0Idx < nL0split; nL0Idx++) {
    uint32_t bNOffset = transpose_B ? 0u : (nL0Idx * nTile * kRound);
    uint32_t c_base =
        ((cl0_base + cL0BufIter) & 1) * (M * nTile); // shared 2-slot L0C select

    for (uint32_t kL0Idx = 0; kL0Idx < kL0split; kL0Idx++) {
      initflag = (clear && (kL0Idx == 0));
      // per-K-tile contraction length: PV (kL0split==1) -> the whole k_actual;
      // QK (kL0split==4) -> 128 per tile, the last is the tail. K accumulates
      // across tiles into the same cL0 slot (initflag only on the first tile).
      uint32_t kRemain = k_actual - kL0Idx * kL0Size;
      uint32_t kSize = (kRemain < kL0Size) ? kRemain : kL0Size;
      // last K sub-tile flushes (0b11); earlier ones accumulate (0b10) --
      // faithful to ComputeMm1/Mm2 (block_cube.h:577-578/910). flush_last=false
      // (a non-final K-chunk) keeps EVERY tile at 0b10 so the cL0 keeps
      // accumulating across chunks; only the final chunk's last tile flushes.
      uint8_t unitFlag =
          (flush_last && kL0Idx == kL0split - 1) ? 0b11 : 0b10;
      uint32_t pp = (tileIdx & 1);

      uint32_t l0a_base = pp * (M * kL0Size);
      uint32_t l0b_base = pp * (nTile * kL0Size);

      WaitFlag<HardEvent::M_MTE1>(mmEv + pp);
      if constexpr (!transpose_A) {
        tl::ascend::copy_l1_to_l0a<T1, M, K>(l0a[l0a_base],
                                             A[kL0Idx * M * kL0Size], M, kSize);
      } else {
        tl::ascend::copy_l1_to_l0a<T1, K, M, true>(
            l0a[l0a_base], A[kL0Idx * 16 * kL0Size], M, kSize);
      }
      if constexpr (!transpose_B) {
        tl::ascend::copy_l1_to_l0b<T1, K, N>(
            l0b[l0b_base], B[bNOffset + kL0Idx * 16 * kL0Size], kSize, nTile);
      } else {
        // transpose_B (QK): load only the n_actual real columns (window rows of
        // K^T); the [n_actual:N] columns stay unloaded (masked downstream).
        tl::ascend::copy_l1_to_l0b<T1, N, K, true>(
            l0b[l0b_base], B[kL0Idx * N * kL0Size], kSize, n_actual);
      }
      SetFlag<HardEvent::MTE1_M>(mmEv + pp);
      WaitFlag<HardEvent::MTE1_M>(mmEv + pp);
      // accumulate this N-tile's K into the selected L0C slot; unitFlag drives
      // the mma->fixpipe pipeline so the next N-tile (other slot) overlaps.
      // Faithful to ComputeMm1/Mm2:577-582/910-914 -- WaitFlag<MTE1_M> then Mmad
      // directly (NO PipeBarrier<PIPE_M> before; that drains the M pipe and
      // defeats the cL0 ping-pong), tiny-tile hazard barrier only AFTER and gated
      // on the RUNTIME mma n (mmadParams.n): nTile for PV (always 128 -> none),
      // n_actual (window width) for QK -- the reference uses the runtime n.
      uint32_t mmaN = transpose_B ? n_actual : nTile;
      tl::ascend::mma<T1, T2, M, nTile>(l0a[l0a_base], l0b[l0b_base], C[c_base],
                                        initflag, kSize, mmaN, unitFlag);
      if ((M / 16u) * (mmaN / 16u) < 10u) {
        PipeBarrier<PIPE_M>();
      }
      SetFlag<HardEvent::M_MTE1>(mmEv + pp);
      tileIdx++;
    }

    // N-tile done: fixpipe C[c_base] -> dst column band. unitFlag 0b11 pairs
    // with the Mmad unitFlag (hardware mma->fixpipe pipeline); no software
    // M_FIX/FIX_M flag -- the next N-tile uses the other L0C slot and overlaps.
    // realTailN = mmaN (the columns the mma actually wrote): the fixpipe's nSize
    // MUST equal the mma's n, faithful to the reference (block_cube.h: mmadParams.n
    // == fixParams.nSize == nL1SizeAlign). For PV mmaN == nTile so realTailN == 0
    // (-> nTile, unchanged); for QK mmaN == n_actual (window width < nTile) -- if
    // the fixpipe instead copied the full nTile, the unitFlag fixpipe would wait
    // for cL0 columns [n_actual:nTile] that no mma ever marked ready -> HANG.
    // do_fixpipe=false (a non-final K-chunk): skip the copy-out, the cL0 slot
    // keeps accumulating; only the final chunk flushes (faithful to the reference
    // doing a single Fixpipe after all kL1 chunks have accumulated, cube.h:591).
    if (do_fixpipe) {
      uint32_t fixN = transpose_B ? n_actual : nTile;
      tl::ascend::copy_l0c_to_gm<T2, T2, LayoutGM, M, nTile>(
          dst[nL0Idx * nTile], C[c_base], N, 0, fixN, 0b11);
    }
    cL0BufIter++;
  }

  // L0AB ring drain: self-contained (prime_drain=true) drains both M_MTE1 slots
  // every call; shared mode (prime_drain=false) leaves them for the caller's
  // once-per-cube-loop FreeEventID drain (= the faithful AllocEventID/FreeEventID
  // structure -- the L0AB ring lives across all ComputeMm1/Mm2 calls).
  if (prime_drain) {
    WaitFlag<HardEvent::M_MTE1>(mmEv);
    WaitFlag<HardEvent::M_MTE1>(mmEv + 1);
  }

  SetFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);
  WaitFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);
}

// 2-way merge sort
template <typename T>
CATLASS_DEVICE void
MergeSort(const LocalTensor<T> &dst, const LocalTensor<uint8_t> &tmp,
          const LocalTensor<T> &src0, const LocalTensor<T> &src1,
          uint32_t blockLen0, uint32_t blockLen1) {
  // Note: tmp parameter is kept for API consistency with PTO backend but not
  // used in AscendC

  AscendC::MrgSort4Info params;
  params.elementLengths[0] = blockLen0;
  params.elementLengths[1] = blockLen1;
  params.elementLengths[2] = 0;
  params.elementLengths[3] = 0;
  params.ifExhaustedSuspension = false;
  params.validBit = 3;

  AscendC::MrgSortSrcList<T> srcList;
  srcList.src1 = src0;
  srcList.src2 = src1;
  srcList.src3 = src0;
  srcList.src4 = src0;

  AscendC::MrgSort<T>(dst, srcList, params);
  PipeBarrier<PIPE_V>();
}

// 3-way merge sort
template <typename T>
CATLASS_DEVICE void
MergeSort(const LocalTensor<T> &dst, const LocalTensor<uint8_t> &tmp,
          const LocalTensor<T> &src0, const LocalTensor<T> &src1,
          const LocalTensor<T> &src2, uint32_t blockLen0, uint32_t blockLen1,
          uint32_t blockLen2) {
  // Note: tmp parameter is kept for API consistency with PTO backend but not
  // used in AscendC

  AscendC::MrgSort4Info params;
  params.elementLengths[0] = blockLen0;
  params.elementLengths[1] = blockLen1;
  params.elementLengths[2] = blockLen2;
  params.elementLengths[3] = 0;
  params.ifExhaustedSuspension = false;
  params.validBit = 7;

  AscendC::MrgSortSrcList<T> srcList;
  srcList.src1 = src0;
  srcList.src2 = src1;
  srcList.src3 = src2;
  srcList.src4 = src0;

  AscendC::MrgSort<T>(dst, srcList, params);
  PipeBarrier<PIPE_V>();
}

// 4-way merge sort
template <typename T>
CATLASS_DEVICE void
MergeSort(const LocalTensor<T> &dst, const LocalTensor<uint8_t> &tmp,
          const LocalTensor<T> &src0, const LocalTensor<T> &src1,
          const LocalTensor<T> &src2, const LocalTensor<T> &src3,
          uint32_t blockLen0, uint32_t blockLen1, uint32_t blockLen2,
          uint32_t blockLen3) {
  // Note: tmp parameter is kept for API consistency with PTO backend but not
  // used in AscendC

  AscendC::MrgSort4Info params;
  params.elementLengths[0] = blockLen0;
  params.elementLengths[1] = blockLen1;
  params.elementLengths[2] = blockLen2;
  params.elementLengths[3] = blockLen3;
  params.ifExhaustedSuspension = false;
  params.validBit = 15;

  AscendC::MrgSortSrcList<T> srcList;
  srcList.src1 = src0;
  srcList.src2 = src1;
  srcList.src3 = src2;
  srcList.src4 = src3;

  AscendC::MrgSort<T>(dst, srcList, params);
  PipeBarrier<PIPE_V>();
}

template <typename T>
CATLASS_DEVICE void GatherMask(const LocalTensor<T> &dst,
                               const LocalTensor<T> &sortedTensor,
                               uint8_t src1Pattern) {
  uint32_t eleNum = sortedTensor.GetSize();
  GatherMaskParams gatherMaskParams;
  gatherMaskParams.repeatTimes = Ceil(eleNum * sizeof(T), 256);
  gatherMaskParams.src0BlockStride = 1;
  gatherMaskParams.src0RepeatStride = 8;
  gatherMaskParams.src1RepeatStride = 0;
  uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数
  GatherMask(dst, sortedTensor, src1Pattern, false, static_cast<uint32_t>(0),
             gatherMaskParams, rsvdCnt);
  PipeBarrier<PIPE_V>();
}

template <typename T, typename U>
CATLASS_DEVICE void GatherMask(const LocalTensor<T> &dst,
                               const LocalTensor<T> &sortedTensor,
                               const LocalTensor<U> &src1Pattern) {
  uint32_t eleNum = sortedTensor.GetSize();
  GatherMaskParams gatherMaskParams;
  gatherMaskParams.repeatTimes = Ceil(eleNum * sizeof(T), 256);
  gatherMaskParams.src0BlockStride = 1;
  gatherMaskParams.src0RepeatStride = 8;
  gatherMaskParams.src1RepeatStride = 0;
  uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数
  GatherMask(dst, sortedTensor, src1Pattern, false, static_cast<uint32_t>(0),
             gatherMaskParams, rsvdCnt);
}

template <typename T>
CATLASS_DEVICE void Gather(const LocalTensor<T> &dst,
                           const LocalTensor<T> &sortedTensor,
                           const LocalTensor<uint32_t> &src1Pattern) {

  int32_t count = src1Pattern.GetSize();
  int32_t scalarValue = sizeof(T);
  LocalTensor<int32_t> offset = const_cast<LocalTensor<uint32_t> &>(src1Pattern)
                                    .template ReinterpretCast<int32_t>();
  AscendC::Muls(offset, offset, scalarValue, count);
  AscendC::Gather(dst, sortedTensor,
                  offset.template ReinterpretCast<uint32_t>(),
                  static_cast<uint32_t>(0), static_cast<uint32_t>(count));
}

template <typename T>
CATLASS_DEVICE void
Gatherb(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
        const LocalTensor<uint32_t> &offset, uint8_t repeat_time,
        uint8_t dst_blk_stride, uint8_t dst_rep_stride) {
  GatherRepeatParams gatherRepeatParams;
  gatherRepeatParams.dstBlkStride = dst_blk_stride;
  gatherRepeatParams.dstRepStride = dst_rep_stride;
  Gatherb(dst.template ReinterpretCast<uint32_t>(),
          src0.template ReinterpretCast<uint32_t>(),
          offset.template ReinterpretCast<uint32_t>(), repeat_time,
          gatherRepeatParams);
  PipeBarrier<PIPE_V>();
}

template <typename T>
CATLASS_DEVICE void InitSortBuf(const LocalTensor<T> &src, int64_t eleNum,
                                int64_t rsv = 0) {
  constexpr int32_t NEG_INF = 0xFF800000;
  constexpr uint8_t VEC_REPEAT_MAX = 255;
  constexpr uint8_t B32_VEC_ELM_NUM = 64;
  uint64_t mask1[2] = {0x5555555555555555, 0};
  uint64_t mask0[2] = {0xaaaaaaaaaaaaaaaa, 0};
  int64_t repeatNum = eleNum / B32_VEC_ELM_NUM;
  int64_t forLoop = repeatNum / VEC_REPEAT_MAX;
  int64_t forRemain = repeatNum % VEC_REPEAT_MAX;
  for (int i = 0; i < forLoop; i++) {
    Duplicate(src.template ReinterpretCast<int32_t>(), NEG_INF, mask1,
              VEC_REPEAT_MAX, 1, 8);
    Duplicate(src.template ReinterpretCast<int32_t>(), -1, mask0,
              VEC_REPEAT_MAX, 1, 8);
  }
  if (forRemain > 0) {
    Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX *
                                                      B32_VEC_ELM_NUM],
              NEG_INF, mask1, forRemain, 1, 8);
    Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX *
                                                      B32_VEC_ELM_NUM],
              -1, mask0, forRemain, 1, 8);
  }
  PipeBarrier<PIPE_V>();
}

template <typename T>
CATLASS_DEVICE void brcb(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
                         const uint8_t repeatTime, const uint16_t dstBlkStride,
                         const uint16_t dstRepStride) {
  AscendC::BrcbRepeatParams repeatParams(dstBlkStride, dstRepStride);
  AscendC::Brcb<T>(dst, src0, repeatTime, repeatParams);
}

// Row-broadcast elementwise: dst[i, j] = src0[i, j] OP src1_col[i], i.e. every
// column of row i is combined with that row's single src1 value -- the faithful
// equivalent of the Ascend C reference's RowDivs / RowMuls
// (swa_block_vector.h: Brcb the [M,1] column into an [M, blk] tile, then
// Div/Sub with src1BlkStride=0, src1RepStride=1). This avoids materialising an
// [M, N] broadcast buffer (which overflowed the vector UB once the cube/vector
// pipeline stopped the planner from reusing the softmax/output buffers). `tmp`
// is an [M, 32/sizeof(T)] scratch for the Brcb. N must be a multiple of the
// per-repeat element count (256/sizeof(T)).
template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void row_expand_div(const LocalTensor<T> &dst,
                                   const LocalTensor<T> &src0,
                                   const LocalTensor<T> &src1_col,
                                   const LocalTensor<T> &tmp) {
  constexpr uint32_t BLK = 32 / sizeof(T);   // elems per 32B block (f32: 8)
  constexpr uint32_t MASK = 256 / sizeof(T); // elems per 256B repeat (f32: 64)
  static_assert(N % MASK == 0,
                "row_expand_div requires N % (256/sizeof(T)) == 0");
  // Only the row-repeat (M-repeat) branch of the reference RowDivs is ported
  // (no column-repeat else branch, no tail) -- valid when N/MASK <= M and rows
  // are contiguous (row pitch == N). Both holds for the SWA caller (M=32, N<=512).
  static_assert(N / MASK <= M,
                "row_expand_div assumes N/MASK <= M (row-repeat branch only)");
  // Brcb writes ceil(M/8)*8 rows (8 per repeat), so M must be a multiple of 8
  // or it overflows the [M, blk] tmp scratch.
  static_assert(M % 8 == 0,
                "row_expand_div requires M % 8 == 0 (Brcb writes 8-row blocks)");
  // Brcb outputs 8 blocks (256B) per repeat regardless of dtype, so the repeat
  // count is ceil(M/8) and dstRepStride is 8 blocks -- NOT BLK, which is only
  // correct for fp32 (BLK=8); for fp16 (BLK=16) it would skip half the rows.
  AscendC::Brcb(tmp, src1_col, (M + 7) / 8, AscendC::BrcbRepeatParams(1, 8));
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::BinaryRepeatParams rp;
  rp.src0BlkStride = 1;
  rp.src1BlkStride = 0;
  rp.dstBlkStride = 1;
  rp.src0RepStride = N / BLK;
  rp.src1RepStride = 1;
  rp.dstRepStride = N / BLK;
  for (uint32_t i = 0; i < N / MASK; i++) {
    AscendC::Div(dst[i * MASK], src0[i * MASK], tmp, MASK, M, rp);
  }
}

// Row-broadcast subtraction: dst[i, j] = src0[i, j] - src1_col[i]. Same scheme
// as row_expand_div (Ascend C broadcasts the per-row max the same way inside its
// softmax); replaces a materialised [M, N] max-broadcast buffer.
template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void row_expand_sub(const LocalTensor<T> &dst,
                                   const LocalTensor<T> &src0,
                                   const LocalTensor<T> &src1_col,
                                   const LocalTensor<T> &tmp) {
  constexpr uint32_t BLK = 32 / sizeof(T);
  constexpr uint32_t MASK = 256 / sizeof(T);
  static_assert(N % MASK == 0,
                "row_expand_sub requires N % (256/sizeof(T)) == 0");
  static_assert(N / MASK <= M,
                "row_expand_sub assumes N/MASK <= M (row-repeat branch only)");
  // Brcb writes ceil(M/8)*8 rows (8 per repeat), so M must be a multiple of 8
  // or it overflows the [M, blk] tmp scratch.
  static_assert(M % 8 == 0,
                "row_expand_sub requires M % 8 == 0 (Brcb writes 8-row blocks)");
  // Brcb outputs 8 blocks (256B) per repeat regardless of dtype, so the repeat
  // count is ceil(M/8) and dstRepStride is 8 blocks -- NOT BLK, which is only
  // correct for fp32 (BLK=8); for fp16 (BLK=16) it would skip half the rows.
  AscendC::Brcb(tmp, src1_col, (M + 7) / 8, AscendC::BrcbRepeatParams(1, 8));
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::BinaryRepeatParams rp;
  rp.src0BlkStride = 1;
  rp.src1BlkStride = 0;
  rp.dstBlkStride = 1;
  rp.src0RepStride = N / BLK;
  rp.src1RepStride = 1;
  rp.dstRepStride = N / BLK;
  for (uint32_t i = 0; i < N / MASK; i++) {
    AscendC::Sub(dst[i * MASK], src0[i * MASK], tmp, MASK, M, rp);
  }
}

// Row-broadcast multiplication: dst[i, j] = src0[i, j] * src1_col[i]. Same scheme
// as row_expand_div (Brcb the [M,1] column into an [M, blk] tile, then Mul with
// src1BlkStride=0 / src1RepStride=1) -- the faithful equivalent of the Ascend C
// reference's RowMuls (cfa/scfa flash-attention PV rescale: multiply the previous
// KV-tile's partial output by the per-row exp(m_old - m_new) factor). The non-PTO
// counterpart of the PTO TROWEXPANDMUL; emitted by tl.ascend_row_expand_mul_nd.
template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void row_expand_mul(const LocalTensor<T> &dst,
                                   const LocalTensor<T> &src0,
                                   const LocalTensor<T> &src1_col,
                                   const LocalTensor<T> &tmp) {
  constexpr uint32_t BLK = 32 / sizeof(T);
  constexpr uint32_t MASK = 256 / sizeof(T);
  static_assert(N % MASK == 0,
                "row_expand_mul requires N % (256/sizeof(T)) == 0");
  static_assert(N / MASK <= M,
                "row_expand_mul assumes N/MASK <= M (row-repeat branch only)");
  // Brcb writes ceil(M/8)*8 rows (8 per repeat), so M must be a multiple of 8
  // or it overflows the [M, blk] tmp scratch.
  static_assert(M % 8 == 0,
                "row_expand_mul requires M % 8 == 0 (Brcb writes 8-row blocks)");
  // Brcb outputs 8 blocks (256B) per repeat regardless of dtype, so the repeat
  // count is ceil(M/8) and dstRepStride is 8 blocks -- NOT BLK, which is only
  // correct for fp32 (BLK=8); for fp16 (BLK=16) it would skip half the rows.
  AscendC::Brcb(tmp, src1_col, (M + 7) / 8, AscendC::BrcbRepeatParams(1, 8));
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::BinaryRepeatParams rp;
  rp.src0BlkStride = 1;
  rp.src1BlkStride = 0;
  rp.dstBlkStride = 1;
  rp.src0RepStride = N / BLK;
  rp.src1RepStride = 1;
  rp.dstRepStride = N / BLK;
  for (uint32_t i = 0; i < N / MASK; i++) {
    AscendC::Mul(dst[i * MASK], src0[i * MASK], tmp, MASK, M, rp);
  }
}

// SoftmaxFlashV2 config: WITHOUT_BRC -> the max/sum/exp outputs are (m,1), not
// broadcast to (m,N). Mirrors the Ascend C reference's
// SAS_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC (sparse_attn_sharedkv_common.h:26).
constexpr SoftmaxConfig kSoftmaxFlashV2CfgWithoutBrc = {
    false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

// Faithful Ascend C SoftmaxFlashV2 (sparse_attn_sharedkv_swa_block_vector.h
// SoftmaxFlashV2Compute): the variable-N online softmax over the SWA window.
//
// In Ascend C the score buffer (mmResUb) is win_align-strided, so the library
// runs on a contiguous [M, columnCount=win_align] tile with actualColumnCount =
// winm. Our `src`/`dst` are a fixed [M, N]-strided UB tile (N = BI), whose first
// `col_count` (= 16-aligned window win_align) columns hold the score and whose
// [col_count, N) tail is uninitialised. Feeding the library N=128 as columnCount
// would make it touch that uninitialised tail (flaky NaN). Instead we replicate
// Ascend C's win_align-contiguous layout: compact src[:, 0:col_count] into the
// contiguous [M, col_count] `compact` buffer, run SoftmaxFlashV2 there (srcK =
// col_count, the library's designed range), then scatter back to dst[:,
// 0:col_count]. The uninitialised [col_count, N) is never read; the >=16
// alignment tail [actual_col, col_count) is exp'd but excluded by the reduce
// (oriSrcK = actual_col), exactly as in Ascend C.
//
// dst (== src, in place via isReuseSource) gets P = exp(score - newMax);
// `sum`/`max` are the running sum/max (m,1) seeded by `in_sum` (1.0) / `in_max`
// (per-row sink); `expmax` (m,1) is the flash rescale factor (unused for a
// single block). `tmp` is the library's uint8 shared scratch. col_count and N
// must both be multiples of 32/sizeof(T).
template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void
softmax_flash_v2(const LocalTensor<T> &dst, const LocalTensor<T> &sum,
                 const LocalTensor<T> &max, const LocalTensor<T> &expmax,
                 const LocalTensor<T> &src, const LocalTensor<T> &in_sum,
                 const LocalTensor<T> &in_max, const LocalTensor<uint8_t> &tmp,
                 const LocalTensor<T> &compact, uint32_t col_count,
                 uint32_t actual_col) {
  constexpr uint32_t BLK = 32 / sizeof(T); // elems per 32B datablock (f32: 8)
  static_assert(N % BLK == 0, "softmax_flash_v2 requires N % (32/sizeof(T)) == 0");
  // compact src[:, 0:col_count] (N-strided) -> compact ([M, col_count] contig).
  AscendC::DataCopy(
      compact, src,
      AscendC::DataCopyParams{static_cast<uint16_t>(M),
                              static_cast<uint16_t>(col_count / BLK),
                              static_cast<uint16_t>((N - col_count) / BLK), 0});
  AscendC::PipeBarrier<PIPE_V>();

  SoftMaxShapeInfo srcShape{M, col_count, M, actual_col};
  SoftMaxTiling tiling = SoftMaxFlashV2TilingFunc(
      srcShape, sizeof(T), sizeof(T), tmp.GetSize(), true, false);
  SoftmaxFlashV2<T, true, true, false, false, kSoftmaxFlashV2CfgWithoutBrc>(
      compact, sum, max, compact, expmax, in_sum, in_max, tmp, tiling, srcShape);
  AscendC::PipeBarrier<PIPE_V>();

  // scatter compact ([M, col_count] contig) -> dst[:, 0:col_count] (N-strided).
  AscendC::DataCopy(
      dst, compact,
      AscendC::DataCopyParams{static_cast<uint16_t>(M),
                              static_cast<uint16_t>(col_count / BLK), 0,
                              static_cast<uint16_t>((N - col_count) / BLK)});
}

template <typename T1, typename T2, typename LayOutL1, typename LayoutGM,
          uint32_t M, uint32_t N, uint32_t K, uint32_t baseM, uint32_t baseN,
          uint32_t baseK, bool init, bool is_transpose_A = false,
          bool is_transpose_B = false, bool enable_relu = false>
CATLASS_DEVICE void gemmL1(LocalTensor<T1> A, LocalTensor<T1> B,
                           GlobalTensor<T1> C, LocalTensor<T1> A2,
                           LocalTensor<T1> B2, LocalTensor<T2> C2) {
  for (uint32_t loopM = 0; loopM < M / baseM; loopM++) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);

    copy_l1_to_l0a<T1, M, K, baseM, baseK>(A2, A[loopM * baseM * 16]);

    for (uint32_t loopN = 0; loopN < N / baseN; loopN++) {
      copy_l1_to_l0b<T1, K, N, baseK, baseN>(B2, B[loopN * baseN * K]);

      AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

      mma<T1, T2, baseM, baseN, baseK, init>(A2, B2, C2);

      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(0);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);

      copy_l0c_to_gm<T1, T2, LayoutGM, baseM, baseN, M, N>(
          C[loopM * baseM * N + loopN * baseN], C2, enable_relu);

      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
CATLASS_DEVICE void
Broadcast(const LocalTensor<T> &dst, const LocalTensor<T> &src,
          LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t dstShape[dim],
          const uint32_t srcShape[dim]) {
  AscendC::Broadcast<T, dim, axis, isReuseSource>(dst, src, dstShape, srcShape,
                                                  sharedTmpBuffer);
}

template <typename T>
CATLASS_DEVICE void Fill(const LocalTensor<T> &dst, const T &scalarValue,
                         const int32_t &count) {
  AscendC::Duplicate<T>(dst, scalarValue, count);
}

template <typename T>
CATLASS_DEVICE void ArithProgression(const LocalTensor<T> &dst,
                                     const T firstValue, const T diffValue,
                                     const int32_t count) {
  AscendC::ArithProgression<T>(dst, firstValue, diffValue, count);
}

template <typename T>
CATLASS_DEVICE void Sort(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                         const LocalTensor<T> &tmp, const int32_t repeatTimes,
                         const int32_t actualCount) {
  if constexpr (sizeof(T) == 2) {
    // B16 (half): MrgSort requires >= 256 bytes per source, but Sort32 only
    // produces 128 bytes per block for B16. Work around by sorting in float.
    //
    // Layout in tmp (N = alignedCount, as float elements via ReinterpretCast):
    //   ftmp[0 .. N*2-1]    = Sort32 output + merge ping-pong buffer A
    //   ftmp[N*2 .. N*4-1]  = Sort<float>'s dst (merge ping-pong buffer B)
    //     - before Sort32: indices at [N*2..N*3), float_src at [N*3..N*4)
    //     - after  Sort32: entire region free for merge
    // Total: 4N float elements = 8N half elements.
    uint32_t N = repeatTimes * 32;

    auto ftmp = tmp.template ReinterpretCast<float>();
    auto float_src = ftmp[N * 3];

    // Cast half → float
    AscendC::Cast(float_src, src, AscendC::RoundMode::CAST_NONE, N);

    // Sort<float> guarantees result in dst (= ftmp[N*2])
    Sort<float>(ftmp[N * 2], float_src, ftmp, repeatTimes, actualCount);

    // Cast float result → half (2*N elements: interleaved [value, index] pairs)
    AscendC::Cast(dst, ftmp[N * 2], AscendC::RoundMode::CAST_RINT, N * 2);
    PipeBarrier<PIPE_V>();
    return;
  }

  constexpr uint32_t blockSize = 32;
  uint32_t alignedCount = repeatTimes * blockSize;
  uint32_t padCount = alignedCount - actualCount;
  uint32_t blockNum = repeatTimes;

  // Generate ascending indices as float values (0.0, 1.0, 2.0, ...) in dst
  // (temporary storage — overwritten by merge later). This allows tmp to
  // be only alignedCount*2 elements instead of alignedCount*4, because dst
  // (which is 2*alignedCount for interleaved output) doubles as the second
  // merge ping-pong buffer.
  AscendC::ArithProgression<T>(dst, T(0), T(1), alignedCount);
  PipeBarrier<PIPE_V>();
  LocalTensor<uint32_t> indices = dst.template ReinterpretCast<uint32_t>();

  // Pad src in-place with -inf for unused positions
  if (padCount > 0) {
    T negInf = -CUDART_INF_F;
    constexpr uint32_t elemPerBlock =
        32 / sizeof(T); // 16 for half, 8 for float
    uint32_t alignedActual = (actualCount / elemPerBlock) * elemPerBlock;
    uint32_t inBlockOffset = actualCount - alignedActual;

    if (inBlockOffset == 0) {
      // actualCount is already 32-byte aligned, simple Duplicate
      AscendC::Duplicate<T>(src[actualCount], negInf, padCount);
    } else {
      // Non-aligned: split into aligned bulk fill + masked partial block
      uint32_t nextAligned = alignedActual + elemPerBlock;
      // Fill full aligned blocks after the partial one
      if (nextAligned < alignedCount) {
        AscendC::Duplicate<T>(src[nextAligned], negInf,
                              alignedCount - nextAligned);
      }
      // Fill partial block using mask to preserve valid elements before
      // actualCount
      uint64_t mask0 = 0;
      for (uint32_t i = inBlockOffset; i < elemPerBlock; i++) {
        mask0 |= (1ULL << i);
      }
      uint64_t masks[2] = {mask0, 0};
      AscendC::Duplicate(src[alignedActual], negInf, masks, (uint8_t)1,
                         (uint16_t)1, (uint8_t)0);
    }
    PipeBarrier<PIPE_V>();
  }

  // Sort32: each 32-element block → tmp[0..alignedCount*2-1] (bufA)
  AscendC::Sort32(tmp, src, indices, repeatTimes);
  PipeBarrier<PIPE_V>();

  // Merge ping-pong between tmp[0..2N-1] and dst[0..2N-1].
  // tmp only needs alignedCount*2 elements (Sort32 output size).

  if (blockNum > 1) {
    uint32_t fullSegSize = blockSize;
    uint32_t lastSegSize = blockSize;
    uint32_t numSegs = blockNum;
    bool readFromTmp = true; // Sort32 output is in tmp

    while (numSegs > 1) {
      uint32_t newNumSegs = 0;
      uint32_t inOffset = 0;
      uint32_t outOffset = 0;

      for (uint32_t g = 0; g < numSegs; g += 4) {
        uint32_t groupCount = numSegs - g;
        if (groupCount > 4) {
          groupCount = 4;
        }
        uint32_t len0 = (g == numSegs - 1) ? lastSegSize : fullSegSize;
        uint32_t len1 = 0, len2 = 0, len3 = 0;
        uint32_t totalElems = len0;
        if (groupCount > 1) {
          len1 = (g + 1 == numSegs - 1) ? lastSegSize : fullSegSize;
          totalElems += len1;
        }
        if (groupCount > 2) {
          len2 = (g + 2 == numSegs - 1) ? lastSegSize : fullSegSize;
          totalElems += len2;
        }
        if (groupCount > 3) {
          len3 = (g + 3 == numSegs - 1) ? lastSegSize : fullSegSize;
          totalElems += len3;
        }

        if (groupCount == 1) {
          if (readFromTmp) {
            AscendC::DataCopy(dst[outOffset], tmp[inOffset], len0 * 2);
          } else {
            AscendC::DataCopy(tmp[outOffset], dst[inOffset], len0 * 2);
          }
        } else {
          AscendC::MrgSort4Info params;
          params.elementLengths[0] = len0;
          params.elementLengths[1] = len1;
          params.elementLengths[2] = groupCount > 2 ? len2 : 0;
          params.elementLengths[3] = groupCount > 3 ? len3 : 0;
          params.ifExhaustedSuspension = false;
          params.validBit = (1 << groupCount) - 1;

          uint32_t off0 = inOffset;
          uint32_t off1 = off0 + len0 * 2;
          uint32_t off2 = off1 + len1 * 2;
          uint32_t off3 = off2 + len2 * 2;

          AscendC::MrgSortSrcList<T> srcList;
          if (readFromTmp) {
            srcList.src1 = tmp[off0];
            srcList.src2 = tmp[off1];
            srcList.src3 = groupCount > 2 ? tmp[off2] : tmp[off0];
            srcList.src4 = groupCount > 3 ? tmp[off3] : tmp[off0];
            AscendC::MrgSort<T>(dst[outOffset], srcList, params);
          } else {
            srcList.src1 = dst[off0];
            srcList.src2 = dst[off1];
            srcList.src3 = groupCount > 2 ? dst[off2] : dst[off0];
            srcList.src4 = groupCount > 3 ? dst[off3] : dst[off0];
            AscendC::MrgSort<T>(tmp[outOffset], srcList, params);
          }
        }

        inOffset += totalElems * 2;
        outOffset += totalElems * 2;
        newNumSegs++;
      }

      PipeBarrier<PIPE_V>();

      uint32_t lastGroupStart = ((numSegs - 1) / 4) * 4;
      uint32_t lastGroupCount = numSegs - lastGroupStart;
      uint32_t newLastSegSize = 0;
      for (uint32_t i = 0; i < lastGroupCount; i++) {
        newLastSegSize +=
            (lastGroupStart + i == numSegs - 1) ? lastSegSize : fullSegSize;
      }

      fullSegSize = (newNumSegs > 1) ? 4 * fullSegSize : newLastSegSize;
      lastSegSize = newLastSegSize;
      numSegs = newNumSegs;
      readFromTmp = !readFromTmp;
    }

    // readFromTmp=true means last round wrote to tmp → result in tmp
    if (readFromTmp) {
      AscendC::DataCopy(dst, tmp, alignedCount * 2);
    }
  } else {
    // Single block: Sort32 output is in tmp, copy to dst
    AscendC::DataCopy(dst, tmp, alignedCount * 2);
  }
}

template <typename T>
CATLASS_DEVICE void ClampMax(const LocalTensor<T> &dst,
                             const LocalTensor<T> &buffer,
                             const LocalTensor<uint8_t> &tmp,
                             const T scalarValue, const int32_t count) {
  AscendC::ClampMax<T>(dst, buffer, tmp, scalarValue, count);
}

template <typename T>
CATLASS_DEVICE void TopK(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                         const LocalTensor<T> &tmp, const int32_t K,
                         const int32_t repeatTimes, const int32_t actualCount) {
  // Use tmp as the full-size sort destination (2 * alignedCount elements).
  // Sort writes its result into tmp's first region; we then copy the top-K
  // portion into dst.
  uint32_t alignedCount = repeatTimes * 32;
  // sortDst needs 2 * alignedCount elements; reuse the tail of tmp.
  // Layout of tmp: [0 .. 2*alignedCount-1] = sortDst, [2*alignedCount ..] =
  // sortTmp
  auto sortDst = tmp;
  auto sortTmp = tmp[alignedCount * 2];
  Sort<T>(sortDst, src, sortTmp, repeatTimes, actualCount);
  PipeBarrier<PIPE_V>();
  // Copy 2*K elements (interleaved value-index pairs) from sorted result to
  // dst. DataCopy requires the byte count to be a multiple of 32 bytes, so
  // round up.
  uint32_t topkElems = 2 * K;
  constexpr uint32_t elemsPerBlock = 32 / sizeof(T);
  uint32_t alignedTopk =
      ((topkElems + elemsPerBlock - 1) / elemsPerBlock) * elemsPerBlock;
  AscendC::DataCopy(dst, sortDst, alignedTopk);
}

template <typename T>
CATLASS_DEVICE void ClampMin(const LocalTensor<T> &dst,
                             const LocalTensor<T> &buffer,
                             const LocalTensor<uint8_t> &tmp,
                             const T scalarValue, const int32_t count) {
  AscendC::ClampMin<T>(dst, buffer, tmp, scalarValue, count);
}

template <typename T>
CATLASS_DEVICE void
Clamp(const LocalTensor<T> &dst, const LocalTensor<T> &buffer,
      const LocalTensor<uint8_t> &tmp, const T minScalarValue,
      const T maxScalarValue, const int32_t count) {
  AscendC::ClampMin<T>(dst, buffer, tmp, minScalarValue, count);
  AscendC::ClampMax<T>(dst, dst, tmp, maxScalarValue, count);
}

template <typename T, typename U>
CATLASS_DEVICE void
GatherMask_experiment(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
                      const LocalTensor<U> &src1Pattern, const bool reduceMode,
                      const uint32_t mask, const uint32_t src0BlockStride,
                      const uint32_t repeatTimes, uint32_t src0RepeatStride,
                      const uint32_t src1RepeatStride, uint64_t rsvdCnt) {
  GatherMaskParams gatherMaskParams;
  gatherMaskParams.repeatTimes = repeatTimes;
  gatherMaskParams.src0BlockStride = src0BlockStride;
  gatherMaskParams.src0RepeatStride = src0RepeatStride;
  gatherMaskParams.src1RepeatStride = src1RepeatStride;
  GatherMask(dst, src0, src1Pattern, reduceMode, mask, gatherMaskParams,
             rsvdCnt);
}

template <typename T>
CATLASS_DEVICE void
Fill_experiment(const LocalTensor<T> &dst, const T &scalarValue, uint64_t mask0,
                const uint8_t repeatTime, const uint16_t dstBlockStride,
                const uint8_t dstRepeatStride) {
  uint64_t mask[1] = {mask0};
  AscendC::Duplicate(dst, scalarValue, mask, repeatTime, dstBlockStride,
                     dstRepeatStride);
}

template <typename T>
CATLASS_DEVICE void
Sum_experiment(const LocalTensor<T> &dst, const LocalTensor<T> &src,
               const uint32_t outter, const uint32_t inner, const uint32_t n) {
  SumParams sumParams;
  sumParams.outter = outter;
  sumParams.inner = inner;
  sumParams.n = n;
  AscendC::Sum(dst, src, sumParams);
}

template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void transpose_16x16(LocalTensor<T> const &dst,
                                    LocalTensor<T> const &src) {
  TransDataTo5HDParams transDataParams;
  transDataParams.dstHighHalf = false;
  transDataParams.srcHighHalf = false;
  transDataParams.repeatTimes = N;
  if (transDataParams.repeatTimes == 1) {
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
  } else {
    transDataParams.dstRepStride = M;
    transDataParams.srcRepStride = 1;
  }

  __ubuf__ T *dstList[16];
  __ubuf__ T *srcList[16];

  if constexpr (sizeof(T) == 4) {
    for (int32_t m = 0; m < 16; m = m + 2) {
      dstList[m] = (__ubuf__ T *)dst[16 * (m / 2)].GetPhyAddr();
      dstList[m + 1] = (__ubuf__ T *)dst[16 * (m / 2) + 16].GetPhyAddr();
    }
    for (int32_t n = 0; n < 16; n++) {
      srcList[n] = (__ubuf__ T *)src[n * 16].GetPhyAddr();
    }
  } else {
    for (int i = 0; i < 16; i++) {
      dstList[i] = (__ubuf__ T *)dst[i * N].GetPhyAddr();
      srcList[i] = (__ubuf__ T *)src[i * M].GetPhyAddr();
    }
  }

  AscendC::TransDataTo5HDImpl<T>(dstList, srcList, transDataParams);
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, uint32_t FullM = 16, uint32_t FullN = 16>
CATLASS_DEVICE void transpose(LocalTensor<T> const &dst,
                              LocalTensor<T> const &src) {
  if constexpr (FullM == 16 && FullN == 16) {
    if constexpr (sizeof(T) == 2) {
      AscendC::Transpose(dst, src);
    } else {
      for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
          dst.SetValue(i * 16 + j, src.GetValue(j * 16 + i));
        }
      }
    }
  } else {
    for (uint32_t ti = 0; ti < FullM / 16; ti++) {
      for (uint32_t tj = 0; tj < FullN / 16; tj++) {
        for (int i = 0; i < 16; i++) {
          for (int j = 0; j < 16; j++) {
            dst.SetValue((tj * 16 + j) * FullM + (ti * 16 + i),
                         src.GetValue((ti * 16 + i) * FullN + (tj * 16 + j)));
          }
        }
      }
    }
  }
}

} // namespace tl::ascend
