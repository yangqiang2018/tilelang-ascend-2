"""Probe: strided WholeReduceSum over the first N (runtime) valid columns of a
wide 512-strided buffer -- the shape a narrow online-softmax needs but the
high-level (contiguous) reduce cannot do.

WholeReduceSum(dst, src, mask, repeatTimes, dstRepStride, srcBlkStride,
srcRepStride) reduces `mask` contiguous elements per repeat, one repeat per
row, with `srcRepStride` (in 32B blocks) skipping the padding between rows. So
mask = N (valid cols), repeatTimes = M, srcRepStride = 512-elem-row / 8-fp32-
per-block = 64 blocks.

We dump the raw dst so we can see (a) whether it strides correctly, (b) how the
M results are laid out (each WholeReduce result may sit on its own 32B block, so
they can be 8-fp32 apart rather than packed), and (c) the largest N that still
reduces in a single repeat (fp32 per-repeat limit is 64 elems = 256B).
"""

import tilelang
from tilelang import language as T
import torch

torch.set_default_device("npu")
torch.manual_seed(42)

tilelang.disable_cache()

M = 16
NBUF = 512
BLK = 8  # fp32 elements per 32B block
STRIDE_BLK = NBUF // BLK  # 64: physical row stride in blocks
dtype = "float"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def build(n_col):
    @tilelang.jit(out_idx=[1], target="ascendc", pass_configs=pass_configs)
    def _k():
        @T.prim_func
        def main(
            Input: T.Tensor([M, NBUF], dtype),
            OutSum: T.Tensor([1, NBUF], dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                in_ub = T.alloc_ub((M, NBUF), dtype)
                sum_ub = T.alloc_ub((1, NBUF), dtype)
                if vid == 0:
                    T.copy(Input, in_ub)
                    # strided reduce: mask=n_col per row, M rows, rows 64 blocks apart
                    T.tile.wholereducesum(sum_ub, in_ub, n_col, M, 1, 1, STRIDE_BLK)
                    T.copy(sum_ub, OutSum)

        return main

    return _k()


input = torch.randn((M, NBUF), dtype=torch.float)

for n_col in [64, 100, 128, 256, 511]:
    func = build(n_col)
    torch.npu.synchronize()
    out = func(input)
    torch.npu.synchronize()
    row = out[0].cpu()  # [NBUF]
    ref = torch.sum(input[:, :n_col].cpu(), dim=-1)  # [M]

    # Try two candidate layouts for the M results: packed [0:M] and 8-spread.
    packed = row[:M]
    spread = row[: M * BLK : BLK]
    err_packed = (packed - ref).abs().max().item()
    err_spread = (spread - ref).abs().max().item()
    print(f"n={n_col:4d}  err(packed[0:M])={err_packed:.4g}  err(spread[::8])={err_spread:.4g}")
    if min(err_packed, err_spread) > 1e-2:
        print(f"   raw dst[0:24] = {[round(x, 3) for x in row[:24].tolist()]}")
        print(f"   ref[0:8]      = {[round(x, 3) for x in ref[:8].tolist()]}")

print("done")
