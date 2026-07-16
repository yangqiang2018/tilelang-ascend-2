"""Row reduce over a runtime (non-constant) valid column count.

The source buffer is a fixed 512-wide UB tile, but only the first ``n`` columns
of each row are logically valid, where ``n`` is a runtime scalar (read from an
input tensor into a register) -- not a compile-time constant. ``real_shape=[M,
n]`` lowers the reduce to the ``reduce_max_rt`` / ``reduce_sum_rt`` intrinsics,
which carry ``n`` as a runtime argument instead of a template parameter, so the
reduce only touches the ``n`` valid columns and ignores the padding.

The padding columns are left as random garbage on purpose: a reduce that
incorrectly walks all 512 columns would fold that garbage into the result and
fail the check, so this doubles as a discriminator for the runtime-extent path.
The same compiled kernel is called with several different ``n`` values to prove
``n`` is resolved at runtime (a single build, not one build per width).
"""

import tilelang
from tilelang import language as T
import torch

torch.set_default_device("npu")
torch.manual_seed(42)

tilelang.disable_cache()

M = 16
NBUF = 512
dtype = "float"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2, 3], target="ascendc", pass_configs=pass_configs)
def reduce_runtime_n():
    @T.prim_func
    def main(
        Input: T.Tensor([M, NBUF], dtype),
        Ncol: T.Tensor([1], "int32"),
        OutMax: T.Tensor([1, NBUF], dtype),
        OutSum: T.Tensor([1, NBUF], dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            in_ub = T.alloc_ub((M, NBUF), dtype)
            max_ub = T.alloc_ub((1, NBUF), dtype)
            sum_ub = T.alloc_ub((1, NBUF), dtype)

            if vid == 0:
                T.copy(Input, in_ub)
                # Runtime scalar column count read straight from a GM param,
                # mirroring the ``act_q_lens[b]`` / ``seqused_kv[b]`` scalar
                # reads that feed the real softmax window width.
                n = Ncol[0]
                # reduce_max first (does not destroy src), then reduce_sum
                # (may use src as scratch) -- same ordering the online-softmax
                # path relies on.
                T.reduce_max(in_ub, max_ub, dim=-1, real_shape=[M, n])
                T.reduce_sum(in_ub, sum_ub, dim=-1, real_shape=[M, n])
                T.copy(max_ub, OutMax)
                T.copy(sum_ub, OutSum)

    return main


func = reduce_runtime_n()
print("init successful!")

# Dump the generated Ascend C so we can see exactly what the reduce_*_rt call
# and its runtime N argument lowered to.
_src = func.get_kernel_source()
with open("/tmp/reduce_rt_cg.cpp", "w") as _f:
    _f.write(_src)
print("=== reduce / Reduce lines ===")
for _i, _ln in enumerate(_src.splitlines()):
    if "reduce" in _ln.lower() or "Reduce" in _ln or "Ncol" in _ln:
        print(f"{_i:5d}: {_ln.strip()[:150]}")
print("=== end ===")

input = torch.randn((M, NBUF), dtype=torch.float)

for n_val in [100, 250, 64, 511, 16]:
    ncol = torch.tensor([n_val], dtype=torch.int32)
    torch.npu.synchronize()
    outmax, outsum = func(input, ncol)
    torch.npu.synchronize()

    ref_max = torch.max(input[:, :n_val], dim=-1).values  # [M]
    ref_sum = torch.sum(input[:, :n_val], dim=-1)  # [M]

    torch.testing.assert_close(ref_max, outmax[0, :M], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ref_sum, outsum[0, :M], rtol=1e-2, atol=1e-2)
    print(f"n={n_val:4d}  max/sum match")

print("Runtime-N reduce OK for all widths!")
