"""Probe: strided masked exp over the first COL (compile-time) valid columns of a
wide N-strided buffer -- exp only the window, in place, without compacting to a
contiguous tile. This is what the narrow online-softmax needs and the contiguous
``AscendC::Exp(dst, src, count)`` cannot do on a strided sub-region of a wider
buffer (a flat count would walk across the row padding into the next row).

``T.tile.exp_experiment`` exps a 64-column (fp32) chunk of every row in one call,
striding the buffer's physical column count (read from the declaration by
ExpExperimentCodegen) between rows. Callers loop ceil(COL/64) chunks over the
valid window; the [COL, N) tail is left untouched and never read. This mirrors the
existing ``row_expand_sub_experiment`` used by xattention -- the unary counterpart
so the softmax's exp step can stay strided-narrow instead of compacting.

We dump the generated Ascend C so we can see the ``exp_mask`` call and its
repeat/stride arguments, then check exp of the first COL columns against torch.
"""

import tilelang
from tilelang import language as T
import torch

torch.set_default_device("npu")
torch.manual_seed(0)

tilelang.disable_cache()

M = 16
N = 512  # physical (padded) row width
CHUNK = 64  # fp32 elems per repeat (256B per-repeat limit)
dtype = "float"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def build(col):
    @tilelang.jit(out_idx=[1], target="ascendc", pass_configs=pass_configs)
    def _k():
        @T.prim_func
        def main(
            Input: T.Tensor([M, N], dtype),
            Out: T.Tensor([M, N], dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                in_ub = T.alloc_ub((M, N), dtype)
                if vid == 0:
                    T.copy(Input, in_ub)
                    # strided masked exp over [0:col], 64-col chunks, all M rows/call
                    for k in range(col // CHUNK):
                        sc = k * CHUNK
                        T.tile.exp_experiment(in_ub[:, sc : sc + CHUNK], in_ub[:, sc : sc + CHUNK])
                    T.copy(in_ub, Out)

        return main

    return _k()


func = build(128)
print("init OK")

# Dump the generated Ascend C so we can see the exp_mask call + repeat/stride.
_src = func.get_kernel_source()
with open("/tmp/exp_exp_cg.cpp", "w") as _f:
    _f.write(_src)
print("=== exp_mask / Exp lines ===")
for _i, _ln in enumerate(_src.splitlines()):
    if "exp_mask" in _ln or "AscendC::Exp" in _ln:
        print(f"{_i:5d}: {_ln.strip()[:140]}")
print("=== end (full dump at /tmp/exp_exp_cg.cpp) ===")

inp = torch.randn((M, N), dtype=torch.float)
for col in (64, 128, 256):
    func = build(col)
    torch.npu.synchronize()
    out = func(inp)
    torch.npu.synchronize()
    got = out[:, :col].cpu()
    ref = torch.exp(inp[:, :col].cpu())  # exp over the valid window
    err = (got - ref).abs().max().item()
    print(f"col={col:4d}  max|exp diff|={err:.4g}  {'OK' if err < 1e-3 else '!! MISMATCH'}")

print("done")
