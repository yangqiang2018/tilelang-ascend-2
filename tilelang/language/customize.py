# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations

import tilelang.language as T
from tilelang.language.tir import op
from tvm.tir import PrimExpr, Buffer, BufferRegion, Var
from tvm import tir
from tilelang.language.ascend import _dtype
import math


def atomic_add(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added

    Returns:
        PrimExpr: Handle to the atomic addition operation
    """
    return T.call_extern("handle", "AtomicAdd", T.address_of(dst), value)


def atomic_addx2(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation with double-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (double-width)

    Returns:
        PrimExpr: Handle to the double-width atomic addition operation
    """
    return T.call_extern("handle", "AtomicAddx2", T.address_of(dst), T.address_of(value))


def atomic_addx4(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation with double-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (double-width)

    Returns:
        PrimExpr: Handle to the double-width atomic addition operation
    """
    return T.call_extern("handle", "AtomicAddx4", T.address_of(dst), T.address_of(value))


def dp4a(A: Buffer, B: Buffer, C: Buffer) -> PrimExpr:
    """Perform a 4-element dot product with accumulation (DP4A).

    Args:
        A (Buffer): First input buffer
        B (Buffer): Second input buffer
        C (Buffer): Accumulation buffer

    Returns:
        PrimExpr: Handle to the DP4A operation
    """
    return T.call_extern("handle", "DP4A", T.address_of(A), T.address_of(B), T.address_of(C))


def clamp(dst: PrimExpr, min_val: PrimExpr, max_val: PrimExpr) -> PrimExpr:
    """Clamps the input value dst between [min_val, max_val]

    Args:
        dst: Input value to be clamped
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Value clamped to the specified range
    """
    dst = T.max(dst, min_val)  # Ensure value is not less than minimum
    dst = T.min(dst, max_val)  # Ensure value is not greater than maximum
    return dst


def reshape(src: Buffer, shape: list[PrimExpr]) -> Buffer:
    """Reshapes the input buffer to the specified shape.

    Args:
        src (Buffer): Input buffer to be reshaped
        shape (list[PrimExpr]): New shape for the buffer

    Returns:
        Buffer: A new buffer view with the specified shape
    """
    return T.Buffer(shape, src.dtype, src.data)


def view(src: Buffer, shape: list[PrimExpr] | None = None, dtype: str | None = None) -> Buffer:
    """Views the input buffer with optionally modified shape and dtype.

    Args:
        src (Buffer): Input buffer to be viewed
        shape (list[PrimExpr] | None, optional): New shape for the buffer. Defaults to None.
        dtype (str | None = None, optional): New dtype for the buffer. Defaults to None.

    Returns:
        Buffer: A new buffer view with the specified shape and dtype
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    return T.Buffer(shape, dtype, src.data)


def npu_gemm(A, B, C, init=False, n_actual=None, unit_flag=None, k_actual=None):
    """NPU GEMM intrinsic. A, B, C can be 2D or higher-order (leading dims must be 1).

    n_actual / unit_flag (both default ``None``): optional trailing args that map
    to the C++ ``mma`` template's ``n_actual`` (runtime output-column count, <= N)
    and ``unitFlag`` (0b10 accumulate / 0b11 flush, drives the hardware mma->fixpipe
    pipeline for the cL0 ping-pong overlap). When BOTH are ``None`` the call emits
    the legacy 6-arg form (``mma<...>(A, B, C, init, K)``) so every existing caller
    is byte-for-byte unchanged (the C++ defaults are ``n_actual=N``, ``unitFlag=0``).
    Used by the kernel-driven ring-aware decomposition (Layer ③) to fuse a
    standalone ``T.mma`` with a following ``T.copy(L0C->GM, unit_flag=0b11)`` fixpipe.

    k_actual (default ``None``): runtime contraction length passed as the C++ mma's
    ``K`` arg, OVERRIDING the value derived from ``A``'s last dim. Lets the operands
    stay FULL buffers (e.g. a [G,128] L0A) while the mma contracts only ``k_actual``
    columns (= the reference's runtime ``kSize`` / ``k_actual=winm``). Needed because
    a symbolic slice operand (``p_l0a[pp,:,0:winm]``) makes ``access_ptr`` call
    ``int(winm)`` on a Var and fail; passing full operands + ``k_actual`` avoids that
    and matches ``gemm_v0_fixp`` (which loads the full tile and contracts ``kSize``).
    """

    def legalize_arguments(arg: Buffer | Var):
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (tir.Buffer | tir.Var: Input argument to legalize

        Returns:
            tir.Buffer | tir.Var: The legalized argument
        """
        if isinstance(arg, Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)

    def retrieve_shape(object: Buffer | BufferRegion) -> list[int]:
        if isinstance(object, Buffer):
            return object.shape
        elif isinstance(object, BufferRegion):
            region = object.region
            shape = []
            for r in region:
                shape.append(r.extent)
            return shape
        else:
            raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    A_shape = retrieve_shape(A)
    B_shape = retrieve_shape(B)
    C_shape = retrieve_shape(C)

    assert len(C_shape) >= 2, "current only support C as a 2D or higher-order tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(C_shape) > 2:
        for i in range(len(C_shape) - 2):
            assert C_shape[i] == 1, (
                "current only support C as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, (
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, (
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )

    M, N = C_shape[-2], C_shape[-1]
    K = A_shape[-1]
    K_B = B_shape[-2]
    assert K == K_B, f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    def retrieve_ptr(object: Buffer | BufferRegion, access_type: str = "r") -> PrimExpr:
        if isinstance(object, Buffer):
            return object.access_ptr(access_type)
        elif isinstance(object, BufferRegion):
            buffer, region = object.buffer, object.region
            indices = []
            for r in region:
                indices.append(r.min)
            strides = []
            stride = 1
            for s in reversed(buffer.shape):
                strides.insert(0, stride)
                stride *= s
            offset = 0
            for i in range(len(indices)):
                offset += indices[i] * strides[i]
            extent = [x.extent for x in object.region]
            size_extent = math.prod(extent)
            return buffer.access_ptr(access_mask=access_type, offset=offset, extent=size_extent)
        else:
            raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    Aptr = retrieve_ptr(A, "r")
    Bptr = retrieve_ptr(B, "r")
    Cptr = retrieve_ptr(C, "w" if init is True else "rw")

    # K runtime arg: k_actual overrides the shape-derived K (operands stay full;
    # the mma contracts only k_actual). The <M,N> template params are unaffected.
    K_runtime = k_actual if k_actual is not None else K
    mma_args = [f"mma<{_dtype(A)}, {_dtype(C)}, {M}, {N}>", Aptr, Bptr, Cptr, init, K_runtime]
    if n_actual is not None or unit_flag is not None:
        # positional in the C++ template: (..., K, n_actual=N, unitFlag=0). To set
        # unit_flag we must also supply n_actual, so default the unset one.
        mma_args.append(n_actual if n_actual is not None else N)
        mma_args.append(unit_flag if unit_flag is not None else 0)
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_mma"), *mma_args)


def loop_break():
    """Break out of the innermost loop."""
    return T.call_intrin("handle", op.Op.get("tl.loop_break"))  # noqa: F821
