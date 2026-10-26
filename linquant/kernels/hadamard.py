# Copyright (c) 2026, Philipp Nazari

"""Kronecker-factored Hadamard rotation of the state axis (R rotation).

Each head's rotation is R_h = diag(s_h) · H where H is the normalized Hadamard
matrix and s_h ∈ {-1, +1}^D is a per-head random sign vector:

    y = x · diag(s) · H  →  sign flip, then Hadamard

The Hadamard is applied as H_2 ⊗ H_{D/2}: a prebuilt H_{D/2} block (bf16, cached per
device) is applied to each half with a tensor-core dot, and the H_2 step is the
butterfly (w0 ± w1) / sqrt(2). This file holds the standalone rotation kernel used
for the key-side FP16 B path; the value-side B/C rotation is fused into
``linquant.kernels.mamba2.fused_split_rotate_xBC``, which reuses
``_resolve_kronecker_params`` from here.
"""

from __future__ import annotations

import inspect
import math
import os

import torch
import triton
import triton.language as tl

# Autotune settings inlined from fla.utils so the kernel does not depend on the
# fla package layout.
IS_AMD = triton.runtime.driver.active.get_current_target().backend == "hip"
_FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"
_SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": _FLA_CACHE_RESULTS} if _SUPPORTS_AUTOTUNE_CACHE else {}

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=nw) for bt in [32, 64] for nw in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _kron2_hadamard_kernel(
    x_ptr,
    y_ptr,
    signs_ptr,
    h_block_ptr,
    T,
    D: tl.constexpr,
    BS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    IN_STRIDE_T: tl.constexpr,
    OUT_STRIDE_T: tl.constexpr,
    BT: tl.constexpr,
    RSQRT_D: tl.constexpr,
):
    """Kronecker H_2 ⊗ H_{D/2} rotation (signs first, then Hadamard).

    Stride-aware: IN_STRIDE_T and OUT_STRIDE_T are the per-token strides along
    the T axis for x and y respectively. For a contiguous (B, T, H, D) tensor
    both equal H*D; for a sliced view like xBC[..., :d_ssm] the input stride is
    the parent's T-stride (xBC_dim).
    """
    i_chunk, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % NUM_HEADS
    in_base = (i_bh // NUM_HEADS) * T * IN_STRIDE_T + i_h * D
    out_base = (i_bh // NUM_HEADS) * T * OUT_STRIDE_T + i_h * D

    p_H = tl.make_block_ptr(h_block_ptr, (BS, BS), (BS, 1), (0, 0), (BS, BS), (1, 0))
    b_H = tl.load(p_H)

    p0 = tl.make_block_ptr(
        x_ptr + in_base, (T, BS), (IN_STRIDE_T, 1), (i_chunk * BT, 0), (BT, BS), (1, 0)
    )
    p1 = tl.make_block_ptr(
        x_ptr + in_base + BS, (T, BS), (IN_STRIDE_T, 1), (i_chunk * BT, 0), (BT, BS), (1, 0)
    )
    b0 = tl.load(p0, boundary_check=(0,)).to(tl.float32)
    b1 = tl.load(p1, boundary_check=(0,)).to(tl.float32)

    s0 = tl.load(signs_ptr + i_h * D + tl.arange(0, BS)).to(tl.float32)
    s1 = tl.load(signs_ptr + i_h * D + BS + tl.arange(0, BS)).to(tl.float32)

    b0 = b0 * s0[None, :]
    b1 = b1 * s1[None, :]

    w0 = tl.dot(b0.to(b_H.dtype), b_H).to(tl.float32)
    w1 = tl.dot(b1.to(b_H.dtype), b_H).to(tl.float32)

    y0 = (w0 + w1) * RSQRT_D
    y1 = (w0 - w1) * RSQRT_D

    q0 = tl.make_block_ptr(
        y_ptr + out_base, (T, BS), (OUT_STRIDE_T, 1), (i_chunk * BT, 0), (BT, BS), (1, 0)
    )
    q1 = tl.make_block_ptr(
        y_ptr + out_base + BS, (T, BS), (OUT_STRIDE_T, 1), (i_chunk * BT, 0), (BT, BS), (1, 0)
    )
    tl.store(q0, y0.to(q0.dtype.element_ty), boundary_check=(0,))
    tl.store(q1, y1.to(q1.dtype.element_ty), boundary_check=(0,))


def hadamard_matrix(d: int, device=None, dtype=None) -> torch.Tensor:
    """Construct a normalized Hadamard matrix of size d (must be power of 2)."""
    if d & (d - 1) != 0:
        raise ValueError(f"Hadamard rotation requires d to be a power of 2, got {d}")
    H = torch.ones(1, 1, device=device, dtype=torch.float32)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / math.sqrt(d)
    if dtype is not None:
        H = H.to(dtype)
    return H


# Module-level cache for prebuilt H_{BS} matrices (one per block size per device)
_h_block_cache: dict[tuple[int, int], torch.Tensor] = {}


def _get_h_block(bs: int, device: torch.device) -> torch.Tensor:
    """Get or create the prebuilt H_{BS} matrix (head-independent)."""
    key = (bs, device.index if device.index is not None else 0)
    if key not in _h_block_cache:
        _h_block_cache[key] = hadamard_matrix(bs, device=device, dtype=torch.bfloat16)
    return _h_block_cache[key]


def _resolve_kronecker_params(D, n_chunks, device):
    """Prebuilt H_{D/n_chunks} block and the H_2-step scale for a Kronecker rotation.

    Prebuilt H_{BS} has entries ±1/√BS → only 1/√n_chunks is needed for the H_2 step.
    """
    h_block = _get_h_block(D // n_chunks, device)
    kron_scale = 1.0 / math.sqrt(n_chunks)
    return h_block, kron_scale


def kronecker_hadamard_fwd(x, signs, num_heads, n_chunks=2):
    """Kronecker-factored Hadamard rotation: returns (x * signs) @ H per head.

    Args:
        x: (B, T, H, D) FP16/BF16, D a power of two ≥ 64 (any T-stride, e.g. a view of xBC).
        signs: (H, D) ±1 per-head sign vectors.
        num_heads: H (kept for call-site readability; the shape is authoritative).
        n_chunks: Kronecker split (H_2 ⊗ H_{D/2} for 2).
    """
    x_shape = x.shape
    B, T, H, D = x_shape
    assert D >= 64, f"Kronecker Hadamard needs D >= 64, got {D}"

    h_block, kron_scale = _resolve_kronecker_params(D, n_chunks, x.device)
    y = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), B * H)

    # Use the input's actual T-stride: a sliced view (e.g. xBC[..., d_ssm:d_ssm+G*N])
    # has stride(1) = parent xBC_dim, which differs from H*D. `empty_like` returns a
    # contiguous output, so the output T-stride is always H*D.
    in_stride_t = x.stride(1)
    out_stride_t = y.stride(1)

    _kron2_hadamard_kernel[grid](
        x,
        y,
        signs,
        h_block,
        T=T,
        D=D,
        BS=D // n_chunks,
        NUM_HEADS=H,
        IN_STRIDE_T=in_stride_t,
        OUT_STRIDE_T=out_stride_t,
        RSQRT_D=kron_scale,
    )
    return y.view(x_shape)
