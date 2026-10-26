# ruff: noqa: N999, N803, N802
"""Fused split + Kronecker-Hadamard R rotation + INT8 quant for B and C.

Reads the B and C lanes of the post-conv1d xBC slab once and emits, in a single launch:
  - q_B:     (B, T, G, N) int8 — (D⁻¹-scaled,) rotated + quantized with B_inv_scale
  - q_C:     (B, T, G, N) int8 — (D-scaled,) rotated + quantized with C_inv_scale

x is never copied: the caller gets it as a view of xBC's first d_ssm lanes.

Replaces the sequence:
  rot_B  → kronecker_hadamard_fwd(B, r_signs, G)
  rot_C  → kronecker_hadamard_fwd(C, r_signs, G)
  + Python clamp/round to INT8 for q_B, q_C.

xBC layout (last axis): [x (d_ssm = H*P), B (G*N), C (G*N)].

HAS_R_SIGNS=False ⇒ B/C branches do split + INT8 only, no rotation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from linquant.kernels.hadamard import _resolve_kronecker_params

NUM_WARPS = [2, 4, 8]


@triton.autotune(
    configs=[triton.Config({"BT": bt}, num_warps=nw) for bt in [16, 32, 64] for nw in NUM_WARPS],
    key=[
        "BC_D",
        "HAS_R_SIGNS",
    ],
)
@triton.jit(do_not_specialize=["T"])
def _fused_split_rotate_xBC_kernel(  # noqa: C901
    # Input: (B, T, xBC_dim) fp16
    xBC_ptr,
    # Outputs
    qB_out_ptr,  # int8 (B, T, G, N)
    qC_out_ptr,  # int8 (B, T, G, N)
    # Signs and H block
    bc_signs_ptr,
    bc_h_block_ptr,
    # Scales
    B_inv_scale_ptr,  # (G,) fp32
    C_inv_scale_ptr,  # (G,) fp32
    dqk_b_ptr,  # (G, N) fp32 — diagonal key scaling on B (HAS_DQK only), applied before the rotation
    dqk_c_ptr,  # (G, N) fp32 — diagonal key scaling on C (HAS_DQK only)
    # Dimensions
    T,
    BC_D: tl.constexpr,  # d_state (e.g. 128)
    BC_BS: tl.constexpr,  # BC_D // 2
    BC_NGROUPS: tl.constexpr,
    # Strides
    XBC_STRIDE_T: tl.constexpr,  # T-stride in xBC: d_ssm + 2*G*N
    BC_STRIDE_T: tl.constexpr,  # G*N (output)
    # Offsets (along last dim)
    B_OFFSET: tl.constexpr,  # d_ssm
    C_OFFSET: tl.constexpr,  # d_ssm + G*N
    # Kronecker normalization scale (1/sqrt(n_chunks) for prebuilt)
    BC_KRON_SCALE: tl.constexpr,
    # Constexpr toggles
    HAS_R_SIGNS: tl.constexpr,  # rotate B/C branches when True; else pure split + INT8
    HAS_DQK: tl.constexpr,
    # Block size along T
    BT: tl.constexpr,
):
    """One program handles one group of B or C for one chunk of T."""
    i_chunk = tl.program_id(0)
    i_bh = tl.program_id(1)

    total_per_batch = BC_NGROUPS + BC_NGROUPS
    i_batch = i_bh // total_per_batch
    i_local = i_bh % total_per_batch

    if i_local < BC_NGROUPS:
        # ---- B branch: (optional) rotate FP, then emit INT8 (vars suffixed _b) ----
        i_g_b = i_local
        in_base_b = i_batch * T * XBC_STRIDE_T + B_OFFSET + i_g_b * BC_D
        out_base_b = i_batch * T * BC_STRIDE_T + i_g_b * BC_D
        inv_scale_b = tl.load(B_inv_scale_ptr + i_g_b).to(tl.float32)

        p0_b = tl.make_block_ptr(
            xBC_ptr + in_base_b,
            (T, BC_BS),
            (XBC_STRIDE_T, 1),
            (i_chunk * BT, 0),
            (BT, BC_BS),
            (1, 0),
        )
        p1_b = tl.make_block_ptr(
            xBC_ptr + in_base_b,
            (T, BC_BS),
            (XBC_STRIDE_T, 1),
            (i_chunk * BT, BC_BS),
            (BT, BC_BS),
            (1, 0),
        )
        b0_b = tl.load(p0_b, boundary_check=(0,)).to(tl.float32)
        b1_b = tl.load(p1_b, boundary_check=(0,)).to(tl.float32)
        if HAS_DQK:
            d0_b = tl.load(dqk_b_ptr + i_g_b * BC_D + tl.arange(0, BC_BS)).to(tl.float32)
            d1_b = tl.load(dqk_b_ptr + i_g_b * BC_D + BC_BS + tl.arange(0, BC_BS)).to(tl.float32)
            b0_b = (b0_b * d0_b[None, :]).to(tl.float16).to(tl.float32)
            b1_b = (b1_b * d1_b[None, :]).to(tl.float16).to(tl.float32)
        if HAS_R_SIGNS:
            signs_base_b = bc_signs_ptr + i_g_b * BC_D
            p_H_b = tl.make_block_ptr(
                bc_h_block_ptr, (BC_BS, BC_BS), (BC_BS, 1), (0, 0), (BC_BS, BC_BS), (1, 0)
            )
            s0_b = tl.load(signs_base_b + tl.arange(0, BC_BS)).to(tl.float32)
            s1_b = tl.load(signs_base_b + BC_BS + tl.arange(0, BC_BS)).to(tl.float32)
            bH_b = tl.load(p_H_b)
            w0_b = tl.dot((b0_b * s0_b[None, :]).to(bH_b.dtype), bH_b).to(tl.float32)
            w1_b = tl.dot((b1_b * s1_b[None, :]).to(bH_b.dtype), bH_b).to(tl.float32)
            # Round-trip via FP16 to match the unfused reference, which stores the
            # rotated B as FP16 (kronecker_hadamard_fwd output) and reloads as FP32
            # before the INT8 multiply. Skipping this drops a few INT8 boundary
            # roundings.
            y0_b = ((w0_b + w1_b) * BC_KRON_SCALE).to(tl.float16).to(tl.float32)
            y1_b = ((w0_b - w1_b) * BC_KRON_SCALE).to(tl.float16).to(tl.float32)
        else:
            # No rotation: identity round-trip via fp16 (matches unfused reference
            # which loads fp16 B from xBC and casts to fp32 before the INT8 mul).
            y0_b = b0_b.to(tl.float16).to(tl.float32)
            y1_b = b1_b.to(tl.float16).to(tl.float32)

        # q_B: per-tensor / per-G scale
        q0_b = tl.maximum(
            tl.minimum(tl.extra.cuda.libdevice.rint(y0_b * inv_scale_b), 127.0), -128.0
        )
        q1_b = tl.maximum(
            tl.minimum(tl.extra.cuda.libdevice.rint(y1_b * inv_scale_b), 127.0), -128.0
        )
        oB0 = tl.make_block_ptr(
            qB_out_ptr + out_base_b,
            (T, BC_BS),
            (BC_STRIDE_T, 1),
            (i_chunk * BT, 0),
            (BT, BC_BS),
            (1, 0),
        )
        oB1 = tl.make_block_ptr(
            qB_out_ptr + out_base_b,
            (T, BC_BS),
            (BC_STRIDE_T, 1),
            (i_chunk * BT, BC_BS),
            (BT, BC_BS),
            (1, 0),
        )
        tl.store(oB0, q0_b.to(tl.int8), boundary_check=(0,))
        tl.store(oB1, q1_b.to(tl.int8), boundary_check=(0,))

    else:
        # ---- C branch: (optional) rotate FP, single INT8 output (vars suffixed _c) ----
        i_g_c = i_local - BC_NGROUPS
        in_base_c = i_batch * T * XBC_STRIDE_T + C_OFFSET + i_g_c * BC_D
        out_base_c = i_batch * T * BC_STRIDE_T + i_g_c * BC_D
        inv_scale_c = tl.load(C_inv_scale_ptr + i_g_c).to(tl.float32)

        p0_c = tl.make_block_ptr(
            xBC_ptr + in_base_c,
            (T, BC_BS),
            (XBC_STRIDE_T, 1),
            (i_chunk * BT, 0),
            (BT, BC_BS),
            (1, 0),
        )
        p1_c = tl.make_block_ptr(
            xBC_ptr + in_base_c,
            (T, BC_BS),
            (XBC_STRIDE_T, 1),
            (i_chunk * BT, BC_BS),
            (BT, BC_BS),
            (1, 0),
        )
        b0_c = tl.load(p0_c, boundary_check=(0,)).to(tl.float32)
        b1_c = tl.load(p1_c, boundary_check=(0,)).to(tl.float32)
        if HAS_DQK:
            d0_c = tl.load(dqk_c_ptr + i_g_c * BC_D + tl.arange(0, BC_BS)).to(tl.float32)
            d1_c = tl.load(dqk_c_ptr + i_g_c * BC_D + BC_BS + tl.arange(0, BC_BS)).to(tl.float32)
            b0_c = (b0_c * d0_c[None, :]).to(tl.float16).to(tl.float32)
            b1_c = (b1_c * d1_c[None, :]).to(tl.float16).to(tl.float32)
        if HAS_R_SIGNS:
            signs_base_c = bc_signs_ptr + i_g_c * BC_D
            p_H_c = tl.make_block_ptr(
                bc_h_block_ptr, (BC_BS, BC_BS), (BC_BS, 1), (0, 0), (BC_BS, BC_BS), (1, 0)
            )
            s0_c = tl.load(signs_base_c + tl.arange(0, BC_BS)).to(tl.float32)
            s1_c = tl.load(signs_base_c + BC_BS + tl.arange(0, BC_BS)).to(tl.float32)
            bH_c = tl.load(p_H_c)
            w0_c = tl.dot((b0_c * s0_c[None, :]).to(bH_c.dtype), bH_c).to(tl.float32)
            w1_c = tl.dot((b1_c * s1_c[None, :]).to(bH_c.dtype), bH_c).to(tl.float32)
            # FP16 round-trip — match the unfused reference (see B branch above).
            y0_c = ((w0_c + w1_c) * BC_KRON_SCALE).to(tl.float16).to(tl.float32)
            y1_c = ((w0_c - w1_c) * BC_KRON_SCALE).to(tl.float16).to(tl.float32)
        else:
            y0_c = b0_c.to(tl.float16).to(tl.float32)
            y1_c = b1_c.to(tl.float16).to(tl.float32)
        q0_c = tl.maximum(
            tl.minimum(tl.extra.cuda.libdevice.rint(y0_c * inv_scale_c), 127.0), -128.0
        )
        q1_c = tl.maximum(
            tl.minimum(tl.extra.cuda.libdevice.rint(y1_c * inv_scale_c), 127.0), -128.0
        )
        oC0 = tl.make_block_ptr(
            qC_out_ptr + out_base_c,
            (T, BC_BS),
            (BC_STRIDE_T, 1),
            (i_chunk * BT, 0),
            (BT, BC_BS),
            (1, 0),
        )
        oC1 = tl.make_block_ptr(
            qC_out_ptr + out_base_c,
            (T, BC_BS),
            (BC_STRIDE_T, 1),
            (i_chunk * BT, BC_BS),
            (BT, BC_BS),
            (1, 0),
        )
        tl.store(oC0, q0_c.to(tl.int8), boundary_check=(0,))
        tl.store(oC1, q1_c.to(tl.int8), boundary_check=(0,))


def prepare_dqk(d: torch.Tensor) -> torch.Tensor:
    """Diagonal key scaling vector as the kernel wants it: fp16-rounded values in a contiguous fp32 tensor
    (rounding first keeps the result equal to the former in-place fp16 multiply on xBC)."""
    return d.reshape(-1).to(torch.float16).to(torch.float32).contiguous()


def fused_split_rotate_quant_xBC(
    xBC: torch.Tensor,
    r_signs: torch.Tensor | None,
    nheads: int,
    ngroups: int,
    d_state: int,
    headdim: int,
    B_inv_scale: torch.Tensor,
    C_inv_scale: torch.Tensor,
    dqk_b: torch.Tensor | None = None,
    dqk_c: torch.Tensor | None = None,
):
    """Fused split + (optional) Kronecker R rotation + INT8 quant for B, C; x as a view.

    r_signs=None ⇒ no R rotation on B/C (B/C are split + INT8 only).

    Args:
        xBC: (B, T, d_ssm + 2*G*N) fp16 — post-conv1d+silu activations.
        r_signs: (G, N) ±1 — Kronecker-Hadamard signs for B/C's R rotation, or None.
        nheads, ngroups, d_state, headdim: dimensions.
        B_inv_scale: (G,) fp32 — INT8 scale for q_B.
        C_inv_scale: (G,) fp32 — INT8 scale for q_C.
        dqk_b, dqk_c: (G*N,) or (G, N) — diagonal key scaling D⁻¹ on B and D on C, applied to the
            conv+SiLU outputs before the rotation (exact: C·D and B·D⁻¹ leave the SSD unchanged). None = off.
            Rounded to fp16 first, so the result equals the former in-place fp16 multiply on xBC bit for bit.

    Returns:
        x_fp:  (B, T, H, P) fp16 — view of xBC's first d_ssm lanes (no copy)
        q_B:   (B, T, G, N) int8 — (R-rotated if r_signs) + quantized
        q_C:   (B, T, G, N) int8 — (R-rotated if r_signs) + quantized
    """
    batch, seq_len, xBC_dim = xBC.shape
    d_ssm = nheads * headdim
    g_n = ngroups * d_state
    assert xBC_dim == d_ssm + 2 * g_n, f"xBC_dim={xBC_dim} != {d_ssm + 2 * g_n}"

    has_r_signs = r_signs is not None
    # Block size for the make_block_ptr calls (used with or without rotation).
    bc_n_chunks = 4 if d_state >= 256 else 2
    bc_BS = d_state // bc_n_chunks

    if has_r_signs:
        bc_h_block, bc_kron_scale = _resolve_kronecker_params(d_state, bc_n_chunks, xBC.device)
        bc_signs_arg = r_signs
    else:
        # Dummy ptrs (kernel never reads under HAS_R_SIGNS=False). Reuse xBC.
        bc_h_block = xBC
        bc_signs_arg = xBC
        bc_kron_scale = 1.0

    x_out = xBC[..., :d_ssm].view(batch, seq_len, nheads, headdim)

    has_dqk = dqk_b is not None
    if has_dqk:
        # callers pass tensors already rounded to fp16 and stored as contiguous fp32 (prepare_dqk); the
        # conversions are not repeated per call because at B=1 the layer is launch-bound (each op ~10 µs)
        dqk_b_arg = dqk_b if dqk_b.dtype == torch.float32 and dqk_b.is_contiguous() else prepare_dqk(dqk_b)
        dqk_c_arg = dqk_c if dqk_c.dtype == torch.float32 and dqk_c.is_contiguous() else prepare_dqk(dqk_c)
    else:
        dqk_b_arg = xBC
        dqk_c_arg = xBC

    q_B = torch.empty(batch, seq_len, ngroups, d_state, device=xBC.device, dtype=torch.int8)
    q_C = torch.empty(batch, seq_len, ngroups, d_state, device=xBC.device, dtype=torch.int8)

    def grid(meta):
        return (
            triton.cdiv(seq_len, meta["BT"]),
            batch * 2 * ngroups,
        )

    _fused_split_rotate_xBC_kernel[grid](
        xBC,
        q_B,
        q_C,
        bc_signs_arg,
        bc_h_block,
        B_inv_scale,
        C_inv_scale,
        dqk_b_arg,
        dqk_c_arg,
        T=seq_len,
        BC_D=d_state,
        BC_BS=bc_BS,
        BC_NGROUPS=ngroups,
        XBC_STRIDE_T=xBC_dim,
        BC_STRIDE_T=ngroups * d_state,
        B_OFFSET=d_ssm,
        C_OFFSET=d_ssm + g_n,
        BC_KRON_SCALE=bc_kron_scale,
        HAS_R_SIGNS=has_r_signs,
        HAS_DQK=has_dqk,
    )
    return x_out, q_B, q_C
