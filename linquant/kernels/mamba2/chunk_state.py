# ruff: noqa: E501, D100, D103, D205, D417, E731, N802, N806, C901, F821
"""Mamba-2 value-side chunk_state: INT8 matmul + static per-(H,P) state quant.

``_quant_chunk_state_fwd_xfp_static_int8`` consumes FP x, quantizes it per tile
with the calibrated per-(H,P) scale ``alpha_hp`` (SSDi8's value-side absorption),
and writes INT8 states directly with the pre-calibrated per-(H,P) state scale and
per-G B-scale, eliminating the intermediate FP32 states tensor.
Ruff suppressions match other Triton kernels in this tree.
"""

import torch
import triton
import triton.language as tl

# =============================================================================
# Fused chunk-state + static per-(H,P) INT8 quantization (SSDi8-aligned).
#
# Single canonical kernel: ``_quant_chunk_state_fwd_xfp_static_int8_kernel``
# takes FP x, single-pass K-loop with calibrated per-P ``inv_alpha``.
# Epilogue: 2D requant ``(alpha / S_out)[:, None] * bscale_gn[None, :]``.
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2
        ),
    ],
    key=[
        "hdim",
        "dstate",
        "chunk_size",
        "ALPHA_PER_HP",
        "OUT_QSCALE_PER_HP",
    ],
)
@triton.jit
def _quant_chunk_state_fwd_xfp_static_int8_kernel(
    # Pointers.
    x_ptr,
    b_ptr,
    alpha_hp_ptr,
    bscale_g_ptr,
    states_ptr,
    out_qscale_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    # Sizes.
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides: x (B, S, H, P).
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    # Strides: B (B, S, G, N) INT8.
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    # Strides: states (B, C, H, P, N) INT8 output.
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    # Strides: dt (B, H, C, K).
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    # Strides: dA_cumsum (B, H, C, K).
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    # Strides: alpha_hp (H, P).
    stride_alpha_head,
    stride_alpha_hdim,
    # Strides: bscale_g (G,) per-group scalar (matches SSDi8).
    stride_bscale_group,
    # Strides: out_qscale (H, P).
    stride_oqs_head,
    stride_oqs_hdim,
    # Constexprs.
    ALPHA_PER_HP: tl.constexpr,
    OUT_QSCALE_PER_HP: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """SSDi8-style single-pass weighted-before-quant chunk state.

    Single K-loop: per-P calibrated ``inv_alpha`` scales ``x_fp`` before gating
    and INT8 quantization.  INT8xINT8->INT32 dot.  Epilogue applies requant
    ``(alpha / S_out)[:, None] * bscale_g`` (per-G scalar, matching SSDi8).

    Granularity constexprs:
        ALPHA_PER_HP: True ⇒ alpha_hp is (H, P), per-(H,P) vector load over P.
                      False ⇒ alpha_hp's "P" stride is 0 (scalar within each pid_h).
        OUT_QSCALE_PER_HP: same convention for out_qscale (S_out).

    The non-PER_HP path covers both per-tensor (stride_head=0) and per-head
    (stride_head≠0) — the only difference is the head-axis offset, which is
    just `pid_h * stride_head` in both cases (zero for per-tensor).
    """
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    gid = pid_h // nheads_ngroups_ratio

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # P
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # N
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # K

    # Base pointers.
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + gid * stride_b_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    )

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # Per-P row scale: inv_alpha = 1 / alpha_hp.  Loaded once, reused every K tile.
    if ALPHA_PER_HP:
        alpha_m = tl.load(
            alpha_hp_ptr + pid_h * stride_alpha_head + offs_m * stride_alpha_hdim,
            mask=offs_m < hdim,
            other=1.0,
        ).to(tl.float32)
        inv_alpha_m = 1.0 / alpha_m
    else:
        # Single scalar load (covers per-tensor stride_head=0 and per-head stride_head≠0).
        alpha_scalar = tl.load(alpha_hp_ptr + pid_h * stride_alpha_head).to(tl.float32)
        inv_alpha_scalar = 1.0 / alpha_scalar

    # Per-G scalar B scale (matches SSDi8 quant_chunk_state.py:62).
    bscale_g = tl.load(bscale_g_ptr + gid * stride_bscale_group).to(tl.float32)

    # ---- Single-pass K-loop: inv_alpha quant → INT8×INT8 → INT32 accum.
    acc_i32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        k_mask = offs_k < chunk_size_limit - k

        # Load FP x tile (M, K).
        x_fp = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & k_mask[None, :], other=0.0).to(
            tl.float32
        )

        # Load INT8 B tile (K, N).
        b_i8 = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < dstate), other=0).to(
            tl.int8
        )

        # Gate = exp(dA_last - dA_k) * dt_k.
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        gate_k = tl.exp(dA_cs_last - dA_cs_k) * dt_k

        # Quantize: x_fp * inv_alpha[:, None] * gate_k[None, :] → round → INT8.
        if ALPHA_PER_HP:
            x_fp *= inv_alpha_m[:, None]
        else:
            x_fp *= inv_alpha_scalar
        x_fp *= gate_k[None, :]
        x_q = tl.where(x_fp >= 0, tl.floor(x_fp + 0.5), tl.ceil(x_fp - 0.5))
        x_q = tl.maximum(tl.minimum(x_q, 127), -127).to(tl.int8)
        acc_i32 += tl.dot(x_q, b_i8, out_dtype=tl.int32)

        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # ---- Epilogue: requant  acc * (alpha / S_out)[:, None] * bscale_g → INT8 (matches SSDi8).
    if OUT_QSCALE_PER_HP:
        S_out = tl.load(
            out_qscale_ptr + pid_h * stride_oqs_head + offs_m * stride_oqs_hdim,
            mask=offs_m < hdim,
            other=1.0,
        ).to(tl.float32)
        if ALPHA_PER_HP:
            req_scale = (alpha_m / S_out)[:, None] * bscale_g
        else:
            req_scale = (alpha_scalar / S_out)[:, None] * bscale_g
    else:
        S_out_scalar = tl.load(out_qscale_ptr + pid_h * stride_oqs_head).to(tl.float32)
        if ALPHA_PER_HP:
            req_scale = (alpha_m / S_out_scalar)[:, None] * bscale_g
        else:
            req_scale = (alpha_scalar / S_out_scalar) * bscale_g
    acc_fp = acc_i32.to(tl.float32) * req_scale

    q = tl.where(acc_fp >= 0, tl.floor(acc_fp + 0.5), tl.ceil(acc_fp - 0.5))
    q = tl.maximum(tl.minimum(q, 127), -127).to(tl.int8)

    states_ptr += (
        pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    )
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, q, mask=c_mask)


def _quant_chunk_state_fwd_xfp_static_int8(
    x_fp,
    q_B,
    dt,
    dA_cumsum,
    alpha_hp,
    bscale_g,
    out_qscale,
):
    """SSDi8-style single-pass chunk state with calibrated requant.

    Args:
        x_fp: (batch, seqlen, nheads, headdim) FP16/BF16/FP32 x activations.
        q_B: (batch, seqlen, ngroups, dstate) INT8 B.
        dt: (batch, nheads, nchunks, chunk_size).
        dA_cumsum: same shape as dt.
        alpha_hp: (nheads, headdim) calibrated per-(H,P) row scale for x.
        bscale_g: (ngroups,) per-G column scale for B (matches SSDi8).
        out_qscale: (nheads, headdim) S_out — output state scale for requant.

    Returns:
        states_int8: (batch, nchunks, nheads, headdim, dstate) INT8.
    """
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = x_fp.shape
    # Caller contract (the canonical SSD orchestrator):
    #   - x_fp: CUDA, FP16/BF16/FP32, contiguous
    #   - alpha_hp: (H, P) FP32 on device
    #   - bscale_g: (G,) FP32 on device
    #   - out_qscale: (H, P) FP32 on device
    # The per-call asserts that previously verified all of this were dead
    # host-side dispatch on the hot path. If a non-canonical caller violates
    # the contract the kernel will fault loudly.

    states_int8 = torch.empty(
        (batch, nchunks, nheads, headdim, dstate), device=x_fp.device, dtype=torch.int8
    )
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"]) * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    # Granularity detection: P-axis stride 0 ⇒ scalar-per-program (per-tensor or per-head).
    alpha_per_hp = alpha_hp.stride(1) != 0
    out_qscale_per_hp = out_qscale.stride(1) != 0
    with torch.cuda.device(x_fp.device.index):
        _quant_chunk_state_fwd_xfp_static_int8_kernel[grid](
            x_fp,
            q_B,
            alpha_hp,
            bscale_g,
            states_int8,
            out_qscale,
            dt,
            dA_cumsum,
            headdim,
            dstate,
            chunk_size,
            batch,
            seqlen,
            nheads // ngroups,
            x_fp.stride(0),
            x_fp.stride(1),
            x_fp.stride(2),
            x_fp.stride(3),
            q_B.stride(0),
            q_B.stride(1),
            q_B.stride(2),
            q_B.stride(-1),
            states_int8.stride(0),
            states_int8.stride(1),
            states_int8.stride(2),
            states_int8.stride(3),
            states_int8.stride(4),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            alpha_hp.stride(0),
            alpha_hp.stride(1),
            bscale_g.stride(0),
            out_qscale.stride(0),
            out_qscale.stride(1),
            ALPHA_PER_HP=alpha_per_hp,
            OUT_QSCALE_PER_HP=out_qscale_per_hp,
        )
    return states_int8
