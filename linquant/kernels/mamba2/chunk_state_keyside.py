# ruff: noqa: E501, D100, D103, D205, D417, E731, N802, N806, C901, F821
"""Key-side absorption chunk_state: in-kernel INT8 x, in-kernel FP→INT8 B'.

Flips SSDi8's value-side absorption: instead of weighting X per-tile and dotting
with pre-quantized B, this kernel quantizes FP16 x in-loop with its calibrated
per-(H, P) scale and weights B per-tile with the gate (dt * exp(A)), quantizes
the weighted B' inside the K-loop with the pre-calibrated per-(H, N) b_gate_scale,
then dots INT8 X × INT8 B' into an INT32 accumulator (single post-loop rescale by
b_gate_vec/127).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        # Keyside-specific configs: b_fp lives in FP32 inside the K-loop, so SHMEM
        # pressure is ~2× value-side at the same (M,N,K). Fewer stages let N=128
        # fit on A5000 (~99KB SHMEM).
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=2,
        ),
        # K=64 configs: halve K-iterations (256/64=4 vs 256/32=8), reducing
        # per-tile B-gating overhead at the cost of larger B tiles.
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        # Smaller tiles for dynamic paths (amax reduction adds register pressure).
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        # 1-stage configs (no software pipelining — less register pressure).
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=1,
            num_warps=4,
        ),
    ],
    key=[
        "hdim",
        "dstate",
        "chunk_size",
    ],
)
@triton.jit
def _chunk_state_keyside_kernel(
    # Pointers.
    x_ptr,  # FP16 (B, S, H, P) — quantized in-loop with x_scale
    b_ptr,  # FP16 (B, S, G, N) — unquantized B
    x_scale_ptr,  # FP32 (H, P) — x quantization scale (in-loop quant + epilogue)
    states_ptr,  # INT8 (B, C, H, P, N) — output
    out_qscale_ptr,  # FP32 (H, P) — output state scale S_out
    dt_ptr,  # FP32 (B, H, C, K)
    dA_cumsum_ptr,  # FP32 (B, H, C, K)
    b_gate_scale_ptr,  # FP32 — calibrated max(|gate*B|)
    # Sizes.
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides: x (B, S, H, P) FP16.
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    # Strides: B (B, S, G, N) FP16.
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
    # Strides: x_scale (H, P).
    stride_xs_head,
    stride_xs_hdim,
    # Strides: out_qscale (H, P).
    stride_oqs_head,
    stride_oqs_hdim,
    # Strides: b_gate_scale (B, C, H, N_or_1).
    stride_bgs_batch,
    stride_bgs_chunk,
    stride_bgs_head,
    stride_bgs_dstate,
    # Constexprs.
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    gid = pid_h // nheads_ngroups_ratio

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Base pointers.
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + gid * stride_b_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    )

    # Tile pointers: x is (M, K), B is (K, N).
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_b_seqlen + offs_n[None, :] * stride_b_dstate)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # x_scale (H, P) for the in-loop quantization and the epilogue.
    x_scale_m = tl.load(
        x_scale_ptr + pid_h * stride_xs_head + offs_m * stride_xs_hdim,
        mask=offs_m < hdim,
        other=1.0,
    ).to(tl.float32)
    x_inv_scale_m = 1.0 / tl.maximum(x_scale_m, 1e-10)

    # Load pre-calibrated b_gate_scale along N — per-(H, N) when
    # stride_bgs_dstate>0, per-H when stride_bgs_dstate=0 (single scalar broadcast
    # across all N lanes).
    _bgs_base = (
        b_gate_scale_ptr
        + pid_b * stride_bgs_batch
        + pid_c * stride_bgs_chunk
        + pid_h * stride_bgs_head
    )
    b_gate_vec = tl.load(
        _bgs_base + offs_n * stride_bgs_dstate,
        mask=offs_n < dstate,
        other=1.0,
    ).to(tl.float32)
    b_inv_vec = 127.0 / b_gate_vec  # (BLOCK_SIZE_N,)

    # ---- K-loop: quantize x, weight FP B with gate, quantize B', dot ----
    # INT32 accumulator (B' lives on a single per-N scale across K).
    acc_i32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        k_mask = offs_k < chunk_size_limit - k
        x_fp = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & k_mask[None, :], other=0.0).to(
            tl.float32
        )
        x_q = x_fp * x_inv_scale_m[:, None]
        x_q = tl.extra.cuda.libdevice.rint(x_q)
        x_i8 = tl.maximum(tl.minimum(x_q, 127.0), -127.0).to(tl.int8)
        b_raw = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < dstate), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        gate_k = tl.exp(dA_cs_last - dA_cs_k) * dt_k

        # gate*B in FP32 (single post-loop rescale).
        gated_b = b_raw.to(tl.float32) * gate_k[:, None]
        b_q_fp = gated_b * b_inv_vec[None, :]

        b_q = tl.extra.cuda.libdevice.rint(b_q_fp)
        b_q = tl.maximum(tl.minimum(b_q, 127.0), -127.0).to(tl.int8)
        sub_i32 = tl.dot(x_i8, b_q)
        acc_i32 += sub_i32

        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # Single post-loop rescale: acc * (b_gate_vec / 127) — per-N broadcast.
    acc_fp = tl.cast(acc_i32, tl.float32) * (b_gate_vec / 127.0)[None, :]

    # ---- Epilogue: requant acc * (x_scale / S_out) → INT8 states.
    S_out = tl.load(
        out_qscale_ptr + pid_h * stride_oqs_head + offs_m * stride_oqs_hdim,
        mask=offs_m < hdim,
        other=1.0,
    ).to(tl.float32)
    req_scale = (x_scale_m / S_out)[:, None]

    acc_fp *= req_scale
    q = tl.extra.cuda.libdevice.rint(acc_fp)
    q = tl.maximum(tl.minimum(q, 127.0), -127.0).to(tl.int8)

    states_ptr += (
        pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    )
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, q, mask=c_mask)


def _bgs_strides(t):
    """Return (batch, chunk, head, dstate) strides for b_gate_scale tensor.

    Zeroes out strides for size-1 dims so the kernel broadcasts correctly.
    """
    assert t.dim() == 4, f"b_gate_scale must be 4D (B,C,H,N), got {t.dim()}D"
    return tuple(t.stride(i) if t.size(i) > 1 else 0 for i in range(4))


def _chunk_state_keyside_fwd(
    x_fp,
    x_scale_hp,
    b_fp,
    dt,
    dA_cumsum,
    out_qscale,
    b_gate_scale,
):
    """Key-side absorption chunk_state: in-kernel INT8 x, FP→INT8 B'.

    Args:
        x_fp: (batch, seqlen, nheads, headdim) FP16 — quantized to INT8 inside the
            kernel with x_scale_hp.
        x_scale_hp: (nheads, headdim) FP32 — per-(H, P) scale used to quantize x.
        b_fp: (batch, seqlen, ngroups, dstate) FP16 — unquantized B.
        dt: (batch, nheads, nchunks, chunk_size) FP32.
        dA_cumsum: same shape as dt.
        out_qscale: (nheads, headdim) FP32 — per-(H, P) output state scale.
        b_gate_scale: (B, C, H, N_or_1) or broadcastable 4D FP32 — calibrated
            max(|gate*B|).

    Returns:
        states_int8: (batch, nchunks, nheads, headdim, dstate) INT8.
    """
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = b_fp.shape
    batch, seqlen, nheads, headdim = x_fp.shape

    states_int8 = torch.empty(
        (batch, nchunks, nheads, headdim, dstate), device=x_fp.device, dtype=torch.int8
    )
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"]) * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    bgs_strides = _bgs_strides(b_gate_scale)

    with torch.cuda.device(x_fp.device.index):
        _chunk_state_keyside_kernel[grid](
            x_fp,
            b_fp,
            x_scale_hp,
            states_int8,
            out_qscale,
            dt,
            dA_cumsum,
            b_gate_scale,
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
            b_fp.stride(0),
            b_fp.stride(1),
            b_fp.stride(2),
            b_fp.stride(-1),
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
            x_scale_hp.stride(0),
            x_scale_hp.stride(1),
            out_qscale.stride(0),
            out_qscale.stride(1),
            *bgs_strides,
        )
    return states_int8
