# ruff: noqa: E501, E731, C901, D205
"""chunk_scan kernel — SSDi8 base.

Copied from SSDi8's quant_chunk_scan.py (their faster kernel): INT8 C × INT8
passed-state dot with per-G/per-(H,P) dequant, INT8 CB (dequantized, gated) ×
FP16 x conv path, FP D residual. Same math, better occupancy.
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
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4
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
    ],
    key=["chunk_size", "hdim", "dstate"],
)
@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers
    cb_ptr,
    x_ptr,
    out_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    C_ptr,
    C_scale,
    prev_states_ptr,
    states_scale,
    D_ptr,
    D_scale,
    CB_scale_ptr,
    # Dims
    chunk_size,
    hdim,
    dstate,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_cb_batch,
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_head,
    stride_out_hdim,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_C_batch,
    stride_C_seqlen,
    stride_C_head,
    stride_C_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_D_head,
    stride_ss_head,
    stride_ss_hdim,
    # Meta
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    STATES_SCALE_PER_HP: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    gid = pid_h // nheads_ngroups_ratio
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + gid * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    )
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + gid * stride_C_head
    prev_states_ptr += (
        pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0
    ).to(tl.float32)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # Per-group scalar & per-(H,P) column scales (or scalar when STATES_SCALE_PER_HP=False).
    gscale = tl.load(C_scale + gid).to(tl.float32)
    if STATES_SCALE_PER_HP:
        ss_cols = tl.load(
            states_scale + pid_h * stride_ss_head + offs_n * stride_ss_hdim,
            mask=(offs_n < hdim),
            other=1.0,
        ).to(tl.float32)
    else:
        ss_scalar = tl.load(states_scale + pid_h * stride_ss_head).to(tl.float32)

    # Row gate
    row_gate = tl.exp(dA_cs_m)

    # ===== States path: INT8×INT8 → INT32 =====
    acc_i32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    k = 0
    while k < dstate:
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k[None, :] * stride_C_dstate)
        S_i8ptr = prev_states_ptr + (
            offs_k[:, None] * stride_states_dstate + offs_n[None, :] * stride_states_hdim
        )
        Ci8 = tl.load(
            C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate), other=0
        ).to(tl.int8)
        Si8 = tl.load(
            S_i8ptr, mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim), other=0
        ).to(tl.int8)
        acc_i32 += tl.dot(Ci8, Si8, out_dtype=tl.int32)
        k += BLOCK_SIZE_K

    acc_prev = tl.cast(acc_i32, tl.float32)
    acc_prev *= gscale
    if STATES_SCALE_PER_HP:
        acc_prev *= ss_cols[None, :]
    else:
        acc_prev *= ss_scalar
    acc = acc_prev * row_gate[:, None]

    # ===== Conv path: CB(int8→fp) @ X(fp) =====
    Scb = tl.load(CB_scale_ptr + gid).to(tl.float32)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)

    conv_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kk in range(0, K_MAX, BLOCK_SIZE_K):
        cb_q = tl.load(
            cb_ptrs,
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - kk),
            other=0,
        ).to(tl.int8)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - kk, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - kk, other=0.0).to(tl.float32)

        cb = tl.cast(cb_q, tl.float32) * Scb
        cb *= tl.exp((dA_cs_m[:, None] - dA_cs_k[None, :]))
        cb *= dt_k
        mask = offs_m[:, None] >= kk + offs_k[None, :]
        cb = tl.where(mask, cb, 0.0)

        Xfp = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - kk) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        conv_acc += tl.dot(cb.to(tl.float16), Xfp.to(tl.float16))

        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    acc += conv_acc

    # ===== Residual D =====
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_scale) * tl.load(
                D_ptr + pid_h * stride_D_head + offs_out_n, mask=offs_out_n < hdim, other=0.0
            ).to(tl.float32)
        else:
            D = tl.load(D_scale) * tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_res = tl.load(
            x_ptr + (offs_out_m[:, None] * stride_x_seqlen + offs_out_n[None, :] * stride_x_hdim),
            mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc += x_res * D

    # ===== Store output =====
    out_ptr += (
        pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    )
    out_ptrs = out_ptr + (
        stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim
    )
    tl.store(
        out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim)
    )


def _quant_chunk_scan_fwd(
    cb_i8,
    x_fp,
    dt,
    dA_cumsum,
    q_C,
    C_scale,
    states,
    q_D,
    D_scale,
    state_dequant_scale,
    cb_scale,
):
    """INT8 chunk scan (SSDi8 layout).

    Always-FP-x: ``x_fp`` is FP16/FP32, fed directly into the kernel without
    any dequant multiply (matches SSDi8). State dequant uses per-(H,P)
    ``state_dequant_scale``.

    Caller contract (the SSD orchestrator on the W4A8 hot path): ``cb_i8`` is
    INT8 with ``cb_scale`` shape (G,) FP32 contiguous; ``C_scale`` is (G,) FP32
    contiguous; ``state_dequant_scale`` is (H, P) FP32 (possibly a stride-0 view
    along P for per-tensor / per-head coarsening). ``q_D`` is the INT8 D vector
    (H,) with scalar ``D_scale``, or None for no residual.
    """
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = q_C.shape
    batch, seqlen, nheads, headdim = x_fp.shape
    device = x_fp.device
    states_scale = state_dequant_scale

    out = torch.empty(batch, seqlen, nheads, headdim, device=device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"]) * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )

    _chunk_scan_fwd_kernel[grid](
        cb_i8,
        x_fp,
        out,
        dt,
        dA_cumsum,
        q_C,
        C_scale,
        states,
        states_scale,
        q_D if q_D is not None else q_C,
        D_scale if D_scale is not None else C_scale,  # dummy when HAS_D=False
        cb_scale,
        # Dims
        chunk_size,
        headdim,
        dstate,
        batch,
        seqlen,
        nheads // ngroups,
        # Strides: cb
        cb_i8.stride(0),
        cb_i8.stride(1),
        cb_i8.stride(2),
        cb_i8.stride(3),
        cb_i8.stride(4),
        # x
        x_fp.stride(0),
        x_fp.stride(1),
        x_fp.stride(2),
        x_fp.stride(3),
        # out
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        # dt
        dt.stride(0),
        dt.stride(2),
        dt.stride(1),
        dt.stride(3),
        # dA_cumsum
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(3),
        # C
        q_C.stride(0),
        q_C.stride(1),
        q_C.stride(2),
        q_C.stride(3),
        # states
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        # D
        q_D.stride(0) if q_D is not None else 0,
        # states_scale
        states_scale.stride(0),
        states_scale.stride(1),
        # Meta
        HAS_D=q_D is not None,
        D_HAS_HDIM=q_D.dim() == 2 if q_D is not None else True,
        # P-axis stride 0 ⇒ scalar per-program (per-tensor or per-head coarsening).
        STATES_SCALE_PER_HP=states_scale.stride(1) != 0,
    )
    return out
