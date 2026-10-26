# ruff: noqa: E501, D100, D205, E731, C901
"""INT8 BMM CB chunk kernel — quantized C @ B^T per-chunk, INT8 out with a static per-G scale."""

import math

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
    # B_IS_FP changes the SHMEM footprint per K-tile (FP16 vs INT8 B), so
    # the value-side and key-side variants must not share one autotune entry.
    key=["chunk_size", "K", "B_IS_FP"],
)
@triton.jit
def _quant_bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr,
    a_scale,
    b_ptr,
    b_scale,
    out_ptr,
    cb_out_scale,
    # Matrix dimensions
    seqlen,
    chunk_size,
    K,
    ngroups,
    stride_a_batch,
    stride_a_seqlen,
    stride_a_head,
    stride_ak,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_bk,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_outm,
    stride_outn,
    # Meta-parameters
    B_IS_FP: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h * stride_b_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    a_sc = tl.load(a_scale + pid_h).to(tl.float32)
    b_sc = tl.load(b_scale + pid_h).to(tl.float32)

    if B_IS_FP:
        # b_sc = amax/127 (SSDi8 convention; verified vs quantize_int8_per_group_torch
        # in libs/SSDi8/.../quant_ssd_combined.py:167 where q = round(t/S)).
        # So b_inv_sc = 1/b_sc = 127/amax — maps b_fp ∈ [-amax, amax] to [-127, 127].
        # NOT 127/b_sc (= 127²/amax) which saturates b_q to ±127 every element.
        b_inv_sc = 1.0 / b_sc
    acc_i32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask_a = (offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        k_mask_b = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit)
        a_i8 = tl.load(a_ptrs, mask=k_mask_a, other=0).to(tl.int8)
        if B_IS_FP:
            b_fp = tl.load(b_ptrs, mask=k_mask_b, other=0.0).to(tl.float32)
            b_q = tl.extra.cuda.libdevice.rint(b_fp * b_inv_sc)
            b_i8 = tl.maximum(tl.minimum(b_q, 127), -127).to(tl.int8)
        else:
            b_i8 = tl.load(b_ptrs, mask=k_mask_b, other=0).to(tl.int8)
        acc_i32 += tl.dot(a_i8, b_i8, out_dtype=tl.int32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_mask = (offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size)

    # Requantize to INT8 using pre-supplied per-group scale.
    # Round-half-even (libdevice.rint, IEEE default) — mathematically unbiased
    # on symmetric distributions. SSDi8 uses round-half-away which introduces
    # a small positive bias; bisect on 04-29 showed half-even gives lower PPL.
    Scb = tl.load(cb_out_scale + pid_h).to(tl.float32)
    inv_scb = (a_sc * b_sc) / Scb
    q = tl.extra.cuda.libdevice.rint(acc_i32.to(tl.float32) * inv_scb)
    cb_max = 127.0
    q = tl.minimum(tl.maximum(q, -cb_max), cb_max)
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, q.to(tl.int8), mask=out_mask)


def _quant_bmm_chunk_fwd(
    a,
    a_scale,
    b,
    b_scale,
    chunk_size,
    static_cb_scale,
    b_is_fp=False,
):
    """Argument:
        a: (batch, seqlen, ngroups, k) — INT8, per-group (G,) scale.
        b: (batch, seqlen, ngroups, k) — INT8 or FP16 (if b_is_fp), per-group (G,) scale.
        static_cb_scale: (ngroups,) FP32 — pre-calibrated per-group INT8 scale of the CB output.

    Return:
        out_int8: (batch, nchunks, ngroups, chunk_size, chunk_size) INT8, scale ``static_cb_scale``.
    """
    batch, seqlen, ngroups, k = a.shape
    nchunks = math.ceil(seqlen / chunk_size)

    # `static_cb_scale` is registered as a FP32 contiguous (ngroups,)
    # buffer at layer construction time; the per-call `.float().contiguous()`
    # was a CPU dispatch cost that drained the GPU stream queue. Fast-path
    # the common case; fall back to adaptation only for off-shape inputs.
    if (
        static_cb_scale.dtype == torch.float32
        and static_cb_scale.shape == (ngroups,)
        and static_cb_scale.is_contiguous()
    ):
        cb_out_scale = static_cb_scale
    else:
        cb_out_scale = static_cb_scale.float().contiguous()
    assert cb_out_scale.shape == (ngroups,), (
        f"static_cb_scale must be (ngroups,), got {cb_out_scale.shape}"
    )

    out = torch.empty(
        (batch, nchunks, ngroups, chunk_size, chunk_size),
        device=a.device,
        dtype=torch.int8,
    )

    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(chunk_size, META["BLOCK_SIZE_N"]),
        batch,
        nchunks * ngroups,
    )
    with torch.cuda.device(a.device.index):
        _quant_bmm_chunk_fwd_kernel[grid](
            a,
            a_scale,
            b,
            b_scale,
            out,
            cb_out_scale,
            seqlen,
            chunk_size,
            k,
            ngroups,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a.stride(-1),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            b.stride(-1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(-2),
            out.stride(-1),
            B_IS_FP=b_is_fp,
        )
    return out
