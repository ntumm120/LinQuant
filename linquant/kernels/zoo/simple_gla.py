"""INT8 chunk_simple_gla (linear attention with no or scalar decay): state kernel + output kernel.

Matches linquant.reference.chunk_core with policy `kernel_static`, `state=8`, `spass=fp32carry`, `readpost=1`,
roles q_read/q_local tied to one per-head scale, `round_away` rounding. eta splits the write decay between the
key factor (Ω^(1-eta)) and the value factor (Ω^eta); eta=1 is FLA's value-side placement.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _q8(x, inv_scale, qmax):
    """Round half away from zero to the INT8 grid (matches the production kernels' rounding)."""
    y = x * inv_scale
    y = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
    return tl.maximum(tl.minimum(y, qmax), -qmax).to(tl.int8)  # qmax: 127 (INT8) or 31 (INT6 codes in INT8)


@triton.jit
def _state_body(
    k, v, g, h_i8, ht, unrot,
    skw, svw, ss,  # (H,K) key-write scale, (H,V) value-write scale, (H,V) state scale (fp32)
    eta,
    qmax,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, STORE_FINAL: tl.constexpr, USE_UNROT: tl.constexpr,
):
    """USE_UNROT (canonical storage under R_V): v arrives rotated; the dequantized write product is multiplied by
    R_Vᵀ (fp16, (V,V)) before it joins the canonical fp32 carry, so the stored state is canonical. Needs BV == V
    (the launch wrapper `_state_kernel_unrot` fixes BV = V; the autotuned `_state_kernel` never un-rotates)."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k, m_v = o_k < K, o_v < V
    b_skw = tl.load(skw + i_h * K + o_k, mask=m_k, other=1.0)  # (BK,)
    b_svw = tl.load(svw + i_h * V + o_v, mask=m_v, other=1.0)  # (BV,)
    b_ss = tl.load(ss + i_h * V + o_v, mask=m_v, other=1.0)  # (BV,)
    inv_skw, inv_svw, inv_ss = 1.0 / b_skw, 1.0 / b_svw, 1.0 / b_ss
    b_h = tl.zeros([BK, BV], dtype=tl.float32)  # fp32 carry
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        o_h = ((i_n * NT + i_t) * H + i_h).to(tl.int64) * K * V
        p_h = tl.make_block_ptr(h_i8 + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # INT8 stored copy read by this chunk's output kernel
        tl.store(p_h, _q8(b_h, inv_ss[None, :], 127.0), boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)  # (BK, BT)
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)  # (BT, BV)
        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            o_t = i_t * BT + tl.arange(0, BT)
            b_g = tl.load(g + bos * H + o_t * H + i_h, mask=o_t < T, other=0.0)
            d = b_g_last - b_g  # (BT,) log of the terminal-write decay Ω_j; split in log space (no exp/log round trip)
            b_k = b_k * tl.exp((1.0 - eta) * d)[None, :]
            b_v = b_v * tl.exp(eta * d)[:, None]
            b_h *= tl.exp(b_g_last)
        kq = _q8(b_k, inv_skw[:, None], qmax)  # (BK, BT) int8, per-key-channel scale (output dim)
        vq = _q8(b_v, inv_svw[None, :], qmax)  # (BT, BV) int8, per-value-channel scale (output dim)
        acc = tl.dot(kq, vq, out_dtype=tl.int32).to(tl.float32) * b_skw[:, None] * b_svw[None, :]
        if USE_UNROT:
            p_r = tl.make_block_ptr(unrot, (V, V), (V, 1), (0, 0), (BV, BV), (1, 0))
            b_r = tl.load(p_r, boundary_check=(0, 1))
            acc = tl.dot(acc.to(tl.float16), b_r).to(tl.float32)  # dequantized product → canonical basis
        b_h += acc
    if STORE_FINAL:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=2),
             triton.Config({"BK": 64, "BV": 128}, num_warps=4, num_stages=2),
             triton.Config({"BK": 32, "BV": 64}, num_warps=2, num_stages=2)],
    key=["BT", "USE_G"],
)
@triton.jit(do_not_specialize=["T"])
def _state_kernel(k, v, g, h_i8, ht, unrot, skw, svw, ss, eta, qmax, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr, STORE_FINAL: tl.constexpr):
    _state_body(k, v, g, h_i8, ht, unrot, skw, svw, ss, eta, qmax, T, H, K, V, BT, BK, BV, USE_G, STORE_FINAL, False)


@triton.jit(do_not_specialize=["T"])
def _state_kernel_unrot(k, v, g, h_i8, ht, unrot, skw, svw, ss, eta, qmax, T, H: tl.constexpr, K: tl.constexpr,
                        V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr,
                        STORE_FINAL: tl.constexpr):
    tl.static_assert(BV == V, "canonical storage needs one V-tile spanning all value channels")
    _state_body(k, v, g, h_i8, ht, unrot, skw, svw, ss, eta, qmax, T, H, K, V, BT, BK, BV, USE_G, STORE_FINAL, True)


@triton.jit
def _output_body(
    q, k, v, h_i8, g, o, unrot,
    sq, sk, svl, ss,  # (H,) q scale, (H,) k scale, (H,V) local value scale, (H,V) state scale
    scale,
    qmax,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_UNROT: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    bos = i_b * T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h_i8 += ((i_b * NT + i_t) * H + i_h).to(tl.int64) * K * V
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    b_sq = tl.load(sq + i_h)
    b_sk = tl.load(sk + i_h)
    b_svl = tl.load(svl + i_h * V + o_v, mask=m_v, other=1.0)
    b_ss = tl.load(ss + i_h * V + o_v, mask=m_v, other=1.0)
    inv_sq, inv_sk, inv_svl = 1.0 / b_sq, 1.0 / b_sk, 1.0 / b_svl
    b_o = tl.zeros([BT, BV], dtype=tl.int32)
    b_A = tl.zeros([BT, BT], dtype=tl.int32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h_i8, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        qq = _q8(tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * scale, inv_sq, qmax)  # (BT, BK): q·scale is the operand
        kq = _q8(tl.load(p_k, boundary_check=(0, 1)).to(tl.float32), inv_sk, qmax)  # (BK, BT)
        hq = tl.load(p_h, boundary_check=(0, 1))  # (BK, BV) int8
        b_o += tl.dot(qq, hq, out_dtype=tl.int32)
        b_A += tl.dot(qq, kq, out_dtype=tl.int32)
    b_of = b_o.to(tl.float32) * b_sq * b_ss[None, :]  # state read, dequantized
    b_Af = b_A.to(tl.float32) * (b_sq * b_sk)  # C·Bᵀ kept fp32 (no requant in the zoo lineage)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_G:
        b_g = tl.load(g + bos * H + o_t * H + i_h, mask=m_t, other=0.0)
        b_of = b_of * tl.exp(b_g)[:, None]  # read-side decay after the product
        b_Af = b_Af * tl.exp(b_g[:, None] - b_g[None, :])
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_Af = tl.where(m_A, b_Af, 0.0)
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    vq = _q8(tl.load(p_v, boundary_check=(0, 1)).to(tl.float32), inv_svl[None, :], qmax)  # INT8 codes ...
    b_loc = tl.dot(b_Af.to(tl.float16), vq.to(tl.float16)) * b_svl[None, :]  # ... fp16 dot (production convention)
    if USE_UNROT:  # canonical storage: the state read is canonical, only the local product is in the rotated basis
        p_r = tl.make_block_ptr(unrot, (V, V), (V, 1), (0, 0), (BV, BV), (1, 0))
        b_r = tl.load(p_r, boundary_check=(0, 1))
        b_loc = tl.dot(b_loc.to(tl.float16), b_r).to(tl.float32)
    b_out = b_of + b_loc  # scale already inside the quantized q
    tl.store(p_o, b_out.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=2),
             triton.Config({"BK": 64, "BV": 128}, num_warps=4, num_stages=2),
             triton.Config({"BK": 32, "BV": 64}, num_warps=2, num_stages=2)],
    key=["BT", "USE_G"],
)
@triton.jit(do_not_specialize=["T"])
def _output_kernel(q, k, v, h_i8, g, o, unrot, sq, sk, svl, ss, scale, qmax, T, H: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr):
    _output_body(q, k, v, h_i8, g, o, unrot, sq, sk, svl, ss, scale, qmax, T, H, K, V, BT, BK, BV, USE_G, False)


@triton.jit(do_not_specialize=["T"])
def _output_kernel_unrot(q, k, v, h_i8, g, o, unrot, sq, sk, svl, ss, scale, qmax, T, H: tl.constexpr, K: tl.constexpr,
                         V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr):
    tl.static_assert(BV == V, "canonical storage needs one V-tile spanning all value channels")
    _output_body(q, k, v, h_i8, g, o, unrot, sq, sk, svl, ss, scale, qmax, T, H, K, V, BT, BK, BV, USE_G, True)


def chunk_simple_gla_int8(q, k, v, g, scales, *, eta=1.0, scale=None, chunk_size=64, output_final_state=False, unrot=None, bits=8, out_dtype=None):
    """q,k [B,T,H,K], v [B,T,H,V] (fp16/bf16), g [B,T,H] fp32 log-decay or None.
    scales: dict with 'q' (H,) [calibrated on q·scale], 'k' (H,), 'kw' (H,K), 'vw' (H,V), 'vl' (H,V), 'state' (H,V),
    all fp32 CUDA (linquant.invariances.FakeQuant.kernel_scales). T must be a multiple of chunk_size.
    unrot: optional (V,V) fp16 R_Vᵀ for canonical storage when v was rotated by R_V (output comes out canonical).
    Returns (o [B,T,H,V] in v.dtype, final fp32 state [B,H,K,V] or None)."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()  # block pointers assume dense [B,T,H,·] layouts
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    scale = K**-0.5 if scale is None else scale
    qmax = float(2 ** (bits - 1) - 1)  # code range: 127 for INT8, 31 for INT6 codes in INT8 containers
    h_i8 = torch.empty(B, NT, H, K, V, dtype=torch.int8, device=q.device)
    ht = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device) if output_final_state else None
    # FLA convention: the kernels consume the chunk-local cumulative log-decay
    gg = g.float().view(B, NT, BT, H).cumsum(dim=2).reshape(B, T, H).contiguous() if g is not None else None
    o = torch.empty(v.shape, dtype=out_dtype or v.dtype, device=v.device)  # fp32 for an exit un-rotation
    if unrot is None:
        grid_h = lambda meta: (triton.cdiv(K, meta["BK"]), triton.cdiv(V, meta["BV"]), B * H)  # noqa: E731
        _state_kernel[grid_h](k, v, gg, h_i8, ht, q, scales["kw"], scales["vw"], scales["state"], float(eta), qmax,
                              T=T, H=H, K=K, V=V, BT=BT, USE_G=g is not None, STORE_FINAL=ht is not None)
        grid_o = lambda meta: (triton.cdiv(V, meta["BV"]), NT, B * H)  # noqa: E731
        _output_kernel[grid_o](q, k, v, h_i8, gg, o, q, scales["q"], scales["k"], scales["vl"], scales["state"], float(scale), qmax,
                               T=T, H=H, K=K, V=V, BT=BT, USE_G=g is not None)
        return o, ht
    # canonical storage under R_V: fixed configs with the V tile spanning all value channels
    assert V in (64, 128), "canonical storage needs one V-tile spanning all value channels"
    R = unrot.to(torch.float16).contiguous()
    BK = 64
    _state_kernel_unrot[(triton.cdiv(K, BK), 1, B * H)](k, v, gg, h_i8, ht, R, scales["kw"], scales["vw"], scales["state"],
                                                          float(eta), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=V,
                                                          USE_G=g is not None, STORE_FINAL=ht is not None, num_warps=4,
                                                          num_stages=2)
    _output_kernel_unrot[(1, NT, B * H)](q, k, v, h_i8, gg, o, R, scales["q"], scales["k"], scales["vl"], scales["state"],
                                          float(scale), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=V, USE_G=g is not None,
                                          num_warps=4, num_stages=2)
    return o, ht
