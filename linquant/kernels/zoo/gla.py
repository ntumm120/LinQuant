"""INT8 chunk_gla (linear attention with diagonal, per-channel decay).

FLA structure: intra-chunk product A via secondary chunks of BC tokens (off-diagonal BC×BC blocks as dots on
anchor-rescaled operands, diagonal blocks as an fp32 loop with exact per-channel weights), state kernel with
h ← Diag(exp(g_last))·h + (k ⊙ exp(g_last − g))ᵀ v, output o = (q ⊙ exp(g))·h + A·v. Quantized: the anchored
q/k operands of the off-diagonal blocks (INT8 dots, per-head static scales, q tied to the read), the diagonal
blocks on the same INT8 codes (fp32 reduction: the per-element decay sits inside the contraction), the
pre-decayed read operand q ⊙ exp(g) (per head), the write operands (per channel), the stored state (per (H,V)).
eta splits the scalar envelope s_t = max_n exp(g_last,n − g_t,n) of the write decay between the key factor
(Ω ⊙ s^−eta) and the value factor (s^eta). R_QK is local only (the anchored operands are rotated before
quantizing; the state axis is not rotatable under diagonal decay); R_V with canonical storage as in simple_gla.
Matches linquant.reference.chunk_core(decay="diagonal") with policy `kernel_static`, `state=8`, `spass=fp32carry`,
`kernel_arith`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .simple_gla import _q8

BC = 16  # secondary chunk (linquant.reference.BC)


@triton.jit(do_not_specialize=["T"])
def _gla_A_inter(
    q, k, g, A, rot, sq, sk, scale, qmax,
    T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    USE_ROT: tl.constexpr,
):
    """Off-diagonal blocks (i_i > i_j): INT8(q ⊙ exp(g − g_anchor)·scale) · INT8(k ⊙ exp(g_anchor − g))ᵀ, the anchor
    being the first row of the i-block; USE_ROT rotates both anchored operands by R_QK (fp16 dot, needs BK == K)."""
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    bos = i_b * T
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return
    b_sq, b_sk = tl.load(sq + i_h), tl.load(sk + i_h)
    inv_sq, inv_sk = 1.0 / b_sq, 1.0 / b_sk
    b_A = tl.zeros([BC, BC], dtype=tl.int32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        b_gn = tl.load(g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k, mask=m_k, other=0.0)  # anchor (BK,)
        b_qg = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * tl.exp(tl.load(p_g, boundary_check=(0, 1)) - b_gn[None, :]) * scale
        b_kg = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * tl.exp(b_gn[:, None] - tl.load(p_gk, boundary_check=(0, 1)))
        if USE_ROT:
            p_r = tl.make_block_ptr(rot, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
            b_r = tl.load(p_r, boundary_check=(0, 1))
            b_qg = tl.dot(b_qg.to(tl.float16), b_r).to(tl.float32)  # Qb·R
            b_kg = tl.dot(tl.trans(b_r), b_kg.to(tl.float16)).to(tl.float32)  # (Kb·R)ᵀ
        b_A += tl.dot(_q8(b_qg, inv_sq, qmax), _q8(b_kg, inv_sk, qmax), out_dtype=tl.int32)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, (b_A.to(tl.float32) * (b_sq * b_sk)).to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _gla_A_intra(
    q, k, g, A, sq, sk, scale, qmax,
    T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
):
    """Diagonal blocks: A_ij = Σ_n q̂_in k̂_jn exp(g_in − g_jn), i ≥ j, on the dequantized INT8 codes (fp32 reduction)."""
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    if i_t * BT + i_i * BC >= T:
        return
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_A = (i_t * BT + i_i * BC + o_i) * H * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + o_i) < T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    b_sq, b_sk = tl.load(sq + i_h), tl.load(sk + i_h)
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    b_q = _q8(tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * scale, 1.0 / b_sq, qmax).to(tl.float32) * b_sq
    b_g = tl.load(p_g, boundary_check=(0, 1))
    p_k = k + (i_t * BT + i_i * BC) * H * K + o_k
    p_gk = g + (i_t * BT + i_i * BC) * H * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = _q8(tl.load(p_k, mask=m_k, other=0.0).to(tl.float32), 1.0 / b_sk, qmax).to(tl.float32) * b_sk
        b_gk = tl.load(p_gk, mask=m_k, other=0.0)
        b_Aj = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        tl.store(A + o_A + j, b_Aj.to(A.dtype.element_ty), mask=m_A & (o_i >= j))
        p_k += H * K
        p_gk += H * K


@triton.jit
def _gla_state_body(
    k, v, g, logenv, h_i8, ht, unrot,
    skw, svw, ss,
    eta,
    qmax,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    STORE_FINAL: tl.constexpr, USE_UNROT: tl.constexpr,
):
    """Per-channel gates: h ← Diag(exp(g_last))·h + INT8(k ⊙ Ω ⊙ s^−eta)ᵀ · INT8(v ⊙ s^eta), Ω = exp(g_last − g) per
    (token, channel), s_t = exp(logenv_t) the scalar envelope max_n Ω_tn. USE_UNROT as in simple_gla (BV == V)."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k, m_v = o_k < K, o_v < V
    b_skw = tl.load(skw + i_h * K + o_k, mask=m_k, other=1.0)
    b_svw = tl.load(svw + i_h * V + o_v, mask=m_v, other=1.0)
    b_ss = tl.load(ss + i_h * V + o_v, mask=m_v, other=1.0)
    inv_skw, inv_svw, inv_ss = 1.0 / b_skw, 1.0 / b_svw, 1.0 / b_ss
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        o_h = ((i_n * NT + i_t) * H + i_h).to(tl.int64) * K * V
        p_h = tl.make_block_ptr(h_i8 + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, _q8(b_h, inv_ss[None, :], 127.0), boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_gk_last = tl.load(g + (bos + last_idx) * H * K + i_h * K + o_k, mask=m_k, other=0.0)  # (BK,)
        b_gk = tl.load(p_gk, boundary_check=(0, 1))  # (BK, BT)
        o_t = i_t * BT + tl.arange(0, BT)
        b_le = tl.load(logenv + bos * H + o_t * H + i_h, mask=o_t < T, other=0.0)  # (BT,) log s_t
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * tl.exp(b_gk_last[:, None] - b_gk - eta * b_le[None, :])
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32) * tl.exp(eta * b_le)[:, None]
        b_h *= tl.exp(b_gk_last)[:, None]
        acc = tl.dot(_q8(b_k, inv_skw[:, None], qmax), _q8(b_v, inv_svw[None, :], qmax), out_dtype=tl.int32).to(tl.float32)
        acc = acc * b_skw[:, None] * b_svw[None, :]
        if USE_UNROT:
            p_r = tl.make_block_ptr(unrot, (V, V), (V, 1), (0, 0), (BV, BV), (1, 0))
            acc = tl.dot(acc.to(tl.float16), tl.load(p_r, boundary_check=(0, 1))).to(tl.float32)
        b_h += acc
    if STORE_FINAL:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=2),
             triton.Config({"BK": 64, "BV": 128}, num_warps=4, num_stages=2),
             triton.Config({"BK": 32, "BV": 64}, num_warps=2, num_stages=2)],
    key=["BT"],
)
@triton.jit(do_not_specialize=["T"])
def _gla_state_kernel(k, v, g, logenv, h_i8, ht, unrot, skw, svw, ss, eta, qmax, T, H: tl.constexpr, K: tl.constexpr,
                      V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, STORE_FINAL: tl.constexpr):
    _gla_state_body(k, v, g, logenv, h_i8, ht, unrot, skw, svw, ss, eta, qmax, T, H, K, V, BT, BK, BV, STORE_FINAL, False)


@triton.jit(do_not_specialize=["T"])
def _gla_state_kernel_unrot(k, v, g, logenv, h_i8, ht, unrot, skw, svw, ss, eta, qmax, T, H: tl.constexpr, K: tl.constexpr,
                            V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, STORE_FINAL: tl.constexpr):
    tl.static_assert(BV == V, "canonical storage needs one V-tile spanning all value channels")
    _gla_state_body(k, v, g, logenv, h_i8, ht, unrot, skw, svw, ss, eta, qmax, T, H, K, V, BT, BK, BV, STORE_FINAL, True)


@triton.jit
def _gla_output_body(
    q, v, g, A, h_i8, o, unrot,
    sq, svl, ss,
    scale,
    qmax,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_UNROT: tl.constexpr,
):
    """o = INT8(q ⊙ exp(g)·scale)·h_int8 (state read, per-head scale) + A·INT8(v) (fp16 dot on the codes)."""
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    bos = i_b * T
    q += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    A += (bos * H + i_h) * BT
    h_i8 += ((i_b * NT + i_t) * H + i_h).to(tl.int64) * K * V
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    b_sq = tl.load(sq + i_h)
    b_svl = tl.load(svl + i_h * V + o_v, mask=m_v, other=1.0)
    b_ss = tl.load(ss + i_h * V + o_v, mask=m_v, other=1.0)
    inv_sq, inv_svl = 1.0 / b_sq, 1.0 / b_svl
    b_o = tl.zeros([BT, BV], dtype=tl.int32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h_i8, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        qq = _q8(tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * tl.exp(tl.load(p_g, boundary_check=(0, 1))) * scale, inv_sq, qmax)
        b_o += tl.dot(qq, tl.load(p_h, boundary_check=(0, 1)), out_dtype=tl.int32)
    b_of = b_o.to(tl.float32) * b_sq * b_ss[None, :]
    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))  # fp32, lower-triangular incl. the diagonal (zeros above)
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    vq = _q8(tl.load(p_v, boundary_check=(0, 1)).to(tl.float32), inv_svl[None, :], qmax)
    b_loc = tl.dot(b_A.to(tl.float16), vq.to(tl.float16)) * b_svl[None, :]
    if USE_UNROT:
        p_r = tl.make_block_ptr(unrot, (V, V), (V, 1), (0, 0), (BV, BV), (1, 0))
        b_loc = tl.dot(b_loc.to(tl.float16), tl.load(p_r, boundary_check=(0, 1))).to(tl.float32)
    tl.store(p_o, (b_of + b_loc).to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=2),
             triton.Config({"BK": 64, "BV": 128}, num_warps=4, num_stages=2),
             triton.Config({"BK": 32, "BV": 64}, num_warps=2, num_stages=2)],
    key=["BT"],
)
@triton.jit(do_not_specialize=["T"])
def _gla_output_kernel(q, v, g, A, h_i8, o, unrot, sq, svl, ss, scale, qmax, T, H: tl.constexpr, K: tl.constexpr,
                       V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    _gla_output_body(q, v, g, A, h_i8, o, unrot, sq, svl, ss, scale, qmax, T, H, K, V, BT, BK, BV, False)


@triton.jit(do_not_specialize=["T"])
def _gla_output_kernel_unrot(q, v, g, A, h_i8, o, unrot, sq, svl, ss, scale, qmax, T, H: tl.constexpr, K: tl.constexpr,
                             V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    tl.static_assert(BV == V, "canonical storage needs one V-tile spanning all value channels")
    _gla_output_body(q, v, g, A, h_i8, o, unrot, sq, svl, ss, scale, qmax, T, H, K, V, BT, BK, BV, True)


def chunk_gla_int8(q, k, v, g, scales, *, eta=0.0, scale=None, chunk_size=64, output_final_state=False, unrot=None,
                   rot=None, bits=8, out_dtype=None):
    """q,k [B,T,H,K], v [B,T,H,V] (fp16/bf16), g [B,T,H,K] fp32 per-channel log-decay.
    scales: 'q' (H,) [on q·scale, tied read/local], 'k' (H,), 'kw' (H,K), 'vw' (H,V), 'vl' (H,V), 'state' (H,V).
    rot: optional (K,K) R_QK applied to the anchored local operands only; unrot: optional (V,V) R_Vᵀ (canonical
    storage, v arrives rotated). Returns (o [B,T,H,V] in v.dtype, final fp32 state or None)."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    assert T % BT == 0 and K <= 128 and BT % BC == 0
    scale = K**-0.5 if scale is None else scale
    qmax = float(2 ** (bits - 1) - 1)  # code range: 127 for INT8, 31 for INT6 codes in INT8 containers
    gc = g.float().view(B, NT, BT, H, K).cumsum(dim=2)  # chunk-local cumulative log-decay (fp32)
    logenv = (gc[:, :, -1:] - gc).amax(-1).reshape(B, T, H).contiguous()  # log of the scalar envelope s_t
    gc = gc.reshape(B, T, H, K).contiguous()
    A = torch.zeros(B, T, H, BT, dtype=torch.float32, device=q.device)
    BKf = triton.next_power_of_2(K)
    NC = BT // BC
    R = q if rot is None else rot.to(torch.float16).contiguous()
    _gla_A_inter[(NT, NC * NC, B * H)](q, k, gc, A, R, scales["q"], scales["k"], float(scale), qmax, T=T, H=H, K=K, BT=BT,
                                       BC=BC, BK=BKf, NC=NC, USE_ROT=rot is not None, num_warps=4)
    _gla_A_intra[(NT, NC, B * H)](q, k, gc, A, scales["q"], scales["k"], float(scale), qmax, T=T, H=H, K=K, BT=BT, BC=BC,
                                  BK=BKf, num_warps=4)
    h_i8 = torch.empty(B, NT, H, K, V, dtype=torch.int8, device=q.device)
    ht = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device) if output_final_state else None
    o = torch.empty(v.shape, dtype=out_dtype or v.dtype, device=v.device)  # fp32 for an exit un-rotation
    if unrot is None:
        grid_h = lambda meta: (triton.cdiv(K, meta["BK"]), triton.cdiv(V, meta["BV"]), B * H)  # noqa: E731
        _gla_state_kernel[grid_h](k, v, gc, logenv, h_i8, ht, q, scales["kw"], scales["vw"], scales["state"], float(eta), qmax,
                                  T=T, H=H, K=K, V=V, BT=BT, STORE_FINAL=ht is not None)
        grid_o = lambda meta: (triton.cdiv(V, meta["BV"]), NT, B * H)  # noqa: E731
        _gla_output_kernel[grid_o](q, v, gc, A, h_i8, o, q, scales["q"], scales["vl"], scales["state"], float(scale), qmax,
                                   T=T, H=H, K=K, V=V, BT=BT)
        return o, ht
    assert V in (64, 128), "canonical storage needs one V-tile spanning all value channels"
    Ru = unrot.to(torch.float16).contiguous()
    _gla_state_kernel_unrot[(triton.cdiv(K, 64), 1, B * H)](k, v, gc, logenv, h_i8, ht, Ru, scales["kw"], scales["vw"],
                                                            scales["state"], float(eta), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=64,
                                                            BV=V, STORE_FINAL=ht is not None, num_warps=4, num_stages=2)
    _gla_output_kernel_unrot[(1, NT, B * H)](q, v, gc, A, h_i8, o, Ru, scales["q"], scales["vl"], scales["state"],
                                             float(scale), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=64, BV=V, num_warps=4,
                                             num_stages=2)
    return o, ht
