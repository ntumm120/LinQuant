"""INT8 chunk_kda (delta rule with diagonal, per-channel decay).

Composition of the delta and GLA pieces: fp32 torch WY prep with per-channel gates (Gram with exp(g_i − g_j)
per channel, unquantized; W = T·(β^σ k ⊙ exp(g)), U = T·(β^σ v), Hw = β^((1−σ)τ) k materialized in the activation
dtype), the GLA A kernels on (q, Hw) (INT8 dots on anchor-rescaled operands, fp32 diagonal blocks), a state
kernel with E = (U − W·S)·fw formed in-loop and per-channel write gates under the scalar η envelope, and the GLA
output kernel on (q, E, A). R_QK is local only (anchored operands), R_V with canonical storage ("align").
Matches linquant.reference.chunk_core(family="delta", decay="diagonal") under `kernel_static`, `state=8`,
`spass=fp32carry`, `kernel_arith`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .gla import BC, _gla_A_inter, _gla_A_intra, _gla_output_kernel, _gla_output_kernel_unrot
from .simple_gla import _q8


def _prep(k, v, gc, beta, *, sigma, tau, chunk_size):
    """fp32 WY prep with per-channel decay. Returns (Hw, W, U, fw) with Hw/W/U in k's dtype."""
    B, T, H, K = k.shape
    C = chunk_size
    nc = T // C
    kf, vf = k.float(), v.float()
    bt = beta.float().unsqueeze(-1)  # [B,T,H,1]
    Lc = (bt**sigma * kf).view(B, nc, C, H, K).transpose(2, 3)  # [B,nc,H,C,K]
    Hg = (bt ** (1 - sigma) * kf).view(B, nc, C, H, K).transpose(2, 3)
    g = gc.view(B, nc, C, H, K).transpose(2, 3)  # [B,nc,H,C,K] chunk-local cumulative log-decay
    # G_ij = Σ_n L_in Hg_jn exp(g_in − g_jn), i > j, via FLA's secondary chunks of BC tokens: anchor-rescaled
    # matmuls for the off-diagonal blocks, an einsum on the (BC×BC) diagonal blocks (exponents ≤ 0: g is
    # non-increasing within a chunk, so nothing overflows)
    G = torch.zeros(B, nc, H, C, C, dtype=torch.float32, device=k.device)
    tril = torch.ones(BC, BC, dtype=torch.bool, device=k.device).tril(-1)
    for s0 in range(0, C, BC):
        rows = slice(s0, s0 + BC)
        anchor = g[:, :, :, s0 : s0 + 1]  # [B,nc,H,1,K]
        if s0 > 0:
            Lb = Lc[:, :, :, rows] * (g[:, :, :, rows] - anchor).exp()
            Hb = Hg[:, :, :, :s0] * (anchor - g[:, :, :, :s0]).exp()
            G[:, :, :, rows, :s0] = Lb @ Hb.transpose(-1, -2)
        gd = g[:, :, :, rows]
        w = (gd.unsqueeze(4) - gd.unsqueeze(3)).masked_fill(~tril[:, :, None], float("-inf")).exp()  # [..,BC,BC,K]
        G[:, :, :, rows, rows] = torch.einsum("bnhik,bnhjk,bnhijk->bnhij", Lc[:, :, :, rows], Hg[:, :, :, rows], w)
    eye = torch.eye(C, dtype=torch.float32, device=k.device)
    Tm = torch.linalg.solve_triangular(eye + G, eye.expand(G.shape).clone(), upper=False, unitriangular=True)
    W = (Tm @ (Lc * g.exp())).transpose(2, 3).reshape(B, T, H, K)
    U = (Tm @ (bt**sigma * vf).view(B, nc, C, H, -1).transpose(2, 3)).transpose(2, 3).reshape(B, T, H, -1)
    Hw = bt ** ((1 - sigma) * tau) * kf
    fw = (bt ** ((1 - sigma) * (1 - tau))).view(B, T, H)
    return Hw.to(k.dtype), W.to(k.dtype), U.to(v.dtype), fw.contiguous()


def _prep_fla(q, k, v, gc, beta, scale, chunk_size):
    """FLA's fused KDA prep (kkt with per-channel gates, solve, recompute_w_u) for the FLA placement σ=τ=1:
    W = T·(βk⊙exp(g)), U = T·(βv), Hw = k, fw = 1. FLA's KDA kernels work in log2 units (gk = g·RCP_LN2)."""
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.utils.constant import RCP_LN2

    B, T, H, K = k.shape
    w, u = chunk_kda_fwd_intra(q=q, k=k, v=v, gk=(gc * RCP_LN2).contiguous(), beta=beta.to(k.dtype),
                               scale=scale, chunk_size=chunk_size)[:2]  # FLA 0.4 returns 5 values, 0.5 six; w, u first
    return k, w.to(k.dtype), u.to(v.dtype), torch.ones(B, T, H, device=k.device)


@triton.jit
def _kda_state_body(
    kw, w, u, fw, g, logenv, h_i8, e_out, ht, unrot,
    skw, sw, ew, ss,
    eta,
    qmax,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    STORE_FINAL: tl.constexpr, USE_UNROT: tl.constexpr,
):
    """One program per (V-tile, batch·head), full K in registers. Per-channel gates: h ← Diag(exp(g_last))·h +
    INT8(Hw ⊙ Ω ⊙ s^−eta)ᵀ · INT8(E ⊙ s^eta); E = (U − W·S)·fw stored fp16 (undecayed) for the output kernel."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k, m_v = o_k < K, o_v < V
    b_skw = tl.load(skw + i_h * K + o_k, mask=m_k, other=1.0)
    b_sw = tl.load(sw + i_h)
    b_ew = tl.load(ew + i_h * V + o_v, mask=m_v, other=1.0)
    b_ss = tl.load(ss + i_h * V + o_v, mask=m_v, other=1.0)
    inv_skw, inv_sw, inv_ew, inv_ss = 1.0 / b_skw, 1.0 / b_sw, 1.0 / b_ew, 1.0 / b_ss
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        o_h = ((i_n * NT + i_t) * H + i_h).to(tl.int64) * K * V
        p_h = tl.make_block_ptr(h_i8 + o_h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        hq = _q8(b_h, inv_ss[None, :], 127.0)
        tl.store(p_h, hq, boundary_check=(0, 1))
        p_kw = tl.make_block_ptr(kw + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_t * BT), (BK, BT), (0, 1))
        p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_t * BT), (BK, BT), (0, 1))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_e = tl.make_block_ptr(e_out + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_kw = tl.load(p_kw, boundary_check=(0, 1)).to(tl.float32)
        b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
        b_u = tl.load(p_u, boundary_check=(0, 1)).to(tl.float32)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        b_fw = tl.load(fw + bos * H + o_t * H + i_h, mask=m_t, other=1.0)
        wq = _q8(b_w, inv_sw, qmax)
        b_ws = tl.dot(wq, hq, out_dtype=tl.int32).to(tl.float32) * b_sw * b_ss[None, :]
        if USE_UNROT:
            p_rot = tl.make_block_ptr(unrot, (V, V), (1, V), (0, 0), (BV, BV), (0, 1))
            b_ws = tl.dot(b_ws.to(tl.float16), tl.load(p_rot, boundary_check=(0, 1))).to(tl.float32)
        b_e = (b_u - b_ws) * b_fw[:, None]
        tl.store(p_e, b_e.to(p_e.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_gk_last = tl.load(g + (bos + last_idx) * H * K + i_h * K + o_k, mask=m_k, other=0.0)  # (BK,)
        b_gk = tl.load(p_gk, boundary_check=(0, 1))  # (BK, BT)
        b_le = tl.load(logenv + bos * H + o_t * H + i_h, mask=m_t, other=0.0)  # (BT,) log s_t
        b_kw = b_kw * tl.exp(b_gk_last[:, None] - b_gk - eta * b_le[None, :])
        b_e = b_e * tl.exp(eta * b_le)[:, None]
        b_h *= tl.exp(b_gk_last)[:, None]
        acc = tl.dot(_q8(b_kw, inv_skw[:, None], qmax), _q8(b_e, inv_ew[None, :], qmax), out_dtype=tl.int32).to(tl.float32)
        acc = acc * b_skw[:, None] * b_ew[None, :]
        if USE_UNROT:
            p_un = tl.make_block_ptr(unrot, (V, V), (V, 1), (0, 0), (BV, BV), (1, 0))
            acc = tl.dot(acc.to(tl.float16), tl.load(p_un, boundary_check=(0, 1))).to(tl.float32)
        b_h += acc
    if STORE_FINAL:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({"BV": BV}, num_warps=w, num_stages=st) for BV in (64, 128) for w in (4, 8) for st in (1, 2)],
    key=["BT"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_state_kernel(kw, w, u, fw, g, logenv, h_i8, e_out, ht, unrot, skw, sw, ew, ss, eta, qmax, T, H: tl.constexpr,
                      K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                      STORE_FINAL: tl.constexpr):
    _kda_state_body(kw, w, u, fw, g, logenv, h_i8, e_out, ht, unrot, skw, sw, ew, ss, eta, qmax, T, H, K, V, BT, BK, BV,
                    STORE_FINAL, False)


@triton.jit(do_not_specialize=["T"])
def _kda_state_kernel_unrot(kw, w, u, fw, g, logenv, h_i8, e_out, ht, unrot, skw, sw, ew, ss, eta, qmax, T, H: tl.constexpr,
                            K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                            STORE_FINAL: tl.constexpr):
    tl.static_assert(BV == V, "canonical storage needs one V-tile spanning all value channels")
    _kda_state_body(kw, w, u, fw, g, logenv, h_i8, e_out, ht, unrot, skw, sw, ew, ss, eta, qmax, T, H, K, V, BT, BK, BV,
                    STORE_FINAL, True)


def chunk_kda_int8(q, k, v, g, beta, scales, *, sigma=1.0, tau=1.0, eta=0.0, scale=None, chunk_size=64,
                   output_final_state=False, unrot=None, rot=None, fla_prep=True, bits=8, out_dtype=None):
    """q,k [B,T,H,K], v [B,T,H,V] (fp16/bf16), g [B,T,H,K] fp32 per-channel log-decay, beta [B,T,H].
    scales: 'q' (H,), 'k' (H,) [Hw local], 'kw' (H,K) [Hw write], 'w' (H,) [W read], 'el'/'ew' (H,V) [E local /
    write], 'state' (H,V). rot: (K,K) R_QK for the anchored local operands; unrot: (V,V) R_Vᵀ (canonical storage).
    Returns (o [B,T,H,V] in v.dtype, final fp32 state or None)."""
    q, k, v, beta = q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous()
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    assert T % BT == 0 and K <= 128 and BT % BC == 0
    scale = K**-0.5 if scale is None else scale
    qmax = float(2 ** (bits - 1) - 1)  # code range: 127 for INT8, 31 for INT6 codes in INT8 containers
    gc = g.float().view(B, NT, BT, H, K).cumsum(dim=2)
    logenv = (gc[:, :, -1:] - gc).amax(-1).reshape(B, T, H).contiguous()
    gc = gc.reshape(B, T, H, K).contiguous()
    if fla_prep and sigma == 1.0 and tau == 1.0:  # FLA's placement: reuse its fused prep kernels
        Hw, W, U, fw = _prep_fla(q, k, v, gc, beta, scale, BT)
    else:
        Hw, W, U, fw = _prep(k, v, gc, beta, sigma=sigma, tau=tau, chunk_size=BT)
    A = torch.zeros(B, T, H, BT, dtype=torch.float32, device=q.device)
    BKf = triton.next_power_of_2(K)
    NC = BT // BC
    R = q if rot is None else rot.to(torch.float16).contiguous()
    _gla_A_inter[(NT, NC * NC, B * H)](q, Hw, gc, A, R, scales["q"], scales["k"], float(scale), qmax, T=T, H=H, K=K, BT=BT,
                                       BC=BC, BK=BKf, NC=NC, USE_ROT=rot is not None, num_warps=4)
    _gla_A_intra[(NT, NC, B * H)](q, Hw, gc, A, scales["q"], scales["k"], float(scale), qmax, T=T, H=H, K=K, BT=BT, BC=BC,
                                  BK=BKf, num_warps=4)
    h_i8 = torch.empty(B, NT, H, K, V, dtype=torch.int8, device=q.device)
    e_out = torch.empty_like(v)
    ht = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device) if output_final_state else None
    o = torch.empty(v.shape, dtype=out_dtype or v.dtype, device=v.device)  # fp32 for an exit un-rotation
    if unrot is None:
        grid_h = lambda meta: (triton.cdiv(V, meta["BV"]), B * H)  # noqa: E731
        _kda_state_kernel[grid_h](Hw, W, U, fw, gc, logenv, h_i8, e_out, ht, q, scales["kw"], scales["w"], scales["ew"],
                                  scales["state"], float(eta), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=BKf, STORE_FINAL=ht is not None)
        grid_o = lambda meta: (triton.cdiv(V, meta["BV"]), NT, B * H)  # noqa: E731
        _gla_output_kernel[grid_o](q, e_out, gc, A, h_i8, o, q, scales["q"], scales["el"], scales["state"], float(scale), qmax,
                                   T=T, H=H, K=K, V=V, BT=BT)
        return o, ht
    assert V in (64, 128), "canonical storage needs one V-tile spanning all value channels"
    Ru = unrot.to(torch.float16).contiguous()
    _kda_state_kernel_unrot[(1, B * H)](Hw, W, U, fw, gc, logenv, h_i8, e_out, ht, Ru, scales["kw"], scales["w"],
                                        scales["ew"], scales["state"], float(eta), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=BKf,
                                        BV=V, STORE_FINAL=ht is not None, num_warps=4, num_stages=1)
    _gla_output_kernel_unrot[(1, NT, B * H)](q, e_out, gc, A, h_i8, o, Ru, scales["q"], scales["el"], scales["state"],
                                             float(scale), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=64, BV=V, num_warps=4,
                                             num_stages=2)
    return o, ht
