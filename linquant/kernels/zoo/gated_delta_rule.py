"""INT8 chunk_gated_delta_rule / chunk_delta_rule (delta family, no or scalar decay).

Structure (FLA): WY prep (Gram, triangular solve, W = T·L₊, U = T·(β^σ V)) → state kernel with the pseudo-values
E = U − W·S computed inside the chunk loop → output kernel. The prep is FLA's fused Triton prep for its own
placement (σ=τ=1) and fp32 torch otherwise (the σ/τ placements are operand relabelings there). Quantized: W (per head) against the INT8 stored state, the readout /
write key factor Hw (per head for the local product, per channel for the write), E (per channel, local and write),
q (per head). Matches linquant.reference.chunk_core with policy `kernel_static`, `state=8`, `spass=fp32carry`,
`readpost=1`, `kernel_arith`. The output kernel is shared with simple_gla (k → Hw, v → E).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .simple_gla import _output_kernel, _output_kernel_unrot, _q8


def _prep(q, k, v, g, beta, *, sigma, tau, chunk_size):
    """fp32 WY prep in torch. Returns (L, Hw, W, U, fw): Gram left factor (unused after), readout/write key
    factor Hw = β^((1−σ)τ) k, state-read operand W = T·(β^σ k ⊙ exp(gc)), pseudo-value RHS U = T·(β^σ v), and the
    value-side share fw = β^((1−σ)(1−τ)) of the outer factor (applied to E in the state kernel)."""
    B, T, H, K = k.shape
    C = chunk_size
    nc = T // C
    kf, vf = k.float(), v.float()
    bt = beta.float().unsqueeze(-1)  # [B,T,H,1]
    gc = g.float().view(B, nc, C, H).cumsum(dim=2).view(B, T, H) if g is not None else None
    Lc = (bt**sigma * kf).view(B, nc, C, H, K).transpose(2, 3)  # [B,nc,H,C,K]
    Hg = (bt ** (1 - sigma) * kf).view(B, nc, C, H, K).transpose(2, 3)
    G = Lc @ Hg.transpose(-1, -2)  # [B,nc,H,C,C]
    tril = torch.ones(C, C, dtype=torch.bool, device=k.device).tril(-1)
    if gc is not None:
        gcc = gc.view(B, nc, C, H).transpose(2, 3)  # [B,nc,H,C]
        lam = (gcc.unsqueeze(-1) - gcc.unsqueeze(-2)).masked_fill(~tril, float("-inf")).exp()
        G = G * lam
    else:
        G = G * tril
    eye = torch.eye(C, dtype=torch.float32, device=k.device)
    Tm = torch.linalg.solve_triangular(eye + G, eye.expand(G.shape).clone(), upper=False, unitriangular=True)
    pq = gcc.exp().unsqueeze(-1) if gc is not None else 1.0
    W = (Tm @ (Lc * pq)).transpose(2, 3).reshape(B, T, H, K)
    U = (Tm @ (bt**sigma * vf).view(B, nc, C, H, -1).transpose(2, 3)).transpose(2, 3).reshape(B, T, H, -1)
    Hw = bt ** ((1 - sigma) * tau) * kf
    fw = (bt ** ((1 - sigma) * (1 - tau))).view(B, T, H)
    return Hw.to(k.dtype), W.to(k.dtype), U.to(v.dtype), fw.contiguous(), gc


@triton.jit
def _delta_state_body(
    kw, w, u, fw, g, h_i8, e_out, ht, unrot,
    skw, sw, ew, ss,  # (H,K) write key scale, (H,) W scale, (H,V) E write scale, (H,V) state scale
    eta,
    qmax,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, STORE_FINAL: tl.constexpr, USE_UNROT: tl.constexpr,
):
    """One program per (V-tile, batch·head), full K in registers (E = U − W·S needs the whole K reduction).
    USE_UNROT (canonical storage under R_V, "align"): U arrives in the rotated value basis; the dequantized
    state read W·S is rotated by R_V (fp16 dot) so E stays rotated, and the dequantized write product is
    un-rotated by R_Vᵀ before it joins the canonical fp32 carry. Needs BV == V (`_delta_state_kernel_unrot`)."""
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
    b_h = tl.zeros([BK, BV], dtype=tl.float32)  # fp32 carry
    for i_t in range(NT):
        o_h = ((i_n * NT + i_t) * H + i_h).to(tl.int64) * K * V
        p_h = tl.make_block_ptr(h_i8 + o_h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        hq = _q8(b_h, inv_ss[None, :], 127.0)  # INT8 stored copy: read by this chunk (state read and E) and the output kernel
        tl.store(p_h, hq, boundary_check=(0, 1))
        p_kw = tl.make_block_ptr(kw + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_t * BT), (BK, BT), (0, 1))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_e = tl.make_block_ptr(e_out + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_kw = tl.load(p_kw, boundary_check=(0, 1)).to(tl.float32)  # (BK, BT)
        b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)  # (BT, BK)
        b_u = tl.load(p_u, boundary_check=(0, 1)).to(tl.float32)  # (BT, BV)
        o_t = i_t * BT + tl.arange(0, BT)
        b_fw = tl.load(fw + bos * H + o_t * H + i_h, mask=o_t < T, other=1.0)  # (BT,)
        # pseudo-values E = (U − W·S_read) ⊙ fw ; W per-head INT8 against the INT8 stored state
        wq = _q8(b_w, inv_sw, qmax)
        b_ws = tl.dot(wq, hq, out_dtype=tl.int32).to(tl.float32) * b_sw * b_ss[None, :]
        if USE_UNROT:  # canonical state read → rotated value basis: (W·S)·R_V, R_V = unrotᵀ read through swapped strides
            p_rot = tl.make_block_ptr(unrot, (V, V), (1, V), (0, 0), (BV, BV), (0, 1))
            b_ws = tl.dot(b_ws.to(tl.float16), tl.load(p_rot, boundary_check=(0, 1))).to(tl.float32)
        b_e = (b_u - b_ws) * b_fw[:, None]
        tl.store(p_e, b_e.to(p_e.dtype.element_ty), boundary_check=(0, 1))  # fp16 E for the output kernel
        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            b_g = tl.load(g + bos * H + o_t * H + i_h, mask=o_t < T, other=0.0)
            d = b_g_last - b_g
            b_kw = b_kw * tl.exp((1.0 - eta) * d)[None, :]
            b_e = b_e * tl.exp(eta * d)[:, None]
            b_h *= tl.exp(b_g_last)
        kq = _q8(b_kw, inv_skw[:, None], qmax)
        eq = _q8(b_e, inv_ew[None, :], qmax)
        acc = tl.dot(kq, eq, out_dtype=tl.int32).to(tl.float32) * b_skw[:, None] * b_ew[None, :]
        if USE_UNROT:
            p_un = tl.make_block_ptr(unrot, (V, V), (V, 1), (0, 0), (BV, BV), (1, 0))
            acc = tl.dot(acc.to(tl.float16), tl.load(p_un, boundary_check=(0, 1))).to(tl.float32)  # → canonical
        b_h += acc
    if STORE_FINAL:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({"BV": BV}, num_warps=w, num_stages=st) for BV in (64, 128) for w in (4, 8) for st in (1, 2)],
    key=["BT", "USE_G"],
)
@triton.jit(do_not_specialize=["T"])
def _delta_state_kernel(kw, w, u, fw, g, h_i8, e_out, ht, unrot, skw, sw, ew, ss, eta, qmax, T, H: tl.constexpr, K: tl.constexpr,
                        V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr,
                        STORE_FINAL: tl.constexpr):
    _delta_state_body(kw, w, u, fw, g, h_i8, e_out, ht, unrot, skw, sw, ew, ss, eta, qmax, T, H, K, V, BT, BK, BV, USE_G,
                      STORE_FINAL, False)


@triton.jit(do_not_specialize=["T"])
def _delta_state_kernel_unrot(kw, w, u, fw, g, h_i8, e_out, ht, unrot, skw, sw, ew, ss, eta, qmax, T, H: tl.constexpr,
                              K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                              USE_G: tl.constexpr, STORE_FINAL: tl.constexpr):
    tl.static_assert(BV == V, "canonical storage needs one V-tile spanning all value channels")
    _delta_state_body(kw, w, u, fw, g, h_i8, e_out, ht, unrot, skw, sw, ew, ss, eta, qmax, T, H, K, V, BT, BK, BV, USE_G,
                      STORE_FINAL, True)


def _prep_fla(k, v, gc, beta, chunk_size):
    """FLA's fused Triton WY prep (kkt + solve_tril + recompute_w_u) for the FLA placement σ=τ=1: W = T·(βk⊙exp(g)),
    U = T·(βv), Hw = k, fw = 1. T is materialized in k's dtype inside FLA (the fp32 torch prep keeps it fp32)."""
    B, T, H, K = k.shape
    b = beta.to(k.dtype)
    try:  # FLA ≥ 0.5: fused kkt + solve_tril + recompute_w_u
        from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra

        w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=gc, beta=b, chunk_size=chunk_size, use_exp2=False)
    except ImportError:  # FLA 0.4.x (cluster env): the same three kernels unfused, as chunk_gated_delta_rule_fwd does
        from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
        from fla.ops.utils import solve_tril

        A = chunk_scaled_dot_kkt_fwd(k=k, g=gc, beta=b, chunk_size=chunk_size, output_dtype=torch.float32)
        A = solve_tril(A=A, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(k=k, v=v, beta=b, A=A, g=gc)
    return k, w.to(k.dtype), u.to(v.dtype), torch.ones(B, T, H, device=k.device)


def chunk_delta_int8(q, k, v, g, beta, scales, *, sigma=1.0, tau=1.0, eta=1.0, scale=None, chunk_size=64,
                     output_final_state=False, unrot=None, fla_prep=True, bits=8, out_dtype=None):
    """q,k [B,T,H,K], v [B,T,H,V] (fp16/bf16), g [B,T,H] fp32 log-decay or None (DeltaNet), beta [B,T,H].
    scales: 'q' (H,) [on q·scale], 'k' (H,) [Hw, local], 'kw' (H,K) [Hw write], 'w' (H,) [W read],
    'el' (H,V) [E local], 'ew' (H,V) [E write], 'state' (H,V). unrot: optional (V,V) R_Vᵀ for canonical storage
    when v was rotated by R_V (E stays rotated, state and output canonical). Returns (o, final state or None)."""
    q, k, v, beta = q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous()  # dense [B,T,H,·] layouts
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    assert T % BT == 0 and K <= 128
    scale = K**-0.5 if scale is None else scale
    qmax = float(2 ** (bits - 1) - 1)  # code range: 127 for INT8, 31 for INT6 codes in INT8 containers
    if fla_prep and sigma == 1.0 and tau == 1.0:  # FLA's placement: reuse its fused prep kernels
        gc = g.float().view(B, NT, BT, H).cumsum(dim=2).reshape(B, T, H).contiguous() if g is not None else None
        Hw, W, U, fw = _prep_fla(k, v, gc, beta, BT)
    else:
        Hw, W, U, fw, gc = _prep(q, k, v, g, beta, sigma=sigma, tau=tau, chunk_size=BT)
    gg = gc.contiguous() if gc is not None else None
    h_i8 = torch.empty(B, NT, H, K, V, dtype=torch.int8, device=q.device)
    e_out = torch.empty_like(v)
    ht = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device) if output_final_state else None
    BK = triton.next_power_of_2(K)
    o = torch.empty(v.shape, dtype=out_dtype or v.dtype, device=v.device)  # fp32 for an exit un-rotation
    if unrot is None:
        grid_h = lambda meta: (triton.cdiv(V, meta["BV"]), B * H)  # noqa: E731
        _delta_state_kernel[grid_h](Hw, W, U, fw, gg, h_i8, e_out, ht, q, scales["kw"], scales["w"], scales["ew"],
                                    scales["state"], float(eta), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=BK, USE_G=g is not None,
                                    STORE_FINAL=ht is not None)
        grid_o = lambda meta: (triton.cdiv(V, meta["BV"]), NT, B * H)  # noqa: E731
        _output_kernel[grid_o](q, Hw, e_out, h_i8, gg, o, q, scales["q"], scales["k"], scales["el"], scales["state"],
                               float(scale), qmax, T=T, H=H, K=K, V=V, BT=BT, USE_G=g is not None)
        return o, ht
    # canonical storage under R_V ("align"): fixed configs with the V tile spanning all value channels; the chunk
    # loop is not software-pipelined (the two (V,V) rotation operands would be multi-buffered past the smem limit)
    assert V in (64, 128), "canonical storage needs one V-tile spanning all value channels"
    R = unrot.to(torch.float16).contiguous()
    _delta_state_kernel_unrot[(1, B * H)](Hw, W, U, fw, gg, h_i8, e_out, ht, R, scales["kw"], scales["w"], scales["ew"],
                                          scales["state"], float(eta), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=V,
                                          USE_G=g is not None, STORE_FINAL=ht is not None, num_warps=4, num_stages=1)
    _output_kernel_unrot[(1, NT, B * H)](q, Hw, e_out, h_i8, gg, o, R, scales["q"], scales["k"], scales["el"],
                                          scales["state"], float(scale), qmax, T=T, H=H, K=K, V=V, BT=BT, BK=64, BV=V,
                                          USE_G=g is not None, num_warps=4, num_stages=2)
    return o, ht
