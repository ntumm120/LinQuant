"""Torch reference chunk core for the six-cell grid, with rotation / decay-split / quant hooks.

FLA conventions: q,k [B,T,H,K], v [B,T,H,V]; state S [K,V] per head; o_t = S_t^T-read via q_t;
S_t = decay_t * S_{t-1} + k_t v_t^T (linear) or the delta-rule update. Log-decay g is per step
([B,T,H] scalar, [B,T,H,K] diagonal). All math in float64; quantization enters only through `qz`.

Transformations:
  rot_qk  persistent K-axis rotation of q,k (no/scalar decay) or post-weighting local rotation of
          the intra-chunk contraction operands (diagonal decay)
  rot_v   persistent V-axis rotation of v with exact un-rotation of the output
  eta     terminal-write decay split: key gets Omega^(1-eta), value gets Omega^eta (scalar envelope
          for diagonal decay)
  rho     delta-rule beta split: Gram left factor beta^rho k, right factor beta^(1-rho) k
  tau     delta-rule placement of the outer beta factor (1 = all on the key)
"""

from __future__ import annotations

import torch

BC = 16  # diagonal-decay sub-block (mirrors FLA's secondary chunking)


def chunk_core(q, k, v, g, beta, *, family, decay, scale, qz, rot_qk=None, rot_v=None, eta=0.0, rho=1.0, C=64,
               read_post=False, storage="rotated", tau=1.0, out_dtype=None, act_dtype=None):
    """Chunkwise forward for one (family, decay) cell. Returns o [B,T,H,V] in q's dtype.

    storage: with rot_v, "rotated" stores the state as S·R_V (production); "canonical" un-rotates every
    value-side product on its dequantized accumulator before it is combined or stored, so the state and the
    output are canonical and no exit un-rotation exists (delta cells rotate the canonical incoming state on
    read and keep E in the rotated basis).
    State pass (zoo kernel lineage): the carry stays fp32 across chunks; with ``qz.state_bits`` only the stored
    copy the next chunk reads is INT8."""
    B, T, H, K = q.shape
    odt = out_dtype or q.dtype
    assert T % C == 0, f"T={T} must be a multiple of the chunk size {C}"
    dt = torch.float64
    # kernel lineages materialize rotated / prepared operands in the activation dtype before quantizing them
    act_dt = act_dtype or q.dtype  # the model's activation dtype (callers may hand over float64 operands)
    mat = (lambda x: x.float().to(act_dt).to(dt)) if getattr(qz, "kernel_arith", False) else (lambda x: x)
    q, k, v = q.to(dt), k.to(dt), v.to(dt)
    persistent = rot_qk is not None and decay != "diagonal"
    if persistent:
        q, k = mat(q @ rot_qk), mat(k @ rot_qk)
    if rot_v is not None:
        v = mat(v @ rot_v)
    q = q * scale
    nc = T // C
    Q = q.reshape(B, nc, C, H, K).permute(0, 3, 1, 2, 4)  # [B,H,NC,C,K]
    Kx = k.reshape(B, nc, C, H, K).permute(0, 3, 1, 2, 4)
    Vx = v.reshape(B, nc, C, H, -1).permute(0, 3, 1, 2, 4)
    gc = None
    if g is not None:
        g = g.to(dt)
        gc = g.reshape(B, nc, C, H, *g.shape[3:]).movedim(3, 1)
        # kernel lineages compute the chunk-local cumulative decay in fp32
        gc = gc.float().cumsum(dim=3).to(dt) if getattr(qz, "kernel_arith", False) else gc.cumsum(dim=3)  # [B,H,NC,C(,K)]
    bt = beta.to(dt).reshape(B, nc, C, H).permute(0, 3, 1, 2).unsqueeze(-1) if beta is not None else None
    S = q.new_zeros(B, H, K, Vx.shape[-1])
    carry = None  # fp32 state-pass accumulator (INT8 stored state only)
    out = torch.empty_like(Vx)
    local_rot = rot_qk if (rot_qk is not None and decay == "diagonal") else None
    unrot = rot_v.T if (rot_v is not None and storage == "canonical") else None  # applied to value-side products
    for c in range(nc):
        gcc = gc[:, :, c] if gc is not None else None
        args = (Q[:, :, c], Kx[:, :, c], Vx[:, :, c], gcc, bt[:, :, c] if bt is not None else None, S, carry)
        fn = _chunk_linear if family == "linear" else _chunk_delta
        out[:, :, c], S, carry = fn(*args, decay=decay, qz=qz, local_rot=local_rot, eta=eta, rho=rho,
                                    read_post=read_post, unrot=unrot, tau=tau, mat=mat)
    o = out.permute(0, 2, 3, 1, 4).reshape(B, T, H, -1)
    if rot_v is not None and unrot is None:
        o = o @ rot_v.T  # exit un-rotation (rotated storage)
    return o.to(odt)


# ----------------------------------------------------------------------------- stored state
def _state_update(S, carry, gam, local, qz):
    """Returns (S_read, carry). S_read is the state the next chunk reads; carry is the pass's own accumulator.

    fp state: S_read = gam*S + local, carry None.
    INT8 state (zoo kernel lineage): the carry stays fp32 across chunks and only the stored copy the next chunk
    reads is INT8 (no error feedback into the carry).
    """
    if not qz.state_bits:
        return gam * S + local, None
    if qz.calibrating:
        # zoo kernels quantize the stored copy from the fp32 carry: the (H,V) state scale observes the carry
        # (calibrating it on the chunk product, SSDi8's convention for its INT8 carry, saturates the copy for
        # slowly forgetting cells — and more so under R_QK, which spreads the product's outlier rows)
        carry = (gam * S + local) if carry is None else (gam * carry + local)
        qz("state", carry)
        return carry, carry
    sc = qz.state_scale(qz.layer, S)
    if sc is None:
        return gam * S + local, None
    qmax = 2 ** (qz.state_bits - 1) - 1
    carry = (gam * S + local) if carry is None else (gam * carry + local)
    if getattr(qz, "kernel_arith", False):
        carry = carry.float().to(carry.dtype)  # fp32 carry in the kernel
        codes = qz.round((carry.float() * (1.0 / sc.float())).double())
    else:
        codes = qz.round(carry / sc)
    return codes.clamp(-qmax, qmax) * sc, carry


# ----------------------------------------------------------------------------- decay helpers
def _decay_terms(gcc, decay, C, K, like):
    """Returns (pq, Om, gam, lam): incoming-read weights, terminal-write weights, state decay, mask."""
    tril = torch.ones(C, C, dtype=torch.bool, device=like.device).tril()
    if gcc is None:
        one = like.new_ones(1, 1, 1, 1)
        return one, one, one, tril.to(like.dtype)
    if decay == "scalar":
        pq = gcc.exp().unsqueeze(-1)  # [B,H,C,1]
        Om = (gcc[..., -1:] - gcc).exp().unsqueeze(-1)
        gam = gcc[..., -1:].exp().unsqueeze(-1)
        lam = (gcc.unsqueeze(-1) - gcc.unsqueeze(-2)).masked_fill(~tril, -torch.inf).exp()  # mask before exp
        return pq, Om, gam, lam
    pq = gcc.exp()  # [B,H,C,K]
    Om = (gcc[..., -1:, :] - gcc).exp()
    gam = gcc[..., -1, :].unsqueeze(-1)  # [B,H,K,1] -> exp applied by caller
    return pq, Om, gam.exp(), None


def _local_prod(Xq, Xk, gcc, lam, decay, qz, rq, rk, local_rot, strict):
    """A_ij = sum_n Xq_in Xk_jn exp(gc_in - gc_jn), i>j (strict) or i>=j. Operands are quantized under the
    roles ``rq`` / ``rk``; None keeps an operand in fp (the delta Gram matrix, as in FLA)."""
    qq = (lambda x: x) if rq is None else (lambda x: qz(rq, x))
    qk = (lambda x: x) if rk is None else (lambda x: qz(rk, x))
    if decay != "diagonal":
        A = (qq(Xq) @ qk(Xk).transpose(-1, -2)) * lam
        return A.tril(-1) if strict else A
    C = Xq.shape[-2]
    A = Xq.new_zeros(*Xq.shape[:-1], C)
    quantized = any(r is not None and qz.scheme_for(r) != "none" for r in (rq, rk))
    for s in range(0, C, BC):
        rows = slice(s, s + BC)
        anchor = gcc[..., s : s + 1, :]
        if s > 0:
            Qb = Xq[..., rows, :] * (gcc[..., rows, :] - anchor).exp()
            Kb = Xk[..., :s, :] * (anchor - gcc[..., :s, :]).exp()
            if local_rot is not None and quantized:
                # kernels rotate the anchored fp32 operands with an fp16 dot (exact no-op on unquantized products)
                Qb, Kb = _un(Qb, local_rot, qz), _un(Kb, local_rot, qz)
            A[..., rows, :s] = qq(Qb) @ qk(Kb).transpose(-1, -2)
        gd = gcc[..., rows, :]
        n = gd.shape[-2]
        mask = torch.ones(n, n, dtype=torch.bool, device=gd.device).tril(-1 if strict else 0).unsqueeze(-1)
        w = (gd.unsqueeze(-2) - gd.unsqueeze(-3)).masked_fill(~mask, -torch.inf).exp()  # [...,i,j,n], masked before exp
        Ad = torch.einsum("...in,...jn,...ijn->...ij", qq(Xq[..., rows, :]), qk(Xk[..., rows, :]), w)
        A[..., rows, rows] = Ad.tril(-1 if strict else 0)
    return A


def _write_weights(Om, decay, eta):
    """(key factor, value factor) for the terminal write under the eta split."""
    if decay != "diagonal":
        return Om ** (1 - eta), Om**eta
    s = Om.amax(dim=-1, keepdim=True)  # scalar envelope per token
    kf = Om * s ** (-eta)
    if eta > 0:  # an envelope that underflowed to 0 gives 0 * inf: that token writes nothing, so its factor is 0
        kf = torch.where(s > 0, kf, torch.zeros_like(kf))
    return kf, s**eta


# ----------------------------------------------------------------------------- cells
def _read(Q, pq, S, decay, qz, read_post):
    """Incoming-state read. read_post (SSDi8): quantize bare Q, apply the decay after the product (scalar decay)."""
    if read_post and decay != "diagonal":
        return (qz("q_read", Q) @ S) * pq
    return qz("q_read", Q * pq) @ S


def _un(x, unrot, qz=None):
    if unrot is None:
        return x
    if qz is not None and getattr(qz, "kernel_arith", False):  # kernels un-rotate the fp32 product cast to fp16
        return (x.float().to(torch.float16).to(x.dtype) @ unrot.to(torch.float16).to(x.dtype)).float().to(x.dtype)
    return x @ unrot


def _local_value_product(A, Vq, qz):
    """A @ Vq; kernel lineages compute this as an fp16 dot (A rounded to fp16; INT8 codes are exact in fp16,
    the per-channel scale is applied after by the caller, fp32 accumulation)."""
    if getattr(qz, "local_fp16", False):
        A = A.float().to(torch.float16).to(A.dtype)  # the kernels hold A in fp32 before the fp16 cast
    return A @ Vq


def _chunk_linear(Q, Kx, Vx, gcc, bt, S, carry, *, decay, qz, local_rot, eta, rho, read_post, unrot, tau, mat):
    C, K = Q.shape[-2:]
    pq, Om, gam, lam = _decay_terms(gcc, decay, C, K, Q)
    Y = _read(Q, pq, S, decay, qz, read_post)  # canonical state under canonical storage → canonical read
    A = _local_prod(Q, Kx, gcc, lam, decay, qz, "q_local", "k_local", local_rot, strict=False)
    Y = Y + _un(_local_value_product(A, qz("v_local", Vx), qz), unrot, qz)  # product un-rotated before combining
    wk, wv = _write_weights(Om, decay, eta)
    local = _un(qz("k_write", Kx * wk).transpose(-1, -2) @ qz("v_write", Vx * wv), unrot, qz)
    S, carry = _state_update(S, carry, gam, local, qz)
    return Y, S, carry


def _chunk_delta(Q, Kx, Vx, gcc, bt, S, carry, *, decay, qz, local_rot, eta, rho, read_post, unrot, tau, mat):
    """rho = Gram exponent sigma (left factor beta^sigma k, right factor beta^(1-sigma) k, RHS beta^sigma v); the
    outer factor beta^(1-sigma) on the pseudo-values is split by tau: beta^((1-sigma)tau) onto the readout/write
    key factor, beta^((1-sigma)(1-tau)) onto the value factor. tau=1 = all on the key (FLA-compatible at rho=1).
    The Gram matrix, the triangular solve and the RHS stay fp (FLA's WY prep); W, Hm and E are quantized."""
    C, K = Q.shape[-2:]
    pq, Om, gam, lam = _decay_terms(gcc, decay, C, K, Q)
    L = bt**rho * Kx  # Gram left factor
    Hg = bt ** (1 - rho) * Kx  # Gram right factor
    Hm = mat(bt ** ((1 - rho) * tau) * Kx)  # readout/write key factor
    fw = bt ** ((1 - rho) * (1 - tau))  # value-side share of the outer factor
    G = _local_prod(L, Hg, gcc, lam, decay, qz, None, None, local_rot, strict=True)
    eye = torch.eye(C, dtype=Q.dtype, device=Q.device)
    Tm = torch.linalg.solve_triangular(eye + G, eye.expand(G.shape).clone(), upper=False, unitriangular=True)
    W = mat(Tm @ (L * pq))  # incoming-state read operand (kernels materialize W, U, Hm like FLA's WY prep)
    U = mat(Tm @ (bt**rho * Vx))  # in the value operand's basis (rotated if rot_v)
    if unrot is not None:  # canonical storage: rotate the canonical incoming-state read (W·S)·R_V, keep E rotated
        F = U - _un(qz("w_read", W) @ S, unrot.T, qz)  # kernels rotate the dequantized INT8 product (fp16 dot)
    else:
        F = U - qz("w_read", W) @ S
    F = F * fw
    Y = _read(Q, pq, S, decay, qz, read_post)
    A = _local_prod(Q, Hm, gcc, lam, decay, qz, "q_local", "k_local", local_rot, strict=False)
    Y = Y + _un(_local_value_product(A, qz("e_local", mat(F)), qz), unrot, qz)  # kernels hand E to the output pass in the activation dtype
    wk, wv = _write_weights(Om, decay, eta)
    local = _un(qz("k_write", Hm * wk).transpose(-1, -2) @ qz("e_write", F * wv), unrot, qz)
    S, carry = _state_update(S, carry, gam, local, qz)
    return Y, S, carry


def hadamard_rotation(n, seed, device=None):
    """Signed Hadamard rotation (orthogonal, n power of two) with a seeded random sign pattern."""
    Hm = torch.ones(1, 1, dtype=torch.float64)
    while Hm.shape[0] < n:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    gen = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=gen).double() * 2 - 1
    return (Hm / n**0.5 * signs).to(device)
