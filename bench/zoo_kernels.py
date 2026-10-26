"""Zoo kernel timing (the kernel panel of the overhead figure): FLA bf16 chunk kernels vs the zoo INT8 kernels (naive INT8 vs INT8 + invariances, same
kernel) for the five families at the zoo cells' shapes (H=8, K=V=128, T=2048).

    python bench/zoo_kernels.py [--batches 1 4 8] [--T 2048] [--out results/zoo/kernel/timing.jsonl]
Times include the torch-side pieces of our path (chunk cumsum, WY prep, operand rotations), i.e. what the
in-model kernel row costs. CUDA-event timing, autotune warmup, median of --reps.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch

from linquant.kernels.zoo.gated_delta_rule import chunk_delta_int8
from linquant.kernels.zoo.gla import chunk_gla_int8
from linquant.kernels.zoo.kda import chunk_kda_int8
from linquant.kernels.zoo.simple_gla import chunk_simple_gla_int8
from linquant.reference import hadamard_rotation


def timeit(fn, reps, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def timeit_q(fn, reps, warmup=10):
    """(median, p25, p75) in ms over reps CUDA-event timings."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    return ts[n // 2], ts[n // 4], ts[(3 * n) // 4]


def make_inputs(B, T, H, K, V, decay, delta, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).to(device, torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).to(device, torch.bfloat16)
    v = torch.randn(B, T, H, V, generator=g).to(device, torch.bfloat16)
    if decay == "scalar":
        gd = (torch.nn.functional.logsigmoid(torch.randn(B, T, H, generator=g) * 3) / 4).to(device)
    elif decay == "diagonal":
        gd = (torch.nn.functional.logsigmoid(torch.randn(B, T, H, K, generator=g) * 3) / 4).to(device)
    else:
        gd = None
    beta = torch.rand(B, T, H, generator=g).to(device) if delta else None
    return q, k, v, gd, beta


def ones_scales(H, K, V, device):
    one = lambda *s: torch.ones(*s, device=device)  # noqa: E731
    return {"q": one(H), "k": one(H), "kw": one(H, K), "vw": one(H, V), "vl": one(H, V), "state": one(H, V),
            "w": one(H), "el": one(H, V), "ew": one(H, V)}


def family_fns(fam, decay):
    """(fla bf16 fn, ours) with a common call signature fn(q,k,v,g,beta, *, rot, unrot)."""
    import fla.ops.delta_rule as dr
    import fla.ops.gated_delta_rule as gdr
    import fla.ops.gla as gla
    import fla.ops.kda as kda
    import fla.ops.simple_gla as sg

    scale = None
    if fam == "linear" and decay != "diagonal":
        ref = lambda q, k, v, g, b: sg.chunk_simple_gla(q, k, v, g, scale=scale)[0]  # noqa: E731
        ours = lambda q, k, v, g, b, sc, rot, unrot: chunk_simple_gla_int8(q, k, v, g, sc, eta=1.0, unrot=unrot)[0]  # noqa: E731
    elif fam == "linear":
        ref = lambda q, k, v, g, b: gla.chunk_gla(q, k, v, g, scale=scale)[0]  # noqa: E731
        ours = lambda q, k, v, g, b, sc, rot, unrot: chunk_gla_int8(q, k, v, g, sc, eta=0.0, rot=rot, unrot=unrot)[0]  # noqa: E731
    elif decay == "none":
        ref = lambda q, k, v, g, b: dr.chunk_delta_rule(q, k, v, b, scale=scale)[0]  # noqa: E731
        ours = lambda q, k, v, g, b, sc, rot, unrot: chunk_delta_int8(q, k, v, None, b, sc, unrot=unrot)[0]  # noqa: E731
    elif decay == "scalar":
        ref = lambda q, k, v, g, b: gdr.chunk_gated_delta_rule(q, k, v, g, b, scale=scale)[0]  # noqa: E731
        ours = lambda q, k, v, g, b, sc, rot, unrot: chunk_delta_int8(q, k, v, g, b, sc, unrot=unrot)[0]  # noqa: E731
    else:
        ref = lambda q, k, v, g, b: kda.chunk_kda(q, k, v, g, b, scale=scale)[0]  # noqa: E731
        ours = lambda q, k, v, g, b, sc, rot, unrot: chunk_kda_int8(q, k, v, g, b, sc, rot=rot, unrot=unrot)[0]  # noqa: E731
    return ref, ours


FAMILIES = [("LA", "linear", "none"), ("Mamba-2", "linear", "scalar"), ("GLA", "linear", "diagonal"),
            ("DeltaNet", "delta", "none"), ("GDN", "delta", "scalar"), ("KDA", "delta", "diagonal")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="results/zoo/kernel/timing.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name()
    H, K, V = args.H, args.K, args.V
    sc = ones_scales(H, K, V, dev)
    Rk, Rv = hadamard_rotation(K, 1, dev), hadamard_rotation(V, 2, dev)
    for name, fam, decay in FAMILIES:
        ref, ours = family_fns(fam, decay)
        for B in args.batches:
            q, k, v, g, beta = make_inputs(B, args.T, H, K, V, decay, fam == "delta", dev)
            persistent = decay != "diagonal"
            Rkb, Rvb = Rk.to(q.dtype), Rv.to(v.dtype)

            def prologue():  # operand rotations as a deployment would run them (bf16 GEMMs, not fused yet)
                qq, kk = (q @ Rkb, k @ Rkb) if persistent else (q, k)
                return qq, kk, v @ Rvb

            def prologue_qk():  # the key-side stack's only runtime cost: R_QK on q and k (D_qk and the absorptions fold into scales)
                return (q @ Rkb, k @ Rkb) if persistent else (q, k)

            qr, kr, vr = prologue()
            t_ref, t_ref25, t_ref75 = timeit_q(lambda: ref(q, k, v, g, beta), args.reps)
            t_naive, t_naive25, t_naive75 = timeit_q(lambda: ours(q, k, v, g, beta, sc, None, None), args.reps)
            # recommended stack without R_V: rotated q/k (persistent) or in-kernel local R_QK (diagonal decay), no un-rotation
            t_key_k, t_key25, t_key75 = timeit_q(lambda: ours(qr, kr, v, g, beta, sc, None if persistent else Rk, None), args.reps)
            t_pro_qk = timeit(prologue_qk, args.reps) if persistent else 0.0
            t_prep = None
            if fam == "delta":  # torch-side WY prep (Gram, solve, W/U) — the part FLA fuses into Triton
                from linquant.kernels.zoo import gated_delta_rule as gdr_mod, kda as kda_mod

                if decay == "diagonal":
                    gc = g.view(B, args.T // 64, 64, H, K).cumsum(dim=2).reshape(B, args.T, H, K).contiguous()
                    t_prep = timeit(lambda: kda_mod._prep_fla(q, k, v, gc, beta, K**-0.5, 64), args.reps)
                else:
                    gcs = g.view(B, args.T // 64, 64, H).cumsum(dim=2).reshape(B, args.T, H).contiguous() if g is not None else None
                    t_prep = timeit(lambda: gdr_mod._prep_fla(k, v, gcs, beta, 64), args.reps)
            t_pro = timeit(prologue, args.reps)
            t_stack_k, t_stack25, t_stack75 = timeit_q(lambda: ours(qr, kr, vr, g, beta, sc, None if persistent else Rk, Rv.T), args.reps)
            # rotated storage: same rotated operands, no un-rotation inside the call (state and output stay in the rotated
            # basis); the one inverse transform is a bf16 GEMM on the chunk output, timed separately
            t_rot_k, t_rot25, t_rot75 = timeit_q(lambda: ours(qr, kr, vr, g, beta, sc, None if persistent else Rk, None), args.reps)
            o_rot = ours(qr, kr, vr, g, beta, sc, None if persistent else Rk, None)
            RvT = Rv.T.to(o_rot.dtype).contiguous()
            t_unrot_out = timeit(lambda: o_rot @ RvT, args.reps)
            rec = dict(family=name, B=B, T=args.T, H=H, K=K, V=V, gpu=gpu, fla_bf16_ms=round(t_ref, 4),
                       fla_bf16_p25=round(t_ref25, 4), fla_bf16_p75=round(t_ref75, 4),
                       int8_naive_ms=round(t_naive, 4), int8_naive_p25=round(t_naive25, 4), int8_naive_p75=round(t_naive75, 4),
                       int8_keystack_kernel_ms=round(t_key_k, 4), int8_keystack_p25=round(t_key25, 4), int8_keystack_p75=round(t_key75, 4),
                       prologue_qk_ms=round(t_pro_qk, 4), int8_keystack_ms=round(t_key_k + t_pro_qk, 4),
                       int8_stacked_kernel_ms=round(t_stack_k, 4), int8_stacked_p25=round(t_stack25, 4), int8_stacked_p75=round(t_stack75, 4),
                       prologue_rot_ms=round(t_pro, 4), int8_stacked_ms=round(t_stack_k + t_pro, 4),
                       int8_rotstore_kernel_ms=round(t_rot_k, 4), int8_rotstore_p25=round(t_rot25, 4), int8_rotstore_p75=round(t_rot75, 4),
                       unrot_out_ms=round(t_unrot_out, 4), int8_rotstore_ms=round(t_rot_k + t_unrot_out + t_pro, 4),
                       prep_ms=None if t_prep is None else round(t_prep, 4),
                       speedup_naive=round(t_ref / t_naive, 3), speedup_stacked_kernel=round(t_ref / t_stack_k, 3),
                       speedup_stacked=round(t_ref / (t_stack_k + t_pro), 3),
                       host=socket.gethostname(), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            print("RECORD", json.dumps(rec), flush=True)
            with open(args.out.replace(".jsonl", f".{socket.gethostname()}.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
