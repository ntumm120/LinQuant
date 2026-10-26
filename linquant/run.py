"""Run a list of arms on one zoo model; append run records to a JSONL file.

Arm spec: comma-separated key=value, e.g. ``policy=kernel_static,bits=6,state=8,spass=fp32carry,readpost=1,rot=qk,eta=0.5``.
Usage: python -m linquant.run --cell mamba2 --size 340m --arms "q=none" "policy=kernel_static,bits=6,state=8,readpost=1,rot=qk"
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time

import torch

from linquant.harness import CELLS, Arm, calib_windows, calibrate, load_zoo, mean_ce, patch_zoo, wt2_ppl
from linquant.invariances import POLICIES, SCHEMES, FakeQuant


def parse_arm(spec):
    """rot=none|qk|v|qkv  q=<scheme> (global scheme for roles the policy leaves out)  bits=8  seed (rotation seed)
    policy=kernel_static|kernel_pt|kernel_ptk  state=8 (INT8 stored state)  spass=fp32carry  readpost=1
    storage=rotated|canonical  deltarv=align  eta rho tau (decay / beta placements)  sqk / sv (diagonal scaling exponents)
    """
    kv = dict(item.split("=", 1) for item in spec.split(",") if item)
    allowed = {"rot", "q", "bits", "eta", "rho", "seed", "policy", "state", "spass", "readpost", "storage", "deltarv",
               "tau", "sqk", "sv"}
    bad = set(kv) - allowed
    assert not bad, f"unknown arm keys {sorted(bad)} in {spec!r}"
    rot = kv.get("rot", "none")
    assert rot in ("none", "qk", "v", "qkv"), f"bad rot {rot!r}"
    assert kv.get("q", "none") in SCHEMES, f"bad scheme {kv.get('q')!r}"
    assert kv.get("spass", "fp32carry") == "fp32carry", f"bad spass {kv.get('spass')!r}"
    assert kv.get("storage", "rotated") in ("rotated", "canonical"), f"bad storage {kv.get('storage')!r}"
    assert kv.get("deltarv", "align") == "align", f"bad deltarv {kv.get('deltarv')!r}"
    assert kv.get("readpost", "0") in ("0", "1"), f"bad readpost {kv.get('readpost')!r}"
    kernel = kv.get("policy", "") != ""
    quant = FakeQuant(kv.get("q", "none"), int(kv.get("bits", 8)), ["ALL"], dict(POLICIES[kv["policy"]]) if kernel else None,
                      int(kv["state"]) if kv.get("state") else None, round_away=kernel)  # kernels round half away from zero
    quant.tie_roles = quant.local_fp16 = quant.kernel_arith = kernel  # one q tile, fp16 local value product, fp32 code arithmetic
    return Arm(rot_qk="qk" in rot, rot_v="v" in rot, eta=float(kv.get("eta", 0)), rho=float(kv.get("rho", 1)),
               rot_seed=int(kv.get("seed", 0)), quant=quant, read_post=kv.get("readpost", "0") == "1",
               storage=kv.get("storage", "rotated"), tau=float(kv.get("tau", 1)),
               smooth_qk=float(kv.get("sqk", 0)), smooth_v=float(kv.get("sv", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="mamba2", choices=list(CELLS))
    ap.add_argument("--size", default="340m")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--calib_n", type=int, default=8)
    ap.add_argument("--calib_seed", type=int, default=0)
    ap.add_argument("--calib_seqlen", type=int, default=2048, help="calibration window length")
    ap.add_argument("--sel_n", type=int, default=8)
    ap.add_argument("--sel_seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="max WT2 test windows")
    ap.add_argument("--out_dir", default="results/zoo", help="records go to <out_dir>/<size>/<cell>.<host>-<pid>.jsonl")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    cell, size = args.cell, args.size
    fam, dec = CELLS[cell]
    model, tok = load_zoo(cell, size)
    cal = calib_windows(tok, args.calib_n, args.calib_seed, seqlen=args.calib_seqlen)
    sel = calib_windows(tok, args.sel_n, args.sel_seed)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out_dir = os.path.join(args.out_dir, size)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{cell}.{socket.gethostname()}-{os.getpid()}.jsonl")  # one file per job
    holder = {"arm": None}
    patch_zoo(model, holder)
    for spec in args.arms:
        arm = holder["arm"] = parse_arm(spec)
        t0 = time.time()
        if arm.quant.scheme not in ("none", "row_dyn") or arm.quant.scheme_by_role or arm.quant.state_bits or arm.need_stats_pass():
            calibrate(model, arm, cal)
        holder["calls"] = 0
        sel_ce = mean_ce(model, sel)
        assert holder["calls"] > 0, "the patched recurrence was never called (fast-path/fallback not routed?)"
        ppl = wt2_ppl(model, tok, limit=args.limit)
        rec = dict(cell=cell, size=size, family=fam, decay=dec, arm=spec, **arm.record(),
                   calib=dict(n=args.calib_n, seed=args.calib_seed, sel_n=args.sel_n, sel_seed=args.sel_seed, seqlen=args.calib_seqlen),
                   sel_ce=sel_ce, wt2_ppl=ppl, limit=args.limit, elapsed_s=round(time.time() - t0, 1), git=git,
                   tag=args.tag, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        print("RECORD", json.dumps(rec), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
