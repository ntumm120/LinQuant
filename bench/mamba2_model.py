# ruff: noqa: E501
r"""Full-model prefill timing for Mamba-2 (the prefill panel of the overhead figure).

Times a full forward over (batch x seqlen) for one timing config. Every mixer is replaced
with a random-weight build from bench/builders.py: random Marlin-packed W4 weights and
synthetic INT8 scales, so kernel paths and dispatch match a calibrated build while the
values are meaningless. For numerical results use scripts/mamba2_w4a8.py.

Usage:
    python bench/mamba2_model.py --config ours_key_r --model state-spaces/mamba2-1.3b \
        --output results/timing/model/ours_key_r.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builders import CONFIGS, build_model  # noqa: E402


def time_cuda(fn, warmup, repeats):
    """Time `fn` via CUDA events. Returns dict with median/p25/p75/min/max ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    n = len(times)
    return {
        "median_ms": times[n // 2],
        "p25_ms": times[max(0, n // 4)],
        "p75_ms": times[min(n - 1, 3 * n // 4)],
        "min_ms": times[0],
        "max_ms": times[-1],
        "n": n,
    }


def bench_prefill(model, input_ids, warmup, repeats):
    """Time a single forward pass over `input_ids`."""
    with torch.no_grad():
        return time_cuda(lambda: model(input_ids), warmup=warmup, repeats=repeats)


def main():
    """Run model-level prefill timing for one config and write JSON output."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--model", default="state-spaces/mamba2-1.3b")
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--prefill_lens", type=int, nargs="+", default=[512, 2048])
    ap.add_argument(
        "--long_context_lens",
        type=int,
        nargs="*",
        default=[4096, 8192],
        help="Extra prefill lengths run at B=1 only (long-context regime, "
        "matches SSDi8's App. H setup). Set to empty to disable.",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    device = "cuda"
    torch.cuda.set_device(0)

    t0 = time.time()
    model = build_model(args.config, model_id=args.model, device=device)
    build_s = time.time() - t0

    gpu_name = torch.cuda.get_device_name(0)
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"\n=== Model prefill timing: {args.config} ===")
    print(f"GPU: {gpu_name}  (free: {free_gb:.1f} GB)")
    print(f"Build time: {build_s:.1f}s")

    # Random input ids — vocab range from the embedding shape. Token *values*
    # don't affect kernel paths / shapes, just need to be valid indices.
    vocab_size = model.backbone.embedding.weight.shape[0]

    # Build the (B, T) cell list: cross-product on standard cells, plus B=1
    # long-context cells if requested.
    cells: list[tuple[int, int]] = [(B, T) for B in args.batch_sizes for T in args.prefill_lens]
    cells += [(1, T) for T in args.long_context_lens]

    max_T = max(T for _, T in cells)
    max_B = max(B for B, _ in cells)
    full_ids = torch.randint(0, vocab_size, (max_B, max_T), device=device, dtype=torch.long)

    torch.cuda.reset_peak_memory_stats()

    results = {
        "config": args.config,
        "model": args.model,
        "gpu": gpu_name,
        "build_seconds": build_s,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "prefill": {},
    }

    print("\n=== Prefill (B × T) ===")
    print(f"{'B':>3}  {'T':>5}  {'median(ms)':>12}  {'p25':>8}  {'p75':>8}  {'tok/s':>10}")
    for B, T in cells:
        input_ids = full_ids[:B, :T].contiguous()
        torch.cuda.reset_peak_memory_stats()
        try:
            st = bench_prefill(model, input_ids, args.warmup, args.repeats)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"{B:>3d}  {T:>5d}  OOM: {exc}")
            results["prefill"][f"B{B}_T{T}"] = {"error": "OOM"}
            torch.cuda.empty_cache()
            continue
        tok_per_sec = (B * T * 1000.0) / max(st["median_ms"], 1e-6)
        st["tokens_per_sec"] = tok_per_sec
        # Per-cell peak (weights + this cell's activations); the run-level peak below is the
        # max over the cells that fit, which is not comparable across configs when some OOM.
        st["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
        results["prefill"][f"B{B}_T{T}"] = st
        print(
            f"{B:>3d}  {T:>5d}  {st['median_ms']:>12.2f}  {st['p25_ms']:>8.2f}  {st['p75_ms']:>8.2f}  {tok_per_sec:>10.1f}"
        )

    results["peak_memory_mb"] = max(
        (v["peak_memory_mb"] for v in results["prefill"].values() if "peak_memory_mb" in v),
        default=0.0,
    )
    print(f"\nPeak memory: {results['peak_memory_mb']:.1f} MB")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
