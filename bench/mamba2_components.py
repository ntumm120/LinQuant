# ruff: noqa: E501
r"""Per-stage timing of one Mamba-2 mixer (the crossover panel of the overhead figure).

Decomposes the kernel-region (post-conv → pre-norm) into its sub-kernel stages
and times each one directly via CUDA events. Stages:

  preproc       — split + (optional) rotate + INT8 quant of x/B/C
                  (`fused_split_rotate_quant_xBC` for ours; per-(G,N) requant
                  for SSDi8; n/a for FP16)
  cumsum        — `_chunk_cumsum_fwd`
  chunk_state   — value-side: `_quant_chunk_state_fwd_xfp_static_int8`
                  key-side:    `_chunk_state_keyside_fwd`
                  SSDi8:       `_quant_chunk_state_fwd`
                  FP16:        `_chunk_state_fwd`
  state_passing — INT8 fixedpoint (ours) or FP (SSDi8, FP16)
  bmm           — C·Bᵀ chunk matrix (`_quant_bmm_chunk_fwd` / `_bmm_chunk_fwd`)
  chunk_scan    — final SSD output

How: monkey-patch each kernel function in the importer's namespace to wrap
itself with a CUDA-event recorder, then run `layer(x)` end-to-end and collect
per-stage event pairs. No orchestrator inlining.

Run one config per process (Triton autotune caches are per process).

Output JSON shape:
  {
    "config": ...,
    "by_shape": { "B{n}_T{m}": { "stage": median_us, ..., "total_us": ... } },
    "raw": { "B{n}_T{m}": { "stage": [µs per rep] } },
  }
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import builders  # noqa: E402
from builders import CONFIGS, MODEL_DIMS, build_layer, set_model_size  # noqa: E402

# (importer_module, attr_name, stage_label). Multiple (module, attr) entries
# can share a stage label — only one fires per forward (depending on path).
# Function-level patches cover (a) the wrapper stages (in_proj / conv1d /
# norm / out_proj are timed by patching the layer's instance methods, see
# ComponentTimer below) and (b) the SSD sub-kernels (patched here at
# module-attribute level).
OURS_PATCHES: list[tuple[str, str, str]] = [
    ("linquant.quantized_mamba2", "causal_conv1d_fn", "conv1d"),
    ("linquant.quantized_mamba2", "fused_split_rotate_quant_xBC", "preproc"),
    ("linquant.kernels.mamba2.ssd_combined", "_quant_chunk_cumsum_fwd", "cumsum"),
    (
        "linquant.kernels.mamba2.ssd_combined",
        "_quant_chunk_state_fwd_xfp_static_int8",
        "chunk_state",
    ),
    (
        "linquant.kernels.mamba2.ssd_combined",
        "_chunk_state_keyside_fwd",
        "chunk_state",
    ),
    (
        "linquant.kernels.mamba2.ssd_combined",
        "_fixedpoint_state_passing_fwd",
        "state_passing",
    ),
    ("linquant.kernels.mamba2.ssd_combined", "_quant_bmm_chunk_fwd", "bmm"),
    ("linquant.kernels.mamba2.ssd_combined", "_chunk_scan_v2", "chunk_scan"),
]

SSDI8_PATCHES: list[tuple[str, str, str]] = [
    ("ssdi8.qMamba2", "causal_conv1d_fn", "conv1d"),
    ("ssdi8.triton.quant_ssd_combined", "quantize_int8_per_group_triton", "preproc"),
    ("ssdi8.triton.quant_ssd_combined", "_quant_chunk_cumsum_fwd", "cumsum"),
    ("ssdi8.triton.quant_ssd_combined", "_quant_chunk_state_fwd", "chunk_state"),
    ("ssdi8.triton.quant_ssd_combined", "_quant_state_passing_fwd", "state_passing"),
    ("ssdi8.triton.quant_ssd_combined", "_quant_bmm_chunk_fwd", "bmm"),
    ("ssdi8.triton.quant_ssd_combined", "_quant_chunk_scan_fwd", "chunk_scan"),
]

FP16_PATCHES: list[tuple[str, str, str]] = [
    ("mamba_ssm.modules.mamba2", "causal_conv1d_fn", "conv1d"),
    ("mamba_ssm.ops.triton.ssd_combined", "_chunk_cumsum_fwd", "cumsum"),
    ("mamba_ssm.ops.triton.ssd_combined", "_chunk_state_fwd", "chunk_state"),
    ("mamba_ssm.ops.triton.ssd_combined", "_state_passing_fwd", "state_passing"),
    ("mamba_ssm.ops.triton.ssd_combined", "_bmm_chunk_fwd", "bmm"),
    ("mamba_ssm.ops.triton.ssd_combined", "_chunk_scan_fwd", "chunk_scan"),
]

# Per-block wrapper stages — timed via instance-method monkey-patches on the
# layer (not module-attribute patches, since these are bound methods and the
# linears use either `forward` or `forward_int8` depending on path). The
# ComponentTimer wraps both attrs and only the one called fires.
WRAPPER_METHODS: list[tuple[str, str, str]] = [
    ("in_proj", "forward", "in_proj"),
    ("in_proj", "forward_int8", "in_proj"),
    ("out_proj", "forward", "out_proj"),
    ("out_proj", "forward_int8", "out_proj"),
    ("norm", "forward", "norm"),
]

# Stage display order — pre-SSD wrappers, SSD sub-kernels, post-SSD wrappers.
STAGES = [
    "in_proj",
    "conv1d",
    "preproc",
    "cumsum",
    "chunk_state",
    "state_passing",
    "bmm",
    "chunk_scan",
    "norm",
    "out_proj",
]


def patches_for_config(cfg: str) -> list[tuple[str, str, str]]:
    """Return the (module, attr, stage) list for the given config."""
    if cfg == "fp16":
        return FP16_PATCHES
    if cfg == "ssdi8":
        return SSDI8_PATCHES
    return OURS_PATCHES


class ComponentTimer:
    """Monkey-patches kernel functions and layer-instance methods to record CUDA events.

    Two patch sources:
      - `module_patches` (passed in): module-attribute patches, e.g.
        `(linquant.kernels.mamba2.ssd_combined, _quant_chunk_cumsum_fwd, "cumsum")`.
        Wraps the function in the importer's namespace.
      - `instance_methods` (passed in via `WRAPPER_METHODS`): instance-method
        patches on the layer (e.g. `(layer.in_proj, "forward_int8", "in_proj")`).
        Wraps the bound method.

    Both share a single events dict. Multiple entries may share a stage label;
    only the one actually called fires per forward. Stage timings sum together
    when several functions for the same stage are invoked per rep
    (e.g., SSDi8's per-group requant runs twice, once for B and once for C).
    """

    def __init__(self, module_patches, layer=None, instance_methods=None):
        self.module_patches = module_patches
        self.layer = layer
        self.instance_methods = instance_methods or []
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._restore_module: list[tuple[object, str, object]] = []
        self._restore_instance: list[tuple[object, str, object]] = []

    def _make_wrap(self, fn, label):
        store = self.events

        def wrapped(*a, **kw):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **kw)
            e.record()
            store.setdefault(label, []).append((s, e))
            return out

        return wrapped

    def __enter__(self):
        """Install the wrappers (module-level + instance-level)."""
        # Module-attribute patches.
        for modname, fname, stage in self.module_patches:
            try:
                mod = importlib.import_module(modname)
            except ImportError:
                continue
            orig = getattr(mod, fname, None)
            if orig is None:
                continue
            self._restore_module.append((mod, fname, orig))
            setattr(mod, fname, self._make_wrap(orig, stage))

        # Instance-method patches (per-layer wrappers: in_proj, out_proj, norm).
        for sub_attr, method_attr, stage in self.instance_methods:
            sub = getattr(self.layer, sub_attr, None) if self.layer is not None else None
            if sub is None:
                continue
            orig = getattr(sub, method_attr, None)
            if orig is None or not callable(orig):
                continue
            self._restore_instance.append((sub, method_attr, orig))
            setattr(sub, method_attr, self._make_wrap(orig, stage))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the originals (module-level + instance-level)."""
        for mod, fname, orig in self._restore_module:
            setattr(mod, fname, orig)
        for sub, method_attr, orig in self._restore_instance:
            try:
                setattr(sub, method_attr, orig)
            except (AttributeError, TypeError):
                # Some bound-method writes are blocked; fall back to delete (re-binds the class method).
                try:
                    delattr(sub, method_attr)
                except AttributeError:
                    pass

    def reset(self):
        """Clear recorded events. Call before each timed forward."""
        self.events.clear()


def collect_stage_medians(events_per_rep: list[dict[str, list[float]]]) -> dict[str, float]:
    """Per stage: sum µs across all calls within a rep, then median across reps.

    Handles both the single-call case (one call/rep) and the multi-call case
    (e.g., SSDi8's bc_requant runs twice/rep — sum per rep).
    """
    out: dict[str, float] = {}
    if not events_per_rep:
        return out
    stages = set()
    for d in events_per_rep:
        stages.update(d.keys())
    for stage in stages:
        per_rep_sum = [sum(d.get(stage, [])) for d in events_per_rep if d.get(stage)]
        if per_rep_sum:
            out[stage] = 1000.0 * statistics.median(per_rep_sum)  # ms → µs
    return out


def time_components_one_shape(layer, x, n_warmup: int, n_repeat: int, patches):
    """Run the layer's forward `n_warmup + n_repeat` times under the patcher."""
    with ComponentTimer(patches, layer=layer, instance_methods=WRAPPER_METHODS) as timer:
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                timer.reset()
                layer(x)
        torch.cuda.synchronize()

        # Real measurement: collect per-rep stage timings
        per_rep: list[dict[str, list[float]]] = []
        with torch.no_grad():
            for _ in range(n_repeat):
                timer.reset()
                layer(x)
                torch.cuda.synchronize()
                rep_ms: dict[str, list[float]] = {}
                for stage, pairs in timer.events.items():
                    rep_ms[stage] = [s.elapsed_time(e) for s, e in pairs]
                per_rep.append(rep_ms)

    medians = collect_stage_medians(per_rep)
    return medians, per_rep


def main():
    """Run per-component timing for one config across (B, T) sweep."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    ap.add_argument(
        "--seq_lens",
        nargs="+",
        type=int,
        default=[256, 512, 1024, 2048],
        help="Cross-product sweep with batch_sizes. Default matches SSDi8 Fig. 4b.",
    )
    ap.add_argument(
        "--long_context_lens",
        nargs="*",
        type=int,
        default=[4096, 6144, 8192, 10240, 12288, 14336],
        help="Extra cells run only at --long_context_batch_size. "
        "Default matches SSDi8 App. H Tab. 13. Empty list disables.",
    )
    ap.add_argument("--long_context_batch_size", type=int, default=8)
    ap.add_argument("--n_warmup", type=int, default=20)
    ap.add_argument("--n_repeat", type=int, default=100)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--model_size", choices=list(MODEL_DIMS.keys()), default="1.3b")
    args = ap.parse_args()

    set_model_size(args.model_size)

    device = "cuda"
    torch.cuda.set_device(0)

    print(f"\n=== Per-stage timing: {args.config} ({args.model_size}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"batch_sizes={args.batch_sizes}, seq_lens={args.seq_lens}")

    t0 = time.time()
    layer = build_layer(args.config, device).eval()
    build_s = time.time() - t0
    print(f"build: {build_s:.1f}s")

    torch.cuda.reset_peak_memory_stats()

    patches = patches_for_config(args.config)

    results = {
        "config": args.config,
        "gpu": torch.cuda.get_device_name(0),
        "build_seconds": build_s,
        "n_warmup": args.n_warmup,
        "n_repeat": args.n_repeat,
        "stages": STAGES,
        "by_shape": {},
    }

    shapes: list[tuple[int, int]] = [(b, t) for b in args.batch_sizes for t in args.seq_lens]
    shapes += [(args.long_context_batch_size, t) for t in args.long_context_lens]

    header = f"  {'B':>3} {'T':>6}  " + " ".join(f"{s:>11}" for s in STAGES) + f"  {'total':>11}"
    print()
    print(header)
    print("-" * len(header))
    for bs, T in shapes:
        try:
            x = torch.randn(bs, T, builders.D_MODEL, dtype=torch.float16, device=device)
            medians_us, _ = time_components_one_shape(
                layer, x, args.n_warmup, args.n_repeat, patches
            )
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  {bs:>3} {T:>6}  OOM: {exc}")
            results["by_shape"][f"B{bs}_T{T}"] = {"error": "OOM"}
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            print(f"  {bs:>3} {T:>6}  FAIL: {type(exc).__name__}: {exc}")
            results["by_shape"][f"B{bs}_T{T}"] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        cells = []
        total = 0.0
        for s in STAGES:
            v = medians_us.get(s)
            if v is None:
                cells.append(f"{'-':>11}")
            else:
                cells.append(f"{v:>11.2f}")
                total += v
        print(f"  {bs:>3} {T:>6}  " + " ".join(cells) + f"  {total:>11.2f}")
        entry = {s: medians_us.get(s) for s in STAGES}
        entry["total_us"] = total
        results["by_shape"][f"B{bs}_T{T}"] = entry

    results["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
    print(f"\nPeak memory: {results['peak_memory_mb']:.1f} MB")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
