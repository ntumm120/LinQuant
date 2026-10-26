# ruff: noqa: C901, D102, D103, D205, D301, E501, F841
"""W4A8 quantization of the Mamba-2 blocks of Nemotron-H 8B (WikiText-2 PPL).

Reproduces the Nemotron-H rows of the main-text production table. Attention and MLP
blocks stay FP16; the 24 mamba blocks get W4 (GPTQ) linears and the INT8 SSD kernel
(SSDi8 pipeline + ours R rotation / D_QK rescaling), all built in memory:

  1. Load FP16 Nemotron-H (HF, trust_remote_code).
  2. Global Hadamard on embedding + every block + lm_head, RMSNorms fused into the
     adjacent linears (`nemotron_h_calibration.apply_global_had_nemotron_h`).
  3. Calibrate the SSD-path activation scales on Pile-uncopyrighted, with D_QK and the
     R rotation applied inside the SSD so every statistic is observed as at runtime.
     Capture the FP out_proj output means for the compensation step.
  4. GPTQ W4 (group 128) on mamba in_proj / out_proj (lm_head stays FP16).
  5. Replace each mamba mixer with `linquant.quantized_mamba2.QuantizedMamba2` carrying
     the calibrated scales, the per-layer random Hadamard signs and D_QK.
  6. Sequential out_proj mean compensation (decay 0.1), then WikiText-2 PPL.

Rows (MODEL in {8b-4k, 8b-128k}):
    python scripts/nemotron_h_w4a8.py --model MODEL --rotation none         # FP16 + SSDi8
    python scripts/nemotron_h_w4a8.py --model MODEL --rotation r            # + R_QK   (value side)
    python scripts/nemotron_h_w4a8.py --model MODEL --rotation r --dqk_alpha 0.5             # + D_QK
    python scripts/nemotron_h_w4a8.py --model MODEL --rotation r --dqk_alpha 0.5 --keyside   # key side
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("SOFT_FILELOCK", "1")

import filelock  # noqa: E402

filelock.FileLock = filelock.SoftFileLock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "libs", "SSDi8"))

import argparse  # noqa: E402
import gc  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch._inductor.config.compile_threads = 1
torch._dynamo.config.suppress_errors = True

import datasets  # noqa: E402

datasets.builder.FileLock = filelock.SoftFileLock

import nemotron_h_calibration as fqn  # noqa: E402
import vllm_kernel_patch  # noqa: E402, F401
from datasets import load_dataset  # noqa: E402
from ssdi8.data_loaders import get_loaders  # noqa: E402
from ssdi8.gptq_utils import GPTQ  # noqa: E402
from ssdi8.hadamard_utils import had_transform  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

NEMOTRON_H_MODELS = {
    "8b-4k": "nvidia/Nemotron-H-8B-Base-8K",
    "8b-128k": "nvidia/Nemotron-H-8B-Reasoning-128K",
}

CAL_SAMPLES, CAL_SEQ_LEN = 512, 2048  # Pile-uncopyrighted calibration windows
GPTQ_SAMPLES, GPTQ_SEQLEN = 128, 1024  # WikiText-2 train windows for GPTQ
COMP_SAMPLES = 256  # Pile windows for the sequential compensation
EVAL_SEQ_LEN = 2048
SEED = 9997


# ---------------------------------------------------------------------------
# Phase 0: Load model
# ---------------------------------------------------------------------------
def load_nemotron_h(model_id, device):
    """Load Nemotron-H in FP16 (cast from bf16 native).

    QuantizedMamba2 expects FP16 throughout the mamba mixer path (Marlin W4 kernels are
    FP16-only; norm/conv1d are used with their loaded dtype). Loading the whole model in
    fp16 keeps everything consistent.
    """
    print(f"Loading {model_id} (FP16, trust_remote_code)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Phase 1: global Hadamard fusion
# ---------------------------------------------------------------------------
def configure_model_hybrid(model):
    """Global Hadamard rotation on embedding + all blocks + lm_head.

    `apply_global_had_nemotron_h` bakes one Hadamard into each mamba out_proj weight
    (paired had_input + had_output); the runtime pre-hook applies the matching
    had_transform to the out_proj input.
    """
    print("\n--- Phase 1: global Hadamard rotation ---")
    model.requires_grad_(False)
    fqn.apply_global_had_nemotron_h(model)
    n_hooks = 0
    for layer in model.backbone.layers:
        if getattr(layer, "block_type", None) == "mamba":
            layer.mixer.out_proj.register_forward_pre_hook(
                lambda m, a: (had_transform(a[0]),) + a[1:]
            )
            n_hooks += 1
    print(f"  Installed {n_hooks} mamba out_proj runtime had_transform hooks")


# ---------------------------------------------------------------------------
# Phase 2: SSD calibration on Pile (with FP out_proj mean capture)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _calibrate_cb_g_scales(model, tokenizer, device, n_samples, seq_len, rotate=False, R_matrix=None):
    """Per-G CB amax at chunk_scan entry (post bmm of C and B).

    Mirrors SSDi8's `cb_cali` ChunkCollector:
        amax_cb = cb.abs().amax(dim=(0, 3, 1, 4))  # (G,)
    where `cb` has shape (B, nchunks, ngroups, K, K). Hooks `_bmm_chunk_fwd` directly.

    With `rotate=True`, the SSD runs end-to-end rotated to match runtime (via
    `_make_rotated_calib_fwd`), so the bmm hook sees C/B already rotated by R.
    """
    import mamba_ssm.ops.triton.ssd_combined as ssd_mod

    samples = fqn._load_cal_samples(tokenizer, n_samples, seq_len)
    mixers = fqn._mamba_mixers(model)
    n_mamba = len(mixers)
    running_max = [None] * n_mamba

    saved_fwd = ssd_mod._mamba_chunk_scan_combined_fwd
    rotated_base = fqn._make_rotated_calib_fwd(fqn._ORIGINAL_FWD, R_matrix if rotate else None)
    ssd_mod._mamba_chunk_scan_combined_fwd = rotated_base

    original_bmm = ssd_mod._bmm_chunk_fwd
    layer_counter = [0]

    def hooked_bmm(*args, **kwargs):
        cb = original_bmm(*args, **kwargs)  # (B, nchunks, ngroups, K, K)
        amax = cb.abs().float().amax(dim=(0, 1, 3, 4))  # (G,)
        idx = layer_counter[0] % n_mamba
        running_max[idx] = amax if running_max[idx] is None else torch.max(running_max[idx], amax)
        layer_counter[0] += 1
        return cb

    ssd_mod._bmm_chunk_fwd = hooked_bmm
    try:
        with torch.no_grad():
            for i, inp in enumerate(samples):
                model(inp.to(device))
                if (i + 1) % 16 == 0:
                    print(f"  CB per-G calibration: {i + 1}/{len(samples)}")
    finally:
        ssd_mod._bmm_chunk_fwd = original_bmm
        ssd_mod._mamba_chunk_scan_combined_fwd = saved_fwd

    return [(rm / 127.0).clamp(min=1e-10).to(device) for rm in running_max]


def dqk_from_stats_nemotron(model, tokenizer, device, n_samples, seq_len, alpha):
    """Per-mamba-layer (D⁻¹, D), D=(max|B|/max|C|)^alpha per (G,N), from the FP model's post-conv+SiLU B/C
    (hooks on in_proj, conv+SiLU recomputed here) over the calibration samples."""
    import torch.nn.functional as F

    samples = fqn._load_cal_samples(tokenizer, n_samples, seq_len)
    mixers = fqn._mamba_mixers(model)
    stats, handles = {}, []
    for li, (_, mx) in enumerate(mixers):
        nh, hd, ng, ds, _, inter = fqn._mamba_mixer_attrs(mx)

        def hook(_m, _i, out, mx=mx, li=li, nh=nh, ng=ng, ds=ds, inter=inter):
            gn = ng * ds
            d_mlp = (out.shape[-1] - 2 * inter - 2 * gn - nh) // 2
            xBC = out[..., 2 * d_mlp + inter : 2 * d_mlp + 2 * inter + 2 * gn]
            w = mx.conv1d.weight
            y = F.conv1d(xBC.transpose(1, 2).to(w.dtype), w, mx.conv1d.bias, groups=w.shape[0], padding=w.shape[-1] - 1)
            y = F.silu(y[..., : xBC.shape[1]]).transpose(1, 2)
            b = y[..., inter : inter + gn].float().abs().amax(dim=(0, 1)).view(ng, ds)
            c = y[..., inter + gn :].float().abs().amax(dim=(0, 1)).view(ng, ds)
            st = stats.setdefault(li, [b, c])
            st[0], st[1] = torch.maximum(st[0], b), torch.maximum(st[1], c)

        handles.append(mx.in_proj.register_forward_hook(hook))
    with torch.no_grad():
        for ids in samples:
            model(ids.to(device))
    for h in handles:
        h.remove()
    out = []
    for li in range(len(mixers)):
        b, c = stats[li]
        D = (b.clamp_min(1e-6) / c.clamp_min(1e-6)).pow(alpha).clamp(1e-2, 1e2)
        out.append((1.0 / D, D))
    return out


def _random_signs_per_layer(n_mamba, nheads, headdim, ngroups, d_state, device, seed, dtype):
    """Per-mamba-layer ±1 sign vectors for the R rotation (random Hadamard transform).

    One generator seeded with `seed`, consumed sequentially per layer: a (nheads, headdim)
    value-axis draw (unused: the value axis is not rotated) followed by the (ngroups,
    d_state) state-axis draw. Both calibration and the runtime mixers call this, so the
    calibration sees exactly the rotation the kernel applies.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    r_signs = []
    for _ in range(n_mamba):
        torch.randint(0, 2, (nheads, headdim), device=device, generator=gen)
        r_signs.append((torch.randint(0, 2, (ngroups, d_state), device=device, generator=gen) * 2 - 1).to(dtype))
    return r_signs


@torch.no_grad()
def calibrate_all(model, tokenizer, device, rotate: bool, rotation_seed: int):
    """Calibrate per-mamba-mixer SSD scales on Pile + capture FP out_proj means.

    Returns (per_layer_scales, per_layer_fp_means) where:
      - per_layer_scales[i]: dict with x_hp, bc_gn, b_gate_hn, x_scaled_hp, state_hp,
        c_g, cb_g, d, in_proj, out_proj (amax/127 except b_gate_hn, per layer).
      - per_layer_fp_means[i]: tensor of out_proj output channel means.
    """
    import scipy.linalg

    mixers = fqn._mamba_mixers(model)
    n_mamba = len(mixers)
    nheads, headdim, ngroups, d_state, _, _ = fqn._mamba_mixer_attrs(mixers[0][1])

    # Per-layer rotation matrices applied to B/C inside the SSD fwd during calibration;
    # must match the runtime Kronecker rotation (same per-group signs).
    R_matrix = None
    if rotate:
        H_n = torch.from_numpy(scipy.linalg.hadamard(d_state)).to(device=device, dtype=torch.bfloat16) / math.sqrt(d_state)
        signs = _random_signs_per_layer(n_mamba, nheads, headdim, ngroups, d_state, device, rotation_seed, torch.bfloat16)
        R_matrix = [H_n[None, :, :] * s[:, None, :] for s in signs]

    print(
        f"\n--- Phase 2: SSD calibration ({n_mamba} mamba layers, "
        f"{CAL_SAMPLES} Pile samples, rotation={'r' if rotate else 'off'}) ---"
    )

    fqn._ORIGINAL_FWD = __import__(
        "mamba_ssm.ops.triton.ssd_combined", fromlist=["_mamba_chunk_scan_combined_fwd"]
    )._mamba_chunk_scan_combined_fwd

    layer_hooks = fqn.install_layer_hooks(model)

    # FP out_proj output means for compensation: running channel mean per mamba layer
    # across all calibration forward passes.
    fp_out_sums = [None] * n_mamba
    fp_out_counts = [0] * n_mamba

    def _make_out_mean_hook(idx):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            if o.dim() == 3:
                s = o.sum(dim=(0, 1)).float().detach()
                n = o.shape[0] * o.shape[1]
            elif o.dim() == 2:
                s = o.sum(dim=0).float().detach()
                n = o.shape[0]
            else:
                return
            if fp_out_sums[idx] is None:
                fp_out_sums[idx] = s
            else:
                fp_out_sums[idx] = fp_out_sums[idx] + s
            fp_out_counts[idx] += int(n)

        return hook

    fp_handles = []
    for i, (_, mix) in enumerate(mixers):
        fp_handles.append(mix.out_proj.register_forward_hook(_make_out_mean_hook(i)))

    kw = dict(n_samples=CAL_SAMPLES, seq_len=CAL_SEQ_LEN, rotate=rotate, R_matrix=R_matrix)
    try:
        # Every SSD calibrator runs the full SSD with D_QK and R applied end-to-end
        # (matching runtime) and observes its own statistic on the rotated activations.
        x_hp = fqn.calibrate_x_hp_scales(model, tokenizer, device, **kw)
        bc_gn = fqn.calibrate_bc_gn_scales(model, tokenizer, device, **kw)
        b_gate_hn = fqn.calibrate_b_gate_hn_scales(model, tokenizer, device, **kw)
        cb_g = _calibrate_cb_g_scales(model, tokenizer, device, **kw)
        x_scaled_hp = fqn.calibrate_x_scaled_hp_scales(model, tokenizer, device, **kw)
        c_g = fqn.calibrate_c_g_scales(model, tokenizer, device, **kw)
        state_hp = fqn.calibrate_state_scales(model, tokenizer, device, **kw)
        d_scales = fqn.compute_d_scales(model, device)
        in_scales, out_scales = fqn.calibrate_linear_input_scales(
            model, tokenizer, device, n_samples=CAL_SAMPLES, seq_len=CAL_SEQ_LEN
        )
    finally:
        for h in layer_hooks:
            h.remove()
        for h in fp_handles:
            h.remove()

    per_layer_scales = []
    per_layer_fp_means = []
    for i in range(n_mamba):
        per_layer_scales.append(
            dict(
                x_hp=x_hp[i],
                bc_gn=bc_gn[i],
                b_gate_hn=b_gate_hn[i],
                x_scaled_hp=x_scaled_hp[i],
                c_g=c_g[i],
                cb_g=cb_g[i],
                state_hp=state_hp[i],
                d=d_scales[i],
                in_proj=in_scales[i],
                out_proj=out_scales[i],
            )
        )
        if fp_out_sums[i] is not None and fp_out_counts[i] > 0:
            per_layer_fp_means.append(fp_out_sums[i] / fp_out_counts[i])
        else:
            per_layer_fp_means.append(None)

    print(
        f"  Calibrated {n_mamba} layers (FP out_proj means captured for "
        f"compensation: {sum(1 for m in per_layer_fp_means if m is not None)}/{n_mamba})"
    )
    return per_layer_scales, per_layer_fp_means


# ---------------------------------------------------------------------------
# Phase 3: GPTQ W4 on mamba in_proj/out_proj (attn/MLP/lm_head stay FP16)
# ---------------------------------------------------------------------------
@torch.no_grad()
def apply_gptq_mamba_only(model, tokenizer, device, nsamples=GPTQ_SAMPLES, seqlen=GPTQ_SEQLEN, group_size=128, percdamp=0.01):
    """GPTQ W4 on mamba in_proj/out_proj. Attention/MLP linears and lm_head keep FP16
    weights (SSDi8 paper Table 7 scope: SSD path only).

    Standard Catcher pattern: capture first-block inputs from WikiText-2 train, then
    propagate layer-by-layer, applying GPTQ only on mamba blocks' linears.
    """
    w_bits = 4
    print(f"\n--- Phase 3: GPTQ W{w_bits} on mamba linears ({nsamples} WT2 samples, seqlen={seqlen}) ---")
    model.eval()
    layers = model.backbone.layers
    nlayers = len(layers)

    dataloader, _ = get_loaders("wikitext2", tokenizer, nsamples=nsamples, seqlen=seqlen)

    model.backbone.embeddings = model.backbone.embeddings.to(device)
    layers[0] = layers[0].to(device)

    d_model = model.config.hidden_size
    model_dtype = next(model.parameters()).dtype
    inps = torch.zeros((nsamples, seqlen, d_model), dtype=model_dtype, device="cpu")
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m
            self.block_type = getattr(m, "block_type", None)

        def forward(self, hidden_states, **kw):
            inps[cache["i"]] = hidden_states.cpu()
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.backbone.embeddings = model.backbone.embeddings.cpu()
    torch.cuda.empty_cache()

    n_mamba_done = 0
    for i in tqdm(range(nlayers), desc=f"GPTQ W{w_bits} (mamba-only)"):
        block = layers[i].to(device)
        bt = getattr(block, "block_type", None)

        if bt == "mamba":
            mixer = block.mixer
            target_linears = {
                "mixer.in_proj": mixer.in_proj,
                "mixer.out_proj": mixer.out_proj,
            }
            solvers = {n: GPTQ(m) for n, m in target_linears.items()}
            handles = [
                m.register_forward_hook(
                    partial(
                        lambda mod, inp, out, gptq: gptq.add_batch(inp[0].data, out.data),
                        gptq=s,
                    )
                )
                for m, s in zip(target_linears.values(), solvers.values())
            ]
            for j in range(nsamples):
                block(inps[j].unsqueeze(0).to(device))
                torch.cuda.empty_cache()
            for h in handles:
                h.remove()

            for _, solver in solvers.items():
                solver.fasterquant(percdamp=percdamp, group_size=group_size, w_bits=w_bits)
                solver.free()
            del solvers

            for _, m in target_linears.items():
                # GPTQ.fasterquant sets `m.group_scale` directly (ssdi8/gptq_utils.py). The
                # metadata flags make W4A8B16O16Linear.from_fp16 consume the GPTQ
                # group_scale instead of falling back to RTN.
                if hasattr(m, "group_scale"):
                    m.apply_gptq = True
                    m.bits = w_bits
                    m.group_size = group_size
            n_mamba_done += 1

        # Propagate inputs to the next layer (all block types).
        for j in range(nsamples):
            yj = block(inps[j].unsqueeze(0).to(device))
            yj = yj[0] if isinstance(yj, tuple) else yj
            inps[j] = yj.squeeze(0).cpu()
            torch.cuda.empty_cache()
        layers[i] = block.cpu()
        del block
        torch.cuda.empty_cache()
        gc.collect()

    print(f"  GPTQ done: {n_mamba_done} mamba blocks ({nlayers - n_mamba_done} attn/MLP blocks + lm_head left FP16)")


# ---------------------------------------------------------------------------
# Phase 4: replace mamba mixers with QuantizedMamba2
# ---------------------------------------------------------------------------
def replace_mamba_mixers(model, per_layer_scales, device, rotate: bool, rotation_seed: int, keyside: bool, dqk_per_layer=None):
    """For each block_type=="mamba", build QuantizedMamba2 (W4A8 linears + INT8 SSD with
    INT8 per-(H,P) states) from the FP16 mixer + calibrated scales, then attach the
    per-(H,P)/per-(G,N) buffers and the rotation signs."""
    from linquant.quantized_mamba2 import QuantizedMamba2

    print(f"\n--- Phase 4: replace mamba mixers (rotation={'r' if rotate else 'off'}, keyside={keyside}) ---")

    mixers = fqn._mamba_mixers(model)
    n_mamba = len(mixers)
    r_signs_per_layer = [None] * n_mamba
    if rotate:
        nheads0, headdim0, ngroups0, d_state0, _, _ = fqn._mamba_mixer_attrs(mixers[0][1])
        r_signs_per_layer = _random_signs_per_layer(n_mamba, nheads0, headdim0, ngroups0, d_state0, device, rotation_seed, torch.float16)

    mamba_idx = 0
    for global_idx, block in enumerate(model.backbone.layers):
        if getattr(block, "block_type", None) != "mamba":
            continue

        # GPTQ left blocks on CPU; move this one to device for from_fp16.
        block.to(device)
        original_mixer = block.mixer
        ls = per_layer_scales[mamba_idx]

        # Per-G B/C scales (Nemotron-H ngroups=8). x is FP into the kernel — no x_scale buffer.
        bc_gn = ls["bc_gn"].float().to(device).clamp_min(1e-10)  # (G, N)
        b_per_g = bc_gn.amax(dim=-1).contiguous()  # (G,)
        c_per_g = ls["c_g"].float().to(device).contiguous()  # (G,)
        a_log_scale = (original_mixer.A_log.detach().float().abs().amax() / 127.0).clamp(min=1e-10).to(device)
        d_scale = ls["d"].float().to(device)
        state_scale = ls["state_hp"].float().to(device).clamp_min(1e-10)  # (H, P)

        scales = {
            "B_scale": b_per_g,
            "B_inv_scale": (1.0 / b_per_g.clamp_min(1e-10)),
            "C_scale": c_per_g,
            "C_inv_scale": (1.0 / c_per_g.clamp_min(1e-10)),
            "A_log_scale": a_log_scale,
            "A_log_inv_scale": (1.0 / a_log_scale.clamp_min(1e-10)),
            "D_scale": d_scale,
            "D_inv_scale": (1.0 / d_scale.clamp_min(1e-10)),
            "in_proj_input_scale": ls["in_proj"].float().to(device),
            "out_proj_input_scale": ls["out_proj"].float().to(device),
        }

        ours = QuantizedMamba2.from_fp16(
            original_mixer=original_mixer,
            scales=scales,
            r_signs=r_signs_per_layer[mamba_idx],
        )
        ours.layer_idx = global_idx
        if dqk_per_layer is not None:
            ours.register_buffer("dqk_b", dqk_per_layer[mamba_idx][0].to(device).flatten())
            ours.register_buffer("dqk_c", dqk_per_layer[mamba_idx][1].to(device).flatten())

        # Per-(H,P) / per-(G,N) buffers (the ones that drive numerics).
        ours.register_buffer("alpha_hp", ls["x_scaled_hp"].float().to(device).clamp_min(1e-8))

        ours.keyside = keyside
        if keyside:
            # Persistent INT8 x (calibrated raw x_hp) and the static per-(H, N) b_gate clip.
            ours.register_buffer("x_quant_scale_hp", ls["x_hp"].float().to(device).clamp_min(1e-8))
            b_gate = ls["b_gate_hn"].to(device).float().clamp_min(1e-10)
            ours.register_buffer("b_gate_scale", b_gate.view(1, 1, ours.nheads, ours.d_state).contiguous())

        # SSDi8 max-collapses (G, N) → (G,) for the chunk_state requant.
        ours.register_buffer("bscale_g", bc_gn.amax(dim=-1).contiguous())

        ours.register_buffer("static_state_scale", state_scale)

        # Static per-G CB scale from `_calibrate_cb_g_scales` (hooks `_bmm_chunk_fwd`).
        ours.register_buffer("static_cb_scale", ls["cb_g"].float().to(device).clamp_min(1e-10).contiguous())

        block.mixer = ours.to(device)
        del original_mixer
        mamba_idx += 1

    print(f"  replaced {mamba_idx} mamba mixers with QuantizedMamba2 ({len(model.backbone.layers) - mamba_idx} attn/MLP left FP16)")


# ---------------------------------------------------------------------------
# Phase 5: sequential out_proj mean compensation
# ---------------------------------------------------------------------------
@torch.no_grad()
def compensate_mamba_outputs(model, tokenizer, device, fp_means, comp_decay, nsamples=COMP_SAMPLES, seq_len=CAL_SEQ_LEN):
    """Sequential per-mamba-mixer out_proj mean compensation.

    Mirrors SSDi8's `mamba_sequential_compensation_ssdout` for hybrid models:
    layer-by-layer forward propagation through the (already-quantized) model,
    capturing each mamba mixer's out_proj quantized output mean and setting
    `mixer.comp_out = (fp_mean - quant_mean) * comp_decay`.

    Non-mamba blocks are propagated through without compensation.
    """
    samples = fqn._load_cal_samples(tokenizer, nsamples, seq_len)
    layers = model.backbone.layers
    L = len(layers)

    print(f"\n--- Phase 5: sequential compensation ({nsamples} Pile samples, decay={comp_decay}) ---")

    model.backbone.embeddings = model.backbone.embeddings.to(device)
    layers[0] = layers[0].to(device)

    inps_list = []

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mod = m
            # Nemotron-H's backbone reads block_type on every block — proxy it.
            self.block_type = getattr(m, "block_type", None)

        def forward(self, hidden_states, *args, **kwargs):
            inps_list.append(
                hidden_states.squeeze(0).detach()
                if (hidden_states.dim() == 3 and hidden_states.size(0) == 1)
                else hidden_states.detach()
            )
            raise ValueError

    layers[0] = Catcher(layers[0])
    for inp in samples:
        try:
            model(inp.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].mod
    layers[0] = layers[0].cpu()
    model.backbone.embeddings = model.backbone.embeddings.cpu()
    torch.cuda.empty_cache()
    assert len(inps_list) == nsamples

    def _propagate(block):
        next_inps = []
        for j in range(nsamples):
            xj = inps_list[j]
            xj = xj.unsqueeze(0) if xj.dim() == 2 else xj
            yj = block(xj.to(device))
            yj = yj[0] if isinstance(yj, tuple) else yj
            next_inps.append(yj.squeeze(0).detach().cpu())
            torch.cuda.empty_cache()
        return next_inps

    mamba_idx = 0
    for li in tqdm(range(L), desc="seq_compensation"):
        block = layers[li].to(device)
        bt = getattr(block, "block_type", None)

        if bt == "mamba":
            mixer = block.mixer
            fp_mean = fp_means[mamba_idx]
            mamba_idx += 1

            if fp_mean is None:
                # No FP mean captured (shouldn't happen): propagate without compensation.
                inps_list = _propagate(block)
                layers[li] = block.cpu()
                del block
                torch.cuda.empty_cache()
                continue

            if not hasattr(mixer, "comp_out") or mixer.comp_out is None:
                mixer.register_buffer(
                    "comp_out",
                    torch.zeros(mixer.out_proj.out_features, dtype=torch.float16, device=device),
                )
            mixer.comp_out.zero_()

            # Phase A: capture quantized out_proj output mean (with comp_out=0).
            q_sum, q_cnt = None, 0

            def _hook(_m, _i, out):
                nonlocal q_sum, q_cnt
                o = out[0] if isinstance(out, tuple) else out
                if o.dim() == 3:
                    s = o.sum(dim=(0, 1)).float().detach()
                    n = o.shape[0] * o.shape[1]
                else:
                    s = o.sum(dim=0).float().detach()
                    n = o.shape[0]
                q_sum = s if q_sum is None else (q_sum + s)
                q_cnt += int(n)

            h = mixer.out_proj.register_forward_hook(_hook)
            for j in range(nsamples):
                xj = inps_list[j]
                xj = xj.unsqueeze(0) if xj.dim() == 2 else xj
                _ = block(xj.to(device))
                torch.cuda.empty_cache()
            h.remove()

            if q_cnt > 0:
                quant_mean = q_sum / q_cnt
                fp_mean_dev = fp_mean.to(device=device, dtype=torch.float32)
                comp = (fp_mean_dev - quant_mean) * comp_decay
                comp = torch.where(torch.isfinite(comp), comp, torch.zeros_like(comp))
                mixer.comp_out.copy_(comp.to(mixer.comp_out.dtype))

        # Phase B (mamba, with comp_out applied) / plain propagation (attn, MLP).
        inps_list = _propagate(block)

        layers[li] = block.cpu()
        del block
        torch.cuda.empty_cache()
        gc.collect()

    model.to(device)
    print(f"  compensation applied to {mamba_idx} mamba mixers")


# ---------------------------------------------------------------------------
# Phase 6: WikiText-2 PPL
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_ppl(model, tokenizer, device, seqlen=EVAL_SEQ_LEN):
    """Standard WT2 test-split PPL at seqlen tokens per window."""
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids
    nsamples = testenc.numel() // seqlen
    nlls = []
    for i in tqdm(range(nsamples), desc="WT2 PPL"):
        inp = testenc[:, i * seqlen : (i + 1) * seqlen].to(device)
        logits = model(inp).logits
        loss = nn.CrossEntropyLoss()(
            logits[:, :-1].reshape(-1, logits.size(-1)), inp[:, 1:].reshape(-1)
        )
        nlls.append(loss.item())
    return torch.exp(torch.tensor(sum(nlls) / len(nlls))).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="8b-128k", help="8b-4k (Base-8K), 8b-128k (Reasoning-128K) or an HF model id.")
    ap.add_argument(
        "--rotation",
        choices=["none", "r"],
        default="r",
        help="'none': plain SSDi8. 'r': R_QK, a per-mamba-layer random Hadamard rotation of the state axis (B/C).",
    )
    ap.add_argument("--rotation_seed", type=int, default=42, help="Seed of the per-layer random rotation signs.")
    ap.add_argument(
        "--dqk_alpha",
        type=float,
        default=0.0,
        help="Diagonal key scaling D=(max|B|/max|C|)^alpha per (group, state channel) on the mamba mixers: C·D, B·D⁻¹ "
        "before the rotation at calibration and runtime. 0 = off; 0.5 = the paper's exponent.",
    )
    ap.add_argument(
        "--keyside",
        action="store_true",
        help="Key-side absorption: persistent INT8 x (calibrated raw x_hp), static per-(H,N) clip for the gated B' in chunk_state.",
    )
    ap.add_argument("--comp_decay", type=float, default=0.1, help="Decay of the sequential out_proj mean compensation.")
    ap.add_argument("--skip_fp16_baseline", action="store_true", help="Skip the FP16 baseline PPL eval.")
    args = ap.parse_args()

    device = "cuda"
    rotate = args.rotation == "r"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model_id = NEMOTRON_H_MODELS.get(args.model, args.model)

    # === Phase 0: Load ===
    model, tokenizer = load_nemotron_h(model_id, device)

    # === FP16 baseline ===
    if not args.skip_fp16_baseline:
        print("\n=== FP16 baseline ===")
        fp16_ppl = eval_ppl(model, tokenizer, device)
        print(f"  FP16 PPL = {fp16_ppl:.4f}")

    # === Phase 1: Hadamard ===
    configure_model_hybrid(model)

    # === Phase 1b: diagonal key scaling statistics on the FP model ===
    dqk_per_layer = None
    if args.dqk_alpha > 0:
        print(f"\n--- Phase 1b: diagonal key scaling statistics (alpha={args.dqk_alpha:g}) ---")
        dqk_per_layer = dqk_from_stats_nemotron(model, tokenizer, device, CAL_SAMPLES, CAL_SEQ_LEN, args.dqk_alpha)
        fqn.DQK_MATRIX = [(db.to(device), dc.to(device)) for db, dc in dqk_per_layer]
        _r = torch.stack([d[1].flatten() for d in dqk_per_layer])
        print(f"[dqk] D range over layers: min {_r.min():.3f} max {_r.max():.3f} median {_r.median():.3f}")

    # === Phase 2: Calibration ===
    per_layer_scales, fp_out_means = calibrate_all(model, tokenizer, device, rotate, args.rotation_seed)
    fqn.DQK_MATRIX = None  # calibration done; the runtime mixers carry D as buffers

    # === Phase 3: GPTQ ===
    apply_gptq_mamba_only(model, tokenizer, device)

    # === Phase 4: Replace mamba mixers ===
    replace_mamba_mixers(model, per_layer_scales, device, rotate, args.rotation_seed, args.keyside, dqk_per_layer=dqk_per_layer)
    model.to(device).eval()

    # === Phase 5: Compensation ===
    compensate_mamba_outputs(model, tokenizer, device, fp_out_means, args.comp_decay)
    model.to(device).eval()

    # === Phase 6: PPL ===
    print("\n=== WikiText-2 PPL ===")
    ppl = eval_ppl(model, tokenizer, device)
    label = f"rot={args.rotation}, dqk_alpha={args.dqk_alpha:g}, {'key' if args.keyside else 'value'} side, comp={args.comp_decay:g}"
    model_short = args.model if args.model in NEMOTRON_H_MODELS else model_id.split("/")[-1]
    print(f"\n  {model_short} W4A8 SSD-only ({label}): WT2 PPL = {ppl:.4f}")


if __name__ == "__main__":
    main()
