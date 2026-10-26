# ruff: noqa: C901, D102, D103, D205, E501
"""Mamba-2 W4A8 with the LinQuant invariances on top of the SSDi8 pipeline.

SSDi8's `quantize_model_mamba` runs unchanged (Pile calibration, GPTQ, output-mean
compensation); the only additions are (1) the R_QK rotation of B/C injected into its
calibration forward, (2) the diagonal key rescaling D_qk = (max|B|/max|C|)^alpha applied
at calibration and at runtime, and (3) the swap of every SSDi8 qMamba2 mixer for our
QuantizedMamba2 (INT8 recurrence with INT8 stored state, key- or value-side absorption).
Compensation is re-measured against our mixers after the swap; value-side rows run with
`--no_comp_out`, key-side rows with `--comp_decay 0.15`. Reports WikiText-2 perplexity.

Usage (main-text rows; add --calib_seed {1,2,3} for the calibration-seed repeats):
    python scripts/mamba2_w4a8.py --model state-spaces/mamba2-1.3b --rotation none --no_comp_out
    python scripts/mamba2_w4a8.py --model state-spaces/mamba2-1.3b --rotation r --no_comp_out
    python scripts/mamba2_w4a8.py --model state-spaces/mamba2-1.3b --rotation r --no_comp_out --dqk_alpha 0.5
    python scripts/mamba2_w4a8.py --model state-spaces/mamba2-1.3b --keyside --rotation r --comp_decay 0.15 --dqk_alpha 0.5
Rotation signs: `--rotation_signs random --rotation_seed 42` (paper); default is the plain Hadamard.
"""

from __future__ import annotations

import os
import sys

import filelock

filelock.FileLock = filelock.SoftFileLock
os.environ["SOFT_FILELOCK"] = "1"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "libs", "SSDi8"))

import datasets as _datasets  # noqa: E402

_datasets.builder.FileLock = filelock.SoftFileLock

import argparse  # noqa: E402
import logging  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402
import vllm_kernel_patch  # noqa: E402, F401
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

N_SAMPLES = 512  # SSDi8's calibration set size (SSDi8_W4A8.sh default)
SEQ_LEN = 512  # SSDi8's calibration sequence length
SP_TOKENIZER = "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"  # Mamba-2 8B (nvidia)


def _reduce_to_2d(t):
    """Reduce a >=2D tensor to its last 2 dims via amax over leading dims."""
    if t is None:
        return None
    while t.dim() > 2:
        t = t.amax(dim=0)
    return t.contiguous()


def translate_layer_to_ours(
    qmixer,
    original_mixer,
    device,
    keyside=False,
    rotate=False,
    r_signs_override: torch.Tensor | None = None,
):
    """Build our QuantizedMamba2 from a fully SSDi8-quantized qMamba2.

    `qmixer` has SSDi8 qMamba2 structure; `original_mixer` is the FP16 Mamba2Simple
    needed by our from_fp16 for shape introspection.

    Scales are kept at SSDi8-native per-(H,P) / per-(G,N) granularity.
    """
    from linquant.quantized_mamba2 import QuantizedMamba2

    qcs = qmixer.qchunk_scan

    # ---- ours QuantizedMamba2 init (re-use existing W4A8 linears) ----
    ours = QuantizedMamba2(
        d_model=qmixer.d_model,
        d_ssm=qmixer.d_ssm,
        nheads=qmixer.nheads,
        headdim=qmixer.headdim,
        d_state=qmixer.d_state,
        ngroups=qmixer.ngroups,
        d_conv=qmixer.d_conv,
        chunk_size=qmixer.chunk_size,
        in_proj=qmixer.in_proj,  # SSDi8 W4A8
        out_proj=qmixer.out_proj,  # SSDi8 W4A8
        conv1d=qmixer.conv1d_origin,  # FP conv1d
        norm=qmixer.norm,  # SSDi8 QRMSNormGated (FP weights)
        A_log=qcs.A_log.to(device),
        D=qcs.D.to(device) if qcs.D is not None else None,
        dt_bias=qcs.dt_bias if qcs.dt_bias is not None else original_mixer.dt_bias,
        dt_limit=qmixer.dt_limit,
        layer_idx=qmixer.layer_idx,
    )

    # ---- per-tensor SSDi8 scales (chunk_scan path; runtime kernel uses bscale_g) ----
    ours.register_buffer("A_log_scale", qcs.A_log_scale.float())
    if qcs.D is not None:
        ours.register_buffer("D_scale", qcs.D_scale.float())
    else:
        ours.register_buffer("D_scale", torch.tensor(1.0, dtype=torch.float32, device=device))

    # ours's QuantizedMamba2.forward Python-pre-quantizes B/C using these
    # *_inv_scale buffers BEFORE calling the kernel. SSDi8 sets B/C scales
    # to 1.0 internally, but that's because SSDi8 quantizes per-tile INSIDE
    # the kernel. Ours quantizes outside, so we need real per-G scales
    # that bound typical activations to keep INT8 range usable.
    # x is consumed as FP (kernel quantizes per-tile), so no x_scale buffer.
    # The orchestrator expects B_scale and C_scale shape (G,). For ngroups=1
    # (1.3B/2.7B) this is identical to a scalar; for ngroups=8 (8B) it preserves
    # per-G B/C precision (otherwise the 7 non-max groups share the max-amax
    # scale, costing ~+0.24 PPL).
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    ngroups = qmixer.ngroups
    # B: per-group, max over N of B_chunk_state_scale_cgn → (G,)
    if getattr(qcs, "B_chunk_state_scale_cgn", None) is not None:
        b_amax = _reduce_to_2d(qcs.B_chunk_state_scale_cgn.to(device).float())
        b_per_g = b_amax.amax(dim=-1).contiguous()  # (G,)
    else:
        b_per_g = one.expand(ngroups).contiguous()
    ours.register_buffer("B_scale", b_per_g)
    ours.register_buffer("B_inv_scale", (1.0 / b_per_g.clamp_min(1e-10)))
    # C: per-group, from C_chunkscan_scale → (G,)
    if qcs.C_chunkscan_scale is not None:
        c_per_g = qcs.C_chunkscan_scale.float().to(device).flatten().contiguous()
        if c_per_g.numel() != ngroups:
            # Reduce leading dims via amax until 1D, then verify shape.
            t = qcs.C_chunkscan_scale.float().to(device)
            while t.dim() > 1:
                t = t.amax(dim=0)
            c_per_g = t.flatten().contiguous()
    else:
        c_per_g = one.expand(ngroups).contiguous()
    ours.register_buffer("C_scale", c_per_g)
    ours.register_buffer("C_inv_scale", (1.0 / c_per_g.clamp_min(1e-10)))

    # ---- per-(H,P) and per-(G,N) scales (SSDi8-native, no coarsening) ----
    if getattr(qcs, "x_chunk_state_scale_chp", None) is not None:
        alpha = qcs.x_chunk_state_scale_chp.to(device).float().clamp_min(1e-8)
        alpha = _reduce_to_2d(alpha)
        ours.register_buffer("alpha_hp", alpha)
    if getattr(qcs, "B_chunk_state_scale_cgn", None) is not None:
        bgn = _reduce_to_2d(qcs.B_chunk_state_scale_cgn.to(device).float())
        bgn_g = bgn.amax(dim=-1).contiguous()
        ours.register_buffer("bscale_g", bgn_g)
    if getattr(qcs, "cb_chunkscan_scale", None) is not None:
        cb = qcs.cb_chunkscan_scale.to(device).float().clamp_min(1e-10)
        while cb.dim() > 1:
            cb = cb.amax(dim=0)
        ours.register_buffer("static_cb_scale", cb.flatten().contiguous())
    if getattr(qcs, "state_chunkscan_scale", None) is not None:
        sss = _reduce_to_2d(qcs.state_chunkscan_scale.to(device).float().clamp_min(1e-10))
        ours.register_buffer("static_state_scale", sss)

    ours.keyside = keyside
    # ---- keyside: persistent x scale from raw x amax (not gate-weighted) ----
    if keyside and getattr(qcs, "ori_x_chunk_state_scale_chp", None) is not None:
        ori_alpha = qcs.ori_x_chunk_state_scale_chp.to(device).float().clamp_min(1e-8)
        ori_alpha = _reduce_to_2d(ori_alpha)
        ours.register_buffer("x_quant_scale_hp", ori_alpha)
    # ---- keyside: static b_gate_scale from calibration (gate×B amax per-(H,N)) ----
    if keyside:
        b_gate_hn = getattr(qcs, "b_gate_scale_hn", None)
        if b_gate_hn is not None:
            b_gate_hn = b_gate_hn.to(device).float().clamp_min(1e-10)
            if b_gate_hn.dim() > 2:
                b_gate_hn = b_gate_hn.amax(dim=0) if b_gate_hn.dim() == 3 else b_gate_hn
            buf = b_gate_hn.view(1, 1, ours.nheads, ours.d_state).contiguous()
            ours.register_buffer("b_gate_scale", buf)

    # ---- compensation buffer (initialized to zero; post-swap pass fills it) ----
    d_model = (
        qmixer.out_proj.out_features
        if hasattr(qmixer.out_proj, "out_features")
        else qmixer.out_proj.weight.shape[0]
    )
    ours.register_buffer("comp_out", torch.zeros(d_model, dtype=torch.float16, device=device))

    # ---- rotation signs: R_QK on B/C (per group); the value-axis rotation is off ----
    if rotate:
        if r_signs_override is not None:
            assert r_signs_override.shape == (ngroups, qmixer.d_state), (
                f"r_signs_override shape mismatch: got {tuple(r_signs_override.shape)}, "
                f"expected ({ngroups}, {qmixer.d_state})"
            )
            r_signs = r_signs_override.to(device=device, dtype=torch.float16)
        else:
            r_signs = torch.ones(ngroups, qmixer.d_state, device=device, dtype=torch.float16)
        ours.register_buffer("r_signs", r_signs)
    else:
        ours.register_buffer("r_signs", None)

    return ours


def eval_ppl(model, tokenizer, device, seqlen=2048):
    """Bit-identical to SSDi8 baseline: delegates to ssdi8.eval_utils.evaluate_ppl.

    Aggregates as exp(sum(loss × seqlen) / (nsamples × seqlen)) instead of
    exp(mean(loss)) — mathematically equivalent for equal-seqlen samples but
    matches SSDi8's FP-arithmetic order exactly.
    """
    from ssdi8.eval_utils import evaluate_ppl as ssdi8_evaluate_ppl

    results = ssdi8_evaluate_ppl(
        model,
        tokenizer,
        "ours-mamba2",
        batch_size=1,
        device=device,
        dataset="wikitext2",
        seq_len=seqlen,
    )
    return float(results["wikitext2"])


def _dqk_calib_samples(calib_seed, tokenizer):
    """Calibration windows for the D statistics: the same domain/seed the SSDi8 pass uses
    (the Pile validation shard, N_SAMPLES x SEQ_LEN tokens)."""
    from datasets import load_dataset

    ds = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
    if calib_seed is not None:
        ds = ds.shuffle(seed=calib_seed)
    out = []
    for row in ds:
        ids = tokenizer(row["text"], return_tensors="pt", max_length=SEQ_LEN, truncation=True).input_ids
        if ids.shape[1] < 2:
            continue
        out.append(ids)
        if len(out) >= N_SAMPLES:
            break
    return out


def dqk_from_stats(model, samples, device, alpha):
    """Per-layer (dqk_b, dqk_c) = (D⁻¹, D), D=(max|B|/max|C|)^alpha per (G,N), from the FP model's post-conv+SiLU
    B/C over the calibration samples (hooks on in_proj; conv+SiLU recomputed here since the fused conv has no hook)."""
    import torch.nn.functional as F

    stats = {}
    handles = []
    for li, block in enumerate(model.backbone.layers):
        mx = block.mixer

        def hook(_m, _i, out, mx=mx, li=li):
            d_ssm, gn, nh = mx.d_ssm, mx.ngroups * mx.d_state, mx.nheads
            d_mlp = (out.shape[-1] - 2 * d_ssm - 2 * gn - nh) // 2
            xBC = out[..., 2 * d_mlp + d_ssm : 2 * d_mlp + d_ssm + d_ssm + 2 * gn]
            w = mx.conv1d.weight  # (conv_dim, 1, d_conv), depthwise
            y = F.conv1d(xBC.transpose(1, 2), w, mx.conv1d.bias, groups=w.shape[0], padding=w.shape[-1] - 1)
            y = F.silu(y[..., : xBC.shape[1]]).transpose(1, 2)
            b = y[..., d_ssm : d_ssm + gn].float().abs().amax(dim=(0, 1)).view(mx.ngroups, mx.d_state)
            c = y[..., d_ssm + gn :].float().abs().amax(dim=(0, 1)).view(mx.ngroups, mx.d_state)
            st = stats.setdefault(li, [b, c])
            st[0], st[1] = torch.maximum(st[0], b), torch.maximum(st[1], c)

        handles.append(mx.in_proj.register_forward_hook(hook))
    with torch.no_grad():
        for ids in samples:
            model(ids.to(device))
    for h in handles:
        h.remove()
    out = []
    for li in range(len(model.backbone.layers)):
        b, c = stats[li]
        D = (b.clamp_min(1e-6) / c.clamp_min(1e-6)).pow(alpha).clamp(1e-2, 1e2)
        out.append((1.0 / D, D))
    return out


def load_tokenizer(model_id):
    """GPT-NeoX tokenizer for 1.3B/2.7B; the nvidia SentencePiece tokenizer (via SSDi8) for 8B."""
    sp_model = os.path.join(model_id, SP_TOKENIZER)
    if not os.path.isfile(sp_model):
        try:
            from huggingface_hub import try_to_load_from_cache

            _cached = try_to_load_from_cache(model_id, SP_TOKENIZER)
            if _cached and isinstance(_cached, str) and os.path.isfile(_cached):
                sp_model = _cached
        except Exception:
            pass
    if not os.path.isfile(sp_model):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    from ssdi8.megatron_utils import _GPTSentencePieceTokenizer

    tokenizer = _GPTSentencePieceTokenizer(sp_model)
    # SSDi8's quantize_model_mamba calls `tokenizer.save(saved_dir)` for 8B; the saved copy is not needed.
    tokenizer.save = lambda *a, **k: None
    print(f"[tokenizer] using SSDi8 _GPTSentencePieceTokenizer from {sp_model}")
    return tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba2-1.3b")
    ap.add_argument(
        "--rotation",
        choices=["none", "r"],
        default="r",
        help="'r': R_QK rotation of B/C (state axis) at calibration and runtime. 'none': no rotation.",
    )
    ap.add_argument(
        "--rotation_signs",
        choices=["ones", "random"],
        default="ones",
        help="Sign vector of the Hadamard rotation. 'ones': the plain Hadamard. 'random': independent "
        "±1 per (group, state channel), shared across layers (randomized Hadamard transform).",
    )
    ap.add_argument(
        "--rotation_seed",
        type=int,
        default=42,
        help="Seed for the random signs (only with --rotation_signs random).",
    )
    ap.add_argument(
        "--no_comp_out",
        action="store_true",
        help="Disable the output-projection mean compensation (comp_out).",
    )
    ap.add_argument(
        "--comp_decay",
        type=float,
        default=None,
        help="Compensation step factor: comp_out = (fp_mean - q_mean) * comp_decay, measured against our "
        "mixers after the swap. Default None resolves to 0.1.",
    )
    ap.add_argument(
        "--keyside",
        action="store_true",
        default=False,
        help="Key-side absorption: persistent INT8 x, dynamic FP→INT8 B' in chunk_state (default: value-side).",
    )
    ap.add_argument(
        "--dqk_alpha",
        type=float,
        default=0.0,
        help="Diagonal key scaling D_n=(max|B_n|/max|C_n|)^alpha per (group, state channel): C·D, B·D⁻¹ before "
        "the rotation, at calibration and at runtime. 0 = off; 0.5 = the paper's exponent.",
    )
    ap.add_argument(
        "--calib_seed",
        type=int,
        default=None,
        help="Seed to shuffle the Pile-Uncopyrighted activation-calibration set. Default None keeps "
        "SSDi8's behavior (first N_SAMPLES in file order). GPTQ's WikiText-2 draw stays at SSDi8's fixed seed.",
    )
    args = ap.parse_args()

    if args.comp_decay is None:
        args.comp_decay = 0.1

    logging.basicConfig(level=logging.INFO)
    device = "cuda"

    from utils import set_deterministic

    set_deterministic(9997)

    print(f"\nLoading FP16 {args.model}...")
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=torch.float16)
    tokenizer = load_tokenizer(args.model)
    for layer in model.backbone.layers:
        layer.mixer.use_mem_eff_path = False
    model.eval()

    # Keep FP model around for shape introspection during translation.
    print(f"Loading FP16 reference {args.model} (for translation shape info)...")
    fp_model = MambaLMHeadModel.from_pretrained(args.model, device="cpu", dtype=torch.float16)
    for layer in fp_model.backbone.layers:
        layer.mixer.use_mem_eff_path = False

    use_rotation = args.rotation != "none"
    ssdi8_args = types.SimpleNamespace(
        a_bits=8,
        w_bits=4,
        apply_hadamard=True,
        apply_gptq=True,
        calib_data_num=N_SAMPLES,
        calib_seqlen=SEQ_LEN,
        compensation=True,
        comp_sam_num=512,
        comp_out_decay=args.comp_decay,
        comp_layers=None,
        rotation=use_rotation,
        rotation_mode="r" if use_rotation else "ru",
        x_scale_gran="per_hp",
        state_scale_gran="per_hp",
        scale_percentile=None,
        # SSDi8 reads other ablation toggles via getattr(args, key, False);
        # omitting them is equivalent to False.
        model=args.model,
        pretrained_dir="pretrained_models",
    )

    import ssdi8.modelutils_mamba as _ssdi8_mu
    from ssdi8.modelutils_mamba import quantize_model_mamba

    # Patch SSDi8's compensation to preserve _fp_means (it normally clears them).
    # We need them for our post-swap compensation pass.
    _orig_sequential_comp = _ssdi8_mu.mamba_sequential_compensation_ssdout

    def _comp_preserve_fp_means(model, *a, **kw):
        saved = None
        if getattr(model, "_fp_means", None) is not None:
            saved = {k: v.clone() for k, v in model._fp_means.items()}
        result = _orig_sequential_comp(model, *a, **kw)
        model._fp_means = saved
        return result

    _ssdi8_mu.mamba_sequential_compensation_ssdout = _comp_preserve_fp_means

    # Inject the R rotation into the calibration forward so SSDi8's run_calibration
    # captures activation statistics for rotated B/C. This sets module-level globals in
    # mamba_calib.ops.triton.ssd_combined; SSDi8's own code just reads them if non-None.
    # The same per-group signs are registered on each swapped mixer below, so calibration
    # and the production kernel apply the same forward rotation.
    n_layers = len(model.backbone.layers)
    r_signs: torch.Tensor | None = None
    if use_rotation:
        import mamba_calib.ops.triton.ssd_combined as _calib_ssd_mod
        import numpy as np
        from scipy.linalg import hadamard as _scipy_hadamard

        mixer0 = model.backbone.layers[0].mixer
        _d_state = mixer0.d_state
        _ngroups = mixer0.ngroups
        H_n = torch.from_numpy(_scipy_hadamard(_d_state).astype(np.float32) / np.sqrt(_d_state)).to(
            device
        )

        if args.rotation_signs == "random":
            # One set of random ±1 signs shared across all layers → a single 3D
            # (per-group) rotation matrix for the calibration path.
            gen = torch.Generator(device=device).manual_seed(args.rotation_seed)
            r_signs = (
                torch.randint(0, 2, (_ngroups, _d_state), device=device, generator=gen) * 2 - 1
            ).to(torch.float16)
            R = (H_n[None, :, :] * r_signs.float()[:, None, :]).to(torch.float32)
        else:
            # Plain Hadamard: 2D global (mamba_calib's 2D matmul path is bit-different
            # from the 3D einsum path under the FP16 cast; keep the historical fast path).
            r_signs = torch.ones(_ngroups, _d_state, device=device, dtype=torch.float16)
            R = H_n
        _calib_ssd_mod._ROTATION_U = None
        _calib_ssd_mod._ROTATION_R = R

        print(
            f"[rotation] Injected R rotation (signs={args.rotation_signs}, seed={args.rotation_seed}, "
            f"layers={n_layers}) into calibration forward"
        )

    dqk_per_layer = None
    if args.dqk_alpha > 0:
        print(f"\n[dqk] statistics pre-pass for the diagonal key scaling (alpha={args.dqk_alpha:g})...")
        dqk_samples = _dqk_calib_samples(args.calib_seed, tokenizer)
        dqk_per_layer = dqk_from_stats(model, dqk_samples, device, args.dqk_alpha)
        del dqk_samples
        _ratios = torch.stack([d[1].flatten() for d in dqk_per_layer])
        print(f"[dqk] D range over layers: min {_ratios.min():.3f} max {_ratios.max():.3f} median {_ratios.median():.3f}")

        # Apply D inside SSDi8's calibration mixer at the conv output, BEFORE its B/C observers (the per-group
        # B/C scales come from those observers, so the SSD-internal injection alone left them unscaled).
        import ssdi8.qMamba2 as _qm2_calib

        _qm2_calib._DQK = {i: (db.to(device).flatten(), dc.to(device).flatten()) for i, (db, dc) in enumerate(dqk_per_layer)}

    print("\nRunning SSDi8 quantize_model_mamba...")
    model.requires_grad_(False)
    calib_dataset = calib_preprocess_fn = None
    if args.calib_seed is not None:
        from functools import partial

        from datasets import load_dataset

        print(f"[calib-seed] Shuffling Pile calibration set with seed {args.calib_seed}")
        calib_dataset = load_dataset(
            "monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train"
        ).shuffle(seed=args.calib_seed)

        def _preprocess(data, tokenizer, max_tokens, device):
            return tokenizer(
                data["text"], return_tensors="pt", max_length=max_tokens, truncation=True
            ).input_ids.to(device)

        calib_preprocess_fn = partial(
            _preprocess, tokenizer=tokenizer, max_tokens=SEQ_LEN, device=device
        )
    model = quantize_model_mamba(
        model,
        "mamba2",
        tokenizer,
        device,
        ssdi8_args,
        calibration_dataset=calib_dataset,
        calib_preprocess_fn=calib_preprocess_fn,
    )
    model.eval()

    # Clear calibration globals (no longer needed after calibration).
    if use_rotation:
        _calib_ssd_mod._ROTATION_U = None
        _calib_ssd_mod._ROTATION_R = None
    if dqk_per_layer is not None:
        _qm2_calib._DQK = None  # calibration done; the swapped-in mixers carry D as buffers

    n = len(model.backbone.layers)
    print(f"\nSwapping mixers to QuantizedMamba2 (n={n})...")
    for i in range(n):
        block = model.backbone.layers[i]
        qmixer = block.mixer
        original_mixer = fp_model.backbone.layers[i].mixer
        ours_layer = translate_layer_to_ours(
            qmixer,
            original_mixer,
            device,
            keyside=args.keyside,
            rotate=use_rotation,
            r_signs_override=r_signs,
        )
        if dqk_per_layer is not None:
            ours_layer.register_buffer("dqk_b", dqk_per_layer[i][0].to(device).flatten())
            ours_layer.register_buffer("dqk_c", dqk_per_layer[i][1].to(device).flatten())
        block.mixer = ours_layer.to(device)
    print(f"[mixer-swap] replaced {n} layers with ours QuantizedMamba2")

    # ---- Post-swap compensation: calibrate comp_out against OUR mixer ----
    fp_means = getattr(model, "_fp_means", None)
    if not args.no_comp_out and fp_means is not None:
        print(f"\n[comp] Post-swap compensation (decay={args.comp_decay})...")
        from datasets import load_dataset as _load_ds

        _pile = _load_ds("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
        comp_n = min(512, len(_pile))
        comp_data = []
        for _ci in range(comp_n):
            _ids = tokenizer(
                _pile[_ci]["text"],
                return_tensors="pt",
                max_length=SEQ_LEN,
                truncation=True,
            ).input_ids.to(device)
            if _ids.shape[1] > 0:
                comp_data.append(_ids)
        comp_n = len(comp_data)

        def _batch_sum_and_count(y):
            if y.dim() == 3:
                return y.sum(dim=(0, 1)).float(), y.shape[0] * y.shape[1]
            if y.dim() == 2:
                return y.sum(dim=0).float(), y.shape[0]
            raise RuntimeError(f"Unexpected shape {tuple(y.shape)}")

        class _Catcher(torch.nn.Module):
            def __init__(self, mod, capture_list):
                super().__init__()
                self.mod = mod
                self.capture_list = capture_list

            def forward(self, x, *a, **kw):
                self.capture_list.append(
                    x.squeeze(0).detach() if x.dim() == 3 and x.size(0) == 1 else x.detach()
                )
                raise ValueError

        # Capture inputs to first block, then run our mixer layer-sequentially
        blocks = model.backbone.layers
        inps_list = []
        blocks[0] = _Catcher(blocks[0], inps_list)
        for j in range(comp_n):
            try:
                model(comp_data[j])
            except ValueError:
                pass
        blocks[0] = blocks[0].mod
        assert len(inps_list) == comp_n

        res_in_list = [None] * comp_n
        n_layers = len(blocks)
        for li in range(n_layers):
            layer = blocks[li]
            mix = layer.mixer
            mix.comp_out.zero_()

            m_fp_out = fp_means.get(f"L{li}/out_proj_out", None)
            if m_fp_out is not None:
                m_fp_out = m_fp_out.to(device=device, dtype=torch.float32)

            out_sum = None
            out_cnt = 0

            def _out_hook(_m, _i, out):
                nonlocal out_sum, out_cnt
                s, c = _batch_sum_and_count(out)
                out_sum = s if out_sum is None else (out_sum + s)
                out_cnt += int(c)

            h = mix.out_proj.register_forward_hook(_out_hook)
            with torch.no_grad():
                for j in range(comp_n):
                    xj = inps_list[j]
                    xj = xj.unsqueeze(0) if xj.dim() == 2 else xj
                    _ = layer(xj, residual=res_in_list[j])
            h.remove()

            if m_fp_out is not None and out_cnt > 0:
                q_mean = out_sum / out_cnt
                q_mean = torch.where(torch.isfinite(q_mean), q_mean, torch.zeros_like(q_mean))
                mix.comp_out.copy_((m_fp_out - q_mean).to(mix.comp_out.dtype) * args.comp_decay)

            next_inps, next_res = [], []
            with torch.no_grad():
                for j in range(comp_n):
                    xj = inps_list[j]
                    xj = xj.unsqueeze(0) if xj.dim() == 2 else xj
                    yj, resj = layer(xj, residual=res_in_list[j])
                    next_inps.append(yj.squeeze(0).detach())
                    next_res.append(resj.detach() if torch.is_tensor(resj) else resj)
            inps_list, res_in_list = next_inps, next_res
            if li < 3 or li == n_layers - 1:
                comp_norm = mix.comp_out.float().norm().item()
                print(f"  L{li} comp_out norm={comp_norm:.4f}")

        del comp_data, inps_list, res_in_list
        model._fp_means = None
        torch.cuda.empty_cache()
        print("[comp] Post-swap compensation done")
    elif args.no_comp_out:
        print("[comp] Compensation disabled (--no_comp_out)")
    elif fp_means is None:
        print("[comp] WARNING: _fp_means not available (SSDi8 calibration may have failed)")

    print("\nEvaluating WikiText-2 PPL...")
    ppl = eval_ppl(model, tokenizer, device)
    side = "key-side" if args.keyside else "value-side"
    comp = "no comp" if args.no_comp_out else f"comp {args.comp_decay:g}"
    print(
        f"\n  WikiText-2 PPL ({args.model}, W4A8, rotation={args.rotation}, D_qk alpha={args.dqk_alpha:g}, "
        f"{side}, {comp}): {ppl:.4f}"
    )


if __name__ == "__main__":
    main()
