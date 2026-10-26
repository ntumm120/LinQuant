# ruff: noqa: E501
"""Single-layer and full-model builders for the Mamba-2 timing benchmarks.

Constructs Mamba-2 mixer blocks with random weights (no calibration or checkpoint
needed: only the kernel paths matter for timing).

Configs:
    fp16              vanilla mamba-ssm Mamba2
    ssdi8             SSDi8 W4A8QMamba2 (value-side absorption, no rotation)
    ours_value_none   our QuantizedMamba2, value-side INT8, no rotation
    ours_value_r      value-side INT8 with R_QK
    ours_key_r        key-side INT8 with R_QK

Used by bench/mamba2_model.py (full-model prefill) and bench/mamba2_components.py
(per-stage recurrence timing).
"""

from __future__ import annotations

import os
import sys

import torch

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "libs", "SSDi8"))
sys.path.insert(0, os.path.join(_root, "scripts"))

# Model dimension presets. The n<N>_p<P> entries are the synthetic geometries of the
# P/N crossover sweep (64 heads throughout; d_model = 32 * headdim keeps nheads = 64);
# Mamba-2 1.3B is the (N=128, P=64) point.
MODEL_DIMS = {
    "1.3b": dict(d_model=2048, d_ssm=4096, d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256),
    "2.7b": dict(d_model=2560, d_ssm=5120, d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256),
    "8b": dict(d_model=4096, d_ssm=8192, d_state=128, d_conv=4, expand=2, headdim=64, ngroups=8, chunk_size=256),
    "n128_p32": dict(d_model=1024, d_ssm=2048, d_state=128, d_conv=4, expand=2, headdim=32, ngroups=1, chunk_size=256),
    "n128_p128": dict(d_model=4096, d_ssm=8192, d_state=128, d_conv=4, expand=2, headdim=128, ngroups=1, chunk_size=256),
    "n128_p256": dict(d_model=8192, d_ssm=16384, d_state=128, d_conv=4, expand=2, headdim=256, ngroups=1, chunk_size=256),
    "n64_p32": dict(d_model=1024, d_ssm=2048, d_state=64, d_conv=4, expand=2, headdim=32, ngroups=1, chunk_size=256),
    "n64_p64": dict(d_model=2048, d_ssm=4096, d_state=64, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256),
    "n64_p128": dict(d_model=4096, d_ssm=8192, d_state=64, d_conv=4, expand=2, headdim=128, ngroups=1, chunk_size=256),
    "n64_p256": dict(d_model=8192, d_ssm=16384, d_state=64, d_conv=4, expand=2, headdim=256, ngroups=1, chunk_size=256),
}

# Active dims (default 1.3B, overridden by set_model_size)
_DIMS = MODEL_DIMS["1.3b"]
D_MODEL = _DIMS["d_model"]
D_SSM = _DIMS["d_ssm"]
D_STATE = _DIMS["d_state"]
D_CONV = _DIMS["d_conv"]
EXPAND = _DIMS["expand"]
HEADDIM = _DIMS["headdim"]
NHEADS = D_SSM // HEADDIM
NGROUPS = _DIMS["ngroups"]
CHUNK_SIZE = _DIMS["chunk_size"]


def set_model_size(size: str):
    """Switch module-level dims to another preset (1.3b, 2.7b, 8b or a crossover geometry)."""
    global D_MODEL, D_SSM, D_STATE, D_CONV, EXPAND, HEADDIM, NHEADS, NGROUPS, CHUNK_SIZE, _DIMS
    _DIMS = MODEL_DIMS[size]
    D_MODEL = _DIMS["d_model"]
    D_SSM = _DIMS["d_ssm"]
    D_STATE = _DIMS["d_state"]
    D_CONV = _DIMS["d_conv"]
    EXPAND = _DIMS["expand"]
    HEADDIM = _DIMS["headdim"]
    NHEADS = D_SSM // HEADDIM
    NGROUPS = _DIMS["ngroups"]
    CHUNK_SIZE = int(os.environ.get("OVERRIDE_CHUNK_SIZE", _DIMS["chunk_size"]))


CONFIGS = ["fp16", "ssdi8", "ours_value_none", "ours_value_r", "ours_key_r"]


def _make_fake_mamba2(device):
    from mamba_ssm.modules.mamba2 import Mamba2

    mixer = Mamba2(
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        headdim=HEADDIM,
        ngroups=NGROUPS,
        chunk_size=CHUNK_SIZE,
        device=device,
        dtype=torch.float16,
    )
    mixer.use_mem_eff_path = False
    mixer.eval()
    return mixer


def _build_fp16(device):
    return _make_fake_mamba2(device)


def _build_ssdi8(device):
    """Vanilla SSDi8 quantized mixer with random weights, no rotation."""
    from ssdi8.qMamba2 import Mamba2Simple, W4A8QMamba2

    mixer = _make_fake_mamba2(device)
    simple = Mamba2Simple(originalLayer=mixer, use_had_transform=True)

    act_scales = {}
    _s = torch.tensor(0.01, device=device)
    for key in [
        "in_proj:input",
        "z_act:input",
        "z_act:output",
        "x_conv_in:input",
        "x_conv_in:output",
        "x_conv_out:input",
        "B_conv_in:input",
        "B_conv_in:output",
        "B_conv_out:input",
        "C_conv_in:input",
        "C_conv_in:output",
        "C_conv_out:input",
        "dt_act:input",
        "dt_act:output",
        "ssm_state_act:input",
        "out_proj:input",
    ]:
        act_scales[key] = _s
    act_scales["chunk_state:B_scale_cgn"] = torch.full((NGROUPS,), 0.01, device=device)
    act_scales["chunk_scan:C_scale"] = torch.full((NGROUPS,), 0.01, device=device)
    act_scales["chunk_scan:cb_scale"] = torch.full((NGROUPS,), 0.01, device=device)
    act_scales["chunk_state:x_scale_chp"] = torch.full((NHEADS, HEADDIM), 0.01, device=device)
    act_scales["chunk_state:ori_x_scale_chp"] = torch.full((NHEADS, HEADDIM), 0.01, device=device)
    # Consumed as a 2-D per-(H,P) tensor by both the chunk_state and chunk_scan kernels,
    # each indexing it as pid_h*stride(0) + col*stride(1) with col < headdim. A (G, H//G, N)
    # buffer would give the wrong head stride and read out of bounds. Must be (NHEADS, HEADDIM).
    act_scales["ssd_combined:state_scale"] = torch.full((NHEADS, HEADDIM), 0.01, device=device)

    q_layer = W4A8QMamba2.from_fp16(originalLayer=simple, act_scales=act_scales, use_had_transform=True)
    q_layer.to(device).eval()
    return q_layer


def _build_ours(device, *, keyside, rotation):
    """Our QuantizedMamba2 via translate_layer_to_ours (the same path as the eval script).

    Synthetic SSDi8 calibration produces some buffers in shapes the SSD kernel rejects
    (e.g. static_state_scale = (H, N) instead of (H, P)). We override those buffers after
    translation to match the kernel contract; values are arbitrary because random-weight
    timing only cares about kernel paths.
    """
    qmixer = _build_ssdi8(device)
    original_mixer = _make_fake_mamba2(device)

    from mamba2_w4a8 import translate_layer_to_ours

    layer = translate_layer_to_ours(
        qmixer,
        original_mixer,
        device,
        keyside=keyside,
        rotate=rotation != "none",
    )

    def reinstall(name, value):
        if name in layer._buffers:
            del layer._buffers[name]
        layer.register_buffer(name, value)

    hp = lambda fill: torch.full((NHEADS, HEADDIM), fill, dtype=torch.float32, device=device)  # noqa: E731
    g = lambda fill: torch.full((NGROUPS,), fill, dtype=torch.float32, device=device)  # noqa: E731

    sss = hp(0.01)
    reinstall("static_state_scale", sss)
    reinstall("static_cb_scale", g(0.01))
    if not hasattr(layer, "alpha_hp") or layer.alpha_hp is None or layer.alpha_hp.shape != (NHEADS, HEADDIM):
        reinstall("alpha_hp", hp(0.01))  # key-side x quant scale, (H, P)
    if keyside:
        reinstall("x_quant_scale_hp", hp(0.01))  # key-side persistent x scale
    # B_inv_scale, C_inv_scale must be (G,) for the value-side path's [None, None, :, None] indexing.
    reinstall("B_inv_scale", g(100.0))
    reinstall("C_inv_scale", g(100.0))
    # bscale_g must be 1-D (G,); synthetic SSDi8 calibration may collapse it to a scalar.
    reinstall("bscale_g", g(0.01))
    # comp_out fairness buffer (zero): SSDi8 layers always pay this add.
    if not hasattr(layer, "comp_out") or layer.comp_out is None:
        reinstall("comp_out", torch.zeros(D_MODEL, dtype=torch.float16, device=device))

    for proj in (layer.in_proj, layer.out_proj):  # synthetic W4A8 input scales
        if getattr(proj, "input_scale", None) is not None:
            proj.input_scale.fill_(0.01)

    if keyside:  # static b_gate_scale (1,1,H,N)
        reinstall("b_gate_scale", torch.full((1, 1, NHEADS, D_STATE), 0.01, dtype=torch.float32, device=device))

    layer.to(device).eval()
    return layer


_BUILDERS = {
    "fp16": lambda d: _build_fp16(d),
    "ssdi8": lambda d: _build_ssdi8(d),
    "ours_value_none": lambda d: _build_ours(d, keyside=False, rotation="none"),
    "ours_value_r": lambda d: _build_ours(d, keyside=False, rotation="r"),
    "ours_key_r": lambda d: _build_ours(d, keyside=True, rotation="r"),
}


def build_layer(config: str, device: str = "cuda"):
    """Construct a single Mamba-2 mixer for the named timing config."""
    if config not in _BUILDERS:
        raise ValueError(f"Unknown config: {config!r}. Valid: {list(_BUILDERS)}")
    return _BUILDERS[config](device)


def build_model(config: str, model_id: str = "state-spaces/mamba2-1.3b", device: str = "cuda"):
    """Construct a full MambaLMHeadModel with the named config's mixer in every layer.

    Random weights matching the layer-level builders: NOT numerically valid, for timing
    only. The model size is detected from model_id. For `fp16` the loaded weights are kept;
    for every other config each mixer is replaced with `build_layer(config, device)`.
    """
    if "8b" in model_id.lower():
        set_model_size("8b")
    elif "2.7b" in model_id.lower():
        set_model_size("2.7b")
    else:
        set_model_size("1.3b")
    import filelock

    filelock.FileLock = filelock.SoftFileLock
    os.environ.setdefault("SOFT_FILELOCK", "1")

    # Routes the W4A8 GEMM of both SSDi8's and our linears through vLLM's Marlin kernel.
    import vllm_kernel_patch  # noqa: F401
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    print(f"[build_model] loading FP16 skeleton {model_id}...")
    model = MambaLMHeadModel.from_pretrained(model_id, device=device, dtype=torch.float16)
    for layer in model.backbone.layers:
        layer.mixer.use_mem_eff_path = False
    model.eval()

    if config == "fp16":
        return model

    n = len(model.backbone.layers)
    print(f"[build_model] building {n} random-weight mixers for config={config}...")
    for i in range(n):
        model.backbone.layers[i].mixer = build_layer(config, device)
    torch.cuda.empty_cache()
    return model
