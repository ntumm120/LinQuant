"""Patch the mixer-zoo UnifiedMixer to route the recurrence through the reference core."""

from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

from linquant.invariances import FakeQuant
from linquant.reference import chunk_core, hadamard_rotation

# The zoo models and their modelling code come from the mixer-zoo training repo (github.com/phnazari/LinQuant).
# MIXERZOO_REPO: a checkout of that repo. MIXERZOO_MODELS: the trained checkpoints, laid out as
# <MIXERZOO_MODELS>/<size>/<cell>/model.safetensors (+ config and tokenizer files), size in {"340m", "1.3b"}.
MIXERZOO_REPO = os.environ.get("MIXERZOO_REPO", "")
MIXERZOO_MODELS = os.environ.get("MIXERZOO_MODELS", "")
CELLS = {  # cell -> (family, decay)
    "linear_none": ("linear", "none"),
    "linear_none_beta": ("linear", "none"),
    "mamba2": ("linear", "scalar"),
    "mamba2_beta": ("linear", "scalar"),
    "gla": ("linear", "diagonal"),
    "gla_beta": ("linear", "diagonal"),
    "deltanet": ("delta", "none"),
    "gated_deltanet": ("delta", "scalar"),
    "kda": ("delta", "diagonal"),
    "linear_none_nonorm": ("linear", "none"),  # no QK norm: is the invariance story norm-dependent?
    "mamba2_nonorm": ("linear", "scalar"),
    "gla_nonorm": ("linear", "diagonal"),
    "linear_none_beta_nonorm": ("linear", "none"),  # gated no-norm cells: β placement × norm
    "mamba2_beta_nonorm": ("linear", "scalar"),
    "gla_beta_nonorm": ("linear", "diagonal"),
}
FLA_FN_CELL = {  # name in unified_mixer's namespace -> (family, decay)
    "chunk_simple_gla": None,  # decay decided by g
    "chunk_gla": ("linear", "diagonal"),
    "chunk_delta_rule": ("delta", "none"),
    "chunk_gated_delta_rule": ("delta", "scalar"),
    "chunk_kda": ("delta", "diagonal"),
}


class Arm:
    """One experimental arm: rotation / eta / rho configuration + quantizer."""

    def __init__(self, rot_qk=False, rot_v=False, eta=0.0, rho=1.0, rot_seed=0, quant: FakeQuant | None = None,
                 read_post=False, storage="rotated", tau=1.0, smooth_qk=0.0, smooth_v=0.0):
        self.rot_qk, self.rot_v, self.eta, self.rho, self.rot_seed = rot_qk, rot_v, eta, rho, rot_seed
        self.quant = quant or FakeQuant("none")
        self.read_post, self.storage, self.tau = read_post, storage, tau  # tau: delta write placement
        self.smooth_qk, self.smooth_v = smooth_qk, smooth_v  # diagonal scaling exponents (0 = off)
        self.smooth_stats: dict = {}  # (layer, name) -> per-(H,D) absmax collected in the stats pass
        self.smooth_D: dict = {}  # (layer, "qk"|"v") -> D per (H,D)
        self.stats_pass = False
        self._rots = {}

    def need_stats_pass(self):
        return (self.smooth_qk > 0 or self.smooth_v > 0) and not self.smooth_D

    def collect(self, layer, name, x):
        a = x.detach().abs().amax(dim=(0, 1)).double()  # [H,D]
        k = (layer, name)
        self.smooth_stats[k] = torch.maximum(self.smooth_stats[k], a) if k in self.smooth_stats else a

    def finalize_smooth(self):
        """Q' = Q D, K' = K D^-1 with D = (max|k| / max|q|)^alpha per (H,N); V' = V Dv, o' = o Dv^-1 with
        Dv = (geomean_p max|v_p| / max|v_p|)^alpha per (H,P) (equalizes value channels)."""
        layers = {k[0] for k in self.smooth_stats}
        for L in layers:
            if self.smooth_qk > 0 and (L, "q") in self.smooth_stats:
                q, k = self.smooth_stats[(L, "q")].clamp_min(1e-8), self.smooth_stats[(L, "k")].clamp_min(1e-8)
                self.smooth_D[(L, "qk")] = (k / q) ** self.smooth_qk
            if self.smooth_v > 0 and (L, "v") in self.smooth_stats:
                v = self.smooth_stats[(L, "v")].clamp_min(1e-8)
                self.smooth_D[(L, "v")] = (v.log().mean(dim=-1, keepdim=True).exp() / v) ** self.smooth_v
        self.stats_pass = False

    def scale_qkv(self, layer, q, k, v):
        """Apply the diagonal scalings (exact with the matching un-scaling of the output)."""
        Dqk, Dv = self.smooth_D.get((layer, "qk")), self.smooth_D.get((layer, "v"))
        if Dqk is not None:
            q, k = q.double() * Dqk, k.double() / Dqk  # float64: chunk_core runs in float64 anyway
        if Dv is not None:
            v = v.double() * Dv
        return q, k, v, Dv

    def rotation(self, n, layer, device):
        key = (n, layer)
        if key not in self._rots:
            self._rots[key] = hadamard_rotation(n, self.rot_seed * 1000 + layer, device)
        return self._rots[key]

    def record(self):
        q = self.quant
        return dict(rot_qk=self.rot_qk, rot_v=self.rot_v, eta=self.eta, rho=self.rho, rot_seed=self.rot_seed,
                    quant=q.scheme, bits=q.bits, qmap=q.scheme_by_role, state_bits=q.state_bits,
                    state_pass="fp32carry", read_post=self.read_post, storage=self.storage, delta_rv="align",
                    tau=self.tau, smooth_qk=self.smooth_qk, smooth_v=self.smooth_v, round_away=q.round_away)


# ----------------------------------------------------------------------------- mixer zoo
_MIXERZOO_MODULES: dict = {}


def _mixerzoo(module):
    """Import ``module`` (dotted path) of the mixer-zoo repo from its file. That repo's package is also called
    ``linquant``, so it cannot be imported by name next to this package; its top-level helpers (``utils``,
    ``unified_mixer``) do not clash and are imported normally once the repo is on ``sys.path``."""
    if not MIXERZOO_REPO:
        raise RuntimeError("set MIXERZOO_REPO to a checkout of the mixer-zoo repo (github.com/phnazari/LinQuant)")
    if MIXERZOO_REPO not in sys.path:
        sys.path.insert(0, MIXERZOO_REPO)
    if module not in _MIXERZOO_MODULES:
        path = os.path.join(MIXERZOO_REPO, *module.split(".")) + ".py"
        spec = importlib.util.spec_from_file_location("mixerzoo_" + module.replace(".", "_"), path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MIXERZOO_MODULES[module] = mod
    return _MIXERZOO_MODULES[module]


def find_model_dir(cell, size):
    if not MIXERZOO_MODELS:
        raise RuntimeError("set MIXERZOO_MODELS to the checkpoint root (<root>/<size>/<cell>/model.safetensors)")
    p = os.path.join(MIXERZOO_MODELS, size, cell)
    if os.path.exists(os.path.join(p, "model.safetensors")):
        return p
    raise FileNotFoundError(f"{cell}@{size}: {p}")


def load_zoo(cell, size, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _mixerzoo("linquant.models.mixer_zoo").unified_mixer_cls()
    path = find_model_dir(cell, size)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def patch_zoo(model, holder: dict):
    """Route every FLA chunk call inside UnifiedMixer through chunk_core with ``holder["arm"]``'s settings."""
    cls = _mixerzoo("linquant.models.mixer_zoo").unified_mixer_cls()
    import unified_mixer as um

    mixers = [m for m in model.modules() if isinstance(m, cls)]
    assert mixers, "patch_zoo: no UnifiedMixer layers found"
    for i, m in enumerate(mixers):
        m.register_forward_pre_hook(lambda mod, args, i=i: setattr(holder["arm"].quant, "layer", i))
        if getattr(m, "_use_beta", False):  # stash the write gate so linear cells can re-place it
            m.b_proj.register_forward_hook(lambda mod, args, out: holder.__setitem__("beta", out.sigmoid()))

    def make(name):
        def fn(q, k, v, g=None, beta=None, **kw):
            arm = holder["arm"]
            fam, dec = FLA_FN_CELL[name] or ("linear", "none" if g is None else "scalar")
            K, V = q.shape[-1], v.shape[-1]
            layer = arm.quant.layer
            v_dtype = v.dtype
            if fam == "linear" and beta is None and arm.rho != 1.0 and "beta" in holder:
                # the mixer pre-multiplied v by beta; move beta^(1-rho) onto k (rho=1: all on v, rho=0: all on k)
                b = holder["beta"].double().unsqueeze(-1)
                v = (v.double() / b.clamp_min(1e-6)) * b**arm.rho  # float64: chunk_core runs in float64 anyway
                k = k.double() * b ** (1 - arm.rho)
            if arm.stats_pass:
                arm.collect(layer, "q", q); arm.collect(layer, "k", k); arm.collect(layer, "v", v)
            q, k, v, Dv = arm.scale_qkv(layer, q, k, v)
            holder["calls"] = holder.get("calls", 0) + 1
            o = chunk_core(q, k, v, g, beta, family=fam, decay=dec, scale=K**-0.5, qz=arm.quant,
                           rot_qk=arm.rotation(K, layer, q.device) if arm.rot_qk else None,
                           rot_v=arm.rotation(V, 10_000 + layer, q.device) if arm.rot_v else None,
                           eta=arm.eta, rho=arm.rho if fam == "delta" else 1.0, read_post=arm.read_post,
                           storage=arm.storage, tau=arm.tau, act_dtype=v_dtype)
            if Dv is not None:
                o = o / Dv.to(o.dtype)
            return o.to(v_dtype), None

        return fn

    for name in FLA_FN_CELL:
        setattr(um, name, make(name))


# ----------------------------------------------------------------------------- data / eval
def calib_windows(tok, n, seed, seqlen=2048):
    _mixerzoo("linquant.eval")  # puts the mixer-zoo repo on sys.path for ``utils``
    from utils.data_utils import get_wikitext2

    return [inp for inp, _ in get_wikitext2(nsamples=n, seed=seed, seqlen=seqlen, tokenizer=tok)]


@torch.no_grad()
def calibrate(model, arm: Arm, windows, device="cuda"):
    if arm.need_stats_pass():  # diagonal scaling: absmax pre-pass on the un-scaled operands, quantizer fully off
        arm.stats_pass = True
        arm.quant.disabled = True
        try:
            for w in windows:
                model(w.to(device))
        finally:
            arm.quant.disabled = False
            arm.stats_pass = False
        arm.finalize_smooth()
    arm.quant.reset()
    arm.quant.calibrating = True
    for w in windows:
        model(w.to(device))
    arm.quant.finalize()


@torch.no_grad()
def mean_ce(model, windows, device="cuda"):
    """Mean next-token CE over windows (calibration-selection metric)."""
    tot = 0.0
    for w in windows:
        w = w.to(device)
        logits = model(w).logits.float()
        tot += F.cross_entropy(logits[0, :-1], w[0, 1:]).item()
    return tot / len(windows)


def wt2_ppl(model, tok, limit=None, device="cuda"):
    evaluate_ppl = _mixerzoo("linquant.eval").evaluate_ppl
    return evaluate_ppl(model, tok, dataset="wikitext2", seqlen=2048, limit=limit, device=device)["wikitext2_ppl"]
