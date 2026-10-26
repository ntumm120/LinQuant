"""Fake quantizers for the Q-control contract: per-role static scales calibrated by absmax."""

from __future__ import annotations

import torch

SCHEMES = ("none", "pt_static", "ph_static", "pc_static", "row_dyn")
ROLES = ("q_read", "q_local", "k_local", "w_read", "v_local", "e_local", "k_write", "v_write", "e_write", "state")

# Realizable scale policies (what the zoo INT8 kernels in linquant/kernels/zoo implement). The delta-rule WY prep
# (Gram matrix, triangular solve, RHS) stays fp32; the state-read operand W, the readout/write key factor and the
# pseudo-values E are quantized. Roles not listed fall back to the arm's global scheme.
POLICIES = {
    "kernel_static": {"q_read": "ph_static", "q_local": "ph_static", "k_local": "ph_static", "w_read": "ph_static",
                      "v_local": "pc_static", "e_local": "pc_static",
                      "k_write": "pc_static", "v_write": "pc_static", "e_write": "pc_static", "state": "pc_static"},
}
# Coarsest realizable profile: one static scalar per role per layer (what SSDi8 does for B, C and x); the INT8
# state keeps its per-(H,V) scale. Same kernels — a per-tensor scale is a per-head/per-channel tensor filled
# with one value.
POLICIES["kernel_pt"] = {r: ("pt_static" if r != "state" else sc) for r, sc in POLICIES["kernel_static"].items()}
# SSDi8 analog: every key role per tensor (SSDi8 reuses one INT8 B for the local product and the write), value
# roles per channel.
_KEY_ROLES = ("q_read", "q_local", "k_local", "k_write", "w_read")
POLICIES["kernel_ptk"] = {**POLICIES["kernel_static"], **{r: "pt_static" for r in _KEY_ROLES}}


class FakeQuant:
    """Callable ``qz(role, x)`` used by the reference core; ``x`` is [B,H,rows,D] in float64.

    scheme: none | pt_static (one scale per layer+role) | ph_static (per head) | pc_static (per head+channel)
            | row_dyn (per-row dynamic absmax; homogeneous control)
    Calibration: set ``calibrating=True``, run windows, ``finalize()``; ``layer`` is set by the harness.
    """

    def __init__(self, scheme="none", bits=8, roles=("ALL",), scheme_by_role=None, state_bits=None, round_away=False):
        assert scheme in SCHEMES, f"unknown scheme {scheme!r}"
        assert all(r in ROLES or r == "ALL" for r in roles), f"unknown role in {roles}"
        self.scheme, self.bits, self.roles = scheme, bits, set(roles)
        self.scheme_by_role = scheme_by_role or {}  # role -> scheme override ("none" disables the role)
        for r, sc in self.scheme_by_role.items():
            assert r in ROLES and sc in SCHEMES, f"bad qmap entry {r}.{sc}"
        self.state_bits = state_bits  # None = state in reference precision; 8 = INT8 per (H,P) static
        self.round_away = round_away  # kernels round half away from zero; torch.round is ties-to-even
        self.tie_roles = False  # set for kernel-lineage policies (see TIES)
        self.local_fp16 = False  # kernel lineages: the intra-chunk value product is an fp16 dot (A rounded to fp16)
        self.kernel_arith = False  # kernel lineages: INT8 codes from x*(1/scale) in fp32, fp32 chunk-local cumsum
        self.qmax = 2 ** (bits - 1) - 1
        self.calibrating, self.layer = False, 0
        self.absmax: dict[tuple, torch.Tensor] = {}
        self.scales: dict[tuple, torch.Tensor] = {}
        self.disabled = False  # True during the diagonal-scaling statistics pre-pass: every role passes through

    TIES = {"q_local": "q_read"}  # kernel lineage: one INT8 q tile serves the state read and the Q·Kᵀ product

    def _key(self, role, layer=None):
        return (self.layer if layer is None else layer, self.TIES.get(role, role) if self.tie_roles else role)

    def scheme_for(self, role):
        if role == "state":
            return (self.scheme_by_role.get("state", "pc_static") if self.state_bits else "none")
        if role in self.scheme_by_role:
            return self.scheme_by_role[role]
        return self.scheme if ("ALL" in self.roles or role in self.roles) else "none"

    def __call__(self, role, x):
        """x is [B,H,rows,D]; role "state" passes the chunk state [B,H,K,V] (scale per (H,V) like SSDi8)."""
        if self.disabled:
            return x
        scheme = self.scheme_for(role)
        if scheme == "none":
            return x
        qmax = 2 ** (self.state_bits - 1) - 1 if role == "state" else self.qmax
        key = self._key(role)
        if scheme == "row_dyn" and self.calibrating:
            return x
        if self.calibrating:
            a = x.abs()
            a = a.amax(dim=(0, 2)) if scheme == "pc_static" else a.amax(dim=(0, 2, 3)) if scheme == "ph_static" else a.amax()
            self.absmax[key] = torch.maximum(self.absmax[key], a) if key in self.absmax else a
            return x
        if scheme == "row_dyn":
            scale = x.abs().amax(dim=-1, keepdim=True) / qmax
        elif scheme == "pc_static":
            scale = self.scales[key].unsqueeze(0).unsqueeze(2)  # [1,H,1,D]
        elif scheme == "ph_static":
            scale = self.scales[key].view(1, -1, 1, 1)  # [1,H,1,1]
        else:
            scale = self.scales[key]
        scale = scale.clamp_min(1e-12)
        if self.kernel_arith:  # mirror the Triton kernels: fp32 operand times fp32 reciprocal, then round
            xq = self.round((x.float() * (1.0 / scale.float())).double())
        else:
            xq = self.round(x / scale)
        return xq.clamp(-qmax, qmax) * scale

    def round(self, x):
        if self.round_away:
            return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))
        return x.round()

    def kernel_scales(self, layer, H, K, V):
        """Scale tensors for the zoo INT8 kernels from a calibrated kernel_static run (float32, CUDA)."""
        def get(role, shape):
            sc = self.scales.get(self._key(role, layer) if role != "state" else (layer, "state"))
            if sc is None:
                return torch.ones(shape, dtype=torch.float32, device="cuda")
            sc = sc.float().clamp_min(1e-6)  # the reference clamps too: an all-zero channel must not export 1/0
            return sc.reshape(shape) if sc.numel() > 1 else sc.expand(shape).contiguous()

        return {"q": get("q_read", (H,)), "k": get("k_local", (H,)), "kw": get("k_write", (H, K)),
                "vw": get("v_write", (H, V)), "vl": get("v_local", (H, V)), "state": get("state", (H, V))}

    def state_scale(self, layer, like):
        """Static (H,V) scale of the INT8 state for ``layer`` (None while calibrating / when the state is fp)."""
        key = (layer, "state")
        if self.disabled or not self.state_bits or self.calibrating:
            return None
        scheme = self.scheme_for("state")
        assert scheme in ("pt_static", "ph_static", "pc_static"), f"state scheme must be static, got {scheme}"
        if key not in self.scales:
            raise KeyError(f"no calibrated state scale for layer {layer}")
        sc = self.scales[key].to(like.dtype).clamp_min(1e-12)
        if self.kernel_arith:
            sc = sc.float().to(like.dtype)  # the kernel holds the state scale in fp32
        if scheme == "pc_static":
            return sc.unsqueeze(0).unsqueeze(2)  # [1,H,1,V]
        if scheme == "ph_static":
            return sc.view(1, -1, 1, 1)  # [1,H,1,1]
        return sc  # scalar

    def finalize(self):
        self.scales = {k: a / ((2 ** (self.state_bits - 1) - 1) if k[1] == "state" else self.qmax)
                       for k, a in self.absmax.items()}
        self.calibrating = False

    def reset(self):
        self.absmax, self.scales = {}, {}
