"""Quantized Mamba-2 mixer: W4A8 linears + FP16 conv+silu + Kronecker R rotation + INT8 SSD.

Drop-in replacement for the Mamba-2 mixer (prefill only). Linear projections are
W4A8 (4-bit Marlin weights, per-tensor static INT8 input). Conv1d and SiLU stay in
FP16. B and C are rotated along the state axis with a per-group Kronecker-Hadamard
R rotation (optionally after the diagonal key rescaling D_qk) and quantized to INT8;
x stays FP16 into the SSD kernels, which quantize it per tile (value-side) or
in-loop (key-side). Chunk states, the inter-chunk recurrence and CB are INT8.

Mamba-2 1.3b dimensions: H=64, P=64, G=1, N=128, d_model=2048, d_ssm=4096, chunk_size=256.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from causal_conv1d import causal_conv1d_fn
from einops import rearrange

from linquant.hadamard_utils import qhad_transform
from linquant.kernels.hadamard import kronecker_hadamard_fwd
from linquant.w4a8_linear import W4A8B16O16Linear, _fused_fp16_to_int8

from .kernels.mamba2.fused_split_rotate_xBC import fused_split_rotate_quant_xBC, prepare_dqk
from .kernels.mamba2.ssd_combined import _quant_mamba_chunk_scan_combined_fwd


def _apply_mask(hidden_states, attention_mask):
    """Zero out padding positions."""
    if attention_mask is not None and attention_mask.shape[1] > 1:  # any nontrivial mask, batch size included
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


class QuantizedMamba2(nn.Module):
    """W4A8 linear + FP16 conv + Kronecker Hadamard R rotation + INT8 SSD Mamba-2 mixer.

    Pipeline:
      1. in_proj (W4A8, FP16 out) -> split z, xBC, dt
      2. causal_conv1d_fn(xBC, conv_weight, conv_bias, activation="silu") -> FP16 xBC
      3. Fused split + (D_qk rescale) + R rotation + INT8 quantize of B, C; x as an FP16 view
      4. INT8 SSD sub-kernels (INT8 chunk states, fixed-point recurrence, INT8 CB) -> FP16 y
      5. RMSNormGated(y, z) -> Hadamard + INT8 quant -> out_proj (W4A8) -> + comp_out

    Calibrated buffers attached after construction (by the eval scripts):
      B_scale/B_inv_scale, C_scale/C_inv_scale (G,), A_log_scale, D_scale (scalars),
      alpha_hp (H, P), bscale_g (G,), static_state_scale (H, P), static_cb_scale (G,),
      r_signs (G, N) or None, comp_out (d_model) or absent, dqk_b/dqk_c (G*N) or absent;
      key-side additionally x_quant_scale_hp (H, P) and b_gate_scale (1, 1, H, N).
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_ssm: int,
        nheads: int,
        headdim: int,
        d_state: int,
        ngroups: int,
        d_conv: int,
        chunk_size: int,
        in_proj: nn.Module,
        out_proj: nn.Module,
        conv1d: nn.Conv1d,
        norm: nn.Module,
        A_log: torch.Tensor,
        D: nn.Parameter,
        dt_bias: nn.Parameter,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        layer_idx: int | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ssm = d_ssm
        self.nheads = nheads
        self.headdim = headdim
        self.d_state = d_state
        self.ngroups = ngroups
        self.d_conv = d_conv
        self.chunk_size = chunk_size
        self.dt_limit = dt_limit
        self.layer_idx = layer_idx

        self.conv_dim = d_ssm + 2 * ngroups * d_state
        # Cached constant to avoid torch.tensor() allocation every forward call
        self.register_buffer("_one", torch.tensor(1.0, dtype=torch.float32))

        # W4A8 linear projections (ours or SSDi8's; both expose `input_scale`)
        self.in_proj = in_proj
        self.out_proj = out_proj

        # FP16 conv1d (depthwise)
        self.conv1d = conv1d

        # Norm
        self.norm = norm

        # SSM parameters
        self.register_buffer("A_log", A_log)  # INT8 (dequantized in-kernel with A_log_scale)
        self.D = D  # INT8 (dequantized in-kernel with D_scale)
        self.dt_bias = dt_bias

    @classmethod
    def from_fp16(  # noqa: D417
        cls,
        original_mixer,
        scales: dict[str, torch.Tensor],
        r_signs: torch.Tensor | None,
    ) -> QuantizedMamba2:
        """Convert a GPTQ'd Mamba2 mixer to a QuantizedMamba2 with our W4A8 linears.

        Args:
            original_mixer: Mamba2 nn.Module with GPTQ'd weights. Supports vanilla
                mamba_ssm.Mamba2 and HF-style wrappers (attribute names are resolved
                via getattr fallbacks).
            scales: dict with keys B_scale, B_inv_scale, C_scale, C_inv_scale (G,),
                A_log_scale, A_log_inv_scale, D_scale, D_inv_scale (scalars),
                in_proj_input_scale, out_proj_input_scale (scalars).
            r_signs: (ngroups, d_state) sign vectors for the R rotation (state-axis), or None.

        Returns:
            QuantizedMamba2 instance; the per-(H,P)/per-(G,N) SSD buffers are attached by the caller.
        """
        # Handle mamba-ssm Mamba2 (nheads, headdim, ...) and HF wrappers (num_heads, head_dim, ...)
        nheads = getattr(original_mixer, "nheads", None) or getattr(
            original_mixer, "num_heads", None
        )
        headdim = getattr(original_mixer, "headdim", None) or getattr(
            original_mixer, "head_dim", None
        )
        d_model = getattr(original_mixer, "d_model", None) or getattr(
            original_mixer, "hidden_size", None
        )
        d_state = getattr(original_mixer, "d_state", None) or getattr(
            original_mixer, "ssm_state_size", None
        )
        ngroups = getattr(original_mixer, "ngroups", None) or getattr(
            original_mixer, "n_groups", None
        )
        d_ssm = nheads * headdim
        d_conv = getattr(original_mixer, "d_conv", None) or getattr(
            original_mixer, "conv_kernel_size", None
        )
        chunk_size = original_mixer.chunk_size
        dt_limit = getattr(
            original_mixer,
            "dt_limit",
            getattr(original_mixer, "time_step_limit", (0.0, float("inf"))),
        )
        if isinstance(dt_limit, list):
            dt_limit = tuple(dt_limit)

        # W4A8 linears: SSDi8-matched kernel (per-tensor static INT8 + Marlin QQQ W4A8 GEMM)
        w4_in_proj = W4A8B16O16Linear.from_fp16(
            original_mixer.in_proj, input_scale=scales["in_proj_input_scale"]
        )
        w4_out_proj = W4A8B16O16Linear.from_fp16(
            original_mixer.out_proj, input_scale=scales["out_proj_input_scale"]
        )

        # Quantize A_log to INT8 (kernel dequants via A_log_scale * int8.to(f32))
        q_A_log = torch.clamp(
            torch.round(original_mixer.A_log.float() * scales["A_log_inv_scale"]), -127, 127
        ).to(torch.int8)

        # Quantize D to INT8 (same as SSDi8: per-tensor absmax)
        q_D = torch.clamp(
            torch.round(original_mixer.D.float() * scales["D_inv_scale"]), -127, 127
        ).to(torch.int8)

        module = cls(
            d_model=d_model,
            d_ssm=d_ssm,
            nheads=nheads,
            headdim=headdim,
            d_state=d_state,
            ngroups=ngroups,
            d_conv=d_conv,
            chunk_size=chunk_size,
            in_proj=w4_in_proj,
            out_proj=w4_out_proj,
            conv1d=original_mixer.conv1d,
            norm=original_mixer.norm,
            A_log=q_A_log,
            D=q_D,
            dt_bias=original_mixer.dt_bias,
            dt_limit=dt_limit,
            layer_idx=original_mixer.layer_idx,
        )

        module.register_buffer("r_signs", r_signs)

        # INT8 quantization scale buffers. x is consumed as FP and the chunk_state
        # kernel quantizes per-tile internally (SSDi8 pattern), so there is no x_scale.
        for name in ("B_scale", "B_inv_scale", "C_scale", "C_inv_scale", "A_log_scale", "D_scale"):
            module.register_buffer(name, scales[name])

        return module

    @torch.no_grad()
    def forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass compatible with both mamba-ssm Block and HF calling conventions.

        mamba-ssm Block calls: mixer(hidden_states, inference_params=...)
        HF layers call: layer(hidden_states, attention_mask=..., past_key_values=...)

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        if past_key_values is not None or kwargs.get("inference_params") is not None:
            raise NotImplementedError(
                "QuantizedMamba2 is prefill-only: cached/stateful decoding (past_key_values, inference_params) is not "
                "implemented; the conv history and the SSM state are not carried between calls"
            )
        batch_size, seq_len, _ = hidden_states.shape

        if hidden_states.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float16)

        # --- 1. W4A8 in_proj ---
        if hasattr(self.in_proj, "forward_int8"):
            # Our W4A8B16O16Linear (from_fp16 path). Fused FP32→INT8 quant (matches
            # SSDi8's QRMSNorm→INT8 path): rint(x.float() / s) clamped to [-128, 127].
            # The kernel computes in FP32 with a true divide, so it is bit-exact with
            # the elementwise chain it replaces.
            s = self.in_proj.input_scale
            h_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            h_i8 = _fused_fp16_to_int8(h_2d, s, clamp_min=-128.0, reciprocal=False)
            h_i8 = h_i8.view(hidden_states.shape)
            projected = self.in_proj.forward_int8(h_i8, s)
        else:
            # SSDi8's W4A8 linear reused from the SSDi8 mixer (Mamba-2 path): it
            # quantizes the FP16 input with its own input_scale.
            projected = self.in_proj(hidden_states)

        # Split: [z0, x0, z, xBC, dt]. d_mlp = 0 for mamba2-1.3b/2.7b/8b, so
        # z0/x0 are empty for those configs. Captured for d_mlp > 0 configs
        # to match SSDi8's qMamba2 forward (qMamba2.py:618-619).
        groups_state_size = self.ngroups * self.d_state
        d_mlp = (projected.shape[-1] - 2 * self.d_ssm - 2 * groups_state_size - self.nheads) // 2

        z0, x0, z, xBC, dt = projected.split(
            [d_mlp, d_mlp, self.d_ssm, self.conv_dim, self.nheads],
            dim=-1,
        )

        # --- 2. Causal conv1d + SiLU (FP16 weights, same as SSDi8) ---
        xBC = _apply_mask(xBC, attention_mask)
        keyside = getattr(self, "keyside", False)

        xBC = causal_conv1d_fn(
            x=xBC.transpose(1, 2),
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation="silu",
        ).transpose(1, 2)
        xBC = _apply_mask(xBC, attention_mask)
        # diagonal key scaling (B·D⁻¹, C·D on the conv+SiLU outputs, where SSDi8 calibrates B/C): fused into the
        # split/rotate/quant kernel below, so it costs nothing at runtime
        dqk_b = getattr(self, "dqk_b", None)
        dqk_c = getattr(self, "dqk_c", None)
        if dqk_b is not None:  # prepared once (fp16-rounded, contiguous fp32): no per-call conversions
            if getattr(self, "_dqk_prepared", None) is None:
                self._dqk_prepared = (prepare_dqk(dqk_b), prepare_dqk(dqk_c))
            dqk_b, dqk_c = self._dqk_prepared

        # --- 3. Split, rotate, and INT8-quantize B/C (x stays an FP16 view of xBC) ---
        x_fp, q_B, q_C = fused_split_rotate_quant_xBC(
            xBC,
            self.r_signs,
            self.nheads,
            self.ngroups,
            self.d_state,
            self.headdim,
            self.B_inv_scale,
            self.C_inv_scale,
            dqk_b=dqk_b,
            dqk_c=dqk_c,
        )
        b_fp_keyside = None
        if keyside:
            d_ssm = self.nheads * self.headdim
            g_n = self.ngroups * self.d_state
            # b_fp_keyside MUST be rotated when r_signs is set — chunk_state_keyside
            # consumes B with the same rotation that q_B (used by bmm_chunk) was
            # quantized under. Slicing raw xBC gives unrotated B; apply R here.
            # Make contiguous before rotation (kernel may not handle stride-T
            # views cleanly).
            b_fp_keyside = xBC[..., d_ssm : d_ssm + g_n]
            if dqk_b is not None:  # the fp B consumed by chunk_state_keyside carries D as well
                b_fp_keyside = b_fp_keyside * dqk_b.to(xBC.dtype)
            b_fp_keyside = b_fp_keyside.reshape(
                xBC.shape[0], xBC.shape[1], self.ngroups, self.d_state
            ).contiguous()
            if self.r_signs is not None:
                b_fp_keyside = kronecker_hadamard_fwd(b_fp_keyside, self.r_signs, self.ngroups)

        del xBC

        # --- 4. INT8 SSD sub-kernels ---
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else {"dt_limit": self.dt_limit}
        )
        dt_scale = self._one.to(device=dt.device)

        keyside_kwargs = {}
        if keyside:
            keyside_kwargs = {
                "keyside": True,
                "x_scale_hp": self.x_quant_scale_hp,
                "b_fp": b_fp_keyside,
                "b_gate_scale": self.b_gate_scale,
            }

        missing = [n for n in ("alpha_hp", "bscale_g", "static_state_scale", "static_cb_scale")
                   if not hasattr(self, n)]
        if missing:
            raise RuntimeError(f"QuantizedMamba2 is not calibrated: missing buffers {missing} "
                               "(from_fp16 builds the module; the calibration step must attach these before inference)")
        y, _ = _quant_mamba_chunk_scan_combined_fwd(
            x_fp=x_fp,
            q_dt=dt,
            dt_scale=dt_scale,
            q_A_log=self.A_log,
            A_log_scale=self.A_log_scale,
            q_B=q_B,
            B_scale=self.B_scale,
            q_C=q_C,
            C_scale=self.C_scale,
            chunk_size=self.chunk_size,
            q_D=self.D,
            D_scale=self.D_scale,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            alpha_hp=self.alpha_hp,
            bscale_g=self.bscale_g,
            static_state_scale=self.static_state_scale,
            static_cb_scale=self.static_cb_scale,
            **keyside_kwargs,
            **dt_limit_kwargs,
        )

        # --- 5. Reshape + RMSNormGated(y, z) ---
        # SSDi8-aligned: z stays FP (see libs/SSDi8/ssdi8/triton/quant_normgated.py:54
        # where the z fake-quant multiply is commented out).
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)

        if d_mlp > 0:
            y = torch.cat([torch.nn.functional.silu(z0) * x0, y], dim=-1)

        # --- 6. Output Hadamard + INT8 quant + W4A8 out_proj ---
        y_i8, scale = qhad_transform(y, self.out_proj.input_scale)
        del y
        if hasattr(self.out_proj, "forward_int8"):
            y = self.out_proj.forward_int8(y_i8, scale)
        else:
            y = self.out_proj(y_i8)  # SSDi8's W4A8 linear: INT8 input on its input_scale
        del y_i8

        # --- 7. Output compensation (SSDi8-style bias correction) ---
        comp_out = getattr(self, "comp_out", None)
        if comp_out is not None:
            y = y + comp_out

        # Match original Mamba2.forward() which returns just the output tensor
        return y
