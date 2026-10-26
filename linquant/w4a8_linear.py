"""Fixed W4A8B16O16 linear layer with correct GPTQ group_scale handling.

The bug in Quamba's original W4A8B16O16Linear.from_fp16():
  When GPTQ group_scale is present, channel_scale was computed from the RAW
  fake-quantized weights: `s_channel = max(abs(W), dim=0) / 127`. But W stores
  the FP16 fake-quantized values, which have sub-group-level precision artifacts.
  The correct reference for channel_scale is w_ref: the result of INT4 group
  quantize-then-dequantize using the GPTQ group scales. Since the Marlin kernel
  internally dequantizes as `w_fp16 = (q_int4 - 8) * s_group * s_channel`, the
  channel_scale must be derived from w_ref to keep the fused scale consistent.

  Concretely, when s_channel is wrong:
    s_group_fused = s_group_orig / s_channel_wrong
  and the Marlin kernel computes:
    w_fp16 = q_int4_offset * s_group_fused * s_channel_wrong
           = q_int4_offset * s_group_orig  (correct cancellation only if s_channel is right)
  But the INT8 activation side divides by s_channel_wrong too, so the overall
  output is scaled by (s_channel_correct / s_channel_wrong) which != 1.

This module:
  1. Uses the GPTQ group_scale for INT4 group quantize-dequantize -> w_ref
  2. Computes channel_scale from w_ref (not from raw weights)
  3. Fuses: s_group_fused = s_group / s_channel
  4. Packs weights using Quamba's Marlin permutation utilities
  5. Uses Quamba's w4a8o16_gemm CUDA kernel for the forward pass
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_fp16_to_int8_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    N,
    stride_x,
    stride_out,
    CLAMP_MIN: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused FP16 → INT8: round(x / scale), clamp, cast. No FP32 temporary.

    RECIPROCAL selects `x * (1/s)` (one fewer division, historical behaviour)
    over `x / s`. The divide form reproduces `x.float() / s` bit-exactly, which
    the SSDi8-parity in_proj path depends on.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask).to(tl.float32)
    s = tl.load(scale_ptr).to(tl.float32)
    # div_rn, not `/`: Triton lowers `/` to an approximate reciprocal, which
    # flips ~1 in 2e6 values across a rounding boundary versus torch's
    # `x.float() / s`. The parity path needs IEEE round-to-nearest.
    scaled = x * (1.0 / s) if RECIPROCAL else tl.extra.cuda.libdevice.div_rn(x, s)
    q = tl.extra.cuda.libdevice.rint(scaled)
    q = tl.minimum(tl.maximum(q, CLAMP_MIN), 127.0)
    tl.store(out_ptr + row * stride_out + offs, q.to(tl.int8), mask=mask)


def _fused_fp16_to_int8(
    x_2d: torch.Tensor,
    scale: torch.Tensor,
    clamp_min: float = -127.0,
    reciprocal: bool = True,
) -> torch.Tensor:
    """Fused FP16→INT8 quant in a single Triton kernel. No FP32 intermediate.

    Args:
        x_2d: [M, N] input.
        scale: 0-dim quantization scale, read on device (no host sync).
        clamp_min: lower clamp bound; -127 by default, -128 for SSDi8 parity.
        reciprocal: multiply by 1/scale instead of dividing. Divide is
            bit-exact with `x.float() / scale`; the reciprocal form is kept as
            the default so existing call sites are unchanged.
    """
    M, N = x_2d.shape
    out = torch.empty(M, N, device=x_2d.device, dtype=torch.int8)
    BLOCK = triton.next_power_of_2(N)
    _fused_fp16_to_int8_kernel[(M,)](
        x_2d,
        out,
        scale,
        N,
        x_2d.stride(0),
        out.stride(0),
        CLAMP_MIN=clamp_min,
        RECIPROCAL=reciprocal,
        BLOCK=BLOCK,
    )
    return out


from .cuda_ext import quant_linear_cuda
from .marlin_utils import (
    MARLIN_QQQ_MAX_PARALLEL,
    MARLIN_QQQ_MIN_THREAD_N,
    get_w4a8_permute_weights,
    get_w4a8_weight_perm,
    w4a8_permute_scales,
)


class W4A8B16O16Linear(nn.Module):
    """W4A8 linear with correct GPTQ group_scale -> channel_scale derivation.

    Stores Marlin-packed INT4 weights, per-group fused scale (s_group / s_channel),
    per-channel INT8 scale (s_channel), and per-tensor input scale. Forward pass
    calls Quamba's w4a8o16_gemm CUDA kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        device=None,
    ):
        factory_kwargs = {"device": device}
        super().__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = out_features

        self.pad_out = 0
        self.size_n, self.size_k = out_features, in_features
        # 256-byte output alignment (Quamba's W4A8 uses 64 which causes alignment issues)
        if self.size_n % 256 != 0:
            self.pad_out = 256 - self.size_n % 256
        self.size_n = self.size_n + self.pad_out

        self.max_par = 16
        max_workspace_size = (self.size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL
        self.register_buffer(
            "workspace", torch.zeros(max_workspace_size, dtype=torch.int, **factory_kwargs)
        )
        self.register_buffer(
            "weight",
            torch.empty(
                (self.size_k // 16, self.size_n * 16 // 8),
                dtype=torch.int32,
                **factory_kwargs,
            ),
        )
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32, **factory_kwargs))
        self.register_buffer(
            "s_channel", torch.empty((1, self.size_n), dtype=torch.float32, **factory_kwargs)
        )
        self.register_buffer(
            "s_group",
            torch.empty(
                (self.size_k // group_size, self.size_n),
                dtype=torch.float16,
                **factory_kwargs,
            ),
        )

    @classmethod
    @torch.no_grad()
    def from_fp16(cls, original_layer: nn.Linear, input_scale: torch.Tensor):
        """Convert an FP16 linear (possibly with GPTQ metadata) to W4A8 Marlin format.

        The key fix: channel_scale is computed from w_ref (group-dequantized INT4
        reference) rather than from the raw FP16 weights.

        Args:
            original_layer: Source FP16 nn.Linear (optionally with GPTQ group_scale).
            input_scale: Per-tensor static INT8 input scale (0-dim FP32).
        """
        assert original_layer.bias is None, "Bias not supported"

        bits = 4
        group_size = 128 if original_layer.in_features > 128 else -1
        group_scale = None  # [n_groups, out_features] from GPTQ
        if hasattr(original_layer, "apply_gptq") and original_layer.apply_gptq:
            bits = original_layer.bits
            group_size = original_layer.group_size
            group_scale = original_layer.group_scale.to(torch.float32)  # fp16 would saturate large GPTQ scales to inf
        assert bits == 4, "Only 4-bit quantization supported"
        if group_size not in [-1, 128]:
            raise ValueError("Only groupsize -1 and 128 are supported.")

        device = original_layer.weight.device
        qlinear = cls(
            original_layer.in_features,
            original_layer.out_features,
            group_size=128,
            device=device,
        )
        qlinear.pad_out = 0

        W = original_layer.weight  # [Dout, Din]
        qlinear.size_n, qlinear.size_k = W.shape
        if W.shape[0] % 256 != 0:
            qlinear.pad_out = 256 - W.shape[0] % 256

        # Transpose: [Dout, Din] -> [Din, Dout] = [size_k, size_n]
        W_t = W.cpu().to(torch.float16).t().contiguous()
        size_k, size_n = W_t.shape

        if group_size == -1:
            group_size = size_k

        quant_type = "per-channel" if group_size == size_k else "per-group"

        if group_scale is not None and group_size < size_k:
            # --- Per-group path with GPTQ group_scale: THE FIX ---
            # group_scale is [n_groups, Dout]. Quamba's w4a8_quantize format expects
            # s_group as [n_groups * Dout] flattened or [n_groups, size_n] after reshape.
            # The GPTQ group_scale maps: w_fp16 = q_int4_centered * group_scale
            #   where q_int4_centered = q_int4_unsigned - 8 (half_q_val).
            #
            # Step 1: Group quantize-dequantize using GPTQ scales to get w_ref.
            # Keep s_group in FP32 throughout this step. Earlier code cast to FP16
            # immediately, but Nemotron-H's GPTQ produces occasional group_scale
            # entries above the FP16 max (65504), which saturate to Inf in FP16
            # and then `0 * Inf = NaN` in `(q_w - 8) * s_expanded` for any INT4
            # code = 8 in that group. The NaN propagates into `s_channel` →
            # Marlin GEMM → entire output is poisoned (PPL spike from 7.49 to
            # 1640 with even one swapped layer on Nemotron-H 8B).
            s_group_orig = group_scale.cpu().to(torch.float32)  # [n_groups, Dout]
            assert torch.isfinite(s_group_orig).all(), "GPTQ group_scale has non-finite entries"

            max_q_val = 2**bits - 1  # 15
            half_q_val = (max_q_val + 1) // 2  # 8

            # Reshape W into groups: [n_groups, group_size, size_n]
            W_grouped = W_t.reshape(-1, group_size, size_n).to(torch.float32)
            # s_group_orig: [n_groups, size_n] -> expand for broadcast
            s_expanded = s_group_orig.unsqueeze(1)  # [n_groups, 1, size_n]

            # Quantize to unsigned INT4 using GPTQ scales
            q_w = torch.round(W_grouped / s_expanded) + half_q_val
            q_w = torch.clamp(q_w, 0, max_q_val).int()

            # Dequantize to get w_ref (the true INT4 reference) in FP32
            w_ref = (q_w - half_q_val).to(torch.float32) * s_expanded

            # Flatten back to [size_k, size_n]
            q_w = q_w.reshape(size_k, size_n).contiguous()
            w_ref = w_ref.reshape(size_k, size_n).contiguous()

            # Step 2: Compute channel_scale FROM w_ref (not from W_t!)
            s_channel = torch.max(torch.abs(w_ref), dim=0, keepdim=True)[0]  # [1, size_n]
            s_channel = s_channel / 127.0
            # Avoid division by zero for dead channels
            s_channel = torch.clamp(s_channel, min=1e-10)
            assert not torch.isnan(s_channel).any(), (
                "s_channel has NaN — group_scale likely overflowed FP16. "
                "Keep w_ref computation in FP32 (already done) and verify "
                "group_scale magnitudes from GPTQ are finite."
            )
            s_channel = s_channel.to(dtype=torch.float32)

            # Step 3: Fuse scales: s_group_fused = s_group / s_channel
            # The Marlin kernel computes:
            #   output = (q_int4_centered * s_group_fused) * s_channel * input_int8
            # So s_group_fused * s_channel = s_group_orig, giving correct dequantization.
            s_group_fused = (s_group_orig.reshape(-1, size_n).contiguous() / s_channel).to(
                dtype=torch.float16
            )
            s_group_fused = torch.nan_to_num(s_group_fused, nan=0.0)

        else:
            # Per-channel path (no GPTQ group_scale) or group_size == size_k:
            # Fall back to Quamba's get_w4a8_quantize_weights which handles both cases correctly.
            from .marlin_utils import get_w4a8_quantize_weights

            w_ref, q_w, s_group_fused, s_channel = get_w4a8_quantize_weights(W_t, bits, group_size)

        # --- Padding ---
        if qlinear.pad_out != 0:
            w_ref = torch.nn.functional.pad(w_ref, (0, qlinear.pad_out, 0, 0), "constant", 0)
            q_w = torch.nn.functional.pad(q_w, (0, qlinear.pad_out, 0, 0), "constant", 0)
            s_group_fused = torch.nn.functional.pad(
                s_group_fused,
                (0, qlinear.pad_out),
                "constant",
                0,
            )
            s_channel = torch.nn.functional.pad(s_channel, (0, qlinear.pad_out), "constant", 1e-6)
            size_k, size_n = w_ref.shape

        # --- Marlin weight permutation & packing ---
        weight_perm = get_w4a8_weight_perm(bits, quant_type)
        packed_w = get_w4a8_permute_weights(q_w, size_k, size_n, bits, weight_perm, group_size)
        s_group_packed, s_channel_packed = w4a8_permute_scales(
            s_group_fused, s_channel, size_k, size_n, group_size
        )

        # --- Store results ---
        qlinear.size_k, qlinear.size_n = size_k, size_n
        qlinear.weight = packed_w.to(device)
        qlinear.input_scale = input_scale.to(device)
        qlinear.s_group = s_group_packed.to(device)
        qlinear.s_channel = s_channel_packed.to(device)
        return qlinear

    def _gemm(self, x_i8, s_tok, x_shape):
        """Run the W4A8 GEMM kernel on pre-quantized INT8 input."""
        M = x_i8.shape[0]
        y = quant_linear_cuda.w4a8o16_gemm(
            x_i8,
            self.weight,
            s_tok,
            self.s_channel,
            self.s_group,
            self.workspace,
            M,
            self.size_n,
            self.size_k,
            False,
        )
        if self.pad_out != 0:
            y = y[:, : -self.pad_out]
        return y.reshape(*x_shape[:-1], -1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input -> W4A8 GEMM -> FP16 output.

        Dispatches on dtype: INT8 input goes straight to GEMM (assumed already
        quantized with self.input_scale); FP16 input gets fused FP16→INT8
        quant first.
        """
        # SSDi8 feeds INT8 in (from QRMSNorm/QHadamard with output_scale).
        # Reinterpreting INT8 bytes as FP16 in `_fused_fp16_to_int8` produces
        # garbage — route INT8 through forward_int8 which expects pre-quantized.
        if x.dtype == torch.int8:
            return self.forward_int8(x, self.input_scale)
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]).contiguous()
        x_i8 = _fused_fp16_to_int8(x_2d, self.input_scale)
        return self._gemm(x_i8, self.input_scale, x_shape)

    @torch.no_grad()
    def forward_int8(self, x_i8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Pre-quantized INT8 input -> W4A8 GEMM -> FP16 output.

        Use with fused QHadamard that outputs INT8 + scale directly.
        """
        x_shape = x_i8.shape
        x_2d = x_i8.reshape(-1, x_shape[-1]).contiguous()
        return self._gemm(x_2d, scale, x_shape)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"W4A8B16O16Linear(in_features={self.in_features}, out_features={self.out_features})"
