"""Fused Hadamard transform + INT8 quantization for the out_proj input (SSDi8's `quant_hadamard` CUDA extension)."""

from __future__ import annotations

import torch
from ssdi8.hadamard_utils import get_qhad_fn


def qhad_transform(w, input_scale):
    """Hadamard-rotate `w` (..., dim) and quantize to INT8 in one kernel launch.

    `input_scale` is the per-tensor scale (amax / 127) as a Python float or a tensor; the float path keeps the
    `had_scale / input_scale` division in 64-bit, matching SSDi8's QHadamard. Returns (w_int8, input_scale).
    """
    dim = w.shape[-1]
    transform_fn, _, had_scale = get_qhad_fn(dim)
    scale_val = input_scale.float().item() if isinstance(input_scale, torch.Tensor) else float(input_scale)
    w_i8 = transform_fn(w.reshape(-1, dim), had_scale / scale_val)
    return w_i8.reshape(w.shape), input_scale
