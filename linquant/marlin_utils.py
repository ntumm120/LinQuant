"""Vendored Marlin weight packing utilities (from Quamba's marlin_utils.py).

Only the W4A8 packing functions needed by our pipeline.
Removes the dependency on the quamba package.
"""

from __future__ import annotations

import numpy as np
import torch

MARLIN_QQQ_MIN_THREAD_N = 64
MARLIN_QQQ_MAX_PARALLEL = 16
GPTQ_MARLIN_TILE = 16


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


@torch.no_grad()
def permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    return q_w


@torch.no_grad()
def get_w4a8_permute_weights(q_w, size_k, size_n, num_bits, perm, group_size):
    q_w = permute_weights(q_w, size_k, size_n, perm)
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device
    q_w = q_w.cpu().numpy().astype(np.uint32)
    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    if group_size == size_k:
        for i in range(pack_factor):
            q_packed |= (q_w[:, i::pack_factor] & 0xF) << num_bits * i
    else:
        for i in range(pack_factor):
            q_packed |= q_w[:, i::pack_factor] << num_bits * i
    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)
    return q_packed


@torch.no_grad()
def get_w4a8_weight_perm(num_bits: int, quant_type: str):
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [4 * (i % 4), 4 * (i % 4) + 1, 4 * (i % 4) + 2, 4 * (i % 4) + 3]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])
    perm = np.array(perm_list)
    assert quant_type in ["per-channel", "per-group"]
    if num_bits == 4:
        interleave = (
            np.array([4, 0, 5, 1, 6, 2, 7, 3])
            if quant_type == "per-channel"
            else np.array([0, 2, 4, 6, 1, 3, 5, 7])
        )
    else:
        raise ValueError(f"num_bits must be 4, got {num_bits}")
    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    return torch.from_numpy(perm)


@torch.no_grad()
def get_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


@torch.no_grad()
def w4a8_permute_scales(s_group, s_channel, size_k, size_n, group_size):
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
        s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
        s_group = s_group.reshape((-1, size_n)).contiguous()
    else:
        s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s_channel = s_channel.reshape((-1, size_n)).contiguous()
    return s_group, s_channel


@torch.no_grad()
def get_w4a8_quantize_weights(w: torch.Tensor, num_bits: int, group_size: int = -1):
    orig_device = w.device
    size_k, size_n = w.shape
    assert w.is_floating_point()
    if group_size == -1:
        group_size = size_k

    if group_size < size_k:
        w = w.reshape((-1, group_size, size_n)).permute(1, 0, 2).reshape((group_size, -1))
        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2
        s_group = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s_group *= 2 / max_q_val
        q_w = torch.round(w / s_group).int() + half_q_val
        q_w = torch.clamp(q_w, 0, max_q_val)
        w_ref = (q_w - half_q_val).half() * s_group

        def reshape_w(w):
            return w.reshape((group_size, -1, size_n)).permute(1, 0, 2).reshape((size_k, size_n)).contiguous()

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)
        s_channel = torch.max(torch.abs(w_ref), 0, keepdim=True)[0] / 127.0
        t_int8 = (w_ref / s_channel).round().clamp(-128, 127).to(torch.int8)
        w_ref = t_int8.half() * s_channel
        s_channel = s_channel.reshape(1, -1).to(dtype=torch.float)
        s_group = (s_group.reshape(-1, size_n).contiguous() / s_channel).to(dtype=torch.half)
    else:
        max_q_val = 2 ** (num_bits - 1) - 1
        s_channel = torch.max(torch.abs(w), 0, keepdim=True)[0] / max_q_val
        q_w = torch.clamp(torch.round(w / s_channel).int(), -max_q_val, max_q_val)
        w_ref = q_w.half() * s_channel
        s_group = torch.tensor([], dtype=torch.half)
        s_channel = (s_channel / (2 ** (8 - num_bits))).reshape(-1, size_n).contiguous().to(torch.float)

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        s_group.to(device=orig_device),
        s_channel.to(device=orig_device),
    )
