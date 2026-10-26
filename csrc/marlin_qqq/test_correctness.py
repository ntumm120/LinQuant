"""Correctness test for standalone Marlin QQQ W4A8 GEMM kernel.

Implements the exact vLLM packing pipeline for Marlin QQQ:
  qqq_quantize_weights -> marlin_permute_weights -> marlin_qqq_weights -> marlin_qqq_permute_scales
Then runs the kernel and compares to FP16 reference.
"""

import os
import sys

import numpy as np
import torch

torch._inductor.config.compile_threads = 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the built marlin_qqq extension lives next to this file
import marlin_qqq

# ---------- Ported from vLLM quant_utils.py ----------

TILE = 16


def qqq_quantize_weights(w, num_bits, group_size):
    """Port of vLLM's qqq_quantize_weights for per-channel mode (group_size == size_k)."""
    size_k, size_n = w.shape

    if group_size == -1:
        group_size = size_k

    if group_size == size_k:
        # Per-channel quantization
        max_q_val = 2 ** (num_bits - 1) - 1  # 7 for 4-bit

        s_channel = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s_channel /= max_q_val

        q_w = torch.round(w / s_channel).int()
        q_w = torch.clamp(q_w, -max_q_val, max_q_val)

        w_ref = q_w.half() * s_channel

        s_group = torch.tensor([], dtype=torch.half)
        # Compensate for the nibble-to-byte shift in the kernel's dequant
        s_channel /= 2 ** (8 - num_bits)
        s_channel = s_channel.reshape(-1, size_n).contiguous().to(torch.float)
    else:
        # Per-group quantization
        w = w.reshape((-1, group_size, size_n)).permute(1, 0, 2).reshape((group_size, -1))

        max_q_val = 2**num_bits - 1  # 15 for 4-bit
        half_q_val = (max_q_val + 1) // 2  # 8

        s_group = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s_group *= 2 / max_q_val

        q_w = torch.round(w / s_group).int()
        q_w += half_q_val
        q_w = torch.clamp(q_w, 0, max_q_val)

        w_ref = (q_w - half_q_val).half() * s_group

        def reshape_w(w_in):
            return (
                w_in.reshape((group_size, -1, size_n))
                .permute(1, 0, 2)
                .reshape((size_k, size_n))
                .contiguous()
            )

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)

        s_channel = torch.max(torch.abs(w_ref), 0, keepdim=True)[0]
        s_channel /= 127.0
        t_int8 = (w_ref / s_channel).round().clamp(-128, 127).to(torch.int8)
        w_ref = t_int8.half() * s_channel
        s_channel = s_channel.reshape(1, -1).to(dtype=torch.float)

        s_group = (s_group.reshape(-1, size_n).contiguous() / s_channel).to(dtype=torch.half)

    return w_ref, q_w, s_group, s_channel


# ---------- Ported from vLLM marlin_utils_test.py ----------


def marlin_permute_weights(q_w, size_k, size_n, perm):
    assert q_w.shape == (size_k, size_n)
    q_w = q_w.reshape((size_k // TILE, TILE, size_n // TILE, TILE))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // TILE, size_n * TILE))
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    return q_w


# ---------- Ported from vLLM marlin_utils_test_qqq.py ----------


def get_qqq_weight_perm(num_bits, quant_type):
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

    if num_bits == 4:
        if quant_type == "per-channel":
            interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        else:
            interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    else:
        raise ValueError(f"num_bits must be 4, got {num_bits}")

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    return torch.from_numpy(perm)


def marlin_qqq_weights(q_w, size_k, size_n, num_bits, perm, group_size):
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    pack_factor = 32 // num_bits  # 8 for 4-bit
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    if group_size == size_k:
        for i in range(pack_factor):
            q_packed |= (q_w[:, i::pack_factor] & 0xF) << (num_bits * i)
    else:
        for i in range(pack_factor):
            q_packed |= q_w[:, i::pack_factor] << (num_bits * i)

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)
    return q_packed


def get_qqq_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_qqq_permute_scales(s_group, s_channel, size_k, size_n, group_size):
    scale_perm, scale_perm_single = get_qqq_scale_perms()
    if group_size < size_k and group_size != -1:
        s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
        s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
        s_group = s_group.reshape((-1, size_n)).contiguous()
    else:
        s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s_channel = s_channel.reshape((-1, size_n)).contiguous()
    return s_group, s_channel


def marlin_qqq_quantize(w, num_bits, group_size):
    """Full Marlin QQQ quantization pipeline (port of vLLM's marlin_qqq_quantize)."""
    size_k, size_n = w.shape
    if group_size == -1:
        effective_group_size = size_k
    else:
        effective_group_size = group_size

    quant_type = "per-channel" if effective_group_size == size_k else "per-group"

    w_ref, q_w, s_group, s_channel = qqq_quantize_weights(w, num_bits, group_size)

    weight_perm = get_qqq_weight_perm(num_bits, quant_type)
    packed_q_w = marlin_qqq_weights(q_w, size_k, size_n, num_bits, weight_perm, effective_group_size)
    perm_s_group, perm_s_channel = marlin_qqq_permute_scales(
        s_group, s_channel, size_k, size_n, effective_group_size
    )

    results = [w_ref, packed_q_w, perm_s_group, perm_s_channel]
    for i in range(len(results)):
        results[i] = results[i].to(w.device)
    return results


# ---------- Test ----------


def test_marlin_qqq(M, K, N, group_size=-1):
    """Test Marlin QQQ kernel correctness at given dimensions."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")

    print(f"\n=== Marlin QQQ Test: M={M}, K={K}, N={N}, group_size={group_size} ===")

    # Random FP16 weight and activation
    w = torch.randn(K, N, dtype=torch.float16, device=device)
    a_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)

    # Quantize weights using the full Marlin QQQ pipeline
    w_ref, b_q, s_group, s_ch = marlin_qqq_quantize(w, num_bits=4, group_size=group_size)

    # Quantize activations to INT8 (per-token)
    a_abs_max = a_fp16.abs().amax(dim=1, keepdim=True)  # [M, 1]
    s_tok = (a_abs_max / 127.0).clamp(min=1e-10).to(torch.float32).squeeze(1)  # [M]
    a_int8 = torch.clamp(torch.round(a_fp16.float() / s_tok.unsqueeze(1)), -128, 127).to(
        torch.int8
    )

    # Dequantized reference: use the dequantized weight w_ref
    # The "ideal" quantized result
    a_dequant = a_int8.half() * s_tok.unsqueeze(1).half()
    quant_ref = a_dequant @ w_ref  # [M, N]

    # FP16 reference (before any quantization)
    fp16_ref = a_fp16 @ w  # [M, N]

    # Prepare kernel inputs
    workspace = torch.zeros(N // 64 * 16, dtype=torch.int32, device=device)

    # s_group: empty for per-channel, FP16 tensor for per-group
    if s_group.numel() == 0:
        s_group_kernel = torch.empty(0, dtype=torch.float16, device=device)
    else:
        s_group_kernel = s_group.to(device)

    print(f"  b_q shape: {b_q.shape}, dtype: {b_q.dtype}")
    print(f"  s_ch shape: {s_ch.shape}, s_tok shape: {s_tok.shape}")
    print(f"  s_group shape: {s_group_kernel.shape}")

    try:
        result = marlin_qqq.marlin_qqq_gemm(
            a_int8, b_q, s_tok, s_ch, s_group_kernel, workspace, M, N, K
        )
    except Exception as e:
        print(f"  KERNEL ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False

    print(f"  result shape: {result.shape}, dtype: {result.dtype}")

    if not result.isfinite().all():
        print("  FAIL: output contains non-finite values")
        return False

    # Compare to quantized reference
    abs_diff = (result.float() - quant_ref.float()).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # Compare to FP16 reference
    fp16_diff = (result.float() - fp16_ref.float()).abs()
    fp16_max = fp16_diff.max().item()
    fp16_mean = fp16_diff.mean().item()

    ref_scale = quant_ref.float().abs().mean().item() + 1e-6
    rel_err = mean_abs / ref_scale

    print(f"  vs quant_ref: max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f}, rel_err={rel_err:.4f}")
    print(f"  vs fp16_ref:  max_abs={fp16_max:.4f}, mean_abs={fp16_mean:.4f}")

    # Tolerance: the kernel should match the quantized reference closely.
    # Small differences come from FP16 accumulation order differences.
    ok = rel_err < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    all_ok = True

    # Per-channel (group_size = -1)
    all_ok &= test_marlin_qqq(16, 128, 128, group_size=-1)
    all_ok &= test_marlin_qqq(16, 256, 256, group_size=-1)
    all_ok &= test_marlin_qqq(32, 256, 256, group_size=-1)
    all_ok &= test_marlin_qqq(48, 256, 256, group_size=-1)
    all_ok &= test_marlin_qqq(64, 256, 256, group_size=-1)

    # Per-group (group_size = 128)
    all_ok &= test_marlin_qqq(16, 256, 256, group_size=128)
    all_ok &= test_marlin_qqq(32, 256, 256, group_size=128)

    # Larger
    all_ok &= test_marlin_qqq(16, 2048, 2048, group_size=-1)
    all_ok &= test_marlin_qqq(1, 2048, 2048, group_size=-1)

    print(f"\n{'=' * 40}")
    print(f"Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
