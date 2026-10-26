"""Zoo INT8 kernel (chunk_simple_gla) vs the harness at the matching profile."""

import pytest
import torch

from linquant.invariances import POLICIES, FakeQuant
from linquant.reference import chunk_core

B, T, H, K, V = 2, 512, 4, 64, 128


def check(out, run, qz, label):
    """The kernel must agree with its fake-quant emulation at least as well as that emulation agrees with exact
    float64 arithmetic (no-decay cells never forget a code flip, so rounding order alone drifts chunk by chunk)."""
    ref = run().float()
    qz.kernel_arith = False
    exact = run().float()
    qz.kernel_arith = True
    d, n = (out.float() - ref).abs(), (ref - exact).abs()
    err_max, err_mean = d.max().item() / ref.abs().max().item(), d.mean().item() / ref.abs().mean().item()
    noise_max, noise_mean = n.max().item() / ref.abs().max().item(), n.mean().item() / ref.abs().mean().item()
    # factor 2: kernel drift and harness noise are two same-magnitude draws (a dataflow bug shows as >= 100x)
    assert torch.isfinite(out).all() and err_max <= max(5e-3, 2 * noise_max) and err_mean <= max(2e-4, 2 * noise_mean), \
        f"{label}: rel max {err_max:.2e} rel mean {err_mean:.2e} (harness arithmetic noise floor {noise_max:.2e} / {noise_mean:.2e})"


def cut(tensors, n):
    """Short horizons (n tokens) isolate the dataflow from the recurrence's accumulated drift."""
    return [t[:, :n] if t is not None else None for t in tensors]


def inputs(decay, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).cuda().half()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).cuda().half()
    v = torch.randn(B, T, H, V, generator=g).cuda().half()
    gd = (torch.nn.functional.logsigmoid(torch.randn(B, T, H, generator=g) * 3) / 4).cuda().float() if decay else None
    return q, k, v, gd


@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("eta", [1.0, 0.0, 0.5])
def test_kernel_matches_harness(decay, eta):
    from linquant.kernels.zoo.simple_gla import chunk_simple_gla_int8

    if not decay and eta != 1.0:
        pytest.skip("eta needs a decay")
    q, k, v, g = inputs(decay)
    qz = FakeQuant("none", 8, ["ALL"], dict(POLICIES["kernel_static"]), state_bits=8, round_away=True)
    qz.tie_roles = True
    qz.local_fp16 = True
    qz.kernel_arith = True
    qz.layer = 0
    run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="scalar" if decay else "none", scale=K**-0.5,  # noqa: E731
                             qz=qz, eta=eta, read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    out, _ = chunk_simple_gla_int8(q, k, v, g, qz.kernel_scales(0, H, K, V), eta=eta, scale=K**-0.5, chunk_size=64)
    check(out, run, qz, f"decay={decay} eta={eta}")


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_kernel_strong_decay_no_nan(eta):
    """AUD-006: strongly negative cumulative decay must not produce NaN codes at any eta."""
    from linquant.kernels.zoo.simple_gla import chunk_simple_gla_int8

    q, k, v, g = inputs(True)
    g = g * 40.0  # push the chunk-local cumulative decay far below float32's exp range
    qz = FakeQuant("none", 8, ["ALL"], dict(POLICIES["kernel_static"]), state_bits=8, round_away=True)
    qz.tie_roles = True
    qz.local_fp16 = True
    qz.kernel_arith = True
    qz.layer = 0
    run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="scalar", scale=K**-0.5, qz=qz, eta=eta,  # noqa: E731
                             read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    out, _ = chunk_simple_gla_int8(q, k, v, g, qz.kernel_scales(0, H, K, V), eta=eta, scale=K**-0.5, chunk_size=64)
    check(out, run, qz, f"strong decay eta={eta}")


@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("n", [128, 512])
def test_kernel_canonical_storage_rv(decay, n):
    """R_V with canonical storage: v rotated in torch, kernel un-rotates each product before the fp32 carry / output."""
    from linquant.kernels.zoo.simple_gla import chunk_simple_gla_int8
    from linquant.reference import hadamard_rotation

    q, k, v, g = cut(inputs(decay), n)
    R = hadamard_rotation(V, 7, q.device)
    qz = FakeQuant("none", 8, ["ALL"], dict(POLICIES["kernel_static"]), state_bits=8, round_away=True)
    qz.tie_roles = qz.local_fp16 = qz.kernel_arith = True
    qz.layer = 0
    run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="scalar" if decay else "none", scale=K**-0.5,  # noqa: E731
                             qz=qz, rot_v=R, storage="canonical", eta=1.0, read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    vr = (v.double() @ R).half()  # the kernel receives the rotated value operand (materialized in the activation dtype)
    out, _ = chunk_simple_gla_int8(q, k, vr, g, qz.kernel_scales(0, H, K, V), eta=1.0, scale=K**-0.5, chunk_size=64,
                                   unrot=R.T)
    check(out, run, qz, f"canonical R_V decay={decay} T={n}")
