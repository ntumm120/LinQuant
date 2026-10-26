"""Zoo INT8 delta-family kernel (gated / plain delta rule) vs the harness at the matching profile."""

import pytest
import torch

from linquant.invariances import POLICIES, FakeQuant
from linquant.reference import chunk_core
from tests.test_kernel_simple_gla import check, cut

B, T, H, K, V = 2, 512, 4, 64, 128


def inputs(decay, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).cuda().half()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).cuda().half()
    v = torch.randn(B, T, H, V, generator=g).cuda().half()
    beta = torch.rand(B, T, H, generator=g).cuda().float()
    gd = (torch.nn.functional.logsigmoid(torch.randn(B, T, H, generator=g) * 3) / 4).cuda().float() if decay else None
    return q, k, v, gd, beta


def _kernel_scales(qz, layer):
    base = qz.kernel_scales(layer, H, K, V)

    def one(role, shape):
        sc = qz.scales.get((layer, role))
        if sc is None:
            return torch.ones(shape, dtype=torch.float32, device="cuda")
        return sc.float().reshape(shape) if sc.numel() > 1 else sc.float().expand(shape).contiguous()

    return {"q": base["q"], "k": base["k"], "kw": base["kw"], "state": base["state"],
            "w": one("w_read", (H,)), "el": one("e_local", (H, V)), "ew": one("e_write", (H, V))}


@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("sigma,tau,eta", [(1.0, 1.0, 1.0), (0.0, 1.0, 1.0), (0.5, 0.5, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.5)])
def test_delta_kernel_matches_harness(decay, sigma, tau, eta):
    from linquant.kernels.zoo.gated_delta_rule import chunk_delta_int8

    if not decay and eta != 1.0:
        pytest.skip("eta needs a decay")
    q, k, v, g, beta = inputs(decay)
    qz = FakeQuant("none", 8, ["ALL"], dict(POLICIES["kernel_static"]), state_bits=8, round_away=True)
    qz.tie_roles = qz.local_fp16 = qz.kernel_arith = True
    qz.layer = 0
    run = lambda: chunk_core(q, k, v, g, beta, family="delta", decay="scalar" if decay else "none", scale=K**-0.5,  # noqa: E731
                             qz=qz, eta=eta, rho=sigma, tau=tau, read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    out, _ = chunk_delta_int8(q, k, v, g, beta, _kernel_scales(qz, 0), sigma=sigma, tau=tau, eta=eta, scale=K**-0.5,
                              chunk_size=64)
    check(out, run, qz, f"decay={decay} sigma={sigma} tau={tau} eta={eta}")


@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("sigma,tau", [(1.0, 1.0), (0.0, 1.0)])
@pytest.mark.parametrize("n", [128, 512])
def test_delta_kernel_canonical_storage_rv(decay, sigma, tau, n):
    """R_V with canonical storage ("align"): v rotated in torch, the kernel rotates the state read into the value
    basis, keeps E rotated, and un-rotates the local and write products."""
    from linquant.kernels.zoo.gated_delta_rule import chunk_delta_int8
    from linquant.reference import hadamard_rotation

    q, k, v, g, beta = cut(inputs(decay), n)
    R = hadamard_rotation(V, 7, q.device)
    qz = FakeQuant("none", 8, ["ALL"], dict(POLICIES["kernel_static"]), state_bits=8, round_away=True)
    qz.tie_roles = qz.local_fp16 = qz.kernel_arith = True
    qz.layer = 0
    run = lambda: chunk_core(q, k, v, g, beta, family="delta", decay="scalar" if decay else "none", scale=K**-0.5,  # noqa: E731
                             qz=qz, rot_v=R, storage="canonical", eta=1.0, rho=sigma, tau=tau,
                             read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    vr = (v.double() @ R).half()
    out, _ = chunk_delta_int8(q, k, vr, g, beta, _kernel_scales(qz, 0), sigma=sigma, tau=tau, eta=1.0, scale=K**-0.5,
                              chunk_size=64, unrot=R.T)
    check(out, run, qz, f"canonical R_V decay={decay} sigma={sigma} tau={tau} T={n}")
