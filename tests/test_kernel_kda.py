"""Zoo INT8 KDA kernel (delta family, diagonal per-channel decay) vs the harness at the matching profile."""

import pytest
import torch

from linquant.reference import chunk_core, hadamard_rotation
from tests.test_kernel_delta import _kernel_scales
from tests.test_kernel_gla import qz_
from tests.test_kernel_simple_gla import check, cut

B, T, H, K, V = 2, 512, 4, 64, 128


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).cuda().half()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).cuda().half()
    v = torch.randn(B, T, H, V, generator=g).cuda().half()
    beta = torch.rand(B, T, H, generator=g).cuda().float()
    gd = (torch.nn.functional.logsigmoid(torch.randn(B, T, H, K, generator=g) * 3) / 4).cuda().float()
    return q, k, v, gd, beta


@pytest.mark.parametrize("sigma,tau,eta", [(1.0, 1.0, 0.0), (0.0, 1.0, 0.5), (0.5, 0.5, 1.0)])
def test_kda_kernel_matches_harness(sigma, tau, eta):
    from linquant.kernels.zoo.kda import chunk_kda_int8

    q, k, v, g, beta = inputs()
    qz = qz_()
    run = lambda: chunk_core(q, k, v, g, beta, family="delta", decay="diagonal", scale=K**-0.5, qz=qz, eta=eta,  # noqa: E731
                             rho=sigma, tau=tau, read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    out, _ = chunk_kda_int8(q, k, v, g, beta, _kernel_scales(qz, 0), sigma=sigma, tau=tau, eta=eta, scale=K**-0.5,
                            chunk_size=64)
    check(out, run, qz, f"kda sigma={sigma} tau={tau} eta={eta}")


@pytest.mark.parametrize("n", [128, 512])
def test_kda_kernel_local_rqk(n):
    from linquant.kernels.zoo.kda import chunk_kda_int8

    q, k, v, g, beta = cut(inputs(), n)
    R = hadamard_rotation(K, 3, q.device)
    qz = qz_()
    run = lambda: chunk_core(q, k, v, g, beta, family="delta", decay="diagonal", scale=K**-0.5, qz=qz, eta=0.0,  # noqa: E731
                             rot_qk=R, read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    out, _ = chunk_kda_int8(q, k, v, g, beta, _kernel_scales(qz, 0), eta=0.0, scale=K**-0.5, chunk_size=64, rot=R)
    check(out, run, qz, f"kda local R_QK T={n}")


@pytest.mark.parametrize("n", [128, 512])
def test_kda_kernel_canonical_storage_rv(n):
    from linquant.kernels.zoo.kda import chunk_kda_int8

    q, k, v, g, beta = cut(inputs(), n)
    R = hadamard_rotation(V, 7, q.device)
    qz = qz_()
    run = lambda: chunk_core(q, k, v, g, beta, family="delta", decay="diagonal", scale=K**-0.5, qz=qz, eta=1.0,  # noqa: E731
                             rot_v=R, storage="canonical", read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    vr = (v.double() @ R).half()
    out, _ = chunk_kda_int8(q, k, vr, g, beta, _kernel_scales(qz, 0), eta=1.0, scale=K**-0.5, chunk_size=64, unrot=R.T)
    check(out, run, qz, f"kda canonical R_V T={n}")
