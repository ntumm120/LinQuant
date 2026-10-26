"""Zoo INT8 GLA kernel (linear family, diagonal per-channel decay) vs the harness at the matching profile."""

import pytest
import torch

from linquant.invariances import POLICIES, FakeQuant
from linquant.reference import chunk_core, hadamard_rotation
from tests.test_kernel_simple_gla import check, cut

B, T, H, K, V = 2, 512, 4, 64, 128


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).cuda().half()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1).cuda().half()
    v = torch.randn(B, T, H, V, generator=g).cuda().half()
    gd = (torch.nn.functional.logsigmoid(torch.randn(B, T, H, K, generator=g) * 3) / 4).cuda().float()
    return q, k, v, gd


def qz_():
    qz = FakeQuant("none", 8, ["ALL"], dict(POLICIES["kernel_static"]), state_bits=8, round_away=True)
    qz.tie_roles = qz.local_fp16 = qz.kernel_arith = True
    qz.layer = 0
    return qz


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_gla_kernel_matches_harness(eta):
    from linquant.kernels.zoo.gla import chunk_gla_int8

    q, k, v, g = inputs()
    qz = qz_()
    run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="diagonal", scale=K**-0.5, qz=qz, eta=eta,  # noqa: E731
                             read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    out, _ = chunk_gla_int8(q, k, v, g, qz.kernel_scales(0, H, K, V), eta=eta, scale=K**-0.5, chunk_size=64)
    check(out, run, qz, f"gla eta={eta}")


@pytest.mark.parametrize("n", [128, 512])
def test_gla_kernel_local_rqk(n):
    """R_QK on a diagonal-decay cell is local only: the anchored operands of the off-diagonal blocks are rotated."""
    from linquant.kernels.zoo.gla import chunk_gla_int8

    q, k, v, g = cut(inputs(), n)
    R = hadamard_rotation(K, 3, q.device)
    qz = qz_()
    run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="diagonal", scale=K**-0.5, qz=qz, eta=0.0,  # noqa: E731
                             rot_qk=R, read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    out, _ = chunk_gla_int8(q, k, v, g, qz.kernel_scales(0, H, K, V), eta=0.0, scale=K**-0.5, chunk_size=64, rot=R)
    check(out, run, qz, f"gla local R_QK T={n}")


@pytest.mark.parametrize("n", [128, 512])
def test_gla_kernel_canonical_storage_rv(n):
    from linquant.kernels.zoo.gla import chunk_gla_int8

    q, k, v, g = cut(inputs(), n)
    R = hadamard_rotation(V, 7, q.device)
    qz = qz_()
    run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="diagonal", scale=K**-0.5, qz=qz, eta=1.0,  # noqa: E731
                             rot_v=R, storage="canonical", read_post=True, C=64)
    qz.calibrating = True
    run()
    qz.finalize()
    vr = (v.double() @ R).half()
    out, _ = chunk_gla_int8(q, k, vr, g, qz.kernel_scales(0, H, K, V), eta=1.0, scale=K**-0.5, chunk_size=64, unrot=R.T)
    check(out, run, qz, f"gla canonical R_V T={n}")
