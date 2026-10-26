"""Zoo INT8 kernels vs the harness across scale regimes (per-tensor policies), code ranges (INT6 codes in INT8
containers) and the two R_V storage bases (docs/narrative_and_todo.md §2c–2d)."""

import pytest
import torch

from linquant.invariances import POLICIES, FakeQuant
from linquant.reference import chunk_core, hadamard_rotation
from tests.test_kernel_simple_gla import check, inputs

B, T, H, K, V = 2, 512, 4, 64, 128


def quantizer(policy, bits):
    qz = FakeQuant("none", bits, ["ALL"], dict(POLICIES[policy]), state_bits=8, round_away=True)
    qz.tie_roles = qz.local_fp16 = qz.kernel_arith = True
    qz.layer = 0
    return qz


def delta_inputs(seed=0):
    q, k, v, g = inputs(True, seed)
    beta = torch.rand(B, T, H, generator=torch.Generator().manual_seed(seed + 7)).cuda().float()
    return q, k, v, g, beta


def diag_gate(seed=0):
    gen = torch.Generator().manual_seed(seed + 3)
    return (torch.nn.functional.logsigmoid(torch.randn(B, T, H, K, generator=gen) * 3) / 4).cuda().float()


@pytest.mark.parametrize("policy,bits", [("kernel_pt", 8), ("kernel_pt", 6), ("kernel_ptk", 8), ("kernel_static", 6)])
@pytest.mark.parametrize("cell", ["simple_gla", "delta", "gla", "kda"])
def test_regime_and_bits(cell, policy, bits):
    qz = quantizer(policy, bits)
    if cell == "simple_gla":
        from linquant.kernels.zoo.simple_gla import chunk_simple_gla_int8

        q, k, v, g = inputs(True)
        run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="scalar", scale=K**-0.5, qz=qz, eta=1.0,  # noqa: E731
                                 read_post=True, C=64)
        kern = lambda: chunk_simple_gla_int8(q, k, v, g, qz.kernel_scales(0, H, K, V), eta=1.0, scale=K**-0.5, bits=bits)[0]  # noqa: E731
    elif cell == "delta":
        from linquant.kernels.zoo.gated_delta_rule import chunk_delta_int8

        q, k, v, g, beta = delta_inputs()
        run = lambda: chunk_core(q, k, v, g, beta, family="delta", decay="scalar", scale=K**-0.5, qz=qz, eta=1.0,  # noqa: E731
                                 read_post=True, C=64)
        kern = lambda: chunk_delta_int8(q, k, v, g, beta, delta_scales(qz), eta=1.0, scale=K**-0.5, bits=bits)[0]  # noqa: E731
    elif cell == "gla":
        from linquant.kernels.zoo.gla import chunk_gla_int8

        q, k, v, _ = inputs(False)
        g = diag_gate()
        run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="diagonal", scale=K**-0.5, qz=qz, eta=0.0,  # noqa: E731
                                 read_post=True, C=64)
        kern = lambda: chunk_gla_int8(q, k, v, g, qz.kernel_scales(0, H, K, V), eta=0.0, scale=K**-0.5, bits=bits)[0]  # noqa: E731
    else:
        from linquant.kernels.zoo.kda import chunk_kda_int8

        q, k, v, _, beta = delta_inputs()
        g = diag_gate()
        run = lambda: chunk_core(q, k, v, g, beta, family="delta", decay="diagonal", scale=K**-0.5, qz=qz, eta=0.0,  # noqa: E731
                                 read_post=True, C=64)
        kern = lambda: chunk_kda_int8(q, k, v, g, beta, delta_scales(qz), eta=0.0, scale=K**-0.5, bits=bits)[0]  # noqa: E731
    qz.calibrating = True
    run()
    qz.finalize()
    check(kern(), run, qz, f"{cell} {policy} bits={bits}")


def delta_scales(qz):
    sc = qz.kernel_scales(0, H, K, V)
    for role, key, shape in (("w_read", "w", (H,)), ("e_local", "el", (H, V)), ("e_write", "ew", (H, V))):
        s = qz.scales.get((0, role))
        sc[key] = torch.ones(shape, device="cuda") if s is None else (s.float().reshape(shape) if s.numel() > 1 else s.float().expand(shape).contiguous())
    return sc


@pytest.mark.parametrize("storage", ["canonical", "rotated"])
@pytest.mark.parametrize("cell", ["simple_gla", "delta", "gla", "kda"])
def test_rv_storage(cell, storage):
    """R_V with the state stored canonically (in-kernel un-rotation) or rotated (fp32 output un-rotated at the exit)."""
    qz = quantizer("kernel_pt", 8)
    R = hadamard_rotation(V, 2, "cuda")
    unrot = R.T if storage == "canonical" else None
    kw = dict(scale=K**-0.5, bits=8, unrot=unrot, out_dtype=None if storage == "canonical" else torch.float32)
    if cell == "simple_gla":
        from linquant.kernels.zoo.simple_gla import chunk_simple_gla_int8

        q, k, v, g = inputs(True)
        vr = (v.double() @ R).half()
        run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="scalar", scale=K**-0.5, qz=qz, eta=1.0, rot_v=R,  # noqa: E731
                                 storage=storage, read_post=True, C=64)
        kern = lambda: chunk_simple_gla_int8(q, k, vr, g, qz.kernel_scales(0, H, K, V), eta=1.0, **kw)[0]  # noqa: E731
    elif cell == "delta":
        from linquant.kernels.zoo.gated_delta_rule import chunk_delta_int8

        q, k, v, g, beta = delta_inputs()
        vr = (v.double() @ R).half()
        run = lambda: chunk_core(q, k, v, g, beta, family="delta", decay="scalar", scale=K**-0.5, qz=qz, eta=1.0, rot_v=R,  # noqa: E731
                                 storage=storage, read_post=True, C=64)
        kern = lambda: chunk_delta_int8(q, k, vr, g, beta, delta_scales(qz), eta=1.0, **kw)[0]  # noqa: E731
    elif cell == "gla":
        from linquant.kernels.zoo.gla import chunk_gla_int8

        q, k, v, _ = inputs(False)
        g = diag_gate()
        vr = (v.double() @ R).half()
        run = lambda: chunk_core(q, k, v, g, None, family="linear", decay="diagonal", scale=K**-0.5, qz=qz, eta=0.0, rot_v=R,  # noqa: E731
                                 storage=storage, read_post=True, C=64)
        kern = lambda: chunk_gla_int8(q, k, vr, g, qz.kernel_scales(0, H, K, V), eta=0.0, **kw)[0]  # noqa: E731
    else:
        from linquant.kernels.zoo.kda import chunk_kda_int8

        q, k, v, _, beta = delta_inputs()
        g = diag_gate()
        vr = (v.double() @ R).half()
        run = lambda: chunk_core(q, k, v, g, beta, family="delta", decay="diagonal", scale=K**-0.5, qz=qz, eta=0.0, rot_v=R,  # noqa: E731
                                 storage=storage, read_post=True, C=64)
        kern = lambda: chunk_kda_int8(q, k, vr, g, beta, delta_scales(qz), eta=0.0, **kw)[0]  # noqa: E731
    qz.calibrating = True
    run()
    qz.finalize()
    out = kern()
    if storage == "rotated":
        out = (out.float() @ R.T.float()).half()
    check(out, run, qz, f"{cell} R_V {storage}")
