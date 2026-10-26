"""Parity + invariance tests for the reference chunk core (run on a GPU node with the pla env)."""

import pytest
import torch

from linquant.harness import CELLS, Arm
from linquant.invariances import FakeQuant
from linquant.reference import chunk_core, hadamard_rotation

B, T, H, K, V = 2, 256, 4, 64, 128
NOQ = FakeQuant("none")


def make_inputs(cell, seed=0):
    fam, dec = CELLS[cell]
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    v = torch.randn(B, T, H, V, generator=g)
    beta = torch.rand(B, T, H, generator=g) if fam == "delta" or cell.endswith("_beta") else None
    gd = None
    if dec == "scalar":
        gd = torch.nn.functional.logsigmoid(torch.randn(B, T, H, generator=g) * 3) / 4
    elif dec == "diagonal":
        gd = torch.nn.functional.logsigmoid(torch.randn(B, T, H, K, generator=g) * 3) / 4
    if fam == "linear" and beta is not None:
        v, beta = v * beta.unsqueeze(-1), None
    return [t.cuda() if t is not None else None for t in (q, k, v, gd, beta)]


def fla_ref(cell, q, k, v, g, beta):
    from fla.ops.delta_rule import chunk_delta_rule
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gla import chunk_gla
    from fla.ops.kda import chunk_kda
    from fla.ops.simple_gla import chunk_simple_gla

    fam, dec = CELLS[cell]
    if fam == "linear" and dec != "diagonal":
        return chunk_simple_gla(q, k, v, g=g)[0]
    if fam == "linear":
        return chunk_gla(q, k, v, g=g)[0]
    if dec == "none":
        return chunk_delta_rule(q, k, v, beta=beta)[0]
    if dec == "scalar":
        return chunk_gated_delta_rule(q, k, v, g=g, beta=beta)[0]
    return chunk_kda(q, k, v, g=g, beta=beta, use_gate_in_kernel=False)[0]


def run(cell, arm, inputs):
    fam, dec = CELLS[cell]
    q, k, v, g, beta = inputs
    return chunk_core(q, k, v, g, beta, family=fam, decay=dec, scale=K**-0.5, qz=arm.quant,
                      rot_qk=arm.rotation(K, 0, q.device) if arm.rot_qk else None,
                      rot_v=arm.rotation(V, 1, q.device) if arm.rot_v else None, eta=arm.eta, rho=arm.rho,
                      read_post=arm.read_post, storage=arm.storage, tau=arm.tau)


@pytest.mark.parametrize("cell", list(CELLS))
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_parity_vs_fla(cell, dtype):
    """fp32 (tight) where FLA supports it; bf16 inputs (kernel-precision tolerance) for every cell."""
    if dtype == "fp32" and CELLS[cell][0] == "delta":
        pytest.skip("FLA delta kernels are bf16-only")
    inp = make_inputs(cell)
    if dtype == "bf16":
        inp = [t.bfloat16() if t is not None and t.dtype == torch.float32 and t.dim() == 4 else t for t in inp]
    ref = fla_ref(cell, *inp).float()
    out = run(cell, Arm(), inp).float()
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    assert err < (2e-3 if dtype == "fp32" else 2.5e-2), f"{cell}/{dtype}: rel max err {err:.2e}"


@pytest.mark.parametrize("cell", list(CELLS))
@pytest.mark.parametrize("kind", ["qk", "v", "eta", "rho"])
def test_invariances(cell, kind):
    fam, dec = CELLS[cell]
    if kind == "rho" and fam != "delta":
        pytest.skip("rho is a delta-family knob")
    if kind == "eta" and dec == "none":
        pytest.skip("no decay to split")
    inp = make_inputs(cell)
    base = run(cell, Arm(), inp)
    arms = {"qk": [Arm(rot_qk=True)], "v": [Arm(rot_v=True)],
            "eta": [Arm(eta=e) for e in (0.25, 0.5, 1.0)], "rho": [Arm(rho=r) for r in (0.0, 0.5)]}[kind]
    for arm in arms:
        out = run(cell, arm, inp)
        err = (out - base).abs().max().item() / base.abs().max().item()
        assert err < 1e-9, f"{cell}/{kind}: {err:.2e}"


@pytest.mark.parametrize("cell", ["mamba2", "gated_deltanet"])
def test_eta_homogeneity_row_dyn(cell):
    """Per-row dynamic scales absorb the per-token write weights: the eta sweep is flat by construction."""
    inp = make_inputs(cell)
    outs = [run(cell, Arm(eta=e, quant=FakeQuant("row_dyn", 8, ["k_write", "e_write", "v_write"])), inp) for e in (0.0, 1.0)]
    err = (outs[0] - outs[1]).abs().max().item() / outs[0].abs().max().item()
    assert err < 1e-6, f"{cell}: {err:.2e}"


def test_hadamard_orthogonal():
    R = hadamard_rotation(64, 3)
    assert torch.allclose(R @ R.T, torch.eye(64, dtype=R.dtype), atol=1e-12)


@pytest.mark.parametrize("cell", list(CELLS))
def test_canonical_storage_exact(cell):
    """R_V with canonical storage (un-rotate each product before combining/storing) is exact without quant."""
    inp = make_inputs(cell)
    base = run(cell, Arm(), inp)
    out = run(cell, Arm(rot_v=True, storage="canonical"), inp)
    err = (out - base).abs().max().item() / base.abs().max().item()
    assert err < 1e-9, f"{cell}: {err:.2e}"


@pytest.mark.parametrize("cell", ["mamba2_beta", "gated_deltanet"])
def test_read_post_exact(cell):
    inp = make_inputs(cell)
    base = run(cell, Arm(), inp)
    out = run(cell, Arm(read_post=True), inp)
    assert (out - base).abs().max().item() / base.abs().max().item() < 1e-9


@pytest.mark.parametrize("cell", ["mamba2_beta", "gated_deltanet", "kda"])
@pytest.mark.parametrize("policy", ["kernel_static", "kernel_pt", "kernel_ptk"])
def test_policies_run_and_state_quant(cell, policy):
    """Realizable policies + INT8 stored state (rotated and canonical) run, calibrate and stay close to fp."""
    from linquant.invariances import POLICIES

    inp = make_inputs(cell)
    base = run(cell, Arm(), inp)
    for storage in ("rotated", "canonical"):
        q = FakeQuant("pt_static", 8, ["ALL"], dict(POLICIES[policy]), state_bits=8)
        arm = Arm(rot_v=True, quant=q, storage=storage)
        q.calibrating = True
        run(cell, arm, inp)
        q.finalize()
        out = run(cell, arm, inp)
        err = (out - base).abs().max().item() / base.abs().max().item()
        assert torch.isfinite(out).all() and err < 0.1, f"{cell}/{policy}/{storage}: {err:.2e}"


@pytest.mark.parametrize("cell", ["deltanet", "gated_deltanet", "kda"])
def test_delta_two_knob_exact(cell):
    """Gram exponent x write placement (3x3) is exact on outputs and on the outgoing state."""
    fam, dec = CELLS[cell]
    q, k, v, g, beta = inp = make_inputs(cell)
    base = run(cell, Arm(), inp)
    for sig in (0.0, 0.5, 1.0):
        for tau in (0.0, 0.5, 1.0):
            out = run(cell, Arm(rho=sig, tau=tau), inp)
            err = (out - base).abs().max().item() / base.abs().max().item()
            assert err < 1e-9, f"{cell} sigma={sig} tau={tau}: {err:.2e}"


@pytest.mark.parametrize("cell", ["linear_none_beta", "mamba2_beta", "gla_beta"])
def test_linear_beta_placement_exact(cell):
    """Moving beta from v onto k in the linear cells is exact."""
    fam, dec = CELLS[cell]
    q, k, v, g, _ = make_inputs(cell)  # v already carries beta (write gate)
    gen = torch.Generator().manual_seed(1)
    beta = torch.rand(B, T, H, generator=gen).cuda().double().unsqueeze(-1)
    vg = v.double() * beta
    base = chunk_core(q, k, vg, g, None, family=fam, decay=dec, scale=K**-0.5, qz=NOQ)
    for rho in (0.0, 0.5):
        out = chunk_core(q, k.double() * beta ** (1 - rho), v.double() * beta**rho, g, None, family=fam, decay=dec,
                         scale=K**-0.5, qz=NOQ)
        assert (out - base).abs().max().item() / base.abs().max().item() < 1e-9


def test_diagonal_scaling_exact():
    """Q D, K D^-1 (and V Dv with output / Dv) leaves the recurrence output unchanged."""
    for cell in ("mamba2_beta", "gla"):
        fam, dec = CELLS[cell]
        q, k, v, g, beta = make_inputs(cell)
        base = chunk_core(q.double(), k.double(), v.double(), g, beta, family=fam, decay=dec, scale=K**-0.5, qz=NOQ)
        gen = torch.Generator().manual_seed(2)
        D = torch.rand(H, K, generator=gen).cuda().double() + 0.5
        Dv = torch.rand(H, V, generator=gen).cuda().double() + 0.5
        out = chunk_core(q.double() * D, k.double() / D, v.double() * Dv, g, beta, family=fam, decay=dec,
                         scale=K**-0.5, qz=NOQ) / Dv
        assert (out - base).abs().max().item() / base.abs().max().item() < 1e-9, cell


def test_calibrate_stats_pass_with_native_policy():
    """AUD-001: the diagonal-scaling pre-pass must run with every quantization route disabled."""
    from linquant.harness import calibrate
    from linquant.run import parse_arm

    arm = parse_arm("policy=kernel_static,sqk=0.5,state=8")

    class Dummy(torch.nn.Module):
        def forward(self, w):
            x = torch.ones(1, 1, 2, 2, dtype=torch.float64, device=w.device)
            for role in ("q", "k", "v"):
                arm.collect(0, role, x)
            arm.quant("q_local", x)
            return arm.quant("state", x)

    calibrate(Dummy(), arm, [torch.ones(1)], device="cpu")
    assert arm.smooth_D and not arm.quant.disabled and not arm.stats_pass


def test_fp32carry_state_pass_no_error_feedback():
    """Zoo-lineage pass: the carry is never quantized; only the read copy is."""
    from linquant.reference import _state_update

    qz = FakeQuant("none", 8, ["ALL"], state_bits=8)
    qz.scales[(0, "state")] = torch.full((1, 1), 0.5, dtype=torch.float64)
    qz.layer = 0
    S, carry = torch.zeros(1, 1, 1, 1, dtype=torch.float64), None
    ref = 0.0
    for lc in (0.3, 0.3, 0.3, 0.3):
        S, carry = _state_update(S, carry, torch.full((1, 1, 1, 1), 0.9, dtype=torch.float64),
                                 torch.full((1, 1, 1, 1), lc, dtype=torch.float64), qz)
        ref = 0.9 * ref + lc
        assert abs(carry.item() - ref) < 1e-12 and abs(S.item() - round(ref / 0.5) * 0.5) < 1e-12


def test_value_axis_scaling_exact_on_delta_cells():
    for cell in ("deltanet", "gated_deltanet", "kda"):
        fam, dec = CELLS[cell]
        q, k, v, g, beta = make_inputs(cell)
        base = chunk_core(q.double(), k.double(), v.double(), g, beta, family=fam, decay=dec, scale=K**-0.5, qz=NOQ)
        gen = torch.Generator().manual_seed(3)
        Dv = torch.rand(H, V, generator=gen).cuda().double() + 0.5
        out = chunk_core(q.double(), k.double(), v.double() * Dv, g, beta, family=fam, decay=dec, scale=K**-0.5, qz=NOQ) / Dv
        assert (out - base).abs().max().item() / base.abs().max().item() < 1e-9, cell


def test_kernel_scales_use_requested_layer():
    """AUD-005: kernel_scales(layer) must not read the mutable current layer for the operand scales."""
    qz = FakeQuant("none", 8, ["ALL"], state_bits=8)
    qz.layer = 1
    for layer, val in ((0, 2.0), (1, 9.0)):
        for role in ("q_read", "k_local", "k_write", "v_write", "v_local", "state"):
            qz.scales[(layer, role)] = torch.tensor(val)
    out = qz.kernel_scales(0, 1, 1, 1)
    assert all(float(v.flatten()[0]) == 2.0 for v in out.values()), {k: v.flatten()[0].item() for k, v in out.items()}
