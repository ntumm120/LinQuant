"""Calibration observers for the Nemotron-H W4A8 pipeline (`scripts/nemotron_h_w4a8.py`).

Pure observation passes over the mamba mixers (`block_type == "mamba"`): each one patches
`mamba_ssm.ops.triton.ssd_combined._mamba_chunk_scan_combined_fwd` (or `_chunk_state_fwd`),
runs the calibration windows, and records amax statistics. Mathematically equivalent to
SSDi8's `ChunkCollector` observers. The SSD runs end-to-end with the diagonal key scaling
D_qk and the R rotation applied to B/C, so every statistic is observed in the regime the
W4A8 kernel sees at runtime.
"""

import os

os.environ.setdefault("SOFT_FILELOCK", "1")

import filelock  # noqa: E402

filelock.FileLock = filelock.SoftFileLock

import torch  # noqa: E402

torch._inductor.config.compile_threads = 1
torch._dynamo.config.suppress_errors = True

import datasets  # noqa: E402

datasets.builder.FileLock = filelock.SoftFileLock

import torch.nn as nn  # noqa: E402
from datasets import load_dataset  # noqa: E402
from mamba_ssm.ops.triton.ssd_combined import _chunk_cumsum_fwd  # noqa: E402
from ssdi8.hadamard_utils import had_transform, is_pow2  # noqa: E402  (libs/SSDi8 on sys.path)


# ---------------------------------------------------------------------------
# Global Hadamard rotation of the residual stream (weights only, done once)
# ---------------------------------------------------------------------------
def had_input(W):
    """Apply H to input dimension (columns): W_new = W @ H."""
    was_cpu = not W.is_cuda
    if was_cpu:
        W = W.cuda()
    W = had_transform(W.float()).half()
    return W.cpu() if was_cpu else W


def had_output(W):
    """Apply H to output dimension (rows): W_new = H @ W."""
    was_cpu = not W.is_cuda
    if was_cpu:
        W = W.cuda()
    W = had_transform(W.T.contiguous().float()).T.contiguous().half()
    return W.cpu() if was_cpu else W


def fuse_norm_into_linear(linear, norm_weight):
    """Fuse RMSNorm weight into a linear layer: W_new = W @ diag(g)."""
    linear.weight.data = linear.weight.data * norm_weight[None, :]


def apply_global_had_nemotron_h(model):
    """Apply global Hadamard rotation to Nemotron-H's residual stream.

    Nemotron-H has 3 block types (mamba, attention, mlp), each with its own norm.
    All blocks consume and produce H-rotated residuals.
    """
    layers = model.backbone.layers

    # 1. Rotate embedding
    model.backbone.embeddings.weight.data = had_input(model.backbone.embeddings.weight.data)
    print("  Rotated embedding")

    n_mamba, n_attn, n_mlp = 0, 0, 0
    for i, block in enumerate(layers):
        norm_w = block.norm.weight.data.clone()
        bt = block.block_type

        if bt == "mamba":
            mixer = block.mixer
            # Fuse norm into in_proj + apply H
            fuse_norm_into_linear(mixer.in_proj, norm_w)
            mixer.in_proj.weight.data = had_input(mixer.in_proj.weight.data)
            # out_proj: output H (H-space for residual)
            d_inner = mixer.out_proj.weight.shape[1]
            can_had_inner = (
                (d_inner % 12 == 0 and is_pow2(d_inner // 12))
                or (d_inner % 20 == 0 and is_pow2(d_inner // 20))
                or (d_inner % 28 == 0 and is_pow2(d_inner // 28))
                or (d_inner % 40 == 0 and is_pow2(d_inner // 40))
                or is_pow2(d_inner)
            )
            if can_had_inner:
                # H·H^T=I split: bake one H into weight, apply other at runtime
                mixer.out_proj.weight.data = had_input(mixer.out_proj.weight.data)
                mixer.out_proj._needs_runtime_had = True
            else:
                mixer.out_proj._needs_runtime_had = False
            mixer.out_proj.weight.data = had_output(mixer.out_proj.weight.data)
            n_mamba += 1

        elif bt == "attention":
            attn = block.mixer
            # Fuse norm into q/k/v_proj + apply H
            for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
                fuse_norm_into_linear(proj, norm_w)
                proj.weight.data = had_input(proj.weight.data)
            # o_proj: output H (H-space for residual)
            attn.o_proj.weight.data = had_output(attn.o_proj.weight.data)
            n_attn += 1

        elif bt == "mlp":
            mlp = block.mixer
            # Fuse norm into up_proj + apply H
            fuse_norm_into_linear(mlp.up_proj, norm_w)
            mlp.up_proj.weight.data = had_input(mlp.up_proj.weight.data)
            # down_proj: output H (H-space for residual)
            mlp.down_proj.weight.data = had_output(mlp.down_proj.weight.data)
            n_mlp += 1

        block.norm.weight.data.fill_(1.0)

    # Final norm + lm_head
    tied = model.backbone.embeddings.weight.data_ptr() == model.lm_head.weight.data_ptr()
    if tied:
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())
        model.lm_head.weight.data = had_input(model.lm_head.weight.data)
        final_norm_w = model.backbone.norm_f.weight.data.clone()
        fuse_norm_into_linear(model.lm_head, final_norm_w)
        model.lm_head.weight.data = had_input(model.lm_head.weight.data)
        print("  Untied embedding/lm_head for separate rotation")
    else:
        final_norm_w = model.backbone.norm_f.weight.data.clone()
        fuse_norm_into_linear(model.lm_head, final_norm_w)
        model.lm_head.weight.data = had_input(model.lm_head.weight.data)
    model.backbone.norm_f.weight.data.fill_(1.0)

    print(f"  Global Hadamard: {n_mamba} mamba + {n_attn} attn + {n_mlp} mlp + embedding + lm_head")


# ---------------------------------------------------------------------------
# SSD-path calibration observers
# ---------------------------------------------------------------------------
# Saved reference to the original (unpatched) SSD forward; set by the caller before calibrating.
_ORIGINAL_FWD = None

DQK_MATRIX = None  # per-mamba-layer (D⁻¹ for B, D for C) diagonal key scaling, applied before the rotation

# Mamba-only index of the mixer currently running (kept current by `install_layer_hooks`);
# selects the per-layer rotation and D_qk inside the patched SSD forward.
FQCFG = {"layer_idx": 0}


def _make_rotated_calib_fwd(base_fwd, R_matrix=None):
    """Wrap an SSD fwd so B/C enter D_qk-scaled and R-rotated, as at runtime.

    `R_matrix` may be a single tensor (shared across layers) or a list/tuple with one
    tensor per mamba layer, looked up via `FQCFG["layer_idx"]`.

    Mirrors SSDi8's calibration injection: the *full* SSD runs on the rotated
    activations, so chunk_state, chunk_scan, and downstream observers all see the
    regime the W4A8 runtime kernel sees. Without it, helpers would rotate locally
    only for the amax observation while downstream layers are calibrated against a
    different distribution than runtime.
    """
    if R_matrix is None and DQK_MATRIX is None:
        return base_fwd

    def _resolve(mat):
        if isinstance(mat, (list, tuple)):
            return mat[FQCFG["layer_idx"]]
        return mat

    def rotated_fwd(x, dt, A, B, C, chunk_size, **kwargs):
        R_l = _resolve(R_matrix)
        if DQK_MATRIX is not None:  # C·D, B·D⁻¹ (exact) before the rotation, like the runtime kernel
            db, dc = DQK_MATRIX[FQCFG["layer_idx"]]
            B = B * db.to(B.device, B.dtype)
            C = C * dc.to(C.device, C.dtype)
        if R_l is not None:
            R = R_l.to(B.device, B.dtype)
            if R.dim() == 3:
                # Per-group Rᵀ stored; forward rotation per group.
                B = torch.einsum("btgn,gmn->btgm", B, R).contiguous()
                C = torch.einsum("btgn,gmn->btgm", C, R).contiguous()
            else:
                R_T = R.T
                B = B @ R_T
                C = C @ R_T
        return base_fwd(x, dt, A, B, C, chunk_size, **kwargs)

    return rotated_fwd


def _compute_x_scaled(x, dt, dA_cumsum, chunk_size):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, _ = dt.shape
    S_total = nchunks * chunk_size
    if seqlen != S_total:
        x_pad = x.new_zeros((batch, S_total, nheads, headdim))
        x_pad[:, :seqlen] = x
        x = x_pad
    dA_last = dA_cumsum[..., -1:]
    scale_lut = torch.exp(dA_last - dA_cumsum) * dt
    X_bcKhd = x.view(batch, nchunks, chunk_size, nheads, headdim)
    scale_bckh1 = scale_lut.permute(0, 2, 3, 1).unsqueeze(-1).to(X_bcKhd.dtype)
    X_scaled = X_bcKhd * scale_bckh1
    return X_scaled, seqlen != S_total, S_total


# ---------------------------------------------------------------------------
# Layer accessors (mamba-only view)
# ---------------------------------------------------------------------------
def _mamba_mixers(model):
    """Return list of (global_block_idx, mixer) for mamba blocks only."""
    out = []
    for i, layer in enumerate(model.backbone.layers):
        if getattr(layer, "block_type", None) == "mamba":
            out.append((i, layer.mixer))
    return out


def _mamba_mixer_attrs(mix):
    """Resolve mixer attributes across mamba-ssm / Nemotron naming."""
    nheads = getattr(mix, "nheads", None) or getattr(mix, "num_heads")
    headdim = getattr(mix, "headdim", None) or getattr(mix, "head_dim")
    ngroups = getattr(mix, "ngroups", None) or getattr(mix, "n_groups")
    d_state = getattr(mix, "d_state", None) or getattr(mix, "ssm_state_size")
    chunk_size = getattr(mix, "chunk_size")
    intermediate = nheads * headdim
    return nheads, headdim, ngroups, d_state, chunk_size, intermediate


# ---------------------------------------------------------------------------
# Calibration helpers (mamba-only, Nemotron-aware)
# ---------------------------------------------------------------------------
def _load_cal_samples(tokenizer, n_samples, seq_len):
    raw = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
    samples = []
    for doc in raw:
        ids = tokenizer(
            doc["text"], return_tensors="pt", truncation=True, max_length=seq_len
        ).input_ids
        if ids.shape[1] >= seq_len:
            samples.append(ids[:, :seq_len])
        if len(samples) >= n_samples:
            break
    return samples


def calibrate_linear_input_scales(model, tokenizer, device, n_samples=32, seq_len=2048):
    """Per-tensor INT8 input scales for mamba-only in_proj / out_proj."""
    samples = _load_cal_samples(tokenizer, n_samples, seq_len)
    mixers = _mamba_mixers(model)
    n_mamba = len(mixers)
    in_max = [None] * n_mamba
    out_max = [None] * n_mamba

    def _make_hook(running, idx):
        def hook(mod, args):
            x = args[0]
            amax = x.abs().float().amax().detach()
            running[idx] = amax if running[idx] is None else torch.max(running[idx], amax)

        return hook

    handles = []
    for i, (_, mix) in enumerate(mixers):
        handles.append(mix.in_proj.register_forward_pre_hook(_make_hook(in_max, i)))
        handles.append(mix.out_proj.register_forward_pre_hook(_make_hook(out_max, i)))

    try:
        with torch.no_grad():
            for i, inp in enumerate(samples):
                model(inp.to(device))
                if (i + 1) % 16 == 0:
                    print(f"  in/out_proj calibration: {i + 1}/{len(samples)}")
    finally:
        for h in handles:
            h.remove()

    in_scales = [(m / 127.0).clamp(min=1e-10).to(device) for m in in_max]
    out_scales = [(m / 127.0).clamp(min=1e-10).to(device) for m in out_max]
    return in_scales, out_scales


def compute_d_scales(model, device):
    """Per-tensor INT8 D scale (amax/127) for every mamba mixer."""
    return [
        (mix.D.abs().float().amax() / 127.0).clamp(min=1e-10).to(device)
        for _, mix in _mamba_mixers(model)
    ]


def calibrate_x_scaled_hp_scales(
    model,
    tokenizer,
    device,
    n_samples=32,
    seq_len=2048,
    rotate=False,
    R_matrix=None,
):
    """Per-(H,P) scale of x_scaled = x * exp(dA_last - dA_cumsum) * dt (SSDi8).

    With `rotate=True`, the SSD runs end-to-end with rotated B/C to match runtime.
    """
    import mamba_ssm.ops.triton.ssd_combined as ssd_mod

    samples = _load_cal_samples(tokenizer, n_samples, seq_len)
    mixers = _mamba_mixers(model)
    n_mamba = len(mixers)
    running_max = [None] * n_mamba

    saved_fwd = ssd_mod._mamba_chunk_scan_combined_fwd
    rotated_base = _make_rotated_calib_fwd(_ORIGINAL_FWD, R_matrix if rotate else None)
    layer_counter = [0]

    def hooked_fwd(x, dt, A, B, C, chunk_size, **kwargs):
        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt,
            A,
            chunk_size,
            dt_bias=kwargs.get("dt_bias"),
            dt_softplus=kwargs.get("dt_softplus", False),
            dt_limit=kwargs.get("dt_limit", (0.0, float("inf"))),
        )
        X_scaled, _, _ = _compute_x_scaled(x, dt_out, dA_cumsum, chunk_size)
        amax = X_scaled.abs().float().amax(dim=(0, 2, 1))  # (H, P)
        idx = layer_counter[0] % n_mamba
        running_max[idx] = amax if running_max[idx] is None else torch.max(running_max[idx], amax)
        layer_counter[0] += 1
        return rotated_base(x, dt, A, B, C, chunk_size, **kwargs)

    ssd_mod._mamba_chunk_scan_combined_fwd = hooked_fwd
    try:
        with torch.no_grad():
            for i, inp in enumerate(samples):
                model(inp.to(device))
                if (i + 1) % 8 == 0:
                    print(f"  x_scaled_hp calibration: {i + 1}/{len(samples)}")
    finally:
        ssd_mod._mamba_chunk_scan_combined_fwd = saved_fwd

    return [(rm / 127.0).clamp(min=1e-10).to(device) for rm in running_max]


def calibrate_c_g_scales(
    model,
    tokenizer,
    device,
    n_samples=32,
    seq_len=2048,
    rotate=False,
    R_matrix=None,
):
    """Per-group (G,) scale of C at chunk_scan entry (SSDi8 C_cali).

    With `rotate=True`, the SSD runs end-to-end with rotated B/C to match
    runtime; the per-G amax observation here uses the same rotated `C`.
    """
    import mamba_ssm.ops.triton.ssd_combined as ssd_mod

    samples = _load_cal_samples(tokenizer, n_samples, seq_len)
    mixers = _mamba_mixers(model)
    n_mamba = len(mixers)
    running_max = [None] * n_mamba

    saved_fwd = ssd_mod._mamba_chunk_scan_combined_fwd
    rotated_base = _make_rotated_calib_fwd(_ORIGINAL_FWD, R_matrix if rotate else None)
    layer_counter = [0]

    def hooked_fwd(x, dt, A, B, C, chunk_size, **kwargs):
        C_obs = C  # observe the D-scaled C the runtime quantizes; rotated_base applies D to the forward itself
        if DQK_MATRIX is not None:
            C_obs = C * DQK_MATRIX[layer_counter[0] % n_mamba][1].to(C.device, C.dtype)
        if rotate and R_matrix is not None:
            R_l = (
                R_matrix[layer_counter[0] % n_mamba]
                if isinstance(R_matrix, (list, tuple))
                else R_matrix
            )
            R = R_l.to(C.device, C.dtype)
            C_cal = torch.einsum("btgn,gmn->btgm", C_obs, R) if R.dim() == 3 else C_obs @ R.T
        else:
            C_cal = C_obs
        amax = C_cal.abs().float().amax(dim=(0, 1, 3))  # (G,)
        idx = layer_counter[0] % n_mamba
        running_max[idx] = amax if running_max[idx] is None else torch.max(running_max[idx], amax)
        layer_counter[0] += 1
        return rotated_base(x, dt, A, B, C, chunk_size, **kwargs)

    ssd_mod._mamba_chunk_scan_combined_fwd = hooked_fwd
    try:
        with torch.no_grad():
            for i, inp in enumerate(samples):
                model(inp.to(device))
                if (i + 1) % 16 == 0:
                    print(f"  C per-G calibration: {i + 1}/{len(samples)}")
    finally:
        ssd_mod._mamba_chunk_scan_combined_fwd = saved_fwd

    return [(rm / 127.0).clamp(min=1e-10).to(device) for rm in running_max]


def calibrate_bc_gn_scales(
    model,
    tokenizer,
    device,
    n_samples=32,
    seq_len=2048,
    rotate=False,
    R_matrix=None,
):
    """Per-(G, N) scale of B at chunk_state entry (reduced to per-G at apply).

    With `rotate=True`, the SSD runs end-to-end with rotated B/C to match
    runtime; the per-(G,N) amax observation here uses the same rotated `B`.
    """
    import mamba_ssm.ops.triton.ssd_combined as ssd_mod

    samples = _load_cal_samples(tokenizer, n_samples, seq_len)
    mixers = _mamba_mixers(model)
    n_mamba = len(mixers)
    running_max = [None] * n_mamba

    saved_fwd = ssd_mod._mamba_chunk_scan_combined_fwd
    rotated_base = _make_rotated_calib_fwd(_ORIGINAL_FWD, R_matrix if rotate else None)
    layer_counter = [0]

    def hooked_fwd(x, dt, A, B, C, chunk_size, **kwargs):
        B_obs = B  # observe the D-scaled B the runtime quantizes; rotated_base applies D to the forward itself
        if DQK_MATRIX is not None:
            B_obs = B * DQK_MATRIX[layer_counter[0] % n_mamba][0].to(B.device, B.dtype)
        if rotate and R_matrix is not None:
            R_l = (
                R_matrix[layer_counter[0] % n_mamba]
                if isinstance(R_matrix, (list, tuple))
                else R_matrix
            )
            R = R_l.to(B.device, B.dtype)
            B_cal = torch.einsum("btgn,gmn->btgm", B_obs, R) if R.dim() == 3 else B_obs @ R.T
        else:
            B_cal = B_obs
        amax = B_cal.abs().float().amax(dim=(0, 1))  # (G, N)
        idx = layer_counter[0] % n_mamba
        running_max[idx] = amax if running_max[idx] is None else torch.max(running_max[idx], amax)
        layer_counter[0] += 1
        return rotated_base(x, dt, A, B, C, chunk_size, **kwargs)

    ssd_mod._mamba_chunk_scan_combined_fwd = hooked_fwd
    try:
        with torch.no_grad():
            for i, inp in enumerate(samples):
                model(inp.to(device))
                if (i + 1) % 8 == 0:
                    print(f"  B/C calibration: {i + 1}/{len(samples)}")
    finally:
        ssd_mod._mamba_chunk_scan_combined_fwd = saved_fwd

    return [(rm / 127.0).clamp(min=1e-10).to(device) for rm in running_max]


def calibrate_b_gate_hn_scales(
    model,
    tokenizer,
    device,
    n_samples=32,
    seq_len=2048,
    rotate=False,
    R_matrix=None,
):
    """Per-(H, N) amax of (gate * B) — the static b_gate_scale of the key-side chunk_state.

    Mirrors SSDi8 chunk_state's "b_gate_hn" observer: scale_lut is built
    from `dA_cumsum` and `dt`; gated_b = scale_lut * B[broadcast_to_H]; the
    amax is reduced over (B, C, K), keeping the per-(H, N) shape.

    With `rotate=True`, B is rotated to match runtime (same dispatch as the
    other rotated calibrations).
    """
    import mamba_ssm.ops.triton.ssd_combined as ssd_mod

    samples = _load_cal_samples(tokenizer, n_samples, seq_len)
    mixers = _mamba_mixers(model)
    n_mamba = len(mixers)
    running_max = [None] * n_mamba

    saved_fwd = ssd_mod._mamba_chunk_scan_combined_fwd
    rotated_base = _make_rotated_calib_fwd(_ORIGINAL_FWD, R_matrix if rotate else None)
    layer_counter = [0]

    def hooked_fwd(x, dt, A, B, C, chunk_size, **kwargs):
        B_obs = B  # observe the D-scaled B the runtime quantizes; rotated_base applies D to the forward itself
        if DQK_MATRIX is not None:
            B_obs = B * DQK_MATRIX[layer_counter[0] % n_mamba][0].to(B.device, B.dtype)
        if rotate and R_matrix is not None:
            R_l = (
                R_matrix[layer_counter[0] % n_mamba]
                if isinstance(R_matrix, (list, tuple))
                else R_matrix
            )
            R = R_l.to(B.device, B.dtype)
            B_cal = torch.einsum("btgn,gmn->btgm", B_obs, R) if R.dim() == 3 else B_obs @ R.T
        else:
            B_cal = B_obs

        dA_cumsum, dt_out = ssd_mod._chunk_cumsum_fwd(
            dt,
            A,
            chunk_size,
            dt_bias=kwargs.get("dt_bias", None),
            dt_softplus=kwargs.get("dt_softplus", False),
            dt_limit=kwargs.get("dt_limit", (0.0, float("inf"))),
        )
        # dA_cumsum, dt_out: (batch, nheads, nchunks, chunk_size)
        dA_last = dA_cumsum[..., -1:].contiguous()
        scale_lut = (torch.exp(dA_last - dA_cumsum) * dt_out).to(B_cal.dtype)  # (B,H,C,K)
        # Reshape B to (B, nchunks, chunk_size, ngroups, dstate)
        batch, seqlen, ngroups, dstate = B_cal.shape
        nheads = dA_cumsum.shape[1]
        nchunks = dA_cumsum.shape[2]
        B_pad = B_cal
        S_total = nchunks * chunk_size
        if seqlen != S_total:
            pad = torch.zeros(
                (batch, S_total - seqlen, ngroups, dstate),
                device=B_cal.device,
                dtype=B_cal.dtype,
            )
            B_pad = torch.cat([B_cal, pad], dim=1)
        b_bcKgn = B_pad.view(batch, nchunks, chunk_size, ngroups, dstate)
        nheads_per_g = nheads // ngroups
        group_idx = torch.arange(nheads, device=B_cal.device) // nheads_per_g
        b_per_head = b_bcKgn[:, :, :, group_idx, :]  # (B,C,K,H,N)
        scale_bcKh1 = scale_lut.permute(0, 2, 3, 1).unsqueeze(-1)  # (B,C,K,H,1)
        gated_b = b_per_head * scale_bcKh1  # (B,C,K,H,N)
        amax = gated_b.abs().float().amax(dim=(0, 1, 2))  # (H, N)
        idx = layer_counter[0] % n_mamba
        running_max[idx] = amax if running_max[idx] is None else torch.max(running_max[idx], amax)
        layer_counter[0] += 1
        return rotated_base(x, dt, A, B, C, chunk_size, **kwargs)

    ssd_mod._mamba_chunk_scan_combined_fwd = hooked_fwd
    try:
        with torch.no_grad():
            for i, inp in enumerate(samples):
                model(inp.to(device))
                if (i + 1) % 8 == 0:
                    print(f"  b_gate_hn calibration: {i + 1}/{len(samples)}")
    finally:
        ssd_mod._mamba_chunk_scan_combined_fwd = saved_fwd

    # Raw amax, NOT amax/127: the keyside chunk_state kernel treats b_gate_scale
    # as the clip threshold itself (quantizes with 127/b_gate, dequantizes with
    # b_gate/127), matching SSDi8's b_gate ChunkCollector(div=1.0).
    return [rm.clamp(min=1e-10).to(device) for rm in running_max]


def calibrate_x_hp_scales(
    model,
    tokenizer,
    device,
    n_samples=32,
    seq_len=2048,
    rotate=False,
    R_matrix=None,
):
    """Per-(H, P) scale of raw x at SSD entry (the persistent INT8 x of the key-side path).

    With `rotate=True`, the SSD runs end-to-end with rotated B/C to match runtime.
    """
    import mamba_ssm.ops.triton.ssd_combined as ssd_mod

    samples = _load_cal_samples(tokenizer, n_samples, seq_len)
    mixers = _mamba_mixers(model)
    n_mamba = len(mixers)
    running_max = [None] * n_mamba

    saved_fwd = ssd_mod._mamba_chunk_scan_combined_fwd
    rotated_base = _make_rotated_calib_fwd(_ORIGINAL_FWD, R_matrix if rotate else None)
    layer_counter = [0]

    def hooked_fwd(x, dt, A, B, C, chunk_size, **kwargs):
        amax = x.abs().float().amax(dim=(0, 1))  # (H, P)
        idx = layer_counter[0] % n_mamba
        running_max[idx] = amax if running_max[idx] is None else torch.max(running_max[idx], amax)
        layer_counter[0] += 1
        return rotated_base(x, dt, A, B, C, chunk_size, **kwargs)

    ssd_mod._mamba_chunk_scan_combined_fwd = hooked_fwd
    try:
        with torch.no_grad():
            for i, inp in enumerate(samples):
                model(inp.to(device))
                if (i + 1) % 8 == 0:
                    print(f"  x_hp calibration: {i + 1}/{len(samples)}")
    finally:
        ssd_mod._mamba_chunk_scan_combined_fwd = saved_fwd

    return [(rm / 127.0).clamp(min=1e-10).to(device) for rm in running_max]


def calibrate_state_scales(
    model,
    tokenizer,
    device,
    n_samples=32,
    seq_len=2048,
    rotate=False,
    R_matrix=None,
):
    """Per-(H, P) chunk-state amax (static state scale for the per-(H,P) path).

    With `rotate=True`, the SSD runs end-to-end rotated (B/C by R.T), matching
    runtime. The chunk_state observer sees B already rotated (by the wrapped SSD
    fwd) and captures the correct rotated-state amax.
    """
    import mamba_ssm.ops.triton.ssd_combined as ssd_mod

    samples = _load_cal_samples(tokenizer, n_samples, seq_len)
    mixers = _mamba_mixers(model)
    n_mamba = len(mixers)
    running_max = [None] * n_mamba

    saved_fwd = ssd_mod._mamba_chunk_scan_combined_fwd
    rotated_base = _make_rotated_calib_fwd(_ORIGINAL_FWD, R_matrix if rotate else None)
    ssd_mod._mamba_chunk_scan_combined_fwd = rotated_base

    original_cs = ssd_mod._chunk_state_fwd
    layer_counter = [0]

    def hooked_cs(B, x, dt, dA_cumsum, **kwargs):
        # x and B are already rotated by `rotated_base` upstream — observe directly.
        states = original_cs(B, x, dt, dA_cumsum, **kwargs)
        amax = states.abs().float().amax(dim=(0, 1, -1))  # (H, P)
        idx = layer_counter[0] % n_mamba
        running_max[idx] = amax if running_max[idx] is None else torch.max(running_max[idx], amax)
        layer_counter[0] += 1
        return states

    ssd_mod._chunk_state_fwd = hooked_cs
    try:
        with torch.no_grad():
            for i, inp in enumerate(samples):
                model(inp.to(device))
                if (i + 1) % 8 == 0:
                    print(f"  State calibration: {i + 1}/{len(samples)}")
    finally:
        ssd_mod._chunk_state_fwd = original_cs
        ssd_mod._mamba_chunk_scan_combined_fwd = saved_fwd

    return [(rm / 127.0).clamp(min=1e-10).to(device) for rm in running_max]


# ---------------------------------------------------------------------------
# Layer counter hook — maps mamba mixers to mamba-only indices
# ---------------------------------------------------------------------------
def install_layer_hooks(model):
    """Forward-pre-hooks on each mamba mixer to update FQCFG['layer_idx'].

    The hook sets FQCFG['layer_idx'] to the mamba-only sequential index,
    used by the patched SSD fwd to pick the right per-layer rotation and D_qk.
    """
    mixers = _mamba_mixers(model)
    hooks = []
    for i, (_, mix) in enumerate(mixers):

        def _mk(idx):
            def hook(module, args):
                FQCFG["layer_idx"] = idx

            return hook

        hooks.append(mix.register_forward_pre_hook(_mk(i)))
    return hooks
