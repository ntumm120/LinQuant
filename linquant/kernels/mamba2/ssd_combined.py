"""INT8 SSD combined forward: cumsum, state, passing, bmm, scan.

Always-FP-x: x is consumed as FP16; the chunk_state kernel quantizes it
per tile internally (matches SSDi8). No INT8-pre-quantized x path.
"""

from .bmm_chunk import _quant_bmm_chunk_fwd
from .chunk_cumsum import _quant_chunk_cumsum_fwd
from .chunk_scan_v2 import _quant_chunk_scan_fwd as _chunk_scan_v2
from .chunk_state import _quant_chunk_state_fwd_xfp_static_int8
from .chunk_state_keyside import _chunk_state_keyside_fwd
from .state_passing import _fixedpoint_state_passing_fwd

# Fixed-point gate fractional bits for the all-INT8 state-passing kernel.
# Matches SSDi8 (Q0.7); never tuned in any canonical or ablation run.
_GATE_SHIFT = 7


def _quant_mamba_chunk_scan_combined_fwd(  # noqa: D417
    x_fp,
    q_dt,
    dt_scale,
    q_A_log,
    A_log_scale,
    q_B,
    B_scale,
    q_C,
    C_scale,
    chunk_size,
    q_D,
    D_scale,
    dt_bias=None,
    dt_softplus=True,
    dt_limit=(0.0, float("inf")),
    static_state_scale=None,
    static_cb_scale=None,
    alpha_hp=None,
    bscale_g=None,
    keyside=False,
    x_scale_hp=None,
    b_fp=None,
    b_gate_scale=None,
):
    """INT8 SSD combined forward — canonical paper-parity path.

    FP-x in, INT8 chunk states (value-side: per-tile x quant with ``alpha_hp``
    against INT8 ``q_B``; key-side: in-kernel x quant with ``x_scale_hp``
    against gated FP16 ``b_fp`` quantized with ``b_gate_scale``), INT8
    fixed-point recurrence, INT8 CB with the static per-G ``static_cb_scale``,
    direct INT8 chunk_scan. Returns (out, None) — `final_states` is discarded by
    every live caller (PPL eval drops it).

    Caller contract (enforced at construction time in QuantizedMamba2 / the
    SSDi8-parity eval scripts): contiguous q_B, q_C, x_fp; pre-staged FP32
    on-device buffers for static_state_scale, alpha_hp, bscale_g,
    static_cb_scale (and x_scale_hp, b_gate_scale when keyside).
    """
    batch, seqlen, nheads, headdim = x_fp.shape
    _, _, ngroups, dstate = q_B.shape

    dA_cumsum, dt = _quant_chunk_cumsum_fwd(
        q_dt,
        dt_scale,
        q_A_log,
        A_log_scale,
        chunk_size,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )

    if keyside:
        states_i8 = _chunk_state_keyside_fwd(
            x_fp,
            x_scale_hp,
            b_fp,
            dt,
            dA_cumsum,
            static_state_scale,
            b_gate_scale,
        )
    else:
        states_i8 = _quant_chunk_state_fwd_xfp_static_int8(
            x_fp,
            q_B,
            dt,
            dA_cumsum,
            alpha_hp,
            bscale_g,
            static_state_scale,
        )

    # Reshape for state_passing: (B, C, H, P, N) -> (B, C, H, P*N). View-safe
    # since `states_i8` is freshly torch.empty'd row-major contiguous.
    states_flat_i8 = states_i8.reshape(batch, -1, nheads, headdim * dstate)

    sp_out_i8, _ = _fixedpoint_state_passing_fwd(
        states_flat_i8,
        dA_cumsum[:, :, :, -1],
        gate_shift=_GATE_SHIFT,
    )

    states = sp_out_i8.reshape(batch, -1, nheads, headdim, dstate)

    # --- BMM CB + chunk_scan ---
    # static_cb_scale is a pre-staged FP32 (G,) contiguous buffer; the kernel
    # emits INT8 CB directly on that scale.
    cb_i8 = _quant_bmm_chunk_fwd(
        q_C,
        C_scale,
        b_fp if keyside else q_B,
        B_scale,
        chunk_size,
        static_cb_scale,
        b_is_fp=keyside,
    )

    out = _chunk_scan_v2(
        cb_i8,
        x_fp,
        dt,
        dA_cumsum,
        q_C,
        C_scale,
        states,
        q_D,
        D_scale,
        static_state_scale,
        static_cb_scale,
    )
    return out, None
