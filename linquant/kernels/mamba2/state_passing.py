"""All-INT8 fixed-point state passing for the INT8 SSD (matches SSDi8).

The inter-chunk recurrence runs in INT32 on the INT8 chunk states with the gate
exp(dA) quantized to Q0.GATE_SHIFT fixed point:
    s32 = ((s32 * gate_q + round) >> GATE_SHIFT) + new_s32
"""


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["dim"],
)
@triton.jit
def _fixedpoint_state_passing_fwd_kernel(
    # All-INT8 fixed-point state passing (matches SSDi8 exactly).
    # Input: INT8 chunk states. Output: INT8 passed states + final state.
    # Recurrence: s32 = (s32 * gate_q + round) >> SHIFT + new_s32
    states_ptr, out_ptr, final_states_ptr,
    dA_cs_ptr,
    dim, nchunks,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head, stride_final_states_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    BLOCK_SIZE: tl.constexpr,
    GATE_SHIFT: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    # Initialize accumulator as INT32 (zero initial state)
    s32 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # Store initial state
    s8_store = tl.minimum(tl.maximum(s32, -127), 127).to(tl.int8)
    tl.store(out_ptrs, s8_store, mask=offs_m < dim)
    out_ptrs += stride_out_chunk

    for c in range(nchunks):
        # Load new chunk state (INT8)
        new_s8 = tl.load(states_ptrs, mask=offs_m < dim, other=0).to(tl.int8)
        new_s32 = tl.cast(new_s8, tl.int32)

        # Gate = exp(dA_last)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        gate_fp = tl.exp(dA_cs)

        # Quantize gate to Q0.SHIFT fixed-point
        S = 1 << GATE_SHIFT
        gate_scaled = gate_fp * S
        gate_r = tl.where(gate_scaled >= 0, tl.floor(gate_scaled + 0.5), tl.ceil(gate_scaled - 0.5))
        gate_r = tl.minimum(tl.maximum(gate_r, -127.0), 127.0)
        gate_q = tl.cast(gate_r, tl.int32)

        # INT32 recurrence
        mul32 = s32 * gate_q
        mul32 = (mul32 + (S >> 1)) >> GATE_SHIFT
        s32 = mul32 + new_s32

        # Store INT8
        if c < (nchunks - 1):
            s8_mid = tl.minimum(tl.maximum(s32, -127), 127).to(tl.int8)
            tl.store(out_ptrs, s8_mid, mask=offs_m < dim)
        else:
            s8_fin = tl.minimum(tl.maximum(s32, -127), 127).to(tl.int8)
            tl.store(final_states_ptrs, s8_fin, mask=offs_m < dim)

        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _fixedpoint_state_passing_fwd(states_i8, dA_chunk_cumsum, gate_shift=7):
    """SSDi8-matched all-INT8 fixed-point state passing (zero initial state).

    Args:
        states_i8: (batch, nchunks, nheads, dim) INT8 chunk states.
        dA_chunk_cumsum: (batch, nheads, nchunks).
        gate_shift: fractional bits of the fixed-point gate (Q0.gate_shift).
    Returns:
        (out_i8, final_states_i8) — both INT8; out_i8[:, c] is the state entering chunk c.
    """
    batch, nchunks, nheads, dim = states_i8.shape
    out = torch.empty_like(states_i8)
    final_states = torch.empty((batch, nheads, dim), device=states_i8.device, dtype=torch.int8)

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), batch, nheads)
    with torch.cuda.device(states_i8.device.index):
        _fixedpoint_state_passing_fwd_kernel[grid](
            states_i8, out, final_states, dA_chunk_cumsum,
            dim, nchunks,
            states_i8.stride(0), states_i8.stride(1), states_i8.stride(2), states_i8.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final_states.stride(0), final_states.stride(1), final_states.stride(2),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            GATE_SHIFT=gate_shift,
        )
    return out, final_states
