"""Zoo-lineage INT8 chunk kernels: FLA's chunk kernels with quantize-before-dot, INT8 stored state, fp32 carry.

Profile (docs/zoo_result_plan.md, "Zoo (FLA-derived, minimal)"): scalar per-head scales on the Q·Kᵀ and state-read
operands, per-channel static scales on the write operands and the stored state, fp32 carry across chunks with an
INT8 stored copy read by the next chunk, C·Bᵀ kept fp32, intra-chunk value product as an fp16 dot on INT8 codes
with the channel scale applied after, read-side decay applied after the product, rotations applied by the caller.
"""
