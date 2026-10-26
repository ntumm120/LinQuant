# LinQuant

Code for *LinQuant: rotation and rescaling invariances for quantizing chunkwise linear attention*.
Everything needed to reproduce the main-text results: the fake-quantized reference recurrence and the
zoo harness (Section 4), the INT8 Triton kernels (kernel timing), and the Mamba-2 / Nemotron-H W4A8
production path on top of SSDi8 (the production table and the prefill timing).

```
linquant/
  reference.py        chunkwise recurrence (all five families) with the R / U rotations, decay and beta placements,
                      diagonal rescalings, INT8 stored state; the single source of truth for the invariances
  invariances.py      fake quantizers: static per-role scales (per tensor / head / channel), the realizable
                      kernel policies kernel_static, kernel_pt, kernel_ptk
  harness.py, run.py  patch a mixer-zoo model to run its recurrence through reference.py, calibrate, eval WikiText-2
  kernels/zoo/        INT8 chunk kernels for linear attention, GLA, DeltaNet, gated DeltaNet, KDA (FLA structure)
  kernels/mamba2/     INT8 Mamba-2 SSD kernels (value-side and key-side absorption, INT8 state passing)
  quantized_mamba2.py our W4A8 Mamba-2 mixer; w4a8_linear.py, marlin_utils.py: W4A8 linears on the Marlin GEMM
scripts/
  mamba2_w4a8.py            Mamba-2 1.3B / 2.7B / 8B production rows (SSDi8 pipeline + ours)
  nemotron_h_w4a8.py        Nemotron-H 8B rows (+ nemotron_h_calibration.py)
bench/
  zoo_kernels.py            kernel timing: FLA bf16 vs INT8 naive vs INT8 + invariances
  mamba2_model.py           full-model prefill timing, Mamba-2 configs fp16 / ssdi8 / ours
  mamba2_components.py      per-stage recurrence timing (the P/N crossover geometries)
tests/                      zoo kernels vs reference.py, reference.py invariances (exact, float64)
libs/SSDi8                  submodule: our fork of SSDi8 (upstream + calibration hooks + Marlin W4A8 patch)
csrc/marlin_qqq             Marlin QQQ W4A8 GEMM (from vLLM), built as a standalone extension
```

## Install

Tested with Python 3.11, CUDA 12.8, torch 2.8.0, triton 3.4.0 on H100 (PPL) and A5000 (timing).

```bash
git clone --recursive <this repo> && cd LinQuant
pip install -e .                                   # pins torch, triton, fla-core, mamba-ssm, causal-conv1d, transformers
cd libs/SSDi8 && pip install . --no-build-isolation && cd ../..     # SSDi8 CUDA extensions (quant_linear_cuda, quant_hadamard, ...)
cd csrc/marlin_qqq && python setup.py build_ext --inplace && cd ../..   # Marlin W4A8 GEMM (sm80 / sm86 / sm90)
```

nvcc needs a host gcc it supports (CUDA 12.8: gcc <= 13). If your environment's default `gcc` is newer (conda-forge
ships gcc 14), point the two builds at a system compiler: `CC=/usr/bin/gcc CXX=/usr/bin/g++ python setup.py build_ext --inplace`.
The SSDi8 build also provides `fast_hadamard_transform`. Model weights and datasets (WikiText-2, Pile-uncopyrighted
validation shard) are fetched from the Hugging Face Hub.

## Zoo (Section 4): invariances with fake quantization

The zoo models (340M, six cells) were trained with the mixer-zoo repo (github.com/phnazari/LinQuant); the harness
imports that repo's modelling code and loads its checkpoints:

```bash
export MIXERZOO_REPO=/path/to/mixer-zoo-checkout
export MIXERZOO_MODELS=/path/to/checkpoints      # <MIXERZOO_MODELS>/340m/<cell>/model.safetensors (+ config, tokenizer)
```

One run = one cell, a list of arms, calibration on 8 WikiText-2 windows (seed 0, 2048 tokens), WikiText-2 PPL.
Arms are `key=value` lists (grammar in `linquant/run.py`):

```bash
# FP16 reference, naive INT6 operands / INT8 state, + R_QK, + D_QK (sqk) and the beta placement (rho)
python -m linquant.run --cell deltanet --size 340m --arms \
    "q=none" \
    "policy=kernel_static,bits=6,state=8" \
    "rot=qk,policy=kernel_static,bits=6,state=8" \
    "rot=qk,sqk=0.5,policy=kernel_static,bits=6,state=8" \
    "rot=qk,rho=0.5,policy=kernel_static,bits=6,state=8"
```

Cells: `linear_none`, `mamba2`, `gla` (linear attention with no / scalar / diagonal decay), their `_beta` gated
variants, `deltanet`, `gated_deltanet`, `kda`. Knobs: `rot=none|qk|v|qkv`, `storage=rotated|canonical`,
`eta` / `rho` / `tau` (decay and beta placements), `sqk` / `sv` (diagonal scaling exponents),
`policy=kernel_static|kernel_pt|kernel_ptk`, `bits`, `state`. Records are appended to
`results/zoo/<size>/<cell>.<host>-<pid>.jsonl`.

`python -m pytest tests/` checks the INT8 zoo kernels against `reference.py` and the invariances themselves.

## Production table: Mamba-2 and Nemotron-H W4A8

The SSDi8 pipeline (Pile calibration, GPTQ W4, output-mean compensation) runs unchanged; we add the R_QK rotation
of B/C inside its calibration forward, the diagonal key rescaling D_QK, and swap each mixer for
`QuantizedMamba2` (INT8 recurrence, INT8 stored state, value- or key-side absorption).

```bash
M=state-spaces/mamba2-1.3b          # or state-spaces/mamba2-2.7b, or a converted mamba2-8b checkpoint
S="--rotation_signs random --rotation_seed 42"
python scripts/mamba2_w4a8.py --model $M --rotation none --no_comp_out $S                 # SSDi8 column
python scripts/mamba2_w4a8.py --model $M --rotation r --no_comp_out $S                    # + R_QK (value side)
python scripts/mamba2_w4a8.py --model $M --rotation r --no_comp_out --dqk_alpha 0.5 $S    # + D_QK (value side)
python scripts/mamba2_w4a8.py --model $M --keyside --rotation r --comp_decay 0.15 --dqk_alpha 0.5 $S   # key side
# add --calib_seed {1,2,3} for the calibration-seed repeats

python scripts/nemotron_h_w4a8.py --model 8b-4k --rotation none                           # FP16 + SSDi8 columns
python scripts/nemotron_h_w4a8.py --model 8b-4k --rotation r --rotation_seed 42
python scripts/nemotron_h_w4a8.py --model 8b-4k --rotation r --rotation_seed 42 --dqk_alpha 0.5
python scripts/nemotron_h_w4a8.py --model 8b-4k --rotation r --rotation_seed 42 --dqk_alpha 0.5 --keyside
```

## Timing

```bash
python bench/zoo_kernels.py --out results/zoo/kernel/timing.jsonl          # kernel panel (H=8, K=V=128, T=2048)
for c in fp16 ssdi8 ours_value_r ours_key_r; do                             # prefill panel
  python bench/mamba2_model.py --config $c --model state-spaces/mamba2-1.3b --output results/timing/model/$c.json
done
for g in n64_p32 n128_p64 n128_p128 n128_p256; do                           # P/N crossover (B=8, T=2048)
  for c in ssdi8 ours_value_r ours_key_r; do
    python bench/mamba2_components.py --config $c --model_size $g --output results/timing/pershape/${g}_$c.json
  done
done
```

The Mamba-2 benches use random weights with the production kernel paths; they are not numerically meaningful.
The `ours_*` configs can abort with `free(): invalid pointer` while the interpreter exits (SSDi8 CUDA extension
teardown); the JSON has been written by then and is complete.

## Notes

* The FP16 and SSDi8 columns of the production table come from the unmodified SSDi8 pipeline (`libs/SSDi8`,
  `SSDi8_W4A8.sh`); `--rotation none --no_comp_out` runs the same quantization through our mixer.
* The zoo `*_nonorm` cells (appendix) need the mixer-zoo repo's `qk_norm="none"` option.
* `libs/SSDi8` carries one numerical fix over upstream: the fused Hadamard + INT8 kernel (`quant_hadamard`,
  used for the out_proj input) cast the rounded float to `int8_t` without a clamp, which is undefined out of range
  and compiles to a saturating conversion with some toolchains and a wrapping one with others. The fork clamps to
  ±127 (the behaviour of the build the paper's numbers were produced with). Rebuild the extension after pulling.
* The production rows are not bit-reproducible run to run (Triton autotuning and cuBLAS pick kernels by timing;
  the sequential compensation amplifies the differences): on Mamba-2 1.3B the same command on the same H100 gives
  WikiText-2 perplexities about 5e-3 apart, and the paper's rows were likewise single runs. Compare rows at that
  resolution. The zoo (fake-quant, float64 reference) is deterministic.
* Figure and table scripts are not included; every figure reads the JSON / JSONL records the commands above write.
