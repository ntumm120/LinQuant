[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](TODO-arxiv-link)

# LinQuant: Functional Invariances for Low-Precision Linear Recurrences

**LinQuant** quantizes the chunkwise prefill of linear recurrences (linear attention, Mamba-2, GLA, DeltaNet, Gated DeltaNet, KDA) by exploiting *functional invariances*: transformations that leave the full-precision recurrence unchanged while making the quantized operands easier to represent. The framework covers seven such invariances, including rotations and diagonal rescalings of the key and value axes and alternative placements of the recurrent decay and write factors, and it is applied on top of [SSDi8](https://github.com/cau-hai-lab/SSDi8), an INT8 prefill pipeline for Mamba-2, to produce W4A8 Mamba-2 and Nemotron-H models with INT8 recurrent states.

This repository contains everything needed to reproduce the paper's main-text results: the fake-quantized reference recurrence and the mixer-zoo harness, the INT8 Triton kernels used for timing, and the Mamba-2 / Nemotron-H W4A8 production path.

```
linquant/
  reference.py        chunkwise recurrence (all families) with the R / U rotations, decay and write-gate placements,
                      diagonal rescalings, INT8 stored state; the single source of truth for the invariances
  invariances.py      fake quantizers: static per-role scales (per tensor / head / channel), the realizable
                      kernel policies kernel_static, kernel_pt, kernel_ptk
  harness.py, run.py  patch a mixer-zoo model to run its recurrence through reference.py, calibrate, eval WikiText-2
  kernels/zoo/        INT8 chunk kernels for linear attention, GLA, DeltaNet, Gated DeltaNet, KDA (FLA structure)
  kernels/mamba2/     INT8 Mamba-2 SSD kernels (value-side and key-side absorption, INT8 state passing)
  quantized_mamba2.py our W4A8 Mamba-2 mixer; w4a8_linear.py, marlin_utils.py: W4A8 linears on the Marlin GEMM
scripts/              production rows: mamba2_w4a8.py, nemotron_h_w4a8.py (+ nemotron_h_calibration.py)
bench/                kernel timing (zoo_kernels.py), full-model prefill (mamba2_model.py), per-stage (mamba2_components.py)
tests/                zoo kernels vs reference.py, reference.py invariances (exact, float64)
libs/SSDi8            submodule: upstream SSDi8, pinned; patches/ssdi8.patch turns it into our fork
csrc/marlin_qqq       Marlin QQQ W4A8 GEMM (from vLLM), built as a standalone extension
```

## Installation

Tested with Python 3.11, CUDA 12.8, torch 2.8.0, triton 3.4.0 on H100 (perplexity) and A5000 (timing).

Clone with submodules and apply our patch to SSDi8 (calibration hooks for the rotation and key rescaling, the Marlin W4A8 projections, an H100 build fix, and one numerical fix described under [Notes](#notes)):

```bash
git clone --recursive https://github.com/ntumm120/LinQuant.git && cd LinQuant
git -C libs/SSDi8 apply ../../patches/ssdi8.patch
```

Install the package and build the two CUDA extensions:

```bash
pip install -e .                                                         # pins torch, triton, fla-core, mamba-ssm, causal-conv1d, transformers
cd libs/SSDi8 && pip install . --no-build-isolation && cd ../..          # SSDi8 extensions (quant_linear_cuda, quant_hadamard, fast_hadamard_transform)
cd csrc/marlin_qqq && python setup.py build_ext --inplace && cd ../..    # Marlin W4A8 GEMM (sm80 / sm86 / sm90)
```

nvcc needs a host gcc it supports (CUDA 12.8: gcc <= 13). If your environment's default `gcc` is newer (conda-forge ships gcc 14), point both builds at a system compiler: `CC=/usr/bin/gcc CXX=/usr/bin/g++ python setup.py build_ext --inplace`. Model weights and datasets (WikiText-2, the Pile-uncopyrighted validation shard) are fetched from the Hugging Face Hub.

`python -m pytest tests/` checks the INT8 zoo kernels against `reference.py` and the invariances themselves; it needs no model weights.

## Reproducing the paper

### Controlled study: invariances with fake quantization (Section 4)

The zoo models (340M and 1.3B, one cell per sequence mixer) come from the [mixer zoo](#mixer-zoo); the harness imports that repo's modelling code and loads its checkpoints:

```bash
export MIXERZOO_REPO=/path/to/mixer-zoo-checkout
export MIXERZOO_MODELS=/path/to/checkpoints      # <MIXERZOO_MODELS>/340m/<cell>/model.safetensors (+ config, tokenizer)
```

One run is one cell and a list of arms: calibration on 8 WikiText-2 windows (seed 0, 2048 tokens), then WikiText-2 perplexity. Arms are `key=value` lists (grammar in `linquant/run.py`):

```bash
# FP16 reference, naive INT6 operands / INT8 state, + R_QK, + D_QK (sqk), and the write-gate placement (rho)
python -m linquant.run --cell deltanet --size 340m --arms \
    "q=none" \
    "policy=kernel_static,bits=6,state=8" \
    "rot=qk,policy=kernel_static,bits=6,state=8" \
    "rot=qk,sqk=0.5,policy=kernel_static,bits=6,state=8" \
    "rot=qk,rho=0.5,policy=kernel_static,bits=6,state=8"
```

Cells: `linear_none`, `mamba2`, `gla` (linear attention with no / scalar / diagonal decay), their `_beta` gated variants, `deltanet`, `gated_deltanet`, `kda`. Knobs: `rot=none|qk|v|qkv`, `storage=rotated|canonical`, `eta` / `rho` / `tau` (decay and write-gate placements), `sqk` / `sv` (diagonal rescaling exponents), `policy=kernel_static|kernel_pt|kernel_ptk`, `bits`, `state`. Records are appended to `results/zoo/<size>/<cell>.<host>-<pid>.jsonl`.

### Production: Mamba-2 and Nemotron-H W4A8 (Section 5)

The SSDi8 pipeline (Pile calibration, GPTQ W4, output-mean compensation) runs unchanged; we add the R_QK rotation of B/C inside its calibration forward, the diagonal key rescaling D_QK, and swap each mixer for `QuantizedMamba2` (INT8 recurrence, INT8 stored state, value- or key-side absorption).

```bash
M=state-spaces/mamba2-1.3b          # or state-spaces/mamba2-2.7b, or a converted mamba2-8b checkpoint
S="--rotation_signs random --rotation_seed 42"
python scripts/mamba2_w4a8.py --model $M --rotation none --no_comp_out $S                 # SSDi8 row
python scripts/mamba2_w4a8.py --model $M --rotation r --no_comp_out $S                    # + R_QK (value side)
python scripts/mamba2_w4a8.py --model $M --rotation r --no_comp_out --dqk_alpha 0.5 $S    # + D_QK (value side)
python scripts/mamba2_w4a8.py --model $M --keyside --rotation r --comp_decay 0.15 --dqk_alpha 0.5 $S   # key side
# add --calib_seed {1,2,3} for the calibration-seed repeats

python scripts/nemotron_h_w4a8.py --model 8b-4k --rotation none                           # FP16 + SSDi8 rows
python scripts/nemotron_h_w4a8.py --model 8b-4k --rotation r --rotation_seed 42
python scripts/nemotron_h_w4a8.py --model 8b-4k --rotation r --rotation_seed 42 --dqk_alpha 0.5
python scripts/nemotron_h_w4a8.py --model 8b-4k --rotation r --rotation_seed 42 --dqk_alpha 0.5 --keyside
```

### Timing

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

The Mamba-2 benches use random weights with the production kernel paths; they are not numerically meaningful. The `ours_*` configs can abort with `free(): invalid pointer` while the interpreter exits (SSDi8 CUDA extension teardown); the JSON has been written by then and is complete.

## Hybrid models

Nemotron-H 8B (Mamba-2 mixers interleaved with attention and MLP blocks) is quantized through `scripts/nemotron_h_w4a8.py`, which runs SSDi8's calibration and GPTQ W4 on the Mamba-2 in/out projections and swaps in `QuantizedMamba2`; the attention and MLP linears and the LM head stay in FP16.

## Mixer zoo

The controlled study uses models that differ only in their sequence mixer (linear attention with no, scalar, or diagonal decay, each with and without a write gate, plus DeltaNet, Gated DeltaNet, and KDA), trained with the [mixer-zoo repository](https://github.com/phnazari/LinQuant). The `*_nonorm` cells of the appendix need that repository's `qk_norm="none"` option.

## Notes

* The FP16 and SSDi8 rows of the production table come from the unmodified SSDi8 pipeline (`libs/SSDi8`, `SSDi8_W4A8.sh`); `--rotation none --no_comp_out` runs the same quantization through our mixer.
* `patches/ssdi8.patch` carries one numerical fix over upstream: the fused Hadamard + INT8 kernel (`quant_hadamard`, used for the out_proj input) cast the rounded float to `int8_t` without a clamp, which is undefined out of range and compiles to a saturating conversion with some toolchains and a wrapping one with others. The patch clamps to ±127 (the behaviour of the build the paper's numbers were produced with). Rebuild the extension after applying it. `UPSTREAM_DIVERGENCE.md` inside the patched tree lists every change.
* The production rows are not bit-reproducible run to run (Triton autotuning and cuBLAS pick kernels by timing; the sequential compensation amplifies the differences): on Mamba-2 1.3B the same command on the same H100 gives WikiText-2 perplexities about 5e-3 apart, and the paper's rows were likewise single runs. Compare rows at that resolution. The zoo (fake quantization against a float64 reference) is deterministic.
* Figure and table scripts are not included; every figure reads the JSON / JSONL records the commands above write.

## Citation

```bibtex
TODO
```

## Dependencies

* [SSDi8](https://github.com/cau-hai-lab/SSDi8) (UT Austin Research License, non-commercial), the production pipeline this work extends
* [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), the chunk-kernel structure of `linquant/kernels/zoo`
* [Mamba](https://github.com/state-spaces/mamba) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
* [vLLM](https://github.com/vllm-project/vllm), source of the Marlin QQQ W4A8 GEMM in `csrc/marlin_qqq`
* [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform), built as part of SSDi8
* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), zero-shot evaluation through SSDi8's wrapper

## Acknowledgements

This work was supported by the Hector Foundation, Max Planck ETH Center for Learning Systems, and the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
