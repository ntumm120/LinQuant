"""Loader for the `quant_linear_cuda` extension (CUTLASS INT8 GEMMs + Marlin W4A8 entry point).

Loads the build artifact of the SSDi8 submodule (`libs/SSDi8/quant_linear_cuda.*.so`, built
with `cd libs/SSDi8 && pip install . --no-build-isolation`) and registers it under the module
name `quant_linear_cuda`, falling back to a pip-installed module of that name. The first
extension loaded under that name wins for the whole process, so this loader runs before
`vllm_kernel_patch`, which then swaps in vLLM's Marlin kernel for the W4A8 GEMM.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

quant_linear_cuda = None

_pyver = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_ssdi8_so = glob.glob(os.path.join(_ROOT, "libs", "SSDi8", f"quant_linear_cuda.{_pyver}*.so")) or glob.glob(
    os.path.join(_ROOT, "libs", "SSDi8", "quant_linear_cuda.cpython-*.so")
)
if _ssdi8_so:
    _spec = importlib.util.spec_from_file_location("quant_linear_cuda", _ssdi8_so[0])
    quant_linear_cuda = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(quant_linear_cuda)
    sys.modules["quant_linear_cuda"] = quant_linear_cuda

if quant_linear_cuda is None:
    try:
        import quant_linear_cuda  # type: ignore[no-redef]  # noqa: F811
    except ImportError:
        raise ImportError(
            "quant_linear_cuda not found. Build it with:\n  cd libs/SSDi8 && python -m pip install . --no-build-isolation"
        )
