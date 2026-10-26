"""Build standalone Marlin QQQ W4A8 GEMM kernel as a PyTorch C++ extension."""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(ROOT, "csrc")

setup(
    name="marlin_qqq",
    ext_modules=[
        CUDAExtension(
            name="marlin_qqq",
            sources=[
                os.path.join(CSRC, "bindings.cpp"),
                os.path.join(
                    CSRC,
                    "quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu",
                ),
            ],
            include_dirs=[CSRC],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--generate-code=arch=compute_80,code=sm_80",
                    "--generate-code=arch=compute_86,code=sm_86",
                    "--generate-code=arch=compute_90,code=sm_90",
                    "--generate-code=arch=compute_90a,code=sm_90a",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--threads=4",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
