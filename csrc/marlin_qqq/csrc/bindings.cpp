// Pybind11 bindings for standalone Marlin QQQ GEMM kernel.
// The actual kernel is compiled separately as a .cu file; we just declare
// the C++ entry point here and bind it.

#include <torch/extension.h>

// Forward declaration of the entry point defined in marlin_qqq_gemm_kernel.cu
torch::Tensor marlin_qqq_gemm(torch::Tensor const& a,
                               torch::Tensor const& b_q_weight,
                               torch::Tensor const& s_tok,
                               torch::Tensor const& s_ch,
                               torch::Tensor const& s_group,
                               torch::Tensor& workspace, int64_t size_m,
                               int64_t size_n, int64_t size_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("marlin_qqq_gemm", &marlin_qqq_gemm,
        "Marlin QQQ W4A8 GEMM (CUDA)",
        py::arg("a"), py::arg("b_q_weight"), py::arg("s_tok"),
        py::arg("s_ch"), py::arg("s_group"), py::arg("workspace"),
        py::arg("size_m"), py::arg("size_n"), py::arg("size_k"));
}
