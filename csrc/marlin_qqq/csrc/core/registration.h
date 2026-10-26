#pragma once

// Standalone replacement for vLLM's registration.h
// We use direct pybind11 binding instead of TORCH_LIBRARY_IMPL, so this
// header just provides the macro as a no-op.

#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  static void _unused_##NAME##_##DEVICE(torch::Library& MODULE)
