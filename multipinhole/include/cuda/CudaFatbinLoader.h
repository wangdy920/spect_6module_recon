#pragma once

#include "cuda/CudaDriverHelpers.h"
#include "cuda/CudaNvrtcArch.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace spect::cuda {

inline bool isFrozenConsumerArch(int major, int minor) {
  return (major == 8 && (minor == 6 || minor == 9)) ||
         (major == 12 && minor == 0);
}

inline void requireFrozenConsumerArch(const cudaDeviceProp &props) {
  if (isFrozenConsumerArch(props.major, props.minor))
    return;
  throw std::runtime_error(
      "GPU " + cudaSmTag(props) +
      " is outside the frozen field set sm_86/89/120. Helper kernels are "
      "build-time fatbins; runtime NVRTC is disabled.");
}

inline void loadEmbeddedFatbin(CUmodule *module, const unsigned char *image,
                               unsigned long long size, const char *name) {
  if (image == nullptr || size == 0) {
    throw std::runtime_error(std::string("embedded fatbin is empty: ") +
                             (name ? name : "(unnamed)"));
  }
  SPECT_CUDRV_CHECK(cuModuleLoadDataEx(module, image, 0, nullptr, nullptr));
}

} // namespace spect::cuda
