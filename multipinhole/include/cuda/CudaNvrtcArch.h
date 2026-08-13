#pragma once

#include <cuda_runtime_api.h>

#include <string>

namespace spect::cuda {

/// Format NVRTC `--gpu-architecture=` for a CUDA compute capability.
/// Examples: (8,9) -> `--gpu-architecture=compute_89`,
///           (12,0) -> `--gpu-architecture=compute_120`.
inline std::string nvrtcGpuArchitectureOption(int major, int minor) {
  return "--gpu-architecture=compute_" + std::to_string(major) +
         std::to_string(minor);
}

inline std::string nvrtcGpuArchitectureOption(const cudaDeviceProp &props) {
  return nvrtcGpuArchitectureOption(props.major, props.minor);
}

/// Human-readable CC tag used in logs (e.g. "sm_89", "sm_120").
inline std::string cudaSmTag(int major, int minor) {
  return "sm_" + std::to_string(major) + std::to_string(minor);
}

inline std::string cudaSmTag(const cudaDeviceProp &props) {
  return cudaSmTag(props.major, props.minor);
}

} // namespace spect::cuda
