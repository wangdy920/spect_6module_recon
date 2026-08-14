#pragma once

#include <cuda_runtime_api.h>

#include <string>

namespace spect::cuda {

/// Human-readable CC tag used in logs (e.g. "sm_89", "sm_120").
inline std::string cudaSmTag(int major, int minor) {
  return "sm_" + std::to_string(major) + std::to_string(minor);
}

inline std::string cudaSmTag(const cudaDeviceProp &props) {
  return cudaSmTag(props.major, props.minor);
}

} // namespace spect::cuda
