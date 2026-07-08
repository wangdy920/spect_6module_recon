#pragma once

#include "cuda/CudaConfig.h"

#include <cstdint>
#include <vector>

namespace spect::cuda {

#if CUDA_ENABLED

bool computeAttenFactorsLineIntegralCUDA(const std::vector<double> &muMap,
                                         int nx, int ny, int nz, double dx,
                                         double dy, double dz, int numAngles,
                                         int rotationSign,
                                         std::vector<uint16_t> &factors);

#else

inline bool computeAttenFactorsLineIntegralCUDA(
    const std::vector<double> &, int, int, int, double, double, double, int, int,
    std::vector<uint16_t> &) {
  return false;
}

#endif

} // namespace spect::cuda
