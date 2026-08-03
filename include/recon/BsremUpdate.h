#pragma once

#include <cstdint>
#include <vector>

namespace spect {

struct BsremUpdateParams {
  const std::vector<uint8_t> *mask = nullptr;
  const std::vector<double> *gradient = nullptr;
  const std::vector<double> *curvature = nullptr;
  const std::vector<double> *kappa = nullptr;
  std::vector<double> *updateFactors = nullptr;
  double betaScaled = 0.0;
  double alpha = 1.0;
  bool nonneg = true;
  int threads = 1;
};

void applyBsremUpdate(std::vector<double> &image,
                      const std::vector<double> &numerator,
                      const std::vector<double> &globalDenom,
                      const std::vector<double> &subsetDenom,
                      const BsremUpdateParams &params);

} // namespace spect