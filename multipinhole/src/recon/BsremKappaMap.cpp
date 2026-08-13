#include "recon/BsremKappaMap.h"

#include <algorithm>
#include <cmath>

namespace spect::main {

std::vector<double> computeKappaMapStandard(
    const std::vector<double> &globalSensitivity, const cli::ReconArgs &args,
    const std::vector<uint8_t> *mask) {
  std::vector<double> kappa(globalSensitivity.size(), 1.0);
  if (!args.bsremUseKappaMap || globalSensitivity.empty())
    return kappa;

  // Exclude <= eps and non-finite voxels from the normalization mean. Feeding
  // 1/sqrt(eps) for dead voxels would inflate the mean and effectively disable
  // regularization inside the object.
  const double eps = std::max(args.bsremKappaEps, 1e-12);
  double sum = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < globalSensitivity.size(); ++i) {
    if (mask && !(*mask)[i])
      continue;
    const double value = globalSensitivity[i];
    if (value > eps && std::isfinite(value)) {
      kappa[i] = 1.0 / std::sqrt(value + eps);
      sum += kappa[i];
      ++count;
    }
  }
  const double mean = (count > 0) ? (sum / static_cast<double>(count)) : 1.0;
  for (size_t i = 0; i < kappa.size(); ++i) {
    double value = kappa[i] / mean;
    if (args.bsremSoftMask && mask && !(*mask)[i])
      value = std::max(value, 1.0 + args.bsremSoftMaskLambdaOut);
    value =
        std::clamp(value, args.bsremKappaClipMin, args.bsremKappaClipMax);
    kappa[i] = value;
  }

  return kappa;
}

} // namespace spect::main