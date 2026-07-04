#include "app/BsremUpdate.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace spect {

namespace {

void requireSize(const char *name, size_t actual, size_t expected) {
  if (actual != expected)
    throw std::runtime_error(std::string("applyBsremUpdate: ") + name +
                             " size mismatch");
}

} // namespace

void applyBsremUpdate(std::vector<double> &image,
                      const std::vector<double> &numerator,
                      const std::vector<double> &globalDenom,
                      const std::vector<double> &subsetDenom,
                      const BsremUpdateParams &params) {
  const size_t nvox = image.size();
  requireSize("numerator", numerator.size(), nvox);
  requireSize("global_denom", globalDenom.size(), nvox);
  requireSize("subset_denom", subsetDenom.size(), nvox);
  if (params.mask)
    requireSize("mask", params.mask->size(), nvox);
  if (params.gradient)
    requireSize("gradient", params.gradient->size(), nvox);
  if (params.curvature)
    requireSize("curvature", params.curvature->size(), nvox);
  if (params.kappa)
    requireSize("kappa", params.kappa->size(), nvox);
  if (params.updateFactors && params.updateFactors->size() != nvox)
    params.updateFactors->assign(nvox, 0.0);

  const double eps = 1e-12;
  constexpr double max_update_factor = 1.0;
  constexpr double min_update_factor = -0.9;
  const int threads = params.threads;

#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(nvox); ++j) {
    const size_t idx = static_cast<size_t>(j);
    if (params.mask && !(*params.mask)[idx]) {
      image[idx] = 0.0;
      if (params.updateFactors)
        (*params.updateFactors)[idx] = 0.0;
      continue;
    }

    const double sens_global = globalDenom[idx];
    if (sens_global <= eps) {
      image[idx] = 0.0;
      if (params.updateFactors)
        (*params.updateFactors)[idx] = 0.0;
      continue;
    }

    const double kappa = params.kappa ? (*params.kappa)[idx] : 1.0;
    const double term_likelihood = numerator[idx] - subsetDenom[idx];
    const double term_penalty = params.gradient
                                    ? params.betaScaled * kappa *
                                          (*params.gradient)[idx]
                                    : 0.0;
    const double gradient_diff = term_likelihood - term_penalty;

    double denom_eff = sens_global;
    if (params.curvature)
      denom_eff += params.betaScaled * std::max((*params.curvature)[idx], 0.0);

    double f_curr = image[idx];
    if (f_curr < eps)
      f_curr = eps;

    double update_factor = params.alpha * (gradient_diff / denom_eff);
    if (update_factor > max_update_factor)
      update_factor = max_update_factor;
    if (update_factor < min_update_factor)
      update_factor = min_update_factor;

    double f_new = f_curr * (1.0 + update_factor);
    if (f_new < eps)
      f_new = params.nonneg ? 0.0 : eps;
    image[idx] = f_new;
    if (params.updateFactors)
      (*params.updateFactors)[idx] = update_factor;
  }
}

} // namespace spect