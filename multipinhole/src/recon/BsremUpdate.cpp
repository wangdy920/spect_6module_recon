#include "recon/BsremUpdate.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace spect {

namespace {

void requireSize(const char *name, size_t actual, size_t expected) {
  if (actual != expected)
    throw std::runtime_error(std::string("applyBsremUpdate: ") + name +
                             " size mismatch");
}

void requireOptionalSize(const char *name, const std::vector<double> *vec,
                         size_t expected) {
  if (vec)
    requireSize(name, vec->size(), expected);
}

void zeroOptional(std::vector<double> *vec, size_t idx) {
  if (vec)
    (*vec)[idx] = 0.0;
}

} // namespace

void applyMlemUpdate(std::vector<double> &image,
                     const std::vector<double> &numerator,
                     const std::vector<double> &denom,
                     const std::vector<uint8_t> *mask, bool nonneg,
                     int threads) {
  const size_t nvox = image.size();
  requireSize("numerator", numerator.size(), nvox);
  requireSize("denom", denom.size(), nvox);
  if (mask)
    requireSize("mask", mask->size(), nvox);
  constexpr double eps = 1e-12;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads))
#endif
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(nvox); ++j) {
    const size_t idx = static_cast<size_t>(j);
    if (mask && !(*mask)[idx]) {
      image[idx] = 0.0;
      continue;
    }
    const double denomEff = denom[idx];
    if (!(denomEff > 0.0)) {
      image[idx] = 0.0;
      continue;
    }
    double value = image[idx] * (numerator[idx] / std::max(denomEff, eps));
    if (nonneg && value < 0.0)
      value = 0.0;
    image[idx] = value;
  }
}

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
  requireOptionalSize("gradient", params.gradient, nvox);
  requireOptionalSize("curvature", params.curvature, nvox);
  requireOptionalSize("kappa", params.kappa, nvox);
  if (params.updateFactors && params.updateFactors->size() != nvox)
    params.updateFactors->assign(nvox, 0.0);
  if (params.rawRelativeSteps && params.rawRelativeSteps->size() != nvox)
    params.rawRelativeSteps->assign(nvox, 0.0);
  if (params.preconditionedSteps &&
      params.preconditionedSteps->size() != nvox)
    params.preconditionedSteps->assign(nvox, 0.0);

  const double imageEps = 1e-12;
  const double mlemEps = 1e-12;
  const int threads = params.threads;
  const bool equiv = params.mlemEquivalenceMode;
  const double alpha = equiv ? 1.0 : params.alpha;
  const double beta = equiv ? 0.0 : params.betaScaled;
  const double denomEps = equiv ? 0.0 : params.denomEps;
  const bool clip = equiv ? false : params.clipRelativeStep;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads))
#endif
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(nvox); ++j) {
    const size_t idx = static_cast<size_t>(j);
    if (params.mask && !(*params.mask)[idx]) {
      image[idx] = 0.0;
      zeroOptional(params.updateFactors, idx);
      zeroOptional(params.rawRelativeSteps, idx);
      zeroOptional(params.preconditionedSteps, idx);
      continue;
    }

    const double sens_global = globalDenom[idx];
    if (equiv) {
      if (!(sens_global > 0.0)) {
        image[idx] = 0.0;
        zeroOptional(params.updateFactors, idx);
        zeroOptional(params.rawRelativeSteps, idx);
        zeroOptional(params.preconditionedSteps, idx);
        continue;
      }
    } else if (!(sens_global > params.sMin)) {
      if (params.zeroLowSensitivity)
        image[idx] = 0.0;
      zeroOptional(params.updateFactors, idx);
      zeroOptional(params.rawRelativeSteps, idx);
      zeroOptional(params.preconditionedSteps, idx);
      continue;
    }

    const double s_eff =
        equiv ? std::max(sens_global, mlemEps) : sens_global;
    const double s_sub =
        equiv ? s_eff : subsetDenom[idx];
    const double kappa = (!equiv && params.kappa) ? (*params.kappa)[idx] : 1.0;
    const double lambda = beta * kappa;
    const double r = (!equiv && params.gradient) ? (*params.gradient)[idx] : 0.0;
    double c = 0.0;
    if (!equiv && params.curvature)
      c = (*params.curvature)[idx];
    const double c_pos = std::max(c, 0.0);

    const double v_num = numerator[idx] - s_sub - lambda * r;
    const double v_den = s_eff + lambda * c_pos + denomEps;
    const double v = v_num / v_den;
    double u = alpha * v;
    if (params.preconditionedSteps)
      (*params.preconditionedSteps)[idx] = v;
    if (params.rawRelativeSteps)
      (*params.rawRelativeSteps)[idx] = u;

    if (clip) {
      if (u > params.clipMax)
        u = params.clipMax;
      if (u < params.clipMin)
        u = params.clipMin;
    }

    double f_curr = image[idx];
    if (f_curr < imageEps)
      f_curr = imageEps;
    double f_new = f_curr * (1.0 + u);
    if (f_new < imageEps)
      f_new = params.nonneg ? 0.0 : imageEps;
    image[idx] = f_new;
    if (params.updateFactors)
      (*params.updateFactors)[idx] = u;
  }
}

} // namespace spect
