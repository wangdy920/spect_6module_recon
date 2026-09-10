#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spect {

/// Host MLEM multiplicative update used by the compact non-resident path.
/// If s_j <= 0 the voxel is zeroed; otherwise x <- x * n / max(s, eps).
/// `globalDenom` (the full-data S, optional) carries the ordered-subset skip
/// rule: a voxel with S_m <= 0 but S > 0 is invisible to this subset only and
/// keeps its value instead of being zeroed, while S <= 0 still zeroes.
void applyMlemUpdate(std::vector<double> &image,
                     const std::vector<double> &numerator,
                     const std::vector<double> &denom,
                     const std::vector<uint8_t> *mask, bool nonneg,
                     int threads,
                     const std::vector<double> *globalDenom = nullptr);

struct BsremUpdateParams {
  const std::vector<uint8_t> *mask = nullptr;
  /// Raw ∇R(x), without beta or kappa.
  const std::vector<double> *gradient = nullptr;
  /// Raw D_reg(x), without beta or kappa.
  const std::vector<double> *curvature = nullptr;
  /// Per-voxel kappa; lambda_j = betaScaled * kappa_j. Null means kappa=1.
  const std::vector<double> *kappa = nullptr;
  /// Clipped relative step ū after alpha and clip (optional).
  std::vector<double> *updateFactors = nullptr;
  /// Unclipped relative step u = alpha * v (optional).
  std::vector<double> *rawRelativeSteps = nullptr;
  /// Preconditioned direction v without alpha (optional).
  std::vector<double> *preconditionedSteps = nullptr;
  /// User regularization strength beta. lambda_j = beta * kappa_j.
  double betaScaled = 0.0;
  double alpha = 1.0;
  /// M: subset count of the current stage; scales the subset data gradient
  /// n - s_m to full-data size (Ahn & Fessler 2003). 1 for a full-data step.
  double subsetScale = 1.0;
  /// Added to the BSREM denominator after lambda * max(D_reg,0).
  double denomEps = 1e-12;
  /// Voxels with s_j <= sMin are not updated with a tiny denominator.
  double sMin = 1e-12;
  bool clipRelativeStep = true;
  double clipMin = -0.9;
  double clipMax = 1.0;
  /// Degenerate to the host MLEM update: no clip, no penalty, s_eff=max(s,eps).
  bool mlemEquivalenceMode = false;
  bool zeroLowSensitivity = true;
  bool nonneg = true;
  int threads = 1;
};

/// BSREM-II relative update (M = subsetScale, s = global, s_m = subset):
///   lambda_j = beta * kappa_j
///   u_j = alpha * (M (n_j - s_m,j) - lambda_j r_j) / (s_j + lambda_j max(c_j,0) + eps)
///   ū_j = clip(u_j, clipMin, clipMax)
///   x <- x * (1 + ū)
void applyBsremUpdate(std::vector<double> &image,
                      const std::vector<double> &numerator,
                      const std::vector<double> &globalDenom,
                      const std::vector<double> &subsetDenom,
                      const BsremUpdateParams &params);

} // namespace spect
