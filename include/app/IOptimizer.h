#pragma once

/// @file IOptimizer.h
/// @brief Unified optimizer interface for iterative reconstruction algorithms.
///
/// Each optimizer wraps a single per-subset step: numerator computation
/// (forward project → ratio → backproject → PSF) plus algorithm-specific
/// image update.  MainProgram calls optimizer->step() once per subset and
/// optimizer->postIteration() after every iteration.

#include "app/ReconConfig.h"
#include "filters/PSFConfig.h"
#include "io/CSRTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace spect {

struct ReconOpts;

/// Parameters passed to IOptimizer::step() each subset.
/// Common fields are always filled.  Algorithm-specific fields carry
/// sensible defaults so that algorithms that ignore them don't need
/// special handling.
struct StepParams {
  // -- Matrix: exactly one is set ---------------------------
  const io::CSR *matrixCSR = nullptr;
  const io::CSRView *matrixView = nullptr;

  // -- Common (always set) ----------------------------------
  const std::vector<int> *subsetRows = nullptr;
  const std::vector<double> *projections = nullptr;
  std::vector<double> *image = nullptr;
  const std::vector<double> *denom = nullptr;
  int Nx = 0;
  int Ny = 0;
  int Nz = 0;
  const ReconOpts *opts = nullptr;
  const filters::PSFKernels *psfKernels = nullptr;
  const std::vector<double> *gradReg = nullptr;
  double beta = 0.0; ///< Regularization strength (scaled per algorithm)

  // -- BSREM-specific (default values are harmless for MLEM) --
  const std::vector<double> *subsetDenom = nullptr;
  double bsremAlpha = 1.0;
  double bsremBetaScaled = 0.0;
  const std::vector<double> *curvature = nullptr;
  const std::vector<double> *kappa = nullptr;
  std::vector<double> *bsremBuffer = nullptr;
};

/// Abstract optimizer: one per algorithm flavour.
class IOptimizer {
public:
  virtual ~IOptimizer() = default;

  /// Execute a single-subset step (numerator + update).
  virtual void step(const StepParams &p) = 0;

  /// Called after all subsets in an iteration (e.g. ADMM z/u update).
  virtual void postIteration(std::vector<double> & /*image*/,
                             int /*iteration*/) {}

  /// Human-readable algorithm name.
  virtual std::string name() const = 0;
};

} // namespace spect
