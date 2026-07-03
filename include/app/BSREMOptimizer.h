#pragma once

#include "app/BSREMReconstructor.h"
#include "app/IOptimizer.h"

namespace spect {

/// Block Sequential Regularized EM optimizer.
/// Wraps the existing BSREM_step function without any behavioural change.
class BSREMOptimizer : public IOptimizer {
public:
  void step(const StepParams &p) override {
    if (p.matrixView) {
      BSREM_step(*p.matrixView, *p.subsetRows, *p.projections, *p.image,
                 *p.denom, *p.subsetDenom,
                 static_cast<int>(p.image->size()), p.Nx, p.Ny, p.Nz, *p.opts,
                 p.psfKernels, p.gradReg, p.bsremBetaScaled, p.curvature,
                 p.kappa, p.bsremBuffer, p.bsremAlpha);
    } else {
      BSREM_step(*p.matrixCSR, *p.subsetRows, *p.projections, *p.image,
                 *p.denom, *p.subsetDenom,
                 static_cast<int>(p.image->size()), p.Nx, p.Ny, p.Nz, *p.opts,
                 p.psfKernels, p.gradReg, p.bsremBetaScaled, p.curvature,
                 p.kappa, p.bsremBuffer, p.bsremAlpha);
    }
  }

  std::string name() const override { return "BSREM"; }
};

} // namespace spect
