#pragma once

#include "app/IOptimizer.h"
#include "app/OSEMReconstructor.h"

namespace spect {

/// MLEM / penalized-MLEM optimizer.
/// Wraps the existing OSEM_step function without any behavioural change.
class MLEMOptimizer : public IOptimizer {
public:
  void step(const StepParams &p) override {
    if (p.matrixView) {
      OSEM_step(*p.matrixView, *p.subsetRows, *p.projections, *p.image,
                *p.denom, p.Nx, p.Ny, p.Nz, *p.opts, p.psfKernels, p.gradReg,
                p.beta, nullptr, 1.0);
    } else {
      OSEM_step(*p.matrixCSR, *p.subsetRows, *p.projections, *p.image,
                *p.denom, p.Nx, p.Ny, p.Nz, *p.opts, p.psfKernels, p.gradReg,
                p.beta, nullptr, 1.0);
    }
  }

  std::string name() const override { return "MLEM"; }
};

} // namespace spect
