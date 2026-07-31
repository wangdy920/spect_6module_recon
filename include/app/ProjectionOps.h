// ProjectionOps.h - Forward and backward projection operations
#pragma once

#include "ReconConfig.h"
#include "core/AttenuationConstants.h"
#include <vector>

namespace spect {

/// Apply PSF in forward direction (image domain blurring)
void apply_psf_forward(
    const std::vector<double>& f,
    std::vector<double>& f_blur,
    int Nx, int Ny, int Nz,
    const ReconOpts& opts,
    const PSFKernels* psfKernels,
    int threads);

/// Apply PSF in backward direction (adjoint of forward PSF)
void apply_psf_backward(
    std::vector<double>& numer,
    int Nx, int Ny, int Nz,
    const ReconOpts& opts,
    const PSFKernels* psfKernels,
    int threads);

} // namespace spect
