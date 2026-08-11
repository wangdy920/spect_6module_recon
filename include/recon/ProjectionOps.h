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

/// Build the sieve save copy: per-bed local extract -> apply_psf_forward with
/// the same kernels/mirror boundary as the iteration path -> accumulate back
/// into a global save volume (honoring bedVirtualZFlip). The iteration state
/// `image` is never modified. Single-bed degrades to one convolution of a
/// copy of `image`.
std::vector<double> build_sieve_save_volume(
    const std::vector<double>& image,
    int Nx, int Ny, int baseNz,
    int bedCount,
    const std::vector<int>& bedZStarts,
    bool bedVirtualZFlip,
    const ReconOpts& opts,
    const PSFKernels* psfKernels,
    int threads);

} // namespace spect
