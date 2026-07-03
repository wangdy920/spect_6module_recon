// ProjectionOps.h - Forward and backward projection operations
#pragma once

#include "ReconConfig.h"
#include "core/AttenuationConstants.h"
#include "io/CSRBuilder.h"
#include <vector>

namespace spect {

// Import CSR type for convenience (CSRView omitted to avoid conflict with core::CSRView)
using io::CSR;

/// Compute normalization A^T * 1 (column sums of system matrix)
std::vector<double> computeNormAT(const CSR& A, int threads);

/// Compute normalization A^T * 1 for CSRView
std::vector<double> computeNormAT(const io::CSRView& A, int threads);

/// Compute normalization with attenuation correction
template <typename MatrixType>
std::vector<double> computeNormAT_Atten(
    const MatrixType& A,
    const std::vector<uint16_t>& atten,
    int numAngles,
    int threads);

/// Compute subset sensitivity for OSEM-like algorithms
template <typename MatrixType>
std::vector<double> computeSubsetSensitivity(
    const MatrixType& A,
    const std::vector<int>& subset_rows,
    int N,
    int threads,
    const ReconOpts& opts);

/// Forward projection: Ax (template for CSR and CSRView)
template <typename MatrixType>
std::vector<double> forward_project(
    const MatrixType& A,
    const std::vector<double>& f,
    int threads);

/// Backprojection: A^T * y (template for CSR and CSRView)
template <typename MatrixType>
std::vector<double> backproject(
    const MatrixType& A,
    const std::vector<double>& y,
    int threads);

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

/// Apply PSF only for PostSmooth mode
void apply_psf_postsmooth(
    std::vector<double>& volume,
    int Nx, int Ny, int Nz,
    const ReconOpts& opts,
    const PSFKernels* psfKernels,
    int threads);

} // namespace spect

// Include template implementations
#include "ProjectionOps.inl"
