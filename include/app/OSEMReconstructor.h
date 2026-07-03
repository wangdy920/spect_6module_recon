// OSEMReconstructor.h - OSEM/MLEM reconstruction algorithm
#pragma once

#include "ReconConfig.h"
#include "ProjectionOps.h"
#include <vector>

namespace spect {

struct OSEMStageTimingSnapshot {
    double inputPSFSeconds = 0.0;
    double forwardSeconds = 0.0;
    double backprojectionSeconds = 0.0;
    double cudaFusedSeconds = 0.0;
    double bufferPSFSeconds = 0.0;
    double updateSeconds = 0.0;
    size_t calls = 0;
    size_t cudaFusedCalls = 0;
};

void resetOSEMStageTiming();
OSEMStageTimingSnapshot getOSEMStageTimingSnapshot();

/**
 * @brief Single OSEM/MLEM iteration step
 * 
 * Implements EM update: f *= (A^T * (p / (Ax))) / (A^T * 1)
 * 
 * Features:
 * - Multi-bed reconstruction with overlap support
 * - CTAC attenuation correction (bedAttenFactors)
 * - Image-domain PSF (forward/backward blur)
 * - Projection-domain or operator-domain PSF
 * - Optional regularization (via gradR, beta)
 * - BSREM relaxation (via bsrem_alpha)
 * - AVX2 optimized forward projection
 * 
 * @param A System matrix (CSR format)
 * @param subset_rows Subset of projection rows to use
 * @param p Measured projection data (M)
 * @param f Image estimate (N) - updated in-place
 * @param denom Normalization A^T*1 (N)
 * @param Nx, Ny, Nz Image dimensions for PSF
 * @param opts Reconstruction options
 * @param psfKernels Optional PSF kernels
 * @param gradR Optional regularization gradient (N)
 * @param beta Regularization strength
 * @param numer_out Optional output for numerator (debugging)
 * @param bsrem_alpha Relaxation parameter (1.0 = standard OSEM)
 */
void OSEM_step(
    const CSR& A,
    const std::vector<int>& subset_rows,
    const std::vector<double>& p,
    std::vector<double>& f,
    const std::vector<double>& denom,
    int Nx, int Ny, int Nz,
    const ReconOpts& opts,
    const PSFKernels* psfKernels = nullptr,
    const std::vector<double>* gradR = nullptr,
    double beta = 0.0,
    std::vector<double>* numer_out = nullptr,
    double bsrem_alpha = 1.0
);

/// CSRView version (avoids copying system matrix)
void OSEM_step(
    const io::CSRView& A,
    const std::vector<int>& subset_rows,
    const std::vector<double>& p,
    std::vector<double>& f,
    const std::vector<double>& denom,
    int Nx, int Ny, int Nz,
    const ReconOpts& opts,
    const PSFKernels* psfKernels = nullptr,
    const std::vector<double>* gradR = nullptr,
    double beta = 0.0,
    std::vector<double>* numer_out = nullptr,
    double bsrem_alpha = 1.0
);

} // namespace spect
