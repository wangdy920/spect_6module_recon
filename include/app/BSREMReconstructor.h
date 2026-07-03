#pragma once

#include "ReconConfig.h"
#include "io/CSRBuilder.h"
#include <vector>

namespace spect {
struct BSREMStageTimingSnapshot {
    double inputPSFSeconds = 0.0;
    double forwardSeconds = 0.0;
    double backprojectionSeconds = 0.0;
    double cudaFusedSeconds = 0.0;
    double bufferPSFSeconds = 0.0;
    double updateSeconds = 0.0;
    size_t calls = 0;
    size_t cudaFusedCalls = 0;
};

void resetBSREMStageTiming();
BSREMStageTimingSnapshot getBSREMStageTimingSnapshot();

/**
 * @brief BSREM (Block Sequential Regularized Expectation Maximization) Reconstruction Algorithm
 * 
 * BSREM is a regularized variant of OSEM that incorporates penalty terms directly
 * into the EM update formula. The algorithm performs subset-based iterations with
 * additive gradient updates.
 * 
 * Update Formula:
 *   f_new = f + alpha * (f / S_global) * (A^T * (y / (A*f)) - S_subset - beta * gradR)
 * 
 * Where:
 *   - f: Current image estimate
 *   - alpha: Relaxation parameter (typically 1.0)
 *   - S_global: Global sensitivity (sum of all subset sensitivities)
 *   - S_subset: Current subset sensitivity
 *   - y: Measured projection data
 *   - A: System matrix
 *   - beta: Regularization strength
 *   - gradR: Gradient of regularization penalty
 * 
 * Key Features:
 *   - Direct regularization integration in the update formula
 *   - Relaxation parameter for convergence control
 *   - Subset-based iterations for acceleration
 *   - Support for various regularizers (TV, LogCosh, etc.)
 *   - Robust numerical stability with update limiters
 * 
 * Advantages over OSEM:
 *   - Better noise control through regularization
 *   - More stable convergence with high beta values
 *   - Flexible balance between data fidelity and smoothness
 * 
 * Typical Use Case:
 *   - Low-count imaging where noise suppression is critical
 *   - Applications requiring edge-preserving reconstruction
 *   - Multi-bed reconstruction with smoothness constraints
 */

/**
 * @brief Perform one BSREM iteration on a subset of projection data
 * 
 * This function implements the BSREM algorithm update for CSR sparse matrices.
 * It includes forward projection, ratio calculation, backprojection, and
 * regularized image update with relaxation.
 * 
 * @param A System matrix (CSR format)
 * @param subset_rows Indices of projection rows in current subset
 * @param p Measured projection data (full array)
 * @param f Image estimate (input/output, modified in-place)
 * @param global_denom Global sensitivity image (sum of all subsets: A^T * 1)
 * @param subset_denom Current subset sensitivity (A_subset^T * 1)
 * @param N Total number of voxels (Nx * Ny * Nz)
 * @param Nx Image width in voxels
 * @param Ny Image height in voxels
 * @param Nz Image depth in voxels
 * @param opts Reconstruction options (PSF, CTAC, threads, etc.)
 * @param psf PSF kernels for convolution (nullptr if not used)
 * @param gradPtr Pointer to regularization gradient vector (nullptr if not used)
 * @param beta_scaled Regularization strength scaled by number of subsets (beta / n_subsets)
 * @param curvPtr Optional regularization curvature surrogate map (non-negative, unscaled)
 * @param kappaPtr Optional spatial beta scaling map (kappa_j), defaults to 1 if null
 * @param buffer Temporary buffer for backprojection (size N)
 * @param alpha Relaxation parameter (0.0 to 1.0, typically 1.0)
 * 
 * @note Forward projection includes CTAC if opts.bedAttenFactors is provided
 * @note PSF can be applied in Operator or PostSmooth modes
 * @note Update includes safety limiters to prevent numerical explosions
 * @note Non-negativity constraint enforced on output image
 */
void BSREM_step(
    const CSR &A,
    const std::vector<int> &subset_rows,
    const std::vector<double> &p,
    std::vector<double> &f,
    const std::vector<double> &global_denom,
    const std::vector<double> &subset_denom,
    int N, int Nx, int Ny, int Nz,
    const ReconOpts &opts,
    const filters::PSFKernels *psf,
    const std::vector<double> *gradPtr,
    double beta_scaled,
    const std::vector<double> *curvPtr,
    const std::vector<double> *kappaPtr,
    std::vector<double> *buffer,
    double alpha
);

/**
 * @brief BSREM step with CSRView (non-owning view)
 * 
 * Identical to the CSR version but accepts a CSRView for zero-copy scenarios.
 * See BSREM_step(CSR) for full documentation.
 */
void BSREM_step(
    const io::CSRView &A,
    const std::vector<int> &subset_rows,
    const std::vector<double> &p,
    std::vector<double> &f,
    const std::vector<double> &global_denom,
    const std::vector<double> &subset_denom,
    int N, int Nx, int Ny, int Nz,
    const ReconOpts &opts,
    const filters::PSFKernels *psf,
    const std::vector<double> *gradPtr,
    double beta_scaled,
    const std::vector<double> *curvPtr,
    const std::vector<double> *kappaPtr,
    std::vector<double> *buffer,
    double alpha
);

} // namespace spect
