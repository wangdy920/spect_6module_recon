#include "app/BSREMReconstructor.h"

#include "app/BsremUpdate.h"
#include "app/ProjectionOps.h"
#include "app/SubsetUtils.h"
#include "cuda/ProjDomainChunked.h"
#include "interfaces/ISparseOps.h"

#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <type_traits>

namespace spect {

namespace {
using StageClock = std::chrono::steady_clock;

struct BSREMStageTimingState {
  double inputPSFSeconds = 0.0;
  double forwardSeconds = 0.0;
  double backprojectionSeconds = 0.0;
  double cudaFusedSeconds = 0.0;
  double bufferPSFSeconds = 0.0;
  double updateSeconds = 0.0;
  size_t calls = 0;
  size_t cudaFusedCalls = 0;
};

BSREMStageTimingState g_bsremStageTiming;

template <typename MatrixType>
void BSREM_step_template(const MatrixType &A, const std::vector<int> &subset_rows,
                         const std::vector<double> &p, std::vector<double> &f,
                         const std::vector<double> &global_denom,
                         const std::vector<double> &subset_denom, int N,
                         int Nx, int Ny, int Nz, const ReconOpts &opts,
                         const filters::PSFKernels *psf,
                         const std::vector<double> *gradPtr,
                         double beta_scaled,
                         const std::vector<double> *curvPtr,
                         const std::vector<double> *kappaPtr,
                         std::vector<double> *buffer, double alpha) {
  if (static_cast<int>(p.size()) != A.rows * opts.bedCount)
    throw std::runtime_error("BSREM_step: p size mismatch");
  if (static_cast<int>(f.size()) != N)
    throw std::runtime_error("BSREM_step: f size mismatch");
  if (static_cast<int>(global_denom.size()) != N)
    throw std::runtime_error("BSREM_step: global_denom size mismatch");
  if (static_cast<int>(subset_denom.size()) != N)
    throw std::runtime_error("BSREM_step: subset_denom size mismatch");
  if (buffer == nullptr)
    throw std::runtime_error("BSREM_step: buffer is null");
  if (kappaPtr && static_cast<int>(kappaPtr->size()) != N)
    throw std::runtime_error("BSREM_step: kappa size mismatch");
  if (curvPtr && static_cast<int>(curvPtr->size()) != N)
    throw std::runtime_error("BSREM_step: curvature size mismatch");
  if (opts.w_proj != nullptr || opts.w_img != nullptr)
    throw std::runtime_error(
        "BSREM_step: w_proj/w_img are not supported by BSREM (their "
        "multiplicative semantics are undefined); remove them or use OSEM.");
  if (!opts.bedOffsetVoxels.empty()) {
    const long long minSize = opts.bedOffsetVoxels.back() + A.cols;
    if (static_cast<long long>(f.size()) < minSize)
      throw std::runtime_error("BSREM_step: f size too small for offsets");
  }
  if (subset_rows.empty())
    return;

  const double eps = 1e-12;
  const int threads = opts.threads;

  std::vector<double> f_blur;
  const std::vector<double> *f_used = &f;
  auto stageStart = StageClock::now();
  if (needs_psf_blur(opts, psf)) {
    apply_psf_forward(f, f_blur, Nx, Ny, Nz, opts, psf, threads);
    f_used = &f_blur;
  }
  g_bsremStageTiming.inputPSFSeconds +=
      std::chrono::duration<double>(StageClock::now() - stageStart).count();

  std::vector<double> yhat(subset_rows.size(), 0.0);
  bool usedCudaFusedSubset = false;
  if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
    if (canUseCudaFusedFullSubsetBsrem(A, subset_rows, opts)) {
      if (buffer->size() != static_cast<size_t>(N))
        buffer->resize(static_cast<size_t>(N));
      std::fill(buffer->begin(), buffer->end(), 0.0);
      stageStart = StageClock::now();
      usedCudaFusedSubset = cuda::runFullSubsetBsremCUDA(
          A, p, f_used->data(), std::max(1, opts.bedCount),
          opts.bedOffsetVoxels.empty() ? nullptr : &opts.bedOffsetVoxels,
          opts.bedAttenFactors.empty() ? nullptr : &opts.bedAttenFactors,
          opts.num_angles,
          threads, buffer->data());
      if (usedCudaFusedSubset) {
        g_bsremStageTiming.cudaFusedSeconds +=
            std::chrono::duration<double>(StageClock::now() - stageStart)
                .count();
        ++g_bsremStageTiming.cudaFusedCalls;
      }
    }
#endif
  }

  if (!usedCudaFusedSubset) {
    stageStart = StageClock::now();
    runSubsetForwardKernel(A, subset_rows, f_used->data(), opts, threads, eps,
                           yhat, false);
    g_bsremStageTiming.forwardSeconds +=
        std::chrono::duration<double>(StageClock::now() - stageStart).count();

    if (buffer->size() != static_cast<size_t>(N))
      buffer->resize(static_cast<size_t>(N));
    std::fill(buffer->begin(), buffer->end(), 0.0);

    stageStart = StageClock::now();
    runSubsetBackprojectKernel<false, true>(
        A, subset_rows, p, yhat, opts, threads, buffer->data(),
        "BSREM_step: bed index out of range");
    g_bsremStageTiming.backprojectionSeconds +=
        std::chrono::duration<double>(StageClock::now() - stageStart).count();
  }

  stageStart = StageClock::now();
  apply_psf_backward(*buffer, Nx, Ny, Nz, opts, psf, threads);
  g_bsremStageTiming.bufferPSFSeconds +=
      std::chrono::duration<double>(StageClock::now() - stageStart).count();

  stageStart = StageClock::now();
  BsremUpdateParams updateParams;
  updateParams.mask = opts.mask;
  updateParams.gradient = gradPtr;
  updateParams.curvature = curvPtr;
  updateParams.kappa = kappaPtr;
  updateParams.betaScaled = beta_scaled;
  updateParams.alpha = alpha;
  updateParams.nonneg = opts.nonneg;
  updateParams.threads = threads;
  applyBsremUpdate(f, *buffer, global_denom, subset_denom, updateParams);

  if (opts.psf.mode == PSFMode::PostSmooth && psf &&
      opts.do_post_smooth_every_iter) {
    apply_psf_postsmooth(f, Nx, Ny, Nz, opts, psf, threads);
  }
  g_bsremStageTiming.updateSeconds +=
      std::chrono::duration<double>(StageClock::now() - stageStart).count();
  ++g_bsremStageTiming.calls;
}

} // namespace

void resetBSREMStageTiming() { g_bsremStageTiming = {}; }

BSREMStageTimingSnapshot getBSREMStageTimingSnapshot() {
  BSREMStageTimingSnapshot snapshot;
  snapshot.inputPSFSeconds = g_bsremStageTiming.inputPSFSeconds;
  snapshot.forwardSeconds = g_bsremStageTiming.forwardSeconds;
  snapshot.backprojectionSeconds = g_bsremStageTiming.backprojectionSeconds;
  snapshot.cudaFusedSeconds = g_bsremStageTiming.cudaFusedSeconds;
  snapshot.bufferPSFSeconds = g_bsremStageTiming.bufferPSFSeconds;
  snapshot.updateSeconds = g_bsremStageTiming.updateSeconds;
  snapshot.calls = g_bsremStageTiming.calls;
  snapshot.cudaFusedCalls = g_bsremStageTiming.cudaFusedCalls;
  return snapshot;
}

void BSREM_step(const CSR &A, const std::vector<int> &subset_rows,
                const std::vector<double> &p, std::vector<double> &f,
                const std::vector<double> &global_denom,
                const std::vector<double> &subset_denom, int N, int Nx, int Ny,
                int Nz, const ReconOpts &opts, const filters::PSFKernels *psf,
                const std::vector<double> *gradPtr, double beta_scaled,
                const std::vector<double> *curvPtr,
                const std::vector<double> *kappaPtr,
                std::vector<double> *buffer, double alpha) {
  BSREM_step_template(A, subset_rows, p, f, global_denom, subset_denom, N, Nx,
                      Ny, Nz, opts, psf, gradPtr, beta_scaled, curvPtr,
                      kappaPtr, buffer, alpha);
}

void BSREM_step(const io::CSRView &A, const std::vector<int> &subset_rows,
                const std::vector<double> &p, std::vector<double> &f,
                const std::vector<double> &global_denom,
                const std::vector<double> &subset_denom, int N, int Nx, int Ny,
                int Nz, const ReconOpts &opts, const filters::PSFKernels *psf,
                const std::vector<double> *gradPtr, double beta_scaled,
                const std::vector<double> *curvPtr,
                const std::vector<double> *kappaPtr,
                std::vector<double> *buffer, double alpha) {
  BSREM_step_template(A, subset_rows, p, f, global_denom, subset_denom, N, Nx,
                      Ny, Nz, opts, psf, gradPtr, beta_scaled, curvPtr,
                      kappaPtr, buffer, alpha);
}

} // namespace spect
