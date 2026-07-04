#include "app/OSEMReconstructor.h"

#include "app/SubsetUtils.h"
#include "cuda/ProjDomainChunked.h"
#include "filters/Convolution.h"
#include "interfaces/ISparseOps.h"

#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>

namespace spect {

namespace {
using StageClock = std::chrono::steady_clock;

struct OSEMStageTimingState {
  double inputPSFSeconds = 0.0;
  double forwardSeconds = 0.0;
  double backprojectionSeconds = 0.0;
  double cudaFusedSeconds = 0.0;
  double bufferPSFSeconds = 0.0;
  double updateSeconds = 0.0;
  size_t calls = 0;
  size_t cudaFusedCalls = 0;
};

OSEMStageTimingState g_osemStageTiming;

} // namespace

void OSEM_step(const CSR &A, const std::vector<int> &subset_rows,
               const std::vector<double> &p, std::vector<double> &f,
               const std::vector<double> &denom, int Nx, int Ny, int Nz,
               const ReconOpts &opts, const PSFKernels *psfKernels,
               const std::vector<double> *gradR, double beta,
               std::vector<double> *numer_out, double bsrem_alpha) {
  const int threads = opts.threads;
  const int M = A.rows * opts.bedCount;
  const int N = static_cast<int>(f.size());
  if (static_cast<int>(p.size()) != M)
    throw std::runtime_error("p size mismatch");
  if (!opts.bedOffsetVoxels.empty()) {
    const long long minSize = opts.bedOffsetVoxels.back() + A.cols;
    if (static_cast<long long>(f.size()) < minSize)
      throw std::runtime_error("f size too small for offsets");
  } else if (static_cast<int>(f.size()) != A.cols * opts.bedCount) {
    throw std::runtime_error("f size mismatch (virtual stacking)");
  }
  if (subset_rows.empty())
    return;

  const double eps = 1e-12;

  std::vector<double> f_blur;
  const std::vector<double> *f_used = &f;
  if (needs_psf_blur(opts, psfKernels)) {
    apply_psf_forward(f, f_blur, Nx, Ny, Nz, opts, psfKernels, threads);
    f_used = &f_blur;
  }

  std::vector<double> yhat(subset_rows.size(), 0.0);
  runSubsetForwardKernel(A, subset_rows, f_used->data(), opts, threads, eps,
                         yhat, true);

  std::vector<double> numer(static_cast<size_t>(N), 0.0);
  runSubsetBackprojectKernel<true, false>(
      A, subset_rows, p, yhat, opts, threads, numer.data(),
      "OSEM_step: bed index out of range");

  apply_psf_backward(numer, Nx, Ny, Nz, opts, psfKernels, threads);

#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(N); ++j) {
    if (opts.mask && !(*opts.mask)[static_cast<size_t>(j)]) {
      f[static_cast<size_t>(j)] = 0.0;
      continue;
    }
    double denom_eff = denom[static_cast<size_t>(j)];
    if (!(denom_eff > 0.0)) {
      f[static_cast<size_t>(j)] = 0.0;
      continue;
    }
    if (gradR && beta > 0.0) {
      const double fj = std::max(f[static_cast<size_t>(j)], eps);
      denom_eff += beta * ((*gradR)[static_cast<size_t>(j)] / fj);
    }

    const double f_val = f[static_cast<size_t>(j)];
    double f_new = f_val * (numer[static_cast<size_t>(j)] / denom_eff);
    if (opts.w_img)
      f_new *= (*opts.w_img)[static_cast<size_t>(j)];
    if (bsrem_alpha != 1.0)
      f_new = (1.0 - bsrem_alpha) * f_val + bsrem_alpha * f_new;
    if (opts.nonneg && f_new < 0.0)
      f_new = 0.0;
    f[static_cast<size_t>(j)] = f_new;
  }

  if (opts.psf.mode == PSFMode::PostSmooth && psfKernels &&
      opts.do_post_smooth_every_iter) {
    apply_psf_postsmooth(f, Nx, Ny, Nz, opts, psfKernels, threads);
  }

  if (numer_out)
    numer_out->swap(numer);
}

void OSEM_step(const io::CSRView &A, const std::vector<int> &subset_rows,
               const std::vector<double> &p, std::vector<double> &f,
               const std::vector<double> &denom, int Nx, int Ny, int Nz,
               const ReconOpts &opts, const PSFKernels *psfKernels,
               const std::vector<double> *gradR, double beta,
               std::vector<double> *numer_out, double bsrem_alpha) {
  const int threads = opts.threads;
  const int M = A.rows * opts.bedCount;
  const int N = static_cast<int>(f.size());
  if (static_cast<int>(p.size()) != M)
    throw std::runtime_error("p size mismatch");
  if (!opts.bedOffsetVoxels.empty()) {
    const long long minSize = opts.bedOffsetVoxels.back() + A.cols;
    if (static_cast<long long>(f.size()) < minSize)
      throw std::runtime_error("f size too small for offsets");
  } else if (static_cast<int>(f.size()) != A.cols * opts.bedCount) {
    throw std::runtime_error("f size mismatch (virtual stacking)");
  }
  if (subset_rows.empty())
    return;

  const double eps = 1e-12;

  std::vector<double> f_blur;
  const std::vector<double> *f_used = &f;
  auto stageStart = StageClock::now();
  if (needs_psf_blur(opts, psfKernels)) {
    apply_psf_forward(f, f_blur, Nx, Ny, Nz, opts, psfKernels, threads);
    f_used = &f_blur;
  }
  g_osemStageTiming.inputPSFSeconds +=
      std::chrono::duration<double>(StageClock::now() - stageStart).count();

  std::vector<double> numer(static_cast<size_t>(N), 0.0);
  bool usedCudaFusedSubset = false;
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
  if (canUseCudaFusedFullSubsetOsem(A, subset_rows, opts)) {
    stageStart = StageClock::now();
    usedCudaFusedSubset = cuda::runFullSubsetEmNumerCUDA(
        A, p, f_used->data(), std::max(1, opts.bedCount),
        opts.bedOffsetVoxels.empty() ? nullptr : &opts.bedOffsetVoxels,
        opts.bedAttenFactors.empty() ? nullptr : &opts.bedAttenFactors,
        opts.num_angles,
        threads, numer.data());
    if (usedCudaFusedSubset) {
      g_osemStageTiming.cudaFusedSeconds +=
          std::chrono::duration<double>(StageClock::now() - stageStart).count();
      ++g_osemStageTiming.cudaFusedCalls;
    }
  }
#endif

  if (!usedCudaFusedSubset) {
    std::vector<double> yhat(subset_rows.size(), 0.0);
    stageStart = StageClock::now();
    runSubsetForwardKernel(A, subset_rows, f_used->data(), opts, threads, eps,
                           yhat, true);
    g_osemStageTiming.forwardSeconds +=
        std::chrono::duration<double>(StageClock::now() - stageStart).count();

    stageStart = StageClock::now();
    runSubsetBackprojectKernel<false, false>(
        A, subset_rows, p, yhat, opts, threads, numer.data(),
        "OSEM_step(CSRView): bed index out of range");
    g_osemStageTiming.backprojectionSeconds +=
        std::chrono::duration<double>(StageClock::now() - stageStart).count();
  }

  stageStart = StageClock::now();
  apply_psf_backward(numer, Nx, Ny, Nz, opts, psfKernels, threads);
  g_osemStageTiming.bufferPSFSeconds +=
      std::chrono::duration<double>(StageClock::now() - stageStart).count();

  stageStart = StageClock::now();
#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(N); ++j) {
    if (opts.mask && !(*opts.mask)[static_cast<size_t>(j)]) {
      f[static_cast<size_t>(j)] = 0.0;
      continue;
    }
    double denom_eff = denom[static_cast<size_t>(j)];
    if (!(denom_eff > 0.0)) {
      f[static_cast<size_t>(j)] = 0.0;
      continue;
    }
    if (gradR && beta > 0.0) {
      const double fj = std::max(f[static_cast<size_t>(j)], eps);
      denom_eff += beta * ((*gradR)[static_cast<size_t>(j)] / fj);
    }

    const double f_val = f[static_cast<size_t>(j)];
    double f_new = f_val * (numer[static_cast<size_t>(j)] / denom_eff);
    if (opts.w_img)
      f_new *= (*opts.w_img)[static_cast<size_t>(j)];
    if (bsrem_alpha != 1.0)
      f_new = (1.0 - bsrem_alpha) * f_val + bsrem_alpha * f_new;
    if (opts.nonneg && f_new < 0.0)
      f_new = 0.0;
    f[static_cast<size_t>(j)] = f_new;
  }

  if (opts.psf.mode == PSFMode::PostSmooth && psfKernels &&
      opts.do_post_smooth_every_iter) {
    apply_psf_postsmooth(f, Nx, Ny, Nz, opts, psfKernels, threads);
  }
  g_osemStageTiming.updateSeconds +=
      std::chrono::duration<double>(StageClock::now() - stageStart).count();
  ++g_osemStageTiming.calls;

  if (numer_out)
    numer_out->swap(numer);
}

void resetOSEMStageTiming() { g_osemStageTiming = {}; }

OSEMStageTimingSnapshot getOSEMStageTimingSnapshot() {
  OSEMStageTimingSnapshot snapshot;
  snapshot.inputPSFSeconds = g_osemStageTiming.inputPSFSeconds;
  snapshot.forwardSeconds = g_osemStageTiming.forwardSeconds;
  snapshot.backprojectionSeconds = g_osemStageTiming.backprojectionSeconds;
  snapshot.cudaFusedSeconds = g_osemStageTiming.cudaFusedSeconds;
  snapshot.bufferPSFSeconds = g_osemStageTiming.bufferPSFSeconds;
  snapshot.updateSeconds = g_osemStageTiming.updateSeconds;
  snapshot.calls = g_osemStageTiming.calls;
  snapshot.cudaFusedCalls = g_osemStageTiming.cudaFusedCalls;
  return snapshot;
}

} // namespace spect
