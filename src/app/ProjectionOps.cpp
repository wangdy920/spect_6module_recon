#include "app/ProjectionOps.h"

#include "cuda/VolumeOpsCUDA.h"
#include "filters/Convolution.h"
#include "interfaces/ISparseOps.h"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

namespace spect {

namespace {

std::atomic<bool> g_cudaPsfDisabled{false};
std::atomic<unsigned long long> g_cudaPsfAnomalyCount{0};

bool preferCudaVolumeOps(const ReconOpts &opts) {
  return !g_cudaPsfDisabled.load(std::memory_order_relaxed) && opts.sparseOps &&
         opts.sparseOps->backend() == BackendType::CUDA &&
         cuda::isCudaVolumeOpsAvailable();
}

struct VolumeStats {
  size_t count = 0;
  size_t finite = 0;
  double maxAbs = 0.0;
  double l1 = 0.0;
};

VolumeStats computeVolumeStats(const std::vector<double> &volume) {
  VolumeStats stats;
  stats.count = volume.size();
  for (const double value : volume) {
    if (!std::isfinite(value))
      continue;
    ++stats.finite;
    const double absValue = std::abs(value);
    stats.maxAbs = std::max(stats.maxAbs, absValue);
    stats.l1 += absValue;
  }
  return stats;
}

bool hasFiniteNonZeroSignal(const VolumeStats &stats) {
  return stats.finite == stats.count && stats.maxAbs > 0.0;
}

bool isFiniteAllZero(const VolumeStats &stats) {
  return stats.count > 0 && stats.finite == stats.count && stats.maxAbs == 0.0;
}

void markCudaPsfAnomaly(const std::string &reason) {
  g_cudaPsfDisabled.store(true, std::memory_order_relaxed);
  const auto occurrence =
      g_cudaPsfAnomalyCount.fetch_add(1, std::memory_order_relaxed) + 1;
  std::cerr << "[WARN] CUDA PSF anomaly #" << occurrence << ": " << reason
            << "; restoring input and using CPU PSF for the remainder of this "
               "process.\n";
}

bool inspectCudaPsfOutput(const VolumeStats &input, const VolumeStats &output,
                          std::string &reason) {
  if (output.count == 0 || output.finite != output.count) {
    reason = "output is empty or non-finite";
    return false;
  }
  if (!hasFiniteNonZeroSignal(input)) {
    if (output.maxAbs != 0.0) {
      reason = "zero input produced nonzero output";
      return false;
    }
    return true;
  }
  if (isFiniteAllZero(output)) {
    reason = "finite all-zero output for nonzero input";
    return false;
  }
  const double maxRatio = output.maxAbs / input.maxAbs;
  const double energyRatio = output.l1 / input.l1;
  if (!(maxRatio >= 1e-8 && maxRatio <= 1e8)) {
    reason = "max ratio out of guard range (input=" +
             std::to_string(input.maxAbs) + ", output=" +
             std::to_string(output.maxAbs) + ", ratio=" +
             std::to_string(maxRatio) + ")";
    return false;
  }
  if (!(energyRatio >= 1e-6 && energyRatio <= 1e6)) {
    reason = "energy ratio out of guard range (input=" +
             std::to_string(input.l1) + ", output=" +
             std::to_string(output.l1) + ", ratio=" +
             std::to_string(energyRatio) + ")";
    return false;
  }
  return true;
}

void applyPsfInplace(std::vector<double> &volume, int Nx, int Ny, int Nz,
                     const ReconOpts &opts, const PSFKernels *psfKernels,
                     int threads, bool allowPostSmooth, bool adjoint) {
  const bool useCuda = preferCudaVolumeOps(opts);
  if (((opts.psf.mode == PSFMode::Operator) ||
       (allowPostSmooth && opts.psf.mode == PSFMode::PostSmooth)) &&
      psfKernels) {
    if (useCuda) {
      std::vector<double> original = volume;
      const VolumeStats inputStats = computeVolumeStats(original);
      bool applied = false;
      try {
        applied = adjoint
                      ? cuda::applySeparableConv3DAdjointInplaceCUDA(
                            volume, Nx, Ny, Nz, psfKernels->kx,
                            psfKernels->ky, psfKernels->kz)
                      : cuda::applySeparableConv3DInplaceCUDA(
                            volume, Nx, Ny, Nz, psfKernels->kx,
                            psfKernels->ky, psfKernels->kz);
      } catch (const std::exception &ex) {
        volume = std::move(original);
        markCudaPsfAnomaly(std::string("CUDA volume operation failed: ") +
                           ex.what());
      }
      if (applied) {
        const VolumeStats outputStats = computeVolumeStats(volume);
        std::string reason;
        if (inspectCudaPsfOutput(inputStats, outputStats, reason))
          return;
        volume = std::move(original);
        markCudaPsfAnomaly(reason);
      } else if (!g_cudaPsfDisabled.load(std::memory_order_relaxed)) {
        volume = std::move(original);
        markCudaPsfAnomaly("CUDA PSF operation reported unavailable");
      }
    }
    if (adjoint) {
      filters::conv3d_separable_adjoint_inplace(
          volume, Nx, Ny, Nz, psfKernels->kx, psfKernels->ky,
          psfKernels->kz, threads);
    } else {
      filters::conv3d_separable_inplace(volume, Nx, Ny, Nz, psfKernels->kx,
                                        psfKernels->ky, psfKernels->kz,
                                        threads);
    }
    return;
  }
}

} // namespace


std::vector<double> computeNormAT(const CSR &A, int threads) {
  threads = effective_threads(threads);
  const size_t nvox = static_cast<size_t>(A.cols);
  std::vector<double> denom(nvox, 0.0);

  if (!should_use_threadlocal_accum(nvox, threads)) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(A.rows); ++row) {
      for (size_t t = A.row_ptr[static_cast<size_t>(row)];
           t < A.row_ptr[static_cast<size_t>(row) + 1]; ++t) {
        const double v = static_cast<double>(A.val[t]);
#pragma omp atomic
        denom[static_cast<size_t>(A.col_idx[t])] += v;
      }
    }
    return denom;
  }

  const size_t threadCount = static_cast<size_t>(threads);
  std::vector<double> tls(threadCount * nvox, 0.0);

#pragma omp parallel num_threads(threads)
  {
    const size_t tid = static_cast<size_t>(omp_get_thread_num());
    double *local = tls.data() + tid * nvox;

#pragma omp for schedule(dynamic)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(A.rows); ++row) {
      for (size_t t = A.row_ptr[static_cast<size_t>(row)];
           t < A.row_ptr[static_cast<size_t>(row) + 1]; ++t) {
        local[static_cast<size_t>(A.col_idx[t])] += static_cast<double>(A.val[t]);
      }
    }
  }

#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(nvox); ++j) {
    double sum = 0.0;
    const size_t jj = static_cast<size_t>(j);
    for (size_t tid = 0; tid < threadCount; ++tid)
      sum += tls[tid * nvox + jj];
    denom[jj] = sum;
  }
  return denom;
}

std::vector<double> computeNormAT(const io::CSRView &A, int threads) {
  threads = effective_threads(threads);
  const size_t nvox = static_cast<size_t>(A.cols);
  std::vector<double> denom(nvox, 0.0);

  if (!should_use_threadlocal_accum(nvox, threads)) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(A.rows); ++row) {
      const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
      const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
      for (size_t t = r0; t < r1; ++t) {
        const double v = static_cast<double>(A.val[t]);
#pragma omp atomic
        denom[static_cast<size_t>(A.col_idx[t])] += v;
      }
    }
    return denom;
  }

  const size_t threadCount = static_cast<size_t>(threads);
  std::vector<double> tls(threadCount * nvox, 0.0);

#pragma omp parallel num_threads(threads)
  {
    const size_t tid = static_cast<size_t>(omp_get_thread_num());
    double *local = tls.data() + tid * nvox;

#pragma omp for schedule(dynamic)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(A.rows); ++row) {
      const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
      const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
      for (size_t t = r0; t < r1; ++t) {
        local[static_cast<size_t>(A.col_idx[t])] += static_cast<double>(A.val[t]);
      }
    }
  }

#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(nvox); ++j) {
    double sum = 0.0;
    const size_t jj = static_cast<size_t>(j);
    for (size_t tid = 0; tid < threadCount; ++tid)
      sum += tls[tid * nvox + jj];
    denom[jj] = sum;
  }
  return denom;
}

void apply_psf_forward(const std::vector<double> &f, std::vector<double> &f_blur,
                       int Nx, int Ny, int Nz, const ReconOpts &opts,
                       const PSFKernels *psfKernels, int threads) {
  f_blur = f;
  applyPsfInplace(f_blur, Nx, Ny, Nz, opts, psfKernels, threads, false,
                  false);
}

void apply_psf_backward(std::vector<double> &numer, int Nx, int Ny, int Nz,
                        const ReconOpts &opts, const PSFKernels *psfKernels,
                        int threads) {
  applyPsfInplace(numer, Nx, Ny, Nz, opts, psfKernels, threads, false, true);
}

void apply_psf_postsmooth(std::vector<double> &volume, int Nx, int Ny, int Nz,
                          const ReconOpts &opts, const PSFKernels *psfKernels,
                          int threads) {
  applyPsfInplace(volume, Nx, Ny, Nz, opts, psfKernels, threads, true, false);
}

} // namespace spect
