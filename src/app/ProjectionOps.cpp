#include "app/ProjectionOps.h"

#include "cuda/VolumeOpsCUDA.h"
#include "filters/Convolution.h"
#include "interfaces/ISparseOps.h"

#include <cmath>
#include <iostream>
#include <utility>

namespace spect {

namespace {

bool preferCudaVolumeOps(const ReconOpts &opts) {
  return opts.sparseOps &&
         opts.sparseOps->backend() == BackendType::CUDA &&
         cuda::isCudaVolumeOpsAvailable();
}

bool hasFiniteNonZeroSignal(const std::vector<double> &volume) {
  for (const double value : volume) {
    if (std::isfinite(value) && value != 0.0)
      return true;
  }
  return false;
}

bool isFiniteAllZero(const std::vector<double> &volume) {
  if (volume.empty())
    return false;
  for (const double value : volume) {
    if (!std::isfinite(value) || value != 0.0)
      return false;
  }
  return true;
}

void applyPsfInplace(std::vector<double> &volume, int Nx, int Ny, int Nz,
                     const ReconOpts &opts, const PSFKernels *psfKernels,
                     int threads, bool allowPostSmooth) {
  const bool useCuda = preferCudaVolumeOps(opts);
  if (((opts.psf.mode == PSFMode::Operator) ||
       (allowPostSmooth && opts.psf.mode == PSFMode::PostSmooth)) &&
      psfKernels) {
    if (useCuda) {
      std::vector<double> original = volume;
      const bool inputHasSignal = hasFiniteNonZeroSignal(original);
      if (cuda::applySeparableConv3DInplaceCUDA(
              volume, Nx, Ny, Nz, psfKernels->kx, psfKernels->ky,
              psfKernels->kz)) {
        if (!(inputHasSignal && isFiniteAllZero(volume)))
          return;
        std::cerr << "[WARN] CUDA PSF returned finite all-zero output for "
                     "nonzero input; falling back to CPU separable convolution.\n";
        volume = std::move(original);
      } else {
        std::cerr << "[WARN] CUDA PSF unavailable or failed; falling back to "
                     "CPU separable convolution.\n";
      }
    }
    filters::conv3d_separable_inplace(volume, Nx, Ny, Nz, psfKernels->kx,
                                      psfKernels->ky, psfKernels->kz,
                                      threads);
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
  applyPsfInplace(f_blur, Nx, Ny, Nz, opts, psfKernels, threads, false);
}

void apply_psf_backward(std::vector<double> &numer, int Nx, int Ny, int Nz,
                        const ReconOpts &opts, const PSFKernels *psfKernels,
                        int threads) {
  applyPsfInplace(numer, Nx, Ny, Nz, opts, psfKernels, threads, false);
}

void apply_psf_postsmooth(std::vector<double> &volume, int Nx, int Ny, int Nz,
                          const ReconOpts &opts, const PSFKernels *psfKernels,
                          int threads) {
  applyPsfInplace(volume, Nx, Ny, Nz, opts, psfKernels, threads, true);
}

} // namespace spect
