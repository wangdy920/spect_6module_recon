#include "app/OSEMReconstructor.h"

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

bool isOrderedFullSubset(const std::vector<int> &subset_rows, int baseRows,
                         int bedCount) {
  const int64_t totalRows =
      static_cast<int64_t>(baseRows) * static_cast<int64_t>(bedCount);
  if (subset_rows.size() != static_cast<size_t>(totalRows)) {
    return false;
  }
  for (int64_t row = 0; row < totalRows; ++row) {
    if (subset_rows[static_cast<size_t>(row)] != row) {
      return false;
    }
  }
  return true;
}

bool canUseCudaFusedFullSubsetOsem(const io::CSRView &A,
                                   const std::vector<int> &subset_rows,
                                   const ReconOpts &opts) {
#if !defined(SPECT_ENABLE_CUDA_SPARSEOPS)
  (void)A;
  (void)subset_rows;
  (void)opts;
  return false;
#else
  if (!opts.sparseOps || opts.sparseOps->backend() != BackendType::CUDA) {
    return false;
  }
  if (opts.w_proj != nullptr || opts.b_proj != nullptr) {
    return false;
  }
  const int bedCount = std::max(1, opts.bedCount);
  if (!isOrderedFullSubset(subset_rows, A.rows, bedCount)) {
    return false;
  }
  if (!opts.bedOffsetVoxels.empty() &&
      opts.bedOffsetVoxels.size() != static_cast<size_t>(bedCount)) {
    return false;
  }
  const size_t outputSize =
      !opts.bedOffsetVoxels.empty()
          ? static_cast<size_t>(opts.bedOffsetVoxels.back()) +
                static_cast<size_t>(A.cols)
          : static_cast<size_t>(bedCount) * static_cast<size_t>(A.cols);
  if (outputSize == 0) {
    return false;
  }
  if (opts.bedAttenFactors.empty()) {
    return true;
  }
  if (opts.num_angles <= 0 ||
      opts.bedAttenFactors.size() != static_cast<size_t>(bedCount)) {
    return false;
  }
  const size_t expectedSize =
      static_cast<size_t>(opts.num_angles) * static_cast<size_t>(A.cols);
  for (const auto &atten : opts.bedAttenFactors) {
    if (atten.size() != expectedSize) {
      return false;
    }
  }
  return true;
#endif
}

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

  const int baseRows = A.rows;
  const int baseCols = A.cols;
  const double eps = 1e-12;
  const size_t NvoxLocal = static_cast<size_t>(baseCols);
  const size_t raysPerAngle = (opts.num_angles > 0) ? (baseRows / opts.num_angles) : 0;

  std::vector<double> f_blur;
  const std::vector<double> *f_used = &f;
  if (needs_psf_blur(opts, psfKernels)) {
    apply_psf_forward(f, f_blur, Nx, Ny, Nz, opts, psfKernels, threads);
    f_used = &f_blur;
  }

  std::vector<double> yhat(subset_rows.size(), 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
    const int global_row = subset_rows[static_cast<size_t>(idx)];
    const int bed = (opts.bedCount > 1) ? (global_row / baseRows) : 0;
    const int row = (opts.bedCount > 1) ? (global_row % baseRows) : global_row;

    const uint16_t *attPtr = nullptr;
    const double invScale = 1.0 / kAttenUint16Scale;
    if (!opts.bedAttenFactors.empty()) {
      int angle = 0;
      if (raysPerAngle > 0)
        angle = row / static_cast<int>(raysPerAngle);
      if (angle >= opts.num_angles)
        angle = opts.num_angles - 1;
      attPtr = opts.bedAttenFactors[bed].data() + static_cast<size_t>(angle) * NvoxLocal;
    }

    const size_t col_offset = !opts.bedOffsetVoxels.empty() ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                                                            : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

    double s = 0.0;
    size_t t = A.row_ptr[static_cast<size_t>(row)];
    const size_t t_end = A.row_ptr[static_cast<size_t>(row) + 1];
    __m256d v_sum = _mm256_setzero_pd();
    __m128i v_col_offset = _mm_set1_epi32(static_cast<int>(col_offset));

    for (; t + 4 <= t_end; t += 4) {
      __m128 v_val_f = _mm_loadu_ps(&A.val[t]);
      if (attPtr) {
        const float a0 = static_cast<float>(attPtr[A.col_idx[t]]) * static_cast<float>(invScale);
        const float a1 = static_cast<float>(attPtr[A.col_idx[t + 1]]) * static_cast<float>(invScale);
        const float a2 = static_cast<float>(attPtr[A.col_idx[t + 2]]) * static_cast<float>(invScale);
        const float a3 = static_cast<float>(attPtr[A.col_idx[t + 3]]) * static_cast<float>(invScale);
        v_val_f = _mm_mul_ps(v_val_f, _mm_set_ps(a3, a2, a1, a0));
      }

      __m256d v_val_d = _mm256_cvtps_pd(v_val_f);
      __m128i v_idx = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&A.col_idx[t]));
      if (opts.bedCount > 1)
        v_idx = _mm_add_epi32(v_idx, v_col_offset);
      const __m256d v_f = _mm256_i32gather_pd(f_used->data(), v_idx, 8);
      v_sum = _mm256_fmadd_pd(v_val_d, v_f, v_sum);
    }

    double buf[4];
    _mm256_storeu_pd(buf, v_sum);
    s = buf[0] + buf[1] + buf[2] + buf[3];
    for (; t < t_end; ++t) {
      double val = static_cast<double>(A.val[t]);
      if (attPtr)
        val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
      s += val * (*f_used)[static_cast<size_t>(A.col_idx[t]) + col_offset];
    }
    if (opts.w_proj)
      s *= (*opts.w_proj)[static_cast<size_t>(global_row)];
    if (opts.b_proj)
      s += (*opts.b_proj)[static_cast<size_t>(global_row)];
    yhat[static_cast<size_t>(idx)] = (s > eps) ? s : eps;
  }

  std::vector<double> numer(static_cast<size_t>(N), 0.0);
  const double invScale = 1.0 / kAttenUint16Scale;
  if (!should_use_threadlocal_accum(NvoxLocal, threads)) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
      const int global_row = subset_rows[static_cast<size_t>(idx)];
      const int bed = (opts.bedCount > 1) ? (global_row / baseRows) : 0;
      const int row = (opts.bedCount > 1) ? (global_row % baseRows) : global_row;

      const uint16_t *attPtr = nullptr;
      if (!opts.bedAttenFactors.empty()) {
        int angle = 0;
        if (raysPerAngle > 0)
          angle = row / static_cast<int>(raysPerAngle);
        if (angle >= opts.num_angles)
          angle = opts.num_angles - 1;
        attPtr = opts.bedAttenFactors[bed].data() + static_cast<size_t>(angle) * NvoxLocal;
      }

      const size_t col_offset = !opts.bedOffsetVoxels.empty() ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                                                              : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);
      const double ratio = p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(idx)];
      for (size_t t = A.row_ptr[static_cast<size_t>(row)]; t < A.row_ptr[static_cast<size_t>(row) + 1]; ++t) {
        double val = static_cast<double>(A.val[t]);
        if (attPtr)
          val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
        const double inc = ratio * val;
#pragma omp atomic
        numer[static_cast<size_t>(A.col_idx[t]) + col_offset] += inc;
      }
    }
  } else {
    const int bedCount = std::max(1, opts.bedCount);
    std::vector<std::vector<int>> bed_indices(static_cast<size_t>(bedCount));
    for (size_t i = 0; i < subset_rows.size(); ++i) {
      const int global_row = subset_rows[i];
      const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
      if (bed < 0 || bed >= bedCount)
        throw std::runtime_error("OSEM_step: bed index out of range");
      bed_indices[static_cast<size_t>(bed)].push_back(static_cast<int>(i));
    }

    const size_t threadCount = static_cast<size_t>(threads);
    std::vector<double> tls(threadCount * NvoxLocal, 0.0);
    for (int bed = 0; bed < bedCount; ++bed) {
      if (bed_indices[static_cast<size_t>(bed)].empty())
        continue;

      const size_t col_offset = !opts.bedOffsetVoxels.empty() ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                                                              : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

#pragma omp parallel num_threads(threads)
      {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        double *local = tls.data() + tid * NvoxLocal;
        std::fill(local, local + NvoxLocal, 0.0);

#pragma omp for schedule(dynamic)
        for (ptrdiff_t ii = 0; ii < static_cast<ptrdiff_t>(bed_indices[static_cast<size_t>(bed)].size()); ++ii) {
          const int idx = bed_indices[static_cast<size_t>(bed)][static_cast<size_t>(ii)];
          const int global_row = subset_rows[static_cast<size_t>(idx)];
          const int row = (bedCount > 1) ? (global_row % baseRows) : global_row;
          const double ratio = p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(idx)];

          const uint16_t *attPtr = nullptr;
          if (!opts.bedAttenFactors.empty()) {
            int angle = 0;
            if (raysPerAngle > 0)
              angle = row / static_cast<int>(raysPerAngle);
            if (angle >= opts.num_angles)
              angle = opts.num_angles - 1;
            attPtr = opts.bedAttenFactors[bed].data() + static_cast<size_t>(angle) * NvoxLocal;
          }

          for (size_t t = A.row_ptr[static_cast<size_t>(row)]; t < A.row_ptr[static_cast<size_t>(row) + 1]; ++t) {
            double val = static_cast<double>(A.val[t]);
            if (attPtr)
              val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
            local[static_cast<size_t>(A.col_idx[t])] += ratio * val;
          }
        }
      }

#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(NvoxLocal); ++j) {
        double sum = 0.0;
        const size_t jj = static_cast<size_t>(j);
        for (size_t tid = 0; tid < threadCount; ++tid)
          sum += tls[tid * NvoxLocal + jj];
        numer[col_offset + jj] += sum;
      }
    }
  }

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

  const int baseRows = A.rows;
  const int baseCols = A.cols;
  const double eps = 1e-12;
  const size_t NvoxLocal = static_cast<size_t>(baseCols);
  const size_t raysPerAngle = (opts.num_angles > 0) ? (baseRows / opts.num_angles) : 0;

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
    SparseSubsetContext subsetCtx;
    subsetCtx.bedCount = std::max(1, opts.bedCount);
    subsetCtx.bedOffsetVoxels =
        opts.bedOffsetVoxels.empty() ? nullptr : &opts.bedOffsetVoxels;
    subsetCtx.bedAttenFactors =
        opts.bedAttenFactors.empty() ? nullptr : &opts.bedAttenFactors;
    subsetCtx.numAngles = opts.num_angles;

    std::vector<double> yhat(subset_rows.size(), 0.0);
    stageStart = StageClock::now();
    if (opts.sparseOps) {
      opts.sparseOps->spmvSubset(A, f_used->data(), subset_rows, subsetCtx,
                                 yhat.data(), threads);
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(subset_rows.size());
           ++idx) {
        const int global_row = subset_rows[static_cast<size_t>(idx)];
        double s = yhat[static_cast<size_t>(idx)];
        if (opts.w_proj)
          s *= (*opts.w_proj)[static_cast<size_t>(global_row)];
        if (opts.b_proj)
          s += (*opts.b_proj)[static_cast<size_t>(global_row)];
        yhat[static_cast<size_t>(idx)] = (s > eps) ? s : eps;
      }
    } else {
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
        const int global_row = subset_rows[static_cast<size_t>(idx)];
        const int bed = (opts.bedCount > 1) ? (global_row / baseRows) : 0;
        const int row = (opts.bedCount > 1) ? (global_row % baseRows) : global_row;

        const uint16_t *attPtr = nullptr;
        const double invScale = 1.0 / kAttenUint16Scale;
        if (!opts.bedAttenFactors.empty()) {
          int angle = 0;
          if (raysPerAngle > 0)
            angle = row / static_cast<int>(raysPerAngle);
          if (angle >= opts.num_angles)
            angle = opts.num_angles - 1;
          attPtr = opts.bedAttenFactors[bed].data() + static_cast<size_t>(angle) * NvoxLocal;
        }

        const size_t col_offset = !opts.bedOffsetVoxels.empty() ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                                                                : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
        double s = 0.0;
        size_t t = r0;
        __m256d v_sum = _mm256_setzero_pd();
        __m128i v_col_offset = _mm_set1_epi32(static_cast<int>(col_offset));

        for (; t + 4 <= r1; t += 4) {
          __m128 v_val_f = _mm_loadu_ps(&A.val[t]);
          if (attPtr) {
            const float a0 = static_cast<float>(attPtr[A.col_idx[t]]) * static_cast<float>(invScale);
            const float a1 = static_cast<float>(attPtr[A.col_idx[t + 1]]) * static_cast<float>(invScale);
            const float a2 = static_cast<float>(attPtr[A.col_idx[t + 2]]) * static_cast<float>(invScale);
            const float a3 = static_cast<float>(attPtr[A.col_idx[t + 3]]) * static_cast<float>(invScale);
            v_val_f = _mm_mul_ps(v_val_f, _mm_set_ps(a3, a2, a1, a0));
          }

          __m256d v_val_d = _mm256_cvtps_pd(v_val_f);
          __m128i v_idx = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&A.col_idx[t]));
          if (opts.bedCount > 1)
            v_idx = _mm_add_epi32(v_idx, v_col_offset);
          const __m256d v_f = _mm256_i32gather_pd(f_used->data(), v_idx, 8);
          v_sum = _mm256_fmadd_pd(v_val_d, v_f, v_sum);
        }

        double buf[4];
        _mm256_storeu_pd(buf, v_sum);
        s = buf[0] + buf[1] + buf[2] + buf[3];
        for (; t < r1; ++t) {
          double val = static_cast<double>(A.val[t]);
          if (attPtr)
            val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
          s += val * (*f_used)[static_cast<size_t>(A.col_idx[t]) + col_offset];
        }
        if (opts.w_proj)
          s *= (*opts.w_proj)[static_cast<size_t>(global_row)];
        if (opts.b_proj)
          s += (*opts.b_proj)[static_cast<size_t>(global_row)];
        yhat[static_cast<size_t>(idx)] = (s > eps) ? s : eps;
      }
    }
    g_osemStageTiming.forwardSeconds +=
        std::chrono::duration<double>(StageClock::now() - stageStart).count();

    const double invScale = 1.0 / kAttenUint16Scale;
    stageStart = StageClock::now();
    if (opts.sparseOps) {
      std::vector<double> ratio(subset_rows.size(), 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(subset_rows.size());
           ++idx) {
        const int global_row = subset_rows[static_cast<size_t>(idx)];
        ratio[static_cast<size_t>(idx)] =
            p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(idx)];
      }
      opts.sparseOps->spmvTSubset(A, ratio.data(), subset_rows, subsetCtx,
                                  numer.data(), threads);
    } else if (!should_use_threadlocal_accum(NvoxLocal, threads)) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
        const int global_row = subset_rows[static_cast<size_t>(idx)];
        const int bed = (opts.bedCount > 1) ? (global_row / baseRows) : 0;
        const int row = (opts.bedCount > 1) ? (global_row % baseRows) : global_row;
        const size_t col_offset = !opts.bedOffsetVoxels.empty() ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                                                                : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

        const uint16_t *attPtr = nullptr;
        if (!opts.bedAttenFactors.empty()) {
          int angle = 0;
          if (raysPerAngle > 0)
            angle = row / static_cast<int>(raysPerAngle);
          if (angle >= opts.num_angles)
            angle = opts.num_angles - 1;
          attPtr = opts.bedAttenFactors[bed].data() + static_cast<size_t>(angle) * NvoxLocal;
        }

        const double ratio = p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(idx)];
        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
        for (size_t t = r0; t < r1; ++t) {
          double val = static_cast<double>(A.val[t]);
          if (attPtr)
            val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
          const double inc = val * ratio;
#pragma omp atomic
          numer[static_cast<size_t>(A.col_idx[t]) + col_offset] += inc;
        }
      }
    } else {
      const int bedCount = std::max(1, opts.bedCount);
      std::vector<std::vector<int>> bed_indices(static_cast<size_t>(bedCount));
      for (size_t i = 0; i < subset_rows.size(); ++i) {
        const int global_row = subset_rows[i];
        const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
        if (bed < 0 || bed >= bedCount)
          throw std::runtime_error("OSEM_step(CSRView): bed index out of range");
        bed_indices[static_cast<size_t>(bed)].push_back(static_cast<int>(i));
      }

      const size_t threadCount = static_cast<size_t>(threads);
      std::vector<double> tls(threadCount * NvoxLocal, 0.0);
      for (int bed = 0; bed < bedCount; ++bed) {
        if (bed_indices[static_cast<size_t>(bed)].empty())
          continue;

        const size_t col_offset = !opts.bedOffsetVoxels.empty() ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                                                                : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

#pragma omp parallel num_threads(threads)
        {
          const size_t tid = static_cast<size_t>(omp_get_thread_num());
          double *local = tls.data() + tid * NvoxLocal;
          std::fill(local, local + NvoxLocal, 0.0);

#pragma omp for schedule(dynamic)
          for (ptrdiff_t ii = 0; ii < static_cast<ptrdiff_t>(bed_indices[static_cast<size_t>(bed)].size()); ++ii) {
            const int idx = bed_indices[static_cast<size_t>(bed)][static_cast<size_t>(ii)];
            const int global_row = subset_rows[static_cast<size_t>(idx)];
            const int row = (bedCount > 1) ? (global_row % baseRows) : global_row;
            const double ratio = p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(idx)];

            const uint16_t *attPtr = nullptr;
            if (!opts.bedAttenFactors.empty()) {
              int angle = 0;
              if (raysPerAngle > 0)
                angle = row / static_cast<int>(raysPerAngle);
              if (angle >= opts.num_angles)
                angle = opts.num_angles - 1;
              attPtr = opts.bedAttenFactors[bed].data() + static_cast<size_t>(angle) * NvoxLocal;
            }

            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            for (size_t t = r0; t < r1; ++t) {
              double val = static_cast<double>(A.val[t]);
              if (attPtr)
                val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
              local[static_cast<size_t>(A.col_idx[t])] += val * ratio;
            }
          }
        }

#pragma omp parallel for schedule(static) num_threads(threads)
        for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(NvoxLocal); ++j) {
          double sum = 0.0;
          const size_t jj = static_cast<size_t>(j);
          for (size_t tid = 0; tid < threadCount; ++tid)
            sum += tls[tid * NvoxLocal + jj];
          numer[col_offset + jj] += sum;
        }
      }
    }
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
