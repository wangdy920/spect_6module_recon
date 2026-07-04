#include "app/BSREMReconstructor.h"

#include "app/BsremUpdate.h"
#include "app/ProjectionOps.h"
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

bool canUseCudaFusedFullSubsetBsrem(const io::CSRView &A,
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
  // w_proj/w_img semantics are undefined for BSREM (physics review items
  // 11/15); the fused kernel ignores them, so gate it off like b_proj.
  if (opts.b_proj != nullptr || opts.w_proj != nullptr ||
      opts.w_img != nullptr) {
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
  const int baseRows = A.rows;
  const int baseCols = A.cols;
  const size_t NvoxLocal = static_cast<size_t>(baseCols);
  const size_t raysPerAngle = (opts.num_angles > 0) ? (static_cast<size_t>(baseRows) / opts.num_angles) : 0;

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
    if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
    SparseSubsetContext subsetCtx;
    subsetCtx.bedCount = std::max(1, opts.bedCount);
    subsetCtx.bedOffsetVoxels =
        opts.bedOffsetVoxels.empty() ? nullptr : &opts.bedOffsetVoxels;
    subsetCtx.bedAttenFactors =
        opts.bedAttenFactors.empty() ? nullptr : &opts.bedAttenFactors;
    subsetCtx.numAngles = opts.num_angles;

    if (opts.sparseOps) {
      opts.sparseOps->spmvSubset(A, f_used->data(), subset_rows, subsetCtx,
                                 yhat.data(), threads);
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(subset_rows.size()); ++i) {
        const int global_row = subset_rows[static_cast<size_t>(i)];
        double s = yhat[static_cast<size_t>(i)];
        if (opts.b_proj)
          s += (*opts.b_proj)[static_cast<size_t>(global_row)];
        yhat[static_cast<size_t>(i)] = (s > eps) ? s : eps;
      }
    } else {
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(subset_rows.size()); ++i) {
        const int global_row = subset_rows[static_cast<size_t>(i)];
        const int bed = (opts.bedCount > 1) ? (global_row / A.rows) : 0;
        const int row = (opts.bedCount > 1) ? (global_row % A.rows) : global_row;
        const size_t col_offset =
            !opts.bedOffsetVoxels.empty()
                ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                : static_cast<size_t>(bed) * static_cast<size_t>(A.cols);

        const uint16_t *attPtr = nullptr;
        const double invScale = 1.0 / kAttenUint16Scale;
        if (!opts.bedAttenFactors.empty()) {
          int angle = 0;
          if (raysPerAngle > 0)
            angle = row / static_cast<int>(raysPerAngle);
          if (angle >= opts.num_angles)
            angle = opts.num_angles - 1;
          attPtr = opts.bedAttenFactors[bed].data() +
                   static_cast<size_t>(angle) * NvoxLocal;
        }

        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
        size_t t = r0;
        double s = 0.0;
        __m256d v_sum = _mm256_setzero_pd();
        __m128i v_col_offset = _mm_set1_epi32(static_cast<int>(col_offset));
        for (; t + 4 <= r1; t += 4) {
          __m128 v_val_f = _mm_loadu_ps(&A.val[t]);
          if (attPtr) {
            const float a0 = static_cast<float>(attPtr[A.col_idx[t]]) *
                             static_cast<float>(invScale);
            const float a1 = static_cast<float>(attPtr[A.col_idx[t + 1]]) *
                             static_cast<float>(invScale);
            const float a2 = static_cast<float>(attPtr[A.col_idx[t + 2]]) *
                             static_cast<float>(invScale);
            const float a3 = static_cast<float>(attPtr[A.col_idx[t + 3]]) *
                             static_cast<float>(invScale);
            v_val_f = _mm_mul_ps(v_val_f, _mm_set_ps(a3, a2, a1, a0));
          }
          const __m256d v_val_d = _mm256_cvtps_pd(v_val_f);
          __m128i v_idx = _mm_loadu_si128(
              reinterpret_cast<const __m128i *>(&A.col_idx[t]));
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

        if (opts.b_proj)
          s += (*opts.b_proj)[static_cast<size_t>(global_row)];
        yhat[static_cast<size_t>(i)] = (s > eps) ? s : eps;
      }
    }
    } else {
#pragma omp parallel for schedule(static) num_threads(threads)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(subset_rows.size()); ++i) {
      const int global_row = subset_rows[static_cast<size_t>(i)];
      const int bed = (opts.bedCount > 1) ? (global_row / A.rows) : 0;
      const int row = (opts.bedCount > 1) ? (global_row % A.rows) : global_row;
      const size_t col_offset =
          !opts.bedOffsetVoxels.empty()
              ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
              : static_cast<size_t>(bed) * static_cast<size_t>(A.cols);

      const uint16_t *attPtr = nullptr;
      const double invScale = 1.0 / kAttenUint16Scale;
      if (!opts.bedAttenFactors.empty()) {
        int angle = 0;
        if (raysPerAngle > 0)
          angle = row / static_cast<int>(raysPerAngle);
        if (angle >= opts.num_angles)
          angle = opts.num_angles - 1;
        attPtr = opts.bedAttenFactors[bed].data() +
                 static_cast<size_t>(angle) * NvoxLocal;
      }

      size_t t = A.row_ptr[static_cast<size_t>(row)];
      const size_t t_end = A.row_ptr[static_cast<size_t>(row) + 1];
      double s = 0.0;
      __m256d v_sum = _mm256_setzero_pd();
      __m128i v_col_offset = _mm_set1_epi32(static_cast<int>(col_offset));
      for (; t + 4 <= t_end; t += 4) {
        __m128 v_val_f = _mm_loadu_ps(&A.val[t]);
        if (attPtr) {
          const float a0 = static_cast<float>(attPtr[A.col_idx[t]]) *
                           static_cast<float>(invScale);
          const float a1 = static_cast<float>(attPtr[A.col_idx[t + 1]]) *
                           static_cast<float>(invScale);
          const float a2 = static_cast<float>(attPtr[A.col_idx[t + 2]]) *
                           static_cast<float>(invScale);
          const float a3 = static_cast<float>(attPtr[A.col_idx[t + 3]]) *
                           static_cast<float>(invScale);
          v_val_f = _mm_mul_ps(v_val_f, _mm_set_ps(a3, a2, a1, a0));
        }
        const __m256d v_val_d = _mm256_cvtps_pd(v_val_f);
        __m128i v_idx = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&A.col_idx[t]));
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

      if (opts.b_proj)
        s += (*opts.b_proj)[static_cast<size_t>(global_row)];
      yhat[static_cast<size_t>(i)] = (s > eps) ? s : eps;
    }
    }
    g_bsremStageTiming.forwardSeconds +=
        std::chrono::duration<double>(StageClock::now() - stageStart).count();

    if (buffer->size() != static_cast<size_t>(N))
      buffer->resize(static_cast<size_t>(N));
    std::fill(buffer->begin(), buffer->end(), 0.0);

    stageStart = StageClock::now();
    if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
    SparseSubsetContext subsetCtx;
    subsetCtx.bedCount = std::max(1, opts.bedCount);
    subsetCtx.bedOffsetVoxels =
        opts.bedOffsetVoxels.empty() ? nullptr : &opts.bedOffsetVoxels;
    subsetCtx.bedAttenFactors =
        opts.bedAttenFactors.empty() ? nullptr : &opts.bedAttenFactors;
    subsetCtx.numAngles = opts.num_angles;

    if (opts.sparseOps) {
      std::vector<double> ratio(subset_rows.size(), 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(subset_rows.size()); ++i) {
        const int global_row = subset_rows[static_cast<size_t>(i)];
        ratio[static_cast<size_t>(i)] =
            p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(i)];
      }
      opts.sparseOps->spmvTSubset(A, ratio.data(), subset_rows, subsetCtx,
                                  buffer->data(), threads);
    } else if (!should_use_threadlocal_accum(NvoxLocal, threads)) {
#pragma omp parallel for schedule(dynamic) num_threads(threads)
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(subset_rows.size()); ++i) {
        const int global_row = subset_rows[static_cast<size_t>(i)];
        const int bed = (opts.bedCount > 1) ? (global_row / A.rows) : 0;
        const int row = (opts.bedCount > 1) ? (global_row % A.rows) : global_row;
        const size_t col_offset =
            !opts.bedOffsetVoxels.empty()
                ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                : static_cast<size_t>(bed) * static_cast<size_t>(A.cols);

        const uint16_t *attPtr = nullptr;
        const double invScale = 1.0 / kAttenUint16Scale;
        if (!opts.bedAttenFactors.empty()) {
          int angle = 0;
          if (raysPerAngle > 0)
            angle = row / static_cast<int>(raysPerAngle);
          if (angle >= opts.num_angles)
            angle = opts.num_angles - 1;
          attPtr = opts.bedAttenFactors[bed].data() +
                   static_cast<size_t>(angle) * NvoxLocal;
        }

        const double r =
            p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(i)];
        if (std::abs(r) < 1e-9)
          continue;

        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
        for (size_t t = r0; t < r1; ++t) {
          double val = static_cast<double>(A.val[t]);
          if (attPtr)
            val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
#pragma omp atomic
          (*buffer)[static_cast<size_t>(A.col_idx[t]) + col_offset] += val * r;
        }
      }
    } else {
      const int bedCount = std::max(1, opts.bedCount);
      std::vector<std::vector<int>> bed_indices(static_cast<size_t>(bedCount));
      for (size_t i = 0; i < subset_rows.size(); ++i) {
        const int global_row = subset_rows[i];
        const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
        if (bed < 0 || bed >= bedCount)
          throw std::runtime_error("BSREM_step: bed index out of range");
        bed_indices[static_cast<size_t>(bed)].push_back(static_cast<int>(i));
      }

      const size_t threadCount = static_cast<size_t>(threads);
      std::vector<double> tls(threadCount * NvoxLocal, 0.0);
      for (int bed = 0; bed < bedCount; ++bed) {
        if (bed_indices[static_cast<size_t>(bed)].empty())
          continue;

        const size_t col_offset =
            !opts.bedOffsetVoxels.empty()
                ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                : static_cast<size_t>(bed) * static_cast<size_t>(A.cols);

#pragma omp parallel num_threads(threads)
        {
          const size_t tid = static_cast<size_t>(omp_get_thread_num());
          double *local = tls.data() + tid * NvoxLocal;
          std::fill(local, local + NvoxLocal, 0.0);

#pragma omp for schedule(dynamic)
          for (ptrdiff_t ii = 0;
               ii <
               static_cast<ptrdiff_t>(bed_indices[static_cast<size_t>(bed)].size());
               ++ii) {
            const int i =
                bed_indices[static_cast<size_t>(bed)][static_cast<size_t>(ii)];
            const int global_row = subset_rows[static_cast<size_t>(i)];
            const int row = (bedCount > 1) ? (global_row % baseRows) : global_row;
            const double r =
                p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(i)];
            if (std::abs(r) < 1e-9)
              continue;

            const uint16_t *attPtr = nullptr;
            const double invScale = 1.0 / kAttenUint16Scale;
            if (!opts.bedAttenFactors.empty()) {
              int angle = 0;
              if (raysPerAngle > 0)
                angle = row / static_cast<int>(raysPerAngle);
              if (angle >= opts.num_angles)
                angle = opts.num_angles - 1;
              attPtr = opts.bedAttenFactors[bed].data() +
                       static_cast<size_t>(angle) * NvoxLocal;
            }

            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            for (size_t t = r0; t < r1; ++t) {
              double val = static_cast<double>(A.val[t]);
              if (attPtr)
                val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
              local[static_cast<size_t>(A.col_idx[t])] += val * r;
            }
          }
        }

#pragma omp parallel for schedule(static) num_threads(threads)
        for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(NvoxLocal); ++j) {
          double sum = 0.0;
          const size_t jj = static_cast<size_t>(j);
          for (size_t tid = 0; tid < threadCount; ++tid)
            sum += tls[tid * NvoxLocal + jj];
          (*buffer)[col_offset + jj] += sum;
        }
      }
    }
    } else if (!should_use_threadlocal_accum(NvoxLocal, threads)) {
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(subset_rows.size()); ++i) {
      const int global_row = subset_rows[static_cast<size_t>(i)];
      const int bed = (opts.bedCount > 1) ? (global_row / A.rows) : 0;
      const int row = (opts.bedCount > 1) ? (global_row % A.rows) : global_row;
      const size_t col_offset = !opts.bedOffsetVoxels.empty() ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                                                              : static_cast<size_t>(bed) * static_cast<size_t>(A.cols);

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

      const double r = p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(i)];
      if (std::abs(r) < 1e-9)
        continue;

      if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
        for (size_t t = r0; t < r1; ++t) {
          double val = static_cast<double>(A.val[t]);
          if (attPtr)
            val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
#pragma omp atomic
          (*buffer)[static_cast<size_t>(A.col_idx[t]) + col_offset] += val * r;
        }
      } else {
        for (size_t t = A.row_ptr[static_cast<size_t>(row)]; t < A.row_ptr[static_cast<size_t>(row) + 1]; ++t) {
          double val = static_cast<double>(A.val[t]);
          if (attPtr)
            val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
#pragma omp atomic
          (*buffer)[static_cast<size_t>(A.col_idx[t]) + col_offset] += val * r;
        }
      }
    }
    } else {
    const int bedCount = std::max(1, opts.bedCount);
    std::vector<std::vector<int>> bed_indices(static_cast<size_t>(bedCount));
    for (size_t i = 0; i < subset_rows.size(); ++i) {
      const int global_row = subset_rows[i];
      const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
      if (bed < 0 || bed >= bedCount)
        throw std::runtime_error("BSREM_step: bed index out of range");
      bed_indices[static_cast<size_t>(bed)].push_back(static_cast<int>(i));
    }

    const size_t threadCount = static_cast<size_t>(threads);
    std::vector<double> tls(threadCount * NvoxLocal, 0.0);
    for (int bed = 0; bed < bedCount; ++bed) {
      if (bed_indices[static_cast<size_t>(bed)].empty())
        continue;

      const size_t col_offset = !opts.bedOffsetVoxels.empty() ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                                                              : static_cast<size_t>(bed) * static_cast<size_t>(A.cols);

#pragma omp parallel num_threads(threads)
      {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        double *local = tls.data() + tid * NvoxLocal;
        std::fill(local, local + NvoxLocal, 0.0);

#pragma omp for schedule(dynamic)
        for (ptrdiff_t ii = 0; ii < static_cast<ptrdiff_t>(bed_indices[static_cast<size_t>(bed)].size()); ++ii) {
          const int i = bed_indices[static_cast<size_t>(bed)][static_cast<size_t>(ii)];
          const int global_row = subset_rows[static_cast<size_t>(i)];
          const int row = (bedCount > 1) ? (global_row % baseRows) : global_row;
          const double r = p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(i)];
          if (std::abs(r) < 1e-9)
            continue;

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

          if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            for (size_t t = r0; t < r1; ++t) {
              double val = static_cast<double>(A.val[t]);
              if (attPtr)
                val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
              local[static_cast<size_t>(A.col_idx[t])] += val * r;
            }
          } else {
            for (size_t t = A.row_ptr[static_cast<size_t>(row)]; t < A.row_ptr[static_cast<size_t>(row) + 1]; ++t) {
              double val = static_cast<double>(A.val[t]);
              if (attPtr)
                val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
              local[static_cast<size_t>(A.col_idx[t])] += val * r;
            }
          }
        }
      }

#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(NvoxLocal); ++j) {
        double sum = 0.0;
        const size_t jj = static_cast<size_t>(j);
        for (size_t tid = 0; tid < threadCount; ++tid)
          sum += tls[tid * NvoxLocal + jj];
        (*buffer)[col_offset + jj] += sum;
      }
    }
    }
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
