#pragma once

#include "app/ReconConfig.h"
#include "interfaces/ISparseOps.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spect {

namespace subset_detail {

template <typename MatrixType>
inline size_t rowBegin(const MatrixType &A, int row) {
  if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
    return static_cast<size_t>(A.row_ptr[row]);
  } else {
    return A.row_ptr[static_cast<size_t>(row)];
  }
}

template <typename MatrixType>
inline size_t rowEnd(const MatrixType &A, int row) {
  if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
    return static_cast<size_t>(A.row_ptr[row + 1]);
  } else {
    return A.row_ptr[static_cast<size_t>(row) + 1];
  }
}

inline size_t columnOffset(const ReconOpts &opts, int bed, int baseCols) {
  return !opts.bedOffsetVoxels.empty()
             ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
             : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);
}

inline const uint16_t *attenuationForRow(const ReconOpts &opts, int bed,
                                         int row, size_t raysPerAngle,
                                         size_t nvoxLocal) {
  if (opts.bedAttenFactors.empty()) {
    return nullptr;
  }
  int angle = 0;
  if (raysPerAngle > 0) {
    angle = row / static_cast<int>(raysPerAngle);
  }
  if (angle >= opts.num_angles) {
    angle = opts.num_angles - 1;
  }
  return opts.bedAttenFactors[bed].data() +
         static_cast<size_t>(angle) * nvoxLocal;
}

inline SparseSubsetContext makeSparseSubsetContext(const ReconOpts &opts) {
  SparseSubsetContext subsetCtx;
  subsetCtx.bedCount = std::max(1, opts.bedCount);
  subsetCtx.bedOffsetVoxels =
      opts.bedOffsetVoxels.empty() ? nullptr : &opts.bedOffsetVoxels;
  subsetCtx.bedAttenFactors =
      opts.bedAttenFactors.empty() ? nullptr : &opts.bedAttenFactors;
  subsetCtx.numAngles = opts.num_angles;
  return subsetCtx;
}

template <bool kRatioTimesValue>
inline double backprojectProduct(double ratio, double value) {
  if constexpr (kRatioTimesValue) {
    return ratio * value;
  } else {
    return value * ratio;
  }
}

} // namespace subset_detail

inline bool isOrderedFullSubset(const std::vector<int> &subset_rows,
                                int baseRows, int bedCount) {
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

inline bool canUseCudaFusedFullSubsetCommon(
    const io::CSRView &A, const std::vector<int> &subset_rows,
    const ReconOpts &opts, bool rejectProjectionWeights,
    bool rejectImageWeights) {
#if !defined(SPECT_ENABLE_CUDA_SPARSEOPS)
  (void)A;
  (void)subset_rows;
  (void)opts;
  (void)rejectProjectionWeights;
  (void)rejectImageWeights;
  return false;
#else
  if (!opts.sparseOps || opts.sparseOps->backend() != BackendType::CUDA) {
    return false;
  }
  if (opts.b_proj != nullptr ||
      (rejectProjectionWeights && opts.w_proj != nullptr) ||
      (rejectImageWeights && opts.w_img != nullptr)) {
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

inline bool canUseCudaFusedFullSubsetOsem(
    const io::CSRView &A, const std::vector<int> &subset_rows,
    const ReconOpts &opts) {
  return canUseCudaFusedFullSubsetCommon(A, subset_rows, opts, true, false);
}

inline bool canUseCudaFusedFullSubsetBsrem(
    const io::CSRView &A, const std::vector<int> &subset_rows,
    const ReconOpts &opts) {
  // w_proj/w_img semantics are undefined for BSREM (physics review items
  // 11/15); the fused kernel ignores them, so gate it off like b_proj.
  return canUseCudaFusedFullSubsetCommon(A, subset_rows, opts, true, true);
}

template <typename MatrixType>
void runSubsetForwardKernel(const MatrixType &A,
                            const std::vector<int> &subset_rows,
                            const double *image, const ReconOpts &opts,
                            int threads, double eps,
                            std::vector<double> &yhat,
                            bool applyProjectionWeights) {
  const int baseRows = A.rows;
  const int baseCols = A.cols;
  const size_t nvoxLocal = static_cast<size_t>(baseCols);
  const size_t raysPerAngle =
      (opts.num_angles > 0) ? (static_cast<size_t>(baseRows) / opts.num_angles)
                            : 0;

  if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
    if (opts.sparseOps) {
      const SparseSubsetContext subsetCtx =
          subset_detail::makeSparseSubsetContext(opts);
      opts.sparseOps->spmvSubset(A, image, subset_rows, subsetCtx, yhat.data(),
                                 threads);
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t idx = 0;
           idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
        const int global_row = subset_rows[static_cast<size_t>(idx)];
        double s = yhat[static_cast<size_t>(idx)];
        if (applyProjectionWeights && opts.w_proj) {
          s *= (*opts.w_proj)[static_cast<size_t>(global_row)];
        }
        if (opts.b_proj) {
          s += (*opts.b_proj)[static_cast<size_t>(global_row)];
        }
        yhat[static_cast<size_t>(idx)] = (s > eps) ? s : eps;
      }
      return;
    }
  }

#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(subset_rows.size());
       ++idx) {
    const int global_row = subset_rows[static_cast<size_t>(idx)];
    const int bed = (opts.bedCount > 1) ? (global_row / baseRows) : 0;
    const int row = (opts.bedCount > 1) ? (global_row % baseRows) : global_row;

    const double invScale = 1.0 / kAttenUint16Scale;
    const uint16_t *attPtr = subset_detail::attenuationForRow(
        opts, bed, row, raysPerAngle, nvoxLocal);
    const size_t col_offset = subset_detail::columnOffset(opts, bed, baseCols);

    size_t t = subset_detail::rowBegin(A, row);
    const size_t t_end = subset_detail::rowEnd(A, row);
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
      __m128i v_idx =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(&A.col_idx[t]));
      if (opts.bedCount > 1) {
        v_idx = _mm_add_epi32(v_idx, v_col_offset);
      }
      const __m256d v_f = _mm256_i32gather_pd(image, v_idx, 8);
      v_sum = _mm256_fmadd_pd(v_val_d, v_f, v_sum);
    }

    double buf[4];
    _mm256_storeu_pd(buf, v_sum);
    s = buf[0] + buf[1] + buf[2] + buf[3];
    for (; t < t_end; ++t) {
      double val = static_cast<double>(A.val[t]);
      if (attPtr) {
        val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
      }
      s += val * image[static_cast<size_t>(A.col_idx[t]) + col_offset];
    }
    if (applyProjectionWeights && opts.w_proj) {
      s *= (*opts.w_proj)[static_cast<size_t>(global_row)];
    }
    if (opts.b_proj) {
      s += (*opts.b_proj)[static_cast<size_t>(global_row)];
    }
    yhat[static_cast<size_t>(idx)] = (s > eps) ? s : eps;
  }
}

template <bool kRatioTimesValue, bool kSkipTinyManualRatios,
          typename MatrixType>
void runSubsetBackprojectKernel(const MatrixType &A,
                                const std::vector<int> &subset_rows,
                                const std::vector<double> &p,
                                const std::vector<double> &yhat,
                                const ReconOpts &opts, int threads,
                                double *output,
                                const char *bedIndexError) {
  const int baseRows = A.rows;
  const int baseCols = A.cols;
  const size_t nvoxLocal = static_cast<size_t>(baseCols);
  const size_t raysPerAngle =
      (opts.num_angles > 0) ? (static_cast<size_t>(baseRows) / opts.num_angles)
                            : 0;

  if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
    if (opts.sparseOps) {
      const SparseSubsetContext subsetCtx =
          subset_detail::makeSparseSubsetContext(opts);
      std::vector<double> ratio(subset_rows.size(), 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t idx = 0;
           idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
        const int global_row = subset_rows[static_cast<size_t>(idx)];
        ratio[static_cast<size_t>(idx)] =
            p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(idx)];
      }
      opts.sparseOps->spmvTSubset(A, ratio.data(), subset_rows, subsetCtx,
                                  output, threads);
      return;
    }
  }

  const double invScale = 1.0 / kAttenUint16Scale;
  if (!should_use_threadlocal_accum(nvoxLocal, threads)) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (ptrdiff_t idx = 0;
         idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
      const int global_row = subset_rows[static_cast<size_t>(idx)];
      const int bed = (opts.bedCount > 1) ? (global_row / baseRows) : 0;
      const int row = (opts.bedCount > 1) ? (global_row % baseRows) : global_row;
      const size_t col_offset = subset_detail::columnOffset(opts, bed, baseCols);
      const uint16_t *attPtr = subset_detail::attenuationForRow(
          opts, bed, row, raysPerAngle, nvoxLocal);
      const double ratio =
          p[static_cast<size_t>(global_row)] / yhat[static_cast<size_t>(idx)];
      if constexpr (kSkipTinyManualRatios) {
        if (std::abs(ratio) < 1e-9) {
          continue;
        }
      }

      for (size_t t = subset_detail::rowBegin(A, row);
           t < subset_detail::rowEnd(A, row); ++t) {
        double value = static_cast<double>(A.val[t]);
        if (attPtr) {
          value *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
        }
#pragma omp atomic
        output[static_cast<size_t>(A.col_idx[t]) + col_offset] +=
            subset_detail::backprojectProduct<kRatioTimesValue>(ratio, value);
      }
    }
  } else {
    const int bedCount = std::max(1, opts.bedCount);
    std::vector<std::vector<int>> bed_indices(static_cast<size_t>(bedCount));
    for (size_t i = 0; i < subset_rows.size(); ++i) {
      const int global_row = subset_rows[i];
      const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
      if (bed < 0 || bed >= bedCount) {
        throw std::runtime_error(bedIndexError);
      }
      bed_indices[static_cast<size_t>(bed)].push_back(static_cast<int>(i));
    }

    const size_t threadCount = static_cast<size_t>(threads);
    std::vector<double> tls(threadCount * nvoxLocal, 0.0);
    for (int bed = 0; bed < bedCount; ++bed) {
      if (bed_indices[static_cast<size_t>(bed)].empty()) {
        continue;
      }

      const size_t col_offset = subset_detail::columnOffset(opts, bed, baseCols);

#pragma omp parallel num_threads(threads)
      {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        double *local = tls.data() + tid * nvoxLocal;
        std::fill(local, local + nvoxLocal, 0.0);

#pragma omp for schedule(dynamic)
        for (ptrdiff_t ii = 0;
             ii < static_cast<ptrdiff_t>(
                      bed_indices[static_cast<size_t>(bed)].size());
             ++ii) {
          const int idx =
              bed_indices[static_cast<size_t>(bed)][static_cast<size_t>(ii)];
          const int global_row = subset_rows[static_cast<size_t>(idx)];
          const int row = (bedCount > 1) ? (global_row % baseRows) : global_row;
          const double ratio = p[static_cast<size_t>(global_row)] /
                               yhat[static_cast<size_t>(idx)];
          if constexpr (kSkipTinyManualRatios) {
            if (std::abs(ratio) < 1e-9) {
              continue;
            }
          }

          const uint16_t *attPtr = subset_detail::attenuationForRow(
              opts, bed, row, raysPerAngle, nvoxLocal);
          for (size_t t = subset_detail::rowBegin(A, row);
               t < subset_detail::rowEnd(A, row); ++t) {
            double value = static_cast<double>(A.val[t]);
            if (attPtr) {
              value *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
            }
            local[static_cast<size_t>(A.col_idx[t])] +=
                subset_detail::backprojectProduct<kRatioTimesValue>(ratio,
                                                                    value);
          }
        }
      }

#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(nvoxLocal); ++j) {
        double sum = 0.0;
        const size_t jj = static_cast<size_t>(j);
        for (size_t tid = 0; tid < threadCount; ++tid) {
          sum += tls[tid * nvoxLocal + jj];
        }
        output[col_offset + jj] += sum;
      }
    }
  }
}

} // namespace spect