// ProjectionOps.inl - Template implementations for projection operations
#pragma once

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <type_traits>

namespace spect {

// ===== computeNormAT_Atten template =====

template <typename MatrixType>
std::vector<double> computeNormAT_Atten(
    const MatrixType& A,
    const std::vector<uint16_t>& atten,
    int numAngles,
    int threads) {
  threads = effective_threads(threads);
  const size_t Nvox = static_cast<size_t>(A.cols);
  std::vector<double> denom(Nvox, 0.0);
  const size_t raysPerAngle = (numAngles > 0) ? (A.rows / numAngles) : 0;
  const double invScale = 1.0 / kAttenUint16Scale;

  if (!should_use_threadlocal_accum(Nvox, threads)) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(A.rows); ++row) {
      int angle = 0;
      if (raysPerAngle > 0)
        angle = static_cast<int>(row / raysPerAngle);
      if (angle >= numAngles)
        angle = numAngles - 1;

      const uint16_t* attPtr = atten.data() + (size_t)angle * Nvox;
      const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
      const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);

      for (size_t t = r0; t < r1; ++t) {
        double v = static_cast<double>(A.val[t]);
        v *= (double)attPtr[A.col_idx[t]] * invScale;
#pragma omp atomic
        denom[static_cast<size_t>(A.col_idx[t])] += v;
      }
    }
    return denom;
  }

  const size_t threadCount = static_cast<size_t>(threads);
  std::vector<double> tls(threadCount * Nvox, 0.0);

#pragma omp parallel num_threads(threads)
  {
    const size_t tid = static_cast<size_t>(omp_get_thread_num());
    double* local = tls.data() + tid * Nvox;

#pragma omp for schedule(dynamic)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(A.rows); ++row) {
      int angle = 0;
      if (raysPerAngle > 0)
        angle = static_cast<int>(row / raysPerAngle);
      if (angle >= numAngles)
        angle = numAngles - 1;

      const uint16_t* attPtr = atten.data() + (size_t)angle * Nvox;
      const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
      const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);

      for (size_t t = r0; t < r1; ++t) {
        double v = static_cast<double>(A.val[t]);
        v *= (double)attPtr[A.col_idx[t]] * invScale;
        local[static_cast<size_t>(A.col_idx[t])] += v;
      }
    }
  }

#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(Nvox); ++j) {
    double sum = 0.0;
    const size_t jj = static_cast<size_t>(j);
    for (size_t tid = 0; tid < threadCount; ++tid)
      sum += tls[tid * Nvox + jj];
    denom[jj] = sum;
  }

  return denom;
}

// ===== forward_project template =====

template <typename MatrixType>
std::vector<double> forward_project(
    const MatrixType& A,
    const std::vector<double>& f,
    int threads) {
  std::vector<double> result(A.rows, 0.0);

#pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(A.rows); ++row) {
    double sum = 0.0;
    
    if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
      const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
      const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
      for (size_t t = r0; t < r1; ++t) {
        sum += static_cast<double>(A.val[t]) * f[A.col_idx[t]];
      }
    } else {
      for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
        sum += static_cast<double>(A.val[t]) * f[A.col_idx[t]];
      }
    }
    
    result[row] = sum;
  }

  return result;
}

// ===== backproject template =====

template <typename MatrixType>
std::vector<double> backproject(
    const MatrixType& A,
    const std::vector<double>& y,
    int threads) {
  threads = effective_threads(threads);
  const size_t nvox = static_cast<size_t>(A.cols);
  std::vector<double> result(nvox, 0.0);

  if (!should_use_threadlocal_accum(nvox, threads)) {
    // Atomic version
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(A.rows); ++row) {
      const double y_val = y[row];
      
      if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
        for (size_t t = r0; t < r1; ++t) {
          const double contrib = static_cast<double>(A.val[t]) * y_val;
#pragma omp atomic
          result[A.col_idx[t]] += contrib;
        }
      } else {
        for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
          const double contrib = static_cast<double>(A.val[t]) * y_val;
#pragma omp atomic
          result[A.col_idx[t]] += contrib;
        }
      }
    }
    return result;
  }

  // Thread-local version
  const size_t threadCount = static_cast<size_t>(threads);
  std::vector<double> tls(threadCount * nvox, 0.0);

#pragma omp parallel num_threads(threads)
  {
    const size_t tid = static_cast<size_t>(omp_get_thread_num());
    double* local = tls.data() + tid * nvox;

#pragma omp for schedule(dynamic)
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(A.rows); ++row) {
      const double y_val = y[row];
      
      if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
        for (size_t t = r0; t < r1; ++t) {
          local[A.col_idx[t]] += static_cast<double>(A.val[t]) * y_val;
        }
      } else {
        for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
          local[A.col_idx[t]] += static_cast<double>(A.val[t]) * y_val;
        }
      }
    }
  }

  // Reduction
#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(nvox); ++j) {
    double sum = 0.0;
    const size_t jj = static_cast<size_t>(j);
    for (size_t tid = 0; tid < threadCount; ++tid)
      sum += tls[tid * nvox + jj];
    result[jj] = sum;
  }

  return result;
}

// ===== computeSubsetSensitivity template =====

template <typename MatrixType>
std::vector<double> computeSubsetSensitivity(
    const MatrixType& A,
    const std::vector<int>& subset_rows,
    int N,
    int threads,
    const ReconOpts& opts) {
  threads = effective_threads(threads);
  std::vector<double> sensitivity(N, 0.0);
  if (subset_rows.empty())
    return sensitivity;

  const int baseRows = A.rows;
  const int baseCols = A.cols;
  const size_t NvoxLocal = static_cast<size_t>(baseCols);
  const size_t raysPerAngle =
      (opts.num_angles > 0) ? (baseRows / opts.num_angles) : 0;

  const int bedCount = std::max(1, opts.bedCount);
  if (!should_use_threadlocal_accum(NvoxLocal, threads)) {
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(subset_rows.size());
         ++idx) {
      int global_row = subset_rows[static_cast<size_t>(idx)];
      int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
      int row = (bedCount > 1) ? (global_row % baseRows) : global_row;

      // Attenuation factors pointer
      const uint16_t* attPtr = nullptr;
      const double invScale = 1.0 / kAttenUint16Scale;
      if (!opts.bedAttenFactors.empty()) {
        int angle = 0;
        if (raysPerAngle > 0)
          angle = static_cast<int>(row / raysPerAngle);
        if (angle >= opts.num_angles)
          angle = opts.num_angles - 1;
        attPtr = opts.bedAttenFactors[bed].data() + static_cast<size_t>(angle) * NvoxLocal;
      }

      size_t col_offset = (!opts.bedOffsetVoxels.empty())
                              ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                              : static_cast<size_t>(bed) * baseCols;

      if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
        for (size_t t = r0; t < r1; ++t) {
          double v = static_cast<double>(A.val[t]);
          if (attPtr)
            v *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
#pragma omp atomic
          sensitivity[static_cast<size_t>(A.col_idx[t]) + col_offset] += v;
        }
      } else {
        for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
          double v = static_cast<double>(A.val[t]);
          if (attPtr)
            v *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
#pragma omp atomic
          sensitivity[static_cast<size_t>(A.col_idx[t]) + col_offset] += v;
        }
      }
    }
    return sensitivity;
  }

  // Thread-local accumulation version
  std::vector<std::vector<int>> bed_indices(bedCount);
  for (size_t i = 0; i < subset_rows.size(); ++i) {
    const int global_row = subset_rows[i];
    const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
    if (bed < 0 || bed >= bedCount)
      throw std::runtime_error(
          "computeSubsetSensitivity: bed index out of range");
    bed_indices[bed].push_back(static_cast<int>(i));
  }

  const size_t threadCount = static_cast<size_t>(threads);
  std::vector<double> tls(threadCount * NvoxLocal, 0.0);

  for (int bed = 0; bed < bedCount; ++bed) {
    if (bed_indices[bed].empty())
      continue;

    const size_t col_offset = (!opts.bedOffsetVoxels.empty())
                                  ? static_cast<size_t>(opts.bedOffsetVoxels[bed])
                                  : static_cast<size_t>(bed) * baseCols;

#pragma omp parallel num_threads(threads)
    {
      const size_t tid = static_cast<size_t>(omp_get_thread_num());
      double* local = tls.data() + tid * NvoxLocal;

      const uint16_t* attPtr = nullptr;
      const double invScale = 1.0 / kAttenUint16Scale;

#pragma omp for schedule(dynamic)
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(bed_indices[bed].size()); ++i) {
        const int idx = bed_indices[bed][static_cast<size_t>(i)];
        const int global_row = subset_rows[static_cast<size_t>(idx)];
        const int row = (bedCount > 1) ? (global_row % baseRows) : global_row;

        if (!opts.bedAttenFactors.empty()) {
          int angle = 0;
          if (raysPerAngle > 0)
            angle = static_cast<int>(row / raysPerAngle);
          if (angle >= opts.num_angles)
            angle = opts.num_angles - 1;
          attPtr = opts.bedAttenFactors[bed].data() + static_cast<size_t>(angle) * NvoxLocal;
        }

        if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
          const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
          const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
          for (size_t t = r0; t < r1; ++t) {
            double v = static_cast<double>(A.val[t]);
            if (attPtr)
              v *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
            local[static_cast<size_t>(A.col_idx[t])] += v;
          }
        } else {
          for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
            double v = static_cast<double>(A.val[t]);
            if (attPtr)
              v *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
            local[static_cast<size_t>(A.col_idx[t])] += v;
          }
        }
      }
    }

    // Reduction to global sensitivity with offset
#pragma omp parallel for schedule(static) num_threads(threads)
    for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(NvoxLocal); ++j) {
      double sum = 0.0;
      const size_t jj = static_cast<size_t>(j);
      for (size_t tid = 0; tid < threadCount; ++tid)
        sum += tls[tid * NvoxLocal + jj];
      sensitivity[jj + col_offset] += sum;
    }

    // Clear thread-local storage for next bed
    std::fill(tls.begin(), tls.end(), 0.0);
  }

  return sensitivity;
}

} // namespace spect
