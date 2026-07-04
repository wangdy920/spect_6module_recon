#pragma once

#include <cstdint>
#include <vector>

namespace spect {

struct SparseSubsetContext;

namespace sparse_detail {

template <typename MatrixType>
void spmvSubsetImpl(const MatrixType& A,
                    const double* x,
                    const std::vector<int>& subset_rows,
                    const SparseSubsetContext& ctx,
                    double* y,
                    int threads);

template <typename MatrixType>
void spmvTSubsetImpl(const MatrixType& A,
                     const double* x,
                     const std::vector<int>& subset_rows,
                     const SparseSubsetContext& ctx,
                     double* y,
                     int threads);

template <typename MatrixType>
void computeNormAT_AttenImpl(const MatrixType& A,
                             const std::vector<uint16_t>& atten,
                             int numAngles,
                             double* norm,
                             int threads);

template <typename MatrixType>
void computeSubsetSensitivityImpl(const MatrixType& A,
                                  const std::vector<int>& subset_rows,
                                  int N,
                                  const SparseSubsetContext& ctx,
                                  double* sensitivity,
                                  int threads);

} // namespace sparse_detail
} // namespace spect