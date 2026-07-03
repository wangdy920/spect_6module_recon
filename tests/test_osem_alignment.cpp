/**
 * @file test_osem_alignment.cpp
 * @brief OSEM template-alignment regression tests
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "app/OSEMReconstructor.h"
#include "app/ReconConfig.h"
#include "io/CSRBuilder.h"
#include "sparse/SparseOpsRef.h"

namespace spect {
namespace test {

namespace {

io::CSR makeOneRowTwoCol() {
  io::CSR A;
  A.rows = 1;
  A.cols = 2;
  A.row_ptr = {0, 2};
  A.col_idx = {0, 1};
  A.val = {1.0f, 1.0f};
  return A;
}

io::CSR makeThreeByThree() {
  io::CSR A;
  A.rows = 3;
  A.cols = 3;
  A.row_ptr = {0, 2, 4, 5};
  A.col_idx = {0, 2, 1, 2, 0};
  A.val = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  return A;
}

io::CSR makeTwoByThree() {
  io::CSR A;
  A.rows = 2;
  A.cols = 3;
  A.row_ptr = {0, 2, 4};
  A.col_idx = {0, 1, 1, 2};
  A.val = {1.0f, 2.0f, 3.0f, 4.0f};
  return A;
}

io::CSRView viewOf(const io::CSR& A) {
  return io::CSRView{A.rows, A.cols, A.row_ptr.data(), A.col_idx.data(),
                     A.val.data(), static_cast<uint64_t>(A.val.size())};
}

std::vector<double> forwardProject(const io::CSR& A, const std::vector<double>& f) {
  std::vector<double> y(static_cast<size_t>(A.rows), 0.0);
  for (int row = 0; row < A.rows; ++row) {
    double sum = 0.0;
    for (size_t t = static_cast<size_t>(A.row_ptr[row]);
         t < static_cast<size_t>(A.row_ptr[row + 1]); ++t) {
      sum += static_cast<double>(A.val[t]) * f[static_cast<size_t>(A.col_idx[t])];
    }
    y[static_cast<size_t>(row)] = sum;
  }
  return y;
}

ReconOpts makeOpts(const ISparseOps* sparseOps = nullptr) {
  ReconOpts opts;
  opts.threads = 1;
  opts.bedCount = 1;
  opts.sparseOps = sparseOps;
  return opts;
}

std::vector<double> forwardProjectMultiBedCTAC(
    const io::CSR& A,
    const std::vector<double>& f,
    int bedCount,
    const std::vector<long long>& bedOffsetVoxels,
    const std::vector<std::vector<uint16_t>>& bedAttenFactors,
    int numAngles,
    const std::vector<double>* w_proj = nullptr,
    const std::vector<double>* b_proj = nullptr,
    bool clamp_to_eps = false) {
  const int baseRows = A.rows;
  const int baseCols = A.cols;
  const int totalRows = baseRows * bedCount;
  const size_t raysPerAngle =
      (numAngles > 0) ? (static_cast<size_t>(baseRows) / static_cast<size_t>(numAngles)) : 0;
  const double invScale = 1.0 / 65535.0;
  const double eps = 1e-12;

  std::vector<double> p(static_cast<size_t>(totalRows), 0.0);
  for (int globalRow = 0; globalRow < totalRows; ++globalRow) {
    const int bed = (bedCount > 1) ? (globalRow / baseRows) : 0;
    const int row = (bedCount > 1) ? (globalRow % baseRows) : globalRow;

    const uint16_t* attPtr = nullptr;
    if (!bedAttenFactors.empty()) {
      int angle = 0;
      if (raysPerAngle > 0)
        angle = static_cast<int>(static_cast<size_t>(row) / raysPerAngle);
      if (angle >= numAngles)
        angle = numAngles - 1;
      attPtr = bedAttenFactors[static_cast<size_t>(bed)].data() +
               static_cast<size_t>(angle) * static_cast<size_t>(baseCols);
    }

    const size_t colOffset =
        !bedOffsetVoxels.empty()
            ? static_cast<size_t>(bedOffsetVoxels[static_cast<size_t>(bed)])
            : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

    double sum = 0.0;
    for (size_t t = static_cast<size_t>(A.row_ptr[row]);
         t < static_cast<size_t>(A.row_ptr[row + 1]); ++t) {
      double val = static_cast<double>(A.val[t]);
      if (attPtr)
        val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
      sum += val * f[colOffset + static_cast<size_t>(A.col_idx[t])];
    }
    if (w_proj)
      sum *= (*w_proj)[static_cast<size_t>(globalRow)];
    if (b_proj)
      sum += (*b_proj)[static_cast<size_t>(globalRow)];
    if (clamp_to_eps && sum <= eps)
      sum = eps;
    p[static_cast<size_t>(globalRow)] = sum;
  }

  return p;
}

void expectNear(const std::vector<double>& actual,
                const std::vector<double>& expected,
                double tol = 1e-12) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], tol) << "Mismatch at index " << i;
  }
}

} // namespace

TEST(OSEMAlignmentTest, SparseOpsSubsetMatchesFallback) {
  const io::CSR A = makeThreeByThree();
  const io::CSRView Aview = viewOf(A);
  const std::vector<int> subset_rows = {1, 2};
  const std::vector<double> f0 = {1.5, 2.0, 0.5};
  const std::vector<double> full_p = forwardProject(A, f0);
  const std::vector<double> denom = {6.0, 3.0, 6.0}; // A^T * 1

  std::vector<double> f_fallback = f0;
  std::vector<double> f_sparse = f0;

  const ReconOpts fallback_opts = makeOpts(nullptr);
  OSEM_step(Aview, subset_rows, full_p, f_fallback, denom,
            3, 1, 1, fallback_opts, nullptr, nullptr, 0.0, nullptr, 1.0);

  const ISparseOps& ops = getSparseOpsRef();
  const ReconOpts sparse_opts = makeOpts(&ops);
  OSEM_step(Aview, subset_rows, full_p, f_sparse, denom,
            3, 1, 1, sparse_opts, nullptr, nullptr, 0.0, nullptr, 1.0);

  expectNear(f_sparse, f_fallback);
}

TEST(OSEMAlignmentTest, MultiBedCTACSparseOpsMatchesFallback) {
  const io::CSR A = makeTwoByThree();
  const std::vector<int> subset_rows = {0, 1, 2, 3};
  const std::vector<long long> bed_offsets = {0, 2}; // overlap by 1 voxel
  const std::vector<std::vector<uint16_t>> bed_atten = {
      {65535, 32768, 65535, 32768, 65535, 32768},
      {49152, 65535, 16384, 65535, 16384, 49152}};
  const int num_angles = 2;
  const std::vector<double> f0 = {1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector<double> p =
      forwardProjectMultiBedCTAC(A, f0, 2, bed_offsets, bed_atten, num_angles);
  const std::vector<double> denom(f0.size(), 1.0);

  std::vector<double> f_fallback = f0;
  std::vector<double> f_sparse = f0;

  ReconOpts fallback_opts = makeOpts(nullptr);
  fallback_opts.bedCount = 2;
  fallback_opts.bedOffsetVoxels = bed_offsets;
  fallback_opts.bedAttenFactors = bed_atten;
  fallback_opts.num_angles = num_angles;

  OSEM_step(A, subset_rows, p, f_fallback, denom, 5, 1, 1, fallback_opts,
            nullptr, nullptr, 0.0, nullptr, 1.0);

  const ISparseOps& ops = getSparseOpsRef();
  ReconOpts sparse_opts = fallback_opts;
  sparse_opts.sparseOps = &ops;

  OSEM_step(A, subset_rows, p, f_sparse, denom, 5, 1, 1, sparse_opts, nullptr,
            nullptr, 0.0, nullptr, 1.0);

  expectNear(f_sparse, f_fallback);
}

TEST(OSEMAlignmentTest, MultiBedCTACWeightedBgSparseOpsMatchesFallback) {
  const io::CSR A = makeTwoByThree();
  const std::vector<int> subset_rows = {0, 1, 2, 3};
  const std::vector<long long> bed_offsets = {0, 2}; // overlap by 1 voxel
  const std::vector<std::vector<uint16_t>> bed_atten = {
      {65535, 32768, 65535, 32768, 65535, 32768},
      {49152, 65535, 16384, 65535, 16384, 49152}};
  const int num_angles = 2;
  const std::vector<double> f0 = {1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector<double> w_proj = {0.8, 1.2, 1.1, 0.9};
  const std::vector<double> b_proj = {0.05, 0.10, 0.20, 0.15};
  const std::vector<double> p = forwardProjectMultiBedCTAC(
      A, f0, 2, bed_offsets, bed_atten, num_angles, &w_proj, &b_proj, true);
  const std::vector<double> denom(f0.size(), 1.0);

  std::vector<double> f_fallback = f0;
  std::vector<double> f_sparse = f0;

  ReconOpts fallback_opts = makeOpts(nullptr);
  fallback_opts.bedCount = 2;
  fallback_opts.bedOffsetVoxels = bed_offsets;
  fallback_opts.bedAttenFactors = bed_atten;
  fallback_opts.num_angles = num_angles;
  fallback_opts.w_proj = &w_proj;
  fallback_opts.b_proj = &b_proj;

  OSEM_step(A, subset_rows, p, f_fallback, denom, 5, 1, 1, fallback_opts,
            nullptr, nullptr, 0.0, nullptr, 1.0);

  const ISparseOps& ops = getSparseOpsRef();
  ReconOpts sparse_opts = fallback_opts;
  sparse_opts.sparseOps = &ops;

  OSEM_step(A, subset_rows, p, f_sparse, denom, 5, 1, 1, sparse_opts, nullptr,
            nullptr, 0.0, nullptr, 1.0);

  expectNear(f_sparse, f_fallback);
}

TEST(OSEMAlignmentTest, MultiBedCTACWeightedBgRegularizedSparseOpsMatchesFallback) {
  const io::CSR A = makeTwoByThree();
  const std::vector<int> subset_rows = {0, 1, 2, 3};
  const std::vector<long long> bed_offsets = {0, 2}; // overlap by 1 voxel
  const std::vector<std::vector<uint16_t>> bed_atten = {
      {65535, 32768, 65535, 32768, 65535, 32768},
      {49152, 65535, 16384, 65535, 16384, 49152}};
  const int num_angles = 2;
  const std::vector<double> f0 = {1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector<double> w_proj = {0.8, 1.2, 1.1, 0.9};
  const std::vector<double> b_proj = {0.05, 0.10, 0.20, 0.15};
  const std::vector<double> gradR = {0.2, 0.1, 0.15, 0.05, 0.12};
  const double beta = 0.25;

  const std::vector<double> p = forwardProjectMultiBedCTAC(
      A, f0, 2, bed_offsets, bed_atten, num_angles, &w_proj, &b_proj, true);
  const std::vector<double> denom = {2.0, 2.5, 3.0, 2.2, 2.8};

  std::vector<double> f_fallback = f0;
  std::vector<double> f_sparse = f0;

  ReconOpts fallback_opts = makeOpts(nullptr);
  fallback_opts.bedCount = 2;
  fallback_opts.bedOffsetVoxels = bed_offsets;
  fallback_opts.bedAttenFactors = bed_atten;
  fallback_opts.num_angles = num_angles;
  fallback_opts.w_proj = &w_proj;
  fallback_opts.b_proj = &b_proj;

  OSEM_step(A, subset_rows, p, f_fallback, denom, 5, 1, 1, fallback_opts,
            nullptr, &gradR, beta, nullptr, 1.0);

  const ISparseOps& ops = getSparseOpsRef();
  ReconOpts sparse_opts = fallback_opts;
  sparse_opts.sparseOps = &ops;

  OSEM_step(A, subset_rows, p, f_sparse, denom, 5, 1, 1, sparse_opts, nullptr,
            &gradR, beta, nullptr, 1.0);

  expectNear(f_sparse, f_fallback);
}

} // namespace test
} // namespace spect
