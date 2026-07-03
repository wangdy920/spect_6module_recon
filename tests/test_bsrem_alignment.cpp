/**
 * @file test_bsrem_alignment.cpp
 * @brief BSREM template-alignment regression tests
 */

#include <gtest/gtest.h>

#include <vector>

#include "app/BSREMReconstructor.h"
#include "app/ReconConfig.h"
#include "io/CSRBuilder.h"
#include "sparse/SparseOpsRef.h"

namespace spect {
namespace test {

namespace {

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

std::vector<double> forwardProjectMultiBedCTAC(
    const io::CSR& A,
    const std::vector<double>& f,
    int bedCount,
    const std::vector<long long>& bedOffsetVoxels,
    const std::vector<std::vector<uint16_t>>& bedAttenFactors,
    int numAngles,
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
    if (b_proj)
      sum += (*b_proj)[static_cast<size_t>(globalRow)];
    if (clamp_to_eps && sum <= eps)
      sum = eps;
    p[static_cast<size_t>(globalRow)] = sum;
  }

  return p;
}

ReconOpts makeOpts(const ISparseOps* sparseOps = nullptr) {
  ReconOpts opts;
  opts.threads = 1;
  opts.bedCount = 1;
  opts.sparseOps = sparseOps;
  return opts;
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

TEST(BSREMAlignmentTest, SubsetSparseOpsMatchesFallback) {
  const io::CSR A = makeThreeByThree();
  const std::vector<int> subset_rows = {0, 2};
  const std::vector<double> f0 = {1.5, 2.0, 0.5};
  const std::vector<double> p = forwardProject(A, f0);
  const std::vector<double> global_denom = {2.8, 2.1, 2.4};
  const std::vector<double> subset_denom = {1.4, 1.2, 1.0};

  std::vector<double> f_fallback = f0;
  std::vector<double> f_sparse = f0;
  std::vector<double> buffer_fallback;
  std::vector<double> buffer_sparse;

  const ReconOpts fallback_opts = makeOpts(nullptr);
  BSREM_step(A, subset_rows, p, f_fallback, global_denom, subset_denom, 3, 3, 1, 1,
             fallback_opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer_fallback, 1.0);

  const ISparseOps& ops = getSparseOpsRef();
  const ReconOpts sparse_opts = makeOpts(&ops);
  BSREM_step(A, subset_rows, p, f_sparse, global_denom, subset_denom, 3, 3, 1, 1,
             sparse_opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer_sparse, 1.0);

  expectNear(f_sparse, f_fallback);
}

TEST(BSREMAlignmentTest, MultiBedCTACSparseOpsMatchesFallback) {
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
  const std::vector<double> global_denom = {2.5, 2.8, 2.0, 2.3, 2.4};
  const std::vector<double> subset_denom = {1.1, 1.2, 1.0, 1.3, 1.1};

  std::vector<double> f_fallback = f0;
  std::vector<double> f_sparse = f0;
  std::vector<double> buffer_fallback;
  std::vector<double> buffer_sparse;

  ReconOpts fallback_opts = makeOpts(nullptr);
  fallback_opts.bedCount = 2;
  fallback_opts.bedOffsetVoxels = bed_offsets;
  fallback_opts.bedAttenFactors = bed_atten;
  fallback_opts.num_angles = num_angles;

  BSREM_step(A, subset_rows, p, f_fallback, global_denom, subset_denom, 5, 5, 1, 1,
             fallback_opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer_fallback, 1.0);

  const ISparseOps& ops = getSparseOpsRef();
  ReconOpts sparse_opts = fallback_opts;
  sparse_opts.sparseOps = &ops;

  BSREM_step(A, subset_rows, p, f_sparse, global_denom, subset_denom, 5, 5, 1, 1,
             sparse_opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer_sparse, 1.0);

  expectNear(f_sparse, f_fallback);
}

TEST(BSREMAlignmentTest, MultiBedCTACBgRegularizedSparseOpsMatchesFallback) {
  const io::CSR A = makeTwoByThree();
  const std::vector<int> subset_rows = {0, 1, 2, 3};
  const std::vector<long long> bed_offsets = {0, 2}; // overlap by 1 voxel
  const std::vector<std::vector<uint16_t>> bed_atten = {
      {65535, 32768, 65535, 32768, 65535, 32768},
      {49152, 65535, 16384, 65535, 16384, 49152}};
  const int num_angles = 2;
  const std::vector<double> f0 = {1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector<double> b_proj = {0.05, 0.10, 0.20, 0.15};
  const std::vector<double> gradR = {0.03, 0.02, 0.04, 0.01, 0.05};
  const double beta = 0.5;

  const std::vector<double> p = forwardProjectMultiBedCTAC(
      A, f0, 2, bed_offsets, bed_atten, num_angles, &b_proj, true);
  const std::vector<double> global_denom = {2.7, 2.9, 2.1, 2.4, 2.5};
  const std::vector<double> subset_denom = {1.2, 1.3, 1.1, 1.4, 1.2};

  std::vector<double> f_fallback = f0;
  std::vector<double> f_sparse = f0;
  std::vector<double> buffer_fallback;
  std::vector<double> buffer_sparse;

  ReconOpts fallback_opts = makeOpts(nullptr);
  fallback_opts.bedCount = 2;
  fallback_opts.bedOffsetVoxels = bed_offsets;
  fallback_opts.bedAttenFactors = bed_atten;
  fallback_opts.num_angles = num_angles;
  fallback_opts.b_proj = &b_proj;

  BSREM_step(A, subset_rows, p, f_fallback, global_denom, subset_denom, 5, 5, 1, 1,
             fallback_opts, nullptr, &gradR, beta, nullptr, nullptr, &buffer_fallback, 1.0);

  const ISparseOps& ops = getSparseOpsRef();
  ReconOpts sparse_opts = fallback_opts;
  sparse_opts.sparseOps = &ops;

  BSREM_step(A, subset_rows, p, f_sparse, global_denom, subset_denom, 5, 5, 1, 1,
             sparse_opts, nullptr, &gradR, beta, nullptr, nullptr, &buffer_sparse, 1.0);

  expectNear(f_sparse, f_fallback);
}

TEST(BSREMAlignmentTest, CurvatureDenominatorReducesStepMagnitude) {
  io::CSR A;
  A.rows = 1;
  A.cols = 1;
  A.row_ptr = {0, 1};
  A.col_idx = {0};
  A.val = {1.0f};

  const std::vector<int> subset_rows = {0};
  const std::vector<double> p = {2.0};
  const std::vector<double> global_denom = {1.0};
  const std::vector<double> subset_denom = {1.0};
  const std::vector<double> curv = {1.0};

  std::vector<double> f_no_curv = {1.0};
  std::vector<double> f_with_curv = {1.0};
  std::vector<double> buffer;
  ReconOpts opts = makeOpts(nullptr);

  BSREM_step(A, subset_rows, p, f_no_curv, global_denom, subset_denom,
             1, 1, 1, 1, opts, nullptr, nullptr, 0.5, nullptr, nullptr, &buffer, 1.0);

  BSREM_step(A, subset_rows, p, f_with_curv, global_denom, subset_denom,
             1, 1, 1, 1, opts, nullptr, nullptr, 0.5, &curv, nullptr, &buffer, 1.0);

  EXPECT_LT(f_with_curv[0], f_no_curv[0]);
  EXPECT_NEAR(f_no_curv[0], 2.0, 1e-12);
  EXPECT_NEAR(f_with_curv[0], 1.6666666667, 1e-9);
}

TEST(BSREMAlignmentTest, KappaMapStrengthensPenalty) {
  io::CSR A;
  A.rows = 1;
  A.cols = 1;
  A.row_ptr = {0, 1};
  A.col_idx = {0};
  A.val = {1.0f};

  const std::vector<int> subset_rows = {0};
  const std::vector<double> p = {2.0};
  const std::vector<double> global_denom = {1.0};
  const std::vector<double> subset_denom = {1.0};
  const std::vector<double> grad = {1.0};
  const std::vector<double> kappa1 = {1.0};
  const std::vector<double> kappa2 = {2.0};

  std::vector<double> f_k1 = {1.0};
  std::vector<double> f_k2 = {1.0};
  std::vector<double> buffer;
  ReconOpts opts = makeOpts(nullptr);

  BSREM_step(A, subset_rows, p, f_k1, global_denom, subset_denom,
             1, 1, 1, 1, opts, nullptr, &grad, 0.5, nullptr, &kappa1, &buffer, 1.0);
  BSREM_step(A, subset_rows, p, f_k2, global_denom, subset_denom,
             1, 1, 1, 1, opts, nullptr, &grad, 0.5, nullptr, &kappa2, &buffer, 1.0);

  EXPECT_NEAR(f_k1[0], 1.5, 1e-12);
  EXPECT_NEAR(f_k2[0], 1.0, 1e-12);
  EXPECT_LT(f_k2[0], f_k1[0]);
}

} // namespace test
} // namespace spect
