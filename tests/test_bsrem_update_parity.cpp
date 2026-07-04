/**
 * @file test_bsrem_update_parity.cpp
 * @brief BSREM update parity and kappa-map golden tests.
 */

#include <gtest/gtest.h>

#include "app/BSREMOptimizer.h"
#include "app/BSREMReconstructor.h"
#include "app/IOptimizer.h"
#include "app/ReconConfig.h"
#include "cli/ReconArgs.h"
#include "io/CSRBuilder.h"
#include "main/BsremKappaMap.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace spect {
namespace test {
namespace {

constexpr int kNx = 4;
constexpr int kNy = 4;
constexpr int kNz = 4;
constexpr int kNvox = kNx * kNy * kNz;
constexpr double kUpdateEps = 1e-12;

io::CSR makeSyntheticCsr() {
  io::CSR matrix;
  matrix.rows = 8;
  matrix.cols = kNvox;
  matrix.row_ptr = {0, 3, 6, 9, 12, 15, 18, 21, 24};
  matrix.col_idx = {0,  7,  19, 3,  18, 42, 6,  23, 55, 9,  31, 63,
                    12, 28, 44, 15, 37, 52, 21, 40, 58, 24, 46, 61};
  matrix.val = {1.00f, 0.25f, 0.50f, 0.75f, 1.25f, 0.30f,
                0.60f, 0.90f, 0.40f, 1.10f, 0.20f, 0.70f,
                0.35f, 0.80f, 1.15f, 0.55f, 0.45f, 0.95f,
                1.30f, 0.65f, 0.25f, 0.50f, 1.05f, 0.85f};
  return matrix;
}

io::CSR makeIdentityCsr() {
  io::CSR matrix;
  matrix.rows = kNvox;
  matrix.cols = kNvox;
  matrix.row_ptr.resize(static_cast<size_t>(kNvox) + 1);
  matrix.col_idx.resize(static_cast<size_t>(kNvox));
  matrix.val.resize(static_cast<size_t>(kNvox), 1.0f);
  for (int i = 0; i <= kNvox; ++i)
    matrix.row_ptr[static_cast<size_t>(i)] = static_cast<size_t>(i);
  for (int i = 0; i < kNvox; ++i)
    matrix.col_idx[static_cast<size_t>(i)] = i;
  return matrix;
}

std::vector<int> allRows(int rows) {
  std::vector<int> subsetRows(static_cast<size_t>(rows));
  for (int i = 0; i < rows; ++i)
    subsetRows[static_cast<size_t>(i)] = i;
  return subsetRows;
}

std::vector<double> makeImage() {
  std::vector<double> image(static_cast<size_t>(kNvox));
  for (size_t i = 0; i < image.size(); ++i)
    image[i] = 0.75 + 0.01 * static_cast<double>((i * 7) % 17);
  return image;
}

std::vector<double> forwardProject(const io::CSR &matrix,
                                   const std::vector<double> &image) {
  std::vector<double> projection(static_cast<size_t>(matrix.rows), 0.0);
  for (int row = 0; row < matrix.rows; ++row) {
    double sum = 0.0;
    for (size_t idx = matrix.row_ptr[static_cast<size_t>(row)];
         idx < matrix.row_ptr[static_cast<size_t>(row) + 1]; ++idx) {
      sum += static_cast<double>(matrix.val[idx]) *
             image[static_cast<size_t>(matrix.col_idx[idx])];
    }
    projection[static_cast<size_t>(row)] =
        sum * (1.03 + 0.02 * static_cast<double>(row % 3));
  }
  return projection;
}

std::vector<double> makePattern(double base, double step, int period) {
  std::vector<double> values(static_cast<size_t>(kNvox));
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = base + step * static_cast<double>(i % static_cast<size_t>(period));
  return values;
}

ReconOpts makeOpts() {
  ReconOpts opts;
  opts.threads = 1;
  opts.bedCount = 1;
  opts.nonneg = true;
  return opts;
}

std::vector<double> updateFactors(const std::vector<double> &before,
                                  const std::vector<double> &after) {
  std::vector<double> factors(before.size(), 0.0);
  for (size_t i = 0; i < before.size(); ++i) {
    double current = before[i];
    if (current < kUpdateEps)
      current = kUpdateEps;
    factors[i] = after[i] / current - 1.0;
  }
  return factors;
}

std::vector<double> compactLegacyBsremCore(
    std::vector<double> image, const std::vector<double> &numer,
    const std::vector<double> &denom, const std::vector<double> *gradPtr,
    double bsremBeta, const std::vector<double> *curvPtr,
    const std::vector<double> *kappaPtr, double bsremAlpha, bool nonneg) {
  constexpr double maxUpdateFactor = 1.0;
  constexpr double minUpdateFactor = -0.9;
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(image.size()); ++j) {
    const size_t idx = static_cast<size_t>(j);
    const double sensGlobal = denom[idx];
    if (!(sensGlobal > kUpdateEps)) {
      image[idx] = 0.0;
      continue;
    }

    const double kappa = kappaPtr ? (*kappaPtr)[idx] : 1.0;
    const double termLikelihood = numer[idx] - denom[idx];
    const double termPenalty =
        gradPtr ? bsremBeta * kappa * (*gradPtr)[idx] : 0.0;
    double denomEff = sensGlobal;
    if (curvPtr)
      denomEff += bsremBeta * std::max((*curvPtr)[idx], 0.0);

    double fCurr = image[idx];
    if (fCurr < kUpdateEps)
      fCurr = kUpdateEps;

    double updateFactor =
        bsremAlpha * ((termLikelihood - termPenalty) / denomEff);
    if (updateFactor > maxUpdateFactor)
      updateFactor = maxUpdateFactor;
    if (updateFactor < minUpdateFactor)
      updateFactor = minUpdateFactor;

    double value = fCurr * (1.0 + updateFactor);
    if (value < kUpdateEps)
      value = nonneg ? 0.0 : kUpdateEps;
    image[idx] = value;
  }
  return image;
}

void expectBitwiseEqual(const std::vector<double> &actual,
                        const std::vector<double> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_DOUBLE_EQ(actual[i], expected[i]) << "index " << i;
}

} // namespace

TEST(BSREMUpdateParityTest, ReconstructorStepMatchesStandardOptimizerWrapper) {
  const io::CSR matrix = makeSyntheticCsr();
  const std::vector<int> subsetRows = allRows(matrix.rows);
  const std::vector<double> image0 = makeImage();
  const std::vector<double> projections = forwardProject(matrix, image0);
  const std::vector<double> globalDenom = makePattern(1.20, 0.03, 11);
  const std::vector<double> subsetDenom = makePattern(0.90, 0.02, 13);
  const std::vector<double> grad = makePattern(-0.06, 0.015, 9);
  const std::vector<double> curv = makePattern(0.10, 0.01, 7);
  const std::vector<double> kappa = makePattern(0.80, 0.04, 5);
  constexpr double betaScaled = 0.17;
  constexpr double alpha = 0.83;

  ReconOpts opts = makeOpts();
  std::vector<double> direct = image0;
  std::vector<double> standard = image0;
  std::vector<double> directBuffer;
  std::vector<double> standardBuffer;

  BSREM_step(matrix, subsetRows, projections, direct, globalDenom, subsetDenom,
             kNvox, kNx, kNy, kNz, opts, nullptr, &grad, betaScaled, &curv,
             &kappa, &directBuffer, alpha);

  StepParams params;
  params.matrixCSR = &matrix;
  params.subsetRows = &subsetRows;
  params.projections = &projections;
  params.image = &standard;
  params.denom = &globalDenom;
  params.Nx = kNx;
  params.Ny = kNy;
  params.Nz = kNz;
  params.opts = &opts;
  params.subsetDenom = &subsetDenom;
  params.bsremAlpha = alpha;
  params.bsremBetaScaled = betaScaled;
  params.curvature = &curv;
  params.kappa = &kappa;
  params.gradReg = &grad;
  params.bsremBuffer = &standardBuffer;
  BSREMOptimizer optimizer;
  optimizer.step(params);

  expectBitwiseEqual(standard, direct);
  expectBitwiseEqual(updateFactors(image0, standard),
                     updateFactors(image0, direct));
}

TEST(BSREMUpdateParityTest, StandardStepMatchesCompactCoreFormula) {
  const io::CSR matrix = makeIdentityCsr();
  const std::vector<int> subsetRows = allRows(matrix.rows);
  const std::vector<double> image0(static_cast<size_t>(kNvox), 1.0);
  const std::vector<double> numer = makePattern(0.85, 0.025, 17);
  const std::vector<double> denom = makePattern(1.05, 0.020, 13);
  const std::vector<double> grad = makePattern(-0.04, 0.010, 9);
  const std::vector<double> curv = makePattern(0.07, 0.015, 7);
  const std::vector<double> kappa = makePattern(0.70, 0.05, 6);
  constexpr double beta = 0.19;
  constexpr double alpha = 0.72;

  ReconOpts opts = makeOpts();
  std::vector<double> standard = image0;
  std::vector<double> buffer;
  BSREM_step(matrix, subsetRows, numer, standard, denom, denom, kNvox, kNx, kNy,
             kNz, opts, nullptr, &grad, beta, &curv, &kappa, &buffer, alpha);

  const std::vector<double> compact = compactLegacyBsremCore(
      image0, numer, denom, &grad, beta, &curv, &kappa, alpha, opts.nonneg);

  expectBitwiseEqual(compact, standard);
  expectBitwiseEqual(updateFactors(image0, compact),
                     updateFactors(image0, standard));
}

TEST(BSREMKappaMapGoldenTest, StandardHistoricalBehavior) {
  cli::ReconArgs args;
  args.bsremUseKappaMap = true;
  args.bsremKappaEps = 1e-6;
  args.bsremKappaClipMin = 0.0;
  args.bsremKappaClipMax = 1.0e9;
  const std::vector<double> sensitivity = {4.0, 0.0, -1.0, 9.0};
  const std::vector<double> expected = {1.1999999666666725,
                                        2.4000002333333175,
                                        2.4000002333333175,
                                        0.8000000333333275};

  const std::vector<double> actual =
      main::computeKappaMapStandard(sensitivity, args, nullptr);

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i], 1e-15) << "index " << i;
}

TEST(BSREMKappaMapGoldenTest, CompactHistoricalBehavior) {
  cli::ReconArgs args;
  args.bsremUseKappaMap = true;
  args.bsremKappaEps = 1e-6;
  args.bsremKappaClipMin = 0.0;
  args.bsremKappaClipMax = 1.0e9;
  const std::vector<double> sensitivity = {4.0, 0.0, -1.0, 9.0};
  const std::vector<double> expected = {0.0009995833819646974,
                                        1.9991670138252242,
                                        1.9991670138252242,
                                        0.0006663889675867998};

  const std::vector<double> actual =
      main::computeKappaMapCompact(sensitivity, args, nullptr);

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i], 1e-15) << "index " << i;
}

} // namespace test
} // namespace spect