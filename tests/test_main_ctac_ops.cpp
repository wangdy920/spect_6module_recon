#include <gtest/gtest.h>

#include "main/CTACOps.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace spect::main::test {

namespace {

double factorAt(const std::vector<uint16_t> &factors, size_t voxels, int angle,
                size_t idx) {
  return static_cast<double>(factors[static_cast<size_t>(angle) * voxels + idx]) /
         65535.0;
}

} // namespace

TEST(CTACOpsTest, ComputeAttenFactorsZeroMuAllOnes) {
  const int nx = 2;
  const int ny = 2;
  const int nz = 2;
  const int numAngles = 3;
  const std::vector<double> muMap(static_cast<size_t>(nx) * ny * nz, 0.0);

  const std::vector<uint16_t> factors = computeAttenFactors(
      muMap, nx, ny, nz, 1.0, 1.0, 1.0, numAngles, 1, 1);

  ASSERT_EQ(factors.size(), static_cast<size_t>(numAngles) * muMap.size());
  for (uint16_t value : factors) {
    EXPECT_EQ(value, static_cast<uint16_t>(65535));
  }
}

TEST(CTACOpsTest, ApplyCTACToCSRZeroMuKeepsWeights) {
  spect::io::CSR csr;
  csr.rows = 2;
  csr.cols = 4;
  csr.row_ptr = {0, 2, 3};
  csr.col_idx = {0, 1, 2};
  csr.val = {1.0f, 2.0f, 3.0f};

  const std::vector<float> before = csr.val;
  const std::vector<double> muMap(4, 0.0);

  applyCTACToCSR(csr, muMap, 2, 2, 1, 1.0, 1.0, 1.0, 1, 1);

  ASSERT_EQ(csr.val.size(), before.size());
  for (size_t i = 0; i < csr.val.size(); ++i) {
    EXPECT_FLOAT_EQ(csr.val[i], before[i]);
  }
}

TEST(CTACOpsTest, ApplyCTACToCSRInvalidAnglesThrows) {
  spect::io::CSR csr;
  csr.rows = 2;
  csr.cols = 4;
  csr.row_ptr = {0, 1, 1};
  csr.col_idx = {0};
  csr.val = {1.0f};

  const std::vector<double> muMap(4, 0.0);

  EXPECT_THROW(
      applyCTACToCSR(csr, muMap, 2, 2, 1, 1.0, 1.0, 1.0, 0, 1),
      std::runtime_error);
}

TEST(CTACOpsTest, ApplyChangsACZeroMuKeepsImage) {
  std::vector<double> image = {1.0, 2.0, 3.0, 4.0};
  const std::vector<double> before = image;
  const std::vector<double> muMap(4, 0.0);

  applyChangsAC(image, muMap, 2, 2, 1, 1.0, 1.0, 1.0, 4, 1);

  ASSERT_EQ(image.size(), before.size());
  for (size_t i = 0; i < image.size(); ++i) {
    EXPECT_NEAR(image[i], before[i], 1e-12);
  }
}

TEST(CTACOpsTest, LineIntegralZeroMuAllOnes) {
  const int nx = 8;
  const int ny = 8;
  const int nz = 2;
  const int numAngles = 4;
  const std::vector<double> muMap(static_cast<size_t>(nx) * ny * nz, 0.0);

  const std::vector<uint16_t> factors = computeAttenFactors(
      muMap, nx, ny, nz, 0.5, 0.5, 0.5, numAngles, 1, 1);

  ASSERT_EQ(factors.size(), static_cast<size_t>(numAngles) * muMap.size());
  for (uint16_t value : factors) {
    EXPECT_EQ(value, static_cast<uint16_t>(65535));
  }
}

// Slab phantom: attenuating slab only at y in [30, 40]. A ray leaving the
// centre voxel toward the detector at angle 0 (ray direction +Y) must cross
// the slab; the opposite view (180 deg) and the lateral view (90 deg) must
// not. This locks the angle dependence that the legacy per-voxel model lacks.
TEST(CTACOpsTest, LineIntegralSlabIsAngleDependent) {
  const int nx = 41;
  const int ny = 41;
  const int nz = 1;
  const int numAngles = 4; // theta = 0, 90, 180, 270 deg
  const double mu = 0.05;  // mm^-1
  std::vector<double> muMap(static_cast<size_t>(nx) * ny * nz, 0.0);
  for (int iy = 30; iy <= 40; ++iy)
    for (int ix = 0; ix < nx; ++ix)
      muMap[static_cast<size_t>(iy) * nx + ix] = mu;

  const std::vector<uint16_t> factors = computeAttenFactors(
      muMap, nx, ny, nz, 1.0, 1.0, 1.0, numAngles, 1, 1);

  const size_t voxels = muMap.size();
  const size_t centre = static_cast<size_t>(20) * nx + 20;

  // theta=0: ray (0,+1) crosses ~11 mm of slab -> exp(-0.55) ~= 0.577.
  const double through = factorAt(factors, voxels, 0, centre);
  EXPECT_NEAR(through, std::exp(-mu * 11.0), 0.05);
  // theta=90 (ray (-1,0)) and theta=180 (ray (0,-1)) miss the slab entirely.
  EXPECT_GT(factorAt(factors, voxels, 1, centre), 0.99);
  EXPECT_GT(factorAt(factors, voxels, 2, centre), 0.99);
  // Angle dependence must be present (legacy model would make these equal).
  EXPECT_LT(through, factorAt(factors, voxels, 2, centre) - 0.3);
}

// Doubling the voxel spacing along the ray doubles the physical path length
// through the same voxel-space slab.
TEST(CTACOpsTest, LineIntegralScalesWithVoxelSpacing) {
  const int nx = 41;
  const int ny = 41;
  const int nz = 1;
  const double mu = 0.05;
  std::vector<double> muMap(static_cast<size_t>(nx) * ny * nz, 0.0);
  for (int iy = 30; iy <= 40; ++iy)
    for (int ix = 0; ix < nx; ++ix)
      muMap[static_cast<size_t>(iy) * nx + ix] = mu;

  const std::vector<uint16_t> f1 = computeAttenFactors(
      muMap, nx, ny, nz, 1.0, 1.0, 1.0, 1, 1, 1);
  const std::vector<uint16_t> f2 = computeAttenFactors(
      muMap, nx, ny, nz, 2.0, 2.0, 2.0, 1, 1, 1);

  const size_t centre = static_cast<size_t>(20) * nx + 20;
  const double a1 = factorAt(f1, muMap.size(), 0, centre);
  const double a2 = factorAt(f2, muMap.size(), 0, centre);
  EXPECT_NEAR(a2, a1 * a1, 0.02);
}

// Slab on the +X side: at theta=90 deg the ray direction depends on the
// rotation sign ((-1,0) for +1 vs (+1,0) for -1), so only one sign crosses
// the slab. This locks the -ctacRotSign semantics.
TEST(CTACOpsTest, LineIntegralRotationSignFlipsLateralView) {
  const int nx = 41;
  const int ny = 41;
  const int nz = 1;
  const int numAngles = 4;
  const double mu = 0.05;
  std::vector<double> muMap(static_cast<size_t>(nx) * ny * nz, 0.0);
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 30; ix <= 40; ++ix)
      muMap[static_cast<size_t>(iy) * nx + ix] = mu;

  const std::vector<uint16_t> plus = computeAttenFactors(
      muMap, nx, ny, nz, 1.0, 1.0, 1.0, numAngles, 1, 1);
  const std::vector<uint16_t> minus = computeAttenFactors(
      muMap, nx, ny, nz, 1.0, 1.0, 1.0, numAngles, -1, 1);

  const size_t voxels = muMap.size();
  const size_t centre = static_cast<size_t>(20) * nx + 20;
  // sign=+1, theta=+90: ray (-1,0) misses the +X slab.
  EXPECT_GT(factorAt(plus, voxels, 1, centre), 0.99);
  // sign=-1, theta=-90: ray (+1,0) crosses it.
  EXPECT_LT(factorAt(minus, voxels, 1, centre), std::exp(-mu * 9.0));
}

} // namespace spect::main::test
