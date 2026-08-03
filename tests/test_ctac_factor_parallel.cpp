#include <gtest/gtest.h>

#include "ctac/CTACOps.h"
#include "cuda/CudaConfig.h"

#include <cstdint>
#include <vector>

namespace spect::main {
namespace {

class CTACFactorParallelTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!spect::cuda::isCudaAvailable()) {
      GTEST_SKIP() << "CUDA is not available";
    }
  }
};

TEST_F(CTACFactorParallelTest, MultiBedMatchesSerial) {
  // Two small beds: a uniform slab and a smaller internal slab.
  const int nx = 4;
  const int ny = 4;
  const int nz = 4;
  const int numAngles = 8;
  const size_t voxels = static_cast<size_t>(nx) * ny * nz;

  std::vector<double> muBed0(voxels, 0.0);
  std::vector<double> muBed1(voxels, 0.0);
  // Bed 0: half-volume attenuator.
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny / 2; ++y) {
      for (int x = 0; x < nx; ++x) {
        muBed0[static_cast<size_t>(z * ny + y) * nx + x] = 0.01;
      }
    }
  }
  // Bed 1: quarter-volume attenuator.
  for (int z = 0; z < nz / 2; ++z) {
    for (int y = 0; y < ny / 2; ++y) {
      for (int x = 0; x < nx / 2; ++x) {
        muBed1[static_cast<size_t>(z * ny + y) * nx + x] = 0.02;
      }
    }
  }

  const std::vector<std::vector<double>> muMaps = {muBed0, muBed1};

  const auto serial0 = computeAttenFactors(muBed0, nx, ny, nz, 1.0, 1.0, 1.0,
                                           numAngles, 1, 1);
  const auto serial1 = computeAttenFactors(muBed1, nx, ny, nz, 1.0, 1.0, 1.0,
                                           numAngles, 1, 1);

  const auto parallel = computeAttenFactorsMultiBed(
      muMaps, nx, ny, nz, 1.0, 1.0, 1.0, numAngles, 1, 4, 0);

  ASSERT_EQ(parallel.size(), 2u);
  ASSERT_EQ(parallel[0].size(), serial0.size());
  ASSERT_EQ(parallel[1].size(), serial1.size());

  for (size_t i = 0; i < serial0.size(); ++i) {
    EXPECT_EQ(parallel[0][i], serial0[i]) << "bed0 mismatch at " << i;
  }
  for (size_t i = 0; i < serial1.size(); ++i) {
    EXPECT_EQ(parallel[1][i], serial1[i]) << "bed1 mismatch at " << i;
  }
}

TEST_F(CTACFactorParallelTest, SingleBedEquivalentToDirect) {
  const int nx = 3;
  const int ny = 3;
  const int nz = 3;
  const int numAngles = 4;
  const size_t voxels = static_cast<size_t>(nx) * ny * nz;

  std::vector<double> mu(voxels, 0.0);
  for (size_t i = 0; i < voxels / 2; ++i) {
    mu[i] = 0.015;
  }

  const auto direct = computeAttenFactors(mu, nx, ny, nz, 1.0, 1.0, 1.0,
                                          numAngles, -1, 1);
  const auto parallel = computeAttenFactorsMultiBed(
      {mu}, nx, ny, nz, 1.0, 1.0, 1.0, numAngles, -1, 4, 0);

  ASSERT_EQ(parallel.size(), 1u);
  ASSERT_EQ(parallel[0].size(), direct.size());
  for (size_t i = 0; i < direct.size(); ++i) {
    EXPECT_EQ(parallel[0][i], direct[i]) << "single-bed mismatch at " << i;
  }
}

} // namespace
} // namespace spect::main
