#include <gtest/gtest.h>

#include "core/AttenuationCoefficients.h"
#include "ctac/AutoMuMap.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace spect::main::test {

namespace {

constexpr double kWaterMu = 0.0152; // mm^-1

size_t idx3(int ix, int iy, int iz, int nx, int ny) {
  return static_cast<size_t>(iz) * nx * ny + static_cast<size_t>(iy) * nx +
         static_cast<size_t>(ix);
}

/// Activity cylinder along Z, centred in the volume.
std::vector<double> makeCylinderActivity(int nx, int ny, int nz, double radius,
                                         double value) {
  std::vector<double> act(static_cast<size_t>(nx) * ny * nz, 0.0);
  const double cx = (nx - 1) / 2.0;
  const double cy = (ny - 1) / 2.0;
  for (int iz = 0; iz < nz; ++iz)
    for (int iy = 0; iy < ny; ++iy)
      for (int ix = 0; ix < nx; ++ix) {
        const double r = std::hypot(ix - cx, iy - cy);
        if (r <= radius)
          act[idx3(ix, iy, iz, nx, ny)] = value;
      }
  return act;
}

} // namespace

TEST(AutoMuMapTest, EmptyActivityGivesEmptyMuMap) {
  const int n = 8;
  const std::vector<double> act(static_cast<size_t>(n) * n * n, 0.0);
  const AutoMuMapResult r =
      buildAutoMuMapFromActivity(act, n, n, n, 0.05, kWaterMu);
  EXPECT_EQ(r.bodyVoxels, 0u);
  for (double v : r.muMap)
    EXPECT_EQ(v, 0.0);
}

TEST(AutoMuMapTest, UniformCylinderSegmentedToWater) {
  const int nx = 32, ny = 32, nz = 12;
  const double radius = 10.0;
  const std::vector<double> act =
      makeCylinderActivity(nx, ny, nz, radius, 1.0);
  const AutoMuMapResult r =
      buildAutoMuMapFromActivity(act, nx, ny, nz, 0.05, kWaterMu);

  // Every activity voxel must be water; far outside must stay zero.
  size_t expected = 0;
  for (size_t i = 0; i < act.size(); ++i)
    if (act[i] > 0.0) {
      ++expected;
      EXPECT_DOUBLE_EQ(r.muMap[i], kWaterMu);
    }
  EXPECT_GE(r.bodyVoxels, expected);
  // Closing may add a thin rim; it must stay bounded (< r+2 voxel radius).
  const double maxBody =
      3.14159265 * (radius + 2.0) * (radius + 2.0) * static_cast<double>(nz);
  EXPECT_LT(static_cast<double>(r.bodyVoxels), maxBody);
  EXPECT_EQ(r.muMap[idx3(0, 0, 0, nx, ny)], 0.0);
  EXPECT_EQ(r.muMap[idx3(nx - 1, ny - 1, nz - 1, nx, ny)], 0.0);
}

// Cold cavity inside an active shell must be filled (it still attenuates).
TEST(AutoMuMapTest, ColdCoreIsHoleFilled) {
  const int nx = 32, ny = 32, nz = 12;
  std::vector<double> act = makeCylinderActivity(nx, ny, nz, 10.0, 1.0);
  const double cx = (nx - 1) / 2.0;
  const double cy = (ny - 1) / 2.0;
  // Carve a cold core of radius 4 in the middle Z range only, so the cavity
  // is fully enclosed by active voxels (not open to the volume border).
  for (int iz = 2; iz < nz - 2; ++iz)
    for (int iy = 0; iy < ny; ++iy)
      for (int ix = 0; ix < nx; ++ix)
        if (std::hypot(ix - cx, iy - cy) <= 4.0)
          act[idx3(ix, iy, iz, nx, ny)] = 0.0;

  const AutoMuMapResult r =
      buildAutoMuMapFromActivity(act, nx, ny, nz, 0.05, kWaterMu);
  EXPECT_DOUBLE_EQ(
      r.muMap[idx3(static_cast<int>(cx), static_cast<int>(cy), nz / 2, nx, ny)],
      kWaterMu);
}

// A tiny disconnected hot speck must be dropped by largest-component
// selection; only the main cylinder survives.
TEST(AutoMuMapTest, DisconnectedSpeckleRemoved) {
  const int nx = 40, ny = 40, nz = 10;
  std::vector<double> act = makeCylinderActivity(nx, ny, nz, 8.0, 1.0);
  act[idx3(1, 1, 1, nx, ny)] = 1.0; // isolated speck far from the cylinder

  const AutoMuMapResult r =
      buildAutoMuMapFromActivity(act, nx, ny, nz, 0.05, kWaterMu,
                                 /*fovMask=*/nullptr,
                                 /*closingRadiusVoxels=*/1);
  EXPECT_EQ(r.muMap[idx3(1, 1, 1, nx, ny)], 0.0);
  EXPECT_DOUBLE_EQ(r.muMap[idx3(nx / 2, ny / 2, nz / 2, nx, ny)], kWaterMu);
}

TEST(AutoMuMapTest, FovMaskExcludesOutsideActivity) {
  const int nx = 40, ny = 40, nz = 10;
  std::vector<double> act = makeCylinderActivity(nx, ny, nz, 8.0, 1.0);
  // Strong artefact outside the FOV (e.g. edge streak) that would otherwise
  // dominate the largest component.
  for (int iz = 0; iz < nz; ++iz)
    for (int iy = 0; iy < ny; ++iy)
      for (int ix = 36; ix < nx; ++ix)
        act[idx3(ix, iy, iz, nx, ny)] = 5.0;

  std::vector<uint8_t> fov(act.size(), 0);
  const double cx = (nx - 1) / 2.0;
  const double cy = (ny - 1) / 2.0;
  for (int iz = 0; iz < nz; ++iz)
    for (int iy = 0; iy < ny; ++iy)
      for (int ix = 0; ix < nx; ++ix)
        if (std::hypot(ix - cx, iy - cy) <= 12.0)
          fov[idx3(ix, iy, iz, nx, ny)] = 1;

  const AutoMuMapResult r =
      buildAutoMuMapFromActivity(act, nx, ny, nz, 0.05, kWaterMu, &fov);
  EXPECT_EQ(r.muMap[idx3(38, ny / 2, nz / 2, nx, ny)], 0.0);
  EXPECT_DOUBLE_EQ(r.muMap[idx3(nx / 2, ny / 2, nz / 2, nx, ny)], kWaterMu);
}

TEST(AutoMuMapTest, WaterValueParameterRespected) {
  const int n = 16;
  const std::vector<double> act = makeCylinderActivity(n, n, n, 5.0, 2.0);
  const double mu = 0.0120; // broad-beam effective value
  const AutoMuMapResult r = buildAutoMuMapFromActivity(act, n, n, n, 0.05, mu);
  EXPECT_DOUBLE_EQ(r.muMap[idx3(n / 2, n / 2, n / 2, n, n)], mu);
}

TEST(AutoMuMapTest, WaterMuNuclideLookupUsesCentralTable) {
  double mu = 0.0;
  EXPECT_TRUE(spect::core::waterMuPerMMForNuclide("Tc-99m", mu));
  EXPECT_DOUBLE_EQ(mu, kWaterMu);
  EXPECT_TRUE(spect::core::waterMuPerMMForNuclide("I123", mu));
  EXPECT_NEAR(mu, 0.014766221821206932, 1e-15);
  EXPECT_TRUE(spect::core::waterMuPerMMForNuclide("In111_245", mu));
  EXPECT_NEAR(mu, 0.012738504550836008, 1e-15);
  EXPECT_FALSE(spect::core::waterMuPerMMForNuclide("unknown", mu));
}

TEST(AutoMuMapTest, InvalidArgumentsThrow) {
  const std::vector<double> act(8, 1.0);
  EXPECT_THROW(buildAutoMuMapFromActivity(act, 2, 2, 3, 0.05, kWaterMu),
               std::runtime_error);
  EXPECT_THROW(buildAutoMuMapFromActivity(act, 2, 2, 2, 0.0, kWaterMu),
               std::runtime_error);
  EXPECT_THROW(buildAutoMuMapFromActivity(act, 2, 2, 2, 1.5, kWaterMu),
               std::runtime_error);
  EXPECT_THROW(buildAutoMuMapFromActivity(act, 2, 2, 2, 0.05, -1.0),
               std::runtime_error);
}

} // namespace spect::main::test
