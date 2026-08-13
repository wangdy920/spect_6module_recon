#include <gtest/gtest.h>

#include "pipeline/VolumeOutput.h"
#include "recon/ReconLoopHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spect {
namespace main {
namespace test {

TEST(VolumeOutputTest, ZeroCheckpointCanFailClosed) {
  const char *previous = std::getenv("SPECT_FAIL_ON_ZERO_CHECKPOINT");
  const std::string saved = previous ? previous : "";
#ifdef _WIN32
  _putenv_s("SPECT_FAIL_ON_ZERO_CHECKPOINT", "1");
#else
  setenv("SPECT_FAIL_ON_ZERO_CHECKPOINT", "1", 1);
#endif
  EXPECT_THROW(
      guardZeroMaxCheckpoint(std::vector<double>(4, 0.0), "test"),
      std::runtime_error);
#ifdef _WIN32
  _putenv_s("SPECT_FAIL_ON_ZERO_CHECKPOINT", previous ? saved.c_str() : "");
#else
  if (previous)
    setenv("SPECT_FAIL_ON_ZERO_CHECKPOINT", saved.c_str(), 1);
  else
    unsetenv("SPECT_FAIL_ON_ZERO_CHECKPOINT");
#endif
}

TEST(VolumeOutputTest, SanitizesAndValidatesCropRange) {
  const CropSettings crop = sanitizeCropSettings(-3, -1);
  EXPECT_EQ(crop.start, 0);
  EXPECT_EQ(crop.end, 0);
  EXPECT_TRUE(isCropSettingsValid(4, CropSettings{1, 1}));
  EXPECT_FALSE(isCropSettingsValid(4, CropSettings{2, 2}));
  EXPECT_FALSE(isCropSettingsValid(4, CropSettings{4, 0}));
}

TEST(VolumeOutputTest, SaveVolumeUsesFixedCenterCrop) {
  namespace fs = std::filesystem;
  const fs::path dir =
      fs::temp_directory_path() / "spect_fixed_geometry_crop_test";
  fs::remove_all(dir);
  fs::create_directories(dir);

  constexpr int nx = 10;
  constexpr int ny = 8;
  constexpr int nz = 6;
  std::vector<double> source(static_cast<size_t>(nx) * ny * nz);
  std::iota(source.begin(), source.end(), 0.0);
  OutputTransform transform;
  transform.centerCropNx = 8;
  transform.centerCropNy = 6;
  transform.centerCropNz = 4;
  transform.baseNz = nz;
  transform.bedZStarts = {0};

  saveVolumeOutput(source, OutputGeometry{nx, ny, nz, 0.3, 0.3, 0.3},
                   dir.string(), "crop", 2, nullptr, CropSettings{0, 0},
                   true, &transform);

  std::ifstream mhd(dir / "crop_it2.mhd");
  const std::string header((std::istreambuf_iterator<char>(mhd)),
                           std::istreambuf_iterator<char>());
  mhd.close();
  EXPECT_NE(header.find("DimSize = 8 6 4"), std::string::npos);
  EXPECT_EQ(fs::file_size(dir / "crop_it2.raw"),
            static_cast<uintmax_t>(8 * 6 * 4 * sizeof(double)));
  std::ifstream raw(dir / "crop_it2.raw", std::ios::binary);
  double first = -1.0;
  raw.read(reinterpret_cast<char *>(&first), sizeof(first));
  raw.close();
  EXPECT_DOUBLE_EQ(first, 91.0);

  fs::remove_all(dir);
}

TEST(VolumeOutputTest, OutputClampCanBeDisabledForLinearParity) {
  namespace fs = std::filesystem;
  const fs::path dir =
      fs::temp_directory_path() / "spect_output_clamp_policy_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const std::vector<double> source{-2.0, 3.0};
  OutputFilterSettings filters;
  filters.nonnegativeClamp = false;
  saveVolumeOutput(source, OutputGeometry{2, 1, 1, 1.0, 1.0, 1.0},
                   dir.string(), "linear", 1, nullptr, CropSettings{}, true,
                   nullptr, &filters);
  filters.nonnegativeClamp = true;
  saveVolumeOutput(source, OutputGeometry{2, 1, 1, 1.0, 1.0, 1.0},
                   dir.string(), "clamped", 1, nullptr, CropSettings{}, true,
                   nullptr, &filters);

  auto firstValue = [](const fs::path &path) {
    std::ifstream raw(path, std::ios::binary);
    double value = 0.0;
    raw.read(reinterpret_cast<char *>(&value), sizeof(value));
    return value;
  };
  EXPECT_DOUBLE_EQ(firstValue(dir / "linear_it1.raw"), -2.0);
  EXPECT_DOUBLE_EQ(firstValue(dir / "clamped_it1.raw"), 0.0);
  fs::remove_all(dir);
}

TEST(VolumeOutputTest, PrepareOutputAppliesMaskAndZeroesEndSlices) {
  const std::vector<double> source = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<uint8_t> mask = {1, 1, 1, 0, 0, 1, 1, 1};

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, 2, 1, 4, &mask, CropSettings{1, 1}, &outNz);

  EXPECT_EQ(outNz, 4);
  EXPECT_EQ(output,
            (std::vector<double>{0, 0, 2, 0, 0, 5, 0, 0}));
}

TEST(VolumeOutputTest, PrepareOutputFlipsSingleBedZBeforeCrop) {
  const std::vector<double> source = {10, 11, 12, 13, 14, 15};
  OutputTransform transform;
  transform.flipZPerBed = true;
  transform.baseNz = 6;
  transform.bedZStarts = {0};

  const auto output = prepareOutputVolume(
      source, 1, 1, 6, nullptr, CropSettings{1, 1}, nullptr, &transform);
  EXPECT_EQ(output, (std::vector<double>{0, 14, 13, 12, 11, 0}));
}

TEST(VolumeOutputTest, CanonicalStitchFitsPrototypeBedGain) {
  constexpr int baseNz = 70;
  std::vector<double> source(2 * baseNz, -100.0);
  std::fill(source.begin() + 2, source.begin() + 68, 10.0);
  std::fill(source.begin() + baseNz + 2,
            source.begin() + baseNz + 68, 20.0);

  OutputTransform transform;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {14};
  transform.canonicalGeom = {66, 56, 10};

  int outNz = 0;
  const auto output = prepareOutputVolume(
      source, 1, 1, 2 * baseNz, nullptr, CropSettings{0, 0}, &outNz,
      &transform, 0.3, 0.3);

  ASSERT_EQ(outNz, 122);
  ASSERT_EQ(output.size(), 122u);
  for (double value : output)
    EXPECT_NEAR(value, 10.0, 1e-12);
}

TEST(VolumeOutputTest, CanonicalGainFitUsesReversedStorageOverlapEnds) {
  constexpr int nx = 5;
  constexpr int ny = 5;
  constexpr int baseNz = 70;
  constexpr int cropStart = 2;
  constexpr int stepLayers = 56;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(2 * baseNz * plane, 1.0);
  for (int z = 0; z < 10; ++z) {
    std::fill_n(source.begin() +
                    static_cast<std::ptrdiff_t>(cropStart + stepLayers + z) *
                        plane,
                plane, 10.0);
    std::fill_n(source.begin() +
                    static_cast<std::ptrdiff_t>(baseNz + cropStart + z) *
                        plane,
                plane, 20.0);
  }

  OutputTransform transform;
  transform.reverseLocalZInReconStorage = true;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.canonicalGeom = {66, stepLayers, 10};

  const auto gains = fitCenter66BedGains(
      source, nx, ny, transform, cropStart, 0.3, 0.3);
  ASSERT_EQ(gains.size(), 2u);
  EXPECT_NEAR(gains[0], 1.0, 1e-12);
  EXPECT_NEAR(gains[1], 0.5, 1e-12);
}

TEST(VolumeOutputTest, CanonicalStitchKeepsOutputZContinuityUnderReversedStorage) {
  constexpr int baseNz = 70;
  constexpr int cropStart = 2;
  constexpr int stepLayers = 56;
  std::vector<double> source(2 * baseNz, 10.0);
  source[cropStart + 63] = 2.0;          // high bed physical crop layer 2
  source[baseNz + cropStart + 7] = 6.0; // low bed physical crop layer 58

  OutputTransform transform;
  transform.reverseLocalZInReconStorage = true;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {14};
  transform.canonicalGeom = {66, stepLayers, 10};

  const auto output = prepareOutputVolume(
      source, 1, 1, 2 * baseNz, nullptr, CropSettings{0, 0}, nullptr,
      &transform, 0.3, 0.3);
  constexpr double pi = 3.14159265358979323846;
  const double lowWeight = std::sin(pi / 9.0) * std::sin(pi / 9.0);
  EXPECT_NEAR(output[40], 10.0, 1e-12);
  EXPECT_NEAR(output[63], 2.0 * lowWeight + 6.0 * (1.0 - lowWeight),
              1e-12);
}

TEST(VolumeOutputTest, CanonicalGainFitRejectsSingleOverlapOutlier) {
  constexpr int nx = 5;
  constexpr int ny = 5;
  constexpr int baseNz = 70;
  constexpr int nz = 2 * baseNz;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 2; z < 68; ++z) {
    std::fill_n(source.begin() + static_cast<std::ptrdiff_t>(z) * plane,
                plane, 10.0);
    std::fill_n(source.begin() +
                    static_cast<std::ptrdiff_t>(baseNz + z) * plane,
                plane, 20.0);
  }
  source[(baseNz + 2) * plane + 2 * nx + 2] = 200000.0;

  OutputTransform transform;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {14};
  transform.canonicalGeom = {66, 56, 10};

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform,
      0.3, 0.3);
  EXPECT_NEAR(output[56 * plane], 10.0, 1e-12);
  EXPECT_NEAR(output.back(), 10.0, 1e-12);
}

TEST(VolumeOutputTest, CanonicalSparseGainFallbackUsesFixedCroppedXY) {
  constexpr int nx = 120;
  constexpr int ny = 120;
  constexpr int baseNz = 70;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(2 * baseNz * plane, 0.0);

  constexpr int blendOverlap = 10; // stepLayers=56, cropLayers=66 => blend=10
  for (int z = 0; z < blendOverlap; ++z) {
    const size_t highBase = static_cast<size_t>(2 + z) * plane;
    const size_t lowBase = static_cast<size_t>(baseNz + 58 + z) * plane;
    const size_t center = static_cast<size_t>(60) * nx + 60;
    source[highBase + center] = 10.0;
    source[lowBase + center] = 20.0;
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        if (x >= 10 && x < 110 && y >= 10 && y < 110)
          continue;
        const size_t xy = static_cast<size_t>(y) * nx + x;
        source[highBase + xy] = 10.0;
        source[lowBase + xy] = 1000.0;
      }
    }
  }

  OutputTransform transform;
  transform.centerCropNx = 100;
  transform.centerCropNy = 100;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.canonicalGeom = {66, 56, 10};

  const auto gains = fitCenter66BedGains(
      source, nx, ny, transform, 2, 0.3, 0.3);
  ASSERT_EQ(gains.size(), 2u);
  EXPECT_NEAR(gains[0], 1.0, 1e-12);
  EXPECT_NEAR(gains[1], 0.5, 1e-12);
}

TEST(VolumeOutputTest, CanonicalGainFitAppliesChainAboveTwo) {
  // Prototype behavior: the full log-ratio chain is always applied, even
  // when an adjacent ratio exceeds 2x (bed1=10 vs bed2=100 -> gain 0.1).
  constexpr int nx = 5;
  constexpr int ny = 5;
  constexpr int baseNz = 70;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(2 * baseNz * plane, 0.0);
  for (int z = 2; z < 68; ++z) {
    std::fill_n(source.begin() + static_cast<std::ptrdiff_t>(z) * plane,
                plane, 10.0);
    std::fill_n(source.begin() +
                    static_cast<std::ptrdiff_t>(baseNz + z) * plane,
                plane, 100.0);
  }
  OutputTransform transform;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {14};
  transform.canonicalGeom = {66, 56, 10};

  const auto gains = fitCenter66BedGains(
      source, nx, ny, transform, 2, 0.3, 0.3);
  EXPECT_NEAR(gains[0], 1.0, 1e-12);
  EXPECT_NEAR(gains[1], 0.1, 1e-12);
}

TEST(VolumeOutputTest, CanonicalStitchCropsMaskWithSameGeometry) {
  constexpr int baseNz = 70;
  const std::vector<double> source(2 * baseNz, 1.0);
  std::vector<uint8_t> mask(2 * baseNz, 0);
  mask[2] = 1;
  mask[baseNz + 67] = 1;
  OutputTransform transform;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {14};
  transform.canonicalGeom = {66, 56, 10};

  const auto output = prepareOutputVolume(
      source, 1, 1, 2 * baseNz, &mask, CropSettings{0, 0}, nullptr,
      &transform);

  ASSERT_EQ(output.size(), 122u);
  EXPECT_DOUBLE_EQ(output.front(), 1.0);
  EXPECT_DOUBLE_EQ(output.back(), 1.0);
  EXPECT_EQ(std::count(output.begin(), output.end(), 1.0), 2);
}

TEST(VolumeOutputTest, CanonicalStitchRejectsNon56LayerStep) {
  constexpr int baseNz = 70;
  const std::vector<double> source(2 * baseNz, 1.0);
  OutputTransform transform;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.canonicalGeom = {66, 56, 10};

  EXPECT_THROW((void)prepareOutputVolume(
                   source, 1, 1, 2 * baseNz, nullptr, CropSettings{0, 0},
                   nullptr, &transform),
               std::invalid_argument);
}

TEST(VolumeOutputTest, WholeAxisFlipsPrecedeEndSliceZeroing) {
  const std::vector<double> zSource = {1, 2, 3, 4};
  OutputTransform zTransform;
  zTransform.flipZWholeOutput = true;
  EXPECT_EQ(prepareOutputVolume(zSource, 1, 1, 4, nullptr,
                                CropSettings{1, 1}, nullptr, &zTransform),
            (std::vector<double>{0, 3, 2, 0}));

  const std::vector<double> xSource = {1, 2, 3, 4, 5, 6};
  OutputTransform xTransform;
  xTransform.flipXWholeOutput = true;
  EXPECT_EQ(prepareOutputVolume(xSource, 3, 2, 1, nullptr,
                                CropSettings{0, 0}, nullptr, &xTransform),
            (std::vector<double>{3, 2, 1, 6, 5, 4}));
}

TEST(VolumeOutputTest, ThrowsWhenSourceSizeMismatch) {
  const std::vector<double> source = {1.0, 2.0, 3.0};
  EXPECT_THROW(
      (void)prepareOutputVolume(source, 2, 2, 2, nullptr,
                                CropSettings{0, 0}),
      std::invalid_argument);
}

TEST(VolumeOutputTest, SeamXyAlignRemovesIntegerOverlapTranslation) {
  constexpr int nx = 40;
  constexpr int ny = 40;
  constexpr int baseNz = 70;
  constexpr int x0 = 18;
  constexpr int y0 = 20;
  constexpr int dxBlob = 3;
  constexpr int dyBlob = -2;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(2 * baseNz * plane, 0.0);
  auto paint = [&](int bed, int cx, int cy) {
    const int z0 = bed * baseNz;
    for (int z = 0; z < baseNz; ++z) {
      for (int y = cy - 5; y <= cy + 5; ++y) {
        for (int x = cx - 5; x <= cx + 5; ++x) {
          const int ddx = x - cx;
          const int ddy = y - cy;
          source[static_cast<size_t>(z0 + z) * plane +
                 static_cast<size_t>(y) * nx + x] =
              100.0 - 2.0 * (ddx * ddx + ddy * ddy);
        }
      }
    }
  };
  paint(0, x0, y0);
  paint(1, x0 + dxBlob, y0 + dyBlob);

  OutputTransform transform;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {14};
  transform.canonicalGeom = {66, 56, 10};

  const auto jumped = prepareOutputVolume(
      source, nx, ny, 2 * baseNz, nullptr, CropSettings{0, 0}, nullptr,
      &transform, 0.3, 0.3);
  auto argmaxXY = [&](const std::vector<double> &vol, int z) {
    const size_t off = static_cast<size_t>(z) * plane;
    size_t best = 0;
    double peak = -1.0;
    for (size_t i = 0; i < plane; ++i) {
      if (vol[off + i] > peak) {
        peak = vol[off + i];
        best = i;
      }
    }
    return std::pair<int, int>{static_cast<int>(best % nx),
                               static_cast<int>(best / nx)};
  };
  const auto before = argmaxXY(jumped, 40);
  const auto after = argmaxXY(jumped, 80);
  EXPECT_EQ(before.first, x0);
  EXPECT_EQ(before.second, y0);
  EXPECT_EQ(after.first, x0 + dxBlob);
  EXPECT_EQ(after.second, y0 + dyBlob);

  transform.seamXyAlign = true;
  const auto aligned = prepareOutputVolume(
      source, nx, ny, 2 * baseNz, nullptr, CropSettings{0, 0}, nullptr,
      &transform, 0.3, 0.3);
  const auto alignedBefore = argmaxXY(aligned, 40);
  const auto alignedAfter = argmaxXY(aligned, 80);
  EXPECT_EQ(alignedBefore.first, x0);
  EXPECT_EQ(alignedBefore.second, y0);
  EXPECT_EQ(alignedAfter.first, x0);
  EXPECT_EQ(alignedAfter.second, y0);

  transform.reverseLocalZInReconStorage = true;
  const auto alignedRev = prepareOutputVolume(
      source, nx, ny, 2 * baseNz, nullptr, CropSettings{0, 0}, nullptr,
      &transform, 0.3, 0.3);
  EXPECT_EQ(argmaxXY(alignedRev, 40).first, x0);
  EXPECT_EQ(argmaxXY(alignedRev, 40).second, y0);
  EXPECT_EQ(argmaxXY(alignedRev, 80).first, x0);
  EXPECT_EQ(argmaxXY(alignedRev, 80).second, y0);
}

} // namespace test
} // namespace main
} // namespace spect
