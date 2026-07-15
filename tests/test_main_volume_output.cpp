#include <gtest/gtest.h>

#include "main/VolumeOutput.h"

#include <cmath>
#include <numeric>
#include <vector>

namespace spect {
namespace main {
namespace test {

TEST(VolumeOutputTest, SanitizesNegativeCropValues) {
  const CropSettings crop = sanitizeCropSettings(-3, -1);
  EXPECT_EQ(crop.start, 0);
  EXPECT_EQ(crop.end, 0);
}

TEST(VolumeOutputTest, ValidatesCropRangeAgainstNz) {
  EXPECT_TRUE(isCropSettingsValid(4, CropSettings{1, 1}));
  EXPECT_FALSE(isCropSettingsValid(4, CropSettings{2, 2}));
  EXPECT_FALSE(isCropSettingsValid(4, CropSettings{4, 0}));
}

TEST(VolumeOutputTest, PrepareOutputAppliesMaskAndCrop) {
  const int nx = 2;
  const int ny = 1;
  const int nz = 4;
  const std::vector<double> source = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<uint8_t> mask = {1, 1, 1, 0, 0, 1, 1, 1};

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, &mask, CropSettings{1, 1}, &outNz);

  EXPECT_EQ(outNz, 2);
  ASSERT_EQ(output.size(), 4u);
  EXPECT_DOUBLE_EQ(output[0], 2.0);
  EXPECT_DOUBLE_EQ(output[1], 0.0);
  EXPECT_DOUBLE_EQ(output[2], 0.0);
  EXPECT_DOUBLE_EQ(output[3], 5.0);
}

TEST(VolumeOutputTest, PrepareOutputFlipsZWithinEachBedBeforeCrop) {
  const int nx = 1;
  const int ny = 1;
  const int nz = 6;
  const std::vector<double> source = {10, 11, 12, 20, 21, 22};
  OutputTransform transform;
  transform.flipZPerBed = true;
  transform.baseNz = 3;
  transform.bedZStarts = {0, 3};

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, &outNz, &transform);

  EXPECT_EQ(outNz, 6);
  ASSERT_EQ(output.size(), source.size());
  const std::vector<double> expected = {12, 11, 10, 22, 21, 20};
  EXPECT_EQ(output, expected);
}

TEST(VolumeOutputTest, HardCutsPhysicalOverlapAfterNonOverlappingRecon) {
  const int nx = 1;
  const int ny = 1;
  const int nz = 8;
  const std::vector<double> source = {1, 2, 3, 4, 5, 6, 7, 8};
  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.baseNz = 4;
  transform.bedZStarts = {0, 4};
  transform.outputBedOverlaps = {2};

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, &outNz, &transform);

  EXPECT_EQ(outNz, 6);
  EXPECT_EQ(output, (std::vector<double>{1, 2, 3, 6, 7, 8}));
}

TEST(VolumeOutputTest, HardCutsMaskWithTheSameSlices) {
  const int nx = 1;
  const int ny = 1;
  const int nz = 8;
  const std::vector<double> source = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<uint8_t> mask = {1, 1, 1, 1, 1, 0, 1, 1};
  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.baseNz = 4;
  transform.bedZStarts = {0, 4};
  transform.outputBedOverlaps = {2};

  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, &mask, CropSettings{0, 0}, nullptr, &transform);

  EXPECT_EQ(output, (std::vector<double>{1, 2, 3, 0, 7, 8}));
}

TEST(VolumeOutputTest, HardCutPrecedesWholeZFlipAndCrop) {
  const int nx = 1;
  const int ny = 1;
  const int nz = 8;
  const std::vector<double> source = {1, 2, 3, 4, 5, 6, 7, 8};
  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.flipZWholeOutput = true;
  transform.baseNz = 4;
  transform.bedZStarts = {0, 4};
  transform.outputBedOverlaps = {2};

  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{1, 1}, nullptr, &transform);

  EXPECT_EQ(output, (std::vector<double>{7, 6, 3, 2}));
}

TEST(VolumeOutputTest, SigmoidBlendMatchesNormalizedK12Reference) {
  const int nx = 1;
  const int ny = 1;
  const int nz = 12;
  std::vector<double> source(6, 10.0);
  source.insert(source.end(), 6, 20.0);

  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.sigmoidBlendOutput = true;
  transform.baseNz = 6;
  transform.bedZStarts = {0, 6};
  transform.outputBedOverlaps = {4};

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, &outNz, &transform);

  ASSERT_EQ(outNz, 8);
  ASSERT_EQ(output.size(), 8u);
  EXPECT_DOUBLE_EQ(output[0], 10.0);
  EXPECT_DOUBLE_EQ(output[1], 10.0);
  for (int j = 0; j < 4; ++j) {
    const double t = static_cast<double>(j) / 4.0;
    const double rightWeight = 1.0 / (1.0 + std::exp(-12.0 * (t - 0.5)));
    EXPECT_NEAR(output[static_cast<size_t>(j + 2)],
                (1.0 - rightWeight) * 10.0 + rightWeight * 20.0, 1e-12);
  }
  EXPECT_DOUBLE_EQ(output[6], 20.0);
  EXPECT_DOUBLE_EQ(output[7], 20.0);
}

TEST(VolumeOutputTest, ObjectAwarePhantomContourInterpolationFillsSeamGap) {
  const int nx = 5;
  const int ny = 5;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 1; y <= 3; ++y) {
      for (int x = 1; x <= 3; ++x)
        source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx + x] = 1.0;
    }
  }
  for (int localZ : {12, 20 + 8}) {
    for (int y = 1; y <= 3; ++y) {
      for (int x = 1; x <= 3; ++x)
        source[static_cast<size_t>(localZ) * plane + static_cast<size_t>(y) * nx + x] = 0.0;
    }
  }

  OutputTransform baselineTransform;
  baselineTransform.bedOverlapInRecon = false;
  baselineTransform.sigmoidBlendOutput = true;
  baselineTransform.baseNz = 20;
  baselineTransform.bedZStarts = {0, 20};
  baselineTransform.outputBedOverlaps = {16};
  OutputTransform processedTransform = baselineTransform;
  processedTransform.objectAwareSeamPostprocess = true;
  processedTransform.objectProfile = "phantom";
  processedTransform.seamHalfWidth = 8;

  int outNz = 0;
  const auto baseline = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, &outNz,
      &baselineTransform);
  const auto processed = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr,
      &processedTransform);

  ASSERT_EQ(outNz, 24);
  ASSERT_EQ(baseline.size(), processed.size());
  const size_t seamCenter = static_cast<size_t>(12) * plane + 2 * nx + 2;
  EXPECT_DOUBLE_EQ(baseline[seamCenter], 0.0);
  EXPECT_NEAR(processed[seamCenter], 1.0, 1e-12);
  EXPECT_EQ(processed.front(), baseline.front());
}

TEST(VolumeOutputTest, PhantomContourInterpolationKeepsTwoComponentsSeparate) {
  const int nx = 9;
  const int ny = 5;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 1; y <= 3; ++y) {
      for (int x : {0, 1, 2, 6, 7, 8})
        source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx + x] = 1.0;
    }
  }
  for (int localZ : {12, 20 + 8}) {
    for (int y = 1; y <= 3; ++y) {
      for (int x : {0, 1, 2, 6, 7, 8})
        source[static_cast<size_t>(localZ) * plane + static_cast<size_t>(y) * nx + x] = 0.0;
    }
  }

  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.sigmoidBlendOutput = true;
  transform.objectAwareSeamPostprocess = true;
  transform.objectProfile = "phantom";
  transform.baseNz = 20;
  transform.bedZStarts = {0, 20};
  transform.outputBedOverlaps = {16};
  transform.seamHalfWidth = 8;

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform);
  const size_t seam = static_cast<size_t>(12) * plane;
  EXPECT_NEAR(output[seam + 2 * nx + 1], 1.0, 1e-12);
  EXPECT_NEAR(output[seam + 2 * nx + 7], 1.0, 1e-12);
  EXPECT_DOUBLE_EQ(output[seam + 2 * nx + 4], 0.0);
}

TEST(VolumeOutputTest, ZimuSmoothingDoesNotExpandBinarySupport) {
  const int nx = 5;
  const int ny = 5;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 1; y <= 3; ++y) {
      for (int x = 1; x <= 3; ++x)
        source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx + x] = 1.0;
    }
  }
  for (int localZ : {12, 20 + 8}) {
    for (int y = 1; y <= 3; ++y) {
      for (int x = 1; x <= 3; ++x)
        source[static_cast<size_t>(localZ) * plane + static_cast<size_t>(y) * nx + x] = 2.0;
    }
  }

  OutputTransform baselineTransform;
  baselineTransform.bedOverlapInRecon = false;
  baselineTransform.sigmoidBlendOutput = true;
  baselineTransform.baseNz = 20;
  baselineTransform.bedZStarts = {0, 20};
  baselineTransform.outputBedOverlaps = {16};
  OutputTransform transform = baselineTransform;
  transform.objectAwareSeamPostprocess = true;
  transform.objectProfile = "zimu";

  const auto baseline = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr,
      &baselineTransform);
  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform);
  ASSERT_EQ(output.size(), baseline.size());
  for (size_t index = 0; index < output.size(); ++index)
    EXPECT_EQ(output[index] > 0.0, baseline[index] > 0.0);
  const size_t seamCenter = static_cast<size_t>(12) * plane + 2 * nx + 2;
  EXPECT_LT(output[seamCenter], baseline[seamCenter]);
  EXPECT_NEAR(std::accumulate(output.begin(), output.end(), 0.0),
              std::accumulate(baseline.begin(), baseline.end(), 0.0), 1e-6);
}

TEST(VolumeOutputTest, PrepareOutputFlipsWholeZAfterPerBedFlip) {
  const int nx = 1;
  const int ny = 1;
  const int nz = 6;
  const std::vector<double> source = {10, 11, 12, 20, 21, 22};
  OutputTransform transform;
  transform.flipZPerBed = true;
  transform.flipZWholeOutput = true;
  transform.baseNz = 3;
  transform.bedZStarts = {0, 3};

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, &outNz, &transform);

  EXPECT_EQ(outNz, 6);
  ASSERT_EQ(output.size(), source.size());
  const std::vector<double> expected = {20, 21, 22, 10, 11, 12};
  EXPECT_EQ(output, expected);
}

TEST(VolumeOutputTest, PrepareOutputWholeZFlipPrecedesCrop) {
  const int nx = 1;
  const int ny = 1;
  const int nz = 4;
  const std::vector<double> source = {1, 2, 3, 4};
  OutputTransform transform;
  transform.flipZWholeOutput = true;

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{1, 1}, &outNz, &transform);

  EXPECT_EQ(outNz, 2);
  const std::vector<double> expected = {3, 2};
  EXPECT_EQ(output, expected);
}

TEST(VolumeOutputTest, PrepareOutputFlipsWholeX) {
  const int nx = 3;
  const int ny = 2;
  const int nz = 1;
  const std::vector<double> source = {1, 2, 3, 4, 5, 6};
  OutputTransform transform;
  transform.flipXWholeOutput = true;

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, &outNz, &transform);

  EXPECT_EQ(outNz, 1);
  const std::vector<double> expected = {3, 2, 1, 6, 5, 4};
  EXPECT_EQ(output, expected);
}

TEST(VolumeOutputTest, AppliesPhysicalCircularFovWithFourVoxelTaper) {
  const int nx = 11;
  const int ny = 11;
  const std::vector<double> source(static_cast<size_t>(nx) * ny, 1.0);
  OutputTransform transform;
  transform.fovRadiusMm = 2.0;

  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, 1, nullptr, CropSettings{0, 0}, nullptr, &transform,
      1.0, 1.0);

  const auto at = [&](int x, int y) { return output[static_cast<size_t>(y) * nx + x]; };
  EXPECT_DOUBLE_EQ(at(5, 5), 1.0);
  EXPECT_DOUBLE_EQ(at(7, 5), 1.0);
  EXPECT_DOUBLE_EQ(at(8, 5), 0.75);
  EXPECT_DOUBLE_EQ(at(10, 5), 0.25);
  EXPECT_DOUBLE_EQ(at(0, 0), 0.0);
}

TEST(VolumeOutputTest, ThrowsWhenSourceSizeMismatch) {
  const std::vector<double> source = {1.0, 2.0, 3.0};
  EXPECT_THROW(
      (void)prepareOutputVolume(source, 2, 2, 2, nullptr, CropSettings{0, 0}),
      std::invalid_argument);
}

} // namespace test
} // namespace main
} // namespace spect
