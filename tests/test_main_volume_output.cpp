#include <gtest/gtest.h>

#include "main/VolumeOutput.h"

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

TEST(VolumeOutputTest, ThrowsWhenSourceSizeMismatch) {
  const std::vector<double> source = {1.0, 2.0, 3.0};
  EXPECT_THROW(
      (void)prepareOutputVolume(source, 2, 2, 2, nullptr, CropSettings{0, 0}),
      std::invalid_argument);
}

} // namespace test
} // namespace main
} // namespace spect
