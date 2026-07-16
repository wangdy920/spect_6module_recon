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

TEST(VolumeOutputTest, PrepareOutputAppliesMaskAndZeroesEndSlices) {
  const int nx = 2;
  const int ny = 1;
  const int nz = 4;
  const std::vector<double> source = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<uint8_t> mask = {1, 1, 1, 0, 0, 1, 1, 1};

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, &mask, CropSettings{1, 1}, &outNz);

  EXPECT_EQ(outNz, 4);
  ASSERT_EQ(output.size(), source.size());
  EXPECT_EQ(output,
            (std::vector<double>{0, 0, 2, 0, 0, 5, 0, 0}));
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

TEST(VolumeOutputTest, HardCutAndWholeZFlipPrecedeEndSliceZeroing) {
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

  EXPECT_EQ(output, (std::vector<double>{0, 7, 6, 3, 2, 0}));
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

TEST(VolumeOutputTest, ObjectAwarePostprocessAcceptsSupportedStepOverlaps) {
  const int nx = 5;
  const int ny = 5;
  const int baseNz = 80;
  const int nz = 2 * baseNz;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<int> overlaps = {21}; // 17.7 mm stride at 0.3 mm/voxel.
  for (int stride = 2; stride <= baseNz; stride += 2)
    overlaps.push_back(baseNz - stride); // Every 0.6 mm step through 24 mm.
  for (int overlap : overlaps) {
    std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
    for (int z = 0; z < nz; ++z) {
      for (int y = 1; y <= 3; ++y) {
        for (int x = 1; x <= 3; ++x)
          source[static_cast<size_t>(z) * plane +
                 static_cast<size_t>(y) * nx + x] = 1.0;
      }
    }
    if (overlap > 0) {
      for (int localZ : {baseNz - (overlap + 1) / 2,
                         baseNz + overlap / 2}) {
        for (int y = 1; y <= 3; ++y) {
          for (int x = 1; x <= 3; ++x)
            source[static_cast<size_t>(localZ) * plane +
                   static_cast<size_t>(y) * nx + x] = 0.0;
        }
      }
    }

    OutputTransform transform;
    transform.bedOverlapInRecon = false;
    transform.sigmoidBlendOutput = true;
    transform.objectAwareSeamPostprocess = true;
    transform.objectProfile = "phantom";
    transform.baseNz = baseNz;
    transform.bedZStarts = {0, baseNz};
    transform.outputBedOverlaps = {overlap};
    transform.seamHalfWidth = 8;

    int outNz = 0;
    const auto output = prepareOutputVolume(
        source, nx, ny, nz, nullptr, CropSettings{0, 0}, &outNz, &transform);

    EXPECT_EQ(outNz, nz - overlap);
    const int seam = baseNz - overlap + overlap / 2;
    const size_t seamCenter = static_cast<size_t>(seam) * plane + 2 * nx + 2;
    EXPECT_NEAR(output[seamCenter], 1.0, 1e-12);
  }
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

TEST(VolumeOutputTest, FovPrecedesPhantomComponentGate) {
  const int nx = 21;
  const int ny = 21;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 9; y <= 11; ++y) {
      for (int x : {6, 7, 8, 12, 13, 14})
        source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx + x] = 1.0;
    }
  }
  // Unequal corner artifacts would make the phantom gate see 3 vs 4
  // components if the final circular FOV were applied too late.
  for (int z = 0; z < 40; ++z) {
    for (int y = 0; y <= 2; ++y) {
      for (int x = 0; x <= 2; ++x)
        source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx + x] = 1.0;
      if (z >= 20) {
        for (int x = 18; x <= 20; ++x)
          source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx + x] = 1.0;
      }
    }
  }
  for (int localZ : {12, 28}) {
    for (int y = 9; y <= 11; ++y) {
      for (int x : {6, 7, 8, 12, 13, 14})
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
  transform.fovRadiusMm = 7.0;

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform,
      1.0, 1.0);
  const size_t seam = static_cast<size_t>(12) * plane;
  EXPECT_GT(output[seam + 10 * nx + 7], 0.9);
  EXPECT_GT(output[seam + 10 * nx + 13], 0.9);
  EXPECT_DOUBLE_EQ(output[seam], 0.0);
}

TEST(VolumeOutputTest, AutoClassifiesEpTubeAsPhantomWhenFovComesFirst) {
  // Single-cylinder EPguan-like support: one compact coronal component.
  // Peripheral FOV-outside arcs must not push auto classification into the
  // gray zone or skip the phantom contour gate.
  const int nx = 21;
  const int ny = 21;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 8; y <= 12; ++y) {
      for (int x = 8; x <= 12; ++x) {
        const double dy = y - 10.0;
        const double dx = x - 10.0;
        if (dy * dy + dx * dx <= 6.25)
          source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx +
                 x] = 1.0;
      }
    }
  }
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y <= 2; ++y) {
      for (int x = 0; x <= 2; ++x)
        source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx +
               x] = 0.8;
      if (z >= 20) {
        for (int x = 18; x <= 20; ++x)
          source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx +
                 x] = 0.8;
      }
    }
  }
  for (int localZ : {12, 28}) {
    for (int y = 8; y <= 12; ++y) {
      for (int x = 8; x <= 12; ++x)
        source[static_cast<size_t>(localZ) * plane +
               static_cast<size_t>(y) * nx + x] = 0.0;
    }
  }

  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.sigmoidBlendOutput = true;
  transform.objectAwareSeamPostprocess = true;
  transform.objectProfile = "auto";
  transform.baseNz = 20;
  transform.bedZStarts = {0, 20};
  transform.outputBedOverlaps = {16};
  transform.seamHalfWidth = 8;
  transform.fovRadiusMm = 7.0;

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform,
      1.0, 1.0);
  const size_t seam = static_cast<size_t>(12) * plane;
  EXPECT_GT(output[seam + 10 * nx + 10], 0.9);
  EXPECT_DOUBLE_EQ(output[seam], 0.0);
  EXPECT_DOUBLE_EQ(output[seam + 2], 0.0);
}

TEST(VolumeOutputTest, AutoClassifiesSeamFragmentedEpTubeViaDominantSupport) {
  // Mimic EPguan: one bright tube plus many tiny coronal speckles that would
  // inflate projection_components past the compact proj<=3 gate.
  const int nx = 21;
  const int ny = 21;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 8; y <= 12; ++y) {
      for (int x = 8; x <= 12; ++x) {
        const double dy = y - 10.0;
        const double dx = x - 10.0;
        if (dy * dy + dx * dx <= 6.25)
          source[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx +
                 x] = 1.0;
      }
    }
  }
  for (int z = 0; z < nz; z += 3) {
    const int x = 2 + (z % 5);
    const int y = 2 + ((z / 3) % 4);
    for (int dy = 0; dy < 3; ++dy)
      for (int dx = 0; dx < 3; ++dx)
        source[static_cast<size_t>(z) * plane +
               static_cast<size_t>(y + dy) * nx + (x + dx)] = 0.85;
  }
  for (int localZ : {12, 28}) {
    for (int y = 8; y <= 12; ++y) {
      for (int x = 8; x <= 12; ++x)
        source[static_cast<size_t>(localZ) * plane +
               static_cast<size_t>(y) * nx + x] = 0.0;
    }
  }

  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.sigmoidBlendOutput = true;
  transform.objectAwareSeamPostprocess = true;
  transform.objectProfile = "auto";
  transform.baseNz = 20;
  transform.bedZStarts = {0, 20};
  transform.outputBedOverlaps = {16};
  transform.seamHalfWidth = 8;
  transform.fovRadiusMm = 8.0;

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform,
      1.0, 1.0);
  const size_t seam = static_cast<size_t>(12) * plane;
  EXPECT_GT(output[seam + 10 * nx + 10], 0.9);
}

TEST(VolumeOutputTest, AutoKeepsLowHighFractionDominantTubeOnEpguanPath) {
  const int nx = 61;
  const int ny = 61;
  const int nz = 272;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.005);
  for (int z = 0; z < nz; ++z) {
    const double value = z % 5 == 0 ? 1.0 : 0.3;
    for (int y = 20; y <= 40; ++y)
      for (int x = 20; x <= 40; ++x)
        source[static_cast<size_t>(z) * plane +
               static_cast<size_t>(y) * nx + x] = value;
  }
  for (int block = 0; block < 5; ++block) {
    const int z0 = 10 + 50 * block;
    const int x0 = 2;
    for (int z = z0; z < z0 + 3; ++z)
      for (int y = 2; y <= 4; ++y)
        for (int x = x0; x < x0 + 3; ++x)
          source[static_cast<size_t>(z) * plane +
                 static_cast<size_t>(y) * nx + x] = 0.3;
  }

  OutputTransform transform;
  transform.sigmoidBlendOutput = true;
  transform.objectAwareSeamPostprocess = true;
  transform.objectProfile = "auto";
  transform.baseNz = 80;
  transform.bedZStarts = {0, 80, 160, 240};
  transform.outputBedOverlaps = {16, 16, 16};
  transform.seamHalfWidth = 8;

  postprocessStitchedOutput(source, nx, ny, nz, transform, 0.3, 0.3);
  EXPECT_NEAR(source[0], 0.005 * transform.epguanBackgroundRetain, 1e-12);
}

TEST(VolumeOutputTest, EpguanBackgroundFilterPreservesTubeAndAttenuatesFarField) {
  const int nx = 61;
  const int ny = 61;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.005);
  for (int z = 0; z < nz; ++z) {
    for (int y = 24; y <= 36; ++y) {
      for (int x = 24; x <= 36; ++x) {
        const double dy = y - 30.0;
        const double dx = x - 30.0;
        if (dy * dy + dx * dx <= 36.0)
          source[static_cast<size_t>(z) * plane +
                 static_cast<size_t>(y) * nx + x] = 1.0;
      }
    }
  }

  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.sigmoidBlendOutput = true;
  transform.objectAwareSeamPostprocess = true;
  transform.objectProfile = "epguan";
  transform.baseNz = 20;
  transform.bedZStarts = {0, 20};
  transform.outputBedOverlaps = {16};
  transform.seamHalfWidth = 8;
  transform.epguanBackgroundRetain = 0.08;
  transform.epguanSupportThreshold = 0.06;
  transform.epguanSupportMarginMm = 1.5;

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform,
      0.3, 0.3);
  EXPECT_DOUBLE_EQ(output[30 * nx + 30], 1.0);
  EXPECT_NEAR(output[0], 0.005 * 0.08, 1e-12);
}

TEST(VolumeOutputTest, EpguanDoesNotPreserveBrightSeamCollarByDefault) {
  const int nx = 9;
  const int ny = 9;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 2; y <= 6; ++y)
      for (int x = 2; x <= 6; ++x)
        source[static_cast<size_t>(z) * plane +
               static_cast<size_t>(y) * nx + x] = 1.0;
  }
  for (int localZ : {12, 28}) {
    for (int y = 2; y <= 6; ++y)
      for (int x = 2; x <= 6; ++x)
        source[static_cast<size_t>(localZ) * plane +
               static_cast<size_t>(y) * nx + x] = 2.0;
  }

  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.sigmoidBlendOutput = true;
  transform.objectAwareSeamPostprocess = true;
  transform.objectProfile = "epguan";
  transform.epguanBackgroundFilter = false;
  transform.baseNz = 20;
  transform.bedZStarts = {0, 20};
  transform.outputBedOverlaps = {16};
  transform.seamHalfWidth = 8;

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform);
  const size_t seam = static_cast<size_t>(12) * plane;
  EXPECT_NEAR(output[seam + 4 * nx + 4], 1.0, 1e-12);
}

TEST(VolumeOutputTest, EpguanFeathersC4AtSeamWindowEdges) {
  const int nx = 9;
  const int ny = 9;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 2; y <= 6; ++y)
      for (int x = 2; x <= 6; ++x)
        source[static_cast<size_t>(z) * plane +
               static_cast<size_t>(y) * nx + x] = 1.0;
  }
  for (const auto [localZ, value] :
       std::vector<std::pair<int, double>>{{4, 2.0}, {0, 2.0}}) {
    const int bed = localZ == 0 ? 1 : 0;
    const int z = bed * 20 + localZ;
    for (int y = 2; y <= 6; ++y)
      for (int x = 2; x <= 6; ++x)
        source[static_cast<size_t>(z) * plane +
               static_cast<size_t>(y) * nx + x] = value;
  }

  OutputTransform transform;
  transform.bedOverlapInRecon = false;
  transform.sigmoidBlendOutput = true;
  transform.objectAwareSeamPostprocess = true;
  transform.objectProfile = "epguan";
  transform.epguanBackgroundFilter = false;
  transform.baseNz = 20;
  transform.bedZStarts = {0, 20};
  transform.outputBedOverlaps = {16};
  transform.seamHalfWidth = 8;
  transform.epguanSeamEdgeStrength = 0.60;
  transform.epguanSeamFeatherSlices = 4;

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform);
  const size_t edge = static_cast<size_t>(4) * plane;
  EXPECT_NEAR(output[edge + 4 * nx + 4], 1.4, 1e-12);
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

TEST(VolumeOutputTest, PrepareOutputWholeZFlipPrecedesEndSliceZeroing) {
  const int nx = 1;
  const int ny = 1;
  const int nz = 4;
  const std::vector<double> source = {1, 2, 3, 4};
  OutputTransform transform;
  transform.flipZWholeOutput = true;

  int outNz = 0;
  const std::vector<double> output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{1, 1}, &outNz, &transform);

  EXPECT_EQ(outNz, 4);
  const std::vector<double> expected = {0, 3, 2, 0};
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
