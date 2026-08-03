#include <gtest/gtest.h>

#include "pipeline/VolumeOutput.h"
#include "recon/ReconLoopHelpers.h"
#include "postprocess/SeamPostprocess.h"
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <cmath>
#include <numeric>
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
  transform.outputBedOverlaps = {0};

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


TEST(VolumeOutputTest, Center66Cosine10MatchesReferenceWeights) {
  const int nx = 1;
  const int ny = 1;
  const int baseNz = 70;
  const int nz = 2 * baseNz;
  std::vector<double> source(static_cast<size_t>(nz), -100.0);
  std::fill(source.begin() + 2, source.begin() + 68, 10.0);
  std::fill(source.begin() + baseNz + 2,
            source.begin() + baseNz + 68, 20.0);

  OutputTransform transform;
  transform.center66CosineOutput = true;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {14};

  int outNz = 0;
  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, &outNz, &transform);

  ASSERT_EQ(outNz, 122);
  ASSERT_EQ(output.size(), 122u);
  EXPECT_DOUBLE_EQ(output.front(), 10.0);
  EXPECT_DOUBLE_EQ(output[55], 10.0);
  constexpr double pi = 3.14159265358979323846;
  for (int overlapIndex = 0; overlapIndex < 10; ++overlapIndex) {
    const double phase = static_cast<double>(overlapIndex) / 9.0;
    const double sine = std::sin(0.5 * pi * phase);
    const double rightWeight = sine * sine;
    EXPECT_NEAR(output[static_cast<size_t>(56 + overlapIndex)],
                (1.0 - rightWeight) * 10.0 + rightWeight * 20.0,
                1e-12);
  }
  EXPECT_DOUBLE_EQ(output[66], 20.0);
  EXPECT_DOUBLE_EQ(output.back(), 20.0);
}

TEST(VolumeOutputTest, Center66Cosine10CropsMaskWithSameGeometry) {
  const int nx = 1;
  const int ny = 1;
  const int baseNz = 70;
  const int nz = 2 * baseNz;
  const std::vector<double> source(static_cast<size_t>(nz), 1.0);
  std::vector<uint8_t> mask(static_cast<size_t>(nz), 0);
  mask[2] = 1;
  mask[baseNz + 2 + 65] = 1;

  OutputTransform transform;
  transform.center66CosineOutput = true;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {14};

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, &mask, CropSettings{0, 0}, nullptr, &transform);

  ASSERT_EQ(output.size(), 122u);
  EXPECT_DOUBLE_EQ(output.front(), 1.0);
  EXPECT_DOUBLE_EQ(output.back(), 1.0);
  EXPECT_EQ(std::count(output.begin(), output.end(), 1.0), 2);
}

TEST(VolumeOutputTest, Center66Cosine10RejectsNon56LayerStep) {
  const int baseNz = 70;
  const std::vector<double> source(static_cast<size_t>(2 * baseNz), 1.0);
  OutputTransform transform;
  transform.center66CosineOutput = true;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {13};

  EXPECT_THROW((void)prepareOutputVolume(
                   source, 1, 1, 2 * baseNz, nullptr, CropSettings{0, 0},
                   nullptr, &transform),
               std::invalid_argument);
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

TEST(VolumeOutputTest, ClassifiesDiffuseEdgeTruncatedObjectAsMouse) {
  const int nx = 21;
  const int ny = 15;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> volume(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int centerX : {4, 10, 16}) {
      for (int y = 6; y <= 8; ++y) {
        for (int x = centerX - 1; x <= centerX + 1; ++x)
          volume[static_cast<size_t>(z) * plane +
                 static_cast<size_t>(y) * nx + x] = 0.2;
      }
      volume[static_cast<size_t>(z) * plane + 7 * nx + centerX] = 1.0;
    }
  }

  EXPECT_EQ(classifyOutputObjectProfile(volume, nx, ny, nz), "mouse");
}

TEST(VolumeOutputTest, DiffuseMouseGateDoesNotCaptureCompactPhantom) {
  const int nx = 11;
  const int ny = 11;
  const int nz = 20;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> volume(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 3; y <= 7; ++y)
      for (int x = 3; x <= 7; ++x)
        volume[static_cast<size_t>(z) * plane +
               static_cast<size_t>(y) * nx + x] = 1.0;
  }

  EXPECT_EQ(classifyOutputObjectProfile(volume, nx, ny, nz), "phantom");
}

namespace {

/// Six triangular rod clusters on one pitch, apex inward, parked in the middle
/// of z so the volume looks like a single-bed acquisition with air at both ends.
std::vector<double> makeRodLatticeVolume(int nx, int ny, int nz, int z0, int z1,
                                         double innerRadius, double pitch,
                                         int sectors) {
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> volume(static_cast<size_t>(nz) * plane, 0.0);
  const double centerY = 0.5 * (ny - 1);
  const double centerX = 0.5 * (nx - 1);
  const double rowStep = pitch * std::sqrt(3.0) / 2.0;
  constexpr double rodRadius = 2.0;
  for (int sector = 0; sector < sectors; ++sector) {
    const double angle = 2.0 * 3.14159265358979323846 * sector / sectors;
    const double dirY = std::sin(angle);
    const double dirX = std::cos(angle);
    for (int row = 0; row < 3; ++row) {
      for (int index = 0; index <= row; ++index) {
        const double radius = innerRadius + row * rowStep;
        const double across = (index - 0.5 * row) * pitch;
        const double rodY = centerY + dirY * radius - dirX * across;
        const double rodX = centerX + dirX * radius + dirY * across;
        for (int y = 0; y < ny; ++y) {
          for (int x = 0; x < nx; ++x) {
            if (std::hypot(y - rodY, x - rodX) > rodRadius)
              continue;
            for (int z = z0; z < z1; ++z)
              volume[static_cast<size_t>(z) * plane +
                     static_cast<size_t>(y) * nx + x] = 1.0;
          }
        }
      }
    }
  }
  return volume;
}

} // namespace

TEST(VolumeOutputTest, ClassifiesSingleBedRodLatticeAsDerenzo) {
  const int nx = 61;
  const int ny = 61;
  const int nz = 30;
  const auto volume = makeRodLatticeVolume(nx, ny, nz, 10, 20, 10.0, 6.0, 6);

  EXPECT_EQ(classifyOutputObjectProfile(volume, nx, ny, nz, 0.5, 0.5),
            "derenzo");
}

TEST(VolumeOutputTest, RodLatticeGateRejectsSingleUnsectoredCluster) {
  const int nx = 61;
  const int ny = 61;
  const int nz = 30;
  const auto volume = makeRodLatticeVolume(nx, ny, nz, 10, 20, 2.0, 6.0, 6);

  EXPECT_NE(classifyOutputObjectProfile(volume, nx, ny, nz, 0.5, 0.5),
            "derenzo");
}

TEST(VolumeOutputTest, RodLatticeGateNeedsVoxelSizes) {
  const int nx = 61;
  const int ny = 61;
  const int nz = 30;
  const auto volume = makeRodLatticeVolume(nx, ny, nz, 10, 20, 10.0, 6.0, 6);

  EXPECT_NE(classifyOutputObjectProfile(volume, nx, ny, nz), "derenzo");
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

TEST(VolumeOutputTest, ZeroAnalysisFovPreservesLegacyOutputExactly) {
  const int nx = 9;
  const int ny = 9;
  const int nz = 40;
  std::vector<double> source(static_cast<size_t>(nx) * ny * nz);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<double>((i * 13) % 29) / 29.0;

  OutputTransform legacy;
  legacy.sigmoidBlendOutput = true;
  legacy.objectAwareSeamPostprocess = true;
  legacy.objectProfile = "phantom";
  legacy.baseNz = 20;
  legacy.bedZStarts = {0, 20};
  legacy.outputBedOverlaps = {16};
  legacy.fovRadiusMm = 3.0;

  const auto expected = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &legacy,
      1.0, 1.0);
  OutputTransform explicitAnalysis = legacy;
  explicitAnalysis.analysisFovRadiusMm = legacy.fovRadiusMm;
  const auto actual = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr,
      &explicitAnalysis, 1.0, 1.0);
  EXPECT_EQ(actual, expected);
}

TEST(VolumeOutputTest, FixedAnalysisFovIgnoresRetainedOuterArtifact) {
  const int nx = 21;
  const int ny = 21;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 9; y <= 11; ++y)
      for (int x = 9; x <= 11; ++x)
        source[static_cast<size_t>(z) * plane +
               static_cast<size_t>(y) * nx + x] = 1.0;
  }
  // This one-bed-only streak is retained by the wider output FOV but must not
  // participate in component matching.
  for (int z = 0; z < 20; ++z)
    source[static_cast<size_t>(z) * plane + 10 * nx + 18] = 1.0;
  for (int localZ : {12, 28})
    for (int y = 9; y <= 11; ++y)
      for (int x = 9; x <= 11; ++x)
        source[static_cast<size_t>(localZ) * plane +
               static_cast<size_t>(y) * nx + x] = 0.0;

  OutputTransform transform;
  transform.sigmoidBlendOutput = true;
  transform.objectAwareSeamPostprocess = true;
  transform.objectProfile = "auto";
  transform.epguanBackgroundFilter = false;
  transform.baseNz = 20;
  transform.bedZStarts = {0, 20};
  transform.outputBedOverlaps = {16};
  transform.seamHalfWidth = 8;
  transform.analysisFovRadiusMm = 3.0;
  transform.fovRadiusMm = 8.0;

  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform,
      1.0, 1.0);
  const size_t seam = static_cast<size_t>(12) * plane;
  EXPECT_GT(output[seam + 10 * nx + 10], 0.9);
  EXPECT_GT(output[seam + 10 * nx + 18], 0.4);
}

TEST(VolumeOutputTest, MouseConfidenceFusionChangesOnlyBilateralMainSupport) {
  const int nx = 15;
  const int ny = 15;
  const int nz = 16;
  const int baseNz = 8;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(static_cast<size_t>(nz) * plane, 0.0);
  for (int z = 0; z < baseNz; ++z) {
    for (int y = 6; y <= 8; ++y)
      for (int x = 6; x <= 8; ++x) {
        source[static_cast<size_t>(z) * plane +
               static_cast<size_t>(y) * nx + x] = 1.0;
        source[static_cast<size_t>(baseNz + z) * plane +
               static_cast<size_t>(y) * nx + x] = 0.1;
      }
    // Seen by only the first bed: this tail must fail closed.
    source[static_cast<size_t>(z) * plane + 2 * nx + 2] = 1.0;
  }

  OutputTransform baselineTransform;
  baselineTransform.sigmoidBlendOutput = true;
  baselineTransform.objectAwareSeamPostprocess = true;
  baselineTransform.objectProfile = "mouse";
  baselineTransform.baseNz = baseNz;
  baselineTransform.bedZStarts = {0, baseNz};
  baselineTransform.outputBedOverlaps = {4};

  const auto baseline = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr,
      &baselineTransform);
  OutputTransform candidateTransform = baselineTransform;
  candidateTransform.mouseSeamStrength = 0.5;
  const auto candidate = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr,
      &candidateTransform);

  const size_t overlapMid = static_cast<size_t>(6) * plane;
  EXPECT_GT(candidate[overlapMid + 7 * nx + 7],
            baseline[overlapMid + 7 * nx + 7]);
  EXPECT_NEAR(candidate[overlapMid + 7 * nx + 7], 0.625, 1e-12);
  EXPECT_DOUBLE_EQ(candidate[overlapMid + 2 * nx + 2],
                   baseline[overlapMid + 2 * nx + 2]);
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
  transform.outputBedOverlaps = {0};

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

TEST(VolumeOutputTest, DerenzoFillImprovesInternalGapWithoutExtendingRods) {
  const int nx = 48;
  const int ny = 48;
  const int nz = 40;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(plane * nz, 0.0);
  const auto addRod = [&](int centerY, int centerX, int start, int stop,
                          int gap) {
    for (int z = start; z < stop; ++z) {
      const double value = z == gap ? 0.15 : 1.0;
      for (int y = centerY - 1; y <= centerY + 1; ++y) {
        for (int x = centerX - 1; x <= centerX + 1; ++x)
          source[static_cast<size_t>(z) * plane +
                 static_cast<size_t>(y) * nx + x] = value;
      }
    }
  };
  addRod(15, 15, 7, 33, 20);
  addRod(32, 31, 9, 35, 22);

  OutputTransform transform;
  transform.objectProfile = "derenzo";
  transform.derenzoAxialPostprocess = true;
  transform.derenzoAxialStrength = 0.5;
  const auto output = prepareOutputVolume(
      source, nx, ny, nz, nullptr, CropSettings{0, 0}, nullptr, &transform,
      0.3, 0.3, 0.3);

  const auto at = [&](const std::vector<double> &volume, int z, int y, int x) {
    return volume[static_cast<size_t>(z) * plane +
                  static_cast<size_t>(y) * nx + x];
  };
  EXPECT_GT(at(output, 20, 15, 15), at(source, 20, 15, 15));
  EXPECT_GT(at(output, 22, 32, 31), at(source, 22, 32, 31));
  EXPECT_EQ(*std::max_element(output.begin(), output.end()), 1.0);
  for (int z = 0; z < nz; ++z) {
    if (z < 7 || z >= 33)
      EXPECT_DOUBLE_EQ(at(output, z, 15, 15), at(source, z, 15, 15));
    if (z < 9 || z >= 35)
      EXPECT_DOUBLE_EQ(at(output, z, 32, 31), at(source, z, 32, 31));
  }
  EXPECT_DOUBLE_EQ(at(output, 20, 24, 24), 0.0);
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
