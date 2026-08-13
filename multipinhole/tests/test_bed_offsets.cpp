#include <gtest/gtest.h>

#include "recon/BedOffsets.h"

namespace spect {
namespace app {
namespace test {

TEST(BedOffsetsTest, ComputeOffsetsWithOverlap) {
    const std::vector<int> overlaps = {16, 16, 16};
    const auto offsets = computeBedZOffsets(4, 100, overlaps);
    ASSERT_EQ(offsets.size(), 4u);
    EXPECT_EQ(offsets[0], 0);
    EXPECT_EQ(offsets[1], 84);
    EXPECT_EQ(offsets[2], 168);
    EXPECT_EQ(offsets[3], 252);
}

TEST(BedOffsetsTest, InferBaseNzFromCombinedNz) {
    const std::vector<int> overlaps = {16, 16, 16};
    EXPECT_EQ(inferBaseNzFromCombinedNz(352, 4, overlaps), 100);
}

TEST(BedOffsetsTest, InferBaseNzFromBaseNzCandidate) {
    const std::vector<int> overlaps = {16, 16, 16};
    EXPECT_EQ(inferBaseNzFromCombinedNz(100, 4, overlaps), 100);
}

TEST(BedOffsetsTest, AccumulatesLocalBedVolumesInOverlap) {
    const int nx = 2;
    const int ny = 1;
    const int baseNz = 3;
    const auto starts = computeBedZOffsets(2, baseNz, {1});
    ASSERT_EQ(starts, (std::vector<int>{0, 2}));

    std::vector<double> global(static_cast<size_t>(nx * ny * 5), 0.0);
    const std::vector<double> bed0 = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    const std::vector<double> bed1 = {
        10.0, 20.0,
        30.0, 40.0,
        50.0, 60.0,
    };

    accumulateBedVolumeToGlobal(global, bed0, nx, ny, baseNz, starts[0]);
    accumulateBedVolumeToGlobal(global, bed1, nx, ny, baseNz, starts[1]);

    const std::vector<double> expected = {
        1.0, 2.0,
        3.0, 4.0,
        15.0, 26.0,
        30.0, 40.0,
        50.0, 60.0,
    };
    EXPECT_EQ(global, expected);

    EXPECT_EQ(extractBedVolumeFromGlobal(global, nx, ny, baseNz, starts[0]),
              (std::vector<double>{1.0, 2.0, 3.0, 4.0, 15.0, 26.0}));
    EXPECT_EQ(extractBedVolumeFromGlobal(global, nx, ny, baseNz, starts[1]),
              (std::vector<double>{15.0, 26.0, 30.0, 40.0, 50.0, 60.0}));
}

TEST(BedOffsetsTest, ZeroOverlapIdenticalLocalBedsAccumulateIndependently) {
    const int nx = 2;
    const int ny = 1;
    const int baseNz = 2;
    const auto starts = computeBedZOffsets(2, baseNz, {0});
    ASSERT_EQ(starts, (std::vector<int>{0, 2}));

    std::vector<double> global(static_cast<size_t>(nx * ny * 4), 0.0);
    const std::vector<double> local = {1.0, 2.0, 3.0, 4.0};
    accumulateBedVolumeToGlobal(global, local, nx, ny, baseNz, starts[0]);
    accumulateBedVolumeToGlobal(global, local, nx, ny, baseNz, starts[1]);

    EXPECT_EQ(global,
              (std::vector<double>{1.0, 2.0, 3.0, 4.0,
                                   1.0, 2.0, 3.0, 4.0}));
    EXPECT_EQ(extractBedVolumeFromGlobal(global, nx, ny, baseNz, starts[0]),
              local);
    EXPECT_EQ(extractBedVolumeFromGlobal(global, nx, ny, baseNz, starts[1]),
              local);
}

TEST(BedOffsetsTest, VirtualZFlipMapsLocalSlicesIntoPhysicalGlobalOrder) {
    const int nx = 1;
    const int ny = 1;
    const int baseNz = 3;
    std::vector<double> global(3, 0.0);
    const std::vector<double> local{1.0, 2.0, 3.0};

    accumulateBedVolumeToGlobal(global, local, nx, ny, baseNz, 0, true);

    EXPECT_EQ(global, (std::vector<double>{3.0, 2.0, 1.0}));
    EXPECT_EQ(extractBedVolumeFromGlobal(global, nx, ny, baseNz, 0, true),
              local);
}

TEST(BedOffsetsTest, VirtualZFlipUsesTheSamePhysicalOverlapForBothBeds) {
    const int nx = 1;
    const int ny = 1;
    const int baseNz = 3;
    const auto starts = computeBedZOffsets(2, baseNz, {1});
    std::vector<double> global(5, 0.0);

    accumulateBedVolumeToGlobal(global, {1.0, 2.0, 3.0}, nx, ny, baseNz,
                                starts[0], true);
    accumulateBedVolumeToGlobal(global, {4.0, 5.0, 6.0}, nx, ny, baseNz,
                                starts[1], true);

    EXPECT_EQ(global, (std::vector<double>{3.0, 2.0, 7.0, 5.0, 4.0}));
}

} // namespace test
} // namespace app
} // namespace spect
