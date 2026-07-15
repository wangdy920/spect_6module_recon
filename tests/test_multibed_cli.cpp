/**
 * @file test_multibed_cli.cpp
 * @brief Split from test_interfaces.cpp (multi-bed CLI).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/ArgumentParser.h"
#include "cli/ConfigParser.h"
#include "tests/common/parser_test_helpers.h"


namespace spect {
namespace io {
namespace test {

TEST(IOTest, MultiBedCodeDefaultsMatchShippedSigmoidMode) {
    spect::cli::ReconArgs args;
    EXPECT_FALSE(args.bedOverlapInRecon);
    EXPECT_EQ(args.bedCropMethod, "sigmoid");
    EXPECT_TRUE(args.bedSeamPostprocess);
    EXPECT_EQ(args.bedObjectProfile, "auto");
    EXPECT_EQ(args.bedSeamHalfWidth, 8);
    EXPECT_DOUBLE_EQ(args.bedZimuCloseStrength, 0.5);
}

TEST(IOTest, MultiBedParserSortsDfListAndBProjListByBedNumber) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "C:/scan/Bed3.bin;C:/scan/Bed1.bin;C:/scan/Bed2.bin",
        "-bProjList", "C:/bg/Bed2.bin;C:/bg/Bed3.bin;C:/bg/Bed1.bin",
        "-loadCSR", "matrix.csrb",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.bedCount, 3);
    EXPECT_EQ(args.projPaths,
              (std::vector<std::string>{"C:/scan/Bed1.bin",
                                        "C:/scan/Bed2.bin",
                                        "C:/scan/Bed3.bin"}));
    EXPECT_EQ(args.bProjPaths,
              (std::vector<std::string>{"C:/bg/Bed1.bin",
                                        "C:/bg/Bed2.bin",
                                        "C:/bg/Bed3.bin"}));
}
TEST(IOTest, MultiBedParserRejectsMoreThanFiveDfListBeds) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "Bed1.bin;Bed2.bin;Bed3.bin;Bed4.bin;Bed5.bin;Bed6.bin",
        "-loadCSR", "matrix.csrb",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}
TEST(IOTest, MultiBedParserRejectsMoreThanFiveBedOverlapSlots) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "Bed1.bin;Bed2.bin;Bed3.bin",
        "-bedOverlap", "1,2,3,4,5,6",
        "-loadCSR", "matrix.csrb",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}
TEST(IOTest, MultiBedParserParsesZFlipSwitches) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-zbedflip");
    argv.push_back("0");
    argv.push_back("-zoutflip");
    argv.push_back("1");
    argv.push_back("-xoutflip");
    argv.push_back("1");
    argv.push_back("-bedCropMethod");
    argv.push_back("direct");
    argv.push_back("-bedSeamPostprocess");
    argv.push_back("0");
    argv.push_back("-bedObjectProfile");
    argv.push_back("zimu");

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_FALSE(args.zBedFlip);
    EXPECT_TRUE(args.zOutFlip);
    EXPECT_TRUE(args.xOutFlip);
    EXPECT_EQ(args.bedCropMethod, "direct");
    EXPECT_EQ(args.bedObjectProfile, "zimu");
}
TEST(IOTest, MultiBedParserRejectsLegacyBlendFill) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-bedCropMethod");
    argv.push_back("blendfill");

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}
TEST(IOTest, MultiBedParserAcceptsCompactChunkCtac) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "Bed1.bin;Bed2.bin",
        "-chunkCSRDir", "chunks",
        "-matrixLayout", "compact-symmetric",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360",
        "-ctac",
        "-attMap", "mu.mhd"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_TRUE(args.doCTAC);
    EXPECT_EQ(args.bedCount, 2);
}
TEST(IOTest, MultiBedConfigReadsOverlapCropAndZFlipDefaults) {
    const auto dir = std::filesystem::temp_directory_path() / "spect_multibed_config_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "spect_config.ini";

    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "[MultiBed]\n";
        out << "bed_overlap = 5\n";
        out << "overlap_in_recon = 0\n";
        out << "crop_method = sigmoid\n";
        out << "seam_postprocess = 1\n";
        out << "object_profile = phantom\n";
        out << "phantom_close_strength = 0.9\n";
        out << "zimu_close_strength = 0.8\n";
        out << "seam_half_width = 5\n";
        out << "crop_end = 2,3\n";
        out << "fov_radius_mm = 15.0\n";
        out << "z_bed_flip = 0\n";
        out << "z_out_flip = 1\n";
        out << "x_out_flip = 1\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", path.string(),
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.bedOverlaps, (std::vector<int>{5}));
    EXPECT_FALSE(args.bedOverlapInRecon);
    EXPECT_EQ(args.bedCropMethod, "sigmoid");
    EXPECT_TRUE(args.bedSeamPostprocess);
    EXPECT_EQ(args.bedObjectProfile, "phantom");
    EXPECT_DOUBLE_EQ(args.bedPhantomCloseStrength, 0.9);
    EXPECT_DOUBLE_EQ(args.bedZimuCloseStrength, 0.8);
    EXPECT_EQ(args.bedSeamHalfWidth, 5);
    EXPECT_EQ(args.cropEndStart, 2);
    EXPECT_EQ(args.cropEndEnd, 3);
    EXPECT_DOUBLE_EQ(args.fovRadiusMm, 15.0);
    EXPECT_FALSE(args.zBedFlip);
    EXPECT_TRUE(args.zOutFlip);
    EXPECT_TRUE(args.xOutFlip);

    std::filesystem::remove(path);
}

} // namespace test
} // namespace io
} // namespace spect

