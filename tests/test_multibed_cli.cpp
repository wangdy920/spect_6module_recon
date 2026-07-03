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

TEST(IOTest, MultiBedParserSortsDfListAndBProjListByBedNumber) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "C:/scan/Bed3.bin;C:/scan/Bed1.bin;C:/scan/Bed2.bin",
        "-bProjList", "C:/bg/Bed2.bin;C:/bg/Bed3.bin;C:/bg/Bed1.bin",
        "-loadCSR", "matrix.csrb",
        "-sparseBackend", "cpu",
        "-alg", "mlem",
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
        "-sparseBackend", "cpu",
        "-alg", "mlem",
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
        "-sparseBackend", "cpu",
        "-alg", "mlem",
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

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_FALSE(args.zBedFlip);
    EXPECT_TRUE(args.zOutFlip);
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
        out << "crop_end = 2,3\n";
        out << "z_bed_flip = 0\n";
        out << "z_out_flip = 1\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", path.string(),
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-sparseBackend", "cpu",
        "-alg", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.bedOverlaps, (std::vector<int>{5}));
    EXPECT_EQ(args.cropEndStart, 2);
    EXPECT_EQ(args.cropEndEnd, 3);
    EXPECT_FALSE(args.zBedFlip);
    EXPECT_TRUE(args.zOutFlip);

    std::filesystem::remove(path);
}

} // namespace test
} // namespace io
} // namespace spect

