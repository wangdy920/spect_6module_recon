/**
 * @file test_multibed_cli.cpp
 * @brief Split from test_interfaces.cpp (multi-bed CLI).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/ArgumentParser.h"
#include "cli/ConfigParser.h"
#include "io/BadDetectorMask.h"
#include "io/DetZRowWeights.h"
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
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
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
TEST(IOTest, TimeSegmentMergeReadsFromGlobalConfig) {
    const auto dir = std::filesystem::temp_directory_path() /
                     "spect_time_segment_config_tests";
    std::filesystem::create_directories(dir);
    const auto config = dir / "spect_config.ini";
    {
        std::ofstream out(config);
        ASSERT_TRUE(out);
        out << "[InputOutput]\nmerge_time_segments = true\n";
    }
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon", "-config", config.string(), "-df", "Bed1.bin",
        "-chunkCSRDir", "chunks", "-opti", "mlem", "-it", "1",
        "-dim", "108,108,80", "-vox", "0.3", "-ang", "360"
    };
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_TRUE(args.mergeTimeSegments);
    std::filesystem::remove(config);
}
TEST(IOTest, MultiBedParserRejectsMoreThanFiveDfListBeds) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "Bed1.bin;Bed2.bin;Bed3.bin;Bed4.bin;Bed5.bin;Bed6.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
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
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
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
    argv.insert(argv.end(), {"-zbedflip", "0", "-zoutflip", "1",
                             "-xoutflip", "1"});

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_FALSE(args.zBedFlip);
    EXPECT_TRUE(args.zOutFlip);
    EXPECT_TRUE(args.xOutFlip);
}

TEST(IOTest, SingleBedStepAlignedDefaultsOnAndCanBeDisabled) {
    spect::cli::ReconArgs defaults;
    ASSERT_TRUE(parseReconArgsForTest(compactBaseArgs(), defaults));
    EXPECT_TRUE(defaults.singleBedStepAligned);

    spect::cli::ReconArgs disabled;
    auto argv = compactBaseArgs();
    argv.insert(argv.end(), {"-singleBedStepAligned", "0"});
    ASSERT_TRUE(parseReconArgsForTest(argv, disabled));
    EXPECT_FALSE(disabled.singleBedStepAligned);
}

TEST(IOTest, MultiBedParserAcceptsCompactChunkCtac) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "Bed1.bin;Bed2.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
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
    const auto dir = std::filesystem::temp_directory_path() /
                     "spect_multibed_config_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "spect_config.ini";

    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "[MultiBed]\n";
        out << "bed_overlap = 164\n";
        out << "single_bed_step_aligned = false\n";
        out << "crop_end = 2,3\n";
        out << "z_bed_flip = 0\n";
        out << "z_out_flip = 1\n";
        out << "x_out_flip = 1\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon", "-config", path.string(), "-df", "proj.bin",
        "-chunkCSRDir", "chunks", "-opti", "mlem", "-it", "1",
        "-dim", "108,108,220", "-vox", "0.3", "-ang", "360"};

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.bedOverlaps, (std::vector<int>{164}));
    EXPECT_FALSE(args.singleBedStepAligned);
    EXPECT_EQ(args.cropEndStart, 2);
    EXPECT_EQ(args.cropEndEnd, 3);
    EXPECT_FALSE(args.zBedFlip);
    EXPECT_TRUE(args.zOutFlip);
    EXPECT_TRUE(args.xOutFlip);

    std::filesystem::remove(path);
}

TEST(IOTest, DetZEdgeCensorDefaultsToOff) {
    spect::cli::ReconArgs args;
    EXPECT_EQ(args.detZEdgeCensor, 0);
}
TEST(IOTest, DetZEdgeCensorParsesCliWithCompactChunkLayout) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "Bed1.bin;Bed2.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360",
        "-detZEdgeCensor", "3"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.detZEdgeCensor, 3);
}
TEST(IOTest, DetZEdgeCensorReadsFromMultiBedConfig) {
    const auto dir = std::filesystem::temp_directory_path() / "spect_multibed_config_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "spect_config_detz_censor.ini";

    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "[MultiBed]\n";
        out << "detz_edge_censor = 3\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", path.string(),
        "-dfList", "Bed1.bin;Bed2.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.detZEdgeCensor, 3);

    std::filesystem::remove(path);
}
TEST(IOTest, DetZEdgeCensorRejectsNegativeDepth) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "Bed1.bin;Bed2.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360",
        "-detZEdgeCensor", "-1"
    };

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}
TEST(IOTest, DetZEdgeCensorRejectsDepthCoveringDetector) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "Bed1.bin;Bed2.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360",
        "-detN", "48,48",
        "-detZEdgeCensor", "24"
    };

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}
TEST(IOTest, DetZEdgeCensorEchoesInDumpConf) {
    spect::cli::ArgumentParser parser;
    spect::cli::ReconArgs args;
    std::vector<std::string> storage = {
        "SPECT_recon", "-detZEdgeCensor", "3", "-dumpConf"};
    std::vector<char*> argv;
    for (std::string& arg : storage)
        argv.push_back(arg.data());

    testing::internal::CaptureStdout();
    const auto result = parser.parse(static_cast<int>(argv.size()), argv.data(), args);
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_EQ(result, spect::cli::ArgumentParser::ParseResult::ExitSuccess);
    EXPECT_NE(out.find("detz_edge_censor = 3"), std::string::npos);
}

namespace {

std::filesystem::path writeTempDetZWeights(const std::string& name,
                                           size_t count, float value,
                                           size_t overrideIndex = std::string::npos,
                                           float overrideValue = 0.0f) {
    const auto dir = std::filesystem::temp_directory_path() / "spect_detz_weights_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / name;
    std::ofstream out(path);
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) out << ' ';
        out << (i == overrideIndex ? overrideValue : value);
    }
    out << '\n';
    return path;
}

std::filesystem::path writeTempProjectionWeights(const std::string& name,
                                                 std::uint32_t angles,
                                                 std::uint32_t rays,
                                                 double value) {
    const auto dir =
        std::filesystem::temp_directory_path() / "spect_detz_weights_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / name;
    std::ofstream out(path, std::ios::binary);
    const std::uint32_t header[2] = {angles, rays};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    for (std::uint64_t i = 0;
         i < static_cast<std::uint64_t>(angles) * rays; ++i) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    return path;
}

std::vector<std::string> detZWeightsChunkArgs(const std::string& weightsPath) {
    return {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-dfList", "Bed1.bin;Bed2.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360",
        "-detN", "48,48",
        "-detZRowWeightsFile", weightsPath
    };
}

} // namespace

TEST(IOTest, DetZRowWeightsDefaultsToOff) {
    spect::cli::ReconArgs args;
    EXPECT_TRUE(args.detZRowWeightsFile.empty());
}
TEST(IOTest, DetZRowWeightsParsesCliWithCompactChunkLayout) {
    const auto weightsPath = writeTempDetZWeights("row48.txt", 48, 0.95f);
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv =
        detZWeightsChunkArgs(weightsPath.string());

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.detZRowWeightsFile, weightsPath.generic_string());

    std::filesystem::remove(weightsPath);
}
TEST(IOTest, DetZRowWeightsParsesPerBinFile) {
    const auto weightsPath =
        writeTempDetZWeights("bin2304.txt", 48 * 48, 1.02f);
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv =
        detZWeightsChunkArgs(weightsPath.string());

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.detZRowWeightsFile, weightsPath.generic_string());

    std::filesystem::remove(weightsPath);
}
TEST(IOTest, NormalizationFactorLoadsProjectionBinAndAlias) {
    const auto weightsPath =
        writeTempProjectionWeights("projection12.bin", 2, 6, 0.91);
    const auto values =
        spect::io::loadDetZRowWeights(weightsPath.string(), 2, 3, 2);
    ASSERT_EQ(values.size(), 12u);
    EXPECT_DOUBLE_EQ(values.front(), 0.91);
    EXPECT_DOUBLE_EQ(values.back(), 0.91);

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "Bed1.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
        "-it", "1",
        "-dim", "2,2,2",
        "-vox", "0.3",
        "-ang", "2",
        "-detN", "2,3",
        "-normFactorFile", weightsPath.string()
    };
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_TRUE(args.detZRowWeightsFile.empty());
    EXPECT_EQ(args.normFactorFile, weightsPath.generic_string());

    std::filesystem::remove(weightsPath);
}
TEST(IOTest, DetZRowWeightsReadsFromMultiBedConfig) {
    const auto weightsPath = writeTempDetZWeights("cfg48.txt", 48, 1.0f);
    const auto dir = std::filesystem::temp_directory_path() / "spect_multibed_config_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "spect_config_detz_weights.ini";

    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "[MultiBed]\n";
        out << "detz_row_weights_file = " << weightsPath.generic_string() << "\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", path.string(),
        "-dfList", "Bed1.bin;Bed2.bin",
        "-chunkCSRDir", "chunks",
        "-opti", "mlem",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360",
        "-detN", "48,48"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.detZRowWeightsFile, weightsPath.generic_string());

    std::filesystem::remove(path);
    std::filesystem::remove(weightsPath);
}
TEST(IOTest, DetZRowWeightsRejectsBadCount) {
    const auto weightsPath = writeTempDetZWeights("bad50.txt", 50, 1.0f);
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv =
        detZWeightsChunkArgs(weightsPath.string());

    EXPECT_FALSE(parseReconArgsForTest(argv, args));

    std::filesystem::remove(weightsPath);
}
TEST(IOTest, DetZRowWeightsRejectsNonPositive) {
    const auto weightsPath =
        writeTempDetZWeights("nonpos.txt", 48, 1.0f, 7, 0.0f);
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv =
        detZWeightsChunkArgs(weightsPath.string());

    EXPECT_FALSE(parseReconArgsForTest(argv, args));

    std::filesystem::remove(weightsPath);
}
TEST(IOTest, DetZRowWeightsRejectsMissingFile) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv =
        detZWeightsChunkArgs("definitely_missing_detz_weights_9f3a.txt");

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}
TEST(IOTest, DetZRowWeightsEchoesInDumpConf) {
    spect::cli::ArgumentParser parser;
    spect::cli::ReconArgs args;
    std::vector<std::string> storage = {
        "SPECT_recon", "-detZRowWeightsFile", "weights.txt", "-dumpConf"};
    std::vector<char*> argv;
    for (std::string& arg : storage)
        argv.push_back(arg.data());

    testing::internal::CaptureStdout();
    const auto result = parser.parse(static_cast<int>(argv.size()), argv.data(), args);
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_EQ(result, spect::cli::ArgumentParser::ParseResult::ExitSuccess);
    EXPECT_NE(out.find("detz_row_weights_file = weights.txt"),
              std::string::npos);
}



namespace {

std::filesystem::path writeTempBadDetectorList(
    const std::string& name, const std::vector<std::string>& identifiers) {
    const auto dir =
        std::filesystem::temp_directory_path() / "spect_bad_detector_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / name;
    std::ofstream out(path);
    out << "[\n";
    for (size_t index = 0; index < identifiers.size(); ++index) {
        out << "  \"" << identifiers[index] << "\"";
        if (index + 1 < identifiers.size()) out << ',';
        out << '\n';
    }
    out << "]\n";
    return path;
}

} // namespace

TEST(IOTest, BadDetectorListAcceptsEmptyAndCanonicalIds) {
    const auto emptyPath = writeTempBadDetectorList("empty", {});
    EXPECT_TRUE(loadBadDetectorList(emptyPath.string()).empty());

    const auto path = writeTempBadDetectorList(
        "valid", {"link-3-FPGA-2-dector-5", "link-4-FPGA-0-dector-7"});
    const auto ids = loadBadDetectorList(path.string());
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(badDetectorName(ids[0]), "link-3-FPGA-2-dector-5");
    EXPECT_EQ(badDetectorName(ids[1]), "link-4-FPGA-0-dector-7");

    std::filesystem::remove(emptyPath);
    std::filesystem::remove(path);
}

TEST(IOTest, BadDetectorListAcceptsGlobalNumbersAndCompletesHardwareIds) {
    const auto dir =
        std::filesystem::temp_directory_path() / "spect_bad_detector_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "global_numbers";
    {
        std::ofstream out(path);
        out << "[\n  68,\n  \"95\"\n]\n";
    }

    const auto ids = loadBadDetectorList(path.string());
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(badDetectorName(ids[0]), "link-4-FPGA-2-dector-4");
    EXPECT_EQ(badDetectorGlobalNumber(ids[0]), 68);
    EXPECT_EQ(badDetectorName(ids[1]), "link-4-FPGA-5-dector-7");
    EXPECT_EQ(badDetectorGlobalNumber(ids[1]), 95);

    std::filesystem::remove(path);
}

TEST(IOTest, BadDetectorListRejectsInvalidAndDuplicateIds) {
    const auto invalidPath =
        writeTempBadDetectorList("invalid", {"link-5-FPGA-0-dector-0"});
    EXPECT_THROW(loadBadDetectorList(invalidPath.string()), std::runtime_error);

    const auto duplicatePath = writeTempBadDetectorList(
        "duplicate", {"20", "link-3-FPGA-2-dector-4"});
    EXPECT_THROW(loadBadDetectorList(duplicatePath.string()), std::runtime_error);

    const auto rangePath = writeTempBadDetectorList("out_of_range", {"96"});
    EXPECT_THROW(loadBadDetectorList(rangePath.string()), std::runtime_error);

    std::filesystem::remove(invalidPath);
    std::filesystem::remove(duplicatePath);
    std::filesystem::remove(rangePath);
}

TEST(IOTest, BadDetectorMaskMapsHardwareBlocksToIdentityProjectionRows) {
    const std::vector<BadDetectorId> ids = {{3, 2, 5}, {4, 2, 5}};
    const auto mask = buildBadDetectorProjectionMask(ids, 360, 48, 48);
    ASSERT_EQ(mask.size(), static_cast<size_t>(360 * 48 * 48));
    EXPECT_EQ(std::count(mask.begin(), mask.end(), uint8_t{1}),
              2 * 60 * 12 * 12);
    const auto index = [](int angle, int z, int x) {
        return static_cast<size_t>(angle) * 48 * 48 +
               static_cast<size_t>(z) * 48 + x;
    };
    EXPECT_EQ(mask[index(120, 12, 12)], 1); // link3 detector5: block (1,1)
    EXPECT_EQ(mask[index(179, 23, 23)], 1);
    EXPECT_EQ(mask[index(120, 36, 12)], 1); // link4 detector5: block (1,3)
    EXPECT_EQ(mask[index(179, 47, 23)], 1);
    EXPECT_EQ(mask[index(119, 12, 12)], 0);
    EXPECT_EQ(mask[index(180, 12, 12)], 0);
    EXPECT_EQ(mask[index(120, 11, 12)], 0);
    EXPECT_THROW(buildBadDetectorProjectionMask(ids, 360, 40, 40),
                 std::runtime_error);
}

TEST(IOTest, BadDetectorMaskMapsPrototypeDetector68Exactly) {
    const auto mask = buildBadDetectorProjectionMask(
        {badDetectorFromGlobalNumber(68)}, 360, 48, 48);
    ASSERT_EQ(std::count(mask.begin(), mask.end(), uint8_t{1}),
              60 * 12 * 12);
    const auto index = [](int angle, int z, int x) {
        return static_cast<size_t>(angle) * 48 * 48 +
               static_cast<size_t>(z) * 48 + x;
    };
    EXPECT_EQ(mask[index(0, 24, 24)], 1);
    EXPECT_EQ(mask[index(59, 35, 35)], 1);
    EXPECT_EQ(mask[index(60, 24, 24)], 0);
    EXPECT_EQ(mask[index(120, 36, 0)], 0);
}

TEST(IOTest, BadDetectorConfigResolvesListRelativeToGlobalConfig) {
    const auto listPath = writeTempBadDetectorList(
        "bad_dector", {"link-3-FPGA-2-dector-5"});
    const auto configPath = listPath.parent_path() / "bad_detector_config.ini";
    {
        std::ofstream out(configPath);
        out << "[Detector]\n";
        out << "bad_detector_file = bad_dector\n";
    }
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon", "-config", configPath.string(),
        "-df", "Bed1.bin", "-chunkCSRDir", "chunks",
        "-opti", "mlem",
        "-it", "1", "-dim", "108,108,80",
        "-vox", "0.3", "-ang", "360", "-detN", "48,48"};

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(std::filesystem::path(args.badDetectorFile).lexically_normal(),
              listPath.lexically_normal());

    std::filesystem::remove(configPath);
    std::filesystem::remove(listPath);
}

TEST(IOTest, EmptyBadDetectorListAcceptedWithChunkLayout) {
    const auto listPath = writeTempBadDetectorList("empty_standard", {});
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon", "-config", "__missing_compact_test_config__.ini",
        "-df", "Bed1.bin", "-chunkCSRDir", "chunks", "-opti", "mlem",
        "-it", "1", "-dim", "108,108,80",
        "-vox", "0.3", "-ang", "360", "-badDetectorFile",
        listPath.string()};

    EXPECT_TRUE(parseReconArgsForTest(argv, args));
    std::filesystem::remove(listPath);
}

} // namespace test
} // namespace io
} // namespace spect
