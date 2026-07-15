/**
 * @file test_parser.cpp
 * @brief Split from test_interfaces.cpp (CLI/parser).
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

#include "cli/StringUtils.h"
#include "cli/ConfigParser.h"


namespace spect {
namespace cli {
namespace test {

TEST(CLITest, StringSplit) {
    auto parts = split("a,b,c", ',');
    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}
TEST(CLITest, StringTrim) {
    EXPECT_EQ(trim("  hello  "), "hello");
    EXPECT_EQ(trim("\t\nworld\r\n"), "world");
    EXPECT_EQ(trim(""), "");
}
TEST(CLITest, ToLower) {
    EXPECT_EQ(toLower("Hello World"), "hello world");
    EXPECT_EQ(toLower("MLEM"), "mlem");
}
TEST(CLITest, NormalizePath) {
    EXPECT_EQ(normalizePath("D:\\path\\to\\file"), "D:/path/to/file");
    EXPECT_EQ(normalizePath("already/forward"), "already/forward");
}
TEST(CLITest, SplitPathList) {
    auto paths = splitPathList("a.mat, b.mat; c.mat");
    ASSERT_EQ(paths.size(), 3);
    EXPECT_EQ(paths[0], "a.mat");
    EXPECT_EQ(paths[1], "b.mat");
    EXPECT_EQ(paths[2], "c.mat");
}
TEST(CLITest, ParseOnOff) {
    bool val = false;
    EXPECT_TRUE(parseOnOff("on", val));
    EXPECT_TRUE(val);
    EXPECT_TRUE(parseOnOff("OFF", val));
    EXPECT_FALSE(val);
    EXPECT_TRUE(parseOnOff("true", val));
    EXPECT_TRUE(val);
    EXPECT_FALSE(parseOnOff("invalid", val));
}
TEST(CLITest, ConfigParserMethods) {
    ConfigParser cfg;
    // 测试空配置的默认值
    EXPECT_EQ(cfg.getString("Section", "key", "default"), "default");
    EXPECT_EQ(cfg.getInt("Section", "key", 42), 42);
    EXPECT_DOUBLE_EQ(cfg.getDouble("Section", "key", 3.14), 3.14);
    EXPECT_FALSE(cfg.getBool("Section", "key", false));
    EXPECT_FALSE(cfg.hasSection("Section"));
    EXPECT_FALSE(cfg.hasKey("Section", "key"));
}

TEST(CLITest, ConfigParserRejectsInvalidTypedValues) {
    const auto path = std::filesystem::temp_directory_path() /
                      "spect_invalid_typed_config.ini";
    {
        std::ofstream out(path);
        ASSERT_TRUE(out.good());
        out << "[Invalid]\nint = 12x\ndouble = 1.2mm\nbool = maybe\n";
    }
    ConfigParser cfg;
    ASSERT_TRUE(cfg.load(path.string()));
    EXPECT_THROW((void)cfg.getInt("Invalid", "int", 0), std::invalid_argument);
    EXPECT_THROW((void)cfg.getDouble("Invalid", "double", 0.0),
                 std::invalid_argument);
    EXPECT_THROW((void)cfg.getBool("Invalid", "bool", false),
                 std::invalid_argument);
    std::filesystem::remove(path);
}

} // namespace test
} // namespace cli
} // namespace spect


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

TEST(IOTest, ParserHelpAndDumpConfReturnExitSuccess) {
    spect::cli::ArgumentParser parser;
    spect::cli::ReconArgs helpArgs;
    {
        std::vector<std::string> storage = {"SPECT_recon", "-help"};
        std::vector<char*> argv;
        for (std::string& arg : storage)
            argv.push_back(arg.data());
        EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), helpArgs),
                  spect::cli::ArgumentParser::ParseResult::ExitSuccess);
    }

    spect::cli::ReconArgs versionArgs;
    {
        std::vector<std::string> storage = {"SPECT_recon", "-v"};
        std::vector<char*> argv;
        for (std::string& arg : storage)
            argv.push_back(arg.data());
        EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), versionArgs),
                  spect::cli::ArgumentParser::ParseResult::ExitSuccess);
        EXPECT_TRUE(versionArgs.versionRequested);
    }

    spect::cli::ReconArgs dumpArgs;
    {
        std::vector<std::string> storage = {
            "SPECT_recon",
            "-dumpConf"
        };
        std::vector<char*> argv;
        for (std::string& arg : storage)
            argv.push_back(arg.data());
        EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), dumpArgs),
                  spect::cli::ArgumentParser::ParseResult::ExitSuccess);
    }
}

TEST(IOTest, ExplicitMissingConfigReturnsExitFailure) {
    spect::cli::ArgumentParser parser;
    spect::cli::ReconArgs args;
    std::vector<std::string> storage = {
        "SPECT_recon", "-config", "__missing_explicit_config__.ini", "-dumpConf"};
    std::vector<char*> argv;
    for (std::string& arg : storage)
        argv.push_back(arg.data());

    EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), args),
              spect::cli::ArgumentParser::ParseResult::ExitFailure);
}

TEST(IOTest, ExplicitMissingAlgoConfigReturnsExitFailure) {
    spect::cli::ArgumentParser parser;
    spect::cli::ReconArgs args;
    std::vector<std::string> storage = {
        "SPECT_recon", "-algoConf", "__missing_algo_config__.conf", "-dumpConf"};
    std::vector<char*> argv;
    for (std::string& arg : storage)
        argv.push_back(arg.data());

    EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), args),
              spect::cli::ArgumentParser::ParseResult::ExitFailure);
}

TEST(IOTest, ProjectionResidualDiagnosticIsExplicitlyOptIn) {
    spect::cli::ArgumentParser parser;
    spect::cli::ReconArgs args;
    std::vector<std::string> storage = {
        "SPECT_recon", "-projectionResidual", "on", "-dumpConf"};
    std::vector<char*> argv;
    for (std::string& arg : storage)
        argv.push_back(arg.data());
    EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), args),
              spect::cli::ArgumentParser::ParseResult::ExitSuccess);
    EXPECT_TRUE(args.projectionResidual);
}

TEST(IOTest, ProjectionResidualDiagnosticRejectsStandardLayout) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon", "-df", "proj.bin", "-loadCSR", "matrix.csrb",
        "-dim", "2,2,2", "-projectionResidual", "on"};
    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, ParserMissingRequiredArgReturnsExitFailure) {
    spect::cli::ArgumentParser parser;
    spect::cli::ReconArgs args;
    std::vector<std::string> storage = {"SPECT_recon", "-df"};
    std::vector<char*> argv;
    for (std::string& arg : storage)
        argv.push_back(arg.data());

    EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), args),
              spect::cli::ArgumentParser::ParseResult::ExitFailure);
}

TEST(IOTest, CompactSymmetricParserDefaultsToStandard) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    EXPECT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.matrixLayout, "standard");
    EXPECT_TRUE(args.compactMetaPath.empty());
}
TEST(IOTest, StandardParserDoesNotEnableCompactFromMetaOverride) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-compactMeta", "custom_metadata.bin",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    EXPECT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.matrixLayout, "standard");
    EXPECT_EQ(args.compactMetaPath, "custom_metadata.bin");
}
TEST(IOTest, CompactSymmetricParserAcceptsExplicitMetaOverride) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-compactMeta");
    argv.push_back("custom_metadata.bin");

    EXPECT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.matrixLayout, "compact-symmetric");
    EXPECT_EQ(args.compactMetaPath, "custom_metadata.bin");
}
TEST(IOTest, CompactSymmetricParserAcceptsDenomCachePath) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-compactDenomCache");
    argv.push_back("cache/psc_260601_global_denom.mhd");

    EXPECT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.compactDenomCachePath,
              "cache/psc_260601_global_denom.mhd");
}
TEST(IOTest, CompactSymmetricParserAcceptsChunkCSRDirWithoutManualGeometry) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-matrixLayout", "compact-symmetric",
        "-chunkCSRDir", "matrix_dir",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1"
    };

    EXPECT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.matrixLayout, "compact-symmetric");
    EXPECT_EQ(args.chunkCSRDir, "matrix_dir");
    EXPECT_TRUE(args.loadCSRPath.empty());
}

TEST(IOTest, CompactSymmetricDefaultsSparseBackendToCuda) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-matrixLayout", "compact-symmetric",
        "-chunkCSRDir", "matrix_dir",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, PublicSparseBackendFlagIsRejected) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-matrixLayout", "compact-symmetric",
        "-chunkCSRDir", "matrix_dir",
        "-sparseBackend", "cpu",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, PublicAlgFlagIsRejected) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-alg", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, OptiSelectsAlgorithm) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-opti", "bsrem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.algorithm, spect::cli::ReconArgs::Algorithm::BSREM);
}

TEST(IOTest, PublicRegularizationRejectsNonTvRdp) {
    auto argv = compactBaseArgs();
    argv.push_back("-reg");
    argv.push_back("Tikhonov,0.01");

    spect::cli::ReconArgs args;
    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, RegisteredCtRegularizationOptionsParse) {
    auto argv = compactBaseArgs();
    argv.insert(argv.end(), {"-opti", "bsrem",
                             "-reg", "RDP,0.01,0.2,0.001",
                             "-regCT", "registered_ct.mhd",
                             "-regCtSigma", "0.025",
                             "-regCtFloor", "0.15"});

    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.reg.type, spect::RegType::RDP);
    EXPECT_NE(args.regCtPath.find("registered_ct.mhd"), std::string::npos);
    EXPECT_DOUBLE_EQ(args.reg.ct_sigma, 0.025);
    EXPECT_DOUBLE_EQ(args.reg.ct_weight_floor, 0.15);
}

TEST(IOTest, PublicPsfProjFlagIsRejected) {
    auto argv = compactBaseArgs();
    argv.push_back("-psf");
    argv.push_back("1.5,4");
    argv.push_back("-psfProj");
    argv.push_back("off");

    spect::cli::ReconArgs args;
    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, PublicPsfModeRejectsNonOperatorModes) {
    auto argv = compactBaseArgs();
    argv.push_back("-psf");
    argv.push_back("1.5,4");
    argv.push_back("-psfMode");
    argv.push_back("PostSmooth");

    spect::cli::ReconArgs args;
    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, PublicCtacModelRejectsLegacyPerVoxel) {
    auto argv = compactBaseArgs();
    argv.push_back("-ctacModel");
    argv.push_back("per-voxel");

    spect::cli::ReconArgs args;
    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, PublicPriorFlagIsRejected) {
    auto argv = compactBaseArgs();
    argv.push_back("-prior");
    argv.push_back("prior.mhd");

    spect::cli::ReconArgs args;
    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, CompactSymmetricParserReadsDefaultChunkCSRDirFromConfig) {
    const auto dir =
        std::filesystem::temp_directory_path() / "spect_matrix_config_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "spect_config.ini";

    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "[SystemMatrix]\n";
        out << "default_chunk_csr_dir = matrix/default_psc\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", path.string(),
        "-df", "proj.bin",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.matrixLayout, "compact-symmetric");
    const std::filesystem::path resolved(args.chunkCSRDir);
    EXPECT_TRUE(resolved.is_absolute());
    EXPECT_EQ(resolved.filename().string(), "default_psc");
    EXPECT_EQ(resolved.parent_path().filename().string(), "matrix");

    std::filesystem::remove(path);
}
TEST(IOTest, ExplicitLoadCSROverridesConfigDefaultChunkCSRDir) {
    const auto dir =
        std::filesystem::temp_directory_path() / "spect_matrix_config_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "spect_config_loadcsr.ini";

    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "[SystemMatrix]\n";
        out << "default_chunk_csr_dir = matrix/default_psc\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", path.string(),
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-matrixLayout", "standard",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.matrixLayout, "standard");
    EXPECT_EQ(args.loadCSRPath, "matrix.csrb");
    EXPECT_TRUE(args.chunkCSRDir.empty());

    std::filesystem::remove(path);
}
TEST(IOTest, PsfFlagDefaultsToOperatorMode) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-psf");
    argv.push_back("1.5,5");

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_TRUE(args.usePSF);
    EXPECT_EQ(args.psf.mode, spect::filters::PSFMode::Operator);
    EXPECT_NEAR(args.psf.fwhm_mm, 1.5, 1e-12);
    EXPECT_NEAR(args.psf.min_sigmas, 5.0, 1e-12);
}
TEST(IOTest, PsfFlagWithoutValueUsesDefaultFwhmAndTruncation) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-psf");

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_TRUE(args.usePSF);
    EXPECT_EQ(args.psf.mode, spect::filters::PSFMode::Operator);
    EXPECT_NEAR(args.psf.fwhm_mm, 1.5, 1e-12);
    EXPECT_NEAR(args.psf.min_sigmas, 4.0, 1e-12);
}
TEST(IOTest, ConfigSupportsXyOnlyOperatorPsf) {
    const auto path = std::filesystem::temp_directory_path() /
                      "spect_xy_only_psf_config.ini";
    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "[PSF]\nmode = Operator\nfwhm = 1.5\nz_blur = false\n";
    }

    auto argv = compactBaseArgs();
    argv[2] = path.string();
    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_TRUE(args.usePSF);
    EXPECT_EQ(args.psf.mode, spect::filters::PSFMode::Operator);
    EXPECT_FALSE(args.psf.z_blur);
    std::filesystem::remove(path);
}
TEST(IOTest, ExplicitPsfModeNoneOverridesPsfDefaultOperator) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-psfMode");
    argv.push_back("None");
    argv.push_back("-psf");
    argv.push_back("1.5");

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_TRUE(args.usePSF);
    EXPECT_EQ(args.psf.mode, spect::filters::PSFMode::None);
}
TEST(IOTest, CompactSymmetricParserRejectsLoadCSRAndChunkCSRDirTogether) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-chunkCSRDir");
    argv.push_back("matrix_dir");

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}
TEST(IOTest, StandardParserRejectsCompactDenomCachePathWithoutCompactLayout) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-compactDenomCache", "cache/psc_260601_global_denom.mhd",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}
TEST(IOTest, CompactSymmetricParserRejectsUnsupportedPhase1Options) {
    auto expectRejected = [](std::vector<std::string> argv) {
        spect::cli::ReconArgs args;
        EXPECT_FALSE(parseReconArgsForTest(argv, args));
    };

    {
        auto argv = compactBaseArgs();
        argv.push_back("-sparseBackend");
        argv.push_back("cpu");
        expectRejected(argv);
    }
    {
        auto argv = compactBaseArgs();
        auto it = std::find(argv.begin(), argv.end(), "mlem");
        ASSERT_NE(it, argv.end());
        *it = "admm";
        expectRejected(argv);
    }
    {
        auto argv = compactBaseArgs();
        auto it = std::find(argv.begin(), argv.end(), "1");
        ASSERT_NE(it, argv.end());
        *it = "2";
        expectRejected(argv);
    }
    {
        auto argv = compactBaseArgs();
        argv.push_back("-ctac");
        expectRejected(argv);
    }
    {
        auto argv = compactBaseArgs();
        argv.push_back("-reg");
        argv.push_back("RDP,0.01,1e-4,2.0");
        expectRejected(argv);
    }
    {
        auto argv = compactBaseArgs();
        argv.push_back("-psf");
        argv.push_back("1.5");
        argv.push_back("-psfProj");
        argv.push_back("on");
        expectRejected(argv);
    }
    {
        auto argv = compactBaseArgs();
        argv.push_back("-psfMode");
        argv.push_back("PostSmooth");
        expectRejected(argv);
    }
}
TEST(IOTest, CompactSymmetricParserAcceptsBSREMRDP) {
    auto argv = compactBaseArgs();
    auto it = std::find(argv.begin(), argv.end(), "mlem");
    ASSERT_NE(it, argv.end());
    *it = "bsrem";
    argv.push_back("-reg");
    argv.push_back("RDP,0.01,1e-4,2.0");
    argv.push_back("-bsrem");
    argv.push_back("1.0,0.0");

    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.algorithm, spect::cli::ReconArgs::Algorithm::BSREM);
    EXPECT_TRUE(args.useReg);
    EXPECT_EQ(args.reg.type, spect::RegType::RDP);
    EXPECT_DOUBLE_EQ(args.reg.beta, 0.01);
    EXPECT_DOUBLE_EQ(args.reg.gamma, 1e-4);
    EXPECT_DOUBLE_EQ(args.reg.delta, 2.0);
    EXPECT_FLOAT_EQ(args.bsrem_alpha0, 1.0f);
    EXPECT_FLOAT_EQ(args.bsrem_gamma, 0.0f);
}
TEST(IOTest, CompactSymmetricParserAcceptsBSREMTV) {
    auto argv = compactBaseArgs();
    auto it = std::find(argv.begin(), argv.end(), "mlem");
    ASSERT_NE(it, argv.end());
    *it = "bsrem";
    argv.push_back("-reg");
    argv.push_back("TV,0.01,1e-6");
    argv.push_back("-bsrem");
    argv.push_back("1.0,0.0");

    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.algorithm, spect::cli::ReconArgs::Algorithm::BSREM);
    EXPECT_TRUE(args.useReg);
    EXPECT_EQ(args.reg.type, spect::RegType::TVIsotropic);
    EXPECT_DOUBLE_EQ(args.reg.beta, 0.01);
    EXPECT_DOUBLE_EQ(args.reg.tv_eps, 1e-6);
}
TEST(IOTest, CompactSymmetricParserAcceptsBackgroundProjection) {
    auto argv = compactBaseArgs();
    argv.push_back("-bProj");
    argv.push_back("background.bin");

    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.matrixLayout, "compact-symmetric");
    EXPECT_EQ(args.bProjPath, "background.bin");
}
TEST(IOTest, ConfigParserStripsInlineComments) {
    const auto dir = std::filesystem::temp_directory_path() / "spect_config_parser_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "spect_config.ini";

    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "; full-line comment\n";
        out << "[Basic]\n";
        out << "dimensions = 108,108,80 ; reconstruction grid\n";
        out << "threads = 16 # worker count\n";
        out << "\n";
        out << "[PSF]\n";
        out << "mode = None ; None | Operator\n";
        out << "fwhm = 3.0 ; mm\n";
    }

    spect::cli::ConfigParser cfg;
    ASSERT_TRUE(cfg.load(path.string()));
    EXPECT_EQ(cfg.getString("Basic", "dimensions"), "108,108,80");
    EXPECT_EQ(cfg.getInt("Basic", "threads", 0), 16);
    EXPECT_EQ(cfg.getString("PSF", "mode"), "None");
    EXPECT_DOUBLE_EQ(cfg.getDouble("PSF", "fwhm", 0.0), 3.0);

    std::filesystem::remove(path);
}

TEST(IOTest, CtacAutoMuNuclideConfigResolvesWaterMu) {
    const auto dir = std::filesystem::temp_directory_path() / "spect_ctac_config_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "spect_config.ini";

    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "[Basic]\n";
        out << "dimensions = 8,8,8\n";
        out << "voxel_size = 1,1,1\n";
        out << "[Detector]\n";
        out << "angles = 1\n";
        out << "[CTAC]\n";
        out << "auto_mu_nuclide = I131\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", path.string(),
        "-df", "proj.bin",
        "-loadCSR", "matrix.csrb",
        "-matrixLayout", "standard",
        "-opti", "mlem",
        "-subsets", "1",
        "-it", "1"
    };
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.ctacAutoMuValueSource, "i131");
    EXPECT_NEAR(args.ctacAutoMuValue, 0.010998703003107506, 1e-15);

    std::filesystem::remove(path);
}

} // namespace test
} // namespace io
} // namespace spect
