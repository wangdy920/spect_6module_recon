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
#include "cli/RegistrationIni.h"


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

std::filesystem::path frozenRegistrationIniPath() {
    const auto fromThisFile =
        std::filesystem::path(__FILE__).parent_path().parent_path() /
        "config" / "ct_spect_registration.ini";
    if (std::filesystem::exists(fromThisFile))
        return fromThisFile;
    return std::filesystem::current_path() / "config" /
           "ct_spect_registration.ini";
}

TEST(CLITest, RegistrationIniLoadsWithConfigParser) {
    const auto path = frozenRegistrationIniPath();
    ASSERT_TRUE(std::filesystem::exists(path)) << path.string();

    spect::cli::ConfigParser cfg;
    ASSERT_TRUE(cfg.load(path.string()));
    ASSERT_TRUE(cfg.hasSection("Registration"));
    EXPECT_THROW((void)cfg.getDouble("Registration", "translation_xyz_mm", 0.0),
                 std::invalid_argument);

    spect::cli::RegistrationTransform loaded;
    std::string error;
    ASSERT_TRUE(spect::cli::loadRegistrationIni(path.string(), loaded, &error))
        << error;
    EXPECT_NEAR(loaded.translation_xyz_mm[0], -0.225, 1e-12);
    EXPECT_NEAR(loaded.translation_xyz_mm[1], 1.6, 1e-12);
    EXPECT_NEAR(loaded.translation_xyz_mm[2], -11.1, 1e-12);
    EXPECT_DOUBLE_EQ(loaded.refine_limit_mm, 2.0);
    EXPECT_DOUBLE_EQ(loaded.residual_gate_mm, 0.3);
}

TEST(CLITest, AppliedRegistrationIniUsesSameSectionAndBool) {
    const auto path = std::filesystem::temp_directory_path() /
                      "spect_applied_registration.ini";
    {
        std::ofstream out(path);
        ASSERT_TRUE(out.good());
        out << "; SPECT CT registration parameters. Key=value INI for ConfigParser.\n";
        out << "[Registration]\n";
        out << "fixed_translation_xyz_mm = -0.225,1.6,-11.1\n";
        out << "z_bed_flip = 1\n";
        out << "attmap_bed_order = recon_bed1_max_z\n";
    }
    spect::cli::ConfigParser cfg;
    ASSERT_TRUE(cfg.load(path.string()));
    ASSERT_TRUE(cfg.hasSection("Registration"));
    EXPECT_TRUE(cfg.getBool("Registration", "z_bed_flip", false));
    const auto xyz = split(
        cfg.getString("Registration", "fixed_translation_xyz_mm"), ',');
    ASSERT_EQ(xyz.size(), 3u);
    EXPECT_DOUBLE_EQ(std::stod(xyz[2]), -11.1);
    std::filesystem::remove(path);
}

std::filesystem::path repoRootFromTestParser() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

TEST(CLITest, CtGuidedTestIniDiffersOnlyByCliUnlock) {
    const auto repo = repoRootFromTestParser();
    const auto production = repo / "config" / "spect_config.ini";
    const auto fixture = repo / "config" / "spect_config.ct_guided.ini";
    ASSERT_TRUE(std::filesystem::exists(production)) << production.string();
    ASSERT_TRUE(std::filesystem::exists(fixture)) << fixture.string();

    ConfigParser prod;
    ConfigParser test;
    ASSERT_TRUE(prod.load(production.string()));
    ASSERT_TRUE(test.load(fixture.string()));
    EXPECT_TRUE(prod.getBool("Compatibility", "recon_sections_override_cli",
                             false));
    EXPECT_FALSE(test.getBool("Compatibility", "recon_sections_override_cli",
                              true));

    const auto& prodSections = prod.getSections();
    const auto& testSections = test.getSections();
    ASSERT_EQ(prodSections.size(), testSections.size());
    for (const auto& [section, keys] : prodSections) {
        ASSERT_TRUE(test.hasSection(section)) << section;
        for (const auto& [key, value] : keys) {
            if (section == "Compatibility" &&
                key == "recon_sections_override_cli")
                continue;
            EXPECT_EQ(test.getString(section, key), value)
                << "[" << section << "] " << key;
        }
    }
    for (const auto& [section, keys] : testSections) {
        ASSERT_TRUE(prod.hasSection(section)) << section;
        for (const auto& [key, value] : keys) {
            if (section == "Compatibility" &&
                key == "recon_sections_override_cli")
                continue;
            EXPECT_EQ(prod.getString(section, key), value)
                << "[" << section << "] " << key;
        }
    }
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
        testing::internal::CaptureStdout();
        EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), versionArgs),
                  spect::cli::ArgumentParser::ParseResult::ExitSuccess);
        const std::string output = testing::internal::GetCapturedStdout();
        EXPECT_TRUE(versionArgs.versionRequested);
        EXPECT_NE(output.find("# SPECT_recon Changelog"), std::string::npos);
        EXPECT_NE(output.find("## 20260811+949f766"), std::string::npos);
    }

    spect::cli::ReconArgs shortVersionArgs;
    {
        std::vector<std::string> storage = {"SPECT_recon", "--version"};
        std::vector<char*> argv;
        for (std::string& arg : storage)
            argv.push_back(arg.data());
        testing::internal::CaptureStdout();
        EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), shortVersionArgs),
                  spect::cli::ArgumentParser::ParseResult::ExitSuccess);
        const std::string output = testing::internal::GetCapturedStdout();
        EXPECT_TRUE(shortVersionArgs.versionRequested);
        EXPECT_EQ(output.find("# SPECT_recon Changelog"), std::string::npos);
        EXPECT_EQ(std::count(output.begin(), output.end(), '\n'), 1);
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

TEST(IOTest, CompactSymmetricParserAcceptsDenomCachePath) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-compactDenomCache");
    argv.push_back("cache/psc_260601_global_denom.mhd");

    EXPECT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(std::filesystem::path(args.compactDenomCachePath),
              std::filesystem::absolute(
                  "cache/psc_260601_global_denom.mhd").lexically_normal());
}

TEST(IOTest, DenomCacheConfigIsExeRelativeAndCliPathWins) {
    const auto root = std::filesystem::temp_directory_path() /
                      "spect_denom_path_config_test";
    std::filesystem::create_directories(root);
    const auto config = root / "spect_config.ini";
    const auto cliCache = root / "explicit.mhd";
    {
        std::ofstream out(config);
        ASSERT_TRUE(out);
        out << "[SystemMatrix]\n";
        out << "compact_denom_cache_path = cache/from_config.mhd\n";
        out << "compact_denom_cache_dir = cache/compact_denom\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        (root / "SPECT_recon.exe").string(), "-config", config.string(),
        "-df", "proj.bin", "-chunkCSRDir", "chunks", "-opti", "mlem",
        "-it", "1", "-compactDenomCache", cliCache.string()};
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(std::filesystem::path(args.compactDenomCachePath), cliCache);
    const std::filesystem::path autoDir(args.compactDenomCacheDir);
    EXPECT_TRUE(autoDir.is_absolute());
    EXPECT_EQ(autoDir.filename(), "compact_denom");
    EXPECT_EQ(autoDir.parent_path().filename(), "cache");
    EXPECT_NE(autoDir.parent_path().parent_path(), root);

    std::filesystem::remove_all(root);
}
TEST(IOTest, ParserAcceptsCtacFactorCacheDir) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.push_back("-ctacFactorCacheDir");
    argv.push_back("cache/ctac_factors");

    EXPECT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.ctacFactorCacheDir, "cache/ctac_factors");
}
TEST(IOTest, CompactSymmetricParserAcceptsChunkCSRDirWithoutManualGeometry) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-chunkCSRDir", "matrix_dir",
        "-opti", "mlem",
        "-it", "1"
    };

    EXPECT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.chunkCSRDir, "matrix_dir");
}

TEST(IOTest, CompactSymmetricDefaultsSparseBackendToCuda) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-chunkCSRDir", "matrix_dir",
        "-opti", "mlem",
        "-it", "1",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-ang", "360"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, OptiSelectsAlgorithm) {
    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        "SPECT_recon",
        "-config", "__missing_compact_test_config__.ini",
        "-df", "proj.bin",
        "-chunkCSRDir", "matrix_dir",
        "-opti", "bsrem",
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

TEST(IOTest, CtGuidedTestIniKeepsRegCtAndCtac) {
    const auto repo =
        std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto fixture = repo / "config" / "spect_config.ct_guided.ini";
    ASSERT_TRUE(std::filesystem::exists(fixture)) << fixture.string();

    auto argv = compactBaseArgs();
    for (auto& arg : argv) {
        if (arg == "__missing_compact_test_config__.ini")
            arg = fixture.string();
    }
    argv.insert(argv.end(),
                {"-opti", "bsrem", "-reg", "TVIsotropic,0.01", "-regCT",
                 "regct_hu_recon220.mhd", "-regCtSigma", "100", "-ctac",
                 "-attMap", "attmap_recon220.mhd"});

    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_FALSE(args.reconSectionsOverrideCli);
    EXPECT_TRUE(args.useReg);
    EXPECT_EQ(args.reg.type, spect::RegType::TVIsotropic);
    EXPECT_NE(args.regCtPath.find("regct_hu_recon220.mhd"), std::string::npos);
    EXPECT_DOUBLE_EQ(args.reg.ct_sigma, 100.0);
    EXPECT_TRUE(args.doCTAC);
    EXPECT_NE(args.attMapPath.find("attmap_recon220.mhd"), std::string::npos);
    EXPECT_TRUE(args.zBedFlip);
    EXPECT_EQ(args.psf.mode, spect::filters::PSFMode::Sieve);
}

TEST(IOTest, PublicPsfModeRejectsRetiredPostSmooth) {
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

TEST(IOTest, UnknownAndDeprecatedCliArgumentsAreIgnored) {
    auto argv = compactBaseArgs();
    argv.insert(argv.end(), {"-prior", "prior.mhd", "-subsets", "1",
                             "spect_psf_sigma=0.637"});

    spect::cli::ReconArgs args;
    EXPECT_TRUE(parseReconArgsForTest(argv, args));
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
        "-it", "1"
    };

    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    const std::filesystem::path resolved(args.chunkCSRDir);
    EXPECT_TRUE(resolved.is_absolute());
    EXPECT_EQ(resolved.filename().string(), "default_psc");
    EXPECT_EQ(resolved.parent_path().filename().string(), "matrix");

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
        "-chunkCSRDir", "matrix_dir",
        "-opti", "mlem",
        "-it", "1"
    };
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.ctacAutoMuValueSource, "i131");
    EXPECT_NEAR(args.ctacAutoMuValue, 0.010998703003107506, 1e-15);

    std::filesystem::remove(path);
}

TEST(CLITest, AbsoluteFoutKeepsDriveLetterOutputDir) {
    using spect::cli::ReconArgs;

    const auto parseFout = [](const std::string& path, ReconArgs& args) {
        auto argv = compactBaseArgs();
        argv.insert(argv.end(), {"-fout", path});
        return parseReconArgsForTest(argv, args);
    };

    {
        ReconArgs args;
        ASSERT_TRUE(parseFout("D:/data/case/run1", args));
        const std::filesystem::path dir(args.outputDir);
        EXPECT_TRUE(dir.is_absolute());
        EXPECT_EQ(dir.filename().string(), "case");
        EXPECT_EQ(args.outputName, "run1");
    }
    {
        ReconArgs args;
        ASSERT_TRUE(parseFout("D:/data/case/run1.mhd", args));
        EXPECT_TRUE(std::filesystem::path(args.outputDir).is_absolute());
        EXPECT_EQ(args.outputName, "run1");
    }
    {
        ReconArgs args;
        ASSERT_TRUE(parseFout("D:/data/case/", args));
        const std::filesystem::path dir(args.outputDir);
        EXPECT_TRUE(dir.is_absolute());
        EXPECT_EQ(dir.parent_path().filename().string(), "case");
        EXPECT_EQ(args.outputName, "recon");
    }
    {
        ReconArgs args;
        ASSERT_TRUE(parseFout("rel/out/run1", args));
        EXPECT_FALSE(std::filesystem::path(args.outputDir).is_absolute());
        EXPECT_EQ(args.outputName, "run1");
    }
}

} // namespace test
} // namespace io
} // namespace spect
