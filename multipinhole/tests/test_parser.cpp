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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/StringUtils.h"
#include "cli/ConfigParser.h"
#include "pipeline/ReconArgsPrinter.h"


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

TEST(CLITest, ConfigParserTreatsBlankValuesAsNotConfigured) {
    const auto dir = std::filesystem::temp_directory_path();
    const auto lowerPath = dir / "spect_blank_lower.ini";
    const auto upperPath = dir / "spect_blank_upper.ini";
    {
        std::ofstream lower(lowerPath);
        lower << "[Detector]\nbad_detector_file = bad_dector\nangles = 360\n";
        std::ofstream upper(upperPath);
        upper << "[Detector]\nbad_detector_file =   ; comment only\nangles =\n"
              << "[Blank]\nkey =\n";
    }
    ConfigParser lower;
    ConfigParser upper;
    ASSERT_TRUE(lower.load(lowerPath.string()));
    ASSERT_TRUE(upper.load(upperPath.string()));
    EXPECT_FALSE(upper.hasKey("Detector", "bad_detector_file"));
    EXPECT_FALSE(upper.hasSection("Blank"));
    lower.overlay(upper);
    EXPECT_EQ(lower.getString("Detector", "bad_detector_file"), "bad_dector");
    EXPECT_EQ(lower.getInt("Detector", "angles", 0), 360);
    std::filesystem::remove(lowerPath);
    std::filesystem::remove(upperPath);
}

std::filesystem::path frozenRegistrationIniPath() {
    return std::filesystem::path(__FILE__).parent_path().parent_path() /
           "config" / "ct_spect_registration.ini";
}

TEST(CLITest, RegistrationIniUsesStringTripleAndTypedLimits) {
    ConfigParser cfg;
    const auto path = frozenRegistrationIniPath();
    ASSERT_TRUE(cfg.load(path.string())) << path.string();
    const auto xyz = split(
        cfg.getString("Registration", "translation_xyz_mm"), ',');
    ASSERT_EQ(xyz.size(), 3u);
    EXPECT_DOUBLE_EQ(std::stod(xyz[0]), -0.225);
    EXPECT_DOUBLE_EQ(std::stod(xyz[1]), 1.6);
    EXPECT_DOUBLE_EQ(std::stod(xyz[2]), -11.1);
    EXPECT_DOUBLE_EQ(
        cfg.getDouble("Registration", "refine_limit_mm", 0.0), 2.0);
    EXPECT_DOUBLE_EQ(
        cfg.getDouble("Registration", "residual_gate_mm", 0.0), 0.3);
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
        out << "regct_layout = reconstruction_storage_z\n";
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

TEST(CLITest, CtGuidedTestIniMatchesUnlockedProductionDefaults) {
    const auto repo = repoRootFromTestParser();
    const auto production = repo / "config" / "spect_config.ini";
    const auto fixture = repo / "config" / "spect_config.ct_guided.ini";
    ASSERT_TRUE(std::filesystem::exists(production)) << production.string();
    ASSERT_TRUE(std::filesystem::exists(fixture)) << fixture.string();

    ConfigParser prod;
    ConfigParser test;
    ASSERT_TRUE(prod.load(production.string()));
    ASSERT_TRUE(test.load(fixture.string()));
    EXPECT_FALSE(prod.getBool("Compatibility", "recon_sections_override_cli",
                             true));
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
#include "pipeline/RunLogger.h"
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
            "-chunkCSRDir", "chunks", "-dumpConf"
        };
        std::vector<char*> argv;
        for (std::string& arg : storage)
            argv.push_back(arg.data());
        EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), dumpArgs),
                  spect::cli::ArgumentParser::ParseResult::ExitSuccess);
    }
}

TEST(IOTest, VersionFlagWithValueIsRejectedAsVerbosity) {
    spect::cli::ArgumentParser parser;
    spect::cli::ReconArgs args;
    std::vector<std::string> storage = {"SPECT_recon", "-v", "1"};
    std::vector<char*> argv;
    for (std::string& arg : storage)
        argv.push_back(arg.data());
    testing::internal::CaptureStderr();
    EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), args),
              spect::cli::ArgumentParser::ParseResult::ExitFailure);
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(args.versionRequested);
    EXPECT_NE(err.find("use -vb"), std::string::npos);
}

TEST(IOTest, BedsOptionParsesCountAndRange) {
    spect::cli::ArgumentParser parser;
    auto parseBeds = [&](const std::string &value, spect::cli::ReconArgs &args) {
        std::vector<std::string> storage = {"SPECT_recon", "-chunkCSRDir", "chunks",
                                            "-beds", value, "-dumpConf"};
        std::vector<char*> argv;
        for (std::string& arg : storage)
            argv.push_back(arg.data());
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        const auto result = parser.parse(static_cast<int>(argv.size()), argv.data(), args);
        testing::internal::GetCapturedStdout();
        testing::internal::GetCapturedStderr();
        return result;
    };
    using Result = spect::cli::ArgumentParser::ParseResult;
    {
        spect::cli::ReconArgs args;
        EXPECT_EQ(parseBeds("12", args), Result::ExitSuccess);
        EXPECT_EQ(args.bedFirst, 1);
        EXPECT_EQ(args.bedLast, 12);
    }
    {
        spect::cli::ReconArgs args;
        EXPECT_EQ(parseBeds("3:7", args), Result::ExitSuccess);
        EXPECT_EQ(args.bedFirst, 3);
        EXPECT_EQ(args.bedLast, 7);
    }
    for (const std::string bad : {"0", "5:3", "x", "2:", "1:4x", "0:3"}) {
        spect::cli::ReconArgs args;
        EXPECT_EQ(parseBeds(bad, args), Result::ExitFailure) << bad;
    }
}

TEST(IOTest, IterAndVerboseAliasesAreRejected) {
    spect::cli::ArgumentParser parser;
    {
        spect::cli::ReconArgs args;
        std::vector<std::string> storage = {"SPECT_recon", "-iter", "20:20"};
        std::vector<char*> argv;
        for (std::string& arg : storage)
            argv.push_back(arg.data());
        testing::internal::CaptureStderr();
        EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), args),
                  spect::cli::ArgumentParser::ParseResult::ExitFailure);
        const std::string err = testing::internal::GetCapturedStderr();
        EXPECT_NE(err.find("use -it"), std::string::npos);
    }
    {
        spect::cli::ReconArgs args;
        std::vector<std::string> storage = {"SPECT_recon", "-verbose", "1"};
        std::vector<char*> argv;
        for (std::string& arg : storage)
            argv.push_back(arg.data());
        testing::internal::CaptureStderr();
        EXPECT_EQ(parser.parse(static_cast<int>(argv.size()), argv.data(), args),
                  spect::cli::ArgumentParser::ParseResult::ExitFailure);
        const std::string err = testing::internal::GetCapturedStderr();
        EXPECT_NE(err.find("use -vb"), std::string::npos);
    }
}

TEST(IOTest, CompactDenomCacheJoinsSplitWindowsDriveLetter) {
    spect::cli::ReconArgs args;
    auto argv = compactBaseArgs();
    argv.insert(argv.end(), {"-compactDenomCache", "D:", "\\cache\\foo.mhd"});
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_NE(args.compactDenomCachePath.find("foo.mhd"), std::string::npos);
    EXPECT_NE(args.compactDenomCachePath, std::string("D:"));
}

TEST(IOTest, IterProgressLineReachesStdoutAtQuietVerbosity) {
    const int previous = spect::main::getVerbosity();
    auto* previousLog = spect::main::getFileLogStream();
    std::ostringstream fileLog;
    spect::main::setVerbosity(0);
    spect::main::setFileLogStream(&fileLog);
    testing::internal::CaptureStdout();
    spect::main::logInfo("[ITER] 1/50 compact");
    spect::main::logDebug("[SUBSET] 1 subset 0/10 (1 of 10)");
    spect::main::logInfo("Reconstruction finished successfully.");
    spect::main::logInfo("Loaded configuration from x");
    const std::string out = testing::internal::GetCapturedStdout();
    spect::main::setFileLogStream(previousLog);
    spect::main::setVerbosity(previous);
    EXPECT_NE(out.find("[ITER] 1/50 compact"), std::string::npos);
    EXPECT_EQ(out.find("[SUBSET]"), std::string::npos);
    EXPECT_EQ(fileLog.str().find("[SUBSET]"), std::string::npos);
    EXPECT_NE(out.find("Reconstruction finished successfully."), std::string::npos);
    EXPECT_EQ(out.find("Loaded configuration"), std::string::npos);
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
        "SPECT_recon", "-chunkCSRDir", "chunks", "-projectionResidual", "on", "-dumpConf"};
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

    { std::ofstream out(config, std::ios::app);
      out << "[Compatibility]\nrecon_sections_override_cli=true\n"; }
    spect::cli::ReconArgs protectedArgs;
    ASSERT_TRUE(parseReconArgsForTest(argv, protectedArgs));
    EXPECT_EQ(std::filesystem::path(protectedArgs.compactDenomCachePath),
              autoDir.parent_path()/"from_config.mhd");
    EXPECT_FALSE(protectedArgs.compactDenomCachePathExplicit);

    std::filesystem::remove_all(root);
}

TEST(IOTest, ConfiguredIterationsProtectedButRuntimeOutputRemainsCli) {
    const auto root = std::filesystem::temp_directory_path() /
                      "spect_cli_iteration_output_priority_test";
    std::filesystem::create_directories(root);
    const auto config = root / "spect_config.ini";
    const auto output = root / "cli_output" / "run";
    {
        std::ofstream out(config);
        ASSERT_TRUE(out);
        out << "[Compatibility]\n";
        out << "recon_sections_override_cli = true\n";
        out << "[Reconstruction]\n";
        out << "iterations = 50:10\n";
        out << "[InputOutput]\n";
        out << "output_dir = config_output\n";
        out << "output_name = config_recon\n";
    }

    spect::cli::ReconArgs args;
    const std::vector<std::string> argv = {
        (root / "SPECT_recon.exe").string(), "-config", config.string(),
        "-df", "proj.bin", "-chunkCSRDir", "chunks", "-opti", "mlem",
        "-it", "20:5", "-save-every", "7", "-fout", output.string()};
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.numIters, 50);
    EXPECT_EQ(args.subsetSchedule, "10");
    EXPECT_EQ(args.saveInterval, 0); // auto-selected legacy recipe explicitly requests final-only
    EXPECT_EQ(std::filesystem::path(args.outputDir), output.parent_path());
    EXPECT_EQ(args.outputName, "run");

    std::filesystem::remove_all(root);
}

TEST(IOTest, NormalizationBindsPerIsotopeInGlobalConfigOnly) {
    const auto root = std::filesystem::temp_directory_path() /
                      "spect_cli_global_normalization_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto config = root / "spect_config.ini";
    const auto recipe = root / "recipe.conf";
    const auto writeText = [](const std::filesystem::path& path, const std::string& text) {
        std::ofstream out(path, std::ios::trunc);
        out << text;
    };
    const std::string protectedHeader = "[Compatibility]\nrecon_sections_override_cli = true\n";
    writeText(config, protectedHeader);

    auto argv = spect::io::test::compactBaseArgs();
    argv[2] = config.string();
    argv.insert(argv.end(), {"-isotope", "Tc99m"});
    spect::cli::ReconArgs probe;
    ASSERT_TRUE(parseReconArgsForTest(argv, probe));
    const std::uint32_t angles = static_cast<std::uint32_t>(probe.numAngles);
    // Without a geometry config the detector size stays 0 and any per-bin count is accepted.
    const std::uint32_t rays = probe.detNz > 0 && probe.detNx > 0
        ? static_cast<std::uint32_t>(probe.detNz * probe.detNx) : 4u;
    ASSERT_GT(angles, 0u);
    const auto writeNorm = [&](const std::filesystem::path& path) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&angles), sizeof(angles));
        out.write(reinterpret_cast<const char*>(&rays), sizeof(rays));
        const std::vector<double> ones(static_cast<size_t>(angles) * rays, 1.0);
        out.write(reinterpret_cast<const char*>(ones.data()),
                  static_cast<std::streamsize>(ones.size() * sizeof(double)));
    };
    const auto tcNorm = root / "tc_norm.bin";
    const auto luNorm = root / "lu_norm.bin";
    const auto cliNorm = root / "cli_norm.bin";
    for (const auto& path : {tcNorm, luNorm, cliNorm})
        writeNorm(path);
    const auto same = [](const std::string& actual, const std::filesystem::path& expected) {
        return std::filesystem::path(actual).lexically_normal() == expected.lexically_normal();
    };

    writeText(config, protectedHeader + "[Normalization]\nfactor_file_Tc99m = " +
                          tcNorm.generic_string() + "\nfactor_file_Lu177 = " +
                          luNorm.generic_string() + "\n");
    spect::cli::ReconArgs tc;
    ASSERT_TRUE(parseReconArgsForTest(argv, tc));
    EXPECT_TRUE(same(tc.normFactorFile, tcNorm)) << tc.normFactorFile;
    auto luArgv = argv;
    luArgv.back() = "Lu177";
    spect::cli::ReconArgs lu;
    ASSERT_TRUE(parseReconArgsForTest(luArgv, lu));
    EXPECT_TRUE(same(lu.normFactorFile, luNorm)) << lu.normFactorFile;

    // An explicit CLI file replaces the configured binding, also under protection.
    auto cliArgv = argv;
    cliArgv.insert(cliArgv.end(), {"-normFactorFile", cliNorm.string()});
    spect::cli::ReconArgs cli;
    ASSERT_TRUE(parseReconArgsForTest(cliArgv, cli));
    EXPECT_TRUE(same(cli.normFactorFile, cliNorm)) << cli.normFactorFile;

    // A blank recipe key is not configured, so older packaged recipes still load.
    auto recipeArgv = argv;
    recipeArgv.insert(recipeArgv.end(), {"-algoConf", recipe.string()});
    writeText(recipe, "[Algorithm]\nname = MLEM\n[Normalization]\nfactor_file =\n");
    spect::cli::ReconArgs blankRecipe;
    ASSERT_TRUE(parseReconArgsForTest(recipeArgv, blankRecipe));
    EXPECT_TRUE(same(blankRecipe.normFactorFile, tcNorm)) << blankRecipe.normFactorFile;

    writeText(recipe, "[Algorithm]\nname = MLEM\n[Normalization]\nfactor_file_Tc99m = " +
                          tcNorm.generic_string() + "\n");
    spect::cli::ReconArgs filledRecipe;
    EXPECT_FALSE(parseReconArgsForTest(recipeArgv, filledRecipe));

    for (const std::string misplaced : {"[Normalization]\nfactor_file = x.bin\n",
                                        "[Normalization]\nfactor_file_Tc99 = x.bin\n",
                                        "[MultiBed]\nnormalization_factor_file = x.bin\n"}) {
        writeText(config, protectedHeader + misplaced);
        spect::cli::ReconArgs rejected;
        EXPECT_FALSE(parseReconArgsForTest(argv, rejected)) << misplaced;
    }

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
    argv.insert(argv.end(), {"-prior", "prior.mhd",
                             "spect_psf_sigma=0.637"});

    spect::cli::ReconArgs args;
    EXPECT_TRUE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, CompactSymmetricParserAcceptsSubsetSchedule) {
    auto argv = compactBaseArgs();
    argv.insert(argv.end(), {"-subsets", "10:3,5:2,1:5"});

    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.subsetSchedule, "10:3,5:2,1:5");
}

TEST(IOTest, IterationsUseSubsetCountAndIndependentCheckpointFrequency) {
    auto argv = compactBaseArgs();
    argv.insert(argv.end(), {"-it", "100:10", "-save-every", "5"});
    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.numIters, 100);
    EXPECT_EQ(args.subsetSchedule, "10");
    EXPECT_EQ(args.saveInterval, 5);

    argv = compactBaseArgs();
    argv.insert(argv.end(), {"-it", "100:1"});
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.subsetSchedule, "1");
    EXPECT_EQ(args.saveInterval, 0);
    for (const std::string value : {"0:1", "10:0", "10:", "10:2:3", "10x:2"}) {
        argv = compactBaseArgs();
        argv.insert(argv.end(), {"-it", value});
        spect::cli::ReconArgs invalid;
        EXPECT_FALSE(parseReconArgsForTest(argv, invalid)) << value;
    }
    for (const bool reverse : {false, true}) {
        argv = compactBaseArgs();
        const std::vector<std::string> extra = reverse
            ? std::vector<std::string>{"-subsets", "5", "-it", "100:10"}
            : std::vector<std::string>{"-it", "100:10", "-subsets", "5"};
        argv.insert(argv.end(), extra.begin(), extra.end());
        spect::cli::ReconArgs invalid;
        EXPECT_FALSE(parseReconArgsForTest(argv, invalid));
    }
}

TEST(IOTest, ShippedConfigDefaultsToFinalOnlyAndButterOutput) {
    const auto config = std::filesystem::path(__FILE__).parent_path().parent_path() /
                        "config" / "spect_config.ini";
    const std::vector<std::string> argv = {"SPECT_recon", "-config", config.string(),
        "-df", "proj.bin", "-chunkCSRDir", "chunks", "-it", "1:1"};
    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.saveInterval, 0);
    EXPECT_EQ(args.smoothApplyStage, "none");
    EXPECT_EQ(args.butterApplyStage, "output");
    EXPECT_FALSE(args.outputFiltersOverwritePrimary);
    auto overridden = argv;
    overridden.insert(overridden.end(), {"-opti", "bsrem"});
    spect::cli::ReconArgs bsrem;
    ASSERT_TRUE(parseReconArgsForTest(overridden, bsrem));
    EXPECT_EQ(bsrem.algorithm, spect::cli::ReconArgs::Algorithm::BSREM);
    EXPECT_FALSE(bsrem.useReg); // -opti changes the optimizer, not the selected recipe.

}

TEST(IOTest, CompactSymmetricParserRejectsRemovedPixelMode) {
    auto argv = compactBaseArgs();
    argv.insert(argv.end(), {"-subset-mode", "posem"});
    spect::cli::ReconArgs args;
    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, CompactSymmetricParserReadsSubsetScheduleFromConfig) {
    const auto path = std::filesystem::temp_directory_path() /
                      "spect_subset_schedule_config.ini";
    {
        std::ofstream out(path);
        ASSERT_TRUE(out);
        out << "[Algorithm]\nsubsets = 6:2,1:3\n";
    }

    const std::vector<std::string> argv = {
        "SPECT_recon", "-algoConf", path.string(), "-df", "proj.bin",
        "-chunkCSRDir", "chunks", "-opti", "mlem", "-it", "5",
        "-vox", "0.3", "-ang", "360"};
    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_EQ(args.subsetSchedule, "6:2,1:3");
    std::filesystem::remove(path);
}

TEST(IOTest, CompactSymmetricParserRejectsEmptySubsetSchedule) {
    auto argv = compactBaseArgs();
    argv.insert(argv.end(), {"-subsets", ""});

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

TEST(IOTest, CompactSymmetricParserAcceptsBsremSensFloor) {
    auto argv = compactBaseArgs();
    auto it = std::find(argv.begin(), argv.end(), "mlem");
    ASSERT_NE(it, argv.end());
    *it = "bsrem";
    argv.insert(argv.end(), {"-bsrem", "1.5,0", "-bsremSensFloor", "1e-4"});

    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_FLOAT_EQ(args.bsremSensFloorRel, 1e-4f);
}

TEST(IOTest, CompactSymmetricParserRejectsSensFloorOutOfRange) {
    auto argv = compactBaseArgs();
    auto it = std::find(argv.begin(), argv.end(), "mlem");
    ASSERT_NE(it, argv.end());
    *it = "bsrem";
    argv.insert(argv.end(), {"-bsrem", "1.5,0", "-bsremSensFloor", "1.5"});

    spect::cli::ReconArgs args;
    EXPECT_FALSE(parseReconArgsForTest(argv, args));
}

TEST(IOTest, CompactSymmetricParserAcceptsBsremEquivAndDiag) {
    auto argv = compactBaseArgs();
    auto it = std::find(argv.begin(), argv.end(), "mlem");
    ASSERT_NE(it, argv.end());
    *it = "bsrem";
    argv.insert(argv.end(),
                {"-bsrem", "1.0,0.0", "-bsremEquiv", "on", "-bsremDiag", "on",
                 "-bsremDiagInterval", "5"});

    spect::cli::ReconArgs args;
    ASSERT_TRUE(parseReconArgsForTest(argv, args));
    EXPECT_TRUE(args.bsremMlemEquivalenceMode);
    EXPECT_TRUE(args.bsremDiagnostics);
    EXPECT_EQ(args.bsremDiagInterval, 5);
    EXPECT_FALSE(args.useReg);
    EXPECT_EQ(args.reg.type, spect::RegType::None);
    EXPECT_FLOAT_EQ(args.bsrem_alpha0, 1.0f);
    EXPECT_FLOAT_EQ(args.bsrem_gamma, 0.0f);
    EXPECT_FALSE(args.bsremUseCurvDenom);
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

TEST(IOTest, RecipesCoverDefaultExplicitAndCliWithBothOverridePolicies) {
    const auto root = std::filesystem::temp_directory_path() / "spect_recipe_priority";
    std::filesystem::create_directories(root);
    const auto global = root / "spect_config.ini";
    const auto selected = root / "selected.conf";
    const auto fallback = root / "default.conf";
    { std::ofstream f(fallback); f << "[Algorithm]\nname=MLEM\niterations=50:10\n[Regularization]\ntype=None\n"; }
    { std::ofstream f(selected); f << "[Algorithm]\nname=BSREM\niterations=75:10\n[BSREM]\nalpha0=1.5\n"
        "[Regularization]\ntype=TV\nbeta=0.0003\n[PSF]\nmode=none\n[Mask]\nuse_cylinder=false\n"
        "[Filtering]\nbutter_apply_stage=none\n"; }
    for (const bool protect : {false, true}) {
        { std::ofstream f(global); f << "[Reconstruction]\ndefault_config=default.conf\n[Compatibility]\nrecon_sections_override_cli="
            << (protect ? "true" : "false") << "\n[PSF]\nmode=sieve\nfwhm=1.3\n[Filtering]\nbutter_apply_stage=output\n"; }
        for (int style=0; style<3; ++style) {
            std::vector<std::string> argv = {"SPECT_recon", "-config", global.string(), "-df", "proj.bin", "-chunkCSRDir", "chunks", "-o", "caller-output"};
            if (style) argv.insert(argv.end(), {"-algoConf", selected.string()});
            if (style==2) argv.insert(argv.end(), {"-it", "4:2", "-reg", "none", "-psfMode", "sieve", "-opti", "mlem"});
            spect::cli::ReconArgs a;
            ASSERT_TRUE(spect::io::test::parseReconArgsForTest(argv,a)) << protect << ':' << style;
            EXPECT_EQ(a.numIters, style==0 ? 50 : (style==2 && !protect ? 4 : 75));
            EXPECT_EQ(a.subsetSchedule, style==2 && !protect ? "2" : "10");
            EXPECT_EQ(a.useReg, style!=0 && (style!=2 || protect));
            EXPECT_EQ(a.usePSF, style==0 || (style==2 && !protect));
            EXPECT_EQ(a.useCylindricalMask, style==0);
            EXPECT_EQ(a.butterApplyStage, style==0 ? "output" : "none");
            EXPECT_EQ(a.outputDir,"caller-output");
            EXPECT_EQ(std::filesystem::path(a.algoConfPath), style ? selected : fallback);
            const int previous = spect::main::getVerbosity();
            auto* previousLog = spect::main::getFileLogStream();
            std::ostringstream log;
            spect::main::setVerbosity(0);
            spect::main::setFileLogStream(&log);
            spect::main::printReconArgs(a);
            spect::main::setFileLogStream(previousLog);
            spect::main::setVerbosity(previous);
            EXPECT_NE(log.str().find("[CONFIG] Effective reconstruction settings"), std::string::npos);
            EXPECT_NE(log.str().find("profile: " + a.algoConfPath), std::string::npos);
            EXPECT_NE(log.str().find(protect ? "profile > global > CLI" : "CLI > profile > global"), std::string::npos);
            const std::string effective = style==0 ? "algorithm=MLEM, iterations=50, subset_schedule=10"
                : (style==2 && !protect ? "algorithm=MLEM, iterations=4, subset_schedule=2"
                                       : "algorithm=BSREM, iterations=75, subset_schedule=10");
            EXPECT_NE(log.str().find(effective), std::string::npos) << log.str();
            if (style!=0 && (style!=2 || protect))
                EXPECT_NE(log.str().find("regularization: TVIsotropic, beta=0.0003"), std::string::npos);
        }
    }
    spect::cli::ReconArgs directRecipe;
    ASSERT_TRUE(spect::io::test::parseReconArgsForTest(
        {"SPECT_recon", "-config", selected.string(), "-df", "proj.bin", "-chunkCSRDir", "chunks"}, directRecipe));
    EXPECT_EQ(directRecipe.numIters,75);
    EXPECT_EQ(directRecipe.reg.type,spect::RegType::TVIsotropic);
    EXPECT_DOUBLE_EQ(directRecipe.reg.beta,.0003);
    EXPECT_FALSE(directRecipe.usePSF);
    std::filesystem::remove_all(root);
}

TEST(IOTest, MatrixIdsSelectUniqueNuclideResourcesAndRejectMismatch) {
    const auto root=std::filesystem::temp_directory_path()/"spect_matrix_ids";
    std::filesystem::create_directories(root/"matrix"/"Lu177"/"peak208");
    std::filesystem::create_directories(root/"matrix"/"Lu177"/"peak113");
    for (const auto* id : {"peak208","peak113"}) { std::ofstream f(root/"matrix"/"Lu177"/id/"metadata.bin"); f << "fixture"; }
    const auto config=root/"spect_config.ini";
    { std::ofstream f(config); f << "[Reconstruction]\ndefault_config=recipe.conf\ndefault_config_Lu177=recipe.conf\n[SystemMatrix]\nroot=" << (root/"matrix").generic_string() << "\n"; }
    { std::ofstream f(root/"recipe.conf"); f << "[Acquisition]\nisotope=Lu177\nprojection_model=joint\n[Algorithm]\niterations=30:10\n[SystemMatrix]\nmatrix_id=peak208\nmatrix_id2=peak113\nenergy_window_weights=0.1036,0.0617\n"; }
    const std::vector<std::string> base={"SPECT_recon","-config",config.string(),"-isotope","Lu177","-df","Bed1.bin","-df2","Bed1_113.bin"};
    spect::cli::ReconArgs a;
    ASSERT_TRUE(spect::io::test::parseReconArgsForTest(base,a));
    EXPECT_EQ(std::filesystem::path(a.chunkCSRDir).filename(),"peak208");
    EXPECT_EQ(std::filesystem::path(a.chunkCSRDir2).filename(),"peak113");
    EXPECT_EQ(a.energyWindowWeights,(std::vector<double>{.1036,.0617}));
    auto mismatch=base; mismatch[4]="Tc99m";
    spect::cli::ReconArgs invalid;
    EXPECT_FALSE(spect::io::test::parseReconArgsForTest(mismatch,invalid));
    std::filesystem::create_directories(root/"matrix"/"peak208");
    { std::ofstream f(root/"matrix"/"peak208"/"metadata.bin"); f << "ambiguous"; }
    spect::cli::ReconArgs ambiguous;
    EXPECT_FALSE(spect::io::test::parseReconArgsForTest(base,ambiguous));
    std::filesystem::remove_all(root);
}

TEST(IOTest, SharedProfileCannotOwnBedGeometryOrMatrixResources) {
    const auto root = std::filesystem::temp_directory_path() / "spect_profile_ownership";
    std::filesystem::create_directories(root);
    const auto config = root / "global.ini";
    const auto recipe = root / "recipe.conf";
    const auto scanner = root / "PSC30-MPH6.ini";
    { std::ofstream f(scanner); f << "[Scanner]\nname=PSC30-MPH6\n[Tc99m]\ninput_mode=single\n"
        "chunk_format=csrb\nsysmat_high=" << (root / "high").generic_string() << "\nsysmat_low=\n"; }
    { std::ofstream f(config); f << "[Reconstruction]\ndefault_config=recipe.conf\n"
        "[SystemMatrix]\nscanner_config=" << scanner.generic_string() << "\n"; }
    { std::ofstream f(recipe); f << "[Algorithm]\nname=MLEM\niterations=1:1\n"
        "[Acquisition]\nisotope=Tc99m\nprojection_model=single\n"
        "[MultiBed]\nbed_step_mm=12\n"; }
    const std::vector<std::string> base = {"SPECT_recon", "-config", config.string(), "-df", "proj.bin"};
    spect::cli::ReconArgs geometry;
    EXPECT_FALSE(spect::io::test::parseReconArgsForTest(base, geometry));
    { std::ofstream f(recipe); f << "[Algorithm]\nname=MLEM\niterations=1:1\n"
        "[Acquisition]\nisotope=Tc99m\nprojection_model=single\n"
        "[SystemMatrix]\nmatrix_id=profile_owned_matrix\n"; }
    spect::cli::ReconArgs matrix;
    EXPECT_FALSE(spect::io::test::parseReconArgsForTest(base, matrix));
    std::filesystem::remove_all(root);
}

TEST(IOTest, MaskModeSpecificCylinderDimensionsAreParsed) {
    const auto root = std::filesystem::temp_directory_path() / "spect_mask_dims";
    std::filesystem::create_directories(root);
    const auto config = root / "mask.ini";
    { std::ofstream f(config); f << "[Mask]\nuse_cylinder=true\n"
        "cylinder_dims_single=28,20\ncylinder_dims_multi=40,60\n"; }
    auto argv = spect::io::test::compactBaseArgs();
    argv[2] = config.string();
    spect::cli::ReconArgs args;
    ASSERT_TRUE(spect::io::test::parseReconArgsForTest(argv, args));
    EXPECT_TRUE(args.useCylindricalMask);
    EXPECT_TRUE(args.cylinderDimsSingleConfigured);
    EXPECT_FLOAT_EQ(args.cylinderDiameterSingleMM, 28.0f);
    EXPECT_FLOAT_EQ(args.cylinderHeightSingleMM, 20.0f);
    EXPECT_TRUE(args.cylinderDimsMultiConfigured);
    EXPECT_FLOAT_EQ(args.cylinderDiameterMultiMM, 40.0f);
    EXPECT_FLOAT_EQ(args.cylinderHeightMultiMM, 60.0f);
    std::filesystem::remove_all(root);
}

TEST(IOTest, JointEnergyWeightsMustBeFiniteAndPositive) {
    for (const auto &weights : {"inf,1", "1,inf", "nan,1", "1,nan", "0,1", "-1,1", "0.1036,0.0617"}) {
        auto argv = spect::io::test::compactBaseArgs();
        argv.insert(argv.end(), {"-chunkCSRDir2", "window2", "-df2", "window2.bin",
                                 "-energyWeights", weights});
        spect::cli::ReconArgs args;
        EXPECT_EQ(spect::io::test::parseReconArgsForTest(argv, args),
                  std::string(weights) == "0.1036,0.0617") << weights;
    }
}

TEST(IOTest, ScannerMatricesPreserveChannelOrderAndSkipUnusedSlots) {
    const auto root = std::filesystem::temp_directory_path() / "spect_scanner_matrices";
    std::filesystem::create_directories(root);
    const auto config = root / "global.ini";
    const auto scanner = root / "PSC30-MPH6.ini";
    { std::ofstream f(config); f << "[Acquisition]\nisotope=Lu177\nprojection_model=joint\n"
        "[SystemMatrix]\nenergy_window_weights=0.1036,0.0617\n"; }
    auto writeScanner = [&](const std::string& low) {
        std::ofstream f(scanner); f << "[Scanner]\nname=PSC30-MPH6\n[Lu177]\ninput_mode=prefer_split\n"
            "sysmat_high=" << (root / "high").generic_string() << "\nsysmat_low=" << low << "\n";
    };
    writeScanner((root / "low").generic_string());
    std::vector<std::string> base = {"SPECT_recon", "-config", config.string(), "-scannerConf", scanner.string(),
                                    "-df", "Bed2_high.bin", "-df2", "Bed2_low.bin"};
    spect::cli::ReconArgs joint;
    ASSERT_TRUE(spect::io::test::parseReconArgsForTest(base, joint));
    EXPECT_EQ(joint.projectionModel, "joint");
    EXPECT_EQ(joint.energyWindowWeights, (std::vector<double>{.1036, .0617}));
    EXPECT_EQ(std::filesystem::path(joint.chunkCSRDir2), root / "low");
    writeScanner("");
    spect::cli::ReconArgs missing;
    EXPECT_FALSE(spect::io::test::parseReconArgsForTest(base, missing));
    base.resize(7);
    spect::cli::ReconArgs partial;
    EXPECT_FALSE(spect::io::test::parseReconArgsForTest(base, partial));
    // Legacy configs without projection_model keep the prefer_split fallback.
    { std::ofstream f(config); f << "[Acquisition]\nisotope=Lu177\n"
        "[SystemMatrix]\nenergy_window_weights=0.1036,0.0617\n"; }
    base.back() = "Bed2.bin";
    spect::cli::ReconArgs merged;
    ASSERT_TRUE(spect::io::test::parseReconArgsForTest(base, merged));
    EXPECT_EQ(merged.projectionModel, "merged");
    EXPECT_TRUE(merged.chunkCSRDir2.empty());
    EXPECT_TRUE(merged.energyWindowWeights.empty());
    EXPECT_EQ(std::filesystem::path(merged.chunkCSRDir), root / "high");
    std::filesystem::remove_all(root);
}

TEST(IOTest, ProjectionModelMustMatchScannerInputMode) {
    const auto root = std::filesystem::temp_directory_path() / "spect_projection_contract";
    std::filesystem::create_directories(root);
    const auto config = root / "global.ini";
    const auto scanner = root / "PSC30-MPH6.ini";
    { std::ofstream f(scanner); f << "[Scanner]\nname=PSC30-MPH6\n[Lu177]\ninput_mode=merged\n"
        "chunk_format=csrb\nsysmat_high=" << (root / "high").generic_string() << "\nsysmat_low=\n"; }
    const std::vector<std::string> base = {"SPECT_recon", "-config", config.string(),
        "-scannerConf", scanner.string(), "-df", "Bed1.bin"};
    { std::ofstream f(config); f << "[Compatibility]\nrecon_sections_override_cli=true\n"
        "[Acquisition]\nisotope=Lu177\nprojection_model=joint\n";
    }
    spect::cli::ReconArgs mismatch;
    EXPECT_FALSE(spect::io::test::parseReconArgsForTest(base, mismatch));
    { std::ofstream f(config); f << "[Compatibility]\nrecon_sections_override_cli=true\n"
        "[Acquisition]\nisotope=Lu177\nprojection_model=merged\n";
    }
    spect::cli::ReconArgs merged;
    ASSERT_TRUE(spect::io::test::parseReconArgsForTest(base, merged));
    EXPECT_EQ(merged.projectionModel, "merged");
    EXPECT_EQ(merged.projectionInputMode, "merged");
    EXPECT_EQ(std::filesystem::path(merged.chunkCSRDir), root / "high");
    // Explicit joint is a first-class scanner mode, not a silent merged rewrite.
    { std::ofstream f(scanner); f << "[Scanner]\nname=PSC30-MPH6\n[Lu177]\ninput_mode=joint\n"
        "chunk_format=csrb\nsysmat_high=" << (root / "high").generic_string()
        << "\nsysmat_low=" << (root / "low").generic_string() << "\n"; }
    { std::ofstream f(config); f << "[Compatibility]\nrecon_sections_override_cli=true\n"
        "[Acquisition]\nisotope=Lu177\nprojection_model=joint\n"
        "[SystemMatrix]\nenergy_window_weights=0.1036,0.0617\n";
    }
    auto jointArgv = base;
    jointArgv.insert(jointArgv.end(), {"-df2", "Bed1_113.bin"});
    spect::cli::ReconArgs joint;
    ASSERT_TRUE(spect::io::test::parseReconArgsForTest(jointArgv, joint));
    EXPECT_EQ(joint.projectionModel, "joint");
    EXPECT_EQ(joint.projectionInputMode, "joint");
    EXPECT_EQ(std::filesystem::path(joint.chunkCSRDir2), root / "low");
    std::filesystem::remove_all(root);
}

TEST(IOTest, ConfiguredScannerOverridesCliAndFailsClosed) {
    const auto root = std::filesystem::temp_directory_path() / "spect_scanner_priority";
    std::filesystem::create_directories(root);
    const auto config = root / "global.ini";
    const auto scanner = root / "PSC30-MPH6.ini";
    {
        std::ofstream f(scanner);
        f << "[Scanner]\nname=PSC30-MPH6\n[Lu177]\ninput_mode=merged\n"
             "sysmat_high=" << (root / "high").generic_string() << "\nsysmat_low=\n";
    }
    const std::vector<std::string> argv = {"SPECT_recon", "-config", config.string(),
        "-scannerConf", "SC60-MPH3.ini", "-df", "Bed1.bin"};
    auto writeConfig = [&](const char* policy, const char* binding) {
        std::ofstream f(config);
        f << "[Compatibility]\nrecon_sections_override_cli=" << policy
          << "\n[Acquisition]\nisotope=Lu177\nprojection_model=merged\n"
             "[SystemMatrix]\nenergy_window_weights=0.1036,0.0617\nscanner_config=" << binding << '\n';
    };
    for (const char* policy : {"true", "false"}) {
        writeConfig(policy, "PSC30-MPH6.ini");
        spect::cli::ReconArgs args;
        ASSERT_TRUE(spect::io::test::parseReconArgsForTest(argv, args));
        EXPECT_EQ(args.scannerName, "PSC30-MPH6");
        EXPECT_EQ(args.projectionModel, "merged");
        EXPECT_EQ(std::filesystem::path(args.chunkCSRDir), root / "high");
        EXPECT_TRUE(args.chunkCSRDir2.empty());
        EXPECT_TRUE(args.energyWindowWeights.empty());
    }
    writeConfig("true", "missing.ini");
    auto validCli = argv;
    validCli[4] = scanner.string();
    spect::cli::ReconArgs invalid;
    EXPECT_FALSE(spect::io::test::parseReconArgsForTest(validCli, invalid));
    writeConfig("true", "");
    spect::cli::ReconArgs fallback;
    ASSERT_TRUE(spect::io::test::parseReconArgsForTest(validCli, fallback));
    EXPECT_EQ(fallback.scannerName, "PSC30-MPH6");
    std::filesystem::remove_all(root);
}
