/**
 * @file test_interfaces.cpp
 * @brief 接口头文件编译验证测试
 * 
 * 验证所有接口头文件可以正确编译，不依赖第三方库。
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

// 包含所有接口头文件，验证编译
#include "core/Types.h"
#include "cli/ArgumentParser.h"
#include "cli/ConfigParser.h"
#include "filters/Convolution.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "io/MHDWriter.h"
#include "io/VolumeLoader.h"

namespace spect {
namespace test {

// ============ Types.h 测试 ============

TEST(InterfaceTest, StatusEnum) {
    EXPECT_EQ(statusToString(Status::Ok), "Ok");
    EXPECT_EQ(statusToString(Status::InvalidArgument), "InvalidArgument");
    EXPECT_EQ(statusToString(Status::FileNotFound), "FileNotFound");
}

TEST(InterfaceTest, Dims3D) {
    Dims3D dims{64, 64, 32};
    EXPECT_EQ(dims.nx, 64);
    EXPECT_EQ(dims.ny, 64);
    EXPECT_EQ(dims.nz, 32);
    EXPECT_EQ(dims.total(), 64 * 64 * 32);
}

TEST(InterfaceTest, VoxelSize) {
    VoxelSize vs{0.5, 0.5, 1.0};
    EXPECT_DOUBLE_EQ(vs.dx, 0.5);
    EXPECT_DOUBLE_EQ(vs.dy, 0.5);
    EXPECT_DOUBLE_EQ(vs.dz, 1.0);
}

TEST(InterfaceTest, MHDWritersUseZeroRawHeaderSize) {
    const auto tmpDir = std::filesystem::temp_directory_path() / "spect_mhd_header_test";
    std::filesystem::create_directories(tmpDir);
    const auto path = (tmpDir / "writer_header").string();

    spect::io::writeMHD(path, std::vector<double>{1.0, 2.0}, 1, 1, 2, 0.3f, 0.3f, 0.3f);

    std::ifstream mhd(tmpDir / "writer_header.mhd");
    ASSERT_TRUE(mhd.good());
    const std::string text((std::istreambuf_iterator<char>(mhd)),
                           std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("HeaderSize = 0"), std::string::npos);
    EXPECT_EQ(text.find("HeaderSize = -1"), std::string::npos);

    const auto outDir = tmpDir / "inline";
    spect::io::saveMHDRAW(std::vector<double>{3.0}, 1, 1, 1, 0.3, 0.3, 0.3,
                          outDir.string(), "inline_header", -1);
    std::ifstream inlineMhd(outDir / "inline_header.mhd");
    ASSERT_TRUE(inlineMhd.good());
    const std::string inlineText((std::istreambuf_iterator<char>(inlineMhd)),
                                 std::istreambuf_iterator<char>());
    EXPECT_NE(inlineText.find("HeaderSize = 0"), std::string::npos);
    EXPECT_EQ(inlineText.find("HeaderSize = -1"), std::string::npos);
}




TEST(InterfaceTest, TruncatedMhdRawThrows) {
    const auto tmpDir = std::filesystem::temp_directory_path() /
                        "spect_mhd_truncated_test";
    std::filesystem::create_directories(tmpDir);
    const auto mhdPath = tmpDir / "truncated.mhd";
    const auto rawPath = tmpDir / "truncated.raw";
    {
        std::ofstream mhd(mhdPath);
        ASSERT_TRUE(mhd.good());
        mhd << "ObjectType = Image\nNDims = 3\nDimSize = 2 1 1\n"
               "ElementSpacing = 1 1 1\nElementType = MET_DOUBLE\n"
               "ElementDataFile = truncated.raw\n";
    }
    {
        std::ofstream raw(rawPath, std::ios::binary);
        const double oneValue = 1.0;
        raw.write(reinterpret_cast<const char*>(&oneValue), sizeof(oneValue));
    }

    int nx = 0, ny = 0, nz = 0;
    float dx = 0, dy = 0, dz = 0;
    EXPECT_THROW((void)spect::io::loadVolumeWithSpacing(
                     mhdPath.string(), nx, ny, nz, dx, dy, dz),
                 std::runtime_error);
    std::filesystem::remove_all(tmpDir);
}
} // namespace test
} // namespace spect


// ============ App模块测试 【T2.3】 ============

namespace spect {
namespace app {
namespace test {

// Integration test: ConfigParser with real config file
TEST(AppTest, ConfigParserIntegration) {
    // Test ConfigParser can parse the default config file
    spect::cli::ConfigParser parser;
    std::string configPath = "config/spect_config.ini";
    
    // Try to load config file (may not exist in test environment)
    if (std::filesystem::exists(configPath)) {
        ASSERT_TRUE(parser.load(configPath));
        
        // Verify some expected values from the config
        EXPECT_EQ(parser.getInt("Basic", "threads", 0), 16);
        EXPECT_EQ(parser.getInt("Basic", "verbosity", 0), 1);
        EXPECT_EQ(parser.getString("Detector", "channels", ""), "48,48");
        EXPECT_EQ(parser.getInt("Detector", "angles", 0), 360);
        EXPECT_EQ(parser.getString("PSF", "mode", ""), "sieve");
        EXPECT_DOUBLE_EQ(parser.getDouble("PSF", "fwhm", 0.0), 1.3);
        EXPECT_DOUBLE_EQ(parser.getDouble("PSF", "min_sigmas", 0.0), 3.0);
        EXPECT_DOUBLE_EQ(
            parser.getDouble("Filtering", "butterworth_cutoff", 0.0), 0.0);
        EXPECT_EQ(parser.getInt("Filtering", "butterworth_order", 0), 2);
        EXPECT_EQ(parser.getString("Filtering", "smooth_apply_stage", ""), "none");

        EXPECT_EQ(parser.getString("Regularization", "type", ""), "None");
    } else {
        GTEST_SKIP() << "Config file not found: " << configPath;
    }
}

} // namespace test
} // namespace app
} // namespace spect
