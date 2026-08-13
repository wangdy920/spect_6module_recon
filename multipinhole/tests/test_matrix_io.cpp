/**
 * @file test_matrix_io.cpp
 * @brief Split from test_interfaces.cpp (matrix IO).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/CSRTypes.h"
#include "io/ProjectionLoader.h"
#include "io/MatFileReader.h"
#include "pipeline/ReconDataLoader.h"


namespace spect {
namespace io {
namespace test {

TEST(IOTest, ProjectionDataEmpty) {
    ProjectionData pd;
    EXPECT_TRUE(pd.empty());
    EXPECT_EQ(pd.size(), 0);
    EXPECT_EQ(pd.numAngles, 0);
}
TEST(IOTest, ReorderProjection3D) {
    // 创建模拟的3D MAT数据 [2 x 3 x 4] = detZ x detX x angles
    MatArray mat;
    mat.dims = {2, 3, 4};  // 2 detZ, 3 detX, 4 angles
    mat.cls = MAT_C_DOUBLE;
    mat.d.resize(2 * 3 * 4);
    
    // 填充测试数据: value = z + 10*x + 100*a
    for (int a = 0; a < 4; ++a) {
        for (int x = 0; x < 3; ++x) {
            for (int z = 0; z < 2; ++z) {
                size_t idx = z + 2 * (x + 3 * a);
                mat.d[idx] = z + 10.0 * x + 100.0 * a;
            }
        }
    }
    
    auto result = reorderProjection(mat, 0);  // 0 = all angles
    EXPECT_EQ(result.size(), 2 * 3 * 4);
    
    // 验证重排序后的数据
    // 输出应为角度优先: angle0所有det, angle1所有det, ...
    size_t idx = 0;
    for (int a = 0; a < 4; ++a) {
        for (int z = 0; z < 2; ++z) {
            for (int x = 0; x < 3; ++x) {
                double expected = z + 10.0 * x + 100.0 * a;
                EXPECT_DOUBLE_EQ(result[idx], expected) << "a=" << a << " z=" << z << " x=" << x;
                ++idx;
            }
        }
    }
}
TEST(IOTest, ReorderProjectionLimitAngles) {
    MatArray mat;
    mat.dims = {2, 2, 10};  // 10 angles available
    mat.cls = MAT_C_DOUBLE;
    mat.d.resize(2 * 2 * 10, 1.0);
    
    auto result = reorderProjection(mat, 3);  // Only 3 angles
    EXPECT_EQ(result.size(), 2 * 2 * 3);
}
TEST(IOTest, LoadProjectionBinAutoDetects) {
    const auto dir = std::filesystem::temp_directory_path() / "spect_projection_loader_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "projection.bin";

    const int32_t numAngles = 3;
    const int32_t raysPerAngle = 4;
    double values[12];
    for (int i = 0; i < 12; ++i)
        values[i] = static_cast<double>(i + 1);

    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        out.write(reinterpret_cast<const char*>(&numAngles), sizeof(numAngles));
        out.write(reinterpret_cast<const char*>(&raysPerAngle), sizeof(raysPerAngle));
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    ProjectionReorderStats stats;
    MatArray mat;
    const ProjectionData projection =
        loadProjection(path.string(), "matrix", 2, &stats, &mat);

    EXPECT_TRUE(projection.binaryFormat);
    EXPECT_FALSE(projection.detectorShapeKnown);
    EXPECT_EQ(projection.numAngles, 2);
    EXPECT_EQ(projection.raysPerAngle, 4u);
    EXPECT_EQ(projection.detNz, 1);
    EXPECT_EQ(projection.detNx, 4);
    ASSERT_EQ(projection.data.size(), 8u);
    for (size_t i = 0; i < projection.data.size(); ++i)
        EXPECT_DOUBLE_EQ(projection.data[i], values[i]);
    EXPECT_TRUE(mat.dims.empty());
    EXPECT_EQ(stats.selectedAngles, 2);
    EXPECT_EQ(stats.detectorsPerAngle, 4u);
    EXPECT_EQ(stats.recordCount, 8u);

    std::filesystem::remove(path);
}
TEST(IOTest, TimeSegmentMergeSumsMatchingBedsAndRejectsMissingBed) {
    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto root = std::filesystem::temp_directory_path() /
                      ("spect_time_segment_merge_" + unique);
    const auto first = root / "case" / "20260810_210556" / "SPECT";
    const auto second = root / "case" / "20260810_211500" / "SPECT";
    std::filesystem::create_directories(first);
    std::filesystem::create_directories(second);

    auto writeProjection = [](const std::filesystem::path &path,
                              double base) {
        const int32_t angles = 2;
        const int32_t rays = 3;
        double values[6];
        for (int i = 0; i < 6; ++i)
            values[i] = base + static_cast<double>(i);
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&angles), sizeof(angles));
        out.write(reinterpret_cast<const char*>(&rays), sizeof(rays));
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    };
    writeProjection(first / "Bed1.bin", 1.0);
    writeProjection(first / "Bed2.bin", 101.0);
    writeProjection(second / "Bed1.bin", 10.0);
    writeProjection(second / "Bed2.bin", 110.0);

    spect::cli::ReconArgs args;
    args.mergeTimeSegments = true;
    args.projPath = first.string();
    spect::main::prepareTimeSegmentProjectionInputs(args);
    ASSERT_EQ(args.projPaths.size(), 2u);
    const auto loaded = spect::main::loadProjectionSet(
        args.projPaths, 2, "Projection Data", "matrix", true);
    ASSERT_EQ(loaded.sourceGroups.size(), 2u);
    EXPECT_EQ(loaded.sourceGroups[0].size(), 2u);
    ASSERT_EQ(loaded.data.size(), 12u);
    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(loaded.data[static_cast<size_t>(i)], 11.0 + 2.0 * i);
        EXPECT_DOUBLE_EQ(loaded.data[static_cast<size_t>(i + 6)],
                         211.0 + 2.0 * i);
    }

    std::filesystem::remove(second / "Bed2.bin");
    EXPECT_THROW(spect::main::loadProjectionSet(
                     args.projPaths, 2, "Projection Data", "matrix", true),
                 std::runtime_error);

    std::filesystem::remove(first / "Bed2.bin");
    args.projPaths.clear();
    args.projPath = first.string();
    spect::main::prepareTimeSegmentProjectionInputs(args);
    EXPECT_TRUE(args.projPaths.empty());
    EXPECT_EQ(std::filesystem::path(args.projPath).filename(), "Bed1.bin");
    std::filesystem::remove_all(root);
}
TEST(IOTest, IoCSRBv1HeaderIs24Bytes) {
    EXPECT_EQ(sizeof(spect::io::CSRHeader), 24u);
    spect::io::CSRHeader header;
    header.magic = 0x43535242u;
    header.version = 1;
    header.rows = 10;
    header.cols = 20;
    header.nnz = 100;
    EXPECT_EQ(header.magic, 0x43535242u);
    EXPECT_EQ(header.version, 1u);
}
TEST(IOTest, ReorderProjectionEmptyThrows) {
    MatArray mat;
    mat.dims = {2, 3, 4};
    // No data
    EXPECT_THROW(reorderProjection(mat, 0), std::runtime_error);
}

} // namespace test
} // namespace io
} // namespace spect

