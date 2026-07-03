/**
 * @file test_matrix_io.cpp
 * @brief Split from test_interfaces.cpp (matrix IO).
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

#include "core/CSRHeader.h"
#include "io/CSRTypes.h"
#include "io/ProjectionLoader.h"
#include "io/MatFileReader.h"


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

