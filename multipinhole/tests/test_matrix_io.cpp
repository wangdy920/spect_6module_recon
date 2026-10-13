/**
 * @file test_matrix_io.cpp
 * @brief Split from test_interfaces.cpp (matrix IO).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "io/ProjectionLoader.h"
#include "io/MatFileReader.h"
#include "io/MHDWriter.h"
#include "pipeline/ReconDataLoader.h"


namespace spect {
namespace io {
namespace test {
namespace {

void writeProjectionMat(const std::filesystem::path &path,
                        std::vector<size_t> dims, void *values,
                        matio_classes cls = MAT_C_DOUBLE,
                        matio_types dtype = MAT_T_DOUBLE) {
    mat_t *file = Mat_CreateVer(path.string().c_str(), nullptr, MAT_FT_MAT5);
    ASSERT_NE(file, nullptr);
    matvar_t *var = Mat_VarCreate("matrix", cls, dtype,
                                 static_cast<int>(dims.size()), dims.data(),
                                 values, 0);
    ASSERT_NE(var, nullptr);
    ASSERT_EQ(Mat_VarWrite(file, var, MAT_COMPRESSION_NONE), 0);
    Mat_VarFree(var);
    Mat_Close(file);
}

} // namespace

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
TEST(IOTest, LoadProjectionMatPreservesSourceAndOptionalStats) {
    const auto path = std::filesystem::temp_directory_path() /
        ("spect_projection_mat_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".mat");
    std::vector<float> floats = {0, 1, 10, 11, 100, 101, 110, 111};
    std::vector<double> values(floats.begin(), floats.end());
    for (bool single : {false, true}) {
        const auto cls = single ? MAT_C_SINGLE : MAT_C_DOUBLE;
        const auto dtype = single ? MAT_T_SINGLE : MAT_T_DOUBLE;
        ASSERT_NO_FATAL_FAILURE(writeProjectionMat(
            path, {2, 2, 2}, single ? static_cast<void *>(floats.data())
                                    : static_cast<void *>(values.data()), cls, dtype));
        for (bool withStats : {false, true}) {
            for (bool withSource : {false, true}) {
                SCOPED_TRACE(::testing::Message() << single << ',' << withStats
                                                 << ',' << withSource);
                ProjectionReorderStats stats{true, 9, 8, 7};
                MatArray source;
                const auto projection = loadProjection(
                    path.string(), "matrix", 1, withStats ? &stats : nullptr,
                    withSource ? &source : nullptr);
                EXPECT_EQ(projection.data, (std::vector<double>{0, 10, 1, 11}));
                EXPECT_EQ(projection.numAngles, 1);
                EXPECT_EQ(projection.detNz, 2);
                EXPECT_EQ(projection.detNx, 2);
                EXPECT_EQ(projection.raysPerAngle, 4u);
                EXPECT_TRUE(projection.detectorShapeKnown);
                EXPECT_FALSE(projection.binaryFormat);
                if (withStats) {
                    EXPECT_FALSE(stats.isEventList);
                    EXPECT_EQ(stats.selectedAngles, 1);
                    EXPECT_EQ(stats.detectorsPerAngle, 4u);
                    EXPECT_EQ(stats.recordCount, 4u);
                }
                if (withSource) {
                    EXPECT_EQ(source.dims, (std::vector<size_t>{2, 2, 2}));
                    EXPECT_EQ(source.cls, cls);
                    EXPECT_EQ(source.dtype, dtype);
                    EXPECT_EQ(source.d, single ? std::vector<double>{} : values);
                    EXPECT_EQ(source.f, single ? floats : std::vector<float>{});
                    EXPECT_TRUE(source.i32.empty());
                }
            }
        }
    }
    std::filesystem::remove(path);
}
TEST(IOTest, LoadProjectionMatFailuresPreserveStageOutputs) {
    const auto path = std::filesystem::temp_directory_path() /
        ("spect_projection_failure_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".mat");
    std::vector<int32_t> integers = {0, 0, 7};
    ASSERT_NO_FATAL_FAILURE(writeProjectionMat(
        path, {1, 3}, integers.data(), MAT_C_INT32, MAT_T_INT32));
    ProjectionReorderStats stats{true, 9, 8, 7};
    MatArray source;
    source.dims = {1, 1};
    source.d = {42};
    EXPECT_THROW(loadProjection(path.string(), "missing", 0, &stats, &source),
                 std::runtime_error);
    EXPECT_EQ(source.dims, (std::vector<size_t>{1, 1}));
    EXPECT_EQ(source.d, (std::vector<double>{42}));
    EXPECT_TRUE(stats.isEventList);
    EXPECT_EQ(stats.selectedAngles, 9);
    EXPECT_EQ(stats.detectorsPerAngle, 8u);
    EXPECT_EQ(stats.recordCount, 7u);

    // The source is available even when its class cannot be reordered.
    EXPECT_THROW(loadProjection(path.string(), "matrix", 0, &stats, &source),
                 std::runtime_error);
    EXPECT_EQ(source.dims, (std::vector<size_t>{1, 3}));
    EXPECT_EQ(source.cls, MAT_C_INT32);
    EXPECT_EQ(source.dtype, MAT_T_INT32);
    EXPECT_EQ(source.i32, integers);
    EXPECT_TRUE(source.d.empty());
    EXPECT_TRUE(source.f.empty());
    EXPECT_TRUE(stats.isEventList);
    EXPECT_EQ(stats.selectedAngles, 9);
    EXPECT_EQ(stats.detectorsPerAngle, 8u);
    EXPECT_EQ(stats.recordCount, 7u);

    std::vector<double> invalidShape = {1, 2, 3, 4};
    ASSERT_NO_FATAL_FAILURE(writeProjectionMat(path, {2, 2}, invalidShape.data()));
    EXPECT_THROW(loadProjection(path.string(), "matrix", 0, &stats, &source),
                 std::runtime_error);
    EXPECT_EQ(source.dims, (std::vector<size_t>{2, 2}));
    EXPECT_EQ(source.d, invalidShape);
    EXPECT_TRUE(source.i32.empty());
    EXPECT_FALSE(stats.isEventList);
    EXPECT_EQ(stats.selectedAngles, 0);
    EXPECT_EQ(stats.detectorsPerAngle, 0u);
    EXPECT_EQ(stats.recordCount, 0u);

    stats = {true, 9, 8, 7};
    EXPECT_THROW(loadProjection(path.string() + ".bin", "matrix", 0, &stats, &source),
                 std::runtime_error);
    EXPECT_TRUE(source.dims.empty());
    EXPECT_TRUE(source.d.empty());
    EXPECT_TRUE(source.f.empty());
    EXPECT_TRUE(source.i32.empty());
    EXPECT_FALSE(stats.isEventList);
    EXPECT_EQ(stats.selectedAngles, 0);
    EXPECT_EQ(stats.detectorsPerAngle, 0u);
    EXPECT_EQ(stats.recordCount, 0u);
    std::filesystem::remove(path);
}
TEST(IOTest, LoadProjectionEmptyEventsPreserveOptionalStatsMetadata) {
    const auto path = std::filesystem::temp_directory_path() /
        ("spect_empty_events_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".mat");
    for (bool rowTriples : {true, false}) {
        const std::vector<size_t> dims = rowTriples ? std::vector<size_t>{2, 3}
                                                   : std::vector<size_t>{3, 2};
        std::vector<double> values(6);
        for (size_t r = 0; r < 2; ++r) {
            const double fields[] = {std::numeric_limits<double>::quiet_NaN(),
                                     static_cast<double>(r), 1.0};
            for (size_t c = 0; c < 3; ++c)
                values[rowTriples ? r + 2 * c : c + 3 * r] = fields[c];
        }
        ASSERT_NO_FATAL_FAILURE(writeProjectionMat(path, dims, values.data()));
        for (bool withStats : {false, true}) {
            ProjectionReorderStats stats{true, 9, 8, 7};
            MatArray source;
            const auto projection = loadProjection(
                path.string(), "matrix", 0, withStats ? &stats : nullptr, &source);
            EXPECT_TRUE(projection.empty());
            EXPECT_EQ(projection.numAngles, 0);
            EXPECT_EQ(projection.detNz, withStats ? 1 : 0);
            EXPECT_EQ(projection.detNx, 0);
            EXPECT_EQ(projection.raysPerAngle, 0u);
            EXPECT_TRUE(projection.detectorShapeKnown);
            EXPECT_FALSE(projection.binaryFormat);
            EXPECT_EQ(source.dims, dims);
            ASSERT_EQ(source.d.size(), values.size());
            EXPECT_EQ(std::memcmp(source.d.data(), values.data(),
                                  values.size() * sizeof(double)), 0);
            if (withStats) {
                EXPECT_FALSE(stats.isEventList);
                EXPECT_EQ(stats.selectedAngles, 0);
                EXPECT_EQ(stats.detectorsPerAngle, 0u);
                EXPECT_EQ(stats.recordCount, 0u);
            }
        }
    }
    std::filesystem::remove(path);
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
    spect::main::prepareProjectionInputs(args);
    ASSERT_EQ(args.projPaths.size(), 2u);
    CompactSymmetryMetadata meta;
    meta.fullNumAngles = 2;
    meta.detNz = 1;
    meta.detNx = 3;
    const auto loaded = spect::main::loadProjectionSet(
        args.projPaths, meta, "Projection Data", "matrix", true);
    ASSERT_EQ(loaded.sourceGroups.size(), 2u);
    EXPECT_EQ(loaded.sourceGroups[0].size(), 2u);
    ASSERT_EQ(loaded.data.size(), 12u);
    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(loaded.data[static_cast<size_t>(i)], 11.0 + 2.0 * i);
        EXPECT_DOUBLE_EQ(loaded.data[static_cast<size_t>(i + 6)],
                         211.0 + 2.0 * i);
    }

    // A negative segment must not be hidden by a positive accumulated count.
    for (const auto &segment : {first, second}) {
        writeProjection(segment / "Bed1.bin", -1.0);
        EXPECT_THROW(spect::main::loadProjectionSet(
                         args.projPaths, meta, "Projection Data", "matrix", true),
                     std::runtime_error);
        writeProjection(segment / "Bed1.bin", segment == first ? 1.0 : 10.0);
    }
    writeProjection(first / "Bed1.bin", std::numeric_limits<double>::max());
    writeProjection(second / "Bed1.bin", std::numeric_limits<double>::max());
    EXPECT_THROW(spect::main::loadProjectionSet(
                     args.projPaths, meta, "Projection Data", "matrix", true),
                 std::runtime_error);
    writeProjection(first / "Bed1.bin", 1.0);
    writeProjection(second / "Bed1.bin", 10.0);

    // Same payload length, different angle/ray layout in a later segment.
    {
        std::fstream out(second / "Bed1.bin", std::ios::binary | std::ios::in | std::ios::out);
        const int32_t header[] = {3, 2};
        out.write(reinterpret_cast<const char*>(header), sizeof(header));
    }
    EXPECT_THROW(spect::main::loadProjectionSet(
                     args.projPaths, meta, "Projection Data", "matrix", true),
                 std::runtime_error);
    writeProjection(second / "Bed1.bin", 10.0);

    std::filesystem::remove(second / "Bed2.bin");
    EXPECT_THROW(spect::main::loadProjectionSet(
                     args.projPaths, meta, "Projection Data", "matrix", true),
                 std::runtime_error);

    std::filesystem::remove(first / "Bed2.bin");
    args.projPaths.clear();
    args.projPath = first.string();
    spect::main::prepareProjectionInputs(args);
    EXPECT_TRUE(args.projPaths.empty());
    EXPECT_EQ(std::filesystem::path(args.projPath).filename(), "Bed1.bin");
    std::filesystem::remove_all(root);
}
TEST(IOTest, BedsSelectionSearchesDirectoryForBedRange) {
    const auto dir = std::filesystem::temp_directory_path() /
        ("spect_beds_select_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    for (int bed = 1; bed <= 12; ++bed)
        std::ofstream(dir / ("Bed" + std::to_string(bed) + ".bin")) << 'x';
    std::ofstream(dir / "Bed3_high.bin") << 'x';

    spect::cli::ReconArgs args;
    args.projPath = dir.string();
    args.bedFirst = 3;
    args.bedLast = 7;
    spect::main::prepareProjectionInputs(args);
    EXPECT_TRUE(args.projPath.empty());
    ASSERT_EQ(args.projPaths.size(), 5u);
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(std::filesystem::path(args.projPaths[i]).filename(),
                  "Bed" + std::to_string(3 + i) + ".bin");

    // A file inside the directory (here via a one-entry -dfList) also works;
    // Bed10 must not sort before Bed2.
    args = {};
    args.projPaths = {(dir / "Bed1.bin").string()};
    args.bedFirst = 1;
    args.bedLast = 12;
    spect::main::prepareProjectionInputs(args);
    ASSERT_EQ(args.projPaths.size(), 12u);
    EXPECT_EQ(std::filesystem::path(args.projPaths[9]).filename(), "Bed10.bin");

    args = {};
    args.projPath = dir.string();
    args.bedFirst = args.bedLast = 4;
    spect::main::prepareProjectionInputs(args);
    EXPECT_TRUE(args.projPaths.empty());
    EXPECT_EQ(std::filesystem::path(args.projPath).filename(), "Bed4.bin");

    args = {};
    args.projPath = dir.string();
    args.bedFirst = 10;
    args.bedLast = 13;
    EXPECT_THROW(spect::main::prepareProjectionInputs(args), std::runtime_error);

    args = {};
    args.projPaths = {(dir / "Bed1.bin").string(), (dir / "Bed2.bin").string()};
    args.bedFirst = 1;
    args.bedLast = 2;
    EXPECT_THROW(spect::main::prepareProjectionInputs(args), std::runtime_error);
    std::filesystem::remove_all(dir);
}
TEST(IOTest, ReconstructionProjectionInputsRequireExactLayoutAndValidCounts) {
    const auto root = std::filesystem::temp_directory_path() /
        ("spect_projection_contract_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    CompactSymmetryMetadata meta;
    meta.fullNumAngles = 8;
    meta.detNz = meta.detNx = 2;
    const auto valid = root / "valid.bin";
    const auto candidate = root / "candidate.bin";
    auto writeBin = [](const std::filesystem::path &path, int32_t angles,
                       int32_t rays, double last) {
        std::vector<double> values(static_cast<size_t>(angles) * rays);
        for (size_t i = 0; i < values.size(); ++i) values[i] = i * 0.125;
        values.front() = -0.0;
        values.back() = last;
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&angles), sizeof(angles));
        out.write(reinterpret_cast<const char*>(&rays), sizeof(rays));
        out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
        return values;
    };
    const auto first = writeBin(valid, 8, 4, 3.875);
    for (const char *label : {"Projection Data", "Projection Data (window 2)",
                              "Background Projection", "Background Projection (window 2)"}) {
        SCOPED_TRACE(label);
        const auto second = writeBin(candidate, 8, 4, 7.5);
        const auto loaded = spect::main::loadProjectionSet(
            {valid.string(), candidate.string()}, meta, label, "matrix");
        ASSERT_EQ(loaded.data.size(), 64u);
        EXPECT_EQ(std::memcmp(loaded.data.data(), first.data(), 32 * sizeof(double)), 0);
        EXPECT_EQ(std::memcmp(loaded.data.data() + 32, second.data(), 32 * sizeof(double)), 0);
        for (const auto dims : {std::pair<int, int>{12, 4}, {4, 4}, {4, 8}, {8, 3}}) {
            SCOPED_TRACE(::testing::Message() << dims.first << 'x' << dims.second);
            writeBin(candidate, dims.first, dims.second, 1.0);
            for (const auto &paths : {std::vector<std::string>{candidate.string()},
                                      std::vector<std::string>{valid.string(), candidate.string()}})
                EXPECT_THROW(spect::main::loadProjectionSet(paths, meta, label, "matrix"),
                             std::runtime_error);
        }
        for (double value : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity()}) {
            writeBin(candidate, 8, 4, value);
            try {
                (void)spect::main::loadProjectionSet(
                    {valid.string(), candidate.string()}, meta, label, "matrix");
                FAIL() << "Invalid count accepted";
            } catch (const std::runtime_error &error) {
                EXPECT_NE(std::string(error.what()).find(candidate.string()), std::string::npos);
                EXPECT_NE(std::string(error.what()).find("index 31"), std::string::npos);
            }
        }
    }

    // MAT supplies detector dimensions; equal-length but transposed shapes fail.
    const auto matPath = root / "projection.mat";
    for (const auto shape : {std::pair<size_t, size_t>{2, 2}, {1, 4}}) {
        std::vector<double> values(32);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<double>(i);
        ASSERT_NO_FATAL_FAILURE(writeProjectionMat(
            matPath, {shape.first, shape.second, 8}, values.data()));
        if (shape.first == 2) {
            const auto loaded = spect::main::loadProjectionSet(
                {matPath.string()}, meta, "Projection Data", "matrix");
            ASSERT_EQ(loaded.data.size(), 32u);
            for (size_t a = 0; a < 8; ++a)
                for (size_t z = 0; z < 2; ++z)
                    for (size_t x = 0; x < 2; ++x)
                        EXPECT_DOUBLE_EQ(loaded.data[a * 4 + z * 2 + x],
                                         values[z + 2 * (x + 2 * a)]);
        } else {
            EXPECT_THROW(spect::main::loadProjectionSet(
                             {matPath.string()}, meta, "Projection Data", "matrix"),
                         std::runtime_error);
        }
    }
    std::filesystem::remove_all(root);
}
TEST(IOTest, ReconstructionEventListRejectsInvalidSourceCounts) {
    const auto path = std::filesystem::temp_directory_path() /
        ("spect_event_contract_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".mat");
    CompactSymmetryMetadata meta;
    meta.fullNumAngles = 8;
    meta.detNz = 1;
    meta.detNx = 2;
    for (bool rowTriples : {true, false}) {
        for (double last : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
            const std::vector<size_t> dims = {rowTriples ? 17u : 3u,
                                             rowTriples ? 3u : 17u};
            std::vector<double> values(51);
            for (size_t r = 0; r < 17; ++r) {
                const double fields[] = {r < 16 ? double(r / 2) : 0.0,
                                          r < 16 ? double(r % 2) : 0.0,
                                          r < 16 ? 1.0 : last};
                for (size_t c = 0; c < 3; ++c)
                    values[rowTriples ? r + 17 * c : c + 3 * r] = fields[c];
            }
            ASSERT_NO_FATAL_FAILURE(writeProjectionMat(path, dims, values.data()));
            if (last == 0.0) {
                for (bool withStats : {false, true}) {
                    ProjectionReorderStats stats;
                    MatArray source;
                    const auto projection = loadProjection(
                        path.string(), "matrix", 2, withStats ? &stats : nullptr, &source);
                    EXPECT_EQ(projection.data, std::vector<double>(4, 1.0));
                    EXPECT_EQ(projection.numAngles, 2);
                    EXPECT_EQ(projection.detNz, 1);
                    EXPECT_EQ(projection.detNx, 2);
                    EXPECT_EQ(projection.raysPerAngle, 2u);
                    EXPECT_EQ(source.dims, dims);
                    EXPECT_EQ(source.d, values);
                    if (withStats) {
                        EXPECT_TRUE(stats.isEventList);
                        EXPECT_EQ(stats.selectedAngles, 2);
                        EXPECT_EQ(stats.detectorsPerAngle, 2u);
                        EXPECT_EQ(stats.recordCount, 17u);
                    }
                }
                const auto loaded = spect::main::loadProjectionSet(
                    {path.string()}, meta, "Projection Data", "matrix");
                EXPECT_EQ(loaded.data, std::vector<double>(16, 1.0));
            } else {
                EXPECT_THROW(spect::main::loadProjectionSet(
                                 {path.string()}, meta, "Projection Data", "matrix"),
                             std::runtime_error);
            }
        }
    }
    std::filesystem::remove(path);
}
TEST(IOTest, ReorderProjectionEmptyThrows) {
    MatArray mat;
    mat.dims = {2, 3, 4};
    // No data
    EXPECT_THROW(reorderProjection(mat, 0), std::runtime_error);
}

TEST(IOTest, CheckedVolumeRejectsWrongGridAndInvalidMuButAllowsNegativeHU) {
    const auto stem = std::filesystem::temp_directory_path() / "spect_checked_mu";
    const std::string path = stem.string() + ".mhd";
    std::vector<double> data(8, 0.0152);
    const auto save = [&](float spacing) {
        writeMHD(stem.string(), data, 2, 2, 2, spacing, spacing, spacing);
    };
    const auto load = [&](bool nonnegative) {
        return spect::main::loadVolumeChecked(path, "test map", 2, 2, 2,
                                               {0.4, 0.4, 0.4}, nonnegative);
    };
    save(0.4f);
    EXPECT_EQ(load(true), data);
    EXPECT_THROW(spect::main::loadVolumeChecked(path, "test map", 2, 2, 3,
                                               {0.4, 0.4, 0.4}, true), std::runtime_error);
    save(0.8f);
    EXPECT_THROW(load(true), std::runtime_error);
    save(0.0f);
    EXPECT_THROW(load(true), std::runtime_error);
    {
        std::ofstream header(path);
        header << "NDims = 3\nDimSize = 2 2 2\nElementType = MET_DOUBLE\n"
               << "ElementDataFile = " << stem.filename().string() << ".raw\n";
    }
    EXPECT_THROW(spect::main::loadVolumeChecked(path, "missing spacing", 2, 2, 2,
                                               {1, 1, 1}, true), std::runtime_error);
    for (double bad : {-0.1, std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity()}) {
        data[3] = bad;
        save(0.4f);
        EXPECT_THROW(load(true), std::runtime_error);
        if (std::isfinite(bad))
            EXPECT_EQ(load(false), data);
        else
            EXPECT_THROW(load(false), std::runtime_error);
    }
    std::filesystem::remove(path);
    std::filesystem::remove(stem.string() + ".raw");
}

} // namespace test
} // namespace io
} // namespace spect

