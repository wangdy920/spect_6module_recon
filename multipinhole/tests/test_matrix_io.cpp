/**
 * @file test_matrix_io.cpp
 * @brief Split from test_interfaces.cpp (matrix IO).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
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

#include "io/CSRTypes.h"
#include "io/ProjectionLoader.h"
#include "io/MatFileReader.h"
#include "io/MHDWriter.h"
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
        size_t dims[] = {shape.first, shape.second, 8};
        mat_t *file = Mat_CreateVer(matPath.string().c_str(), nullptr, MAT_FT_MAT5);
        ASSERT_NE(file, nullptr);
        matvar_t *var = Mat_VarCreate("matrix", MAT_C_DOUBLE, MAT_T_DOUBLE, 3,
                                     dims, values.data(), 0);
        ASSERT_NE(var, nullptr);
        ASSERT_EQ(Mat_VarWrite(file, var, MAT_COMPRESSION_NONE), 0);
        Mat_VarFree(var);
        Mat_Close(file);
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
            size_t dims[] = {rowTriples ? 17u : 3u, rowTriples ? 3u : 17u};
            std::vector<double> values(51);
            for (size_t r = 0; r < 17; ++r) {
                const double fields[] = {r < 16 ? double(r / 2) : 0.0,
                                          r < 16 ? double(r % 2) : 0.0,
                                          r < 16 ? 1.0 : last};
                for (size_t c = 0; c < 3; ++c)
                    values[rowTriples ? r + 17 * c : c + 3 * r] = fields[c];
            }
            mat_t *file = Mat_CreateVer(path.string().c_str(), nullptr, MAT_FT_MAT5);
            ASSERT_NE(file, nullptr);
            matvar_t *var = Mat_VarCreate("matrix", MAT_C_DOUBLE, MAT_T_DOUBLE, 2,
                                         dims, values.data(), 0);
            ASSERT_NE(var, nullptr);
            ASSERT_EQ(Mat_VarWrite(file, var, MAT_COMPRESSION_NONE), 0);
            Mat_VarFree(var);
            Mat_Close(file);
            if (last == 0.0) {
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

