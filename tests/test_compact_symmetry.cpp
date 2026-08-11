/**
 * @file test_compact_symmetry.cpp
 * @brief Split from test_interfaces.cpp (compact symmetry).
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

#include "filters/Convolution.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "tests/common/compact_symmetry_test_helpers.h"


namespace spect {
namespace io {
namespace test {

void writeCompactMetadata(const std::filesystem::path &path, int use90,
                          int useZ, int use45, int symCode,
                          int symmetryOrder) {
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out);
    auto scalar = [&](const auto &value) {
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    };
    const int totalRows = 90 * 48 * 24;
    const int totalVoxels = 108 * 108 * 80;
    const int maxNnz = 1;
    const int compactNumAngles = 90;
    const int detNx = 48;
    const int compactDetZ = 24;
    const int compactRaysPerAngle = 48 * 24;
    const int fullNumAngles = 360;
    const int nx = 108;
    const int ny = 108;
    const int nz = 80;
    const float voxelSize = 0.3f;
    const int detNxDuplicate = 48;
    const int detNz = 48;
    const float pixelSize = 1.0f;
    const float detectorY = 1.0f;
    const float detectorThickness = 1.0f;
    scalar(totalRows);
    scalar(totalVoxels);
    scalar(maxNnz);
    scalar(compactNumAngles);
    scalar(detNx);
    scalar(compactDetZ);
    scalar(compactRaysPerAngle);
    scalar(fullNumAngles);
    scalar(nx);
    scalar(ny);
    scalar(nz);
    scalar(voxelSize);
    scalar(detNxDuplicate);
    scalar(detNz);
    scalar(pixelSize);
    scalar(detectorY);
    scalar(detectorThickness);
    scalar(use90);
    scalar(useZ);
    scalar(use45);
    scalar(symCode);
    scalar(symmetryOrder);
}

TEST(IOTest, CompactSymmetryMetadataAcceptsCanonical90ZContract) {
    const auto path = std::filesystem::temp_directory_path() /
                      "spect_compact_meta_90z.bin";
    writeCompactMetadata(path, 1, 1, 0, 90, 4);
    ASSERT_EQ(std::filesystem::file_size(path),
              spect::io::kCompactMetadataBytes);

    const auto meta = spect::io::loadCompactSymmetryMetadata(path.string());
    EXPECT_EQ(meta.compactRows(), 90 * 48 * 24);
    EXPECT_EQ(meta.fullRows(), 360 * 48 * 48);
    EXPECT_EQ(meta.xySymmetryCount(), 4);
    EXPECT_EQ(meta.zBranchCount(), 2);
    EXPECT_STREQ(spect::io::compactSymmetryModeName(), "90z");
    std::filesystem::remove(path);
}

TEST(IOTest, CompactSymmetryMetadataRejectsLegacy45ZContract) {
    const auto path = std::filesystem::temp_directory_path() /
                      "spect_compact_meta_45z.bin";
    writeCompactMetadata(path, 0, 1, 1, 45, 8);
    EXPECT_THROW(spect::io::loadCompactSymmetryMetadata(path.string()),
                 std::runtime_error);
    std::filesystem::remove(path);
}
TEST(IOTest, CompactSymmetryMetadataMissingFileThrows) {
    const auto path = std::filesystem::temp_directory_path() /
                      "spect_missing_compact_metadata.bin";
    std::filesystem::remove(path);

    EXPECT_THROW(spect::io::loadCompactSymmetryMetadata(path.string()),
                 std::runtime_error);
}
TEST(IOTest, CompactSymmetryMetadataRejectsTruncatedExtendedHeader) {
    const auto dir = std::filesystem::temp_directory_path() /
                     "spect_compact_meta_size_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "metadata_truncated.bin";

    {
        std::vector<char> bytes(spect::io::kCompactMetadataBytes - 8, 0);
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    EXPECT_THROW(spect::io::loadCompactSymmetryMetadata(path.string()),
                 std::runtime_error);
    std::filesystem::remove(path);
}
TEST(IOTest, CompactSymmetryMetadataRejectsUnexpectedTrailingBytes) {
    const auto dir = std::filesystem::temp_directory_path() /
                     "spect_compact_meta_size_tests";
    std::filesystem::create_directories(dir);
    const auto path = dir / "metadata_trailing.bin";

    {
        std::vector<char> bytes(spect::io::kCompactMetadataBytes + 4, 0);
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    EXPECT_THROW(spect::io::loadCompactSymmetryMetadata(path.string()),
                 std::runtime_error);
    std::filesystem::remove(path);
}
TEST(IOTest, Real260601MetadataParsesWhenPresent) {
    const std::filesystem::path metaPath =
        std::filesystem::path("test_data") / "6module" / "260601" / "metadata.bin";
    if (!std::filesystem::exists(metaPath)) {
        GTEST_SKIP() << "test_data/6module/260601 not available";
    }

    const auto meta =
        spect::io::loadCompactSymmetryMetadata(metaPath.string());
    EXPECT_EQ(meta.compactRows(), 90 * 48 * 24);
    EXPECT_EQ(meta.fullRows(), 360 * 48 * 48);
    EXPECT_EQ(meta.symCode, 90);
    EXPECT_EQ(meta.symmetryOrder, 4);
}
TEST(IOTest, Real260601Sym90ZCoversAllFullProjectionRowsWhenPresent) {
    const std::filesystem::path metaPath =
        std::filesystem::path("test_data") / "6module" / "260601" / "metadata.bin";
    if (!std::filesystem::exists(metaPath)) {
        GTEST_SKIP() << "test_data/6module/260601 not available";
    }

    const auto meta =
        spect::io::loadCompactSymmetryMetadata(metaPath.string());

    std::vector<unsigned char> seen(
        static_cast<size_t>(meta.fullRows()), 0);
    size_t duplicateMappings = 0;
    for (int symId = 0; symId < meta.xySymmetryCount(); ++symId) {
        for (int zMirror = 0; zMirror < meta.zBranchCount(); ++zMirror) {
            for (int compactRow = 0; compactRow < meta.totalRows; ++compactRow) {
                uint64_t fullIndex = 0;
                ASSERT_TRUE(spect::io::fullProjectionIndexFromCompactRow(
                    meta, compactRow, symId, zMirror != 0, fullIndex));
                if (seen[fullIndex] != 0) {
                    ++duplicateMappings;
                }
                seen[fullIndex] = 1;
            }
        }
    }
    EXPECT_EQ(duplicateMappings, 0u);
    for (unsigned char value : seen) {
        EXPECT_EQ(value, 1);
    }
}
TEST(IOTest, Real260601ChunkDirectoryReadsWhenPresent) {
    const std::filesystem::path dir =
        std::filesystem::path("test_data") / "6module" / "260601";
    const auto metaPath = dir / "metadata.bin";
    if (!std::filesystem::exists(metaPath)) {
        GTEST_SKIP() << "test_data/6module/260601 not available";
    }

    const auto meta =
        spect::io::loadCompactSymmetryMetadata(metaPath.string());
    const auto chunks =
        spect::io::openCompactChunkCSRDirectory(dir.string(), meta, 1024 * 1024);
    if (chunks.chunks().empty()) {
        GTEST_SKIP() << "no chunk_*.csrb files under test_data/6module/260601";
    }

    EXPECT_EQ(chunks.totalRows(), static_cast<uint64_t>(meta.compactRows()));
    EXPECT_EQ(chunks.cols(), meta.totalVoxels);

    const auto first = chunks.loadChunk(0);
    ASSERT_TRUE(first);
    const auto view = first->view();
    EXPECT_EQ(view.rows, static_cast<int>(chunks.chunks()[0].rows));
    EXPECT_EQ(view.cols, meta.totalVoxels);
    EXPECT_GT(view.nnz, 0u);
}
TEST(IOTest, CompactSymmetryMapsSixModule90ZFullProjectionRowsOnce) {
    spect::io::CompactSymmetryMetadata meta;
    meta.totalRows = 90 * 48 * 24;
    meta.totalVoxels = 108 * 108 * 80;
    meta.compactNumAngles = 90;
    meta.detNx = 48;
    meta.compactDetZ = 24;
    meta.compactRaysPerAngle = 48 * 24;
    meta.fullNumAngles = 360;
    meta.Nx = 108;
    meta.Ny = 108;
    meta.Nz = 80;
    meta.detNz = 48;
    meta.symCode = 90;
    meta.symmetryOrder = 4;

    ASSERT_EQ(meta.compactRows(), 103680);
    ASSERT_EQ(meta.fullRows(), 829440);
    ASSERT_EQ(meta.xySymmetryCount(), 4);
    ASSERT_EQ(meta.zBranchCount(), 2);

    std::vector<unsigned char> seen(static_cast<size_t>(meta.fullRows()), 0);
    size_t validMappings = 0;
    size_t invalidMappings = 0;
    size_t duplicateMappings = 0;
    for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
        for (int z = 0; z < meta.zBranchCount(); ++z) {
            const bool zMirror = z != 0;
            for (int row = 0; row < meta.totalRows; ++row) {
                uint64_t fullIndex = 0;
                if (!spect::io::fullProjectionIndexFromCompactRow(
                        meta, row, sym, zMirror, fullIndex)) {
                    ++invalidMappings;
                    continue;
                }
                ASSERT_LT(fullIndex, static_cast<uint64_t>(meta.fullRows()));
                duplicateMappings += seen[static_cast<size_t>(fullIndex)] != 0;
                seen[static_cast<size_t>(fullIndex)] = 1;
                ++validMappings;
            }
        }
    }

    EXPECT_EQ(validMappings, static_cast<size_t>(meta.fullRows()));
    EXPECT_EQ(invalidMappings, 0u);
    EXPECT_EQ(duplicateMappings, 0u);
    for (unsigned char value : seen) {
        EXPECT_EQ(value, 1);
    }
    EXPECT_NO_THROW(spect::io::validateCompactChunkSymmetricInputs(
        meta, static_cast<uint64_t>(meta.compactRows()), meta.totalVoxels,
        static_cast<size_t>(meta.fullRows()), 48, 48, 360, 108, 108, 80));
}
TEST(IOTest, CompactChunkCSRDirectoryReadsToyChunks) {
    const auto dir = std::filesystem::temp_directory_path() /
                     "spect_compact_chunk_csrb_tests";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    auto writeChunk = [](const std::filesystem::path& path, int32_t rows,
                         int32_t angleOffset, int32_t angleCount,
                         const std::vector<int64_t>& rowPtr,
                         const std::vector<int32_t>& colIdx,
                         const std::vector<float>& values) {
        ASSERT_EQ(rowPtr.size(), static_cast<size_t>(rows) + 1u);
        ASSERT_EQ(colIdx.size(), values.size());
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        const uint32_t magic = 0x42525343u;
        const uint32_t version = 1u;
        const uint64_t nnz = static_cast<uint64_t>(values.size());
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&angleOffset), sizeof(angleOffset));
        out.write(reinterpret_cast<const char*>(&angleCount), sizeof(angleCount));
        out.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
        out.write(reinterpret_cast<const char*>(rowPtr.data()),
                  static_cast<std::streamsize>(rowPtr.size() * sizeof(int64_t)));
        out.write(reinterpret_cast<const char*>(colIdx.data()),
                  static_cast<std::streamsize>(colIdx.size() * sizeof(int32_t)));
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    };

    writeChunk(dir / "chunk_000.csrb", 2, 0, 1, {0, 2, 3}, {0, 2, 4},
               {1.0f, 2.0f, 3.0f});
    writeChunk(dir / "chunk_001.csrb", 2, 1, 1, {0, 1, 2}, {1, 3},
               {4.0f, 5.0f});

    spect::io::CompactSymmetryMetadata meta;
    meta.totalRows = 4;
    meta.totalVoxels = 5;
    meta.compactNumAngles = 2;
    meta.compactRaysPerAngle = 2;

    const auto chunks =
        spect::io::openCompactChunkCSRDirectory(dir.string(), meta, 1024 * 1024);
    ASSERT_EQ(chunks.chunks().size(), 2u);
    EXPECT_EQ(chunks.totalRows(), 4u);
    EXPECT_EQ(chunks.totalNnz(), 5u);
    EXPECT_EQ(chunks.cols(), 5);

    const auto first = chunks.loadChunk(0);
    ASSERT_TRUE(first);
    const auto view = first->view();
    EXPECT_EQ(view.rows, 2);
    EXPECT_EQ(view.cols, 5);
    EXPECT_EQ(view.nnz, 3u);
    ASSERT_EQ(first->rowPtr.size(), 3u);
    EXPECT_EQ(first->rowPtr[0], 0u);
    EXPECT_EQ(first->rowPtr[1], 2u);
    EXPECT_EQ(first->rowPtr[2], 3u);
    ASSERT_EQ(first->colIdx.size(), 3u);
    EXPECT_EQ(first->colIdx[2], 4u);
    EXPECT_FLOAT_EQ(first->values[1], 2.0f);

    std::filesystem::remove_all(dir);
}

TEST(IOTest, CompactChunkCSRDirectoryRejectsNoncanonicalVersion) {
    const auto dir = std::filesystem::temp_directory_path() /
                     "spect_compact_chunk_csrb_version_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const auto path = dir / "chunk_000.csrb";
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out);
    const uint32_t magic = 0x42525343u;
    const uint32_t version = 2u;
    const int32_t rows = 1;
    const int32_t angleOffset = 0;
    const int32_t angleCount = 1;
    const uint64_t nnz = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&angleOffset), sizeof(angleOffset));
    out.write(reinterpret_cast<const char*>(&angleCount), sizeof(angleCount));
    out.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
    out.close();

    spect::io::CompactSymmetryMetadata meta;
    meta.totalRows = 1;
    meta.totalVoxels = 1;
    meta.compactNumAngles = 1;
    meta.compactRaysPerAngle = 1;
    EXPECT_THROW(spect::io::openCompactChunkCSRDirectory(
                     dir.string(), meta, 1024 * 1024),
                 std::runtime_error);

    std::filesystem::remove_all(dir);
}

TEST(IOTest, CompactChunkCSRSampleValidationRejectsTailOutOfRangeColIdx) {
    const auto dir = std::filesystem::temp_directory_path() /
                     "spect_compact_chunk_sample_validation";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const auto path = dir / "chunk_000.csrb";
    const int32_t rows = 2;
    const int32_t angleOffset = 0;
    const int32_t angleCount = 1;
    const std::vector<int64_t> rowPtr = {0, 2, 3};
    const std::vector<int32_t> colIdx = {0, 1, 99};
    const std::vector<float> values = {1.0f, 2.0f, 3.0f};
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out);
    const uint32_t magic = 0x42525343u;
    const uint32_t version = 1u;
    const uint64_t nnz = static_cast<uint64_t>(values.size());
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&angleOffset), sizeof(angleOffset));
    out.write(reinterpret_cast<const char*>(&angleCount), sizeof(angleCount));
    out.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
    out.write(reinterpret_cast<const char*>(rowPtr.data()),
              static_cast<std::streamsize>(rowPtr.size() * sizeof(int64_t)));
    out.write(reinterpret_cast<const char*>(colIdx.data()),
              static_cast<std::streamsize>(colIdx.size() * sizeof(int32_t)));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    out.close();

    spect::io::CompactSymmetryMetadata meta;
    meta.totalRows = 2;
    meta.totalVoxels = 5;
    meta.compactNumAngles = 1;
    meta.compactRaysPerAngle = 2;

    const auto chunks =
        spect::io::openCompactChunkCSRDirectory(dir.string(), meta, 1024 * 1024);
    EXPECT_THROW((void)chunks.loadChunk(0), std::runtime_error);
    std::filesystem::remove_all(dir);
}
TEST(IOTest, CompactSymmetryToyMetadataAnd90ZMappingMatchContract) {
    const auto meta = makeToySym90ZMeta();
    ASSERT_EQ(meta.compactNumAngles, meta.fullNumAngles / 4);
    ASSERT_EQ(meta.detNz, 2 * meta.compactDetZ);
    ASSERT_EQ(meta.xySymmetryCount(), 4);
    ASSERT_EQ(meta.zBranchCount(), 2);

    const std::vector<spect::io::CompactVoxelCoord> expected = {
        {0, 1, 0}, {2, 0, 0}, {3, 2, 0}, {1, 3, 0},
    };
    for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
        const auto mapped = spect::io::mapCompactVoxelCoord(
            0, 1, 0, 4, 5, sym, false);
        EXPECT_EQ(mapped.x, expected[static_cast<size_t>(sym)].x) << sym;
        EXPECT_EQ(mapped.y, expected[static_cast<size_t>(sym)].y) << sym;
        EXPECT_EQ(mapped.z, 0) << sym;
    }
    const auto mirrored =
        spect::io::mapCompactVoxelCoord(0, 1, 1, 4, 5, 0, true);
    EXPECT_EQ(mirrored.x, 0);
    EXPECT_EQ(mirrored.y, 1);
    EXPECT_EQ(mirrored.z, 3);

    bool valid = false;
    EXPECT_EQ(spect::io::fullAngleFromCompactAngle(meta, 0, 0, valid), 0);
    EXPECT_TRUE(valid);
    EXPECT_EQ(spect::io::fullAngleFromCompactAngle(meta, 2, 1, valid), 5);
    EXPECT_TRUE(valid);
    EXPECT_EQ(spect::io::fullAngleFromCompactAngle(meta, 2, 3, valid), 11);
    EXPECT_TRUE(valid);
    EXPECT_EQ(spect::io::fullAngleFromCompactAngle(meta, 3, 0, valid), 0);
    EXPECT_FALSE(valid);
    EXPECT_EQ(spect::io::fullAngleFromCompactAngle(meta, 0, 4, valid), 0);
    EXPECT_FALSE(valid);
}
TEST(IOTest, CompactSymmetryToyExpansionCoversFullProjectionOnce) {
    const auto meta = makeToySym90ZMeta();
    std::vector<unsigned char> seen(static_cast<size_t>(meta.fullRows()), 0);
    size_t validMappings = 0;
    size_t invalidMappings = 0;
    for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
        for (int z = 0; z < meta.zBranchCount(); ++z) {
            const bool zMirror = z != 0;
            for (int row = 0; row < meta.totalRows; ++row) {
                uint64_t fullIndex = 0;
                if (!spect::io::fullProjectionIndexFromCompactRow(
                        meta, row, sym, zMirror, fullIndex)) {
                    ++invalidMappings;
                    continue;
                }
                ASSERT_LT(fullIndex, static_cast<uint64_t>(meta.fullRows()));
                EXPECT_EQ(seen[static_cast<size_t>(fullIndex)], 0) << fullIndex;
                seen[static_cast<size_t>(fullIndex)] = 1;
                ++validMappings;
            }
        }
    }
    EXPECT_EQ(validMappings, static_cast<size_t>(meta.fullRows()));
    EXPECT_EQ(invalidMappings, 0u);
    for (unsigned char value : seen) {
        EXPECT_EQ(value, 1);
    }
}
TEST(IOTest, CompactSymmetryToyNumeratorDenominatorMatchExplicitFull) {
    const auto meta = makeToySym90ZMeta();
    const auto compact = makeToyCompactCSR();
    const auto full = expandCompactToyToFullDense(meta, compact);
    const std::vector<double> image = {1.0, 1.3, 1.7, 2.1,
                                       0.8, 1.2, 1.6, 2.0};
    std::vector<double> projection(static_cast<size_t>(meta.fullRows()), 0.0);
    for (size_t i = 0; i < projection.size(); ++i) {
        projection[i] = 2.0 + 0.05 * static_cast<double>((i * 7u) % 13u);
    }

    const auto fullDenom = denseDenom(full, static_cast<int>(meta.fullRows()),
                                      meta.totalVoxels);
    const auto compactDenom = compactToyDenom(meta, compact);
    expectVectorsNear(compactDenom, fullDenom, 1e-12);

    const auto fullNumer = denseNumerator(
        full, projection, image, static_cast<int>(meta.fullRows()),
        meta.totalVoxels);
    const auto compactNumer = compactToyNumerator(meta, compact, projection, image);
    expectVectorsNear(compactNumer, fullNumer, 1e-12);
}
TEST(IOTest, CompactSymmetryToyMlemOneIterationMatchesExplicitFull) {
    const auto meta = makeToySym90ZMeta();
    const auto compact = makeToyCompactCSR();
    const auto full = expandCompactToyToFullDense(meta, compact);
    const std::vector<double> imageInit = {1.0, 1.3, 1.7, 2.1,
                                             0.8, 1.2, 1.6, 2.0};
    std::vector<double> projection(static_cast<size_t>(meta.fullRows()), 0.0);
    for (size_t i = 0; i < projection.size(); ++i) {
        projection[i] = 2.0 + 0.05 * static_cast<double>((i * 11u + 3u) % 17u);
    }

    const auto fullDenom = denseDenom(full, static_cast<int>(meta.fullRows()),
                                      meta.totalVoxels);
    const auto compactDenom = compactToyDenom(meta, compact);
    const auto fullNumer = denseNumerator(
        full, projection, imageInit, static_cast<int>(meta.fullRows()),
        meta.totalVoxels);
    const auto compactNumer =
        compactToyNumerator(meta, compact, projection, imageInit);

    std::vector<double> imageFull(imageInit.size(), 0.0);
    std::vector<double> imageCompact(imageInit.size(), 0.0);
    for (size_t i = 0; i < imageInit.size(); ++i) {
        imageFull[i] = imageInit[i] * fullNumer[i] /
                       std::max(fullDenom[i], 1e-10);
        imageCompact[i] = imageInit[i] * compactNumer[i] /
                          std::max(compactDenom[i], 1e-10);
    }
    expectVectorsNear(imageCompact, imageFull, 1e-12);
}
TEST(IOTest, CompactSymmetryToyZeroPsfIsIdentityAndGaussianCommutesWithC4) {
    const std::vector<double> image = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        1.5, 2.5, 3.5,
        4.5, 5.5, 6.5,
        7.5, 8.5, 9.5,
    };
    const int nx = 3;
    const int ny = 3;
    const int nz = 2;

    auto zeroPsf = image;
    const auto identityKernel = spect::filters::makeGaussian1D(0.0, 1.0);
    spect::filters::conv3d_separable_inplace(
        zeroPsf, nx, ny, nz, identityKernel, identityKernel, identityKernel, 1);
    expectVectorsNear(zeroPsf, image, 1e-12);

    auto blurred = image;
    const auto xyKernel = spect::filters::makeGaussian1D(1.4, 1.0, 3.0);
    const std::vector<double> zIdentity = {1.0};
    spect::filters::conv3d_separable_inplace(
        blurred, nx, ny, nz, xyKernel, xyKernel, zIdentity, 1);

    for (int sym = 0; sym < 4; ++sym) {
        auto transformedThenBlurred =
            transformedImage(image, nx, ny, nz, sym, false);
        spect::filters::conv3d_separable_inplace(
            transformedThenBlurred, nx, ny, nz, xyKernel, xyKernel, zIdentity, 1);
        const auto blurredThenTransformed =
            transformedImage(blurred, nx, ny, nz, sym, false);
        expectVectorsNear(transformedThenBlurred, blurredThenTransformed, 1e-12);
    }
}

} // namespace test
} // namespace io
} // namespace spect

