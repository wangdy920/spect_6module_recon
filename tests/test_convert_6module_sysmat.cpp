/**
 * @file test_convert_6module_sysmat.cpp
 * @brief Minimal regression for tools/convert_6module_sysmat_to_csrb (Epic 8 §4.3).
 */

#include <gtest/gtest.h>

#include "io/CSRBuilder.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <spawn.h>
#include <unistd.h>
#endif

#ifndef SPECT_TEST_BINARY_DIR
#define SPECT_TEST_BINARY_DIR "."
#endif

namespace {

void writeExact(std::ofstream& out, const void* data, std::size_t bytes) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    ASSERT_TRUE(out.good());
}

void writeMetadata(const std::filesystem::path& path, int32_t totalRows,
                   int32_t totalVoxels) {
    std::vector<int32_t> words(24, 0);
    words[0] = totalRows;
    words[1] = totalVoxels;
    words[3] = totalRows;
    words[4] = 1;
    words[5] = 1;
    words[6] = 1;
    words[7] = totalRows;
    words[8] = 1;
    words[9] = 1;
    words[10] = totalVoxels;

    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out.is_open());
    writeExact(out, words.data(), words.size() * sizeof(int32_t));
}

void writeChunkRow(std::ofstream& out, int32_t col, float value) {
    const int32_t nnz = 1;
    writeExact(out, &nnz, sizeof(nnz));
    writeExact(out, &col, sizeof(col));
    writeExact(out, &value, sizeof(value));
}

void writeChunk(const std::filesystem::path& path, int32_t rows) {
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out.is_open());
    const int32_t angleOffset = 0;
    const int32_t angleCount = rows;
    writeExact(out, &rows, sizeof(rows));
    writeExact(out, &angleOffset, sizeof(angleOffset));
    writeExact(out, &angleCount, sizeof(angleCount));
    writeChunkRow(out, 0, 1.0f);
    if (rows > 1)
        writeChunkRow(out, 2, 2.0f);
}

std::filesystem::path converterExecutable() {
#if defined(_WIN32)
    const char* name = "spect_convert_6module_sysmat_to_csrb.exe";
#else
    const char* name = "spect_convert_6module_sysmat_to_csrb";
#endif
    const std::filesystem::path root(SPECT_TEST_BINARY_DIR);
    const std::vector<std::filesystem::path> candidates = {
        root / "bin" / name,
        root / "bin" / "Release" / name,
        root / "bin" / "Debug" / name,
        root / "bin" / "RelWithDebInfo" / name,
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate))
            return candidate;
    }
    return candidates.front();
}

int runConverter(const std::filesystem::path& tool,
               const std::filesystem::path& matrixDir,
               const std::filesystem::path& outPath) {
#if defined(_WIN32)
    const std::wstring exe = tool.wstring();
    const std::wstring matrixDirW = matrixDir.wstring();
    const std::wstring outPathW = outPath.wstring();
    const wchar_t* argv[] = {exe.c_str(),      L"--matrix-dir",
                             matrixDirW.c_str(), L"--out", outPathW.c_str(),
                             L"--quiet",       nullptr};
    return _wspawnvp(_P_WAIT, exe.c_str(), argv);
#else
    const std::string exe = tool.string();
    const std::string matrixDirS = matrixDir.string();
    const std::string outPathS = outPath.string();
    const char* argv[] = {exe.c_str(), "--matrix-dir", matrixDirS.c_str(), "--out",
                          outPathS.c_str(), "--quiet", nullptr};
    return spawnvp(P_WAIT, exe.c_str(), const_cast<char* const*>(argv));
#endif
}

} // namespace

namespace spect {
namespace tools {
namespace test {

TEST(Convert6moduleTest, ToyChunkToCsrbHeaderMatches) {
    const std::filesystem::path tool = converterExecutable();
    if (!std::filesystem::exists(tool)) {
        GTEST_SKIP() << "Converter executable not found: " << tool;
    }

    const auto root =
        std::filesystem::temp_directory_path() / "spect_convert_6module_tests";
    const auto matrixDir = root / "matrix";
    std::filesystem::create_directories(matrixDir);

    writeMetadata(matrixDir / "metadata.bin", 2, 3);
    writeChunk(matrixDir / "chunk_000.bin", 2);

    const auto outPath = root / "matrix.csrb";
    std::error_code ec;
    std::filesystem::remove(outPath, ec);

    const int exitCode = runConverter(tool, matrixDir, outPath);
    ASSERT_EQ(exitCode, 0) << "converter failed for " << tool;
    ASSERT_TRUE(std::filesystem::exists(outPath));

    const spect::io::CSR csr = spect::io::loadCSRBinary(outPath.string());
    EXPECT_EQ(csr.rows, 2);
    EXPECT_EQ(csr.cols, 3);
    ASSERT_EQ(csr.row_ptr.size(), 3u);
    EXPECT_EQ(csr.row_ptr.front(), 0u);
    EXPECT_EQ(csr.row_ptr.back(), 2u);
    ASSERT_EQ(csr.col_idx.size(), 2u);
    EXPECT_EQ(csr.col_idx[0], 0u);
    EXPECT_EQ(csr.col_idx[1], 2u);
    ASSERT_EQ(csr.val.size(), 2u);
    EXPECT_FLOAT_EQ(csr.val[0], 1.0f);
    EXPECT_FLOAT_EQ(csr.val[1], 2.0f);
}

} // namespace test
} // namespace tools
} // namespace spect
