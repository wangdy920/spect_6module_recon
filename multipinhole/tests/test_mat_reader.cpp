/**
 * @file test_mat_reader.cpp
 * @brief MAT 文件读入路径覆盖测试（double/float/int32）
 */

#include <gtest/gtest.h>
#include <matio.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "io/MatFileReader.h"

namespace spect {
namespace test {
namespace {

std::filesystem::path makeTempMatPath() {
    const auto base = std::filesystem::temp_directory_path() / "spect_mat_tests";
    std::filesystem::create_directories(base);
    const auto stamp = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    return base / ("mat_reader_" + std::to_string(stamp) + ".mat");
}

template <typename T>
void fillMatrix(std::vector<T>& data, int rows, int cols, T base) {
    data.resize(static_cast<size_t>(rows) * cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            const size_t idx = spect::io::matlab2dElem(
                static_cast<size_t>(i), static_cast<size_t>(j),
                static_cast<size_t>(rows));
            data[idx] = static_cast<T>(base + static_cast<T>(i + 10 * j));
        }
    }
}

void writeMatFile(const std::filesystem::path& path,
                  const std::vector<double>& a,
                  const std::vector<float>& b,
                  const std::vector<int32_t>& c,
                  int rows,
                  int cols) {
    mat_t* mat = Mat_CreateVer(path.string().c_str(), nullptr, MAT_FT_MAT5);
    ASSERT_NE(mat, nullptr);

    size_t dims[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};

    matvar_t* varA = Mat_VarCreate("A", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims,
                                   const_cast<double*>(a.data()), 0);
    ASSERT_NE(varA, nullptr);
    ASSERT_EQ(Mat_VarWrite(mat, varA, MAT_COMPRESSION_NONE), 0);
    Mat_VarFree(varA);

    matvar_t* varB = Mat_VarCreate("B", MAT_C_SINGLE, MAT_T_SINGLE, 2, dims,
                                   const_cast<float*>(b.data()), 0);
    ASSERT_NE(varB, nullptr);
    ASSERT_EQ(Mat_VarWrite(mat, varB, MAT_COMPRESSION_NONE), 0);
    Mat_VarFree(varB);

    matvar_t* varC = Mat_VarCreate("C", MAT_C_INT32, MAT_T_INT32, 2, dims,
                                   const_cast<int32_t*>(c.data()), 0);
    ASSERT_NE(varC, nullptr);
    ASSERT_EQ(Mat_VarWrite(mat, varC, MAT_COMPRESSION_NONE), 0);
    Mat_VarFree(varC);

    Mat_Close(mat);
}

}  // namespace

TEST(MatFileReaderTest, ReadsDoubleFloatInt32AndFirstVar) {
    const int rows = 2;
    const int cols = 3;

    std::vector<double> a;
    std::vector<float> b;
    std::vector<int32_t> c;
    fillMatrix(a, rows, cols, 1.0);
    fillMatrix(b, rows, cols, 2.0f);
    fillMatrix(c, rows, cols, static_cast<int32_t>(3));

    const auto path = makeTempMatPath();
    writeMatFile(path, a, b, c, rows, cols);

    const auto a_read = spect::io::readMat(path.string(), "A");
    EXPECT_EQ(a_read.cls, MAT_C_DOUBLE);
    EXPECT_EQ(a_read.dtype, MAT_T_DOUBLE);
    EXPECT_EQ(a_read.dims, (std::vector<size_t>{static_cast<size_t>(rows),
                                                static_cast<size_t>(cols)}));
    ASSERT_EQ(a_read.d.size(), a.size());
    const size_t idx = spect::io::matlab2dElem(1, 2, rows);
    EXPECT_DOUBLE_EQ(a_read.d[idx], a[idx]);

    const auto first = spect::io::readMat(path.string());
    EXPECT_EQ(first.cls, MAT_C_DOUBLE);
    ASSERT_EQ(first.d.size(), a.size());
    EXPECT_DOUBLE_EQ(first.d[idx], a[idx]);

    const auto b_read = spect::io::readMat(path.string(), "B");
    EXPECT_EQ(b_read.cls, MAT_C_SINGLE);
    EXPECT_EQ(b_read.dtype, MAT_T_SINGLE);
    ASSERT_EQ(b_read.f.size(), b.size());
    EXPECT_FLOAT_EQ(b_read.f[idx], b[idx]);

    const auto c_read = spect::io::readMat(path.string(), "C");
    EXPECT_EQ(c_read.cls, MAT_C_INT32);
    EXPECT_EQ(c_read.dtype, MAT_T_INT32);
    ASSERT_EQ(c_read.i32.size(), c.size());
    EXPECT_EQ(c_read.i32[idx], c[idx]);

    std::filesystem::remove(path);
}

}  // namespace test
}  // namespace spect
