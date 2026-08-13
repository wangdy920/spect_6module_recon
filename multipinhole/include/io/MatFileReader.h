/**
 * @file MatFileReader.h
 * @brief MATLAB MAT file reading
 * 
 * Migrated from original_source_backup/dataLoader.h
 */

#pragma once
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <matio.h>

namespace spect {
namespace io {

/**
 * @brief MATLAB array data container
 * 
 * Supports double, float, int32 types
 */
struct MatArray {
    std::vector<size_t> dims;  // MATLAB dimensions (column-major order)
    matio_classes cls = MAT_C_EMPTY;
    matio_types dtype = MAT_T_UNKNOWN;

    std::vector<double> d;      // MAT_C_DOUBLE
    std::vector<float> f;       // MAT_C_SINGLE
    std::vector<int32_t> i32;   // MAT_C_INT32

    size_t numel() const {
        size_t n = 1;
        for (size_t dim : dims)
            n *= dim;
        return n;
    }
};

/**
 * @brief 从 MAT 文件读取变量
 * 
 * @param path MAT 文件路径
 * @param wantVar 变量名（空字符串表示读取第一个变量）
 * @return MatArray 读取的数组数据
 * @throws std::runtime_error 文件不存在或变量未找到
 */
MatArray readMat(const std::string &path, const std::string &wantVar = "");

/**
 * @brief 打印 MatArray 信息
 */
void printMatInfo(const MatArray &arr, const std::string &name);

/**
 * @brief MATLAB 2D 数组元素索引（列主序）
 */
inline size_t matlab2dElem(size_t i, size_t j, size_t rows) {
    return i + rows * j;
}

} // namespace io
} // namespace spect
