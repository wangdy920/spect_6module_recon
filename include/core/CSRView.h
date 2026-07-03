/**
 * @file CSRView.h
 * @brief CSR稀疏矩阵视图 (POD结构，不拥有内存)
 * 
 * 此文件不应包含任何第三方头文件
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace spect {

/**
 * @brief CSR稀疏矩阵视图
 * 
 * POD结构，仅持有指针，不管理内存生命周期。
 * 用于在不同组件间传递稀疏矩阵数据。
 * 
 * 格式约束:
 * - row_ptr[0] == 0
 * - row_ptr[i] <= row_ptr[i+1] (单调递增)
 * - row_ptr[rows] == nnz
 * - col_idx[k] 属于 [0, cols) 对于所有 k
 */
struct CSRView {
    const int64_t* row_ptr = nullptr;  ///< 行指针数组 (rows+1)
    const int32_t* col_idx = nullptr;  ///< 列索引数组 (nnz)
    const double*  values  = nullptr;  ///< 值数组 (nnz)
    int64_t rows = 0;                  ///< 行数
    int64_t cols = 0;                  ///< 列数
    int64_t nnz  = 0;                  ///< 非零元素数
    
    /**
     * @brief 检查视图是否有效
     */
    bool isValid() const {
        return row_ptr != nullptr && 
               col_idx != nullptr && 
               values != nullptr && 
               rows > 0 && 
               cols > 0 && 
               nnz >= 0;
    }
    
    /**
     * @brief 获取指定行的非零元素范围
     * @param row 行索引
     * @param start 输出: 起始位置
     * @param end 输出: 结束位置
     */
    void getRowRange(int64_t row, int64_t& start, int64_t& end) const {
        start = row_ptr[row];
        end = row_ptr[row + 1];
    }
    
    /**
     * @brief 获取指定行的非零元素数
     */
    int64_t getRowNnz(int64_t row) const {
        return row_ptr[row + 1] - row_ptr[row];
    }
};

} // namespace spect
