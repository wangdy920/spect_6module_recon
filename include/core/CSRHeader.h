/**
 * @file CSRHeader.h
 * @brief Legacy internal double-precision CSR layout (NOT on-disk CSRB v1).
 *
 * @deprecated This header describes an in-memory draft used by spect::CSRView
 * and ISparseOps (int64 row_ptr, double values). It is NOT the CSRB file format.
 * For binary I/O use spect::io::CSRHeader / spect::io::CSRView in io/CSRTypes.h
 * (24-byte header, uint32 dimensions, float values).
 */

#pragma once

#include <cstdint>

namespace spect {

/**
 * @brief CSR二进制文件魔数 'CSRB' (0x43535242)
 */
constexpr uint32_t CSR_MAGIC = 0x43535242;

/**
 * @brief CSR二进制文件头
 * 
 * 文件布局:
 *   [Header: 32 bytes]
 *   [row_ptr: (rows+1) * sizeof(int64_t)]
 *   [col_idx: nnz * sizeof(int32_t)]
 *   [values: nnz * sizeof(double)]
 */
#pragma pack(push, 1)
struct CSRHeader {
    uint32_t magic;     ///< 魔数: 0x43535242 ('CSRB')
    uint32_t version;   ///< 版本号 (当前为1)
    int64_t rows;       ///< 行数
    int64_t cols;       ///< 列数
    int64_t nnz;        ///< 非零元素数
    
    /**
     * @brief 验证头部有效性
     */
    bool isValid() const {
        return magic == CSR_MAGIC && 
               version >= 1 && 
               rows >= 0 && 
               cols >= 0 && 
               nnz >= 0;
    }
    
    /**
     * @brief 计算row_ptr数组的字节偏移
     */
    size_t rowPtrOffset() const {
        return sizeof(CSRHeader);
    }
    
    /**
     * @brief 计算col_idx数组的字节偏移
     */
    size_t colIdxOffset() const {
        return rowPtrOffset() + (rows + 1) * sizeof(int64_t);
    }
    
    /**
     * @brief 计算values数组的字节偏移
     */
    size_t valuesOffset() const {
        return colIdxOffset() + nnz * sizeof(int32_t);
    }
    
    /**
     * @brief 计算文件总大小
     */
    size_t totalFileSize() const {
        return valuesOffset() + nnz * sizeof(double);
    }
};
#pragma pack(pop)

static_assert(sizeof(CSRHeader) == 32, "CSRHeader must be 32 bytes");

} // namespace spect
