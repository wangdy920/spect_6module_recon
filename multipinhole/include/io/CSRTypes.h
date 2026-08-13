/**
 * @file CSRTypes.h
 * @brief CSR sparse matrix types (IO-facing, lightweight)
 *
 * This header defines CSR/CSRView/CSRHeader without pulling heavy IO deps.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace spect {
namespace io {

// ================= CSR 基本类型 =================

/**
 * @brief CSR (Compressed Sparse Row) sparse matrix
 *
 * Standard CSR format storage (float values, uint32 indices)
 */
struct CSR {
    int rows = 0;                   // M (number of rows)
    int cols = 0;                   // Nvox (number of columns)
    std::vector<uint32_t> row_ptr;  // size = rows+1
    std::vector<uint32_t> col_idx;  // size = nnz
    std::vector<float> val;         // size = nnz
};

/**
 * @brief Non-owning view of CSR matrix (pointer-based access)
 *
 * Used to avoid data copying
 */
struct CSRView {
    int rows = 0;
    int cols = 0;
    const uint32_t* row_ptr = nullptr; // length = rows+1
    const uint32_t* col_idx = nullptr; // length = nnz
    const float* val = nullptr;        // length = nnz
    uint64_t nnz = 0;

    bool valid() const {
        return row_ptr && col_idx && val && rows >= 0 && cols >= 0;
    }
};

/**
 * @brief CSR binary file header (IO format)
 */
struct CSRHeader {
    uint32_t magic = 0x43535242;   // 'CSRB'
    uint32_t version = 1;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint64_t nnz = 0;
};

static_assert(sizeof(CSRHeader) == 24, "io::CSRHeader must be 24 bytes (CSRB v1 on disk)");

} // namespace io
} // namespace spect
