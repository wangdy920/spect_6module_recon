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
 * @brief One contiguous col_idx/val range of a CSRView (the nnz of one or
 * more adjacent compact angles). Non-owning.
 */
struct CSRSegment {
    const uint32_t* col_idx = nullptr;
    const float* val = nullptr;
    uint64_t nnz = 0;
};

/**
 * @brief Ordered-subset slice of one compact chunk: the rows of the listed
 * compact angles (rowsPerBlock rows each), concatenated in block order. The
 * whole chunk is the slice of all its angles (one segment, tag 0).
 *
 * row_ptr is re-based over the concatenated segments, so the slice is a
 * valid CSR matrix once the segments are copied back to back. Adjacent
 * angles share one segment; blockAngles[b] is the absolute compact angle of
 * row block b, which the ratio / sensitivity kernels use instead of the
 * contiguous compactRowBase mapping. sourceRowPtr (the chunk's row pointer
 * array) and tag identify the (chunk, subset) for device caching; the
 * device key also carries the first segment's pointers.
 */
struct CSRSubsetView {
    int rows = 0;
    int cols = 0;
    uint64_t nnz = 0;
    int rowsPerBlock = 0;
    std::vector<uint32_t> row_ptr;      // rows + 1, re-based
    std::vector<CSRSegment> segments;   // coalesced nnz ranges, block order
    std::vector<int> blockAngles;       // absolute compact angle per block
    const void* sourceRowPtr = nullptr; // chunk identity
    uint64_t tag = 0;                   // subset identity within the chunk

    int blockCount() const { return static_cast<int>(blockAngles.size()); }
    bool empty() const { return rows == 0; }
    bool valid() const {
        return rows >= 0 && cols >= 0 && rowsPerBlock > 0 &&
               row_ptr.size() == static_cast<size_t>(rows) + 1u &&
               (rows == 0 || !segments.empty()) &&
               segments.size() <= blockAngles.size() &&
               static_cast<size_t>(rows) ==
                   blockAngles.size() * static_cast<size_t>(rowsPerBlock);
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
