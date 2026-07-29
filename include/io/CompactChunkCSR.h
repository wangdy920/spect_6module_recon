#pragma once

#include "io/CSRTypes.h"
#include "io/CompactSymmetryMetadata.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace spect::io {

struct CompactChunkCSRInfo {
  std::filesystem::path path;
  uint32_t version = 1u;
  int rows = 0;
  int angleOffset = 0;
  int angleCount = 0;
  uint64_t nnz = 0;
  uint64_t fileSizeBytes = 0;
};

struct CSRQuantizedView {
  int rows = 0;
  int cols = 0;
  const uint32_t *row_ptr = nullptr;     // length = rows+1
  const uint32_t *col_idx = nullptr;     // length = nnz
  const uint16_t *quantized_values = nullptr; // length = nnz
  const float *row_scales = nullptr;     // length = rows
  uint64_t nnz = 0;

  bool valid() const {
    return row_ptr && col_idx && quantized_values && row_scales &&
           rows >= 0 && cols >= 0 && nnz > 0;
  }
};

struct CompactChunkCSRData {
  CompactChunkCSRInfo info;
  int cols = 0;
  std::vector<uint32_t> rowPtr;
  std::vector<uint32_t> colIdx;
  std::vector<float> values;
  // Quantized payload (version 2, per-row scaled uint16 values).
  std::vector<uint16_t> quantizedValues;
  std::vector<float> rowScales;
  std::shared_ptr<const void> mappedPayloadOwner;
  const uint32_t *mappedColIdx = nullptr;
  const float *mappedValues = nullptr;
  const uint16_t *mappedQuantizedValues = nullptr;
  const float *mappedRowScales = nullptr;
  uint64_t mappedPayloadBytes = 0;

  CSRView view() const;
  CSRQuantizedView quantizedView() const;
  uint64_t residentBytes() const;
  bool isQuantized() const;
  const uint32_t *colIdxData() const;
  const float *valuesData() const;
  const uint16_t *quantizedValuesData() const;
  const float *rowScalesData() const;
  size_t colIdxCount() const;
  size_t valuesCount() const;
  size_t quantizedValuesCount() const;
  size_t rowScalesCount() const;
};

class CompactChunkCSRDirectory {
public:
  CompactChunkCSRDirectory() = default;
  CompactChunkCSRDirectory(std::filesystem::path directory,
                           std::vector<CompactChunkCSRInfo> chunks,
                           int cols, uint64_t cacheBudgetBytes);

  const std::filesystem::path &directory() const { return directory_; }
  const std::vector<CompactChunkCSRInfo> &chunks() const { return chunks_; }
  int cols() const { return cols_; }
  uint64_t totalRows() const { return totalRows_; }
  uint64_t totalNnz() const { return totalNnz_; }
  uint64_t cacheBudgetBytes() const { return cacheBudgetBytes_; }

  std::shared_ptr<const CompactChunkCSRData> loadChunk(size_t index) const;

private:
  std::filesystem::path directory_;
  std::vector<CompactChunkCSRInfo> chunks_;
  int cols_ = 0;
  uint64_t totalRows_ = 0;
  uint64_t totalNnz_ = 0;
  uint64_t cacheBudgetBytes_ = 0;
  mutable uint64_t cachedBytes_ = 0;
  mutable std::vector<std::shared_ptr<const CompactChunkCSRData>> cache_;
};

uint64_t defaultCompactChunkHostCacheBudgetBytes();

CompactChunkCSRDirectory openCompactChunkCSRDirectory(
    const std::string &directory, const CompactSymmetryMetadata &meta,
    uint64_t cacheBudgetBytes = defaultCompactChunkHostCacheBudgetBytes());

} // namespace spect::io
