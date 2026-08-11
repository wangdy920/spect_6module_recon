#pragma once

#include "io/CSRTypes.h"
#include "io/CompactSymmetryMetadata.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
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


struct CompactChunkCSRData {
  CompactChunkCSRInfo info;
  int cols = 0;
  std::vector<uint32_t> rowPtr;
  std::vector<uint32_t> colIdx;
  std::vector<float> values;
  std::shared_ptr<const void> mappedPayloadOwner;
  const uint32_t *mappedColIdx = nullptr;
  const float *mappedValues = nullptr;
  uint64_t mappedPayloadBytes = 0;

  CSRView view() const;
  const uint32_t *colIdxData() const;
  const float *valuesData() const;
  size_t colIdxCount() const;
  size_t valuesCount() const;
  uint64_t residentBytes() const;
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
  std::shared_ptr<void> acceleratorLease() const { return acceleratorLease_; }
  void setAcceleratorLease(std::shared_ptr<void> lease) const {
    acceleratorLease_ = std::move(lease);
  }

private:
  std::filesystem::path directory_;
  std::vector<CompactChunkCSRInfo> chunks_;
  int cols_ = 0;
  uint64_t totalRows_ = 0;
  uint64_t totalNnz_ = 0;
  uint64_t cacheBudgetBytes_ = 0;
  mutable uint64_t cachedBytes_ = 0;
  mutable std::vector<std::shared_ptr<const CompactChunkCSRData>> cache_;
  // Declared after cache_: release accelerator registrations before chunk data.
  mutable std::shared_ptr<void> acceleratorLease_;
};

uint64_t defaultCompactChunkHostCacheBudgetBytes();

CompactChunkCSRDirectory openCompactChunkCSRDirectory(
    const std::string &directory, const CompactSymmetryMetadata &meta,
    uint64_t cacheBudgetBytes = defaultCompactChunkHostCacheBudgetBytes());

} // namespace spect::io
