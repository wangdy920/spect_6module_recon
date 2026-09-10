#pragma once

#include "io/CSRTypes.h"
#include "io/CompactSymmetryMetadata.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace spect::io {

// Two on-disk chunk encodings share one directory abstraction:
//   CsrbFloat32  chunk_*.csrb  canonical: int64 row_ptr, uint32 col_idx, f32 val
//   CsrqRunU16   chunk_*.csrq  derived (tools/convert_csrb_to_csrq.py): sorted
//                columns run-indexed, u16 sqrt-companded values with one f32
//                row max. Decode: v = rowMax * (q / 65535)^2 in fp32.
enum class CompactChunkFormat { CsrbFloat32, CsrqRunU16 };

// CSRQ run word: (startCol << kCompactRunLenBits) | (len - 1), len in 1..64.
constexpr int kCompactRunLenBits = 6;
constexpr int kCompactRunMaxLen = 1 << kCompactRunLenBits;
constexpr uint32_t kCompactRunMaxStartCol = 1u << (32 - kCompactRunLenBits);

struct CompactChunkCSRInfo {
  std::filesystem::path path;
  CompactChunkFormat format = CompactChunkFormat::CsrbFloat32;
  uint32_t version = 1u;
  int rows = 0;
  int angleOffset = 0;
  int angleCount = 0;
  uint64_t nnz = 0;
  uint64_t runCount = 0; // CSRQ only
  uint64_t fileSizeBytes = 0;
};

// Host view of one CSRQ chunk (all arrays chunk-local, nnz/run offsets u32).
struct CompactRunU16View {
  int rows = 0;
  int cols = 0;
  uint64_t nnz = 0;
  uint64_t runCount = 0;
  const uint32_t *rowNnzPtr = nullptr; // rows + 1
  const uint32_t *rowRunPtr = nullptr; // rows + 1
  const float *rowMax = nullptr;       // rows
  const uint32_t *runs = nullptr;      // runCount packed run words
  const uint16_t *valuesQ = nullptr;   // nnz, run order within each row

  bool valid() const {
    return rows > 0 && cols > 0 && rowNnzPtr != nullptr &&
           rowRunPtr != nullptr && rowMax != nullptr &&
           (runCount == 0 || runs != nullptr) &&
           (nnz == 0 || valuesQ != nullptr);
  }
};


struct CompactChunkCSRData {
  CompactChunkCSRInfo info;
  int cols = 0;
  // CSRB: rowPtr/colIdx/values. CSRQ: rowPtr (nnz offsets) + the four arrays
  // below; colIdx/values stay empty.
  std::vector<uint32_t> rowPtr;
  std::vector<uint32_t> colIdx;
  std::vector<float> values;
  std::vector<uint32_t> rowRunPtr;
  std::vector<float> rowMax;
  std::vector<uint32_t> runs;
  std::vector<uint16_t> valuesQ;
  std::shared_ptr<const void> mappedPayloadOwner;
  const uint32_t *mappedColIdx = nullptr;
  const float *mappedValues = nullptr;
  uint64_t mappedPayloadBytes = 0;

  CSRView view() const;          // CSRB only; throws for CSRQ
  CompactRunU16View runView() const; // CSRQ only; throws for CSRB
  // CSRB only: the rows of the given absolute compact angles (strictly
  // ascending) that fall inside this chunk, as an ordered-subset slice with
  // adjacent angles merged into one segment. Empty when none do. tag
  // identifies the subset for device caching (0 = whole chunk).
  CSRSubsetView subsetView(const std::vector<int> &compactAngles,
                           int raysPerAngle, uint64_t tag) const;
  // CSRB only: the canonical rows of one POSEM pixel subset (rows r of
  // absolute angle a with compactPixelSubsetId(a, r, count) == index),
  // ascending, as a slice with rowCompactIndex mapping. count > 0,
  // 0 <= index < count.
  CSRSubsetView pixelSubsetView(int count, int index, int raysPerAngle,
                                uint64_t tag) const;
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
  CompactChunkFormat format() const {
    return chunks_.empty() ? CompactChunkFormat::CsrbFloat32 : chunks_.front().format;
  }
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
  mutable std::vector<std::shared_ptr<std::mutex>> cacheMutexes_;
  mutable std::shared_ptr<std::mutex> cacheBudgetMutex_ =
      std::make_shared<std::mutex>();
  // Declared after cache_: release accelerator registrations before chunk data.
  mutable std::shared_ptr<void> acceleratorLease_;
};

CompactChunkCSRDirectory openCompactChunkCSRDirectory(
    const std::string &directory, const CompactSymmetryMetadata &meta);

CompactChunkCSRDirectory openCompactChunkCSRDirectory(
    const std::string &directory, const CompactSymmetryMetadata &meta,
    uint64_t cacheBudgetBytes);

uint64_t recommendCompactChunkHostCacheBudgetBytes(
    uint64_t matrixBytes, uint64_t totalMemoryBytes,
    uint64_t availableMemoryBytes);

} // namespace spect::io
