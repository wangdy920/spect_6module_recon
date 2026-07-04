#pragma once

#include "cuda/CudaConfig.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include "io/CSRTypes.h"
#include "sparse/SparseOpsBackend.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace spect {

constexpr uint64_t kProjDomainChunkMinNnz = 8ull * 1024ull * 1024ull;
constexpr uint64_t kProjDomainChunkDefaultNnz = 48ull * 1024ull * 1024ull;
constexpr uint64_t kProjDomainChunkMaxNnz = 256ull * 1024ull * 1024ull;
constexpr size_t kTransientRangeSlotCount = 2;

template <typename T> class PinnedHostArray {
public:
    PinnedHostArray() = default;
    ~PinnedHostArray() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
        }
    }

    PinnedHostArray(const PinnedHostArray &) = delete;
    PinnedHostArray &operator=(const PinnedHostArray &) = delete;
    PinnedHostArray(PinnedHostArray &&other) noexcept
            : ptr_(other.ptr_), capacity_(other.capacity_) {
        other.ptr_ = nullptr;
        other.capacity_ = 0;
    }
    PinnedHostArray &operator=(PinnedHostArray &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
        }
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        other.ptr_ = nullptr;
        other.capacity_ = 0;
        return *this;
    }

    T *data() const { return ptr_; }
    size_t capacity() const { return capacity_; }

    void allocate(size_t count) {
        if (count == 0 || count <= capacity_) {
            return;
        }
        if (ptr_ != nullptr) {
            CUDA_CHECK(cudaFreeHost(ptr_));
            ptr_ = nullptr;
            capacity_ = 0;
        }
        CUDA_CHECK(
                cudaMallocHost(reinterpret_cast<void **>(&ptr_), count * sizeof(T)));
        capacity_ = count;
    }

    void copyFromHost(const T *host, size_t count) {
        allocate(count);
        if (count == 0) {
            return;
        }
        std::memcpy(ptr_, host, count * sizeof(T));
    }

private:
    T *ptr_ = nullptr;
    size_t capacity_ = 0;
};

struct PinnedHostCSRData {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    size_t rowPtrCount = 0;
    size_t colIdxCount = 0;
    size_t valuesCount = 0;
    PinnedHostArray<int32_t> rowPtr;
    PinnedHostArray<int32_t> colIdx;
    PinnedHostArray<float> values;
};

struct RowChunkRange {
    int rowBegin = 0;
    int rowEnd = 0;
};

struct PreparedChunkedHostCSR {
    std::vector<RowChunkRange> ranges;
    std::vector<PinnedHostCSRData> chunks;
    std::vector<uint64_t> chunkBytes;
    std::vector<uint64_t> touchStamps;
    std::vector<uint8_t> ready;
    uint64_t residentBytes = 0;
    uint64_t maxResidentBytes = 0;
    uint64_t nextTouchStamp = 1;
};

struct MatrixCacheKey {
    const void *rowPtr = nullptr;
    const void *colIdx = nullptr;
    const void *values = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;

    bool operator==(const MatrixCacheKey &other) const {
        return rowPtr == other.rowPtr && colIdx == other.colIdx &&
                     values == other.values && rows == other.rows && cols == other.cols &&
                     nnz == other.nnz;
    }
};

struct SubsetCacheKey {
    MatrixCacheKey matrix;
    const int *subsetRows = nullptr;
    size_t subsetSize = 0;

    bool operator==(const SubsetCacheKey &other) const {
        return matrix == other.matrix && subsetRows == other.subsetRows &&
                     subsetSize == other.subsetSize;
    }
};

struct RangeCacheKey {
    MatrixCacheKey matrix;
    int rowBegin = 0;
    int rowEnd = 0;

    bool operator==(const RangeCacheKey &other) const {
        return matrix == other.matrix && rowBegin == other.rowBegin &&
                     rowEnd == other.rowEnd;
    }
};

struct ChunkRangeCacheKey {
    MatrixCacheKey matrix;
    uint64_t nnzBudget = 0;

    bool operator==(const ChunkRangeCacheKey &other) const {
        return matrix == other.matrix && nnzBudget == other.nnzBudget;
    }
};

struct PreparedChunkRangeCacheKey {
    ChunkRangeCacheKey range;
    uint64_t hostCacheBudgetBytes = 0;

    bool operator==(const PreparedChunkRangeCacheKey &other) const {
        return range == other.range;
    }
};

struct PreparedChunkedHostCSRCacheEntry {
    PreparedChunkRangeCacheKey key;
    PreparedChunkedHostCSR prepared;
};

struct AttenChunkRangeCacheKey {
    ChunkRangeCacheKey range;
    const uint16_t *atten = nullptr;
    int numAngles = 0;

    bool operator==(const AttenChunkRangeCacheKey &other) const {
        return range == other.range && atten == other.atten &&
                     numAngles == other.numAngles;
    }
};

struct PreparedAttenChunkRangeCacheKey {
    AttenChunkRangeCacheKey range;
    uint64_t hostCacheBudgetBytes = 0;

    bool operator==(const PreparedAttenChunkRangeCacheKey &other) const {
        return range == other.range;
    }
};

struct PreparedAttenChunkedHostCSRCacheEntry {
    PreparedAttenChunkRangeCacheKey key;
    PreparedChunkedHostCSR prepared;
};

struct AttenCacheKey {
    MatrixCacheKey matrix;
    const uint16_t *atten = nullptr;
    int numAngles = 0;

    bool operator==(const AttenCacheKey &other) const {
        return matrix == other.matrix && atten == other.atten &&
                     numAngles == other.numAngles;
    }
};

struct CudaChunkStageTimingState {
  double chunkBuildSeconds = 0.0;
  double h2dChunkSeconds = 0.0;
  double uploadWaitOnComputeSeconds = 0.0;
  double fusedBedCycleSeconds = 0.0;
  double cusparseForwardSeconds = 0.0;
  double cusparseForwardHostSeconds = 0.0;
  double ratioSeconds = 0.0;
  double cusparseBackwardSeconds = 0.0;
  double cusparseBackwardHostSeconds = 0.0;
  double compactMatrixUploadSeconds = 0.0;
  double compactInputH2DSeconds = 0.0;
  double compactProjectionH2DSeconds = 0.0;
  double compactOutputD2HSeconds = 0.0;
  double compactForwardOutputD2HSeconds = 0.0;
  double compactBackwardOutputD2HSeconds = 0.0;
  size_t chunkBuildCalls = 0;
  size_t h2dChunkCalls = 0;
  size_t uploadWaitOnComputeCalls = 0;
  size_t fusedBedCycleCalls = 0;
  size_t cusparseForwardCalls = 0;
  size_t cusparseForwardHostCalls = 0;
  size_t ratioCalls = 0;
  size_t cusparseBackwardCalls = 0;
  size_t cusparseBackwardHostCalls = 0;
  size_t compactMatrixUploadCalls = 0;
  size_t compactMatrixReuseCalls = 0;
  size_t compactInputH2DCalls = 0;
  size_t compactProjectionH2DCalls = 0;
  uint64_t compactProjectionH2DBytes = 0;
  size_t compactOutputD2HCalls = 0;
  size_t compactForwardOutputD2HCalls = 0;
  size_t compactBackwardOutputD2HCalls = 0;
  uint64_t compactForwardOutputD2HBytes = 0;
  uint64_t compactBackwardOutputD2HBytes = 0;
  size_t deviceCacheHitCalls = 0;
};

extern CudaChunkStageTimingState g_cudaChunkStageTiming;
extern std::atomic<uint64_t> g_cudaChunkNnzBudgetOverride;
extern std::atomic<uint64_t> g_cudaPreparedChunkHostCacheBytesOverride;

uint64_t recommendProjDomainChunkNnzBudget(size_t freeBytes);
uint64_t selectCudaChunkNnzBudget();

void noteCudaChunkBuild(double seconds);
void noteCudaH2DChunk(double seconds);
void noteCudaUploadWaitOnCompute(double seconds, size_t calls);
void noteCudaFusedBedCycle(double seconds, size_t calls);
void noteCudaCusparseForward(double seconds, size_t calls);
void noteCudaCusparseForwardHost(double seconds);
void noteCudaRatio(double seconds, size_t calls);
void noteCudaCusparseBackward(double seconds, size_t calls);
void noteCudaCusparseBackwardHost(double seconds);
void noteCudaCompactMatrixUpload(double seconds);
void noteCudaCompactMatrixReuseHit();
void noteCudaCompactInputH2D(double seconds);
void noteCudaCompactProjectionH2D(double seconds, size_t bytes);
void noteCudaCompactOutputD2H(double seconds, size_t bytes, bool backward);

uint64_t selectPreparedChunkHostCacheBudgetCapBytes();
uint64_t splitPreparedChunkHostCacheBudgetBytes(uint64_t totalBudgetBytes,
                                                size_t activePreparedCaches);
uint64_t selectPreparedChunkHostCacheBudgetBytes(
    size_t activePreparedCaches = 1);
uint64_t selectPreparedChunkHostCacheBudgetBytes(
    const io::CSRView &matrix, size_t activePreparedCaches = 1);

MatrixCacheKey makeCacheKey(const CSRView &matrix);
MatrixCacheKey makeCacheKey(const io::CSRView &matrix);
SubsetCacheKey makeSubsetCacheKey(const io::CSRView &matrix,
                                  const std::vector<int> &subset_rows);
AttenCacheKey makeAttenCacheKey(const io::CSRView &matrix,
                                const std::vector<uint16_t> &atten,
                                int numAngles);
RangeCacheKey makeRangeCacheKey(const io::CSRView &matrix, int rowBegin,
                                int rowEnd);
ChunkRangeCacheKey makeChunkRangeCacheKey(const io::CSRView &matrix,
                                          uint64_t nnzBudget);
PreparedChunkRangeCacheKey makePreparedChunkRangeCacheKey(
    const io::CSRView &matrix, uint64_t nnzBudget,
    uint64_t hostCacheBudgetBytes);
AttenChunkRangeCacheKey makeAttenChunkRangeCacheKey(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, uint64_t nnzBudget);
PreparedAttenChunkRangeCacheKey makePreparedAttenChunkRangeCacheKey(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, uint64_t nnzBudget, uint64_t hostCacheBudgetBytes);

bool needsWholeMatrixChunking(const io::CSRView &matrix);
uint64_t estimatePreparedChunkHostBytes(const io::CSRView &matrix);
uint64_t estimateDeviceCsrBytes(int64_t rows, int64_t nnz);
void touchPreparedChunk(PreparedChunkedHostCSR &prepared, size_t chunkIndex);
void buildContiguousSubsetCSR(const io::CSRView &matrix, int rowBegin,
                              int rowEnd, int threads,
                              PinnedHostCSRData &host);
void buildAttenuatedContiguousSubsetCSR(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, int rowBegin, int rowEnd, int threads,
    PinnedHostCSRData &host);
std::vector<RowChunkRange> computeContiguousRowChunks(
    const io::CSRView &matrix, uint64_t nnzBudget);
const std::vector<RowChunkRange> &getContiguousRowChunks(
    const io::CSRView &matrix, uint64_t nnzBudget);
void initializePreparedChunkedHostCacheEntry(
    PreparedChunkedHostCSR &prepared,
    const std::vector<RowChunkRange> &ranges,
    uint64_t maxResidentBytes);
PreparedChunkedHostCSR *getPreparedContiguousHostChunks(
    const io::CSRView &matrix, uint64_t nnzBudget, int threads,
    bool eagerPrepare = false, uint64_t hostCacheBudgetBytes = 0);
const PinnedHostCSRData &ensurePreparedContiguousHostChunk(
    PreparedChunkedHostCSR &prepared, const io::CSRView &matrix,
    size_t chunkIndex, int threads);
PreparedChunkedHostCSR *getPreparedAttenuatedHostChunks(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, uint64_t nnzBudget, int threads,
    bool eagerPrepare = false, uint64_t hostCacheBudgetBytes = 0);
const PinnedHostCSRData &ensurePreparedAttenuatedHostChunk(
    PreparedChunkedHostCSR &prepared, const io::CSRView &matrix,
    const std::vector<uint16_t> &atten, int numAngles, size_t chunkIndex,
    int threads);

} // namespace spect

#endif