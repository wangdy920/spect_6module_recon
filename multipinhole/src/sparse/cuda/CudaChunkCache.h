#pragma once

#include "cuda/CudaConfig.h"
#include "cuda/CudaLaunchConfig.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include "io/CSRTypes.h"
#include "io/CompactChunkCSR.h"
#include "sparse/SparseOpsCUDA.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace spect {

constexpr uint64_t kProjDomainChunkMinNnz = cuda::kProjDomainChunkMinNnz;
constexpr uint64_t kProjDomainChunkDefaultNnz = cuda::kProjDomainChunkDefaultNnz;
constexpr uint64_t kProjDomainChunkMaxNnz = cuda::kProjDomainChunkMaxNnz;
constexpr size_t kTransientRangeSlotCount = cuda::kTransientRangeSlotCount;

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
  double transformBackwardSeconds = 0.0;
  double compactMatrixUploadSeconds = 0.0;
  double compactInputH2DSeconds = 0.0;
  double compactProjectionH2DSeconds = 0.0;
  double compactOutputD2HSeconds = 0.0;
  double compactForwardOutputD2HSeconds = 0.0;
  double compactBackwardOutputD2HSeconds = 0.0;
  double csrqForwardSeconds = 0.0;
  double csrqBackwardSeconds = 0.0;
  size_t csrqForwardCalls = 0;
  size_t csrqBackwardCalls = 0;
  size_t chunkBuildCalls = 0;
  size_t h2dChunkCalls = 0;
  size_t uploadWaitOnComputeCalls = 0;
  size_t fusedBedCycleCalls = 0;
  size_t cusparseForwardCalls = 0;
  size_t cusparseForwardHostCalls = 0;
  size_t ratioCalls = 0;
  size_t cusparseBackwardCalls = 0;
  size_t cusparseBackwardHostCalls = 0;
  size_t transformBackwardCalls = 0;
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
void noteCudaTransformBackward(double seconds, size_t calls);
void noteCudaCsrqForward(double seconds, size_t calls);
void noteCudaCsrqBackward(double seconds, size_t calls);
void noteCudaCompactMatrixUpload(double seconds);
void noteCudaCompactMatrixReuseHit();
void noteCudaCompactInputH2D(double seconds);
void noteCudaCompactProjectionH2D(double seconds, size_t bytes);
void noteCudaCompactOutputD2H(double seconds, size_t bytes, bool backward);

uint64_t selectPreparedChunkHostCacheBudgetCapBytes();

MatrixCacheKey makeCacheKey(const io::CSRView &matrix);
SubsetCacheKey makeSubsetCacheKey(const io::CSRView &matrix,
                                  const std::vector<int> &subset_rows);
RangeCacheKey makeRangeCacheKey(const io::CSRView &matrix, int rowBegin,
                                int rowEnd);

uint64_t estimatePreparedChunkHostBytes(const io::CSRView &matrix);
uint64_t estimateDeviceCsrBytes(int64_t rows, int64_t nnz);

void buildContiguousSubsetCSR(const io::CSRView &matrix, int rowBegin,
                              int rowEnd, int threads,
                              PinnedHostCSRData &host);

} // namespace spect

#endif