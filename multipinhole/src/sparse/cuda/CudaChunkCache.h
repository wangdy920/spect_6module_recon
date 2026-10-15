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

// Test hook (setPinnedAllocationLimitForTestsCUDA): while limit >= 0,
// PinnedHostArray::tryAllocate refuses once `limit` arrays it allocated are
// alive, like a driver that has run out of page-lockable memory; releasing an
// array frees its slot. allocate() is not affected.
struct PinnedTryAllocateTestLimit {
    static inline std::atomic<int> limit{-1};
    static inline std::atomic<int> live{0};
};

template <typename T> class PinnedHostArray {
public:
    PinnedHostArray() = default;
    ~PinnedHostArray() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
        }
        uncount();
    }

    PinnedHostArray(const PinnedHostArray &) = delete;
    PinnedHostArray &operator=(const PinnedHostArray &) = delete;
    PinnedHostArray(PinnedHostArray &&other) noexcept
            : ptr_(other.ptr_), capacity_(other.capacity_), counted_(other.counted_) {
        other.ptr_ = nullptr;
        other.capacity_ = 0;
        other.counted_ = false;
    }
    PinnedHostArray &operator=(PinnedHostArray &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
        }
        uncount();
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        counted_ = other.counted_;
        other.ptr_ = nullptr;
        other.capacity_ = 0;
        other.counted_ = false;
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

    // Like allocate, but a refused page-locked allocation leaves the array
    // empty and returns false instead of throwing.
    bool tryAllocate(size_t count) {
        if (count == 0 || count <= capacity_) {
            return true;
        }
        release();
        const int limit = PinnedTryAllocateTestLimit::limit.load();
        if (limit >= 0 && PinnedTryAllocateTestLimit::live.load() >= limit) {
            return false;
        }
        T *ptr = nullptr;
        if (cudaMallocHost(reinterpret_cast<void **>(&ptr), count * sizeof(T)) !=
            cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        ptr_ = ptr;
        capacity_ = count;
        if (limit >= 0) {
            counted_ = true;
            ++PinnedTryAllocateTestLimit::live;
        }
        return true;
    }

    void release() {
        if (ptr_ != nullptr) {
            CUDA_CHECK(cudaFreeHost(ptr_));
            ptr_ = nullptr;
            capacity_ = 0;
        }
        uncount();
    }

    void copyFromHost(const T *host, size_t count) {
        allocate(count);
        if (count == 0) {
            return;
        }
        std::memcpy(ptr_, host, count * sizeof(T));
    }

private:
    void uncount() {
        if (counted_) {
            --PinnedTryAllocateTestLimit::live;
            counted_ = false;
        }
    }

    T *ptr_ = nullptr;
    size_t capacity_ = 0;
    bool counted_ = false;
};

// Identity of an uploaded CSRB matrix: the host arrays it was copied from
// (for a subset slice: the chunk's row pointer array and the first segment)
// plus its shape and the subset tag (0 for a whole chunk, so a whole-chunk
// slice and the chunk's own CSRView share one device copy).
struct MatrixCacheKey {
    const void *rowPtr = nullptr;
    const void *colIdx = nullptr;
    const void *values = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    uint64_t tag = 0;

    bool operator==(const MatrixCacheKey &other) const {
        return rowPtr == other.rowPtr && colIdx == other.colIdx &&
                     values == other.values && rows == other.rows && cols == other.cols &&
                     nnz == other.nnz && tag == other.tag;
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
  size_t csrqPassCalls = 0;
  double csrqDiskReadSeconds = 0.0;
  uint64_t csrqDiskReadBytes = 0;
  size_t csrqDiskReadCalls = 0;
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
  uint64_t csrbMatrixH2DBytes = 0;
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

void noteCudaCusparseForward(double seconds, size_t calls);
void noteCudaCusparseForwardHost(double seconds);
void noteCudaRatio(double seconds, size_t calls);
void noteCudaCusparseBackward(double seconds, size_t calls);
void noteCudaCusparseBackwardHost(double seconds);
void noteCudaTransformBackward(double seconds, size_t calls);
void noteCudaCsrqPass();
void noteCudaCsrqDiskRead(double seconds, uint64_t bytes);
void noteCudaCompactMatrixUpload(double seconds);
void noteCudaCompactMatrixReuseHit();
void noteCudaCompactInputH2D(double seconds);
void noteCudaCompactProjectionH2D(double seconds, size_t bytes);
void noteCudaCompactOutputD2H(double seconds, size_t bytes, bool backward);

MatrixCacheKey makeCacheKey(const io::CSRView &matrix);
MatrixCacheKey makeCacheKey(const io::CSRSubsetView &slice);
uint64_t estimateDeviceCsrBytes(int64_t rows, int64_t nnz);

} // namespace spect

#endif
