#include "CudaChunkCache.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include "core/AttenuationConstants.h"

#include <algorithm>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <memoryapi.h>
#endif


namespace spect {

namespace {

struct SystemMemoryInfo {
  uint64_t totalBytes = 0;
  uint64_t availableBytes = 0;
};

#if defined(_WIN32)
SystemMemoryInfo querySystemMemoryInfo() {
  MEMORYSTATUSEX memStatus{};
  memStatus.dwLength = sizeof(memStatus);
  if (!GlobalMemoryStatusEx(&memStatus)) {
    return {};
  }
  return {static_cast<uint64_t>(memStatus.ullTotalPhys),
          static_cast<uint64_t>(memStatus.ullAvailPhys)};
}
#else
SystemMemoryInfo querySystemMemoryInfo() { return {}; }
#endif

} // namespace

CudaChunkStageTimingState g_cudaChunkStageTiming;

uint64_t recommendProjDomainChunkNnzBudget(size_t freeBytes) {
  uint64_t usableBytes = static_cast<uint64_t>(freeBytes);
  if (usableBytes > cuda::kProjDomainChunkFreeMemoryReserveBytes) {
    usableBytes -= cuda::kProjDomainChunkFreeMemoryReserveBytes;
  }

    const uint64_t targetBytes =
    usableBytes >= cuda::kProjDomainChunkHighMemThresholdBytes
        ? std::min<uint64_t>(
          usableBytes / cuda::kProjDomainChunkHighMemDivisor,
          cuda::kProjDomainChunkTargetBytesCap)
        : std::min<uint64_t>(
          usableBytes / cuda::kProjDomainChunkLowMemDivisor,
          cuda::kProjDomainChunkLowMemTargetBytesCap);
  if (targetBytes == 0) {
    return kProjDomainChunkDefaultNnz;
  }

  return std::clamp<uint64_t>(targetBytes / cuda::kProjDomainChunkBytesPerNnz,
                              kProjDomainChunkMinNnz,
                              kProjDomainChunkMaxNnz);
}

uint64_t selectCudaChunkNnzBudget() {
  try {
    if (!cuda::isCudaAvailable()) {
      return kProjDomainChunkDefaultNnz;
    }
    return recommendProjDomainChunkNnzBudget(cuda::getMemoryInfo().free);
  } catch (...) {
    return kProjDomainChunkDefaultNnz;
  }
}

void noteCudaCusparseForward(double seconds, size_t calls) {
  g_cudaChunkStageTiming.cusparseForwardSeconds += seconds;
  g_cudaChunkStageTiming.cusparseForwardCalls += calls;
}

void noteCudaCusparseForwardHost(double seconds) {
  g_cudaChunkStageTiming.cusparseForwardHostSeconds += seconds;
  ++g_cudaChunkStageTiming.cusparseForwardHostCalls;
}

void noteCudaRatio(double seconds, size_t calls) {
  g_cudaChunkStageTiming.ratioSeconds += seconds;
  g_cudaChunkStageTiming.ratioCalls += calls;
}

void noteCudaCusparseBackward(double seconds, size_t calls) {
  g_cudaChunkStageTiming.cusparseBackwardSeconds += seconds;
  g_cudaChunkStageTiming.cusparseBackwardCalls += calls;
}

void noteCudaCusparseBackwardHost(double seconds) {
  g_cudaChunkStageTiming.cusparseBackwardHostSeconds += seconds;
  ++g_cudaChunkStageTiming.cusparseBackwardHostCalls;
}

void noteCudaTransformBackward(double seconds, size_t calls) {
  g_cudaChunkStageTiming.transformBackwardSeconds += seconds;
  g_cudaChunkStageTiming.transformBackwardCalls += calls;
}

void noteCudaCsrqForward(double seconds, size_t calls) {
  g_cudaChunkStageTiming.csrqForwardSeconds += seconds;
  g_cudaChunkStageTiming.csrqForwardCalls += calls;
}

void noteCudaCsrqBackward(double seconds, size_t calls) {
  g_cudaChunkStageTiming.csrqBackwardSeconds += seconds;
  g_cudaChunkStageTiming.csrqBackwardCalls += calls;
}

void noteCudaCompactMatrixUpload(double seconds) {
  g_cudaChunkStageTiming.compactMatrixUploadSeconds += seconds;
  ++g_cudaChunkStageTiming.compactMatrixUploadCalls;
}

void noteCudaCompactMatrixReuseHit() {
  ++g_cudaChunkStageTiming.compactMatrixReuseCalls;
}

void noteCudaCompactInputH2D(double seconds) {
  g_cudaChunkStageTiming.compactInputH2DSeconds += seconds;
  ++g_cudaChunkStageTiming.compactInputH2DCalls;
}

void noteCudaCompactProjectionH2D(double seconds, size_t bytes) {
  g_cudaChunkStageTiming.compactProjectionH2DSeconds += seconds;
  ++g_cudaChunkStageTiming.compactProjectionH2DCalls;
  g_cudaChunkStageTiming.compactProjectionH2DBytes +=
      static_cast<uint64_t>(bytes);
}

void noteCudaCompactOutputD2H(double seconds, size_t bytes, bool backward) {
  g_cudaChunkStageTiming.compactOutputD2HSeconds += seconds;
  ++g_cudaChunkStageTiming.compactOutputD2HCalls;
  if (backward) {
    g_cudaChunkStageTiming.compactBackwardOutputD2HSeconds += seconds;
    ++g_cudaChunkStageTiming.compactBackwardOutputD2HCalls;
    g_cudaChunkStageTiming.compactBackwardOutputD2HBytes +=
        static_cast<uint64_t>(bytes);
  } else {
    g_cudaChunkStageTiming.compactForwardOutputD2HSeconds += seconds;
    ++g_cudaChunkStageTiming.compactForwardOutputD2HCalls;
    g_cudaChunkStageTiming.compactForwardOutputD2HBytes +=
        static_cast<uint64_t>(bytes);
  }
}

uint64_t selectPreparedChunkHostCacheBudgetCapBytes() {
  const SystemMemoryInfo memory = querySystemMemoryInfo();
  uint64_t cap = cuda::kPreparedChunkHostCacheMinBytes;
  // P3-A: Use 60 GiB threshold instead of 64 GiB — Windows reserves ~0.5 GiB
  // for hardware (BIOS, GPU BAR, etc.), so a 64 GB system reports ~63.8 GiB.
  if (memory.totalBytes >= cuda::kPreparedChunkHostCacheHighRamThresholdBytes) {
    cap = cuda::kPreparedChunkHostCacheMaxBytes;
  } else if (memory.totalBytes >=
             cuda::kPreparedChunkHostCacheMidRamThresholdBytes) {
    cap = cuda::kPreparedChunkHostCacheMidBytes;
  }
  // Code.CUDA.41: On high-RAM systems, Windows MEMORYSTATUSEX.ullAvailPhys
  // counts memory-mapped file pages (e.g. the 21 GB CSR file) as "in use",
  // even though file-backed pages are trivially evictable by the OS.
  // Use totalBytes*2/3 as a floor for effectiveAvailable to prevent the
  // cache budget from being suppressed below the matrix size.
  const uint64_t effectiveAvailable =
      (memory.totalBytes >= cuda::kPreparedChunkHostCacheHighRamThresholdBytes)
            ? std::max(
              memory.availableBytes,
              memory.totalBytes *
                cuda::kPreparedChunkHostCacheEffectiveAvailableNumerator /
                cuda::kPreparedChunkHostCacheEffectiveAvailableDenominator)
          : memory.availableBytes;
  if (effectiveAvailable > cuda::kPreparedChunkHostCacheSafetyReserveBytes) {
    cap = std::min<uint64_t>(
        cap, effectiveAvailable - cuda::kPreparedChunkHostCacheSafetyReserveBytes);
  } else {
    cap = std::min<uint64_t>(cap, cuda::kPreparedChunkHostCacheEmergencyBytes);
  }
  return cap;
}



MatrixCacheKey makeCacheKey(const io::CSRView &matrix) {
  MatrixCacheKey key;
  key.rowPtr = matrix.row_ptr;
  key.colIdx = matrix.col_idx;
  key.values = matrix.val;
  key.rows = matrix.rows;
  key.cols = matrix.cols;
  key.nnz = static_cast<int64_t>(matrix.nnz);
  return key;
}

MatrixCacheKey makeCacheKey(const io::CSRSubsetView &slice) {
  MatrixCacheKey key;
  key.rowPtr = slice.sourceRowPtr;
  if (!slice.segments.empty()) {
    key.colIdx = slice.segments.front().col_idx;
    key.values = slice.segments.front().val;
  }
  key.rows = slice.rows;
  key.cols = slice.cols;
  key.nnz = static_cast<int64_t>(slice.nnz);
  key.tag = slice.tag;
  return key;
}


uint64_t estimateDeviceCsrBytes(int64_t rows, int64_t nnz) {
  if (rows <= 0 || nnz <= 0) {
    return 0;
  }
  return (static_cast<uint64_t>(rows) + 1ull) * sizeof(int32_t) +
         static_cast<uint64_t>(nnz) * sizeof(int32_t) +
         static_cast<uint64_t>(nnz) * sizeof(float);
}



} // namespace spect

#endif
