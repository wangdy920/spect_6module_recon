#include "CudaChunkCache.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include "CudaCsrBuilder.h"
#include "core/AttenuationConstants.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <memoryapi.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
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

void noteCudaChunkBuild(double seconds) {
  g_cudaChunkStageTiming.chunkBuildSeconds += seconds;
  ++g_cudaChunkStageTiming.chunkBuildCalls;
}

void noteCudaH2DChunk(double seconds) {
  g_cudaChunkStageTiming.h2dChunkSeconds += seconds;
  ++g_cudaChunkStageTiming.h2dChunkCalls;
}

void noteCudaUploadWaitOnCompute(double seconds, size_t calls) {
  g_cudaChunkStageTiming.uploadWaitOnComputeSeconds += seconds;
  g_cudaChunkStageTiming.uploadWaitOnComputeCalls += calls;
}

void noteCudaFusedBedCycle(double seconds, size_t calls) {
  g_cudaChunkStageTiming.fusedBedCycleSeconds += seconds;
  g_cudaChunkStageTiming.fusedBedCycleCalls += calls;
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


SubsetCacheKey makeSubsetCacheKey(const io::CSRView &matrix,
                                  const std::vector<int> &subset_rows) {
  SubsetCacheKey key;
  key.matrix = makeCacheKey(matrix);
  key.subsetRows = subset_rows.empty() ? nullptr : subset_rows.data();
  key.subsetSize = subset_rows.size();
  return key;
}



RangeCacheKey makeRangeCacheKey(const io::CSRView &matrix, int rowBegin,
                                int rowEnd) {
  RangeCacheKey key;
  key.matrix = makeCacheKey(matrix);
  key.rowBegin = rowBegin;
  key.rowEnd = rowEnd;
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



void buildContiguousSubsetCSR(const io::CSRView &matrix, int rowBegin,
                              int rowEnd, int threads,
                              PinnedHostCSRData &host) {
  if (rowBegin < 0 || rowEnd < rowBegin || rowEnd > matrix.rows) {
    throw std::runtime_error(
        "CUDA subset sparse backend received an out-of-range contiguous row block.");
  }

  host.rows = static_cast<int64_t>(rowEnd - rowBegin);
  host.cols = matrix.cols;
  host.nnz = 0;
  host.rowPtrCount = static_cast<size_t>(rowEnd - rowBegin) + 1;
  host.colIdxCount = 0;
  host.valuesCount = 0;
  host.rowPtr.allocate(host.rowPtrCount);
  if (rowBegin == rowEnd) {
    if (host.rowPtrCount > 0) {
      host.rowPtr.data()[0] = 0;
    }
    return;
  }

  const uint32_t baseOffset = matrix.row_ptr[static_cast<size_t>(rowBegin)];
  const uint32_t endOffset = matrix.row_ptr[static_cast<size_t>(rowEnd)];
  const uint64_t nnz = static_cast<uint64_t>(endOffset) -
                       static_cast<uint64_t>(baseOffset);
  if (nnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(
        "CUDA subset sparse backend only supports subset nnz representable as int32.");
  }

  for (int row = rowBegin; row <= rowEnd; ++row) {
    host.rowPtr.data()[static_cast<size_t>(row - rowBegin)] =
        static_cast<int32_t>(matrix.row_ptr[static_cast<size_t>(row)] -
                             baseOffset);
  }

  host.colIdxCount = static_cast<size_t>(nnz);
  host.valuesCount = static_cast<size_t>(nnz);
  host.colIdx.allocate(host.colIdxCount);
  host.values.allocate(host.valuesCount);
  const int parallelThreads = getHostParallelThreads(threads, nnz);
  std::atomic<bool> invalidColumn{false};
#pragma omp parallel for schedule(static) if(parallelThreads > 1) num_threads(parallelThreads)
  for (ptrdiff_t parallelOffset = 0;
       parallelOffset < static_cast<ptrdiff_t>(nnz); ++parallelOffset) {
    const uint64_t offset = static_cast<uint64_t>(parallelOffset);
    const size_t sourceIndex =
        static_cast<size_t>(baseOffset) + static_cast<size_t>(offset);
    const uint32_t col = matrix.col_idx[sourceIndex];
    if (col > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      invalidColumn.store(true, std::memory_order_relaxed);
      host.colIdx.data()[static_cast<size_t>(offset)] = 0;
      host.values.data()[static_cast<size_t>(offset)] = 0.0f;
      continue;
    }
    host.colIdx.data()[static_cast<size_t>(offset)] = static_cast<int32_t>(col);
    host.values.data()[static_cast<size_t>(offset)] = matrix.val[sourceIndex];
  }
  if (invalidColumn.load(std::memory_order_relaxed)) {
    throw std::runtime_error(
        "CUDA sparse backend only supports column indices representable as int32.");
  }

  host.nnz = static_cast<int64_t>(nnz);
}



} // namespace spect

#endif
