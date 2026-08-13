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

uint64_t splitPreparedChunkHostCacheBudgetBytes(uint64_t totalBudgetBytes,
                                                size_t activePreparedCaches) {
  if (totalBudgetBytes == 0) {
    return 0;
  }
  const uint64_t cacheCount =
      static_cast<uint64_t>(std::max<size_t>(1, activePreparedCaches));
  return std::max<uint64_t>(1ull, totalBudgetBytes / cacheCount);
}

uint64_t selectPreparedChunkHostCacheBudgetBytes(
    size_t activePreparedCaches) {
  const uint64_t cap = selectPreparedChunkHostCacheBudgetCapBytes();
  if (cap == 0) {
    return splitPreparedChunkHostCacheBudgetBytes(
        cuda::kPreparedChunkHostCacheEmergencyBytes, activePreparedCaches);
  }
  return splitPreparedChunkHostCacheBudgetBytes(cap, activePreparedCaches);
}

uint64_t selectPreparedChunkHostCacheBudgetBytes(
    const io::CSRView &matrix, size_t activePreparedCaches) {
  if (!matrix.valid() || matrix.row_ptr == nullptr) {
    return 0;
  }
  return selectPreparedChunkHostCacheBudgetBytes(activePreparedCaches);
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

AttenCacheKey makeAttenCacheKey(const io::CSRView &matrix,
                                const std::vector<uint16_t> &atten,
                                int numAngles) {
  AttenCacheKey key;
  key.matrix = makeCacheKey(matrix);
  key.atten = atten.data();
  key.numAngles = numAngles;
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

ChunkRangeCacheKey makeChunkRangeCacheKey(const io::CSRView &matrix,
                                          uint64_t nnzBudget) {
  ChunkRangeCacheKey key;
  key.matrix = makeCacheKey(matrix);
  key.nnzBudget = nnzBudget;
  return key;
}

PreparedChunkRangeCacheKey makePreparedChunkRangeCacheKey(
    const io::CSRView &matrix, uint64_t nnzBudget,
    uint64_t hostCacheBudgetBytes) {
  PreparedChunkRangeCacheKey key;
  key.range = makeChunkRangeCacheKey(matrix, nnzBudget);
  key.hostCacheBudgetBytes = hostCacheBudgetBytes;
  return key;
}

AttenChunkRangeCacheKey makeAttenChunkRangeCacheKey(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, uint64_t nnzBudget) {
  AttenChunkRangeCacheKey key;
  key.range = makeChunkRangeCacheKey(matrix, nnzBudget);
  key.atten = atten.data();
  key.numAngles = numAngles;
  return key;
}

PreparedAttenChunkRangeCacheKey makePreparedAttenChunkRangeCacheKey(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, uint64_t nnzBudget, uint64_t hostCacheBudgetBytes) {
  PreparedAttenChunkRangeCacheKey key;
  key.range = makeAttenChunkRangeCacheKey(matrix, atten, numAngles, nnzBudget);
  key.hostCacheBudgetBytes = hostCacheBudgetBytes;
  return key;
}

bool needsWholeMatrixChunking(const io::CSRView &matrix) {
  if (!matrix.valid() || matrix.row_ptr == nullptr) {
    return false;
  }
  const uint64_t nnz =
      static_cast<uint64_t>(matrix.row_ptr[static_cast<size_t>(matrix.rows)]);
  return nnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
}

uint64_t estimatePreparedChunkHostBytes(const io::CSRView &matrix) {
  if (!matrix.valid() || matrix.row_ptr == nullptr) {
    return 0;
  }
  const uint64_t nnz =
      static_cast<uint64_t>(matrix.row_ptr[static_cast<size_t>(matrix.rows)]);
  const uint64_t rowPtrBytes =
      (static_cast<uint64_t>(matrix.rows) + 1ull) * sizeof(int32_t);
  const uint64_t colIdxBytes = nnz * sizeof(int32_t);
  const uint64_t valuesBytes = nnz * sizeof(float);
  return rowPtrBytes + colIdxBytes + valuesBytes;
}

uint64_t estimateDeviceCsrBytes(int64_t rows, int64_t nnz) {
  if (rows <= 0 || nnz <= 0) {
    return 0;
  }
  return (static_cast<uint64_t>(rows) + 1ull) * sizeof(int32_t) +
         static_cast<uint64_t>(nnz) * sizeof(int32_t) +
         static_cast<uint64_t>(nnz) * sizeof(float);
}

namespace {

uint64_t estimatePinnedHostChunkBytes(const PinnedHostCSRData &host) {
  return host.rowPtr.capacity() * sizeof(int32_t) +
         host.colIdx.capacity() * sizeof(int32_t) +
         host.values.capacity() * sizeof(float);
}

void releasePreparedChunk(PreparedChunkedHostCSR &prepared,
                          size_t chunkIndex) {
  if (chunkIndex >= prepared.chunks.size() ||
      !prepared.ready[chunkIndex]) {
    return;
  }
  prepared.residentBytes -= prepared.chunkBytes[chunkIndex];
  prepared.chunkBytes[chunkIndex] = 0;
  prepared.touchStamps[chunkIndex] = 0;
  prepared.ready[chunkIndex] = 0;
  prepared.chunks[chunkIndex] = PinnedHostCSRData{};
}

bool evictPreparedChunkLru(PreparedChunkedHostCSR &prepared,
                           size_t protectedIndex = SIZE_MAX) {
  size_t victimIndex = SIZE_MAX;
  uint64_t victimStamp = std::numeric_limits<uint64_t>::max();
  for (size_t index = 0; index < prepared.chunks.size(); ++index) {
    if (!prepared.ready[index] || index == protectedIndex) {
      continue;
    }
    if (prepared.touchStamps[index] < victimStamp) {
      victimStamp = prepared.touchStamps[index];
      victimIndex = index;
    }
  }
  if (victimIndex == SIZE_MAX) {
    return false;
  }
  releasePreparedChunk(prepared, victimIndex);
  return true;
}

template <typename BuildChunkFn>
const PinnedHostCSRData &ensurePreparedChunkResident(
    PreparedChunkedHostCSR &prepared, size_t chunkIndex,
    BuildChunkFn &&buildChunk) {
  if (chunkIndex >= prepared.chunks.size()) {
    throw std::runtime_error(
        "CUDA prepared chunk resident index out of range.");
  }
  if (prepared.ready[chunkIndex]) {
    touchPreparedChunk(prepared, chunkIndex);
    return prepared.chunks[chunkIndex];
  }

  PinnedHostCSRData built;
  buildChunk(built);
  const uint64_t chunkBytes = estimatePinnedHostChunkBytes(built);

  if (prepared.residentBytes + chunkBytes <= prepared.maxResidentBytes) {
    // Fits within budget: cache permanently.
    prepared.chunks[chunkIndex] = std::move(built);
    prepared.chunkBytes[chunkIndex] = chunkBytes;
    prepared.ready[chunkIndex] = 1;
    prepared.residentBytes += chunkBytes;
    touchPreparedChunk(prepared, chunkIndex);
    return prepared.chunks[chunkIndex];
  }

  // Over budget: do NOT evict — sequential iteration would thrash the LRU.
  // Instead, evict only if the victim has NOT been touched in the current
  // iteration epoch (i.e. was only populated by warmup, not by recent use).
  // This prevents the pathological case where accessing chunks 0..N-1
  // sequentially evicts chunks that chunk N+1 needs.
  while (prepared.residentBytes > 0 &&
         prepared.residentBytes + chunkBytes > prepared.maxResidentBytes) {
    if (!evictPreparedChunkLru(prepared, chunkIndex)) {
      break;
    }
  }

  prepared.chunks[chunkIndex] = std::move(built);
  prepared.chunkBytes[chunkIndex] = chunkBytes;
  prepared.ready[chunkIndex] = 1;
  prepared.residentBytes += chunkBytes;
  touchPreparedChunk(prepared, chunkIndex);
  return prepared.chunks[chunkIndex];
}

template <typename BuildChunkFn>
void warmPreparedChunksUntilBudget(PreparedChunkedHostCSR &prepared,
                                   BuildChunkFn &&buildChunk) {
  for (size_t chunkIndex = 0; chunkIndex < prepared.chunks.size();
       ++chunkIndex) {
    if (prepared.ready[chunkIndex]) {
      continue;
    }

    PinnedHostCSRData built;
    buildChunk(chunkIndex, built);
    const uint64_t chunkBytes = estimatePinnedHostChunkBytes(built);
    if (prepared.residentBytes > 0 &&
        prepared.residentBytes + chunkBytes > prepared.maxResidentBytes) {
      break;
    }

    prepared.chunks[chunkIndex] = std::move(built);
    prepared.chunkBytes[chunkIndex] = chunkBytes;
    prepared.ready[chunkIndex] = 1;
    prepared.residentBytes += chunkBytes;
    touchPreparedChunk(prepared, chunkIndex);
  }
}

} // namespace

void touchPreparedChunk(PreparedChunkedHostCSR &prepared, size_t chunkIndex) {
  if (chunkIndex >= prepared.touchStamps.size()) {
    throw std::runtime_error("CUDA prepared chunk touch index out of range.");
  }
  prepared.touchStamps[chunkIndex] = prepared.nextTouchStamp++;
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

void buildAttenuatedContiguousSubsetCSR(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, int rowBegin, int rowEnd, int threads,
    PinnedHostCSRData &host) {
  if (numAngles <= 0) {
    throw std::runtime_error(
        "CUDA attenuation chunk builder requires numAngles > 0.");
  }
  const size_t voxels = static_cast<size_t>(matrix.cols);
  if (atten.size() != static_cast<size_t>(numAngles) * voxels) {
    throw std::runtime_error(
        "CUDA attenuation chunk builder received an attenuation table with invalid size.");
  }

  buildContiguousSubsetCSR(matrix, rowBegin, rowEnd, threads, host);
  if (host.nnz == 0) {
    return;
  }

  const size_t rows = static_cast<size_t>(rowEnd - rowBegin);
  const size_t raysPerAngle =
      static_cast<size_t>(matrix.rows) / static_cast<size_t>(numAngles);
  const float invScale = 1.0f / static_cast<float>(kAttenUint16Scale);

  for (size_t localRow = 0; localRow < rows; ++localRow) {
    int angle = 0;
    if (raysPerAngle > 0) {
      angle = static_cast<int>((static_cast<size_t>(rowBegin) + localRow) /
                               raysPerAngle);
    }
    if (angle >= numAngles) {
      angle = numAngles - 1;
    }
    const uint16_t *attPtr =
        atten.data() + static_cast<size_t>(angle) * voxels;
    const size_t r0 = static_cast<size_t>(host.rowPtr.data()[localRow]);
    const size_t r1 = static_cast<size_t>(host.rowPtr.data()[localRow + 1]);
    for (size_t index = r0; index < r1; ++index) {
      host.values.data()[index] *=
          static_cast<float>(
              attPtr[static_cast<size_t>(host.colIdx.data()[index])]) *
          invScale;
    }
  }
}

std::vector<RowChunkRange> computeContiguousRowChunks(
    const io::CSRView &matrix, uint64_t nnzBudget) {
  if (!matrix.valid()) {
    throw std::runtime_error(
        "Invalid io::CSRView passed to CUDA chunked sparse backend.");
  }
  std::vector<RowChunkRange> chunks;
  if (matrix.rows <= 0) {
    return chunks;
  }

  const uint64_t effectiveBudget =
      std::max<uint64_t>(1ull, std::min<uint64_t>(
                                   nnzBudget,
                                   static_cast<uint64_t>(
                                       std::numeric_limits<int32_t>::max())));

  int rowBegin = 0;
  while (rowBegin < matrix.rows) {
    const uint64_t baseOffset =
        static_cast<uint64_t>(matrix.row_ptr[static_cast<size_t>(rowBegin)]);
    int rowEnd = rowBegin + 1;
    while (rowEnd <= matrix.rows) {
      const uint64_t endOffset =
          static_cast<uint64_t>(matrix.row_ptr[static_cast<size_t>(rowEnd)]);
      const uint64_t chunkNnz = endOffset - baseOffset;
      if (chunkNnz > effectiveBudget) {
        if (rowEnd == rowBegin + 1) {
          throw std::runtime_error(
              "CUDA chunked sparse backend encountered a row chunk larger than the int32/budget limit.");
        }
        --rowEnd;
        break;
      }
      if (rowEnd == matrix.rows) {
        break;
      }
      ++rowEnd;
    }
    if (rowEnd <= rowBegin) {
      throw std::runtime_error(
          "CUDA chunked sparse backend failed to make forward progress.");
    }
    chunks.push_back(RowChunkRange{rowBegin, rowEnd});
    rowBegin = rowEnd;
  }

  return chunks;
}

const std::vector<RowChunkRange> &getContiguousRowChunks(
    const io::CSRView &matrix, uint64_t nnzBudget) {
  static std::mutex cacheMutex;
  static std::deque<
      std::pair<ChunkRangeCacheKey, std::vector<RowChunkRange>>>
      cache;

  const ChunkRangeCacheKey key = makeChunkRangeCacheKey(matrix, nnzBudget);
  std::lock_guard<std::mutex> lock(cacheMutex);
  for (auto &entry : cache) {
    if (entry.first == key) {
      return entry.second;
    }
  }
  if (cache.size() >= 4) {
    cache.erase(cache.begin());
  }
  cache.emplace_back(key, computeContiguousRowChunks(matrix, nnzBudget));
  return cache.back().second;
}

void initializePreparedChunkedHostCacheEntry(
    PreparedChunkedHostCSR &prepared,
    const std::vector<RowChunkRange> &ranges,
    uint64_t maxResidentBytes) {
  if (!prepared.ranges.empty()) {
    return;
  }
  prepared.ranges = ranges;
  prepared.chunks.resize(ranges.size());
  prepared.chunkBytes.assign(ranges.size(), 0);
  prepared.touchStamps.assign(ranges.size(), 0);
  prepared.ready.assign(ranges.size(), 0);
  prepared.residentBytes = 0;
  prepared.maxResidentBytes = maxResidentBytes;
  prepared.nextTouchStamp = 1;
}

PreparedChunkedHostCSR *getPreparedContiguousHostChunks(
    const io::CSRView &matrix, uint64_t nnzBudget, int threads,
    bool eagerPrepare, uint64_t hostCacheBudgetBytes) {
  const uint64_t effectiveHostCacheBudgetBytes =
      hostCacheBudgetBytes > 0 ? hostCacheBudgetBytes
                               : selectPreparedChunkHostCacheBudgetBytes(matrix);
  if (effectiveHostCacheBudgetBytes == 0) {
    return nullptr;
  }
  static std::mutex cacheMutex;
  static std::deque<PreparedChunkedHostCSRCacheEntry> cache;

  const PreparedChunkRangeCacheKey key =
      makePreparedChunkRangeCacheKey(matrix, nnzBudget,
                                     effectiveHostCacheBudgetBytes);
  std::lock_guard<std::mutex> lock(cacheMutex);
  for (auto &entry : cache) {
    if (entry.key == key) {
      if (eagerPrepare) {
        warmPreparedChunksUntilBudget(
            entry.prepared, [&](size_t chunkIndex, PinnedHostCSRData &host) {
              const RowChunkRange &range = entry.prepared.ranges[chunkIndex];
              buildContiguousSubsetCSR(matrix, range.rowBegin, range.rowEnd,
                                       threads, host);
            });
      }
      return &entry.prepared;
    }
  }
  if (cache.size() >= 2) {
    cache.erase(cache.begin());
  }
  PreparedChunkedHostCSRCacheEntry entry;
  entry.key = key;
  initializePreparedChunkedHostCacheEntry(entry.prepared,
                                          getContiguousRowChunks(matrix,
                                                                 nnzBudget),
                                          effectiveHostCacheBudgetBytes);
  if (eagerPrepare) {
    warmPreparedChunksUntilBudget(
        entry.prepared, [&](size_t chunkIndex, PinnedHostCSRData &host) {
          const RowChunkRange &range = entry.prepared.ranges[chunkIndex];
          buildContiguousSubsetCSR(matrix, range.rowBegin, range.rowEnd, threads,
                                   host);
        });
  }
  cache.push_back(std::move(entry));
  return &cache.back().prepared;
}

const PinnedHostCSRData &ensurePreparedContiguousHostChunk(
    PreparedChunkedHostCSR &prepared, const io::CSRView &matrix,
    size_t chunkIndex, int threads) {
  return ensurePreparedChunkResident(
      prepared, chunkIndex, [&](PinnedHostCSRData &host) {
        const RowChunkRange &range = prepared.ranges[chunkIndex];
        const auto buildStart = std::chrono::steady_clock::now();
        buildContiguousSubsetCSR(matrix, range.rowBegin, range.rowEnd, threads,
                                 host);
        noteCudaChunkBuild(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          buildStart)
                .count());
      });
}

PreparedChunkedHostCSR *getPreparedAttenuatedHostChunks(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, uint64_t nnzBudget, int threads, bool eagerPrepare,
    uint64_t hostCacheBudgetBytes) {
  const uint64_t effectiveHostCacheBudgetBytes =
      hostCacheBudgetBytes > 0 ? hostCacheBudgetBytes
                               : selectPreparedChunkHostCacheBudgetBytes(matrix);
  if (effectiveHostCacheBudgetBytes == 0) {
    return nullptr;
  }
  static std::mutex cacheMutex;
  static std::deque<PreparedAttenChunkedHostCSRCacheEntry> cache;

  const PreparedAttenChunkRangeCacheKey key =
      makePreparedAttenChunkRangeCacheKey(matrix, atten, numAngles, nnzBudget,
                                          effectiveHostCacheBudgetBytes);
  std::lock_guard<std::mutex> lock(cacheMutex);
  for (auto &entry : cache) {
    if (entry.key == key) {
      if (eagerPrepare) {
        warmPreparedChunksUntilBudget(
            entry.prepared, [&](size_t chunkIndex, PinnedHostCSRData &host) {
              const RowChunkRange &range = entry.prepared.ranges[chunkIndex];
              buildAttenuatedContiguousSubsetCSR(matrix, atten, numAngles,
                                                 range.rowBegin, range.rowEnd,
                                                 threads, host);
            });
      }
      return &entry.prepared;
    }
  }
  if (cache.size() >= 2) {
    cache.erase(cache.begin());
  }
  PreparedAttenChunkedHostCSRCacheEntry entry;
  entry.key = key;
  initializePreparedChunkedHostCacheEntry(entry.prepared,
                                          getContiguousRowChunks(matrix,
                                                                 nnzBudget),
                                          effectiveHostCacheBudgetBytes);
  if (eagerPrepare) {
    warmPreparedChunksUntilBudget(
        entry.prepared, [&](size_t chunkIndex, PinnedHostCSRData &host) {
          const RowChunkRange &range = entry.prepared.ranges[chunkIndex];
          buildAttenuatedContiguousSubsetCSR(matrix, atten, numAngles,
                                             range.rowBegin, range.rowEnd,
                                             threads, host);
        });
  }
  cache.push_back(std::move(entry));
  return &cache.back().prepared;
}

const PinnedHostCSRData &ensurePreparedAttenuatedHostChunk(
    PreparedChunkedHostCSR &prepared, const io::CSRView &matrix,
    const std::vector<uint16_t> &atten, int numAngles, size_t chunkIndex,
    int threads) {
  return ensurePreparedChunkResident(
      prepared, chunkIndex, [&](PinnedHostCSRData &host) {
        const RowChunkRange &range = prepared.ranges[chunkIndex];
        const auto buildStart = std::chrono::steady_clock::now();
        buildAttenuatedContiguousSubsetCSR(matrix, atten, numAngles,
                                           range.rowBegin, range.rowEnd, threads,
                                           host);
        noteCudaChunkBuild(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          buildStart)
                .count());
      });
}

} // namespace spect

#endif