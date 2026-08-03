#include "CudaCsrBuilder.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include "core/AttenuationConstants.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace spect {

std::atomic<int> g_cudaHostChunkBuildThreadsOverride{0};

namespace {

unsigned int detectPhysicalCoreCount() {
  static const unsigned int cached = []() -> unsigned int {
#if defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && length > 0) {
      std::vector<unsigned char> buffer(length);
      auto *info =
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
              buffer.data());
      if (GetLogicalProcessorInformationEx(RelationProcessorCore, info,
                                           &length)) {
        unsigned int cores = 0;
        unsigned char *cursor = buffer.data();
        const unsigned char *end = buffer.data() + length;
        while (cursor < end) {
          const auto *entry =
              reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
                  cursor);
          if (entry->Relationship == RelationProcessorCore) {
            ++cores;
          }
          cursor += entry->Size;
        }
        if (cores > 0) {
          return cores;
        }
      }
    }
#endif
#if defined(_OPENMP)
    return static_cast<unsigned int>(std::max(1, omp_get_num_procs()));
#else
    return std::max(1u, std::thread::hardware_concurrency());
#endif
  }();
  return cached;
}

int resolveHostChunkBuildThreadBudget(int requestedThreads) {
  const int override =
      g_cudaHostChunkBuildThreadsOverride.load(std::memory_order_relaxed);
  if (override > 0) {
    return std::max(1, override);
  }
  const int safeRequested = std::max(1, requestedThreads);
  const int physicalCores =
      std::max(1, static_cast<int>(detectPhysicalCoreCount()));
  const int headroom = physicalCores >= 12 ? 2 : 1;
  return std::max(1, std::min(safeRequested, physicalCores - headroom));
}

} // namespace

int getHostParallelThreads(int requestedThreads, uint64_t workItems) {
  if (requestedThreads <= 1 || workItems < 262144ull) {
    return 1;
  }
#if defined(_OPENMP)
  return resolveHostChunkBuildThreadBudget(requestedThreads);
#else
  (void)requestedThreads;
  return 1;
#endif
}

HostCSRData convertCSR(const io::CSRView &matrix) {
  if (!matrix.valid()) {
    throw std::runtime_error("Invalid io::CSRView passed to CUDA sparse backend.");
  }

  HostCSRData host;
  host.rows = matrix.rows;
  host.cols = matrix.cols;
  host.nnz = static_cast<int64_t>(matrix.nnz);

  host.rowPtr.resize(static_cast<size_t>(matrix.rows) + 1);
  for (int row = 0; row <= matrix.rows; ++row) {
    const uint32_t offset = matrix.row_ptr[static_cast<size_t>(row)];
    if (offset > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error(
          "CUDA sparse backend only supports row offsets representable as int32.");
    }
    host.rowPtr[static_cast<size_t>(row)] = static_cast<int32_t>(offset);
  }

  host.colIdx.resize(static_cast<size_t>(matrix.nnz));
  host.values.resize(static_cast<size_t>(matrix.nnz));
  for (size_t index = 0; index < matrix.nnz; ++index) {
    const uint32_t col = matrix.col_idx[index];
    if (col > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error(
          "CUDA sparse backend only supports column indices representable as int32.");
    }
    host.colIdx[index] = static_cast<int32_t>(col);
    host.values[index] = matrix.val[index];
  }

  return host;
}

void buildContiguousSubsetCSR(const io::CSRView &matrix, int rowBegin,
                              int rowEnd, int threads, HostCSRData &host) {
  if (rowBegin < 0 || rowEnd < rowBegin || rowEnd > matrix.rows) {
    throw std::runtime_error(
        "CUDA subset sparse backend received an out-of-range contiguous row block.");
  }

  host.rows = static_cast<int64_t>(rowEnd - rowBegin);
  host.cols = matrix.cols;
  host.rowPtr.clear();
  host.colIdx.clear();
  host.values.clear();
  if (rowBegin == rowEnd) {
    host.rowPtr = {0};
    host.nnz = 0;
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

  host.rowPtr.resize(static_cast<size_t>(rowEnd - rowBegin) + 1);
  for (int row = rowBegin; row <= rowEnd; ++row) {
    host.rowPtr[static_cast<size_t>(row - rowBegin)] = static_cast<int32_t>(
        matrix.row_ptr[static_cast<size_t>(row)] - baseOffset);
  }

  host.colIdx.resize(static_cast<size_t>(nnz));
  host.values.resize(static_cast<size_t>(nnz));
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
      host.colIdx[static_cast<size_t>(offset)] = 0;
      host.values[static_cast<size_t>(offset)] = 0.0f;
      continue;
    }
    host.colIdx[static_cast<size_t>(offset)] = static_cast<int32_t>(col);
    host.values[static_cast<size_t>(offset)] = matrix.val[sourceIndex];
  }
  if (invalidColumn.load(std::memory_order_relaxed)) {
    throw std::runtime_error(
        "CUDA sparse backend only supports column indices representable as int32.");
  }

  host.nnz = static_cast<int64_t>(nnz);
}

void buildAttenuatedContiguousSubsetCSR(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, int rowBegin, int rowEnd, int threads, HostCSRData &host) {
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
    const size_t r0 = static_cast<size_t>(host.rowPtr[localRow]);
    const size_t r1 = static_cast<size_t>(host.rowPtr[localRow + 1]);
    for (size_t index = r0; index < r1; ++index) {
      host.values[index] *=
          static_cast<float>(attPtr[static_cast<size_t>(host.colIdx[index])]) *
          invScale;
    }
  }
}

void buildSubsetCSR(const io::CSRView &matrix,
                    const std::vector<int> &subset_rows, int threads,
                    HostCSRData &host) {
  if (!matrix.valid()) {
    throw std::runtime_error(
        "Invalid io::CSRView passed to CUDA subset sparse backend.");
  }

  host.rows = static_cast<int64_t>(subset_rows.size());
  host.cols = matrix.cols;
  host.rowPtr.clear();
  host.colIdx.clear();
  host.values.clear();
  if (subset_rows.empty()) {
    host.rowPtr = {0};
    host.nnz = 0;
    return;
  }

  auto isContiguousAscending = [&]() -> bool {
    const int firstRow = subset_rows.front();
    for (size_t index = 0; index < subset_rows.size(); ++index) {
      if (subset_rows[index] != firstRow + static_cast<int>(index)) {
        return false;
      }
    }
    return true;
  };

  if (isContiguousAscending()) {
    const int rowBegin = subset_rows.front();
    const int rowEnd = rowBegin + static_cast<int>(subset_rows.size());
    buildContiguousSubsetCSR(matrix, rowBegin, rowEnd, threads, host);
    return;
  }

  uint64_t totalNnz = 0;
  for (size_t index = 0; index < subset_rows.size(); ++index) {
    const int row = subset_rows[index];
    if (row < 0 || row >= matrix.rows) {
      throw std::runtime_error(
          "CUDA subset sparse backend received an out-of-range row index.");
    }
    const uint32_t r0 = matrix.row_ptr[static_cast<size_t>(row)];
    const uint32_t r1 = matrix.row_ptr[static_cast<size_t>(row) + 1];
    totalNnz += static_cast<uint64_t>(r1) - static_cast<uint64_t>(r0);
  }
  if (totalNnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(
        "CUDA subset sparse backend only supports subset nnz representable as int32.");
  }

  host.rowPtr.reserve(subset_rows.size() + 1);
  host.colIdx.reserve(static_cast<size_t>(totalNnz));
  host.values.reserve(static_cast<size_t>(totalNnz));
  host.rowPtr.push_back(0);

  for (size_t index = 0; index < subset_rows.size(); ++index) {
    const int row = subset_rows[index];
    const uint32_t r0 = matrix.row_ptr[static_cast<size_t>(row)];
    const uint32_t r1 = matrix.row_ptr[static_cast<size_t>(row) + 1];
    for (uint32_t t = r0; t < r1; ++t) {
      const uint32_t col = matrix.col_idx[static_cast<size_t>(t)];
      if (col > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(
            "CUDA sparse backend only supports column indices representable as int32.");
      }
      host.colIdx.push_back(static_cast<int32_t>(col));
      host.values.push_back(matrix.val[static_cast<size_t>(t)]);
    }

    const size_t nnz = host.colIdx.size();
    if (nnz > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error(
          "CUDA subset sparse backend only supports subset nnz representable as int32.");
    }
    host.rowPtr.push_back(static_cast<int32_t>(nnz));
  }

  host.nnz = static_cast<int64_t>(host.colIdx.size());
}

void buildAttenuatedCSR(const io::CSRView &matrix,
                        const std::vector<uint16_t> &atten, int numAngles,
                        int threads, HostCSRData &host) {
  if (!matrix.valid()) {
    throw std::runtime_error(
        "Invalid io::CSRView passed to CUDA attenuation sparse backend.");
  }
  if (numAngles <= 0) {
    throw std::runtime_error(
        "CUDA attenuation sparse backend requires numAngles > 0.");
  }

  host = convertCSR(matrix);
  const size_t voxels = static_cast<size_t>(matrix.cols);
  if (atten.size() != static_cast<size_t>(numAngles) * voxels) {
    throw std::runtime_error(
        "CUDA attenuation sparse backend received an attenuation table with invalid size.");
  }

  const size_t raysPerAngle = static_cast<size_t>(matrix.rows) /
                              static_cast<size_t>(numAngles);
  const float invScale = 1.0f / static_cast<float>(kAttenUint16Scale);
  const int parallelThreads =
      getHostParallelThreads(threads, static_cast<uint64_t>(matrix.rows));
#pragma omp parallel for schedule(static) if(parallelThreads > 1) num_threads(parallelThreads)
  for (int row = 0; row < matrix.rows; ++row) {
    int angle = 0;
    if (raysPerAngle > 0) {
      angle = static_cast<int>(static_cast<size_t>(row) / raysPerAngle);
    }
    if (angle >= numAngles) {
      angle = numAngles - 1;
    }
    const uint16_t *attPtr =
        atten.data() + static_cast<size_t>(angle) * voxels;
    const size_t r0 = static_cast<size_t>(matrix.row_ptr[row]);
    const size_t r1 = static_cast<size_t>(matrix.row_ptr[row + 1]);
    for (size_t t = r0; t < r1; ++t) {
      host.values[t] *= static_cast<float>(attPtr[matrix.col_idx[t]]) *
                        invScale;
    }
  }
}

} // namespace spect

#endif