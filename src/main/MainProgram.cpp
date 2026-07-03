#include "main/MainProgram.h"

#include "app/ADMMOptimizer.h"
#include "app/BedOffsets.h"
#include "app/BSREMOptimizer.h"
#include "app/BSREMReconstructor.h"
#include "app/IOptimizer.h"
#include "app/MLEMOptimizer.h"
#include "app/OSEMReconstructor.h"
#include "app/ProjectionOps.h"
#include "app/ReconMetrics.h"
#include "cli/ArgumentParser.h"
#include "cli/ExitCodes.h"
#include "core/Masks.h"
#include "core/ReconLimits.h"
#include "cuda/CompactSymmetricProjector.h"
#include "cuda/VolumeOpsCUDA.h"
#include "cuda/ProjDomainChunked.h"
#include "filters/ButterworthFilter.h"
#include "filters/Convolution.h"
#include "io/CSRBuilder.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "io/MHDWriter.h"
#include "io/ProjectionLoader.h"
#include "io/VolumeLoader.h"
#include "main/AutoMuMap.h"
#include "main/CTACOps.h"
#include "main/MatVarPath.h"
#include "main/ReconDataLoader.h"
#include "main/ReconArgsPrinter.h"
#include "main/ReconPreparation.h"
#include "main/ReconSetup.h"
#include "main/RunLogging.h"
#include "main/RunLogger.h"
#include "main/SmoothingOps.h"
#include "main/VolumeOutput.h"
#include "regularizer/RegularizerCompat.h"
#include "sparse/SparseOpsBackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
#include <cuda_runtime.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spect::main {
namespace {

using ProfileClock = std::chrono::steady_clock;

struct ReconstructionStageTiming {
  double regGradientSeconds = 0.0;
  double subsetSensitivitySeconds = 0.0;
  double cudaChunkWarmupSeconds = 0.0;
  double osemCoreSeconds = 0.0;
  double bsremCoreSeconds = 0.0;
  double butterworthSeconds = 0.0;
  double smoothingSeconds = 0.0;
  double saveSeconds = 0.0;
  size_t regGradientCalls = 0;
  size_t subsetSensitivityCalls = 0;
  size_t cudaChunkWarmupCalls = 0;
  size_t osemCoreCalls = 0;
  size_t bsremCoreCalls = 0;
  size_t butterworthCalls = 0;
  size_t smoothingCalls = 0;
  size_t saveCalls = 0;
};

struct CompactExecutionTiming {
  double projectionLoadSeconds = 0.0;
  double csrMapOrLoadSeconds = 0.0;
  double metadataLoadAndValidateSeconds = 0.0;
  double psfPrepareSeconds = 0.0;
  double compactSensitivitySeconds = 0.0;
  double compactForwardRatioBackwardSeconds = 0.0;
  double regGradientSeconds = 0.0;
  size_t regGradientCalls = 0;
  double mlemUpdateSeconds = 0.0;
  double butterworthSeconds = 0.0;
  double outputWriteSeconds = 0.0;
  double totalLoadToIterationDoneSeconds = 0.0;
  double fullCommandWallSeconds = 0.0;
};

struct MemoryUsageSnapshot {
  bool hostAvailable = false;
  uint64_t hostWorkingSetBytes = 0;
  uint64_t hostPeakWorkingSetBytes = 0;
  bool cudaAvailable = false;
  uint64_t cudaUsedBytes = 0;
  uint64_t cudaFreeBytes = 0;
  uint64_t cudaTotalBytes = 0;
};

MemoryUsageSnapshot queryMemoryUsageSnapshot() {
  MemoryUsageSnapshot snapshot;
#ifdef _WIN32
  using K32GetProcessMemoryInfoFn =
      BOOL(WINAPI *)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
  HMODULE kernel32 = GetModuleHandleA("kernel32.dll");
  auto *getProcessMemoryInfo =
      kernel32 ? reinterpret_cast<K32GetProcessMemoryInfoFn>(
                     GetProcAddress(kernel32, "K32GetProcessMemoryInfo"))
               : nullptr;
  if (getProcessMemoryInfo) {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (getProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&counters),
            sizeof(counters))) {
      snapshot.hostAvailable = true;
      snapshot.hostWorkingSetBytes =
          static_cast<uint64_t>(counters.WorkingSetSize);
      snapshot.hostPeakWorkingSetBytes =
          static_cast<uint64_t>(counters.PeakWorkingSetSize);
    }
  }
#endif
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess &&
      totalBytes >= freeBytes) {
    snapshot.cudaAvailable = true;
    snapshot.cudaFreeBytes = static_cast<uint64_t>(freeBytes);
    snapshot.cudaTotalBytes = static_cast<uint64_t>(totalBytes);
    snapshot.cudaUsedBytes = static_cast<uint64_t>(totalBytes - freeBytes);
  }
#endif
  return snapshot;
}

void mergeMemoryPeak(MemoryUsageSnapshot &peak,
                     const MemoryUsageSnapshot &sample) {
  if (sample.hostAvailable) {
    peak.hostAvailable = true;
    peak.hostWorkingSetBytes =
        std::max(peak.hostWorkingSetBytes, sample.hostWorkingSetBytes);
    peak.hostPeakWorkingSetBytes =
        std::max(peak.hostPeakWorkingSetBytes,
                 sample.hostPeakWorkingSetBytes);
  }
  if (sample.cudaAvailable) {
    peak.cudaAvailable = true;
    peak.cudaUsedBytes = std::max(peak.cudaUsedBytes, sample.cudaUsedBytes);
    peak.cudaFreeBytes = sample.cudaFreeBytes;
    peak.cudaTotalBytes = std::max(peak.cudaTotalBytes, sample.cudaTotalBytes);
  }
}

std::string formatMemoryEntry(const std::string &label,
                              const MemoryUsageSnapshot &snapshot) {
  std::ostringstream os;
  os << "[MEM] " << label;
  if (snapshot.hostAvailable) {
    os << " host_working_set_peak_bytes=" << snapshot.hostWorkingSetBytes
       << " host_process_peak_working_set_bytes="
       << snapshot.hostPeakWorkingSetBytes;
  } else {
    os << " host_unavailable=1";
  }
  if (snapshot.cudaAvailable) {
    os << " cuda_used_peak_bytes=" << snapshot.cudaUsedBytes
       << " cuda_total_bytes=" << snapshot.cudaTotalBytes
       << " cuda_free_bytes_last=" << snapshot.cudaFreeBytes;
  } else {
    os << " cuda_unavailable=1";
  }
  return os.str();
}

struct ProjDomainChunkCacheKey {
  const void *rowPtr = nullptr;
  const void *colIdx = nullptr;
  const void *values = nullptr;
  int rows = 0;
  int cols = 0;
  uint64_t nnz = 0;
  uint64_t frameSize = 0;
  uint64_t chunkNnzBudget = 0;

  bool operator==(const ProjDomainChunkCacheKey &other) const {
    return rowPtr == other.rowPtr && colIdx == other.colIdx &&
           values == other.values && rows == other.rows && cols == other.cols &&
           nnz == other.nnz && frameSize == other.frameSize &&
           chunkNnzBudget == other.chunkNnzBudget;
  }
};

inline bool canUseFrameAlignedProjDomainChunks(size_t frameSize, int rows) {
  return frameSize > 0 &&
         static_cast<size_t>(rows) % frameSize == 0;
}

template <typename MatrixType>
bool canUseProjDomainCudaChunking(const MatrixType &, const ReconOpts &) {
  return false;
}

inline bool canUseProjDomainCudaChunking(const io::CSRView &matrix,
                                         const ReconOpts &opts) {
  return opts.sparseOps != nullptr &&
         opts.sparseOps->backend() == BackendType::CUDA &&
         matrix.nnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
}

template <typename MatrixType>
std::vector<std::vector<int>> buildProjDomainCudaChunks(const MatrixType &matrix,
                                                        size_t frameSize,
                                                        uint64_t chunkNnzBudget) {
  const uint64_t kMaxChunkNnz = std::clamp<uint64_t>(
      chunkNnzBudget, 8ull * 1024ull * 1024ull,
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
  const int rows = static_cast<int>(matrix.rows);
  std::vector<std::vector<int>> chunks;
  chunks.reserve(4);

  auto rowOffset = [&](int row) -> uint64_t {
    return static_cast<uint64_t>(matrix.row_ptr[static_cast<size_t>(row)]);
  };

  auto appendChunk = [&](int rowStart, int rowEnd) {
    std::vector<int> chunk(static_cast<size_t>(rowEnd - rowStart));
    std::iota(chunk.begin(), chunk.end(), rowStart);
    chunks.push_back(std::move(chunk));
  };

  if (canUseFrameAlignedProjDomainChunks(frameSize, rows)) {
    const int frameRows = static_cast<int>(frameSize);
    int rowStart = 0;
    while (rowStart < rows) {
      const uint64_t startOffset = rowOffset(rowStart);
      int rowEnd = rowStart;
      while (rowEnd < rows) {
        const int nextEnd = std::min(rows, rowEnd + frameRows);
        const uint64_t endOffset = rowOffset(nextEnd);
        if (endOffset < startOffset) {
          throw std::runtime_error(
              "Proj-domain MLEM CUDA chunking encountered non-monotonic CSR row offsets.");
        }
        if (endOffset - startOffset > kMaxChunkNnz) {
          break;
        }
        rowEnd = nextEnd;
      }

      if (rowEnd == rowStart) {
        throw std::runtime_error(
            "Proj-domain MLEM CUDA chunking found a single detector frame whose nnz exceeds the int32 subset limit.");
      }

      appendChunk(rowStart, rowEnd);
      rowStart = rowEnd;
    }
    return chunks;
  }

  int rowStart = 0;
  while (rowStart < rows) {
    const uint64_t startOffset = rowOffset(rowStart);
    int rowEnd = rowStart;
    while (rowEnd < rows) {
      const uint64_t endOffset = rowOffset(rowEnd + 1);
      if (endOffset < startOffset) {
        throw std::runtime_error(
            "Proj-domain MLEM CUDA chunking encountered non-monotonic CSR row offsets.");
      }
      if (endOffset - startOffset > kMaxChunkNnz) {
        break;
      }
      ++rowEnd;
    }

    if (rowEnd == rowStart) {
      throw std::runtime_error(
          "Proj-domain MLEM CUDA chunking found a single row whose nnz exceeds the int32 subset limit.");
    }

    appendChunk(rowStart, rowEnd);
    rowStart = rowEnd;
  }

  return chunks;
}

inline const std::vector<std::vector<int>> &getProjDomainCudaChunks(
    const io::CSRView &matrix, size_t frameSize, uint64_t chunkNnzBudget) {
  static std::mutex cacheMutex;
  static std::vector<std::pair<ProjDomainChunkCacheKey,
                               std::vector<std::vector<int>>>>
      cache;

  const ProjDomainChunkCacheKey key{matrix.row_ptr, matrix.col_idx, matrix.val,
                                    matrix.rows, matrix.cols, matrix.nnz,
                                    static_cast<uint64_t>(frameSize),
                                    chunkNnzBudget};
  std::lock_guard<std::mutex> lock(cacheMutex);
  for (auto &entry : cache) {
    if (entry.first == key) {
      return entry.second;
    }
  }

  if (cache.size() >= 4) {
    cache.erase(cache.begin());
  }
  cache.emplace_back(key,
                     buildProjDomainCudaChunks(matrix, frameSize,
                                               chunkNnzBudget));
  return cache.back().second;
}

inline void runProjDomainChunkedForward(const io::CSRView &matrix,
                                        const ReconOpts &opts,
                                        size_t frameSize,
                                        uint64_t chunkNnzBudget,
                                        const double *x, double *y) {
  const SparseSubsetContext chunkCtx{};
  const auto &chunks = getProjDomainCudaChunks(matrix, frameSize,
                                               chunkNnzBudget);
  for (const auto &chunkRows : chunks) {
    std::vector<double> local(static_cast<size_t>(chunkRows.size()), 0.0);
    opts.sparseOps->spmvSubset(matrix, x, chunkRows, chunkCtx, local.data(),
                               opts.threads);
    std::copy(local.begin(), local.end(),
              y + static_cast<size_t>(chunkRows.front()));
  }
}

inline void runProjDomainChunkedTranspose(const io::CSRView &matrix,
                                          const ReconOpts &opts,
                                          size_t frameSize,
                                          uint64_t chunkNnzBudget,
                                          const double *x, double *y) {
  const SparseSubsetContext chunkCtx{};
  const auto &chunks = getProjDomainCudaChunks(matrix, frameSize,
                                               chunkNnzBudget);
  std::vector<double> local(static_cast<size_t>(matrix.cols), 0.0);
  for (const auto &chunkRows : chunks) {
    const size_t chunkSize = static_cast<size_t>(chunkRows.size());
    std::vector<double> localX(chunkSize, 0.0);
    std::copy_n(x + static_cast<size_t>(chunkRows.front()), chunkSize,
                localX.data());
    std::fill(local.begin(), local.end(), 0.0);
    opts.sparseOps->spmvTSubset(matrix, localX.data(), chunkRows, chunkCtx,
                                local.data(), opts.threads);
    for (size_t index = 0; index < local.size(); ++index) {
      y[index] += local[index];
    }
  }
}

std::string formatProfileEntry(const std::string &label, double seconds,
                               size_t calls, uint64_t bytes = 0) {
  std::ostringstream oss;
  oss << "[PROFILE] " << label << ": total=" << std::fixed
      << std::setprecision(3) << seconds << "s";
  if (calls > 0)
    oss << ", calls=" << calls << ", avg=" << (seconds / static_cast<double>(calls))
        << "s";
  if (bytes > 0)
    oss << ", bytes=" << bytes;
  return oss.str();
}

void logCudaChunkTimingSnapshot(
    const spect::cuda::CudaChunkStageTimingSnapshot &cudaChunkTiming) {
  logInfo(formatProfileEntry("cuda.chunk_build", cudaChunkTiming.chunkBuildSeconds,
                             cudaChunkTiming.chunkBuildCalls));
  logInfo(formatProfileEntry("cuda.pinned_stage",
                             cudaChunkTiming.pinnedStageSeconds,
                             cudaChunkTiming.pinnedStageCalls));
  logInfo(formatProfileEntry("cuda.h2d_chunk", cudaChunkTiming.h2dChunkSeconds,
                             cudaChunkTiming.h2dChunkCalls));
  logInfo(formatProfileEntry("cuda.upload_wait",
                             cudaChunkTiming.uploadWaitOnComputeSeconds,
                             cudaChunkTiming.uploadWaitOnComputeCalls));
  logInfo(formatProfileEntry("cuda.fused_bed_cycle",
                             cudaChunkTiming.fusedBedCycleSeconds,
                             cudaChunkTiming.fusedBedCycleCalls));
  logInfo(formatProfileEntry("cuda.cusparse_fwd",
                             cudaChunkTiming.cusparseForwardSeconds,
                             cudaChunkTiming.cusparseForwardCalls));
  logInfo(formatProfileEntry("cuda.cusparse_fwd_host",
                             cudaChunkTiming.cusparseForwardHostSeconds,
                             cudaChunkTiming.cusparseForwardHostCalls));
  logInfo(formatProfileEntry("cuda.ratio", cudaChunkTiming.ratioSeconds,
                             cudaChunkTiming.ratioCalls));
  logInfo(formatProfileEntry("cuda.transpose_chunk",
                             cudaChunkTiming.transposeChunkSeconds,
                             cudaChunkTiming.transposeChunkCalls));
  logInfo(formatProfileEntry("cuda.cusparse_bwd",
                             cudaChunkTiming.cusparseBackwardSeconds,
                             cudaChunkTiming.cusparseBackwardCalls));
  logInfo(formatProfileEntry("cuda.cusparse_bwd_host",
                             cudaChunkTiming.cusparseBackwardHostSeconds,
                             cudaChunkTiming.cusparseBackwardHostCalls));
  if (cudaChunkTiming.compactMatrixUploadCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_matrix_upload",
                               cudaChunkTiming.compactMatrixUploadSeconds,
                               cudaChunkTiming.compactMatrixUploadCalls));
  }
  if (cudaChunkTiming.compactMatrixReuseCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_matrix_reuse_hit", 0.0,
                               cudaChunkTiming.compactMatrixReuseCalls));
  }
  if (cudaChunkTiming.compactInputH2DCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_input_h2d",
                               cudaChunkTiming.compactInputH2DSeconds,
                               cudaChunkTiming.compactInputH2DCalls));
  }
  if (cudaChunkTiming.compactProjectionH2DCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_projection_h2d",
                               cudaChunkTiming.compactProjectionH2DSeconds,
                               cudaChunkTiming.compactProjectionH2DCalls,
                               cudaChunkTiming.compactProjectionH2DBytes));
  }
  if (cudaChunkTiming.compactOutputD2HCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_output_d2h",
                               cudaChunkTiming.compactOutputD2HSeconds,
                               cudaChunkTiming.compactOutputD2HCalls));
  }
  if (cudaChunkTiming.compactForwardOutputD2HCalls > 0) {
    logInfo(formatProfileEntry(
        "cuda.compact_forward_output_d2h",
        cudaChunkTiming.compactForwardOutputD2HSeconds,
        cudaChunkTiming.compactForwardOutputD2HCalls,
        cudaChunkTiming.compactForwardOutputD2HBytes));
  }
  if (cudaChunkTiming.compactBackwardOutputD2HCalls > 0) {
    logInfo(formatProfileEntry(
        "cuda.compact_backward_output_d2h",
        cudaChunkTiming.compactBackwardOutputD2HSeconds,
        cudaChunkTiming.compactBackwardOutputD2HCalls,
        cudaChunkTiming.compactBackwardOutputD2HBytes));
  }
  if (cudaChunkTiming.deviceCacheHitCalls > 0) {
    logInfo(formatProfileEntry("cuda.device_cache_hit", 0.0,
                               cudaChunkTiming.deviceCacheHitCalls));
  }
}

struct VectorStats {
  size_t count = 0;
  size_t finite = 0;
  size_t nonzero = 0;
  double min = 0.0;
  double max = 0.0;
  double sum = 0.0;
};

VectorStats computeVectorStats(const std::vector<double> &values) {
  VectorStats stats;
  stats.count = values.size();
  bool initialized = false;
  for (const double value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    ++stats.finite;
    if (value != 0.0) {
      ++stats.nonzero;
    }
    if (!initialized) {
      stats.min = value;
      stats.max = value;
      initialized = true;
    } else {
      stats.min = std::min(stats.min, value);
      stats.max = std::max(stats.max, value);
    }
    stats.sum += value;
  }
  return stats;
}

std::string formatStatsEntry(const std::string &label,
                             const VectorStats &stats) {
  std::ostringstream oss;
  oss << "[STATS] " << label << ": count=" << stats.count
      << ", finite=" << stats.finite << ", nonzero=" << stats.nonzero
      << ", min=" << std::setprecision(12) << stats.min
      << ", max=" << stats.max << ", sum=" << stats.sum;
  return oss.str();
}

std::string outputPrefixFromArgs(const ReconArgs &args) {
  return args.outputName.empty() ? std::string("recon") : args.outputName;
}

std::filesystem::path normalizePathForCache(
    const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path normalized = std::filesystem::weakly_canonical(path, ec);
  if (ec)
    normalized = std::filesystem::absolute(path, ec);
  if (ec)
    normalized = path.lexically_normal();
  return normalized;
}

std::string psfModeKey(filters::PSFMode mode) {
  switch (mode) {
  case filters::PSFMode::None:
    return "None";
  case filters::PSFMode::Operator:
    return "Operator";
  case filters::PSFMode::PostSmooth:
    return "PostSmooth";
  case filters::PSFMode::ProjDomain:
    return "ProjDomain";
  case filters::PSFMode::Gaussian:
    return "Gaussian";
  case filters::PSFMode::File:
    return "File";
  }
  return "Unknown";
}

std::string psfInterpKey(filters::PSFInterp interp) {
  switch (interp) {
  case filters::PSFInterp::Linear:
    return "Linear";
  case filters::PSFInterp::Nearest:
    return "Nearest";
  }
  return "Unknown";
}

void appendFileIdentity(std::ostringstream &oss, const std::string &label,
                        const std::filesystem::path &path) {
  const auto normalized = normalizePathForCache(path);
  oss << label << "=" << normalized.string() << '\n';
  std::error_code ec;
  const auto size = std::filesystem::file_size(normalized, ec);
  if (ec) {
    oss << label << "_size=UNAVAILABLE\n";
  } else {
    oss << label << "_size=" << size << '\n';
  }
  ec.clear();
  const auto writeTime = std::filesystem::last_write_time(normalized, ec);
  if (ec) {
    oss << label << "_mtime=UNAVAILABLE\n";
  } else {
    oss << label << "_mtime=" << writeTime.time_since_epoch().count() << '\n';
  }
}

template <typename T>
void appendVectorLine(std::ostringstream &oss, const std::string &label,
                      const std::vector<T> &values) {
  oss << label << '=';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      oss << ',';
    oss << values[i];
  }
  oss << '\n';
}

std::filesystem::path resolveCompactDenomCacheMhdPath(const ReconArgs &args) {
  if (args.compactDenomCachePath.empty())
    return {};
  std::filesystem::path path(args.compactDenomCachePath);
  if (path.extension() != ".mhd")
    path.replace_extension(".mhd");
  return normalizePathForCache(path);
}

std::filesystem::path compactDenomManifestPath(
    const std::filesystem::path &mhdPath) {
  std::filesystem::path manifest = mhdPath;
  manifest.replace_extension(".manifest.txt");
  return manifest;
}

std::string abbreviateManifestLine(const std::string &line) {
  constexpr size_t kMaxManifestLine = 160;
  if (line.size() <= kMaxManifestLine)
    return line;
  return line.substr(0, kMaxManifestLine) + "...";
}

std::string summarizeManifestMismatch(const std::string &expected,
                                      const std::string &actual) {
  std::istringstream expectedStream(expected);
  std::istringstream actualStream(actual);
  std::string expectedLine;
  std::string actualLine;
  size_t lineNo = 1;
  while (true) {
    const bool haveExpected =
        static_cast<bool>(std::getline(expectedStream, expectedLine));
    const bool haveActual =
        static_cast<bool>(std::getline(actualStream, actualLine));
    if (!haveExpected && !haveActual)
      return "cache manifest mismatch";
    if (haveExpected != haveActual || expectedLine != actualLine) {
      std::ostringstream oss;
      oss << "cache manifest mismatch at line " << lineNo << ": expected '"
          << (haveExpected ? abbreviateManifestLine(expectedLine) : "<EOF>")
          << "', actual '"
          << (haveActual ? abbreviateManifestLine(actualLine) : "<EOF>")
          << "'";
      return oss.str();
    }
    ++lineNo;
  }
}

bool isCompactDenomManifestMismatch(const std::string &status) {
  return status.rfind("cache manifest mismatch", 0) == 0;
}

std::string buildCompactDenomCacheManifest(
    const ReconArgs &args, const OutputGeometry &geometry,
    const spect::io::CompactSymmetryMetadata &compactMeta,
    const std::string &compactMetaPath, bool compactChunkLayout,
    const spect::io::CompactChunkCSRDirectory *compactChunks,
    const std::vector<int> &bedZStarts,
    const spect::filters::PSFKernels *psfKernels) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << "cache_schema=compact_global_denom_v1\n";
  oss << "matrix_layout=" << args.matrixLayout << '\n';
  oss << "matrix_source_kind="
      << (compactChunkLayout ? "chunkCSRDir" : "loadCSR") << '\n';
  appendFileIdentity(oss, "compact_meta", compactMetaPath);
  oss << "compact_mode="
      << spect::io::compactSymmetryModeName(compactMeta.mode) << '\n';
  oss << "compact_dims=" << compactMeta.Nx << ',' << compactMeta.Ny << ','
      << compactMeta.Nz << '\n';
  oss << "compact_voxel_size=" << compactMeta.voxelSize << '\n';
  oss << "compact_det=" << compactMeta.detNz << ',' << compactMeta.detNx
      << '\n';
  oss << "compact_angles=" << compactMeta.fullNumAngles << '\n';
  oss << "compact_rows=" << compactMeta.compactRows() << '\n';
  oss << "full_rows=" << compactMeta.fullRows() << '\n';
  oss << "matrix_cols=" << compactMeta.totalVoxels << '\n';
  if (compactChunkLayout) {
    oss << "chunk_dir="
        << normalizePathForCache(compactChunks->directory()).string() << '\n';
    oss << "chunk_count=" << compactChunks->chunks().size() << '\n';
    oss << "chunk_total_rows=" << compactChunks->totalRows() << '\n';
    oss << "chunk_total_nnz=" << compactChunks->totalNnz() << '\n';
    for (size_t i = 0; i < compactChunks->chunks().size(); ++i) {
      const auto &chunk = compactChunks->chunks()[i];
      const std::string prefix = "chunk_" + std::to_string(i);
      appendFileIdentity(oss, prefix, chunk.path);
      oss << prefix << "_rows=" << chunk.rows << '\n';
      oss << prefix << "_angle_offset=" << chunk.angleOffset << '\n';
      oss << prefix << "_angle_count=" << chunk.angleCount << '\n';
      oss << prefix << "_nnz=" << chunk.nnz << '\n';
      oss << prefix << "_file_size_bytes=" << chunk.fileSizeBytes << '\n';
    }
  } else {
    appendFileIdentity(oss, "load_csr", args.loadCSRPath);
  }
  oss << "global_dims=" << geometry.nx << ',' << geometry.ny << ','
      << geometry.nz << '\n';
  oss << "global_spacing=" << geometry.vx << ',' << geometry.vy << ','
      << geometry.vz << '\n';
  // CTAC bakes attenuation into the matrix values, so the sensitivity depends
  // on the mu-map; keep these in the manifest to invalidate stale caches.
  // Lines are emitted only when CTAC is active so pre-existing no-CTAC caches
  // keep matching their stored manifests.
  if (args.doCTAC) {
    oss << "ctac=1\n";
    oss << "ctac_model=" << args.ctacModel << '\n';
    oss << "ctac_rot_sign=" << args.ctacRotSign << '\n';
    if (!args.attMapPath.empty()) {
      appendFileIdentity(oss, "ctac_mu_map", args.attMapPath);
    }
    if (!args.ctacAutoMuFrom.empty()) {
      appendFileIdentity(oss, "ctac_auto_mu_from", args.ctacAutoMuFrom);
      oss << "ctac_auto_mu_thresh=" << args.ctacAutoMuThresh << '\n';
      oss << "ctac_auto_mu_value=" << args.ctacAutoMuValue << '\n';
    }
  }
  oss << "bed_count=" << bedZStarts.size() << '\n';
  appendVectorLine(oss, "bed_z_starts", bedZStarts);
  oss << "joseph_passes=" << args.josephPasses << '\n';
  oss << "psf_mode=" << psfModeKey(args.psf.mode) << '\n';
  oss << "psf_fwhm=" << args.psf.fwhm_x << ',' << args.psf.fwhm_y << ','
      << args.psf.fwhm_z << ',' << args.psf.fwhm_mm << '\n';
  oss << "psf_vox=" << args.psf.vox_x << ',' << args.psf.vox_y << ','
      << args.psf.vox_z << '\n';
  oss << "psf_truncation_mm=" << args.psf.truncation_mm << '\n';
  oss << "psf_z_dependent=" << (args.psf.z_dependent ? 1 : 0) << '\n';
  oss << "psf_interp=" << psfInterpKey(args.psf.interp) << '\n';
  oss << "psf_min_sigmas=" << args.psf.min_sigmas << '\n';
  if (args.psf.mode == spect::filters::PSFMode::Operator &&
      bedZStarts.size() > 1) {
    oss << "psf_multibed_scope=global_volume\n";
  }
  if (psfKernels != nullptr) {
    appendVectorLine(oss, "psf_kernel_x", psfKernels->kx);
    appendVectorLine(oss, "psf_kernel_y", psfKernels->ky);
    appendVectorLine(oss, "psf_kernel_z", psfKernels->kz);
  } else {
    oss << "psf_kernel_x=\npsf_kernel_y=\npsf_kernel_z=\n";
  }
  return oss.str();
}

bool tryLoadCompactDenomCache(const std::filesystem::path &cacheMhdPath,
                              const std::string &expectedManifest,
                              const OutputGeometry &geometry,
                              std::vector<double> &denom,
                              std::string *status) {
  const auto manifestPath = compactDenomManifestPath(cacheMhdPath);
  if (!std::filesystem::exists(cacheMhdPath) ||
      !std::filesystem::exists(manifestPath)) {
    if (status)
      *status = "cache files not found";
    return false;
  }

  std::ifstream manifestIn(manifestPath, std::ios::binary);
  if (!manifestIn) {
    if (status)
      *status = "failed to open cache manifest";
    return false;
  }
  std::ostringstream actualManifest;
  actualManifest << manifestIn.rdbuf();
  if (actualManifest.str() != expectedManifest) {
    if (status)
      *status = summarizeManifestMismatch(expectedManifest,
                                          actualManifest.str());
    return false;
  }

  try {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    std::vector<double> loaded = spect::io::loadVolumeWithSpacing(
        cacheMhdPath.string(), nx, ny, nz, dx, dy, dz);
    const auto closeEnough = [](double a, double b) {
      return std::abs(a - b) <= 1e-9;
    };
    if (nx != geometry.nx || ny != geometry.ny || nz != geometry.nz ||
        !closeEnough(dx, geometry.vx) || !closeEnough(dy, geometry.vy) ||
        !closeEnough(dz, geometry.vz) || loaded.size() != denom.size()) {
      if (status)
        *status = "cache volume geometry mismatch";
      return false;
    }
    const VectorStats stats = computeVectorStats(loaded);
    if (stats.count == 0 || stats.finite != stats.count || stats.nonzero == 0 ||
        !(stats.sum > 0.0)) {
      if (status)
        *status = "cache volume stats invalid";
      return false;
    }
    denom.swap(loaded);
    if (status)
      *status = "cache hit";
    return true;
  } catch (const std::exception &ex) {
    if (status)
      *status = ex.what();
    return false;
  }
}

void saveCompactDenomCache(const std::filesystem::path &cacheMhdPath,
                           const std::string &manifest,
                           const OutputGeometry &geometry,
                           const std::vector<double> &denom) {
  spect::io::writeMHD(cacheMhdPath.string(), denom, geometry.nx, geometry.ny,
                      geometry.nz, static_cast<float>(geometry.vx),
                      static_cast<float>(geometry.vy),
                      static_cast<float>(geometry.vz), true);
  const auto manifestPath = compactDenomManifestPath(cacheMhdPath);
  std::ofstream manifestOut(manifestPath, std::ios::binary);
  if (!manifestOut)
    throw std::runtime_error("Failed writing compact denom cache manifest: " +
                             manifestPath.string());
  manifestOut << manifest;
  if (!manifestOut)
    throw std::runtime_error("Failed writing compact denom cache manifest: " +
                             manifestPath.string());
}

std::vector<double> computeKappaMap(const std::vector<double> &globalSensitivity,
                                    const ReconArgs &args,
                                    const std::vector<uint8_t> *mask) {
  std::vector<double> kappa(globalSensitivity.size(), 1.0);
  if (!args.bsremUseKappaMap || globalSensitivity.empty())
    return kappa;

  const double eps = std::max(args.bsremKappaEps, 1e-12);
  double sum = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < globalSensitivity.size(); ++i) {
    if (mask && !(*mask)[i])
      continue;
    const double value = 1.0 / std::sqrt(std::max(globalSensitivity[i], 0.0) + eps);
    sum += value;
    ++count;
    kappa[i] = value;
  }

  const double mean = (count > 0) ? (sum / static_cast<double>(count)) : 1.0;
  for (size_t i = 0; i < kappa.size(); ++i) {
    double value = kappa[i] / mean;
    if (args.bsremSoftMask && mask && !(*mask)[i])
      value = std::max(value, 1.0 + args.bsremSoftMaskLambdaOut);
    value = std::clamp(value, args.bsremKappaClipMin, args.bsremKappaClipMax);
    kappa[i] = value;
  }

  return kappa;
}

bool tryComputeRegGradientCUDA(const std::vector<double> &image,
                               const ReconArgs &args, const ReconOpts &opts,
                               const std::vector<uint8_t> *maskForReg,
                               std::vector<double> &gradReg) {
  if (!opts.sparseOps || opts.sparseOps->backend() != BackendType::CUDA)
    return false;
  return cuda::computeRegGradientCUDA(
      image, args.Nx, args.Ny, args.Nz, args.reg, maskForReg, gradReg);
}

bool tryApplySeparableConvCUDA(std::vector<double> &volume, int nx, int ny,
                               int nz, const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz,
                               const ReconOpts &opts) {
  return opts.sparseOps && opts.sparseOps->backend() == BackendType::CUDA &&
         cuda::isCudaVolumeOpsAvailable() &&
         cuda::applySeparableConv3DInplaceCUDA(volume, nx, ny, nz, kx, ky, kz);
}

void applySeparableConvRuntime(std::vector<double> &volume, int nx, int ny,
                               int nz, const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz,
                               const ReconOpts &opts, int threads) {
  if (!tryApplySeparableConvCUDA(volume, nx, ny, nz, kx, ky, kz, opts)) {
    filters::conv3d_separable_inplace(volume, nx, ny, nz, kx, ky, kz,
                                      threads);
  }
}

void applyIterationFilters(std::vector<double> &image, int nx, int ny, int nz,
                           const filters::Spacing3 &spacing,
                           const SmoothConf &smoothConf,
                           const filters::ButterworthConf &butterConf,
                           int iteration, int threads) {
  if (smoothConf.enabled) {
    filters::conv3d_separable_inplace(image, nx, ny, nz, smoothConf.kx,
                                      smoothConf.ky, smoothConf.kz, threads);
  }
  if (butterConf.enabled && iteration % std::max(1, butterConf.every_k_iters) == 0) {
    filters::apply_butterworth_inplace(image, nx, ny, nz, spacing,
                                       butterConf, threads);
  }
}

void applyJosephLike(std::vector<double> &buffer, int nx, int ny, int nz,
                     int passes, int threads, bool clampImage) {
  if (passes <= 0)
    return;
  applyJosephSmoothing(buffer, nx, ny, nz, passes, threads);
  if (clampImage)
    clampNonNegativeInPlace(buffer, threads);
}

template <typename MatrixType>
void runProjDomainMLEMStep(const MatrixType &matrix,
                           const std::vector<double> &projections,
                           std::vector<double> &image,
                           const std::vector<double> &denom,
                           const ReconArgs &args, const ReconOpts &opts,
                           const filters::PSFKernels &psfKernels,
                           const std::vector<double> *gradReg,
                           const std::vector<uint8_t> *maskForReg,
                           const std::vector<uint8_t> *maskPtr,
                           const filters::ButterworthConf &butterConf,
                           const SmoothConf &smoothConf,
                           const filters::Spacing3 &spacing) {
  (void)maskForReg;
  const int rows = matrix.rows;
  const int nvox = static_cast<int>(image.size());
  const double eps = 1e-12;
  const size_t frameSize = static_cast<size_t>(opts.det_nz) *
                           static_cast<size_t>(opts.det_nx);
  uint64_t chunkNnzBudget = 8ull * 1024ull * 1024ull;
  bool useChunkedCudaForward = false;
  bool useStreamingChunkedCuda = false;
  bool usedSparseForward = false;
  std::vector<double> numer(static_cast<size_t>(nvox), 0.0);

  if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
    useChunkedCudaForward = canUseProjDomainCudaChunking(matrix, opts);
    if (useChunkedCudaForward) {
      chunkNnzBudget = cuda::getProjDomainChunkNnzBudgetCUDA();
    }
    useStreamingChunkedCuda =
        useChunkedCudaForward &&
        canUseFrameAlignedProjDomainChunks(frameSize, rows);
    if (useStreamingChunkedCuda) {
      const auto &chunks =
          getProjDomainCudaChunks(matrix, frameSize, chunkNnzBudget);
      logInfo("Proj-domain CUDA chunking: chunks=" +
              std::to_string(chunks.size()) + ", frameRows=" +
              std::to_string(frameSize) + ", nnzBudget=" +
              std::to_string(chunkNnzBudget));
      usedSparseForward = cuda::runProjDomainChunkedStreamingCUDA(
          matrix, chunks, projections, psfKernels, frameSize, opts.det_nz,
          opts.det_nx, image.data(), opts.threads, numer.data());
    }
  }

  if (!usedSparseForward) {
    std::vector<double> yhat(static_cast<size_t>(rows), 0.0);

    if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
      if (useChunkedCudaForward) {
        runProjDomainChunkedForward(matrix, opts, frameSize, chunkNnzBudget,
                                    image.data(), yhat.data());
        usedSparseForward = true;
      } else {
        try {
          opts.sparseOps->spmv(matrix, image.data(), yhat.data(), opts.threads);
          usedSparseForward = true;
        } catch (const std::runtime_error &ex) {
          const std::string message = ex.what() ? ex.what() : "";
          if (message.find("row offsets representable as int32") ==
              std::string::npos) {
            throw;
          }
          std::cerr << "[WARN] Proj-domain MLEM forward projection fell back to CPU because the CUDA sparse backend does not support this matrix row-offset range.\n";
        }
      }
    } else {
      spect::io::CSRView view{};
      view.rows = matrix.rows;
      view.cols = matrix.cols;
      view.row_ptr = matrix.row_ptr.data();
      view.col_idx = matrix.col_idx.data();
      view.val = matrix.val.data();
      view.nnz = matrix.val.size();
      try {
        opts.sparseOps->spmv(view, image.data(), yhat.data(), opts.threads);
        usedSparseForward = true;
      } catch (const std::runtime_error &ex) {
        const std::string message = ex.what() ? ex.what() : "";
        if (message.find("row offsets representable as int32") ==
            std::string::npos) {
          throw;
        }
        std::cerr << "[WARN] Proj-domain MLEM forward projection fell back to CPU because the CUDA sparse backend does not support this matrix row-offset range.\n";
      }
    }

    if (!usedSparseForward) {
      yhat = spect::forward_project(matrix, image, opts.threads);
    }

    std::vector<double> ratio = projections;
#pragma omp parallel for schedule(static) num_threads(opts.threads)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(ratio.size()); ++i) {
      ratio[static_cast<size_t>(i)] /=
          std::max(yhat[static_cast<size_t>(i)], eps);
    }

    if (frameSize > 0) {
#pragma omp parallel num_threads(opts.threads) if(opts.threads > 1 && opts.num_angles > 1)
      {
        std::vector<double> frame(frameSize);
#pragma omp for schedule(static)
        for (int angle = 0; angle < opts.num_angles; ++angle) {
          const size_t offset = static_cast<size_t>(angle) * frameSize;
          std::copy_n(ratio.data() + offset, frameSize, frame.data());
          filters::conv2d_separable_inplace(frame, opts.det_nz, opts.det_nx,
                                            psfKernels.ky, psfKernels.kx, 1);
          std::copy_n(frame.data(), frameSize, ratio.data() + offset);
        }
      }
    }

    if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
      if (useChunkedCudaForward) {
        runProjDomainChunkedTranspose(matrix, opts, frameSize, chunkNnzBudget,
                                      ratio.data(), numer.data());
      } else {
        opts.sparseOps->spmvT(matrix, ratio.data(), numer.data(), opts.threads);
      }
    } else {
      spect::io::CSRView view{};
      view.rows = matrix.rows;
      view.cols = matrix.cols;
      view.row_ptr = matrix.row_ptr.data();
      view.col_idx = matrix.col_idx.data();
      view.val = matrix.val.data();
      view.nnz = matrix.val.size();
      opts.sparseOps->spmvT(view, ratio.data(), numer.data(), opts.threads);
    }
  }

#pragma omp parallel for schedule(static) num_threads(opts.threads)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(image.size()); ++j) {
    const size_t index = static_cast<size_t>(j);
    if (maskPtr && !(*maskPtr)[index]) {
      image[index] = 0.0;
      continue;
    }
    double denomEff = denom[index];
    if (!(denomEff > 0.0)) {
      image[index] = 0.0;
      continue;
    }
    if (gradReg) {
      const double fj = std::max(image[index], eps);
      denomEff += args.reg.beta * ((*gradReg)[index] / fj);
      if (!(denomEff > 0.0))
        denomEff = denom[index] + args.reg.beta * std::abs((*gradReg)[index]) / fj + eps;
    }
    image[index] *= numer[index] / denomEff;
    if (opts.nonneg && image[index] < 0.0)
      image[index] = 0.0;
  }

  applyJosephLike(image, args.Nx, args.Ny, args.Nz, args.josephPasses,
                  opts.threads, true);
  if (butterConf.enabled) {
    filters::apply_butterworth_inplace(image, args.Nx, args.Ny, args.Nz,
                                       spacing, butterConf, opts.threads);
  }
  if (smoothConf.enabled) {
    applySeparableConvRuntime(image, args.Nx, args.Ny, args.Nz, smoothConf.kx,
                              smoothConf.ky, smoothConf.kz, opts,
                              opts.threads);
    if (maskPtr)
      spect::core::applyMaskInplace(image, *maskPtr);
  }
}

void reportConfigStatus(const ReconArgs &args) {
  if (args.configExplicit) {
    if (!args.configFound) {
      logWarn("Specified config file '" + args.configPath +
              "' was not found. Using internal defaults.");
    } else if (args.configLoaded) {
      logInfo("Loaded configuration from " + args.configPath);
    } else {
      logWarn("Failed to load config file '" + args.configPath +
              "'. Using internal defaults.");
    }
    return;
  }

  if (args.configFound && args.configLoaded) {
    logInfo("Loaded configuration from " + args.configPath);
  } else if (!args.configFound) {
    logInfo("Default config not found in exe-local config/ or cwd fallbacks. Using internal defaults.");
  } else {
    logWarn("Failed to load default config '" + args.configPath +
            "'. Using internal defaults.");
  }
}

} // namespace

int runProgram(int argc, char **argv) {
  const auto fullCommandStart = ProfileClock::now();
  if (argc <= 1) {
    spect::cli::ArgumentParser::printUsage();
    return spect::cli::kExitSuccess;
  }

  ReconArgs args;
  spect::cli::ArgumentParser parser;
  const auto parseResult = parser.parse(argc, argv, args);
  if (parseResult == spect::cli::ArgumentParser::ParseResult::ExitSuccess)
    return spect::cli::kExitSuccess;
  if (parseResult != spect::cli::ArgumentParser::ParseResult::Success)
    return spect::cli::kExitUserError;

  setVerbosity(std::clamp(args.verbosity, 0, 2));
  args.verbosity = getVerbosity();

  std::string logDirError;
  auto logDir = resolveLogDirectory(args, logDirError);
  ReconArgs logArgs = args;
  if (logDir.empty())
    logDir = ".";
  logArgs.outputDir = logDir.string();
  ConsoleTeeGuard tee(makeDefaultLogPath(logArgs));
  if (tee.enabled()) {
    logInfo("Logging console output to: " + tee.logPath().string());
    if (!logDirError.empty()) {
      logWarn("Failed to create output directory '" + args.outputDir +
              "' for logging (" + logDirError + "); using current directory.");
    }
  } else {
    logWarn("Failed to create log file at: " + makeDefaultLogPath(logArgs).string());
  }
  logInfo("Command line: " + makeCommandLineString(argc, argv));
  reportConfigStatus(args);

#ifdef _OPENMP
  omp_set_num_threads(std::max(1, args.numThreads));
#endif

  try {
    CompactExecutionTiming compactTiming;
    const auto loadToIterationStart = ProfileClock::now();
    const bool compactLayoutRequested =
        args.matrixLayout == "compact-symmetric";
    const bool compactChunkLayout =
        compactLayoutRequested && !args.chunkCSRDir.empty();
    spect::io::CompactSymmetryMetadata compactMeta;
    std::string compactMetaPath;

    if (compactChunkLayout) {
      compactMetaPath = resolveCompactMetadataPath(args);
      compactMeta = spect::io::loadCompactSymmetryMetadata(compactMetaPath);
      applyCompactMetadataGeometry(args, compactMeta);
      logInfo("compact chunk metadata geometry loaded from " + compactMetaPath +
              " (mode=" + spect::io::compactSymmetryModeName(compactMeta.mode) +
              ", compactRows=" + std::to_string(compactMeta.compactRows()) +
              ", fullRows=" + std::to_string(compactMeta.fullRows()) + ")");
    }

    std::string dimHintWarning;
    if (adoptReconDimsFromHint(args, &dimHintWarning) && !dimHintWarning.empty())
      logWarn(dimHintWarning);

    std::string dimsWarning;
    if (ensureReconDims(args, &dimsWarning) && !dimsWarning.empty())
      logWarn(dimsWarning);

    const BedSetupResult bedSetup = prepareBedSetup(args);
    for (const std::string &warning : bedSetup.warnings)
      logWarn(warning);
    logInfo(bedSetup.summary);
    printReconArgs(args);

    std::filesystem::create_directories(args.outputDir);

    const std::vector<std::string> projPaths = bedSetup.projectionPaths;
    if (projPaths.empty() || projPaths.front().empty())
      throw std::runtime_error("Projection path is required.");

    auto stageStart = ProfileClock::now();
    const LoadedProjectionSet projectionSet =
        loadProjectionSet(projPaths, args.numAngles, "Projection Data", "matrix");
    compactTiming.projectionLoadSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
    if (projectionSet.data.empty())
      throw std::runtime_error("Projection data is empty after loading.");

    args.numAngles = projectionSet.numAngles;
    if (projectionSet.detectorShapeKnown) {
      args.detNz = projectionSet.detNz;
      args.detNx = projectionSet.detNx;
    } else {
      if (projectionSet.numAngles <= 0 ||
          projectionSet.raysPerBed % static_cast<size_t>(projectionSet.numAngles) != 0) {
        throw std::runtime_error("Projection BIN geometry is inconsistent.");
      }
      const size_t raysPerAngle =
          projectionSet.raysPerBed / static_cast<size_t>(projectionSet.numAngles);
      const size_t configuredRays =
          (args.detNz > 0 && args.detNx > 0)
              ? static_cast<size_t>(args.detNz) * static_cast<size_t>(args.detNx)
              : 0;
      if (configuredRays == raysPerAngle) {
        logInfo("Projection BIN does not store detZ/detX; using configured detector shape " +
                std::to_string(args.detNz) + "x" + std::to_string(args.detNx) + ".");
      } else {
        if (raysPerAngle > static_cast<size_t>(std::numeric_limits<int>::max()))
          throw std::runtime_error("Projection BIN rays_per_angle exceeds int range.");
        args.detNz = 1;
        args.detNx = static_cast<int>(raysPerAngle);
        logWarn("Projection BIN does not store detZ/detX and configured -detN does not match rays_per_angle. Falling back to 1x" +
                std::to_string(args.detNx) +
                "; set -detN <detZ,detX> when projection-domain PSF or detector geometry is required.");
      }
    }
    const int bedCount = std::max(1, bedSetup.bedCount);
    const int baseNx = bedSetup.baseNx;
    const int baseNy = bedSetup.baseNy;
    const int baseNz = bedSetup.baseNz;
    const size_t baseNvox = static_cast<size_t>(baseNx) * static_cast<size_t>(baseNy) *
                            static_cast<size_t>(baseNz);
    const size_t globalNvox = static_cast<size_t>(args.Nx) * static_cast<size_t>(args.Ny) *
                              static_cast<size_t>(args.Nz);

    logInfo("Loaded projection rays: total=" +
            std::to_string(projectionSet.data.size()) + ", per bed=" +
            std::to_string(projectionSet.raysPerBed));

    std::vector<double> backgroundProj;
    std::vector<std::string> backgroundPaths = args.bProjPaths;
    if (backgroundPaths.empty() && !args.bProjPath.empty())
      backgroundPaths = {args.bProjPath};
    if (!backgroundPaths.empty()) {
      if (bedCount > 1 && backgroundPaths.size() != static_cast<size_t>(bedCount))
        throw std::runtime_error("Background projection count must match bed count.");
      if (bedCount == 1 && backgroundPaths.size() > 1)
        throw std::runtime_error("Multiple background projections were provided for single-bed mode.");

      const LoadedProjectionSet backgroundSet =
          loadProjectionSet(backgroundPaths, args.numAngles,
                            "Background Projection", "b_proj");
      if (backgroundSet.raysPerBed != projectionSet.raysPerBed)
        throw std::runtime_error("Background projection geometry does not match emission projections.");
      backgroundProj = backgroundSet.data;
      logInfo("Loaded background projection term with " +
              std::to_string(backgroundProj.size()) + " rays.");
    }

    if (args.algorithm == ReconArgs::Algorithm::BSREM && args.psfProj) {
      throw std::runtime_error(
          "BSREM does not support projection-domain PSF. Use operator-domain PSF instead.");
    }

    const auto spacingArray = voxelSpacingFromArgs(args);
    const filters::Spacing3 spacing{spacingArray[0], spacingArray[1], spacingArray[2]};
    const OutputGeometry outputGeometry{args.Nx, args.Ny, args.Nz, spacing.dx,
                      spacing.dy, spacing.dz};

    std::vector<double> sharedAnatomical;
    if (!args.priorPath.empty() && args.priorPath != "none") {
      logInfo("Loading anatomical prior from " + args.priorPath);
      sharedAnatomical = loadVolumeChecked(args.priorPath, "Prior volume", args.Nx,
                                          args.Ny, args.Nz);
    } else if ((args.useReg && (args.reg.type == RegType::Bowsher ||
                                args.reg.type == RegType::PLS)) &&
               !args.attMapPath.empty()) {
      logInfo("No explicit prior was provided. Reusing attenuation map as anatomical prior.");
      sharedAnatomical = loadVolumeChecked(args.attMapPath, "Attenuation volume",
                                          args.Nx, args.Ny, args.Nz);
    }
    if (!sharedAnatomical.empty())
      args.reg.anatomical_image = &sharedAnatomical;

    std::vector<uint8_t> mask;
    const std::vector<uint8_t> *maskPtr =
        maybeMakeCylindricalMask(mask, args.Nx, args.Ny, args.Nz, spacingArray, args);

    std::vector<double> ctacMuMap;
    if (args.doCTAC) {
      if (!args.attMapPath.empty()) {
        logInfo("Loading attenuation map from " + args.attMapPath);
        ctacMuMap = loadVolumeChecked(args.attMapPath, "Attenuation map",
                                      args.Nx, args.Ny, args.Nz);
      } else if (!args.ctacAutoMuFrom.empty()) {
        logInfo("Auto mu-map: segmenting early-iteration reconstruction " +
                args.ctacAutoMuFrom);
        logWarn("Auto mu-map assumes activity roughly fills the attenuating "
                "volume (water-equivalent phantom). For hot-rods-in-cold-body "
                "objects supply a measured -attMap instead.");
        const std::vector<double> earlyRecon = loadVolumeChecked(
            args.ctacAutoMuFrom, "Auto-mu source reconstruction", args.Nx,
            args.Ny, args.Nz);
        const spect::main::AutoMuMapResult autoMu =
            spect::main::buildAutoMuMapFromActivity(
                earlyRecon, args.Nx, args.Ny, args.Nz, args.ctacAutoMuThresh,
                args.ctacAutoMuValue, maskPtr);
        if (autoMu.bodyVoxels == 0)
          throw std::runtime_error(
              "Auto mu-map segmentation found no body voxels; check "
              "-ctacAutoMu input volume and -ctacAutoMuThresh.");
        logInfo("Auto mu-map: body voxels=" +
                std::to_string(autoMu.bodyVoxels) + ", threshold=" +
                std::to_string(autoMu.thresholdUsed) + " (P99.9=" +
                std::to_string(autoMu.robustMax) + "), mu=" +
                std::to_string(args.ctacAutoMuValue) + " mm^-1.");
        ctacMuMap = autoMu.muMap;
        saveVolumeOutput(ctacMuMap, outputGeometry, args.outputDir,
                         outputPrefixFromArgs(args) + "_automu", -1, nullptr,
                         sanitizeCropSettings(0, 0));
      } else {
        throw std::runtime_error(
            "CTAC requires -attMap <file> or -ctacAutoMu <early recon volume>.");
      }
    }
    if (!std::isnan(args.maskFromAttThreshold)) {
      const std::vector<double> *maskSource = nullptr;
      if (!ctacMuMap.empty())
        maskSource = &ctacMuMap;
      else if (!sharedAnatomical.empty())
        maskSource = &sharedAnatomical;
      if (!maskSource)
        throw std::runtime_error("-maskFromAtt requires -attMap or -prior.");

      if (mask.empty())
        mask.assign(globalNvox, 1);
      for (size_t i = 0; i < mask.size(); ++i) {
        if ((*maskSource)[i] <= args.maskFromAttThreshold)
          mask[i] = 0;
      }
      maskPtr = &mask;
      logInfo("Applied attenuation-derived body mask.");
    }

    if (maskPtr) {
      const std::vector<double> maskVolume(mask.begin(), mask.end());
      saveVolumeOutput(maskVolume, outputGeometry, args.outputDir,
                       outputPrefixFromArgs(args) + "_mask", -1, nullptr,
                       sanitizeCropSettings(0, 0));
    }

    spect::io::CSR sysCSR;
    spect::io::CSRMap csrMap;
    bool useMappedCSR = false;
    std::unique_ptr<spect::io::CompactChunkCSRDirectory> compactChunks;

    stageStart = ProfileClock::now();
    if (compactChunkLayout) {
      logInfo("Opening compact chunk CSR directory from " + args.chunkCSRDir);
      auto opened =
          spect::io::openCompactChunkCSRDirectory(args.chunkCSRDir, compactMeta);
      logInfo("Compact chunk CSR directory opened: chunks=" +
              std::to_string(opened.chunks().size()) +
              ", compactRows=" + std::to_string(opened.totalRows()) +
              ", nnz=" + std::to_string(opened.totalNnz()) +
              ", hostCacheBudgetBytes=" +
              std::to_string(opened.cacheBudgetBytes()));
      compactChunks =
          std::make_unique<spect::io::CompactChunkCSRDirectory>(std::move(opened));
    } else if (!args.loadCSRPath.empty()) {
      std::error_code csrExistsEc;
      const bool csrExists =
          std::filesystem::exists(args.loadCSRPath, csrExistsEc);
      if (csrExistsEc) {
        throw std::runtime_error("Cannot access CSR cache path " +
                                 args.loadCSRPath + ": " +
                                 csrExistsEc.message());
      }
      if (!csrExists) {
        throw std::runtime_error("CSR cache file does not exist: " +
                                 args.loadCSRPath);
      }
      const uint64_t csrFileSize =
          static_cast<uint64_t>(std::filesystem::file_size(args.loadCSRPath));
      const bool implicitCompactMap =
          compactLayoutRequested &&
          spect::io::shouldUseImplicitCSRMap(csrFileSize);
      if (args.mapCSR || implicitCompactMap) {
        logInfo("Mapping CSR cache from " + args.loadCSRPath +
                (implicitCompactMap && !args.mapCSR
                     ? " (implicit compact single-file large-matrix path)"
                     : ""));
        try {
          csrMap.open(args.loadCSRPath);
          useMappedCSR = true;
        } catch (const std::exception &ex) {
          if (args.mapCSR ||
              spect::io::shouldRefuseFullCopyCSRLoad(
                  csrFileSize, spect::io::availablePhysicalMemoryBytes())) {
            throw;
          }
          logWarn("CSR mmap failed for " + args.loadCSRPath + ": " +
                  ex.what() + "; falling back to full-copy load.");
          sysCSR = spect::io::loadCSRBinary(args.loadCSRPath);
        }
      } else {
        logInfo("Loading CSR cache into memory from " + args.loadCSRPath);
        sysCSR = spect::io::loadCSRBinary(args.loadCSRPath);
      }
    } else {
      if (args.sysIndexPath.empty() || args.sysWeightPath.empty()) {
        throw std::runtime_error(
            "System matrix inputs (-indexMat and -weightMat) are required unless -loadCSR is used.");
      }

      std::string indexPath = args.sysIndexPath;
      std::string weightPath = args.sysWeightPath;
      const std::string indexVar = resolveMatVarName(indexPath, "");
      const std::string weightVar = resolveMatVarName(weightPath, "");
      const std::vector<uint8_t> *maskForCSR =
          (maskPtr && args.ApruneMask && bedCount == 1) ? maskPtr : nullptr;
      if (maskPtr && args.ApruneMask && bedCount > 1)
        logWarn("ApruneMask is ignored for multi-bed CSR build.");

      if (args.Astream) {
        logInfo("Building CSR via streaming MAT reader.");
        sysCSR = spect::io::buildCSR_from_mat_stream(
            indexPath, indexVar, weightPath, weightVar, baseNvox,
            args.weightPruneEps, maskForCSR);
      } else {
        logInfo("Building CSR via in-memory MAT load.");
        const auto indexMat = spect::io::readMat(indexPath, indexVar);
        const auto weightMat = spect::io::readMat(weightPath, weightVar);
        sysCSR = spect::io::buildCSR_from_mat(indexMat, weightMat, baseNvox,
                                              args.weightPruneEps, maskForCSR);
      }
      if (!args.saveCSRPath.empty()) {
        logInfo("Saving CSR cache to " + args.saveCSRPath);
        spect::io::saveCSRBinary(sysCSR, args.saveCSRPath);
      }
    }
    compactTiming.csrMapOrLoadSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();

    const int matrixRows = compactChunkLayout
                               ? static_cast<int>(compactChunks->totalRows())
                               : (useMappedCSR ? csrMap.view().rows : sysCSR.rows);
    const int matrixCols =
        compactChunkLayout ? compactChunks->cols()
                           : (useMappedCSR ? csrMap.view().cols : sysCSR.cols);
    spect::io::CSRView matrixViewForValidation{};
    if (compactChunkLayout) {
      matrixViewForValidation.rows = matrixRows;
      matrixViewForValidation.cols = matrixCols;
      matrixViewForValidation.nnz = compactChunks->totalNnz();
    } else if (useMappedCSR) {
      matrixViewForValidation = csrMap.view();
    } else {
      matrixViewForValidation.rows = sysCSR.rows;
      matrixViewForValidation.cols = sysCSR.cols;
      matrixViewForValidation.row_ptr = sysCSR.row_ptr.data();
      matrixViewForValidation.col_idx = sysCSR.col_idx.data();
      matrixViewForValidation.val = sysCSR.val.data();
      matrixViewForValidation.nnz = sysCSR.val.size();
    }
    if (matrixCols != static_cast<int>(baseNvox)) {
      throw std::runtime_error("System matrix voxel count does not match the per-bed reconstruction grid.");
    }

    const bool compactLayout = compactLayoutRequested;
    if (compactLayout) {
      stageStart = ProfileClock::now();
      if (compactMetaPath.empty()) {
        compactMetaPath = resolveCompactMetadataPath(args);
        compactMeta = spect::io::loadCompactSymmetryMetadata(compactMetaPath);
      }
      if (compactChunkLayout) {
        spect::io::validateCompactChunkSymmetricInputs(
            compactMeta, compactChunks->totalRows(), compactChunks->cols(),
            projectionSet.raysPerBed, args.detNz, args.detNx, args.numAngles,
            baseNx, baseNy, baseNz);
      } else {
        spect::io::validateCompactSymmetricInputs(
            compactMeta, matrixViewForValidation, projectionSet.raysPerBed,
            args.detNz, args.detNx, args.numAngles, baseNx, baseNy, baseNz);
      }
      compactTiming.metadataLoadAndValidateSeconds =
          std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
      logInfo("compact-symmetric layout enabled via " + compactMetaPath +
              " (mode=" + spect::io::compactSymmetryModeName(compactMeta.mode) +
              ", compactRows=" + std::to_string(compactMeta.totalRows) +
              ", fullRows=" + std::to_string(compactMeta.fullRows()) +
              (compactChunkLayout ? ", matrix=chunkCSRDir)" : ", matrix=loadCSR)"));
      if (!compactChunkLayout) {
        logWarn("[DEPRECATED] single-file compact-symmetric CSRB "
                "(-loadCSR -compactMeta) is legacy 260525 compatibility. "
                "Prefer -chunkCSRDir with 260601 metadata.");
      }
      if (compactChunkLayout && bedCount > spect::core::kMaxBeds) {
        throw std::runtime_error(
            "compact chunk multi-bed CUDA path supports at most " +
            std::to_string(spect::core::kMaxBeds) + " beds.");
      }
    }

    const size_t expectedRowsPerBed =
        compactLayout ? static_cast<size_t>(compactMeta.fullRows())
                      : static_cast<size_t>(matrixRows);
    const size_t expectedRays = expectedRowsPerBed * static_cast<size_t>(bedCount);
    std::vector<double> projections = projectionSet.data;
    if (projections.size() < expectedRays) {
      throw std::runtime_error("Projection count is smaller than the required system-matrix rows across all beds.");
    }
    if (projections.size() > expectedRays) {
      if (compactLayout) {
        throw std::runtime_error(
            "Compact-symmetric layout requires exact full projection length; refusing to truncate.");
      }
      logWarn("Projection count exceeds required rays. Truncating to match the system matrix.");
      projections.resize(expectedRays);
    }
    if (!backgroundProj.empty() && backgroundProj.size() != projections.size()) {
      throw std::runtime_error("Background projection size does not match projection size.");
    }

    const std::vector<int> bedZStarts =
        spect::app::computeBedZOffsets(bedCount, baseNz, args.bedOverlaps);

    std::vector<std::vector<uint16_t>> dynamicAttenFactors;
    if (args.doCTAC) {
      const spect::main::CTACModel ctacModel =
          (args.ctacModel == "per-voxel")
              ? spect::main::CTACModel::PerVoxelApprox
              : spect::main::CTACModel::LineIntegral;
      if (ctacModel == spect::main::CTACModel::PerVoxelApprox) {
        logWarn("CTAC model 'per-voxel' is the legacy exp(-mu) approximation "
                "replicated per angle; it is not quantitative line-integral "
                "CTAC. Use -ctacModel line-integral for the physical model.");
      } else {
        logInfo("CTAC model: angle-dependent parallel-ray line integral "
                "(rotation sign " + std::to_string(args.ctacRotSign) +
                "). Attenuation map must be mu in mm^-1.");
      }
      if (useMappedCSR || bedCount > 1 || compactChunkLayout) {
        logInfo("Using dynamic CTAC attenuation factors.");
        dynamicAttenFactors.resize(static_cast<size_t>(bedCount));
        const size_t plane = static_cast<size_t>(baseNx) * static_cast<size_t>(baseNy);
        for (int bed = 0; bed < bedCount; ++bed) {
          std::vector<double> localMu(baseNvox, 0.0);
          for (int z = 0; z < baseNz; ++z) {
            const int globalZ = bedZStarts[static_cast<size_t>(bed)] + z;
            const size_t srcOffset = static_cast<size_t>(globalZ) * plane;
            const size_t dstOffset = static_cast<size_t>(z) * plane;
            std::copy_n(ctacMuMap.data() + srcOffset, plane, localMu.data() + dstOffset);
          }
          dynamicAttenFactors[static_cast<size_t>(bed)] =
              computeAttenFactors(ctacModel, localMu, baseNx, baseNy, baseNz,
                                  spacing.dx, spacing.dy, spacing.dz,
                                  args.numAngles, args.ctacRotSign,
                                  args.numThreads);
        }
      } else {
        logInfo("Applying CTAC directly to in-memory CSR.");
        applyCTACToCSR(sysCSR, ctacMuMap, baseNx, baseNy, baseNz, spacing.dx,
                       spacing.dy, spacing.dz, args.numAngles, args.numThreads,
                       ctacModel, args.ctacRotSign);
      }
    }

    SparseBackendChoice resolvedBackend;
    bool backendParsed = false;
    const ISparseOps &sparseOps =
        selectSparseOps(args.sparseBackend, &resolvedBackend, &backendParsed);
    if (!backendParsed)
      logWarn("Unknown sparse backend '" + args.sparseBackend + "'. Falling back to auto.");
    logInfo(std::string("Sparse backend: ") + sparseBackendChoiceName(resolvedBackend));
    if (compactLayout && resolvedBackend != SparseBackendChoice::CUDA) {
      throw std::runtime_error(
          "compact-symmetric layout requires an enabled CUDA sparse backend.");
    }

    ReconOpts opts;
    opts.threads = std::max(1, args.numThreads);
    opts.n_subsets = std::max(1, args.numSubsets);
    opts.nonneg = true;
    opts.mask = maskPtr;
    opts.w_proj = nullptr;
    opts.w_img = nullptr;
    opts.b_proj = backgroundProj.empty() ? nullptr : &backgroundProj;
    opts.sparseOps = &sparseOps;
    opts.bedCount = bedCount;
    opts.bedOverlaps = args.bedOverlaps;
    opts.bedAttenFactors = dynamicAttenFactors;
    opts.bedOffsetVoxels.resize(static_cast<size_t>(bedCount));
    for (int bed = 0; bed < bedCount; ++bed) {
      opts.bedOffsetVoxels[static_cast<size_t>(bed)] =
          static_cast<long long>(bedZStarts[static_cast<size_t>(bed)]) *
          static_cast<long long>(baseNx) * static_cast<long long>(baseNy);
    }
    opts.det_nz = args.detNz;
    opts.det_nx = args.detNx;
    opts.num_angles = args.numAngles;

    stageStart = ProfileClock::now();
    auto prepared = prepareReconPreparedParams(args, args.psf);
    auto runtimePsf = preparePSFForRuntime(opts, args, spacing, bedCount,
                                           projections.size(), baseNx);
    compactTiming.psfPrepareSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
    for (const std::string &message : runtimePsf.logs.infoMessages)
      logInfo(message);
    for (const std::string &message : runtimePsf.logs.warningMessages)
      logWarn(message);

    const filters::PSFKernels *psfPtr = nullptr;
    if (!runtimePsf.psfKernels.kx.empty()) {
      psfPtr = &runtimePsf.psfKernels;
    } else if (!prepared.psfKernels.kx.empty()) {
      psfPtr = &prepared.psfKernels;
    }
    const CropSettings crop = sanitizeCropSettings(args.cropEndStart, args.cropEndEnd);
    if ((crop.start != 0 || crop.end != 0) && !isCropSettingsValid(args.Nz, crop))
      throw std::runtime_error("Invalid output crop range.");
    OutputTransform outputTransform;
    const OutputTransform *outputTransformPtr = nullptr;
    const bool dfListMode = !args.projPaths.empty();
    const bool enableBedZFlip = args.zBedFlip && dfListMode;
    if (enableBedZFlip || args.zOutFlip) {
      outputTransform.flipZPerBed = enableBedZFlip;
      outputTransform.flipZWholeOutput = args.zOutFlip;
      outputTransform.baseNz = baseNz;
      outputTransform.bedZStarts = bedZStarts;
      outputTransformPtr = &outputTransform;
      if (enableBedZFlip)
        logInfo("Output transform: per-bed Z flip enabled for dfList bed segments.");
      if (args.zOutFlip)
        logInfo("Output transform: whole-output Z flip enabled.");
    }

    const auto runReconstruction = [&](const auto &matrix) {
      using MatrixType = std::decay_t<decltype(matrix)>;

      if (compactLayout) {
        spect::io::CSRView compactMatrix{};
        if (!compactChunkLayout) {
          if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
            compactMatrix = matrix;
          } else {
            compactMatrix.rows = matrix.rows;
            compactMatrix.cols = matrix.cols;
            compactMatrix.row_ptr = matrix.row_ptr.data();
            compactMatrix.col_idx = matrix.col_idx.data();
            compactMatrix.val = matrix.val.data();
            compactMatrix.nnz = matrix.val.size();
          }
        }

#if !defined(SPECT_ENABLE_CUDA_SPARSEOPS)
        throw std::runtime_error(
            "compact-symmetric layout requires a CUDA-enabled build.");
#else
        const OutputGeometry geometry = outputGeometry;
        const std::string outputPrefix = outputPrefixFromArgs(args);
        const uint16_t *compactAttFactors = nullptr;
        if (!opts.bedAttenFactors.empty()) {
          if (!compactChunkLayout) {
            throw std::runtime_error(
                "compact-symmetric CTAC requires the chunk directory layout.");
          }
          if (bedCount > 1) {
            throw std::runtime_error(
                "compact-symmetric CTAC currently supports a single bed.");
          }
          compactAttFactors = opts.bedAttenFactors[0].data();
          logInfo("compact CTAC: attenuation factors baked into chunk values "
                  "on device (fused kernels disabled for this run).");
        }
        std::vector<double> denom(globalNvox, 0.0);
        const bool compactPsfActive = needs_psf_blur(opts, psfPtr);
        const bool compactMultiBedGlobalPsf = compactPsfActive && bedCount > 1;
        spect::cuda::CompactSymmetricStageTiming compactCudaTiming;
        MemoryUsageSnapshot compactPeakMemory;
        auto observeCompactMemory = [&]() {
          mergeMemoryPeak(compactPeakMemory, queryMemoryUsageSnapshot());
        };
        observeCompactMemory();
        const auto sensStart = ProfileClock::now();
        const std::filesystem::path compactDenomCachePath =
            resolveCompactDenomCacheMhdPath(args);
        const std::string compactDenomManifest =
            compactDenomCachePath.empty()
                ? std::string()
                : buildCompactDenomCacheManifest(
                      args, geometry, compactMeta, compactMetaPath,
                      compactChunkLayout, compactChunks.get(), bedZStarts, psfPtr);
        bool loadedCompactDenomCache = false;
        if (!compactDenomCachePath.empty()) {
          std::string cacheStatus;
          loadedCompactDenomCache = tryLoadCompactDenomCache(
              compactDenomCachePath, compactDenomManifest, geometry, denom,
              &cacheStatus);
          if (loadedCompactDenomCache) {
            logInfo("compact denom cache hit: " +
                    compactDenomCachePath.string());
            if (compactChunkLayout) {
              spect::cuda::prepareCompactChunkExecutionCUDA(*compactChunks);
            }
          } else {
            if (isCompactDenomManifestMismatch(cacheStatus)) {
              throw std::runtime_error(
                  "compact denom cache manifest mismatch for " +
                  compactDenomCachePath.string() + ": " + cacheStatus +
                  ". Use a distinct -compactDenomCache path for no-PSF, "
                  "PSF/FWHM, Joseph, and bed-geometry variants, or delete the "
                  "stale cache intentionally.");
            }
            logInfo("compact denom cache miss: " + cacheStatus);
          }
        }
        if (!loadedCompactDenomCache) {
          std::vector<double> localDenom(baseNvox, 0.0);
          const bool ok =
              compactChunkLayout
                  ? spect::cuda::computeCompactChunkSensitivityCUDA(
                        *compactChunks, compactMeta, opts.threads,
                        localDenom.data(), &compactCudaTiming,
                        compactAttFactors)
                  : spect::cuda::computeCompactSymmetricSensitivityCUDA(
                        compactMatrix, compactMeta, opts.threads,
                        localDenom.data(), &compactCudaTiming);
          if (!ok) {
            throw std::runtime_error(
                "compact-symmetric sensitivity requires CUDA sparse backend.");
          }
          if (compactMultiBedGlobalPsf) {
            for (int bed = 0; bed < bedCount; ++bed) {
              spect::app::accumulateBedVolumeToGlobal(
                  denom, localDenom, baseNx, baseNy, baseNz,
                  bedZStarts[static_cast<size_t>(bed)]);
              observeCompactMemory();
            }
            logInfo(formatStatsEntry("compact_denom_pre_psf_global",
                                     computeVectorStats(denom)));
            apply_psf_backward(denom, args.Nx, args.Ny, args.Nz, opts, psfPtr,
                               opts.threads);
            logInfo(formatStatsEntry("compact_denom_post_psf_global",
                                     computeVectorStats(denom)));
          } else {
            apply_psf_backward(localDenom, baseNx, baseNy, baseNz, opts, psfPtr,
                               opts.threads);
            for (int bed = 0; bed < bedCount; ++bed) {
              spect::app::accumulateBedVolumeToGlobal(
                  denom, localDenom, baseNx, baseNy, baseNz,
                  bedZStarts[static_cast<size_t>(bed)]);
              observeCompactMemory();
            }
          }

          if (args.josephPasses > 0) {
            applyJosephLike(denom, args.Nx, args.Ny, args.Nz, args.josephPasses,
                            opts.threads, false);
            logInfo("Denom smoothing (Joseph-like) complete.");
          }
          if (!compactDenomCachePath.empty()) {
            try {
              saveCompactDenomCache(compactDenomCachePath, compactDenomManifest,
                                    geometry, denom);
              logInfo("compact denom cache saved: " +
                      compactDenomCachePath.string());
            } catch (const std::exception &ex) {
              logWarn(std::string("Failed saving compact denom cache: ") +
                      ex.what());
            }
          }
        }
        compactTiming.compactSensitivitySeconds +=
            std::chrono::duration<double>(ProfileClock::now() - sensStart)
                .count();
        const VectorStats denomStats = computeVectorStats(denom);
        logInfo(formatStatsEntry("compact_denom", denomStats));
        if (denomStats.count == 0 || denomStats.finite != denomStats.count ||
            denomStats.nonzero == 0 || !(denomStats.sum > 0.0)) {
          throw std::runtime_error(
              "compact-symmetric denominator is empty, non-finite, or zero.");
        }
        const VectorStats projectionStats = computeVectorStats(projections);
        logInfo(formatStatsEntry("compact_projection", projectionStats));
        const bool compactPositiveProjectionInput =
            projectionStats.count > 0 &&
            projectionStats.finite == projectionStats.count &&
            projectionStats.nonzero > 0 && projectionStats.sum > 0.0;

        std::vector<double> image(globalNvox, 1.0);
        std::vector<double> gradReg(globalNvox, 0.0);
        std::vector<double> curvMap;
        std::vector<double> kappaMap =
            (args.algorithm == ReconArgs::Algorithm::BSREM)
                ? computeKappaMap(denom, args, maskPtr)
                : std::vector<double>();
        const bool useRegularization =
            args.algorithm == ReconArgs::Algorithm::BSREM && args.useReg &&
            args.reg.beta > 0.0 && args.reg.type != RegType::None;
        const bool useBsremPenaltyUpdate =
            args.algorithm == ReconArgs::Algorithm::BSREM && useRegularization;
        const std::vector<uint8_t> *maskForReg =
            (maskPtr && useRegularization && args.reg.skip_outside_mask)
                ? maskPtr
                : nullptr;
        bool compactChunkReverseOrderNext =
            compactChunkLayout && !loadedCompactDenomCache;
        auto lastIterationDone = ProfileClock::now();

        for (int it = 1; it <= args.numIters; ++it) {
          logInfo("Iteration " + std::to_string(it) + "/" +
                  std::to_string(args.numIters) +
                  " (compact-symmetric CUDA path)");
          std::vector<double> numer(globalNvox, 0.0);
          std::vector<double> globalPsfImage;
          const std::vector<double> *bedImageSource = &image;
          if (compactMultiBedGlobalPsf) {
            logInfo(formatStatsEntry(
                "compact_global_image.it" + std::to_string(it),
                computeVectorStats(image)));
            apply_psf_forward(image, globalPsfImage, args.Nx, args.Ny, args.Nz,
                              opts, psfPtr, opts.threads);
            logInfo(formatStatsEntry(
                "compact_global_psf_forward.it" + std::to_string(it),
                computeVectorStats(globalPsfImage)));
            bedImageSource = &globalPsfImage;
          }

          if (compactChunkLayout) {
            std::vector<std::vector<double>> localImages(
                static_cast<size_t>(bedCount));
            std::vector<std::vector<double>> localUsed(
                static_cast<size_t>(bedCount));
            std::vector<std::vector<double>> localNumers(
                static_cast<size_t>(bedCount),
                std::vector<double>(baseNvox, 0.0));
            std::vector<const double *> projectionPtrs(
                static_cast<size_t>(bedCount), nullptr);
            std::vector<const double *> backgroundPtrs(
                static_cast<size_t>(bedCount), nullptr);
            std::vector<const double *> imagePtrs(static_cast<size_t>(bedCount),
                                                  nullptr);
            std::vector<double *> numerPtrs(static_cast<size_t>(bedCount),
                                            nullptr);

            for (int bed = 0; bed < bedCount; ++bed) {
              const int zStart = bedZStarts[static_cast<size_t>(bed)];
              localImages[static_cast<size_t>(bed)] =
                  spect::app::extractBedVolumeFromGlobal(
                      *bedImageSource, baseNx, baseNy, baseNz, zStart);
              imagePtrs[static_cast<size_t>(bed)] =
                  localImages[static_cast<size_t>(bed)].data();
              if (compactMultiBedGlobalPsf) {
                logInfo(formatStatsEntry(
                    "compact_local_psf_forward.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localImages[static_cast<size_t>(bed)])));
              } else if (compactPsfActive) {
                logInfo(formatStatsEntry(
                    "compact_local_image.it" + std::to_string(it) + ".bed" +
                        std::to_string(bed),
                    computeVectorStats(localImages[static_cast<size_t>(bed)])));
                apply_psf_forward(localImages[static_cast<size_t>(bed)],
                                  localUsed[static_cast<size_t>(bed)], baseNx,
                                  baseNy, baseNz, opts, psfPtr, opts.threads);
                logInfo(formatStatsEntry(
                    "compact_psf_forward.it" + std::to_string(it) + ".bed" +
                        std::to_string(bed),
                    computeVectorStats(localUsed[static_cast<size_t>(bed)])));
                imagePtrs[static_cast<size_t>(bed)] =
                    localUsed[static_cast<size_t>(bed)].data();
              }

              const size_t projectionOffset =
                  static_cast<size_t>(bed) *
                  static_cast<size_t>(compactMeta.fullRows());
              projectionPtrs[static_cast<size_t>(bed)] =
                  projections.data() + projectionOffset;
              if (!backgroundProj.empty()) {
                backgroundPtrs[static_cast<size_t>(bed)] =
                    backgroundProj.data() + projectionOffset;
              }
              numerPtrs[static_cast<size_t>(bed)] =
                  localNumers[static_cast<size_t>(bed)].data();
            }

            std::vector<const uint16_t *> attPtrs;
            if (compactAttFactors != nullptr) {
              attPtrs.assign(static_cast<size_t>(bedCount), nullptr);
              attPtrs[0] = compactAttFactors;
            }
            const auto numerStart = ProfileClock::now();
            const bool ok = spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
                *compactChunks, compactMeta, projectionPtrs.data(),
                backgroundProj.empty() ? nullptr : backgroundPtrs.data(),
                imagePtrs.data(), bedCount, opts.threads, numerPtrs.data(),
                compactChunkReverseOrderNext, &compactCudaTiming,
                attPtrs.empty() ? nullptr : attPtrs.data());
            if (!ok) {
              throw std::runtime_error(
                  "compact-symmetric numerator requires CUDA sparse backend.");
            }
            compactChunkReverseOrderNext = !compactChunkReverseOrderNext;
            compactTiming.compactForwardRatioBackwardSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - numerStart)
                    .count();

            for (int bed = 0; bed < bedCount; ++bed) {
              const int zStart = bedZStarts[static_cast<size_t>(bed)];
              std::vector<double> &localNumer =
                  localNumers[static_cast<size_t>(bed)];
              if (compactPsfActive) {
                logInfo(formatStatsEntry(
                    "compact_local_numer_pre_psf.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localNumer)));
              }
              if (compactMultiBedGlobalPsf) {
                spect::app::accumulateBedVolumeToGlobal(
                    numer, localNumer, baseNx, baseNy, baseNz, zStart);
              } else {
                apply_psf_backward(localNumer, baseNx, baseNy, baseNz, opts,
                                   psfPtr, opts.threads);
                if (compactPsfActive) {
                  logInfo(formatStatsEntry(
                      "compact_local_numer_post_psf.it" + std::to_string(it) +
                          ".bed" + std::to_string(bed),
                      computeVectorStats(localNumer)));
                }
                spect::app::accumulateBedVolumeToGlobal(
                    numer, localNumer, baseNx, baseNy, baseNz, zStart);
              }
            }
            if (compactMultiBedGlobalPsf) {
              logInfo(formatStatsEntry(
                  "compact_numer_pre_psf_global.it" + std::to_string(it),
                  computeVectorStats(numer)));
              apply_psf_backward(numer, args.Nx, args.Ny, args.Nz, opts, psfPtr,
                                 opts.threads);
              logInfo(formatStatsEntry(
                  "compact_numer_post_psf_global.it" + std::to_string(it),
                  computeVectorStats(numer)));
            }
          } else {
            for (int bed = 0; bed < bedCount; ++bed) {
              const int zStart = bedZStarts[static_cast<size_t>(bed)];
              std::vector<double> localImage =
                  spect::app::extractBedVolumeFromGlobal(
                      *bedImageSource, baseNx, baseNy, baseNz, zStart);
              std::vector<double> localUsed;
              const double *localImageForProjection = localImage.data();
              if (compactMultiBedGlobalPsf) {
                logInfo(formatStatsEntry(
                    "compact_local_psf_forward.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localImage)));
              } else if (compactPsfActive) {
                logInfo(formatStatsEntry(
                    "compact_local_image.it" + std::to_string(it) + ".bed" +
                        std::to_string(bed),
                    computeVectorStats(localImage)));
                apply_psf_forward(localImage, localUsed, baseNx, baseNy, baseNz,
                                  opts, psfPtr, opts.threads);
                logInfo(formatStatsEntry(
                    "compact_psf_forward.it" + std::to_string(it) + ".bed" +
                        std::to_string(bed),
                    computeVectorStats(localUsed)));
                localImageForProjection = localUsed.data();
              }

              std::vector<double> localNumer(baseNvox, 0.0);
              const size_t projectionOffset =
                  static_cast<size_t>(bed) *
                  static_cast<size_t>(compactMeta.fullRows());
              const double *localBackground =
                  backgroundProj.empty() ? nullptr
                                         : backgroundProj.data() +
                                               projectionOffset;
              const auto numerStart = ProfileClock::now();
              const bool ok = spect::cuda::runCompactSymmetricMlemNumerCUDA(
                  compactMatrix, compactMeta, projections.data() + projectionOffset,
                  localBackground, localImageForProjection, opts.threads,
                  localNumer.data(), &compactCudaTiming);
              if (!ok) {
                throw std::runtime_error(
                    "compact-symmetric numerator requires CUDA sparse backend.");
              }
              compactTiming.compactForwardRatioBackwardSeconds +=
                  std::chrono::duration<double>(ProfileClock::now() -
                                                numerStart)
                      .count();

              if (compactPsfActive) {
                logInfo(formatStatsEntry(
                    "compact_local_numer_pre_psf.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localNumer)));
              }
              if (compactMultiBedGlobalPsf) {
                spect::app::accumulateBedVolumeToGlobal(
                    numer, localNumer, baseNx, baseNy, baseNz, zStart);
              } else {
                apply_psf_backward(localNumer, baseNx, baseNy, baseNz, opts,
                                   psfPtr, opts.threads);
                if (compactPsfActive) {
                  logInfo(formatStatsEntry(
                      "compact_local_numer_post_psf.it" + std::to_string(it) +
                          ".bed" + std::to_string(bed),
                      computeVectorStats(localNumer)));
                }
                spect::app::accumulateBedVolumeToGlobal(
                    numer, localNumer, baseNx, baseNy, baseNz, zStart);
              }
            }
            if (compactMultiBedGlobalPsf) {
              logInfo(formatStatsEntry(
                  "compact_numer_pre_psf_global.it" + std::to_string(it),
                  computeVectorStats(numer)));
              apply_psf_backward(numer, args.Nx, args.Ny, args.Nz, opts, psfPtr,
                                 opts.threads);
              logInfo(formatStatsEntry(
                  "compact_numer_post_psf_global.it" + std::to_string(it),
                  computeVectorStats(numer)));
            }
          }
          const VectorStats numerStats = computeVectorStats(numer);
          logInfo(formatStatsEntry("compact_numer.it" + std::to_string(it),
                                   numerStats));
          if (numerStats.count == 0 || numerStats.finite != numerStats.count ||
              (compactPositiveProjectionInput &&
               (numerStats.nonzero == 0 || !(numerStats.sum > 0.0)))) {
            throw std::runtime_error(
                "compact-symmetric numerator is empty, non-finite, or zero "
                "despite positive projection input.");
          }

          const std::vector<double> *gradPtr = nullptr;
          if (useRegularization) {
            const auto regStart = ProfileClock::now();
            if (!tryComputeRegGradientCUDA(image, args, opts, maskForReg,
                                           gradReg)) {
              spect::regularizer::compute_reg_gradient(
                  image, args.Nx, args.Ny, args.Nz, args.reg, maskForReg,
                  gradReg, opts.threads);
            }
            compactTiming.regGradientSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - regStart)
                    .count();
            ++compactTiming.regGradientCalls;
            gradPtr = &gradReg;
          }

          double bsremAlpha = 1.0;
          double bsremBeta = 0.0;
          const std::vector<double> *curvPtr = nullptr;
          const std::vector<double> *kappaPtr = nullptr;
          if (useBsremPenaltyUpdate) {
            const int k = it - 1;
            bsremAlpha =
                args.bsrem_alpha0 /
                (1.0 + static_cast<double>(args.bsrem_gamma) * k);
            bsremBeta = useRegularization ? args.reg.beta : 0.0;
            if (args.bsremUseCurvDenom) {
              const bool refreshCurvature =
                  curvMap.empty() || !args.bsremFreezeCurv ||
                  it <= std::max(1, args.bsremCurvWarmupIters);
              if (refreshCurvature) {
                curvMap.assign(globalNvox, 0.0);
                if (gradPtr) {
                  for (size_t i = 0; i < globalNvox; ++i) {
                    const double denomBase =
                        std::max(image[i], args.bsremCurvEps);
                    curvMap[i] = std::abs(gradReg[i]) / denomBase;
                  }
                }
              }
              curvPtr = &curvMap;
            }
            kappaPtr = args.bsremUseKappaMap ? &kappaMap : nullptr;
          }

          const auto updateStart = ProfileClock::now();
          constexpr double eps = 1e-12;
          if (!useBsremPenaltyUpdate) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(opts.threads)
#endif
            for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(globalNvox); ++j) {
              const size_t idx = static_cast<size_t>(j);
              if (maskPtr && !(*maskPtr)[idx]) {
                image[idx] = 0.0;
                continue;
              }
              const double denomEff = denom[idx];
              if (!(denomEff > 0.0)) {
                image[idx] = 0.0;
                continue;
              }
              double value = image[idx] * (numer[idx] / std::max(denomEff, eps));
              if (opts.nonneg && value < 0.0)
                value = 0.0;
              image[idx] = value;
            }
          } else {
            constexpr double maxUpdateFactor = 1.0;
            constexpr double minUpdateFactor = -0.9;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(opts.threads)
#endif
            for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(globalNvox); ++j) {
              const size_t idx = static_cast<size_t>(j);
              if (maskPtr && !(*maskPtr)[idx]) {
                image[idx] = 0.0;
                continue;
              }
              const double sensGlobal = denom[idx];
              if (!(sensGlobal > eps)) {
                image[idx] = 0.0;
                continue;
              }

              const double kappa = kappaPtr ? (*kappaPtr)[idx] : 1.0;
              const double termLikelihood = numer[idx] - denom[idx];
              const double termPenalty =
                  gradPtr ? bsremBeta * kappa * (*gradPtr)[idx] : 0.0;
              double denomEff = sensGlobal;
              if (curvPtr)
                denomEff += bsremBeta * std::max((*curvPtr)[idx], 0.0);

              double fCurr = image[idx];
              if (fCurr < eps)
                fCurr = eps;

              double updateFactor =
                  bsremAlpha * ((termLikelihood - termPenalty) / denomEff);
              if (updateFactor > maxUpdateFactor)
                updateFactor = maxUpdateFactor;
              if (updateFactor < minUpdateFactor)
                updateFactor = minUpdateFactor;

              double value = fCurr * (1.0 + updateFactor);
              if (value < eps)
                value = opts.nonneg ? 0.0 : eps;
              image[idx] = value;
            }
          }
          compactTiming.mlemUpdateSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - updateStart)
                  .count();

          applyJosephLike(image, args.Nx, args.Ny, args.Nz, args.josephPasses,
                          opts.threads, true);

          if (prepared.butterConf.enabled) {
            const auto butterStart = ProfileClock::now();
            filters::apply_butterworth_inplace(image, args.Nx, args.Ny,
                                               args.Nz, spacing,
                                               prepared.butterConf,
                                               opts.threads);
            compactTiming.butterworthSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - butterStart)
                    .count();
          }

          if (prepared.smoothConf.enabled) {
            applySeparableConvRuntime(image, args.Nx, args.Ny, args.Nz,
                                      prepared.smoothConf.kx,
                                      prepared.smoothConf.ky,
                                      prepared.smoothConf.kz, opts,
                                      opts.threads);
          }

          clampNonNegativeInPlace(image, opts.threads);
          if (maskPtr)
            spect::core::applyMaskInplace(image, *maskPtr);
          lastIterationDone = ProfileClock::now();
          observeCompactMemory();

          if (args.saveInterval > 0 &&
              (it % args.saveInterval == 0 || it == args.numIters)) {
            const auto saveStart = ProfileClock::now();
            saveVolumeOutput(image, geometry, args.outputDir, outputPrefix, it,
                             maskPtr, crop, true, outputTransformPtr);
            compactTiming.outputWriteSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - saveStart)
                    .count();
          }
        }

        if (args.saveInterval <= 0 ||
            (args.numIters % args.saveInterval) != 0) {
          const auto saveStart = ProfileClock::now();
          saveVolumeOutput(image, geometry, args.outputDir, outputPrefix,
                           args.numIters, maskPtr, crop, true,
                           outputTransformPtr);
          compactTiming.outputWriteSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - saveStart)
                  .count();
          observeCompactMemory();
        }

        compactTiming.totalLoadToIterationDoneSeconds =
            std::chrono::duration<double>(lastIterationDone -
                                          loadToIterationStart)
                .count();
        compactTiming.fullCommandWallSeconds =
            std::chrono::duration<double>(ProfileClock::now() -
                                          fullCommandStart)
                .count();

        logInfo(formatProfileEntry("projection_load",
                                   compactTiming.projectionLoadSeconds, 1));
        logInfo(formatProfileEntry("csr_map_or_load",
                                   compactTiming.csrMapOrLoadSeconds, 1));
        logInfo(formatProfileEntry(
            "metadata_load_and_validate",
            compactTiming.metadataLoadAndValidateSeconds, 1));
        logInfo(formatProfileEntry("psf_prepare",
                                   compactTiming.psfPrepareSeconds, 1));
        logInfo(formatProfileEntry("compact_sensitivity",
                                   compactTiming.compactSensitivitySeconds,
                                   compactCudaTiming.sensitivityCalls));
        logInfo(formatProfileEntry(
            "compact_forward_ratio_backward",
            compactTiming.compactForwardRatioBackwardSeconds,
            compactCudaTiming.numeratorCalls));
        logInfo(formatProfileEntry("compact.transform_forward",
                                   compactCudaTiming.transformForwardSeconds,
                                   compactCudaTiming.numeratorCalls));
        logInfo(formatProfileEntry("compact.forward",
                                   compactCudaTiming.forwardSeconds,
                                   compactCudaTiming.numeratorCalls));
        logInfo(formatProfileEntry(
            "compact.ratio", compactCudaTiming.ratioSeconds,
            compactCudaTiming.numeratorCalls + compactCudaTiming.sensitivityCalls));
        logInfo(formatProfileEntry(
            "compact.backward", compactCudaTiming.backwardSeconds,
            compactCudaTiming.numeratorCalls + compactCudaTiming.sensitivityCalls));
        logInfo(formatProfileEntry(
            "compact.transform_backward",
            compactCudaTiming.transformBackwardSeconds,
            compactCudaTiming.numeratorCalls + compactCudaTiming.sensitivityCalls));
        if (compactTiming.regGradientCalls > 0) {
          logInfo(formatProfileEntry("compact.reg_gradient",
                                     compactTiming.regGradientSeconds,
                                     compactTiming.regGradientCalls));
        }
        logInfo(formatProfileEntry(useBsremPenaltyUpdate
                                       ? "bsrem_update"
                                       : "mlem_update",
                                   compactTiming.mlemUpdateSeconds,
                                   static_cast<size_t>(args.numIters)));
        logInfo(formatProfileEntry("butterworth",
                                   compactTiming.butterworthSeconds,
                                   prepared.butterConf.enabled
                                       ? static_cast<size_t>(args.numIters)
                                       : 0));
        logInfo(formatProfileEntry("output_write",
                                   compactTiming.outputWriteSeconds, 1));
        logInfo(formatProfileEntry(
            "total_load_to_iteration_done",
            compactTiming.totalLoadToIterationDoneSeconds, 1));
        logInfo(formatProfileEntry("full_command_wall_time",
                                   compactTiming.fullCommandWallSeconds, 1));
        const spect::cuda::CudaChunkStageTimingSnapshot cudaChunkTiming =
            spect::cuda::getCudaChunkStageTimingSnapshotCUDA();
        logCudaChunkTimingSnapshot(cudaChunkTiming);
        logInfo(formatMemoryEntry("compact_peak_memory", compactPeakMemory));
        return image;
#endif
      }

      std::vector<double> denom(globalNvox, 0.0);
      // PSF operator model is P = A * G, so sensitivity must be G^T A^T 1
      // (matches numerator G^T A^T r and the compact-symmetric path).
      if (!opts.bedAttenFactors.empty()) {
        for (int bed = 0; bed < bedCount; ++bed) {
          std::vector<double> localDenom = computeNormAT_Atten<MatrixType>(
              matrix, opts.bedAttenFactors[static_cast<size_t>(bed)],
              args.numAngles, opts.threads);
          apply_psf_backward(localDenom, baseNx, baseNy, baseNz, opts, psfPtr,
                             opts.threads);
          spect::app::accumulateBedVolumeToGlobal(
              denom, localDenom, baseNx, baseNy, baseNz,
              bedZStarts[static_cast<size_t>(bed)]);
        }
      } else {
        std::vector<double> localDenom = computeNormAT(matrix, opts.threads);
        apply_psf_backward(localDenom, baseNx, baseNy, baseNz, opts, psfPtr,
                           opts.threads);
        for (int bed = 0; bed < bedCount; ++bed) {
          spect::app::accumulateBedVolumeToGlobal(
              denom, localDenom, baseNx, baseNy, baseNz,
              bedZStarts[static_cast<size_t>(bed)]);
        }
      }

      const std::vector<std::vector<int>> subsets =
          make_simple_subsets(matrix.rows * bedCount, opts.n_subsets);
      if (subsets.empty())
        throw std::runtime_error("Failed to create projection subsets.");
      const auto isOrderedFullSubsetRows =
          [&](const std::vector<int> &subsetRows) {
            const int64_t totalRows =
                static_cast<int64_t>(matrix.rows) *
                static_cast<int64_t>(bedCount);
            if (subsetRows.size() != static_cast<size_t>(totalRows))
              return false;
            for (int64_t row = 0; row < totalRows; ++row) {
              if (subsetRows[static_cast<size_t>(row)] != row)
                return false;
            }
            return true;
          };
      const bool reuseFullSubsetBsremDenom =
          args.algorithm == ReconArgs::Algorithm::BSREM &&
          opts.n_subsets == 1 && isOrderedFullSubsetRows(subsets.front());

      std::vector<double> image(globalNvox, 1.0);
      std::vector<double> gradReg(globalNvox, 0.0);
      std::vector<double> subsetDenom;
      std::vector<double> bsremBuffer(globalNvox, 0.0);
      std::vector<double> curvMap;
      std::vector<double> kappaMap = computeKappaMap(denom, args, maskPtr);
      const std::vector<uint8_t> *maskForReg =
          (maskPtr && args.useReg && args.reg.skip_outside_mask) ? maskPtr : nullptr;
          const bool useRegularization =
              args.useReg && args.reg.beta > 0.0 && args.reg.type != RegType::None;
          ReconstructionStageTiming stageTiming;
          resetOSEMStageTiming();
          resetBSREMStageTiming();
          spect::cuda::resetCudaChunkStageTimingCUDA();

      // --- Create optimizer ---
      std::unique_ptr<IOptimizer> optimizer;
      if (args.algorithm == ReconArgs::Algorithm::BSREM) {
        optimizer = std::make_unique<BSREMOptimizer>();
      } else if (args.algorithm == ReconArgs::Algorithm::ADMM) {
        ADMMConfig admmCfg;
        admmCfg.rho = args.admmRho;
        admmCfg.reg = args.reg;
        admmCfg.beta = useRegularization ? args.reg.beta : 0.0;
        admmCfg.mask = maskForReg;
        admmCfg.threads = opts.threads;
        admmCfg.tvInnerIters = args.admmTvInner;
        admmCfg.adaptive = args.admmAdaptive;
        auto admm = std::make_unique<ADMMOptimizer>(std::move(admmCfg));
        admm->setDimensions(args.Nx, args.Ny, args.Nz);
        optimizer = std::move(admm);
      } else {
        optimizer = std::make_unique<MLEMOptimizer>();
      }

      if (args.josephPasses > 0) {
        applyJosephLike(denom, args.Nx, args.Ny, args.Nz, args.josephPasses,
                        opts.threads, false);
        logInfo("Denom smoothing (Joseph-like) complete.");
      }
      if (reuseFullSubsetBsremDenom) {
        logInfo("BSREM full-subset sensitivity reuses precomputed denom.");
      }

      const OutputGeometry geometry = outputGeometry;
      const std::string outputPrefix = outputPrefixFromArgs(args);
      const bool canProjPSF = opts.psf.mode == filters::PSFMode::ProjDomain &&
                              opts.n_subsets == 1 && bedCount == 1 &&
                              psfPtr != nullptr && opts.det_nz > 0 && opts.det_nx > 0 &&
                              static_cast<size_t>(opts.det_nz) *
                                      static_cast<size_t>(opts.det_nx) *
                                      static_cast<size_t>(opts.num_angles) ==
                                  projections.size();

      if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
        if (resolvedBackend == SparseBackendChoice::CUDA && args.numIters > 1 &&
            !canProjPSF) {
          const auto warmupStart = ProfileClock::now();
          const bool warmed = spect::cuda::warmupFullSubsetChunkCacheCUDA(
              matrix, bedCount,
              opts.bedAttenFactors.empty() ? nullptr : &opts.bedAttenFactors,
              args.numAngles, opts.threads);
          if (warmed) {
            stageTiming.cudaChunkWarmupSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - warmupStart)
                    .count();
            ++stageTiming.cudaChunkWarmupCalls;
            logInfo("CUDA chunk warmup complete.");
          }
        }
      }

      for (int it = 1; it <= args.numIters; ++it) {
        logInfo("Iteration " + std::to_string(it) + "/" +
                std::to_string(args.numIters) +
                (canProjPSF ? " (Proj-domain PSF path)" : " (standard path)"));

        if (canProjPSF) {
          const std::vector<double> *gradPtr = nullptr;
          if (useRegularization) {
            const auto stageStart = ProfileClock::now();
            if (!tryComputeRegGradientCUDA(image, args, opts, maskForReg,
                                           gradReg)) {
              spect::regularizer::compute_reg_gradient(
                  image, args.Nx, args.Ny, args.Nz, args.reg, maskForReg,
                  gradReg, opts.threads);
            }
            stageTiming.regGradientSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
            ++stageTiming.regGradientCalls;
            gradPtr = &gradReg;
          }
          runProjDomainMLEMStep(matrix, projections, image, denom, args, opts,
                                *psfPtr, gradPtr, maskForReg, maskPtr,
                                prepared.butterConf, prepared.smoothConf,
                                spacing);
        } else {
          for (int subsetIndex = 0; subsetIndex < opts.n_subsets; ++subsetIndex) {
            const std::vector<int> &subsetRows =
                subsets[static_cast<size_t>(subsetIndex)];
            const std::vector<double> *gradPtr = nullptr;
            if (useRegularization) {
              const auto stageStart = ProfileClock::now();
              if (!tryComputeRegGradientCUDA(image, args, opts, maskForReg,
                                             gradReg)) {
                spect::regularizer::compute_reg_gradient(
                    image, args.Nx, args.Ny, args.Nz, args.reg, maskForReg,
                    gradReg, opts.threads);
              }
              stageTiming.regGradientSeconds +=
                  std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
              ++stageTiming.regGradientCalls;
              gradPtr = &gradReg;
            }

            // -- BSREM pre-processing (no-op for other algorithms) --
            const std::vector<double> *activeSubsetDenom = &denom;
            double bsremAlpha = 1.0;
            double bsremBetaScaled = 0.0;
            const std::vector<double> *curvPtr = nullptr;
            const std::vector<double> *kappaPtr = nullptr;

            if (args.algorithm == ReconArgs::Algorithm::BSREM) {
              const int k = (it - 1) * opts.n_subsets + subsetIndex;
              bsremAlpha =
                  args.bsrem_alpha0 /
                  (1.0 + static_cast<double>(args.bsrem_gamma) * k);
              bsremBetaScaled =
                  useRegularization
                      ? (args.reg.beta / static_cast<double>(opts.n_subsets))
                      : 0.0;
              if (reuseFullSubsetBsremDenom) {
                activeSubsetDenom = &denom;
              } else {
                const auto subsetSensitivityStart = ProfileClock::now();
                subsetDenom = computeSubsetSensitivity<MatrixType>(
                    matrix, subsetRows, static_cast<int>(globalNvox),
                    opts.threads, opts);
                stageTiming.subsetSensitivitySeconds +=
                    std::chrono::duration<double>(ProfileClock::now() -
                                                  subsetSensitivityStart)
                        .count();
                ++stageTiming.subsetSensitivityCalls;
                apply_psf_backward(subsetDenom, args.Nx, args.Ny, args.Nz,
                                   opts, psfPtr, opts.threads);
                if (args.josephPasses > 0) {
                  applyJosephLike(subsetDenom, args.Nx, args.Ny, args.Nz,
                                  args.josephPasses, opts.threads, false);
                }
                activeSubsetDenom = &subsetDenom;
              }

              if (args.bsremUseCurvDenom) {
                const bool refreshCurvature =
                    curvMap.empty() || !args.bsremFreezeCurv ||
                    it <= std::max(1, args.bsremCurvWarmupIters);
                if (refreshCurvature) {
                  curvMap.assign(globalNvox, 0.0);
                  for (size_t i = 0; i < globalNvox; ++i) {
                    const double denomBase =
                        std::max(image[i], args.bsremCurvEps);
                    curvMap[i] = std::abs(gradReg[i]) / denomBase;
                  }
                }
                curvPtr = &curvMap;
              }

              kappaPtr = args.bsremUseKappaMap ? &kappaMap : nullptr;
            }

            // -- Build StepParams and dispatch via optimizer ----
            StepParams sp;
            if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
              sp.matrixView = &matrix;
            } else {
              sp.matrixCSR = &matrix;
            }
            sp.subsetRows = &subsetRows;
            sp.projections = &projections;
            sp.image = &image;
            sp.denom = &denom;
            sp.Nx = args.Nx;
            sp.Ny = args.Ny;
            sp.Nz = args.Nz;
            sp.opts = &opts;
            sp.psfKernels = psfPtr;
            sp.gradReg = gradPtr;
            sp.beta = useRegularization ? args.reg.beta : 0.0;
            sp.subsetDenom = activeSubsetDenom;
            sp.bsremAlpha = bsremAlpha;
            sp.bsremBetaScaled = bsremBetaScaled;
            sp.curvature = curvPtr;
            sp.kappa = kappaPtr;
            sp.bsremBuffer = &bsremBuffer;

            const auto stepStart = ProfileClock::now();
            optimizer->step(sp);
            const double stepSec =
                std::chrono::duration<double>(ProfileClock::now() - stepStart)
                    .count();
            if (args.algorithm == ReconArgs::Algorithm::BSREM) {
              stageTiming.bsremCoreSeconds += stepSec;
              ++stageTiming.bsremCoreCalls;
            } else {
              stageTiming.osemCoreSeconds += stepSec;
              ++stageTiming.osemCoreCalls;
            }

            applyJosephLike(image, args.Nx, args.Ny, args.Nz,
                            args.josephPasses, opts.threads, true);

            if (prepared.butterConf.enabled &&
                prepared.butterConf.apply_after_each_subset) {
              const auto stageStart = ProfileClock::now();
              filters::apply_butterworth_inplace(
                  image, args.Nx, args.Ny, args.Nz, spacing,
                  prepared.butterConf, opts.threads);
              stageTiming.butterworthSeconds +=
                  std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
              ++stageTiming.butterworthCalls;
            }
          }
        }

        // --- Post-iteration (ADMM u/d updates; no-op for other algorithms) ---
        optimizer->postIteration(image, it);

        if (prepared.butterConf.enabled &&
            !prepared.butterConf.apply_after_each_subset) {
          const auto stageStart = ProfileClock::now();
          filters::apply_butterworth_inplace(image, args.Nx, args.Ny,
                                             args.Nz, spacing,
                                             prepared.butterConf,
                                             opts.threads);
          stageTiming.butterworthSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
          ++stageTiming.butterworthCalls;
        }

        if (prepared.smoothConf.enabled) {
          const auto stageStart = ProfileClock::now();
          applySeparableConvRuntime(image, args.Nx, args.Ny, args.Nz,
                                    prepared.smoothConf.kx,
                                    prepared.smoothConf.ky,
                                    prepared.smoothConf.kz, opts,
                                    opts.threads);
          if (maskPtr)
            spect::core::applyMaskInplace(image, *maskPtr);
                            stageTiming.smoothingSeconds +=
                              std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
                            ++stageTiming.smoothingCalls;
        }

        clampNonNegativeInPlace(image, opts.threads);
        if (maskPtr)
          spect::core::applyMaskInplace(image, *maskPtr);

        if (args.saveInterval > 0 &&
            (it % args.saveInterval == 0 || it == args.numIters)) {
          const auto stageStart = ProfileClock::now();
          saveVolumeOutput(image, geometry, args.outputDir, outputPrefix, it,
                           maskPtr, crop, true, outputTransformPtr);
          stageTiming.saveSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
          ++stageTiming.saveCalls;
          logDebug("Saved volume at iteration " + std::to_string(it));
        }
      }

      if (args.saveInterval <= 0 || (args.numIters % args.saveInterval) != 0) {
        const auto stageStart = ProfileClock::now();
        saveVolumeOutput(image, geometry, args.outputDir, outputPrefix,
                         args.numIters, maskPtr, crop, true,
                         outputTransformPtr);
        stageTiming.saveSeconds +=
            std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
        ++stageTiming.saveCalls;
      }

      if (useRegularization) {
        const double penalty = spect::regularizer::compute_reg_energy(
            image, args.Nx, args.Ny, args.Nz, args.reg, maskForReg);
        const auto metric = app::computePoissonLogLikelihoodRuntime(
          matrix, image, projections, opts, args.Nx, args.Ny, args.Nz,
          psfPtr, opts.threads);
        logInfo(app::formatMetricLine(args.reg.beta, metric.likelihood,
                        penalty, metric.modelTag));
      }

      logInfo(formatProfileEntry("reg_gradient", stageTiming.regGradientSeconds,
                                 stageTiming.regGradientCalls));
      logInfo(formatProfileEntry("subset_sensitivity",
                                 stageTiming.subsetSensitivitySeconds,
                                 stageTiming.subsetSensitivityCalls));
      logInfo(formatProfileEntry("osem_core", stageTiming.osemCoreSeconds,
                                 stageTiming.osemCoreCalls));
      logInfo(formatProfileEntry("bsrem_core", stageTiming.bsremCoreSeconds,
                                 stageTiming.bsremCoreCalls));
      logInfo(formatProfileEntry("butterworth", stageTiming.butterworthSeconds,
                                 stageTiming.butterworthCalls));
      logInfo(formatProfileEntry("smoothing", stageTiming.smoothingSeconds,
                                 stageTiming.smoothingCalls));
      logInfo(formatProfileEntry("save_volume", stageTiming.saveSeconds,
                                 stageTiming.saveCalls));
      logInfo(formatProfileEntry("cuda.chunk_warmup",
                                 stageTiming.cudaChunkWarmupSeconds,
                                 stageTiming.cudaChunkWarmupCalls));

      const BSREMStageTimingSnapshot bsremTiming = getBSREMStageTimingSnapshot();
      const OSEMStageTimingSnapshot osemTiming = getOSEMStageTimingSnapshot();
      logInfo(formatProfileEntry("osem.input_psf", osemTiming.inputPSFSeconds,
                                 osemTiming.calls));
      logInfo(formatProfileEntry("osem.forward", osemTiming.forwardSeconds,
                                 osemTiming.calls));
      logInfo(formatProfileEntry("osem.backprojection",
                                 osemTiming.backprojectionSeconds,
                                 osemTiming.calls));
      logInfo(formatProfileEntry("osem.cuda_fused",
                                 osemTiming.cudaFusedSeconds,
                                 osemTiming.cudaFusedCalls));
      logInfo(formatProfileEntry("osem.buffer_psf",
                                 osemTiming.bufferPSFSeconds,
                                 osemTiming.calls));
      logInfo(formatProfileEntry("osem.update", osemTiming.updateSeconds,
                                 osemTiming.calls));
      logInfo(formatProfileEntry("bsrem.input_psf", bsremTiming.inputPSFSeconds,
                                 bsremTiming.calls));
      logInfo(formatProfileEntry("bsrem.forward", bsremTiming.forwardSeconds,
                                 bsremTiming.calls));
      logInfo(formatProfileEntry("bsrem.backprojection",
                                 bsremTiming.backprojectionSeconds,
                                 bsremTiming.calls));
      logInfo(formatProfileEntry("bsrem.cuda_fused",
                                 bsremTiming.cudaFusedSeconds,
                                 bsremTiming.cudaFusedCalls));
      logInfo(formatProfileEntry("bsrem.buffer_psf",
                                 bsremTiming.bufferPSFSeconds,
                                 bsremTiming.calls));
      logInfo(formatProfileEntry("bsrem.update", bsremTiming.updateSeconds,
                                 bsremTiming.calls));
      const spect::cuda::CudaChunkStageTimingSnapshot cudaChunkTiming =
          spect::cuda::getCudaChunkStageTimingSnapshotCUDA();
      logCudaChunkTimingSnapshot(cudaChunkTiming);

      if (!args.attMapPath.empty() && !args.doCTAC) {
        const spect::main::CTACModel changModel =
            (args.ctacModel == "per-voxel")
                ? spect::main::CTACModel::PerVoxelApprox
                : spect::main::CTACModel::LineIntegral;
        if (changModel == spect::main::CTACModel::PerVoxelApprox) {
          logWarn("Chang's AC output uses the legacy per-voxel exp(-mu) "
                  "approximation (-ctacModel per-voxel).");
        } else {
          logInfo("Chang's AC output uses angle-averaged parallel-ray line "
                  "integrals of the attenuation map (mu in mm^-1).");
        }
        std::vector<double> muMap;
        if (!sharedAnatomical.empty() && args.attMapPath == args.priorPath) {
          muMap = sharedAnatomical;
        } else {
          muMap = loadVolumeChecked(args.attMapPath, "Attenuation map", args.Nx,
                                    args.Ny, args.Nz);
        }
        std::vector<double> acImage = image;
        applyChangsAC(acImage, muMap, args.Nx, args.Ny, args.Nz, spacing.dx,
                      spacing.dy, spacing.dz, args.numAngles, opts.threads,
                      changModel, args.ctacRotSign);
        saveVolumeOutput(acImage, geometry, args.outputDir,
                         outputPrefix + "_AC", -1, maskPtr, crop, true,
                         outputTransformPtr);
      }

      return image;
    };

    if (compactChunkLayout)
      (void)runReconstruction(spect::io::CSRView{});
    else if (useMappedCSR)
      (void)runReconstruction(csrMap.view());
    else
      (void)runReconstruction(sysCSR);

    logInfo("Reconstruction finished successfully.");
    return spect::cli::kExitSuccess;
  } catch (const std::exception &ex) {
    std::cerr << "[FATAL] " << ex.what() << '\n';
    return spect::cli::kExitRuntimeError;
  }
}

} // namespace spect::main
