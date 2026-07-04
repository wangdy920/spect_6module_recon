#include "sparse/SparseOpsBackend.h"
#include "core/AttenuationConstants.h"
#include "cuda/CudaDriverHelpers.h"
#include "cuda/ProjDomainChunked.h"
#include "cuda/SparseOpsCudaKernels.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)

#include "CudaCsrBuilder.h"
#include "cuda/CudaConfig.h"
#include "filters/Convolution.h"

#if CUDA_ENABLED

#include <cuda.h>
#include <cusparse.h>
#include <nvrtc.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <memory>
#include <thread>
#include <cstring>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

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

bool isFullSingleBedSubset(const std::vector<int> &subset_rows,
                           int64_t rowCount,
                           const SparseSubsetContext &ctx) {
  if (ctx.bedCount != 1) {
    return false;
  }
  if (ctx.bedOffsetVoxels && !ctx.bedOffsetVoxels->empty() &&
      (*ctx.bedOffsetVoxels)[0] != 0) {
    return false;
  }
  if (ctx.bedAttenFactors && !ctx.bedAttenFactors->empty()) {
    return false;
  }
  if (subset_rows.size() != static_cast<size_t>(rowCount)) {
    return false;
  }
  for (int64_t row = 0; row < rowCount; ++row) {
    if (subset_rows[static_cast<size_t>(row)] != row) {
      return false;
    }
  }
  return true;
}

bool isOrderedFullSubset(const std::vector<int> &subset_rows,
                         int64_t totalRowCount) {
  if (subset_rows.size() != static_cast<size_t>(totalRowCount)) {
    return false;
  }
  for (int64_t row = 0; row < totalRowCount; ++row) {
    if (subset_rows[static_cast<size_t>(row)] != row) {
      return false;
    }
  }
  return true;
}

bool hasNoAttenuation(const SparseSubsetContext &ctx) {
  return !ctx.bedAttenFactors || ctx.bedAttenFactors->empty();
}

bool hasUsableAttenuation(const SparseSubsetContext &ctx, int64_t baseCols) {
  if (!ctx.bedAttenFactors || ctx.bedAttenFactors->empty() ||
      ctx.numAngles <= 0) {
    return false;
  }
  if (ctx.bedAttenFactors->size() !=
      static_cast<size_t>(std::max(1, ctx.bedCount))) {
    return false;
  }
  const size_t expectedSize =
      static_cast<size_t>(ctx.numAngles) * static_cast<size_t>(baseCols);
  for (const auto &atten : *ctx.bedAttenFactors) {
    if (atten.size() != expectedSize) {
      return false;
    }
  }
  return true;
}

bool hasUsableOffsets(const SparseSubsetContext &ctx, int64_t baseCols) {
  if (!ctx.bedOffsetVoxels || ctx.bedOffsetVoxels->empty()) {
    return false;
  }
  if (ctx.bedOffsetVoxels->size() !=
      static_cast<size_t>(std::max(1, ctx.bedCount))) {
    return false;
  }
  for (long long offset : *ctx.bedOffsetVoxels) {
    if (offset < 0 || offset > std::numeric_limits<int64_t>::max() - baseCols) {
      return false;
    }
  }
  return true;
}

bool isFullMultiBedSubset(const std::vector<int> &subset_rows,
                          int64_t rowCount,
                          const SparseSubsetContext &ctx) {
  if (ctx.bedCount <= 1) {
    return false;
  }
  if (!hasNoAttenuation(ctx)) {
    return false;
  }
  return isOrderedFullSubset(subset_rows,
                             rowCount * static_cast<int64_t>(ctx.bedCount));
}

bool isContiguousAscending(const std::vector<int> &rows) {
  if (rows.empty()) {
    return true;
  }
  const int firstRow = rows.front();
  for (size_t index = 0; index < rows.size(); ++index) {
    if (rows[index] != firstRow + static_cast<int>(index)) {
      return false;
    }
  }
  return true;
}

constexpr uint64_t kProjDomainChunkMinNnz = 8ull * 1024ull * 1024ull;
constexpr uint64_t kProjDomainChunkDefaultNnz = 48ull * 1024ull * 1024ull;
constexpr uint64_t kProjDomainChunkMaxNnz = 256ull * 1024ull * 1024ull;
// P3-A: 64GB system can afford 24GiB pinned cache — enough to hold all ~10
// float chunks (~20GB total) so chunk_build drops to zero after warmup.
constexpr uint64_t kPreparedChunkHostCacheMaxBytes =
    24ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t kPreparedChunkHostCacheMidBytes =
    8ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t kPreparedChunkHostCacheMinBytes =
    2ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t kPreparedChunkHostCacheEmergencyBytes =
    512ull * 1024ull * 1024ull;
// With P0 float (8B/nnz), chunks are smaller; 4GiB reserve is sufficient
// to keep the OS and other processes stable on a 64GB system.
constexpr uint64_t kPreparedChunkHostCacheSafetyReserveBytes =
    4ull * 1024ull * 1024ull * 1024ull;
constexpr size_t kTransientRangeSlotCount = 2;
constexpr int kCompactDeviceBackwardMaxAccumulators = 5;

cusparseSpMVAlg_t parseSpmvAlgName(const char *text,
                                   cusparseSpMVAlg_t fallback) {
  if (text == nullptr || *text == '\0') {
    return fallback;
  }
  std::string value(text);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (value == "default") {
    return CUSPARSE_SPMV_ALG_DEFAULT;
  }
  if (value == "csr1" || value == "csr_alg1" || value == "alg1") {
    return CUSPARSE_SPMV_CSR_ALG1;
  }
  if (value == "csr2" || value == "csr_alg2" || value == "alg2") {
    return CUSPARSE_SPMV_CSR_ALG2;
  }
  return fallback;
}

cusparseSpMVAlg_t selectedSpmvAlg(bool backward) {
  static const cusparseSpMVAlg_t globalAlg = parseSpmvAlgName(
      std::getenv("SPECT_CUDA_SPMV_ALG"), CUSPARSE_SPMV_ALG_DEFAULT);
  static const cusparseSpMVAlg_t forwardAlg = parseSpmvAlgName(
      std::getenv("SPECT_CUDA_SPMV_FWD_ALG"), globalAlg);
  static const cusparseSpMVAlg_t backwardAlg = parseSpmvAlgName(
      std::getenv("SPECT_CUDA_SPMV_BWD_ALG"), globalAlg);
  return backward ? backwardAlg : forwardAlg;
}

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

CudaChunkStageTimingState g_cudaChunkStageTiming;
std::atomic<uint64_t> g_cudaChunkNnzBudgetOverride{0};
std::atomic<uint64_t> g_cudaPreparedChunkHostCacheBytesOverride{0};

uint64_t recommendProjDomainChunkNnzBudget(size_t freeBytes) {
  constexpr uint64_t kBytesPerNnz =
      static_cast<uint64_t>(sizeof(int32_t) + sizeof(float));
  constexpr uint64_t kReserveBytes = 768ull * 1024ull * 1024ull;
  constexpr uint64_t kTargetBytesCap = 4ull * 1024ull * 1024ull * 1024ull;
  constexpr uint64_t kHighMemThreshold = 12ull * 1024ull * 1024ull * 1024ull;

  uint64_t usableBytes = static_cast<uint64_t>(freeBytes);
  if (usableBytes > kReserveBytes) {
    usableBytes -= kReserveBytes;
  }

  const uint64_t targetBytes = usableBytes >= kHighMemThreshold
                                   ? std::min<uint64_t>(usableBytes / 2ull,
                                                        kTargetBytesCap)
                                   : std::min<uint64_t>(usableBytes / 3ull,
                                                        3ull * 1024ull *
                                                            1024ull * 1024ull);
  if (targetBytes == 0) {
    return kProjDomainChunkDefaultNnz;
  }

  return std::clamp<uint64_t>(targetBytes / kBytesPerNnz,
                              kProjDomainChunkMinNnz,
                              kProjDomainChunkMaxNnz);
}

uint64_t selectCudaChunkNnzBudget() {
  const uint64_t overrideBudget =
      g_cudaChunkNnzBudgetOverride.load(std::memory_order_relaxed);
  if (overrideBudget > 0) {
    return overrideBudget;
  }
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

inline uint64_t estimatePreparedChunkHostBytes(const io::CSRView &matrix);

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

uint64_t selectPreparedChunkHostCacheBudgetCapBytes() {
  const SystemMemoryInfo memory = querySystemMemoryInfo();
  uint64_t cap = kPreparedChunkHostCacheMinBytes;
  // P3-A: Use 60 GiB threshold instead of 64 GiB — Windows reserves ~0.5 GiB
  // for hardware (BIOS, GPU BAR, etc.), so a 64 GB system reports ~63.8 GiB.
  if (memory.totalBytes >= 60ull * 1024ull * 1024ull * 1024ull) {
    cap = kPreparedChunkHostCacheMaxBytes;
  } else if (memory.totalBytes >= 28ull * 1024ull * 1024ull * 1024ull) {
    cap = kPreparedChunkHostCacheMidBytes;
  }
  // Code.CUDA.41: On high-RAM systems, Windows MEMORYSTATUSEX.ullAvailPhys
  // counts memory-mapped file pages (e.g. the 21 GB CSR file) as "in use",
  // even though file-backed pages are trivially evictable by the OS.
  // Use totalBytes*2/3 as a floor for effectiveAvailable to prevent the
  // cache budget from being suppressed below the matrix size.
  const uint64_t effectiveAvailable =
      (memory.totalBytes >= 60ull * 1024ull * 1024ull * 1024ull)
          ? std::max(memory.availableBytes, memory.totalBytes * 2ull / 3ull)
          : memory.availableBytes;
  if (effectiveAvailable > kPreparedChunkHostCacheSafetyReserveBytes) {
    cap = std::min<uint64_t>(
        cap, effectiveAvailable - kPreparedChunkHostCacheSafetyReserveBytes);
  } else {
    cap = std::min<uint64_t>(cap, kPreparedChunkHostCacheEmergencyBytes);
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
    size_t activePreparedCaches = 1) {
  const uint64_t overrideBytes =
      g_cudaPreparedChunkHostCacheBytesOverride.load(std::memory_order_relaxed);
  if (overrideBytes > 0) {
    return splitPreparedChunkHostCacheBudgetBytes(overrideBytes,
                                                  activePreparedCaches);
  }
  const uint64_t cap = selectPreparedChunkHostCacheBudgetCapBytes();
  if (cap == 0) {
    return splitPreparedChunkHostCacheBudgetBytes(
        kPreparedChunkHostCacheEmergencyBytes, activePreparedCaches);
  }
  return splitPreparedChunkHostCacheBudgetBytes(cap, activePreparedCaches);
}

uint64_t selectPreparedChunkHostCacheBudgetBytes(
    const io::CSRView &matrix, size_t activePreparedCaches = 1) {
  if (!matrix.valid() || matrix.row_ptr == nullptr) {
    return 0;
  }
  return selectPreparedChunkHostCacheBudgetBytes(activePreparedCaches);
}

struct ProjDomainKernelCache {
  CUmodule module = nullptr;
  CUfunction ratioFunction = nullptr;
  CUfunction compactRatioFunction = nullptr;
  CUfunction compactTransformBackwardFunction = nullptr;
  CUfunction compactBakeAttenFunction = nullptr;
  CUfunction convYFunction = nullptr;
  CUfunction convXFunction = nullptr;
  int device = -1;
  int computeMajor = -1;
  int computeMinor = -1;
  std::mutex mutex;

  ~ProjDomainKernelCache() {
    if (module != nullptr) {
      cuModuleUnload(module);
    }
  }
};

ProjDomainKernelCache g_projDomainKernelCache;

void ensureProjDomainKernelsLoaded() {
  CUDA_CHECK(cudaFree(nullptr));
  SPECT_CUDRV_CHECK(cuInit(0));

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));

  std::lock_guard<std::mutex> lock(g_projDomainKernelCache.mutex);
  if (g_projDomainKernelCache.ratioFunction != nullptr &&
      g_projDomainKernelCache.compactRatioFunction != nullptr &&
      g_projDomainKernelCache.compactTransformBackwardFunction != nullptr &&
      g_projDomainKernelCache.compactBakeAttenFunction != nullptr &&
      g_projDomainKernelCache.convYFunction != nullptr &&
      g_projDomainKernelCache.convXFunction != nullptr &&
      g_projDomainKernelCache.device == device &&
      g_projDomainKernelCache.computeMajor == props.major &&
      g_projDomainKernelCache.computeMinor == props.minor) {
    return;
  }

  const char *source = cuda::projectionDomainKernelSource();

  nvrtcProgram program;
  SPECT_NVRTC_CHECK(
      nvrtcCreateProgram(&program, source, "projdomain_chunk_ops.cu", 0,
                         nullptr, nullptr));

  std::string arch = "--gpu-architecture=compute_" +
                     std::to_string(props.major) +
                     std::to_string(props.minor);
  const char *options[] = {arch.c_str()};
  const nvrtcResult compileStatus = nvrtcCompileProgram(program, 1, options);

  size_t logSize = 0;
  SPECT_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &logSize));
  std::string log(logSize, '\0');
  if (logSize > 1) {
    SPECT_NVRTC_CHECK(nvrtcGetProgramLog(program, log.data()));
  }
  if (compileStatus != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error(std::string("NVRTC compile failed: ") + log);
  }

  size_t ptxSize = 0;
  SPECT_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptxSize));
  std::string ptx(ptxSize, '\0');
  SPECT_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
  SPECT_NVRTC_CHECK(nvrtcDestroyProgram(&program));

  CUcontext context = nullptr;
  SPECT_CUDRV_CHECK(cuCtxGetCurrent(&context));
  if (context == nullptr) {
    throw std::runtime_error(
        "CUDA context is not initialized for projection-domain helper kernels.");
  }

  if (g_projDomainKernelCache.module != nullptr) {
    SPECT_CUDRV_CHECK(cuModuleUnload(g_projDomainKernelCache.module));
    g_projDomainKernelCache.module = nullptr;
    g_projDomainKernelCache.ratioFunction = nullptr;
    g_projDomainKernelCache.compactRatioFunction = nullptr;
    g_projDomainKernelCache.compactTransformBackwardFunction = nullptr;
    g_projDomainKernelCache.compactBakeAttenFunction = nullptr;
    g_projDomainKernelCache.convYFunction = nullptr;
    g_projDomainKernelCache.convXFunction = nullptr;
  }

  SPECT_CUDRV_CHECK(cuModuleLoadDataEx(&g_projDomainKernelCache.module,
                                       ptx.data(), 0, nullptr, nullptr));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.ratioFunction, g_projDomainKernelCache.module,
      "compute_ratio_chunk"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactRatioFunction,
      g_projDomainKernelCache.module, "compute_compact_ratio_chunk"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactTransformBackwardFunction,
      g_projDomainKernelCache.module, "compact_transform_backward_add"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactBakeAttenFunction,
      g_projDomainKernelCache.module, "compact_bake_atten_values"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.convYFunction, g_projDomainKernelCache.module,
      "conv2d_y_mirror_frames"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.convXFunction, g_projDomainKernelCache.module,
      "conv2d_x_mirror_frames"));
  g_projDomainKernelCache.device = device;
  g_projDomainKernelCache.computeMajor = props.major;
  g_projDomainKernelCache.computeMinor = props.minor;
}

void launchProjDomainRatioKernel(const float *yhat, const float *projections,
                                 size_t projectionOffset, float *ratio,
                                 size_t count, float eps,
                                 cudaStream_t stream = nullptr) {
  ensureProjDomainKernelsLoaded();
  if (count == 0) {
    return;
  }
  CUdeviceptr yhatPtr = reinterpret_cast<CUdeviceptr>(const_cast<float *>(yhat));
  CUdeviceptr projPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(projections));
  CUdeviceptr ratioPtr = reinterpret_cast<CUdeviceptr>(ratio);
  unsigned long long offset = static_cast<unsigned long long>(projectionOffset);
  unsigned long long total = static_cast<unsigned long long>(count);
  void *args[] = {&yhatPtr, &projPtr, &offset, &ratioPtr, &total, &eps};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((count + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(g_projDomainKernelCache.ratioFunction,
                                   gridSize, 1, 1, blockSize, 1, 1, 0,
                                   reinterpret_cast<CUstream>(stream),
                                   args, nullptr));
}

void launchCompactRatioKernel(
    const float *yhat, const float *fullProjection, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &params, float *ratio, size_t count,
    float eps, cudaStream_t stream = nullptr) {
  ensureProjDomainKernelsLoaded();
  if (count == 0) {
    return;
  }
  CUdeviceptr yhatPtr = reinterpret_cast<CUdeviceptr>(const_cast<float *>(yhat));
  CUdeviceptr projectionPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(fullProjection));
  CUdeviceptr ratioPtr = reinterpret_cast<CUdeviceptr>(ratio);
  unsigned long long projectionCount =
      static_cast<unsigned long long>(fullProjectionCount);
  unsigned long long total = static_cast<unsigned long long>(count);
  int zMirror = params.zMirror ? 1 : 0;
  void *args[] = {&yhatPtr,
                  &projectionPtr,
                  &projectionCount,
                  const_cast<int *>(&params.symmetryMode),
                  const_cast<int *>(&params.compactRowBase),
                  const_cast<int *>(&params.compactRaysPerAngle),
                  const_cast<int *>(&params.compactNumAngles),
                  const_cast<int *>(&params.detNx),
                  const_cast<int *>(&params.detNz),
                  const_cast<int *>(&params.compactDetZ),
                  const_cast<int *>(&params.fullNumAngles),
                  const_cast<int *>(&params.fullRaysPerAngle),
                  const_cast<int *>(&params.symId),
                  &zMirror,
                  &ratioPtr,
                  &total,
                  &eps};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((count + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactRatioFunction, gridSize, 1, 1, blockSize,
      1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactTransformBackwardAddKernel(
    const float *rotated, double *accum, size_t nvox,
    const cuda::CompactDeviceTransformParams &params,
    cudaStream_t stream = nullptr) {
  ensureProjDomainKernelsLoaded();
  if (nvox == 0) {
    return;
  }
  CUdeviceptr rotatedPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(rotated));
  CUdeviceptr accumPtr = reinterpret_cast<CUdeviceptr>(accum);
  unsigned long long total = static_cast<unsigned long long>(nvox);
  int zMirror = params.zMirror ? 1 : 0;
  void *args[] = {&rotatedPtr,
                  &accumPtr,
                  &total,
                  const_cast<int *>(&params.symmetryMode),
                  const_cast<int *>(&params.nx),
                  const_cast<int *>(&params.ny),
                  const_cast<int *>(&params.nz),
                  const_cast<int *>(&params.symId),
                  &zMirror};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((nvox + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactTransformBackwardFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactBakeAttenKernel(const float *values, const int32_t *rowPtr,
                                  const int32_t *colIdx, const uint16_t *att,
                                  float *baked, int rows,
                                  const cuda::CompactAttenBakeParams &params,
                                  cudaStream_t stream = nullptr) {
  ensureProjDomainKernelsLoaded();
  if (rows <= 0) {
    return;
  }
  CUdeviceptr valuesPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(values));
  CUdeviceptr rowPtrPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<int32_t *>(rowPtr));
  CUdeviceptr colIdxPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<int32_t *>(colIdx));
  CUdeviceptr attPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint16_t *>(att));
  CUdeviceptr bakedPtr = reinterpret_cast<CUdeviceptr>(baked);
  int zMirror = params.zMirror ? 1 : 0;
  void *args[] = {&valuesPtr,
                  &rowPtrPtr,
                  &colIdxPtr,
                  &attPtr,
                  &bakedPtr,
                  &rows,
                  const_cast<int *>(&params.symmetryMode),
                  const_cast<int *>(&params.compactRowBase),
                  const_cast<int *>(&params.compactRaysPerAngle),
                  const_cast<int *>(&params.compactNumAngles),
                  const_cast<int *>(&params.fullNumAngles),
                  const_cast<int *>(&params.symId),
                  &zMirror,
                  const_cast<int *>(&params.nx),
                  const_cast<int *>(&params.ny),
                  const_cast<int *>(&params.nz)};
  const unsigned int blockSize = 256;
  const unsigned int gridSize = static_cast<unsigned int>(rows);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactBakeAttenFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchProjDomainConvYKernel(const float *input, float *output,
                                 const float *kernel, int radius, int detNz,
                                 int detNx, size_t frameCount,
                                 cudaStream_t stream = nullptr) {
  ensureProjDomainKernelsLoaded();
  const size_t total = frameCount * static_cast<size_t>(detNz) *
                       static_cast<size_t>(detNx);
  if (total == 0) {
    return;
  }
  CUdeviceptr inputPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(input));
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  CUdeviceptr kernelPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(kernel));
  unsigned long long frames = static_cast<unsigned long long>(frameCount);
  void *args[] = {&inputPtr, &outputPtr, &kernelPtr, &radius, &detNz, &detNx,
                  &frames};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(g_projDomainKernelCache.convYFunction,
                                   gridSize, 1, 1, blockSize, 1, 1, 0,
                                   reinterpret_cast<CUstream>(stream),
                                   args, nullptr));
}

void launchProjDomainConvXKernel(const float *input, float *output,
                                 const float *kernel, int radius, int detNz,
                                 int detNx, size_t frameCount,
                                 cudaStream_t stream = nullptr) {
  ensureProjDomainKernelsLoaded();
  const size_t total = frameCount * static_cast<size_t>(detNz) *
                       static_cast<size_t>(detNx);
  if (total == 0) {
    return;
  }
  CUdeviceptr inputPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(input));
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  CUdeviceptr kernelPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(kernel));
  unsigned long long frames = static_cast<unsigned long long>(frameCount);
  void *args[] = {&inputPtr, &outputPtr, &kernelPtr, &radius, &detNz, &detNx,
                  &frames};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(g_projDomainKernelCache.convXFunction,
                                   gridSize, 1, 1, blockSize, 1, 1, 0,
                                   reinterpret_cast<CUstream>(stream),
                                   args, nullptr));
}

class CudaStreamGuard {
public:
  CudaStreamGuard() = default;
  ~CudaStreamGuard() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  CudaStreamGuard(const CudaStreamGuard &) = delete;
  CudaStreamGuard &operator=(const CudaStreamGuard &) = delete;

  cudaStream_t get() {
    if (stream_ == nullptr) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
    return stream_;
  }

private:
  cudaStream_t stream_ = nullptr;
};

class CudaEventGuard {
public:
  CudaEventGuard() = default;
  ~CudaEventGuard() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  CudaEventGuard(const CudaEventGuard &) = delete;
  CudaEventGuard &operator=(const CudaEventGuard &) = delete;

  cudaEvent_t get() {
    if (event_ == nullptr) {
      CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDefault));
    }
    return event_;
  }

  void record(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(get(), stream)); }
  void wait(cudaStream_t stream) { CUDA_CHECK(cudaStreamWaitEvent(stream, get(), 0)); }

private:
  cudaEvent_t event_ = nullptr;
};

class TimedCudaStagePair {
public:
  TimedCudaStagePair() {
    CUDA_CHECK(cudaEventCreateWithFlags(&start_, cudaEventDefault));
    CUDA_CHECK(cudaEventCreateWithFlags(&stop_, cudaEventDefault));
  }

  ~TimedCudaStagePair() {
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
  }

  TimedCudaStagePair(const TimedCudaStagePair &) = delete;
  TimedCudaStagePair &operator=(const TimedCudaStagePair &) = delete;

  TimedCudaStagePair(TimedCudaStagePair &&other) noexcept
      : start_(other.start_), stop_(other.stop_) {
    other.start_ = nullptr;
    other.stop_ = nullptr;
  }

  TimedCudaStagePair &operator=(TimedCudaStagePair &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
    start_ = other.start_;
    stop_ = other.stop_;
    other.start_ = nullptr;
    other.stop_ = nullptr;
    return *this;
  }

  void recordStart(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
  void recordStop(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(stop_, stream)); }
  void synchronizeStop() const { CUDA_CHECK(cudaEventSynchronize(stop_)); }

  double elapsedSeconds() const {
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start_, stop_));
    return static_cast<double>(elapsedMs) / 1000.0;
  }

private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

double accumulateTimedCudaPairs(const std::vector<TimedCudaStagePair> &pairs) {
  double seconds = 0.0;
  for (const auto &pair : pairs) {
    seconds += pair.elapsedSeconds();
  }
  return seconds;
}

template <typename T> class DeviceArray {
public:
  DeviceArray() = default;

  explicit DeviceArray(size_t count) { allocate(count); }

  ~DeviceArray() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;

  T *get() const { return ptr_; }
  size_t size() const { return size_; }

  void allocate(size_t count) {
    if (count == 0) {
      return;
    }
    if (ptr_ != nullptr && count <= size_) {
      return;
    }
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      size_ = 0;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(T)));
    size_ = count;
  }

  void copyFromHost(const T *host, size_t count) {
    allocate(count);
    if (count == 0) {
      return;
    }
    CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copyFromHostAsync(const T *host, size_t count, cudaStream_t stream) {
    allocate(count);
    if (count == 0) {
      return;
    }
    CUDA_CHECK(cudaMemcpyAsync(ptr_, host, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

  void copyToHost(T *host, size_t count) const {
    if (count == 0) {
      return;
    }
    if (count > size_) {
      throw std::runtime_error("CUDA device buffer is smaller than requested host copy.");
    }
    CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void copyToHostAsync(T *host, size_t count, cudaStream_t stream) const {
    if (count == 0) {
      return;
    }
    if (count > size_) {
      throw std::runtime_error("CUDA device buffer is smaller than requested host copy.");
    }
    CUDA_CHECK(cudaMemcpyAsync(host, ptr_, count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
  }

  void reset() {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      size_ = 0;
    }
  }

private:
  T *ptr_ = nullptr;
  size_t size_ = 0;
};

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

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void ensure(size_t bytes) {
    if (bytes <= size_) {
      return;
    }
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      size_ = 0;
    }
    if (bytes == 0) {
      return;
    }
    CUDA_CHECK(cudaMalloc(&ptr_, bytes));
    size_ = bytes;
  }

  void *get() const { return ptr_; }

private:
  void *ptr_ = nullptr;
  size_t size_ = 0;
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
    // P3-A: Exclude hostCacheBudgetBytes from comparison.  The budget
    // depends on available RAM which drops after warmup pins memory,
    // causing iteration-time lookups to miss the warmed cache entry.
    // The budget only matters at initialization (maxResidentBytes).
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
    // P3-A: Same rationale as PreparedChunkRangeCacheKey — exclude budget
    // from comparison so pinned-memory allocation doesn't invalidate the key.
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

inline MatrixCacheKey makeCacheKey(const CSRView &matrix) {
  MatrixCacheKey key;
  key.rowPtr = matrix.row_ptr;
  key.colIdx = matrix.col_idx;
  key.values = matrix.values;
  key.rows = matrix.rows;
  key.cols = matrix.cols;
  key.nnz = matrix.nnz;
  return key;
}

inline MatrixCacheKey makeCacheKey(const io::CSRView &matrix) {
  MatrixCacheKey key;
  key.rowPtr = matrix.row_ptr;
  key.colIdx = matrix.col_idx;
  key.values = matrix.val;
  key.rows = matrix.rows;
  key.cols = matrix.cols;
  key.nnz = static_cast<int64_t>(matrix.nnz);
  return key;
}

inline SubsetCacheKey makeSubsetCacheKey(const io::CSRView &matrix,
                                         const std::vector<int> &subset_rows) {
  SubsetCacheKey key;
  key.matrix = makeCacheKey(matrix);
  key.subsetRows = subset_rows.empty() ? nullptr : subset_rows.data();
  key.subsetSize = subset_rows.size();
  return key;
}

inline AttenCacheKey makeAttenCacheKey(const io::CSRView &matrix,
                                       const std::vector<uint16_t> &atten,
                                       int numAngles) {
  AttenCacheKey key;
  key.matrix = makeCacheKey(matrix);
  key.atten = atten.data();
  key.numAngles = numAngles;
  return key;
}

inline RangeCacheKey makeRangeCacheKey(const io::CSRView &matrix, int rowBegin,
                                       int rowEnd) {
  RangeCacheKey key;
  key.matrix = makeCacheKey(matrix);
  key.rowBegin = rowBegin;
  key.rowEnd = rowEnd;
  return key;
}

inline ChunkRangeCacheKey makeChunkRangeCacheKey(const io::CSRView &matrix,
                                                 uint64_t nnzBudget) {
  ChunkRangeCacheKey key;
  key.matrix = makeCacheKey(matrix);
  key.nnzBudget = nnzBudget;
  return key;
}

inline PreparedChunkRangeCacheKey
makePreparedChunkRangeCacheKey(const io::CSRView &matrix, uint64_t nnzBudget,
                               uint64_t hostCacheBudgetBytes) {
  PreparedChunkRangeCacheKey key;
  key.range = makeChunkRangeCacheKey(matrix, nnzBudget);
  key.hostCacheBudgetBytes = hostCacheBudgetBytes;
  return key;
}

inline AttenChunkRangeCacheKey
makeAttenChunkRangeCacheKey(const io::CSRView &matrix,
                            const std::vector<uint16_t> &atten, int numAngles,
                            uint64_t nnzBudget) {
  AttenChunkRangeCacheKey key;
  key.range = makeChunkRangeCacheKey(matrix, nnzBudget);
  key.atten = atten.data();
  key.numAngles = numAngles;
  return key;
}

inline PreparedAttenChunkRangeCacheKey
makePreparedAttenChunkRangeCacheKey(const io::CSRView &matrix,
                                    const std::vector<uint16_t> &atten,
                                    int numAngles, uint64_t nnzBudget,
                                    uint64_t hostCacheBudgetBytes) {
  PreparedAttenChunkRangeCacheKey key;
  key.range = makeAttenChunkRangeCacheKey(matrix, atten, numAngles, nnzBudget);
  key.hostCacheBudgetBytes = hostCacheBudgetBytes;
  return key;
}

inline void ensureCudaRuntimeAvailable() {
  if (!cuda::isCudaAvailable()) {
    throw std::runtime_error(
        "CUDA sparse backend was selected, but no CUDA device is available at runtime.");
  }
}

inline bool needsWholeMatrixChunking(const io::CSRView &matrix) {
  if (!matrix.valid() || matrix.row_ptr == nullptr) {
    return false;
  }
  const uint64_t nnz =
      static_cast<uint64_t>(matrix.row_ptr[static_cast<size_t>(matrix.rows)]);
  const uint64_t overrideBudget =
      g_cudaChunkNnzBudgetOverride.load(std::memory_order_relaxed);
  if (overrideBudget > 0 && nnz > overrideBudget) {
    return true;
  }
  return nnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
}

inline uint64_t estimatePreparedChunkHostBytes(const io::CSRView &matrix) {
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

inline uint64_t estimatePinnedHostChunkBytes(const PinnedHostCSRData &host) {
  return host.rowPtr.capacity() * sizeof(int32_t) +
         host.colIdx.capacity() * sizeof(int32_t) +
         host.values.capacity() * sizeof(float);
}

inline uint64_t estimateDeviceCsrBytes(int64_t rows, int64_t nnz) {
  if (rows <= 0 || nnz <= 0) {
    return 0;
  }
  return (static_cast<uint64_t>(rows) + 1ull) * sizeof(int32_t) +
         static_cast<uint64_t>(nnz) * sizeof(int32_t) +
         static_cast<uint64_t>(nnz) * sizeof(float);
}

inline void releasePreparedChunk(PreparedChunkedHostCSR &prepared,
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

inline void touchPreparedChunk(PreparedChunkedHostCSR &prepared,
                               size_t chunkIndex) {
  if (chunkIndex >= prepared.touchStamps.size()) {
    throw std::runtime_error("CUDA prepared chunk touch index out of range.");
  }
  prepared.touchStamps[chunkIndex] = prepared.nextTouchStamp++;
}

inline bool evictPreparedChunkLru(PreparedChunkedHostCSR &prepared,
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
inline const PinnedHostCSRData &ensurePreparedChunkResident(
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
inline void warmPreparedChunksUntilBudget(PreparedChunkedHostCSR &prepared,
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

inline void buildContiguousSubsetCSR(const io::CSRView &matrix, int rowBegin,
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

inline void buildAttenuatedContiguousSubsetCSR(
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

inline std::vector<RowChunkRange> computeContiguousRowChunks(
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

inline const std::vector<RowChunkRange> &getContiguousRowChunks(
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

inline void initializePreparedChunkedHostCacheEntry(
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

inline PreparedChunkedHostCSR *getPreparedContiguousHostChunks(
    const io::CSRView &matrix, uint64_t nnzBudget, int threads,
    bool eagerPrepare = false, uint64_t hostCacheBudgetBytes = 0) {
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

inline const PinnedHostCSRData &ensurePreparedContiguousHostChunk(
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

inline PreparedChunkedHostCSR *getPreparedAttenuatedHostChunks(
    const io::CSRView &matrix, const std::vector<uint16_t> &atten,
    int numAngles, uint64_t nnzBudget, int threads, bool eagerPrepare = false,
    uint64_t hostCacheBudgetBytes = 0) {
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

inline const PinnedHostCSRData &ensurePreparedAttenuatedHostChunk(
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

class SparseOpsCUDAState {
public:
  template <typename MatrixType>
  void executeSpMV(const MatrixType &matrix, const double *x, double *y,
                   cusparseOperation_t operation) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const CachedMatrix &cached = ensureMatrixUploaded(matrix);
    const int64_t inputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                  ? cached.rows
                                  : cached.cols;
    const int64_t outputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                   ? cached.cols
                                   : cached.rows;

    uploadDouble(dInput_, x, static_cast<size_t>(inputSize));
    dOutput_.allocate(static_cast<size_t>(outputSize));

    CuSparseVecGuard vecX;
    CuSparseVecGuard vecY;
    vecX.create(inputSize, dInput_.get());
    vecY.create(outputSize, dOutput_.get());
    executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f);
    downloadDouble(dOutput_, y, static_cast<size_t>(outputSize));
  }

  void executeSpMVFloat(const io::CSRView &matrix, const float *x, float *y,
                        cusparseOperation_t operation) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t inputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                  ? cached.rows
                                  : cached.cols;
    const int64_t outputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                   ? cached.cols
                                   : cached.rows;
    const bool backward = operation == CUSPARSE_OPERATION_TRANSPOSE;

    const auto stageStart = std::chrono::steady_clock::now();
    dInput_.copyFromHost(x, static_cast<size_t>(inputSize));
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      stageStart)
            .count());
    dOutput_.allocate(static_cast<size_t>(outputSize));

    CuSparseVecGuard vecX;
    CuSparseVecGuard vecY;
    vecX.create(inputSize, dInput_.get());
    vecY.create(outputSize, dOutput_.get());
    TimedCudaStagePair deviceTiming;
    deviceTiming.recordStart(nullptr);
    executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f);
    deviceTiming.recordStop(nullptr);
    const auto outputStart = std::chrono::steady_clock::now();
    dOutput_.copyToHost(y, static_cast<size_t>(outputSize));
    const double outputSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      outputStart)
            .count();
    noteCudaCompactOutputD2H(outputSeconds,
                             static_cast<size_t>(outputSize) * sizeof(float),
                             backward);
    const double deviceSeconds = deviceTiming.elapsedSeconds();
    if (backward) {
      noteCudaCusparseBackward(deviceSeconds, 1);
    } else {
      noteCudaCusparseForward(deviceSeconds, 1);
    }
  }

  void executeCompactAttenSpMVFloat(const io::CSRView &matrix,
                                    const uint16_t *attFactors,
                                    size_t attCount,
                                    const cuda::CompactAttenBakeParams &params,
                                    const float *x, float *y,
                                    cusparseOperation_t operation) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    if (params.symmetryMode != 45 && params.symmetryMode != 90) {
      throw std::runtime_error(
          "compact attenuation-baked SpMV requires symmetryMode 45 or 90.");
    }
    if (attFactors == nullptr || attCount == 0) {
      throw std::runtime_error(
          "compact attenuation-baked SpMV received an empty factor table.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);

    // Upload the full-angle attenuation table once; cache by host pointer.
    if (attenTableHostPtr_ != attFactors || attenTableCount_ != attCount) {
      dAttenTable_.copyFromHost(attFactors, attCount);
      attenTableHostPtr_ = attFactors;
      attenTableCount_ = attCount;
      bakedValid_ = false;
    }

    // Re-bake only when the (matrix, branch, angle window, table) key changes;
    // a forward spmv followed by the transposed spmv of the same chunk/branch
    // reuses the baked values.
    // Comparing device pointers as well guards against the same key being
    // re-uploaded to a different device allocation (single-slot cache churn).
    const bool bakeHit =
        bakedValid_ && bakedMatrixKey_ == cached.key &&
        bakedRowPtrDev_ == static_cast<const void *>(cached.dRowPtr.get()) &&
        bakedColIdxDev_ == static_cast<const void *>(cached.dColIdx.get()) &&
        bakedAttPtr_ == static_cast<const void *>(attFactors) &&
        bakedParams_.symmetryMode == params.symmetryMode &&
        bakedParams_.compactRowBase == params.compactRowBase &&
        bakedParams_.symId == params.symId &&
        bakedParams_.zMirror == params.zMirror;
    if (!bakeHit) {
      dBakedValues_.allocate(static_cast<size_t>(cached.nnz));
      launchCompactBakeAttenKernel(
          cached.dValues.get(), cached.dRowPtr.get(), cached.dColIdx.get(),
          dAttenTable_.get(), dBakedValues_.get(),
          static_cast<int>(cached.rows), params, computeStream_.get());
      bakedMat_.create(cached.rows, cached.cols, cached.nnz,
                       cached.dRowPtr.get(), cached.dColIdx.get(),
                       dBakedValues_.get());
      bakedBufferSizeValid_ = {};
      bakedMatrixKey_ = cached.key;
      bakedRowPtrDev_ = cached.dRowPtr.get();
      bakedColIdxDev_ = cached.dColIdx.get();
      bakedAttPtr_ = attFactors;
      bakedParams_ = params;
      bakedValid_ = true;
      CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    }

    const int64_t inputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                  ? cached.rows
                                  : cached.cols;
    const int64_t outputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                   ? cached.cols
                                   : cached.rows;
    const bool backward = operation == CUSPARSE_OPERATION_TRANSPOSE;

    const auto stageStart = std::chrono::steady_clock::now();
    dInput_.copyFromHost(x, static_cast<size_t>(inputSize));
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      stageStart)
            .count());
    dOutput_.allocate(static_cast<size_t>(outputSize));

    CuSparseVecGuard vecX;
    CuSparseVecGuard vecY;
    vecX.create(inputSize, dInput_.get());
    vecY.create(outputSize, dOutput_.get());
    TimedCudaStagePair deviceTiming;
    deviceTiming.recordStart(nullptr);
    const size_t bufferIndex = spmvBufferIndex(operation);
    executeSpMVDescriptorLocked(bakedMat_.get(),
                                bakedBufferSizes_[bufferIndex],
                                bakedBufferSizeValid_[bufferIndex], operation,
                                vecX.get(), vecY.get(), 0.0f, backward);
    deviceTiming.recordStop(nullptr);
    const auto outputStart = std::chrono::steady_clock::now();
    dOutput_.copyToHost(y, static_cast<size_t>(outputSize));
    const double outputSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      outputStart)
            .count();
    noteCudaCompactOutputD2H(outputSeconds,
                             static_cast<size_t>(outputSize) * sizeof(float),
                             backward);
    const double deviceSeconds = deviceTiming.elapsedSeconds();
    if (backward) {
      noteCudaCusparseBackward(deviceSeconds, 1);
    } else {
      noteCudaCusparseForward(deviceSeconds, 1);
    }
  }

  void executeCompactDeviceRatioSpmvTFloat(
      const io::CSRView &matrix, const float *x, const double *fullProjection,
      size_t fullProjectionCount,
      const cuda::CompactDeviceRatioParams &params, float *y,
      cuda::CompactDeviceRatioTiming *timing) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    if (params.symmetryMode != 45 && params.symmetryMode != 90) {
      throw std::runtime_error(
          "compact device-ratio CUDA helper requires symmetryMode 45 or 90.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t rows = cached.rows;
    const int64_t cols = cached.cols;

    const auto inputStart = std::chrono::steady_clock::now();
    dInput_.copyFromHost(x, static_cast<size_t>(cols));
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      inputStart)
            .count());
    ensureCompactFullProjectionUploaded(fullProjection, fullProjectionCount);
    ensureProjDomainKernelsLoaded();

    dOutput_.allocate(static_cast<size_t>(rows));
    dChunkInput_.allocate(static_cast<size_t>(rows));
    dFullOutput_.allocate(static_cast<size_t>(cols));

    CuSparseVecGuard vecXForward;
    CuSparseVecGuard vecYForward;
    vecXForward.create(cols, dInput_.get());
    vecYForward.create(rows, dOutput_.get());
    TimedCudaStagePair forwardTiming;
    forwardTiming.recordStart(nullptr);
    executeSpMVLocked(cached, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      vecXForward.get(), vecYForward.get(), 0.0f);
    forwardTiming.recordStop(nullptr);

    TimedCudaStagePair ratioTiming;
    ratioTiming.recordStart(nullptr);
    launchCompactRatioKernel(dOutput_.get(), dCompactProjectionInput_.get(),
                             fullProjectionCount, params, dChunkInput_.get(),
                             static_cast<size_t>(rows), 1e-10f);
    ratioTiming.recordStop(nullptr);

    CuSparseVecGuard vecXBackward;
    CuSparseVecGuard vecYBackward;
    vecXBackward.create(rows, dChunkInput_.get());
    vecYBackward.create(cols, dFullOutput_.get());
    TimedCudaStagePair backwardTiming;
    backwardTiming.recordStart(nullptr);
    executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                      vecXBackward.get(), vecYBackward.get(), 0.0f);
    backwardTiming.recordStop(nullptr);

    const auto outputStart = std::chrono::steady_clock::now();
    dFullOutput_.copyToHost(y, static_cast<size_t>(cols));
    const double outputSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      outputStart)
            .count();
    noteCudaCompactOutputD2H(outputSeconds,
                             static_cast<size_t>(cols) * sizeof(float),
                             true);
    const double forwardSeconds = forwardTiming.elapsedSeconds();
    const double ratioSeconds = ratioTiming.elapsedSeconds();
    const double backwardSeconds = backwardTiming.elapsedSeconds();
    noteCudaCusparseForward(forwardSeconds, 1);
    noteCudaRatio(ratioSeconds, 1);
    noteCudaCusparseBackward(backwardSeconds, 1);
    if (timing != nullptr) {
      timing->forwardSeconds += forwardSeconds;
      timing->ratioSeconds += ratioSeconds;
      timing->backwardSeconds += backwardSeconds;
    }
  }

  void beginCompactDeviceBackward(size_t voxelCount, int accumulatorIndex) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    accum.allocate(voxelCount);
    if (voxelCount > 0) {
      CUDA_CHECK(cudaMemset(accum.get(), 0,
                            voxelCount * sizeof(double)));
    }
    compactBackwardAccumSize_[slot] = voxelCount;
  }

  void executeCompactDeviceBackwardSpmvTAccumulateFloat(
      const io::CSRView &matrix, const float *x,
      const cuda::CompactDeviceTransformParams &params,
      cuda::CompactDeviceBackwardTiming *timing, int accumulatorIndex) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    const size_t accumSize = compactBackwardAccumSize_[slot];
    if (accumSize == 0 || accum.get() == nullptr) {
      throw std::runtime_error(
          "compact device-backward accumulator was not initialized.");
    }
    if (params.symmetryMode != 45 && params.symmetryMode != 90) {
      throw std::runtime_error(
          "compact device-backward CUDA helper requires symmetryMode 45 or 90.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t rows = cached.rows;
    const int64_t cols = cached.cols;
    if (static_cast<size_t>(cols) != accumSize) {
      throw std::runtime_error(
          "compact device-backward accumulator size does not match matrix columns.");
    }

    const auto inputStart = std::chrono::steady_clock::now();
    dInput_.copyFromHost(x, static_cast<size_t>(rows));
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      inputStart)
            .count());
    dFullOutput_.allocate(static_cast<size_t>(cols));
    ensureProjDomainKernelsLoaded();

    CuSparseVecGuard vecXBackward;
    CuSparseVecGuard vecYBackward;
    vecXBackward.create(rows, dInput_.get());
    vecYBackward.create(cols, dFullOutput_.get());
    TimedCudaStagePair backwardTiming;
    backwardTiming.recordStart(nullptr);
    executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                      vecXBackward.get(), vecYBackward.get(), 0.0f);
    backwardTiming.recordStop(nullptr);

    TimedCudaStagePair transformTiming;
    transformTiming.recordStart(nullptr);
    launchCompactTransformBackwardAddKernel(
        dFullOutput_.get(), accum.get(), accumSize, params);
    transformTiming.recordStop(nullptr);

    backwardTiming.synchronizeStop();
    transformTiming.synchronizeStop();
    const double backwardSeconds = backwardTiming.elapsedSeconds();
    const double transformSeconds = transformTiming.elapsedSeconds();
    noteCudaCusparseBackward(backwardSeconds, 1);
    if (timing != nullptr) {
      timing->backwardSeconds += backwardSeconds;
      timing->transformBackwardSeconds += transformSeconds;
    }
  }

  void executeCompactDeviceRatioSpmvTAccumulateFloat(
      const io::CSRView &matrix, const float *x, const double *fullProjection,
      size_t fullProjectionCount,
      const cuda::CompactDeviceRatioParams &ratioParams,
      const cuda::CompactDeviceTransformParams &transformParams,
      cuda::CompactDeviceRatioTiming *ratioTimingOut,
      cuda::CompactDeviceBackwardTiming *backwardTimingOut,
      int accumulatorIndex) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    const size_t accumSize = compactBackwardAccumSize_[slot];
    if (accumSize == 0 || accum.get() == nullptr) {
      throw std::runtime_error(
          "compact device accumulator was not initialized.");
    }
    if (ratioParams.symmetryMode != 45 && ratioParams.symmetryMode != 90) {
      throw std::runtime_error(
          "compact device-ratio accumulator helper requires symmetryMode 45 or 90.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t rows = cached.rows;
    const int64_t cols = cached.cols;
    if (static_cast<size_t>(cols) != accumSize) {
      throw std::runtime_error(
          "compact device accumulator size does not match matrix columns.");
    }

    const auto inputStart = std::chrono::steady_clock::now();
    dInput_.copyFromHost(x, static_cast<size_t>(cols));
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      inputStart)
            .count());
    ensureCompactFullProjectionUploaded(fullProjection, fullProjectionCount);
    ensureProjDomainKernelsLoaded();

    dOutput_.allocate(static_cast<size_t>(rows));
    dChunkInput_.allocate(static_cast<size_t>(rows));
    dFullOutput_.allocate(static_cast<size_t>(cols));

    CuSparseVecGuard vecXForward;
    CuSparseVecGuard vecYForward;
    vecXForward.create(cols, dInput_.get());
    vecYForward.create(rows, dOutput_.get());
    TimedCudaStagePair forwardTiming;
    forwardTiming.recordStart(nullptr);
    executeSpMVLocked(cached, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      vecXForward.get(), vecYForward.get(), 0.0f);
    forwardTiming.recordStop(nullptr);

    TimedCudaStagePair ratioTiming;
    ratioTiming.recordStart(nullptr);
    launchCompactRatioKernel(dOutput_.get(), dCompactProjectionInput_.get(),
                             fullProjectionCount, ratioParams,
                             dChunkInput_.get(), static_cast<size_t>(rows),
                             1e-10f);
    ratioTiming.recordStop(nullptr);

    CuSparseVecGuard vecXBackward;
    CuSparseVecGuard vecYBackward;
    vecXBackward.create(rows, dChunkInput_.get());
    vecYBackward.create(cols, dFullOutput_.get());
    TimedCudaStagePair backwardTiming;
    backwardTiming.recordStart(nullptr);
    executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                      vecXBackward.get(), vecYBackward.get(), 0.0f);
    backwardTiming.recordStop(nullptr);

    TimedCudaStagePair transformTiming;
    transformTiming.recordStart(nullptr);
    launchCompactTransformBackwardAddKernel(
        dFullOutput_.get(), accum.get(), accumSize, transformParams);
    transformTiming.recordStop(nullptr);

    transformTiming.synchronizeStop();
    const double forwardSeconds = forwardTiming.elapsedSeconds();
    const double ratioSeconds = ratioTiming.elapsedSeconds();
    const double backwardSeconds = backwardTiming.elapsedSeconds();
    const double transformSeconds = transformTiming.elapsedSeconds();
    noteCudaCusparseForward(forwardSeconds, 1);
    noteCudaRatio(ratioSeconds, 1);
    noteCudaCusparseBackward(backwardSeconds, 1);
    if (ratioTimingOut != nullptr) {
      ratioTimingOut->forwardSeconds += forwardSeconds;
      ratioTimingOut->ratioSeconds += ratioSeconds;
      ratioTimingOut->backwardSeconds += backwardSeconds;
    }
    if (backwardTimingOut != nullptr) {
      backwardTimingOut->backwardSeconds += backwardSeconds;
      backwardTimingOut->transformBackwardSeconds += transformSeconds;
    }
  }

  void downloadCompactDeviceBackward(double *y, size_t voxelCount,
                                     int accumulatorIndex) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    const DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    if (voxelCount != compactBackwardAccumSize_[slot] ||
        accum.get() == nullptr) {
      throw std::runtime_error(
          "compact device-backward download size does not match accumulator.");
    }
    accum.copyToHost(y, voxelCount);
  }

  void executeSubsetSpMV(const io::CSRView &matrix,
                         const std::vector<int> &subset_rows, const double *x,
                         double *y, cusparseOperation_t operation,
                         int threads) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const CachedMatrix &cached = ensureSubsetUploaded(matrix, subset_rows, threads);
    const int64_t inputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                  ? cached.rows
                                  : cached.cols;
    const int64_t outputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                   ? cached.cols
                                   : cached.rows;

    uploadDouble(dInput_, x, static_cast<size_t>(inputSize));
    dOutput_.allocate(static_cast<size_t>(outputSize));

    CuSparseVecGuard vecX;
    CuSparseVecGuard vecY;
    vecX.create(inputSize, dInput_.get());
    vecY.create(outputSize, dOutput_.get());
    executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f);
    downloadDouble(dOutput_, y, static_cast<size_t>(outputSize));
  }

  void executeAttenuatedSpMV(const io::CSRView &matrix,
                             const std::vector<uint16_t> &atten,
                             int numAngles, const double *x, double *y,
                             cusparseOperation_t operation,
                             int threads) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const CachedMatrix &cached =
        ensureAttenuatedUploaded(matrix, atten, numAngles, threads);
    const int64_t inputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                  ? cached.rows
                                  : cached.cols;
    const int64_t outputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                   ? cached.cols
                                   : cached.rows;

    uploadDouble(dInput_, x, static_cast<size_t>(inputSize));
    dOutput_.allocate(static_cast<size_t>(outputSize));

    CuSparseVecGuard vecX;
    CuSparseVecGuard vecY;
    vecX.create(inputSize, dInput_.get());
    vecY.create(outputSize, dOutput_.get());
    executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f);
    downloadDouble(dOutput_, y, static_cast<size_t>(outputSize));
  }

  void runProjDomainChunkedStreaming(
      const io::CSRView &matrix, const std::vector<std::vector<int>> &chunkRows,
      const std::vector<double> &projections,
      const filters::PSFKernels &psfKernels, size_t frameSize, int detNz,
      int detNx, const double *image, int threads, double *numer) {
    ensureCudaRuntimeAvailable();
    if (image == nullptr || numer == nullptr) {
      throw std::runtime_error(
          "Proj-domain CUDA chunked streaming requires valid input and output buffers.");
    }
    if (frameSize == 0) {
      throw std::runtime_error(
          "Proj-domain CUDA chunked streaming requires a non-zero detector frame size.");
    }
    if (projections.size() < static_cast<size_t>(matrix.rows)) {
      throw std::runtime_error(
          "Proj-domain CUDA chunked streaming received an undersized projection vector.");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    uploadDouble(dInput_, image, static_cast<size_t>(matrix.cols));
    dAccumOutput_.allocate(static_cast<size_t>(matrix.cols));
    if (matrix.cols > 0) {
      CUDA_CHECK(cudaMemset(dAccumOutput_.get(), 0,
                            static_cast<size_t>(matrix.cols) *
                                sizeof(float)));
    }

    constexpr float eps = 1e-12f;
    std::vector<double> localYhat;
    std::vector<double> localRatio;
    bool canUseDeviceChunkOps = true;
    try {
      ensureProjDomainKernelWeightsUploaded(psfKernels);
      uploadDouble(dProjectionInput_, projections.data(), projections.size());
    } catch (...) {
      canUseDeviceChunkOps = false;
    }

    for (const auto &rows : chunkRows) {
      if (rows.empty()) {
        continue;
      }
      if (!isContiguousAscending(rows)) {
        throw std::runtime_error(
            "Proj-domain CUDA chunked streaming expects contiguous ascending chunk rows.");
      }

      const CachedMatrix &cached =
          ensureStreamingSubsetUploaded(matrix, rows, threads);
      const size_t chunkSize = static_cast<size_t>(cached.rows);
      if (chunkSize % frameSize != 0) {
        throw std::runtime_error(
            "Proj-domain CUDA chunked streaming requires detector-frame-aligned chunks.");
      }

      dOutput_.allocate(chunkSize);
      CuSparseVecGuard vecXForward;
      CuSparseVecGuard vecYForward;
      vecXForward.create(cached.cols, dInput_.get());
      vecYForward.create(cached.rows, dOutput_.get());
      executeSpMVLocked(cached, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        vecXForward.get(), vecYForward.get(), 0.0f);

      if (canUseDeviceChunkOps) {
        try {
          applyProjDomainChunkOpsOnDevice(
              static_cast<size_t>(rows.front()), chunkSize, frameSize, detNz,
              detNx, psfKernels, eps);
        } catch (...) {
          canUseDeviceChunkOps = false;
        }
      }

      if (!canUseDeviceChunkOps) {
        localYhat.resize(chunkSize);
        downloadDouble(dOutput_, localYhat.data(), chunkSize);

        localRatio.resize(chunkSize);
        for (size_t index = 0; index < chunkSize; ++index) {
          if (rows[index] < 0) {
            throw std::runtime_error(
                "Proj-domain CUDA chunked streaming received a negative row index.");
          }
          const size_t globalRow = static_cast<size_t>(rows[index]);
          localRatio[index] =
              projections[globalRow] / std::max(localYhat[index], static_cast<double>(eps));
        }
        applyProjDomainChunkPSF(localRatio, frameSize, detNz, detNx,
                                psfKernels, threads);
        uploadDouble(dChunkInput_, localRatio.data(), chunkSize);
      }

      CuSparseVecGuard vecXTranspose;
      CuSparseVecGuard vecYTranspose;
      vecXTranspose.create(cached.rows, dChunkInput_.get());
      vecYTranspose.create(cached.cols, dAccumOutput_.get());
      executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                        vecXTranspose.get(), vecYTranspose.get(), 1.0f);
    }

    downloadDouble(dAccumOutput_, numer, static_cast<size_t>(matrix.cols));
  }

private:
  // Double↔float host-device conversion helpers.
  // Pageable→device cudaMemcpy(Async) is host-synchronous, so a single
  // scratch buffer (hostFloatScratch_) is safe to reuse between calls.
  void uploadDouble(DeviceArray<float> &dev, const double *src, size_t count) {
    hostFloatScratch_.resize(count);
    for (size_t i = 0; i < count; ++i)
      hostFloatScratch_[i] = static_cast<float>(src[i]);
    dev.copyFromHost(hostFloatScratch_.data(), count);
  }

  void ensureCompactFullProjectionUploaded(const double *src, size_t count) {
    if (src == nullptr || count == 0) {
      throw std::runtime_error(
          "compact device-ratio CUDA helper received an empty projection.");
    }
    if (compactProjectionHostKey_ == src &&
        compactProjectionCount_ == count &&
        dCompactProjectionInput_.get() != nullptr &&
        dCompactProjectionInput_.size() >= count) {
      return;
    }
    const auto uploadStart = std::chrono::steady_clock::now();
    uploadDouble(dCompactProjectionInput_, src, count);
    compactProjectionHostKey_ = src;
    compactProjectionCount_ = count;
    noteCudaCompactProjectionH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      uploadStart)
            .count(),
        count * sizeof(float));
  }

  void uploadDoubleAsync(DeviceArray<float> &dev, const double *src,
                         size_t count, cudaStream_t stream) {
    hostFloatScratch_.resize(count);
    for (size_t i = 0; i < count; ++i)
      hostFloatScratch_[i] = static_cast<float>(src[i]);
    dev.copyFromHostAsync(hostFloatScratch_.data(), count, stream);
  }

  void downloadDouble(const DeviceArray<float> &dev, double *dst,
                      size_t count) {
    hostFloatScratch_.resize(count);
    dev.copyToHost(hostFloatScratch_.data(), count);
    for (size_t i = 0; i < count; ++i)
      dst[i] = static_cast<double>(hostFloatScratch_[i]);
  }

  struct CachedMatrix {
    MatrixCacheKey key;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    DeviceArray<int32_t> dRowPtr;
    DeviceArray<int32_t> dColIdx;
    DeviceArray<float> dValues;
    CuSparseMatGuard matA;
    mutable std::array<size_t, 2> spmvBufferSizes{};
    mutable std::array<bool, 2> spmvBufferSizeValid{};

    void reset() {
      key = MatrixCacheKey{};
      rows = 0;
      cols = 0;
      nnz = 0;
      matA.reset();
      dRowPtr.reset();
      dColIdx.reset();
      dValues.reset();
      spmvBufferSizes = {};
      spmvBufferSizeValid = {};
    }
  };

  // P3-D: Persistent device-side chunk cache.  Keeps as many chunks as fit
  // in VRAM across iterations so that their H2D upload can be skipped.
  // Uses fixed-residency assignment (not LRU) to avoid sequential-access
  // thrashing: the first N chunks that fit are permanently cached.
  struct DeviceChunkCache {
    std::vector<std::unique_ptr<CachedMatrix>> slots;
    std::vector<bool> valid;
    MatrixCacheKey matrixKey{};
    uint64_t nnzBudget = 0;
    size_t numCached = 0;
    bool initialized = false;

    bool isHit(size_t chunkIndex, const MatrixCacheKey &key) const {
      return initialized && matrixKey == key &&
             chunkIndex < valid.size() && valid[chunkIndex];
    }

    void invalidate() {
      slots.clear();
      valid.clear();
      numCached = 0;
      initialized = false;
    }
  };

  struct ResidentMatrixSetCache {
    std::vector<std::unique_ptr<CachedMatrix>> slots;
    std::vector<MatrixCacheKey> keys;
    std::vector<bool> valid;
    uint64_t reserveBytes = 0;
    size_t numCached = 0;
    bool initialized = false;

    bool matches(const std::vector<MatrixCacheKey> &otherKeys,
                 uint64_t otherReserveBytes) const {
      if (!initialized || reserveBytes != otherReserveBytes ||
          keys.size() != otherKeys.size()) {
        return false;
      }
      for (size_t i = 0; i < keys.size(); ++i) {
        if (!(keys[i] == otherKeys[i])) {
          return false;
        }
      }
      return true;
    }

    const CachedMatrix *find(const MatrixCacheKey &key) const {
      if (!initialized) {
        return nullptr;
      }
      for (size_t i = 0; i < keys.size(); ++i) {
        if (i < valid.size() && valid[i] && keys[i] == key &&
            i < slots.size() && slots[i] != nullptr) {
          return slots[i].get();
        }
      }
      return nullptr;
    }

    void invalidate() {
      slots.clear();
      keys.clear();
      valid.clear();
      reserveBytes = 0;
      numCached = 0;
      initialized = false;
    }
  };

  void ensureProjDomainKernelWeightsUploaded(
      const filters::PSFKernels &psfKernels) {
    if (projKernelXHost_.size() != psfKernels.kx.size() ||
        !std::equal(projKernelXHost_.begin(), projKernelXHost_.end(),
                    psfKernels.kx.begin(),
                    [](float a, double b) { return a == static_cast<float>(b); })) {
      projKernelXHost_.resize(psfKernels.kx.size());
      for (size_t i = 0; i < psfKernels.kx.size(); ++i)
        projKernelXHost_[i] = static_cast<float>(psfKernels.kx[i]);
      dProjKernelX_.copyFromHost(projKernelXHost_.data(), projKernelXHost_.size());
    }
    if (projKernelYHost_.size() != psfKernels.ky.size() ||
        !std::equal(projKernelYHost_.begin(), projKernelYHost_.end(),
                    psfKernels.ky.begin(),
                    [](float a, double b) { return a == static_cast<float>(b); })) {
      projKernelYHost_.resize(psfKernels.ky.size());
      for (size_t i = 0; i < psfKernels.ky.size(); ++i)
        projKernelYHost_[i] = static_cast<float>(psfKernels.ky[i]);
      dProjKernelY_.copyFromHost(projKernelYHost_.data(), projKernelYHost_.size());
    }
  }

  void applyProjDomainChunkOpsOnDevice(size_t projectionOffset, size_t chunkSize,
                                       size_t frameSize, int detNz, int detNx,
                                       const filters::PSFKernels &psfKernels,
                                       float eps) {
    if (chunkSize == 0) {
      return;
    }
    if (frameSize == 0 || chunkSize % frameSize != 0) {
      throw std::runtime_error(
          "Proj-domain CUDA device helper requires frame-aligned chunks.");
    }

    const size_t frameCount = chunkSize / frameSize;
    dChunkInput_.allocate(chunkSize);
    launchProjDomainRatioKernel(dOutput_.get(), dProjectionInput_.get(),
                                projectionOffset, dChunkInput_.get(),
                                chunkSize, eps);

    float *current = dChunkInput_.get();
    if (!projKernelYHost_.empty() && projKernelYHost_.size() > 1) {
      dProjChunkScratch_.allocate(chunkSize);
      launchProjDomainConvYKernel(
          current, dProjChunkScratch_.get(), dProjKernelY_.get(),
          static_cast<int>(projKernelYHost_.size() / 2), detNz, detNx,
          frameCount);
      current = dProjChunkScratch_.get();
    }

    if (!projKernelXHost_.empty() && projKernelXHost_.size() > 1) {
      dProjChunkScratch_.allocate(chunkSize);
      float *target =
          (current == dChunkInput_.get()) ? dProjChunkScratch_.get()
                                          : dChunkInput_.get();
      launchProjDomainConvXKernel(
          current, target, dProjKernelX_.get(),
          static_cast<int>(projKernelXHost_.size() / 2), detNz, detNx,
          frameCount);
      current = target;
    }

    if (current != dChunkInput_.get()) {
      CUDA_CHECK(cudaMemcpy(dChunkInput_.get(), current,
                            chunkSize * sizeof(float),
                            cudaMemcpyDeviceToDevice));
    }
  }

  static void applyProjDomainChunkPSF(std::vector<double> &ratioChunk,
                                      size_t frameSize, int detNz, int detNx,
                                      const filters::PSFKernels &psfKernels,
                                      int threads) {
    if (ratioChunk.empty()) {
      return;
    }

    const ptrdiff_t frameCount =
        static_cast<ptrdiff_t>(ratioChunk.size() / frameSize);
#pragma omp parallel num_threads(threads) if(threads > 1 && frameCount > 1)
    {
      std::vector<double> frame(frameSize, 0.0);
#pragma omp for schedule(static)
      for (ptrdiff_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        const size_t offset = static_cast<size_t>(frameIndex) * frameSize;
        std::copy_n(ratioChunk.data() + offset, frameSize, frame.data());
        filters::conv2d_separable_inplace(frame, detNz, detNx, psfKernels.ky,
                                          psfKernels.kx, 1);
        std::copy_n(frame.data(), frameSize, ratioChunk.data() + offset);
      }
    }
  }

  static size_t spmvBufferIndex(cusparseOperation_t operation) {
    return operation == CUSPARSE_OPERATION_TRANSPOSE ? 1u : 0u;
  }

  void executeSpMVDescriptorLocked(cusparseSpMatDescr_t mat,
                                   size_t &bufferSizeCache,
                                   bool &bufferSizeValid,
                                   cusparseOperation_t operation,
                                   cusparseDnVecDescr_t vecX,
                                   cusparseDnVecDescr_t vecY, float beta,
                                   bool profileAsBackward,
                                   cudaStream_t stream = nullptr) {
    const auto hostStart = std::chrono::steady_clock::now();
    const float alpha = 1.0f;
    const cusparseSpMVAlg_t alg = selectedSpmvAlg(profileAsBackward);
    handle_.setStream(stream);
    size_t bufferSize = bufferSizeCache;
    if (!bufferSizeValid) {
      SPECT_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
          handle_.get(), operation, &alpha, mat, vecX, &beta, vecY,
          CUDA_R_32F, alg, &bufferSize));
      bufferSizeCache = bufferSize;
      bufferSizeValid = true;
    }
    dBuffer_.ensure(bufferSize);
    SPECT_CUSPARSE_CHECK(cusparseSpMV(
        handle_.get(), operation, &alpha, mat, vecX, &beta, vecY,
        CUDA_R_32F, alg, dBuffer_.get()));
    const double hostSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      hostStart)
            .count();
    if (profileAsBackward) {
      noteCudaCusparseBackwardHost(hostSeconds);
    } else {
      noteCudaCusparseForwardHost(hostSeconds);
    }
  }

  void executeSpMVLocked(const CachedMatrix &cached,
                         cusparseOperation_t operation,
                         cusparseDnVecDescr_t vecX,
                         cusparseDnVecDescr_t vecY, float beta,
                         cudaStream_t stream = nullptr) {
    const size_t bufferIndex = spmvBufferIndex(operation);
    executeSpMVDescriptorLocked(
        cached.matA.get(), cached.spmvBufferSizes[bufferIndex],
        cached.spmvBufferSizeValid[bufferIndex], operation, vecX, vecY, beta,
        operation == CUSPARSE_OPERATION_TRANSPOSE, stream);
  }

  void uploadHostCSR(const HostCSRData &host, const MatrixCacheKey &key,
                     CachedMatrix *cache) {
    cache->key = key;
    cache->rows = host.rows;
    cache->cols = host.cols;
    cache->nnz = host.nnz;
    cache->dRowPtr.copyFromHost(host.rowPtr.data(), host.rowPtr.size());
    cache->dColIdx.copyFromHost(host.colIdx.data(), host.colIdx.size());
    cache->dValues.copyFromHost(host.values.data(), host.values.size());
    cache->matA.create(host.rows, host.cols, host.nnz, cache->dRowPtr.get(),
                       cache->dColIdx.get(), cache->dValues.get());
  }

  void uploadHostCSR(const PinnedHostCSRData &host, const MatrixCacheKey &key,
                     CachedMatrix *cache) {
    cache->key = key;
    cache->rows = host.rows;
    cache->cols = host.cols;
    cache->nnz = host.nnz;
    cache->dRowPtr.copyFromHost(host.rowPtr.data(), host.rowPtrCount);
    cache->dColIdx.copyFromHost(host.colIdx.data(), host.colIdxCount);
    cache->dValues.copyFromHost(host.values.data(), host.valuesCount);
    cache->matA.create(host.rows, host.cols, host.nnz, cache->dRowPtr.get(),
                       cache->dColIdx.get(), cache->dValues.get());
  }

  void uploadHostCSRAsync(const PinnedHostCSRData &host,
                          const MatrixCacheKey &key, CachedMatrix *cache,
                          cudaStream_t stream) {
    cache->key = key;
    cache->rows = host.rows;
    cache->cols = host.cols;
    cache->nnz = host.nnz;
    cache->dRowPtr.copyFromHostAsync(host.rowPtr.data(), host.rowPtrCount,
                                     stream);
    cache->dColIdx.copyFromHostAsync(host.colIdx.data(), host.colIdxCount,
                                     stream);
    cache->dValues.copyFromHostAsync(host.values.data(), host.valuesCount,
                                     stream);
    cache->matA.create(host.rows, host.cols, host.nnz, cache->dRowPtr.get(),
                       cache->dColIdx.get(), cache->dValues.get());
  }

  void uploadIoCsrView(const io::CSRView &matrix, const MatrixCacheKey &key,
                       CachedMatrix *cache) {
    cache->key = key;
    cache->rows = matrix.rows;
    cache->cols = matrix.cols;
    cache->nnz = static_cast<int64_t>(matrix.nnz);
    cache->dRowPtr.copyFromHost(
        reinterpret_cast<const int32_t *>(matrix.row_ptr),
        static_cast<size_t>(matrix.rows) + 1u);
    cache->dColIdx.copyFromHost(
        reinterpret_cast<const int32_t *>(matrix.col_idx),
        static_cast<size_t>(matrix.nnz));
    cache->dValues.copyFromHost(matrix.val, static_cast<size_t>(matrix.nnz));
    cache->matA.create(cache->rows, cache->cols, cache->nnz,
                       cache->dRowPtr.get(), cache->dColIdx.get(),
                       cache->dValues.get());
  }

  void uploadIoCsrViewAsync(const io::CSRView &matrix, const MatrixCacheKey &key,
                            CachedMatrix *cache, cudaStream_t stream) {
    cache->key = key;
    cache->rows = matrix.rows;
    cache->cols = matrix.cols;
    cache->nnz = static_cast<int64_t>(matrix.nnz);
    cache->dRowPtr.copyFromHostAsync(
        reinterpret_cast<const int32_t *>(matrix.row_ptr),
        static_cast<size_t>(matrix.rows) + 1u, stream);
    cache->dColIdx.copyFromHostAsync(
        reinterpret_cast<const int32_t *>(matrix.col_idx),
        static_cast<size_t>(matrix.nnz), stream);
    cache->dValues.copyFromHostAsync(matrix.val, static_cast<size_t>(matrix.nnz),
                                     stream);
    cache->matA.create(cache->rows, cache->cols, cache->nnz,
                       cache->dRowPtr.get(), cache->dColIdx.get(),
                       cache->dValues.get());
  }

  void validateIoCsrViewForUpload(const io::CSRView &matrix) const {
    if (!matrix.valid()) {
      throw std::runtime_error(
          "Invalid io::CSRView passed to CUDA sparse backend.");
    }
    const uint32_t nnz = matrix.row_ptr[static_cast<size_t>(matrix.rows)];
    if (nnz > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        matrix.nnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error(
          "CUDA sparse backend only supports nnz representable as int32.");
    }
    if (matrix.cols > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error(
          "CUDA sparse backend only supports column dimensions representable as int32.");
    }
  }

  template <typename MatrixType>
  const CachedMatrix &ensureMatrixUploaded(const MatrixType &matrix) {
    const MatrixCacheKey key = makeCacheKey(matrix);
    if (cache_.key == key && cache_.matA.get() != nullptr) {
      return cache_;
    }

    const HostCSRData host = convertCSR(matrix);
    uploadHostCSR(host, key, &cache_);
    return cache_;
  }

  const CachedMatrix &ensureMatrixUploaded(const io::CSRView &matrix) {
    const MatrixCacheKey key = makeCacheKey(matrix);
    if (const CachedMatrix *resident = residentMatrixSetCache_.find(key)) {
      return *resident;
    }
    if (cache_.key == key && cache_.matA.get() != nullptr) {
      return cache_;
    }
    validateIoCsrViewForUpload(matrix);
    uploadIoCsrView(matrix, key, &cache_);
    return cache_;
  }

  const CachedMatrix &ensureCompactFloatMatrixUploaded(
      const io::CSRView &matrix) {
    const MatrixCacheKey key = makeCacheKey(matrix);
    if (const CachedMatrix *resident = residentMatrixSetCache_.find(key)) {
      noteCudaCompactMatrixReuseHit();
      return *resident;
    }
    if (cache_.key == key && cache_.matA.get() != nullptr) {
      noteCudaCompactMatrixReuseHit();
      return cache_;
    }
    validateIoCsrViewForUpload(matrix);
    const auto uploadStart = std::chrono::steady_clock::now();
    uploadIoCsrView(matrix, key, &cache_);
    noteCudaCompactMatrixUpload(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      uploadStart)
            .count());
    return cache_;
  }

  const CachedMatrix &ensureSubsetUploaded(const io::CSRView &matrix,
                                           const std::vector<int> &subset_rows,
                                           int threads) {
    const SubsetCacheKey key = makeSubsetCacheKey(matrix, subset_rows);
    for (size_t index = 0; index < subsetCache_.size(); ++index) {
      if (subsetCacheValid_[index] && subsetCacheKeys_[index] == key &&
          subsetCache_[index].matA.get() != nullptr) {
        return subsetCache_[index];
      }
    }

    const size_t slot = nextSubsetCacheSlot_;
    nextSubsetCacheSlot_ = (nextSubsetCacheSlot_ + 1) % subsetCache_.size();

    buildSubsetCSR(matrix, subset_rows, threads, subsetHostScratch_);
    uploadHostCSR(subsetHostScratch_, key.matrix, &subsetCache_[slot]);
    subsetCacheKeys_[slot] = key;
    subsetCacheValid_[slot] = true;
    return subsetCache_[slot];
  }

  const CachedMatrix &ensureStreamingSubsetUploaded(
      const io::CSRView &matrix, const std::vector<int> &subset_rows,
      int threads) {
    const SubsetCacheKey key = makeSubsetCacheKey(matrix, subset_rows);
    if (streamingSubsetValid_ && streamingSubsetKey_ == key &&
        streamingSubsetCache_.matA.get() != nullptr) {
      return streamingSubsetCache_;
    }

    buildSubsetCSR(matrix, subset_rows, threads, subsetHostScratch_);
    uploadHostCSR(subsetHostScratch_, key.matrix, &streamingSubsetCache_);
    streamingSubsetKey_ = key;
    streamingSubsetValid_ = true;
    return streamingSubsetCache_;
  }

  const CachedMatrix &ensureStreamingRangeUploaded(const io::CSRView &matrix,
                                                   int rowBegin, int rowEnd,
                                                   int threads) {
    const RangeCacheKey key = makeRangeCacheKey(matrix, rowBegin, rowEnd);
    if (streamingRangeValid_ && streamingRangeKey_ == key &&
        streamingRangeCache_.matA.get() != nullptr) {
      return streamingRangeCache_;
    }

    buildContiguousSubsetCSR(matrix, rowBegin, rowEnd, threads,
                             subsetHostScratch_);
    uploadHostCSR(subsetHostScratch_, key.matrix, &streamingRangeCache_);
    streamingRangeKey_ = key;
    streamingRangeValid_ = true;
    return streamingRangeCache_;
  }

  const CachedMatrix &
  ensurePreparedStreamingRangeUploaded(const io::CSRView &matrix, int rowBegin,
                                       int rowEnd,
                                       const PinnedHostCSRData &hostChunk) {
    const RangeCacheKey key = makeRangeCacheKey(matrix, rowBegin, rowEnd);
    if (streamingRangeValid_ && streamingRangeKey_ == key &&
        streamingRangeCache_.matA.get() != nullptr) {
      return streamingRangeCache_;
    }

    uploadHostCSR(hostChunk, key.matrix, &streamingRangeCache_);
    streamingRangeKey_ = key;
    streamingRangeValid_ = true;
    return streamingRangeCache_;
  }

  void queueTransientRangeUpload(size_t slot, const MatrixCacheKey &key,
                                 const PinnedHostCSRData &hostChunk) {
    if (slot >= transientRangeCache_.size()) {
      throw std::runtime_error("CUDA transient range slot out of range.");
    }
    finalizeTransientRangeUploadTiming(slot);
    cudaStream_t stream = transientUploadStreams_[slot].get();
    if (transientRangeSlotUsed_[slot]) {
      transientRangeReusable_[slot].wait(stream);
    }
    transientRangeUploadStart_[slot].record(stream);
    uploadHostCSRAsync(hostChunk, key, &transientRangeCache_[slot], stream);
    transientRangeReady_[slot].record(stream);
    transientRangeValid_[slot] = true;
    transientRangeSlotUsed_[slot] = true;
    transientRangeUploadTimingPending_[slot] = true;
    transientRangeReusableReady_[slot] = false;
  }

  const CachedMatrix &waitTransientRangeUpload(
      size_t slot, TimedCudaStagePair *waitTiming = nullptr) {
    if (slot >= transientRangeCache_.size()) {
      throw std::runtime_error("CUDA transient range slot is out of range.");
    }
    if (!transientRangeValid_[slot]) {
      throw std::runtime_error("CUDA transient range slot is not ready.");
    }
    if (waitTiming != nullptr) {
      waitTiming->recordStart(computeStream_.get());
    }
    transientRangeReady_[slot].wait(computeStream_.get());
    if (waitTiming != nullptr) {
      waitTiming->recordStop(computeStream_.get());
    }
    return transientRangeCache_[slot];
  }

  void finalizeTransientRangeUploadTiming(size_t slot) {
    if (slot >= transientRangeCache_.size() ||
        !transientRangeUploadTimingPending_[slot]) {
      return;
    }
    const cudaError_t status = cudaEventQuery(transientRangeReady_[slot].get());
    if (status == cudaErrorNotReady) {
      cudaGetLastError();
      return;
    }
    CUDA_CHECK(status);
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs,
                                    transientRangeUploadStart_[slot].get(),
                                    transientRangeReady_[slot].get()));
    noteCudaH2DChunk(static_cast<double>(elapsedMs) / 1000.0);
    transientRangeUploadTimingPending_[slot] = false;
  }

  void waitTransientPinnedScratchReusable(size_t slot) {
    if (slot >= transientRangeCache_.size() || !transientRangeSlotUsed_[slot]) {
      return;
    }
    // The upload stream wait only orders future H2D work on that stream; it
    // does not stop the host from overwriting the pinned staging buffer early.
    // Reusing the slot-local pinned scratch therefore requires a host-side wait
    // until the previous upload on that slot has actually completed.
    CUDA_CHECK(cudaEventSynchronize(transientRangeReady_[slot].get()));
  }

  void markTransientRangeReusable(size_t slot) {
    if (slot >= transientRangeCache_.size() || !transientRangeValid_[slot]) {
      return;
    }
    finalizeTransientRangeUploadTiming(slot);
    transientRangeReusable_[slot].record(computeStream_.get());
    transientRangeReusableReady_[slot] = true;
  }

  void resetTransientRangeState() {
    transientRangeValid_.fill(false);
    transientRangeSlotUsed_.fill(false);
    transientRangeUploadTimingPending_.fill(false);
    transientRangeReusableReady_.fill(false);
  }

  const CachedMatrix &ensureAttenuatedUploaded(const io::CSRView &matrix,
                                               const std::vector<uint16_t> &atten,
                                               int numAngles,
                                               int threads) {
    const AttenCacheKey key = makeAttenCacheKey(matrix, atten, numAngles);
    for (size_t index = 0; index < attenCache_.size(); ++index) {
      if (attenCacheValid_[index] && attenCacheKeys_[index] == key &&
          attenCache_[index].matA.get() != nullptr) {
        return attenCache_[index];
      }
    }

    const size_t slot = nextAttenCacheSlot_;
    nextAttenCacheSlot_ = (nextAttenCacheSlot_ + 1) % attenCache_.size();

    buildAttenuatedCSR(matrix, atten, numAngles, threads, attenHostScratch_);
    uploadHostCSR(attenHostScratch_, key.matrix, &attenCache_[slot]);
    attenCacheKeys_[slot] = key;
    attenCacheValid_[slot] = true;
    return attenCache_[slot];
  }

  // P3-D: Populate device-side chunk cache from prepared host cache.
  // Called during warmup.  Uploads as many chunks as available VRAM
  // permits while reserving space for transient slots and workspace.
  void populateDeviceChunkCacheLocked(
      const PreparedChunkedHostCSR &prepared,
      const MatrixCacheKey &matrixKey,
      uint64_t nnzBudget) {
    if (deviceChunkCache_.initialized &&
        deviceChunkCache_.matrixKey == matrixKey) {
      return;  // Already populated for this matrix.
    }
    if (deviceChunkCache_.initialized) {
      deviceChunkCache_.invalidate();
    }

    const auto &ranges = prepared.ranges;
    if (ranges.empty()) {
      return;
    }

    // Estimate the largest per-chunk device CSR size.
    uint64_t maxChunkBytes = 0;
    for (size_t i = 0; i < ranges.size(); ++i) {
      if (i < prepared.chunks.size() && prepared.ready[i]) {
        const uint64_t csrBytes = estimateDeviceCsrBytes(
            prepared.chunks[i].rows, prepared.chunks[i].nnz);
        maxChunkBytes = std::max(maxChunkBytes, csrBytes);
      }
    }
    if (maxChunkBytes == 0) {
      return;
    }

    // Reserve space for 2 transient slots + cuSPARSE/cuFFT workspace.
    constexpr uint64_t kDeviceCacheReserveBytes =
        2ull * 3ull * 1024ull * 1024ull * 1024ull;  // ~6 GiB headroom

    uint64_t freeBytes = 0;
    try {
      freeBytes = static_cast<uint64_t>(cuda::getMemoryInfo().free);
    } catch (...) {
      return;
    }
    if (freeBytes <= kDeviceCacheReserveBytes) {
      return;
    }
    const uint64_t availableForCache = freeBytes - kDeviceCacheReserveBytes;
    size_t numToCacheMax =
        static_cast<size_t>(availableForCache / maxChunkBytes);
    numToCacheMax = std::min(numToCacheMax, ranges.size());
    if (numToCacheMax == 0) {
      return;
    }

    deviceChunkCache_.slots.resize(numToCacheMax);
    deviceChunkCache_.valid.resize(numToCacheMax, false);
    deviceChunkCache_.matrixKey = matrixKey;
    deviceChunkCache_.nnzBudget = nnzBudget;

    for (size_t i = 0; i < numToCacheMax; ++i) {
      if (i >= prepared.chunks.size() || !prepared.ready[i]) {
        continue;
      }
      deviceChunkCache_.slots[i] = std::make_unique<CachedMatrix>();
      uploadHostCSRAsync(prepared.chunks[i], matrixKey,
                         deviceChunkCache_.slots[i].get(), computeStream_.get());
      deviceChunkCache_.valid[i] = true;
      ++deviceChunkCache_.numCached;
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    deviceChunkCache_.initialized = true;

    const uint64_t usedMiB =
        deviceChunkCache_.numCached * maxChunkBytes / (1024ull * 1024ull);
    fprintf(stderr,
            "[P3-D] Device chunk cache: %zu / %zu chunks cached "
            "(%llu MiB VRAM)\n",
            deviceChunkCache_.numCached, ranges.size(),
            static_cast<unsigned long long>(usedMiB));
  }

public:
  bool warmupResidentMatrixSet(const std::vector<io::CSRView> &matrices,
                               uint64_t reserveBytes) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<MatrixCacheKey> keys;
    keys.reserve(matrices.size());
    for (const auto &matrix : matrices) {
      validateIoCsrViewForUpload(matrix);
      keys.push_back(makeCacheKey(matrix));
    }
    if (residentMatrixSetCache_.matches(keys, reserveBytes)) {
      return residentMatrixSetCache_.numCached > 0;
    }

    residentMatrixSetCache_.invalidate();
    if (matrices.empty()) {
      return false;
    }

    uint64_t freeBytes = 0;
    try {
      freeBytes = static_cast<uint64_t>(cuda::getMemoryInfo().free);
    } catch (...) {
      return false;
    }
    if (freeBytes <= reserveBytes) {
      return false;
    }
    const uint64_t availableForCache = freeBytes - reserveBytes;

    std::vector<uint64_t> matrixBytesPerEntry;
    matrixBytesPerEntry.reserve(matrices.size());
    uint64_t fitBytes = 0;
    size_t fitCount = 0;
    for (const auto &matrix : matrices) {
      const uint64_t bytes =
          estimateDeviceCsrBytes(matrix.rows, static_cast<int64_t>(matrix.nnz));
      matrixBytesPerEntry.push_back(bytes);
      if (bytes == 0 || fitBytes + bytes > availableForCache) {
        break;
      }
      fitBytes += bytes;
      ++fitCount;
    }
    const size_t requiredFitCount = std::min<size_t>(2, matrices.size());
    if (fitCount < requiredFitCount) {
      return false;
    }

    residentMatrixSetCache_.slots.resize(matrices.size());
    residentMatrixSetCache_.keys = keys;
    residentMatrixSetCache_.valid.resize(matrices.size(), false);
    residentMatrixSetCache_.reserveBytes = reserveBytes;

    uint64_t usedBytes = 0;
    for (size_t index = 0; index < matrices.size(); ++index) {
      const io::CSRView &matrix = matrices[index];
      const uint64_t matrixBytes = matrixBytesPerEntry[index];
      if (matrixBytes == 0 || usedBytes + matrixBytes > availableForCache) {
        break;
      }
      auto cached = std::make_unique<CachedMatrix>();
      uploadIoCsrViewAsync(matrix, keys[index], cached.get(),
                           computeStream_.get());
      residentMatrixSetCache_.slots[index] = std::move(cached);
      residentMatrixSetCache_.valid[index] = true;
      usedBytes += matrixBytes;
      ++residentMatrixSetCache_.numCached;
    }

    if (residentMatrixSetCache_.numCached == 0) {
      residentMatrixSetCache_.invalidate();
      return false;
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    residentMatrixSetCache_.initialized = true;
    fprintf(stderr,
            "[CompactChunkCUDA] Resident matrix cache: %zu / %zu matrices "
            "cached (%llu MiB reserve=%llu MiB)\n",
            residentMatrixSetCache_.numCached, matrices.size(),
            static_cast<unsigned long long>(usedBytes / (1024ull * 1024ull)),
            static_cast<unsigned long long>(reserveBytes /
                                            (1024ull * 1024ull)));
    return true;
  }

public:
  bool warmupFullSubsetChunkCache(
      const io::CSRView &matrix, int bedCount,
      const std::vector<std::vector<uint16_t>> *bedAttenFactors, int numAngles,
      int threads) {
    ensureCudaRuntimeAvailable();
    if (!needsWholeMatrixChunking(matrix)) {
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t nnzBudget = selectCudaChunkNnzBudget();
    const bool useAttenuation =
        bedAttenFactors != nullptr && !bedAttenFactors->empty();
    if (!useAttenuation) {
      const auto *prepared =
          getPreparedContiguousHostChunks(matrix, nnzBudget, threads, true);
      if (prepared != nullptr) {
        // P3-D: populate device-side chunk cache from the prepared host data.
        const MatrixCacheKey matrixKey = makeCacheKey(matrix);
        populateDeviceChunkCacheLocked(*prepared, matrixKey, nnzBudget);
        return true;
      }
      const auto &chunks = getContiguousRowChunks(matrix, nnzBudget);
      const MatrixCacheKey matrixKey = makeCacheKey(matrix);
      for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        const auto buildStart = std::chrono::steady_clock::now();
        waitTransientPinnedScratchReusable(0);
        buildContiguousSubsetCSR(matrix, chunk.rowBegin, chunk.rowEnd, threads,
                                 transientRangePinnedScratch_[0]);
        noteCudaChunkBuild(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          buildStart)
                .count());
        queueTransientRangeUpload(0, matrixKey, transientRangePinnedScratch_[0]);
        (void)waitTransientRangeUpload(0);
        markTransientRangeReusable(0);
      }
      CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
      return true;
    }

    if (bedCount <= 0 || static_cast<size_t>(bedCount) != bedAttenFactors->size()) {
      throw std::runtime_error(
          "CUDA chunk warmup received inconsistent attenuation inputs.");
    }

    bool preparedAny = true;
    const uint64_t perBedPreparedHostCacheBudgetBytes =
        selectPreparedChunkHostCacheBudgetBytes(
            matrix, static_cast<size_t>(bedCount));
    for (int bed = 0; bed < bedCount; ++bed) {
      const auto *prepared = getPreparedAttenuatedHostChunks(
          matrix, (*bedAttenFactors)[static_cast<size_t>(bed)], numAngles,
          nnzBudget, threads, true, perBedPreparedHostCacheBudgetBytes);
      preparedAny = preparedAny && (prepared != nullptr);
    }
    if (preparedAny) {
      return true;
    }
    const auto &chunks = getContiguousRowChunks(matrix, nnzBudget);
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
      for (int bed = 0; bed < bedCount; ++bed) {
        for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
          const RowChunkRange &chunk = chunks[chunkIndex];
          const auto buildStart = std::chrono::steady_clock::now();
          waitTransientPinnedScratchReusable(0);
          buildAttenuatedContiguousSubsetCSR(
              matrix, (*bedAttenFactors)[static_cast<size_t>(bed)], numAngles,
              chunk.rowBegin, chunk.rowEnd, threads,
              transientRangePinnedScratch_[0]);
          noteCudaChunkBuild(
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            buildStart)
                  .count());
          queueTransientRangeUpload(0, matrixKey, transientRangePinnedScratch_[0]);
          (void)waitTransientRangeUpload(0);
          markTransientRangeReusable(0);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    return true;
  }

  void executeChunkedWholeMatrixSpMV(const io::CSRView &matrix, const double *x,
                                     double *y,
                                     cusparseOperation_t operation,
                                     int threads) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t nnzBudget = selectCudaChunkNnzBudget();
    PreparedChunkedHostCSR *prepared =
        getPreparedContiguousHostChunks(matrix, nnzBudget, threads);
    const auto &chunks =
        prepared ? prepared->ranges : getContiguousRowChunks(matrix, nnzBudget);
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
    const size_t transientSlotCount = transientRangeCache_.size();
    resetTransientRangeState();

    const auto queueChunkUpload = [&](size_t chunkIndex, size_t slot) {
      const RowChunkRange &chunk = chunks[chunkIndex];
      if (prepared != nullptr && prepared->ready[chunkIndex]) {
        touchPreparedChunk(*prepared, chunkIndex);
        queueTransientRangeUpload(slot, matrixKey, prepared->chunks[chunkIndex]);
        return;
      }
      const auto buildStart = std::chrono::steady_clock::now();
      waitTransientPinnedScratchReusable(slot);
      buildContiguousSubsetCSR(matrix, chunk.rowBegin, chunk.rowEnd, threads,
                               transientRangePinnedScratch_[slot]);
      noteCudaChunkBuild(
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        buildStart)
              .count());
      queueTransientRangeUpload(slot, matrixKey, transientRangePinnedScratch_[slot]);
    };

    if (!chunks.empty()) {
      queueChunkUpload(0, 0);
      if (chunks.size() > 1 && transientSlotCount > 1) {
        queueChunkUpload(1, 1);
      }
    }

    if (operation == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      std::vector<TimedCudaStagePair> uploadWaitTimings;
      std::vector<TimedCudaStagePair> forwardTimings;
      uploadWaitTimings.reserve(chunks.size());
      forwardTimings.reserve(chunks.size());
      uploadDoubleAsync(dInput_, x, static_cast<size_t>(matrix.cols),
                        computeStream_.get());
      for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        const size_t slot =
            transientSlotCount > 0 ? chunkIndex % transientSlotCount : 0;
        uploadWaitTimings.emplace_back();
        const CachedMatrix &cached =
            waitTransientRangeUpload(slot, &uploadWaitTimings.back());
        const size_t chunkRows =
            static_cast<size_t>(chunk.rowEnd - chunk.rowBegin);
        dOutput_.allocate(chunkRows);

        CuSparseVecGuard vecX;
        CuSparseVecGuard vecY;
        vecX.create(cached.cols, dInput_.get());
        vecY.create(cached.rows, dOutput_.get());
        forwardTimings.emplace_back();
        forwardTimings.back().recordStart(computeStream_.get());
        executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f,
                          computeStream_.get());
        forwardTimings.back().recordStop(computeStream_.get());
        markTransientRangeReusable(slot);
        if (chunkIndex + transientSlotCount < chunks.size()) {
          queueChunkUpload(chunkIndex + transientSlotCount, slot);
        }
        hostFloatScratch_.resize(chunkRows);
        dOutput_.copyToHostAsync(hostFloatScratch_.data(), chunkRows,
                                 computeStream_.get());
        CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
        for (size_t i = 0; i < chunkRows; ++i)
          y[static_cast<size_t>(chunk.rowBegin) + i] =
              static_cast<double>(hostFloatScratch_[i]);
      }
      noteCudaUploadWaitOnCompute(accumulateTimedCudaPairs(uploadWaitTimings),
                                  uploadWaitTimings.size());
      noteCudaCusparseForward(accumulateTimedCudaPairs(forwardTimings),
                              forwardTimings.size());
      return;
    }

    std::vector<TimedCudaStagePair> uploadWaitTimings;
    std::vector<TimedCudaStagePair> backwardTimings;
    uploadWaitTimings.reserve(chunks.size());
    backwardTimings.reserve(chunks.size());
    uploadDoubleAsync(dInput_, x, static_cast<size_t>(matrix.rows),
                      computeStream_.get());
    dAccumOutput_.allocate(static_cast<size_t>(matrix.cols));
    if (matrix.cols > 0) {
      CUDA_CHECK(cudaMemsetAsync(dAccumOutput_.get(), 0,
                                 static_cast<size_t>(matrix.cols) *
                                     sizeof(float),
                                 computeStream_.get()));
    }

    for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
      const RowChunkRange &chunk = chunks[chunkIndex];
      const size_t slot =
          transientSlotCount > 0 ? chunkIndex % transientSlotCount : 0;
      uploadWaitTimings.emplace_back();
      const CachedMatrix &cached =
          waitTransientRangeUpload(slot, &uploadWaitTimings.back());

      CuSparseVecGuard vecX;
      CuSparseVecGuard vecY;
      vecX.create(cached.rows,
                  dInput_.get() + static_cast<size_t>(chunk.rowBegin));
      vecY.create(cached.cols, dAccumOutput_.get());
      backwardTimings.emplace_back();
      backwardTimings.back().recordStart(computeStream_.get());
      executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 1.0f,
                        computeStream_.get());
      backwardTimings.back().recordStop(computeStream_.get());
      markTransientRangeReusable(slot);
      if (chunkIndex + transientSlotCount < chunks.size()) {
        queueChunkUpload(chunkIndex + transientSlotCount, slot);
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    noteCudaUploadWaitOnCompute(accumulateTimedCudaPairs(uploadWaitTimings),
                                uploadWaitTimings.size());
    noteCudaCusparseBackward(accumulateTimedCudaPairs(backwardTimings),
                             backwardTimings.size());
    downloadDouble(dAccumOutput_, y, static_cast<size_t>(matrix.cols));
  }

  void executeChunkedFullMultiBedSpMV(
      const io::CSRView &matrix, const double *x, double *y,
      const std::vector<long long> &bedOffsets, int bedCount,
      cusparseOperation_t operation, int threads) {
    ensureCudaRuntimeAvailable();
    if (bedCount <= 0 || static_cast<size_t>(bedCount) != bedOffsets.size()) {
      throw std::runtime_error(
          "CUDA full multi-bed chunked SpMV received inconsistent bed offsets.");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t nnzBudget = selectCudaChunkNnzBudget();
    PreparedChunkedHostCSR *prepared =
        getPreparedContiguousHostChunks(matrix, nnzBudget, threads);
    const auto &chunks =
        prepared ? prepared->ranges : getContiguousRowChunks(matrix, nnzBudget);
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
    const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
    const size_t baseCols = static_cast<size_t>(matrix.cols);
    const size_t totalRows = rowsPerBed * static_cast<size_t>(bedCount);
    const size_t outputSize = static_cast<size_t>(bedOffsets.back()) + baseCols;
    const auto queueChunkUpload = [&](size_t chunkIndex, size_t slot) {
      const RowChunkRange &chunk = chunks[chunkIndex];
      if (prepared != nullptr && prepared->ready[chunkIndex]) {
        touchPreparedChunk(*prepared, chunkIndex);
        queueTransientRangeUpload(
            slot, matrixKey, prepared->chunks[chunkIndex]);
        return;
      }
      waitTransientPinnedScratchReusable(slot);
      buildContiguousSubsetCSR(matrix, chunk.rowBegin, chunk.rowEnd, threads,
                               transientRangePinnedScratch_[slot]);
      queueTransientRangeUpload(slot, matrixKey,
                                transientRangePinnedScratch_[slot]);
    };

    if (!chunks.empty()) {
      queueChunkUpload(0, 0);
      if (chunks.size() > 1) {
        queueChunkUpload(1, 1);
      }
    }

    if (operation == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      uploadDoubleAsync(dInput_, x, outputSize, computeStream_.get());
      dFullOutput_.allocate(totalRows);

      for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        const size_t slot = chunkIndex % transientRangeCache_.size();
        const CachedMatrix &cached = waitTransientRangeUpload(slot);
        const size_t chunkRows =
            static_cast<size_t>(chunk.rowEnd - chunk.rowBegin);

        for (int bed = 0; bed < bedCount; ++bed) {
          const size_t colOffset =
              static_cast<size_t>(bedOffsets[static_cast<size_t>(bed)]);
          const size_t rowOffset =
              static_cast<size_t>(bed) * rowsPerBed +
              static_cast<size_t>(chunk.rowBegin);

          CuSparseVecGuard vecX;
          CuSparseVecGuard vecY;
          vecX.create(cached.cols, dInput_.get() + colOffset);
          vecY.create(cached.rows, dFullOutput_.get() + rowOffset);
          executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f,
                            computeStream_.get());
        }
        markTransientRangeReusable(slot);
        if (chunkIndex + transientRangeCache_.size() < chunks.size()) {
          queueChunkUpload(chunkIndex + transientRangeCache_.size(), slot);
        }
      }

      CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
      downloadDouble(dFullOutput_, y, totalRows);
      return;
    }

    uploadDoubleAsync(dInput_, x, totalRows, computeStream_.get());
    dAccumOutput_.allocate(outputSize);
    if (outputSize > 0) {
      CUDA_CHECK(cudaMemsetAsync(dAccumOutput_.get(), 0,
                                 outputSize * sizeof(float),
                                 computeStream_.get()));
    }

    for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
      const RowChunkRange &chunk = chunks[chunkIndex];
      const size_t slot = chunkIndex % transientRangeCache_.size();
      const CachedMatrix &cached = waitTransientRangeUpload(slot);

      for (int bed = 0; bed < bedCount; ++bed) {
        const size_t rowOffset =
            static_cast<size_t>(bed) * rowsPerBed +
            static_cast<size_t>(chunk.rowBegin);
        const size_t colOffset =
            static_cast<size_t>(bedOffsets[static_cast<size_t>(bed)]);

        CuSparseVecGuard vecX;
        CuSparseVecGuard vecY;
        vecX.create(cached.rows, dInput_.get() + rowOffset);
        vecY.create(cached.cols, dAccumOutput_.get() + colOffset);
        executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 1.0f,
                          computeStream_.get());
      }
      markTransientRangeReusable(slot);
      if (chunkIndex + transientRangeCache_.size() < chunks.size()) {
        queueChunkUpload(chunkIndex + transientRangeCache_.size(), slot);
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    downloadDouble(dAccumOutput_, y, outputSize);
  }

  void executeChunkedFullMultiBedAttenSpMV(
      const io::CSRView &matrix, const double *x, double *y,
      const std::vector<long long> &bedOffsets,
      const std::vector<std::vector<uint16_t>> &bedAttenFactors,
      int numAngles, int bedCount, cusparseOperation_t operation,
      int threads) {
    ensureCudaRuntimeAvailable();
    if (bedCount <= 0 || static_cast<size_t>(bedCount) != bedOffsets.size() ||
        static_cast<size_t>(bedCount) != bedAttenFactors.size()) {
      throw std::runtime_error(
          "CUDA full multi-bed attenuated chunked SpMV received inconsistent bed data.");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t nnzBudget = selectCudaChunkNnzBudget();
    std::vector<PreparedChunkedHostCSR *> preparedPerBed(
        static_cast<size_t>(bedCount), nullptr);
    const uint64_t perBedPreparedHostCacheBudgetBytes =
        selectPreparedChunkHostCacheBudgetBytes(
            matrix, static_cast<size_t>(bedCount));
    for (int bed = 0; bed < bedCount; ++bed) {
      preparedPerBed[static_cast<size_t>(bed)] = getPreparedAttenuatedHostChunks(
          matrix, bedAttenFactors[static_cast<size_t>(bed)], numAngles,
          nnzBudget, threads, false, perBedPreparedHostCacheBudgetBytes);
    }
    const auto &chunks = getContiguousRowChunks(matrix, nnzBudget);
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
    const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
    const size_t baseCols = static_cast<size_t>(matrix.cols);
    const size_t totalRows = rowsPerBed * static_cast<size_t>(bedCount);
    const size_t outputSize = static_cast<size_t>(bedOffsets.back()) + baseCols;

    struct AttenTask {
      size_t chunkIndex = 0;
      int bed = 0;
    };
    std::vector<AttenTask> tasks;
    tasks.reserve(chunks.size() * static_cast<size_t>(bedCount));
    for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
      for (int bed = 0; bed < bedCount; ++bed) {
        tasks.push_back(AttenTask{chunkIndex, bed});
      }
    }

    const auto queueTaskUpload = [&](size_t taskIndex, size_t slot) {
      const AttenTask &task = tasks[taskIndex];
      const RowChunkRange &chunk = chunks[task.chunkIndex];
      PreparedChunkedHostCSR *prepared =
          preparedPerBed[static_cast<size_t>(task.bed)];
      if (prepared != nullptr && prepared->ready[task.chunkIndex]) {
        touchPreparedChunk(*prepared, task.chunkIndex);
        queueTransientRangeUpload(
            slot, matrixKey,
            prepared->chunks[task.chunkIndex]);
        return;
      }
      waitTransientPinnedScratchReusable(slot);
      buildAttenuatedContiguousSubsetCSR(
          matrix, bedAttenFactors[static_cast<size_t>(task.bed)], numAngles,
          chunk.rowBegin, chunk.rowEnd, threads,
          transientRangePinnedScratch_[slot]);
      queueTransientRangeUpload(slot, matrixKey,
                                transientRangePinnedScratch_[slot]);
    };

    if (!tasks.empty()) {
      queueTaskUpload(0, 0);
      if (tasks.size() > 1) {
        queueTaskUpload(1, 1);
      }
    }

    if (operation == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      uploadDoubleAsync(dInput_, x, outputSize, computeStream_.get());
      dFullOutput_.allocate(totalRows);

      for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
        const AttenTask &task = tasks[taskIndex];
        const RowChunkRange &chunk = chunks[task.chunkIndex];
        const size_t slot = taskIndex % transientRangeCache_.size();
        const CachedMatrix &cached = waitTransientRangeUpload(slot);
        const size_t colOffset =
            static_cast<size_t>(bedOffsets[static_cast<size_t>(task.bed)]);
        const size_t rowOffset =
            static_cast<size_t>(task.bed) * rowsPerBed +
            static_cast<size_t>(chunk.rowBegin);

        CuSparseVecGuard vecX;
        CuSparseVecGuard vecY;
        vecX.create(cached.cols, dInput_.get() + colOffset);
        vecY.create(cached.rows, dFullOutput_.get() + rowOffset);
        executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f,
                          computeStream_.get());

        markTransientRangeReusable(slot);
        if (taskIndex + transientRangeCache_.size() < tasks.size()) {
          queueTaskUpload(taskIndex + transientRangeCache_.size(), slot);
        }
      }

      CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
      downloadDouble(dFullOutput_, y, totalRows);
      return;
    }

    uploadDoubleAsync(dInput_, x, totalRows, computeStream_.get());
    dAccumOutput_.allocate(outputSize);
    if (outputSize > 0) {
      CUDA_CHECK(cudaMemsetAsync(dAccumOutput_.get(), 0,
                                 outputSize * sizeof(float),
                                 computeStream_.get()));
    }

    for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
      const AttenTask &task = tasks[taskIndex];
      const RowChunkRange &chunk = chunks[task.chunkIndex];
      const size_t slot = taskIndex % transientRangeCache_.size();
      const CachedMatrix &cached = waitTransientRangeUpload(slot);
      const size_t rowOffset =
          static_cast<size_t>(task.bed) * rowsPerBed +
          static_cast<size_t>(chunk.rowBegin);
      const size_t colOffset =
          static_cast<size_t>(bedOffsets[static_cast<size_t>(task.bed)]);

      CuSparseVecGuard vecX;
      CuSparseVecGuard vecY;
      vecX.create(cached.rows, dInput_.get() + rowOffset);
      vecY.create(cached.cols, dAccumOutput_.get() + colOffset);
      executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 1.0f,
                        computeStream_.get());

      markTransientRangeReusable(slot);
      if (taskIndex + transientRangeCache_.size() < tasks.size()) {
        queueTaskUpload(taskIndex + transientRangeCache_.size(), slot);
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    downloadDouble(dAccumOutput_, y, outputSize);
  }

  void runFullSubsetEmNumer(const io::CSRView &matrix,
                            const std::vector<double> &projections,
                            const double *image, int bedCount,
                            const std::vector<long long> *bedOffsets,
                            const std::vector<std::vector<uint16_t>> *bedAttenFactors,
                            int numAngles, int threads, double *numer) {
    ensureCudaRuntimeAvailable();
    if (image == nullptr || numer == nullptr) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper requires valid input/output buffers.");
    }
    if (bedCount <= 0) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper requires a positive bed count.");
    }
    if (bedOffsets != nullptr &&
        bedOffsets->size() != static_cast<size_t>(bedCount)) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper received inconsistent bed offsets.");
    }
    if (bedAttenFactors != nullptr && !bedAttenFactors->empty() &&
        (numAngles <= 0 ||
         bedAttenFactors->size() != static_cast<size_t>(bedCount))) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper received inconsistent attenuation inputs.");
    }

    const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
    const size_t baseCols = static_cast<size_t>(matrix.cols);
    const size_t totalRows = rowsPerBed * static_cast<size_t>(bedCount);
    if (projections.size() != totalRows) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper received a projection vector with unexpected size.");
    }

    const auto resolveColOffset = [&](int bed) -> size_t {
      if (bedOffsets != nullptr && !bedOffsets->empty()) {
        return static_cast<size_t>((*bedOffsets)[static_cast<size_t>(bed)]);
      }
      return static_cast<size_t>(bed) * baseCols;
    };
    const size_t outputSize =
        resolveColOffset(bedCount - 1) + baseCols;

    std::lock_guard<std::mutex> lock(mutex_);
    // P3-D: Reuse the device cache's nnzBudget when available to keep
    // chunk boundaries consistent with cached device-side data.
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
    const uint64_t nnzBudget =
        (deviceChunkCache_.initialized &&
         deviceChunkCache_.matrixKey == matrixKey)
            ? deviceChunkCache_.nnzBudget
            : selectCudaChunkNnzBudget();
    const bool useAttenuation =
        bedAttenFactors != nullptr && !bedAttenFactors->empty();
    PreparedChunkedHostCSR *preparedHost = nullptr;
    if (!useAttenuation) {
      preparedHost = getPreparedContiguousHostChunks(matrix, nnzBudget, threads);
    }
    std::vector<PreparedChunkedHostCSR *> preparedPerBed;
    if (useAttenuation) {
      preparedPerBed.resize(static_cast<size_t>(bedCount), nullptr);
      const uint64_t perBedPreparedHostCacheBudgetBytes =
          selectPreparedChunkHostCacheBudgetBytes(
              matrix, static_cast<size_t>(bedCount));
      for (int bed = 0; bed < bedCount; ++bed) {
        preparedPerBed[static_cast<size_t>(bed)] =
            getPreparedAttenuatedHostChunks(
                matrix, (*bedAttenFactors)[static_cast<size_t>(bed)], numAngles,
                nnzBudget, threads, false,
                perBedPreparedHostCacheBudgetBytes);
      }
    }
    const auto &chunks = getContiguousRowChunks(matrix, nnzBudget);
    std::vector<TimedCudaStagePair> uploadWaitTimings;
    std::vector<TimedCudaStagePair> bedCycleTimings;
    std::vector<TimedCudaStagePair> forwardTimings;
    std::vector<TimedCudaStagePair> ratioTimings;
    std::vector<TimedCudaStagePair> backwardTimings;
    if (!useAttenuation) {
      const size_t opCount = chunks.size() * static_cast<size_t>(bedCount);
      uploadWaitTimings.reserve(chunks.size());
      bedCycleTimings.reserve(opCount);
      forwardTimings.reserve(opCount);
      ratioTimings.reserve(opCount);
      backwardTimings.reserve(opCount);
    }

    uploadDoubleAsync(dInput_, image, outputSize, computeStream_.get());
    uploadDoubleAsync(dProjectionInput_, projections.data(), projections.size(),
                                        computeStream_.get());
    dAccumOutput_.allocate(outputSize);
    if (outputSize > 0) {
      CUDA_CHECK(cudaMemsetAsync(dAccumOutput_.get(), 0,
                                 outputSize * sizeof(float),
                                 computeStream_.get()));
    }
    resetTransientRangeState();

    constexpr float eps = 1e-12f;
    if (!useAttenuation) {
      const size_t transientSlotCount = transientRangeCache_.size();
      const auto tryQueueChunkUploadToSlot = [&](size_t chunkIndex,
                                                  size_t slot) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        if (slot >= transientRangeCache_.size()) {
          return false;
        }
        if (transientRangeSlotUsed_[slot] &&
            !transientRangeReusableReady_[slot]) {
          return false;
        }
        // Use prepared cache for resident chunks — avoids rebuilding.
        // Non-resident chunks go through the transient scratch path
        // to prevent LRU thrashing (sequential access would evict
        // chunks needed in the next iteration).
        if (preparedHost != nullptr &&
            preparedHost->ready[chunkIndex]) {
          touchPreparedChunk(*preparedHost, chunkIndex);
          queueTransientRangeUpload(
              slot, matrixKey,
              preparedHost->chunks[chunkIndex]);
          return true;
        }
        const auto buildStart = std::chrono::steady_clock::now();
        waitTransientPinnedScratchReusable(slot);
        buildContiguousSubsetCSR(matrix, chunk.rowBegin, chunk.rowEnd, threads,
                                 transientRangePinnedScratch_[slot]);
        noteCudaChunkBuild(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          buildStart)
                .count());
        queueTransientRangeUpload(slot, matrixKey,
                                  transientRangePinnedScratch_[slot]);
        return true;
      };
      const auto runChunkCompute = [&](size_t chunkIndex,
                                       const CachedMatrix &cached) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        const size_t chunkRows =
            static_cast<size_t>(chunk.rowEnd - chunk.rowBegin);
        dOutput_.allocate(chunkRows);
        dChunkInput_.allocate(chunkRows);

        for (int bed = 0; bed < bedCount; ++bed) {
          bedCycleTimings.emplace_back();
          bedCycleTimings.back().recordStart(computeStream_.get());
          const size_t colOffset = resolveColOffset(bed);
          const size_t projOffset =
              static_cast<size_t>(bed) * rowsPerBed +
              static_cast<size_t>(chunk.rowBegin);

          CuSparseVecGuard vecXForward;
          CuSparseVecGuard vecYForward;
          vecXForward.create(cached.cols, dInput_.get() + colOffset);
          vecYForward.create(cached.rows, dOutput_.get());
          forwardTimings.emplace_back();
          forwardTimings.back().recordStart(computeStream_.get());
          executeSpMVLocked(cached, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            vecXForward.get(), vecYForward.get(), 0.0f,
                            computeStream_.get());
          forwardTimings.back().recordStop(computeStream_.get());

          ratioTimings.emplace_back();
          ratioTimings.back().recordStart(computeStream_.get());
          launchProjDomainRatioKernel(
              dOutput_.get(), dProjectionInput_.get(), projOffset,
              dChunkInput_.get(), chunkRows, eps, computeStream_.get());
          ratioTimings.back().recordStop(computeStream_.get());

          CuSparseVecGuard vecXTranspose;
          CuSparseVecGuard vecYTranspose;
          vecXTranspose.create(cached.rows, dChunkInput_.get());
          vecYTranspose.create(cached.cols, dAccumOutput_.get() + colOffset);
          backwardTimings.emplace_back();
          backwardTimings.back().recordStart(computeStream_.get());
          executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                            vecXTranspose.get(), vecYTranspose.get(), 1.0f,
                            computeStream_.get());
          backwardTimings.back().recordStop(computeStream_.get());
          bedCycleTimings.back().recordStop(computeStream_.get());
        }
      };

      // P3-C+D: Build interleaved schedule that overlaps cached-chunk
      // compute with uncached-chunk uploads.  When device-cached and
      // transient chunks coexist, processing them in original order
      // (all cached first) wastes the GPU compute time of cached chunks
      // that could hide upload latency.  Instead, we interleave:
      //   [prime U0] [C0] [wait U0, prime U1] [U0] [C1] [wait U1, ...]
      // Each cached chunk's compute time (~0.043s) overlaps with the
      // next uncached chunk's upload (~0.058s), hiding most of the
      // transfer cost.
      std::vector<size_t> cachedChunks;
      std::vector<size_t> uncachedChunks;
      cachedChunks.reserve(chunks.size());
      uncachedChunks.reserve(chunks.size());
      for (size_t ci = 0; ci < chunks.size(); ++ci) {
        if (deviceChunkCache_.isHit(ci, matrixKey)) {
          cachedChunks.push_back(ci);
        } else {
          uncachedChunks.push_back(ci);
        }
      }

      // Build interleaved schedule: alternate uncached→cached so that
      // each cached-chunk compute overlaps with the next upload.
      // Pattern: U, C, U, C, ... then remaining of whichever is longer.
      std::vector<size_t> schedule;
      schedule.reserve(chunks.size());
      size_t ci = 0, ui = 0;
      while (ui < uncachedChunks.size() || ci < cachedChunks.size()) {
        if (ui < uncachedChunks.size()) schedule.push_back(uncachedChunks[ui++]);
        if (ci < cachedChunks.size())   schedule.push_back(cachedChunks[ci++]);
      }

      // Prime transient slots with first uncached chunks from schedule.
      size_t nextUncachedIdx = 0;  // index into uncachedChunks
      while (nextUncachedIdx < uncachedChunks.size() &&
             nextUncachedIdx < transientSlotCount) {
        const size_t slot = nextUncachedIdx % transientSlotCount;
        const bool queued =
            tryQueueChunkUploadToSlot(uncachedChunks[nextUncachedIdx], slot);
        if (!queued) {
          throw std::runtime_error(
              "CUDA chunk upload failed to initialize stable slots.");
        }
        ++nextUncachedIdx;
      }

      // Process chunks in interleaved order.
      size_t transientCounter = 0;  // counts uncached chunks processed
      for (const size_t chunkIndex : schedule) {
        if (deviceChunkCache_.isHit(chunkIndex, matrixKey)) {
          // Device cache hit — compute directly, no upload needed.
          // GPU compute here overlaps with any in-flight upload.
          ++g_cudaChunkStageTiming.deviceCacheHitCalls;
          runChunkCompute(chunkIndex, *deviceChunkCache_.slots[chunkIndex]);
        } else {
          // Transient upload path.
          const size_t slot = transientCounter % transientSlotCount;
          uploadWaitTimings.emplace_back();
          const CachedMatrix &cached =
              waitTransientRangeUpload(slot, &uploadWaitTimings.back());
          runChunkCompute(chunkIndex, cached);

          markTransientRangeReusable(slot);
          ++transientCounter;

          // Queue next uncached chunk if available.
          if (nextUncachedIdx < uncachedChunks.size()) {
            const size_t nextSlot = nextUncachedIdx % transientSlotCount;
            const bool queued = tryQueueChunkUploadToSlot(
                uncachedChunks[nextUncachedIdx], nextSlot);
            if (!queued) {
              throw std::runtime_error(
                  "CUDA chunk upload lost stable slot sequencing.");
            }
            ++nextUncachedIdx;
          }
        }
      }
    } else {
      struct AttenTask {
        size_t chunkIndex = 0;
        int bed = 0;
      };
      std::vector<AttenTask> tasks;
      tasks.reserve(chunks.size() * static_cast<size_t>(bedCount));
      for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        for (int bed = 0; bed < bedCount; ++bed) {
          tasks.push_back(AttenTask{chunkIndex, bed});
        }
      }

      uploadWaitTimings.reserve(tasks.size());
      bedCycleTimings.reserve(tasks.size());
      forwardTimings.reserve(tasks.size());
      ratioTimings.reserve(tasks.size());
      backwardTimings.reserve(tasks.size());
      const auto tryQueueTaskUpload = [&](size_t taskIndex) {
        const size_t slot = taskIndex % transientRangeCache_.size();
        const AttenTask &task = tasks[taskIndex];
        const RowChunkRange &chunk = chunks[task.chunkIndex];
        if (transientRangeSlotUsed_[slot] &&
            !transientRangeReusableReady_[slot]) {
          return false;
        }
        // Use prepared cache for resident chunks; otherwise build
        // into transient scratch to avoid LRU thrashing.
        PreparedChunkedHostCSR *preparedAtten =
            preparedPerBed[static_cast<size_t>(task.bed)];
        if (preparedAtten != nullptr &&
            preparedAtten->ready[task.chunkIndex]) {
          touchPreparedChunk(*preparedAtten, task.chunkIndex);
          queueTransientRangeUpload(
              slot, matrixKey,
              preparedAtten->chunks[task.chunkIndex]);
          return true;
        }
        const auto buildStart = std::chrono::steady_clock::now();
        waitTransientPinnedScratchReusable(slot);
        buildAttenuatedContiguousSubsetCSR(
            matrix, (*bedAttenFactors)[static_cast<size_t>(task.bed)],
            numAngles, chunk.rowBegin, chunk.rowEnd, threads,
            transientRangePinnedScratch_[slot]);
        noteCudaChunkBuild(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          buildStart)
                .count());
        queueTransientRangeUpload(slot, matrixKey,
                                  transientRangePinnedScratch_[slot]);
        return true;
      };

      size_t nextTaskToQueue = 0;
      while (nextTaskToQueue < tasks.size() &&
             nextTaskToQueue < transientRangeCache_.size()) {
        const bool queued = tryQueueTaskUpload(nextTaskToQueue);
        if (!queued) {
          throw std::runtime_error(
              "CUDA attenuated chunk upload failed to initialize stable slots.");
        }
        ++nextTaskToQueue;
      }

      for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
        const AttenTask &task = tasks[taskIndex];
        const RowChunkRange &chunk = chunks[task.chunkIndex];
        const size_t slot = taskIndex % transientRangeCache_.size();
        uploadWaitTimings.emplace_back();
        const CachedMatrix &cached =
            waitTransientRangeUpload(slot, &uploadWaitTimings.back());
        const size_t chunkRows =
            static_cast<size_t>(chunk.rowEnd - chunk.rowBegin);
        dOutput_.allocate(chunkRows);
        dChunkInput_.allocate(chunkRows);

        bedCycleTimings.emplace_back();
        bedCycleTimings.back().recordStart(computeStream_.get());
        const size_t colOffset = resolveColOffset(task.bed);
        const size_t projOffset =
            static_cast<size_t>(task.bed) * rowsPerBed +
            static_cast<size_t>(chunk.rowBegin);

        CuSparseVecGuard vecXForward;
        CuSparseVecGuard vecYForward;
        vecXForward.create(cached.cols, dInput_.get() + colOffset);
        vecYForward.create(cached.rows, dOutput_.get());
        forwardTimings.emplace_back();
        forwardTimings.back().recordStart(computeStream_.get());
        executeSpMVLocked(cached, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          vecXForward.get(), vecYForward.get(), 0.0f,
                          computeStream_.get());
        forwardTimings.back().recordStop(computeStream_.get());

        ratioTimings.emplace_back();
        ratioTimings.back().recordStart(computeStream_.get());
        launchProjDomainRatioKernel(dOutput_.get(), dProjectionInput_.get(),
                                    projOffset, dChunkInput_.get(), chunkRows,
                                    eps, computeStream_.get());
        ratioTimings.back().recordStop(computeStream_.get());

        CuSparseVecGuard vecXTranspose;
        CuSparseVecGuard vecYTranspose;
        vecXTranspose.create(cached.rows, dChunkInput_.get());
        vecYTranspose.create(cached.cols, dAccumOutput_.get() + colOffset);
        backwardTimings.emplace_back();
        backwardTimings.back().recordStart(computeStream_.get());
        executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                          vecXTranspose.get(), vecYTranspose.get(), 1.0f,
                          computeStream_.get());
        backwardTimings.back().recordStop(computeStream_.get());
        bedCycleTimings.back().recordStop(computeStream_.get());

        markTransientRangeReusable(slot);
        if (nextTaskToQueue < tasks.size()) {
          const bool queued = tryQueueTaskUpload(nextTaskToQueue);
          if (!queued) {
            throw std::runtime_error(
                "CUDA attenuated chunk upload lost stable slot sequencing.");
          }
          ++nextTaskToQueue;
        }
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    noteCudaUploadWaitOnCompute(accumulateTimedCudaPairs(uploadWaitTimings),
                                uploadWaitTimings.size());
    noteCudaFusedBedCycle(accumulateTimedCudaPairs(bedCycleTimings),
                          bedCycleTimings.size());
    noteCudaCusparseForward(accumulateTimedCudaPairs(forwardTimings),
                            forwardTimings.size());
    noteCudaRatio(accumulateTimedCudaPairs(ratioTimings), ratioTimings.size());
    noteCudaCusparseBackward(accumulateTimedCudaPairs(backwardTimings),
                             backwardTimings.size());
    downloadDouble(dAccumOutput_, numer, outputSize);
  }

  void runFullSubsetBsrem(const io::CSRView &matrix,
                          const std::vector<double> &projections,
                          const double *image, int bedCount,
                          const std::vector<long long> *bedOffsets,
                          const std::vector<std::vector<uint16_t>> *bedAttenFactors,
                          int numAngles, int threads, double *numer) {
    runFullSubsetEmNumer(matrix, projections, image, bedCount, bedOffsets,
                         bedAttenFactors, numAngles, threads, numer);
  }

private:
  size_t compactBackwardAccumulatorSlot(int accumulatorIndex) const {
    if (accumulatorIndex < 0 ||
        accumulatorIndex >= kCompactDeviceBackwardMaxAccumulators) {
      throw std::runtime_error(
          "compact device-backward accumulator index is out of range.");
    }
    return static_cast<size_t>(accumulatorIndex);
  }

  CuSparseHandleGuard handle_;
  CachedMatrix cache_;
  std::array<CachedMatrix, 4> subsetCache_;
  std::array<SubsetCacheKey, 4> subsetCacheKeys_{};
  std::array<bool, 4> subsetCacheValid_{};
  size_t nextSubsetCacheSlot_ = 0;
  CachedMatrix streamingSubsetCache_;
  SubsetCacheKey streamingSubsetKey_{};
  bool streamingSubsetValid_ = false;
  CachedMatrix streamingRangeCache_;
  RangeCacheKey streamingRangeKey_{};
  bool streamingRangeValid_ = false;
  std::array<CachedMatrix, kTransientRangeSlotCount> transientRangeCache_;
  std::array<bool, kTransientRangeSlotCount> transientRangeValid_{};
  std::array<bool, kTransientRangeSlotCount> transientRangeSlotUsed_{};
  std::array<bool, kTransientRangeSlotCount> transientRangeReusableReady_{};
  std::array<PinnedHostCSRData, kTransientRangeSlotCount>
      transientRangePinnedScratch_;
  std::array<CudaEventGuard, kTransientRangeSlotCount> transientRangeUploadStart_;
  std::array<CudaEventGuard, kTransientRangeSlotCount> transientRangeReady_;
  std::array<CudaEventGuard, kTransientRangeSlotCount> transientRangeReusable_;
  std::array<bool, kTransientRangeSlotCount> transientRangeUploadTimingPending_{};
  DeviceChunkCache deviceChunkCache_;            // P3-D
  ResidentMatrixSetCache residentMatrixSetCache_;
  HostCSRData subsetHostScratch_;
  std::array<CachedMatrix, 4> attenCache_;
  std::array<AttenCacheKey, 4> attenCacheKeys_{};
  std::array<bool, 4> attenCacheValid_{};
  size_t nextAttenCacheSlot_ = 0;
  HostCSRData attenHostScratch_;
  DeviceArray<float> dInput_;
  DeviceArray<float> dOutput_;
  DeviceArray<float> dProjectionInput_;
  DeviceArray<float> dCompactProjectionInput_;
  DeviceArray<float> dChunkInput_;
  DeviceArray<float> dProjChunkScratch_;
  DeviceArray<float> dProjKernelX_;
  DeviceArray<float> dProjKernelY_;
  DeviceArray<float> dAccumOutput_;
  DeviceArray<float> dFullOutput_;
  std::array<DeviceArray<double>, kCompactDeviceBackwardMaxAccumulators>
      dCompactBackwardAccum_;
  std::vector<float> projKernelXHost_;
  std::vector<float> projKernelYHost_;
  std::vector<float> hostFloatScratch_;
  const double *compactProjectionHostKey_ = nullptr;
  size_t compactProjectionCount_ = 0;
  std::array<size_t, kCompactDeviceBackwardMaxAccumulators>
      compactBackwardAccumSize_{};
  DeviceBuffer dBuffer_;
  // Compact CTAC: resident attenuation factor table + baked-values scratch.
  DeviceArray<uint16_t> dAttenTable_;
  const void *attenTableHostPtr_ = nullptr;
  size_t attenTableCount_ = 0;
  DeviceArray<float> dBakedValues_;
  CuSparseMatGuard bakedMat_;
  MatrixCacheKey bakedMatrixKey_{};
  cuda::CompactAttenBakeParams bakedParams_{};
  const void *bakedAttPtr_ = nullptr;
  const void *bakedRowPtrDev_ = nullptr;
  const void *bakedColIdxDev_ = nullptr;
  bool bakedValid_ = false;
  std::array<size_t, 2> bakedBufferSizes_{};
  std::array<bool, 2> bakedBufferSizeValid_{};
  CudaStreamGuard computeStream_;
  std::array<CudaStreamGuard, kTransientRangeSlotCount> transientUploadStreams_;
  std::mutex mutex_;
};

SparseOpsCUDAState &cudaState() {
  // CUDA runtime teardown order at process exit is not stable across driver and
  // MSVC CRT versions. Keep the backend state alive until process termination;
  // the OS/driver reclaims these resources and we avoid use-after-runtime
  // crashes from static destructors.
  static SparseOpsCUDAState *state = new SparseOpsCUDAState();
  return *state;
}

class SparseOpsCUDAImpl final : public ISparseOps {
public:
  BackendType backend() const override { return BackendType::CUDA; }

  void spmv(const CSRView &matrix, const double *x, double *y,
            int threads) const override {
    (void)threads;
    cudaState().executeSpMV(matrix, x, y, CUSPARSE_OPERATION_NON_TRANSPOSE);
  }

  void spmvT(const CSRView &matrix, const double *x, double *y,
             int threads) const override {
    (void)threads;
    cudaState().executeSpMV(matrix, x, y, CUSPARSE_OPERATION_TRANSPOSE);
  }

  void computeNormAT(const CSRView &matrix, double *norm,
                     int threads) const override {
    (void)threads;
    std::vector<double> ones(static_cast<size_t>(matrix.rows), 1.0);
    std::fill(norm, norm + static_cast<size_t>(matrix.cols), 0.0);
    spmvT(matrix, ones.data(), norm, threads);
  }

  void spmv(const io::CSRView &matrix, const double *x, double *y,
            int threads = 1) const override {
    if (needsWholeMatrixChunking(matrix)) {
      cudaState().executeChunkedWholeMatrixSpMV(
          matrix, x, y, CUSPARSE_OPERATION_NON_TRANSPOSE, threads);
      return;
    }
    cudaState().executeSpMV(matrix, x, y, CUSPARSE_OPERATION_NON_TRANSPOSE);
  }

  void spmvT(const io::CSRView &matrix, const double *x, double *y,
             int threads = 1) const override {
    if (needsWholeMatrixChunking(matrix)) {
      cudaState().executeChunkedWholeMatrixSpMV(
          matrix, x, y, CUSPARSE_OPERATION_TRANSPOSE, threads);
      return;
    }
    cudaState().executeSpMV(matrix, x, y, CUSPARSE_OPERATION_TRANSPOSE);
  }

  void computeNormAT(const io::CSRView &matrix, double *norm,
                     int threads = 1) const override {
    std::vector<double> ones(static_cast<size_t>(matrix.rows), 1.0);
    std::fill(norm, norm + static_cast<size_t>(matrix.cols), 0.0);
    spmvT(matrix, ones.data(), norm, threads);
  }

  void computeNormAT_Atten(const io::CSRView &matrix,
                           const std::vector<uint16_t> &atten, int numAngles,
                           double *norm, int threads = 1) const override {
    std::vector<double> ones(static_cast<size_t>(matrix.rows), 1.0);
    std::fill(norm, norm + static_cast<size_t>(matrix.cols), 0.0);
    cudaState().executeAttenuatedSpMV(matrix, atten, numAngles, ones.data(),
                                      norm, CUSPARSE_OPERATION_TRANSPOSE,
                                      threads);
  }

  void spmvSubset(const io::CSRView &matrix, const double *x,
                  const std::vector<int> &subset_rows,
                  const SparseSubsetContext &ctx, double *y,
                  int threads = 1) const override {
    const int bedCount = std::max(1, ctx.bedCount);
    const bool orderedFullSubset =
        isOrderedFullSubset(subset_rows,
                            matrix.rows * static_cast<int64_t>(bedCount));
    const bool usableOffsets =
        (bedCount == 1) || hasUsableOffsets(ctx, matrix.cols);
    const bool usableAtten = hasUsableAttenuation(ctx, matrix.cols);
    if (isFullSingleBedSubset(subset_rows, matrix.rows, ctx)) {
      spmv(matrix, x, y, threads);
      return;
    }
    if (orderedFullSubset && usableAtten && usableOffsets) {
      const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
      for (int bed = 0; bed < bedCount; ++bed) {
        const size_t colOffset =
            (bedCount == 1)
                ? 0
                : static_cast<size_t>(
                      (*ctx.bedOffsetVoxels)[static_cast<size_t>(bed)]);
        cudaState().executeAttenuatedSpMV(
            matrix, (*ctx.bedAttenFactors)[static_cast<size_t>(bed)],
            ctx.numAngles, x + colOffset,
            y + rowsPerBed * static_cast<size_t>(bed),
            CUSPARSE_OPERATION_NON_TRANSPOSE, threads);
      }
      return;
    }
    if (ctx.bedCount == 1 && (!ctx.bedOffsetVoxels || ctx.bedOffsetVoxels->empty()) &&
        (!ctx.bedAttenFactors || ctx.bedAttenFactors->empty())) {
      cudaState().executeSubsetSpMV(matrix, subset_rows, x, y,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    threads);
      return;
    }
    if (isFullMultiBedSubset(subset_rows, matrix.rows, ctx) &&
        hasUsableOffsets(ctx, matrix.cols)) {
      const int bedCount = std::max(1, ctx.bedCount);
      const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
      if (needsWholeMatrixChunking(matrix)) {
        cudaState().executeChunkedFullMultiBedSpMV(
            matrix, x, y, *ctx.bedOffsetVoxels, bedCount,
            CUSPARSE_OPERATION_NON_TRANSPOSE, threads);
        return;
      }
      for (int bed = 0; bed < bedCount; ++bed) {
        const size_t colOffset = static_cast<size_t>(
            (*ctx.bedOffsetVoxels)[static_cast<size_t>(bed)]);
        spmv(matrix, x + colOffset,
             y + rowsPerBed * static_cast<size_t>(bed), threads);
      }
      return;
    }
    ISparseOps::spmvSubset(matrix, x, subset_rows, ctx, y, threads);
  }

  void spmvTSubset(const io::CSRView &matrix, const double *x,
                   const std::vector<int> &subset_rows,
                   const SparseSubsetContext &ctx, double *y,
                   int threads = 1) const override {
    const int bedCount = std::max(1, ctx.bedCount);
    const bool orderedFullSubset =
        isOrderedFullSubset(subset_rows,
                            matrix.rows * static_cast<int64_t>(bedCount));
    const bool usableOffsets =
        (bedCount == 1) || hasUsableOffsets(ctx, matrix.cols);
    const bool usableAtten = hasUsableAttenuation(ctx, matrix.cols);
    if (isFullSingleBedSubset(subset_rows, matrix.rows, ctx)) {
      std::fill(y, y + static_cast<size_t>(matrix.cols), 0.0);
      spmvT(matrix, x, y, threads);
      return;
    }
    if (orderedFullSubset && usableAtten && usableOffsets) {
      const size_t outputSize = (bedCount == 1)
                                    ? static_cast<size_t>(matrix.cols)
                                    : static_cast<size_t>(
                                          (*ctx.bedOffsetVoxels)
                                              [static_cast<size_t>(bedCount - 1)] +
                                          matrix.cols);
      std::fill(y, y + outputSize, 0.0);
      const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
      std::vector<double> local(static_cast<size_t>(matrix.cols), 0.0);
      for (int bed = 0; bed < bedCount; ++bed) {
        cudaState().executeAttenuatedSpMV(
            matrix, (*ctx.bedAttenFactors)[static_cast<size_t>(bed)],
            ctx.numAngles, x + rowsPerBed * static_cast<size_t>(bed),
            local.data(), CUSPARSE_OPERATION_TRANSPOSE, threads);
        const size_t colOffset =
            (bedCount == 1)
                ? 0
                : static_cast<size_t>(
                      (*ctx.bedOffsetVoxels)[static_cast<size_t>(bed)]);
        for (size_t index = 0; index < local.size(); ++index) {
          y[colOffset + index] += local[index];
        }
      }
      return;
    }
    if (ctx.bedCount == 1 && (!ctx.bedOffsetVoxels || ctx.bedOffsetVoxels->empty()) &&
        (!ctx.bedAttenFactors || ctx.bedAttenFactors->empty())) {
      std::fill(y, y + static_cast<size_t>(matrix.cols), 0.0);
      cudaState().executeSubsetSpMV(matrix, subset_rows, x, y,
                                    CUSPARSE_OPERATION_TRANSPOSE, threads);
      return;
    }
    if (isFullMultiBedSubset(subset_rows, matrix.rows, ctx) &&
        hasUsableOffsets(ctx, matrix.cols)) {
      const size_t outputSize = static_cast<size_t>(
          (*ctx.bedOffsetVoxels)[static_cast<size_t>(ctx.bedCount - 1)] +
          matrix.cols);
      std::fill(y, y + outputSize, 0.0);
      const int bedCount = std::max(1, ctx.bedCount);
      const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
      if (needsWholeMatrixChunking(matrix)) {
        cudaState().executeChunkedFullMultiBedSpMV(
            matrix, x, y, *ctx.bedOffsetVoxels, bedCount,
            CUSPARSE_OPERATION_TRANSPOSE, threads);
        return;
      }
      std::vector<double> local(static_cast<size_t>(matrix.cols), 0.0);
      for (int bed = 0; bed < bedCount; ++bed) {
        spmvT(matrix, x + rowsPerBed * static_cast<size_t>(bed), local.data(),
              threads);
        const size_t colOffset = static_cast<size_t>(
            (*ctx.bedOffsetVoxels)[static_cast<size_t>(bed)]);
        for (size_t index = 0; index < local.size(); ++index) {
          y[colOffset + index] += local[index];
        }
      }
      return;
    }
    ISparseOps::spmvTSubset(matrix, x, subset_rows, ctx, y, threads);
  }

  void computeSubsetSensitivity(const io::CSRView &matrix,
                                const std::vector<int> &subset_rows, int N,
                                const SparseSubsetContext &ctx,
                                double *sensitivity,
                                int threads = 1) const override {
    const int bedCount = std::max(1, ctx.bedCount);
    const bool orderedFullSubset =
        isOrderedFullSubset(subset_rows,
                            matrix.rows * static_cast<int64_t>(bedCount));
    const bool usableOffsets =
        (bedCount == 1) || hasUsableOffsets(ctx, matrix.cols);
    const bool usableAtten = hasUsableAttenuation(ctx, matrix.cols);
    if (isFullSingleBedSubset(subset_rows, matrix.rows, ctx) &&
        N == matrix.cols) {
      computeNormAT(matrix, sensitivity, threads);
      return;
    }
    if (orderedFullSubset && usableAtten && usableOffsets && N >= matrix.cols) {
      std::fill(sensitivity, sensitivity + static_cast<size_t>(N), 0.0);
      std::vector<double> local(static_cast<size_t>(matrix.cols), 0.0);
      for (int bed = 0; bed < bedCount; ++bed) {
        computeNormAT_Atten(matrix,
                            (*ctx.bedAttenFactors)[static_cast<size_t>(bed)],
                            ctx.numAngles, local.data(), threads);
        const size_t colOffset =
            (bedCount == 1)
                ? 0
                : static_cast<size_t>(
                      (*ctx.bedOffsetVoxels)[static_cast<size_t>(bed)]);
        for (size_t index = 0; index < local.size(); ++index) {
          sensitivity[colOffset + index] += local[index];
        }
      }
      return;
    }
    if (ctx.bedCount == 1 && N == matrix.cols &&
        (!ctx.bedOffsetVoxels || ctx.bedOffsetVoxels->empty()) &&
        (!ctx.bedAttenFactors || ctx.bedAttenFactors->empty())) {
      std::vector<double> ones(subset_rows.size(), 1.0);
      std::fill(sensitivity, sensitivity + static_cast<size_t>(matrix.cols), 0.0);
      cudaState().executeSubsetSpMV(matrix, subset_rows, ones.data(),
                                    sensitivity, CUSPARSE_OPERATION_TRANSPOSE,
                                    threads);
      return;
    }
    if (isFullMultiBedSubset(subset_rows, matrix.rows, ctx) && N >= matrix.cols &&
        hasUsableOffsets(ctx, matrix.cols)) {
      std::fill(sensitivity, sensitivity + static_cast<size_t>(N), 0.0);
      std::vector<double> local(static_cast<size_t>(matrix.cols), 0.0);
      const int bedCount = std::max(1, ctx.bedCount);
      computeNormAT(matrix, local.data(), threads);
      for (int bed = 0; bed < bedCount; ++bed) {
        const size_t colOffset = static_cast<size_t>(
            (*ctx.bedOffsetVoxels)[static_cast<size_t>(bed)]);
        for (size_t index = 0; index < local.size(); ++index) {
          sensitivity[colOffset + index] += local[index];
        }
      }
      return;
    }
    ISparseOps::computeSubsetSensitivity(matrix, subset_rows, N, ctx,
                                         sensitivity, threads);
  }
};

} // namespace

const ISparseOps &getSparseOpsCUDA() {
  static const SparseOpsCUDAImpl instance;
  return instance;
}

namespace cuda {

uint64_t getProjDomainChunkNnzBudgetCUDA() {
  return selectCudaChunkNnzBudget();
}

uint64_t getCudaPreparedChunkHostCacheBudgetCapCUDA() {
  return selectPreparedChunkHostCacheBudgetCapBytes();
}

void setCudaChunkNnzBudgetOverrideForTestingCUDA(uint64_t nnzBudget) {
  g_cudaChunkNnzBudgetOverride.store(nnzBudget, std::memory_order_relaxed);
}

void setCudaPreparedChunkHostCacheBytesOverrideForTestingCUDA(uint64_t bytes) {
  g_cudaPreparedChunkHostCacheBytesOverride.store(bytes,
                                                  std::memory_order_relaxed);
}

void setCudaHostChunkBuildThreadsOverrideForTestingCUDA(int threads) {
  g_cudaHostChunkBuildThreadsOverride.store(threads,
                                            std::memory_order_relaxed);
}

void setCudaSkipBackwardTransposeForTestingCUDA(bool /*skip*/) {
  // Code.CUDA.40: backward transpose scratch removed; this API is now a no-op.
}

int resolveCudaHostChunkBuildThreadsForTestingCUDA(int requestedThreads,
                                                   uint64_t workItems) {
  return getHostParallelThreads(requestedThreads, workItems);
}

uint64_t resolveCudaPreparedChunkHostCacheBudgetForTestingCUDA(
    size_t activePreparedCaches) {
  return selectPreparedChunkHostCacheBudgetBytes(activePreparedCaches);
}

bool runProjDomainChunkedStreamingCUDA(
    const io::CSRView &matrix, const std::vector<std::vector<int>> &chunkRows,
    const std::vector<double> &projections,
    const filters::PSFKernels &psfKernels, size_t frameSize, int detNz,
    int detNx, const double *image, int threads, double *numer) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().runProjDomainChunkedStreaming(
      matrix, chunkRows, projections, psfKernels, frameSize, detNz, detNx,
      image, threads, numer);
  return true;
}

bool runFullSubsetBsremCUDA(const io::CSRView &matrix,
                            const std::vector<double> &projections,
                            const double *image, int bedCount,
                            const std::vector<long long> *bedOffsets,
                            const std::vector<std::vector<uint16_t>> *bedAttenFactors,
                            int numAngles,
                            int threads, double *numer) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().runFullSubsetEmNumer(matrix, projections, image, bedCount,
                                   bedOffsets, bedAttenFactors, numAngles,
                                   threads, numer);
  return true;
}

bool runFullSubsetEmNumerCUDA(const io::CSRView &matrix,
                              const std::vector<double> &projections,
                              const double *image, int bedCount,
                              const std::vector<long long> *bedOffsets,
                              const std::vector<std::vector<uint16_t>> *bedAttenFactors,
                              int numAngles,
                              int threads, double *numer) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().runFullSubsetEmNumer(matrix, projections, image, bedCount,
                                   bedOffsets, bedAttenFactors, numAngles,
                                   threads, numer);
  return true;
}

bool warmupFullSubsetChunkCacheCUDA(
    const io::CSRView &matrix, int bedCount,
    const std::vector<std::vector<uint16_t>> *bedAttenFactors, int numAngles,
    int threads) {
  if (!isCudaAvailable()) {
    return false;
  }
  return cudaState().warmupFullSubsetChunkCache(matrix, bedCount,
                                                bedAttenFactors, numAngles,
                                                threads);
}

bool warmupResidentMatrixSetCUDA(const std::vector<io::CSRView> &matrices,
                                 uint64_t reserveBytes) {
  if (!isCudaAvailable()) {
    return false;
  }
  return cudaState().warmupResidentMatrixSet(matrices, reserveBytes);
}

bool spmvFloatCUDA(const io::CSRView &matrix, const float *x, float *y) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeSpMVFloat(matrix, x, y, CUSPARSE_OPERATION_NON_TRANSPOSE);
  return true;
}

bool spmvTFloatCUDA(const io::CSRView &matrix, const float *x, float *y) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeSpMVFloat(matrix, x, y, CUSPARSE_OPERATION_TRANSPOSE);
  return true;
}

bool compactAttenSpmvFloatCUDA(const io::CSRView &matrix,
                               const uint16_t *attFactors, size_t attCount,
                               const cuda::CompactAttenBakeParams &params,
                               const float *x, float *y) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactAttenSpMVFloat(matrix, attFactors, attCount,
                                           params, x, y,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE);
  return true;
}

bool compactAttenSpmvTFloatCUDA(const io::CSRView &matrix,
                                const uint16_t *attFactors, size_t attCount,
                                const cuda::CompactAttenBakeParams &params,
                                const float *x, float *y) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactAttenSpMVFloat(matrix, attFactors, attCount,
                                           params, x, y,
                                           CUSPARSE_OPERATION_TRANSPOSE);
  return true;
}

bool compactDeviceBackwardBeginCUDA(size_t voxelCount, int accumulatorIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().beginCompactDeviceBackward(voxelCount, accumulatorIndex);
  return true;
}

bool compactDeviceBackwardSpmvTAccumulateFloatCUDA(
    const io::CSRView &matrix, const float *x,
    const cuda::CompactDeviceTransformParams &params,
    cuda::CompactDeviceBackwardTiming *timing, int accumulatorIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactDeviceBackwardSpmvTAccumulateFloat(matrix, x,
                                                               params, timing,
                                                               accumulatorIndex);
  return true;
}

bool compactDeviceBackwardDownloadCUDA(double *y, size_t voxelCount,
                                       int accumulatorIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().downloadCompactDeviceBackward(y, voxelCount, accumulatorIndex);
  return true;
}

bool compactDeviceRatioSpmvTAccumulateFloatCUDA(
    const io::CSRView &matrix, const float *x, const double *fullProjection,
    size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &ratioParams,
    const cuda::CompactDeviceTransformParams &transformParams,
    cuda::CompactDeviceRatioTiming *ratioTiming,
    cuda::CompactDeviceBackwardTiming *backwardTiming, int accumulatorIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactDeviceRatioSpmvTAccumulateFloat(
      matrix, x, fullProjection, fullProjectionCount, ratioParams,
      transformParams, ratioTiming, backwardTiming, accumulatorIndex);
  return true;
}

bool compactDeviceRatioSpmvTFloatCUDA(
    const io::CSRView &matrix, const float *x, const double *fullProjection,
    size_t fullProjectionCount, const cuda::CompactDeviceRatioParams &params,
    float *y, cuda::CompactDeviceRatioTiming *timing) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactDeviceRatioSpmvTFloat(
      matrix, x, fullProjection, fullProjectionCount, params, y, timing);
  return true;
}

void resetCudaChunkStageTimingCUDA() { g_cudaChunkStageTiming = {}; }

CudaChunkStageTimingSnapshot getCudaChunkStageTimingSnapshotCUDA() {
  CudaChunkStageTimingSnapshot snapshot;
  snapshot.chunkBuildSeconds = g_cudaChunkStageTiming.chunkBuildSeconds;
  // pinnedStageSeconds: removed in Code.CUDA.40, defaults to 0.0
  snapshot.h2dChunkSeconds = g_cudaChunkStageTiming.h2dChunkSeconds;
  snapshot.uploadWaitOnComputeSeconds =
      g_cudaChunkStageTiming.uploadWaitOnComputeSeconds;
  snapshot.fusedBedCycleSeconds = g_cudaChunkStageTiming.fusedBedCycleSeconds;
  snapshot.cusparseForwardSeconds =
      g_cudaChunkStageTiming.cusparseForwardSeconds;
  snapshot.cusparseForwardHostSeconds =
      g_cudaChunkStageTiming.cusparseForwardHostSeconds;
  snapshot.ratioSeconds = g_cudaChunkStageTiming.ratioSeconds;
  // transposeChunkSeconds: removed in Code.CUDA.40, defaults to 0.0
  snapshot.cusparseBackwardSeconds =
      g_cudaChunkStageTiming.cusparseBackwardSeconds;
  snapshot.cusparseBackwardHostSeconds =
      g_cudaChunkStageTiming.cusparseBackwardHostSeconds;
  snapshot.compactMatrixUploadSeconds =
      g_cudaChunkStageTiming.compactMatrixUploadSeconds;
  snapshot.compactInputH2DSeconds =
      g_cudaChunkStageTiming.compactInputH2DSeconds;
  snapshot.compactProjectionH2DSeconds =
      g_cudaChunkStageTiming.compactProjectionH2DSeconds;
  snapshot.compactOutputD2HSeconds =
      g_cudaChunkStageTiming.compactOutputD2HSeconds;
  snapshot.compactForwardOutputD2HSeconds =
      g_cudaChunkStageTiming.compactForwardOutputD2HSeconds;
  snapshot.compactBackwardOutputD2HSeconds =
      g_cudaChunkStageTiming.compactBackwardOutputD2HSeconds;
  snapshot.chunkBuildCalls = g_cudaChunkStageTiming.chunkBuildCalls;
  // pinnedStageCalls: removed in Code.CUDA.40, defaults to 0
  snapshot.h2dChunkCalls = g_cudaChunkStageTiming.h2dChunkCalls;
  snapshot.uploadWaitOnComputeCalls =
      g_cudaChunkStageTiming.uploadWaitOnComputeCalls;
  snapshot.fusedBedCycleCalls = g_cudaChunkStageTiming.fusedBedCycleCalls;
  snapshot.cusparseForwardCalls =
      g_cudaChunkStageTiming.cusparseForwardCalls;
  snapshot.cusparseForwardHostCalls =
      g_cudaChunkStageTiming.cusparseForwardHostCalls;
  snapshot.ratioCalls = g_cudaChunkStageTiming.ratioCalls;
  // transposeChunkCalls: removed in Code.CUDA.40, defaults to 0
  snapshot.cusparseBackwardCalls =
      g_cudaChunkStageTiming.cusparseBackwardCalls;
  snapshot.cusparseBackwardHostCalls =
      g_cudaChunkStageTiming.cusparseBackwardHostCalls;
  snapshot.compactMatrixUploadCalls =
      g_cudaChunkStageTiming.compactMatrixUploadCalls;
  snapshot.compactMatrixReuseCalls =
      g_cudaChunkStageTiming.compactMatrixReuseCalls;
  snapshot.compactInputH2DCalls =
      g_cudaChunkStageTiming.compactInputH2DCalls;
  snapshot.compactProjectionH2DCalls =
      g_cudaChunkStageTiming.compactProjectionH2DCalls;
  snapshot.compactProjectionH2DBytes =
      g_cudaChunkStageTiming.compactProjectionH2DBytes;
  snapshot.compactOutputD2HCalls =
      g_cudaChunkStageTiming.compactOutputD2HCalls;
  snapshot.compactForwardOutputD2HCalls =
      g_cudaChunkStageTiming.compactForwardOutputD2HCalls;
  snapshot.compactBackwardOutputD2HCalls =
      g_cudaChunkStageTiming.compactBackwardOutputD2HCalls;
  snapshot.compactForwardOutputD2HBytes =
      g_cudaChunkStageTiming.compactForwardOutputD2HBytes;
  snapshot.compactBackwardOutputD2HBytes =
      g_cudaChunkStageTiming.compactBackwardOutputD2HBytes;
  snapshot.deviceCacheHitCalls = g_cudaChunkStageTiming.deviceCacheHitCalls;
  return snapshot;
}

} // namespace cuda

} // namespace spect

#undef SPECT_CUSPARSE_CHECK

#else

namespace spect {

namespace {

class SparseOpsCUDAUnavailable final : public ISparseOps {
public:
  BackendType backend() const override { return BackendType::CUDA; }

  void spmv(const CSRView &, const double *, double *, int) const override {
    throw std::runtime_error(
        "CUDA sparse backend was enabled for selection, but this build does not "
        "contain SPECT_HAS_CUDA support.");
  }

  void spmvT(const CSRView &, const double *, double *, int) const override {
    throw std::runtime_error(
        "CUDA sparse backend was enabled for selection, but this build does not "
        "contain SPECT_HAS_CUDA support.");
  }

  void computeNormAT(const CSRView &, double *, int) const override {
    throw std::runtime_error(
        "CUDA sparse backend was enabled for selection, but this build does not "
        "contain SPECT_HAS_CUDA support.");
  }
};

} // namespace

const ISparseOps &getSparseOpsCUDA() {
  static const SparseOpsCUDAUnavailable instance;
  return instance;
}

namespace cuda {

uint64_t getProjDomainChunkNnzBudgetCUDA() {
  return kProjDomainChunkMinNnz;
}

uint64_t getCudaPreparedChunkHostCacheBudgetCapCUDA() { return 0; }

bool runProjDomainChunkedStreamingCUDA(
    const io::CSRView &, const std::vector<std::vector<int>> &,
    const std::vector<double> &, const filters::PSFKernels &, size_t, int, int,
    const double *, int, double *) {
  return false;
}

bool runFullSubsetBsremCUDA(const io::CSRView &, const std::vector<double> &,
                            const double *, int,
                            const std::vector<long long> *,
                            const std::vector<std::vector<uint16_t>> *, int,
                            int, double *) {
  return false;
}

bool runFullSubsetEmNumerCUDA(const io::CSRView &,
                              const std::vector<double> &, const double *, int,
                              const std::vector<long long> *,
                              const std::vector<std::vector<uint16_t>> *, int,
                              int, double *) {
  return false;
}

bool warmupFullSubsetChunkCacheCUDA(
    const io::CSRView &, int, const std::vector<std::vector<uint16_t>> *, int,
    int) {
  return false;
}

bool warmupResidentMatrixSetCUDA(const std::vector<io::CSRView> &,
                                 uint64_t) {
  return false;
}

bool spmvFloatCUDA(const io::CSRView &, const float *, float *) {
  return false;
}

bool spmvTFloatCUDA(const io::CSRView &, const float *, float *) {
  return false;
}

bool compactAttenSpmvFloatCUDA(const io::CSRView &, const uint16_t *, size_t,
                               const cuda::CompactAttenBakeParams &,
                               const float *, float *) {
  return false;
}

bool compactAttenSpmvTFloatCUDA(const io::CSRView &, const uint16_t *, size_t,
                                const cuda::CompactAttenBakeParams &,
                                const float *, float *) {
  return false;
}

void resetCudaChunkStageTimingCUDA() {}

CudaChunkStageTimingSnapshot getCudaChunkStageTimingSnapshotCUDA() {
  return {};
}

void setCudaChunkNnzBudgetOverrideForTestingCUDA(uint64_t) {}
void setCudaPreparedChunkHostCacheBytesOverrideForTestingCUDA(uint64_t) {}
void setCudaHostChunkBuildThreadsOverrideForTestingCUDA(int) {}
int resolveCudaHostChunkBuildThreadsForTestingCUDA(int, uint64_t) { return 1; }

} // namespace cuda

} // namespace spect

#endif

#endif
