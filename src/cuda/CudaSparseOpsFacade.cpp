#include "sparse/SparseOpsBackend.h"
#include "core/AttenuationConstants.h"
#include "cuda/CudaDriverHelpers.h"
#include "cuda/ProjDomainChunked.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)

#include "CudaChunkCache.h"
#include "CudaCsrBuilder.h"
#include "CudaProjKernelLauncher.h"
#include "cuda/CudaConfig.h"
#include "filters/Convolution.h"

#if CUDA_ENABLED

#include <cuda.h>
#include <cusparse.h>

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

#include "CudaSparseOpsState.h"

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
                               const float *x, float *y, int cacheSlot) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactAttenSpMVFloat(matrix, attFactors, attCount,
                                           params, x, y,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           cacheSlot);
  return true;
}

bool compactAttenSpmvTFloatCUDA(const io::CSRView &matrix,
                                const uint16_t *attFactors, size_t attCount,
                                const cuda::CompactAttenBakeParams &params,
                                const float *x, float *y, int cacheSlot) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactAttenSpMVFloat(matrix, attFactors, attCount,
                                           params, x, y,
                                           CUSPARSE_OPERATION_TRANSPOSE,
                                           cacheSlot);
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
    cuda::CompactDeviceBackwardTiming *backwardTiming, int accumulatorIndex,
    int cacheSlot) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactDeviceRatioSpmvTAccumulateFloat(
      matrix, x, fullProjection, fullProjectionCount, ratioParams,
      transformParams, ratioTiming, backwardTiming, accumulatorIndex, cacheSlot);
  return true;
}

bool compactDeviceRatioAttenSpmvTAccumulateFloatCUDA(
    const io::CSRView &matrix, const float *x, const uint16_t *attFactors,
    size_t attCount, const cuda::CompactAttenBakeParams &attenParams,
    const double *fullProjection, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &ratioParams,
    const cuda::CompactDeviceTransformParams &transformParams,
    cuda::CompactDeviceRatioTiming *ratioTiming,
    cuda::CompactDeviceBackwardTiming *backwardTiming, int accumulatorIndex,
    int cacheSlot) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactDeviceRatioAttenSpmvTAccumulateFloat(
      matrix, x, attFactors, attCount, attenParams, fullProjection,
      fullProjectionCount, ratioParams, transformParams, ratioTiming,
      backwardTiming, accumulatorIndex, cacheSlot);
  return true;
}

bool compactDeviceRatioSpmvTFloatCUDA(
    const io::CSRView &matrix, const float *x, const double *fullProjection,
    size_t fullProjectionCount, const cuda::CompactDeviceRatioParams &params,
    float *y, cuda::CompactDeviceRatioTiming *timing, int cacheSlot) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactDeviceRatioSpmvTFloat(
      matrix, x, fullProjection, fullProjectionCount, params, y, timing,
      cacheSlot);
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
