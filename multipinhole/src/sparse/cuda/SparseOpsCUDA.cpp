#include "sparse/SparseOpsCUDA.h"
#include "core/AttenuationConstants.h"
#include "cuda/CudaDriverHelpers.h"
#include "sparse/ProjDomainChunked.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)

#include "CudaChunkCache.h"
#include "CudaCsrBuilder.h"
#include "CudaProjKernelLauncher.h"
#include "filters/cuda/VolumeOpsCUDA.h"
#include "core/ReconLimits.h"
#include "cuda/CudaConfig.h"
#include "cuda/CudaLaunchConfig.h"
#include "io/CompactChunkCSR.h"

#if CUDA_ENABLED

#include <cuda.h>
#include <cusparse.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <thread>
#include <cstring>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

#include "CudaSparseOpsState.h"

} // namespace


namespace cuda {

uint64_t getProjDomainChunkNnzBudgetCUDA() {
  return selectCudaChunkNnzBudget();
}

uint64_t getCudaPreparedChunkHostCacheBudgetCapCUDA() {
  return selectPreparedChunkHostCacheBudgetCapBytes();
}

bool warmupResidentMatrixSetCUDA(const std::vector<io::CSRView> &matrices,
                                 uint64_t reserveBytes) {
  if (!isCudaAvailable()) {
    return false;
  }
  return cudaState().warmupResidentMatrixSet(matrices, reserveBytes);
}

void noteCompactCsrLocalityCUDA(bool sortedHint) {
  if (!isCudaAvailable()) {
    return;
  }
  cudaState().noteCompactCsrLocality(sortedHint);
}

bool refreshCompactProjectionCUDA(const double *projection, size_t count,
                                  int cacheSlot, const double *background) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().refreshCompactProjection(projection, count, cacheSlot,
                                       background);
  return true;
}

bool refreshCompactBranchBatchInputCUDA(const float *input, size_t count) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().refreshCompactBranchBatchInput(input, count);
  return true;
}

bool prepareCompactBranchBatchInputFromGlobalCUDA(
    const double *globalImage, size_t globalImageCount, int globalNz,
    const int *bedZStarts, int bedCount, bool flipLocalZ,
    bool uploadGlobalImage,
    const cuda::CompactDeviceTransformParams &commonTransformParams,
    int branchCount, int zBranchCount,
    const std::vector<double> *castorKx,
    const std::vector<double> *castorKy,
    const std::vector<double> *castorKz) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().prepareCompactBranchBatchInputFromGlobal(
      globalImage, globalImageCount, globalNz, bedZStarts, bedCount,
      flipLocalZ, uploadGlobalImage, commonTransformParams, branchCount,
      zBranchCount, castorKx, castorKy, castorKz);
  return true;
}

bool finishCompactResidentMlemIterationCUDA(
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ,
    const cuda::CompactDeviceTransformParams &commonTransformParams,
    const double *denom, const uint8_t *mask,
    const std::vector<double> *castorKx,
    const std::vector<double> *castorKy,
    const std::vector<double> *castorKz,
    bool requirePositiveNumerator, bool refreshStaticInputs,
    bool *imageAllZero) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().finishCompactResidentMlemIteration(
      globalImageCount, globalNz, bedZStarts, bedCount, flipLocalZ,
      commonTransformParams, denom, mask, castorKx, castorKy, castorKz,
      requirePositiveNumerator, refreshStaticInputs, imageAllZero);
  return true;
}

bool downloadCompactResidentImageCUDA(double *image, size_t count) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().downloadCompactResidentImage(image, count);
  return true;
}

bool prefetchCompactFloatMatrixCUDA(const io::CSRView &next,
                                    const io::CSRView &current) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().prefetchCompactFloatMatrix(next, current);
  return true;
}

bool prepareCompactTransposeHostCacheCUDA(const io::CSRView &matrix) {
  if (!isCudaAvailable()) {
    return false;
  }
  return cudaState().prepareCompactTransposeHost(matrix);
}

bool releaseCompactExecutionCUDA() {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().releaseCompactExecution();
  return true;
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

bool compactDeviceRatioBedBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRView &matrix, const double *const *fullProjections,
    size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &commonRatioParams,
    const cuda::CompactDeviceTransformParams &commonTransformParams,
    const int *symIds, const int *zMirrors, int branchCount, int bedCount,
    cuda::CompactDeviceRatioTiming *ratioTiming,
    cuda::CompactDeviceBackwardTiming *backwardTiming) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactBedBranchBatchDeviceRatioSpmvTAccumulateFloat(
      matrix, fullProjections, fullProjectionCount, commonRatioParams,
      commonTransformParams, symIds, zMirrors, branchCount, bedCount,
      ratioTiming, backwardTiming);
  return true;
}

bool compactSensitivityBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRView &matrix,
    const cuda::CompactDeviceTransformParams &commonTransformParams,
    const int *symIds, const int *zMirrors, int batchSize,
    cuda::CompactDeviceBackwardTiming *backwardTiming, int accumulatorIndex,
    int cacheSlot, const cuda::CompactDeviceRatioParams *commonRatioParams,
    const double *rowWeights, size_t rowWeightCount) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactSensitivityBranchBatchSpmvTAccumulateFloat(
      matrix, commonTransformParams, symIds, zMirrors, batchSize,
      backwardTiming, accumulatorIndex, cacheSlot, commonRatioParams,
      rowWeights, rowWeightCount);
  return true;
}

bool csrqWarmupBeginCUDA(int matrixIndex, int cols,
                         const std::vector<CsrqChunkShape> &shapes,
                         uint64_t reserveBytes, bool forceStreamed,
                         bool *streamed) {
  if (!isCudaAvailable()) {
    return false;
  }
  const bool isStreamed = cudaState().beginCsrqWarmup(
      matrixIndex, cols, shapes, reserveBytes, forceStreamed);
  if (streamed != nullptr) {
    *streamed = isStreamed;
  }
  return true;
}

bool csrqWarmupChunkCUDA(int matrixIndex, size_t chunkIndex,
                         const io::CompactRunU16View &view) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().warmupCsrqChunk(matrixIndex, chunkIndex, view);
  return true;
}

bool csrqWarmupEndCUDA(int matrixIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().endCsrqWarmup(matrixIndex);
  return true;
}

bool csrqResidentCUDA(int matrixIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  return cudaState().csrqResident(matrixIndex);
}

bool csrqRunNumerCUDA(int matrixIndex, int bedCount, int branchCount,
                      const CompactDeviceRatioParams &commonRatioParams,
                      const CompactDeviceTransformParams &commonTransformParams,
                      const int *symIds, const int *zMirrors,
                      const double *const *fullProjections,
                      const double *const *fullBackgrounds,
                      size_t fullProjectionCount, double weight,
                      int accumulatorBase, int cacheSlotBase,
                      CompactDeviceRatioTiming *ratioTiming,
                      CompactDeviceBackwardTiming *backwardTiming) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().csrqRunNumer(matrixIndex, bedCount, branchCount,
                           commonRatioParams, commonTransformParams, symIds,
                           zMirrors, fullProjections, fullBackgrounds,
                           fullProjectionCount, weight, accumulatorBase,
                           cacheSlotBase, ratioTiming, backwardTiming);
  return true;
}

bool csrqRunSensitivityCUDA(
    int matrixIndex, int branchCount, double weight,
    const CompactDeviceRatioParams &commonRatioParams,
    const CompactDeviceTransformParams &commonTransformParams,
    const int *symIds, const int *zMirrors, int accumulatorIndex,
    size_t fullProjectionCount, const double *rowWeights,
    CompactDeviceBackwardTiming *backwardTiming) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().csrqRunSensitivity(
      matrixIndex, branchCount, weight, commonRatioParams,
      commonTransformParams, symIds, zMirrors, accumulatorIndex, rowWeights,
      rowWeights == nullptr ? 0 : fullProjectionCount, fullProjectionCount,
      backwardTiming);
  return true;
}

bool csrqRunForwardCUDA(int matrixIndex, int branchCount,
                        float *hostYhatPacked, size_t count) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().csrqRunForward(matrixIndex, branchCount, hostYhatPacked, count);
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
  snapshot.transformBackwardSeconds =
      g_cudaChunkStageTiming.transformBackwardSeconds;
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
  snapshot.transformBackwardCalls =
      g_cudaChunkStageTiming.transformBackwardCalls;
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
  snapshot.csrqForwardSeconds = g_cudaChunkStageTiming.csrqForwardSeconds;
  snapshot.csrqBackwardSeconds = g_cudaChunkStageTiming.csrqBackwardSeconds;
  snapshot.csrqForwardCalls = g_cudaChunkStageTiming.csrqForwardCalls;
  snapshot.csrqBackwardCalls = g_cudaChunkStageTiming.csrqBackwardCalls;
  return snapshot;
}

} // namespace cuda

} // namespace spect

#undef SPECT_CUSPARSE_CHECK

#else

namespace spect {



namespace cuda {

uint64_t getProjDomainChunkNnzBudgetCUDA() {
  return kProjDomainChunkMinNnz;
}

uint64_t getCudaPreparedChunkHostCacheBudgetCapCUDA() { return 0; }

bool warmupResidentMatrixSetCUDA(const std::vector<io::CSRView> &,
                                 uint64_t) {
  return false;
}

bool refreshCompactProjectionCUDA(const double *, size_t, int,
                                  const double *) {
  return false;
}

bool refreshCompactBranchBatchInputCUDA(const float *, size_t) {
  return false;
}

bool prepareCompactBranchBatchInputFromGlobalCUDA(
    const double *, size_t, int, const int *, int, bool, bool,
    const cuda::CompactDeviceTransformParams &, int, int,
    const std::vector<double> *, const std::vector<double> *,
    const std::vector<double> *) {
  return false;
}

bool finishCompactResidentMlemIterationCUDA(
    size_t, int, const int *, int, bool,
    const cuda::CompactDeviceTransformParams &, const double *,
    const uint8_t *, const std::vector<double> *, const std::vector<double> *,
    const std::vector<double> *, bool, bool, bool *) {
  return false;
}

bool downloadCompactResidentImageCUDA(double *, size_t) { return false; }

bool prefetchCompactFloatMatrixCUDA(const io::CSRView &, const io::CSRView &) {
  return false;
}

bool prepareCompactTransposeHostCacheCUDA(const io::CSRView &) {
  return false;
}

bool releaseCompactExecutionCUDA() { return false; }

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

bool csrqWarmupBeginCUDA(int, int, const std::vector<CsrqChunkShape> &,
                         uint64_t, bool, bool *) {
  return false;
}

bool csrqWarmupChunkCUDA(int, size_t, const io::CompactRunU16View &) {
  return false;
}

bool csrqWarmupEndCUDA(int) { return false; }

bool csrqResidentCUDA(int) { return false; }

bool csrqRunNumerCUDA(int, int, int, const CompactDeviceRatioParams &,
                      const CompactDeviceTransformParams &, const int *,
                      const int *, const double *const *,
                      const double *const *, size_t, double, int, int,
                      CompactDeviceRatioTiming *,
                      CompactDeviceBackwardTiming *) {
  return false;
}

bool csrqRunSensitivityCUDA(int, int, double,
                            const CompactDeviceRatioParams &,
                            const CompactDeviceTransformParams &, const int *,
                            const int *, int, size_t, const double *,
                            CompactDeviceBackwardTiming *) {
  return false;
}

bool csrqRunForwardCUDA(int, int, float *, size_t) { return false; }

void resetCudaChunkStageTimingCUDA() {}

CudaChunkStageTimingSnapshot getCudaChunkStageTimingSnapshotCUDA() {
  return {};
}

} // namespace cuda

} // namespace spect

#endif

#endif
