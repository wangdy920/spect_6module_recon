#include "sparse/SparseOpsCUDA.h"
#include "core/AttenuationConstants.h"
#include "cuda/CudaDriverHelpers.h"
#include "sparse/ProjDomainChunked.h"
#include "sparse/CompactSubsets.h"

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
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
    const std::vector<double> *castorKz, const uint8_t *localSupportMask) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().prepareCompactBranchBatchInputFromGlobal(
      globalImage, globalImageCount, globalNz, bedZStarts, bedCount,
      flipLocalZ, uploadGlobalImage, commonTransformParams, branchCount,
      zBranchCount, castorKx, castorKy, castorKz, localSupportMask);
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
    bool *imageAllZero, int subsetIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().finishCompactResidentMlemIteration(
      globalImageCount, globalNz, bedZStarts, bedCount, flipLocalZ,
      commonTransformParams, denom, mask, castorKx, castorKy, castorKz,
      requirePositiveNumerator, refreshStaticInputs, imageAllZero, subsetIndex);
  return true;
}

bool setCompactSubsetSensitivitiesCUDA(const double *const *hostSensitivities,
                                       int count, size_t globalImageCount) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().setCompactSubsetSensitivities(hostSensitivities, count,
                                            globalImageCount);
  return true;
}

bool downloadCompactResidentImageCUDA(double *image, size_t count) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().downloadCompactResidentImage(image, count);
  return true;
}

bool finishCompactResidentBsremIterationCUDA(
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ,
    const cuda::CompactDeviceTransformParams &commonTransformParams,
    const double *denom, const uint8_t *mask,
    const std::vector<double> *castorKx,
    const std::vector<double> *castorKy,
    const std::vector<double> *castorKz,
    const cuda::CompactBsremUpdateParams &params,
    bool requirePositiveNumerator, bool refreshStaticInputs,
    bool *imageAllZero, int subsetIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().finishCompactResidentBsremIteration(
      globalImageCount, globalNz, bedZStarts, bedCount, flipLocalZ,
      commonTransformParams, denom, mask, castorKx, castorKy, castorKz, params,
      requirePositiveNumerator, refreshStaticInputs, imageAllZero, subsetIndex);
  return true;
}

const double *compactResidentImageDeviceCUDA(size_t count) {
  if (!isCudaAvailable()) {
    return nullptr;
  }
  return cudaState().compactResidentImageDevice(count);
}

double *compactResidentGradientDeviceCUDA(size_t count) {
  if (!isCudaAvailable()) {
    return nullptr;
  }
  return cudaState().compactResidentGradientDevice(count);
}

bool prefetchCompactFloatMatrixCUDA(const io::CSRSubsetView &next,
                                    const io::CSRSubsetView &current) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().prefetchCompactFloatMatrix(next, current);
  return true;
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

bool compactDeviceBackwardBeginCUDA(size_t voxelCount, int accumulatorIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().beginCompactDeviceBackward(voxelCount, accumulatorIndex);
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

bool compactDeviceRatioBedBranchBatchAttenSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView &matrix, const double *const *fullProjections,
    size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &commonRatioParams,
    const cuda::CompactDeviceTransformParams &commonTransformParams,
    const int *symIds, const int *zMirrors, int branchCount, int bedCount,
    const uint16_t *const *bedAttFactors, size_t attCount, double attPower,
    cuda::CompactDeviceRatioTiming *ratioTiming,
    cuda::CompactDeviceBackwardTiming *backwardTiming, int cacheSlotBase,
    const std::function<io::CSRSubsetView()> &loadNext) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactBedBranchBatchAttenDeviceRatioSpmvTAccumulateFloat(
      matrix, fullProjections, fullProjectionCount, commonRatioParams,
      commonTransformParams, symIds, zMirrors, branchCount, bedCount,
      bedAttFactors, attCount, attPower, ratioTiming, backwardTiming, 0,
      cacheSlotBase, loadNext);
  return true;
}

bool compactSensitivityBranchBatchAttenSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView &matrix,
    const cuda::CompactDeviceRatioParams &commonRatioParams,
    const cuda::CompactDeviceTransformParams &commonTransformParams,
    const int *symIds, const int *zMirrors, int branchCount,
    const uint16_t *attFactors, size_t attCount, double attPower,
    const double *rowWeights, size_t fullProjectionCount,
    cuda::CompactDeviceBackwardTiming *backwardTiming, int accumulatorIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactSensitivityBranchBatchAttenSpmvTAccumulateFloat(
      matrix, commonRatioParams, commonTransformParams, symIds, zMirrors,
      branchCount, attFactors, attCount, attPower, rowWeights,
      fullProjectionCount, backwardTiming, accumulatorIndex);
  return true;
}

bool compactBranchBatchAttenForwardCUDA(
    const io::CSRView &matrix,
    const cuda::CompactDeviceRatioParams &commonRatioParams,
    const cuda::CompactDeviceTransformParams &commonTransformParams,
    const int *symIds, const int *zMirrors, int branchCount,
    const uint16_t *attFactors, size_t attCount, double attPower,
    float *hostYhatBatch, size_t count) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactBranchBatchAttenForward(
      matrix, commonRatioParams, commonTransformParams, symIds, zMirrors,
      branchCount, attFactors, attCount, attPower, hostYhatBatch, count);
  return true;
}

bool compactDeviceRatioBedBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView &matrix, const double *const *fullProjections,
    size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &commonRatioParams,
    const cuda::CompactDeviceTransformParams &commonTransformParams,
    const int *symIds, const int *zMirrors, int branchCount, int bedCount,
    cuda::CompactDeviceRatioTiming *ratioTiming,
    cuda::CompactDeviceBackwardTiming *backwardTiming, int cacheSlotBase,
    const std::function<io::CSRSubsetView()> &loadNext) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().executeCompactBedBranchBatchDeviceRatioSpmvTAccumulateFloat(
      matrix, fullProjections, fullProjectionCount, commonRatioParams,
      commonTransformParams, symIds, zMirrors, branchCount, bedCount,
      ratioTiming, backwardTiming, 0, cacheSlotBase, loadNext);
  return true;
}

bool compactSensitivityBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView &matrix,
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

bool csrqWarmupBeginCUDA(int matrixIndex, int cols, int branchCount,
                         const std::vector<CsrqChunkShape> &shapes,
                         uint64_t reserveBytes, bool forceStreamed,
                         bool *streamed, const CsrqHostPlan *hostPlan) {
  if (!isCudaAvailable()) {
    return false;
  }
  const bool isStreamed = cudaState().beginCsrqWarmup(
      matrixIndex, cols, branchCount, shapes, reserveBytes, forceStreamed,
      hostPlan);
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

void setPinnedAllocationLimitForTestsCUDA(int limit) {
  PinnedTryAllocateTestLimit::limit.store(std::max(-1, limit));
}

bool setCsrqPackedBackwardCUDA(bool enabled) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().setCsrqPackedBackward(enabled);
  return true;
}

bool csrqPackedBackwardCUDA() {
  if (!isCudaAvailable()) {
    return false;
  }
  return cudaState().csrqPackedBackward();
}

bool csrqRunNumerCUDA(int matrixIndex, int bedCount, int branchCount,
                      const CompactDeviceRatioParams &commonRatioParams,
                      const CompactDeviceTransformParams &commonTransformParams,
                      const int *symIds, const int *zMirrors,
                      const double *const *fullProjections,
                      const double *const *fullBackgrounds,
                      size_t fullProjectionCount, double weight,
                      int accumulatorBase, int cacheSlotBase,
                      const uint16_t *const *bedAttFactors, size_t attCount,
                      double attPower, int subsetCount, int subsetIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().csrqRunNumer(matrixIndex, bedCount, branchCount,
                           commonRatioParams, commonTransformParams, symIds,
                           zMirrors, fullProjections, fullBackgrounds,
                           fullProjectionCount, weight, accumulatorBase,
                           cacheSlotBase, bedAttFactors, attCount, attPower,
                           subsetCount, subsetIndex);
  return true;
}

bool csrqRunSensitivityCUDA(
    int matrixIndex, int branchCount, double weight,
    const CompactDeviceRatioParams &commonRatioParams,
    const CompactDeviceTransformParams &commonTransformParams,
    const int *symIds, const int *zMirrors, int accumulatorIndex,
    size_t fullProjectionCount, const double *rowWeights,
    const uint16_t *attFactors, size_t attCount, double attPower,
    int subsetCount, int subsetIndex) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().csrqRunSensitivity(
      matrixIndex, branchCount, weight, commonRatioParams,
      commonTransformParams, symIds, zMirrors, accumulatorIndex, rowWeights,
      rowWeights == nullptr ? 0 : fullProjectionCount, fullProjectionCount,
      attFactors, attCount, attPower, subsetCount, subsetIndex);
  return true;
}

bool csrqRunForwardCUDA(int matrixIndex, int branchCount,
                        float *hostYhatPacked, size_t count,
                        const CompactDeviceRatioParams *attenRatioParams,
                        const CompactDeviceTransformParams *attenTransformParams,
                        const int *symIds, const int *zMirrors,
                        const uint16_t *attFactors, size_t attCount,
                        double attPower) {
  if (!isCudaAvailable()) {
    return false;
  }
  cudaState().csrqRunForward(matrixIndex, branchCount, hostYhatPacked, count,
                             attenRatioParams, attenTransformParams, symIds,
                             zMirrors, attFactors, attCount, attPower);
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
  snapshot.csrbMatrixH2DBytes = g_cudaChunkStageTiming.csrbMatrixH2DBytes;
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
  snapshot.csrqPassCalls = g_cudaChunkStageTiming.csrqPassCalls;
  snapshot.csrqDiskReadSeconds = g_cudaChunkStageTiming.csrqDiskReadSeconds;
  snapshot.csrqDiskReadBytes = g_cudaChunkStageTiming.csrqDiskReadBytes;
  snapshot.csrqDiskReadCalls = g_cudaChunkStageTiming.csrqDiskReadCalls;
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
    const std::vector<double> *, const uint8_t *) {
  return false;
}

bool setCompactSubsetSensitivitiesCUDA(const double *const *, int, size_t) {
  return false;
}

bool finishCompactResidentMlemIterationCUDA(
    size_t, int, const int *, int, bool,
    const cuda::CompactDeviceTransformParams &, const double *,
    const uint8_t *, const std::vector<double> *, const std::vector<double> *,
    const std::vector<double> *, bool, bool, bool *, int) {
  return false;
}

bool downloadCompactResidentImageCUDA(double *, size_t) { return false; }

bool finishCompactResidentBsremIterationCUDA(
    size_t, int, const int *, int, bool,
    const cuda::CompactDeviceTransformParams &, const double *,
    const uint8_t *, const std::vector<double> *, const std::vector<double> *,
    const std::vector<double> *, const cuda::CompactBsremUpdateParams &, bool,
    bool, bool *, int) {
  return false;
}

const double *compactResidentImageDeviceCUDA(size_t) { return nullptr; }

double *compactResidentGradientDeviceCUDA(size_t) { return nullptr; }

bool prefetchCompactFloatMatrixCUDA(const io::CSRSubsetView &,
                                    const io::CSRSubsetView &) {
  return false;
}

bool releaseCompactExecutionCUDA() { return false; }

bool spmvFloatCUDA(const io::CSRView &, const float *, float *) {
  return false;
}

bool spmvTFloatCUDA(const io::CSRView &, const float *, float *) {
  return false;
}

bool compactDeviceRatioBedBranchBatchAttenSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView &, const double *const *, size_t,
    const CompactDeviceRatioParams &, const CompactDeviceTransformParams &,
    const int *, const int *, int, int, const uint16_t *const *, size_t, double,
    CompactDeviceRatioTiming *, CompactDeviceBackwardTiming *, int) {
  return false;
}

bool compactSensitivityBranchBatchAttenSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView &, const CompactDeviceRatioParams &,
    const CompactDeviceTransformParams &, const int *, const int *, int,
    const uint16_t *, size_t, double, const double *, size_t,
    CompactDeviceBackwardTiming *, int) {
  return false;
}

bool compactBranchBatchAttenForwardCUDA(const io::CSRView &,
                                        const CompactDeviceRatioParams &,
                                        const CompactDeviceTransformParams &,
                                        const int *, const int *, int,
                                        const uint16_t *, size_t, double,
                                        float *, size_t) {
  return false;
}

bool csrqWarmupBeginCUDA(int, int, int, const std::vector<CsrqChunkShape> &,
                         uint64_t, bool, bool *, const CsrqHostPlan *) {
  return false;
}

bool csrqWarmupChunkCUDA(int, size_t, const io::CompactRunU16View &) {
  return false;
}

bool csrqWarmupEndCUDA(int) { return false; }

bool csrqResidentCUDA(int) { return false; }

bool setCsrqPackedBackwardCUDA(bool) { return false; }

bool csrqPackedBackwardCUDA() { return false; }

bool csrqRunNumerCUDA(int, int, int, const CompactDeviceRatioParams &,
                      const CompactDeviceTransformParams &, const int *,
                      const int *, const double *const *,
                      const double *const *, size_t, double, int, int,
                      const uint16_t *const *, size_t, double, int, int) {
  return false;
}

bool csrqRunSensitivityCUDA(int, int, double,
                            const CompactDeviceRatioParams &,
                            const CompactDeviceTransformParams &, const int *,
                            const int *, int, size_t, const double *,
                            const uint16_t *, size_t, double, int, int) {
  return false;
}

bool csrqRunForwardCUDA(int, int, float *, size_t,
                        const CompactDeviceRatioParams *,
                        const CompactDeviceTransformParams *, const int *,
                        const int *, const uint16_t *, size_t, double) {
  return false;
}

void resetCudaChunkStageTimingCUDA() {}

CudaChunkStageTimingSnapshot getCudaChunkStageTimingSnapshotCUDA() {
  return {};
}

} // namespace cuda

} // namespace spect

#endif

#endif
