#pragma once

#include "cuda/CudaLaunchConfig.h"

#include "io/CSRTypes.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spect {
namespace cuda {

struct CudaChunkStageTimingSnapshot {
  double chunkBuildSeconds = 0.0;
  double pinnedStageSeconds = 0.0;
  double h2dChunkSeconds = 0.0;
  double uploadWaitOnComputeSeconds = 0.0;
  double fusedBedCycleSeconds = 0.0;
  double cusparseForwardSeconds = 0.0;
  double cusparseForwardHostSeconds = 0.0;
  double ratioSeconds = 0.0;
  double transposeChunkSeconds = 0.0;
  double cusparseBackwardSeconds = 0.0;
  double cusparseBackwardHostSeconds = 0.0;
  double transformBackwardSeconds = 0.0;
  double compactMatrixUploadSeconds = 0.0;
  double compactInputH2DSeconds = 0.0;
  double compactProjectionH2DSeconds = 0.0;
  double compactOutputD2HSeconds = 0.0;
  double compactForwardOutputD2HSeconds = 0.0;
  double compactBackwardOutputD2HSeconds = 0.0;
  size_t chunkBuildCalls = 0;
  size_t pinnedStageCalls = 0;
  size_t h2dChunkCalls = 0;
  size_t uploadWaitOnComputeCalls = 0;
  size_t fusedBedCycleCalls = 0;
  size_t cusparseForwardCalls = 0;
  size_t cusparseForwardHostCalls = 0;
  size_t ratioCalls = 0;
  size_t transposeChunkCalls = 0;
  size_t cusparseBackwardCalls = 0;
  size_t cusparseBackwardHostCalls = 0;
  size_t transformBackwardCalls = 0;
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
  // P3-D: device-side chunk cache hit counter
  size_t deviceCacheHitCalls = 0;
  // CSRQ resident-matrix custom kernels (per chunk pass).
  double csrqForwardSeconds = 0.0;
  double csrqBackwardSeconds = 0.0;
  size_t csrqForwardCalls = 0;
  size_t csrqBackwardCalls = 0;
};

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)

void resetCudaChunkStageTimingCUDA();
CudaChunkStageTimingSnapshot getCudaChunkStageTimingSnapshotCUDA();

uint64_t getProjDomainChunkNnzBudgetCUDA();
uint64_t getCudaPreparedChunkHostCacheBudgetCapCUDA();

#else

inline void resetCudaChunkStageTimingCUDA() {}
inline CudaChunkStageTimingSnapshot getCudaChunkStageTimingSnapshotCUDA() {
  return {};
}

inline uint64_t getProjDomainChunkNnzBudgetCUDA() { return kProjDomainChunkMinNnz; }
inline uint64_t getCudaPreparedChunkHostCacheBudgetCapCUDA() { return 0; }

#endif

} // namespace cuda
} // namespace spect
