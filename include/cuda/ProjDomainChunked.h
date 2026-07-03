#pragma once

#include "filters/PSFConfig.h"
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
};

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)

bool runProjDomainChunkedStreamingCUDA(
    const io::CSRView &matrix, const std::vector<std::vector<int>> &chunkRows,
    const std::vector<double> &projections,
    const filters::PSFKernels &psfKernels, size_t frameSize, int detNz,
    int detNx, const double *image, int threads, double *numer);

bool runFullSubsetBsremCUDA(const io::CSRView &matrix,
                            const std::vector<double> &projections,
                            const double *image, int bedCount,
                            const std::vector<long long> *bedOffsets,
                            const std::vector<std::vector<uint16_t>> *bedAttenFactors,
                            int numAngles,
                            int threads, double *numer);

bool runFullSubsetEmNumerCUDA(const io::CSRView &matrix,
                              const std::vector<double> &projections,
                              const double *image, int bedCount,
                              const std::vector<long long> *bedOffsets,
                              const std::vector<std::vector<uint16_t>> *bedAttenFactors,
                              int numAngles,
                              int threads, double *numer);

bool warmupFullSubsetChunkCacheCUDA(
    const io::CSRView &matrix, int bedCount,
    const std::vector<std::vector<uint16_t>> *bedAttenFactors, int numAngles,
    int threads);

void resetCudaChunkStageTimingCUDA();
CudaChunkStageTimingSnapshot getCudaChunkStageTimingSnapshotCUDA();
void setCudaChunkNnzBudgetOverrideForTestingCUDA(uint64_t nnzBudget);
void setCudaPreparedChunkHostCacheBytesOverrideForTestingCUDA(uint64_t bytes);
void setCudaHostChunkBuildThreadsOverrideForTestingCUDA(int threads);
void setCudaSkipBackwardTransposeForTestingCUDA(bool skip);
int resolveCudaHostChunkBuildThreadsForTestingCUDA(int requestedThreads,
                                                   uint64_t workItems);
uint64_t resolveCudaPreparedChunkHostCacheBudgetForTestingCUDA(
    size_t activePreparedCaches);

uint64_t getProjDomainChunkNnzBudgetCUDA();
uint64_t getCudaPreparedChunkHostCacheBudgetCapCUDA();

#else

inline bool runProjDomainChunkedStreamingCUDA(
    const io::CSRView &, const std::vector<std::vector<int>> &,
    const std::vector<double> &, const filters::PSFKernels &, size_t, int, int,
    const double *, int, double *) {
  return false;
}

inline bool runFullSubsetBsremCUDA(const io::CSRView &,
                                   const std::vector<double> &,
                                   const double *, int,
                                   const std::vector<long long> *,
                                   const std::vector<std::vector<uint16_t>> *,
                                   int, int,
                                   double *) {
  return false;
}

inline bool runFullSubsetEmNumerCUDA(const io::CSRView &,
                                     const std::vector<double> &,
                                     const double *, int,
                                     const std::vector<long long> *,
                                     const std::vector<std::vector<uint16_t>> *,
                                     int, int,
                                     double *) {
  return false;
}

inline bool warmupFullSubsetChunkCacheCUDA(
    const io::CSRView &, int, const std::vector<std::vector<uint16_t>> *, int,
    int) {
  return false;
}

inline void resetCudaChunkStageTimingCUDA() {}
inline CudaChunkStageTimingSnapshot getCudaChunkStageTimingSnapshotCUDA() {
  return {};
}
inline void setCudaChunkNnzBudgetOverrideForTestingCUDA(uint64_t) {}
inline void setCudaPreparedChunkHostCacheBytesOverrideForTestingCUDA(uint64_t) {}
inline void setCudaHostChunkBuildThreadsOverrideForTestingCUDA(int) {}
inline void setCudaSkipBackwardTransposeForTestingCUDA(bool) {}
inline int resolveCudaHostChunkBuildThreadsForTestingCUDA(int, uint64_t) {
  return 1;
}
inline uint64_t resolveCudaPreparedChunkHostCacheBudgetForTestingCUDA(size_t) {
  return 0;
}

inline uint64_t getProjDomainChunkNnzBudgetCUDA() { return 8ull * 1024ull * 1024ull; }
inline uint64_t getCudaPreparedChunkHostCacheBudgetCapCUDA() { return 0; }

#endif

} // namespace cuda
} // namespace spect
