#pragma once

#include "cuda/CudaConfig.h"
#include "cuda/CudaLaunchConfig.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include "sparse/SparseOpsCUDA.h"

#include <cstddef>
#include <cstdint>

namespace spect {

void ensureProjDomainKernelsLoaded();

void launchProjDomainRatioKernel(const float *yhat, const float *projections,
                                 size_t projectionOffset, float *ratio,
                                 size_t count, float eps,
                                 cudaStream_t stream = nullptr);

// fullBackground: device additive background (same layout as fullProjection)
// or nullptr for none.
void launchCompactRatioKernel(
    const float *yhat, const double *fullProjection,
    const double *fullBackground, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &params, float *ratio, size_t count,
    double eps, cudaStream_t stream = nullptr);

void launchCompactRatioBatchKernel(
    const float *yhatBatch, const double *fullProjection,
    const double *fullBackground, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &commonParams, const int *dSymIds,
    const int *dZMirrors, int batchSize, float *ratioBatch, size_t rows,
    double eps, cudaStream_t stream = nullptr);

void launchCompactGatherBedDoubleToFloatKernel(
    const double *globalImage, size_t globalCount, float *bedImage, int nx,
    int ny, int nz, int globalNz, int zStart, bool flipLocalZ,
    cudaStream_t stream = nullptr);

void launchCompactConvertDoubleToFloatKernel(
    const double *input, float *output, size_t count,
    cudaStream_t stream = nullptr);

void launchCompactAccumulateBedFloatToGlobalDoubleKernel(
    const float *bedImage, double *globalImage, size_t globalCount, int nx,
    int ny, int nz, int globalNz, int zStart, bool flipLocalZ,
    cudaStream_t stream = nullptr);

void launchCompactMlemUpdateKernel(
    double *image, const double *numer, const double *denom,
    const uint8_t *mask, bool hasMask, size_t count, uint32_t *flags,
    double eps, cudaStream_t stream = nullptr);

void launchCompactTransformForwardBatchKernel(
    const float *bedImage, float *rotatedBatch, size_t nvox, int branchCount,
    int zBranchCount, const cuda::CompactDeviceTransformParams &commonParams,
    cudaStream_t stream = nullptr);

void launchCompactTransformBackwardAddKernel(
    const float *rotated, double *accum, size_t nvox,
    const cuda::CompactDeviceTransformParams &params,
    cudaStream_t stream = nullptr);

void launchCompactTransformBackwardAddBatchKernel(
    const float *rotatedBatch, double *accum, size_t nvox, int batchSize,
    const cuda::CompactDeviceTransformParams &commonParams, const int *dSymIds,
    const int *dZMirrors, cudaStream_t stream = nullptr);

void launchCompactBakeAttenKernel(const float *values, const int32_t *rowPtr,
                                  const int32_t *colIdx, const uint16_t *att,
                                  float *baked, int rows,
                                  const cuda::CompactAttenBakeParams &params,
                                  cudaStream_t stream = nullptr);

void launchCompactAttenFusedForwardKernel(
    const float *values, const int32_t *rowPtr, const int32_t *colIdx,
    const uint16_t *att, const float *x, float *y, int rows,
    const cuda::CompactAttenBakeParams &params,
    cudaStream_t stream = nullptr);

void launchCompactAttenFusedBackwardKernel(
    const float *values, const int32_t *rowPtr, const int32_t *colIdx,
    const uint16_t *att, const float *x, float *y, int rows,
    const cuda::CompactAttenBakeParams &params,
    cudaStream_t stream = nullptr);

void launchCompactPackColPairToFloat2Kernel(const float *x0, const float *x1,
                                            float *xPair, int cols,
                                            cudaStream_t stream = nullptr);

void launchCompactCsrSpmvXPairKernel(const float *values, const int32_t *rowPtr,
                                     const int32_t *colIdx, const float *xPair,
                                     float *y0, float *y1, int rows,
                                     cudaStream_t stream = nullptr);

void launchCompactPackColQuadToFloat4Kernel(const float *x0, const float *x1,
                                            const float *x2, const float *x3,
                                            float *xQuad, int cols,
                                            cudaStream_t stream = nullptr);

void launchCompactCsrSpmvXQuadKernel(const float *values, const int32_t *rowPtr,
                                     const int32_t *colIdx, const float *xQuad,
                                     float *y0, float *y1, float *y2, float *y3,
                                     int rows, cudaStream_t stream = nullptr);

// CSRQ resident-matrix kernels (projdomain_chunk_ops.cu, "CSRQ kernels").
// K = 8 right-hand sides packed [voxel][8]; block size is fixed by the
// kernel's shared-memory tile layout.
constexpr int kCsrqK = cuda::kCsrqBranchCount;
constexpr unsigned int kCsrqBlockSize = 256;

struct CsrqDeviceChunkPtrs {
  const uint32_t *rowNnzPtr = nullptr;
  const uint32_t *rowRunPtr = nullptr;
  const float *rowMax = nullptr;
  const uint32_t *runs = nullptr;
  const uint16_t *valuesQ = nullptr;
  int rows = 0;
};

// yPacked[row][8] = A x for the eight branches (prediction output).
void launchCsrqForwardK8Kernel(const CsrqDeviceChunkPtrs &chunk,
                               const float *xPacked, float *yPacked,
                               cudaStream_t stream = nullptr);

// Same kernel, ratio mode: ratioPacked[row][8] =
//   scale * p / max(scale * A x + background, eps), 0 where the (row, branch)
// has no full projection row. fullBackground may be null.
void launchCsrqForwardRatioK8Kernel(
    const CsrqDeviceChunkPtrs &chunk, const float *xPacked, float *ratioPacked,
    const double *fullProjection, const double *fullBackground,
    size_t fullProjectionCount, double scale,
    const cuda::CompactDeviceRatioParams &params, const int *dSymIds,
    const int *dZMirrors, double eps, cudaStream_t stream = nullptr);

// ratioPacked is [row][8]; yBranch is branch-major [8][nvox] (the layout
// launchCompactTransformBackwardAddBatchKernel consumes). Only branches
// [kFirst, kFirst + kCount) are scattered.
void launchCsrqBackwardK8Kernel(const CsrqDeviceChunkPtrs &chunk,
                                const float *ratioPacked, float *yBranch,
                                size_t nvox, int kFirst, int kCount,
                                cudaStream_t stream = nullptr);

// Branches scattered per backward pass over the matrix. Measured on the
// 120x120x220 Tc99m matrix (RTX 4070 Ti SUPER, ms per full-matrix backward):
// 1 branch x 8 passes 207, 2 x 4 150, 4 x 2 179, 8 x 1 191. Two branches keep
// the 25 MB atomic target inside the 48 MB L2 while the per-pass matrix read
// and tile expansion stay amortized; the kernel has a dedicated two-branch
// path. The remaining floor is L2 atomic throughput (~22 G sector atomics/s),
// i.e. the atomic count itself, not their locality.
constexpr int kCsrqBackwardBranchesPerPass = 2;

void launchCsrqPackBranch8Kernel(const float *branchMajor, float *packed,
                                 size_t nvox, cudaStream_t stream = nullptr);

// ratio = value * rowWeights[fullIndex] (value alone when rowWeights is null)
// where the (row, branch) has a full projection row, else 0.
void launchCsrqUnitRatioK8Kernel(float *ratioPacked, float value,
                                 const double *rowWeights,
                                 size_t fullProjectionCount,
                                 const cuda::CompactDeviceRatioParams &params,
                                 const int *dSymIds, const int *dZMirrors,
                                 int rows, cudaStream_t stream = nullptr);

// CSRB branch-batch sensitivity input: ratioBatch[branch*rows + row] =
// rowWeights[fullIndex] (1 when null) for valid (row, branch), else 0.
void launchCompactUnitRatioBatchKernel(
    const double *rowWeights, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &commonParams, const int *dSymIds,
    const int *dZMirrors, int batchSize, float *ratioBatch, size_t rows,
    cudaStream_t stream = nullptr);

} // namespace spect

#endif
