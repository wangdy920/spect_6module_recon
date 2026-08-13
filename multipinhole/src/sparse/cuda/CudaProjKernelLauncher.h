#pragma once

#include "cuda/CudaConfig.h"

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

void launchCompactRatioKernel(
    const float *yhat, const double *fullProjection, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &params, float *ratio, size_t count,
    double eps, cudaStream_t stream = nullptr);

void launchCompactRatioBatchKernel(
    const float *yhatBatch, const double *fullProjection,
    size_t fullProjectionCount,
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


} // namespace spect

#endif
