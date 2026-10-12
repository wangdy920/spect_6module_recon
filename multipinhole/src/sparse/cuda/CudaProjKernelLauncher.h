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

// fullBackground: device additive background (same layout as fullProjection)
// or nullptr for none.
void launchCompactRatioKernel(
    const float *yhat, const double *fullProjection,
    const double *fullBackground, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &params, float *ratio, size_t count,
    double eps, cudaStream_t stream = nullptr);

// Null yhatBatch prepares a binary p!=0 activity mask in ratioBatch.
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
    cudaStream_t stream = nullptr, const uint8_t *mask = nullptr);

void launchCompactAccumulateBedFloatToGlobalDoubleKernel(
    const float *bedImage, double *globalImage, size_t globalCount, int nx,
    int ny, int nz, int globalNz, int zStart, bool flipLocalZ,
    cudaStream_t stream = nullptr);

void launchCompactMlemUpdateKernel(
    double *image, const double *numer, const double *denom,
    const double *globalDenom, const uint8_t *mask, bool hasMask, size_t count,
    uint32_t *flags, double eps, cudaStream_t stream = nullptr);

// curvature = |gradient| / max(image, eps) (BSREM curvature surrogate).
void launchCompactBsremCurvatureKernel(const double *image,
                                       const double *gradient, double eps,
                                       double *curvature, size_t count,
                                       cudaStream_t stream = nullptr);

// BSREM-II relative update in place; gradient / curvature / kappa / mask may
// be null. Flags share the MLEM update layout.
// subsetDenom: sensitivity of the rows in this update (== denom for a
// full-data step); params.subsetScale is the subset count M.
void launchCompactBsremUpdateKernel(
    double *image, const double *numer, const double *denom,
    const double *subsetDenom, const double *gradient,
    const double *curvature, const double *kappa, const uint8_t *mask,
    bool hasMask, size_t count, const cuda::CompactBsremUpdateParams &params,
    uint32_t *flags, cudaStream_t stream = nullptr);

void launchCompactTransformForwardBatchKernel(
    const float *bedImage, float *rotatedBatch, size_t nvox, int branchCount,
    int zBranchCount, const cuda::CompactDeviceTransformParams &commonParams,
    cudaStream_t stream = nullptr, const uint8_t *mask = nullptr);

void launchCompactTransformBackwardAddKernel(
    const float *rotated, double *accum, size_t nvox,
    const cuda::CompactDeviceTransformParams &params,
    cudaStream_t stream = nullptr);

void launchCompactTransformBackwardAddBatchKernel(
    const float *rotatedBatch, double *accum, size_t nvox, int batchSize,
    const cuda::CompactDeviceTransformParams &commonParams, const int *dSymIds,
    const int *dZMirrors, cudaStream_t stream = nullptr);

// CTAC pre-scaling of a branch-major batch (CSRB layout, batchSize dense
// columns of nvox): out[k][v] = in[k][v] * (attSlices[sliceIndex[k]][M_k(v)]
// / 65535)^power (factor 1 where sliceIndex[k] < 0).
void launchCompactBranchBatchAttenScaleKernel(
    const float *branchMajor, const uint16_t *attSlices, const int *dSliceIndex,
    float *out, size_t nvox, int batchSize,
    const cuda::CompactDeviceTransformParams &commonParams, const int *dSymIds,
    const int *dZMirrors, float power, cudaStream_t stream = nullptr);

void launchCompactPackColPairToFloat2Kernel(const float *x0, const float *x1,
                                            float *xPair, int cols,
                                            cudaStream_t stream = nullptr);

void launchCompactCsrSpmvXPairKernel(const float *values, const int32_t *rowPtr,
                                     const int32_t *colIdx, const float *xPair,
                                     float *y0, float *y1, int rows,
                                     cudaStream_t stream = nullptr,
                                     bool zeroProjectionMask = false);

// Transpose scatter y_k[col] += a * x_k[row] for two / four RHS with zero-row
// skipping (rows whose ratios are all exactly zero read no matrix bytes).
// Outputs accumulate; the caller zeroes them.
void launchCompactCsrSpmvTPairKernel(const float *values, const int32_t *rowPtr,
                                     const int32_t *colIdx, const float *x0,
                                     const float *x1, float *y0, float *y1,
                                     int rows, cudaStream_t stream = nullptr);
void launchCompactCsrSpmvTQuadKernel(const float *values, const int32_t *rowPtr,
                                     const int32_t *colIdx, const float *x0,
                                     const float *x1, const float *x2,
                                     const float *x3, float *y0, float *y1,
                                     float *y2, float *y3, int rows,
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
// K = 4 or 8 right-hand sides packed [voxel][K], one shared implementation.
constexpr unsigned int kCsrqBlockSize = 256;

struct CsrqDeviceChunkPtrs {
  const uint32_t *rowNnzPtr = nullptr;
  const uint32_t *rowRunPtr = nullptr;
  const float *rowMax = nullptr;
  const uint32_t *runs = nullptr;
  const uint16_t *valuesQ = nullptr;
  int rows = 0;
  int branchCount = 0;
  // Logical launch rows select interleaved angle groups in the unchanged
  // chunk. Ratios retain physical row offsets; no matrix/ratio gathering.
  int rowOffset = 0;
  int rowGroupSize = 1;
  int rowGroupStride = 1;
  int selectedRows() const {
    if (rowOffset >= rows) return 0;
    const int span = rowGroupSize * rowGroupStride;
    return ((rows - rowOffset + span - 1) / span) * rowGroupSize;
  }
};

// yPacked[row][K] = A x for the active branches (prediction output).
void launchCsrqForwardKernel(const CsrqDeviceChunkPtrs &chunk,
                               const float *xPacked, float *yPacked,
                               cudaStream_t stream = nullptr);

// Same kernel, ratio mode: ratioPacked[row][K] =
//   scale * p / max(scale * A x + background, eps), 0 where the (row, branch)
// has no full projection row. fullBackground may be null.
void launchCsrqForwardRatioKernel(
    const CsrqDeviceChunkPtrs &chunk, const float *xPacked, float *ratioPacked,
    const double *fullProjection, const double *fullBackground,
    size_t fullProjectionCount, double scale,
    const cuda::CompactDeviceRatioParams &params, const int *dSymIds,
    const int *dZMirrors, double eps, cudaStream_t stream = nullptr);

// ratioPacked is [row][K]; yBranch is branch-major [K][nvox] (the layout
// launchCompactTransformBackwardAddBatchKernel consumes). Only branches
// [kFirst, kFirst + kCount) are scattered.
void launchCsrqBackwardKernel(const CsrqDeviceChunkPtrs &chunk,
                                const float *ratioPacked, float *yBranch,
                                size_t nvox, int kFirst, int kCount,
                                cudaStream_t stream = nullptr);

// Packed-output backward (opt-in, sm_120 vector-atomic trial): all active
// branches in one pass into yPacked[nvox][K] (float4 atomics on sm_90+,
// scalar fallback below).
void launchCsrqBackwardPackedKernel(const CsrqDeviceChunkPtrs &chunk,
                                      const float *ratioPacked, float *yPacked,
                                      cudaStream_t stream = nullptr);

// accum[M_k(v)] += yPacked[v][k] (times attSlices[sliceIndex[k]][M_k(v)]^power
// when attSlices is non-null); consumer of the packed backward.
void launchCompactTransformBackwardAddPackedKernel(
    const float *yPacked, double *accum, size_t nvox, int branchCount,
    const cuda::CompactDeviceTransformParams &commonParams, const int *dSymIds,
    const int *dZMirrors, const uint16_t *attSlices = nullptr,
    const int *dSliceIndex = nullptr, float power = 1.0f,
    cudaStream_t stream = nullptr);

// Branches scattered per backward pass over the matrix. Measured on the
// 120x120x220 Tc99m matrix (RTX 4070 Ti SUPER, ms per full-matrix backward):
// 1 branch x 8 passes 207, 2 x 4 150, 4 x 2 179, 8 x 1 191. Two branches keep
// the 25 MB atomic target inside the 48 MB L2 while the per-pass matrix read
// and tile expansion stay amortized; the kernel has a dedicated two-branch
// path. The remaining floor is L2 atomic throughput (~22 G sector atomics/s),
// i.e. the atomic count itself, not their locality.
constexpr int kCsrqBackwardBranchesPerPass = 2;

void launchCsrqPackBranchKernel(const float *branchMajor, float *packed,
                                 size_t nvox, int branchCount, cudaStream_t stream = nullptr);

// CTAC variant for one compact angle: packed[v][k] = branchMajor[k][v] *
// (attSlices[sliceIndex[k]][M_k(v)] / 65535)^power (factor 1 where
// sliceIndex[k] < 0). attSlices holds the staged full-angle factor slices
// ([slice][nvox] uint16); sliceIndex points at the K per-branch slice ids
// of this compact angle; power rescales the table to the window's photon
// energy (1 = as stored).
void launchCsrqPackBranchAttenKernel(
    const float *branchMajor, const uint16_t *attSlices, const int *dSliceIndex,
    float *packed, size_t nvox, int branchCount,
    const cuda::CompactDeviceTransformParams &commonParams, const int *dSymIds,
    const int *dZMirrors, float power = 1.0f, cudaStream_t stream = nullptr);

// launchCompactTransformBackwardAddBatchKernel with the CTAC factor applied at
// the unrotated voxel (adjoint of launchCsrqPackBranchAttenKernel).
void launchCompactTransformBackwardAddBatchAttenKernel(
    const float *rotatedBatch, double *accum, size_t nvox, int batchSize,
    const cuda::CompactDeviceTransformParams &commonParams, const int *dSymIds,
    const int *dZMirrors, const uint16_t *attSlices, const int *dSliceIndex,
    float power = 1.0f, cudaStream_t stream = nullptr);

// ratio = value * rowWeights[fullIndex] (value alone when rowWeights is null)
// where the (row, branch) has a full projection row, else 0.
void launchCsrqUnitRatioKernel(float *ratioPacked, float value,
                                 const double *rowWeights,
                                 size_t fullProjectionCount,
                                 const cuda::CompactDeviceRatioParams &params,
                                 const int *dSymIds, const int *dZMirrors,
                                 int rows, int branchCount, cudaStream_t stream = nullptr);

// CSRB branch-batch sensitivity input: ratioBatch[branch*rows + row] =
// rowWeights[fullIndex] (1 when null) for valid (row, branch), else 0.
void launchCompactUnitRatioBatchKernel(
    const double *rowWeights, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &commonParams, const int *dSymIds,
    const int *dZMirrors, int batchSize, float *ratioBatch, size_t rows,
    cudaStream_t stream = nullptr);

} // namespace spect

#endif
