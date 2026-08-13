/**
 * @file SparseOpsCUDA.h
 * @brief CUDA sparse-operation declarations used by the compact workflow
 */

#pragma once

#include "io/CSRTypes.h"
#include "io/CompactChunkCSR.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spect {


namespace cuda {
struct CompactDeviceRatioParams {
    int symmetryMode = 0; // 45 or 90
    int compactRowBase = 0;
    int compactRaysPerAngle = 0;
    int compactNumAngles = 0;
    int detNx = 0;
    int detNz = 0;
    int compactDetZ = 0;
    int fullNumAngles = 0;
    int fullRaysPerAngle = 0;
    int symId = 0;
    bool zMirror = false;
};

struct CompactDeviceRatioTiming {
    double forwardSeconds = 0.0;
    double ratioSeconds = 0.0;
    double backwardSeconds = 0.0;
};

struct CompactDeviceTransformParams {
    int symmetryMode = 0; // 45 or 90
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int symId = 0;
    bool zMirror = false;
};

struct CompactDeviceBackwardTiming {
    double backwardSeconds = 0.0;
    double transformBackwardSeconds = 0.0;
};

/// Parameters for compact CTAC attenuation lookup:
/// effective[t] = val[t] * att[fullAngle(row(t)), unrotatedVoxel(col(t))].
/// Combines the row->full-angle mapping of CompactDeviceRatioParams with the
/// voxel mapping of CompactDeviceTransformParams.
struct CompactAttenBakeParams {
    int symmetryMode = 0; // 45 or 90
    int compactRowBase = 0;
    int compactRaysPerAngle = 0;
    int compactNumAngles = 0;
    int fullNumAngles = 0;
    int symId = 0;
    bool zMirror = false;
    int nx = 0;
    int ny = 0;
    int nz = 0;
};

} // namespace cuda


namespace cuda {
bool warmupResidentMatrixSetCUDA(const std::vector<io::CSRView>& matrices,
                                 uint64_t reserveBytes = 0);
bool refreshCompactProjectionCUDA(const double* projection, size_t count,
                                  int cacheSlot = 0);
bool refreshCompactBranchBatchInputCUDA(const float* input, size_t count);
bool prepareCompactBranchBatchInputFromGlobalCUDA(
    const double *globalImage, size_t globalImageCount, int globalNz,
    const int *bedZStarts, int bedCount, bool flipLocalZ,
    bool uploadGlobalImage,
    const CompactDeviceTransformParams &commonTransformParams,
    int branchCount, int zBranchCount,
    const std::vector<double> *castorKx = nullptr,
    const std::vector<double> *castorKy = nullptr,
    const std::vector<double> *castorKz = nullptr);
bool finishCompactResidentMlemIterationCUDA(
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ,
    const CompactDeviceTransformParams &commonTransformParams,
    const double *denom, const uint8_t *mask,
    const std::vector<double> *castorKx,
    const std::vector<double> *castorKy,
    const std::vector<double> *castorKz,
    bool requirePositiveNumerator, bool refreshStaticInputs,
    bool *imageAllZero);
bool downloadCompactResidentImageCUDA(double *image, size_t count);
bool releaseCompactExecutionCUDA();
bool spmvFloatCUDA(const io::CSRView& matrix, const float* x, float* y);
bool spmvTFloatCUDA(const io::CSRView& matrix, const float* x, float* y);
/// spmv/spmvT with compact CTAC factors fused into the CUDA nnz loop.
/// attFactors is the full-angle table (fullNumAngles * nx*ny*nz uint16,
/// scale 1/65535) in the unrotated image frame; it is uploaded once and cached
/// by host pointer. The lookup is algebraically equivalent to the baked
/// reference values.
bool compactAttenSpmvFloatCUDA(const io::CSRView& matrix,
                               const uint16_t* attFactors,
                               size_t attCount,
                               const CompactAttenBakeParams& params,
                               const float* x, float* y,
                               int cacheSlot = 0);
bool compactAttenSpmvTFloatCUDA(const io::CSRView& matrix,
                                const uint16_t* attFactors,
                                size_t attCount,
                                const CompactAttenBakeParams& params,
                                const float* x, float* y,
                                int cacheSlot = 0);
bool compactDeviceBackwardBeginCUDA(size_t voxelCount,
                                    int accumulatorIndex = 0);
bool compactDeviceBackwardSpmvTAccumulateFloatCUDA(
    const io::CSRView& matrix,
    const float* x,
    const CompactDeviceTransformParams& params,
    CompactDeviceBackwardTiming* timing = nullptr,
    int accumulatorIndex = 0);
bool compactDeviceBackwardDownloadCUDA(double* y, size_t voxelCount,
                                       int accumulatorIndex = 0);
bool compactDeviceRatioSpmvTAccumulateFloatCUDA(
    const io::CSRView& matrix,
    const float* x,
    const double* fullProjection,
    size_t fullProjectionCount,
    const CompactDeviceRatioParams& ratioParams,
    const CompactDeviceTransformParams& transformParams,
    CompactDeviceRatioTiming* ratioTiming = nullptr,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int accumulatorIndex = 0,
    int cacheSlot = 0);
/// Device-ratio + device-backward accumulator path with on-the-fly CTAC
/// attenuation. attFactors is the full-angle table (fullNumAngles * nx*ny*nz
/// uint16) cached by host pointer per cacheSlot.
bool compactDeviceRatioAttenSpmvTAccumulateFloatCUDA(
    const io::CSRView& matrix,
    const float* x,
    const uint16_t* attFactors,
    size_t attCount,
    const cuda::CompactAttenBakeParams& attenParams,
    const double* fullProjection,
    size_t fullProjectionCount,
    const CompactDeviceRatioParams& ratioParams,
    const CompactDeviceTransformParams& transformParams,
    CompactDeviceRatioTiming* ratioTiming = nullptr,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int accumulatorIndex = 0,
    int cacheSlot = 0);
bool compactDeviceRatioSpmvTFloatCUDA(const io::CSRView& matrix,
                                      const float* x,
                                      const double* fullProjection,
                                      size_t fullProjectionCount,
                                      const CompactDeviceRatioParams& params,
                                      float* y,
                                      CompactDeviceRatioTiming* timing = nullptr,
                                      int cacheSlot = 0);
/// Batched compact device-ratio + device-backward accumulator path. The input
/// refreshed by refreshCompactBranchBatchInputCUDA is ordered bed-major, with
/// branchCount columns per bed; each bed keeps its own projection and backward
/// accumulator.
bool compactDeviceRatioBedBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRView& matrix,
    const double* const* fullProjections,
    size_t fullProjectionCount,
    const CompactDeviceRatioParams& commonRatioParams,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int branchCount,
    int bedCount,
    CompactDeviceRatioTiming* ratioTiming = nullptr,
    CompactDeviceBackwardTiming* backwardTiming = nullptr);
/// Branch-batch compact sensitivity accumulator path.  Computes A^T * 1 for all
/// symmetry branches in one SpMM call and scatters the results into the device
/// backward accumulator.  symIds/zMirrors describe the branch symmetry for each
/// column.  Only used when attenuation is not enabled; otherwise falls back to
/// the per-branch attenuated path.
bool compactSensitivityBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRView& matrix,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int batchSize,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int accumulatorIndex = 0,
    int cacheSlot = 0);
}

} // namespace spect
