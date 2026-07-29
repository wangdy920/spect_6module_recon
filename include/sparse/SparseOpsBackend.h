/**
 * @file SparseOpsBackend.h
 * @brief Sparse backend selection helpers (CPU/Reference/CUDA placeholder)
 */

#pragma once

#include "interfaces/ISparseOps.h"
#include "io/CSRTypes.h"
#include "io/CompactChunkCSR.h"
#include "sparse/SparseOpsCPU.h"
#include "sparse/SparseOpsRef.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace spect {

inline bool isCudaSparseOpsAvailable() {
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)
    return true;
#else
    return false;
#endif
}

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

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)
const ISparseOps& getSparseOpsCUDA();

namespace cuda {
bool warmupResidentMatrixSetCUDA(const std::vector<io::CSRView>& matrices,
                                 uint64_t reserveBytes = 0);
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
/// Same as compactDeviceRatioSpmvTAccumulateFloatCUDA but launches on a
/// per-bed CUDA stream and skips the final synchronization, allowing the
/// caller to overlap multiple beds. The caller must later synchronize the
/// stream (e.g. via compactDeviceRatioSynchronizeBedCUDA).
bool compactDeviceRatioSpmvTAccumulateFloatAsyncCUDA(
    const io::CSRView& matrix,
    const float* x,
    const double* fullProjection,
    size_t fullProjectionCount,
    const CompactDeviceRatioParams& ratioParams,
    const CompactDeviceTransformParams& transformParams,
    int accumulatorIndex = 0,
    int cacheSlot = 0);
bool compactDeviceRatioSynchronizeBedCUDA(int cacheSlot);
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
/// Branch-batch compact device-ratio + device-backward accumulator path.
/// xBatch is cols x batchSize column-major (one rotated-image branch per
/// column). symIds/zMirrors describe the branch symmetry for each column.
bool compactDeviceRatioBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRView& matrix,
    const float* xBatch,
    const double* fullProjection,
    size_t fullProjectionCount,
    const CompactDeviceRatioParams& commonRatioParams,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int batchSize,
    CompactDeviceRatioTiming* ratioTiming = nullptr,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int accumulatorIndex = 0,
    int cacheSlot = 0);
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
bool compactDeviceRatioBranchBatchQuantizedSpmvTAccumulateFloatCUDA(
    const io::CSRQuantizedView& matrix,
    const float* xBatch,
    const double* fullProjection,
    size_t fullProjectionCount,
    const CompactDeviceRatioParams& commonRatioParams,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int batchSize,
    CompactDeviceRatioTiming* ratioTiming = nullptr,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int accumulatorIndex = 0,
    int cacheSlot = 0);
bool compactSensitivityBranchBatchQuantizedSpmvTAccumulateFloatCUDA(
    const io::CSRQuantizedView& matrix,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int batchSize,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int accumulatorIndex = 0,
    int cacheSlot = 0);
}
#endif

#if !(defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL))
inline const ISparseOps& getSparseOpsCUDA() {
    throw std::runtime_error(
        "SPECT_recon requires a CUDA-enabled sparse backend build.");
}

namespace cuda {
inline bool warmupResidentMatrixSetCUDA(const std::vector<io::CSRView>&,
                                        uint64_t = 0) {
    return false;
}
inline bool spmvFloatCUDA(const io::CSRView&, const float*, float*) {
    return false;
}
inline bool spmvTFloatCUDA(const io::CSRView&, const float*, float*) {
    return false;
}
inline bool compactAttenSpmvFloatCUDA(const io::CSRView&, const uint16_t*,
                                      size_t, const CompactAttenBakeParams&,
                                      const float*, float*, int = 0) {
    return false;
}
inline bool compactAttenSpmvTFloatCUDA(const io::CSRView&, const uint16_t*,
                                       size_t, const CompactAttenBakeParams&,
                                       const float*, float*, int = 0) {
    return false;
}
inline bool compactDeviceBackwardBeginCUDA(size_t, int = 0) {
    return false;
}
inline bool compactDeviceBackwardSpmvTAccumulateFloatCUDA(
    const io::CSRView&, const float*, const CompactDeviceTransformParams&,
    CompactDeviceBackwardTiming* = nullptr, int = 0) {
    return false;
}
inline bool compactDeviceBackwardDownloadCUDA(double*, size_t, int = 0) {
    return false;
}
inline bool compactDeviceRatioSpmvTAccumulateFloatCUDA(
    const io::CSRView&, const float*, const double*, size_t,
    const CompactDeviceRatioParams&, const CompactDeviceTransformParams&,
    CompactDeviceRatioTiming* = nullptr,
    CompactDeviceBackwardTiming* = nullptr, int = 0, int = 0) {
    return false;
}
inline bool compactDeviceRatioSpmvTAccumulateFloatAsyncCUDA(
    const io::CSRView&, const float*, const double*, size_t,
    const CompactDeviceRatioParams&, const CompactDeviceTransformParams&,
    int = 0, int = 0) {
    return false;
}
inline bool compactDeviceRatioSynchronizeBedCUDA(int) { return false; }
inline bool compactDeviceRatioAttenSpmvTAccumulateFloatCUDA(
    const io::CSRView&, const float*, const uint16_t*, size_t,
    const cuda::CompactAttenBakeParams&, const double*, size_t,
    const CompactDeviceRatioParams&, const CompactDeviceTransformParams&,
    CompactDeviceRatioTiming* = nullptr,
    CompactDeviceBackwardTiming* = nullptr, int = 0, int = 0) {
    return false;
}
inline bool compactDeviceRatioSpmvTFloatCUDA(
    const io::CSRView&, const float*, const double*, size_t,
    const CompactDeviceRatioParams&, float*,
    CompactDeviceRatioTiming* = nullptr, int = 0) {
    return false;
}
inline bool compactDeviceRatioBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRView&, const float*, const double*, size_t,
    const CompactDeviceRatioParams&, const CompactDeviceTransformParams&,
    const int*, const int*, int,
    CompactDeviceRatioTiming* = nullptr,
    CompactDeviceBackwardTiming* = nullptr, int = 0, int = 0) {
    return false;
}
inline bool compactSensitivityBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRView&, const CompactDeviceTransformParams&,
    const int*, const int*, int,
    CompactDeviceBackwardTiming* = nullptr, int = 0, int = 0) {
    return false;
}
inline bool compactDeviceRatioBranchBatchQuantizedSpmvTAccumulateFloatCUDA(
    const io::CSRQuantizedView&, const float*, const double*, size_t,
    const CompactDeviceRatioParams&, const CompactDeviceTransformParams&,
    const int*, const int*, int,
    CompactDeviceRatioTiming* = nullptr,
    CompactDeviceBackwardTiming* = nullptr, int = 0, int = 0) {
    return false;
}
inline bool compactSensitivityBranchBatchQuantizedSpmvTAccumulateFloatCUDA(
    const io::CSRQuantizedView&, const CompactDeviceTransformParams&,
    const int*, const int*, int,
    CompactDeviceBackwardTiming* = nullptr, int = 0, int = 0) {
    return false;
}
}
#endif

} // namespace spect
