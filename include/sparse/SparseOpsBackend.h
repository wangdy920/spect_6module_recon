/**
 * @file SparseOpsBackend.h
 * @brief Sparse backend selection helpers (CPU/Reference/CUDA placeholder)
 */

#pragma once

#include "interfaces/ISparseOps.h"
#include "io/CSRTypes.h"
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

/// Parameters for baking per-angle attenuation factors into compact chunk
/// matrix values on the device: baked[t] = val[t] * att[fullAngle(row(t)),
/// unrotatedVoxel(col(t))]. Combines the row->full-angle mapping of
/// CompactDeviceRatioParams with the voxel mapping of
/// CompactDeviceTransformParams.
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
/// spmv/spmvT with attenuation-baked values. attFactors is the full-angle
/// table (fullNumAngles * nx*ny*nz uint16, scale 1/65535) in the unrotated
/// image frame; it is uploaded once and cached by host pointer.
bool compactAttenSpmvFloatCUDA(const io::CSRView& matrix,
                               const uint16_t* attFactors,
                               size_t attCount,
                               const CompactAttenBakeParams& params,
                               const float* x, float* y);
bool compactAttenSpmvTFloatCUDA(const io::CSRView& matrix,
                                const uint16_t* attFactors,
                                size_t attCount,
                                const CompactAttenBakeParams& params,
                                const float* x, float* y);
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
    int accumulatorIndex = 0);
bool compactDeviceRatioSpmvTFloatCUDA(const io::CSRView& matrix,
                                      const float* x,
                                      const double* fullProjection,
                                      size_t fullProjectionCount,
                                      const CompactDeviceRatioParams& params,
                                      float* y,
                                      CompactDeviceRatioTiming* timing = nullptr);
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
                                      const float*, float*) {
    return false;
}
inline bool compactAttenSpmvTFloatCUDA(const io::CSRView&, const uint16_t*,
                                       size_t, const CompactAttenBakeParams&,
                                       const float*, float*) {
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
    CompactDeviceBackwardTiming* = nullptr, int = 0) {
    return false;
}
inline bool compactDeviceRatioSpmvTFloatCUDA(
    const io::CSRView&, const float*, const double*, size_t,
    const CompactDeviceRatioParams&, float*,
    CompactDeviceRatioTiming* = nullptr) {
    return false;
}
}
#endif

} // namespace spect
