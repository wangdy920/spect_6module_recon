/**
 * @file SparseOpsBackend.h
 * @brief Sparse backend selection helpers (CPU/Reference/CUDA placeholder)
 */

#pragma once

#include "interfaces/ISparseOps.h"
#include "io/CSRTypes.h"
#include "sparse/SparseOpsCPU.h"
#include "sparse/SparseOpsRef.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spect {

enum class SparseBackendChoice {
    Auto,
    CPU,
    Reference,
    CUDA
};

inline const char* sparseBackendChoiceName(SparseBackendChoice choice) {
    switch (choice) {
    case SparseBackendChoice::Auto:
        return "auto";
    case SparseBackendChoice::CPU:
        return "cpu";
    case SparseBackendChoice::Reference:
        return "ref";
    case SparseBackendChoice::CUDA:
        return "cuda";
    default:
        return "auto";
    }
}

inline bool parseSparseBackendChoice(const std::string& raw,
                                     SparseBackendChoice& choice) {
    std::string value = raw;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value.empty() || value == "auto") {
        choice = SparseBackendChoice::Auto;
        return true;
    }
    if (value == "cpu") {
        choice = SparseBackendChoice::CPU;
        return true;
    }
    if (value == "ref" || value == "reference") {
        choice = SparseBackendChoice::Reference;
        return true;
    }
    if (value == "cuda" || value == "gpu") {
        choice = SparseBackendChoice::CUDA;
        return true;
    }
    return false;
}

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

inline const ISparseOps& selectSparseOps(SparseBackendChoice choice,
                                         SparseBackendChoice* resolved = nullptr) {
    const ISparseOps* selected = nullptr;
    SparseBackendChoice out = choice;

    switch (choice) {
    case SparseBackendChoice::Auto:
        selected = &getDefaultSparseOps();
        out = (selected->backend() == BackendType::Reference)
                  ? SparseBackendChoice::Reference
                  : SparseBackendChoice::CPU;
        break;
    case SparseBackendChoice::CPU:
        selected = &getSparseOpsCPU();
        break;
    case SparseBackendChoice::Reference:
        selected = &getSparseOpsRef();
        break;
    case SparseBackendChoice::CUDA:
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)
        selected = &getSparseOpsCUDA();
#else
        selected = &getDefaultSparseOps();
        out = (selected->backend() == BackendType::Reference)
                  ? SparseBackendChoice::Reference
                  : SparseBackendChoice::CPU;
#endif
        break;
    }

    if (resolved) {
        *resolved = out;
    }
    return *selected;
}

inline const ISparseOps& selectSparseOps(const std::string& backend,
                                         SparseBackendChoice* resolved = nullptr,
                                         bool* parsedOk = nullptr) {
    SparseBackendChoice parsed = SparseBackendChoice::Auto;
    const bool ok = parseSparseBackendChoice(backend, parsed);
    if (parsedOk) {
        *parsedOk = ok;
    }
    if (!ok) {
        parsed = SparseBackendChoice::Auto;
    }
    return selectSparseOps(parsed, resolved);
}

} // namespace spect
