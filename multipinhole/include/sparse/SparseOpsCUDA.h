/**
 * @file SparseOpsCUDA.h
 * @brief CUDA sparse-operation declarations used by the compact workflow
 */

#pragma once

#include "io/CSRTypes.h"
#include "io/CompactChunkCSR.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
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
    double windowWeight = 1.0; // A_window = windowWeight * A; background is unscaled.
    /// Device table of the absolute compact angle of every compactRaysPerAngle
    /// row block (rows / compactRaysPerAngle entries). Set by the state for an
    /// ordered-subset slice, whose blocks are not consecutive angles; nullptr
    /// keeps the contiguous mapping angle = (compactRowBase + row) / rpa.
    const int* blockAngles = nullptr;
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

/// Device BSREM-II relative update on the resident image, same contract as
/// the host BsremUpdateParams / applyBsremUpdate (Ahn & Fessler 2003 scaling
/// with M subsets; M = 1 and S_m = S is the plain full-data step):
///   lambda = betaScaled * kappa
///   u = alpha * (M (q_m - S_m) - lambda * r) / (S + lambda * max(c, 0) + denomEps)
///   x <- max(x, 1e-12) * (1 + clip(u))
/// The regularizer gradient r is a device array produced by the volume ops
/// (computeRegGradientDeviceCUDA) from the resident image; the curvature
/// surrogate c = |r| / max(x, curvEps) is kept on the device and refreshed only
/// when refreshCurvature is set; kappa is a host array uploaded once.
struct CompactBsremUpdateParams {
    double alpha = 1.0;
    double betaScaled = 0.0;
    /// M: number of subsets of the current stage (scales the subset data
    /// gradient q_m - S_m to full-data size).
    double subsetScale = 1.0;
    double denomEps = 1e-12;
    double sMin = 1e-12;
    bool clipRelativeStep = true;
    double clipMin = -0.9;
    double clipMax = 1.0;
    bool nonneg = true;
    bool zeroLowSensitivity = true;
    /// Device regularizer gradient (nullptr: no penalty term).
    const double* deviceGradient = nullptr;
    bool useCurvature = false;
    bool refreshCurvature = false;
    double curvEps = 1e-12;
    /// Host kappa map (nullptr: kappa = 1); uploaded once per host pointer.
    const double* kappa = nullptr;
};

} // namespace cuda


namespace cuda {
bool warmupResidentMatrixSetCUDA(const std::vector<io::CSRView>& matrices,
                                 uint64_t reserveBytes = 0);
void noteCompactCsrLocalityCUDA(bool sortedHint);
/// Uploads the measured projection (and, when given, its background /
/// scatter estimate) into cacheSlot. Slots [0, kMaxBeds) belong to the CSRB
/// path and CSRQ window 0; CSRQ window m uses slot m * kMaxBeds + bed.
bool refreshCompactProjectionCUDA(const double* projection, size_t count,
                                  int cacheSlot = 0,
                                  const double* background = nullptr);
bool refreshCompactBranchBatchInputCUDA(const float* input, size_t count);
bool prepareCompactBranchBatchInputFromGlobalCUDA(
    const double *globalImage, size_t globalImageCount, int globalNz,
    const int *bedZStarts, int bedCount, bool flipLocalZ,
    bool uploadGlobalImage,
    const CompactDeviceTransformParams &commonTransformParams,
    int branchCount, int zBranchCount,
    const std::vector<double> *castorKx = nullptr,
    const std::vector<double> *castorKy = nullptr,
    const std::vector<double> *castorKz = nullptr,
    const uint8_t *localSupportMask = nullptr);
/// Ordered subsets: keeps the subset sensitivities S_0..S_{count-1} (count
/// host arrays of globalImageCount doubles) resident so the update tails
/// below can select one by index without a per-sub-iteration upload. count
/// 0 releases them. Replaces the previous set.
bool setCompactSubsetSensitivitiesCUDA(const double *const *hostSensitivities,
                                       int count, size_t globalImageCount);
/// Resident MLEM (OSEM) iteration tail: x <- x q_m / S_m over the accumulated
/// per-bed backward outputs. subsetIndex < 0 divides by the global denom
/// (uploaded when refreshStaticInputs); otherwise by the resident subset
/// sensitivity of that index.
bool finishCompactResidentMlemIterationCUDA(
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ,
    const CompactDeviceTransformParams &commonTransformParams,
    const double *denom, const uint8_t *mask,
    const std::vector<double> *castorKx,
    const std::vector<double> *castorKy,
    const std::vector<double> *castorKz,
    bool requirePositiveNumerator, bool refreshStaticInputs,
    bool *imageAllZero, int subsetIndex = -1);
bool downloadCompactResidentImageCUDA(double *image, size_t count);
/// Resident BSREM-II iteration tail: accumulates q like the MLEM variant and
/// applies the relative update with the given penalty terms. subsetIndex < 0
/// uses S_m = S (global denom); otherwise the resident subset sensitivity.
bool finishCompactResidentBsremIterationCUDA(
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ,
    const CompactDeviceTransformParams &commonTransformParams,
    const double *denom, const uint8_t *mask,
    const std::vector<double> *castorKx,
    const std::vector<double> *castorKy,
    const std::vector<double> *castorKz,
    const CompactBsremUpdateParams &params,
    bool requirePositiveNumerator, bool refreshStaticInputs,
    bool *imageAllZero, int subsetIndex = -1);
/// Device pointer of the resident image (count doubles) for the volume ops;
/// nullptr when CUDA is unavailable.
const double *compactResidentImageDeviceCUDA(size_t count);
/// Device scratch (count doubles) that receives the regularizer gradient.
double *compactResidentGradientDeviceCUDA(size_t count);
/// Streams the next slice into the free device slot on the upload stream
/// while `current` (already resident) is being consumed.
bool prefetchCompactFloatMatrixCUDA(const io::CSRSubsetView &next,
                                    const io::CSRSubsetView &current);
bool releaseCompactExecutionCUDA();
bool spmvFloatCUDA(const io::CSRView& matrix, const float* x, float* y);
bool spmvTFloatCUDA(const io::CSRView& matrix, const float* x, float* y);
bool compactDeviceBackwardBeginCUDA(size_t voxelCount,
                                    int accumulatorIndex = 0);
bool compactDeviceBackwardDownloadCUDA(double* y, size_t voxelCount,
                                       int accumulatorIndex = 0);
// --- CSRB batch passes over a chunk slice ------------------------------------
// The matrix operand of the four passes below is an io::CSRSubsetView: the
// rows of the compact angles of one ordered subset that fall in one chunk, or
// the whole chunk (all its angles, tag 0). The slice is streamed into a device
// slot (segments back to back, re-based row pointers) and its per-block angle
// table drives the row -> full projection mapping, so the kernels never assume
// consecutive angles. commonRatioParams.compactRowBase is ignored for slices
// with more than one block; the state fills blockAngles.

/// CSRB CTAC numerator on the branch batch: H_b = A att_b^attPower, the
/// factor applied per compact angle in the image domain (see the CSRQ
/// counterpart csrqRunNumerCUDA). bedAttFactors holds one full-angle uint16
/// table (attCount entries) per bed.
bool compactDeviceRatioBedBranchBatchAttenSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView& matrix,
    const double* const* fullProjections,
    size_t fullProjectionCount,
    const CompactDeviceRatioParams& commonRatioParams,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int branchCount,
    int bedCount,
    const uint16_t* const* bedAttFactors,
    size_t attCount,
    double attPower = 1.0,
    CompactDeviceRatioTiming* ratioTiming = nullptr,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int cacheSlotBase = 0,
    const std::function<io::CSRSubsetView()>& loadNext = {});
/// CSRB CTAC sensitivity of one chunk: H^T W 1 for one bed's table added to
/// accumulatorIndex. fullProjectionCount is the full projection row count
/// (row -> projection validity); rowWeights (that many doubles) may be
/// nullptr for W = 1.
bool compactSensitivityBranchBatchAttenSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView& matrix,
    const CompactDeviceRatioParams& commonRatioParams,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int branchCount,
    const uint16_t* attFactors,
    size_t attCount,
    double attPower,
    const double* rowWeights,
    size_t fullProjectionCount,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int accumulatorIndex = 0);
/// CSRB CTAC forward projection of one chunk over the refreshed single-bed
/// branch batch; hostYhatBatch receives [branch][chunkRow] floats.
bool compactBranchBatchAttenForwardCUDA(
    const io::CSRView& matrix,
    const CompactDeviceRatioParams& commonRatioParams,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int branchCount,
    const uint16_t* attFactors,
    size_t attCount,
    double attPower,
    float* hostYhatBatch,
    size_t count);
/// Batched compact device-ratio + device-backward accumulator path. The input
/// refreshed by refreshCompactBranchBatchInputCUDA is ordered bed-major, with
/// branchCount columns per bed; each bed keeps its own projection and backward
/// accumulator. loadNext runs after GPU submission and before the final wait; it
/// must only load host data, never re-enter the CUDA sparse state.
bool compactDeviceRatioBedBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView& matrix,
    const double* const* fullProjections,
    size_t fullProjectionCount,
    const CompactDeviceRatioParams& commonRatioParams,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int branchCount,
    int bedCount,
    CompactDeviceRatioTiming* ratioTiming = nullptr,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int cacheSlotBase = 0,
    const std::function<io::CSRSubsetView()>& loadNext = {});
/// Branch-batch compact sensitivity accumulator path.  Computes A^T W 1 for all
/// symmetry branches in one SpMM call and scatters the results into the device
/// backward accumulator.  symIds/zMirrors describe the branch symmetry for each
/// column.  rowWeights (one double per full projection row) may be nullptr for
/// W = 1 (all ones, the production default); with weights, commonRatioParams
/// supplies the row -> full projection mapping. CTAC runs use the attenuated
/// variant above.
bool compactSensitivityBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRSubsetView& matrix,
    const CompactDeviceTransformParams& commonTransformParams,
    const int* symIds,
    const int* zMirrors,
    int batchSize,
    CompactDeviceBackwardTiming* backwardTiming = nullptr,
    int accumulatorIndex = 0,
    int cacheSlot = 0,
    const CompactDeviceRatioParams* commonRatioParams = nullptr,
    const double* rowWeights = nullptr,
    size_t rowWeightCount = 0);

// --- CSRQ matrices (CompactChunkFormat::CsrqRunU16) ------------------------
// Each energy window's matrix is loaded once into device slot matrixIndex
// through the incremental warmup below (chunk by chunk, so the host holds one
// chunk at a time). The state keeps it resident when it fits next to
// kCsrqResidentDeviceReserveBytes, otherwise streams it from pinned host
// memory through two device chunk slots; hard failure when even that does
// not fit. Every pass walks the chunks of one matrix chunk-outer with the
// custom K=8 kernels. Inputs come from the same branch-batch buffer as the
// CSRB path (refreshCompactBranchBatchInputCUDA /
// prepareCompactBranchBatchInputFromGlobalCUDA, layout [bed][branch][voxel]);
// outputs are added to the compact device backward accumulators
// (compactDeviceBackwardBeginCUDA / DownloadCUDA), so several windows simply
// accumulate into the same per-bed slot (joint multi-window MLEM).
struct CsrqChunkShape {
    int rows = 0;
    uint64_t nnz = 0;
    uint64_t runCount = 0;
    int compactRowBase = 0;

    uint64_t deviceBytes() const {
        return 2ull * sizeof(uint32_t) * (static_cast<uint64_t>(rows) + 1ull) +
               sizeof(float) * static_cast<uint64_t>(rows) +
               sizeof(uint32_t) * runCount + sizeof(uint16_t) * nnz;
    }
    // A reusable slot needs each array's maximum, not the largest whole chunk.
    void includeDimensions(const CsrqChunkShape& other) {
        rows = std::max(rows, other.rows);
        nnz = std::max(nnz, other.nnz);
        runCount = std::max(runCount, other.runCount);
    }
};
/// Thrown by csrqWarmupBeginCUDA when the matrix does not fit on the device
/// even streamed (two chunk slots + scratch + reserve exceed the free memory).
/// The driver may then demote an earlier resident matrix to streamed and
/// retry instead of aborting.
struct CsrqDeviceMemoryError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};
/// Announces the matrix; *streamed reports whether it will be streamed
/// rather than resident (forceStreamed skips the fit test). Drops whatever
/// slot matrixIndex held.
bool csrqWarmupBeginCUDA(int matrixIndex, int cols, int branchCount,
                         const std::vector<CsrqChunkShape>& shapes,
                         uint64_t reserveBytes, bool forceStreamed = false,
                         bool* streamed = nullptr);
/// Hands over one chunk (uploaded or pinned before returning).
bool csrqWarmupChunkCUDA(int matrixIndex, size_t chunkIndex,
                         const io::CompactRunU16View& view);
/// Marks the matrix usable once every chunk was handed over.
bool csrqWarmupEndCUDA(int matrixIndex);
/// True while matrix matrixIndex is loaded (cleared by
/// releaseCompactExecutionCUDA).
bool csrqResidentCUDA(int matrixIndex);
/// Packed [nvox][K] CSRQ backward: all active branches scattered in one pass
/// with float4 atomics (native on sm_90+, scalar fallback on sm_86/89 where it
/// is ~5x slower than the branch-major default). Default from
/// SPECT_CSRQ_BACKWARD_PACKED=1; exposed for the sm_120 trial and tests.
bool setCsrqPackedBackwardCUDA(bool enabled);
bool csrqPackedBackwardCUDA();
/// Forward + ratio + backward of one window for beds [0, bedCount) in one
/// chunk-outer pass. Bed b's mean is weight * H_b x_b + background_b
/// (fullBackgrounds may be null or hold null entries); the backward adds
/// weight * H_b^T(p_b / mean_b) to accumulator accumulatorBase + b and caches
/// the projection in slot cacheSlotBase + b. commonRatioParams.compactRowBase
/// is replaced per chunk. H_b = A without CTAC; with bedAttFactors (one
/// full-angle uint16 table of attCount entries per bed, entries may be null)
/// H_b = A att_b^attPower, the factor being applied in the image domain per
/// compact angle (pack / transform-add), never inside the nnz loop. attPower
/// = mu_w(E_window)/mu_w(E_map) rescales the table to the window's energy
/// (1 = as stored).
bool csrqRunNumerCUDA(int matrixIndex, int bedCount, int branchCount,
                      const CompactDeviceRatioParams& commonRatioParams,
                      const CompactDeviceTransformParams& commonTransformParams,
                      const int* symIds, const int* zMirrors,
                      const double* const* fullProjections,
                      const double* const* fullBackgrounds,
                      size_t fullProjectionCount, double weight,
                      int accumulatorBase, int cacheSlotBase,
                      CompactDeviceRatioTiming* ratioTiming = nullptr,
                      CompactDeviceBackwardTiming* backwardTiming = nullptr,
                      const uint16_t* const* bedAttFactors = nullptr,
                      size_t attCount = 0, double attPower = 1.0,
                            int subsetCount = 1, int subsetIndex = 0);
/// weight * H^T W 1 over rows with a valid full projection counterpart.
/// rowWeights (fullProjectionCount doubles: detector masks, censoring,
/// efficiency) may be nullptr for W = 1; attFactors (one bed's full-angle
/// table) may be nullptr for H = A, otherwise H = A att^attPower.
bool csrqRunSensitivityCUDA(int matrixIndex, int branchCount, double weight,
                            const CompactDeviceRatioParams& commonRatioParams,
                            const CompactDeviceTransformParams& commonTransformParams,
                            const int* symIds, const int* zMirrors,
                            int accumulatorIndex, size_t fullProjectionCount,
                            const double* rowWeights = nullptr,
                            CompactDeviceBackwardTiming* backwardTiming = nullptr,
                            const uint16_t* attFactors = nullptr,
                            size_t attCount = 0, double attPower = 1.0,
                            int subsetCount = 1, int subsetIndex = 0);
/// Forward projection of one window over the refreshed single-bed branch
/// batch; hostYhatPacked receives [compactRow][branch] floats (rows * K).
/// With attFactors the projection is H x = A att^attPower x and the angle /
/// branch parameters are required.
bool csrqRunForwardCUDA(int matrixIndex, int branchCount,
                        float* hostYhatPacked, size_t count,
                        const CompactDeviceRatioParams* attenRatioParams = nullptr,
                        const CompactDeviceTransformParams* attenTransformParams = nullptr,
                        const int* symIds = nullptr, const int* zMirrors = nullptr,
                        const uint16_t* attFactors = nullptr, size_t attCount = 0,
                        double attPower = 1.0);
}

} // namespace spect
