/**
 * @file SparseOpsCUDA.h
 * @brief CUDA sparse-operation declarations used by the compact workflow
 */

#pragma once

#include "io/CSRTypes.h"
#include "io/CompactChunkCSR.h"

#include <cstddef>
#include <cstdint>
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

/// Device BSREM-II relative update on the resident image (subset = 1), same
/// contract as the host BsremUpdateParams / applyBsremUpdate:
///   lambda = betaScaled * kappa
///   u = alpha * (q - S - lambda * r) / (S + lambda * max(c, 0) + denomEps)
///   x <- max(x, 1e-12) * (1 + clip(u))
/// The regularizer gradient r is a device array produced by the volume ops
/// (computeRegGradientDeviceCUDA) from the resident image; the curvature
/// surrogate c = |r| / max(x, curvEps) is kept on the device and refreshed only
/// when refreshCurvature is set; kappa is a host array uploaded once.
struct CompactBsremUpdateParams {
    double alpha = 1.0;
    double betaScaled = 0.0;
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
/// Resident BSREM-II iteration tail: accumulates q like the MLEM variant and
/// applies the relative update with the given penalty terms.
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
    bool *imageAllZero);
/// Device pointer of the resident image (count doubles) for the volume ops;
/// nullptr when CUDA is unavailable.
const double *compactResidentImageDeviceCUDA(size_t count);
/// Device scratch (count doubles) that receives the regularizer gradient.
double *compactResidentGradientDeviceCUDA(size_t count);
bool prefetchCompactFloatMatrixCUDA(const io::CSRView &next,
                                    const io::CSRView &current);
// csrt experiment (SPECT_COMPACT_T_SPMM=csrt): pre-derive host CSR(A^T)
// halves for a chunk during matrix prepare. No-op in other modes.
bool prepareCompactTransposeHostCacheCUDA(const io::CSRView &matrix);
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
/// Branch-batch compact sensitivity accumulator path.  Computes A^T W 1 for all
/// symmetry branches in one SpMM call and scatters the results into the device
/// backward accumulator.  symIds/zMirrors describe the branch symmetry for each
/// column.  rowWeights (one double per full projection row) may be nullptr for
/// W = 1 (all ones, the production default); with weights, commonRatioParams
/// supplies the row -> full projection mapping.  Only used when attenuation is
/// not enabled; otherwise falls back to the per-branch attenuated path.
bool compactSensitivityBranchBatchSpmvTAccumulateFloatCUDA(
    const io::CSRView& matrix,
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
bool csrqWarmupBeginCUDA(int matrixIndex, int cols,
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
/// Forward + ratio + backward of one window for beds [0, bedCount) in one
/// chunk-outer pass. Bed b's mean is weight * H_b x_b + background_b
/// (fullBackgrounds may be null or hold null entries); the backward adds
/// weight * H_b^T(p_b / mean_b) to accumulator accumulatorBase + b and caches
/// the projection in slot cacheSlotBase + b. commonRatioParams.compactRowBase
/// is replaced per chunk. H_b = A without CTAC; with bedAttFactors (one
/// full-angle uint16 table of attCount entries per bed, entries may be null)
/// H_b = A att_b, the factor being applied in the image domain per compact
/// angle (pack / transform-add), never inside the nnz loop.
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
                      size_t attCount = 0);
/// weight * H^T W 1 over rows with a valid full projection counterpart.
/// rowWeights (fullProjectionCount doubles: detector masks, censoring,
/// efficiency) may be nullptr for W = 1; attFactors (one bed's full-angle
/// table) may be nullptr for H = A.
bool csrqRunSensitivityCUDA(int matrixIndex, int branchCount, double weight,
                            const CompactDeviceRatioParams& commonRatioParams,
                            const CompactDeviceTransformParams& commonTransformParams,
                            const int* symIds, const int* zMirrors,
                            int accumulatorIndex, size_t fullProjectionCount,
                            const double* rowWeights = nullptr,
                            CompactDeviceBackwardTiming* backwardTiming = nullptr,
                            const uint16_t* attFactors = nullptr,
                            size_t attCount = 0);
/// Forward projection of one window over the refreshed single-bed branch
/// batch; hostYhatPacked receives [compactRow][branch] floats (rows * 8).
/// With attFactors the projection is H x = A att x and the angle / branch
/// parameters are required.
bool csrqRunForwardCUDA(int matrixIndex, int branchCount,
                        float* hostYhatPacked, size_t count,
                        const CompactDeviceRatioParams* attenRatioParams = nullptr,
                        const CompactDeviceTransformParams* attenTransformParams = nullptr,
                        const int* symIds = nullptr, const int* zMirrors = nullptr,
                        const uint16_t* attFactors = nullptr, size_t attCount = 0);
}

} // namespace spect
