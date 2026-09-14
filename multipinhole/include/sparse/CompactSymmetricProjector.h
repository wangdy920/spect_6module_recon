#pragma once

#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "filters/PSFConfig.h"
#include "sparse/CompactSubsets.h"
#include "sparse/SparseOpsCUDA.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spect::cuda {

struct CompactSymmetricStageTiming {
  double transformForwardSeconds = 0.0;
  double forwardSeconds = 0.0;
  double ratioSeconds = 0.0;
  double backwardSeconds = 0.0;
  double transformBackwardSeconds = 0.0;
  double sensitivitySeconds = 0.0;
  double numeratorSeconds = 0.0;
  size_t sensitivityCalls = 0;
  size_t numeratorCalls = 0;
};

// One measurement channel of the Poisson model: an energy window with its own
// system matrix directory and the scalar w_k that turns the matrix into the
// window mean (w_k * A_k x + b_k), i.e. emission intensity x window efficiency
// when the matrix is per emitted photon. A single-window run is the
// one-channel case with weight 1. Channels are measurement groupings, never
// subsets: every channel and chunk accumulates into the same statistics and
// the image is updated once per iteration. Only CSRQ directories may be
// combined; CSRB stays single-channel.
struct CompactMatrixChannel {
  const spect::io::CompactChunkCSRDirectory *chunks = nullptr;
  double weight = 1.0;
  /// CTAC energy scaling: the per-bed factor table att (computed at the mu
  /// map's energy) enters this window as att^attenPower, with attenPower =
  /// mu_water(E_window) / mu_water(E_map) (1 = table used as is). Exact
  /// under the water-scaled mu model shared by auto-mu and the HU->mu
  /// pipeline, since att = exp(-integral mu).
  double attenPower = 1.0;
  const double* sensitivityRowWeights = nullptr; ///< Optional channel-specific W, fullRows entries; borrowed during calls.
};

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)

// --- Poisson data-term contract ---------------------------------------------
// Every optimizer (MLEM, BSREM-II) consumes the same two sufficient statistics
// of the Poisson log-likelihood over all channels k:
//   q = sum_k H_k^T ( p_k / (H_k x + b_k) )   backprojected ratio (per bed)
//   S = sum_k H_k^T 1                          sensitivity (once, cached)
// with H_k = w_k A_k. The data gradient is q - S. Nothing below knows which
// optimizer or penalty will use the result.

// Ordered subsets (CSRB, or CSRQ without CTAC): both statistics may be
// restricted to one subset of compact angles (CompactSubsetSpec, see
// CompactSubsets.h). The matrix is then streamed as (chunk, subset) slices —
// only the subset's rows cross PCIe — and S_m / q_m are the sums over those
// rows, so sum_m S_m = S. nullptr or count 1 is the full-data pass.

// S: joint sensitivity over all channels, S = sum_k w_k A_k^T W 1.
// rowWeights (optional, meta.fullRows() doubles indexed like the measured
// projection) carries the per-row sensitivity weights: detector masks and
// detZ censoring as 0/1, normalization efficiency as 1/c. nullptr means W = 1.
// Measured bins are never scaled by W; censoring zeroes them upstream.
bool computePoissonSensitivityCUDA(
    const std::vector<CompactMatrixChannel> &channels,
    const spect::io::CompactSymmetryMetadata &meta, int threads, double *denom,
    CompactSymmetricStageTiming *timing = nullptr,
    const uint16_t *attFactors = nullptr,
    const double *rowWeights = nullptr,
    const CompactSubsetSpec *subset = nullptr);

// q: joint backprojected ratio per bed.
// fullProjections / fullBackgrounds are indexed [channel * bedCount + bed]
// (fullBackgrounds may be nullptr, or hold nullptr entries).
// nextSubset (optional): the subset of the following call, whose first slice
// is prefetched behind this pass's last one. The caller flips
// reverseChunkOrder between consecutive calls, so that first slice is the
// last chunk of this call's order.
bool accumulatePoissonBackprojectedRatioCUDA(
    const std::vector<CompactMatrixChannel> &channels,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *const *fullProjections,
    const double *const *fullBackgrounds,
    const double *const *smoothedImages, int bedCount, int threads,
    double *const *numers, bool reverseChunkOrder = false,
    CompactSymmetricStageTiming *timing = nullptr,
    const uint16_t *const *attFactors = nullptr,
    bool refreshProjectionInputs = true,
    bool branchBatchInputPrepared = false,
    bool downloadBackwardOutputs = true,
    const CompactSubsetSpec *subset = nullptr,
    const CompactSubsetSpec *nextSubset = nullptr);

// Assigns device matrix slots and uploads every channel's CSRQ matrix (or
// warms the CSRB resident cache). Must run before the per-channel calls.
bool prepareChannelExecutionCUDA(
    const std::vector<CompactMatrixChannel> &channels);

// CSRQ residency policy. By default a matrix stays resident when it fits next
// to kCsrqResidentDeviceReserveBytes and is streamed from pinned host memory
// otherwise; forcing streaming keeps the GPU memory free for other work (or
// exercises the streamed path deliberately). Applies to matrices warmed
// after the call.
void setCsrqForceStreamingCUDA(bool force);
bool csrqForceStreamingCUDA();
// Test hook: after `skip` automatic (not forced-streamed) CSRQ warmups, the
// next `count` of them report "does not fit even streamed", exercising the
// demote-and-retry path of the residency planner without depending on the
// device's free memory. (0, 0) disables the hook.
void setCsrqWarmupFitFailuresForTestsCUDA(int skip, int count);


// --- Single-directory forms -------------------------------------------------
// Thin wrappers over the one-channel case, kept so the CSRB production callers
// and tests are untouched.
//
// attFactors (optional, CTAC): full-angle attenuation table in the unrotated
// image frame, laid out as att[fullAngle * totalVoxels + voxel] (uint16,
// scale 1/65535), fullNumAngles * totalVoxels entries. nullptr disables CTAC.
bool computeCompactChunkSensitivityCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta, int threads, double *denom,
    CompactSymmetricStageTiming *timing = nullptr,
    const uint16_t *attFactors = nullptr);

bool prepareCompactChunkExecutionCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks);

bool prepareCompactBranchBatchInputFromGlobalCUDA(
    const spect::io::CompactSymmetryMetadata &meta,
    const double *globalImage, size_t globalImageCount, int globalNz,
    const int *bedZStarts, int bedCount, bool flipLocalZ,
    const spect::filters::PSFKernels *castorPsf = nullptr,
    bool uploadGlobalImage = true);

// subsetIndex >= 0 divides by the resident subset sensitivity of that index
// (setCompactSubsetSensitivitiesCUDA); -1 is the full-data step over denom.
bool finishCompactResidentMlemIterationCUDA(
    const spect::io::CompactSymmetryMetadata &meta,
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ, const double *denom, const uint8_t *mask,
    const spect::filters::PSFKernels *castorPsf,
    bool requirePositiveNumerator, bool refreshStaticInputs,
    bool *imageAllZero, int subsetIndex = -1);

// BSREM-II counterpart: same data-term accumulation, then the relative update
// with params (penalty gradient already on the device, see
// compactResidentImageDeviceCUDA / compactResidentGradientDeviceCUDA).
bool finishCompactResidentBsremIterationCUDA(
    const spect::io::CompactSymmetryMetadata &meta,
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ, const double *denom, const uint8_t *mask,
    const spect::filters::PSFKernels *castorPsf,
    const CompactBsremUpdateParams &params, bool requirePositiveNumerator,
    bool refreshStaticInputs, bool *imageAllZero, int subsetIndex = -1);

bool runCompactChunkMlemNumerCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *fullProjection, const double *fullBackground,
    const double *smoothedImage, int threads, double *numer,
    bool reverseChunkOrder = false,
    CompactSymmetricStageTiming *timing = nullptr,
    const uint16_t *attFactors = nullptr,
    bool refreshProjectionInput = true);

// attenPower: CTAC energy exponent of this window (see
// CompactMatrixChannel::attenPower); values other than 1 need the CSRQ path.
bool computeCompactChunkForwardProjectionCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *smoothedImage, int threads, double *fullExpected,
    const uint16_t *attFactors = nullptr, double attenPower = 1.0);

// attFactors: optional per-bed pointer array (bedCount entries); each entry
// may be nullptr for "no attenuation for this bed".
bool runCompactChunkMlemNumerMultiBedCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *const *fullProjections,
    const double *const *fullBackgrounds,
    const double *const *smoothedImages, int bedCount, int threads,
    double *const *numers, bool reverseChunkOrder = false,
    CompactSymmetricStageTiming *timing = nullptr,
    const uint16_t *const *attFactors = nullptr,
    bool refreshProjectionInputs = true,
    bool branchBatchInputPrepared = false,
    bool downloadBackwardOutputs = true);

#endif

} // namespace spect::cuda
