#pragma once

#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "filters/PSFConfig.h"

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

// One energy window of a joint multi-window (MB-JR) reconstruction: its own
// system matrix directory and the scalar w that turns the matrix into the
// window mean (w * A x + background), i.e. emission intensity x window
// efficiency when the matrix is per emitted photon. A single-window run is
// the one-channel case with weight 1. Only CSRQ directories may be combined;
// CSRB stays single-channel.
struct CompactMatrixChannel {
  const spect::io::CompactChunkCSRDirectory *chunks = nullptr;
  double weight = 1.0;
};

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)

// Joint sensitivity sum_k w_k A_k^T 1 over all channels.
bool computeCompactChunkSensitivityCUDA(
    const std::vector<CompactMatrixChannel> &channels,
    const spect::io::CompactSymmetryMetadata &meta, int threads, double *denom,
    CompactSymmetricStageTiming *timing = nullptr,
    const uint16_t *attFactors = nullptr);

// Joint numerator sum_k w_k A_k^T(p_k / (w_k A_k x + b_k)) per bed.
// fullProjections / fullBackgrounds are indexed [channel * bedCount + bed]
// (fullBackgrounds may be nullptr, or hold nullptr entries).
bool runCompactChunkMlemNumerMultiBedCUDA(
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
    bool downloadBackwardOutputs = true);

// Uploads every channel's CSRQ matrix (or warms the CSRB resident cache).
bool prepareCompactChunkExecutionCUDA(
    const std::vector<CompactMatrixChannel> &channels);

// CSRQ residency policy. By default a matrix stays resident when it fits next
// to kCsrqResidentDeviceReserveBytes and is streamed from pinned host memory
// otherwise; forcing streaming keeps the GPU memory free for other work (or
// exercises the streamed path deliberately). Applies to matrices warmed
// after the call.
void setCsrqForceStreamingCUDA(bool force);
bool csrqForceStreamingCUDA();


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

bool finishCompactResidentMlemIterationCUDA(
    const spect::io::CompactSymmetryMetadata &meta,
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ, const double *denom, const uint8_t *mask,
    const spect::filters::PSFKernels *castorPsf,
    bool requirePositiveNumerator, bool refreshStaticInputs,
    bool *imageAllZero);

bool runCompactChunkMlemNumerCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *fullProjection, const double *fullBackground,
    const double *smoothedImage, int threads, double *numer,
    bool reverseChunkOrder = false,
    CompactSymmetricStageTiming *timing = nullptr,
    const uint16_t *attFactors = nullptr,
    bool refreshProjectionInput = true);

bool computeCompactChunkForwardProjectionCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *smoothedImage, int threads, double *fullExpected,
    const uint16_t *attFactors = nullptr);

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
