#pragma once

#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"

#include <cstddef>
#include <cstdint>

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

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)


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

bool runCompactChunkMlemNumerCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *fullProjection, const double *fullBackground,
    const double *smoothedImage, int threads, double *numer,
    bool reverseChunkOrder = false,
    CompactSymmetricStageTiming *timing = nullptr,
    const uint16_t *attFactors = nullptr);

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
    const uint16_t *const *attFactors = nullptr);

#endif

} // namespace spect::cuda
