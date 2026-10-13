// ReconConfig.h - Reconstruction configuration and options
#pragma once

#include "filters/PSFConfig.h"
#include "io/CSRTypes.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace spect {


// Import filter types into spect namespace for convenience
using filters::PSFConf;
using filters::PSFKernels;
using filters::PSFMode;

// Import CSR type from io namespace
using io::CSR;

/// Reconstruction options and parameters
struct ReconOpts {
  int threads = 16;
  bool nonneg = true;
  const std::vector<double>* w_proj = nullptr;
  const std::vector<double>* b_proj = nullptr;
  PSFConf psf;

  // Angle count for per-angle attenuation in projection metrics.
  int num_angles = 0;

  // Multi-bed configuration
  int bedCount = 1;
  std::vector<long long> bedOffsetVoxels; // Pre-calculated start index in image vector for each bed

  // Per-Bed Attenuation Factors for On-the-fly CTAC
  // Outer: Bed Index. Inner: [Angle][Voxel_in_Base_Grid] -> flattened to Angle*Nvox.
  // Stored as uint16_t (0-65535) mapped to 0.0-1.0.
  std::vector<std::vector<uint16_t>> bedAttenFactors;

};

/// Check if PSF blurring is needed
inline bool needs_psf_blur(const ReconOpts& opts, const PSFKernels* psfKernels) {
  // Operator/Sieve: apply space-invariant PSF during forward/back projection
  if (opts.psf.mode != PSFMode::None)
    return (psfKernels != nullptr);
  return false;
}

} // namespace spect
