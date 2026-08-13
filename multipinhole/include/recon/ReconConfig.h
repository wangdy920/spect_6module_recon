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
  const std::vector<uint8_t>* mask = nullptr;
  const std::vector<double>* w_img = nullptr;
  const std::vector<double>* w_proj = nullptr;
  const std::vector<double>* b_proj = nullptr;
  PSFConf psf;
  bool do_post_smooth_every_iter = false;

  // Optional projection-domain PSF geometry (used only if enabled and valid)
  int det_nz = 0;
  int det_nx = 0;
  int num_angles = 0;

  // Multi-bed configuration
  int bedCount = 1;
  std::vector<int> bedOverlaps; // Overlap pixels between beds (Stride = BaseNz - Overlap)
  std::vector<long long> bedOffsetVoxels; // Pre-calculated start index in image vector for each bed

  // Per-Bed Attenuation Factors for On-the-fly CTAC
  // Outer: Bed Index. Inner: [Angle][Voxel_in_Base_Grid] -> flattened to Angle*Nvox.
  // Stored as uint16_t (0-65535) mapped to 0.0-1.0.
  std::vector<std::vector<uint16_t>> bedAttenFactors;

};

/// Check if thread-local accumulation should be used based on memory constraints
inline int effective_threads(int threads) {
  if (threads <= 0)
    threads = 1;
#ifdef SPECT_DETERMINISTIC
  (void)threads;
  return 1;
#else
  return threads;
#endif
}

inline bool should_use_threadlocal_accum(size_t nvox, int threads) {
  threads = effective_threads(threads);
  if (threads <= 1 || nvox == 0)
    return false;

  // Cap temporary memory to avoid regressions on large volumes / many-core CPUs.
  // Fallback path uses atomics (slower but memory-safe).
  constexpr size_t kMaxThreadLocalBytes = 512ull * 1024ull * 1024ull; // 512 MiB
  constexpr size_t kMaxElems = kMaxThreadLocalBytes / sizeof(double);
  const size_t t = static_cast<size_t>(threads);
  return nvox <= (kMaxElems / t);
}

/// Check if PSF blurring is needed
inline bool needs_psf_blur(const ReconOpts& opts, const PSFKernels* psfKernels) {
  // Operator/Sieve: apply space-invariant PSF during forward/back projection
  if (opts.psf.mode != PSFMode::None)
    return (psfKernels != nullptr);
  return false;
}

} // namespace spect
