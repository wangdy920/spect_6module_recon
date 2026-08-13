#include "recon/ProjectionOps.h"

#include "filters/cuda/VolumeOpsCUDA.h"
#include "filters/Convolution.h"
#include "recon/BedOffsets.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace spect {

namespace {

bool preferCudaVolumeOps(const ReconOpts &) {
  return cuda::isCudaVolumeOpsAvailable();
}

struct VolumeStats {
  size_t count = 0;
  size_t finite = 0;
  double maxAbs = 0.0;
  double l1 = 0.0;
};

VolumeStats computeVolumeStats(const std::vector<double> &volume) {
  VolumeStats stats;
  stats.count = volume.size();
  for (const double value : volume) {
    if (!std::isfinite(value))
      continue;
    ++stats.finite;
    const double absValue = std::abs(value);
    stats.maxAbs = std::max(stats.maxAbs, absValue);
    stats.l1 += absValue;
  }
  return stats;
}

bool hasFiniteNonZeroSignal(const VolumeStats &stats) {
  return stats.finite == stats.count && stats.maxAbs > 0.0;
}

bool isFiniteAllZero(const VolumeStats &stats) {
  return stats.count > 0 && stats.finite == stats.count && stats.maxAbs == 0.0;
}

bool inspectCudaPsfOutput(const VolumeStats &input, const VolumeStats &output,
                          std::string &reason) {
  if (output.count == 0 || output.finite != output.count) {
    reason = "output is empty or non-finite";
    return false;
  }
  if (!hasFiniteNonZeroSignal(input)) {
    if (output.maxAbs != 0.0) {
      reason = "zero input produced nonzero output";
      return false;
    }
    return true;
  }
  if (isFiniteAllZero(output)) {
    reason = "finite all-zero output for nonzero input";
    return false;
  }
  const double maxRatio = output.maxAbs / input.maxAbs;
  const double energyRatio = output.l1 / input.l1;
  if (!(maxRatio >= 1e-8 && maxRatio <= 1e8)) {
    reason = "max ratio out of guard range (input=" +
             std::to_string(input.maxAbs) + ", output=" +
             std::to_string(output.maxAbs) + ", ratio=" +
             std::to_string(maxRatio) + ")";
    return false;
  }
  if (!(energyRatio >= 1e-6 && energyRatio <= 1e6)) {
    reason = "energy ratio out of guard range (input=" +
             std::to_string(input.l1) + ", output=" +
             std::to_string(output.l1) + ", ratio=" +
             std::to_string(energyRatio) + ")";
    return false;
  }
  return true;
}

void applyPsfInplace(std::vector<double> &volume, int Nx, int Ny, int Nz,
                     const ReconOpts &opts, const PSFKernels *psfKernels,
                     int threads, bool adjoint) {
  const bool useCuda = preferCudaVolumeOps(opts);
  if (opts.psf.mode != PSFMode::None && psfKernels) {
    if (opts.psf.kernel == filters::PSFKernel::Castor) {
      // CASToR 严格对齐：反向复用正向卷积器（C=D_z^-1*K 再施一次）。
      // CUDA path uses zero padding plus z-boundary normalization.
      (void)adjoint;
      if (useCuda) {
        const VolumeStats inputStats = computeVolumeStats(volume);
        bool applied = false;
        try {
          applied = cuda::applyCastorConv3DInplaceCUDA(
              volume, Nx, Ny, Nz, psfKernels->kx, psfKernels->ky,
              psfKernels->kz);
        } catch (const std::exception &ex) {
          throw std::runtime_error(std::string("CUDA CASToR PSF failed: ") +
                                   ex.what());
        }
        if (!applied)
          throw std::runtime_error(
              "CUDA CASToR PSF operation reported unavailable");
        const VolumeStats outputStats = computeVolumeStats(volume);
        std::string reason;
        if (!inspectCudaPsfOutput(inputStats, outputStats, reason))
          throw std::runtime_error("CUDA CASToR PSF anomaly: " + reason);
        return;
      }
      filters::conv3d_castor_forward_inplace(volume, Nx, Ny, Nz,
                                             psfKernels->kx, psfKernels->ky,
                                             psfKernels->kz, threads);
      return;
    }
    if (useCuda) {
      const VolumeStats inputStats = computeVolumeStats(volume);
      bool applied = false;
      try {
        applied = adjoint
                      ? cuda::applySeparableConv3DAdjointInplaceCUDA(
                            volume, Nx, Ny, Nz, psfKernels->kx,
                            psfKernels->ky, psfKernels->kz)
                      : cuda::applySeparableConv3DInplaceCUDA(
                            volume, Nx, Ny, Nz, psfKernels->kx,
                            psfKernels->ky, psfKernels->kz);
      } catch (const std::exception &ex) {
        throw std::runtime_error(std::string("CUDA PSF operation failed: ") +
                                 ex.what());
      }
      if (!applied)
        throw std::runtime_error("CUDA PSF operation reported unavailable");
      const VolumeStats outputStats = computeVolumeStats(volume);
      std::string reason;
      if (!inspectCudaPsfOutput(inputStats, outputStats, reason))
        throw std::runtime_error("CUDA PSF anomaly: " + reason);
      return;
    }
    if (adjoint) {
      filters::conv3d_separable_adjoint_inplace(
          volume, Nx, Ny, Nz, psfKernels->kx, psfKernels->ky,
          psfKernels->kz, threads);
    } else {
      filters::conv3d_separable_inplace(volume, Nx, Ny, Nz, psfKernels->kx,
                                        psfKernels->ky, psfKernels->kz,
                                        threads);
    }
    return;
  }
}

} // namespace

void apply_psf_forward(const std::vector<double> &f, std::vector<double> &f_blur,
                       int Nx, int Ny, int Nz, const ReconOpts &opts,
                       const PSFKernels *psfKernels, int threads) {
  f_blur = f;
  applyPsfInplace(f_blur, Nx, Ny, Nz, opts, psfKernels, threads, false);
}

void apply_psf_backward(std::vector<double> &numer, int Nx, int Ny, int Nz,
                        const ReconOpts &opts, const PSFKernels *psfKernels,
                        int threads) {
  applyPsfInplace(numer, Nx, Ny, Nz, opts, psfKernels, threads, true);
}

std::vector<double> build_sieve_save_volume(
    const std::vector<double> &image, int Nx, int Ny, int baseNz, int bedCount,
    const std::vector<int> &bedZStarts, bool bedVirtualZFlip,
    const ReconOpts &opts, const PSFKernels *psfKernels, int threads) {
  if (bedCount <= 1) {
    // Single bed: one convolution of a copy of the global state.
    std::vector<double> saveVolume;
    apply_psf_forward(image, saveVolume, Nx, Ny, baseNz, opts, psfKernels,
                      threads);
    return saveVolume;
  }
  std::vector<double> saveVolume(image.size(), 0.0);
  for (int bed = 0; bed < bedCount; ++bed) {
    std::vector<double> localImage =
        spect::app::extractBedVolumeFromGlobal(image, Nx, Ny, baseNz,
                                               bedZStarts[static_cast<size_t>(bed)],
                                               bedVirtualZFlip);
    std::vector<double> localBlurred;
    apply_psf_forward(localImage, localBlurred, Nx, Ny, baseNz, opts,
                      psfKernels, threads);
    spect::app::accumulateBedVolumeToGlobal(
        saveVolume, localBlurred, Nx, Ny, baseNz,
        bedZStarts[static_cast<size_t>(bed)], bedVirtualZFlip);
  }
  return saveVolume;
}

} // namespace spect
