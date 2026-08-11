#include "pipeline/ReconPreparation.h"

#include "core/Masks.h"
#include "filters/ButterworthFilter.h"
#include "filters/Convolution.h"
#include "filters/cuda/VolumeOpsCUDA.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spect::main {

std::array<double, 3> voxelSpacingFromArgs(const ReconArgs &args) {
  return {args.dx > 0.0 ? args.dx : static_cast<double>(args.Vx),
          args.dy > 0.0 ? args.dy : static_cast<double>(args.Vy),
          args.dz > 0.0 ? args.dz : static_cast<double>(args.Vz)};
}

const std::vector<uint8_t> *maybeMakeCylindricalMask(
    std::vector<uint8_t> &mask, int nx, int ny, int nz,
    const std::array<double, 3> &spacing, const ReconArgs &args) {
  if (!args.useCylindricalMask)
    return nullptr;

  const int bedCount = std::max(1, args.bedCount);
  if (nz % bedCount != 0)
    throw std::runtime_error(
        "Cylindrical ROI requires an integer local Z size per bed.");
  const int localNz = nz / bedCount;
  const std::vector<uint8_t> localMask = spect::core::makeCylMask_mm(
      nx, ny, localNz, spacing[0], spacing[1], spacing[2],
      args.cylinderDiameterMM, args.cylinderHeightMM, args.cylCenterX,
      args.cylCenterY, args.cylCenterZ);
  if (bedCount == 1) {
    mask = localMask;
    return &mask;
  }

  mask.resize(localMask.size() * static_cast<size_t>(bedCount));
  for (int bed = 0; bed < bedCount; ++bed) {
    std::copy(localMask.begin(), localMask.end(),
              mask.begin() + static_cast<std::ptrdiff_t>(bed) *
                                 static_cast<std::ptrdiff_t>(localMask.size()));
  }
  return &mask;
}

filters::PSFKernels preparePSFKernels(const filters::PSFConf &psfConf) {
  if (psfConf.mode == filters::PSFMode::None)
    return {};
  return filters::make_psf_kernels(psfConf);
}

filters::ButterworthConf prepareButterworthConf(const ReconArgs &args) {
  filters::ButterworthConf conf;
  // Only the in_recon stage feeds the per-iteration filter.
  if (!args.useButter || args.butterApplyStage != "in_recon")
    return conf;
  conf.enabled = true;
  conf.type = filters::BWType::Lowpass;
  conf.order = std::max(1, args.butterOrder);
  conf.cutoff_hi_frac = args.butterCutoff;
  conf.cutoff_lo_frac = 0.0;
  conf.apply_after_each_subset = false;
  conf.every_k_iters = 1;
  return conf;
}

SmoothConf prepareSmoothingConf(const ReconArgs &args) {
  SmoothConf conf;
  // Only the in_recon stage feeds the per-iteration smoother.
  if (!args.useSmooth || args.smoothApplyStage != "in_recon" ||
      args.smoothSigma <= 0.0f)
    return conf;

  int kernel = std::max(1, args.smoothKernel);
  if ((kernel % 2) == 0)
    ++kernel;
  const int radius = kernel / 2;
  const double sigma = std::max<double>(args.smoothSigma, 1e-6);
  conf.kx.resize(static_cast<size_t>(kernel));
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    const double value = std::exp(-0.5 * (static_cast<double>(i * i) / (sigma * sigma)));
    conf.kx[static_cast<size_t>(i + radius)] = value;
    sum += value;
  }
  for (double &v : conf.kx)
    v /= sum;
  conf.ky = conf.kx;
  conf.kz = conf.kx;
  conf.enabled = true;
  return conf;
}

ReconPreparedParams prepareReconPreparedParams(const ReconArgs &args,
                                               const filters::PSFConf &psfConf) {
  ReconPreparedParams prepared;
  prepared.psfKernels = preparePSFKernels(psfConf);
  prepared.butterConf = prepareButterworthConf(args);
  prepared.smoothConf = prepareSmoothingConf(args);
  return prepared;
}

PSFPreparationResult preparePSFForRuntime(ReconOpts &opts, const ReconArgs &args,
                                          const filters::Spacing3 &spacing,
                                          int, size_t,
                                          int) {
  PSFPreparationResult result;
  opts.psf = args.psf;
  opts.psf.vox_x = spacing.dx;
  opts.psf.vox_y = spacing.dy;
  opts.psf.vox_z = spacing.dz;

  if (!args.usePSF) {
    opts.psf.mode = filters::PSFMode::None;
    return result;
  }

  if (opts.psf.mode != filters::PSFMode::None) {
    if (!cuda::isCudaVolumeOpsAvailable())
      throw std::runtime_error(
          "PSF-enabled reconstruction requires CUDA volume operations; "
          "SPECT_CUDA_VOLUME_OPS=0 and CPU fallback are not supported.");
    result.psfKernels = preparePSFKernels(opts.psf);
  }

  return result;
}

} // namespace spect::main
