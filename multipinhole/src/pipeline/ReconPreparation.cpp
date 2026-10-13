#include "pipeline/ReconPreparation.h"

#include "core/Masks.h"
#include "filters/ButterworthFilter.h"
#include "filters/Convolution.h"
#include "filters/cuda/VolumeOpsCUDA.h"
#include "recon/BedOffsets.h"

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
  const int stackedNz =
      nz + spect::app::computeTotalOverlap(bedCount, args.bedOverlaps);
  if (stackedNz % bedCount != 0)
    throw std::runtime_error(
        "Cylindrical ROI requires an integer local Z size per bed.");
  const int localNz = stackedNz / bedCount;
  const std::vector<uint8_t> localMask = spect::core::makeCylMask_mm(
      nx, ny, localNz, spacing[0], spacing[1], spacing[2],
      args.cylinderDiameterMM, args.cylinderHeightMM, args.cylCenterX,
      args.cylCenterY, args.cylCenterZ);
  if (bedCount == 1) {
    mask = localMask;
    return &mask;
  }

  // Union of the bed grids; overlapping beds share voxels of one volume.
  const std::vector<int> starts =
      spect::app::computeBedZOffsets(bedCount, localNz, args.bedOverlaps);
  if (starts.back() + localNz != nz)
    throw std::runtime_error("Cylindrical ROI bed layout does not match Nz.");
  mask.assign(static_cast<size_t>(nx) * static_cast<size_t>(ny) *
                  static_cast<size_t>(nz),
              0);
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  for (int start : starts) {
    const size_t offset = static_cast<size_t>(start) * plane;
    for (size_t i = 0; i < localMask.size(); ++i)
      mask[offset + i] = std::max(mask[offset + i], localMask[i]);
  }
  return &mask;
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

ReconPreparedParams prepareReconPreparedParams(const ReconArgs &args) {
  ReconPreparedParams prepared;
  prepared.butterConf = prepareButterworthConf(args);
  prepared.smoothConf = prepareSmoothingConf(args);
  return prepared;
}

filters::PSFKernels preparePSFForRuntime(ReconOpts &opts, const ReconArgs &args,
                                         const filters::Spacing3 &spacing) {
  opts.psf = args.psf;
  opts.psf.vox_x = spacing.dx;
  opts.psf.vox_y = spacing.dy;
  opts.psf.vox_z = spacing.dz;

  if (!args.usePSF)
    opts.psf.mode = filters::PSFMode::None;
  if (opts.psf.mode == filters::PSFMode::None)
    return {};
  if (!cuda::isCudaVolumeOpsAvailable())
    throw std::runtime_error(
        "PSF-enabled reconstruction requires CUDA volume operations; "
        "SPECT_CUDA_VOLUME_OPS=0 and CPU fallback are not supported.");
  return filters::make_psf_kernels(opts.psf);
}

} // namespace spect::main
