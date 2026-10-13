#pragma once

#include "recon/ReconConfig.h"
#include "cli/ReconArgs.h"
#include "filters/PSFConfig.h"

#include <array>
#include <cstdint>
#include <vector>

namespace spect::main {

struct SmoothConf {
  bool enabled = false;
  std::vector<double> kx;
  std::vector<double> ky;
  std::vector<double> kz;
};

struct ReconPreparedParams {
  filters::ButterworthConf butterConf;
  SmoothConf smoothConf;
};

std::array<double, 3> voxelSpacingFromArgs(const ReconArgs &args);

const std::vector<uint8_t> *
maybeMakeCylindricalMask(std::vector<uint8_t> &mask, int nx, int ny, int nz,
                         const std::array<double, 3> &spacing,
                         const ReconArgs &args);

filters::ButterworthConf prepareButterworthConf(const ReconArgs &args);

SmoothConf prepareSmoothingConf(const ReconArgs &args);

ReconPreparedParams prepareReconPreparedParams(const ReconArgs &args);

filters::PSFKernels preparePSFForRuntime(ReconOpts &opts, const ReconArgs &args,
                                         const filters::Spacing3 &spacing);

} // namespace spect::main
