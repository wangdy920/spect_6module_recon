#pragma once

#include "recon/ReconConfig.h"
#include "cli/ReconArgs.h"
#include "filters/PSFConfig.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spect::main {

struct SmoothConf {
  bool enabled = false;
  std::vector<double> kx;
  std::vector<double> ky;
  std::vector<double> kz;
};

struct ReconPreparedParams {
  filters::PSFKernels psfKernels;
  filters::ButterworthConf butterConf;
  SmoothConf smoothConf;
};

struct ReconPreparationLogs {
  std::vector<std::string> infoMessages;
  std::vector<std::string> warningMessages;
};

struct PSFPreparationResult {
  filters::PSFKernels psfKernels;
  ReconPreparationLogs logs;
};

std::array<double, 3> voxelSpacingFromArgs(const ReconArgs &args);

const std::vector<uint8_t> *
maybeMakeCylindricalMask(std::vector<uint8_t> &mask, int nx, int ny, int nz,
                         const std::array<double, 3> &spacing,
                         const ReconArgs &args);

filters::PSFKernels preparePSFKernels(const filters::PSFConf &psfConf);

filters::ButterworthConf prepareButterworthConf(const ReconArgs &args);

SmoothConf prepareSmoothingConf(const ReconArgs &args);

ReconPreparedParams prepareReconPreparedParams(const ReconArgs &args,
                                               const filters::PSFConf &psfConf);

PSFPreparationResult
preparePSFForRuntime(ReconOpts &opts, const ReconArgs &args,
                     const filters::Spacing3 &spacing, int bedCount,
                     size_t projectionCount, int imageNx);

} // namespace spect::main
