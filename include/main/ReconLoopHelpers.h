#pragma once

#include "app/ReconConfig.h"
#include "cli/ReconArgs.h"
#include "cuda/ProjDomainChunked.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spect::main {

std::string formatProfileEntry(const std::string &label, double seconds,
                               size_t calls, uint64_t bytes = 0);

void logCudaChunkTimingSnapshot(
    const spect::cuda::CudaChunkStageTimingSnapshot &cudaChunkTiming);

std::string outputPrefixFromArgs(const ReconArgs &args);

bool tryComputeRegGradientCUDA(const std::vector<double> &image,
                               const ReconArgs &args, const ReconOpts &opts,
                               const std::vector<uint8_t> *maskForReg,
                               std::vector<double> &gradReg);

bool tryApplySeparableConvCUDA(std::vector<double> &volume, int nx, int ny,
                               int nz, const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz,
                               const ReconOpts &opts);

void applySeparableConvRuntime(std::vector<double> &volume, int nx, int ny,
                               int nz, const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz,
                               const ReconOpts &opts, int threads);

void applyJosephLike(std::vector<double> &buffer, int nx, int ny, int nz,
                     int passes, int threads, bool clampImage);

} // namespace spect::main