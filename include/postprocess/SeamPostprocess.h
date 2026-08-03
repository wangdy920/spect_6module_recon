#pragma once

#include <string>
#include <vector>

namespace spect {
namespace main {

struct OutputTransform;

std::string classifyOutputObjectProfile(const std::vector<double> &volume,
                                        int nx, int ny, int nz);

void postprocessStitchedOutput(std::vector<double> &volume, int nx, int ny,
                               int nz, const OutputTransform &transform,
                               double voxelSizeXmm = 1.0,
                               double voxelSizeYmm = 1.0,
                               const std::vector<double> *analysisVolume = nullptr);

void postprocessDerenzoOutput(std::vector<double> &volume, int nx, int ny,
                             int nz, double voxelSizeXmm, double voxelSizeYmm,
                             double voxelSizeZmm, double axialStrength);

} // namespace main
} // namespace spect
