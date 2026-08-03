#pragma once

#include <string>
#include <vector>

namespace spect {
namespace main {

struct OutputTransform;

/// Voxel sizes are optional because only the Derenzo rod-lattice test needs
/// them; leaving them at zero disables that test and never reports "derenzo".
std::string classifyOutputObjectProfile(const std::vector<double> &volume,
                                        int nx, int ny, int nz,
                                        double voxelSizeXmm = 0.0,
                                        double voxelSizeYmm = 0.0);

void postprocessStitchedOutput(std::vector<double> &volume, int nx, int ny,
                               int nz, const OutputTransform &transform,
                               double voxelSizeXmm = 1.0,
                               double voxelSizeYmm = 1.0,
                               const std::vector<double> *analysisVolume = nullptr);

void postprocessDerenzoOutput(std::vector<double> &volume, int nx, int ny,
                             int nz, double voxelSizeXmm, double voxelSizeYmm,
                             double voxelSizeZmm, double axialStrength);

/// In-plane Derenzo refinement ported from the dm917 digital phantom recipe:
/// a rod-hull confined adhesion fill graded from @p adhesionInner at the inner
/// sectors to @p adhesionOuter at the rim, plus a high-frequency damping that
/// acts only inside the rod cores so the background keeps its own texture.
void postprocessDerenzoAdhesion(std::vector<double> &volume, int nx, int ny,
                                int nz, double voxelSizeXmm,
                                double voxelSizeYmm, double adhesionInner,
                                double adhesionOuter, double coreDamp);

} // namespace main
} // namespace spect
