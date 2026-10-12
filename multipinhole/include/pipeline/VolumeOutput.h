#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/SixModuleGeometry.h"

namespace spect {
namespace main {

/// Canonical output geometry derived from baseNz and bedOverlap.
/// stepLayers = baseNz - bedOverlap  (physical bed step in layers)
/// cropLayers = stepLayers + blendMargin (PSC), or baseNz (matrix grid)
/// blendOverlap = cropLayers - stepLayers
struct CanonicalOutputGeometry {
  int cropLayers = 0;
  int stepLayers = 0;
  int blendOverlap = 0;
};

/// Compute canonical output geometry from baseNz and bedOverlap.
/// Validates that all overlaps produce the same step, that the step
/// matches the scanner bed step (bedStepMm, default 16.8 mm) at the
/// configured voxel spacing within half-layer quantization, and that the
/// physical bed step is respected within tolerance. The blend margin
/// defaults to 10 layers; if cropLayers would exceed baseNz, it is
/// clamped and the blend margin shrinks accordingly.
CanonicalOutputGeometry computeCanonicalOutputGeometry(
    int baseNz, const std::vector<int>& outputBedOverlaps,
    double voxelSizeZmm, double bedStepMm = 0.0);

struct CropSettings {
  int start = 0;
  int end = 0;
};

struct OutputGeometry {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  double vx = 1.0;
  double vy = 1.0;
  double vz = 1.0;
};

struct OutputTransform {
  bool matrixGridOutput = false; // Full bed grids, no prototype gain fitting.
  bool flipZPerBed = false;
  bool flipZWholeOutput = false;
  bool flipXWholeOutput = false;
  double xyRotationDeg = 0.0; // After flips/crop/mask; positive maps +X toward +Y.
  bool reverseLocalZInReconStorage = false;
  /// Opt-in: integer XY translation of each bed so the 10-layer physical
  /// overlap matches its neighbour (NCC search). Default off. Does not
  /// change output Nz.
  bool seamXyAlign = false;
  int centerCropNx = 0;
  int centerCropNy = 0;
  int centerCropNz = 0;
  int baseNz = 0;
  CanonicalOutputGeometry canonicalGeom;
  std::vector<int> bedZStarts;
  std::vector<int> outputBedOverlaps;
};

CropSettings sanitizeCropSettings(int cropStart, int cropEnd);
bool isCropSettingsValid(int nz, const CropSettings &crop);

/// Output filters never modify the iteration state. Suffixed copies remain
/// independent; overwritePrimary optionally chains Smooth then Butterworth into
/// the unsuffixed saved volume.
struct OutputFilterSettings {
  bool smoothEnabled = false;
  double smoothFwhmMm = 0.30;
  double smoothSigmas = 3.0;
  bool butterEnabled = false;
  double butterCutoffCpcm = 5.0;
  int butterOrder = 8;
  bool nonnegativeClamp = true;
  bool overwritePrimary = false;
};

std::vector<double> prepareOutputVolume(const std::vector<double> &source,
                                        int nx, int ny, int nz,
                                        const std::vector<uint8_t> *mask,
                                        const CropSettings &crop,
                                        int *outNz = nullptr,
                                        const OutputTransform *transform = nullptr,
                                        double voxelSizeXmm = 1.0,
                                        double voxelSizeYmm = 1.0,
                                        double voxelSizeZmm = 1.0);

/// Prototype robust per-bed scalar gains: positive finite log-ratio samples
/// inside the 30 mm overlap ROI, median/MAD outlier rejection, Bed1 anchored
/// at 1.0, with the full adjacent-bed chain always applied. Exposed so the run
/// manifest records exactly the gains applied to the final saved volume.
std::vector<double> fitCenter66BedGains(const std::vector<double> &source,
                                        int nx, int ny,
                                        const OutputTransform &transform,
                                        int cropStart, double voxelSizeXmm,
                                        double voxelSizeYmm);

void saveVolumeOutput(const std::vector<double> &source,
                      const OutputGeometry &geometry,
                      const std::string &outputDir,
                      const std::string &outputPrefix, int iteration,
                      const std::vector<uint8_t> *mask,
                      const CropSettings &crop,
                      bool littleEndian = true,
                      const OutputTransform *transform = nullptr,
                      const OutputFilterSettings *outputFilters = nullptr);

} // namespace main
} // namespace spect
