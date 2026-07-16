#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace spect {
namespace main {

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
  bool flipZPerBed = false;
  bool jointVirtualZFlip = false;
  bool flipZWholeOutput = false;
  bool flipXWholeOutput = false;
  bool bedOverlapInRecon = true;
  bool sigmoidBlendOutput = false;
  bool objectAwareSeamPostprocess = false;
  std::string objectProfile = "auto";
  double phantomCloseStrength = 1.0;
  double epguanCloseStrength = 1.0;
  bool epguanWallMaxPreserve = false;
  double epguanSeamEdgeStrength = 0.60;
  int epguanSeamFeatherSlices = 4;
  bool epguanBackgroundFilter = true;
  double epguanBackgroundRetain = 0.08;
  double epguanSupportThreshold = 0.06;
  double epguanSupportMarginMm = 1.5;
  double zimuCloseStrength = 0.5;
  int seamHalfWidth = 8;
  int baseNz = 0;
  std::vector<int> bedZStarts;
  std::vector<int> outputBedOverlaps;
  double fovRadiusMm = 0.0;
};

CropSettings sanitizeCropSettings(int cropStart, int cropEnd);
bool isCropSettingsValid(int nz, const CropSettings &crop);

void postprocessStitchedOutput(std::vector<double> &volume, int nx, int ny,
                               int nz, const OutputTransform &transform,
                               double voxelSizeXmm = 1.0,
                               double voxelSizeYmm = 1.0);

std::vector<double> prepareOutputVolume(const std::vector<double> &source,
                                        int nx, int ny, int nz,
                                        const std::vector<uint8_t> *mask,
                                        const CropSettings &crop,
                                        int *outNz = nullptr,
                                        const OutputTransform *transform = nullptr,
                                        double voxelSizeXmm = 1.0,
                                        double voxelSizeYmm = 1.0);

void saveVolumeOutput(const std::vector<double> &source,
                      const OutputGeometry &geometry,
                      const std::string &outputDir,
                      const std::string &outputPrefix, int iteration,
                      const std::vector<uint8_t> *mask,
                      const CropSettings &crop,
                      bool littleEndian = true,
                      const OutputTransform *transform = nullptr);

} // namespace main
} // namespace spect
