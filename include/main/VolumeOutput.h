#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "main/SeamPostprocess.h"

namespace spect {
namespace main {

inline constexpr int kCenter66CropLayers = 66;
inline constexpr int kCenter66BedStepLayers = 56;
inline constexpr int kCenter66OverlapLayers =
    kCenter66CropLayers - kCenter66BedStepLayers;

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
  bool flipZWholeOutput = false;
  bool flipXWholeOutput = false;
  bool sigmoidBlendOutput = false;
  bool center66CosineOutput = false;
  bool objectAwareSeamPostprocess = false;
  std::string objectProfile = "auto";
  bool derenzoAxialPostprocess = false;
  double derenzoAxialStrength = 0.30;
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
  double mouseSeamStrength = 0.0;
  int seamHalfWidth = 8;
  int baseNz = 0;
  std::vector<int> bedZStarts;
  std::vector<int> outputBedOverlaps;
  double fovRadiusMm = 0.0;
  double analysisFovRadiusMm = 0.0;
};

CropSettings sanitizeCropSettings(int cropStart, int cropEnd);
bool isCropSettingsValid(int nz, const CropSettings &crop);

std::vector<double> prepareOutputVolume(const std::vector<double> &source,
                                        int nx, int ny, int nz,
                                        const std::vector<uint8_t> *mask,
                                        const CropSettings &crop,
                                        int *outNz = nullptr,
                                        const OutputTransform *transform = nullptr,
                                        double voxelSizeXmm = 1.0,
                                        double voxelSizeYmm = 1.0,
                                        double voxelSizeZmm = 1.0);

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
