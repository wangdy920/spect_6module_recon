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
  bool flipZWholeOutput = false;
  int baseNz = 0;
  std::vector<int> bedZStarts;
};

CropSettings sanitizeCropSettings(int cropStart, int cropEnd);
bool isCropSettingsValid(int nz, const CropSettings &crop);

std::vector<double> prepareOutputVolume(const std::vector<double> &source,
                                        int nx, int ny, int nz,
                                        const std::vector<uint8_t> *mask,
                                        const CropSettings &crop,
                                        int *outNz = nullptr,
                                        const OutputTransform *transform = nullptr);

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
