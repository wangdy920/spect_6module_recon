#pragma once

#include "cli/ReconArgs.h"
#include "io/CompactSymmetryMetadata.h"

#include <cstddef>
#include <string>
#include <vector>

namespace spect::main {

struct LoadedProjectionSet {
  std::vector<double> data;
  std::size_t raysPerBed = 0;
  int detNz = 0;
  int detNx = 0;
  int numAngles = 0;
  bool detectorShapeKnown = true;
  std::vector<std::vector<std::string>> sourceGroups;
};

std::string resolveCompactMetadataPath(const ReconArgs &args);

void applyCompactMetadataGeometry(
    ReconArgs &args, const spect::io::CompactSymmetryMetadata &meta);

void prepareTimeSegmentProjectionInputs(ReconArgs &args);

LoadedProjectionSet loadProjectionSet(const std::vector<std::string> &paths,
                                      int requestedAngles,
                                      const std::string &label,
                                      const std::string &defaultVar,
                                      bool mergeTimeSegments = false);

std::vector<double> loadVolumeChecked(const std::string &path,
                                      const std::string &label,
                                      int expectedNx, int expectedNy,
                                      int expectedNz);

} // namespace spect::main
