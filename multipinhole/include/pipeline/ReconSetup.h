#pragma once

#include "cli/ReconArgs.h"
#include "pipeline/VolumeOutput.h"

#include <string>
#include <vector>

namespace spect {
namespace main {

struct BedSetupResult {
  int baseNx = 0;
  int baseNy = 0;
  int baseNz = 0;
  int bedCount = 1;
  int totalOverlap = 0;
  std::vector<std::string> projectionPaths;
  std::vector<std::string> warnings;
  std::string summary;
  CanonicalOutputGeometry canonicalGeom;
};

bool adoptReconDimsFromHint(ReconArgs &args, std::string *warningMessage = nullptr);
bool ensureReconDims(ReconArgs &args, std::string *warningMessage = nullptr);
BedSetupResult prepareBedSetup(ReconArgs &args);

} // namespace main
} // namespace spect
