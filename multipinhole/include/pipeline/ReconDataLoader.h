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

// Expands directory inputs to BedK.bin lists: -beds N|A:B searches the -df
// (or single -df2) directory, and merge_time_segments expands a
// timestamp/SPECT directory to all of its beds.
void prepareProjectionInputs(ReconArgs &args);

// Reconstruction consumes complete projections matching the matrix geometry.
LoadedProjectionSet loadProjectionSet(const std::vector<std::string> &paths,
                                      const io::CompactSymmetryMetadata &meta,
                                      const std::string &label,
                                      const std::string &defaultVar,
                                      bool mergeTimeSegments = false);

std::vector<double> loadVolumeChecked(const std::string &path,
                                      const std::string &label,
                                      int expectedNx, int expectedNy,
                                      int expectedNz,
                                      const filters::Spacing3 &expectedSpacing,
                                      bool nonnegative);

std::vector<double> loadAutoMuActivity(const std::string &path, int nx, int ny,
                                       int nz, const filters::Spacing3 &spacing);

} // namespace spect::main
