#pragma once

#include <string>
#include <vector>

#include "cli/ReconArgs.h"
#include "io/CompactSymmetryMetadata.h"
#include "pipeline/VolumeOutput.h"

namespace spect {
namespace main {

/// Everything the run manifest writer needs. All pointers are non-owning and
/// must outlive the writeRunManifest() call.
struct RunManifestInput {
  const cli::ReconArgs *args = nullptr;
  const io::CompactSymmetryMetadata *compactMeta = nullptr;
  const std::string *compactMetaPath = nullptr;
  const std::vector<int> *bedZStarts = nullptr;
  const OutputTransform *outputTransform = nullptr;
  /// The exact volume passed to the final save; the manifest re-derives the
  /// applied per-bed gains from it (deterministic, bit-identical to the fit
  /// performed inside prepareOutputVolume on the same input).
  const std::vector<double> *finalSaveVolume = nullptr;
  const OutputGeometry *geometry = nullptr;
  int savedIteration = -1;
  std::string outputPrefix;
};

/// Writes `<outputDir>/<prefix>_manifest.json` next to the saved volumes:
/// matrix metadata, projection bed mapping (Bed1 = min physical Z), algorithm
/// policy, multibed stitch configuration with fitted gains, and produced files.
/// Throws std::runtime_error on I/O failure (fail fast).
void writeRunManifest(const RunManifestInput &input);

} // namespace main
} // namespace spect
