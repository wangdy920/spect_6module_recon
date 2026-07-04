#pragma once

#include "app/ReconConfig.h"
#include "cli/ReconArgs.h"
#include "filters/PSFConfig.h"
#include "io/CSRBuilder.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "main/ReconPreparation.h"
#include "main/VolumeOutput.h"

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace spect::main {

struct CompactExecutionTiming {
  double projectionLoadSeconds = 0.0;
  double csrMapOrLoadSeconds = 0.0;
  double metadataLoadAndValidateSeconds = 0.0;
  double psfPrepareSeconds = 0.0;
  double compactSensitivitySeconds = 0.0;
  double compactForwardRatioBackwardSeconds = 0.0;
  double regGradientSeconds = 0.0;
  size_t regGradientCalls = 0;
  double mlemUpdateSeconds = 0.0;
  double butterworthSeconds = 0.0;
  double outputWriteSeconds = 0.0;
  double totalLoadToIterationDoneSeconds = 0.0;
  double fullCommandWallSeconds = 0.0;
};

struct CompactSymmetricReconLoopContext {
  const ReconArgs *args = nullptr;
  const ReconOpts *opts = nullptr;
  const ReconPreparedParams *prepared = nullptr;
  const std::vector<double> *projections = nullptr;
  const std::vector<double> *backgroundProj = nullptr;
  const std::vector<uint8_t> *mask = nullptr;
  const filters::PSFKernels *psfKernels = nullptr;
  filters::Spacing3 spacing{};
  OutputGeometry outputGeometry{};
  CropSettings crop{};
  const OutputTransform *outputTransform = nullptr;
  const io::CompactSymmetryMetadata *compactMeta = nullptr;
  const std::string *compactMetaPath = nullptr;
  bool compactChunkLayout = false;
  const io::CompactChunkCSRDirectory *compactChunks = nullptr;
  CompactExecutionTiming *timing = nullptr;
  std::chrono::steady_clock::time_point loadToIterationStart{};
  std::chrono::steady_clock::time_point fullCommandStart{};
  int bedCount = 1;
  int baseNx = 0;
  int baseNy = 0;
  int baseNz = 0;
  size_t baseNvox = 0;
  size_t globalNvox = 0;
  const std::vector<int> *bedZStarts = nullptr;
};

class CompactSymmetricReconLoop {
public:
  explicit CompactSymmetricReconLoop(CompactSymmetricReconLoopContext context);

  std::vector<double> run(const io::CSR &matrix) const;
  std::vector<double> run(const io::CSRView &matrix) const;

private:
  CompactSymmetricReconLoopContext context_;
};

} // namespace spect::main