#pragma once

#include "app/ReconConfig.h"
#include "cli/ReconArgs.h"
#include "filters/PSFConfig.h"
#include "io/CSRBuilder.h"
#include "main/ReconPreparation.h"
#include "main/VolumeOutput.h"
#include "sparse/SparseOpsBackend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spect::main {

struct StandardReconLoopContext {
  const ReconArgs *args = nullptr;
  const ReconOpts *opts = nullptr;
  const ReconPreparedParams *prepared = nullptr;
  const std::vector<double> *projections = nullptr;
  const std::vector<uint8_t> *mask = nullptr;
  const std::vector<double> *sharedAnatomical = nullptr;
  const filters::PSFKernels *psfKernels = nullptr;
  filters::Spacing3 spacing{};
  OutputGeometry outputGeometry{};
  CropSettings crop{};
  const OutputTransform *outputTransform = nullptr;
  SparseBackendChoice resolvedBackend = SparseBackendChoice::Auto;
  int bedCount = 1;
  int baseNx = 0;
  int baseNy = 0;
  int baseNz = 0;
  size_t globalNvox = 0;
  const std::vector<int> *bedZStarts = nullptr;
};

class StandardReconLoop {
public:
  explicit StandardReconLoop(StandardReconLoopContext context);

  std::vector<double> run(const io::CSR &matrix) const;
  std::vector<double> run(const io::CSRView &matrix) const;

private:
  StandardReconLoopContext context_;
};

} // namespace spect::main
