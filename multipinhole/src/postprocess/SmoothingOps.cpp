#include "postprocess/SmoothingOps.h"

#include <vector>

namespace spect::main {

void clampNonNegativeInPlace(std::vector<double> &buffer, int threads) {
#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(buffer.size()); ++i) {
    if (buffer[static_cast<size_t>(i)] < 0.0)
      buffer[static_cast<size_t>(i)] = 0.0;
  }
}

} // namespace spect::main