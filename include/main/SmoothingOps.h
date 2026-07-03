#pragma once

#include <vector>

namespace spect::main {

void applyJosephSmoothing(std::vector<double> &buffer, int nx, int ny, int nz,
                          int passes, int threads);

void clampNonNegativeInPlace(std::vector<double> &buffer, int threads);

} // namespace spect::main

