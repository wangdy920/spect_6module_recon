#pragma once

#include <vector>

namespace spect::main {

void clampNonNegativeInPlace(std::vector<double> &buffer, int threads);

} // namespace spect::main

