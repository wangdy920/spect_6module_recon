#pragma once

#include "cli/ReconArgs.h"

#include <cstdint>
#include <vector>

namespace spect::main {

std::vector<double> computeKappaMapStandard(
    const std::vector<double> &globalSensitivity, const cli::ReconArgs &args,
    const std::vector<uint8_t> *mask);

std::vector<double> computeKappaMapCompact(
    const std::vector<double> &globalSensitivity, const cli::ReconArgs &args,
    const std::vector<uint8_t> *mask);

} // namespace spect::main