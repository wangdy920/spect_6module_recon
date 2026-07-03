#pragma once

#include <string>

namespace spect {
namespace main {

std::string extractMatVarName(std::string &path);
std::string resolveMatVarName(std::string &path, const std::string &fallback);

} // namespace main
} // namespace spect

