#pragma once

#include <string>

namespace spect {
namespace main {

void setVerbosity(int verbosity);
int getVerbosity();

void logInfo(const std::string &message);
void logDebug(const std::string &message);
void logWarn(const std::string &message);

} // namespace main
} // namespace spect

