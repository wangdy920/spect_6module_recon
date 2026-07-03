#include "main/RunLogger.h"

#include <iostream>

namespace spect {
namespace main {

static int gVerbosity = 1;

void setVerbosity(int verbosity) { gVerbosity = verbosity; }
int getVerbosity() { return gVerbosity; }

void logInfo(const std::string &message) {
  if (gVerbosity >= 1)
    std::cout << message << '\n';
}

void logDebug(const std::string &message) {
  if (gVerbosity >= 2)
    std::cout << message << '\n';
}

void logWarn(const std::string &message) {
  if (gVerbosity >= 0)
    std::cerr << "[WARN] " << message << '\n';
}

} // namespace main
} // namespace spect