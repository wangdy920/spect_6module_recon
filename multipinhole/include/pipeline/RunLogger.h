#pragma once

#include <iosfwd>
#include <string>

namespace spect {
namespace main {

void setVerbosity(int verbosity);
int getVerbosity();

/// Optional file sink for quiet runs: when verbosity suppresses console
/// `logInfo`/`logDebug`, lines are still written here so the per-run `.log`
/// stays a complete record.  `[ITER]` progress lines always go to cout
/// (tee'd into the same file) even at `-vb 0`.  When verbosity allows
/// console output, the ConsoleTeeGuard already captures cout/cerr — do
/// not also write here in that case (would duplicate).
void setFileLogStream(std::ostream *stream);
std::ostream *getFileLogStream();

void logInfo(const std::string &message);
void logDebug(const std::string &message);
void logWarn(const std::string &message);

} // namespace main
} // namespace spect
