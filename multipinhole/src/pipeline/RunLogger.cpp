#include "pipeline/RunLogger.h"
#include "pipeline/RunLogging.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>

namespace spect {
namespace main {

static int gVerbosity = 1;
static std::ostream *gFileLog = nullptr;

void setVerbosity(int verbosity) { gVerbosity = verbosity; }
int getVerbosity() { return gVerbosity; }

void setFileLogStream(std::ostream *stream) { gFileLog = stream; }
std::ostream *getFileLogStream() { return gFileLog; }

namespace {

// Terminal timestamp prefix "YYYY-MM-DD HH:MM:SS" (local time).
std::string timestampNow() {
  const std::time_t t = std::chrono::system_clock::to_time_t(
      std::chrono::system_clock::now());
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  return buf;
}

void writeLine(std::ostream &os, const std::string &prefix,
               const std::string &message) {
  os << timestampNow() << prefix << message << '\n';
  // Progress-bar hosts read a redirected stdout pipe; without this the
  // [ITER] line can sit in the 4 KiB buffer until the process exits.
  os.flush();
}

} // namespace

void logInfo(const std::string &message) {
  // Progress and completion always reach cout (and the tee'd .log).
  // Other INFO stays quiet at -vb 0 and is
  // written only to the file sink so the console is not duplicated.
  if (isMinimalConsoleLine(message) || gVerbosity >= 1)
    writeLine(std::cout, " ", message);
  else if (gFileLog)
    writeLine(*gFileLog, " ", message);
}

void logDebug(const std::string &message) {
  if (gVerbosity >= 2)
    writeLine(std::cout, " ", message);
}

void logWarn(const std::string &message) {
  if (gVerbosity >= 0)
    writeLine(std::cerr, " [NOTE] ", message);
}

} // namespace main
} // namespace spect
