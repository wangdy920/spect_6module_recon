#pragma once

#include "cli/ReconArgs.h"

#include <filesystem>
#include <memory>
#include <string>

namespace spect {
namespace main {

/// Console contract of -vb 0 (deployment default): only `[ITER] n/N compact`
/// progress lines and error lines (`Error`, `[ERROR]`, `[FATAL]`), with or
/// without the `YYYY-MM-DD HH:MM:SS ` prefix. `line` may end in '\n'.
bool isMinimalConsoleLine(const std::string &line);

/// Tees cout/cerr into `logPath`. At verbosity 0 (read at construction) the
/// console only shows isMinimalConsoleLine lines; the file keeps everything.
class ConsoleTeeGuard {
public:
  explicit ConsoleTeeGuard(const std::filesystem::path &logPath);
  ConsoleTeeGuard(const ConsoleTeeGuard &) = delete;
  ConsoleTeeGuard &operator=(const ConsoleTeeGuard &) = delete;
  ~ConsoleTeeGuard();

  bool enabled() const;
  const std::filesystem::path &logPath() const;
  /// Empty when enabled(); otherwise why the tee could not open.
  const std::string &errorMessage() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::filesystem::path resolveLogDirectory(const ReconArgs &args,
                                          std::string &errorMessage);
std::filesystem::path makeDefaultLogPath(const ReconArgs &args);
std::string makeCommandLineString(int argc, char **argv);

} // namespace main
} // namespace spect

