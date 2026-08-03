#pragma once

#include "cli/ReconArgs.h"

#include <filesystem>
#include <memory>
#include <string>

namespace spect {
namespace main {

class ConsoleTeeGuard {
public:
  explicit ConsoleTeeGuard(const std::filesystem::path &logPath);
  ConsoleTeeGuard(const ConsoleTeeGuard &) = delete;
  ConsoleTeeGuard &operator=(const ConsoleTeeGuard &) = delete;
  ~ConsoleTeeGuard();

  bool enabled() const;
  const std::filesystem::path &logPath() const;

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

