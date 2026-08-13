#include "pipeline/RunLogging.h"

#include "pipeline/RunLogger.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

class TeeBuf : public std::streambuf {
public:
  TeeBuf(std::streambuf *primary, std::streambuf *secondary)
      : primary_(primary), secondary_(secondary) {}

protected:
  int overflow(int ch) override {
    if (ch == EOF)
      return 0;
    const char c = static_cast<char>(ch);
    const int r1 = primary_ ? primary_->sputc(c) : ch;
    const int r2 = secondary_ ? secondary_->sputc(c) : ch;
    // Flush the log file at line ends so a kill/crash keeps a usable tail.
    // (ofstream unitbuf does not apply: writes go through filebuf::sputc.)
    if (secondary_ && c == '\n')
      secondary_->pubsync();
    return (r1 == EOF || r2 == EOF) ? EOF : ch;
  }

  int sync() override {
    const int r1 = primary_ ? primary_->pubsync() : 0;
    const int r2 = secondary_ ? secondary_->pubsync() : 0;
    return (r1 == 0 && r2 == 0) ? 0 : -1;
  }

private:
  std::streambuf *primary_ = nullptr;
  std::streambuf *secondary_ = nullptr;
};

std::string sanitizeFilename(std::string value) {
  for (char &ch : value) {
    switch (ch) {
    case '<':
    case '>':
    case ':':
    case '"':
    case '/':
    case '\\':
    case '|':
    case '?':
    case '*':
      ch = '_';
      break;
    default:
      break;
    }
  }
  if (value.empty())
    value = "recon";
  return value;
}

std::string timestampForFilename() {
  const auto now = std::chrono::system_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
  char out[80];
  std::snprintf(out, sizeof(out), "%s_%03d", buf, static_cast<int>(ms.count()));
  return std::string(out);
}

} // namespace

namespace spect {
namespace main {

struct ConsoleTeeGuard::Impl {
  std::filesystem::path path;
  std::ofstream file;
  std::unique_ptr<TeeBuf> teeCout;
  std::unique_ptr<TeeBuf> teeCerr;
  std::streambuf *oldCout = nullptr;
  std::streambuf *oldCerr = nullptr;
  std::string error;
  bool active = false;
};

ConsoleTeeGuard::ConsoleTeeGuard(const std::filesystem::path &logPath)
    : impl_(std::make_unique<Impl>()) {
  impl_->path = logPath;
  if (impl_->path.empty()) {
    impl_->error = "empty log path";
    return;
  }

  std::error_code error;
  const auto parent = impl_->path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, error);
    if (error) {
      impl_->error = "create_directories(" + parent.string() +
                     ") failed: " + error.message();
      return;
    }
  }

  // Disable the filebuf buffer before open so line-sync in TeeBuf is enough
  // and ofstream::unitbuf is not relied upon (tee writes bypass ofstream).
  impl_->file.rdbuf()->pubsetbuf(nullptr, 0);
  impl_->file.open(impl_->path, std::ios::out | std::ios::binary);
  if (!impl_->file) {
    impl_->error = "open(" + impl_->path.string() + ") failed";
    return;
  }

  impl_->oldCout = std::cout.rdbuf();
  impl_->oldCerr = std::cerr.rdbuf();
  impl_->teeCout =
      std::make_unique<TeeBuf>(impl_->oldCout, impl_->file.rdbuf());
  impl_->teeCerr =
      std::make_unique<TeeBuf>(impl_->oldCerr, impl_->file.rdbuf());
  std::cout.rdbuf(impl_->teeCout.get());
  std::cerr.rdbuf(impl_->teeCerr.get());
  // Quiet runs (-vb 0) suppress logInfo on cout; keep the durable log filled.
  setFileLogStream(&impl_->file);
  impl_->active = true;
}

ConsoleTeeGuard::~ConsoleTeeGuard() {
  if (!impl_ || !impl_->active)
    return;
  if (getFileLogStream() == &impl_->file)
    setFileLogStream(nullptr);
  std::cout.flush();
  std::cerr.flush();
  std::cout.rdbuf(impl_->oldCout);
  std::cerr.rdbuf(impl_->oldCerr);
  impl_->file.flush();
}

bool ConsoleTeeGuard::enabled() const { return impl_ && impl_->active; }

const std::filesystem::path &ConsoleTeeGuard::logPath() const {
  return impl_->path;
}

const std::string &ConsoleTeeGuard::errorMessage() const {
  static const std::string kEmpty;
  return impl_ ? impl_->error : kEmpty;
}

std::filesystem::path resolveLogDirectory(const ReconArgs &args,
                                          std::string &errorMessage) {
  try {
    std::filesystem::path dir(args.outputDir.empty() ? "output" : args.outputDir);
    std::filesystem::create_directories(dir);
    errorMessage.clear();
    std::error_code absError;
    auto abs = std::filesystem::absolute(dir, absError);
    return absError ? dir : abs;
  } catch (const std::exception &ex) {
    errorMessage = ex.what();
    return {};
  }
}

std::filesystem::path makeDefaultLogPath(const ReconArgs &args) {
  const std::filesystem::path dir =
      args.outputDir.empty() ? std::filesystem::path(".")
                             : std::filesystem::path(args.outputDir);
  const std::string stem =
      sanitizeFilename(args.outputName.empty() ? std::string("recon")
                                               : args.outputName);
  const std::filesystem::path relative =
      dir / (stem + "_" + timestampForFilename() + ".log");
  std::error_code absError;
  auto abs = std::filesystem::absolute(relative, absError);
  return absError ? relative : abs;
}

std::string makeCommandLineString(int argc, char **argv) {
  std::ostringstream oss;
  for (int i = 0; i < argc; ++i) {
    if (i > 0)
      oss << ' ';
    std::string arg = argv[i] ? std::string(argv[i]) : std::string();
    const bool needsQuotes = (arg.find(' ') != std::string::npos) ||
                             (arg.find('\t') != std::string::npos);
    if (needsQuotes)
      oss << '"' << arg << '"';
    else
      oss << arg;
  }
  return oss.str();
}

} // namespace main
} // namespace spect