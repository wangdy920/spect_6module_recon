#include "pipeline/RunLogging.h"

#include "pipeline/RunLogger.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>


namespace {

class TeeBuf : public std::streambuf {
public:
  // quietConsole (-vb 0): the console only receives lines accepted by
  // spect::main::isMinimalConsoleLine; the file still receives every byte.
  TeeBuf(std::streambuf *primary, std::streambuf *secondary, bool quietConsole)
      : primary_(primary), secondary_(secondary), quietConsole_(quietConsole) {}

  // Emits a trailing line without '\n' (quiet mode holds it until then).
  void flushPendingLine() {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    emitPendingLocked();
  }

protected:
  int overflow(int ch) override {
    if (ch == EOF)
      return 0;
    const char c = static_cast<char>(ch);
    const int r2 = secondary_ ? secondary_->sputc(c) : ch;
    int r1 = ch;
    if (!quietConsole_) {
      r1 = primary_ ? primary_->sputc(c) : ch;
    } else {
      std::lock_guard<std::mutex> lock(pendingMutex_);
      pending_.push_back(c);
      if (c == '\n')
        r1 = emitPendingLocked();
    }
    // Flush both the log file and the console/pipe at line ends so a
    // kill/crash keeps a usable tail and a progress-bar host reading
    // redirected stdout sees `[ITER]` immediately.
    if (c == '\n') {
      if (secondary_)
        secondary_->pubsync();
      if (primary_)
        primary_->pubsync();
    }
    return (r1 == EOF || r2 == EOF) ? EOF : ch;
  }

  int sync() override {
    const int r1 = primary_ ? primary_->pubsync() : 0;
    const int r2 = secondary_ ? secondary_->pubsync() : 0;
    return (r1 == 0 && r2 == 0) ? 0 : -1;
  }

private:
  int emitPendingLocked() {
    int result = 0;
    if (primary_ && spect::main::isMinimalConsoleLine(pending_)) {
      const auto size = static_cast<std::streamsize>(pending_.size());
      result = primary_->sputn(pending_.data(), size) == size ? 0 : EOF;
    }
    pending_.clear();
    return result;
  }

  std::streambuf *primary_ = nullptr;
  std::streambuf *secondary_ = nullptr;
  const bool quietConsole_ = false;
  std::mutex pendingMutex_;
  std::string pending_;
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
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
  return std::string(buf);
}

} // namespace

namespace spect {
namespace main {

bool isMinimalConsoleLine(const std::string &line) {
  size_t start = 0;
  const auto digitsAt = [&line](size_t pos, size_t count) {
    for (size_t i = pos; i < pos + count; ++i)
      if (line[i] < '0' || line[i] > '9')
        return false;
    return true;
  };
  // Optional "YYYY-MM-DD HH:MM:SS " prefix written by RunLogger.
  if (line.size() >= 20 && digitsAt(0, 4) && line[4] == '-' && digitsAt(5, 2) &&
      line[7] == '-' && digitsAt(8, 2) && line[10] == ' ' && digitsAt(11, 2) &&
      line[13] == ':' && digitsAt(14, 2) && line[16] == ':' && digitsAt(17, 2) &&
      line[19] == ' ')
    start = 20;
  for (const char *tag : {"[ITER] ", "Error", "[ERROR]", "[FATAL]"})
    if (line.compare(start, std::char_traits<char>::length(tag), tag) == 0)
      return true;
  return false;
}

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
  const bool quietConsole = getVerbosity() < 1;
  impl_->teeCout = std::make_unique<TeeBuf>(impl_->oldCout, impl_->file.rdbuf(),
                                            quietConsole);
  impl_->teeCerr = std::make_unique<TeeBuf>(impl_->oldCerr, impl_->file.rdbuf(),
                                            quietConsole);
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
  impl_->teeCout->flushPendingLine();
  impl_->teeCerr->flushPendingLine();
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
