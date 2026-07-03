#include "main/MatVarPath.h"

namespace spect {
namespace main {

std::string extractMatVarName(std::string &path) {
  const size_t pos = path.rfind(':');
  if (pos == std::string::npos)
    return {};
  if (pos == 1 && path.size() >= 3 && (path[2] == '/' || path[2] == '\\'))
    return {};

  const std::string suffix = path.substr(pos + 1);
  if (suffix.empty() || suffix.find('/') != std::string::npos || suffix.find('\\') != std::string::npos)
    return {};

  path = path.substr(0, pos);
  return suffix;
}

std::string resolveMatVarName(std::string &path, const std::string &fallback) {
  const std::string extracted = extractMatVarName(path);
  return extracted.empty() ? fallback : extracted;
}

} // namespace main
} // namespace spect