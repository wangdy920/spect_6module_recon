#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace spect::core {

inline std::string normalizePathSeparators(std::string path) {
  for (char &ch : path) {
    if (ch == '\\')
      ch = '/';
  }
  return path;
}

inline int extractBedNumberFromPath(const std::string &path) {
  std::string lower = path;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  std::size_t pos = 0;
  while ((pos = lower.find("bed", pos)) != std::string::npos) {
    std::size_t p = pos + 3;
    while (p < lower.size() &&
           (lower[p] == '_' || lower[p] == '-' ||
            std::isspace(static_cast<unsigned char>(lower[p])))) {
      ++p;
    }
    if (p < lower.size() &&
        std::isdigit(static_cast<unsigned char>(lower[p]))) {
      int value = 0;
      while (p < lower.size() &&
             std::isdigit(static_cast<unsigned char>(lower[p]))) {
        value = value * 10 + (lower[p] - '0');
        ++p;
      }
      return value;
    }
    pos += 3;
  }
  return -1;
}

inline std::vector<std::string> sortPathsByBedOrder(
    std::vector<std::string> paths) {
  for (std::string &path : paths)
    path = normalizePathSeparators(std::move(path));
  std::stable_sort(paths.begin(), paths.end(),
                   [](const std::string &a, const std::string &b) {
                     const int left = extractBedNumberFromPath(a);
                     const int right = extractBedNumberFromPath(b);
                     const int leftOrder =
                         left < 0 ? std::numeric_limits<int>::max() : left;
                     const int rightOrder =
                         right < 0 ? std::numeric_limits<int>::max() : right;
                     return leftOrder < rightOrder;
                   });
  return paths;
}

} // namespace spect::core
