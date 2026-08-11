// BadDetectorMask.h - Load hardware detector IDs and map them to projection rows.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace spect {
namespace io {

struct BadDetectorId {
  int link = 0;
  int fpga = 0;
  int detector = 0;
};

inline int badDetectorGlobalNumber(const BadDetectorId &id) {
  return ((id.link == 4 ? 6 : 0) + id.fpga) * 8 + id.detector;
}

inline BadDetectorId badDetectorFromGlobalNumber(int globalDetector) {
  if (globalDetector < 0 || globalDetector >= 96) {
    throw std::runtime_error("bad detector global number out of range [0,95]: " +
                             std::to_string(globalDetector));
  }

  const int boardIndex = globalDetector / 8;
  return {boardIndex < 6 ? 3 : 4, boardIndex % 6, globalDetector % 8};
}

inline std::string badDetectorName(const BadDetectorId &id) {
  return "link-" + std::to_string(id.link) + "-FPGA-" +
         std::to_string(id.fpga) + "-dector-" +
         std::to_string(id.detector);
}

inline std::string trimBadDetectorLine(std::string value) {
  const size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

inline BadDetectorId parseBadDetectorId(const std::string &value) {
  static const std::regex pattern(
      R"(^link-([34])-FPGA-([0-5])-dector-([0-7])$)",
      std::regex::icase);
  std::smatch match;
  if (!std::regex_match(value, match, pattern)) {
    throw std::runtime_error(
        "invalid bad detector identifier '" + value +
        "' (expected link-<3|4>-FPGA-<0..5>-dector-<0..7>)");
  }
  return {std::stoi(match[1].str()), std::stoi(match[2].str()),
          std::stoi(match[3].str())};
}

inline BadDetectorId parseBadDetectorEntry(const std::string &value) {
  const bool isGlobalNumber =
      !value.empty() &&
      std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0;
      });
  if (!isGlobalNumber)
    return parseBadDetectorId(value);

  try {
    size_t used = 0;
    const int globalDetector = std::stoi(value, &used);
    if (used != value.size())
      throw std::runtime_error("invalid bad detector global number: " + value);
    return badDetectorFromGlobalNumber(globalDetector);
  } catch (const std::invalid_argument &) {
    throw std::runtime_error("invalid bad detector global number: " + value);
  } catch (const std::out_of_range &) {
    throw std::runtime_error(
        "bad detector global number out of range [0,95]: " + value);
  }
}

inline std::vector<BadDetectorId>
loadBadDetectorList(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("bad detector list is not readable: " + path);

  std::vector<BadDetectorId> result;
  std::set<std::tuple<int, int, int>> seen;
  bool opened = false;
  bool closed = false;
  std::string line;
  size_t lineNumber = 0;
  while (std::getline(input, line)) {
    ++lineNumber;
    const size_t comment = std::min(line.find('#'), line.find(';'));
    if (comment != std::string::npos)
      line.erase(comment);
    line = trimBadDetectorLine(line);
    if (line.empty())
      continue;
    if (!opened) {
      if (line != "[")
        throw std::runtime_error("bad detector list must start with '[': " +
                                 path + ":" + std::to_string(lineNumber));
      opened = true;
      continue;
    }
    if (closed)
      throw std::runtime_error("unexpected content after bad detector list: " +
                               path + ":" + std::to_string(lineNumber));
    if (line == "]") {
      closed = true;
      continue;
    }
    if (line.back() == ',') {
      line.pop_back();
      line = trimBadDetectorLine(line);
    }
    const bool quoted =
        line.size() >= 2 && line.front() == '"' && line.back() == '"';
    const std::string entry =
        quoted ? line.substr(1, line.size() - 2) : line;
    const bool bareGlobalNumber =
        !entry.empty() &&
        std::all_of(entry.begin(), entry.end(), [](unsigned char ch) {
          return std::isdigit(ch) != 0;
        });
    if (!quoted && !bareGlobalNumber) {
      throw std::runtime_error(
          "bad detector entries must be global numbers or quoted hardware IDs: " +
          path + ":" + std::to_string(lineNumber));
    }
    const BadDetectorId id = parseBadDetectorEntry(entry);
    const auto key = std::make_tuple(id.link, id.fpga, id.detector);
    if (!seen.insert(key).second)
      throw std::runtime_error("duplicate bad detector identifier '" +
                               badDetectorName(id) + "': " + path);
    result.push_back(id);
  }
  if (!opened || !closed)
    throw std::runtime_error("bad detector list must be enclosed by '[' and ']': " +
                             path);
  return result;
}

inline std::vector<uint8_t> buildBadDetectorProjectionMask(
    const std::vector<BadDetectorId> &detectors, int fullAngles, int detNz,
    int detNx) {
  if (detectors.empty())
    return {};
  constexpr int kFpgaCount = 6;
  constexpr int kPhysicalAnglesPerFpga = 60;
  constexpr int kDetectorBlocksPerAxis = 4;
  constexpr int kBlockSize = 12;
  if (fullAngles != kFpgaCount * kPhysicalAnglesPerFpga || detNz != 48 ||
      detNx != 48) {
    throw std::runtime_error(
        "bad detector masking requires canonical 360-angle, 48x48 identity "
        "projection geometry");
  }

  const size_t raysPerAngle = static_cast<size_t>(detNz) * detNx;
  std::vector<uint8_t> mask(static_cast<size_t>(fullAngles) * raysPerAngle, 0);
  for (const BadDetectorId &id : detectors) {
    // The protected 6-module prototype assigns this hardware channel to a
    // fixed acquisition-space block that differs from the generic ENG map.
    const bool prototypeDetector68 =
        id.link == 4 && id.fpga == 2 && id.detector == 4;
    const int localDetector = (id.link == 4 ? 8 : 0) + id.detector;
    const int blockX =
        prototypeDetector68 ? 2 : localDetector % kDetectorBlocksPerAxis;
    const int blockZ =
        prototypeDetector68 ? 2 : localDetector / kDetectorBlocksPerAxis;
    const int angleBegin = prototypeDetector68
                               ? 0
                               : id.fpga * kPhysicalAnglesPerFpga;
    const int angleEnd = angleBegin + kPhysicalAnglesPerFpga;
    for (int angle = angleBegin; angle < angleEnd; ++angle) {
      const size_t angleOffset = static_cast<size_t>(angle) * raysPerAngle;
      for (int z = blockZ * kBlockSize; z < (blockZ + 1) * kBlockSize; ++z) {
        const size_t rowOffset = angleOffset + static_cast<size_t>(z) * detNx;
        std::fill(mask.begin() + static_cast<std::ptrdiff_t>(rowOffset +
                                                             blockX * kBlockSize),
                  mask.begin() + static_cast<std::ptrdiff_t>(rowOffset +
                                                             (blockX + 1) * kBlockSize),
                  static_cast<uint8_t>(1));
      }
    }
  }
  return mask;
}

} // namespace io
} // namespace spect
