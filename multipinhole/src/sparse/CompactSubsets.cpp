#include "sparse/CompactSubsets.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace spect {

std::vector<int> compactSubsetAngles(int compactNumAngles,
                                     const CompactSubsetSpec &spec) {
  if (compactNumAngles <= 0 || spec.count <= 0 || spec.index < 0 ||
      spec.index >= spec.count || compactNumAngles % spec.count != 0) {
    throw std::runtime_error(
        "compact subset spec must have 0 <= index < count and count dividing "
        "the compact angle count (" +
        std::to_string(compactNumAngles) + ").");
  }
  std::vector<int> angles;
  angles.reserve(static_cast<size_t>(compactNumAngles / spec.count));
  for (int angle = spec.index; angle < compactNumAngles; angle += spec.count) {
    angles.push_back(angle);
  }
  return angles;
}

std::vector<int> compactSubsetOrder(int count) {
  if (count <= 0) {
    throw std::runtime_error("compact subset count must be positive.");
  }
  auto circular = [count](int a, int b) {
    const int d = std::abs(a - b);
    return std::min(d, count - d);
  };
  std::vector<int> order;
  order.reserve(static_cast<size_t>(count));
  std::vector<bool> used(static_cast<size_t>(count), false);
  order.push_back(0);
  used[0] = true;
  while (static_cast<int>(order.size()) < count) {
    // Rank a candidate by (min distance to every chosen subset, then the
    // distances to the most recent choices, newest first). The first key
    // spreads the subsets; the tie-break keeps consecutive sub-iterations
    // apart once every gap has closed to the same width (N = 10:
    // 0 5 2 7 1 6 3 8 4 9 instead of ... 1 3 4 6 8 9).
    auto rank = [&](int candidate) {
      std::vector<int> key;
      key.reserve(order.size() + 1);
      int minDistance = count;
      for (const int chosen : order) {
        minDistance = std::min(minDistance, circular(candidate, chosen));
      }
      key.push_back(minDistance);
      for (auto it = order.rbegin(); it != order.rend(); ++it) {
        key.push_back(circular(candidate, *it));
      }
      return key;
    };
    int best = -1;
    std::vector<int> bestKey;
    for (int candidate = 0; candidate < count; ++candidate) {
      if (used[static_cast<size_t>(candidate)]) {
        continue;
      }
      std::vector<int> key = rank(candidate);
      if (best < 0 || key > bestKey) {
        best = candidate;
        bestKey = std::move(key);
      }
    }
    order.push_back(best);
    used[static_cast<size_t>(best)] = true;
  }
  return order;
}

std::vector<CompactSubsetStage>
parseCompactSubsetSchedule(const std::string &text, int numIters,
                           int compactNumAngles, CompactSubsetMode mode) {
  std::vector<CompactSubsetStage> schedule;
  if (numIters <= 0) {
    throw std::runtime_error("subset schedule needs a positive iteration count.");
  }
  auto parseCount = [&](const std::string &token) {
    size_t consumed = 0;
    int value = 0;
    try {
      value = std::stoi(token, &consumed);
    } catch (const std::exception &) {
      consumed = 0;
    }
    if (consumed != token.size() || value <= 0) {
      throw std::runtime_error("invalid subset count '" + token +
                               "' in -subsets " + text);
    }
    if (value > kMaxCompactSubsetCount) {
      throw std::runtime_error(
          "-subsets count '" + token + "' exceeds the low-count cap " +
          std::to_string(kMaxCompactSubsetCount) +
          " (Lu-177 runs 74-90 % zero bins; thinner subsets destabilise).");
    }
    if (mode == CompactSubsetMode::Angles && compactNumAngles > 0 &&
        compactNumAngles % value != 0) {
      throw std::runtime_error(
          "-subsets " + token + " must divide the compact angle count " +
          std::to_string(compactNumAngles) + ".");
    }
    return value;
  };
  if (text.find(':') == std::string::npos) {
    schedule.push_back(CompactSubsetStage{parseCount(text), numIters});
    return schedule;
  }
  std::stringstream stream(text);
  std::string stage;
  int total = 0;
  while (std::getline(stream, stage, ',')) {
    const size_t colon = stage.find(':');
    if (colon == std::string::npos) {
      throw std::runtime_error("subset schedule stage '" + stage +
                               "' must be subsets:iterations.");
    }
    CompactSubsetStage parsed;
    parsed.subsets = parseCount(stage.substr(0, colon));
    const std::string iterationsText = stage.substr(colon + 1);
    size_t consumed = 0;
    try {
      parsed.iterations = std::stoi(iterationsText, &consumed);
    } catch (const std::exception &) {
      consumed = 0;
    }
    if (consumed != iterationsText.size() || parsed.iterations <= 0) {
      throw std::runtime_error("invalid iteration count in subset stage '" +
                               stage + "'.");
    }
    if (!schedule.empty() && schedule.back().subsets % parsed.subsets != 0) {
      throw std::runtime_error(
          "subset schedule must refine coarsely: " +
          std::to_string(parsed.subsets) + " does not divide " +
          std::to_string(schedule.back().subsets) + ".");
    }
    total += parsed.iterations;
    schedule.push_back(parsed);
  }
  if (schedule.empty()) {
    throw std::runtime_error("empty subset schedule.");
  }
  if (total != numIters) {
    throw std::runtime_error(
        "subset schedule iterations (" + std::to_string(total) +
        ") must sum to -it " + std::to_string(numIters) + ".");
  }
  return schedule;
}

std::string formatCompactSubsetSchedule(
    const std::vector<CompactSubsetStage> &schedule) {
  std::string out;
  for (const auto &stage : schedule) {
    if (!out.empty()) {
      out += ',';
    }
    out += std::to_string(stage.subsets) + ':' + std::to_string(stage.iterations);
  }
  return out;
}

CompactSubsetMode parseCompactSubsetMode(const std::string &text) {
  std::string lowered;
  lowered.reserve(text.size());
  for (const char c : text) {
    lowered.push_back(static_cast<char>(std::tolower(
        static_cast<unsigned char>(c))));
  }
  if (lowered == "angle" || lowered == "angles") {
    return CompactSubsetMode::Angles;
  }
  if (lowered == "pixel" || lowered == "pixels" || lowered == "posem") {
    return CompactSubsetMode::Pixels;
  }
  throw std::runtime_error(
      "invalid subset mode '" + text +
      "' (expected angle|pixels|posem for -subset-mode).");
}

std::string formatCompactSubsetMode(CompactSubsetMode mode) {
  return mode == CompactSubsetMode::Pixels ? "pixels" : "angles";
}

} // namespace spect
