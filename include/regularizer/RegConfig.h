#pragma once

#include <string>
#include <vector>

namespace spect {

enum class RegType { None, TVIsotropic, RDP, Unknown };

inline RegType stringToRegType(const std::string &str) {
  if (str == "None" || str == "none" || str.empty())
    return RegType::None;
  if (str == "TVIsotropic" || str == "tv" || str == "TV")
    return RegType::TVIsotropic;
  if (str == "RDP" || str == "rdp")
    return RegType::RDP;
  return RegType::Unknown;
}

inline std::string regTypeToString(RegType type) {
  switch (type) {
  case RegType::None:
    return "None";
  case RegType::TVIsotropic:
    return "TVIsotropic";
  case RegType::RDP:
    return "RDP";
  default:
    return "Unknown";
  }
}

struct RegConf {
  RegType type = RegType::None;
  double beta = 0.0;
  double tv_eps = 1e-6;
  double delta = 0.01;
  double gamma = 0.0;
  bool skip_outside_mask = true;

  // Optional registered CT/mu-map. A fixed bilateral edge weight is applied
  // to TV/RDP; nullptr preserves the activity-only equations exactly.
  const std::vector<double> *ct_image = nullptr;
  double ct_sigma = 0.01;
  double ct_weight_floor = 0.05;
};

} // namespace spect
