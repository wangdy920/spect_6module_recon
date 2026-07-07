#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace spect::core {

struct WaterAttenuationPreset {
  const char *key;
  const char *label;
  double energyKeV;
  double muPerMM;
};

inline std::string normalizeNuclideKey(std::string_view raw) {
  std::string out;
  out.reserve(raw.size());
  for (const char ch : raw) {
    const auto c = static_cast<unsigned char>(ch);
    if (std::isalnum(c))
      out.push_back(static_cast<char>(std::tolower(c)));
  }
  return out;
}

inline const WaterAttenuationPreset *waterAttenuationPreset(
    std::string_view raw) {
  const std::string key = normalizeNuclideKey(raw);
  // Values are water linear attenuation coefficients in mm^-1. Tc99m keeps the
  // historical validated default; the other entries are NIST XCOM liquid-water
  // log-log interpolations at common SPECT photopeak energies.
  static constexpr WaterAttenuationPreset presets[] = {
      {"tc99m", "Tc-99m 140.5 keV water", 140.5, 0.0152},
      {"i123", "I-123 159.0 keV water", 159.0, 0.014766221821206932},
      {"in111171", "In-111 171.3 keV water", 171.3, 0.014411118865640875},
      {"in111245", "In-111 245.4 keV water", 245.4, 0.012738504550836008},
      {"lu177", "Lu-177 208.4 keV water", 208.4, 0.013500970784546906},
      {"i131", "I-131 364.5 keV water", 364.5, 0.010998703003107506},
  };
  for (const auto &preset : presets) {
    if (key == preset.key)
      return &preset;
  }
  return nullptr;
}

inline bool waterMuPerMMForNuclide(std::string_view nuclide,
                                   double &muPerMM) {
  const WaterAttenuationPreset *preset = waterAttenuationPreset(nuclide);
  if (!preset)
    return false;
  muPerMM = preset->muPerMM;
  return true;
}

inline const char *supportedWaterMuNuclides() {
  return "Tc99m, I123, In111_171, In111_245, Lu177, I131";
}

} // namespace spect::core
