#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <string_view>

namespace spect::core {

struct WaterAttenuationPreset {
  const char *key;
  const char *label;
  double energyKeV;
  double muPerMM;
};

// NIST XCOM liquid water, total mass attenuation coefficient with coherent
// scattering (cm^2/g); density 1.0 g/cm^3. The nuclide presets below are the
// log-log interpolation of this table at their photopeak energies (Tc-99m
// keeps the historical validated 0.0152 instead of the interpolated 0.01536).
struct WaterMassAttenuationPoint {
  double energyKeV;
  double muOverRhoCm2PerG;
};

inline const WaterMassAttenuationPoint *waterMassAttenuationTable(size_t *count) {
  static constexpr WaterMassAttenuationPoint table[] = {
      {20.0, 0.8096},  {30.0, 0.3756},  {40.0, 0.2683},  {50.0, 0.2269},
      {60.0, 0.2059},  {80.0, 0.1837},  {100.0, 0.1707}, {150.0, 0.1505},
      {200.0, 0.1370}, {300.0, 0.1186}, {400.0, 0.1061}, {500.0, 0.09687},
      {600.0, 0.08956}, {800.0, 0.07865}, {1000.0, 0.07072},
  };
  *count = sizeof(table) / sizeof(table[0]);
  return table;
}

// Water linear attenuation coefficient (mm^-1) at an arbitrary photon energy
// by log-log interpolation of the NIST table; false outside [20, 1000] keV.
inline bool waterMuPerMMAtEnergyKeV(double energyKeV, double &muPerMM) {
  size_t count = 0;
  const WaterMassAttenuationPoint *table = waterMassAttenuationTable(&count);
  if (!(energyKeV >= table[0].energyKeV) ||
      !(energyKeV <= table[count - 1].energyKeV)) {
    return false;
  }
  size_t hi = 1;
  while (hi + 1 < count && table[hi].energyKeV < energyKeV) {
    ++hi;
  }
  const WaterMassAttenuationPoint &a = table[hi - 1];
  const WaterMassAttenuationPoint &b = table[hi];
  const double t = std::log(energyKeV / a.energyKeV) /
                   std::log(b.energyKeV / a.energyKeV);
  const double muOverRho =
      a.muOverRhoCm2PerG *
      std::pow(b.muOverRhoCm2PerG / a.muOverRhoCm2PerG, t);
  muPerMM = muOverRho * 1.0 /* g/cm^3 */ / 10.0 /* cm -> mm */;
  return true;
}

// Ratio mu_water(E) / mu_water(E_ref): the exponent that turns a CTAC factor
// table computed at E_ref into the table at E under the single-material (water
// scaled) mu model used by the auto-mu and HU->mu pipelines
// (att(E) = att(E_ref)^ratio, since att = exp(-integral mu)).
inline bool waterMuEnergyRatio(double energyKeV, double referenceKeV,
                               double &ratio) {
  double mu = 0.0;
  double muRef = 0.0;
  if (!waterMuPerMMAtEnergyKeV(energyKeV, mu) ||
      !waterMuPerMMAtEnergyKeV(referenceKeV, muRef) || !(muRef > 0.0)) {
    return false;
  }
  ratio = mu / muRef;
  return true;
}

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
