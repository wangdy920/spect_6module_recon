#include "ctac/CTACFactorCache.h"

#include "cli/ReconArgs.h"
#include "io/VolumeLoader.h"
#include "pipeline/RunLogger.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace spect::main {

namespace {

uint64_t fnv1a64(const std::string &text) {
  uint64_t value = 14695981039346656037ull;
  for (const unsigned char ch : text) {
    value ^= static_cast<uint64_t>(ch);
    value *= 1099511628211ull;
  }
  return value;
}

std::filesystem::path normalizePathForCache(const std::filesystem::path &path) {
  std::error_code ec;
  auto canonical = std::filesystem::weakly_canonical(path, ec);
  if (ec)
    return path;
  return canonical;
}

void appendVolumeIdentity(std::ostringstream &out, const std::string &label,
                          const std::string &path) {
  const auto append = [&](const std::string &key, const std::filesystem::path &file) {
    const auto fullPath = std::filesystem::canonical(file);
    out << key << '=' << fullPath.generic_string() << '\n'
        << key << "_size=" << std::filesystem::file_size(fullPath) << '\n'
        << key << "_mtime="
        << std::filesystem::last_write_time(fullPath).time_since_epoch().count() << '\n';
  };
  append(label, path);
  const auto info = io::readVolumeInfo(path);
  append(label + "_data", info.rawPath);
}

} // namespace

std::string buildCTACFactorManifest(const cli::ReconArgs &args, int baseNx,
                                    int baseNy, int baseNz,
                                    const filters::Spacing3 &spacing,
                                    int bedZStart) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "cache_schema=ctac_atten_factors_v2_source_data\n"
      << "ctac_model=" << args.ctacModel << '\n'
      << "nx=" << baseNx << "\nny=" << baseNy << "\nnz=" << baseNz << '\n'
      << "dx=" << spacing.dx << "\ndy=" << spacing.dy << "\ndz=" << spacing.dz << '\n'
      << "num_angles=" << args.numAngles << '\n'
      << "rotation_sign=" << args.ctacRotSign << '\n'
      << "map_z_layout=local_matrix_z\n"
      << "z_bed_flip=" << (args.zBedFlip ? 1 : 0) << '\n'
      << "bed_z_start=" << bedZStart << '\n';
  if (!args.attMapPath.empty())
    appendVolumeIdentity(out, "att_map", args.attMapPath);
  if (!args.ctacAutoMuFrom.empty()) {
    appendVolumeIdentity(out, "auto_mu_from", args.ctacAutoMuFrom);
    out << "auto_mu_thresh=" << args.ctacAutoMuThresh << '\n'
        << "auto_mu_value=" << args.ctacAutoMuValue << '\n'
        << "auto_mu_cylinder=" << args.useCylindricalMask << '\n';
    if (args.useCylindricalMask)
      out << "cylinder_diameter_mm=" << args.cylinderDiameterMM << '\n'
          << "cylinder_height_mm=" << args.cylinderHeightMM << '\n'
          << "cylinder_center=" << args.cylCenterX << ',' << args.cylCenterY
          << ',' << args.cylCenterZ << '\n';
  }
  return out.str();
}

CTACFactorCache::CTACFactorCache(std::filesystem::path cacheDir)
    : cacheDir_(normalizePathForCache(std::move(cacheDir))) {}

std::filesystem::path CTACFactorCache::manifestPath(
    const std::string &manifest) const {
  std::ostringstream name;
  name << "auto_" << std::hex << fnv1a64(manifest) << ".manifest.txt";
  return cacheDir_ / name.str();
}

std::filesystem::path CTACFactorCache::factorsPath(
    const std::string &manifest) const {
  std::ostringstream name;
  name << "auto_" << std::hex << fnv1a64(manifest) << ".raw";
  return cacheDir_ / name.str();
}

std::optional<std::vector<uint16_t>> CTACFactorCache::tryLoad(
    const std::string &manifest) const {
  if (manifest.empty())
    return std::nullopt;

  const auto manifestFile = manifestPath(manifest);
  const auto rawFile = factorsPath(manifest);

  std::error_code ec;
  if (!std::filesystem::exists(manifestFile, ec) || ec) {
    logDebug("CTAC factor cache miss: manifest not found for " +
             manifestFile.string());
    return std::nullopt;
  }
  if (!std::filesystem::exists(rawFile, ec) || ec) {
    logDebug("CTAC factor cache miss: raw factors not found for " +
             rawFile.string());
    return std::nullopt;
  }

  std::ifstream inManifest(manifestFile, std::ios::binary);
  if (!inManifest) {
    logWarn("CTAC factor cache manifest unreadable: " + manifestFile.string());
    return std::nullopt;
  }
  std::string cachedManifest((std::istreambuf_iterator<char>(inManifest)),
                             std::istreambuf_iterator<char>());
  if (cachedManifest != manifest) {
    logWarn("CTAC factor cache manifest mismatch for " + manifestFile.string());
    return std::nullopt;
  }

  const auto rawSize = std::filesystem::file_size(rawFile, ec);
  if (ec) {
    logWarn("CTAC factor cache raw file size unavailable: " + rawFile.string());
    return std::nullopt;
  }
  if (rawSize % sizeof(uint16_t) != 0) {
    logWarn("CTAC factor cache raw file size is not a multiple of uint16: " +
            rawFile.string());
    return std::nullopt;
  }

  std::vector<uint16_t> factors(rawSize / sizeof(uint16_t));
  std::ifstream inRaw(rawFile, std::ios::binary);
  if (!inRaw) {
    logWarn("CTAC factor cache raw file unreadable: " + rawFile.string());
    return std::nullopt;
  }
  inRaw.read(reinterpret_cast<char *>(factors.data()),
             static_cast<std::streamsize>(rawSize));
  if (!inRaw) {
    logWarn("CTAC factor cache raw file incomplete read: " + rawFile.string());
    return std::nullopt;
  }

  logInfo("CTAC factor cache hit: " + rawFile.string());
  return factors;
}

void CTACFactorCache::save(const std::string &manifest,
                           const std::vector<uint16_t> &factors,
                           size_t expectedCount) const {
  if (manifest.empty())
    return;
  if (factors.size() != expectedCount) {
    throw std::runtime_error("CTACFactorCache::save: factor count mismatch " +
                             std::to_string(factors.size()) + " vs " +
                             std::to_string(expectedCount));
  }

  std::error_code ec;
  std::filesystem::create_directories(cacheDir_, ec);
  if (ec) {
    logWarn("Failed to create CTAC factor cache directory " +
            cacheDir_.string() + ": " + ec.message());
    return;
  }

  const auto manifestFile = manifestPath(manifest);
  const auto rawFile = factorsPath(manifest);

  {
    std::ofstream outManifest(manifestFile, std::ios::binary);
    if (!outManifest) {
      logWarn("Failed to write CTAC factor cache manifest " +
              manifestFile.string());
      return;
    }
    outManifest.write(manifest.data(),
                      static_cast<std::streamsize>(manifest.size()));
    outManifest.flush();
    if (!outManifest) {
      logWarn("Failed to flush CTAC factor cache manifest " +
              manifestFile.string());
      return;
    }
  }

  {
    std::ofstream outRaw(rawFile, std::ios::binary);
    if (!outRaw) {
      logWarn("Failed to write CTAC factor cache raw " + rawFile.string());
      return;
    }
    outRaw.write(reinterpret_cast<const char *>(factors.data()),
                 static_cast<std::streamsize>(factors.size() * sizeof(uint16_t)));
    outRaw.flush();
    if (!outRaw) {
      logWarn("Failed to flush CTAC factor cache raw " + rawFile.string());
      return;
    }
  }

  logInfo("CTAC factor cache saved: " + rawFile.string());
}

} // namespace spect::main
