#include "main/CTACFactorCache.h"

#include "core/AttenuationConstants.h"
#include "main/CTACOps.h"
#include "main/RunLogger.h"

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

} // namespace

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
  }

  {
    std::ofstream outRaw(rawFile, std::ios::binary);
    if (!outRaw) {
      logWarn("Failed to write CTAC factor cache raw " + rawFile.string());
      return;
    }
    outRaw.write(reinterpret_cast<const char *>(factors.data()),
                 static_cast<std::streamsize>(factors.size() * sizeof(uint16_t)));
  }

  logInfo("CTAC factor cache saved: " + rawFile.string());
}

std::vector<uint16_t> computeAttenFactorsCached(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, double dz, int numAngles, int rotationSign, int threads,
    const CTACFactorCache *cache, const std::string &manifest) {
  const size_t expectedCount =
      static_cast<size_t>(numAngles) * static_cast<size_t>(nx) * ny * nz;

  if (cache != nullptr && !manifest.empty()) {
    auto cached = cache->tryLoad(manifest);
    if (cached.has_value()) {
      if (cached->size() == expectedCount) {
        return std::move(*cached);
      }
      logWarn("CTAC factor cache hit size mismatch: " +
              std::to_string(cached->size()) + " vs " +
              std::to_string(expectedCount) + ", regenerating.");
    }
  }

  std::vector<uint16_t> factors = computeAttenFactors(
      muMap, nx, ny, nz, dx, dy, dz, numAngles, rotationSign, threads);

  if (cache != nullptr && !manifest.empty()) {
    cache->save(manifest, factors, expectedCount);
  }

  return factors;
}

} // namespace spect::main
