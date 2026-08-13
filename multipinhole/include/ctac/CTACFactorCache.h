#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace spect::main {

/**
 * @brief Disk cache for CTAC attenuation factors.
 *
 * The cache stores a uint16 factor table together with a manifest file that
 * uniquely identifies the geometry, mu-map source, and bed offset that the
 * factors were generated from. The manifest is provided by the caller so that
 * higher-level code can decide which parameters form the cache key.
 */
class CTACFactorCache {
public:
  explicit CTACFactorCache(std::filesystem::path cacheDir);

  /**
   * @brief Try to load a previously cached factor table.
   *
   * On miss or manifest mismatch returns std::nullopt. A mismatch is logged
   * through the existing logging system by the caller; this helper only checks
   * that the manifest and raw file exist and are consistent.
   */
  std::optional<std::vector<uint16_t>> tryLoad(
      const std::string &manifest) const;

  /**
   * @brief Save a factor table to disk, overwriting any existing entry.
   *
   * @param manifest cache key manifest
   * @param factors  uint16 factor table (65535 == 1.0)
   * @param expectedCount expected number of elements, used as a sanity check
   */
  void save(const std::string &manifest, const std::vector<uint16_t> &factors,
            size_t expectedCount) const;

  const std::filesystem::path &cacheDir() const { return cacheDir_; }

private:
  std::filesystem::path manifestPath(const std::string &manifest) const;
  std::filesystem::path factorsPath(const std::string &manifest) const;

  std::filesystem::path cacheDir_;
};

/**
 * @brief Compute attenuation factors with optional disk caching.
 *
 * If @p cache is non-null and @p manifest is non-empty, the function first
 * attempts to load previously cached factors. On miss it generates them via
 * computeAttenFactors() and writes them to the cache.
 *
 * This is a convenience wrapper; callers that need fine-grained control over
 * the cache key can use CTACFactorCache directly.
 */
std::vector<uint16_t> computeAttenFactorsCached(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, double dz, int numAngles, int rotationSign, int threads,
    const CTACFactorCache *cache, const std::string &manifest);

} // namespace spect::main
