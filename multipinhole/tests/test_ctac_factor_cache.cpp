#include <gtest/gtest.h>

#include "ctac/CTACFactorCache.h"
#include "cli/ReconArgs.h"
#include "io/MHDWriter.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>

namespace spect::main {
namespace {

class CTACFactorCacheTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir_ = std::filesystem::temp_directory_path() /
               ("spect_ctac_cache_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    std::filesystem::remove_all(testDir_);
    std::filesystem::create_directories(testDir_);
  }

  void TearDown() override {
    std::filesystem::remove_all(testDir_);
  }

  std::filesystem::path testDir_;
};

TEST_F(CTACFactorCacheTest, SaveAndLoadRoundTrip) {
  CTACFactorCache cache(testDir_);
  const std::string manifest = "cache_schema=ctac_factors_v1\ngeom=2,2,2\n";
  std::vector<uint16_t> factors = {1, 2, 3, 4, 5, 6, 7, 8};

  cache.save(manifest, factors, factors.size());

  auto loaded = cache.tryLoad(manifest);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded.value(), factors);
}

TEST_F(CTACFactorCacheTest, MissReturnsNullopt) {
  CTACFactorCache cache(testDir_);
  const std::string manifest = "cache_schema=ctac_factors_v1\ngeom=2,2,2\n";

  auto loaded = cache.tryLoad(manifest);
  EXPECT_FALSE(loaded.has_value());
}

TEST_F(CTACFactorCacheTest, DifferentManifestsAreIsolated) {
  CTACFactorCache cache(testDir_);
  const std::string manifestA = "cache_schema=ctac_factors_v1\nkey=A\n";
  const std::string manifestB = "cache_schema=ctac_factors_v1\nkey=B\n";
  std::vector<uint16_t> factorsA = {10, 20, 30};
  std::vector<uint16_t> factorsB = {40, 50, 60};

  cache.save(manifestA, factorsA, factorsA.size());
  cache.save(manifestB, factorsB, factorsB.size());

  auto loadedA = cache.tryLoad(manifestA);
  auto loadedB = cache.tryLoad(manifestB);

  ASSERT_TRUE(loadedA.has_value());
  ASSERT_TRUE(loadedB.has_value());
  EXPECT_EQ(loadedA.value(), factorsA);
  EXPECT_EQ(loadedB.value(), factorsB);
}

TEST_F(CTACFactorCacheTest, ManifestMismatchReturnsNullopt) {
  CTACFactorCache cache(testDir_);
  const std::string manifest = "cache_schema=ctac_factors_v1\ngeom=2,2,2\n";
  std::vector<uint16_t> factors = {1, 2, 3, 4};

  cache.save(manifest, factors, factors.size());

  auto loaded = cache.tryLoad(manifest + "extra");
  EXPECT_FALSE(loaded.has_value());
}

TEST_F(CTACFactorCacheTest, CorruptRawFileSizeReturnsNullopt) {
  CTACFactorCache cache(testDir_);
  const std::string manifest = "cache_schema=ctac_factors_v1\ngeom=2,2,2\n";
  std::vector<uint16_t> factors = {1, 2, 3, 4};

  cache.save(manifest, factors, factors.size());

  // Corrupt the raw file by appending a single byte so size is not a multiple
  // of sizeof(uint16_t). Discover the raw file by listing the cache directory.
  std::vector<std::filesystem::path> rawFiles;
  for (const auto &entry : std::filesystem::directory_iterator(testDir_)) {
    if (entry.path().extension() == ".raw")
      rawFiles.push_back(entry.path());
  }
  ASSERT_EQ(rawFiles.size(), 1u);
  {
    std::ofstream out(rawFiles[0], std::ios::app | std::ios::binary);
    char byte = 0;
    out.write(&byte, 1);
  }

  auto loaded = cache.tryLoad(manifest);
  EXPECT_FALSE(loaded.has_value());
}

TEST_F(CTACFactorCacheTest, SaveRejectsMismatchedCount) {
  CTACFactorCache cache(testDir_);
  const std::string manifest = "cache_schema=ctac_factors_v1\ngeom=2,2,2\n";
  std::vector<uint16_t> factors = {1, 2, 3, 4};

  EXPECT_THROW(cache.save(manifest, factors, 99u), std::runtime_error);
}

TEST_F(CTACFactorCacheTest, EmptyManifestNeverCaches) {
  CTACFactorCache cache(testDir_);
  std::vector<uint16_t> factors = {1, 2, 3};
  // Should not throw and should not write anything.
  cache.save("", factors, factors.size());
  auto loaded = cache.tryLoad("");
  EXPECT_FALSE(loaded.has_value());
  // Directory should still be empty (save with empty manifest is a no-op).
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(testDir_),
                          std::filesystem::directory_iterator{}),
            0);
}

TEST_F(CTACFactorCacheTest, SourcePayloadAndAutoMuParametersInvalidateFactors) {
  const auto stem = testDir_ / "mu";
  io::writeMHD(stem.string(), std::vector<double>(8, 0.0152),
               2, 2, 2, 0.4f, 0.4f, 0.4f);
  cli::ReconArgs args;
  args.attMapPath = stem.string() + ".mhd";
  const auto manifest = [&] {
    return buildCTACFactorManifest(args, 2, 2, 2, {0.4, 0.4, 0.4}, 0);
  };
  const auto headerTime = std::filesystem::last_write_time(args.attMapPath);
  const auto before = manifest();
  CTACFactorCache cache(testDir_ / "cache");
  cache.save(before, {1, 2}, 2);
  ASSERT_TRUE(cache.tryLoad(before));
  const auto raw = stem.string() + ".raw";
  const auto rawTime = std::filesystem::last_write_time(raw);
  {
    std::fstream out(raw, std::ios::in | std::ios::out | std::ios::binary);
    const double changed = 0.03;
    out.write(reinterpret_cast<const char *>(&changed), sizeof(changed));
  }
  std::filesystem::last_write_time(raw, rawTime + std::chrono::seconds(2));
  EXPECT_EQ(std::filesystem::last_write_time(args.attMapPath), headerTime);
  EXPECT_NE(before, manifest());
  EXPECT_FALSE(cache.tryLoad(manifest()));

  const auto changedTime = manifest();
  {
    std::ofstream out(raw, std::ios::binary | std::ios::app);
    out.put(0);
  }
  std::filesystem::last_write_time(raw, rawTime + std::chrono::seconds(2));
  EXPECT_NE(changedTime, manifest());

  args.ctacAutoMuFrom = args.attMapPath;
  args.attMapPath.clear();
  args.useCylindricalMask = false;
  const auto unmasked = manifest();
  args.useCylindricalMask = true;
  const auto masked = manifest();
  EXPECT_NE(unmasked, masked);
  args.cylinderDiameterMM += 1.0;
  EXPECT_NE(masked, manifest());
  const auto beforeThreshold = manifest();
  args.ctacAutoMuThresh *= 0.5;
  EXPECT_NE(beforeThreshold, manifest());
  std::filesystem::remove(raw);
  EXPECT_THROW(manifest(), std::filesystem::filesystem_error);

  const auto hdr = testDir_ / "mu.hdr";
  const auto img = testDir_ / "mu.img";
  {
    std::ofstream header(hdr);
    header << "!INTERFILE :=\nmatrix size [1] := 2\nmatrix size [2] := 2\n"
           << "matrix size [3] := 2\nname of data file := mu.img\n";
    std::ofstream payload(img, std::ios::binary);
    const float values[8] = {};
    payload.write(reinterpret_cast<const char *>(values), sizeof(values));
  }
  args.ctacAutoMuFrom.clear();
  args.attMapPath = hdr.string();
  const auto hdrManifest = manifest();
  std::filesystem::last_write_time(img,
      std::filesystem::last_write_time(img) + std::chrono::seconds(2));
  EXPECT_NE(hdrManifest, manifest());
}

} // namespace
} // namespace spect::main
