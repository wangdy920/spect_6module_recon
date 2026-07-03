#include <gtest/gtest.h>

#include "main/ReconSetup.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spect {
namespace main {
namespace test {

namespace {

std::filesystem::path makeTempMhdPath() {
  const auto base = std::filesystem::temp_directory_path() / "spect_recon_setup_tests";
  std::filesystem::create_directories(base);
  const auto stamp = std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count();
  return base / ("dim_hint_" + std::to_string(stamp) + ".mhd");
}

void writeMinimalMhd(const std::filesystem::path &path, int nx, int ny, int nz) {
  std::ofstream out(path);
  ASSERT_TRUE(out.is_open());
  out << "ObjectType = Image\n";
  out << "NDims = 3\n";
  out << "DimSize = " << nx << ' ' << ny << ' ' << nz << "\n";
  out << "ElementSpacing = 1 1 1\n";
  out << "ElementType = MET_FLOAT\n";
  out << "ElementDataFile = dummy.raw\n";
}

} // namespace

TEST(ReconSetupTest, EnsureReconDimsUsesDefaultsWhenMissing) {
  ReconArgs args;
  args.Nx = 0;
  args.Ny = -1;
  args.Nz = 0;

  std::string warning;
  const bool changed = ensureReconDims(args, &warning);

  EXPECT_TRUE(changed);
  EXPECT_EQ(args.Nx, 200);
  EXPECT_EQ(args.Ny, 200);
  EXPECT_EQ(args.Nz, 200);
  EXPECT_FALSE(warning.empty());
}

TEST(ReconSetupTest, AdoptReconDimsFromCtacHintWhenNotExplicit) {
  const auto path = makeTempMhdPath();
  writeMinimalMhd(path, 64, 65, 66);

  ReconArgs args;
  args.doCTAC = true;
  args.attMapPath = path.string();

  std::string warning;
  const bool changed = adoptReconDimsFromHint(args, &warning);

  EXPECT_TRUE(changed);
  EXPECT_EQ(args.Nx, 64);
  EXPECT_EQ(args.Ny, 65);
  EXPECT_EQ(args.Nz, 66);
  EXPECT_NE(warning.find("adopting 64x65x66"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(ReconSetupTest, AdoptReconDimsFromHintSkipsExplicitDims) {
  const auto path = makeTempMhdPath();
  writeMinimalMhd(path, 64, 65, 66);

  ReconArgs args;
  args.dimExplicit = true;
  args.Nx = 100;
  args.Ny = 101;
  args.Nz = 102;
  args.attMapPath = path.string();
  args.doCTAC = true;

  std::string warning;
  const bool changed = adoptReconDimsFromHint(args, &warning);

  EXPECT_FALSE(changed);
  EXPECT_EQ(args.Nx, 100);
  EXPECT_EQ(args.Ny, 101);
  EXPECT_EQ(args.Nz, 102);
  EXPECT_TRUE(warning.empty());

  std::filesystem::remove(path);
}

TEST(ReconSetupTest, PrepareBedSetupReplicatesSingleOverlap) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat", "bed4.mat"};
  args.bedOverlaps = {16};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(result.baseNz, 100);
  EXPECT_EQ(result.bedCount, 4);
  EXPECT_EQ(args.bedCount, 4);
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{16, 16, 16}));
  EXPECT_EQ(result.totalOverlap, 48);
  EXPECT_EQ(args.Nz, 352);
  EXPECT_TRUE(result.warnings.empty());
}

TEST(ReconSetupTest, PrepareBedSetupUsesFixedFiveBedOverlapTable) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat", "bed4.mat"};
  args.bedOverlaps = {10, 20, 30, 40, 50};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{10, 20, 30}));
  EXPECT_EQ(result.totalOverlap, 60);
  EXPECT_EQ(args.Nz, 340);
  EXPECT_TRUE(result.warnings.empty());
}

TEST(ReconSetupTest, PrepareBedSetupPadsMismatchedOverlapList) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat", "bed4.mat"};
  args.bedOverlaps = {16, 16};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{16, 16, 0}));
  ASSERT_EQ(result.warnings.size(), 1u);
  EXPECT_NE(result.warnings[0].find("Missing fixed slots use zero"),
            std::string::npos);
  EXPECT_EQ(args.Nz, 368);
}

TEST(ReconSetupTest, PrepareBedSetupClampsInvalidOverlapValues) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat", "bed4.mat"};
  args.bedOverlaps = {-4, 120, 16};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0, 99, 16}));
  EXPECT_EQ(result.totalOverlap, 115);
  EXPECT_EQ(args.Nz, 285);
  ASSERT_GE(result.warnings.size(), 1u);
  bool sawClamped = false;
  for (const std::string &warning : result.warnings) {
    sawClamped = sawClamped || warning.find("clamped") != std::string::npos;
  }
  EXPECT_TRUE(sawClamped);
}

TEST(ReconSetupTest, PrepareBedSetupRejectsMoreThanFiveBeds) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat",
                    "bed4.mat", "bed5.mat", "bed6.mat"};

  EXPECT_THROW((void)prepareBedSetup(args), std::runtime_error);
}

TEST(ReconSetupTest, PrepareBedSetupMatchesFixedOverlapTableForSubsetBeds) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"case_Bed2.bin", "case_Bed3.bin", "case_Bed4.bin"};
  args.bedOverlaps = {11, 22, 33, 44, 55};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{22, 33}));
  EXPECT_EQ(result.totalOverlap, 55);
  EXPECT_EQ(args.Nz, 245);
  EXPECT_TRUE(result.warnings.empty());
}

TEST(ReconSetupTest, PrepareBedSetupUsesSequentialBedsWhenPathHasNoBedNumber) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"first.bin", "second.bin", "third.bin"};
  args.bedOverlaps = {5, 6, 7, 8, 9};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{5, 6}));
  EXPECT_EQ(result.totalOverlap, 11);
  EXPECT_EQ(args.Nz, 289);
}

TEST(ReconSetupTest, PrepareBedSetupUsesZeroForNonAdjacentSubsetBeds) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"case_Bed1.bin", "case_Bed3.bin"};
  args.bedOverlaps = {5, 6, 7, 8, 9};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0}));
  EXPECT_EQ(result.totalOverlap, 0);
  EXPECT_EQ(args.Nz, 200);
  ASSERT_EQ(result.warnings.size(), 1u);
  EXPECT_NE(result.warnings[0].find("Non-adjacent beds"), std::string::npos);
}

} // namespace test
} // namespace main
} // namespace spect
