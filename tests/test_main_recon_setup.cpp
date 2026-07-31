#include <gtest/gtest.h>

#include "main/ReconSetup.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/BedPathUtils.h"

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

TEST(ReconSetupTest, EnsureReconDimsRejectsMatrixInputWithoutGeometry) {
  ReconArgs args;
  args.Nx = 0;
  args.Ny = 0;
  args.Nz = 0;
  args.chunkCSRDir = "matrix_chunks";

  EXPECT_THROW((void)ensureReconDims(args), std::runtime_error);
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

TEST(ReconSetupTest, AdoptReconDimsFromRegisteredCtHintWhenNotExplicit) {
  const auto path = makeTempMhdPath();
  writeMinimalMhd(path, 54, 55, 56);

  ReconArgs args;
  args.useReg = true;
  args.regCtPath = path.string();

  std::string warning;
  const bool changed = adoptReconDimsFromHint(args, &warning);

  EXPECT_TRUE(changed);
  EXPECT_EQ(args.Nx, 54);
  EXPECT_EQ(args.Ny, 55);
  EXPECT_EQ(args.Nz, 56);
  EXPECT_NE(warning.find("adopting 54x55x56"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(ReconSetupTest, PrepareBedSetupCanReserveOverlapForOutputOnly) {
  ReconArgs args;
  args.Nx = 108;
  args.Ny = 108;
  args.Nz = 80;
  args.projPaths = {"Bed1.bin", "Bed2.bin", "Bed3.bin", "Bed4.bin"};
  args.bedOverlaps = {8, 8, 8, 0, 0};
  args.bedCropMethod = "sigmoid";

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{8, 8, 8}));
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0, 0, 0}));
  EXPECT_EQ(result.totalOverlap, 0);
  EXPECT_EQ(args.Nz, 320);
  EXPECT_NE(result.summary.find("overlapMode=output-sigmoid"),
            std::string::npos);
}

TEST(ReconSetupTest, PrepareBedSetupReplicatesSingleOverlap) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat", "bed4.mat"};
  args.bedOverlaps = {16};
  args.bedCropMethod = "sigmoid";

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(result.baseNz, 100);
  EXPECT_EQ(result.bedCount, 4);
  EXPECT_EQ(args.bedCount, 4);
  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{16, 16, 16}));
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0, 0, 0}));
  EXPECT_EQ(result.totalOverlap, 0);
  EXPECT_EQ(args.Nz, 400);
  EXPECT_TRUE(result.warnings.empty());
}

TEST(ReconSetupTest, PrepareBedSetupPadsMismatchedOverlapList) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat", "bed4.mat"};
  args.bedOverlaps = {16, 16};
  args.bedCropMethod = "sigmoid";

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{16, 16, 0}));
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0, 0, 0}));
  ASSERT_EQ(result.warnings.size(), 1u);
  EXPECT_NE(result.warnings[0].find("Missing fixed slots use zero"),
            std::string::npos);
  EXPECT_EQ(args.Nz, 400);
}

TEST(ReconSetupTest, PrepareBedSetupClampsInvalidOverlapValues) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat", "bed4.mat"};
  args.bedOverlaps = {-4, 120, 16, 0, 0};
  args.bedCropMethod = "sigmoid";

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{0, 99, 16}));
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0, 0, 0}));
  EXPECT_EQ(result.totalOverlap, 0);
  EXPECT_EQ(args.Nz, 400);
  ASSERT_GE(result.warnings.size(), 1u);
  bool sawClamped = false;
  for (const std::string &warning : result.warnings) {
    sawClamped = sawClamped || warning.find("clamped") != std::string::npos;
  }
  EXPECT_TRUE(sawClamped);
}

TEST(ReconSetupTest, PrepareBedSetupMatchesFixedOverlapTableForSubsetBeds) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"case_Bed2.bin", "case_Bed3.bin", "case_Bed4.bin"};
  args.bedOverlaps = {11, 22, 33, 44, 55};
  args.bedCropMethod = "sigmoid";

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{22, 33}));
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0, 0}));
  EXPECT_EQ(result.totalOverlap, 0);
  EXPECT_EQ(args.Nz, 300);
  EXPECT_TRUE(result.warnings.empty());
}

TEST(ReconSetupTest, PrepareBedSetupUsesSequentialBedsWhenPathHasNoBedNumber) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"first.bin", "second.bin", "third.bin"};
  args.bedOverlaps = {5, 6, 7, 8, 9};
  args.bedCropMethod = "sigmoid";

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{5, 6}));
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0, 0}));
  EXPECT_EQ(args.Nz, 300);
}

TEST(ReconSetupTest, PrepareBedSetupUsesZeroForNonAdjacentSubsetBeds) {
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.projPaths = {"case_Bed1.bin", "case_Bed3.bin"};
  args.bedOverlaps = {5, 6, 7, 8, 9};
  args.bedCropMethod = "sigmoid";

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{0}));
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0}));
  EXPECT_EQ(result.totalOverlap, 0);
  EXPECT_EQ(args.Nz, 200);
  ASSERT_EQ(result.warnings.size(), 1u);
  EXPECT_NE(result.warnings[0].find("Non-adjacent beds"), std::string::npos);
}

TEST(ReconSetupTest, PrepareBedSetupAcceptsCenter66FixedStepGeometry) {
  ReconArgs args;
  args.Nx = 108;
  args.Ny = 108;
  args.Nz = 160;
  args.projPaths = {"Bed1.bin", "Bed2.bin", "Bed3.bin"};
  args.bedOverlaps = {104, 104, 104, 104, 0};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(result.baseNz, 160);
  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{104, 104}));
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0, 0}));
  EXPECT_NE(result.summary.find("overlapMode=output-center66_cosine10"),
            std::string::npos);
}

TEST(ReconSetupTest, PrepareBedSetupRejectsCenter66WrongBedStep) {
  ReconArgs args;
  args.Nx = 108;
  args.Ny = 108;
  args.Nz = 160;
  args.projPaths = {"Bed1.bin", "Bed2.bin"};
  args.bedOverlaps = {103, 103, 103, 103, 0};

  EXPECT_THROW((void)prepareBedSetup(args), std::runtime_error);
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

TEST(ReconSetupTest, SharedBedPathParserHandlesSeparatorsAndCase) {
  EXPECT_EQ(spect::core::extractBedNumberFromPath("C:/Scan/BED_12.bin"), 12);
  EXPECT_EQ(spect::core::extractBedNumberFromPath("case/bed-3.bin"), 3);
  EXPECT_EQ(spect::core::extractBedNumberFromPath("case/no_bed.bin"), -1);

  std::vector<std::string> paths = {"C:\\scan\\Bed3.bin", "C:\\scan\\Bed1.bin",
                                    "C:\\scan\\Bed2.bin", "tail.bin"};
  paths = spect::core::sortPathsByBedOrder(std::move(paths));
  EXPECT_NE(paths[0].find("Bed1.bin"), std::string::npos);
  EXPECT_NE(paths[1].find("Bed2.bin"), std::string::npos);
  EXPECT_NE(paths[2].find("Bed3.bin"), std::string::npos);
  EXPECT_NE(paths[3].find("tail.bin"), std::string::npos);
}

} // namespace test
} // namespace main
} // namespace spect
