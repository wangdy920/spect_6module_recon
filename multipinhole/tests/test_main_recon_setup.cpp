#include <gtest/gtest.h>

#include "pipeline/ReconSetup.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/BedPathUtils.h"
#include "core/ReconLimits.h"
#include "recon/BedOffsets.h"

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



TEST(ReconSetupTest, PrepareBedSetupUsesFixedSingleBedRoi) {
  ReconArgs args;
  args.Nx = 108;
  args.Ny = 108;
  args.Nz = 80;
  args.dz = 0.3;
  args.projPath = "Bed1.bin";

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(result.bedCount, 1);
  EXPECT_FLOAT_EQ(args.cylinderDiameterMM, 30.0f);
  EXPECT_FLOAT_EQ(args.cylinderHeightMM, 21.6f);
  EXPECT_EQ(result.canonicalGeom.stepLayers, 56);
  EXPECT_EQ(result.canonicalGeom.cropLayers, 66);
  EXPECT_EQ(result.canonicalGeom.blendOverlap, 10);
}

TEST(ReconSetupTest, PrepareBedSetupCanKeepLegacySingleBedCrop) {
  ReconArgs args;
  args.Nx = 120;
  args.Ny = 120;
  args.Nz = 220;
  args.dz = 0.3;
  args.projPath = "Bed1.bin";
  args.singleBedStepAligned = false;

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(result.bedCount, 1);
  EXPECT_EQ(result.canonicalGeom.cropLayers, 0);
  EXPECT_NE(result.summary.find("legacy 72-layer"), std::string::npos);
}

TEST(ReconSetupTest, PrepareBedSetupUsesConfiguredSingleBedRoi) {
  ReconArgs args;
  args.Nx = 120;
  args.Ny = 120;
  args.Nz = 220;
  args.dz = 0.3;
  args.projPath = "Bed1.bin";
  args.cylinderDimsSingleConfigured = true;
  args.cylinderDiameterSingleMM = 28.0f;
  args.cylinderHeightSingleMM = 20.0f;
  args.cylinderDimsMultiConfigured = true;
  args.cylinderDiameterMultiMM = 40.0f;
  args.cylinderHeightMultiMM = 60.0f;

  (void)prepareBedSetup(args);

  EXPECT_FLOAT_EQ(args.cylinderDiameterMM, 28.0f);
  EXPECT_FLOAT_EQ(args.cylinderHeightMM, 20.0f);
}

TEST(ReconSetupTest, PrepareBedSetupUsesConfiguredMultiBedRoi) {
  ReconArgs args;
  args.Nx = 120;
  args.Ny = 120;
  args.Nz = 220;
  args.dz = 0.3;
  args.projPaths = {"Bed1.bin", "Bed2.bin"};
  args.cylinderDimsSingleConfigured = true;
  args.cylinderDiameterSingleMM = 28.0f;
  args.cylinderHeightSingleMM = 20.0f;
  args.cylinderDimsMultiConfigured = true;
  args.cylinderDiameterMultiMM = 40.0f;
  args.cylinderHeightMultiMM = 60.0f;

  (void)prepareBedSetup(args);

  EXPECT_FLOAT_EQ(args.cylinderDiameterMM, 40.0f);
  EXPECT_FLOAT_EQ(args.cylinderHeightMM, 60.0f);
}

TEST(ReconSetupTest, PrepareBedSetupPreservesExplicitCylinderRoi) {
  ReconArgs args;
  args.Nx = 120;
  args.Ny = 120;
  args.Nz = 220;
  args.dz = 0.3;
  args.projPaths = {"Bed1.bin", "Bed2.bin"};
  args.cylinderDiameterMM = 28.0f;
  args.cylinderHeightMM = 24.0f;
  args.cylinderDimsExplicit = true;

  (void)prepareBedSetup(args);

  EXPECT_FLOAT_EQ(args.cylinderDiameterMM, 28.0f);
  EXPECT_FLOAT_EQ(args.cylinderHeightMM, 24.0f);
}
TEST(ReconSetupTest, PrepareBedSetupAcceptsCenter66FixedStepGeometry) {
  ReconArgs args;
  args.Nx = 108;
  args.Ny = 108;
  args.Nz = 160;
  args.Vz = 0.3f;
  args.dz = 0.3;
  args.projPaths = {"Bed1.bin", "Bed2.bin", "Bed3.bin"};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(result.baseNz, 160);
  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{104, 104}));
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{0, 0}));
  EXPECT_FLOAT_EQ(args.cylinderDiameterMM, 36.0f);
  EXPECT_FLOAT_EQ(args.cylinderHeightMM, 66.0f);
  EXPECT_NE(result.summary.find("overlapMode=output-center66_cosine10"),
            std::string::npos);
}

TEST(ReconSetupTest, PrepareBedSetupDerivesOverlapFromBedStep) {
  ReconArgs args;
  args.Nx = 108;
  args.Ny = 108;
  args.Nz = 160;
  args.dz = 0.3;
  args.projPaths = {"Bed1.bin", "Bed2.bin", "Bed3.bin"};

  const BedSetupResult result = prepareBedSetup(args);

  // Default 16.8 mm step -> 56 layers, overlap = 160 - 56 = 104 per joint.
  EXPECT_EQ(result.baseNz, 160);
  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{104, 104}));
  EXPECT_EQ(result.canonicalGeom.stepLayers, 56);
  EXPECT_EQ(result.canonicalGeom.cropLayers, 66);
  EXPECT_EQ(result.canonicalGeom.blendOverlap, 10);
  EXPECT_TRUE(result.warnings.empty());
}

TEST(ReconSetupTest, PrepareBedSetupDerivesOverlapForDifferentVoxel) {
  // 0.4 mm voxels + default 16.8 mm step -> 42 layers, overlap = 160 - 42.
  ReconArgs args;
  args.Nx = 108;
  args.Ny = 108;
  args.Nz = 160;
  args.Vz = 0.4f;
  args.dz = 0.4;
  args.projPaths = {"Bed1.bin", "Bed2.bin"};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{118}));
  EXPECT_EQ(result.canonicalGeom.stepLayers, 42);
  EXPECT_EQ(result.canonicalGeom.cropLayers, 52);
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

TEST(ReconSetupTest, AdoptReconDimsUsesPerBedNzFromTaggedSubsetMap) {
  const auto path = makeTempMhdPath();
  {
    std::ofstream out(path);
    ASSERT_TRUE(out.is_open());
    out << "ObjectType = Image\nNDims = 3\nDimSize = 120 120 660\n"
           "ElementSpacing = 0.3 0.3 0.3\nElementType = MET_FLOAT\n"
           "SPECTMapRole = regularizer_reconstruction_storage\n"
           "SPECTBedOrder = increasing_bed_number\n"
           "SPECTBeds = 2,3,4\n"
           "SPECTZLayout = reconstruction_storage_z\n"
           "SPECTIntensityUnits = HU\n"
           "ElementDataFile = dummy.raw\n";
  }

  ReconArgs args;
  args.useReg = true;
  args.regCtPath = path.string();

  EXPECT_TRUE(adoptReconDimsFromHint(args));
  EXPECT_EQ(args.Nx, 120);
  EXPECT_EQ(args.Ny, 120);
  EXPECT_EQ(args.Nz, 220);
  std::filesystem::remove(path);
}

TEST(ReconSetupTest, PrepareBedSetupAcceptsContinuousBedSubset) {
  ReconArgs args;
  args.Nx = 120;
  args.Ny = 120;
  args.Nz = 220;
  args.dz = 0.3;
  args.projPaths = {"Bed4.bin", "Bed2.bin", "Bed3.bin"};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(result.bedNumbers, (std::vector<int>{2, 3, 4}));
  EXPECT_NE(result.projectionPaths.front().find("Bed2.bin"), std::string::npos);
  EXPECT_EQ(args.Nz, 660);
}

TEST(ReconSetupTest, PrepareBedSetupRejectsNonContinuousBeds) {
  ReconArgs args;
  args.Nx = 120;
  args.Ny = 120;
  args.Nz = 220;
  args.projPaths = {"Bed2.bin", "Bed4.bin"};

  EXPECT_THROW((void)prepareBedSetup(args), std::runtime_error);
}

TEST(ReconSetupTest, MatrixGridOutputRequiresExplicitAlignedMultiBedStep) {
  ReconArgs args;
  args.Nx = args.Ny = 90;
  args.Nz = 240;
  args.dz = 0.4;
  args.matrixGridOutput = true;
  args.singleBedStepAligned = true;
  args.projPaths = {"Bed1.bin"};
  const auto result = prepareBedSetup(args);
  EXPECT_EQ(args.Nz, 240);
  EXPECT_EQ(result.canonicalGeom.cropLayers, 0);
  EXPECT_NE(result.summary.find("full matrix-grid"), std::string::npos);
  args.projPaths.push_back("Bed2.bin");
  EXPECT_THROW((void)prepareBedSetup(args), std::runtime_error);
  args.bedStepMm = 8.1;
  EXPECT_THROW((void)prepareBedSetup(args), std::runtime_error);
  args.bedStepMm = 8.0;
  args.cylinderDiameterMM = 32.8f;
  args.cylinderHeightMM = 12.0f;
  const auto multi = prepareBedSetup(args);
  // Joint global volume: 240 + 20 layers, beds overlap by 220 layers.
  EXPECT_EQ(args.Nz, 260);
  EXPECT_EQ(multi.canonicalGeom.cropLayers, 240);
  EXPECT_EQ(multi.canonicalGeom.stepLayers, 20);
  EXPECT_EQ(multi.canonicalGeom.blendOverlap, 220);
  EXPECT_EQ(args.bedOverlaps, (std::vector<int>{220}));
  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{220}));
  EXPECT_FLOAT_EQ(args.cylinderHeightMM, 12.0f);
  EXPECT_NE(multi.summary.find("joint"), std::string::npos);
}

TEST(ReconSetupTest, MatrixGridJointVolumeSupportsSixteenBeds) {
  ReconArgs args;
  args.Nx = args.Ny = 90;
  args.Nz = 240;
  args.dz = 0.4;
  args.matrixGridOutput = true;
  args.bedStepMm = 6.0;
  for (int bed = 1; bed <= spect::core::kMaxBeds; ++bed)
    args.projPaths.push_back("Bed" + std::to_string(bed) + ".bin");
  const auto result = prepareBedSetup(args);
  ASSERT_EQ(spect::core::kMaxBeds, 16);
  EXPECT_EQ(result.bedNumbers.back(), 16);
  EXPECT_EQ(args.Nz, 240 + 15 * 15);
  EXPECT_EQ(args.bedOverlaps, std::vector<int>(15, 225));
  EXPECT_EQ(spect::app::computeBedZOffsets(16, 240, args.bedOverlaps).back(),
            15 * 15);

  ReconArgs tooMany = args;
  tooMany.Nz = 240;
  tooMany.projPaths.push_back("Bed17.bin");
  EXPECT_THROW((void)prepareBedSetup(tooMany), std::runtime_error);

  ReconArgs ctac = args;
  ctac.Nz = 240;
  ctac.doCTAC = true;
  EXPECT_THROW((void)prepareBedSetup(ctac), std::runtime_error);

  ReconArgs sixModule;
  sixModule.Nx = sixModule.Ny = 120;
  sixModule.Nz = 220;
  sixModule.dz = 0.3;
  for (int bed = 1; bed <= 6; ++bed)
    sixModule.projPaths.push_back("Bed" + std::to_string(bed) + ".bin");
  EXPECT_THROW((void)prepareBedSetup(sixModule), std::runtime_error);
}

TEST(ReconSetupTest, PrepareBedSetupDerivesOverlapForSmallMatrixNz) {
  // Device scenario: sampling-system matrix with Nz=100 (metadata override).
  // The scanner step (56 layers) alone gives overlap 44 for any baseNz.
  ReconArgs args;
  args.Nx = 100;
  args.Ny = 100;
  args.Nz = 100;
  args.dz = 0.3;
  args.projPaths = {"Bed1.bin", "Bed2.bin"};

  const BedSetupResult result = prepareBedSetup(args);

  EXPECT_EQ(args.outputBedOverlaps, (std::vector<int>{44}));
  EXPECT_EQ(result.canonicalGeom.stepLayers, 56);
  EXPECT_EQ(result.canonicalGeom.cropLayers, 66);
  EXPECT_TRUE(result.warnings.empty());
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
