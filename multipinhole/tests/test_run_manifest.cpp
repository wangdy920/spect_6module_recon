#include <gtest/gtest.h>
#include <algorithm>

#include <filesystem>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <regex>
#include <string>
#include <vector>

#include "cli/ReconArgs.h"
#include "io/CompactSymmetryMetadata.h"
#include "pipeline/RunLogging.h"
#include "pipeline/RunManifest.h"

namespace fs = std::filesystem;

namespace spect {
namespace test {

TEST(RunManifestTest, WritesSchemaGainsAndBedMapping) {
  const fs::path dir = fs::temp_directory_path() / "spect_run_manifest_test";
  fs::remove_all(dir);
  fs::create_directories(dir);

  cli::ReconArgs args;
  args.outputDir = dir.string();
  args.chunkCSRDir = "F:\\matrix\\chunks";
  args.projPaths = {"F:\\data\\Bed1.bin", "F:\\data\\Bed2.bin"};
  args.mergeTimeSegments = true;
  args.projectionSourceGroups = {
      {"F:/case/20260810_210556/SPECT/Bed1.bin",
       "F:/case/20260810_211500/SPECT/Bed1.bin"},
      {"F:/case/20260810_210556/SPECT/Bed2.bin",
       "F:/case/20260810_211500/SPECT/Bed2.bin"}};
  args.badDetectorFile = "bad_detector.txt";
  args.zBedFlip = true;
  args.Nx = 120;
  args.Ny = 120;
  args.Nz = 140;
  // Deliberately differ from the saved geometry; the manifest must describe
  // the actual save contract rather than these reconstruction arguments.
  args.Vx = 9.0f;
  args.Vy = 9.0f;
  args.Vz = 9.0f;
  args.numIters = 50;
  args.saveInterval = 50;
  args.algorithm = cli::ReconArgs::Algorithm::BSREM;
  args.bsrem_alpha0 = 0.75f;
  args.bsrem_gamma = 0.125f;
  args.bsremUseCurvDenom = true;
  args.bsremFreezeCurv = false;
  args.bsremCurvWarmupIters = 4;
  args.bsremCurvEps = 0.0002;
  args.bsremUseKappaMap = true;
  args.bsremKappaEps = 0.0003;
  args.bsremKappaClipMin = 0.25;
  args.bsremKappaClipMax = 4.0;
  args.bsremSoftMask = true;
  args.bsremSoftMaskLambdaOut = 42.0;
  args.useReg = true;
  args.regCtPath = "F:/maps/regct.mhd";
  args.attMapPath = "F:/maps/attmap.mhd";
  args.reg.type = RegType::RDP;
  args.reg.beta = 0.0125;
  args.reg.delta = 0.2;
  args.reg.gamma = 0.01;
  args.reg.skip_outside_mask = false;
  args.psf.mode = filters::PSFMode::Sieve;
  args.psf.kernel = filters::PSFKernel::Castor;
  args.psf.fwhm_mm = 1.3;
  args.psf.min_sigmas = 3.0;
  args.smoothApplyStage = "output";
  args.butterApplyStage = "output";
  args.outputSmoothFwhmMm = 0.30f;
  args.outputSmoothSigmas = 3.0f;
  args.outputButterCutoffCpcm = 5.0f;
  args.outputButterOrder = 8;
  args.outputNonnegativeClamp = false;
  args.outputFiltersOverwritePrimary = true;

  args.outputBedOverlaps = {14};

  io::CompactSymmetryMetadata meta;
  meta.Nx = 120;
  meta.Ny = 120;
  meta.Nz = 220;
  meta.voxelSize = 0.3f;
  meta.fullNumAngles = 360;
  meta.detNx = 48;
  meta.detNz = 48;
  meta.symmetryOrder = 4;

  const int baseNz = 70;
  const int nz = 2 * baseNz;
  const int nx = 3;
  const int ny = 2;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> source(plane * static_cast<size_t>(nz), 0.0);
  std::fill(source.begin() + static_cast<std::ptrdiff_t>(2 * plane),
            source.begin() + static_cast<std::ptrdiff_t>(68 * plane), 10.0);
  std::fill(
      source.begin() +
          static_cast<std::ptrdiff_t>((baseNz + 2) * plane),
      source.begin() +
          static_cast<std::ptrdiff_t>((baseNz + 68) * plane),
      20.0);

  main::OutputTransform transform;
  transform.centerCropNx = 2;
  transform.centerCropNy = 2;
  transform.baseNz = baseNz;
  transform.bedZStarts = {0, baseNz};
  transform.outputBedOverlaps = {14};
  transform.canonicalGeom = {66, 56, 10};


  main::OutputGeometry geometry;
  geometry.nx = nx;
  geometry.ny = ny;
  geometry.nz = nz;
  geometry.vx = 0.3125;
  geometry.vy = 0.625;
  geometry.vz = 1.25;

  main::RunManifestInput input;
  input.args = &args;
  input.compactMeta = &meta;
  input.bedZStarts = &transform.bedZStarts;
  input.outputTransform = &transform;
  input.finalSaveVolume = &source;
  input.geometry = &geometry;
  input.savedIteration = 50;
  input.outputPrefix = "recon";

  main::writeRunManifest(input);

  const fs::path manifestPath = dir / "recon_manifest.json";
  ASSERT_TRUE(fs::exists(manifestPath));
  std::ifstream in(manifestPath);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();

  // Any comma immediately before an object/array terminator is invalid JSON.
  EXPECT_FALSE(std::regex_search(text, std::regex(R"(,\s*[}\]])")));

  EXPECT_NE(text.find("\"schema\": \"spect_recon_run_manifest_v1\""),
            std::string::npos);
  EXPECT_NE(text.find("\"symmetry_mode\": \"sym90z\""), std::string::npos);
  EXPECT_NE(text.find("\"det_nz\": 48\n  },\n  \"execution\": {"),
            std::string::npos);
  // Path backslashes must be JSON-escaped.
  EXPECT_NE(text.find("F:\\\\matrix\\\\chunks"), std::string::npos);
  EXPECT_NE(text.find("Bed1=min_physical_z"), std::string::npos);
  EXPECT_NE(text.find("\"merge_time_segments\": true"), std::string::npos);
  EXPECT_NE(text.find("\"time_segment_count\": 2"), std::string::npos);
  EXPECT_NE(text.find("20260810_211500/SPECT/Bed2.bin"),
            std::string::npos);
  EXPECT_NE(text.find("\"z_bed_flip\": true"), std::string::npos);
  EXPECT_NE(text.find("\"physical_z_offsets_mm\": [0.00000000000000000, "
                      "70.00000000000000000]"),
            std::string::npos);
  EXPECT_NE(text.find("\"local_z_direction\": "
                      "\"increasing_local_z=-storage_z\""),
            std::string::npos);
  EXPECT_NE(text.find("\"scheduler\": "
                      "\"iteration_major_compact_chunk\""),
            std::string::npos);
  EXPECT_NE(text.find("\"kernel\": \"castor\""), std::string::npos);
  EXPECT_NE(text.find("\"opti\": \"bsrem\""), std::string::npos);
  EXPECT_NE(text.find("\"alpha0\": 0.75000000000000000"),
            std::string::npos);
  EXPECT_NE(text.find("\"curv_denom\": true"), std::string::npos);
  EXPECT_NE(text.find("\"curv_freeze\": false"), std::string::npos);
  EXPECT_NE(text.find("\"curv_warmup_iters\": 4"), std::string::npos);
  EXPECT_NE(text.find("\"kappa_map\": true"), std::string::npos);
  EXPECT_NE(text.find("\"kappa_clip\": [0.25000000000000000, "
                      "4.00000000000000000]"),
            std::string::npos);
  EXPECT_NE(text.find("\"soft_mask_lambda\": 42.00000000000000000"),
            std::string::npos);
  EXPECT_NE(text.find("\"mlem_equivalence\": false"), std::string::npos);
  EXPECT_NE(text.find("\"diagnostics\": false"), std::string::npos);
  EXPECT_NE(text.find("\"regularization_enabled\": true"),
            std::string::npos);
  EXPECT_NE(text.find("\"regularizer_type\": \"RDP\""),
            std::string::npos);
  EXPECT_NE(text.find("\"attenuation_map_z_layout\": \"local_matrix_z\""),
            std::string::npos);
  EXPECT_NE(text.find("\"regularizer_ct_z_layout\": "
                      "\"reconstruction_storage_z\""),
            std::string::npos);
  EXPECT_NE(text.find("\"dimensions\": [2, 2, 122]"),
            std::string::npos);
  EXPECT_NE(text.find("\"voxel_mm\": [0.31250000000000000, "
                      "0.62500000000000000, 1.25000000000000000]"),
            std::string::npos);
  EXPECT_EQ(text.find("\"dimensions\": [120, 120, 140]"),
            std::string::npos);
  // Prototype log-median/MAD fit on the 10/20 phantom: Bed1=1, Bed2=0.5.
  EXPECT_NE(text.find("\"fitted_bed_gains\": [1.00000000000000000, "
                      "0.50000000000000000]"),
            std::string::npos);
  EXPECT_NE(text.find("\"gain_fit_method\": "
                      "\"adjacent_overlap_log_median_mad\""),
            std::string::npos);
  EXPECT_NE(text.find("\"stitch_method\": \"center66_cosine10\""),
            std::string::npos);
  EXPECT_NE(text.find("\"gain_fit_reliability_policy\": "
                      "\"prototype_log_chain_no_gate\""),
            std::string::npos);

  EXPECT_NE(text.find("\"increasing_z_index_direction\": "
                      "\"+physical_z\""),
            std::string::npos);
  EXPECT_NE(text.find("\"nonnegative_clamp\": false"), std::string::npos);
  EXPECT_NE(text.find("\"overwrite_primary\": true"), std::string::npos);
  EXPECT_NE(text.find("\"primary_derivation\": \"smooth_then_butter\""),
            std::string::npos);
  EXPECT_NE(text.find("\"derivation\": \"independent_from_prepared_primary\""),
            std::string::npos);
  EXPECT_NE(text.find("\"files\": [\"recon_it50.mhd\", \"recon_it50.raw\", "
                      "\"recon_it50_smooth.mhd\", \"recon_it50_smooth.raw\", "
                      "\"recon_it50_butter.mhd\", \"recon_it50_butter.raw\""),
            std::string::npos);

  in.close();
  fs::remove_all(dir);
}


TEST(RunManifestTest, MissingOutputDirFailsFast) {
  const fs::path dir =
      fs::temp_directory_path() / "spect_run_manifest_missing_dir_test";
  fs::remove_all(dir);

  cli::ReconArgs args;
  args.outputDir = (dir / "does_not_exist").string();

  io::CompactSymmetryMetadata meta;
  std::vector<int> bedZStarts = {0};
  main::OutputTransform transform;
  main::OutputGeometry geometry{1, 1, 1, 0.3125, 0.625, 1.25};

  main::RunManifestInput input;
  input.args = &args;
  input.compactMeta = &meta;
  input.bedZStarts = &bedZStarts;
  input.outputTransform = &transform;
  input.geometry = &geometry;
  input.savedIteration = 1;
  input.outputPrefix = "recon";

  EXPECT_THROW(main::writeRunManifest(input), std::runtime_error);
}

TEST(RunLoggingTest, DefaultLogNameUsesSecondTimestampOnly) {
  cli::ReconArgs args;
  args.outputDir = "C:/tmp/spect_log_name";
  args.outputName = "recon";
  const auto path = main::makeDefaultLogPath(args);
  const std::string name = path.filename().string();
  EXPECT_TRUE(std::regex_match(name,
                               std::regex(R"(recon_\d{8}_\d{6}\.log)")));
}

} // namespace test
} // namespace spect
