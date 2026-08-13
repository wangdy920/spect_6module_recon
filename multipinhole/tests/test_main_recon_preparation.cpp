#include <gtest/gtest.h>

#include "filters/ButterworthFilter.h"
#include "pipeline/ReconPreparation.h"

namespace spect::main::test {

TEST(ReconPreparationTest, ButterworthDisabledWhenFlagOff) {
  ReconArgs args;
  args.useButter = false;

  const auto conf = prepareButterworthConf(args);

  EXPECT_FALSE(conf.enabled);
}

TEST(ReconPreparationTest, ButterworthUsesConfiguredOrderAndCutoff) {
  ReconArgs args;
  args.useButter = true;
  args.butterApplyStage = "in_recon";
  args.butterOrder = 0;
  args.butterCutoff = 0.42f;

  const auto conf = prepareButterworthConf(args);

  EXPECT_TRUE(conf.enabled);
  EXPECT_EQ(conf.order, 1);
  EXPECT_NEAR(conf.cutoff_hi_frac, 0.42, 1e-6);
}

TEST(ReconPreparationTest, ButterworthDisabledWhenStageNotInRecon) {
  ReconArgs args;
  args.useButter = true;
  args.butterApplyStage = "none";

  const auto conf = prepareButterworthConf(args);

  EXPECT_FALSE(conf.enabled);
}

TEST(ReconPreparationTest, ButterworthPreparationHasNoBackendChoice) {
  ReconArgs args;
  args.useButter = true;
  args.butterApplyStage = "in_recon";

  const auto conf = prepareButterworthConf(args);

  EXPECT_TRUE(conf.enabled);
  EXPECT_EQ(conf.order, args.butterOrder);
}

TEST(ReconPreparationTest, SmoothingDisabledWhenSigmaNonPositive) {
  ReconArgs args;
  args.useSmooth = true;
  args.smoothApplyStage = "in_recon";
  args.smoothSigma = 0.0f;

  const auto conf = prepareSmoothingConf(args);

  EXPECT_FALSE(conf.enabled);
  EXPECT_TRUE(conf.kx.empty());
}

TEST(ReconPreparationTest, SmoothingDisabledWhenStageNotInRecon) {
  ReconArgs args;
  args.useSmooth = true;
  args.smoothSigma = 1.0f;
  args.smoothApplyStage = "none";

  const auto conf = prepareSmoothingConf(args);

  EXPECT_FALSE(conf.enabled);
  EXPECT_TRUE(conf.kx.empty());
}

TEST(ReconPreparationTest, SmoothingBuildsOddKernel) {
  ReconArgs args;
  args.useSmooth = true;
  args.smoothApplyStage = "in_recon";
  args.smoothSigma = 1.0f;
  args.smoothKernel = 4;

  const auto conf = prepareSmoothingConf(args);

  ASSERT_TRUE(conf.enabled);
  ASSERT_FALSE(conf.kx.empty());
  EXPECT_EQ(conf.kx.size() % 2, static_cast<size_t>(1));
  EXPECT_EQ(conf.kx, conf.ky);
  EXPECT_EQ(conf.kx, conf.kz);
}

TEST(ReconPreparationTest, CombinedPreparationBuildsAllParts) {
  ReconArgs args;
  args.useSmooth = true;
  args.smoothApplyStage = "in_recon";
  args.smoothSigma = 1.0f;
  args.smoothKernel = 3;
  args.useButter = true;
  args.butterApplyStage = "in_recon";
  args.butterOrder = 2;
  args.butterCutoff = 0.35f;

  spect::filters::PSFConf psf;
  psf.mode = spect::filters::PSFMode::Operator;
  psf.fwhm_x = 2.0;
  psf.fwhm_y = 2.0;
  psf.fwhm_z = 2.0;
  psf.vox_x = 1.0;
  psf.vox_y = 1.0;
  psf.vox_z = 1.0;

  const auto prepared = prepareReconPreparedParams(args, psf);

  EXPECT_TRUE(prepared.butterConf.enabled);
  EXPECT_TRUE(prepared.smoothConf.enabled);
  EXPECT_FALSE(prepared.psfKernels.kx.empty());
}

TEST(ReconPreparationTest, CylindricalMaskHelperReturnsNullWhenDisabled) {
  ReconArgs args;
  args.useCylindricalMask = false;
  std::vector<uint8_t> mask;
  const std::array<double, 3> spacing = {1.0, 1.0, 1.0};

  const auto *ptr =
      maybeMakeCylindricalMask(mask, 4, 4, 2, spacing, args);

  EXPECT_EQ(ptr, nullptr);
  EXPECT_TRUE(mask.empty());
}

TEST(ReconPreparationTest, CylindricalMaskRepeatsForIndependentBeds) {
  ReconArgs args;
  args.useCylindricalMask = true;
  args.bedCount = 2;
  args.cylinderDiameterMM = 6.0f;
  args.cylinderHeightMM = 4.0f;
  std::vector<uint8_t> mask;
  const std::array<double, 3> spacing = {1.0, 1.0, 1.0};

  const auto *ptr =
      maybeMakeCylindricalMask(mask, 10, 10, 12, spacing, args);

  ASSERT_EQ(ptr, &mask);
  ASSERT_EQ(mask.size(), 1200u);
  const auto middle = mask.begin() + 600;
  EXPECT_TRUE(std::equal(mask.begin(), middle, middle));
  const size_t active =
      static_cast<size_t>(std::count(mask.begin(), middle, uint8_t{1}));
  EXPECT_GT(active, 0u);
  EXPECT_LT(active, 600u);
}

} // namespace spect::main::test
