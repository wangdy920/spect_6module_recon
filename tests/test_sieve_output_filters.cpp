/**
 * @file test_sieve_output_filters.cpp
 * @brief Sieve PSF、smooth/butter apply_stage 与独立滤波算子的测试。
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "cli/ArgumentParser.h"
#include "common/parser_test_helpers.h"
#include "cuda/CudaConfig.h"
#include "filters/cuda/VolumeOpsCUDA.h"
#include "filters/ButterworthFilter.h"
#include "filters/Convolution.h"
#include "pipeline/ReconPreparation.h"
#include "pipeline/VolumeOutput.h"
#include "recon/ProjectionOps.h"
#include "recon/ReconConfig.h"

using spect::apply_psf_forward;
using spect::build_sieve_save_volume;
using spect::filters::PSFMode;
using spect::filters::make_psf_kernels;

namespace spect {
namespace test {

namespace {

std::vector<double> makeRampVolume(int nx, int ny, int nz) {
  std::vector<double> v(static_cast<size_t>(nx) * ny * nz);
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = 0.001 * static_cast<double>(i % 97) + 0.01;
  return v;
}

double maxAbsDiff(const std::vector<double> &a, const std::vector<double> &b) {
  double d = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

class ScopedEnvironmentOverride {
public:
  ScopedEnvironmentOverride(const char *name, const char *value) : name_(name) {
    const char *previous = std::getenv(name);
    if (previous) {
      hadPrevious_ = true;
      previous_ = previous;
    }
    set(name_.c_str(), value);
  }

  ~ScopedEnvironmentOverride() {
    if (hadPrevious_) {
      set(name_.c_str(), previous_.c_str());
      return;
    }
#ifdef _WIN32
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }

  ScopedEnvironmentOverride(const ScopedEnvironmentOverride &) = delete;
  ScopedEnvironmentOverride &
  operator=(const ScopedEnvironmentOverride &) = delete;

private:
  static void set(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
  }

  std::string name_;
  std::string previous_;
  bool hadPrevious_ = false;
};

spect::filters::PSFConf makeTestPsf() {
  spect::filters::PSFConf psf;
  psf.mode = PSFMode::Sieve;
  psf.fwhm_x = psf.fwhm_y = psf.fwhm_z = 1.2;
  psf.fwhm_mm = 1.2;
  psf.vox_x = psf.vox_y = psf.vox_z = 0.6;
  psf.min_sigmas = 3.0;
  return psf;
}

// ---------------------------------------------------------------- parser ---

struct ParseOutcome {
  bool ok = false;
  spect::cli::ReconArgs args;
};

ParseOutcome parseWithExtra(std::vector<std::string> extra) {
  auto storage = spect::io::test::compactBaseArgs();
  storage.insert(storage.end(), extra.begin(), extra.end());
  ParseOutcome out;
  out.ok = spect::io::test::parseReconArgsForTest(storage, out.args);
  return out;
}

ParseOutcome parseWithIni(const std::string &iniBody) {
  const auto path = std::filesystem::temp_directory_path() /
                    "spect_sieve_stage_test.ini";
  {
    std::ofstream ofs(path, std::ios::trunc);
    ofs << iniBody;
  }
  // 自建基参数，避免与 compactBaseArgs 的占位 -config 重复。
  std::vector<std::string> storage = {
      "SPECT_recon", "-config", path.string(), "-df", "proj.bin",
      "-chunkCSRDir", "chunks", "-opti", "mlem", "-it", "1",
      "-dim", "108,108,80", "-vox", "0.3", "-ang", "360"};
  ParseOutcome out;
  std::vector<char *> argv;
  argv.reserve(storage.size());
  for (std::string &arg : storage)
    argv.push_back(arg.data());
  spect::cli::ArgumentParser parser;
  out.ok = parser.parse(static_cast<int>(argv.size()), argv.data(),
                        out.args) ==
           spect::cli::ArgumentParser::ParseResult::Success;
  return out;
}

} // namespace

TEST(SieveParserTest, CliPsfModeSieveAccepted) {
  const auto out = parseWithExtra({"-psfMode", "sieve"});
  ASSERT_TRUE(out.ok);
  EXPECT_EQ(out.args.psf.mode, PSFMode::Sieve);
}

TEST(SieveParserTest, IniPsfModeSieveAccepted) {
  const auto out = parseWithIni("[PSF]\nmode = sieve\n");
  ASSERT_TRUE(out.ok);
  EXPECT_EQ(out.args.psf.mode, PSFMode::Sieve);
}

TEST(SieveParserTest, SieveRejectsInReconSmoothViaCli) {
  const auto out = parseWithExtra({"-psfMode", "sieve", "-smooth", "1.0"});
  EXPECT_FALSE(out.ok);
}

TEST(SieveParserTest, SieveRejectsInReconButterViaCli) {
  const auto out = parseWithExtra({"-psfMode", "sieve", "-butter", "0.35,2"});
  EXPECT_FALSE(out.ok);
}

TEST(SieveParserTest, RemovedOutputStageIsRejected) {
  const auto out = parseWithIni(
      "[PSF]\nmode = sieve\n"
      "[Filtering]\nsmooth_apply_stage = output\nbutter_apply_stage = none\n");
  EXPECT_FALSE(out.ok);
}

TEST(SieveParserTest, SmoothCliImpliesInReconStage) {
  const auto out = parseWithExtra({"-smooth", "1.2"});
  ASSERT_TRUE(out.ok);
  EXPECT_EQ(out.args.smoothApplyStage, "in_recon");
  EXPECT_TRUE(out.args.useSmooth);
}

TEST(SieveParserTest, ButterCliImpliesInReconStage) {
  const auto out = parseWithExtra({"-butter", "0.4,3"});
  ASSERT_TRUE(out.ok);
  EXPECT_EQ(out.args.butterApplyStage, "in_recon");
  EXPECT_FLOAT_EQ(out.args.butterCutoff, 0.4f);
  EXPECT_EQ(out.args.butterOrder, 3);
}

TEST(SieveParserTest, IniCanEnableInReconFilters) {
  const auto out = parseWithIni(
      "[Filtering]\nbutterworth_cutoff = 0.4\ngaussian_sigma = 1.2\n"
      "smooth_apply_stage = in_recon\nbutter_apply_stage = in_recon\n");
  ASSERT_TRUE(out.ok);
  EXPECT_TRUE(out.args.useSmooth);
  EXPECT_TRUE(out.args.useButter);
}

TEST(SieveParserTest, InReconStageRequiresPositiveParams) {
  // -smooth 0 显式触发 in_recon 且 sigma 非正 → 拒绝。
  EXPECT_FALSE(parseWithExtra({"-smooth", "0"}).ok);
  // -butter 0（截止 0）同理。
  EXPECT_FALSE(parseWithExtra({"-butter", "0,2"}).ok);
}

TEST(SieveParserTest, InvalidStageRejected) {
  const auto out = parseWithIni("[Filtering]\nbutter_apply_stage = midloop\n");
  EXPECT_FALSE(out.ok);
}

TEST(SieveRuntimeTest, DisabledCudaVolumeOpsFailsPreparation) {
  ScopedEnvironmentOverride volumeOps("SPECT_CUDA_VOLUME_OPS", "0");
  spect::cli::ReconArgs args;
  args.usePSF = true;
  args.psf = makeTestPsf();
  ReconOpts opts;
  const spect::filters::Spacing3 spacing{0.6, 0.6, 0.6};

  EXPECT_THROW(spect::main::preparePSFForRuntime(
                   opts, args, spacing, 1, 8 * 8 * 6, 1),
               std::runtime_error);
}


TEST(SieveParserTest, ZeroInReconParamsFromIniOverrideLegacyDefaults) {
  const auto out = parseWithIni(
      "[Filtering]\nbutterworth_cutoff = 0\ngaussian_sigma = 0\n"
      "smooth_apply_stage = none\nbutter_apply_stage = none\n");
  ASSERT_TRUE(out.ok);
  EXPECT_FLOAT_EQ(out.args.butterCutoff, 0.0f);
  EXPECT_FLOAT_EQ(out.args.smoothSigma, 0.0f);
  EXPECT_FALSE(out.args.useSmooth);
}

// ---------------------------------------------------- sieve save copy -----

TEST(SieveSaveCopyTest, SingleBedSaveCopyEqualsForwardConvolution) {
  const int nx = 8, ny = 8, nz = 6;
  const std::vector<double> image = makeRampVolume(nx, ny, nz);
  ReconOpts opts;
  opts.threads = 1;
  opts.psf = makeTestPsf();
  const auto kernels = make_psf_kernels(opts.psf);

  const std::vector<double> saveCopy = build_sieve_save_volume(
      image, nx, ny, nz, 1, {0}, false, opts, &kernels, 1);

  std::vector<double> expected;
  apply_psf_forward(image, expected, nx, ny, nz, opts, &kernels, 1);
  EXPECT_EQ(saveCopy.size(), image.size());
  EXPECT_LT(maxAbsDiff(saveCopy, expected), 1e-12);
  EXPECT_GT(maxAbsDiff(saveCopy, image), 1e-9);
}

TEST(SieveSaveCopyTest, TwoBedSaveCopyConvolvesPerBedLocally) {
  const int nx = 6, ny = 6, baseNz = 4;
  const int bedCount = 2;
  const std::vector<int> bedZStarts = {0, 4};
  const std::vector<double> image = makeRampVolume(nx, ny, baseNz * bedCount);
  ReconOpts opts;
  opts.threads = 1;
  opts.psf = makeTestPsf();
  const auto kernels = make_psf_kernels(opts.psf);

  const std::vector<double> saveCopy =
      build_sieve_save_volume(image, nx, ny, baseNz, bedCount, bedZStarts,
                              false, opts, &kernels, 1);

  const size_t bedVox = static_cast<size_t>(nx) * ny * baseNz;
  std::vector<double> expected(image.size(), 0.0);
  for (int bed = 0; bed < bedCount; ++bed) {
    std::vector<double> local(image.begin() + bed * bedVox,
                              image.begin() + (bed + 1) * bedVox);
    std::vector<double> blurred;
    apply_psf_forward(local, blurred, nx, ny, baseNz, opts, &kernels, 1);
    std::copy(blurred.begin(), blurred.end(), expected.begin() + bed * bedVox);
  }
  EXPECT_LT(maxAbsDiff(saveCopy, expected), 1e-12);
}

TEST(SieveSaveCopyTest, InputStateIsNotModified) {
  const int nx = 6, ny = 6, nz = 4;
  std::vector<double> image = makeRampVolume(nx, ny, nz);
  const std::vector<double> original = image;
  ReconOpts opts;
  opts.threads = 1;
  opts.psf = makeTestPsf();
  const auto kernels = make_psf_kernels(opts.psf);
  (void)build_sieve_save_volume(image, nx, ny, nz, 1, {0}, false, opts,
                                &kernels, 1);
  EXPECT_EQ(image, original);
}

// ---------------------------------------------------------- castor kernel --

TEST(CastorKernelTest, MatchesErfVoxelIntegral) {
  // fwhm=1.3mm, vox=0.3mm, 3σ：σ=0.552mm，raw=5.52 → round → half=6。
  const auto k = spect::filters::makeCastorErf1D(1.3, 0.3, 3.0);
  ASSERT_EQ(k.size(), 13u);
  const double sigma = 1.3 / (2.0 * std::sqrt(2.0 * std::log(2.0)));
  const double denom = std::sqrt(2.0) * sigma;
  double sum = 0.0;
  for (int i = -6; i <= 6; ++i)
    sum += 0.5 * (std::erf((i + 0.5) * 0.3 / denom) -
                  std::erf((i - 0.5) * 0.3 / denom));
  for (int i = -6; i <= 6; ++i) {
    const double expected =
        0.5 * (std::erf((i + 0.5) * 0.3 / denom) -
               std::erf((i - 0.5) * 0.3 / denom)) /
        sum;
    EXPECT_NEAR(k[static_cast<size_t>(i + 6)], expected, 1e-15);
  }
}

TEST(CastorKernelTest, RoundsHalfDownNotCeil) {
  // raw = sigma*N/vox = 1.49 → half=1（round 与 ceil 的分界：ceil 会给 2）。
  const double c = 2.0 * std::sqrt(2.0 * std::log(2.0));
  EXPECT_EQ(spect::filters::makeCastorErf1D(c * 1.49, 1.0, 1.0).size(), 3u);
  EXPECT_EQ(spect::filters::makeCastorErf1D(c * 1.51, 1.0, 1.0).size(), 5u);
}

TEST(CastorKernelTest, ZNormMatchesInRangeTapSum) {
  const std::vector<double> kz = {0.25, 0.5, 0.25};
  const auto norm = spect::filters::makeCastorZNorm1D(kz, 4);
  ASSERT_EQ(norm.size(), 4u);
  EXPECT_DOUBLE_EQ(norm[0], 0.75);
  EXPECT_DOUBLE_EQ(norm[1], 1.0);
  EXPECT_DOUBLE_EQ(norm[2], 1.0);
  EXPECT_DOUBLE_EQ(norm[3], 0.75);
}

TEST(CastorKernelTest, ForwardRestoresZAndAttenuatesXYBorders) {
  const int n = 12, nz = 6;
  std::vector<double> v(static_cast<size_t>(n) * n * nz, 1.0);
  const auto k = spect::filters::makeCastorErf1D(0.6, 0.6, 3.0);
  spect::filters::conv3d_castor_forward_inplace(v, n, n, nz, k, k, k, 1);
  const auto at = [&](int x, int y, int z) {
    return v[static_cast<size_t>(z) * n * n + y * n + x];
  };
  // z 边界被 D_z 补偿：顶/底面中心 ≈ 1。
  EXPECT_NEAR(at(6, 6, 0), 1.0, 1e-12);
  EXPECT_NEAR(at(6, 6, nz - 1), 1.0, 1e-12);
  // x/y 边界不补偿：角列明显衰减。
  EXPECT_LT(at(0, 6, 3), 0.95);
  // 中心体素保持 1。
  EXPECT_NEAR(at(6, 6, 3), 1.0, 1e-12);
}

TEST(CastorKernelTest, BackwardReusesForwardExactly) {
  const int n = 8;
  std::vector<double> a = makeRampVolume(n, n, n);
  std::vector<double> b = a;
  ReconOpts opts;
  opts.threads = 1;
  opts.psf = makeTestPsf();
  opts.psf.kernel = spect::filters::PSFKernel::Castor;
  const auto kernels = make_psf_kernels(opts.psf);
  apply_psf_forward(a, a, n, n, n, opts, &kernels, 1);
  spect::apply_psf_backward(b, n, n, n, opts, &kernels, 1);
  EXPECT_LT(maxAbsDiff(a, b), 1e-14);
}

TEST(CastorKernelTest, CudaMatchesCpuReference) {
  if (!spect::cuda::isCudaAvailable())
    GTEST_SKIP() << "CUDA unavailable";
  const int n = 16;
  std::vector<double> gpu = makeRampVolume(n, n, n);
  std::vector<double> cpu = gpu;
  const auto k = spect::filters::makeCastorErf1D(0.9, 0.6, 3.0);
  ASSERT_TRUE(spect::cuda::applyCastorConv3DInplaceCUDA(gpu, n, n, n, k, k, k));
  spect::filters::conv3d_castor_forward_inplace(cpu, n, n, n, k, k, k, 1);
  double maxAbs = 0.0, maxRef = 0.0;
  for (size_t i = 0; i < gpu.size(); ++i) {
    maxAbs = std::max(maxAbs, std::abs(gpu[i] - cpu[i]));
    maxRef = std::max(maxRef, std::abs(cpu[i]));
  }
  EXPECT_LT(maxAbs / maxRef, 1e-4);  // 默认 float 路径精度
}

TEST(CastorKernelTest, ForcedFloat32StagingIgnoresPsfDoubleEnvironment) {
  if (!spect::cuda::isCudaAvailable())
    GTEST_SKIP() << "CUDA unavailable";

  const int nx = 2, ny = 2, nz = 2;
  const std::vector<double> input = {
      1.0 + 1e-9, 0.1, 1.0 / 3.0, 16777217.0,
      0.123456789, 42.0000001, 7.00000001, 3.141592653589793};
  const std::vector<double> identity{1.0};
  std::vector<double> forced = input;
  std::vector<double> environmentSelected = input;
  ScopedEnvironmentOverride psfDouble("SPECT_CUDA_PSF_DOUBLE", "1");

  ASSERT_TRUE(
      spect::cuda::applyCastorConv3DInplaceCUDAForcedFloat32Staging(
          forced, nx, ny, nz, identity, identity, identity));
  ASSERT_TRUE(spect::cuda::applyCastorConv3DInplaceCUDA(
      environmentSelected, nx, ny, nz, identity, identity, identity, true));

  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_DOUBLE_EQ(forced[i],
                     static_cast<double>(static_cast<float>(input[i])));
  }
  EXPECT_GT(maxAbsDiff(forced, input), 0.0);
  EXPECT_EQ(environmentSelected, input);
}

TEST(CastorKernelTest, SieveCastorSaveCopyIdentity) {
  const int nx = 8, ny = 8, nz = 6;
  const std::vector<double> image = makeRampVolume(nx, ny, nz);
  ReconOpts opts;
  opts.threads = 1;
  opts.psf = makeTestPsf();
  opts.psf.kernel = spect::filters::PSFKernel::Castor;
  const auto kernels = make_psf_kernels(opts.psf);
  const std::vector<double> saveCopy = build_sieve_save_volume(
      image, nx, ny, nz, 1, {0}, false, opts, &kernels, 1);
  std::vector<double> expected;
  apply_psf_forward(image, expected, nx, ny, nz, opts, &kernels, 1);
  EXPECT_LT(maxAbsDiff(saveCopy, expected), 1e-12);
}

TEST(CastorKernelTest, ParserAcceptsPsfKernelCastor) {
  const auto out = parseWithExtra({"-psfMode", "sieve", "-psfKernel", "castor"});
  ASSERT_TRUE(out.ok);
  EXPECT_EQ(out.args.psf.kernel, spect::filters::PSFKernel::Castor);
  EXPECT_EQ(out.args.psf.mode, PSFMode::Sieve);
}

TEST(CastorKernelTest, ParserRejectsUnknownPsfKernel) {
  EXPECT_FALSE(parseWithExtra({"-psfKernel", "omega"}).ok);
}

// ------------------------------------------------- output-stage butter ----

TEST(OutputButterTest, ThrowsBeyondVoxelNyquist) {
  std::vector<double> v(4 * 4 * 4, 1.0);
  const spect::filters::Spacing3 sp{1.0, 1.0, 1.0};  // Nyquist = 5 cpcm
  EXPECT_THROW(spect::filters::apply_butterworth_output_cufft(
                   v, 4, 4, 4, sp, 100.0, 8),
               std::runtime_error);
}

TEST(OutputButterTest, RejectsNonPositiveCutoff) {
  std::vector<double> v(8, 1.0);
  const spect::filters::Spacing3 sp{1.0, 1.0, 1.0};
  EXPECT_THROW(spect::filters::apply_butterworth_output_cufft(v, 2, 2, 2, sp,
                                                              0.0, 8),
               std::runtime_error);
}

TEST(OutputButterTest, SmoothBlobPassesNearlyUnchangedAndPreservesSidelobes) {
  if (!spect::cuda::isCudaAvailable())
    GTEST_SKIP() << "CUDA unavailable";
  const int n = 16;
  std::vector<double> v(static_cast<size_t>(n) * n * n, 0.0);
  // 中心高斯团（边缘≈0，贴近真实重建体的零背景）。
  const double sigma2 = 2.0 * 2.0;
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const double dx = x - 7.5, dy = y - 7.5, dz = z - 7.5;
        v[static_cast<size_t>(z) * n * n + y * n + x] =
            std::exp(-(dx * dx + dy * dy + dz * dz) / (2 * sigma2));
      }
  const std::vector<double> input = v;
  const spect::filters::Spacing3 sp{1.0, 1.0, 1.0};
  // Nyquist = 5 cpcm；4.5 cpcm/8 阶对低频团块近乎透明。
  spect::filters::apply_butterworth_output_cufft(v, n, n, n, sp, 4.5, 8);
  const double minV = *std::min_element(v.begin(), v.end());
  EXPECT_LT(minV, 0.0);
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < v.size(); ++i) {
    num += (v[i] - input[i]) * (v[i] - input[i]);
    den += input[i] * input[i];
  }
  EXPECT_LT(std::sqrt(num / den), 0.02);
}

TEST(OutputButterTest, ZeroPaddingPreventsCircularWrap) {
  if (!spect::cuda::isCudaAvailable())
    GTEST_SKIP() << "CUDA unavailable";
  const int n = 16;
  std::vector<double> v(static_cast<size_t>(n) * n * n, 0.0);
  v[0] = 1.0;  // delta at the (0,0,0) corner
  const spect::filters::Spacing3 sp{1.0, 1.0, 1.0};
  spect::filters::apply_butterworth_output_cufft(v, n, n, n, sp, 4.5, 8);
  // 无 2× 补零时循环卷积会把角点响应环绕到远端 (n-1,n-1,n-1)。
  // 脉冲响应逐点可能有零交叉，改为比较远端 8 层以外区域的最大响应。
  double farMax = 0.0;
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        if (std::max({x, y, z}) >= n / 2)
          farMax = std::max(farMax, std::abs(
                                        v[static_cast<size_t>(z) * n * n + y * n + x]));
      }
  EXPECT_GT(v[0], 1e-3);
  EXPECT_LT(farMax, 1e-3 * v[0]);
}

// ------------------------------------------- derived files via saveVolume --


} // namespace test
} // namespace spect
