#include <gtest/gtest.h>

#include "recon/BsremDiagnostics.h"
#include "recon/BsremUpdate.h"
#include "regularizer/RegularizerCompat.h"
#include "tests/common/compact_symmetry_test_helpers.h"

#include <cmath>
#include <vector>

using spect::io::test::denseNumerator;
using spect::io::test::expandCompactToyToFullDense;
using spect::io::test::makeToyCompactCSR;
using spect::io::test::makeToySym90ZMeta;

namespace {

double relL2(const std::vector<double> &a, const std::vector<double> &b) {
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return std::sqrt(num) / (std::sqrt(den) + 1e-30);
}

void applyHostMlem(std::vector<double> &x, const std::vector<double> &n,
                   const std::vector<double> &s) {
  spect::applyMlemUpdate(x, n, s, nullptr, true, 1);
}

} // namespace

TEST(BsremUpdateTest, EquivalenceMatchesMlemOnSynthetic) {
  const size_t n = 8;
  std::vector<double> x0(n);
  std::vector<double> numer(n);
  std::vector<double> denom(n);
  for (size_t i = 0; i < n; ++i) {
    x0[i] = 0.25 + 0.1 * static_cast<double>(i);
    numer[i] = 1.2 + 0.05 * static_cast<double>(i);
    denom[i] = 0.8 + 0.07 * static_cast<double>(i);
  }
  denom[3] = 0.0;

  auto xMlem = x0;
  auto xBsrem = x0;
  applyHostMlem(xMlem, numer, denom);

  spect::BsremUpdateParams p;
  p.alpha = 1.0;
  p.betaScaled = 0.0;
  p.mlemEquivalenceMode = true;
  p.threads = 1;
  spect::applyBsremUpdate(xBsrem, numer, denom, denom, p);

  EXPECT_LT(relL2(xBsrem, xMlem), 1e-15);
  EXPECT_EQ(xMlem[3], 0.0);
  EXPECT_EQ(xBsrem[3], 0.0);
}

TEST(BsremUpdateTest, ClipAppliesToFullRelativeStepIncludingAlpha) {
  std::vector<double> x = {1.0};
  std::vector<double> numer = {4.0};
  std::vector<double> denom = {1.0};
  std::vector<double> raw(1, 0.0);
  std::vector<double> clipped(1, 0.0);

  spect::BsremUpdateParams p;
  p.alpha = 2.0;
  p.betaScaled = 0.0;
  p.clipRelativeStep = true;
  p.clipMin = -0.9;
  p.clipMax = 1.0;
  p.denomEps = 0.0;
  p.rawRelativeSteps = &raw;
  p.updateFactors = &clipped;
  spect::applyBsremUpdate(x, numer, denom, denom, p);

  // v = (4-1)/1 = 3; u = 2*3 = 6 before clip; ū = 1; x = 2.
  EXPECT_DOUBLE_EQ(raw[0], 6.0);
  EXPECT_DOUBLE_EQ(clipped[0], 1.0);
  EXPECT_DOUBLE_EQ(x[0], 2.0);
}

TEST(BsremUpdateTest, SensitivityFloorFreezesLowSupportVoxels) {
  // C.3: voxels with s <= sMin are frozen (zeroed with zeroLowSensitivity),
  // voxels above the floor update normally.
  std::vector<double> x = {1.0, 1.0};
  std::vector<double> numer = {4.0, 4.0};
  std::vector<double> denom = {0.5, 2.0};

  spect::BsremUpdateParams p;
  p.alpha = 1.5;
  p.betaScaled = 0.0;
  p.denomEps = 0.0;
  p.sMin = 1.0; // floor between the two sensitivities
  p.zeroLowSensitivity = true;
  spect::applyBsremUpdate(x, numer, denom, denom, p);

  EXPECT_DOUBLE_EQ(x[0], 0.0); // s=0.5 below floor: frozen to zero
  // s=2: v = (4-2)/2 = 1, u = 1.5 -> clipped to 1 -> x = 2.
  EXPECT_DOUBLE_EQ(x[1], 2.0);
}

TEST(BsremUpdateTest, LambdaMultipliesGradientAndCurvature) {
  std::vector<double> x = {1.0};
  std::vector<double> numer = {2.0};
  std::vector<double> denom = {1.0};
  std::vector<double> grad = {0.5};
  std::vector<double> curv = {2.0};
  std::vector<double> kappa = {4.0};
  std::vector<double> v(1, 0.0);

  spect::BsremUpdateParams p;
  p.alpha = 1.0;
  p.betaScaled = 0.5;
  p.gradient = &grad;
  p.curvature = &curv;
  p.kappa = &kappa;
  p.denomEps = 0.0;
  p.clipRelativeStep = false;
  p.preconditionedSteps = &v;
  spect::applyBsremUpdate(x, numer, denom, denom, p);

  // lambda = 0.5 * 4 = 2
  // v = (2 - 1 - 2*0.5) / (1 + 2*2) = 0 / 5 = 0
  EXPECT_NEAR(v[0], 0.0, 1e-15);
}

TEST(BsremUpdateTest, ProductionClampIsNotEquivalenceProof) {
  std::vector<double> x0 = {1.0};
  std::vector<double> numer = {4.0};
  std::vector<double> denom = {1.0};
  auto xMlem = x0;
  auto xA2 = x0;
  applyHostMlem(xMlem, numer, denom);
  spect::BsremUpdateParams p;
  p.alpha = 1.0;
  p.betaScaled = 0.0;
  p.denomEps = 0.0;
  p.clipRelativeStep = true;
  spect::applyBsremUpdate(xA2, numer, denom, denom, p);
  // MLEM allows x*=4; A2 clips u=(4-1)/1=3 to 1 so x=2.
  EXPECT_DOUBLE_EQ(xMlem[0], 4.0);
  EXPECT_DOUBLE_EQ(xA2[0], 2.0);
}

TEST(BsremUpdateTest, ToyDenseA0A1StayWithin1e4) {
  const auto meta = makeToySym90ZMeta();
  const auto compact = makeToyCompactCSR();
  const auto full = expandCompactToyToFullDense(meta, compact);
  const int rows = meta.fullRows();
  const int cols = meta.totalVoxels;
  std::vector<double> x0(static_cast<size_t>(cols), 1.0);
  std::vector<double> y(static_cast<size_t>(rows));
  for (int i = 0; i < rows; ++i)
    y[static_cast<size_t>(i)] = 1.5 + 0.01 * static_cast<double>(i);

  auto xMlem = x0;
  auto xBsrem = x0;
  const auto s = spect::io::test::denseDenom(full, rows, cols);
  spect::BsremUpdateParams p;
  p.mlemEquivalenceMode = true;
  p.threads = 1;
  for (int it = 0; it < 10; ++it) {
    const auto nM = denseNumerator(full, y, xMlem, rows, cols);
    const auto nB = denseNumerator(full, y, xBsrem, rows, cols);
    applyHostMlem(xMlem, nM, s);
    spect::applyBsremUpdate(xBsrem, nB, s, s, p);
    if (it + 1 == 1 || it + 1 == 2 || it + 1 == 5 || it + 1 == 10) {
      EXPECT_LT(relL2(xBsrem, xMlem), 1e-12) << "iter " << (it + 1);
    }
  }
}

TEST(BsremDiagnosticsTest, PoissonYZeroIsMuAndPositiveYRejectsNonpositiveMu) {
  const double mu[] = {2.0, 0.0, -1.0};
  const double y[] = {0.0, 3.0, 1.0};
  const auto stats = spect::accumulatePoissonDataObjective(mu, y, 3, 0.0);
  EXPECT_DOUBLE_EQ(stats.dataObjective, 2.0);
  EXPECT_EQ(stats.invalidMuCount, 2u);
}

TEST(BsremUpdateTest, StabilityC0C3FiniteAndNotAllClamped) {
  const size_t n = 16;
  std::vector<double> x(n, 1.0);
  std::vector<double> s(n, 2.0);
  std::vector<double> numer(n, 2.4);
  std::vector<double> grad(n, 0.1);
  std::vector<double> curv(n, 0.5);
  std::vector<double> raw(n, 0.0);

  auto step = [&](bool rdp, bool curvOn, double alpha) {
    x.assign(n, 1.0);
    spect::BsremUpdateParams p;
    p.alpha = alpha;
    p.betaScaled = rdp ? 0.3 : 0.0;
    p.gradient = rdp ? &grad : nullptr;
    p.curvature = (rdp && curvOn) ? &curv : nullptr;
    p.rawRelativeSteps = &raw;
    p.clipRelativeStep = true;
    p.denomEps = 1e-12;
    spect::applyBsremUpdate(x, numer, s, s, p);
    size_t clamps = 0;
    for (double u : raw) {
      if (u < -0.9 || u > 1.0)
        ++clamps;
      EXPECT_TRUE(std::isfinite(u));
    }
    for (double v : x)
      EXPECT_TRUE(std::isfinite(v));
    EXPECT_LT(clamps, n);
  };

  step(false, false, 1.0); // C0
  step(true, false, 1.0);  // C1
  step(true, true, 1.0);   // C2
  step(true, true, 0.5);   // C3 conservative
}

TEST(BsremDiagnosticsTest, LambdaIsBetaTimesKappaInMetrics) {
  std::vector<double> x = {1.0, 2.0};
  std::vector<double> s = {4.0, 4.0};
  std::vector<double> n = {3.0, 3.0};
  std::vector<double> kappa = {2.0, 2.0};
  std::vector<double> prev = {1.0, 2.0};
  spect::BsremIterationInputs in;
  in.image = &x;
  in.imagePrev = &prev;
  in.numerator = &n;
  in.globalDenom = &s;
  in.kappa = &kappa;
  in.beta = 0.3;
  in.regularizerRaw = 10.0;
  const auto m = spect::computeBsremIterationMetrics(1, in);
  EXPECT_DOUBLE_EQ(m.kappaMean, 2.0);
  EXPECT_DOUBLE_EQ(m.lambdaMean, 0.6);
  EXPECT_DOUBLE_EQ(m.regularizerWeighted, 6.0);
}
