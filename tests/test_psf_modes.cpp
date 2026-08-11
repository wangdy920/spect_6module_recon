/**
 * @file test_psf_modes.cpp
 * @brief PSF 模式行为测试（Operator）
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "recon/ProjectionOps.h"
#include "recon/ReconConfig.h"
#include "recon/BedOffsets.h"
#include "filters/Convolution.h"

using spect::apply_psf_backward;
using spect::apply_psf_forward;
using spect::filters::make_psf_kernels;

namespace spect {
namespace test {

namespace {

std::vector<double> makePatternVolume(int Nx, int Ny, int Nz) {
    std::vector<double> f(static_cast<size_t>(Nx) * Ny * Nz, 0.0);
    const int cx = Nx / 2;
    const int cy = Ny / 2;
    const int cz = Nz / 2;
    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                const size_t idx = static_cast<size_t>(x) +
                                   static_cast<size_t>(Nx) * y +
                                   static_cast<size_t>(Nx) * Ny * z;
                if (x == cx && y == cy && z == cz)
                    f[idx] = 1.0;
            }
        }
    }
    return f;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double maxDiff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        maxDiff = std::max(maxDiff, std::abs(a[i] - b[i]));
    }
    return maxDiff;
}

PSFConf makeBasicPSF() {
    PSFConf psf;
    psf.mode = PSFMode::Operator;
    psf.fwhm_x = 2.0;
    psf.fwhm_y = 2.0;
    psf.fwhm_z = 2.0;
    psf.vox_x = 1.0;
    psf.vox_y = 1.0;
    psf.vox_z = 1.0;
    psf.min_sigmas = 2.0;
    return psf;
}

} // namespace

TEST(PSFModeTest, OperatorModeBlursForwardAndBackward) {
    const int Nx = 5, Ny = 3, Nz = 1;
    const auto input = makePatternVolume(Nx, Ny, Nz);

    ReconOpts opts;
    opts.threads = 1;
    opts.psf = makeBasicPSF();
    opts.psf.mode = PSFMode::Operator;

    const auto kernels = make_psf_kernels(opts.psf);

    std::vector<double> forward;
    apply_psf_forward(input, forward, Nx, Ny, Nz, opts, &kernels, opts.threads);
    EXPECT_GT(maxAbsDiff(input, forward), 1e-12);

    std::vector<double> backward = input;
    apply_psf_backward(backward, Nx, Ny, Nz, opts, &kernels, opts.threads);
    EXPECT_GT(maxAbsDiff(input, backward), 1e-12);
}

TEST(PSFModeTest, OperatorPsfForwardBackwardAreAdjointAtBoundaries) {
    const int Nx = 5, Ny = 4, Nz = 3;
    const size_t total = static_cast<size_t>(Nx) * Ny * Nz;
    std::vector<double> x(total);
    std::vector<double> y(total);
    for (size_t i = 0; i < total; ++i) {
        x[i] = 0.2 + static_cast<double>((7 * i + 3) % 19) / 11.0;
        y[i] = 0.1 + static_cast<double>((5 * i + 1) % 23) / 13.0;
    }
    x.front() += 3.0;
    x.back() += 2.0;
    y[static_cast<size_t>(Nx) * Ny - 1] += 4.0;

    ReconOpts opts;
    opts.threads = 1;
    opts.psf = makeBasicPSF();
    const auto kernels = make_psf_kernels(opts.psf);

    std::vector<double> gx;
    apply_psf_forward(x, gx, Nx, Ny, Nz, opts, &kernels, opts.threads);
    auto gty = y;
    apply_psf_backward(gty, Nx, Ny, Nz, opts, &kernels, opts.threads);

    const double lhs = std::inner_product(gx.begin(), gx.end(), y.begin(), 0.0);
    const double rhs = std::inner_product(x.begin(), x.end(), gty.begin(), 0.0);
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    // Production PSF staging is float32 on CUDA; verify adjointness at its
    // numerical precision rather than the retired CPU-double path.
    EXPECT_NEAR(lhs, rhs, 2e-8 * scale);
}

TEST(PSFModeTest, GaussianKernelUsesConfiguredSigmaTruncation) {
    constexpr double fwhmForOneVoxelSigma = 2.354820045;
    EXPECT_EQ(spect::filters::makeGaussian1D(fwhmForOneVoxelSigma, 1.0, 2.0).size(),
              5u);
    EXPECT_EQ(spect::filters::makeGaussian1D(fwhmForOneVoxelSigma, 1.0).size(),
              9u);
}

TEST(PSFModeTest, XyOnlyModeBuildsIdentityZKernel) {
    auto psf = makeBasicPSF();
    psf.z_blur = false;
    const auto kernels = make_psf_kernels(psf);
    EXPECT_GT(kernels.kx.size(), 1u);
    EXPECT_GT(kernels.ky.size(), 1u);
    EXPECT_EQ(kernels.kz, (std::vector<double>{1.0}));
}

TEST(PSFModeTest, GlobalVolumePsfSpansBedStartWhereLocalBedPsfMissesHalo) {
    const int Nx = 3;
    const int Ny = 3;
    const int baseNz = 5;
    const int bedCount = 2;
    const std::vector<int> overlaps{2};
    const auto starts = spect::app::computeBedZOffsets(bedCount, baseNz, overlaps);
    ASSERT_EQ(starts, (std::vector<int>{0, 3}));
    const int globalNz =
        bedCount * baseNz - spect::app::computeTotalOverlap(bedCount, overlaps);
    ASSERT_EQ(globalNz, 8);

    auto idx = [Nx, Ny](int x, int y, int z) {
        return static_cast<size_t>(x) +
               static_cast<size_t>(Nx) * static_cast<size_t>(y) +
               static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                   static_cast<size_t>(z);
    };

    std::vector<double> global(static_cast<size_t>(Nx) * Ny * globalNz, 0.0);
    global[idx(1, 1, 2)] = 1.0; // One slice before bed 1 starts.

    ReconOpts opts;
    opts.threads = 1;
    opts.psf.mode = PSFMode::Operator;
    opts.psf.fwhm_x = 0.0;
    opts.psf.fwhm_y = 0.0;
    opts.psf.fwhm_z = 2.0;
    opts.psf.vox_x = 1.0;
    opts.psf.vox_y = 1.0;
    opts.psf.vox_z = 1.0;
    const auto kernels = make_psf_kernels(opts.psf);

    std::vector<double> globalBlurred;
    apply_psf_forward(global, globalBlurred, Nx, Ny, globalNz, opts, &kernels,
                      opts.threads);
    const auto bed1FromGlobal = spect::app::extractBedVolumeFromGlobal(
        globalBlurred, Nx, Ny, baseNz, starts[1]);

    const auto bed1Local = spect::app::extractBedVolumeFromGlobal(
        global, Nx, Ny, baseNz, starts[1]);
    std::vector<double> bed1LocalBlurred;
    apply_psf_forward(bed1Local, bed1LocalBlurred, Nx, Ny, baseNz, opts,
                      &kernels, opts.threads);

    EXPECT_GT(bed1FromGlobal[idx(1, 1, 0)], 0.0);
    EXPECT_NEAR(bed1LocalBlurred[idx(1, 1, 0)], 0.0, 1e-12);
}

} // namespace test
} // namespace spect
