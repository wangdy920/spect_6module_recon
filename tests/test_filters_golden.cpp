/**
 * @file test_filters_golden.cpp
 * @brief Filters golden tests (conv3d + Butterworth mask)
 */

#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "filters/ButterworthFilter.h"
#include "filters/Convolution.h"

namespace spect {
namespace test {
namespace {

inline int mirror_index(int i, int n) {
    if (i < 0)
        return -i;
    if (i >= n)
        return 2 * n - 2 - i;
    return i;
}

inline size_t idx3(int x, int y, int z, int Nx, int Ny) {
    return static_cast<size_t>(x) +
           static_cast<size_t>(Nx) * static_cast<size_t>(y) +
           static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
               static_cast<size_t>(z);
}

std::vector<double> conv3d_reference(const std::vector<double>& f,
                                     int Nx,
                                     int Ny,
                                     int Nz,
                                     const std::vector<double>& kx,
                                     const std::vector<double>& ky,
                                     const std::vector<double>& kz) {
    std::vector<double> out(static_cast<size_t>(Nx) * Ny * Nz, 0.0);
    const int rx = static_cast<int>(kx.size()) / 2;
    const int ry = static_cast<int>(ky.size()) / 2;
    const int rz = static_cast<int>(kz.size()) / 2;

    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                double sum = 0.0;
                for (int tz = -rz; tz <= rz; ++tz) {
                    const int zz = mirror_index(z + tz, Nz);
                    const double kzv = kz[static_cast<size_t>(tz + rz)];
                    for (int ty = -ry; ty <= ry; ++ty) {
                        const int yy = mirror_index(y + ty, Ny);
                        const double kyv = ky[static_cast<size_t>(ty + ry)];
                        for (int tx = -rx; tx <= rx; ++tx) {
                            const int xx = mirror_index(x + tx, Nx);
                            const double kxv =
                                kx[static_cast<size_t>(tx + rx)];
                            sum += f[idx3(xx, yy, zz, Nx, Ny)] * kxv * kyv * kzv;
                        }
                    }
                }
                out[idx3(x, y, z, Nx, Ny)] = sum;
            }
        }
    }
    return out;
}

double butterworth_expected(int x,
                            int y,
                            int z,
                            int Nx,
                            int Ny,
                            int Nz,
                            const spect::filters::ButterworthConf& bw) {
    const double denomX = 0.5 * std::max(1, Nx);
    const double denomY = 0.5 * std::max(1, Ny);
    const double denomZ = 0.5 * std::max(1, Nz);
    const double norm = std::sqrt(3.0);
    const int n = std::max(1, bw.order);
    const double fc_lo = bw.cutoff_lo_frac;
    const double fc_hi = bw.cutoff_hi_frac;

    const int kx = (x <= Nx / 2) ? x : x - Nx;
    const int ky = (y <= Ny / 2) ? y : y - Ny;
    const int kz = (z <= Nz / 2) ? z : z - Nz;
    const double fx = static_cast<double>(kx) / denomX;
    const double fy = static_cast<double>(ky) / denomY;
    const double fz = static_cast<double>(kz) / denomZ;
    const double r = std::sqrt(fx * fx + fy * fy + fz * fz) / norm;

    auto H_lp = [&](double rr, double fc) -> double {
        if (fc <= 0.0)
            return 0.0;
        if (rr <= 0.0)
            return 1.0;
        const double ratio = rr / fc;
        const double t = std::pow(ratio, 2 * n);
        return 1.0 / (1.0 + t);
    };
    auto H_hp = [&](double rr, double fc) -> double { return 1.0 - H_lp(rr, fc); };

    switch (bw.type) {
    case spect::filters::BWType::Lowpass:
        return H_lp(r, fc_hi);
    case spect::filters::BWType::Highpass:
        return H_hp(r, fc_hi);
    case spect::filters::BWType::Bandpass:
        return H_hp(r, fc_lo) * H_lp(r, fc_hi);
    case spect::filters::BWType::Bandstop:
        return 1.0 - (H_hp(r, fc_lo) * H_lp(r, fc_hi));
    default:
        return 1.0;
    }
}

}  // namespace

TEST(FiltersGolden, Conv3DSeparableInplaceMatchesReference) {
    const int Nx = 4;
    const int Ny = 3;
    const int Nz = 2;
    const std::vector<double> k = {0.25, 0.5, 0.25};

    std::vector<double> f(static_cast<size_t>(Nx) * Ny * Nz, 0.0);
    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                f[idx3(x, y, z, Nx, Ny)] =
                    static_cast<double>(x + 10 * y + 100 * z);
            }
        }
    }

    const auto expected = conv3d_reference(f, Nx, Ny, Nz, k, k, k);
    auto actual = f;
    spect::filters::conv3d_separable_inplace(actual, Nx, Ny, Nz, k, k, k, 1);

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-12);
    }
}

TEST(FiltersGolden, ButterworthMaskSamplesMatchFormula) {
    const int Nx = 6;
    const int Ny = 6;
    const int Nz = 4;

    spect::filters::ButterworthConf bw;
    bw.enabled = true;
    bw.type = spect::filters::BWType::Bandpass;
    bw.order = 2;
    bw.cutoff_lo_frac = 0.2;
    bw.cutoff_hi_frac = 0.6;

    const auto mask =
        spect::filters::make_butterworth_mask_3d(Nx, Ny, Nz, {1.0, 1.0, 1.0}, bw,
                                                 1);
    const std::vector<std::tuple<int, int, int>> samples = {
        {0, 0, 0}, {1, 0, 0}, {Nx / 2, Ny / 2, Nz / 2}, {Nx - 1, Ny - 1, Nz - 1}};

    for (const auto& s : samples) {
        const int x = std::get<0>(s);
        const int y = std::get<1>(s);
        const int z = std::get<2>(s);
        const double expected = butterworth_expected(x, y, z, Nx, Ny, Nz, bw);
        const double actual = mask[idx3(x, y, z, Nx, Ny)];
        EXPECT_NEAR(actual, expected, 1e-12);
        EXPECT_GE(actual, 0.0);
        EXPECT_LE(actual, 1.0);
    }
}

TEST(FiltersGolden, ButterworthFFTWThreadedPathMatchesSingleThreadReference) {
    const int Nx = 24;
    const int Ny = 20;
    const int Nz = 12;

    spect::filters::ButterworthConf bw;
    bw.enabled = true;
    bw.type = spect::filters::BWType::Lowpass;
    bw.order = 2;
    bw.cutoff_hi_frac = 0.35;

    std::vector<double> input(static_cast<size_t>(Nx) * Ny * Nz, 0.0);
    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                input[idx3(x, y, z, Nx, Ny)] =
                    0.1 * static_cast<double>(x) +
                    0.2 * static_cast<double>(y) +
                    0.3 * static_cast<double>(z) +
                    std::sin(static_cast<double>(x + 2 * y + 3 * z));
            }
        }
    }

    auto single_thread = input;
    auto threaded = input;

    spect::filters::apply_butterworth_inplace_fftw(
        single_thread, Nx, Ny, Nz, {1.0, 1.0, 1.0}, bw, 1);
    spect::filters::apply_butterworth_inplace_fftw(
        threaded, Nx, Ny, Nz, {1.0, 1.0, 1.0}, bw, 4);

    ASSERT_EQ(single_thread.size(), threaded.size());
    for (size_t i = 0; i < single_thread.size(); ++i) {
        EXPECT_NEAR(threaded[i], single_thread[i], 1e-9);
        EXPECT_GE(threaded[i], 0.0);
    }
}

}  // namespace test
}  // namespace spect
