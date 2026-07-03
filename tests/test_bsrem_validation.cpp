/**
 * @file test_bsrem_validation.cpp
 * @brief BSREM 输入校验测试
 */

#include <gtest/gtest.h>
#include <vector>

#include "app/BSREMReconstructor.h"
#include "app/ReconConfig.h"
#include "io/CSRBuilder.h"

using spect::BSREM_step;
using spect::ReconOpts;
using spect::io::CSR;

namespace spect {
namespace test {

namespace {

CSR makeTinyCSR() {
    CSR A;
    A.rows = 1;
    A.cols = 2;
    A.row_ptr = {0, 2};
    A.col_idx = {0, 1};
    A.val = {1.0f, 1.0f};
    return A;
}

ReconOpts makeDefaultOpts() {
    ReconOpts opts;
    opts.threads = 1;
    opts.bedCount = 1;
    return opts;
}

} // namespace

TEST(BSREMValidationTest, ThrowsOnSizeMismatch) {
    CSR A = makeTinyCSR();
    ReconOpts opts = makeDefaultOpts();

    const int Nx = 2, Ny = 1, Nz = 1;
    const int N = Nx * Ny * Nz;

    std::vector<int> subset_rows = {0};
    std::vector<double> p_ok = {1.0};
    std::vector<double> f_ok = {1.0, 1.0};
    std::vector<double> global_denom_ok = {1.0, 1.0};
    std::vector<double> subset_denom_ok = {1.0, 1.0};
    std::vector<double> buffer;

    // p size mismatch
    EXPECT_THROW(
        BSREM_step(A, subset_rows, std::vector<double>{}, f_ok,
                   global_denom_ok, subset_denom_ok, N, Nx, Ny, Nz,
                   opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer, 1.0),
        std::runtime_error);

    // f size mismatch
    EXPECT_THROW(
        BSREM_step(A, subset_rows, p_ok, std::vector<double>{1.0},
                   global_denom_ok, subset_denom_ok, N, Nx, Ny, Nz,
                   opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer, 1.0),
        std::runtime_error);

    // global denom mismatch
    EXPECT_THROW(
        BSREM_step(A, subset_rows, p_ok, f_ok,
                   std::vector<double>{1.0}, subset_denom_ok, N, Nx, Ny, Nz,
                   opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer, 1.0),
        std::runtime_error);

    // subset denom mismatch
    EXPECT_THROW(
        BSREM_step(A, subset_rows, p_ok, f_ok,
                   global_denom_ok, std::vector<double>{1.0}, N, Nx, Ny, Nz,
                   opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer, 1.0),
        std::runtime_error);

    // buffer null
    EXPECT_THROW(
        BSREM_step(A, subset_rows, p_ok, f_ok,
                   global_denom_ok, subset_denom_ok, N, Nx, Ny, Nz,
                   opts, nullptr, nullptr, 0.0, nullptr, nullptr, nullptr, 1.0),
        std::runtime_error);
}

TEST(BSREMValidationTest, ThrowsOnOffsetSizeMismatch) {
    CSR A = makeTinyCSR();
    ReconOpts opts = makeDefaultOpts();
    opts.bedCount = 2;
    opts.bedOffsetVoxels = {0, 2};

    const int Nx = 3, Ny = 1, Nz = 1;
    const int N = Nx * Ny * Nz;

    std::vector<int> subset_rows = {0};
    std::vector<double> p_ok = {1.0, 1.0};
    std::vector<double> f_small = {1.0, 1.0, 1.0}; // size 3 < offset+cols (4)
    std::vector<double> global_denom = {1.0, 1.0, 1.0};
    std::vector<double> subset_denom = {1.0, 1.0, 1.0};
    std::vector<double> buffer;

    EXPECT_THROW(
        BSREM_step(A, subset_rows, p_ok, f_small,
                   global_denom, subset_denom, N, Nx, Ny, Nz,
                   opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer, 1.0),
        std::runtime_error);
}

TEST(BSREMValidationTest, ThrowsOnKappaSizeMismatch) {
    CSR A = makeTinyCSR();
    ReconOpts opts = makeDefaultOpts();

    const int Nx = 2, Ny = 1, Nz = 1;
    const int N = Nx * Ny * Nz;

    std::vector<int> subset_rows = {0};
    std::vector<double> p_ok = {2.0};
    std::vector<double> f_ok = {1.0, 1.0};
    std::vector<double> global_denom = {1.0, 1.0};
    std::vector<double> subset_denom = {1.0, 1.0};
    std::vector<double> grad = {0.1, 0.1};
    std::vector<double> bad_kappa = {1.0};
    std::vector<double> buffer;

    EXPECT_THROW(
        BSREM_step(A, subset_rows, p_ok, f_ok, global_denom, subset_denom, N,
                   Nx, Ny, Nz, opts, nullptr, &grad, 0.1, nullptr, &bad_kappa,
                   &buffer, 1.0),
        std::runtime_error);
}

TEST(BSREMValidationTest, ZeroSensitivityVoxelIsForcedToZero) {
    CSR A;
    A.rows = 1;
    A.cols = 2;
    A.row_ptr = {0, 1};
    A.col_idx = {0};
    A.val = {1.0f};

    ReconOpts opts = makeDefaultOpts();
    const int Nx = 2, Ny = 1, Nz = 1;
    const int N = Nx * Ny * Nz;

    std::vector<int> subset_rows = {0};
    std::vector<double> p = {1.0};
    std::vector<double> f = {1.0, 1.0};
    std::vector<double> global_denom = {1.0, 0.0};
    std::vector<double> subset_denom = {1.0, 0.0};
    std::vector<double> buffer;

    BSREM_step(A, subset_rows, p, f, global_denom, subset_denom, N, Nx, Ny, Nz,
               opts, nullptr, nullptr, 0.0, nullptr, nullptr, &buffer, 1.0);

    EXPECT_DOUBLE_EQ(f[0], 1.0);
    EXPECT_DOUBLE_EQ(f[1], 0.0);
}

} // namespace test
} // namespace spect
