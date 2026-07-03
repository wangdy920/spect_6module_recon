#include <gtest/gtest.h>

#include <vector>

#include "app/ProjectionOps.h"
#include "app/ReconConfig.h"
#include "io/CSRBuilder.h"
#include "sparse/SparseOpsCPU.h"
#include "sparse/SparseOpsRef.h"

namespace spect {
namespace test {

namespace {

io::CSR makeOverlapToyCSR() {
    io::CSR A;
    A.rows = 2;
    A.cols = 3;
    A.row_ptr = {0, 1, 2};
    A.col_idx = {2, 0};
    A.val = {1.0f, 1.0f};
    return A;
}

io::CSRView toView(const io::CSR& A) {
    io::CSRView view;
    view.rows = A.rows;
    view.cols = A.cols;
    view.nnz = static_cast<uint64_t>(A.val.size());
    view.row_ptr = A.row_ptr.data();
    view.col_idx = A.col_idx.data();
    view.val = A.val.data();
    return view;
}

void expectVectorEq(const std::vector<double>& actual,
                    const std::vector<double>& expected,
                    double tol = 1e-12) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol) << "index=" << i;
    }
}

} // namespace

TEST(MultiBedOverlapTest, SubsetSensitivityOverlapAccumulates) {
    const io::CSR A = makeOverlapToyCSR();
    const io::CSRView view = toView(A);

    ReconOpts opts;
    opts.bedCount = 2;
    opts.bedOffsetVoxels = {0, 2};

    // baseRows=2:
    // bed0 rows: 0,1
    // bed1 rows: 2,3
    // with offsets [0,2], row0->global col2, row3(row1@bed1)->global col2
    const std::vector<int> subset_rows = {0, 1, 2, 3};
    const int N = 5; // global voxels

    const std::vector<double> sens = computeSubsetSensitivity(
        view, subset_rows, N, 1, opts);
    const std::vector<double> expected = {1.0, 0.0, 2.0, 0.0, 1.0};
    expectVectorEq(sens, expected);

    SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &opts.bedOffsetVoxels;

    std::vector<double> sens_ref(static_cast<size_t>(N), 0.0);
    std::vector<double> sens_cpu(static_cast<size_t>(N), 0.0);
    getSparseOpsRefImpl().computeSubsetSensitivity(
        view, subset_rows, N, ctx, sens_ref.data(), 1);
    getSparseOpsCPUImpl().computeSubsetSensitivity(
        view, subset_rows, N, ctx, sens_cpu.data(), 4);

    expectVectorEq(sens_ref, expected);
    expectVectorEq(sens_cpu, expected);
}

TEST(MultiBedOverlapTest, ImpulseMapsToSameGlobalVoxelInOverlap) {
    const io::CSR A = makeOverlapToyCSR();
    const io::CSRView view = toView(A);

    const std::vector<long long> offsets = {0, 2};
    SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &offsets;

    // bed0 use row0 => local col2 -> global col2
    // bed1 use row1 => local col0 + offset2 -> global col2
    const std::vector<int> subset_rows = {0, 3};

    std::vector<double> x_global = {0.0, 0.0, 1.0, 0.0, 0.0}; // impulse@overlap
    std::vector<double> y_ref(subset_rows.size(), 0.0);
    std::vector<double> y_cpu(subset_rows.size(), 0.0);

    getSparseOpsRefImpl().spmvSubset(
        view, x_global.data(), subset_rows, ctx, y_ref.data(), 1);
    getSparseOpsCPUImpl().spmvSubset(
        view, x_global.data(), subset_rows, ctx, y_cpu.data(), 4);

    const std::vector<double> expected_forward = {1.0, 1.0};
    expectVectorEq(y_ref, expected_forward);
    expectVectorEq(y_cpu, expected_forward);

    std::vector<double> ratio = {1.0, 1.0};
    std::vector<double> back_ref(5, 0.0);
    std::vector<double> back_cpu(5, 0.0);

    getSparseOpsRefImpl().spmvTSubset(
        view, ratio.data(), subset_rows, ctx, back_ref.data(), 1);
    getSparseOpsCPUImpl().spmvTSubset(
        view, ratio.data(), subset_rows, ctx, back_cpu.data(), 4);

    const std::vector<double> expected_back = {0.0, 0.0, 2.0, 0.0, 0.0};
    expectVectorEq(back_ref, expected_back);
    expectVectorEq(back_cpu, expected_back);
}

} // namespace test
} // namespace spect

