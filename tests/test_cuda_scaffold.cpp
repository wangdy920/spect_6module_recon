#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <vector>

#include "app/BSREMReconstructor.h"
#include "app/OSEMReconstructor.h"
#include "app/ProjectionOps.h"
#include "app/ReconConfig.h"
#include "cuda/CudaConfig.h"
#include "cuda/CompactSymmetricProjector.h"
#include "cuda/ProjDomainChunked.h"
#include "cuda/VolumeOpsCUDA.h"
#include "filters/ButterworthFilter.h"
#include "filters/Convolution.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "regularizer/RegularizerCompat.h"
#include "sparse/SparseOpsCPU.h"
#include "sparse/SparseOpsBackend.h"

namespace {

spect::CSRView makeTinyMatrix(const int64_t *rowPtr, const int32_t *colIdx,
                              const double *values, int64_t rows,
                              int64_t cols, int64_t nnz) {
    spect::CSRView matrix{};
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = nnz;
    matrix.row_ptr = rowPtr;
    matrix.col_idx = colIdx;
    matrix.values = values;
    return matrix;
}

spect::io::CSRView makeTinyIoMatrix(const uint32_t *rowPtr, const uint32_t *colIdx,
                                    const float *values, int rows, int cols,
                                    uint64_t nnz) {
    spect::io::CSRView matrix{};
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = nnz;
    matrix.row_ptr = rowPtr;
    matrix.col_idx = colIdx;
    matrix.val = values;
    return matrix;
}

struct OwnedTinyIoMatrix {
    std::vector<uint32_t> rowPtr;
    std::vector<uint32_t> colIdx;
    std::vector<float> values;
    int rows = 0;
    int cols = 0;

    spect::io::CSRView view() const {
        return makeTinyIoMatrix(rowPtr.data(), colIdx.data(), values.data(), rows,
                                cols, values.size());
    }
};

OwnedTinyIoMatrix makeChunkedTinyIoMatrix() {
    OwnedTinyIoMatrix matrix;
    matrix.rowPtr = {0, 2, 4, 6, 8};
    matrix.colIdx = {0, 1, 1, 2, 0, 2, 0, 1};
    matrix.values = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 1.0f, 1.0f, 4.0f};
    matrix.rows = 4;
    matrix.cols = 3;
    return matrix;
}

class ScopedCudaChunkBudgetOverride {
public:
    explicit ScopedCudaChunkBudgetOverride(uint64_t nnzBudget) {
        spect::cuda::setCudaChunkNnzBudgetOverrideForTestingCUDA(nnzBudget);
    }

    ~ScopedCudaChunkBudgetOverride() {
        spect::cuda::setCudaChunkNnzBudgetOverrideForTestingCUDA(0);
    }
};

class ScopedCudaPreparedHostCacheOverride {
public:
    explicit ScopedCudaPreparedHostCacheOverride(uint64_t bytes) {
        spect::cuda::setCudaPreparedChunkHostCacheBytesOverrideForTestingCUDA(bytes);
    }

    ~ScopedCudaPreparedHostCacheOverride() {
        spect::cuda::setCudaPreparedChunkHostCacheBytesOverrideForTestingCUDA(0);
    }
};

class ScopedCudaHostChunkBuildThreadsOverride {
public:
    explicit ScopedCudaHostChunkBuildThreadsOverride(int threads) {
        spect::cuda::setCudaHostChunkBuildThreadsOverrideForTestingCUDA(threads);
    }

    ~ScopedCudaHostChunkBuildThreadsOverride() {
        spect::cuda::setCudaHostChunkBuildThreadsOverrideForTestingCUDA(0);
    }
};

struct ChunkedEmProbeResult {
    std::vector<double> image;
    std::vector<double> numer;
    spect::cuda::CudaChunkStageTimingSnapshot timing;
};

ChunkedEmProbeResult runCpuChunkedEmProbe(
    const spect::io::CSRView& matrix, const std::vector<double>& projections,
    const std::vector<double>& initialImage, const std::vector<double>& denom,
    const std::vector<long long>& bedOffsets, int iterations) {
    const std::vector<int> subsetRows = {0, 1, 2, 3, 4, 5, 6, 7};
    spect::SparseSubsetContext ctx;
    ctx.bedCount = static_cast<int>(bedOffsets.size());
    ctx.bedOffsetVoxels = &bedOffsets;

    std::vector<double> image = initialImage;
    std::vector<double> yhat(subsetRows.size(), 0.0);
    std::vector<double> ratio(subsetRows.size(), 0.0);
    std::vector<double> numer(initialImage.size(), 0.0);
    for (int iter = 0; iter < iterations; ++iter) {
        std::fill(yhat.begin(), yhat.end(), 0.0);
        std::fill(ratio.begin(), ratio.end(), 0.0);
        std::fill(numer.begin(), numer.end(), 0.0);
        spect::getSparseOpsCPU().spmvSubset(matrix, image.data(), subsetRows, ctx,
                                            yhat.data(), 1);
        for (size_t i = 0; i < ratio.size(); ++i) {
            ratio[i] = projections[i] / std::max(yhat[i], 1e-12);
        }
        spect::getSparseOpsCPU().spmvTSubset(matrix, ratio.data(), subsetRows, ctx,
                                             numer.data(), 1);
        for (size_t voxel = 0; voxel < image.size(); ++voxel) {
            if (denom[voxel] > 1e-12) {
                image[voxel] *= numer[voxel] / denom[voxel];
            }
        }
    }
    return {image, numer, {}};
}

ChunkedEmProbeResult runChunkedEmProbe(
    const spect::io::CSRView& matrix, const std::vector<double>& projections,
    const std::vector<double>& initialImage, const std::vector<double>& denom,
    const std::vector<long long>& bedOffsets, int iterations, bool warmup) {
    if (warmup) {
        EXPECT_TRUE(spect::cuda::warmupFullSubsetChunkCacheCUDA(matrix, 2, nullptr, 0,
                                                                1));
    }
    std::vector<double> image = initialImage;
    std::vector<double> numer(initialImage.size(), 0.0);
    spect::cuda::resetCudaChunkStageTimingCUDA();
    for (int iter = 0; iter < iterations; ++iter) {
        std::fill(numer.begin(), numer.end(), 0.0);
        EXPECT_TRUE(spect::cuda::runFullSubsetEmNumerCUDA(
            matrix, projections, image.data(), 2, &bedOffsets, nullptr, 0, 1,
            numer.data()));
        for (size_t voxel = 0; voxel < image.size(); ++voxel) {
            if (denom[voxel] > 1e-12) {
                image[voxel] *= numer[voxel] / denom[voxel];
            }
        }
    }
    return {image, numer, spect::cuda::getCudaChunkStageTimingSnapshotCUDA()};
}

TEST(CudaScaffoldTest, BackendIsSelectableWhenExperimentalImplIsLinked) {
    EXPECT_TRUE(spect::isCudaSparseOpsAvailable());

    const auto& ops = spect::getSparseOpsCUDA();

    EXPECT_EQ(ops.backend(), spect::BackendType::CUDA);
}

TEST(CudaScaffoldTest, SpmvMatchesExpectedWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const auto& ops = spect::getSparseOpsCUDA();

    const int64_t rowPtr[] = {0, 2, 4};
    const int32_t colIdx[] = {0, 2, 1, 2};
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    const auto matrix = makeTinyMatrix(rowPtr, colIdx, values, 2, 3, 4);

    const double x[3] = {1.0, 2.0, 3.0};
    double y[2] = {0.0, 0.0};

    ops.spmv(matrix, x, y, 1);

    EXPECT_DOUBLE_EQ(y[0], 7.0);
    EXPECT_DOUBLE_EQ(y[1], 18.0);
}

TEST(CudaScaffoldTest, SpmvTransposeAndNormMatchExpectedWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const auto& ops = spect::getSparseOpsCUDA();

    const int64_t rowPtr[] = {0, 2, 4};
    const int32_t colIdx[] = {0, 2, 1, 2};
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    const auto matrix = makeTinyMatrix(rowPtr, colIdx, values, 2, 3, 4);

    const double xT[2] = {5.0, 6.0};
    double yT[3] = {0.0, 0.0, 0.0};
    double norm[3] = {0.0, 0.0, 0.0};

    ops.spmvT(matrix, xT, yT, 1);
    ops.computeNormAT(matrix, norm, 1);

    EXPECT_DOUBLE_EQ(yT[0], 5.0);
    EXPECT_DOUBLE_EQ(yT[1], 18.0);
    EXPECT_DOUBLE_EQ(yT[2], 34.0);

    EXPECT_DOUBLE_EQ(norm[0], 1.0);
    EXPECT_DOUBLE_EQ(norm[1], 3.0);
    EXPECT_DOUBLE_EQ(norm[2], 6.0);
}

TEST(CudaScaffoldTest, RepeatedSpmvCallsReuseSameMatrixCorrectly) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const auto& ops = spect::getSparseOpsCUDA();

    const int64_t rowPtr[] = {0, 2, 4};
    const int32_t colIdx[] = {0, 2, 1, 2};
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    const auto matrix = makeTinyMatrix(rowPtr, colIdx, values, 2, 3, 4);

    const double x1[3] = {1.0, 2.0, 3.0};
    const double x2[3] = {2.0, 1.0, 0.5};
    double y1[2] = {0.0, 0.0};
    double y2[2] = {123.0, 456.0};

    ops.spmv(matrix, x1, y1, 1);
    ops.spmv(matrix, x2, y2, 1);

    EXPECT_DOUBLE_EQ(y1[0], 7.0);
    EXPECT_DOUBLE_EQ(y1[1], 18.0);
    EXPECT_DOUBLE_EQ(y2[0], 3.0);
    EXPECT_DOUBLE_EQ(y2[1], 5.0);
}

TEST(CudaScaffoldTest, OsemSingleBedSingleSubsetMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const uint32_t rowPtr[] = {0, 2, 4};
    const uint32_t colIdx[] = {0, 2, 1, 2};
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 2, 3, 4);

    std::vector<int> subsetRows = {0, 1};
    std::vector<double> projections = {7.0, 18.0};
    std::vector<double> imageCpu = {1.0, 1.0, 1.0};
    std::vector<double> imageCuda = imageCpu;
    std::vector<double> denomCpu(3, 0.0);
    std::vector<double> denomCuda(3, 0.0);

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 1;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();

    spect::ReconOpts optsCuda = optsCpu;
    optsCuda.sparseOps = &spect::getSparseOpsCUDA();

    optsCpu.sparseOps->computeNormAT(matrix, denomCpu.data(), 1);
    optsCuda.sparseOps->computeNormAT(matrix, denomCuda.data(), 1);

    spect::OSEM_step(matrix, subsetRows, projections, imageCpu, denomCpu,
                     3, 1, 1, optsCpu, nullptr, nullptr, 0.0, nullptr, 1.0);
    spect::OSEM_step(matrix, subsetRows, projections, imageCuda, denomCuda,
                     3, 1, 1, optsCuda, nullptr, nullptr, 0.0, nullptr, 1.0);

    ASSERT_EQ(imageCpu.size(), imageCuda.size());
    for (size_t i = 0; i < imageCpu.size(); ++i) {
        EXPECT_NEAR(imageCuda[i], imageCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, OsemSingleBedMultiSubsetMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const uint32_t rowPtr[] = {0, 2, 4, 6, 8};
    const uint32_t colIdx[] = {0, 1, 1, 2, 0, 2, 0, 1};
    const float values[] = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 1.0f, 1.0f, 4.0f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 4, 3, 8);

    const std::vector<double> truth = {1.0, 2.0, 1.5};
    std::vector<double> projections(4, 0.0);
    spect::getSparseOpsCPU().spmv(matrix, truth.data(), projections.data(), 1);

    std::vector<double> imageCpu = {1.0, 1.0, 1.0};
    std::vector<double> imageCuda = imageCpu;
    std::vector<double> denom(3, 0.0);
    spect::getSparseOpsCPU().computeNormAT(matrix, denom.data(), 1);

    const auto subsets = spect::make_simple_subsets(matrix.rows, 2);

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 1;
    optsCpu.n_subsets = 2;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();

    spect::ReconOpts optsCuda = optsCpu;
    optsCuda.sparseOps = &spect::getSparseOpsCUDA();

    for (const auto& subsetRows : subsets) {
        spect::OSEM_step(matrix, subsetRows, projections, imageCpu, denom,
                         3, 1, 1, optsCpu, nullptr, nullptr, 0.0, nullptr,
                         1.0);
        spect::OSEM_step(matrix, subsetRows, projections, imageCuda, denom,
                         3, 1, 1, optsCuda, nullptr, nullptr, 0.0, nullptr,
                         1.0);
    }

    ASSERT_EQ(imageCpu.size(), imageCuda.size());
    for (size_t i = 0; i < imageCpu.size(); ++i) {
        EXPECT_NEAR(imageCuda[i], imageCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, OsemMultiBedFullSubsetMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const uint32_t rowPtr[] = {0, 1, 2};
    const uint32_t colIdx[] = {2, 0};
    const float values[] = {1.0f, 1.0f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 2, 3, 2);

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 2;
    optsCpu.n_subsets = 1;
    optsCpu.bedOffsetVoxels = {0, 2};
    optsCpu.sparseOps = &spect::getSparseOpsCPU();

    spect::ReconOpts optsCuda = optsCpu;
    optsCuda.sparseOps = &spect::getSparseOpsCUDA();

    const std::vector<int> subsetRows = {0, 1, 2, 3};
    const int globalN = 5;
    const std::vector<double> truth = {0.5, 0.25, 1.0, 0.75, 0.5};

    spect::SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &optsCpu.bedOffsetVoxels;

    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), 1);

    std::vector<double> imageCpu(globalN, 1.0);
    std::vector<double> imageCuda = imageCpu;
    std::vector<double> denom = spect::computeSubsetSensitivity(
        matrix, subsetRows, globalN, 1, optsCpu);

    spect::OSEM_step(matrix, subsetRows, projections, imageCpu, denom,
                     globalN, 1, 1, optsCpu, nullptr, nullptr, 0.0, nullptr,
                     1.0);
    spect::OSEM_step(matrix, subsetRows, projections, imageCuda, denom,
                     globalN, 1, 1, optsCuda, nullptr, nullptr, 0.0, nullptr,
                     1.0);

    ASSERT_EQ(imageCpu.size(), imageCuda.size());
    for (size_t i = 0; i < imageCpu.size(); ++i) {
        EXPECT_NEAR(imageCuda[i], imageCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, FullSubsetEmNumerMultiBedMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const uint32_t rowPtr[] = {0, 2, 4};
    const uint32_t colIdx[] = {0, 2, 1, 2};
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 2, 3, 4);

    const std::vector<int> subsetRows = {0, 1, 2, 3};
    const std::vector<long long> bedOffsets = {0, 2};
    const int globalN = 5;

    spect::SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &bedOffsets;

    const std::vector<double> truth = {0.6, 0.2, 1.1, 0.8, 0.4};
    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), 1);

    const std::vector<double> image = {1.0, 0.9, 0.7, 1.1, 0.8};
    std::vector<double> yhatCpu(subsetRows.size(), 0.0);
    std::vector<double> ratioCpu(subsetRows.size(), 0.0);
    std::vector<double> numerCpu(globalN, 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, image.data(), subsetRows, ctx,
                                        yhatCpu.data(), 1);
    for (size_t i = 0; i < subsetRows.size(); ++i) {
        ratioCpu[i] = projections[i] / yhatCpu[i];
    }
    spect::getSparseOpsCPU().spmvTSubset(matrix, ratioCpu.data(), subsetRows,
                                         ctx, numerCpu.data(), 1);

    std::vector<double> numerCuda(globalN, 0.0);
    ASSERT_TRUE(spect::cuda::runFullSubsetEmNumerCUDA(
        matrix, projections, image.data(), 2, &bedOffsets, nullptr, 0, 1,
        numerCuda.data()));

    ASSERT_EQ(numerCpu.size(), numerCuda.size());
    for (size_t i = 0; i < numerCpu.size(); ++i) {
        EXPECT_NEAR(numerCuda[i], numerCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, FullSubsetEmNumerMultiBedCtacMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const uint32_t rowPtr[] = {0, 2, 4};
    const uint32_t colIdx[] = {0, 1, 1, 2};
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 2, 3, 4);

    const std::vector<int> subsetRows = {0, 1, 2, 3};
    const std::vector<long long> bedOffsets = {0, 2};
    const int globalN = 5;
    const int numAngles = 2;
    const std::vector<std::vector<uint16_t>> bedAtten = {
        {65535, 32768, 49152, 32768, 65535, 49152},
        {49152, 65535, 16384, 65535, 16384, 49152}};

    spect::SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &bedOffsets;
    ctx.bedAttenFactors = &bedAtten;
    ctx.numAngles = numAngles;

    const std::vector<double> truth = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), 1);

    const std::vector<double> image = {1.0, 0.8, 1.2, 0.9, 1.1};
    std::vector<double> yhatCpu(subsetRows.size(), 0.0);
    std::vector<double> ratioCpu(subsetRows.size(), 0.0);
    std::vector<double> numerCpu(globalN, 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, image.data(), subsetRows, ctx,
                                        yhatCpu.data(), 1);
    for (size_t i = 0; i < subsetRows.size(); ++i) {
        ratioCpu[i] = projections[i] / yhatCpu[i];
    }
    spect::getSparseOpsCPU().spmvTSubset(matrix, ratioCpu.data(), subsetRows,
                                         ctx, numerCpu.data(), 1);

    std::vector<double> numerCuda(globalN, 0.0);
    ASSERT_TRUE(spect::cuda::runFullSubsetEmNumerCUDA(
        matrix, projections, image.data(), 2, &bedOffsets, &bedAtten,
        numAngles, 1, numerCuda.data()));

    ASSERT_EQ(numerCpu.size(), numerCuda.size());
    for (size_t i = 0; i < numerCpu.size(); ++i) {
        EXPECT_NEAR(numerCuda[i], numerCpu[i], 1e-5) << "index=" << i;
    }
}

// CPU reference for the compact attenuation bake: value[t] is multiplied by
// att[fullAngle(row) * nvox + mappedVoxel(col)], where mappedVoxel applies the
// same rotation/mirror as transformBackwardAdd (Sym90Z here).
TEST(CudaScaffoldTest, CompactAttenBakedSpmvMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    // 2x2x2 image (nvox=8), Sym90Z: compactNumAngles=2, fullNumAngles=8.
    // One ray per angle -> 2 compact rows in this chunk (compactRowBase=0).
    const int nx = 2, ny = 2, nz = 2;
    const int nvox = nx * ny * nz;
    const int compactNumAngles = 2;
    const int fullNumAngles = 8;

    const uint32_t rowPtr[] = {0, 3, 6};
    const uint32_t colIdx[] = {0, 1, 7, 2, 4, 6};
    const float values[] = {1.0f, 2.0f, 0.5f, 1.5f, 3.0f, 0.25f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 2, nvox, 6);

    std::vector<uint16_t> att(static_cast<size_t>(fullNumAngles) * nvox);
    for (size_t i = 0; i < att.size(); ++i) {
        att[i] = static_cast<uint16_t>(10000 + ((941 * i) % 50000));
    }

    const std::vector<float> x = {1.0f, 0.5f, 2.0f, 1.5f,
                                  0.75f, 1.25f, 0.4f, 1.8f};
    const std::vector<float> r = {0.8f, 1.3f};

    for (int symId = 0; symId < 4; ++symId) {
      for (const bool zMirror : {false, true}) {
        spect::cuda::CompactAttenBakeParams params;
        params.symmetryMode = 90;
        params.compactRowBase = 0;
        params.compactRaysPerAngle = 1;
        params.compactNumAngles = compactNumAngles;
        params.fullNumAngles = fullNumAngles;
        params.symId = symId;
        params.zMirror = zMirror;
        params.nx = nx;
        params.ny = ny;
        params.nz = nz;

        // CPU-side baked values via the shared voxel/angle mappings.
        std::vector<double> baked(6, 0.0);
        for (int row = 0; row < 2; ++row) {
            const int fullAngle = row + symId * compactNumAngles;
            for (uint32_t t = rowPtr[row]; t < rowPtr[row + 1]; ++t) {
                const int col = static_cast<int>(colIdx[t]);
                const int plane = nx * ny;
                const int iz = col / plane;
                const int rem = col - iz * plane;
                const int iy = rem / nx;
                const int ix = rem - iy * nx;
                const auto mapped = spect::io::mapSym90ZVoxelCoord(
                    ix, iy, iz, nx, nz, symId, zMirror);
                const size_t mappedIdx = spect::io::compactVoxelLinearIndex(
                    mapped, nx, ny);
                baked[t] = static_cast<double>(values[t]) *
                           att[static_cast<size_t>(fullAngle) * nvox +
                               mappedIdx] /
                           65535.0;
            }
        }

        std::vector<double> yRef(2, 0.0);
        for (int row = 0; row < 2; ++row) {
            for (uint32_t t = rowPtr[row]; t < rowPtr[row + 1]; ++t) {
                yRef[row] += baked[t] * x[colIdx[t]];
            }
        }
        std::vector<double> bpRef(nvox, 0.0);
        for (int row = 0; row < 2; ++row) {
            for (uint32_t t = rowPtr[row]; t < rowPtr[row + 1]; ++t) {
                bpRef[colIdx[t]] += baked[t] * r[row];
            }
        }
        std::vector<double> sensRef(nvox, 0.0);
        for (int row = 0; row < 2; ++row) {
            for (uint32_t t = rowPtr[row]; t < rowPtr[row + 1]; ++t) {
                sensRef[colIdx[t]] += baked[t];
            }
        }

        std::vector<float> yCuda(2, 0.0f);
        ASSERT_TRUE(spect::cuda::compactAttenSpmvFloatCUDA(
            matrix, att.data(), att.size(), params, x.data(), yCuda.data()))
            << "symId=" << symId << " zMirror=" << zMirror;
        std::vector<float> bpCuda(nvox, 0.0f);
        ASSERT_TRUE(spect::cuda::compactAttenSpmvTFloatCUDA(
            matrix, att.data(), att.size(), params, r.data(), bpCuda.data()))
            << "symId=" << symId << " zMirror=" << zMirror;
        const std::vector<float> ones = {1.0f, 1.0f};
        std::vector<float> sensCuda(nvox, 0.0f);
        ASSERT_TRUE(spect::cuda::compactAttenSpmvTFloatCUDA(
            matrix, att.data(), att.size(), params, ones.data(),
            sensCuda.data()))
            << "symId=" << symId << " zMirror=" << zMirror;

        for (int row = 0; row < 2; ++row) {
            EXPECT_NEAR(yCuda[row], yRef[row], 1e-5)
                << "symId=" << symId << " zMirror=" << zMirror
                << " row=" << row;
        }
        for (int v = 0; v < nvox; ++v) {
            EXPECT_NEAR(bpCuda[v], bpRef[v], 1e-5)
                << "symId=" << symId << " zMirror=" << zMirror
                << " voxel=" << v;
            EXPECT_NEAR(sensCuda[v], sensRef[v], 1e-5)
                << "symId=" << symId << " zMirror=" << zMirror
                << " sensitivity voxel=" << v;
        }
        const double lhs = static_cast<double>(yCuda[0]) * r[0] +
                           static_cast<double>(yCuda[1]) * r[1];
        double rhs = 0.0;
        for (int v = 0; v < nvox; ++v) {
            rhs += static_cast<double>(x[v]) * bpCuda[v];
        }
        EXPECT_NEAR(lhs, rhs, 1e-5)
            << "symId=" << symId << " zMirror=" << zMirror;
      }
    }

    // A unit table (all 65535) must reduce to the plain spmv.
    std::vector<uint16_t> unitAtt(att.size(), static_cast<uint16_t>(65535));
    spect::cuda::CompactAttenBakeParams unitParams;
    unitParams.symmetryMode = 90;
    unitParams.compactRowBase = 0;
    unitParams.compactRaysPerAngle = 1;
    unitParams.compactNumAngles = compactNumAngles;
    unitParams.fullNumAngles = fullNumAngles;
    unitParams.symId = 1;
    unitParams.zMirror = false;
    unitParams.nx = nx;
    unitParams.ny = ny;
    unitParams.nz = nz;
    std::vector<float> yUnit(2, 0.0f);
    std::vector<float> yPlain(2, 0.0f);
    ASSERT_TRUE(spect::cuda::compactAttenSpmvFloatCUDA(
        matrix, unitAtt.data(), unitAtt.size(), unitParams, x.data(),
        yUnit.data()));
    ASSERT_TRUE(spect::cuda::spmvFloatCUDA(matrix, x.data(), yPlain.data()));
    for (int row = 0; row < 2; ++row) {
        EXPECT_NEAR(yUnit[row], yPlain[row], 1e-6) << "row=" << row;
    }
    std::vector<float> bpUnit(nvox, 0.0f);
    std::vector<float> bpPlain(nvox, 0.0f);
    ASSERT_TRUE(spect::cuda::compactAttenSpmvTFloatCUDA(
        matrix, unitAtt.data(), unitAtt.size(), unitParams, r.data(),
        bpUnit.data()));
    ASSERT_TRUE(spect::cuda::spmvTFloatCUDA(matrix, r.data(), bpPlain.data()));
    for (int v = 0; v < nvox; ++v) {
        EXPECT_NEAR(bpUnit[v], bpPlain[v], 1e-6) << "voxel=" << v;
    }
}

TEST(CudaScaffoldTest, OsemMultiBedCtacFullSubsetMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const uint32_t rowPtr[] = {0, 2, 4};
    const uint32_t colIdx[] = {0, 1, 1, 2};
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 2, 3, 4);

    const std::vector<int> subsetRows = {0, 1, 2, 3};
    const int globalN = 5;
    const int numAngles = 2;
    const std::vector<long long> bedOffsets = {0, 2};
    const std::vector<std::vector<uint16_t>> bedAtten = {
        {65535, 32768, 49152, 32768, 65535, 49152},
        {49152, 65535, 16384, 65535, 16384, 49152}};
    const std::vector<double> truth = {1.0, 2.0, 3.0, 4.0, 5.0};

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 2;
    optsCpu.n_subsets = 1;
    optsCpu.num_angles = numAngles;
    optsCpu.bedOffsetVoxels = bedOffsets;
    optsCpu.bedAttenFactors = bedAtten;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();

    spect::ReconOpts optsCuda = optsCpu;
    optsCuda.sparseOps = &spect::getSparseOpsCUDA();

    spect::SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &optsCpu.bedOffsetVoxels;
    ctx.bedAttenFactors = &optsCpu.bedAttenFactors;
    ctx.numAngles = numAngles;

    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), 1);

    std::vector<double> imageCpu(globalN, 1.0);
    std::vector<double> imageCuda = imageCpu;
    std::vector<double> denomCpu = spect::computeSubsetSensitivity(
        matrix, subsetRows, globalN, 1, optsCpu);
    std::vector<double> denomCuda = spect::computeSubsetSensitivity(
        matrix, subsetRows, globalN, 1, optsCuda);

    spect::OSEM_step(matrix, subsetRows, projections, imageCpu, denomCpu,
                     globalN, 1, 1, optsCpu, nullptr, nullptr, 0.0, nullptr,
                     1.0);
    spect::OSEM_step(matrix, subsetRows, projections, imageCuda, denomCuda,
                     globalN, 1, 1, optsCuda, nullptr, nullptr, 0.0, nullptr,
                     1.0);

    ASSERT_EQ(imageCpu.size(), imageCuda.size());
    for (size_t i = 0; i < imageCpu.size(); ++i) {
        EXPECT_NEAR(imageCuda[i], imageCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, ProjDomainChunkedStreamingMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const uint32_t rowPtr[] = {0, 2, 4, 6, 8};
    const uint32_t colIdx[] = {0, 1, 1, 2, 0, 2, 0, 1};
    const float values[] = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 1.0f, 1.0f, 4.0f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 4, 3, 8);

    const std::vector<double> image = {1.0, 2.0, 1.5};
    std::vector<double> projections(4, 0.0);
    spect::getSparseOpsCPU().spmv(matrix, image.data(), projections.data(), 1);

    const std::vector<std::vector<int>> chunkRows = {{0, 1}, {2, 3}};
    spect::filters::PSFKernels psf;
    psf.kx = {1.0};
    psf.ky = {1.0};
    psf.kz = {1.0};

    std::vector<double> numerCuda(3, 0.0);
    ASSERT_TRUE(spect::cuda::runProjDomainChunkedStreamingCUDA(
        matrix, chunkRows, projections, psf, 2, 1, 2, image.data(), 1,
        numerCuda.data()));

    std::vector<double> numerCpu(3, 0.0);
    const spect::SparseSubsetContext ctx{};
    for (const auto& rows : chunkRows) {
        std::vector<double> localYhat(rows.size(), 0.0);
        std::vector<double> localRatio(rows.size(), 0.0);
        std::vector<double> localNumer(3, 0.0);
        spect::getSparseOpsCPU().spmvSubset(matrix, image.data(), rows, ctx,
                                            localYhat.data(), 1);
        for (size_t index = 0; index < rows.size(); ++index) {
            localRatio[index] =
                projections[static_cast<size_t>(rows[index])] / localYhat[index];
        }
        spect::getSparseOpsCPU().spmvTSubset(matrix, localRatio.data(), rows,
                                             ctx, localNumer.data(), 1);
        for (size_t index = 0; index < numerCpu.size(); ++index) {
            numerCpu[index] += localNumer[index];
        }
    }

    for (size_t index = 0; index < numerCpu.size(); ++index) {
        EXPECT_NEAR(numerCuda[index], numerCpu[index], 1e-5)
            << "index=" << index;
    }
}

TEST(CudaScaffoldTest, VolumePsfMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    std::vector<double> cpu = {
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0,
        12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0};
    std::vector<double> gpu = cpu;
    const std::vector<double> kx = {0.25, 0.5, 0.25};
    const std::vector<double> ky = {0.2, 0.6, 0.2};
    const std::vector<double> kz = {0.1, 0.8, 0.1};

    spect::filters::conv3d_separable_inplace(cpu, 4, 3, 2, kx, ky, kz, 1);
    ASSERT_TRUE(spect::cuda::applySeparableConv3DInplaceCUDA(gpu, 4, 3, 2, kx,
                                                             ky, kz));

    ASSERT_EQ(cpu.size(), gpu.size());
    for (size_t i = 0; i < cpu.size(); ++i) {
        EXPECT_NEAR(gpu[i], cpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, CompactFinalForwardProjectionMatchesExplicitToyMatrix) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    spect::io::CompactSymmetryMetadata meta;
    meta.Nx = 2;
    meta.Ny = 2;
    meta.Nz = 2;
    meta.totalVoxels = 8;
    meta.fullNumAngles = 16;
    meta.compactNumAngles = 3;
    meta.detNx = 2;
    meta.detNz = 2;
    meta.compactDetZ = 1;
    meta.compactRaysPerAngle = 2;
    meta.totalRows = 6;
    meta.useZSym = true;
    meta.use45Sym = true;
    meta.symCode = 45;
    meta.symmetryOrder = 8;
    meta.mode = spect::io::CompactSymmetryMode::Sym45Z;

    const std::vector<uint32_t> rowPtr = {0, 2, 3, 5, 6, 8, 10};
    const std::vector<uint32_t> colIdx = {0, 3, 5, 1, 6, 2, 4, 7, 0, 6};
    const std::vector<float> values = {1.0f, 0.5f, 0.75f, 0.2f, 1.1f,
                                       0.9f, 0.4f, 1.2f, 0.3f, 0.8f};
    const auto compact = makeTinyIoMatrix(
        rowPtr.data(), colIdx.data(), values.data(), meta.totalRows,
        meta.totalVoxels, values.size());
    const std::vector<double> image = {0.5, 1.0, 1.5, 2.0,
                                       2.5, 3.0, 3.5, 4.0};
    std::vector<double> expected(static_cast<size_t>(meta.fullRows()), 0.0);
    for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
        for (int z = 0; z < meta.zBranchCount(); ++z) {
            const bool zMirror = z != 0;
            for (int row = 0; row < meta.totalRows; ++row) {
                uint64_t fullRow = 0;
                if (!spect::io::fullProjectionIndexFromCompactRow(
                        meta, row, sym, zMirror, fullRow)) {
                    continue;
                }
                for (uint32_t k = rowPtr[static_cast<size_t>(row)];
                     k < rowPtr[static_cast<size_t>(row) + 1u]; ++k) {
                    const size_t fullCol = spect::io::mapSym45ZVoxelIndex(
                        colIdx[static_cast<size_t>(k)], meta.Nx, meta.Ny,
                        meta.Nz, sym, zMirror);
                    expected[static_cast<size_t>(fullRow)] +=
                        static_cast<double>(values[static_cast<size_t>(k)]) *
                        image[fullCol];
                }
            }
        }
    }

    std::vector<double> actual(expected.size(), 0.0);
    ASSERT_TRUE(spect::cuda::computeCompactSymmetricForwardProjectionCUDA(
        compact, meta, image.data(), 1, actual.data()));
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5) << "row=" << i;
    }
}

TEST(CudaScaffoldTest, VolumePsfForwardAdjointDotProductAtBoundaries) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

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
    const std::vector<double> kx = {0.1, 0.2, 0.4, 0.2, 0.1};
    const std::vector<double> ky = {0.2, 0.6, 0.2};
    const std::vector<double> kz = {0.15, 0.7, 0.15};

    auto gx = x;
    auto gty = y;
    ASSERT_TRUE(spect::cuda::applySeparableConv3DInplaceCUDA(
        gx, Nx, Ny, Nz, kx, ky, kz, true));
    ASSERT_TRUE(spect::cuda::applySeparableConv3DAdjointInplaceCUDA(
        gty, Nx, Ny, Nz, kx, ky, kz, true));

    const double lhs = std::inner_product(gx.begin(), gx.end(), y.begin(), 0.0);
    const double rhs = std::inner_product(x.begin(), x.end(), gty.begin(), 0.0);
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    EXPECT_NEAR(lhs, rhs, 2e-6 * scale);
}

TEST(CudaScaffoldTest, RdpGradientMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const int Nx = 3;
    const int Ny = 3;
    const int Nz = 2;
    std::vector<double> image = {
        1.0, 0.5, 2.0,
        1.5, 3.0, 0.75,
        2.5, 1.25, 0.9,
        1.1, 0.8, 1.7,
        2.2, 3.1, 0.6,
        1.4, 0.7, 2.8};
    std::vector<uint8_t> mask = {
        1, 1, 1,
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1};
    std::vector<double> ct(image.size(), 0.0);
    for (size_t i = image.size() / 2; i < image.size(); ++i)
        ct[i] = 0.03;

    spect::RegConf conf;
    conf.type = spect::RegType::RDP;
    conf.delta = 1e-3;
    conf.gamma = 0.2;
    conf.skip_outside_mask = true;
    conf.ct_image = &ct;

    std::vector<double> gradCpu;
    std::vector<double> gradCuda;
    spect::regularizer::compute_reg_gradient(image, Nx, Ny, Nz, conf, &mask,
                                             gradCpu, 1);
    ASSERT_TRUE(spect::cuda::computeRDPGradientCUDA(
        image, Nx, Ny, Nz, conf.delta, conf.gamma, &mask,
        conf.skip_outside_mask, gradCuda, conf.ct_image, conf.ct_sigma,
        conf.ct_weight_floor));

    ASSERT_EQ(gradCpu.size(), gradCuda.size());
    for (size_t i = 0; i < gradCpu.size(); ++i) {
        EXPECT_NEAR(gradCuda[i], gradCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, TvGradientMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const int Nx = 3;
    const int Ny = 3;
    const int Nz = 2;
    std::vector<double> image = {
        1.0, 0.5, 2.0,
        1.5, 3.0, 0.75,
        2.5, 1.25, 0.9,
        1.1, 0.8, 1.7,
        2.2, 3.1, 0.6,
        1.4, 0.7, 2.8};
    std::vector<uint8_t> mask = {
        1, 1, 1,
        1, 1, 1,
        1, 0, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1};
    std::vector<double> ct(image.size(), 0.0);
    for (size_t i = 0; i < image.size(); ++i)
        ct[i] = 0.002 * static_cast<double>(i % 7);

    spect::RegConf conf;
    conf.type = spect::RegType::TVIsotropic;
    conf.tv_eps = 1e-6;
    conf.skip_outside_mask = true;
    conf.ct_image = &ct;

    std::vector<double> gradCpu;
    std::vector<double> gradCuda;
    spect::regularizer::compute_reg_gradient(image, Nx, Ny, Nz, conf, &mask,
                                             gradCpu, 1);
    ASSERT_TRUE(spect::cuda::computeTVGradientCUDA(
        image, Nx, Ny, Nz, conf.tv_eps, &mask,
        conf.skip_outside_mask, gradCuda, conf.ct_image, conf.ct_sigma,
        conf.ct_weight_floor));

    ASSERT_EQ(gradCpu.size(), gradCuda.size());
    for (size_t i = 0; i < gradCpu.size(); ++i) {
        EXPECT_NEAR(gradCuda[i], gradCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, TvProxRegisteredCtPreservesCtEdge) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const int Nx = 6, Ny = 1, Nz = 1;
    const std::vector<double> image = {1.0, 1.0, 1.0, 4.0, 4.0, 4.0};
    const std::vector<double> constantCt(image.size(), 0.0);
    const std::vector<double> edgeCt = {0.0, 0.0, 0.0, 0.08, 0.08, 0.08};
    std::vector<double> activityOnly;
    std::vector<double> constantCtResult;
    std::vector<double> edgeCtResult;

    ASSERT_TRUE(spect::cuda::computeTVProxCUDA(
        image, Nx, Ny, Nz, 0.5, 80, activityOnly));
    ASSERT_TRUE(spect::cuda::computeTVProxCUDA(
        image, Nx, Ny, Nz, 0.5, 80, constantCtResult, &constantCt, 0.01, 0.05));
    ASSERT_TRUE(spect::cuda::computeTVProxCUDA(
        image, Nx, Ny, Nz, 0.5, 80, edgeCtResult, &edgeCt, 0.01, 0.05));

    for (size_t i = 0; i < image.size(); ++i)
        EXPECT_NEAR(constantCtResult[i], activityOnly[i], 1e-10);
    EXPECT_GT(edgeCtResult[3] - edgeCtResult[2],
              activityOnly[3] - activityOnly[2]);
}

TEST(CudaScaffoldTest, FloatPsfMatchesDoublePsfWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    std::vector<double> data = {
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0,
        12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0};
    std::vector<double> dataFloat = data;
    std::vector<double> dataDouble = data;
    const std::vector<double> kx = {0.25, 0.5, 0.25};
    const std::vector<double> ky = {0.2, 0.6, 0.2};
    const std::vector<double> kz = {0.1, 0.8, 0.1};

    // CPU double reference
    spect::filters::conv3d_separable_inplace(data, 4, 3, 2, kx, ky, kz, 1);

    // CUDA double path
    ASSERT_TRUE(spect::cuda::applySeparableConv3DInplaceCUDA(
        dataDouble, 4, 3, 2, kx, ky, kz, false));

    // CUDA float path
    ASSERT_TRUE(spect::cuda::applySeparableConv3DInplaceCUDA(
        dataFloat, 4, 3, 2, kx, ky, kz, true));

    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(dataDouble[i], data[i], 1e-10) << "double index=" << i;
        EXPECT_NEAR(dataFloat[i], data[i], 1e-4) << "float index=" << i;
    }
}

TEST(CudaScaffoldTest, VolumePsfMatchesCpuOnCompactShareGeometry) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const int Nx = 108;
    const int Ny = 108;
    const int Nz = 80;
    const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                         static_cast<size_t>(Nz);
    const std::vector<double> kernel =
        spect::filters::makeGaussian1D(1.5, 0.3);

    std::vector<double> ones(total, 1.0);
    ASSERT_TRUE(spect::cuda::applySeparableConv3DInplaceCUDA(
        ones, Nx, Ny, Nz, kernel, kernel, kernel, true));
    for (double value : ones) {
        ASSERT_NEAR(value, 1.0, 1e-5);
    }

    std::vector<double> cpu(total);
    for (size_t idx = 0; idx < total; ++idx) {
        const int x = static_cast<int>(idx % static_cast<size_t>(Nx));
        const int y =
            static_cast<int>((idx / static_cast<size_t>(Nx)) %
                             static_cast<size_t>(Ny));
        const int z =
            static_cast<int>(idx / (static_cast<size_t>(Nx) *
                                    static_cast<size_t>(Ny)));
        cpu[idx] = 0.5 + 0.01 * static_cast<double>(x) +
                   0.02 * static_cast<double>(y) +
                   0.03 * static_cast<double>(z) +
                   0.001 * static_cast<double>(idx % 17);
    }
    std::vector<double> cudaDouble = cpu;
    std::vector<double> cudaFloat = cpu;

    spect::filters::conv3d_separable_inplace(cpu, Nx, Ny, Nz, kernel, kernel,
                                             kernel, 8);
    ASSERT_TRUE(spect::cuda::applySeparableConv3DInplaceCUDA(
        cudaDouble, Nx, Ny, Nz, kernel, kernel, kernel, false));
    ASSERT_TRUE(spect::cuda::applySeparableConv3DInplaceCUDA(
        cudaFloat, Nx, Ny, Nz, kernel, kernel, kernel, true));

    for (size_t idx = 0; idx < total; idx += 997) {
        EXPECT_NEAR(cudaDouble[idx], cpu[idx], 1e-10) << "double index=" << idx;
        EXPECT_NEAR(cudaFloat[idx], cpu[idx], 1e-4) << "float index=" << idx;
    }
}

TEST(CudaScaffoldTest, ButterworthCuFftProducesFiniteOutputWhenAvailable) {
    if (!spect::cuda::isCudaAvailable() ||
        !spect::filters::isButterworthCUFFTAvailable()) {
        GTEST_SKIP() << "cuFFT Butterworth backend is not available.";
    }

    spect::filters::ButterworthConf cufftConf;
    cufftConf.enabled = true;
    cufftConf.order = 2;
    cufftConf.cutoff_lo_frac = 0.35;

    spect::filters::Spacing3 spacing{1.0, 1.0, 1.0};
    std::vector<double> gpu(64, 0.0);
    double beforeSum = 0.0;
    for (size_t i = 0; i < gpu.size(); ++i) {
        gpu[i] = static_cast<double>((i % 11) + 1);
        beforeSum += gpu[i];
    }
    spect::filters::apply_butterworth_inplace(gpu, 4, 4, 4, spacing, cufftConf,
                                              1);

    double afterSum = 0.0;
    for (double value : gpu) {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GE(value, 0.0);
        afterSum += value;
    }
    EXPECT_GT(beforeSum, 0.0);
    EXPECT_GT(afterSum, 0.0);
}

TEST(CudaScaffoldTest, BsremSingleBedMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const uint32_t rowPtr[] = {0, 2, 4, 6, 8};
    const uint32_t colIdx[] = {0, 1, 1, 2, 0, 2, 0, 1};
    const float values[] = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 1.0f, 1.0f, 4.0f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 4, 3, 8);

    const std::vector<double> truth = {1.0, 2.0, 1.5};
    std::vector<double> projections(4, 0.0);
    spect::getSparseOpsCPU().spmv(matrix, truth.data(), projections.data(), 1);

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 1;
    optsCpu.n_subsets = 1;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();

    spect::ReconOpts optsCuda = optsCpu;
    optsCuda.sparseOps = &spect::getSparseOpsCUDA();

    const std::vector<int> subsetRows = {0, 1, 2, 3};
    std::vector<double> imageCpu = {1.0, 1.0, 1.0};
    std::vector<double> imageCuda = imageCpu;
    std::vector<double> denom(3, 0.0);
    spect::getSparseOpsCPU().computeNormAT(matrix, denom.data(), 1);
    std::vector<double> subsetDenom = denom;
    std::vector<double> bufferCpu(3, 0.0);
    std::vector<double> bufferCuda(3, 0.0);

    spect::BSREM_step(matrix, subsetRows, projections, imageCpu, denom,
                      subsetDenom, 3, 3, 1, 1, optsCpu, nullptr, nullptr, 0.0,
                      nullptr, nullptr, &bufferCpu, 1.0);
    spect::BSREM_step(matrix, subsetRows, projections, imageCuda, denom,
                      subsetDenom, 3, 3, 1, 1, optsCuda, nullptr, nullptr, 0.0,
                      nullptr, nullptr, &bufferCuda, 1.0);

    ASSERT_EQ(imageCpu.size(), imageCuda.size());
    for (size_t i = 0; i < imageCpu.size(); ++i) {
        EXPECT_NEAR(imageCuda[i], imageCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, BsremMultiBedFullSubsetMatchesCpuWhenCudaAvailable) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const uint32_t rowPtr[] = {0, 2, 4};
    const uint32_t colIdx[] = {0, 2, 1, 2};
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto matrix = makeTinyIoMatrix(rowPtr, colIdx, values, 2, 3, 4);

    const std::vector<int> subsetRows = {0, 1, 2, 3};
    const std::vector<long long> bedOffsets = {0, 2};
    const int globalN = 5;

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 2;
    optsCpu.n_subsets = 1;
    optsCpu.bedOffsetVoxels = bedOffsets;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();

    spect::ReconOpts optsCuda = optsCpu;
    optsCuda.sparseOps = &spect::getSparseOpsCUDA();

    spect::SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &optsCpu.bedOffsetVoxels;

    const std::vector<double> truth = {0.6, 0.2, 1.1, 0.8, 0.4};
    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), 1);

    std::vector<double> imageCpu = {1.0, 0.9, 0.7, 1.1, 0.8};
    std::vector<double> imageCuda = imageCpu;
    std::vector<double> denom =
        spect::computeSubsetSensitivity(matrix, subsetRows, globalN, 1, optsCpu);
    std::vector<double> subsetDenom = denom;
    std::vector<double> bufferCpu(globalN, 0.0);
    std::vector<double> bufferCuda(globalN, 0.0);

    spect::BSREM_step(matrix, subsetRows, projections, imageCpu, denom,
                      subsetDenom, globalN, 5, 1, 1, optsCpu, nullptr, nullptr,
                      0.0, nullptr, nullptr, &bufferCpu, 1.0);
    spect::BSREM_step(matrix, subsetRows, projections, imageCuda, denom,
                      subsetDenom, globalN, 5, 1, 1, optsCuda, nullptr, nullptr,
                      0.0, nullptr, nullptr, &bufferCuda, 1.0);

    ASSERT_EQ(bufferCpu.size(), bufferCuda.size());
    for (size_t i = 0; i < bufferCpu.size(); ++i) {
        EXPECT_NEAR(bufferCuda[i], bufferCpu[i], 1e-5) << "buffer index=" << i;
    }

    ASSERT_EQ(imageCpu.size(), imageCuda.size());
    for (size_t i = 0; i < imageCpu.size(); ++i) {
        EXPECT_NEAR(imageCuda[i], imageCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest, ChunkWarmupRemovesBuildAndPinnedStageForRepeatedMlem) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    ScopedCudaChunkBudgetOverride budgetOverride(3);
    const std::vector<int> subsetRows = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<long long> bedOffsets = {0, 2};
    const int globalN = 5;
    const std::vector<double> truth = {0.6, 0.2, 1.1, 0.8, 0.4};
    const std::vector<double> initialImage = {1.0, 0.9, 0.7, 1.1, 0.8};

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 2;
    optsCpu.n_subsets = 1;
    optsCpu.bedOffsetVoxels = bedOffsets;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();
    spect::SparseSubsetContext ctxCpu;
    ctxCpu.bedCount = optsCpu.bedCount;
    ctxCpu.bedOffsetVoxels = &optsCpu.bedOffsetVoxels;

    const OwnedTinyIoMatrix baselineOwned = makeChunkedTinyIoMatrix();
    const auto baselineMatrix = baselineOwned.view();
    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(baselineMatrix, truth.data(), subsetRows, ctxCpu,
                                        projections.data(), 1);
    const std::vector<double> denom =
        spect::computeSubsetSensitivity(baselineMatrix, subsetRows, globalN, 1,
                                        optsCpu);
    std::vector<double> imageCpu = initialImage;
    std::vector<double> yhatCpu(subsetRows.size(), 0.0);
    std::vector<double> ratioCpu(subsetRows.size(), 0.0);
    std::vector<double> numerCpu(globalN, 0.0);
    for (int iter = 0; iter < 6; ++iter) {
        std::fill(yhatCpu.begin(), yhatCpu.end(), 0.0);
        std::fill(numerCpu.begin(), numerCpu.end(), 0.0);
        spect::getSparseOpsCPU().spmvSubset(baselineMatrix, imageCpu.data(),
                                            subsetRows, ctxCpu, yhatCpu.data(), 1);
        for (size_t i = 0; i < ratioCpu.size(); ++i) {
            ratioCpu[i] = projections[i] / std::max(yhatCpu[i], 1e-12);
        }
        spect::getSparseOpsCPU().spmvTSubset(baselineMatrix, ratioCpu.data(),
                                             subsetRows, ctxCpu, numerCpu.data(), 1);
        for (int voxel = 0; voxel < globalN; ++voxel) {
            if (denom[voxel] > 1e-12) {
                imageCpu[static_cast<size_t>(voxel)] *=
                    numerCpu[static_cast<size_t>(voxel)] / denom[voxel];
            }
        }
    }

    const OwnedTinyIoMatrix coldOwned = makeChunkedTinyIoMatrix();
    const auto coldMatrix = coldOwned.view();
    std::vector<double> imageCudaCold = initialImage;
    std::vector<double> numerCudaCold(globalN, 0.0);
    spect::cuda::resetCudaChunkStageTimingCUDA();
    for (int iter = 0; iter < 6; ++iter) {
        std::fill(numerCudaCold.begin(), numerCudaCold.end(), 0.0);
        ASSERT_TRUE(spect::cuda::runFullSubsetEmNumerCUDA(
            coldMatrix, projections, imageCudaCold.data(), 2, &bedOffsets, nullptr,
            0, 1, numerCudaCold.data()));
        for (int voxel = 0; voxel < globalN; ++voxel) {
            if (denom[voxel] > 1e-12) {
                imageCudaCold[static_cast<size_t>(voxel)] *=
                    numerCudaCold[static_cast<size_t>(voxel)] / denom[voxel];
            }
        }
    }
    const auto coldSnapshot = spect::cuda::getCudaChunkStageTimingSnapshotCUDA();

    const OwnedTinyIoMatrix warmOwned = makeChunkedTinyIoMatrix();
    const auto warmMatrix = warmOwned.view();
    ASSERT_TRUE(spect::cuda::warmupFullSubsetChunkCacheCUDA(warmMatrix, 2, nullptr, 0,
                                                            1));
    std::vector<double> imageCudaWarm = initialImage;
    std::vector<double> numerCudaWarm(globalN, 0.0);
    spect::cuda::resetCudaChunkStageTimingCUDA();
    for (int iter = 0; iter < 6; ++iter) {
        std::fill(numerCudaWarm.begin(), numerCudaWarm.end(), 0.0);
        ASSERT_TRUE(spect::cuda::runFullSubsetEmNumerCUDA(
            warmMatrix, projections, imageCudaWarm.data(), 2, &bedOffsets, nullptr,
            0, 1, numerCudaWarm.data()));
        for (int voxel = 0; voxel < globalN; ++voxel) {
            if (denom[voxel] > 1e-12) {
                imageCudaWarm[static_cast<size_t>(voxel)] *=
                    numerCudaWarm[static_cast<size_t>(voxel)] / denom[voxel];
            }
        }
    }
    const auto warmSnapshot = spect::cuda::getCudaChunkStageTimingSnapshotCUDA();

    ASSERT_EQ(imageCpu.size(), imageCudaWarm.size());
    for (size_t i = 0; i < imageCpu.size(); ++i) {
        EXPECT_NEAR(imageCudaWarm[i], imageCpu[i], 1e-5) << "index=" << i;
    }
    ASSERT_EQ(imageCpu.size(), imageCudaCold.size());
    for (size_t i = 0; i < imageCpu.size(); ++i) {
        EXPECT_NEAR(imageCudaCold[i], imageCpu[i], 1e-5) << "cold index=" << i;
    }

    EXPECT_GT(coldSnapshot.chunkBuildCalls, 0u);
    EXPECT_EQ(coldSnapshot.pinnedStageCalls, 0u);
    // P2: backward transpose is skipped by default (direct TRANSPOSE SpMV)
    EXPECT_EQ(coldSnapshot.transposeChunkCalls, 0u);
    EXPECT_EQ(warmSnapshot.chunkBuildCalls, 0u);
    EXPECT_EQ(warmSnapshot.pinnedStageCalls, 0u);
    // P3-D: device chunk cache holds all chunks for this small matrix,
    // so H2D transfers during iterations drop to zero.
    // If device cache is active, h2dChunkCalls == 0 and hits > 0.
    if (warmSnapshot.deviceCacheHitCalls > 0) {
      EXPECT_EQ(warmSnapshot.h2dChunkCalls, 0u);
    } else {
      EXPECT_GT(warmSnapshot.h2dChunkCalls, 0u);
    }
    EXPECT_EQ(warmSnapshot.transposeChunkCalls, 0u);
}

TEST(CudaScaffoldTest, Real260601FirstChunkWarmupForwardMatchesCpuWhenPresent) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    const std::filesystem::path dir = "test_data/6module/260601";
    const std::filesystem::path metaPath = dir / "metadata.bin";
    if (!std::filesystem::exists(metaPath)) {
        GTEST_SKIP() << "no 260601 compact chunk metadata under test_data";
    }

    const auto meta = spect::io::loadCompactSymmetryMetadata(metaPath.string());
    const auto chunks = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, 0);
    ASSERT_FALSE(chunks.chunks().empty());
    const auto chunk = chunks.loadChunk(0);
    ASSERT_TRUE(chunk);
    const auto view = chunk->view();
    ASSERT_GT(view.rows, 0);
    ASSERT_GT(view.cols, 0);

    ASSERT_TRUE(spect::cuda::warmupResidentMatrixSetCUDA({view}));

    std::vector<float> x(static_cast<size_t>(view.cols), 1.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = 1.0f + 0.01f * static_cast<float>(i % 13u);
    }
    std::vector<float> yCuda(static_cast<size_t>(view.rows), 0.0f);
    ASSERT_TRUE(spect::cuda::spmvFloatCUDA(view, x.data(), yCuda.data()));

    const std::vector<int> rows = {
        0,
        std::min(1, view.rows - 1),
        view.rows / 2,
        view.rows - 1,
    };
    for (int row : rows) {
        double yCpu = 0.0;
        for (uint32_t k = view.row_ptr[row]; k < view.row_ptr[row + 1]; ++k) {
            yCpu += static_cast<double>(view.val[k]) *
                    static_cast<double>(x[view.col_idx[k]]);
        }
        const double tol = std::max(1e-2, std::fabs(yCpu) * 2e-4);
        EXPECT_NEAR(static_cast<double>(yCuda[static_cast<size_t>(row)]), yCpu, tol)
            << "row=" << row;
    }
}

TEST(CudaScaffoldTest, ChunkWarmupKeepsRepeatedBsremNumericallyAligned) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    ScopedCudaChunkBudgetOverride budgetOverride(3);
    const OwnedTinyIoMatrix owned = makeChunkedTinyIoMatrix();
    const auto matrix = owned.view();
    const std::vector<int> subsetRows = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<long long> bedOffsets = {0, 2};
    const int globalN = 5;
    const std::vector<double> truth = {0.6, 0.2, 1.1, 0.8, 0.4};
    const std::vector<double> initialImage = {1.0, 0.9, 0.7, 1.1, 0.8};

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 2;
    optsCpu.n_subsets = 1;
    optsCpu.bedOffsetVoxels = bedOffsets;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();

    spect::ReconOpts optsCuda = optsCpu;
    optsCuda.sparseOps = &spect::getSparseOpsCUDA();

    spect::SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &optsCpu.bedOffsetVoxels;

    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), 1);

    const std::vector<double> denom =
        spect::computeSubsetSensitivity(matrix, subsetRows, globalN, 1, optsCpu);
    const std::vector<double> subsetDenom = denom;

    ASSERT_TRUE(
        spect::cuda::warmupFullSubsetChunkCacheCUDA(matrix, 2, nullptr, 0, 1));

    std::vector<double> imageCpu = initialImage;
    std::vector<double> imageCuda = initialImage;
    std::vector<double> bufferCpu(globalN, 0.0);
    std::vector<double> bufferCuda(globalN, 0.0);
    for (int iter = 1; iter <= 6; ++iter) {
        spect::BSREM_step(matrix, subsetRows, projections, imageCpu, denom,
                          subsetDenom, globalN, 6, iter, 1, optsCpu, nullptr,
                          nullptr, 0.0, nullptr, nullptr, &bufferCpu, 1.0);
        spect::BSREM_step(matrix, subsetRows, projections, imageCuda, denom,
                          subsetDenom, globalN, 6, iter, 1, optsCuda, nullptr,
                          nullptr, 0.0, nullptr, nullptr, &bufferCuda, 1.0);
    }

    ASSERT_EQ(bufferCpu.size(), bufferCuda.size());
    for (size_t i = 0; i < bufferCpu.size(); ++i) {
        EXPECT_NEAR(bufferCuda[i], bufferCpu[i], 1e-5) << "buffer index=" << i;
    }
    ASSERT_EQ(imageCpu.size(), imageCuda.size());
    for (size_t i = 0; i < imageCpu.size(); ++i) {
        EXPECT_NEAR(imageCuda[i], imageCpu[i], 1e-5) << "index=" << i;
    }
}

TEST(CudaScaffoldTest,
     StableHostWithoutWarmupMatchesCpuAndAvoidsPinnedStageCopy) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    ScopedCudaChunkBudgetOverride budgetOverride(3);
    const OwnedTinyIoMatrix owned = makeChunkedTinyIoMatrix();
    const auto matrix = owned.view();
    const std::vector<long long> bedOffsets = {0, 2};
    const std::vector<double> truth = {0.6, 0.2, 1.1, 0.8, 0.4};
    const std::vector<double> initialImage = {1.0, 0.9, 0.7, 1.1, 0.8};
    const int globalN = static_cast<int>(initialImage.size());
    const std::vector<int> subsetRows = {0, 1, 2, 3, 4, 5, 6, 7};

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 2;
    optsCpu.n_subsets = 1;
    optsCpu.bedOffsetVoxels = bedOffsets;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();

    spect::SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &optsCpu.bedOffsetVoxels;

    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), 1);
    const std::vector<double> denom =
        spect::computeSubsetSensitivity(matrix, subsetRows, globalN, 1, optsCpu);

    const auto cpu = runCpuChunkedEmProbe(matrix, projections, initialImage, denom,
                                          bedOffsets, 6);
    const auto stable = runChunkedEmProbe(
        matrix, projections, initialImage, denom, bedOffsets, 6, false);

    ASSERT_EQ(cpu.numer.size(), stable.numer.size());
    for (size_t i = 0; i < cpu.numer.size(); ++i) {
        EXPECT_NEAR(stable.numer[i], cpu.numer[i], 1e-5) << "numer index=" << i;
    }
    ASSERT_EQ(cpu.image.size(), stable.image.size());
    for (size_t i = 0; i < cpu.image.size(); ++i) {
        EXPECT_NEAR(stable.image[i], cpu.image[i], 1e-5) << "image index=" << i;
    }
    EXPECT_GT(stable.timing.chunkBuildCalls, 0u);
    EXPECT_GT(stable.timing.h2dChunkCalls, 0u);
    EXPECT_EQ(stable.timing.pinnedStageCalls, 0u);
    // P2: backward transpose is skipped by default
    EXPECT_EQ(stable.timing.transposeChunkCalls, 0u);
}

TEST(CudaScaffoldTest,
     HostChunkBuildThreadBudgetSupportsOverrideAndTinyWorkShortcut) {
    const int defaultLarge =
        spect::cuda::resolveCudaHostChunkBuildThreadsForTestingCUDA(16, 1ull << 20);
    EXPECT_GE(defaultLarge, 1);
    EXPECT_LE(defaultLarge, 16);
    EXPECT_EQ(spect::cuda::resolveCudaHostChunkBuildThreadsForTestingCUDA(16, 1024),
              1);

    ScopedCudaHostChunkBuildThreadsOverride overrideThreads(7);
    const int overrideLarge =
        spect::cuda::resolveCudaHostChunkBuildThreadsForTestingCUDA(16, 1ull << 20);
    if (defaultLarge > 1) {
        EXPECT_EQ(overrideLarge, 7);
    } else {
        EXPECT_EQ(overrideLarge, 1);
    }
}

TEST(CudaScaffoldTest,
     PreparedHostCacheBudgetSplitsAcrossActiveResidencySets) {
    ScopedCudaPreparedHostCacheOverride cacheOverride(1024);
    EXPECT_EQ(
        spect::cuda::resolveCudaPreparedChunkHostCacheBudgetForTestingCUDA(0),
        1024u);
    EXPECT_EQ(
        spect::cuda::resolveCudaPreparedChunkHostCacheBudgetForTestingCUDA(1),
        1024u);
    EXPECT_EQ(
        spect::cuda::resolveCudaPreparedChunkHostCacheBudgetForTestingCUDA(2),
        512u);
    EXPECT_EQ(
        spect::cuda::resolveCudaPreparedChunkHostCacheBudgetForTestingCUDA(4),
        256u);
}

TEST(CudaScaffoldTest,
     PartialPreparedHostCacheBudgetKeepsStablePathNumericallyAligned) {
    if (!spect::cuda::isCudaAvailable()) {
        GTEST_SKIP() << "No CUDA device available in the current runtime.";
    }

    ScopedCudaChunkBudgetOverride budgetOverride(3);
    ScopedCudaPreparedHostCacheOverride cacheOverride(64);

    const OwnedTinyIoMatrix owned = makeChunkedTinyIoMatrix();
    const auto matrix = owned.view();
    const std::vector<long long> bedOffsets = {0, 2};
    const std::vector<double> truth = {0.6, 0.2, 1.1, 0.8, 0.4};
    const std::vector<double> initialImage = {1.0, 0.9, 0.7, 1.1, 0.8};
    const int globalN = static_cast<int>(initialImage.size());
    const std::vector<int> subsetRows = {0, 1, 2, 3, 4, 5, 6, 7};

    spect::ReconOpts optsCpu;
    optsCpu.threads = 1;
    optsCpu.bedCount = 2;
    optsCpu.n_subsets = 1;
    optsCpu.bedOffsetVoxels = bedOffsets;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();

    spect::SparseSubsetContext ctx;
    ctx.bedCount = 2;
    ctx.bedOffsetVoxels = &optsCpu.bedOffsetVoxels;

    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), 1);
    const std::vector<double> denom =
        spect::computeSubsetSensitivity(matrix, subsetRows, globalN, 1, optsCpu);

    const auto cpu = runCpuChunkedEmProbe(matrix, projections, initialImage, denom,
                                          bedOffsets, 6);
    const auto partialWarm = runChunkedEmProbe(
        matrix, projections, initialImage, denom, bedOffsets, 6, true);

    ASSERT_EQ(cpu.numer.size(), partialWarm.numer.size());
    for (size_t i = 0; i < cpu.numer.size(); ++i) {
        EXPECT_NEAR(partialWarm.numer[i], cpu.numer[i], 1e-5)
            << "numer index=" << i;
    }
    ASSERT_EQ(cpu.image.size(), partialWarm.image.size());
    for (size_t i = 0; i < cpu.image.size(); ++i) {
        EXPECT_NEAR(partialWarm.image[i], cpu.image[i], 1e-5)
            << "image index=" << i;
    }
    EXPECT_EQ(partialWarm.timing.pinnedStageCalls, 0u);
    // P2: backward transpose is skipped by default
    EXPECT_EQ(partialWarm.timing.transposeChunkCalls, 0u);
}

} // namespace
