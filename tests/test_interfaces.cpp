/**
 * @file test_interfaces.cpp
 * @brief 接口头文件编译验证测试
 * 
 * 验证所有接口头文件可以正确编译，不依赖第三方库。
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

// 包含所有接口头文件，验证编译
#include "core/Types.h"
#include "core/CSRHeader.h"
#include "core/CSRView.h"
#include "cli/ArgumentParser.h"
#include "cli/ConfigParser.h"
#include "filters/Convolution.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "io/MHDWriter.h"
#include "io/VolumeLoader.h"
#include "interfaces/ISparseOps.h"
#include "interfaces/IProjector.h"
#include "interfaces/IRegularizer.h"
#include "interfaces/IPSFOperator.h"
#include "interfaces/IGeometry.h"
#include "interfaces/IReconstructor.h"
#include "spect/Forward.h"

namespace spect {
namespace test {

// ============ Types.h 测试 ============

TEST(InterfaceTest, StatusEnum) {
    EXPECT_EQ(statusToString(Status::Ok), "Ok");
    EXPECT_EQ(statusToString(Status::InvalidArgument), "InvalidArgument");
    EXPECT_EQ(statusToString(Status::FileNotFound), "FileNotFound");
}

TEST(InterfaceTest, Dims3D) {
    Dims3D dims{64, 64, 32};
    EXPECT_EQ(dims.nx, 64);
    EXPECT_EQ(dims.ny, 64);
    EXPECT_EQ(dims.nz, 32);
    EXPECT_EQ(dims.total(), 64 * 64 * 32);
}

TEST(InterfaceTest, VoxelSize) {
    VoxelSize vs{0.5, 0.5, 1.0};
    EXPECT_DOUBLE_EQ(vs.dx, 0.5);
    EXPECT_DOUBLE_EQ(vs.dy, 0.5);
    EXPECT_DOUBLE_EQ(vs.dz, 1.0);
}

// ============ CSRHeader.h — legacy double draft (NOT on-disk CSRB v1) ============

TEST(InterfaceTest, LegacyDoubleCSRHeaderMagic) {
    EXPECT_EQ(CSR_MAGIC, 0x43535242);  // 'CSRB'
}

TEST(InterfaceTest, LegacyDoubleCSRHeaderDraftSize) {
    EXPECT_EQ(sizeof(CSRHeader), 32u);
}

TEST(InterfaceTest, CSRHeaderValid) {
    CSRHeader header;
    header.magic = CSR_MAGIC;
    header.version = 1;
    header.rows = 100;
    header.cols = 50;
    header.nnz = 1000;
    
    EXPECT_TRUE(header.isValid());
    
    header.magic = 0;
    EXPECT_FALSE(header.isValid());
}

TEST(InterfaceTest, CSRHeaderOffsets) {
    CSRHeader header;
    header.magic = CSR_MAGIC;
    header.version = 1;
    header.rows = 10;
    header.cols = 5;
    header.nnz = 30;
    
    // row_ptr: 11 * 8 = 88 bytes
    // col_idx: 30 * 4 = 120 bytes
    // values: 30 * 8 = 240 bytes
    EXPECT_EQ(header.rowPtrOffset(), 32);
    EXPECT_EQ(header.colIdxOffset(), 32 + 11 * 8);
    EXPECT_EQ(header.valuesOffset(), 32 + 11 * 8 + 30 * 4);
    EXPECT_EQ(header.totalFileSize(), 32 + 88 + 120 + 240);
}

TEST(InterfaceTest, MHDWritersUseZeroRawHeaderSize) {
    const auto tmpDir = std::filesystem::temp_directory_path() / "spect_mhd_header_test";
    std::filesystem::create_directories(tmpDir);
    const auto path = (tmpDir / "writer_header").string();

    spect::io::writeMHD(path, std::vector<double>{1.0, 2.0}, 1, 1, 2, 0.3f, 0.3f, 0.3f);

    std::ifstream mhd(tmpDir / "writer_header.mhd");
    ASSERT_TRUE(mhd.good());
    const std::string text((std::istreambuf_iterator<char>(mhd)),
                           std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("HeaderSize = 0"), std::string::npos);
    EXPECT_EQ(text.find("HeaderSize = -1"), std::string::npos);

    const auto outDir = tmpDir / "inline";
    spect::io::saveMHDRAW(std::vector<double>{3.0}, 1, 1, 1, 0.3, 0.3, 0.3,
                          outDir.string(), "inline_header", -1);
    std::ifstream inlineMhd(outDir / "inline_header.mhd");
    ASSERT_TRUE(inlineMhd.good());
    const std::string inlineText((std::istreambuf_iterator<char>(inlineMhd)),
                                 std::istreambuf_iterator<char>());
    EXPECT_NE(inlineText.find("HeaderSize = 0"), std::string::npos);
    EXPECT_EQ(inlineText.find("HeaderSize = -1"), std::string::npos);
}

// ============ CSRView.h 测试 ============

TEST(InterfaceTest, CSRViewDefault) {
    CSRView view;
    EXPECT_FALSE(view.isValid());
    EXPECT_EQ(view.row_ptr, nullptr);
}

TEST(InterfaceTest, CSRViewRowRange) {
    int64_t row_ptr[] = {0, 3, 5, 10};
    int32_t col_idx[] = {0, 1, 2, 0, 2, 0, 1, 2, 3, 4};
    double values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    CSRView view;
    view.row_ptr = row_ptr;
    view.col_idx = col_idx;
    view.values = values;
    view.rows = 3;
    view.cols = 5;
    view.nnz = 10;
    
    EXPECT_TRUE(view.isValid());
    
    int64_t start, end;
    view.getRowRange(0, start, end);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(end, 3);
    
    EXPECT_EQ(view.getRowNnz(0), 3);
    EXPECT_EQ(view.getRowNnz(1), 2);
    EXPECT_EQ(view.getRowNnz(2), 5);
}

// ============ IPSFOperator.h 测试 ============

TEST(InterfaceTest, NoPSFIdentity) {
    NoPSF nopsf;
    EXPECT_EQ(nopsf.type(), PSFType::None);
    EXPECT_TRUE(nopsf.isIdentity());
    EXPECT_TRUE(nopsf.isSymmetric());
    
    Dims3D dims{4, 4, 4};
    std::vector<double> input(dims.total(), 1.0);
    std::vector<double> output(dims.total(), 0.0);
    
    nopsf.apply(input.data(), output.data(), dims);
    
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_DOUBLE_EQ(input[i], output[i]);
    }
}

// ============ 抽象接口虚函数表测试 ============

// 测试纯虚接口可以被继承
class MockSparseOps : public ISparseOps {
public:
    BackendType backend() const override { return BackendType::Reference; }
    
    void spmv(const CSRView& A, const double* x, double* y, int threads) const override {
        // 简单实现
        for (int64_t i = 0; i < A.rows; ++i) {
            double sum = 0.0;
            for (int64_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                sum += A.values[k] * x[A.col_idx[k]];
            }
            y[i] = sum;
        }
    }
    
    void spmvT(const CSRView& A, const double* x, double* y, int threads) const override {
        for (int64_t j = 0; j < A.cols; ++j) y[j] = 0.0;
        for (int64_t i = 0; i < A.rows; ++i) {
            for (int64_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                y[A.col_idx[k]] += A.values[k] * x[i];
            }
        }
    }
    
    void computeNormAT(const CSRView& A, double* norm, int threads) const override {
        for (int64_t j = 0; j < A.cols; ++j) norm[j] = 0.0;
        for (int64_t i = 0; i < A.rows; ++i) {
            for (int64_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                norm[A.col_idx[k]] += A.values[k];
            }
        }
    }
};

TEST(InterfaceTest, MockSparseOpsCompiles) {
    MockSparseOps ops;
    EXPECT_EQ(ops.backend(), BackendType::Reference);
}

// ============ Forward.h 测试 ============

TEST(InterfaceTest, ForwardDeclarationsCompile) {
    // 仅验证前向声明可以编译
    // 使用指针/引用避免需要完整类型
    spect::ISparseOps* sparse_ptr = nullptr;
    spect::IProjector* proj_ptr = nullptr;
    spect::IRegularizer* reg_ptr = nullptr;
    
    EXPECT_EQ(sparse_ptr, nullptr);
    EXPECT_EQ(proj_ptr, nullptr);
    EXPECT_EQ(reg_ptr, nullptr);
}

// ============ SparseOpsRef测试 【T3.1】 ============

} // namespace test
} // namespace spect

// SparseOpsRef头文件 (在spect命名空间外)
#include "sparse/SparseOpsRef.h"

namespace spect {
namespace sparse {
namespace test {

// 辅助函数：创建简单的测试CSR矩阵
// 3x4矩阵:
// [1 0 2 0]
// [0 3 0 4]
// [5 0 6 0]
struct TestMatrix {
    std::vector<int64_t> row_ptr = {0, 2, 4, 6};
    std::vector<int32_t> col_idx = {0, 2, 1, 3, 0, 2};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    CSRView view() const {
        CSRView v;
        v.rows = 3;
        v.cols = 4;
        v.nnz = 6;
        v.row_ptr = row_ptr.data();
        v.col_idx = col_idx.data();
        v.values = values.data();
        return v;
    }
};

TEST(SparseOpsRefTest, Backend) {
    const auto& ops = getSparseOpsRef();
    EXPECT_EQ(ops.backend(), BackendType::Reference);
}

TEST(SparseOpsRefTest, SpMV) {
    TestMatrix A;
    const auto& ops = getSparseOpsRef();
    
    // x = [1, 2, 3, 4]
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y(3, 0.0);
    
    ops.spmv(A.view(), x.data(), y.data(), 1);
    
    // y[0] = 1*1 + 2*3 = 7
    // y[1] = 3*2 + 4*4 = 22
    // y[2] = 5*1 + 6*3 = 23
    EXPECT_DOUBLE_EQ(y[0], 7.0);
    EXPECT_DOUBLE_EQ(y[1], 22.0);
    EXPECT_DOUBLE_EQ(y[2], 23.0);
}

TEST(SparseOpsRefTest, SpMVT) {
    TestMatrix A;
    const auto& ops = getSparseOpsRef();
    
    // x = [1, 2, 3]
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y(4, 0.0);  // 必须初始化为0
    
    ops.spmvT(A.view(), x.data(), y.data(), 1);
    
    // y = A^T * x
    // col 0: 1*1 + 5*3 = 16
    // col 1: 3*2 = 6
    // col 2: 2*1 + 6*3 = 20
    // col 3: 4*2 = 8
    EXPECT_DOUBLE_EQ(y[0], 16.0);
    EXPECT_DOUBLE_EQ(y[1], 6.0);
    EXPECT_DOUBLE_EQ(y[2], 20.0);
    EXPECT_DOUBLE_EQ(y[3], 8.0);
}

TEST(SparseOpsRefTest, ComputeNormAT) {
    TestMatrix A;
    const auto& ops = getSparseOpsRef();
    
    std::vector<double> norm(4);
    ops.computeNormAT(A.view(), norm.data(), 1);
    
    // norm = A^T * 1 = column sums
    // col 0: 1 + 5 = 6
    // col 1: 3
    // col 2: 2 + 6 = 8
    // col 3: 4
    EXPECT_DOUBLE_EQ(norm[0], 6.0);
    EXPECT_DOUBLE_EQ(norm[1], 3.0);
    EXPECT_DOUBLE_EQ(norm[2], 8.0);
    EXPECT_DOUBLE_EQ(norm[3], 4.0);
}

TEST(SparseOpsRefTest, SpMVScaled) {
    TestMatrix A;
    const auto& ops = getSparseOpsRef();
    
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {1.0, 1.0, 1.0};  // 初始值
    
    // y = 2.0 * A * x + 0.5 * y
    ops.spmvScaled(A.view(), x.data(), y.data(), 2.0, 0.5, 1);
    
    // y[0] = 2.0 * 7 + 0.5 * 1 = 14.5
    // y[1] = 2.0 * 22 + 0.5 * 1 = 44.5
    // y[2] = 2.0 * 23 + 0.5 * 1 = 46.5
    EXPECT_DOUBLE_EQ(y[0], 14.5);
    EXPECT_DOUBLE_EQ(y[1], 44.5);
    EXPECT_DOUBLE_EQ(y[2], 46.5);
}

} // namespace test
} // namespace sparse
} // namespace spect

// ============ SparseOpsCPU测试 【T3.2】 ============

#include "sparse/SparseOpsCPU.h"
#include "sparse/SparseOpsBackend.h"

namespace spect {
namespace sparse {
namespace cpu_test {

// 使用与Ref相同的测试矩阵
struct TestMatrix {
    std::vector<int64_t> row_ptr = {0, 2, 4, 6};
    std::vector<int32_t> col_idx = {0, 2, 1, 3, 0, 2};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    CSRView view() const {
        CSRView v;
        v.rows = 3;
        v.cols = 4;
        v.nnz = 6;
        v.row_ptr = row_ptr.data();
        v.col_idx = col_idx.data();
        v.values = values.data();
        return v;
    }
};

TEST(SparseOpsCPUTest, Backend) {
    const auto& ops = getSparseOpsCPU();
    EXPECT_EQ(ops.backend(), BackendType::CPU);
}

TEST(SparseOpsCPUTest, SpMV_MatchesRef) {
    TestMatrix A;
    const auto& ref = getSparseOpsRef();
    const auto& cpu = getSparseOpsCPU();
    
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y_ref(3), y_cpu(3);
    
    ref.spmv(A.view(), x.data(), y_ref.data(), 1);
    cpu.spmv(A.view(), x.data(), y_cpu.data(), 4);  // 使用4线程
    
    // 验证结果与参考实现一致
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(y_cpu[i], y_ref[i]) << "Mismatch at index " << i;
    }
}

TEST(SparseOpsCPUTest, SpMVT_MatchesRef) {
    TestMatrix A;
    const auto& ref = getSparseOpsRef();
    const auto& cpu = getSparseOpsCPU();
    
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y_ref(4, 0.0), y_cpu(4, 0.0);
    
    ref.spmvT(A.view(), x.data(), y_ref.data(), 1);
    cpu.spmvT(A.view(), x.data(), y_cpu.data(), 4);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(y_cpu[i], y_ref[i]) << "Mismatch at index " << i;
    }
}

TEST(SparseOpsCPUTest, ComputeNormAT_MatchesRef) {
    TestMatrix A;
    const auto& ref = getSparseOpsRef();
    const auto& cpu = getSparseOpsCPU();
    
    std::vector<double> norm_ref(4), norm_cpu(4);
    
    ref.computeNormAT(A.view(), norm_ref.data(), 1);
    cpu.computeNormAT(A.view(), norm_cpu.data(), 4);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(norm_cpu[i], norm_ref[i]) << "Mismatch at index " << i;
    }
}

TEST(SparseOpsCPUTest, SpMVScaled_MatchesRef) {
    TestMatrix A;
    const auto& ref = getSparseOpsRef();
    const auto& cpu = getSparseOpsCPU();
    
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y_ref = {1.0, 1.0, 1.0};
    std::vector<double> y_cpu = {1.0, 1.0, 1.0};
    
    ref.spmvScaled(A.view(), x.data(), y_ref.data(), 2.0, 0.5, 1);
    cpu.spmvScaled(A.view(), x.data(), y_cpu.data(), 2.0, 0.5, 4);
    
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(y_cpu[i], y_ref[i]) << "Mismatch at index " << i;
    }
}

// 大矩阵多线程测试
TEST(SparseOpsCPUTest, LargeMatrix_MultiThread) {
    // 创建1000x1000稀疏矩阵，每行约10个非零元素
    const int N = 1000;
    const int nnz_per_row = 10;
    
    std::vector<int64_t> row_ptr(N + 1);
    std::vector<int32_t> col_idx;
    std::vector<double> values;
    
    row_ptr[0] = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < nnz_per_row; ++j) {
            int col = (i + j * 100) % N;
            col_idx.push_back(col);
            values.push_back(1.0 + (i % 10) * 0.1);
        }
        row_ptr[i + 1] = row_ptr[i] + nnz_per_row;
    }
    
    CSRView A;
    A.rows = N;
    A.cols = N;
    A.nnz = static_cast<int64_t>(values.size());
    A.row_ptr = row_ptr.data();
    A.col_idx = col_idx.data();
    A.values = values.data();
    
    std::vector<double> x(N, 1.0);
    std::vector<double> y_ref(N), y_cpu(N);
    
    const auto& ref = getSparseOpsRef();
    const auto& cpu = getSparseOpsCPU();
    
    ref.spmv(A, x.data(), y_ref.data(), 1);
    cpu.spmv(A, x.data(), y_cpu.data(), 4);
    
    // 验证误差在容许范围内
    double maxErr = 0.0;
    for (int i = 0; i < N; ++i) {
        maxErr = std::max(maxErr, std::abs(y_cpu[i] - y_ref[i]));
    }
    EXPECT_LT(maxErr, 1e-10) << "Large matrix SpMV mismatch";
}

TEST(SparseOpsCPUTest, CudaAvailabilityFlag) {
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)
    EXPECT_TRUE(isCudaSparseOpsAvailable());
#else
    EXPECT_FALSE(isCudaSparseOpsAvailable());
#endif
}

TEST(InterfaceTest, TruncatedMhdRawThrows) {
    const auto tmpDir = std::filesystem::temp_directory_path() /
                        "spect_mhd_truncated_test";
    std::filesystem::create_directories(tmpDir);
    const auto mhdPath = tmpDir / "truncated.mhd";
    const auto rawPath = tmpDir / "truncated.raw";
    {
        std::ofstream mhd(mhdPath);
        ASSERT_TRUE(mhd.good());
        mhd << "ObjectType = Image\nNDims = 3\nDimSize = 2 1 1\n"
               "ElementSpacing = 1 1 1\nElementType = MET_DOUBLE\n"
               "ElementDataFile = truncated.raw\n";
    }
    {
        std::ofstream raw(rawPath, std::ios::binary);
        const double oneValue = 1.0;
        raw.write(reinterpret_cast<const char*>(&oneValue), sizeof(oneValue));
    }

    int nx = 0, ny = 0, nz = 0;
    float dx = 0, dy = 0, dz = 0;
    EXPECT_THROW((void)spect::io::loadVolumeWithSpacing(
                     mhdPath.string(), nx, ny, nz, dx, dy, dz),
                 std::runtime_error);
    std::filesystem::remove_all(tmpDir);
}
} // namespace cpu_test
} // namespace sparse
} // namespace spect


// ============ App模块测试 【T2.3】 ============
#include "app/ReconContext.h"
#include "app/ReconContextAdapter.h"

namespace spect {
namespace app {
namespace test {

TEST(AppTest, ReconContextInit) {
    ReconContext ctx;
    ctx.args.Nx = 100;
    ctx.args.Ny = 100;
    ctx.args.Nz = 50;
    ctx.args.Vx = 0.5f;
    ctx.args.Vy = 0.5f;
    ctx.args.Vz = 1.0f;
    ctx.args.numAngles = 72;
    ctx.args.detNz = 40;
    ctx.args.detNx = 40;
    ctx.args.numThreads = 4;
    
    ctx.initFromArgs();
    
    EXPECT_EQ(ctx.Nx, 100);
    EXPECT_EQ(ctx.Ny, 100);
    EXPECT_EQ(ctx.Nz, 50);
    EXPECT_EQ(ctx.baseNz, 50);
    EXPECT_DOUBLE_EQ(ctx.voxelSpacing[0], 0.5);
    EXPECT_EQ(ctx.numVoxels, 100 * 100 * 50);
    EXPECT_EQ(ctx.numAngles, 72);
    EXPECT_EQ(ctx.raysPerAngle, 40 * 40);
}

TEST(AppTest, ReconContextSingleBed) {
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    ctx.bedCount = 1;
    ctx.baseNz = 32;
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    EXPECT_EQ(ctx.combinedNz, 32);
    EXPECT_EQ(ctx.Nz, 32);
    EXPECT_FALSE(ctx.isMultiBed());
    ASSERT_EQ(ctx.bedZStarts.size(), 1);
    EXPECT_EQ(ctx.bedZStarts[0], 0);
}

TEST(AppTest, ReconContextMultiBedNoOverlap) {
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    // Simulate multi-bed by setting projPaths
    ctx.args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat"};
    ctx.args.bedOverlaps = {0, 0};  // No overlap
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    EXPECT_EQ(ctx.bedCount, 3);
    EXPECT_EQ(ctx.combinedNz, 32 * 3);  // 96
    EXPECT_TRUE(ctx.isMultiBed());
    ASSERT_EQ(ctx.bedZStarts.size(), 3);
    EXPECT_EQ(ctx.bedZStarts[0], 0);
    EXPECT_EQ(ctx.bedZStarts[1], 32);
    EXPECT_EQ(ctx.bedZStarts[2], 64);
}

TEST(AppTest, ReconContextMultiBedWithOverlap) {
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    // Simulate 4 beds
    ctx.args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat", "bed4.mat"};
    ctx.args.bedOverlaps = {8, 8, 8};  // 8 voxel overlap between each pair
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    EXPECT_EQ(ctx.bedCount, 4);
    // Total = 32 * 4 - (8 + 8 + 8) = 128 - 24 = 104
    EXPECT_EQ(ctx.combinedNz, 104);
    EXPECT_TRUE(ctx.isMultiBed());
    ASSERT_EQ(ctx.bedZStarts.size(), 4);
    EXPECT_EQ(ctx.bedZStarts[0], 0);
    EXPECT_EQ(ctx.bedZStarts[1], 24);  // 32 - 8
    EXPECT_EQ(ctx.bedZStarts[2], 48);  // 24 + 24
    EXPECT_EQ(ctx.bedZStarts[3], 72);  // 48 + 24
    
    // Test getBedZRange
    auto range0 = ctx.getBedZRange(0);
    EXPECT_EQ(range0.first, 0);
    EXPECT_EQ(range0.second, 32);
    
    auto range3 = ctx.getBedZRange(3);
    EXPECT_EQ(range3.first, 72);
    EXPECT_EQ(range3.second, 104);
}

// ============ T5.1 Tests: BedInfo and FusionStrategy ============

TEST(AppTest, ReconContextBedInfoSingleBed) {
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    ctx.args.projPaths = {"data.mat"};
    ctx.raysPerBed = 1000;
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    // Single bed should have one BedInfo
    ASSERT_EQ(ctx.bedInfos.size(), 1);
    const auto& info = ctx.getBedInfo(0);
    EXPECT_EQ(info.index, 0);
    EXPECT_EQ(info.zStart, 0);
    EXPECT_EQ(info.zEnd, 32);
    EXPECT_EQ(info.localNz, 32);
    EXPECT_DOUBLE_EQ(info.attenFactor, 1.0);
    EXPECT_EQ(info.rayOffset, 0);
    EXPECT_EQ(info.rayCount, 1000);
}

TEST(AppTest, ReconContextBedInfoMultiBed) {
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    ctx.args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat"};
    ctx.args.bedOverlaps = {8, 8};  // 8 voxel overlap
    ctx.raysPerBed = 2000;
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    // Should have 3 BedInfos
    ASSERT_EQ(ctx.bedInfos.size(), 3);
    
    // Bed 0
    const auto& info0 = ctx.getBedInfo(0);
    EXPECT_EQ(info0.index, 0);
    EXPECT_EQ(info0.zStart, 0);
    EXPECT_EQ(info0.localNz, 32);
    EXPECT_EQ(info0.rayOffset, 0);
    EXPECT_EQ(info0.rayCount, 2000);
    
    // Bed 1
    const auto& info1 = ctx.getBedInfo(1);
    EXPECT_EQ(info1.index, 1);
    EXPECT_EQ(info1.zStart, 24);  // 32 - 8
    EXPECT_EQ(info1.rayOffset, 2000);
    
    // Bed 2
    const auto& info2 = ctx.getBedInfo(2);
    EXPECT_EQ(info2.index, 2);
    EXPECT_EQ(info2.zStart, 48);  // 24 + 24
    EXPECT_EQ(info2.rayOffset, 4000);
}

TEST(AppTest, ReconContextSetBedAttenFactor) {
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    ctx.args.projPaths = {"bed1.mat", "bed2.mat"};
    ctx.args.bedOverlaps = {8};
    ctx.raysPerBed = 1000;
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    // Set custom attenuation factors
    ctx.setBedAttenFactor(0, 1.2);
    ctx.setBedAttenFactor(1, 0.85);
    
    EXPECT_DOUBLE_EQ(ctx.getBedInfo(0).attenFactor, 1.2);
    EXPECT_DOUBLE_EQ(ctx.getBedInfo(1).attenFactor, 0.85);
    
    // Invalid index should not crash
    ctx.setBedAttenFactor(-1, 1.5);
    ctx.setBedAttenFactor(10, 1.5);
}

TEST(AppTest, ReconContextGetBedInfoOutOfRange) {
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    ctx.args.projPaths = {"data.mat"};
    ctx.raysPerBed = 1000;
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    // Valid access
    EXPECT_NO_THROW(ctx.getBedInfo(0));
    
    // Invalid access should throw
    EXPECT_THROW(ctx.getBedInfo(-1), std::out_of_range);
    EXPECT_THROW(ctx.getBedInfo(1), std::out_of_range);
}

TEST(AppTest, ReconContextFusionStrategy) {
    ReconContext ctx;
    
    // Default strategy should be Weighted
    EXPECT_EQ(ctx.fusionStrategy, BedFusionStrategy::Weighted);
    
    // Can change strategy
    ctx.fusionStrategy = BedFusionStrategy::Average;
    EXPECT_EQ(ctx.fusionStrategy, BedFusionStrategy::Average);
    
    ctx.fusionStrategy = BedFusionStrategy::Max;
    EXPECT_EQ(ctx.fusionStrategy, BedFusionStrategy::Max);
}

TEST(AppTest, ReconContextOverlapWeight) {
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    ctx.args.projPaths = {"bed1.mat", "bed2.mat"};
    ctx.args.bedOverlaps = {8};  // 8 voxel overlap
    ctx.raysPerBed = 1000;
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    // Bed 0: zStart=0, zEnd=32
    // Bed 1: zStart=24, zEnd=56
    // Overlap region: z=24 to z=32
    
    // Single bed returns 1.0
    EXPECT_DOUBLE_EQ(ctx.getOverlapWeight(10, 0), 1.0);  // Bed 0 non-overlap
    EXPECT_DOUBLE_EQ(ctx.getOverlapWeight(40, 1), 1.0);  // Bed 1 non-overlap
    
    // In overlap region - weights should transition
    // At z=24 (overlap start): Bed 0 weight = 1.0-0.0 = 1.0, Bed 1 weight = 0.0
    // At z=28 (middle): Bed 0 weight ~= 0.5, Bed 1 weight ~= 0.5
    // At z=31 (near end): Bed 0 weight ~= 0.125, Bed 1 weight ~= 0.875
    
    // Bed 0 in overlap region: weight decreases
    double w0_24 = ctx.getOverlapWeight(24, 0);
    double w0_28 = ctx.getOverlapWeight(28, 0);
    EXPECT_GE(w0_24, w0_28);  // Should decrease
    
    // Bed 1 in overlap region: weight increases
    double w1_24 = ctx.getOverlapWeight(24, 1);
    double w1_28 = ctx.getOverlapWeight(28, 1);
    EXPECT_LE(w1_24, w1_28);  // Should increase
}

// Integration test: ConfigParser with real config file
TEST(AppTest, ConfigParserIntegration) {
    // Test ConfigParser can parse the default config file
    spect::cli::ConfigParser parser;
    std::string configPath = "config/spect_config.ini";
    
    // Try to load config file (may not exist in test environment)
    if (std::filesystem::exists(configPath)) {
        ASSERT_TRUE(parser.load(configPath));
        
        // Verify some expected values from the config
        EXPECT_EQ(parser.getInt("Basic", "threads", 0), 16);
        EXPECT_EQ(parser.getInt("Basic", "verbosity", 0), 1);
        EXPECT_EQ(parser.getInt("Detector", "channels", 0), 40);
        EXPECT_EQ(parser.getInt("Detector", "angles", 0), 360);
        EXPECT_TRUE(parser.getBool("SystemMatrix", "map_csr", false));
        EXPECT_EQ(parser.getString("PSF", "mode", ""), "None");
        EXPECT_DOUBLE_EQ(parser.getDouble("PSF", "fwhm", 0.0), 1.5);
        EXPECT_DOUBLE_EQ(parser.getDouble("PSF", "min_sigmas", 0.0), 4.0);
        EXPECT_DOUBLE_EQ(
            parser.getDouble("Filtering", "butterworth_cutoff", 0.0), 0.35);
        EXPECT_EQ(parser.getInt("Filtering", "butterworth_order", 0), 2);
        EXPECT_EQ(parser.getString("Regularization", "type", ""), "None");
    } else {
        GTEST_SKIP() << "Config file not found: " << configPath;
    }
}

// ============ T5.2 Tests: ReconContextAdapter ============

TEST(AppTest, ReconContextAdapterBasic) {
    using namespace spect::app;
    
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    ctx.args.numAngles = 360;
    ctx.args.detectorChannels = 40;
    ctx.args.numThreads = 8;
    ctx.args.projPaths = {"data.mat"};
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    ReconOpts opts = createReconOpts(ctx);
    
    ASSERT_NE(opts.sparseOps, nullptr);
    EXPECT_EQ(opts.threads, 8);
    EXPECT_EQ(opts.bedCount, 1);
    EXPECT_EQ(opts.num_angles, 360);
    EXPECT_EQ(opts.det_nz, 40);
    EXPECT_EQ(opts.det_nx, 40);
    EXPECT_EQ(opts.psf.mode, PSFMode::None);
}

TEST(AppTest, ReconContextAdapterMultiBed) {
    using namespace spect::app;
    
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    ctx.args.numAngles = 360;
    ctx.args.detectorChannels = 40;
    ctx.args.numThreads = 16;
    ctx.args.projPaths = {"bed1.mat", "bed2.mat", "bed3.mat"};
    ctx.args.bedOverlaps = {8, 8};
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    ReconOpts opts = createReconOpts(ctx);
    
    ASSERT_NE(opts.sparseOps, nullptr);
    EXPECT_EQ(opts.bedCount, 3);
    ASSERT_EQ(opts.bedOffsetVoxels.size(), 3);
    
    // Bed 0: offset = 0
    EXPECT_EQ(opts.bedOffsetVoxels[0], 0);
    
    // Bed 1: offset = zStart[1] * sliceSize = 24 * 64 * 64
    int sliceSize = 64 * 64;
    EXPECT_EQ(opts.bedOffsetVoxels[1], 24 * sliceSize);
    
    // Bed 2: offset = zStart[2] * sliceSize = 48 * 64 * 64
    EXPECT_EQ(opts.bedOffsetVoxels[2], 48 * sliceSize);
}

TEST(AppTest, ReconContextAdapterFromArgs) {
    using namespace spect::app;
    
    spect::cli::ReconArgs args;
    args.Nx = 100;
    args.Ny = 100;
    args.Nz = 50;
    args.numAngles = 180;
    args.detectorChannels = 64;
    args.numThreads = 4;
    args.numSubsets = 12;
    args.projPaths = {"bed1.mat", "bed2.mat"};
    args.bedOverlaps = {10};
    
    ReconOpts opts = createReconOpts(args);
    
    EXPECT_EQ(opts.threads, 4);
    EXPECT_EQ(opts.n_subsets, 12);
    EXPECT_EQ(opts.bedCount, 2);
    EXPECT_EQ(opts.num_angles, 180);
    
    // Bed offsets
    ASSERT_EQ(opts.bedOffsetVoxels.size(), 2);
    EXPECT_EQ(opts.bedOffsetVoxels[0], 0);
    // zStart[1] = baseNz - overlap = 50 - 10 = 40
    EXPECT_EQ(opts.bedOffsetVoxels[1], 40LL * 100 * 100);
}

TEST(AppTest, ReconContextAdapterUsesCudaRegardlessOfArgBackend) {
    using namespace spect::app;

    spect::cli::ReconArgs args;
    args.Nx = 32;
    args.Ny = 32;
    args.Nz = 16;
    args.numAngles = 90;
    args.detectorChannels = 20;
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)
    ReconOpts opts = createReconOpts(args);
    ASSERT_NE(opts.sparseOps, nullptr);
    EXPECT_EQ(opts.sparseOps->backend(), BackendType::CUDA);
#else
    EXPECT_THROW(createReconOpts(args), std::runtime_error);
#endif
}

TEST(AppTest, ReconContextAdapterUsesCuda) {
    using namespace spect::app;

    spect::cli::ReconArgs args;
    args.Nx = 32;
    args.Ny = 32;
    args.Nz = 16;
    args.numAngles = 90;
    args.detectorChannels = 20;
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)
    ReconOpts opts = createReconOpts(args);
    ASSERT_NE(opts.sparseOps, nullptr);
    EXPECT_EQ(opts.sparseOps->backend(), BackendType::CUDA);
#else
    EXPECT_THROW(createReconOpts(args), std::runtime_error);
#endif
}

TEST(AppTest, ReconContextAdapterHelpers) {
    using namespace spect::app;
    
    ReconContext ctx;
    ctx.args.Nx = 64;
    ctx.args.Ny = 64;
    ctx.args.Nz = 32;
    ctx.args.numAngles = 360;
    ctx.args.detNz = 40;
    ctx.args.detNx = 50;
    ctx.args.projPaths = {"data.mat"};
    
    ctx.initFromArgs();
    ctx.calculateCombinedNz();
    
    // Total voxels
    EXPECT_EQ(calculateTotalVoxels(ctx), 64 * 64 * 32);
    
    // Rays per bed
    EXPECT_EQ(calculateRaysPerBed(ctx), 40 * 50 * 360);
}

} // namespace test
} // namespace app
} // namespace spect




