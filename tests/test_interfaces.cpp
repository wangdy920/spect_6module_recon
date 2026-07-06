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

TEST(SparseOpsCPUTest, SparseBackendChoiceParse) {
    SparseBackendChoice choice = SparseBackendChoice::Auto;
    EXPECT_TRUE(parseSparseBackendChoice("auto", choice));
    EXPECT_EQ(choice, SparseBackendChoice::Auto);
    EXPECT_TRUE(parseSparseBackendChoice("cpu", choice));
    EXPECT_EQ(choice, SparseBackendChoice::CPU);
    EXPECT_TRUE(parseSparseBackendChoice("reference", choice));
    EXPECT_EQ(choice, SparseBackendChoice::Reference);
    EXPECT_TRUE(parseSparseBackendChoice("cuda", choice));
    EXPECT_EQ(choice, SparseBackendChoice::CUDA);
    EXPECT_FALSE(parseSparseBackendChoice("invalid", choice));
}

TEST(SparseOpsCPUTest, CudaAvailabilityFlag) {
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)
    EXPECT_TRUE(isCudaSparseOpsAvailable());
#else
    EXPECT_FALSE(isCudaSparseOpsAvailable());
#endif
}

TEST(SparseOpsCPUTest, SparseBackendSelectorFallback) {
    SparseBackendChoice resolved = SparseBackendChoice::Auto;
    bool parsedOk = false;
    const auto& ops = selectSparseOps("cuda", &resolved, &parsedOk);
    EXPECT_TRUE(parsedOk);
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)
    EXPECT_EQ(ops.backend(), BackendType::CUDA);
    EXPECT_EQ(resolved, SparseBackendChoice::CUDA);
#else
    EXPECT_NE(ops.backend(), BackendType::CUDA);
    EXPECT_TRUE(resolved == SparseBackendChoice::CPU ||
                resolved == SparseBackendChoice::Reference);
#endif
}
} // namespace cpu_test
} // namespace sparse
} // namespace spect

// ============ 正则化器测试 【T4.1】 ============

// ============ 新正则化器层次结构测试 【T4.1】 ============
#include "regularizer/RegularizerBase.h"
#include "regularizer/TVRegularizer.h"

namespace spect {
namespace regularizer {
namespace test {

// 使用不同的测试套件名，避免与旧fixture冲突
TEST(NewRegularizerTest, NoneRegularizer) {
    NoneRegularizer none;
    EXPECT_EQ(none.name(), "None");
    
    Dims3D dims{4, 4, 4};
    std::vector<double> f(64, 1.0);
    std::vector<double> grad(64);
    
    EXPECT_DOUBLE_EQ(none.energy(f.data(), dims, nullptr), 0.0);
    none.gradient(f.data(), grad.data(), dims, nullptr, 1);
    
    for (double g : grad) {
        EXPECT_DOUBLE_EQ(g, 0.0);
    }
}

TEST(NewRegularizerTest, TVIsotropicEnergy) {
    TVIsotropicRegularizer tv;
    tv.setBeta(1.0);
    
    Dims3D dims{3, 3, 3};
    std::vector<double> f(27, 0.0);
    
    // 常数图像应有eps导致的微小能量 -> 只验证很小
    double E0 = tv.energy(f.data(), dims);
    EXPECT_LT(E0, 1e-3);  // 允许eps导致的误差
    
    // 中心设为1，其余为0 -> 有梯度
    f[13] = 1.0;  // 中心 (1,1,1)
    double E = tv.energy(f.data(), dims);
    EXPECT_GT(E, E0);  // 非均匀图像能量更高
}

TEST(NewRegularizerTest, TVIsotropicGradient) {
    TVIsotropicRegularizer tv(1e-8);  // 小eps
    tv.setBeta(1.0);
    
    Dims3D dims{5, 5, 5};
    std::vector<double> f(125, 0.0);
    std::vector<double> grad(125, 0.0);
    
    // 设置中心为1
    f[62] = 1.0;  // 中心 (2,2,2)
    
    tv.gradient(f.data(), grad.data(), dims, nullptr, 1);
    
    // 中心梯度应为正（向外）
    EXPECT_GT(grad[62], 0.0);
    
    // 邻居梯度应为负（从中心拉）
    // 这里只验证不全为0
    double sum = 0.0;
    for (double g : grad) sum += std::abs(g);
    EXPECT_GT(sum, 0.0);
}

TEST(NewRegularizerTest, TVHuberEnergy) {
    TVHuberRegularizer huber(0.01);  // delta = 0.01
    huber.setBeta(1.0);
    
    Dims3D dims{3, 3, 3};
    std::vector<double> f(27, 0.0);
    
    // 常数图像
    EXPECT_NEAR(huber.energy(f.data(), dims), 0.0, 1e-10);
    
    // 有梯度
    f[13] = 1.0;
    double E = huber.energy(f.data(), dims);
    EXPECT_GT(E, 0.0);
}

TEST(NewRegularizerTest, RegularizerFactory) {
    auto& factory = RegularizerFactory::instance();
    
    // 注册测试类型
    factory.registerType("TestNone", []{ return std::make_unique<NoneRegularizer>(); });
    
    EXPECT_TRUE(factory.hasType("TestNone"));
    EXPECT_FALSE(factory.hasType("NonExistent"));
    
    auto reg = factory.create("TestNone");
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->name(), "None");
    
    // 不存在的类型返回nullptr
    auto null_reg = factory.create("NonExistent");
    EXPECT_EQ(null_reg, nullptr);
}

TEST(NewRegularizerTest, RegularizerRegistrar) {
    // 使用模板自动注册
    static RegularizerRegistrar<TVIsotropicRegularizer> reg_tv("TVIso_Test");
    
    auto& factory = RegularizerFactory::instance();
    EXPECT_TRUE(factory.hasType("TVIso_Test"));
    
    auto reg = factory.create("TVIso_Test");
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->name(), "TVIsotropic");
}

} // namespace test
} // namespace regularizer
} // namespace spect

// ============ 新增正则化器测试 【T4.2】 ============
#include "regularizer/TikhonovRegularizer.h"
#include "regularizer/MRPRegularizer.h"
#include "regularizer/RDPRegularizer.h"
#include "regularizer/LogCoshRegularizer.h"
#include "regularizer/NLMRegularizer.h"
#include "regularizer/AllRegularizers.h"

namespace spect {
namespace regularizer {
namespace test {

// Tikhonov 正则化器测试
TEST(T42RegularizerTest, TikhonovBasic) {
    TikhonovRegularizer tik;
    tik.setBeta(1.0);
    EXPECT_EQ(tik.name(), "Tikhonov");
    
    Dims3D dims{3, 3, 3};
    std::vector<double> f(27, 1.0);  // 常数图像
    std::vector<double> grad(27);
    
    // 常数图像梯度为0
    tik.gradient(f.data(), grad.data(), dims, nullptr, 1);
    for (double g : grad) {
        EXPECT_NEAR(g, 0.0, 1e-10);
    }
    
    // 中心设为2，产生梯度
    f[13] = 2.0;
    tik.gradient(f.data(), grad.data(), dims, nullptr, 1);
    EXPECT_GT(grad[13], 0.0);  // 中心向下
}

// MRP 正则化器测试
TEST(T42RegularizerTest, MRPBasic) {
    MRPRegularizer mrp;
    mrp.setBeta(1.0);
    EXPECT_EQ(mrp.name(), "MRP");
    
    Dims3D dims{3, 3, 3};
    std::vector<double> f(27, 1.0);
    std::vector<double> grad(27);
    
    mrp.gradient(f.data(), grad.data(), dims, nullptr, 1);
    
    // 常数图像，所有邻居相等，梯度约为0
    for (size_t i = 0; i < 27; ++i) {
        EXPECT_NEAR(grad[i], 0.0, 1e-6);
    }
}

// RDP 正则化器测试
TEST(T42RegularizerTest, RDPBasic) {
    RDPRegularizer rdp(1e-6, 0.0);
    rdp.setBeta(1.0);
    EXPECT_EQ(rdp.name(), "RDP");
    
    Dims3D dims{3, 3, 3};
    std::vector<double> f(27, 1.0);
    std::vector<double> grad(27);
    
    // 常数图像
    rdp.gradient(f.data(), grad.data(), dims, nullptr, 1);
    for (double g : grad) {
        EXPECT_NEAR(g, 0.0, 1e-10);
    }
    
    // 非均匀图像
    f[13] = 2.0;
    rdp.gradient(f.data(), grad.data(), dims, nullptr, 1);
    EXPECT_GT(grad[13], 0.0);
}

// LogCosh 正则化器测试
TEST(T42RegularizerTest, LogCoshBasic) {
    LogCoshRegularizer lc(0.1);
    lc.setBeta(1.0);
    EXPECT_EQ(lc.name(), "LogCosh");
    
    Dims3D dims{3, 3, 3};
    std::vector<double> f(27, 0.5);
    std::vector<double> grad(27);
    
    // 常数图像
    lc.gradient(f.data(), grad.data(), dims, nullptr, 1);
    for (double g : grad) {
        EXPECT_NEAR(g, 0.0, 1e-10);
    }
    
    // 能量应为0
    double E = lc.energy(f.data(), dims);
    EXPECT_NEAR(E, 0.0, 1e-10);
}

// NLM 正则化器测试
TEST(T42RegularizerTest, NLMBasic) {
    NLMRegularizer nlm(0.1, 2, 1);
    nlm.setBeta(1.0);
    EXPECT_EQ(nlm.name(), "NLM");
    
    Dims3D dims{5, 5, 5};
    std::vector<double> f(125, 1.0);
    std::vector<double> grad(125);
    
    // 常数图像，梯度应接近0
    nlm.gradient(f.data(), grad.data(), dims, nullptr, 1);
    double maxGrad = 0.0;
    for (double g : grad) {
        maxGrad = std::max(maxGrad, std::abs(g));
    }
    EXPECT_LT(maxGrad, 1e-8);  // 应该非常小
}

// AllRegularizers 注册测试
TEST(T42RegularizerTest, RegisterBuiltinRegularizers) {
    registerBuiltinRegularizers();
    
    auto& factory = RegularizerFactory::instance();
    
    // 检查所有内置类型都已注册
    EXPECT_TRUE(factory.hasType("None"));
    EXPECT_TRUE(factory.hasType("TVIsotropic"));
    EXPECT_TRUE(factory.hasType("TVHuber"));
    EXPECT_TRUE(factory.hasType("Tikhonov"));
    EXPECT_TRUE(factory.hasType("MRP"));
    EXPECT_TRUE(factory.hasType("RDP"));
    EXPECT_TRUE(factory.hasType("LogCosh"));
    EXPECT_TRUE(factory.hasType("NLM"));
    EXPECT_TRUE(factory.hasType("Bowsher"));
    EXPECT_TRUE(factory.hasType("PLS"));
}

// 便捷创建函数测试
TEST(T42RegularizerTest, CreateRegularizerFunction) {
    registerBuiltinRegularizers();
    
    auto tv = createRegularizer("TVIsotropic", 0.5);
    ASSERT_NE(tv, nullptr);
    EXPECT_EQ(tv->name(), "TVIsotropic");
    EXPECT_DOUBLE_EQ(tv->beta(), 0.5);
    
    auto none = createRegularizer("None", 1.0);
    ASSERT_NE(none, nullptr);
    EXPECT_EQ(none->name(), "None");
    
    auto invalid = createRegularizer("InvalidType", 1.0);
    EXPECT_EQ(invalid, nullptr);
}

} // namespace test
} // namespace regularizer
} // namespace spect

// ============ 兼容层测试 【T4.3】 ============
#include "regularizer/RegularizerCompat.h"

namespace spect {
namespace regularizer {
namespace test {

TEST(RegularizerCompatTest, CreateFromRegType) {
    auto tv = createFromRegType(RegType::TVIsotropic, 0.5);
    ASSERT_NE(tv, nullptr);
    EXPECT_EQ(tv->name(), "TVIsotropic");
    EXPECT_DOUBLE_EQ(tv->beta(), 0.5);
    
    auto none = createFromRegType(RegType::None, 1.0);
    ASSERT_NE(none, nullptr);
    EXPECT_EQ(none->name(), "None");
    
    auto unknown = createFromRegType(RegType::Unknown, 1.0);
    EXPECT_EQ(unknown, nullptr);
}

TEST(RegularizerCompatTest, CreateFromRegConf) {
    RegConf conf;
    conf.type = RegType::RDP;
    conf.beta = 0.01;
    conf.delta = 1e-5;
    conf.gamma = 0.5;
    
    auto rdp = createFromRegConf(conf);
    ASSERT_NE(rdp, nullptr);
    EXPECT_EQ(rdp->name(), "RDP");
    EXPECT_DOUBLE_EQ(rdp->beta(), 0.01);
}

TEST(RegularizerCompatTest, ComputeGradientNewVsOld) {
    // 比较新旧API结果一致性
    const int N = 5;
    std::vector<double> f(N * N * N);
    for (size_t i = 0; i < f.size(); ++i) {
        f[i] = static_cast<double>(i) * 0.01;
    }
    
    RegConf conf;
    conf.type = RegType::Tikhonov;
    conf.beta = 1.0;
    
    std::vector<double> gradOld(N * N * N);
    std::vector<double> gradNew(N * N * N);
    
    // 旧API
    compute_reg_gradient(f, N, N, N, conf, nullptr, gradOld, 1);
    
    // 新API
    computeGradientNew(f, N, N, N, conf, nullptr, gradNew, 1);
    
    // 比较
    for (size_t i = 0; i < f.size(); ++i) {
        EXPECT_NEAR(gradNew[i], gradOld[i], 1e-10) << "Mismatch at index " << i;
    }
}

} // namespace test
} // namespace regularizer
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
        EXPECT_EQ(parser.getString("SystemMatrix", "sparse_backend", ""), "auto");
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

TEST(AppTest, ReconContextAdapterSparseBackendRef) {
    using namespace spect::app;

    spect::cli::ReconArgs args;
    args.Nx = 32;
    args.Ny = 32;
    args.Nz = 16;
    args.numAngles = 90;
    args.detectorChannels = 20;
    args.sparseBackend = "ref";

    ReconOpts opts = createReconOpts(args);

    ASSERT_NE(opts.sparseOps, nullptr);
    EXPECT_EQ(opts.sparseOps->backend(), BackendType::Reference);
}

TEST(AppTest, ReconContextAdapterSparseBackendCudaFallback) {
    using namespace spect::app;

    spect::cli::ReconArgs args;
    args.Nx = 32;
    args.Ny = 32;
    args.Nz = 16;
    args.numAngles = 90;
    args.detectorChannels = 20;
    args.sparseBackend = "cuda";

    ReconOpts opts = createReconOpts(args);

    ASSERT_NE(opts.sparseOps, nullptr);
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL)
    EXPECT_EQ(opts.sparseOps->backend(), BackendType::CUDA);
#else
    EXPECT_NE(opts.sparseOps->backend(), BackendType::CUDA);
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




