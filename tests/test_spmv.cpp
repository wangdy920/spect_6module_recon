/**
 * @file test_spmv.cpp
 * @brief SpMV (稀疏矩阵-向量乘法) 正确性和bitwise一致性测试
 * 
 * 测试内容:
 * 1. SpMV正确性测试 (与标量参考实现比较)
 * 2. Forward projection (y = A * x)
 * 3. Backward projection (x = A^T * y)  
 * 4. 多线程bitwise一致性测试
 * 5. computeNormAT 正确性测试
 */

#include <gtest/gtest.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <omp.h>

// 包含模块化头文件
#include "io/CSRBuilder.h"
#include "io/MatFileReader.h"
#include "sparse/SparseOpsRef.h"
#include "sparse/SparseOpsCPU.h"
#include "app/ProjectionOps.h"

using spect::computeNormAT;

namespace spect {
namespace test {

/**
 * @brief SpMV 测试夹具类
 */
class SpMVTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置随机种子以保证可重复性
        gen_.seed(42);
    }
    
    std::mt19937 gen_;
    
    /**
     * @brief 创建一个简单的测试CSR矩阵
     * 
     * 3x4 矩阵:
     * [1.0  0    2.0  0  ]
     * [0    3.0  0    4.0]
     * [5.0  0    0    6.0]
     */
    CSR createSimpleCSR() {
        CSR A;
        A.rows = 3;
        A.cols = 4;
        A.row_ptr = {0, 2, 4, 6};
        A.col_idx = {0, 2, 1, 3, 0, 3};
        A.val = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        return A;
    }
    
    /**
     * @brief 创建随机CSR矩阵
     */
    CSR createRandomCSR(int rows, int cols, int avgNnzPerRow) {
        CSR A;
        A.rows = rows;
        A.cols = cols;
        A.row_ptr.resize(rows + 1);
        A.row_ptr[0] = 0;
        
        std::uniform_int_distribution<int> nnzDist(1, avgNnzPerRow * 2);
        std::uniform_int_distribution<int> colDist(0, cols - 1);
        std::uniform_real_distribution<float> valDist(0.0f, 1.0f);
        
        for (int r = 0; r < rows; ++r) {
            int nnz = std::min(nnzDist(gen_), cols);
            std::vector<bool> used(cols, false);
            
            for (int k = 0; k < nnz; ++k) {
                int col = colDist(gen_);
                if (!used[col]) {
                    used[col] = true;
                    A.col_idx.push_back(static_cast<uint32_t>(col));
                    A.val.push_back(valDist(gen_));
                }
            }
            
            A.row_ptr[r + 1] = static_cast<uint32_t>(A.col_idx.size());
        }
        
        return A;
    }
    
    /**
     * @brief 标量参考实现: 前向投影 y = A * x
     */
    std::vector<double> refForwardProject(const CSR& A, const std::vector<double>& x) {
        std::vector<double> y(A.rows, 0.0);
        for (int row = 0; row < A.rows; ++row) {
            double sum = 0.0;
            for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
                sum += static_cast<double>(A.val[t]) * x[A.col_idx[t]];
            }
            y[row] = sum;
        }
        return y;
    }
    
    /**
     * @brief 标量参考实现: 反向投影 x = A^T * y
     */
    std::vector<double> refBackwardProject(const CSR& A, const std::vector<double>& y) {
        std::vector<double> x(A.cols, 0.0);
        for (int row = 0; row < A.rows; ++row) {
            for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
                x[A.col_idx[t]] += static_cast<double>(A.val[t]) * y[row];
            }
        }
        return x;
    }
    
    /**
     * @brief 标量参考实现: 计算 A^T 列和 (sensitivity)
     */
    std::vector<double> refComputeNormAT(const CSR& A) {
        std::vector<double> denom(A.cols, 0.0);
        for (int row = 0; row < A.rows; ++row) {
            for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
                denom[A.col_idx[t]] += static_cast<double>(A.val[t]);
            }
        }
        return denom;
    }
    
    /**
     * @brief 比较两个向量是否近似相等
     */
    bool vectorsNearlyEqual(const std::vector<double>& a, 
                           const std::vector<double>& b, 
                           double rtol = 1e-10, 
                           double atol = 1e-14) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = std::abs(a[i] - b[i]);
            double max_val = std::max(std::abs(a[i]), std::abs(b[i]));
            if (diff > atol + rtol * max_val) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * @brief 比较两个向量是否bitwise相同
     */
    bool vectorsBitwiseEqual(const std::vector<double>& a, 
                            const std::vector<double>& b) {
        if (a.size() != b.size()) return false;
        return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
    }
};

// ============ 基本正确性测试 ============

TEST_F(SpMVTest, ForwardProjectSimple) {
    CSR A = createSimpleCSR();
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    
    std::vector<double> y = refForwardProject(A, x);
    
    // 手工计算预期结果:
    // y[0] = 1*1 + 2*3 = 7
    // y[1] = 3*2 + 4*4 = 22
    // y[2] = 5*1 + 6*4 = 29
    EXPECT_DOUBLE_EQ(y[0], 7.0);
    EXPECT_DOUBLE_EQ(y[1], 22.0);
    EXPECT_DOUBLE_EQ(y[2], 29.0);
}

TEST_F(SpMVTest, BackwardProjectSimple) {
    CSR A = createSimpleCSR();
    std::vector<double> y = {1.0, 1.0, 1.0};
    
    std::vector<double> x = refBackwardProject(A, y);
    
    // 手工计算预期结果:
    // x[0] = 1*1 + 5*1 = 6
    // x[1] = 3*1 = 3
    // x[2] = 2*1 = 2
    // x[3] = 4*1 + 6*1 = 10
    EXPECT_DOUBLE_EQ(x[0], 6.0);
    EXPECT_DOUBLE_EQ(x[1], 3.0);
    EXPECT_DOUBLE_EQ(x[2], 2.0);
    EXPECT_DOUBLE_EQ(x[3], 10.0);
}

TEST_F(SpMVTest, NormATSimple) {
    CSR A = createSimpleCSR();
    
    std::vector<double> norm = refComputeNormAT(A);
    
    // 列和 = A^T 的行和
    // col 0: 1 + 5 = 6
    // col 1: 3 = 3
    // col 2: 2 = 2
    // col 3: 4 + 6 = 10
    EXPECT_DOUBLE_EQ(norm[0], 6.0);
    EXPECT_DOUBLE_EQ(norm[1], 3.0);
    EXPECT_DOUBLE_EQ(norm[2], 2.0);
    EXPECT_DOUBLE_EQ(norm[3], 10.0);
}

// ============ computeNormAT 与参考实现比较 ============

TEST_F(SpMVTest, ComputeNormATMatchesReference) {
    CSR A = createRandomCSR(100, 50, 5);
    
    // 参考实现
    std::vector<double> refNorm = refComputeNormAT(A);
    
    // 被测实现 (1线程)
    std::vector<double> testNorm = computeNormAT(A, 1);
    
    ASSERT_EQ(refNorm.size(), testNorm.size());
    EXPECT_TRUE(vectorsNearlyEqual(refNorm, testNorm));
}

TEST_F(SpMVTest, ComputeNormATMatchesReferenceMultiThread) {
    CSR A = createRandomCSR(500, 200, 10);
    
    // 参考实现 (单线程)
    std::vector<double> refNorm = refComputeNormAT(A);
    
    // 被测实现 (多线程)
    int threads = omp_get_max_threads();
    if (threads < 2) threads = 2;  // 确保至少2线程
    
    std::vector<double> testNorm = computeNormAT(A, threads);
    
    ASSERT_EQ(refNorm.size(), testNorm.size());
    EXPECT_TRUE(vectorsNearlyEqual(refNorm, testNorm));
}

// ============ CSRView 版本测试 ============

TEST_F(SpMVTest, ComputeNormATCSRView) {
    CSR A = createRandomCSR(100, 50, 5);
    
    // 参考实现
    std::vector<double> refNorm = refComputeNormAT(A);
    
    // 创建CSRView (模拟mmap情况)
    // 注意: spect::io::CSRView 使用 uint32_t 类型指针
    spect::io::CSRView view;
    view.rows = A.rows;
    view.cols = A.cols;
    view.nnz = A.val.size();
    view.row_ptr = A.row_ptr.data();
    view.col_idx = A.col_idx.data();
    view.val = A.val.data();
    
    // 被测实现
    std::vector<double> testNorm = computeNormAT(view, 1);
    
    EXPECT_TRUE(vectorsNearlyEqual(refNorm, testNorm));
}

TEST_F(SpMVTest, SparseOpsIOCSRViewMatchesReference) {
    CSR A = createSimpleCSR();
    
    spect::io::CSRView view;
    view.rows = A.rows;
    view.cols = A.cols;
    view.nnz = static_cast<uint64_t>(A.val.size());
    view.row_ptr = A.row_ptr.data();
    view.col_idx = A.col_idx.data();
    view.val = A.val.data();
    
    const auto& ref = getSparseOpsRefImpl();
    const auto& cpu = getSparseOpsCPUImpl();
    
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y_ref(A.rows, 0.0), y_cpu(A.rows, 0.0);
    ref.spmv(view, x.data(), y_ref.data(), 1);
    cpu.spmv(view, x.data(), y_cpu.data(), 4);
    EXPECT_TRUE(vectorsNearlyEqual(y_ref, y_cpu));
    
    std::vector<double> norm_ref(A.cols, 0.0), norm_cpu(A.cols, 0.0);
    ref.computeNormAT(view, norm_ref.data(), 1);
    cpu.computeNormAT(view, norm_cpu.data(), 4);
    EXPECT_TRUE(vectorsNearlyEqual(norm_ref, norm_cpu));
}

// ============ 多线程 Bitwise 一致性测试 ============

#ifdef SPECT_DETERMINISTIC

/**
 * @brief 测试多线程情况下结果的bitwise一致性
 * 
 * 根据 Refactor_Consensus_Plan.md 的确定性契约:
 * 同一平台/编译器/指令集下，不同线程数必须产生bitwise相同的结果
 * 
 * 回归测试模式下可通过 SPECT_DETERMINISTIC 强制单线程归约，
 * 从而保证不同线程数调用时结果 bitwise 一致。
 */
TEST_F(SpMVTest, MultiThreadBitwiseDeterminism) {
    // 使用较大矩阵确保触发多线程路径
    CSR A = createRandomCSR(1000, 500, 20);
    
    // 运行多次，不同线程数
    std::vector<double> result1 = computeNormAT(A, 1);
    std::vector<double> result2 = computeNormAT(A, 2);
    std::vector<double> result4 = computeNormAT(A, 4);
    std::vector<double> result8 = computeNormAT(A, 8);
    
    // 应该bitwise相同
    EXPECT_TRUE(vectorsBitwiseEqual(result1, result2)) 
        << "1 thread vs 2 threads should be bitwise identical";
    EXPECT_TRUE(vectorsBitwiseEqual(result1, result4))
        << "1 thread vs 4 threads should be bitwise identical";
    EXPECT_TRUE(vectorsBitwiseEqual(result1, result8))
        << "1 thread vs 8 threads should be bitwise identical";
}

/**
 * @brief 测试相同线程数多次运行的一致性
 */
TEST_F(SpMVTest, SameThreadCountConsistency) {
    CSR A = createRandomCSR(500, 200, 10);
    int threads = 4;
    
    std::vector<double> result1 = computeNormAT(A, threads);
    std::vector<double> result2 = computeNormAT(A, threads);
    std::vector<double> result3 = computeNormAT(A, threads);
    
    // 相同线程数应该始终产生相同结果
    EXPECT_TRUE(vectorsBitwiseEqual(result1, result2));
    EXPECT_TRUE(vectorsBitwiseEqual(result1, result3));
}

#endif  // SPECT_DETERMINISTIC

// ============ 边界条件测试 ============

TEST_F(SpMVTest, EmptyMatrix) {
    CSR A;
    A.rows = 0;
    A.cols = 10;
    A.row_ptr = {0};
    
    std::vector<double> norm = computeNormAT(A, 1);
    
    ASSERT_EQ(norm.size(), 10u);
    for (const auto& v : norm) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST_F(SpMVTest, SingleRowMatrix) {
    CSR A;
    A.rows = 1;
    A.cols = 5;
    A.row_ptr = {0, 3};
    A.col_idx = {0, 2, 4};
    A.val = {1.0f, 2.0f, 3.0f};
    
    std::vector<double> norm = computeNormAT(A, 1);
    
    EXPECT_DOUBLE_EQ(norm[0], 1.0);
    EXPECT_DOUBLE_EQ(norm[1], 0.0);
    EXPECT_DOUBLE_EQ(norm[2], 2.0);
    EXPECT_DOUBLE_EQ(norm[3], 0.0);
    EXPECT_DOUBLE_EQ(norm[4], 3.0);
}

TEST_F(SpMVTest, DiagonalMatrix) {
    // 对角矩阵
    CSR A;
    A.rows = 4;
    A.cols = 4;
    A.row_ptr = {0, 1, 2, 3, 4};
    A.col_idx = {0, 1, 2, 3};
    A.val = {1.0f, 2.0f, 3.0f, 4.0f};
    
    std::vector<double> x = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> y = refForwardProject(A, x);
    
    // 对角矩阵: y = x * diag
    EXPECT_DOUBLE_EQ(y[0], 1.0);
    EXPECT_DOUBLE_EQ(y[1], 2.0);
    EXPECT_DOUBLE_EQ(y[2], 3.0);
    EXPECT_DOUBLE_EQ(y[3], 4.0);
}

// ============ 大规模矩阵测试 ============

TEST_F(SpMVTest, LargeMatrixCorrectness) {
    // 模拟实际SPECT重建规模
    CSR A = createRandomCSR(10000, 5000, 50);
    
    std::vector<double> refNorm = refComputeNormAT(A);
    std::vector<double> testNorm = computeNormAT(A, omp_get_max_threads());
    
    EXPECT_TRUE(vectorsNearlyEqual(refNorm, testNorm, 1e-8, 1e-12));
}

// ============ 数值精度测试 ============

TEST_F(SpMVTest, NumericalPrecision) {
    // 测试小数值和大数值混合情况
    CSR A;
    A.rows = 2;
    A.cols = 3;
    A.row_ptr = {0, 2, 4};
    A.col_idx = {0, 1, 1, 2};
    A.val = {1e-6f, 1e6f, 1e-6f, 1e6f};  // 极端值
    
    std::vector<double> norm = computeNormAT(A, 1);
    
    EXPECT_NEAR(norm[0], 1e-6, 1e-12);
    EXPECT_NEAR(norm[1], 1e6 + 1e-6, 1.0);  // 精度损失可接受
    EXPECT_NEAR(norm[2], 1e6, 1.0);
}

}  // namespace test
}  // namespace spect
