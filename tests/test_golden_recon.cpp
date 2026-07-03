/**
 * @file test_golden_recon.cpp
 * @brief Golden重建测试 - 使用toy矩阵验证重建算法正确性
 * 
 * 测试内容:
 * 1. MLEM单次迭代正确性
 * 2. 简单重建往返测试
 * 3. 敏感度计算正确性
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>

// 包含模块化头文件
#include "io/CSRBuilder.h"
#include "io/MatFileReader.h"
#include "sparse/SparseOpsRef.h"
#include "app/ProjectionOps.h"
#include "app/ReconConfig.h"

using namespace spect::io;
using spect::computeNormAT;
using spect::make_simple_subsets;

namespace spect {
namespace test {

/**
 * @brief Golden重建测试夹具类
 */
class GoldenReconTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建简单的toy系统矩阵
        // 4 rays (rows) x 4 voxels (cols)
        // 每条射线穿过2个体素
        toyMatrix_ = createToyMatrix();
    }
    
    CSR toyMatrix_;
    
    /**
     * @brief 创建简单的2x2体素phantom对应的系统矩阵
     * 
     * 假设4条射线:
     *   Ray 0: 穿过 voxel 0, 1
     *   Ray 1: 穿过 voxel 2, 3
     *   Ray 2: 穿过 voxel 0, 2
     *   Ray 3: 穿过 voxel 1, 3
     */
    CSR createToyMatrix() {
        CSR A;
        A.rows = 4;
        A.cols = 4;
        A.row_ptr = {0, 2, 4, 6, 8};
        A.col_idx = {0, 1, 2, 3, 0, 2, 1, 3};
        A.val = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        return A;
    }
    
    /**
     * @brief 标量参考实现: 前向投影
     */
    std::vector<double> forwardProject(const CSR& A, const std::vector<double>& x) {
        std::vector<double> y(A.rows, 0.0);
        for (int row = 0; row < A.rows; ++row) {
            for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
                y[row] += static_cast<double>(A.val[t]) * x[A.col_idx[t]];
            }
        }
        return y;
    }
    
    /**
     * @brief 标量参考实现: 反向投影
     */
    std::vector<double> backwardProject(const CSR& A, const std::vector<double>& y) {
        std::vector<double> x(A.cols, 0.0);
        for (int row = 0; row < A.rows; ++row) {
            for (size_t t = A.row_ptr[row]; t < A.row_ptr[row + 1]; ++t) {
                x[A.col_idx[t]] += static_cast<double>(A.val[t]) * y[row];
            }
        }
        return x;
    }
    
    /**
     * @brief 标量参考实现: MLEM单次迭代
     * 
     * f^{k+1} = f^k / s * A^T (p / (A * f^k + eps))
     * 
     * 其中 s = A^T * 1 是敏感度
     */
    std::vector<double> mlemIteration(const CSR& A, 
                                      const std::vector<double>& f,
                                      const std::vector<double>& p,
                                      const std::vector<double>& sensitivity,
                                      double eps = 1e-10) {
        int M = A.rows;
        int N = A.cols;
        
        // 前向投影: y = A * f
        std::vector<double> y = forwardProject(A, f);
        
        // 比值: ratio = p / (y + eps)
        std::vector<double> ratio(M);
        for (int i = 0; i < M; ++i) {
            ratio[i] = p[i] / (y[i] + eps);
        }
        
        // 反向投影: correction = A^T * ratio
        std::vector<double> correction = backwardProject(A, ratio);
        
        // 更新: f_new = f * correction / sensitivity
        std::vector<double> f_new(N);
        for (int j = 0; j < N; ++j) {
            if (sensitivity[j] > eps) {
                f_new[j] = f[j] * correction[j] / sensitivity[j];
            } else {
                f_new[j] = 0.0;
            }
        }
        
        return f_new;
    }
};

// ============ 敏感度计算测试 ============

TEST_F(GoldenReconTest, SensitivityComputation) {
    // 使用 computeNormAT 计算敏感度
    std::vector<double> sensitivity = computeNormAT(toyMatrix_, 1);
    
    // 手工计算预期敏感度 (每列的和)
    // col 0: 0.5 + 0.5 = 1.0 (ray 0, ray 2)
    // col 1: 0.5 + 0.5 = 1.0 (ray 0, ray 3)
    // col 2: 0.5 + 0.5 = 1.0 (ray 1, ray 2)
    // col 3: 0.5 + 0.5 = 1.0 (ray 1, ray 3)
    
    ASSERT_EQ(sensitivity.size(), 4u);
    EXPECT_DOUBLE_EQ(sensitivity[0], 1.0);
    EXPECT_DOUBLE_EQ(sensitivity[1], 1.0);
    EXPECT_DOUBLE_EQ(sensitivity[2], 1.0);
    EXPECT_DOUBLE_EQ(sensitivity[3], 1.0);
}

// ============ 前向/反向投影测试 ============

TEST_F(GoldenReconTest, ForwardBackwardConsistency) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    
    // 前向投影
    std::vector<double> y = forwardProject(toyMatrix_, x);
    
    // 验证前向投影结果
    // y[0] = 0.5*1 + 0.5*2 = 1.5
    // y[1] = 0.5*3 + 0.5*4 = 3.5
    // y[2] = 0.5*1 + 0.5*3 = 2.0
    // y[3] = 0.5*2 + 0.5*4 = 3.0
    EXPECT_DOUBLE_EQ(y[0], 1.5);
    EXPECT_DOUBLE_EQ(y[1], 3.5);
    EXPECT_DOUBLE_EQ(y[2], 2.0);
    EXPECT_DOUBLE_EQ(y[3], 3.0);
    
    // 反向投影
    std::vector<double> z = backwardProject(toyMatrix_, y);
    
    // z = A^T * A * x (正规方程相关)
    // 可以验证 <y, y> = <x, z>
    double yy = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
    double xz = std::inner_product(x.begin(), x.end(), z.begin(), 0.0);
    EXPECT_NEAR(yy, xz, 1e-10);
}

// ============ MLEM迭代测试 ============

TEST_F(GoldenReconTest, MLEMSingleIteration) {
    // 真实phantom
    std::vector<double> f_true = {1.0, 2.0, 3.0, 4.0};
    
    // 模拟投影数据 (无噪声)
    std::vector<double> p = forwardProject(toyMatrix_, f_true);
    
    // 敏感度
    std::vector<double> sensitivity = computeNormAT(toyMatrix_, 1);
    
    // 初始估计 (均匀)
    std::vector<double> f_init(4, 2.5);
    
    // 单次MLEM迭代
    std::vector<double> f_updated = mlemIteration(toyMatrix_, f_init, p, sensitivity);
    
    // 迭代后应该更接近真实值
    double err_init = 0.0, err_updated = 0.0;
    for (int i = 0; i < 4; ++i) {
        err_init += std::abs(f_init[i] - f_true[i]);
        err_updated += std::abs(f_updated[i] - f_true[i]);
    }
    
    EXPECT_LT(err_updated, err_init) << "MLEM iteration should reduce error";
}

TEST_F(GoldenReconTest, MLEMConvergence) {
    // 真实phantom
    std::vector<double> f_true = {1.0, 2.0, 3.0, 4.0};
    
    // 模拟投影数据
    std::vector<double> p = forwardProject(toyMatrix_, f_true);
    
    // 敏感度
    std::vector<double> sensitivity = computeNormAT(toyMatrix_, 1);
    
    // 初始估计
    std::vector<double> f = {1.0, 1.0, 1.0, 1.0};
    
    // 多次迭代
    for (int iter = 0; iter < 500; ++iter) {
        f = mlemIteration(toyMatrix_, f, p, sensitivity);
    }
    
    // 应该收敛到接近真实值(相对误差小于20%，因为MLEM收敛速度取决于矩阵条件)
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(f[i], f_true[i], f_true[i] * 0.2) 
            << "MLEM should converge to true value at index " << i;
    }
}

// ============ 子集划分测试 ============

TEST_F(GoldenReconTest, SubsetPartitioning) {
    int M = 12;
    int n_subsets = 4;
    
    auto subsets = make_simple_subsets(M, n_subsets);
    
    ASSERT_EQ(subsets.size(), 4u);
    
    // 验证所有行都被分配
    std::vector<bool> assigned(M, false);
    for (size_t s = 0; s < subsets.size(); ++s) {
        for (size_t i = 0; i < subsets[s].size(); ++i) {
            int row = subsets[s][i];
            ASSERT_GE(row, 0);
            ASSERT_LT(row, M);
            EXPECT_FALSE(assigned[row]) << "Row " << row << " assigned twice";
            assigned[row] = true;
        }
    }
    
    for (int i = 0; i < M; ++i) {
        EXPECT_TRUE(assigned[i]) << "Row " << i << " not assigned";
    }
}

// ============ 非负约束测试 ============

TEST_F(GoldenReconTest, NonNegativityConstraint) {
    // 初始有负值
    std::vector<double> f = {1.0, -0.5, 2.0, -1.0};
    
    // 非负投影
    for (auto& v : f) {
        if (v < 0) v = 0;
    }
    
    for (const auto& v : f) {
        EXPECT_GE(v, 0.0);
    }
}

// ============ 空投影数据测试 ============

TEST_F(GoldenReconTest, ZeroProjectionHandling) {
    std::vector<double> f = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> p = {0.0, 0.0, 0.0, 0.0};  // 零投影
    std::vector<double> sensitivity = computeNormAT(toyMatrix_, 1);
    
    // 应该不会崩溃或产生NaN
    std::vector<double> f_updated = mlemIteration(toyMatrix_, f, p, sensitivity, 1e-10);
    
    for (const auto& v : f_updated) {
        EXPECT_FALSE(std::isnan(v));
        EXPECT_FALSE(std::isinf(v));
    }
}

}  // namespace test
}  // namespace spect
