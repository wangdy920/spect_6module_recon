/**
 * @file test_regularizers.cpp
 * @brief 正则化器梯度有限差分测试和基本功能测试
 * 
 * 测试内容:
 * 1. 各类正则化器梯度有限差分验证
 * 2. 正则化器能量函数计算
 * 3. 边界条件处理
 * 4. 掩码支持
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>

// 包含被测试的头文件 (使用新的模块化路径)
#include "regularizer/RegConfig.h"
#include "regularizer/RegularizerCompat.h"

// 导入命名空间以保持兼容
using namespace spect::regularizer;

namespace spect {
namespace test {

/**
 * @brief 正则化器测试夹具类
 */
class RegularizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_.seed(42);
        
        // 默认测试图像尺寸 (小尺寸便于调试)
        Nx_ = 8;
        Ny_ = 8;
        Nz_ = 8;
        N_ = Nx_ * Ny_ * Nz_;
    }
    
    std::mt19937 gen_;
    int Nx_, Ny_, Nz_, N_;
    
    /**
     * @brief 创建平滑测试图像
     */
    std::vector<double> createSmoothImage() {
        std::vector<double> f(N_);
        for (int z = 0; z < Nz_; ++z) {
            for (int y = 0; y < Ny_; ++y) {
                for (int x = 0; x < Nx_; ++x) {
                    int idx = x + y * Nx_ + z * Nx_ * Ny_;
                    // 平滑的高斯分布
                    double cx = (x - Nx_ / 2.0) / (Nx_ / 4.0);
                    double cy = (y - Ny_ / 2.0) / (Ny_ / 4.0);
                    double cz = (z - Nz_ / 2.0) / (Nz_ / 4.0);
                    f[idx] = std::exp(-(cx*cx + cy*cy + cz*cz)) + 0.1;
                }
            }
        }
        return f;
    }
    
    /**
     * @brief 创建随机测试图像
     */
    std::vector<double> createRandomImage() {
        std::vector<double> f(N_);
        std::uniform_real_distribution<double> dist(0.1, 1.0);
        for (int i = 0; i < N_; ++i) {
            f[i] = dist(gen_);
        }
        return f;
    }
    
    /**
     * @brief 创建全1掩码
     */
    std::vector<uint8_t> createFullMask() {
        return std::vector<uint8_t>(N_, 1);
    }
    
    /**
     * @brief 创建球形掩码
     */
    std::vector<uint8_t> createSphereMask(double radius) {
        std::vector<uint8_t> mask(N_, 0);
        double cx = Nx_ / 2.0;
        double cy = Ny_ / 2.0;
        double cz = Nz_ / 2.0;
        
        for (int z = 0; z < Nz_; ++z) {
            for (int y = 0; y < Ny_; ++y) {
                for (int x = 0; x < Nx_; ++x) {
                    double dx = x - cx;
                    double dy = y - cy;
                    double dz = z - cz;
                    if (dx*dx + dy*dy + dz*dz < radius * radius) {
                        mask[x + y * Nx_ + z * Nx_ * Ny_] = 1;
                    }
                }
            }
        }
        return mask;
    }
    
    /**
     * @brief 有限差分梯度验证
     * 
     * 使用中心差分: (R(f + eps*e_i) - R(f - eps*e_i)) / (2*eps)
     * 比较解析梯度与数值梯度
     */
    bool verifyGradientFD(RegType type, double beta = 1.0, 
                         double rtol = 1e-4, double atol = 1e-6) {
        RegConf conf;
        conf.type = type;
        conf.beta = beta;
        conf.tv_eps = 1e-6;
        conf.mrp_eps = 1e-8;
        conf.delta = 0.01;
        conf.gamma = 0.1;
        conf.nlm_h = 0.1;
        conf.nlm_search = 1;
        conf.nlm_patch = 1;
        
        auto f = createSmoothImage();
        auto mask = createFullMask();
        
        // 计算解析梯度
        std::vector<double> gradR(N_, 0.0);
        compute_reg_gradient(f, Nx_, Ny_, Nz_, conf, &mask, gradR, 1);
        
        // 数值梯度验证 (只验证部分点)
        const double eps = 1e-5;
        int numChecks = std::min(50, N_);
        std::uniform_int_distribution<int> idxDist(0, N_ - 1);
        
        double maxRelError = 0.0;
        for (int c = 0; c < numChecks; ++c) {
            int i = idxDist(gen_);
            if (mask[i] == 0) continue;
            
            // f + eps
            f[i] += eps;
            double Rplus = compute_reg_energy(f, Nx_, Ny_, Nz_, conf, &mask);
            
            // f - 2*eps
            f[i] -= 2 * eps;
            double Rminus = compute_reg_energy(f, Nx_, Ny_, Nz_, conf, &mask);
            
            // 恢复
            f[i] += eps;
            
            // 数值梯度 (能量函数本身不含beta，梯度函数也不含beta)
            double numGrad = (Rplus - Rminus) / (2 * eps);
            double anaGrad = gradR[i];
            
            // 相对误差
            double absError = std::abs(numGrad - anaGrad);
            double maxVal = std::max({std::abs(numGrad), std::abs(anaGrad), 1e-10});
            double relError = absError / maxVal;
            
            maxRelError = std::max(maxRelError, relError);
            
            if (absError > atol + rtol * maxVal) {
                std::cerr << "Gradient mismatch at index " << i 
                         << ": numerical=" << numGrad 
                         << ", analytical=" << anaGrad 
                         << ", relError=" << relError << std::endl;
                return false;
            }
        }
        
        return true;
    }
};

// ============ 字符串转换测试 ============

TEST_F(RegularizerTest, StringToRegType) {
    EXPECT_EQ(stringToRegType("TVIsotropic"), RegType::TVIsotropic);
    EXPECT_EQ(stringToRegType("TVHuber"), RegType::TVHuber);
    EXPECT_EQ(stringToRegType("Tikhonov"), RegType::Tikhonov);
    EXPECT_EQ(stringToRegType("MRP"), RegType::MRP);
    EXPECT_EQ(stringToRegType("RDP"), RegType::RDP);
    EXPECT_EQ(stringToRegType("LogCosh"), RegType::LogCosh);
    EXPECT_EQ(stringToRegType("None"), RegType::None);
    EXPECT_EQ(stringToRegType("invalid"), RegType::Unknown);
}

TEST_F(RegularizerTest, RegTypeToString) {
    EXPECT_EQ(regTypeToString(RegType::TVIsotropic), "TVIsotropic");
    EXPECT_EQ(regTypeToString(RegType::TVHuber), "TVHuber");
    EXPECT_EQ(regTypeToString(RegType::None), "None");
}

// ============ 梯度有限差分测试 ============
// 注意: 这些测试验证compute_reg_gradient与compute_reg_energy的一致性
// 某些正则化器可能存在能量-梯度不一致问题(如因子2差异)，这些在Phase 2修复

TEST_F(RegularizerTest, GradientFD_TVIsotropic) {
    EXPECT_TRUE(verifyGradientFD(RegType::TVIsotropic, 1.0, 1e-3, 1e-5));
}

TEST_F(RegularizerTest, GradientFD_TVHuber) {
    EXPECT_TRUE(verifyGradientFD(RegType::TVHuber, 1.0, 1e-3, 1e-5));
}

// Fixed: Tikhonov梯度已修复，添加因子2
TEST_F(RegularizerTest, GradientFD_Tikhonov) {
    EXPECT_TRUE(verifyGradientFD(RegType::Tikhonov, 1.0, 1e-3, 1e-5));
}

TEST_F(RegularizerTest, GradientFD_MRP) {
    EXPECT_TRUE(verifyGradientFD(RegType::MRP, 1.0, 1e-3, 1e-5));
}

// Fixed: RDP能量函数已添加gamma项，与梯度一致
TEST_F(RegularizerTest, GradientFD_RDP) {
    EXPECT_TRUE(verifyGradientFD(RegType::RDP, 1.0, 1e-3, 1e-5));
}

// Fixed: LogCosh能量和梯度函数已完善
TEST_F(RegularizerTest, GradientFD_LogCosh) {
    EXPECT_TRUE(verifyGradientFD(RegType::LogCosh, 1.0, 1e-3, 1e-5));
}

// ============ 基本功能测试 ============

TEST_F(RegularizerTest, NoneRegularizerZeroGradient) {
    RegConf conf;
    conf.type = RegType::None;
    conf.beta = 1.0;
    
    auto f = createSmoothImage();
    auto mask = createFullMask();
    
    std::vector<double> gradR(N_, 0.0);
    compute_reg_gradient(f, Nx_, Ny_, Nz_, conf, &mask, gradR, 1);
    
    // None 正则化器应该产生零梯度
    for (int i = 0; i < N_; ++i) {
        EXPECT_DOUBLE_EQ(gradR[i], 0.0);
    }
}

TEST_F(RegularizerTest, ZeroBetaZeroEnergy) {
    // compute_reg_gradient 不检查 beta，但 compute_reg_energy 在 beta=0 时返回 0
    RegConf conf;
    conf.type = RegType::TVIsotropic;
    conf.beta = 0.0;  // beta = 0
    
    auto f = createRandomImage();
    auto mask = createFullMask();
    
    double energy = compute_reg_energy(f, Nx_, Ny_, Nz_, conf, &mask);
    
    // beta=0 应该返回零能量
    EXPECT_DOUBLE_EQ(energy, 0.0);
}

TEST_F(RegularizerTest, UniformImageZeroTVGradient) {
    RegConf conf;
    conf.type = RegType::TVIsotropic;
    conf.beta = 1.0;
    conf.tv_eps = 1e-10;
    
    // 均匀图像
    std::vector<double> f(N_, 1.0);
    auto mask = createFullMask();
    
    std::vector<double> gradR(N_, 0.0);
    compute_reg_gradient(f, Nx_, Ny_, Nz_, conf, &mask, gradR, 1);
    
    // 均匀图像的TV梯度应该接近零
    for (int i = 0; i < N_; ++i) {
        EXPECT_NEAR(gradR[i], 0.0, 1e-6);
    }
}

// ============ 掩码测试 ============

TEST_F(RegularizerTest, MaskOutsideZeroGradient) {
    RegConf conf;
    conf.type = RegType::TVIsotropic;
    conf.beta = 1.0;
    conf.skip_outside_mask = true;
    
    auto f = createRandomImage();
    auto mask = createSphereMask(Nx_ / 3.0);
    
    std::vector<double> gradR(N_, 0.0);
    compute_reg_gradient(f, Nx_, Ny_, Nz_, conf, &mask, gradR, 1);
    
    // 掩码外的梯度应该为零
    for (int i = 0; i < N_; ++i) {
        if (mask[i] == 0) {
            EXPECT_DOUBLE_EQ(gradR[i], 0.0);
        }
    }
}

// ============ 能量函数测试 ============

TEST_F(RegularizerTest, EnergyNonNegative) {
    RegConf conf;
    conf.type = RegType::TVIsotropic;
    conf.beta = 1.0;
    
    auto f = createRandomImage();
    auto mask = createFullMask();
    
    double energy = compute_reg_energy(f, Nx_, Ny_, Nz_, conf, &mask);
    
    EXPECT_GE(energy, 0.0);
}

TEST_F(RegularizerTest, UniformImageZeroEnergy) {
    RegConf conf;
    conf.type = RegType::TVIsotropic;
    conf.beta = 1.0;
    conf.tv_eps = 0.0;  // 精确TV
    
    std::vector<double> f(N_, 1.0);
    auto mask = createFullMask();
    
    double energy = compute_reg_energy(f, Nx_, Ny_, Nz_, conf, &mask);
    
    // 均匀图像的TV能量应该为零
    EXPECT_NEAR(energy, 0.0, 1e-10);
}

// ============ 多线程测试 ============

TEST_F(RegularizerTest, MultiThreadConsistency) {
    RegConf conf;
    conf.type = RegType::TVIsotropic;
    conf.beta = 1.0;
    
    auto f = createRandomImage();
    auto mask = createFullMask();
    
    std::vector<double> gradR1(N_, 0.0);
    std::vector<double> gradR4(N_, 0.0);
    
    compute_reg_gradient(f, Nx_, Ny_, Nz_, conf, &mask, gradR1, 1);
    compute_reg_gradient(f, Nx_, Ny_, Nz_, conf, &mask, gradR4, 4);
    
    // 单线程和多线程应该产生相同结果
    for (int i = 0; i < N_; ++i) {
        EXPECT_NEAR(gradR1[i], gradR4[i], 1e-10);
    }
}

// ============ 边界条件测试 ============

TEST_F(RegularizerTest, SmallImageHandling) {
    // 最小尺寸: 2x2x2
    int smallNx = 2, smallNy = 2, smallNz = 2;
    int smallN = smallNx * smallNy * smallNz;
    
    RegConf conf;
    conf.type = RegType::TVIsotropic;
    conf.beta = 1.0;
    
    std::vector<double> f(smallN, 1.0);
    std::vector<uint8_t> mask(smallN, 1);
    std::vector<double> gradR(smallN, 0.0);
    
    // 应该不会崩溃
    EXPECT_NO_THROW({
        compute_reg_gradient(f, smallNx, smallNy, smallNz, conf, &mask, gradR, 1);
    });
}

}  // namespace test
}  // namespace spect
