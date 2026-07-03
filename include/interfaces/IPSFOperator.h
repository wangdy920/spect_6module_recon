/**
 * @file IPSFOperator.h
 * @brief PSF/卷积算子接口
 * 
 * 支持空间不变/空间变PSF，空域/频域实现。
 */

#pragma once

#include "../core/Types.h"

namespace spect {

/**
 * @brief PSF类型
 */
enum class PSFType {
    None,           ///< 无PSF (恒等算子)
    Invariant,      ///< 空间不变PSF
    DepthDependent, ///< 深度依赖PSF
    FullySpatialVariant ///< 完全空间变PSF
};

/**
 * @brief PSF算子接口
 */
class IPSFOperator {
public:
    virtual ~IPSFOperator() = default;
    
    /**
     * @brief 获取PSF类型
     */
    virtual PSFType type() const = 0;
    
    /**
     * @brief 应用PSF (卷积)
     * 
     * @param input 输入图像
     * @param output 输出图像
     * @param dims 图像维度
     * @param threads 线程数
     */
    virtual void apply(const double* input, 
                      double* output,
                      const Dims3D& dims,
                      int threads = 1) const = 0;
    
    /**
     * @brief 应用PSF转置 (用于反投影)
     * 
     * 对于对称PSF，transpose与apply相同。
     */
    virtual void applyTranspose(const double* input, 
                               double* output,
                               const Dims3D& dims,
                               int threads = 1) const = 0;
    
    /**
     * @brief 是否为恒等算子
     */
    virtual bool isIdentity() const { return type() == PSFType::None; }
    
    /**
     * @brief 是否对称
     */
    virtual bool isSymmetric() const { return true; }
};

/**
 * @brief 无操作PSF (恒等算子)
 */
class NoPSF : public IPSFOperator {
public:
    PSFType type() const override { return PSFType::None; }
    
    void apply(const double* input, 
              double* output,
              const Dims3D& dims,
              int threads = 1) const override {
        // 直接复制
        size_t n = dims.total();
        for (size_t i = 0; i < n; ++i) {
            output[i] = input[i];
        }
    }
    
    void applyTranspose(const double* input, 
                       double* output,
                       const Dims3D& dims,
                       int threads = 1) const override {
        apply(input, output, dims, threads);
    }
};

} // namespace spect
