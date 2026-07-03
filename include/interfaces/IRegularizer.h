/**
 * @file IRegularizer.h
 * @brief 正则化器统一接口
 * 
 * 设计约束:
 * - 不允许用switch-case扩展正则，必须新增派生类+工厂注册
 * - 先落地grad()，prox()作为扩展位(ADMM/PDHG)
 */

#pragma once

#include "../core/Types.h"
#include <vector>
#include <string>
#include <memory>

namespace spect {

/**
 * @brief 正则化器接口
 * 
 * 提供正则项的能量计算、梯度计算和近端算子。
 */
class IRegularizer {
public:
    virtual ~IRegularizer() = default;
    
    /**
     * @brief 获取正则化器名称
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief 计算正则化能量 R(f)
     * 
     * @param f 输入图像
     * @param dims 图像维度
     * @param mask 可选掩码 (nullptr表示全图)
     * @return 正则化能量值
     */
    virtual double energy(const double* f, 
                         const Dims3D& dims,
                         const uint8_t* mask = nullptr) const = 0;
    
    /**
     * @brief 计算正则化梯度 ∇R(f)
     * 
     * @param f 输入图像
     * @param grad 输出梯度 (与f同尺寸)
     * @param dims 图像维度
     * @param mask 可选掩码
     * @param threads 线程数
     */
    virtual void gradient(const double* f, 
                         double* grad,
                         const Dims3D& dims,
                         const uint8_t* mask = nullptr,
                         int threads = 1) const = 0;
    
    /**
     * @brief 近端算子 prox_{λR}(f) = argmin_x { 0.5||x-f||² + λR(x) }
     * 
     * 可选实现，用于ADMM/PDHG等算法。
     * 
     * @param f 输入图像
     * @param out 输出图像
     * @param lambda 正则化权重
     * @param dims 图像维度
     * @param mask 可选掩码
     * @param threads 线程数
     * @return Status::Ok 或 Status::NotImplemented
     */
    virtual Status prox(const double* f,
                       double* out,
                       double lambda,
                       const Dims3D& dims,
                       const uint8_t* mask = nullptr,
                       int threads = 1) const {
        return Status::NotImplemented;
    }
    
    /**
     * @brief 获取正则化权重 β
     */
    virtual double beta() const = 0;
    
    /**
     * @brief 设置正则化权重 β
     */
    virtual void setBeta(double b) = 0;
    
    /**
     * @brief 是否支持prox操作
     */
    virtual bool hasProx() const { return false; }
};

/**
 * @brief 正则化器工厂
 */
class RegularizerFactory {
public:
    /**
     * @brief 创建正则化器实例
     * 
     * @param type 正则化器类型名称 (如 "TVIsotropic", "Tikhonov")
     * @param beta 正则化权重
     * @param params 额外参数 (如 delta, eps 等)
     * @return 正则化器实例
     */
    static std::unique_ptr<IRegularizer> create(
        const std::string& type,
        double beta = 1.0,
        const std::vector<double>& params = {});
};

} // namespace spect
