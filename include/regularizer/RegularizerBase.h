/**
 * @file RegularizerBase.h
 * @brief 正则化器基类和工厂
 * 
 * 【T4.1】正则化器继承体系设计
 * 
 * 设计原则：
 * - 新增正则化器只需：1)继承IRegularizer, 2)注册到工厂
 * - 不使用switch-case分发
 * - 兼容现有RegConf配置
 */

#pragma once

#include "interfaces/IRegularizer.h"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace spect {
namespace regularizer {

/**
 * @brief 正则化器基类
 * 
 * 提供通用功能和默认实现。
 * 派生类只需实现energy()和gradient()。
 */
class RegularizerBase : public IRegularizer {
public:
    explicit RegularizerBase(double beta = 0.0) : beta_(beta) {}
    
    /**
     * @brief 获取正则化权重beta
     */
    double beta() const { return beta_; }
    
    /**
     * @brief 设置正则化权重
     */
    void setBeta(double beta) { beta_ = beta; }
    
protected:
    double beta_;
    
    /**
     * @brief 计算6邻域偏移
     */
    static void getStencilOffsets(const Dims3D& dims, int offsets[6]) {
        offsets[0] = -1;                              // -x
        offsets[1] = 1;                               // +x
        offsets[2] = -dims.nx;                        // -y
        offsets[3] = dims.nx;                         // +y
        offsets[4] = -dims.nx * dims.ny;              // -z
        offsets[5] = dims.nx * dims.ny;               // +z
    }
    
    /**
     * @brief 检查边界
     */
    static bool isBoundary(int x, int y, int z, const Dims3D& dims) {
        return x == 0 || x == dims.nx - 1 ||
               y == 0 || y == dims.ny - 1 ||
               z == 0 || z == dims.nz - 1;
    }
};

// ============ 工厂 ============

/**
 * @brief 正则化器创建函数类型
 */
using RegularizerCreator = std::function<std::unique_ptr<IRegularizer>()>;

/**
 * @brief 正则化器工厂
 * 
 * 使用示例：
 * ```cpp
 * // 注册
 * RegularizerFactory::instance().registerType("TVIsotropic", 
 *     []{ return std::make_unique<TVIsotropicRegularizer>(); });
 * 
 * // 创建
 * auto reg = RegularizerFactory::instance().create("TVIsotropic");
 * ```
 */
class RegularizerFactory {
public:
    static RegularizerFactory& instance() {
        static RegularizerFactory factory;
        return factory;
    }
    
    /**
     * @brief 注册正则化器类型
     */
    void registerType(const std::string& name, RegularizerCreator creator) {
        creators_[name] = std::move(creator);
    }
    
    /**
     * @brief 创建正则化器实例
     */
    std::unique_ptr<IRegularizer> create(const std::string& name) const {
        auto it = creators_.find(name);
        if (it != creators_.end()) {
            return it->second();
        }
        return nullptr;
    }
    
    /**
     * @brief 检查类型是否已注册
     */
    bool hasType(const std::string& name) const {
        return creators_.find(name) != creators_.end();
    }
    
    /**
     * @brief 获取所有已注册类型名
     */
    std::vector<std::string> registeredTypes() const {
        std::vector<std::string> names;
        names.reserve(creators_.size());
        for (const auto& [name, _] : creators_) {
            names.push_back(name);
        }
        return names;
    }
    
private:
    RegularizerFactory() = default;
    std::map<std::string, RegularizerCreator> creators_;
};

/**
 * @brief 自动注册辅助类
 * 
 * 使用示例：
 * ```cpp
 * // 在.cpp文件中：
 * static RegularizerRegistrar<TVIsotropicRegularizer> 
 *     reg_tv("TVIsotropic");
 * ```
 */
template<typename T>
class RegularizerRegistrar {
public:
    explicit RegularizerRegistrar(const std::string& name) {
        RegularizerFactory::instance().registerType(name, 
            []{ return std::make_unique<T>(); });
    }
};

// ============ 空正则化器 ============

/**
 * @brief 空正则化器（无正则化）
 */
class NoneRegularizer : public RegularizerBase {
public:
    std::string name() const override { return "None"; }
    
    double energy(const double*, const Dims3D&, const uint8_t*) const override {
        return 0.0;
    }
    
    void gradient(const double*, double* grad, const Dims3D& dims,
                 const uint8_t*, int) const override {
        const size_t N = static_cast<size_t>(dims.nx) * dims.ny * dims.nz;
        std::memset(grad, 0, N * sizeof(double));
    }
};

} // namespace regularizer
} // namespace spect
