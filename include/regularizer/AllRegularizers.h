// Regularizers.h
// 任务 T4.2 - 统一包含所有新正则化器的头文件
// 用于逐步替换旧的 Regularizers.h
#pragma once

#include "regularizer/RegularizerBase.h"
#include "regularizer/TVRegularizer.h"
#include "regularizer/TikhonovRegularizer.h"
#include "regularizer/MRPRegularizer.h"
#include "regularizer/RDPRegularizer.h"
#include "regularizer/LogCoshRegularizer.h"
#include "regularizer/NLMRegularizer.h"
#include "regularizer/BowsherRegularizer.h"
#include "regularizer/PLSRegularizer.h"

namespace spect {
namespace regularizer {

/**
 * @brief 注册所有内置正则化器到工厂
 * 
 * 在程序启动时调用此函数以确保所有正则化器可用。
 */
inline void registerBuiltinRegularizers() {
    auto& factory = RegularizerFactory::instance();
    
    // 基本类型
    factory.registerType("None", []{ return std::make_unique<NoneRegularizer>(); });
    
    // TV 变体
    factory.registerType("TVIsotropic", []{ return std::make_unique<TVIsotropicRegularizer>(); });
    factory.registerType("TVHuber", []{ return std::make_unique<TVHuberRegularizer>(); });
    
    // 二次/L2 类型
    factory.registerType("Tikhonov", []{ return std::make_unique<TikhonovRegularizer>(); });
    
    // 统计类型
    factory.registerType("MRP", []{ return std::make_unique<MRPRegularizer>(); });
    
    // 边缘保持类型
    factory.registerType("RDP", []{ return std::make_unique<RDPRegularizer>(); });
    factory.registerType("LogCosh", []{ return std::make_unique<LogCoshRegularizer>(); });
    
    // 非局部/解剖先验类型
    factory.registerType("NLM", []{ return std::make_unique<NLMRegularizer>(); });
    factory.registerType("Bowsher", []{ return std::make_unique<BowsherRegularizer>(); });
    factory.registerType("PLS", []{ return std::make_unique<PLSRegularizer>(); });
}

/**
 * @brief 从字符串创建正则化器（便捷函数）
 * 
 * @param name 正则化器名称
 * @param beta 正则化权重
 * @return 正则化器实例，如果名称无效则返回nullptr
 */
inline std::unique_ptr<IRegularizer> createRegularizer(const std::string& name, double beta = 1.0) {
    auto reg = RegularizerFactory::instance().create(name);
    if (reg) {
        // 由于IRegularizer没有setBeta，需要动态转换
        if (auto* base = dynamic_cast<RegularizerBase*>(reg.get())) {
            base->setBeta(beta);
        }
    }
    return reg;
}

} // namespace regularizer
} // namespace spect
