/**
 * @file RegularizerCompat.h
 * @brief 正则化器兼容层
 * 
 * 【T4.3】提供从旧RegConf API到新工厂API的桥接。
 * 允许渐进式迁移，最终将被完全移除。
 */

#pragma once

#include "regularizer/AllRegularizers.h"
#include "regularizer/RegConfig.h"  // 新模块化API

namespace spect {
namespace regularizer {

// ================= 辅助函数（从 original Regularizers.h 迁移）=================

/**
 * @brief 体素线性索引
 */
inline size_t IDX(int x, int y, int z, int Nx, int Ny) {
    return static_cast<size_t>(z) * static_cast<size_t>(Ny) * static_cast<size_t>(Nx) 
         + static_cast<size_t>(y) * static_cast<size_t>(Nx) 
         + static_cast<size_t>(x);
}

/**
 * @brief 边界检查
 */
inline bool inb(int x, int y, int z, int Nx, int Ny, int Nz) {
    return (x >= 0 && x < Nx && y >= 0 && y < Ny && z >= 0 && z < Nz);
}

/**
 * @brief 6-邻域偏移表（顺序：-x,+x,-y,+y,-z,+z）
 */
inline void stencil_offsets(int /*Nx*/, int /*Ny*/, int /*Nz*/, 
                            int dx[6], int dy[6], int dz[6]) {
    dx[0] = -1; dy[0] = 0;  dz[0] = 0;
    dx[1] = 1;  dy[1] = 0;  dz[1] = 0;
    dx[2] = 0;  dy[2] = -1; dz[2] = 0;
    dx[3] = 0;  dy[3] = 1;  dz[3] = 0;
    dx[4] = 0;  dy[4] = 0;  dz[4] = -1;
    dx[5] = 0;  dy[5] = 0;  dz[5] = 1;
}

// ================= 接口函数声明 =================

/**
 * @brief 计算正则化梯度
 */
void compute_reg_gradient(const std::vector<double> &f, int Nx, int Ny, int Nz,
                          const RegConf &reg, const std::vector<uint8_t> *mask,
                          std::vector<double> &gradR, int threads);

/**
 * @brief OSL 更新（原地修改）
 */
void osl_update_inplace(std::vector<double> &f,
                        const std::vector<double> &q,
                        const std::vector<double> &d,
                        const std::vector<double> &gradR,
                        double beta, const std::vector<uint8_t> *mask,
                        bool nonneg, int threads);

/**
 * @brief 计算正则化能量
 */
double compute_reg_energy(const std::vector<double> &f, int Nx, int Ny, int Nz,
                          const RegConf &reg,
                          const std::vector<uint8_t> *mask);

// ================= 工厂函数 =================

/**
 * @brief 从RegType创建新正则化器
 * 
 * 桥接函数：将旧枚举映射到新工厂。
 */
inline std::unique_ptr<IRegularizer> createFromRegType(RegType type, double beta = 1.0) {
    // 确保工厂已初始化
    static bool initialized = false;
    if (!initialized) {
        registerBuiltinRegularizers();
        initialized = true;
    }
    
    std::string name;
    switch (type) {
        case RegType::None:        name = "None"; break;
        case RegType::TVIsotropic: name = "TVIsotropic"; break;
        case RegType::TVHuber:     name = "TVHuber"; break;
        case RegType::Tikhonov:    name = "Tikhonov"; break;
        case RegType::MRP:         name = "MRP"; break;
        case RegType::RDP:         name = "RDP"; break;
        case RegType::LogCosh:     name = "LogCosh"; break;
        case RegType::NLM:         name = "NLM"; break;
        case RegType::Bowsher:     name = "Bowsher"; break;
        case RegType::PLS:         name = "PLS"; break;
        default:                   return nullptr;
    }
    return createRegularizer(name, beta);
}

/**
 * @brief 从RegConf创建并配置正则化器
 * 
 * 桥接函数：从旧配置结构创建新正则化器，并设置参数。
 */
inline std::unique_ptr<IRegularizer> createFromRegConf(const RegConf& conf) {
    auto reg = createFromRegType(conf.type, conf.beta);
    if (!reg) return nullptr;
    
    // 配置特定参数
    if (auto* tv = dynamic_cast<TVIsotropicRegularizer*>(reg.get())) {
        // TVIsotropic 使用 tv_eps
        *tv = TVIsotropicRegularizer(conf.tv_eps);
        tv->setBeta(conf.beta);
    }
    else if (auto* huber = dynamic_cast<TVHuberRegularizer*>(reg.get())) {
        *huber = TVHuberRegularizer(conf.delta);
        huber->setBeta(conf.beta);
    }
    else if (auto* mrp = dynamic_cast<MRPRegularizer*>(reg.get())) {
        *mrp = MRPRegularizer(conf.mrp_eps);
        mrp->setBeta(conf.beta);
    }
    else if (auto* rdp = dynamic_cast<RDPRegularizer*>(reg.get())) {
        *rdp = RDPRegularizer(conf.delta, conf.gamma);
        rdp->setBeta(conf.beta);
    }
    else if (auto* lc = dynamic_cast<LogCoshRegularizer*>(reg.get())) {
        *lc = LogCoshRegularizer(conf.delta);
        lc->setBeta(conf.beta);
    }
    else if (auto* nlm = dynamic_cast<NLMRegularizer*>(reg.get())) {
        *nlm = NLMRegularizer(conf.nlm_h, conf.nlm_search, conf.nlm_patch);
        nlm->setBeta(conf.beta);
    }
    else if (auto* bow = dynamic_cast<BowsherRegularizer*>(reg.get())) {
        *bow = BowsherRegularizer(conf.bowsher_k);
        bow->setBeta(conf.beta);
        if (conf.anatomical_image) {
            bow->setAnatomicalImage(conf.anatomical_image->data(), 
                                    conf.anatomical_image->size());
        }
    }
    else if (auto* pls = dynamic_cast<PLSRegularizer*>(reg.get())) {
        *pls = PLSRegularizer(conf.pls_eta);
        pls->setBeta(conf.beta);
        if (conf.anatomical_image) {
            pls->setAnatomicalImage(conf.anatomical_image->data(),
                                    conf.anatomical_image->size());
        }
    }
    
    return reg;
}

/**
 * @brief 使用新API计算梯度（兼容旧接口）
 * 
 * 直接替换 compute_reg_gradient()
 */
inline void computeGradientNew(const std::vector<double>& f,
                                int Nx, int Ny, int Nz,
                                const RegConf& conf,
                                const std::vector<uint8_t>* mask,
                                std::vector<double>& gradR,
                                int threads) {
    // 确保工厂已初始化
    static bool initialized = false;
    if (!initialized) {
        registerBuiltinRegularizers();
        initialized = true;
    }
    
    auto reg = createFromRegConf(conf);
    if (!reg) {
        gradR.assign(Nx * Ny * Nz, 0.0);
        return;
    }
    
    Dims3D dims{Nx, Ny, Nz};
    gradR.resize(dims.total());
    
    const uint8_t* maskPtr = mask ? mask->data() : nullptr;
    reg->gradient(f.data(), gradR.data(), dims, maskPtr, threads);
}

/**
 * @brief 使用新API计算能量（兼容旧接口）
 * 
 * 直接替换 compute_reg_energy()
 */
inline double computeEnergyNew(const std::vector<double>& f,
                                int Nx, int Ny, int Nz,
                                const RegConf& conf,
                                const std::vector<uint8_t>* mask) {
    // 确保工厂已初始化
    static bool initialized = false;
    if (!initialized) {
        registerBuiltinRegularizers();
        initialized = true;
    }
    
    auto reg = createFromRegConf(conf);
    if (!reg) return 0.0;
    
    Dims3D dims{Nx, Ny, Nz};
    const uint8_t* maskPtr = mask ? mask->data() : nullptr;
    return reg->energy(f.data(), dims, maskPtr);
}

} // namespace regularizer
} // namespace spect
