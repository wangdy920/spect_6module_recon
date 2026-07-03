/**
 * @file RegularizerImpl.h
 * @brief 正则化器通用配置和类型定义
 * 
 * 从 original_source_backup/Regularizers.h 迁移
 */

#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace spect {
namespace regularizer {

// ================= 正则化器类型 =================

enum class RegType {
    None,
    TVIsotropic,   // TV 各向同性
    TVHuber,       // Huber TV
    Tikhonov,      // 二次正则
    MRP,           // Median Root Prior
    NLM,           // Non-Local Means
    Bowsher,       // Bowsher 解剖先验
    PLS,           // Parallel Level Sets
    RDP,           // Relative Difference Penalty
    LogCosh,       // LogCosh 边缘保持
    Unknown
};

// ================= 正则化器配置 =================

struct RegConf {
    RegType type = RegType::None;
    double beta = 0.0;             // 正则权重
    double tv_eps = 1e-6;          // TV 稳定项
    double mrp_eps = 1e-8;         // MRP 稳定项
    bool skip_outside_mask = true; // 掩码外跳过

    // NLM 参数
    double nlm_h = 0.1;
    int nlm_search = 3;  // 搜索窗口半径
    int nlm_patch = 1;   // patch 半径

    // 解剖先验
    const std::vector<double> *anatomical_image = nullptr;
    
    // Bowsher
    int bowsher_k = 13;  // 邻域数量
    mutable std::vector<int> bowsher_indices;
    mutable bool bowsher_precomputed = false;
    
    // PLS
    double pls_eta = 1.0;
    mutable std::vector<float> pls_nx, pls_ny, pls_nz;
    mutable bool pls_precomputed = false;

    // Huber / RDP / LogCosh
    double delta = 0.01;
    double gamma = 0.0;  // RDP gamma

    // 其他
    double sigma = 1.0;
    double power = 2.0;
    int kernel = 3;

    // 更新调度
    int numIter = 1;
    int numSubsets = 1;
    int updateFreq = 1;
    int updateStart = 0;
    int updateEnd = 9999;
    int updateInterval = 1;
    enum class UpdateType { Subset, Iteration };
    UpdateType updateType = UpdateType::Subset;
};

// ================= 辅助函数 =================

RegType stringToRegType(const std::string &str);
std::string regTypeToString(RegType type);

/**
 * @brief 计算正则化梯度
 * 
 * @param f 当前图像
 * @param Nx, Ny, Nz 图像尺寸
 * @param reg 正则化配置
 * @param mask 可选掩码
 * @param gradR 输出梯度
 * @param threads 线程数
 */
void compute_reg_gradient(
    const std::vector<double> &f,
    int Nx, int Ny, int Nz,
    const RegConf &reg,
    const std::vector<uint8_t> *mask,
    std::vector<double> &gradR,
    int threads
);

/**
 * @brief OSL 更新步骤
 * 
 * f *= q / (d + beta * gradR/f)
 */
void osl_update_inplace(
    std::vector<double> &f,
    const std::vector<double> &q,
    const std::vector<double> &d,
    const std::vector<double> &gradR,
    double beta,
    const std::vector<uint8_t> *mask,
    bool nonneg,
    int threads
);

} // namespace regularizer
} // namespace spect
