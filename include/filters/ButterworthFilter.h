/**
 * @file ButterworthFilter.h
 * @brief Butterworth 频域滤波器
 * 
 * 从 original_source_backup/Filters.h 迁移
 */

#pragma once
#include "PSFConfig.h"
#include <vector>

namespace spect {
namespace filters {

inline bool isButterworthCUFFTAvailable() {
#if defined(SPECT_BUTTERWORTH_HAVE_CUFFT)
    return true;
#else
    return false;
#endif
}

/**
 * @brief 构建 3D Butterworth 频域掩膜
 * 
 * @param Nx, Ny, Nz 体积尺寸
 * @param sp 体素间距
 * @param bw Butterworth 配置
 * @param threads 线程数
 * @return 频域掩膜（未 shift 布局）
 */
std::vector<double> make_butterworth_mask_3d(
    int Nx, int Ny, int Nz,
    Spacing3 sp,
    const ButterworthConf& bw,
    int threads = 1
);

/**
 * @brief 应用 Butterworth 滤波器（统一调度入口）
 *
 * Production execution is CUDA/cuFFT only. Builds without cuFFT fail fast when
 * this operation is requested.
 */
void apply_butterworth_inplace(
    std::vector<double>& f,
    int Nx, int Ny, int Nz,
    Spacing3 sp,
    const ButterworthConf& bw,
    int threads = 1
);

/**
 * @brief 应用 Butterworth 滤波器（cuFFT 后端）
 *
 * 当未编入 cuFFT 实现时，该函数会抛出运行时错误。
 */
void apply_butterworth_inplace_cufft(
    std::vector<double>& f,
    int Nx, int Ny, int Nz,
    Spacing3 sp,
    const ButterworthConf& bw,
    int threads = 1
);

/**
 * @brief 输出级 3D Butterworth 低通（严格对齐 6module 原型）
 *
 * 与原型 spect_recon_PSC_6module_24pin_YSO_bedmerge.cu 的口径一致：
 *  - 物理截止频率 cutoff_cpcm（cycles/cm），fc_mm = cutoff_cpcm / 10；
 *    H(f) = 1 / (1 + (f/fc)^(2*order))，f 为按各轴 spacing 计算的径向
 *    物理频率（cycles/mm）。
 *  - cutoff_cpcm 超过各轴最小 Nyquist（5 / max(spacing) cycles/cm）时直接
 *    抛出 std::runtime_error。
 *  - 每轴 2 倍零填充（2Nx,2Ny,2Nz），原图居中嵌入，cuFFT float R2C/C2R，
 *    逆变换乘以 1/N_padded，裁剪回原始尺寸。滤波保持线性（负旁瓣在
 *    保存阶段由 saveVolumeOutput 统一钳为非负）。
 *
 * 仅作用于传入的副本；不改动迭代状态。当未编入 cuFFT 实现时抛出运行时错误。
 */
void apply_butterworth_output_cufft(
    std::vector<double>& f,
    int Nx, int Ny, int Nz,
    Spacing3 sp,
    double cutoff_cpcm,
    int order
);

} // namespace filters
} // namespace spect
