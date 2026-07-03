/**
 * @file ButterworthFilter.h
 * @brief Butterworth 频域滤波器
 * 
 * 从 original_source_backup/Filters.h 迁移
 */

#pragma once
#include "PSFConfig.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace spect {
namespace filters {

inline const char* butterworthBackendChoiceName(
    ButterworthBackendChoice choice) {
    switch (choice) {
    case ButterworthBackendChoice::Auto:
        return "auto";
    case ButterworthBackendChoice::FFTW:
        return "fftw";
    case ButterworthBackendChoice::CUFFT:
        return "cufft";
    default:
        return "auto";
    }
}

inline bool parseButterworthBackendChoice(const std::string& raw,
                                          ButterworthBackendChoice& choice) {
    std::string value = raw;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value.empty() || value == "auto") {
        choice = ButterworthBackendChoice::Auto;
        return true;
    }
    if (value == "fftw" || value == "cpu") {
        choice = ButterworthBackendChoice::FFTW;
        return true;
    }
    if (value == "cufft" || value == "cuda" || value == "gpu") {
        choice = ButterworthBackendChoice::CUFFT;
        return true;
    }
    return false;
}

inline bool isButterworthCUFFTAvailable() {
#if defined(SPECT_BUTTERWORTH_HAVE_CUFFT)
    return true;
#else
    return false;
#endif
}

inline ButterworthBackendChoice defaultButterworthBackendChoice() {
#if defined(SPECT_BUTTERWORTH_DEFAULT_BACKEND_CUFFT)
    return ButterworthBackendChoice::CUFFT;
#elif defined(SPECT_BUTTERWORTH_DEFAULT_BACKEND_FFTW)
    return ButterworthBackendChoice::FFTW;
#else
    return ButterworthBackendChoice::Auto;
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
 * 当前默认调度到 FFTW CPU 实现；后续可在此处接入 cuFFT 等后端，
 * 而不需要修改调用方。
 */
void apply_butterworth_inplace(
    std::vector<double>& f,
    int Nx, int Ny, int Nz,
    Spacing3 sp,
    const ButterworthConf& bw,
    int threads = 1
);

/**
 * @brief 应用 Butterworth 滤波器（原地操作，使用 FFTW3）
 * 
 * 执行 FFT -> 频域乘法 -> IFFT。
 * 使用 thread-local 缓存避免重复分配/计划。
 * 
 * @param f 输入/输出体数据
 * @param Nx, Ny, Nz 体积尺寸
 * @param sp 体素间距
 * @param bw Butterworth 配置
 * @param threads 线程数
 */
void apply_butterworth_inplace_fftw(
    std::vector<double>& f,
    int Nx, int Ny, int Nz,
    Spacing3 sp,
    const ButterworthConf& bw,
    int threads = 1
);

/**
 * @brief 应用 Butterworth 滤波器（cuFFT 后端）
 *
 * 当未编入 cuFFT 实现时，该函数会抛出运行时错误；调度入口会在调用前
 * 根据可用性决定是否回退到 FFTW。
 */
void apply_butterworth_inplace_cufft(
    std::vector<double>& f,
    int Nx, int Ny, int Nz,
    Spacing3 sp,
    const ButterworthConf& bw,
    int threads = 1
);

} // namespace filters
} // namespace spect
