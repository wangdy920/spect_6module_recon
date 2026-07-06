/**
 * @file PSFConfig.h
 * @brief PSF configuration structures
 * 
 * 从 original_source_backup/Filters.h 迁移
 */

#pragma once
#include <vector>
#include <string>

namespace spect {
namespace filters {

// ================= PSF 模式与配置 =================

enum class PSFMode { 
    None, 
    Operator,         // PSF 作为算子在迭代中应用
    PostSmooth,       // PSF 作为后处理平滑
    ProjDomain,       // 投影域 PSF
    Gaussian,         // 简单高斯 PSF
    File              // 从文件加载 PSF
};

enum class PSFInterp { Linear, Nearest };

// 三维体素间距
struct Spacing3 { 
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
};

/// PSF 配置上下文
struct PSFConf {
    PSFMode mode = PSFMode::None;
    
    // 各向同性/各向异性高斯 FWHM（单位：mm）
    double fwhm_x = 0.0;
    double fwhm_y = 0.0;
    double fwhm_z = 0.0;
    double fwhm_mm = 1.5;  // 简化接口，单位 mm
    
    // 体素尺寸（mm）
    double vox_x = 1.0;
    double vox_y = 1.0;
    double vox_z = 1.0;
    
    // 高级 PSF
    std::string filePath;
    double truncation_mm = 3.0;
    bool z_dependent = false;
    PSFInterp interp = PSFInterp::Linear;
    double min_sigmas = 4.0;         // 核截断阈值：N * sigma
};

// ================= Butterworth 滤波器配置 =================

enum class BWType {
    None,
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop
};

enum class ButterworthBackendChoice {
    Auto,
    FFTW,
    CUFFT
};

struct ButterworthConf {
    bool enabled = false;
    BWType type = BWType::Lowpass;
    ButterworthBackendChoice backend = ButterworthBackendChoice::Auto;
    int order = 4;
    
    // 截止频率（Nyquist 分数：0~1）
    double cutoff_lo_frac = 0.35;
    double cutoff_hi_frac = 0.35;
    
    // 应用时机
    bool apply_after_each_subset = false;
    int every_k_iters = 1;
};

// ================= PSF 核数据结构 =================

/// 三向可分离核
struct PSFKernels {
    std::vector<double> kx;
    std::vector<double> ky;
    std::vector<double> kz;
};

} // namespace filters
} // namespace spect
