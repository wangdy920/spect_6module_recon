/**
 * @file Convolution.h
 * @brief 3D/2D 可分离卷积操作
 * 
 * 从 original_source_backup/Filters.h 迁移
 */

#pragma once
#include "PSFConfig.h"
#include <vector>
#include <cstddef>

namespace spect {
namespace filters {

// ================= 工具函数 =================

/// 列主序线性索引: (i, j, k) -> i + Nx * j + Nx * Ny * k
inline size_t idx3(size_t i, size_t j, size_t k, size_t Nx, size_t Ny) {
    return i + Nx * j + Nx * Ny * k;
}

/// 生成 1D 高斯核（归一化）
std::vector<double> makeGaussian1D(double fwhm_mm, double vox_mm,
                                   double min_sigmas = 4.0);

// CASToR 严格对齐的 erf 体素积分核：k[i] = 0.5*(erf((i+0.5)*vox/(sqrt2*sigma))
// - erf((i-0.5)*vox/(sqrt2*sigma)))，半径 = castor_round(sigma*N_sigmas/vox)
// （小数部分 > 0.5 才进位，恰 0.5 舍去；半径可为 0）。
std::vector<double> makeCastorErf1D(double fwhm_mm, double vox_mm,
                                    double nb_sigmas);

// CASToR z 边界补偿因子：norm[z] = 落入 [0,Nz) 的 kz 权重和（与 x,y 无关）。
std::vector<double> makeCastorZNorm1D(const std::vector<double>& kz, int Nz);

// CASToR 前向卷积 C = D_z^-1 * K：零扩展可分离卷积（x/y/z 均不补偿），
// 随后逐体素除以 z 边界补偿因子。反向（伴随侧）按 CASToR 源码复用本函数。
void conv3d_castor_forward_inplace(std::vector<double>& f, int Nx, int Ny,
                                   int Nz, const std::vector<double>& kx,
                                   const std::vector<double>& ky,
                                   const std::vector<double>& kz,
                                   int threads);

// ================= 1D 卷积（内部使用）=================

void conv1d_x_mirror(const std::vector<double>& in, std::vector<double>& out,
    size_t Nx, size_t Ny, size_t Nz, const std::vector<double>& kx);

void conv1d_y_mirror(const std::vector<double>& in, std::vector<double>& out,
    size_t Nx, size_t Ny, size_t Nz, const std::vector<double>& ky);

void conv1d_z_mirror(const std::vector<double>& in, std::vector<double>& out,
    size_t Nx, size_t Ny, size_t Nz, const std::vector<double>& kz);

// ================= 3D 可分离卷积 =================

/**
 * @brief 3D 可分离卷积（支持权重归一化）
 * 
 * 应用三次 1D 卷积（X->Y->Z），可选归一化以处理边界/掩码效应。
 */
void conv3_separable_norm(
    const std::vector<double>& in,
    std::vector<double>& out,
    size_t Nx, size_t Ny, size_t Nz,
    const std::vector<double>& kx,
    const std::vector<double>& ky,
    const std::vector<double>& kz,
    const std::vector<double>* weight = nullptr,
    double eps = 1e-12
);

/**
 * @brief 3D 可分离卷积（原地操作）
 * 
 * 使用 ping-pong 缓冲区减少内存占用。
 */
void conv3d_separable_inplace(
    std::vector<double>& f,
    int Nx, int Ny, int Nz,
    const std::vector<double>& kx,
    const std::vector<double>& ky,
    const std::vector<double>& kz,
    int threads
);

/**
 * @brief Exact transpose of conv3d_separable_inplace.
 *
 * Mirror extension is not a symmetric boundary operator, so the adjoint must
 * accumulate boundary samples with the transpose of each 1D pass.
 */
void conv3d_separable_adjoint_inplace(
    std::vector<double>& f,
    int Nx, int Ny, int Nz,
    const std::vector<double>& kx,
    const std::vector<double>& ky,
    const std::vector<double>& kz,
    int threads
);

// ================= PSF 核生成 =================

/// 根据 PSF 配置生成三向核
PSFKernels make_psf_kernels(const PSFConf& psf);

} // namespace filters
} // namespace spect
