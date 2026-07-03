// NLMRegularizer.h
// 任务 T4.2 - Non-Local Means (NLM) 正则化器
// 基于 Buades (2005) 的非局部均值降噪
// 使用patch相似性来计算权重，适合保留纹理和重复结构
#pragma once

#include "RegularizerBase.h"
#include <algorithm>
#include <omp.h>

namespace spect {
namespace regularizer {

/**
 * @brief Non-Local Means (NLM) Regularizer
 * 
 * 使用patch相似性计算像素间权重，
 * 允许远距离像素相互影响（如果patch相似）。
 */
class NLMRegularizer : public RegularizerBase {
public:
    explicit NLMRegularizer(double h = 0.1, int searchRadius = 3, int patchRadius = 1) 
        : h_(h), searchRadius_(searchRadius), patchRadius_(patchRadius) {}
    
    std::string name() const override { return "NLM"; }
    
    void setH(double h) { h_ = std::max(h, 1e-8); }
    void setSearchRadius(int r) { searchRadius_ = std::max(r, 1); }
    void setPatchRadius(int r) { patchRadius_ = std::max(r, 0); }
    
    double h() const { return h_; }
    int searchRadius() const { return searchRadius_; }
    int patchRadius() const { return patchRadius_; }
    
    double energy(const double* f, const Dims3D& dims,
                  const uint8_t* mask = nullptr) const override {
        // NLM的能量函数较复杂，这里采用简化形式
        // E = 0.5 * Σ_i Σ_j w_ij (f_i - f_j)²
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        const double h2 = std::max(h_ * h_, 1e-16);
        
        double E = 0.0;
        for (size_t idx = 0; idx < N; ++idx) {
            if (mask && !mask[idx]) continue;
            
            int z = static_cast<int>(idx / (Nx * Ny));
            int y = static_cast<int>((idx - z * Nx * Ny) / Nx);
            int x = static_cast<int>(idx - z * Nx * Ny - y * Nx);
            
            double val_i = f[idx];
            
            // 搜索窗口
            for (int dz = -searchRadius_; dz <= searchRadius_; ++dz) {
                for (int dy = -searchRadius_; dy <= searchRadius_; ++dy) {
                    for (int dx = -searchRadius_; dx <= searchRadius_; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        
                        int nx = x + dx, ny = y + dy, nz = z + dz;
                        if (nx < 0 || nx >= Nx || ny < 0 || ny >= Ny || nz < 0 || nz >= Nz)
                            continue;
                        
                        size_t nid = static_cast<size_t>(nz) * Ny * Nx + ny * Nx + nx;
                        if (mask && !mask[nid]) continue;
                        
                        // 计算patch距离
                        double dist = computePatchDistance(f, x, y, z, nx, ny, nz, 
                                                           dims, mask);
                        double w = std::exp(-dist / h2);
                        double diff = val_i - f[nid];
                        E += 0.5 * w * diff * diff;
                    }
                }
            }
        }
        return beta_ * E;
    }
    
    void gradient(const double* f, double* grad, const Dims3D& dims,
                  const uint8_t* mask = nullptr, int threads = 1) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        const double h2 = std::max(h_ * h_, 1e-16);
        
        std::fill(grad, grad + N, 0.0);
        
        omp_set_num_threads(threads);
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t z = 0; z < Nz; ++z) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    size_t id = static_cast<size_t>(z) * Ny * Nx + y * Nx + x;
                    if (mask && !mask[id]) continue;
                    
                    double g = 0.0;
                    double val_i = f[id];
                    
                    // 搜索窗口
                    for (int dz = -searchRadius_; dz <= searchRadius_; ++dz) {
                        for (int dy = -searchRadius_; dy <= searchRadius_; ++dy) {
                            for (int dx = -searchRadius_; dx <= searchRadius_; ++dx) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                
                                int nx = x + dx, ny = y + dy, nz = static_cast<int>(z) + dz;
                                if (nx < 0 || nx >= Nx || ny < 0 || ny >= Ny || 
                                    nz < 0 || nz >= Nz) continue;
                                
                                size_t nid = static_cast<size_t>(nz) * Ny * Nx + ny * Nx + nx;
                                if (mask && !mask[nid]) continue;
                                
                                // 计算patch距离
                                double dist = computePatchDistance(f, x, y, static_cast<int>(z), 
                                                                   nx, ny, nz, dims, mask);
                                double w = std::exp(-dist / h2);
                                g += w * (val_i - f[nid]);
                            }
                        }
                    }
                    grad[id] = beta_ * g;
                }
            }
        }
    }
    
private:
    double h_;
    int searchRadius_;
    int patchRadius_;
    
    inline double computePatchDistance(const double* f, 
                                        int ix, int iy, int iz,
                                        int jx, int jy, int jz,
                                        const Dims3D& dims, const uint8_t* mask) const {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        double dist = 0.0;
        int count = 0;
        
        for (int pz = -patchRadius_; pz <= patchRadius_; ++pz) {
            for (int py = -patchRadius_; py <= patchRadius_; ++py) {
                for (int px = -patchRadius_; px <= patchRadius_; ++px) {
                    int ix2 = ix + px, iy2 = iy + py, iz2 = iz + pz;
                    int jx2 = jx + px, jy2 = jy + py, jz2 = jz + pz;
                    
                    if (ix2 >= 0 && ix2 < Nx && iy2 >= 0 && iy2 < Ny && iz2 >= 0 && iz2 < Nz &&
                        jx2 >= 0 && jx2 < Nx && jy2 >= 0 && jy2 < Ny && jz2 >= 0 && jz2 < Nz) {
                        size_t id1 = static_cast<size_t>(iz2) * Ny * Nx + iy2 * Nx + ix2;
                        size_t id2 = static_cast<size_t>(jz2) * Ny * Nx + jy2 * Nx + jx2;
                        
                        if ((!mask || mask[id1]) && (!mask || mask[id2])) {
                            double diff = f[id1] - f[id2];
                            dist += diff * diff;
                            count++;
                        }
                    }
                }
            }
        }
        return (count > 0) ? dist / count : 0.0;
    }
};

} // namespace regularizer
} // namespace spect
