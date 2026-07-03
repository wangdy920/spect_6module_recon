// TikhonovRegularizer.h
// 任务 T4.2 - Tikhonov正则化器 (二次平滑)
// 能量: E = β * Σ_j Σ_k∈N(j) (f_j - f_k)²
// 梯度: ∇E_j = 2β * Σ_k∈N(j) (f_j - f_k)
#pragma once

#include "RegularizerBase.h"
#include <omp.h>

namespace spect {
namespace regularizer {

/**
 * @brief Tikhonov (Quadratic) Regularizer
 * 
 * 也称为L2正则化或拉普拉斯平滑。
 * 对所有梯度施加二次惩罚。
 */
class TikhonovRegularizer : public RegularizerBase {
public:
    TikhonovRegularizer() = default;
    
    std::string name() const override { return "Tikhonov"; }
    
    double energy(const double* f, const Dims3D& dims, 
                  const uint8_t* mask = nullptr) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        
        double E = 0.0;
        for (size_t idx = 0; idx < N; ++idx) {
            if (mask && !mask[idx]) continue;
            
            int z = static_cast<int>(idx / (Nx * Ny));
            int y = static_cast<int>((idx - z * Nx * Ny) / Nx);
            int x = static_cast<int>(idx - z * Nx * Ny - y * Nx);
            
            double fj = f[idx];
            auto accumEnergy = [&](int xx, int yy, int zz) {
                if (xx < 0 || xx >= Nx || yy < 0 || yy >= Ny || zz < 0 || zz >= Nz)
                    return;
                size_t nid = static_cast<size_t>(zz) * Ny * Nx + yy * Nx + xx;
                if (mask && !mask[nid]) return;
                double diff = fj - f[nid];
                E += diff * diff;
            };
            // 只计算正方向邻居，避免重复
            if (x + 1 < Nx) accumEnergy(x + 1, y, z);
            if (y + 1 < Ny) accumEnergy(x, y + 1, z);
            if (z + 1 < Nz) accumEnergy(x, y, z + 1);
        }
        return 0.5 * beta_ * E;  // 0.5因为每对邻居算两次
    }
    
    void gradient(const double* f, double* grad, const Dims3D& dims,
                  const uint8_t* mask = nullptr, int threads = 1) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        
        std::fill(grad, grad + N, 0.0);
        
        omp_set_num_threads(threads);
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t z = 0; z < Nz; ++z) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    size_t id = static_cast<size_t>(z) * Ny * Nx + y * Nx + x;
                    if (mask && !mask[id]) continue;
                    
                    double fj = f[id];
                    double g = 0.0;
                    
                    auto accum = [&](int xx, int yy, int zz) {
                        if (xx < 0 || xx >= Nx || yy < 0 || yy >= Ny || zz < 0 || zz >= Nz)
                            return;
                        size_t nid = static_cast<size_t>(zz) * Ny * Nx + yy * Nx + xx;
                        if (mask && !mask[nid]) return;
                        g += (fj - f[nid]);
                    };
                    
                    accum(x - 1, y, z);
                    accum(x + 1, y, z);
                    accum(x, y - 1, z);
                    accum(x, y + 1, z);
                    accum(x, y, z - 1);
                    accum(x, y, z + 1);
                    
                    // 梯度 = 2 * β * Σ (f_j - f_k)
                    // 因子2来自 d/df_j [0.5 * (f_j - f_k)^2] = (f_j - f_k)
                    // 但每对邻居贡献两次，所以需要因子2
                    grad[id] = 2.0 * beta_ * g;
                }
            }
        }
    }
};

} // namespace regularizer
} // namespace spect
