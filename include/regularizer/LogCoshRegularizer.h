// LogCoshRegularizer.h
// 任务 T4.2 - Log-Cosh 正则化器
// 能量: E = β * Σ_j Σ_k∈N(j) δ² * log(cosh((f_j - f_k) / δ))
// 梯度: ∇E_j = β * Σ_k∈N(j) δ * tanh((f_j - f_k) / δ)
// 
// 性质:
// - |x| << δ: 行为类似L2 (二次)
// - |x| >> δ: 行为类似L1 (线性)
// - 平滑过渡提供边缘保持正则化
#pragma once

#include "RegularizerBase.h"
#include <algorithm>
#include <omp.h>

namespace spect {
namespace regularizer {

/**
 * @brief Log-Cosh Regularizer
 * 
 * 鲁棒的惩罚函数，在小差分处表现为L2，
 * 在大差分处表现为L1，可保持边缘。
 */
class LogCoshRegularizer : public RegularizerBase {
public:
    explicit LogCoshRegularizer(double delta = 0.01) 
        : delta_(std::max(delta, 1e-9)) {}
    
    std::string name() const override { return "LogCosh"; }
    
    void setDelta(double delta) { delta_ = std::max(delta, 1e-9); }
    double delta() const { return delta_; }
    
    double energy(const double* f, const Dims3D& dims,
                  const uint8_t* mask = nullptr) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        const double inv_delta = 1.0 / delta_;
        
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
                double scaled = diff * inv_delta;
                // log(cosh(x)) 数值稳定实现
                double abs_scaled = std::abs(scaled);
                if (abs_scaled < 20.0) {
                    E += std::log(std::cosh(scaled));
                } else {
                    // 对于大|x|, log(cosh(x)) ≈ |x| - log(2)
                    E += abs_scaled - 0.693147180559945;
                }
            };
            // 只计算正方向邻居，避免重复
            if (x + 1 < Nx) accumEnergy(x + 1, y, z);
            if (y + 1 < Ny) accumEnergy(x, y + 1, z);
            if (z + 1 < Nz) accumEnergy(x, y, z + 1);
        }
        return beta_ * delta_ * delta_ * E;
    }
    
    void gradient(const double* f, double* grad, const Dims3D& dims,
                  const uint8_t* mask = nullptr, int threads = 1) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        const double inv_delta = 1.0 / delta_;
        
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
                        
                        double diff = fj - f[nid];
                        double scaled = diff * inv_delta;
                        // tanh 是稳定的
                        g += std::tanh(scaled);
                    };
                    
                    accum(x - 1, y, z);
                    accum(x + 1, y, z);
                    accum(x, y - 1, z);
                    accum(x, y + 1, z);
                    accum(x, y, z - 1);
                    accum(x, y, z + 1);
                    
                    // 梯度 = δ * tanh(diff/δ)
                    grad[id] = beta_ * delta_ * g;
                }
            }
        }
    }
    
private:
    double delta_;
};

} // namespace regularizer
} // namespace spect
