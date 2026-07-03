// RDPRegularizer.h
// 任务 T4.2 - Relative Difference Penalty (RDP) 正则化器
// 基于 Nuyts (2002) 的相对差分惩罚
// 能量: E = 0.5 * Σ_j Σ_k∈N(j) (f_j - f_k)² / (f_j + f_k + γ|f_j-f_k| + 2δ)
#pragma once

#include "RegularizerBase.h"
#include <algorithm>
#include <omp.h>

namespace spect {
namespace regularizer {

/**
 * @brief Relative Difference Penalty (RDP) Regularizer
 * 
 * 相对差分惩罚对相对变化进行惩罚而非绝对变化，
 * 这使得它对高强度区域的边缘更加宽容。
 */
class RDPRegularizer : public RegularizerBase {
public:
    explicit RDPRegularizer(double delta = 1e-6, double gamma = 0.0) 
        : delta_(delta), gamma_(gamma) {}
    
    std::string name() const override { return "RDP"; }
    
    void setDelta(double delta) { delta_ = std::max(delta, 1e-9); }
    void setGamma(double gamma) { gamma_ = gamma; }
    double delta() const { return delta_; }
    double gamma() const { return gamma_; }
    
    double energy(const double* f, const Dims3D& dims,
                  const uint8_t* mask = nullptr) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        const double eps = std::max(delta_, 1e-9);
        
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
                double fk = f[nid];
                double diff = fj - fk;
                double abs_diff = std::abs(diff);
                double denom = fj + fk + gamma_ * abs_diff + 2.0 * eps;
                E += (diff * diff) / denom;
            };
            // 只计算正方向邻居，避免重复
            if (x + 1 < Nx) accumEnergy(x + 1, y, z);
            if (y + 1 < Ny) accumEnergy(x, y + 1, z);
            if (z + 1 < Nz) accumEnergy(x, y, z + 1);
        }
        return 0.5 * beta_ * E;
    }
    
    void gradient(const double* f, double* grad, const Dims3D& dims,
                  const uint8_t* mask = nullptr, int threads = 1) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        const double eps = std::max(delta_, 1e-9);
        
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
                        
                        double fk = f[nid];
                        double diff = fj - fk;
                        double abs_diff = std::abs(diff);
                        double sign_diff = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
                        
                        double denom = fj + fk + gamma_ * abs_diff + 2.0 * eps;
                        if (denom < 1e-9) denom = 1e-9;
                        
                        // d(diff^2/S)/d(fj) = [2*diff*S - diff^2 * (1 + gamma*sign(diff))] / S^2
                        double term_num = 2.0 * diff * denom - (diff * diff) * (1.0 + gamma_ * sign_diff);
                        g += term_num / (denom * denom);
                    };
                    
                    accum(x - 1, y, z);
                    accum(x + 1, y, z);
                    accum(x, y - 1, z);
                    accum(x, y + 1, z);
                    accum(x, y, z - 1);
                    accum(x, y, z + 1);
                    
                    grad[id] = beta_ * g;
                }
            }
        }
    }
    
private:
    double delta_;
    double gamma_;
};

} // namespace regularizer
} // namespace spect
