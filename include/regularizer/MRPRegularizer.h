// MRPRegularizer.h
// 任务 T4.2 - Median Root Prior (MRP) 正则化器
// 能量: E = β * Σ_j (√f_j - median(√f_k))²
// 梯度采用简化形式: ∇E_j ∝ Σ_k (1 - √f_k/√f_j)
#pragma once

#include "RegularizerBase.h"
#include <algorithm>
#include <omp.h>

namespace spect {
namespace regularizer {

/**
 * @brief Median Root Prior (MRP) Regularizer
 * 
 * 基于Alenius & Ruotsalainen (1997)的MRP先验。
 * 使用平方根变换来处理Poisson噪声。
 */
class MRPRegularizer : public RegularizerBase {
public:
    explicit MRPRegularizer(double eps = 1e-8) : eps_(eps) {}
    
    std::string name() const override { return "MRP"; }
    
    double energy(const double* f, const Dims3D& dims,
                  const uint8_t* mask = nullptr) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        const double eps_root = eps_ * eps_;
        
        double E = 0.0;
        std::vector<double> neighbors(6);
        
        for (size_t idx = 0; idx < N; ++idx) {
            if (mask && !mask[idx]) continue;
            
            int z = static_cast<int>(idx / (Nx * Ny));
            int y = static_cast<int>((idx - z * Nx * Ny) / Nx);
            int x = static_cast<int>(idx - z * Nx * Ny - y * Nx);
            
            double sj = std::sqrt(std::max(f[idx], 0.0) + eps_root);
            
            int count = 0;
            auto collectNeighbor = [&](int xx, int yy, int zz) {
                if (xx < 0 || xx >= Nx || yy < 0 || yy >= Ny || zz < 0 || zz >= Nz)
                    return;
                size_t nid = static_cast<size_t>(zz) * Ny * Nx + yy * Nx + xx;
                if (mask && !mask[nid]) return;
                neighbors[count++] = std::sqrt(std::max(f[nid], 0.0) + eps_root);
            };
            
            collectNeighbor(x - 1, y, z);
            collectNeighbor(x + 1, y, z);
            collectNeighbor(x, y - 1, z);
            collectNeighbor(x, y + 1, z);
            collectNeighbor(x, y, z - 1);
            collectNeighbor(x, y, z + 1);
            
            if (count > 0) {
                std::nth_element(neighbors.begin(), neighbors.begin() + count/2, 
                                 neighbors.begin() + count);
                double median = neighbors[count/2];
                double diff = sj - median;
                E += diff * diff;
            }
        }
        return beta_ * E;
    }
    
    void gradient(const double* f, double* grad, const Dims3D& dims,
                  const uint8_t* mask = nullptr, int threads = 1) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        const double eps_root = eps_ * eps_;
        
        std::fill(grad, grad + N, 0.0);
        
        omp_set_num_threads(threads);
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t z = 0; z < Nz; ++z) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    size_t id = static_cast<size_t>(z) * Ny * Nx + y * Nx + x;
                    if (mask && !mask[id]) continue;
                    
                    double sj = std::sqrt(std::max(f[id], 0.0) + eps_root);
                    double g = 0.0;
                    
                    auto accum = [&](int xx, int yy, int zz) {
                        if (xx < 0 || xx >= Nx || yy < 0 || yy >= Ny || zz < 0 || zz >= Nz)
                            return;
                        size_t nid = static_cast<size_t>(zz) * Ny * Nx + yy * Nx + xx;
                        if (mask && !mask[nid]) return;
                        double sk = std::sqrt(std::max(f[nid], 0.0) + eps_root);
                        g += (1.0 - sk / sj);  // = (√f_j - √f_k)/√f_j
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
    double eps_;
};

} // namespace regularizer
} // namespace spect
