// BowsherRegularizer.h
// 任务 T4.2 - Bowsher Prior 正则化器
// 基于 Bowsher (2004) 的解剖先验
// 从26邻域中选择解剖图像差异最小的K个邻居进行平滑
#pragma once

#include "RegularizerBase.h"
#include <algorithm>
#include <omp.h>
#include <vector>

namespace spect {
namespace regularizer {

/**
 * @brief Bowsher Prior Regularizer
 * 
 * 使用解剖图像（如CT或MRI）来指导正则化，
 * 只在解剖相似的邻居之间进行平滑。
 */
class BowsherRegularizer : public RegularizerBase {
public:
    explicit BowsherRegularizer(int k = 13) : k_(k) {}
    
    std::string name() const override { return "Bowsher"; }
    
    void setK(int k) { k_ = std::max(1, std::min(k, 26)); }
    int k() const { return k_; }
    
    void setAnatomicalImage(const double* anat, size_t size) {
        anatomicalImage_ = anat;
        anatomicalSize_ = size;
        precomputed_ = false;
        cachedIndices_.clear();
    }
    
    double energy(const double* f, const Dims3D& dims,
                  const uint8_t* mask = nullptr) const override {
        if (!anatomicalImage_) return 0.0;
        
        ensurePrecomputed(f, dims, mask);
        
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        
        double E = 0.0;
        for (size_t id = 0; id < N; ++id) {
            if (mask && !mask[id]) continue;
            
            double fj = f[id];
            size_t base = id * k_;
            for (int i = 0; i < k_; ++i) {
                int nid = cachedIndices_[base + i];
                if (nid < 0) break;
                double diff = fj - f[nid];
                E += diff * diff;
            }
        }
        return 0.5 * beta_ * E;
    }
    
    void gradient(const double* f, double* grad, const Dims3D& dims,
                  const uint8_t* mask = nullptr, int threads = 1) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        
        std::fill(grad, grad + N, 0.0);
        
        if (!anatomicalImage_) return;
        
        ensurePrecomputed(f, dims, mask);
        
        omp_set_num_threads(threads);
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t id = 0; id < static_cast<ptrdiff_t>(N); ++id) {
            if (mask && !mask[id]) continue;
            
            double fj = f[id];
            double g = 0.0;
            size_t base = id * k_;
            
            for (int i = 0; i < k_; ++i) {
                int nid = cachedIndices_[base + i];
                if (nid < 0) break;
                g += (fj - f[nid]);
            }
            grad[id] = beta_ * g;
        }
    }
    
private:
    int k_;
    const double* anatomicalImage_ = nullptr;
    size_t anatomicalSize_ = 0;
    
    mutable bool precomputed_ = false;
    mutable std::vector<int> cachedIndices_;
    mutable Dims3D cachedDims_{0, 0, 0};
    
    void ensurePrecomputed(const double* /*f*/, const Dims3D& dims, 
                           const uint8_t* mask) const {
        if (precomputed_ && 
            dims.nx == cachedDims_.nx && 
            dims.ny == cachedDims_.ny && 
            dims.nz == cachedDims_.nz) {
            return;
        }
        
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        
        cachedIndices_.assign(N * k_, -1);
        
        struct Neighbor {
            double absDiff;
            int nid;
        };
        
        for (size_t id = 0; id < N; ++id) {
            if (mask && !mask[id]) continue;
            
            int z = static_cast<int>(id / (Nx * Ny));
            int y = static_cast<int>((id - z * Nx * Ny) / Nx);
            int x = static_cast<int>(id - z * Nx * Ny - y * Nx);
            
            Neighbor neighbors[26];
            int count = 0;
            double valAnat = anatomicalImage_[id];
            
            // 26邻域
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        
                        int nx = x + dx, ny = y + dy, nz = z + dz;
                        if (nx < 0 || nx >= Nx || ny < 0 || ny >= Ny || 
                            nz < 0 || nz >= Nz) continue;
                        
                        size_t nid = static_cast<size_t>(nz) * Ny * Nx + ny * Nx + nx;
                        if (mask && !mask[nid]) continue;
                        
                        double dAnat = std::abs(valAnat - anatomicalImage_[nid]);
                        neighbors[count++] = {dAnat, static_cast<int>(nid)};
                    }
                }
            }
            
            int kEff = std::min(k_, count);
            if (kEff > 0) {
                std::partial_sort(neighbors, neighbors + kEff, neighbors + count,
                                  [](const Neighbor& a, const Neighbor& b) {
                                      return a.absDiff < b.absDiff;
                                  });
                for (int i = 0; i < kEff; ++i) {
                    cachedIndices_[id * k_ + i] = neighbors[i].nid;
                }
            }
        }
        
        cachedDims_ = dims;
        precomputed_ = true;
    }
};

} // namespace regularizer
} // namespace spect
