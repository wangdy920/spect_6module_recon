// PLSRegularizer.h
// 任务 T4.2 - Parallel Level Sets (PLS) 正则化器
// 基于 Ehrhardt (2015) 的平行水平集先验
// 只惩罚与解剖边缘不平行的图像梯度
#pragma once

#include "RegularizerBase.h"
#include <algorithm>
#include <omp.h>
#include <vector>

namespace spect {
namespace regularizer {

/**
 * @brief Parallel Level Sets (PLS) Regularizer
 * 
 * 允许图像梯度与解剖梯度平行，
 * 只惩罚垂直于解剖边缘的梯度分量。
 * R(f) = Σ |P * ∇f|², 其中 P = I - n*n^T, n = ∇anat/|∇anat|
 */
class PLSRegularizer : public RegularizerBase {
public:
    explicit PLSRegularizer(double eta = 1.0) : eta_(eta) {}
    
    std::string name() const override { return "PLS"; }
    
    void setEta(double eta) { eta_ = eta; }
    double eta() const { return eta_; }
    
    void setAnatomicalImage(const double* anat, size_t size) {
        anatomicalImage_ = anat;
        anatomicalSize_ = size;
        precomputed_ = false;
        normalsX_.clear();
        normalsY_.clear();
        normalsZ_.clear();
    }
    
    double energy(const double* f, const Dims3D& dims,
                  const uint8_t* mask = nullptr) const override {
        if (!anatomicalImage_) return 0.0;
        
        ensureNormalsComputed(dims, mask);
        
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        
        double E = 0.0;
        for (size_t id = 0; id < N; ++id) {
            if (mask && !mask[id]) continue;
            
            int z = static_cast<int>(id / (Nx * Ny));
            int y = static_cast<int>((id - z * Nx * Ny) / Nx);
            int x = static_cast<int>(id - z * Nx * Ny - y * Nx);
            
            // 计算f的梯度 (中心差分)
            double gfx = computeGrad(f, x, y, z, 0, dims, mask);
            double gfy = computeGrad(f, x, y, z, 1, dims, mask);
            double gfz = computeGrad(f, x, y, z, 2, dims, mask);
            
            // 投影: J = P * ∇f = ∇f - (∇f · n)n
            double nx = normalsX_[id], ny = normalsY_[id], nz = normalsZ_[id];
            double dot = gfx * nx + gfy * ny + gfz * nz;
            double Jx = gfx - dot * nx;
            double Jy = gfy - dot * ny;
            double Jz = gfz - dot * nz;
            
            E += Jx * Jx + Jy * Jy + Jz * Jz;
        }
        return beta_ * eta_ * E;
    }
    
    void gradient(const double* f, double* grad, const Dims3D& dims,
                  const uint8_t* mask = nullptr, int threads = 1) const override {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        
        std::fill(grad, grad + N, 0.0);
        
        if (!anatomicalImage_) return;
        
        ensureNormalsComputed(dims, mask);
        
        // 计算通量 J = P * ∇f
        std::vector<float> Jx(N), Jy(N), Jz(N);
        
        omp_set_num_threads(threads);
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t id = 0; id < static_cast<ptrdiff_t>(N); ++id) {
            if (mask && !mask[id]) {
                Jx[id] = 0; Jy[id] = 0; Jz[id] = 0;
                continue;
            }
            
            int z = static_cast<int>(id / (Nx * Ny));
            int y = static_cast<int>((id - z * Nx * Ny) / Nx);
            int x = static_cast<int>(id - z * Nx * Ny - y * Nx);
            
            double gfx = computeGrad(f, x, y, z, 0, dims, mask);
            double gfy = computeGrad(f, x, y, z, 1, dims, mask);
            double gfz = computeGrad(f, x, y, z, 2, dims, mask);
            
            double nx = normalsX_[id], ny = normalsY_[id], nz = normalsZ_[id];
            double dot = gfx * nx + gfy * ny + gfz * nz;
            
            Jx[id] = static_cast<float>(gfx - dot * nx);
            Jy[id] = static_cast<float>(gfy - dot * ny);
            Jz[id] = static_cast<float>(gfz - dot * nz);
        }
        
        // 计算散度: ∇E = -div(J)
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t z = 0; z < Nz; ++z) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    size_t id = static_cast<size_t>(z) * Ny * Nx + y * Nx + x;
                    if (mask && !mask[id]) continue;
                    
                    double dJx_dx = computeDiv(Jx.data(), x, y, static_cast<int>(z), 0, dims);
                    double dJy_dy = computeDiv(Jy.data(), x, y, static_cast<int>(z), 1, dims);
                    double dJz_dz = computeDiv(Jz.data(), x, y, static_cast<int>(z), 2, dims);
                    
                    grad[id] = -beta_ * eta_ * (dJx_dx + dJy_dy + dJz_dz);
                }
            }
        }
    }
    
private:
    double eta_;
    const double* anatomicalImage_ = nullptr;
    size_t anatomicalSize_ = 0;
    
    mutable bool precomputed_ = false;
    mutable std::vector<float> normalsX_, normalsY_, normalsZ_;
    mutable Dims3D cachedDims_{0, 0, 0};
    
    void ensureNormalsComputed(const Dims3D& dims, const uint8_t* mask) const {
        if (precomputed_ && 
            dims.nx == cachedDims_.nx && 
            dims.ny == cachedDims_.ny && 
            dims.nz == cachedDims_.nz) {
            return;
        }
        
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        const size_t N = static_cast<size_t>(Nx) * Ny * Nz;
        
        normalsX_.assign(N, 0.0f);
        normalsY_.assign(N, 0.0f);
        normalsZ_.assign(N, 0.0f);
        
        for (size_t id = 0; id < N; ++id) {
            if (mask && !mask[id]) continue;
            
            int z = static_cast<int>(id / (Nx * Ny));
            int y = static_cast<int>((id - z * Nx * Ny) / Nx);
            int x = static_cast<int>(id - z * Nx * Ny - y * Nx);
            
            double gax = computeGrad(anatomicalImage_, x, y, z, 0, dims, nullptr);
            double gay = computeGrad(anatomicalImage_, x, y, z, 1, dims, nullptr);
            double gaz = computeGrad(anatomicalImage_, x, y, z, 2, dims, nullptr);
            
            double len = std::sqrt(gax * gax + gay * gay + gaz * gaz) + 1e-12;
            normalsX_[id] = static_cast<float>(gax / len);
            normalsY_[id] = static_cast<float>(gay / len);
            normalsZ_[id] = static_cast<float>(gaz / len);
        }
        
        cachedDims_ = dims;
        precomputed_ = true;
    }
    
    inline double computeGrad(const double* img, int x, int y, int z, int axis,
                               const Dims3D& dims, const uint8_t* /*mask*/) const {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        size_t id = static_cast<size_t>(z) * Ny * Nx + y * Nx + x;
        
        if (axis == 0) { // x方向
            double sub = (x > 0) ? img[id - 1] : img[id];
            double add = (x < Nx - 1) ? img[id + 1] : img[id];
            return (add - sub) * 0.5;
        } else if (axis == 1) { // y方向
            double sub = (y > 0) ? img[id - Nx] : img[id];
            double add = (y < Ny - 1) ? img[id + Nx] : img[id];
            return (add - sub) * 0.5;
        } else { // z方向
            double sub = (z > 0) ? img[id - Nx * Ny] : img[id];
            double add = (z < Nz - 1) ? img[id + Nx * Ny] : img[id];
            return (add - sub) * 0.5;
        }
    }
    
    inline double computeDiv(const float* J, int x, int y, int z, int axis,
                              const Dims3D& dims) const {
        const int Nx = dims.nx, Ny = dims.ny, Nz = dims.nz;
        size_t id = static_cast<size_t>(z) * Ny * Nx + y * Nx + x;
        
        if (axis == 0) {
            if (x > 0 && x < Nx - 1) return (J[id + 1] - J[id - 1]) * 0.5;
            if (x == 0 && Nx > 1) return J[id + 1] - J[id];
            if (x == Nx - 1 && Nx > 1) return J[id] - J[id - 1];
            return 0.0;
        } else if (axis == 1) {
            if (y > 0 && y < Ny - 1) return (J[id + Nx] - J[id - Nx]) * 0.5;
            if (y == 0 && Ny > 1) return J[id + Nx] - J[id];
            if (y == Ny - 1 && Ny > 1) return J[id] - J[id - Nx];
            return 0.0;
        } else {
            if (z > 0 && z < Nz - 1) return (J[id + Nx*Ny] - J[id - Nx*Ny]) * 0.5;
            if (z == 0 && Nz > 1) return J[id + Nx*Ny] - J[id];
            if (z == Nz - 1 && Nz > 1) return J[id] - J[id - Nx*Ny];
            return 0.0;
        }
    }
};

} // namespace regularizer
} // namespace spect
