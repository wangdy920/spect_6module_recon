/**
 * @file TVRegularizer.h
 * @brief Total Variation正则化器
 * 
 * 【T4.2】独立正则化器类示例
 */

#pragma once

#include "RegularizerBase.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spect {
namespace regularizer {

/**
 * @brief 各向同性Total Variation正则化器
 * 
 * R(f) = Σ_i √(|∇f_i|² + ε²)
 * ∇R(f)_i = Σ_j (f_i - f_j) / √(|∇f_i|² + ε²)
 */
class TVIsotropicRegularizer : public RegularizerBase {
public:
    explicit TVIsotropicRegularizer(double eps = 1e-6) 
        : RegularizerBase(1.0), eps_(eps) {}
    
    std::string name() const override { return "TVIsotropic"; }
    
    void setEpsilon(double eps) { eps_ = eps; }
    double epsilon() const { return eps_; }
    
    double energy(const double* f, 
                  const Dims3D& dims,
                  const uint8_t* mask = nullptr) const override {
        const int Nx = dims.nx;
        const int Ny = dims.ny;
        const int Nz = dims.nz;
        const double eps2 = eps_ * eps_;
        double total = 0.0;
        
        for (int z = 0; z < Nz; ++z) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    const int idx = x + y * Nx + z * Nx * Ny;
                    if (mask && !mask[idx]) continue;
                    
                    const double fc = f[idx];
                    double sum_sq = 0.0;
                    
                    // 6邻域梯度平方和
                    if (x > 0) {
                        double d = fc - f[idx - 1];
                        sum_sq += d * d;
                    }
                    if (x < Nx - 1) {
                        double d = fc - f[idx + 1];
                        sum_sq += d * d;
                    }
                    if (y > 0) {
                        double d = fc - f[idx - Nx];
                        sum_sq += d * d;
                    }
                    if (y < Ny - 1) {
                        double d = fc - f[idx + Nx];
                        sum_sq += d * d;
                    }
                    if (z > 0) {
                        double d = fc - f[idx - Nx * Ny];
                        sum_sq += d * d;
                    }
                    if (z < Nz - 1) {
                        double d = fc - f[idx + Nx * Ny];
                        sum_sq += d * d;
                    }
                    
                    total += std::sqrt(sum_sq + eps2);
                }
            }
        }
        
        return beta_ * total;
    }
    
    void gradient(const double* f, 
                  double* grad,
                  const Dims3D& dims,
                  const uint8_t* mask = nullptr,
                  int threads = 1) const override {
        const int Nx = dims.nx;
        const int Ny = dims.ny;
        const int Nz = dims.nz;
        const int stride_y = Nx;
        const int stride_z = Nx * Ny;
        const double eps2 = eps_ * eps_;
        
#ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(static)
#else
        (void)threads;
#endif
        for (int z = 0; z < Nz; ++z) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    const int idx = x + y * stride_y + z * stride_z;
                    if (mask && !mask[idx]) {
                        grad[idx] = 0.0;
                        continue;
                    }
                    
                    const double fc = f[idx];
                    double sum_sq = 0.0;
                    double sum_diff = 0.0;
                    
                    // 计算梯度幅值和差分和
                    if (x > 0) {
                        double d = fc - f[idx - 1];
                        sum_sq += d * d;
                        sum_diff += d;
                    }
                    if (x < Nx - 1) {
                        double d = fc - f[idx + 1];
                        sum_sq += d * d;
                        sum_diff += d;
                    }
                    if (y > 0) {
                        double d = fc - f[idx - stride_y];
                        sum_sq += d * d;
                        sum_diff += d;
                    }
                    if (y < Ny - 1) {
                        double d = fc - f[idx + stride_y];
                        sum_sq += d * d;
                        sum_diff += d;
                    }
                    if (z > 0) {
                        double d = fc - f[idx - stride_z];
                        sum_sq += d * d;
                        sum_diff += d;
                    }
                    if (z < Nz - 1) {
                        double d = fc - f[idx + stride_z];
                        sum_sq += d * d;
                        sum_diff += d;
                    }
                    
                    // ∂R/∂f_i = Σ_j (f_i - f_j) / |∇f|
                    const double norm = std::sqrt(sum_sq + eps2);
                    grad[idx] = beta_ * sum_diff / norm;
                }
            }
        }
    }
    
private:
    double eps_;
};

/**
 * @brief Huber TV正则化器
 * 
 * 在|∇f|<δ时使用L2，否则使用L1，更平滑。
 */
class TVHuberRegularizer : public RegularizerBase {
public:
    TVHuberRegularizer(double delta = 0.01, double eps = 1e-6)
        : RegularizerBase(1.0), delta_(delta), eps_(eps) {}
    
    std::string name() const override { return "TVHuber"; }
    
    double energy(const double* f, 
                  const Dims3D& dims,
                  const uint8_t* mask = nullptr) const override {
        const int Nx = dims.nx;
        const int Ny = dims.ny;
        const int Nz = dims.nz;
        double total = 0.0;
        
        for (int z = 0; z < Nz; ++z) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    const int idx = x + y * Nx + z * Nx * Ny;
                    if (mask && !mask[idx]) continue;
                    
                    const double fc = f[idx];
                    double sum_sq = 0.0;
                    
                    if (x > 0) { double d = fc - f[idx - 1]; sum_sq += d * d; }
                    if (x < Nx - 1) { double d = fc - f[idx + 1]; sum_sq += d * d; }
                    if (y > 0) { double d = fc - f[idx - Nx]; sum_sq += d * d; }
                    if (y < Ny - 1) { double d = fc - f[idx + Nx]; sum_sq += d * d; }
                    if (z > 0) { double d = fc - f[idx - Nx * Ny]; sum_sq += d * d; }
                    if (z < Nz - 1) { double d = fc - f[idx + Nx * Ny]; sum_sq += d * d; }
                    
                    const double norm = std::sqrt(sum_sq);
                    // Huber: δ² * (√(1 + (|∇|/δ)²) - 1)
                    total += delta_ * delta_ * (std::sqrt(1.0 + sum_sq / (delta_ * delta_)) - 1.0);
                }
            }
        }
        
        return beta_ * total;
    }
    
    void gradient(const double* f, 
                  double* grad,
                  const Dims3D& dims,
                  const uint8_t* mask = nullptr,
                  int threads = 1) const override {
        const int Nx = dims.nx;
        const int Ny = dims.ny;
        const int Nz = dims.nz;
        const int stride_y = Nx;
        const int stride_z = Nx * Ny;
        
#ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(static)
#else
        (void)threads;
#endif
        for (int z = 0; z < Nz; ++z) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    const int idx = x + y * stride_y + z * stride_z;
                    if (mask && !mask[idx]) {
                        grad[idx] = 0.0;
                        continue;
                    }
                    
                    const double fc = f[idx];
                    double sum_sq = 0.0;
                    double sum_diff = 0.0;
                    
                    if (x > 0) { double d = fc - f[idx - 1]; sum_sq += d * d; sum_diff += d; }
                    if (x < Nx - 1) { double d = fc - f[idx + 1]; sum_sq += d * d; sum_diff += d; }
                    if (y > 0) { double d = fc - f[idx - stride_y]; sum_sq += d * d; sum_diff += d; }
                    if (y < Ny - 1) { double d = fc - f[idx + stride_y]; sum_sq += d * d; sum_diff += d; }
                    if (z > 0) { double d = fc - f[idx - stride_z]; sum_sq += d * d; sum_diff += d; }
                    if (z < Nz - 1) { double d = fc - f[idx + stride_z]; sum_sq += d * d; sum_diff += d; }
                    
                    // Huber gradient: Σ(f_i - f_j) / √(δ² + |∇f|²)
                    const double denom = std::sqrt(delta_ * delta_ + sum_sq);
                    grad[idx] = beta_ * sum_diff / denom;
                }
            }
        }
    }
    
private:
    double delta_;
    double eps_;
};

} // namespace regularizer
} // namespace spect
