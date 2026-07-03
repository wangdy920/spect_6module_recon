#pragma once

/// @file ADMMOptimizer.h
/// @brief ADMM (Alternating Direction Method of Multipliers) optimizer.
///
/// Phase 1: Supports Quadratic/RDP regularizers via gradient-step proximal.
/// Phase 2: Supports TV (isotropic) via Chambolle-Pock primal-dual proximal.
/// Adaptive ρ (penalty parameter) based on Boyd (2011) / CASToR ADMMLim.
///
/// f-update: EM-ADMM step reusing OSEM_step numerator computation.
/// u-update: prox_{βR/ρ}(f + d).
///   - TV: Chambolle-Pock iteration (GPU or CPU).
///   - RDP/Quad: gradient-step (f+d) − (β/ρ)∇R(f+d).
/// d-update: d += f − u.

#include "app/IOptimizer.h"
#include "app/OSEMReconstructor.h"
#include "cuda/VolumeOpsCUDA.h"
#include "regularizer/RegConfig.h"
#include "regularizer/RegularizerCompat.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace spect {

/// Configuration for the ADMM optimizer.
struct ADMMConfig {
  double rho = 1.0;                        ///< Augmented Lagrangian penalty ρ
  RegConf reg;                             ///< Regularizer config for u-update
  double beta = 0.0;                       ///< Regularization strength β
  const std::vector<uint8_t> *mask = nullptr; ///< Mask for regularizer gradient
  int threads = 1;

  // TV proximal parameters
  int tvInnerIters = 20;                   ///< Chambolle-Pock inner iterations

  // Adaptive ρ parameters (Boyd 2011 / CASToR ADMMLim)
  bool adaptive = true;                    ///< Enable adaptive ρ adjustment
  double admmMu = 2.0;                     ///< Primal/dual residual balance factor
  double admmTau = 2.0;                    ///< ρ scaling factor
  double admmTauMax = 100.0;               ///< Maximum cumulative ρ scaling
};

/// ADMM optimizer implementing the IOptimizer interface.
///
/// Algorithm per outer iteration (= one MainProgram iteration):
///   For each subset s:
///     1. Compute EM numerator: numer = A_s^T(y_s / A_s f)  [via OSEM_step]
///     2. ADMM f-update: f = (numer + ρ·z) / (denom + ρ),  z = u − d
///   After all subsets:
///     3. u-update: u = prox_{βR/ρ}(f + d)
///     4. d-update: d += f − u
///     5. Adaptive ρ: adjust based on primal/dual residual balance
class ADMMOptimizer : public IOptimizer {
public:
  explicit ADMMOptimizer(ADMMConfig cfg) : cfg_(std::move(cfg)) {}

  void step(const StepParams &p) override {
    const size_t N = p.image->size();
    if (!initialized_) {
      u_ = *p.image;
      d_.assign(N, 0.0);
      gradBuf_.resize(N, 0.0);
      u_prev_.assign(N, 0.0);
      rhoInitial_ = cfg_.rho;
      initialized_ = true;
    }

    // --- Compute EM numerator via OSEM_step with bsrem_alpha=0 ---
    std::vector<double> numer;
    if (p.matrixView) {
      OSEM_step(*p.matrixView, *p.subsetRows, *p.projections, *p.image,
                *p.denom, p.Nx, p.Ny, p.Nz, *p.opts, p.psfKernels,
                nullptr, 0.0, &numer, 0.0);
    } else {
      OSEM_step(*p.matrixCSR, *p.subsetRows, *p.projections, *p.image,
                *p.denom, p.Nx, p.Ny, p.Nz, *p.opts, p.psfKernels,
                nullptr, 0.0, &numer, 0.0);
    }

    // --- ADMM f-update: f = (numer + ρ·z) / (denom + ρ) ---
    auto &f = *p.image;
    const auto &denom = *p.denom;
    const double rho = cfg_.rho;
    const int threads = p.opts->threads;

#pragma omp parallel for schedule(static) num_threads(threads)
    for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(N); ++j) {
      const auto jj = static_cast<size_t>(j);
      if (p.opts->mask && !(*p.opts->mask)[jj]) {
        f[jj] = 0.0;
        continue;
      }
      const double dj = denom[jj];
      if (!(dj > 0.0)) {
        f[jj] = 0.0;
        continue;
      }
      const double z_j = u_[jj] - d_[jj];
      double f_new = (numer[jj] + rho * z_j) / (dj + rho);
      if (p.opts->nonneg && f_new < 0.0)
        f_new = 0.0;
      f[jj] = f_new;
    }
  }

  /// u-update (proximal), d-update, and adaptive ρ after all subsets complete.
  void postIteration(std::vector<double> &image, int iteration) override {
    const size_t N = image.size();
    if (N != u_.size())
      return;

    const double rho = cfg_.rho;
    const double beta = cfg_.beta;
    const int threads = cfg_.threads;

    // Save u_prev for dual residual computation (before u-update)
    u_prev_ = u_;

    // v = f + d
    std::vector<double> v(N);
    for (size_t j = 0; j < N; ++j)
      v[j] = image[j] + d_[j];

    // --- u-update: proximal operator ---
    if (beta > 0.0 && cfg_.reg.type != RegType::None) {
      const double lambda = beta / rho; // effective TV/reg weight

      if (cfg_.reg.type == RegType::TVIsotropic) {
        // TV proximal via Chambolle-Pock (GPU then CPU fallback)
        if (!tryComputeTVProxCUDA(v, lambda)) {
          computeTVProxCPU(v, lambda);
        }
      } else {
        // Gradient-step proximal for RDP/Quadratic/etc.
        std::fill(gradBuf_.begin(), gradBuf_.end(), 0.0);
        if (!tryComputeRegGradCUDA(v)) {
          spect::regularizer::compute_reg_gradient(
              v, Nx_, Ny_, Nz_, cfg_.reg, cfg_.mask, gradBuf_, threads);
        }

        const double scale = lambda;
#pragma omp parallel for schedule(static) num_threads(threads)
        for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(N); ++j) {
          const auto jj = static_cast<size_t>(j);
          double u_new = v[jj] - scale * gradBuf_[jj];
          if (u_new < 0.0)
            u_new = 0.0;
          u_[jj] = u_new;
        }
      }
    } else {
      // No regularization: u = v (identity proximal)
      u_ = v;
    }

    // --- d-update: d += f − u ---
#pragma omp parallel for schedule(static) num_threads(threads)
    for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(N); ++j) {
      const auto jj = static_cast<size_t>(j);
      d_[jj] += image[jj] - u_[jj];
    }

    // --- Adaptive ρ (Boyd 2011 / CASToR ADMMLim) ---
    if (cfg_.adaptive) {
      adaptRho(image, iteration);
    }
  }

  std::string name() const override { return "ADMM"; }

  /// Store image dimensions (called from MainProgram before iteration loop).
  void setDimensions(int Nx, int Ny, int Nz) {
    Nx_ = Nx;
    Ny_ = Ny;
    Nz_ = Nz;
  }

private:
  ADMMConfig cfg_;
  std::vector<double> u_;       ///< Auxiliary (primal) variable
  std::vector<double> d_;       ///< Scaled dual variable
  std::vector<double> u_prev_;  ///< Previous u for dual residual
  std::vector<double> gradBuf_; ///< Scratch for regularizer gradient
  int Nx_ = 0, Ny_ = 0, Nz_ = 0;
  bool initialized_ = false;
  double rhoInitial_ = 1.0;    ///< Initial ρ for clamping adaptive range

  /// Try CUDA path for regularizer gradient at v.
  bool tryComputeRegGradCUDA(const std::vector<double> &v) {
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
    return cuda::computeRegGradientCUDA(v, Nx_, Ny_, Nz_, cfg_.reg, cfg_.mask,
                                        gradBuf_);
#else
    (void)v;
    return false;
#endif
  }

  /// Try CUDA path for TV proximal (Chambolle-Pock).
  bool tryComputeTVProxCUDA(const std::vector<double> &v, double lambda) {
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
    return cuda::computeTVProxCUDA(v, Nx_, Ny_, Nz_, lambda,
                                   cfg_.tvInnerIters, u_);
#else
    (void)v;
    (void)lambda;
    return false;
#endif
  }

  /// CPU fallback for TV proximal via Chambolle-Pock primal-dual.
  void computeTVProxCPU(const std::vector<double> &v, double lambda) {
    const size_t N = v.size();
    const int Nx = Nx_, Ny = Ny_, Nz = Nz_;
    const size_t slice = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    const int threads = cfg_.threads;
    const int iters = cfg_.tvInnerIters;
    const double sigma = 1.0 / std::sqrt(12.0);
    const double tau = 1.0 / std::sqrt(12.0);
    const double theta = 1.0;

    // Initialize: u = v, u_bar = v, p = 0
    u_ = v;
    std::vector<double> u_bar = v;
    std::vector<double> px(N, 0.0), py(N, 0.0), pz(N, 0.0);

    for (int k = 0; k < iters; ++k) {
      // Dual update: p = proj_{||·||≤lambda}(p + sigma * grad(u_bar))
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(N); ++j) {
        const auto jj = static_cast<size_t>(j);
        const int x = static_cast<int>(jj % static_cast<size_t>(Nx));
        const int y = static_cast<int>((jj / static_cast<size_t>(Nx)) %
                                        static_cast<size_t>(Ny));

        const double u0 = u_bar[jj];
        double gx = 0.0, gy = 0.0, gz = 0.0;
        if (x + 1 < Nx) gx = u_bar[jj + 1] - u0;
        if (y + 1 < Ny) gy = u_bar[jj + static_cast<size_t>(Nx)] - u0;
        if (jj + slice < N) gz = u_bar[jj + slice] - u0;

        double qx = px[jj] + sigma * gx;
        double qy = py[jj] + sigma * gy;
        double qz = pz[jj] + sigma * gz;

        const double norm = std::sqrt(qx * qx + qy * qy + qz * qz);
        if (norm > lambda) {
          const double s = lambda / norm;
          qx *= s; qy *= s; qz *= s;
        }
        px[jj] = qx; py[jj] = qy; pz[jj] = qz;
      }

      // Primal update + extrapolation
#pragma omp parallel for schedule(static) num_threads(threads)
      for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(N); ++j) {
        const auto jj = static_cast<size_t>(j);
        const int x = static_cast<int>(jj % static_cast<size_t>(Nx));
        const int y = static_cast<int>((jj / static_cast<size_t>(Nx)) %
                                        static_cast<size_t>(Ny));

        double div_val = px[jj] + py[jj] + pz[jj];
        if (x > 0) div_val -= px[jj - 1];
        if (y > 0) div_val -= py[jj - static_cast<size_t>(Nx)];
        if (jj >= slice) div_val -= pz[jj - slice];

        const double u_old = u_[jj];
        double u_new =
            (tau * v[jj] + u_old + tau * div_val) / (1.0 + tau);
        if (u_new < 0.0) u_new = 0.0;

        u_[jj] = u_new;
        u_bar[jj] = u_new + theta * (u_new - u_old);
      }
    }
  }

  /// Adaptive ρ adjustment based on primal/dual residual balance.
  /// Reference: Boyd et al. (2011) §3.4.1; CASToR ADMMLim default config.
  void adaptRho(const std::vector<double> &image, int iteration) {
    const size_t N = image.size();
    const int threads = cfg_.threads;

    // Primal residual: ||f - u||_2
    double primalSq = 0.0;
    // Dual residual: ρ * ||u^k - u^{k-1}||_2
    double dualSq = 0.0;

#pragma omp parallel for schedule(static) num_threads(threads) \
    reduction(+:primalSq, dualSq)
    for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(N); ++j) {
      const auto jj = static_cast<size_t>(j);
      const double r = image[jj] - u_[jj];
      primalSq += r * r;
      const double s = u_[jj] - u_prev_[jj];
      dualSq += s * s;
    }

    const double primalRes = std::sqrt(primalSq);
    const double dualRes = cfg_.rho * std::sqrt(dualSq);
    const double mu = cfg_.admmMu;
    const double tau = cfg_.admmTau;
    const double maxScale = cfg_.admmTauMax;

    if (primalRes > mu * dualRes &&
        cfg_.rho * tau <= rhoInitial_ * maxScale) {
      // Primal residual too large: increase ρ to enforce f ≈ u
      cfg_.rho *= tau;
      for (auto &d : d_) d /= tau; // rescale scaled dual variable
      std::cerr << "[ADMM] it=" << iteration
                << " rho increased to " << cfg_.rho
                << " (primal=" << primalRes << " dual=" << dualRes << ")\n";
    } else if (dualRes > mu * primalRes &&
               cfg_.rho / tau >= rhoInitial_ / maxScale) {
      // Dual residual too large: decrease ρ to improve dual convergence
      cfg_.rho /= tau;
      for (auto &d : d_) d *= tau;
      std::cerr << "[ADMM] it=" << iteration
                << " rho decreased to " << cfg_.rho
                << " (primal=" << primalRes << " dual=" << dualRes << ")\n";
    }
  }
};

} // namespace spect
