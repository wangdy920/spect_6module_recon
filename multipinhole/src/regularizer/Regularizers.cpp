#include "regularizer/RegularizerCompat.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>

namespace spect::regularizer {
namespace {

void validateInputs(const std::vector<double> &f, int nx, int ny, int nz,
                    const RegConf &reg, const std::vector<uint8_t> *mask) {
  const size_t n = static_cast<size_t>(nx) * ny * nz;
  if (nx <= 0 || ny <= 0 || nz <= 0 || f.size() != n)
    throw std::runtime_error("Regularizer image size mismatch");
  if (mask && mask->size() != n)
    throw std::runtime_error("Regularizer mask size mismatch");
  if (reg.ct_image && reg.ct_image->size() != n)
    throw std::runtime_error("Registered CT image size mismatch");
  if (reg.ct_image && !(reg.ct_sigma > 0.0))
    throw std::runtime_error("CT regularization sigma must be positive");
  if (reg.ct_weight_floor < 0.0 || reg.ct_weight_floor > 1.0)
    throw std::runtime_error("CT regularization weight floor must be in [0,1]");
}

inline double ctWeight(const RegConf &reg, size_t a, size_t b) {
  if (!reg.ct_image)
    return 1.0;
  const double d = ((*reg.ct_image)[a] - (*reg.ct_image)[b]) / reg.ct_sigma;
  return reg.ct_weight_floor +
         (1.0 - reg.ct_weight_floor) * std::exp(-0.5 * d * d);
}

void gradTV(const std::vector<double> &f, int nx, int ny, int nz,
            const RegConf &reg, const std::vector<uint8_t> *mask,
            std::vector<double> &grad, int threads) {
  const size_t n = f.size();
  grad.assign(n, 0.0);
  const double eps = std::max(reg.tv_eps, 1e-12);

  struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
  };
  auto masked = [&](size_t i) { return mask && (*mask)[i] == 0; };
  auto p = [&](int x, int y, int z) {
    const size_t i = IDX(x, y, z, nx, ny);
    Vec3 out;
    if (masked(i))
      return out;
    double mag2 = eps * eps;
    auto edge = [&](int xx, int yy, int zz, double &component) {
      if (!inb(xx, yy, zz, nx, ny, nz))
        return;
      const size_t j = IDX(xx, yy, zz, nx, ny);
      if (reg.skip_outside_mask && masked(j))
        return;
      const double w = ctWeight(reg, i, j);
      const double d = f[j] - f[i];
      component = w * d;
      mag2 += w * d * d;
    };
    edge(x + 1, y, z, out.x);
    edge(x, y + 1, z, out.y);
    edge(x, y, z + 1, out.z);
    const double invMag = 1.0 / std::sqrt(mag2);
    out.x *= invMag;
    out.y *= invMag;
    out.z *= invMag;
    return out;
  };

  omp_set_num_threads(std::max(1, threads));
#pragma omp parallel for schedule(static)
  for (ptrdiff_t z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const size_t i = IDX(x, y, static_cast<int>(z), nx, ny);
        if (masked(i))
          continue;
        const Vec3 here = p(x, y, static_cast<int>(z));
        double div = here.x + here.y + here.z;
        if (x > 0 && (!reg.skip_outside_mask || !masked(i - 1)))
          div -= p(x - 1, y, static_cast<int>(z)).x;
        if (y > 0 && (!reg.skip_outside_mask || !masked(i - nx)))
          div -= p(x, y - 1, static_cast<int>(z)).y;
        const size_t plane = static_cast<size_t>(nx) * ny;
        if (z > 0 && (!reg.skip_outside_mask || !masked(i - plane)))
          div -= p(x, y, static_cast<int>(z) - 1).z;
        grad[i] = -div;
      }
    }
  }
}

void gradRDP(const std::vector<double> &f, int nx, int ny, int nz,
             const RegConf &reg, const std::vector<uint8_t> *mask,
             std::vector<double> &grad, int threads) {
  grad.assign(f.size(), 0.0);
  const double eps = std::max(reg.delta, 1e-6);
  auto masked = [&](size_t i) { return mask && (*mask)[i] == 0; };

  omp_set_num_threads(std::max(1, threads));
#pragma omp parallel for schedule(static)
  for (ptrdiff_t z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const size_t i = IDX(x, y, static_cast<int>(z), nx, ny);
        if (masked(i))
          continue;
        const double fi = f[i];
        double g = 0.0;
        auto add = [&](int xx, int yy, int zz) {
          if (!inb(xx, yy, zz, nx, ny, nz))
            return;
          const size_t j = IDX(xx, yy, zz, nx, ny);
          if (reg.skip_outside_mask && masked(j))
            return;
          const double diff = fi - f[j];
          const double absDiff = std::abs(diff);
          const double sign = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
          const double denom =
              std::max(fi + f[j] + reg.gamma * absDiff + 2.0 * eps, 1e-9);
          const double numerator =
              2.0 * diff * denom - diff * diff * (1.0 + reg.gamma * sign);
          g += ctWeight(reg, i, j) * numerator / (denom * denom);
        };
        add(x - 1, y, static_cast<int>(z));
        add(x + 1, y, static_cast<int>(z));
        add(x, y - 1, static_cast<int>(z));
        add(x, y + 1, static_cast<int>(z));
        add(x, y, static_cast<int>(z) - 1);
        add(x, y, static_cast<int>(z) + 1);
        grad[i] = g;
      }
    }
  }
}

} // namespace

void compute_reg_gradient(const std::vector<double> &f, int nx, int ny, int nz,
                          const RegConf &reg,
                          const std::vector<uint8_t> *mask,
                          std::vector<double> &grad, int threads) {
  validateInputs(f, nx, ny, nz, reg, mask);
  if (reg.type == RegType::TVIsotropic)
    gradTV(f, nx, ny, nz, reg, mask, grad, threads);
  else if (reg.type == RegType::RDP)
    gradRDP(f, nx, ny, nz, reg, mask, grad, threads);
  else
    grad.assign(f.size(), 0.0);
}

void osl_update_inplace(std::vector<double> &f, const std::vector<double> &q,
                        const std::vector<double> &d,
                        const std::vector<double> &grad, double beta,
                        const std::vector<uint8_t> *mask, bool nonneg,
                        int threads) {
  const double eps = 1e-12;
  omp_set_num_threads(std::max(1, threads));
#pragma omp parallel for schedule(static)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(f.size()); ++j) {
    const size_t i = static_cast<size_t>(j);
    if (mask && (*mask)[i] == 0) {
      f[i] = 0.0;
      continue;
    }
    double denom = d[i] + beta * grad[i] / std::max(f[i], eps);
    if (!(denom > 0.0))
      denom = d[i] + beta * std::abs(grad[i]) / std::max(f[i], eps) + eps;
    f[i] *= q[i] / denom;
    if (nonneg && f[i] < 0.0)
      f[i] = 0.0;
  }
}

double compute_reg_energy_raw(const std::vector<double> &f, int nx, int ny,
                              int nz, const RegConf &reg,
                              const std::vector<uint8_t> *mask) {
  validateInputs(f, nx, ny, nz, reg, mask);
  if (reg.type == RegType::None)
    return 0.0;

  const double eps = std::max(reg.delta, 1e-6);
  double energy = 0.0;
  auto masked = [&](size_t i) { return mask && (*mask)[i] == 0; };
#pragma omp parallel for reduction(+ : energy) schedule(static)
  for (ptrdiff_t z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const size_t i = IDX(x, y, static_cast<int>(z), nx, ny);
        if (masked(i))
          continue;
        if (reg.type == RegType::TVIsotropic) {
          double mag2 = reg.tv_eps * reg.tv_eps;
          auto add = [&](int xx, int yy, int zz) {
            if (!inb(xx, yy, zz, nx, ny, nz))
              return;
            const size_t j = IDX(xx, yy, zz, nx, ny);
            if (reg.skip_outside_mask && masked(j))
              return;
            const double diff = f[j] - f[i];
            mag2 += ctWeight(reg, i, j) * diff * diff;
          };
          add(x + 1, y, static_cast<int>(z));
          add(x, y + 1, static_cast<int>(z));
          add(x, y, static_cast<int>(z) + 1);
          energy += std::sqrt(mag2);
        } else if (reg.type == RegType::RDP) {
          auto add = [&](int xx, int yy, int zz) {
            if (!inb(xx, yy, zz, nx, ny, nz))
              return;
            const size_t j = IDX(xx, yy, zz, nx, ny);
            if (reg.skip_outside_mask && masked(j))
              return;
            const double diff = f[i] - f[j];
            const double denom = std::max(
                f[i] + f[j] + reg.gamma * std::abs(diff) + 2.0 * eps, eps);
            energy += ctWeight(reg, i, j) * diff * diff / denom;
          };
          add(x + 1, y, static_cast<int>(z));
          add(x, y + 1, static_cast<int>(z));
          add(x, y, static_cast<int>(z) + 1);
        }
      }
    }
  }
  return energy;
}

double compute_reg_energy(const std::vector<double> &f, int nx, int ny, int nz,
                          const RegConf &reg,
                          const std::vector<uint8_t> *mask) {
  if (reg.type == RegType::None || reg.beta <= 0.0)
    return 0.0;
  return reg.beta * compute_reg_energy_raw(f, nx, ny, nz, reg, mask);
}

} // namespace spect::regularizer
