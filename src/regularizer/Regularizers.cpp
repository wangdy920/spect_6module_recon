#include "regularizer/RegularizerCompat.h"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace spect {
namespace regularizer {

static void grad_TV_Isotropic(const std::vector<double> &f, int Nx, int Ny,
                              int Nz, double eps,
                              const std::vector<uint8_t> *mask,
                              bool skip_outside_mask,
                              std::vector<double> &out_grad, int threads) {
  const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  out_grad.assign(N, 0.0);

  auto get = [&](int x, int y, int z) -> double {
    return f[IDX(x, y, z, Nx, Ny)];
  };
  auto masked = [&](int x, int y, int z) -> bool {
    return mask && !(*mask)[IDX(x, y, z, Nx, Ny)];
  };

  struct Vec3 {
    double x;
    double y;
    double z;
  };

  auto get_p = [&](int x, int y, int z) -> Vec3 {
    if (mask && (*mask)[IDX(x, y, z, Nx, Ny)] == 0)
      return {0.0, 0.0, 0.0};

    double gx = 0.0;
    double gy = 0.0;
    double gz = 0.0;
    if (x + 1 < Nx && (!skip_outside_mask || !masked(x + 1, y, z)))
      gx = get(x + 1, y, z) - get(x, y, z);
    if (y + 1 < Ny && (!skip_outside_mask || !masked(x, y + 1, z)))
      gy = get(x, y + 1, z) - get(x, y, z);
    if (z + 1 < Nz && (!skip_outside_mask || !masked(x, y, z + 1)))
      gz = get(x, y, z + 1) - get(x, y, z);

    const double mag = std::sqrt(gx * gx + gy * gy + gz * gz + eps * eps);
    return {gx / mag, gy / mag, gz / mag};
  };

  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (ptrdiff_t z = 0; z < Nz; ++z) {
    for (int y = 0; y < Ny; ++y) {
      for (int x = 0; x < Nx; ++x) {
        if (mask && (*mask)[IDX(x, y, static_cast<int>(z), Nx, Ny)] == 0)
          continue;

        const Vec3 p = get_p(x, y, static_cast<int>(z));
        double div = 0.0;

        if (x > 0 && (!skip_outside_mask || !masked(x - 1, y, static_cast<int>(z)))) {
          const Vec3 prev = get_p(x - 1, y, static_cast<int>(z));
          div += p.x - prev.x;
        } else {
          div += p.x;
        }

        if (y > 0 && (!skip_outside_mask || !masked(x, y - 1, static_cast<int>(z)))) {
          const Vec3 prev = get_p(x, y - 1, static_cast<int>(z));
          div += p.y - prev.y;
        } else {
          div += p.y;
        }

        if (z > 0 && (!skip_outside_mask || !masked(x, y, static_cast<int>(z) - 1))) {
          const Vec3 prev = get_p(x, y, static_cast<int>(z) - 1);
          div += p.z - prev.z;
        } else {
          div += p.z;
        }

        out_grad[IDX(x, y, static_cast<int>(z), Nx, Ny)] = -div;
      }
    }
  }
}

static void grad_TV_Huber(const double *f, int Nx, int Ny, int Nz,
                          std::vector<double> &grad, double delta,
                          const uint8_t *mask) {
  const int N = Nx * Ny * Nz;
  grad.assign(static_cast<size_t>(N), 0.0);
  int dx[6], dy[6], dz[6];
  stencil_offsets(Nx, Ny, Nz, dx, dy, dz);

#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static)
  for (ptrdiff_t idx = 0; idx < N; ++idx) {
    if (mask && !mask[idx])
      continue;

    const int z = static_cast<int>(idx) / (Nx * Ny);
    const int y = (static_cast<int>(idx) - z * Nx * Ny) / Nx;
    const int x = static_cast<int>(idx) - z * Nx * Ny - y * Nx;

    double dR = 0.0;
    for (int k = 0; k < 6; ++k) {
      const int xx = x + dx[k];
      const int yy = y + dy[k];
      const int zz = z + dz[k];
      if (!inb(xx, yy, zz, Nx, Ny, Nz))
        continue;
      const int j = zz * Ny * Nx + yy * Nx + xx;
      const double t = f[idx] - f[j];
      const double abs_t = std::fabs(t);
      dR += (abs_t <= delta) ? t : delta * ((t > 0.0) ? 1.0 : -1.0);
    }
    grad[static_cast<size_t>(idx)] = dR;
  }
}

static void grad_Tikhonov(const std::vector<double> &f, int Nx, int Ny, int Nz,
                          const std::vector<uint8_t> *mask,
                          bool skip_outside_mask,
                          std::vector<double> &out_grad, int threads) {
  const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  out_grad.assign(N, 0.0);

  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (ptrdiff_t z = 0; z < Nz; ++z) {
    for (int y = 0; y < Ny; ++y) {
      for (int x = 0; x < Nx; ++x) {
        const size_t id = IDX(x, y, static_cast<int>(z), Nx, Ny);
        if (mask && (*mask)[id] == 0)
          continue;

        const double fj = f[id];
        double g = 0.0;
        auto accum = [&](int xx, int yy, int zz) {
          if (!inb(xx, yy, zz, Nx, Ny, Nz))
            return;
          const size_t nid = IDX(xx, yy, zz, Nx, Ny);
          if (skip_outside_mask && mask && (*mask)[nid] == 0)
            return;
          g += fj - f[nid];
        };

        accum(x - 1, y, static_cast<int>(z));
        accum(x + 1, y, static_cast<int>(z));
        accum(x, y - 1, static_cast<int>(z));
        accum(x, y + 1, static_cast<int>(z));
        accum(x, y, static_cast<int>(z) - 1);
        accum(x, y, static_cast<int>(z) + 1);
        out_grad[id] = 2.0 * g;
      }
    }
  }
}

static void grad_MRP(const std::vector<double> &f, int Nx, int Ny, int Nz,
                     double eps, const std::vector<uint8_t> *mask,
                     bool skip_outside_mask, std::vector<double> &out_grad,
                     int threads) {
  const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  out_grad.assign(N, 0.0);
  const double eps_root = eps * eps;

  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (ptrdiff_t z = 0; z < Nz; ++z) {
    for (int y = 0; y < Ny; ++y) {
      for (int x = 0; x < Nx; ++x) {
        const size_t id = IDX(x, y, static_cast<int>(z), Nx, Ny);
        if (mask && (*mask)[id] == 0)
          continue;

        const double sj = std::sqrt(std::max(f[id], 0.0) + eps_root);
        double g = 0.0;
        auto accum = [&](int xx, int yy, int zz) {
          if (!inb(xx, yy, zz, Nx, Ny, Nz))
            return;
          const size_t nid = IDX(xx, yy, zz, Nx, Ny);
          if (skip_outside_mask && mask && (*mask)[nid] == 0)
            return;
          const double sk = std::sqrt(std::max(f[nid], 0.0) + eps_root);
          g += 1.0 - (sk / sj);
        };

        accum(x - 1, y, static_cast<int>(z));
        accum(x + 1, y, static_cast<int>(z));
        accum(x, y - 1, static_cast<int>(z));
        accum(x, y + 1, static_cast<int>(z));
        accum(x, y, static_cast<int>(z) - 1);
        accum(x, y, static_cast<int>(z) + 1);
        out_grad[id] = g;
      }
    }
  }
}

static void grad_RDP(const std::vector<double> &f, int Nx, int Ny, int Nz,
                     double delta, double gamma,
                     const std::vector<uint8_t> *mask,
                     bool skip_outside_mask,
                     std::vector<double> &out_grad, int threads) {
  const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  out_grad.assign(N, 0.0);
  const double eps = std::max(delta, 1e-6);

  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (ptrdiff_t z = 0; z < Nz; ++z) {
    for (int y = 0; y < Ny; ++y) {
      for (int x = 0; x < Nx; ++x) {
        const size_t id = IDX(x, y, static_cast<int>(z), Nx, Ny);
        if (mask && (*mask)[id] == 0)
          continue;

        const double fj = f[id];
        double g = 0.0;
        auto accum = [&](int xx, int yy, int zz) {
          if (!inb(xx, yy, zz, Nx, Ny, Nz))
            return;
          const size_t nid = IDX(xx, yy, zz, Nx, Ny);
          if (skip_outside_mask && mask && (*mask)[nid] == 0)
            return;

          const double fk = f[nid];
          const double diff = fj - fk;
          const double abs_diff = std::abs(diff);
          const double sign_diff = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
          double denom = fj + fk + gamma * abs_diff + 2.0 * eps;
          if (denom < 1e-9)
            denom = 1e-9;

          const double term_num = 2.0 * diff * denom - (diff * diff) * (1.0 + gamma * sign_diff);
          g += term_num / (denom * denom);
        };

        accum(x - 1, y, static_cast<int>(z));
        accum(x + 1, y, static_cast<int>(z));
        accum(x, y - 1, static_cast<int>(z));
        accum(x, y + 1, static_cast<int>(z));
        accum(x, y, static_cast<int>(z) - 1);
        accum(x, y, static_cast<int>(z) + 1);
        out_grad[id] = g;
      }
    }
  }
}

static void grad_LogCosh(const std::vector<double> &f, int Nx, int Ny, int Nz,
                         double delta, const std::vector<uint8_t> *mask,
                         bool skip_outside_mask,
                         std::vector<double> &out_grad, int threads) {
  const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  out_grad.assign(N, 0.0);
  const double eps = std::max(delta, 1e-6);
  const double inv_delta = 1.0 / eps;

  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (ptrdiff_t z = 0; z < Nz; ++z) {
    for (int y = 0; y < Ny; ++y) {
      for (int x = 0; x < Nx; ++x) {
        const size_t id = IDX(x, y, static_cast<int>(z), Nx, Ny);
        if (mask && (*mask)[id] == 0)
          continue;

        const double fj = f[id];
        double g = 0.0;
        auto accum = [&](int xx, int yy, int zz) {
          if (!inb(xx, yy, zz, Nx, Ny, Nz))
            return;
          const size_t nid = IDX(xx, yy, zz, Nx, Ny);
          if (skip_outside_mask && mask && (*mask)[nid] == 0)
            return;
          g += eps * std::tanh((fj - f[nid]) * inv_delta);
        };

        accum(x - 1, y, static_cast<int>(z));
        accum(x + 1, y, static_cast<int>(z));
        accum(x, y - 1, static_cast<int>(z));
        accum(x, y + 1, static_cast<int>(z));
        accum(x, y, static_cast<int>(z) - 1);
        accum(x, y, static_cast<int>(z) + 1);
        out_grad[id] = g;
      }
    }
  }
}

static void grad_NLM(const std::vector<double> &f, int Nx, int Ny, int Nz,
                     double h, int search_window, int patch_size,
                     const std::vector<uint8_t> *mask,
                     bool skip_outside_mask,
                     std::vector<double> &out_grad, int threads) {
  const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  out_grad.assign(N, 0.0);
  double h2 = h * h;
  if (h2 < 1e-8)
    h2 = 1e-8;

  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (ptrdiff_t z = 0; z < Nz; ++z) {
    for (int y = 0; y < Ny; ++y) {
      for (int x = 0; x < Nx; ++x) {
        const size_t id = IDX(x, y, static_cast<int>(z), Nx, Ny);
        if (mask && (*mask)[id] == 0)
          continue;

        double g = 0.0;
        const double val_i = f[id];
        for (int dz = -search_window; dz <= search_window; ++dz) {
          for (int dy = -search_window; dy <= search_window; ++dy) {
            for (int dx = -search_window; dx <= search_window; ++dx) {
              if (dx == 0 && dy == 0 && dz == 0)
                continue;

              const int nx = x + dx;
              const int ny = y + dy;
              const int nz = static_cast<int>(z) + dz;
              if (!inb(nx, ny, nz, Nx, Ny, Nz))
                continue;

              const size_t nid = IDX(nx, ny, nz, Nx, Ny);
              if (skip_outside_mask && mask && (*mask)[nid] == 0)
                continue;

              double dist = 0.0;
              int count = 0;
              for (int pz = -patch_size; pz <= patch_size; ++pz) {
                for (int py = -patch_size; py <= patch_size; ++py) {
                  for (int px = -patch_size; px <= patch_size; ++px) {
                    const int ix = x + px;
                    const int iy = y + py;
                    const int iz = static_cast<int>(z) + pz;
                    const int jx = nx + px;
                    const int jy = ny + py;
                    const int jz = nz + pz;
                    if (inb(ix, iy, iz, Nx, Ny, Nz) && inb(jx, jy, jz, Nx, Ny, Nz)) {
                      const double diff = f[IDX(ix, iy, iz, Nx, Ny)] - f[IDX(jx, jy, jz, Nx, Ny)];
                      dist += diff * diff;
                      ++count;
                    }
                  }
                }
              }
              if (count > 0)
                dist /= count;

              const double w = std::exp(-dist / h2);
              g += w * (val_i - f[nid]);
            }
          }
        }
        out_grad[id] = g;
      }
    }
  }
}

static void grad_Bowsher(const std::vector<double> &f, int Nx, int Ny, int Nz,
                         const RegConf &reg, const std::vector<uint8_t> *mask,
                         std::vector<double> &out_grad, int threads) {
  const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  out_grad.assign(N, 0.0);
  const auto &anat = *reg.anatomical_image;
  const int K = reg.bowsher_k;
  const bool skip = reg.skip_outside_mask;

  if (!reg.bowsher_precomputed) {
#pragma omp critical(BowsherCache)
    {
      if (!reg.bowsher_precomputed) {
        reg.bowsher_indices.resize(N * static_cast<size_t>(K), -1);
        for (ptrdiff_t id = 0; id < static_cast<ptrdiff_t>(N); ++id) {
          if (mask && (*mask)[static_cast<size_t>(id)] == 0)
            continue;

          const int z = static_cast<int>(id) / (Nx * Ny);
          const int y = (static_cast<int>(id) - z * Nx * Ny) / Nx;
          const int x = static_cast<int>(id) - z * Nx * Ny - y * Nx;

          struct Neighbor { double abs_diff; int nid; };
          Neighbor neighbors[26];
          int count = 0;
          const double val_anat = anat[static_cast<size_t>(id)];

          for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                const int nx = x + dx;
                const int ny = y + dy;
                const int nz = z + dz;
                if (!inb(nx, ny, nz, Nx, Ny, Nz))
                  continue;
                const size_t nid = IDX(nx, ny, nz, Nx, Ny);
                if (skip && mask && (*mask)[nid] == 0)
                  continue;
                neighbors[count++] = {std::abs(val_anat - anat[nid]), static_cast<int>(nid)};
              }
            }
          }

          const int k_eff = std::min(K, count);
          if (k_eff > 0) {
            std::partial_sort(neighbors, neighbors + k_eff, neighbors + count,
                              [](const Neighbor &a, const Neighbor &b) {
                                return a.abs_diff < b.abs_diff;
                              });
            for (int i = 0; i < k_eff; ++i)
              reg.bowsher_indices[static_cast<size_t>(id) * static_cast<size_t>(K) + static_cast<size_t>(i)] = neighbors[i].nid;
          }
        }
        reg.bowsher_precomputed = true;
      }
    }
  }

  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (ptrdiff_t id = 0; id < static_cast<ptrdiff_t>(N); ++id) {
    if (mask && (*mask)[static_cast<size_t>(id)] == 0)
      continue;
    double g = 0.0;
    const double val_f = f[static_cast<size_t>(id)];
    const size_t base = static_cast<size_t>(id) * static_cast<size_t>(K);
    for (int i = 0; i < K; ++i) {
      const int nid = reg.bowsher_indices[base + static_cast<size_t>(i)];
      if (nid < 0)
        break;
      g += val_f - f[static_cast<size_t>(nid)];
    }
    out_grad[static_cast<size_t>(id)] = g;
  }
}

static void grad_PLS(const std::vector<double> &f, int Nx, int Ny, int Nz,
                     const RegConf &reg, const std::vector<uint8_t> *mask,
                     bool, std::vector<double> &out_grad, int threads) {
  const size_t N = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  out_grad.assign(N, 0.0);
  const auto &anat = *reg.anatomical_image;

  if (!reg.pls_precomputed) {
#pragma omp critical(PLSCache)
    {
      if (!reg.pls_precomputed) {
        reg.pls_nx.assign(N, 0.0f);
        reg.pls_ny.assign(N, 0.0f);
        reg.pls_nz.assign(N, 0.0f);
#pragma omp parallel for schedule(static) num_threads(threads)
        for (ptrdiff_t id = 0; id < static_cast<ptrdiff_t>(N); ++id) {
          if (mask && (*mask)[static_cast<size_t>(id)] == 0)
            continue;

          const int z = static_cast<int>(id) / (Nx * Ny);
          const int y = (static_cast<int>(id) - z * Nx * Ny) / Nx;
          const int x = static_cast<int>(id) - z * Nx * Ny - y * Nx;

          const double ga_x = (((x < Nx - 1) ? anat[static_cast<size_t>(id) + 1] : anat[static_cast<size_t>(id)]) -
                               ((x > 0) ? anat[static_cast<size_t>(id) - 1] : anat[static_cast<size_t>(id)])) * 0.5;
          const double ga_y = (((y < Ny - 1) ? anat[static_cast<size_t>(id) + static_cast<size_t>(Nx)] : anat[static_cast<size_t>(id)]) -
                               ((y > 0) ? anat[static_cast<size_t>(id) - static_cast<size_t>(Nx)] : anat[static_cast<size_t>(id)])) * 0.5;
          const double ga_z = (((z < Nz - 1) ? anat[static_cast<size_t>(id) + static_cast<size_t>(Nx) * static_cast<size_t>(Ny)] : anat[static_cast<size_t>(id)]) -
                               ((z > 0) ? anat[static_cast<size_t>(id) - static_cast<size_t>(Nx) * static_cast<size_t>(Ny)] : anat[static_cast<size_t>(id)])) * 0.5;

          const double len = std::sqrt(ga_x * ga_x + ga_y * ga_y + ga_z * ga_z) + 1e-12;
          reg.pls_nx[static_cast<size_t>(id)] = static_cast<float>(ga_x / len);
          reg.pls_ny[static_cast<size_t>(id)] = static_cast<float>(ga_y / len);
          reg.pls_nz[static_cast<size_t>(id)] = static_cast<float>(ga_z / len);
        }
        reg.pls_precomputed = true;
      }
    }
  }

  std::vector<float> Jx(N), Jy(N), Jz(N);
#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t id = 0; id < static_cast<ptrdiff_t>(N); ++id) {
    if (mask && (*mask)[static_cast<size_t>(id)] == 0) {
      Jx[static_cast<size_t>(id)] = 0.0f;
      Jy[static_cast<size_t>(id)] = 0.0f;
      Jz[static_cast<size_t>(id)] = 0.0f;
      continue;
    }

    const int z = static_cast<int>(id) / (Nx * Ny);
    const int y = (static_cast<int>(id) - z * Nx * Ny) / Nx;
    const int x = static_cast<int>(id) - z * Nx * Ny - y * Nx;

    const double gf_x = (((x < Nx - 1) ? f[static_cast<size_t>(id) + 1] : f[static_cast<size_t>(id)]) -
                         ((x > 0) ? f[static_cast<size_t>(id) - 1] : f[static_cast<size_t>(id)])) * 0.5;
    const double gf_y = (((y < Ny - 1) ? f[static_cast<size_t>(id) + static_cast<size_t>(Nx)] : f[static_cast<size_t>(id)]) -
                         ((y > 0) ? f[static_cast<size_t>(id) - static_cast<size_t>(Nx)] : f[static_cast<size_t>(id)])) * 0.5;
    const double gf_z = (((z < Nz - 1) ? f[static_cast<size_t>(id) + static_cast<size_t>(Nx) * static_cast<size_t>(Ny)] : f[static_cast<size_t>(id)]) -
                         ((z > 0) ? f[static_cast<size_t>(id) - static_cast<size_t>(Nx) * static_cast<size_t>(Ny)] : f[static_cast<size_t>(id)])) * 0.5;

    const double nx = reg.pls_nx[static_cast<size_t>(id)];
    const double ny = reg.pls_ny[static_cast<size_t>(id)];
    const double nz = reg.pls_nz[static_cast<size_t>(id)];
    const double dot = gf_x * nx + gf_y * ny + gf_z * nz;
    Jx[static_cast<size_t>(id)] = static_cast<float>(gf_x - dot * nx);
    Jy[static_cast<size_t>(id)] = static_cast<float>(gf_y - dot * ny);
    Jz[static_cast<size_t>(id)] = static_cast<float>(gf_z - dot * nz);
  }

#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t z = 0; z < Nz; ++z) {
    for (int y = 0; y < Ny; ++y) {
      for (int x = 0; x < Nx; ++x) {
        const size_t id = IDX(x, y, static_cast<int>(z), Nx, Ny);
        if (mask && (*mask)[id] == 0)
          continue;

        const double dJx_dx = (x > 0 && x < Nx - 1) ? (Jx[id + 1] - Jx[id - 1]) * 0.5 :
                             ((x == 0 && Nx > 1) ? Jx[id + 1] - Jx[id] : ((x == Nx - 1 && Nx > 1) ? Jx[id] - Jx[id - 1] : 0.0));
        const double dJy_dy = (y > 0 && y < Ny - 1) ? (Jy[id + static_cast<size_t>(Nx)] - Jy[id - static_cast<size_t>(Nx)]) * 0.5 :
                             ((y == 0 && Ny > 1) ? Jy[id + static_cast<size_t>(Nx)] - Jy[id] : ((y == Ny - 1 && Ny > 1) ? Jy[id] - Jy[id - static_cast<size_t>(Nx)] : 0.0));
        const size_t z_stride = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
        const double dJz_dz = (z > 0 && z < Nz - 1) ? (Jz[id + z_stride] - Jz[id - z_stride]) * 0.5 :
                             ((z == 0 && Nz > 1) ? Jz[id + z_stride] - Jz[id] : ((z == Nz - 1 && Nz > 1) ? Jz[id] - Jz[id - z_stride] : 0.0));
        out_grad[id] = -(dJx_dx + dJy_dy + dJz_dz) * reg.pls_eta;
      }
    }
  }
}

void compute_reg_gradient(const std::vector<double> &f, int Nx, int Ny, int Nz,
                          const RegConf &reg, const std::vector<uint8_t> *mask,
                          std::vector<double> &gradR, int threads) {
  switch (reg.type) {
  case RegType::TVIsotropic:
    grad_TV_Isotropic(f, Nx, Ny, Nz, reg.tv_eps, mask, reg.skip_outside_mask, gradR, threads);
    break;
  case RegType::TVHuber:
    grad_TV_Huber(f.data(), Nx, Ny, Nz, gradR, reg.tv_eps, mask ? mask->data() : nullptr);
    break;
  case RegType::Tikhonov:
    grad_Tikhonov(f, Nx, Ny, Nz, mask, reg.skip_outside_mask, gradR, threads);
    break;
  case RegType::MRP:
    grad_MRP(f, Nx, Ny, Nz, reg.mrp_eps, mask, reg.skip_outside_mask, gradR, threads);
    break;
  case RegType::RDP:
    grad_RDP(f, Nx, Ny, Nz, reg.delta, reg.gamma, mask, reg.skip_outside_mask, gradR, threads);
    break;
  case RegType::NLM:
    grad_NLM(f, Nx, Ny, Nz, reg.nlm_h, reg.nlm_search, reg.nlm_patch, mask, reg.skip_outside_mask, gradR, threads);
    break;
  case RegType::Bowsher:
    if (reg.anatomical_image)
      grad_Bowsher(f, Nx, Ny, Nz, reg, mask, gradR, threads);
    else
      gradR.assign(static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz), 0.0);
    break;
  case RegType::PLS:
    if (reg.anatomical_image)
      grad_PLS(f, Nx, Ny, Nz, reg, mask, reg.skip_outside_mask, gradR, threads);
    else
      gradR.assign(static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz), 0.0);
    break;
  case RegType::LogCosh:
    grad_LogCosh(f, Nx, Ny, Nz, reg.delta, mask, reg.skip_outside_mask, gradR, threads);
    break;
  default:
    gradR.assign(static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz), 0.0);
    break;
  }
}

void osl_update_inplace(std::vector<double> &f, const std::vector<double> &q,
                        const std::vector<double> &d,
                        const std::vector<double> &gradR, double beta,
                        const std::vector<uint8_t> *mask, bool nonneg,
                        int threads) {
  const size_t N = f.size();
  const double eps = 1e-12;

  omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(N); ++j) {
    if (mask && (*mask)[static_cast<size_t>(j)] == 0) {
      f[static_cast<size_t>(j)] = 0.0;
      continue;
    }

    const double fj = f[static_cast<size_t>(j)];
    double denom_eff = d[static_cast<size_t>(j)] + beta * (gradR[static_cast<size_t>(j)] / std::max(fj, eps));
    if (!(denom_eff > 0.0)) {
      denom_eff = d[static_cast<size_t>(j)] + beta * std::abs(gradR[static_cast<size_t>(j)]) / std::max(fj, eps) + eps;
    }

    f[static_cast<size_t>(j)] *= q[static_cast<size_t>(j)] / denom_eff;
    if (nonneg && f[static_cast<size_t>(j)] < 0.0)
      f[static_cast<size_t>(j)] = 0.0;
  }
}

double compute_reg_energy(const std::vector<double> &f, int Nx, int Ny, int Nz,
                          const RegConf &reg,
                          const std::vector<uint8_t> *mask) {
  if (reg.type == RegType::None || reg.beta <= 0.0)
    return 0.0;

  const size_t N = f.size();
  double energy = 0.0;
  int dx[6], dy[6], dz[6];
  stencil_offsets(Nx, Ny, Nz, dx, dy, dz);
  const bool skip = reg.skip_outside_mask;

  if (reg.type == RegType::TVIsotropic || reg.type == RegType::Tikhonov ||
      reg.type == RegType::MRP || reg.type == RegType::TVHuber) {
#pragma omp parallel for reduction(+ : energy)
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(N); ++idx) {
      if (mask && (*mask)[static_cast<size_t>(idx)] == 0)
        continue;

      const int z = static_cast<int>(idx) / (Nx * Ny);
      const int y = (static_cast<int>(idx) - z * Nx * Ny) / Nx;
      const int x = static_cast<int>(idx) - z * Nx * Ny - y * Nx;

      if (reg.type == RegType::TVIsotropic) {
        double gx = 0.0, gy = 0.0, gz = 0.0;
        if (x + 1 < Nx && (!skip || (mask && (*mask)[IDX(x + 1, y, z, Nx, Ny)])))
          gx = f[IDX(x + 1, y, z, Nx, Ny)] - f[static_cast<size_t>(idx)];
        if (y + 1 < Ny && (!skip || (mask && (*mask)[IDX(x, y + 1, z, Nx, Ny)])))
          gy = f[IDX(x, y + 1, z, Nx, Ny)] - f[static_cast<size_t>(idx)];
        if (z + 1 < Nz && (!skip || (mask && (*mask)[IDX(x, y, z + 1, Nx, Ny)])))
          gz = f[IDX(x, y, z + 1, Nx, Ny)] - f[static_cast<size_t>(idx)];
        energy += std::sqrt(gx * gx + gy * gy + gz * gz + reg.tv_eps * reg.tv_eps);
      } else if (reg.type == RegType::Tikhonov) {
        double local_e = 0.0;
        for (int k = 0; k < 6; ++k) {
          const int nx = x + dx[k], ny = y + dy[k], nz = z + dz[k];
          if (inb(nx, ny, nz, Nx, Ny, Nz)) {
            if (skip && mask && (*mask)[IDX(nx, ny, nz, Nx, Ny)] == 0)
              continue;
            const double diff = f[static_cast<size_t>(idx)] - f[IDX(nx, ny, nz, Nx, Ny)];
            local_e += diff * diff;
          }
        }
        energy += 0.5 * local_e;
      } else if (reg.type == RegType::MRP) {
        const double eps2 = reg.mrp_eps * reg.mrp_eps;
        const double si = std::sqrt(std::max(f[static_cast<size_t>(idx)], 0.0) + eps2);
        double local_e = 0.0;
        for (int k = 0; k < 6; ++k) {
          const int nx = x + dx[k], ny = y + dy[k], nz = z + dz[k];
          if (inb(nx, ny, nz, Nx, Ny, Nz)) {
            if (skip && mask && (*mask)[IDX(nx, ny, nz, Nx, Ny)] == 0)
              continue;
            const double sj = std::sqrt(std::max(f[IDX(nx, ny, nz, Nx, Ny)], 0.0) + eps2);
            const double diff = si - sj;
            local_e += diff * diff;
          }
        }
        energy += 0.5 * local_e;
      } else if (reg.type == RegType::TVHuber) {
        double local_e = 0.0;
        const double d = reg.tv_eps;
        for (int k = 0; k < 6; ++k) {
          const int nx = x + dx[k], ny = y + dy[k], nz = z + dz[k];
          if (inb(nx, ny, nz, Nx, Ny, Nz)) {
            if (skip && mask && (*mask)[IDX(nx, ny, nz, Nx, Ny)] == 0)
              continue;
            const double diff = f[static_cast<size_t>(idx)] - f[IDX(nx, ny, nz, Nx, Ny)];
            const double ad = std::abs(diff);
            local_e += (ad <= d) ? (0.5 * diff * diff) : (d * (ad - 0.5 * d));
          }
        }
        energy += 0.5 * local_e;
      }
    }
  } else if (reg.type == RegType::LogCosh) {
    const double delta = std::max(reg.delta, 1e-6);
    const double inv_delta = 1.0 / delta;
#pragma omp parallel for reduction(+ : energy) schedule(static)
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(N); ++idx) {
      if (mask && (*mask)[static_cast<size_t>(idx)] == 0)
        continue;
      const int z = static_cast<int>(idx) / (Nx * Ny);
      const int y = (static_cast<int>(idx) - z * Nx * Ny) / Nx;
      const int x = static_cast<int>(idx) - z * Nx * Ny - y * Nx;
      const double fj = f[static_cast<size_t>(idx)];
      for (int k = 0; k < 6; ++k) {
        const int nx = x + dx[k], ny = y + dy[k], nz = z + dz[k];
        if (!inb(nx, ny, nz, Nx, Ny, Nz))
          continue;
        if (skip && mask && (*mask)[IDX(nx, ny, nz, Nx, Ny)] == 0)
          continue;
        const double fk = f[IDX(nx, ny, nz, Nx, Ny)];
        const double abs_x = std::abs((fj - fk) * inv_delta);
        const double logcosh = abs_x + std::log1p(std::exp(-2.0 * abs_x)) - std::log(2.0);
        energy += delta * delta * logcosh;
      }
    }
    energy *= 0.5;
  } else if (reg.type == RegType::RDP) {
    const double eps = std::max(reg.delta, 1e-6);
#pragma omp parallel for reduction(+ : energy)
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(N); ++idx) {
      if (mask && (*mask)[static_cast<size_t>(idx)] == 0)
        continue;
      const int z = static_cast<int>(idx) / (Nx * Ny);
      const int y = (static_cast<int>(idx) - z * Nx * Ny) / Nx;
      const int x = static_cast<int>(idx) - z * Nx * Ny - y * Nx;
      const double fj = f[static_cast<size_t>(idx)];
      for (int k = 0; k < 6; ++k) {
        const int nx = x + dx[k], ny = y + dy[k], nz = z + dz[k];
        if (!inb(nx, ny, nz, Nx, Ny, Nz))
          continue;
        if (skip && mask && (*mask)[IDX(nx, ny, nz, Nx, Ny)] == 0)
          continue;
        const double fk = f[IDX(nx, ny, nz, Nx, Ny)];
        const double a = fj - fk;
        double b = fj + fk + reg.gamma * std::abs(a) + 2.0 * eps;
        if (b < eps)
          b = eps;
        energy += (a * a) / b;
      }
    }
    energy *= 0.5;
  }

  return reg.beta * energy;
}

} // namespace regularizer
} // namespace spect