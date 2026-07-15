#include "filters/Convolution.h"
#include "filters/ButterworthFilter.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>

namespace spect {
namespace filters {

namespace {

int mirrorIndex(int index, int size) {
  if (size <= 1)
    return 0;
  const int period = 2 * size - 2;
  index %= period;
  if (index < 0)
    index += period;
  return index < size ? index : period - index;
}

} // namespace

std::vector<double> makeGaussian1D(double fwhm_mm, double vox_mm,
                                   double min_sigmas) {
  if (fwhm_mm <= 0.0 || vox_mm <= 0.0)
    return {1.0};

  const double sigma_vox = (fwhm_mm / 2.354820045) / vox_mm;
  const double trunc = (min_sigmas > 0.0) ? min_sigmas : 4.0;
  const int radius = std::max(1, static_cast<int>(std::ceil(trunc * sigma_vox)));
  const int kernel_size = 2 * radius + 1;

  std::vector<double> kernel(static_cast<size_t>(kernel_size));
  double sum = 0.0;
  for (int i = 0; i < kernel_size; ++i) {
    const double x = static_cast<double>(i - radius);
    const double value = std::exp(-0.5 * (x * x) / (sigma_vox * sigma_vox));
    kernel[static_cast<size_t>(i)] = value;
    sum += value;
  }
  for (double &value : kernel)
    value /= sum;
  return kernel;
}

static void conv1d_x_avx2(const std::vector<double> &in, std::vector<double> &out,
                          size_t Nx, size_t Ny, size_t Nz,
                          const std::vector<double> &kx) {
  out.resize(in.size());
  const int radius = static_cast<int>(kx.size()) / 2;

#pragma omp parallel for
  for (ptrdiff_t k = 0; k < static_cast<ptrdiff_t>(Nz); ++k) {
    for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(Ny); ++j) {
      const double *src_row = &in[idx3(0, static_cast<size_t>(j), static_cast<size_t>(k), Nx, Ny)];
      double *dst_row = &out[idx3(0, static_cast<size_t>(j), static_cast<size_t>(k), Nx, Ny)];

      ptrdiff_t i = 0;
      for (; i < std::min<ptrdiff_t>(radius, static_cast<ptrdiff_t>(Nx)); ++i) {
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t) {
          const int ii = mirrorIndex(static_cast<int>(i) + t,
                                     static_cast<int>(Nx));
          sum += src_row[ii] * kx[static_cast<size_t>(t + radius)];
        }
        dst_row[i] = sum;
      }

      for (; i <= static_cast<ptrdiff_t>(Nx) - radius - 4; i += 4) {
        __m256d v_sum = _mm256_setzero_pd();
        for (int t = -radius; t <= radius; ++t) {
          __m256d v_k = _mm256_broadcast_sd(&kx[static_cast<size_t>(t + radius)]);
          __m256d v_in = _mm256_loadu_pd(&src_row[i + t]);
          v_sum = _mm256_fmadd_pd(v_k, v_in, v_sum);
        }
        _mm256_storeu_pd(&dst_row[i], v_sum);
      }

      for (; i < static_cast<ptrdiff_t>(Nx); ++i) {
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t) {
          const int ii = mirrorIndex(static_cast<int>(i) + t,
                                     static_cast<int>(Nx));
          sum += src_row[ii] * kx[static_cast<size_t>(t + radius)];
        }
        dst_row[i] = sum;
      }
    }
  }
}

void conv1d_x_mirror(const std::vector<double> &in, std::vector<double> &out,
                     size_t Nx, size_t Ny, size_t Nz,
                     const std::vector<double> &kx) {
  conv1d_x_avx2(in, out, Nx, Ny, Nz, kx);
}

void conv1d_y_mirror(const std::vector<double> &in, std::vector<double> &out,
                     size_t Nx, size_t Ny, size_t Nz,
                     const std::vector<double> &ky) {
  out.resize(in.size());
  const int radius = static_cast<int>(ky.size()) / 2;

#pragma omp parallel for collapse(2)
  for (ptrdiff_t k = 0; k < static_cast<ptrdiff_t>(Nz); ++k) {
    for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(Ny); ++j) {
      double *dst_row = &out[idx3(0, static_cast<size_t>(j), static_cast<size_t>(k), Nx, Ny)];

      ptrdiff_t i = 0;
      for (; i <= static_cast<ptrdiff_t>(Nx) - 4; i += 4) {
        __m256d v_sum = _mm256_setzero_pd();
        for (int t = -radius; t <= radius; ++t) {
          const int jj = mirrorIndex(static_cast<int>(j) + t,
                                     static_cast<int>(Ny));

          __m256d v_k = _mm256_broadcast_sd(&ky[static_cast<size_t>(t + radius)]);
          __m256d v_in = _mm256_loadu_pd(&in[idx3(static_cast<size_t>(i), static_cast<size_t>(jj), static_cast<size_t>(k), Nx, Ny)]);
          v_sum = _mm256_fmadd_pd(v_k, v_in, v_sum);
        }
        _mm256_storeu_pd(&dst_row[i], v_sum);
      }

      for (; i < static_cast<ptrdiff_t>(Nx); ++i) {
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t) {
          const int jj = mirrorIndex(static_cast<int>(j) + t,
                                     static_cast<int>(Ny));
          sum += in[idx3(static_cast<size_t>(i), static_cast<size_t>(jj), static_cast<size_t>(k), Nx, Ny)] *
                 ky[static_cast<size_t>(t + radius)];
        }
        dst_row[i] = sum;
      }
    }
  }
}

void conv1d_z_mirror(const std::vector<double> &in, std::vector<double> &out,
                     size_t Nx, size_t Ny, size_t Nz,
                     const std::vector<double> &kz) {
  out.resize(in.size());
  const int radius = static_cast<int>(kz.size()) / 2;

#pragma omp parallel for collapse(2)
  for (ptrdiff_t k = 0; k < static_cast<ptrdiff_t>(Nz); ++k) {
    for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(Ny); ++j) {
      double *dst_row = &out[idx3(0, static_cast<size_t>(j), static_cast<size_t>(k), Nx, Ny)];

      ptrdiff_t i = 0;
      for (; i <= static_cast<ptrdiff_t>(Nx) - 4; i += 4) {
        __m256d v_sum = _mm256_setzero_pd();
        for (int t = -radius; t <= radius; ++t) {
          const int kk = mirrorIndex(static_cast<int>(k) + t,
                                     static_cast<int>(Nz));

          __m256d v_k = _mm256_broadcast_sd(&kz[static_cast<size_t>(t + radius)]);
          __m256d v_in = _mm256_loadu_pd(&in[idx3(static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(kk), Nx, Ny)]);
          v_sum = _mm256_fmadd_pd(v_k, v_in, v_sum);
        }
        _mm256_storeu_pd(&dst_row[i], v_sum);
      }

      for (; i < static_cast<ptrdiff_t>(Nx); ++i) {
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t) {
          const int kk = mirrorIndex(static_cast<int>(k) + t,
                                     static_cast<int>(Nz));
          sum += in[idx3(static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(kk), Nx, Ny)] *
                 kz[static_cast<size_t>(t + radius)];
        }
        dst_row[i] = sum;
      }
    }
  }
}

void conv3_separable_norm(const std::vector<double> &in, std::vector<double> &out,
                          size_t Nx, size_t Ny, size_t Nz,
                          const std::vector<double> &kx,
                          const std::vector<double> &ky,
                          const std::vector<double> &kz,
                          const std::vector<double> *weight, double eps) {
  std::vector<double> tmp1;
  std::vector<double> tmp2;
  conv1d_x_avx2(in, tmp1, Nx, Ny, Nz, kx);
  conv1d_y_mirror(tmp1, tmp2, Nx, Ny, Nz, ky);
  conv1d_z_mirror(tmp2, out, Nx, Ny, Nz, kz);

  if (weight) {
    std::vector<double> w1;
    std::vector<double> w2;
    std::vector<double> wn;
    conv1d_x_avx2(*weight, w1, Nx, Ny, Nz, kx);
    conv1d_y_mirror(w1, w2, Nx, Ny, Nz, ky);
    conv1d_z_mirror(w2, wn, Nx, Ny, Nz, kz);

#pragma omp parallel for
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(out.size()); ++idx) {
      const double w = wn[static_cast<size_t>(idx)];
      out[static_cast<size_t>(idx)] = (w > eps) ? out[static_cast<size_t>(idx)] / w : 0.0;
    }
  }
}

void conv3d_separable_inplace(std::vector<double> &f, int Nx, int Ny, int Nz,
                              const std::vector<double> &kx,
                              const std::vector<double> &ky,
                              const std::vector<double> &kz, int threads) {
  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  thread_local std::vector<double> tmp;
  if (tmp.capacity() < total)
    tmp.reserve(total);

  omp_set_num_threads(threads);
  conv1d_x_avx2(f, tmp, static_cast<size_t>(Nx), static_cast<size_t>(Ny), static_cast<size_t>(Nz), kx);
  conv1d_y_mirror(tmp, f, static_cast<size_t>(Nx), static_cast<size_t>(Ny), static_cast<size_t>(Nz), ky);
  conv1d_z_mirror(f, tmp, static_cast<size_t>(Nx), static_cast<size_t>(Ny), static_cast<size_t>(Nz), kz);
  f.swap(tmp);
}

void conv3d_separable_adjoint_inplace(
    std::vector<double> &f, int Nx, int Ny, int Nz,
    const std::vector<double> &kx, const std::vector<double> &ky,
    const std::vector<double> &kz, int threads) {
  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::invalid_argument("PSF adjoint input size mismatch");

  std::vector<double> tmp(total, 0.0);
  const int rz = static_cast<int>(kz.size()) / 2;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      for (int z = 0; z < Nz; ++z) {
        const double value = f[idx3(static_cast<size_t>(x),
                                    static_cast<size_t>(y),
                                    static_cast<size_t>(z), Nx, Ny)];
        for (int t = -rz; t <= rz; ++t) {
          const int zz = mirrorIndex(z + t, Nz);
          tmp[idx3(static_cast<size_t>(x), static_cast<size_t>(y),
                   static_cast<size_t>(zz), Nx, Ny)] +=
              kz[static_cast<size_t>(t + rz)] * value;
        }
      }
    }
  }

  std::fill(f.begin(), f.end(), 0.0);
  const int ry = static_cast<int>(ky.size()) / 2;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
  for (int z = 0; z < Nz; ++z) {
    for (int x = 0; x < Nx; ++x) {
      for (int y = 0; y < Ny; ++y) {
        const double value = tmp[idx3(static_cast<size_t>(x),
                                      static_cast<size_t>(y),
                                      static_cast<size_t>(z), Nx, Ny)];
        for (int t = -ry; t <= ry; ++t) {
          const int yy = mirrorIndex(y + t, Ny);
          f[idx3(static_cast<size_t>(x), static_cast<size_t>(yy),
                 static_cast<size_t>(z), Nx, Ny)] +=
              ky[static_cast<size_t>(t + ry)] * value;
        }
      }
    }
  }

  std::fill(tmp.begin(), tmp.end(), 0.0);
  const int rx = static_cast<int>(kx.size()) / 2;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
  for (int z = 0; z < Nz; ++z) {
    for (int y = 0; y < Ny; ++y) {
      for (int x = 0; x < Nx; ++x) {
        const double value = f[idx3(static_cast<size_t>(x),
                                    static_cast<size_t>(y),
                                    static_cast<size_t>(z), Nx, Ny)];
        for (int t = -rx; t <= rx; ++t) {
          const int xx = mirrorIndex(x + t, Nx);
          tmp[idx3(static_cast<size_t>(xx), static_cast<size_t>(y),
                   static_cast<size_t>(z), Nx, Ny)] +=
              kx[static_cast<size_t>(t + rx)] * value;
        }
      }
    }
  }
  f.swap(tmp);
}

void conv2d_separable_inplace(std::vector<double> &img, int rows, int cols,
                              const std::vector<double> &ky,
                              const std::vector<double> &kx, int threads) {
  if (rows <= 0 || cols <= 0)
    return;

  const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  thread_local std::vector<double> tmp;
  tmp.resize(total);
  auto idx2 = [cols](int r, int c) {
    return static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
  };

  const int ry = static_cast<int>(ky.size()) / 2;
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      double sum = 0.0;
      for (int t = -ry; t <= ry; ++t) {
        int rr = r + t;
        if (rr < 0)
          rr = -rr;
        else if (rr >= rows)
          rr = rows - 1 - (rr - (rows - 1));
        sum += ky[static_cast<size_t>(t + ry)] * img[idx2(rr, c)];
      }
      tmp[idx2(r, c)] = sum;
    }
  }

  const int rx = static_cast<int>(kx.size()) / 2;
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      double sum = 0.0;
      for (int t = -rx; t <= rx; ++t) {
        int cc = c + t;
        if (cc < 0)
          cc = -cc;
        else if (cc >= cols)
          cc = cols - 1 - (cc - (cols - 1));
        sum += kx[static_cast<size_t>(t + rx)] * tmp[idx2(r, cc)];
      }
      img[idx2(r, c)] = sum;
    }
  }
}

PSFKernels make_psf_kernels(const PSFConf &psf) {
  PSFKernels kernels;
  const double fwhm_x = (psf.fwhm_x > 0.0) ? psf.fwhm_x : psf.fwhm_mm;
  const double fwhm_y = (psf.fwhm_y > 0.0) ? psf.fwhm_y : psf.fwhm_mm;
  const double fwhm_z =
      psf.z_blur ? ((psf.fwhm_z > 0.0) ? psf.fwhm_z : psf.fwhm_mm) : 0.0;
  kernels.kx = makeGaussian1D(fwhm_x, psf.vox_x, psf.min_sigmas);
  kernels.ky = makeGaussian1D(fwhm_y, psf.vox_y, psf.min_sigmas);
  kernels.kz = makeGaussian1D(fwhm_z, psf.vox_z, psf.min_sigmas);
  return kernels;
}

std::vector<double> make_butterworth_mask_3d(int Nx, int Ny, int Nz, Spacing3,
                                             const ButterworthConf &bw,
                                             int threads) {
  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  std::vector<double> mask(total, 1.0);
  if (Nx <= 0 || Ny <= 0 || Nz <= 0)
    return mask;

  const double denomX = 0.5 * std::max(1, Nx);
  const double denomY = 0.5 * std::max(1, Ny);
  const double denomZ = 0.5 * std::max(1, Nz);
  const double norm = std::sqrt(3.0);
  const int order = std::max(1, bw.order);
  const double fc_lo = bw.cutoff_lo_frac;
  const double fc_hi = bw.cutoff_hi_frac;

  auto lowpass = [order](double r, double fc) {
    if (fc <= 0.0)
      return 0.0;
    if (r <= 0.0)
      return 1.0;
    const double ratio = r / fc;
    return 1.0 / (1.0 + std::pow(ratio, 2 * order));
  };
  auto highpass = [&lowpass](double r, double fc) { return 1.0 - lowpass(r, fc); };

#pragma omp parallel for num_threads(threads) schedule(static) collapse(2)
  for (int z = 0; z < Nz; ++z) {
    for (int y = 0; y < Ny; ++y) {
      const int kz = (z <= Nz / 2) ? z : z - Nz;
      const int ky = (y <= Ny / 2) ? y : y - Ny;
      const double fz = static_cast<double>(kz) / denomZ;
      const double fy = static_cast<double>(ky) / denomY;

      for (int x = 0; x < Nx; ++x) {
        const int kx = (x <= Nx / 2) ? x : x - Nx;
        const double fx = static_cast<double>(kx) / denomX;
        const double r = std::sqrt(fx * fx + fy * fy + fz * fz) / norm;

        double value = 1.0;
        switch (bw.type) {
        case BWType::None:
          value = 1.0;
          break;
        case BWType::Lowpass:
          value = lowpass(r, fc_hi);
          break;
        case BWType::Highpass:
          value = highpass(r, fc_hi);
          break;
        case BWType::Bandpass:
          value = highpass(r, fc_lo) * lowpass(r, fc_hi);
          break;
        case BWType::Bandstop:
          value = 1.0 - (highpass(r, fc_lo) * lowpass(r, fc_hi));
          break;
        }
        mask[static_cast<size_t>(z) * static_cast<size_t>(Ny) * static_cast<size_t>(Nx) +
             static_cast<size_t>(y) * static_cast<size_t>(Nx) + static_cast<size_t>(x)] = value;
      }
    }
  }
  return mask;
}

void apply_butterworth_inplace(std::vector<double> &f, int Nx, int Ny, int Nz,
                               Spacing3 sp, const ButterworthConf &bw,
                               int threads) {
  apply_butterworth_inplace_cufft(f, Nx, Ny, Nz, sp, bw, threads);
}

#if !defined(SPECT_BUTTERWORTH_HAVE_CUFFT)
void apply_butterworth_inplace_cufft(std::vector<double> &, int, int, int,
                                     Spacing3, const ButterworthConf &, int) {
  throw std::runtime_error(
      "Butterworth cuFFT backend was requested but is not compiled in this build.");
}
#endif

} // namespace filters
} // namespace spect
