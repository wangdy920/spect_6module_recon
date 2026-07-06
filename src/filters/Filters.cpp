#include "filters/Convolution.h"
#include "filters/ButterworthFilter.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <mutex>
#include <omp.h>
#include <stdexcept>

#include <fftw3.h>

namespace spect {
namespace filters {

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
      for (; i < radius; ++i) {
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t) {
          int ii = static_cast<int>(i) + t;
          if (ii < 0)
            ii = -ii;
          if (ii >= static_cast<int>(Nx))
            ii = 2 * static_cast<int>(Nx) - 2 - ii;
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
          int ii = static_cast<int>(i) + t;
          if (ii < 0)
            ii = -ii;
          if (ii >= static_cast<int>(Nx))
            ii = 2 * static_cast<int>(Nx) - 2 - ii;
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
          int jj = static_cast<int>(j) + t;
          if (jj < 0)
            jj = -jj;
          if (jj >= static_cast<int>(Ny))
            jj = 2 * static_cast<int>(Ny) - 2 - jj;

          __m256d v_k = _mm256_broadcast_sd(&ky[static_cast<size_t>(t + radius)]);
          __m256d v_in = _mm256_loadu_pd(&in[idx3(static_cast<size_t>(i), static_cast<size_t>(jj), static_cast<size_t>(k), Nx, Ny)]);
          v_sum = _mm256_fmadd_pd(v_k, v_in, v_sum);
        }
        _mm256_storeu_pd(&dst_row[i], v_sum);
      }

      for (; i < static_cast<ptrdiff_t>(Nx); ++i) {
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t) {
          int jj = static_cast<int>(j) + t;
          if (jj < 0)
            jj = -jj;
          if (jj >= static_cast<int>(Ny))
            jj = 2 * static_cast<int>(Ny) - 2 - jj;
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
          int kk = static_cast<int>(k) + t;
          if (kk < 0)
            kk = -kk;
          if (kk >= static_cast<int>(Nz))
            kk = 2 * static_cast<int>(Nz) - 2 - kk;

          __m256d v_k = _mm256_broadcast_sd(&kz[static_cast<size_t>(t + radius)]);
          __m256d v_in = _mm256_loadu_pd(&in[idx3(static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(kk), Nx, Ny)]);
          v_sum = _mm256_fmadd_pd(v_k, v_in, v_sum);
        }
        _mm256_storeu_pd(&dst_row[i], v_sum);
      }

      for (; i < static_cast<ptrdiff_t>(Nx); ++i) {
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t) {
          int kk = static_cast<int>(k) + t;
          if (kk < 0)
            kk = -kk;
          if (kk >= static_cast<int>(Nz))
            kk = 2 * static_cast<int>(Nz) - 2 - kk;
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
  const double fwhm_z = (psf.fwhm_z > 0.0) ? psf.fwhm_z : psf.fwhm_mm;
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
  ButterworthBackendChoice resolved = bw.backend;
  if (resolved == ButterworthBackendChoice::Auto) {
    resolved = isButterworthCUFFTAvailable()
                   ? ButterworthBackendChoice::CUFFT
                   : ButterworthBackendChoice::FFTW;
  }

  if (resolved == ButterworthBackendChoice::CUFFT) {
    if (isButterworthCUFFTAvailable()) {
      apply_butterworth_inplace_cufft(f, Nx, Ny, Nz, sp, bw, threads);
      return;
    }
    apply_butterworth_inplace_fftw(f, Nx, Ny, Nz, sp, bw, threads);
    return;
  }

  apply_butterworth_inplace_fftw(f, Nx, Ny, Nz, sp, bw, threads);
}

#if !defined(SPECT_BUTTERWORTH_HAVE_CUFFT)
void apply_butterworth_inplace_cufft(std::vector<double> &, int, int, int,
                                     Spacing3, const ButterworthConf &, int) {
  throw std::runtime_error(
      "Butterworth cuFFT backend was requested but is not compiled in this build.");
}
#endif

void apply_butterworth_inplace_fftw(std::vector<double> &f, int Nx, int Ny, int Nz,
                                    Spacing3 sp, const ButterworthConf &bw,
                                    int threads) {
  if (!bw.enabled)
    return;

  // Keep the FFTW execution path conservative until the backend is abstracted.
  // The broader reconstruction pipeline may still use the caller's thread count.
  const int fftw_safe_threads = 1;

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::runtime_error("Butterworth input size mismatch");

  struct FFTWCache {
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;
#ifdef USE_FFTW_THREADS
    int plan_threads = 1;
#endif
    Spacing3 sp;
    ButterworthConf bw;
    fftw_complex *F = nullptr;
    fftw_complex *Fk = nullptr;
    fftw_plan fwd = nullptr;
    fftw_plan bwd = nullptr;
    std::vector<double> H_cache;

    ~FFTWCache() {
      if (fwd)
        fftw_destroy_plan(fwd);
      if (bwd)
        fftw_destroy_plan(bwd);
      if (F)
        fftw_free(F);
      if (Fk)
        fftw_free(Fk);
    }
  };

  thread_local FFTWCache cache;

  static std::once_flag fftw_init_flag;
  static std::mutex fftw_plan_mutex;
#ifdef USE_FFTW_THREADS
  std::call_once(fftw_init_flag, []() {
    if (fftw_init_threads() == 0)
      throw std::runtime_error("FFTW thread initialization failed");
  });
#endif

  auto approx_eq = [](double a, double b) {
    return std::abs(a - b) <= 1e-9 * std::max(1.0, std::max(std::abs(a), std::abs(b)));
  };

  const int requested_plan_threads = fftw_safe_threads;
  const bool rebuild_plan = Nx != cache.Nx || Ny != cache.Ny || Nz != cache.Nz
#ifdef USE_FFTW_THREADS
                            || requested_plan_threads != cache.plan_threads
#endif
      ;
  const bool rebuild_mask = rebuild_plan || !approx_eq(sp.dx, cache.sp.dx) ||
                            !approx_eq(sp.dy, cache.sp.dy) || !approx_eq(sp.dz, cache.sp.dz) ||
                            bw.type != cache.bw.type || bw.order != cache.bw.order ||
                            !approx_eq(bw.cutoff_lo_frac, cache.bw.cutoff_lo_frac) ||
                            !approx_eq(bw.cutoff_hi_frac, cache.bw.cutoff_hi_frac);

  if (rebuild_plan) {
    if (cache.fwd) {
      fftw_destroy_plan(cache.fwd);
      cache.fwd = nullptr;
    }
    if (cache.bwd) {
      fftw_destroy_plan(cache.bwd);
      cache.bwd = nullptr;
    }
    if (cache.F) {
      fftw_free(cache.F);
      cache.F = nullptr;
    }
    if (cache.Fk) {
      fftw_free(cache.Fk);
      cache.Fk = nullptr;
    }

    cache.F = fftw_alloc_complex(total);
    cache.Fk = fftw_alloc_complex(total);
    if (!cache.F || !cache.Fk)
      throw std::bad_alloc();

    cache.Nx = Nx;
    cache.Ny = Ny;
    cache.Nz = Nz;

    {
      std::lock_guard<std::mutex> lock(fftw_plan_mutex);
#ifdef USE_FFTW_THREADS
      fftw_plan_with_nthreads(requested_plan_threads);
      cache.plan_threads = requested_plan_threads;
#endif
      cache.fwd = fftw_plan_dft_3d(Nz, Ny, Nx, cache.F, cache.Fk, FFTW_FORWARD, FFTW_ESTIMATE);
      cache.bwd = fftw_plan_dft_3d(Nz, Ny, Nx, cache.Fk, cache.F, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    if (!cache.fwd || !cache.bwd)
      throw std::runtime_error("FFTW plan creation failed");
  }

  if (rebuild_mask) {
    cache.H_cache = make_butterworth_mask_3d(Nx, Ny, Nz, sp, bw, 1);
    cache.sp = sp;
    cache.bw = bw;
    const bool any_nonzero = std::any_of(cache.H_cache.begin(), cache.H_cache.end(),
                                         [](double value) { return value != 0.0; });
    if (!any_nonzero)
      throw std::runtime_error("Butterworth mask is all zero");
  }

#pragma omp parallel for schedule(static) num_threads(fftw_safe_threads)
  for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(total); ++i) {
    cache.F[i][0] = f[static_cast<size_t>(i)];
    cache.F[i][1] = 0.0;
  }

  fftw_execute(cache.fwd);

#pragma omp parallel for schedule(static) num_threads(fftw_safe_threads)
  for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(total); ++i) {
    const double h = cache.H_cache[static_cast<size_t>(i)];
    cache.Fk[i][0] *= h;
    cache.Fk[i][1] *= h;
  }

  fftw_execute(cache.bwd);

  const double scale = 1.0 / static_cast<double>(total);
#pragma omp parallel for schedule(static) num_threads(fftw_safe_threads)
  for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(total); ++i) {
    const double value = cache.F[i][0] * scale;
    f[static_cast<size_t>(i)] = (value < 0.0) ? 0.0 : value;
  }
}

} // namespace filters
} // namespace spect
