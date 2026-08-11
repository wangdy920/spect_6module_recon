#include "filters/cuda/VolumeOpsCUDA.h"
#include "filters/Convolution.h"

#if CUDA_ENABLED

#include "cuda/CudaConfig.h"
#include "cuda/CudaDriverHelpers.h"
#include "cuda/CudaLaunchConfig.h"
#include "cuda/CudaNvrtcArch.h"
#include "cuda/CudaNvrtcLoader.h"

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace spect::cuda {
namespace {

struct VolumeKernelCache {
  CUmodule module = nullptr;
  CUfunction convX = nullptr;
  CUfunction convY = nullptr;
  CUfunction convZ = nullptr;
  CUfunction convXF = nullptr;
  CUfunction convYF = nullptr;
  CUfunction convZF = nullptr;
  CUfunction convAdjoint = nullptr;
  CUfunction convAdjointF = nullptr;
  CUfunction convXZero = nullptr;
  CUfunction convYZero = nullptr;
  CUfunction convZZero = nullptr;
  CUfunction divZnorm = nullptr;
  CUfunction convXZeroF = nullptr;
  CUfunction convYZeroF = nullptr;
  CUfunction convZZeroF = nullptr;
  CUfunction divZnormF = nullptr;
  CUfunction attenLineIntegral = nullptr;
  CUfunction rdpGrad = nullptr;
  CUfunction tvGrad = nullptr;
  CUfunction tvProxDual = nullptr;
  CUfunction tvProxPrimal = nullptr;
  int device = -1;
  int computeMajor = -1;
  int computeMinor = -1;
  std::mutex mutex;

  ~VolumeKernelCache() {
    if (module)
      cuModuleUnload(module);
  }
};

VolumeKernelCache g_volumeKernelCache;
std::atomic<bool> g_volumeOpsDisabled{false};

bool envFlag(const char *name, bool defaultValue) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return defaultValue;
  return std::string(value) != "0" && std::string(value) != "false" &&
         std::string(value) != "off";
}

bool cudaVolumeOpsRequested() {
  return envFlag("SPECT_CUDA_VOLUME_OPS", true);
}

bool cudaPsfUseFloat(bool requested) {
  return requested && !envFlag("SPECT_CUDA_PSF_DOUBLE", false);
}

bool runVolumeKernelGoldenSelfTest(CUfunction convX, CUfunction convXF);

void disableVolumeOps(const char *reason) {
  g_volumeOpsDisabled.store(true, std::memory_order_relaxed);
  // Use cerr (not fprintf) so ConsoleTeeGuard captures this in the run log.
  std::cerr << "[NOTE] CUDA volume ops disabled after NVRTC self-test/load "
               "failure: "
            << reason << '\n';
}
void ensureVolumeKernelsLoaded() {
  if (!cudaVolumeOpsRequested() ||
      g_volumeOpsDisabled.load(std::memory_order_relaxed)) {
    throw std::runtime_error(
        "CUDA volume ops disabled by SPECT_CUDA_VOLUME_OPS or prior load failure.");
  }
  CUDA_CHECK(cudaFree(nullptr));
  SPECT_CUDRV_CHECK(cuInit(0));

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));

  std::lock_guard<std::mutex> lock(g_volumeKernelCache.mutex);
  if (g_volumeKernelCache.module &&
      g_volumeKernelCache.device == device &&
      g_volumeKernelCache.computeMajor == props.major &&
      g_volumeKernelCache.computeMinor == props.minor) {
    return;
  }

  const char *source = R"CU(
extern "C" __device__ int spect_mirror_index(int idx, int dim) {
  if (dim <= 1) {
    return 0;
  }
  const int period = 2 * dim - 2;
  idx %= period;
  if (idx < 0) idx += period;
  return idx < dim ? idx : period - idx;
}

extern "C" __global__ void conv_x_mirror_3d(const double* input,
                                             double* output,
                                             int Nx, int Ny, int Nz,
                                             const double* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int xx = spect_mirror_index(x + t, Nx);
    const unsigned long long src =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_y_mirror_3d(const double* input,
                                             double* output,
                                             int Nx, int Ny, int Nz,
                                             const double* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int yy = spect_mirror_index(y + t, Ny);
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_z_mirror_3d(const double* input,
                                             double* output,
                                             int Nx, int Ny, int Nz,
                                             const double* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int zz = spect_mirror_index(z + t, Nz);
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_x_zero_3d(const double* input,
                                           double* output,
                                           int Nx, int Ny, int Nz,
                                           const double* kernel,
                                           int radius,
                                           unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int xx = x + t;
    if (xx < 0 || xx >= Nx) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_y_zero_3d(const double* input,
                                           double* output,
                                           int Nx, int Ny, int Nz,
                                           const double* kernel,
                                           int radius,
                                           unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int yy = y + t;
    if (yy < 0 || yy >= Ny) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_z_zero_3d(const double* input,
                                           double* output,
                                           int Nx, int Ny, int Nz,
                                           const double* kernel,
                                           int radius,
                                           unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int zz = z + t;
    if (zz < 0 || zz >= Nz) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

// CASToR z 边界补偿：逐体素除以 norm[z]（norm 为长度 Nz 的每z层因子）。
extern "C" __global__ void div_znorm_3d(double* data,
                                         const double* znorm,
                                         int Nx, int Ny, int Nz,
                                         unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  double d = znorm[z];
  if (!isfinite(d) || d < 1e-12) d = 1e-12;
  data[index] /= d;
}

extern "C" __global__ void conv_mirror_adjoint_3d(
    const double* input, double* output, int Nx, int Ny, int Nz,
    const double* kernel, int radius, int axis, unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  const int coord = axis == 0 ? x : (axis == 1 ? y : z);
  const int dim = axis == 0 ? Nx : (axis == 1 ? Ny : Nz);

  if (dim <= 1) {
    double kernelSum = 0.0;
    for (int t = -radius; t <= radius; ++t)
      kernelSum += kernel[t + radius];
    output[index] = kernelSum * input[index];
    return;
  }

  const int period = 2 * dim - 2;
  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int lower = t;
    const int upper = t + dim - 1;
    int q = coord;
    while (q < lower) q += period;
    while (q - period >= lower) q -= period;
    for (; q <= upper; q += period) {
      const int sourceCoord = q - t;
      const unsigned long long src =
          axis == 0
              ? static_cast<unsigned long long>(sourceCoord) +
                    static_cast<unsigned long long>(Nx) * y +
                    static_cast<unsigned long long>(Nx) * Ny * z
              : (axis == 1
                     ? static_cast<unsigned long long>(x) +
                           static_cast<unsigned long long>(Nx) * sourceCoord +
                           static_cast<unsigned long long>(Nx) * Ny * z
                     : static_cast<unsigned long long>(x) +
                           static_cast<unsigned long long>(Nx) * y +
                           static_cast<unsigned long long>(Nx) * Ny *
                               sourceCoord);
      sum += kernel[t + radius] * input[src];
    }
    if (coord != 0 && coord != dim - 1) {
      q = period - coord;
      while (q < lower) q += period;
      while (q - period >= lower) q -= period;
      for (; q <= upper; q += period) {
        const int sourceCoord = q - t;
        const unsigned long long src =
            axis == 0
                ? static_cast<unsigned long long>(sourceCoord) +
                      static_cast<unsigned long long>(Nx) * y +
                      static_cast<unsigned long long>(Nx) * Ny * z
                : (axis == 1
                       ? static_cast<unsigned long long>(x) +
                             static_cast<unsigned long long>(Nx) * sourceCoord +
                             static_cast<unsigned long long>(Nx) * Ny * z
                       : static_cast<unsigned long long>(x) +
                             static_cast<unsigned long long>(Nx) * y +
                             static_cast<unsigned long long>(Nx) * Ny *
                                 sourceCoord);
        sum += kernel[t + radius] * input[src];
      }
    }
  }
  output[index] = sum;
}

)CU"
  R"CU(
extern "C" __global__ void conv_x_mirror_3d_f(const float* input,
                                               float* output,
                                               int Nx, int Ny, int Nz,
                                               const float* kernel,
                                               int radius,
                                               unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int xx = spect_mirror_index(x + t, Nx);
    const unsigned long long src =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_y_mirror_3d_f(const float* input,
                                               float* output,
                                               int Nx, int Ny, int Nz,
                                               const float* kernel,
                                               int radius,
                                               unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int yy = spect_mirror_index(y + t, Ny);
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_z_mirror_3d_f(const float* input,
                                               float* output,
                                               int Nx, int Ny, int Nz,
                                               const float* kernel,
                                               int radius,
                                               unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int zz = spect_mirror_index(z + t, Nz);
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_x_zero_3d_f(const float* input,
                                             float* output,
                                             int Nx, int Ny, int Nz,
                                             const float* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int xx = x + t;
    if (xx < 0 || xx >= Nx) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_y_zero_3d_f(const float* input,
                                             float* output,
                                             int Nx, int Ny, int Nz,
                                             const float* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int yy = y + t;
    if (yy < 0 || yy >= Ny) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_z_zero_3d_f(const float* input,
                                             float* output,
                                             int Nx, int Ny, int Nz,
                                             const float* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int zz = z + t;
    if (zz < 0 || zz >= Nz) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void div_znorm_3d_f(float* data,
                                           const float* znorm,
                                           int Nx, int Ny, int Nz,
                                           unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  float d = znorm[z];
  if (!isfinite(d) || d < 1e-12f) d = 1e-12f;
  data[index] /= d;
}

extern "C" __global__ void conv_mirror_adjoint_3d_f(
    const float* input, float* output, int Nx, int Ny, int Nz,
    const float* kernel, int radius, int axis, unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  const int coord = axis == 0 ? x : (axis == 1 ? y : z);
  const int dim = axis == 0 ? Nx : (axis == 1 ? Ny : Nz);

  if (dim <= 1) {
    float kernelSum = 0.0f;
    for (int t = -radius; t <= radius; ++t)
      kernelSum += kernel[t + radius];
    output[index] = kernelSum * input[index];
    return;
  }

  const int period = 2 * dim - 2;
  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int lower = t;
    const int upper = t + dim - 1;
    int q = coord;
    while (q < lower) q += period;
    while (q - period >= lower) q -= period;
    for (; q <= upper; q += period) {
      const int sourceCoord = q - t;
      const unsigned long long src =
          axis == 0
              ? static_cast<unsigned long long>(sourceCoord) +
                    static_cast<unsigned long long>(Nx) * y +
                    static_cast<unsigned long long>(Nx) * Ny * z
              : (axis == 1
                     ? static_cast<unsigned long long>(x) +
                           static_cast<unsigned long long>(Nx) * sourceCoord +
                           static_cast<unsigned long long>(Nx) * Ny * z
                     : static_cast<unsigned long long>(x) +
                           static_cast<unsigned long long>(Nx) * y +
                           static_cast<unsigned long long>(Nx) * Ny *
                               sourceCoord);
      sum += kernel[t + radius] * input[src];
    }
    if (coord != 0 && coord != dim - 1) {
      q = period - coord;
      while (q < lower) q += period;
      while (q - period >= lower) q -= period;
      for (; q <= upper; q += period) {
        const int sourceCoord = q - t;
        const unsigned long long src =
            axis == 0
                ? static_cast<unsigned long long>(sourceCoord) +
                      static_cast<unsigned long long>(Nx) * y +
                      static_cast<unsigned long long>(Nx) * Ny * z
                : (axis == 1
                       ? static_cast<unsigned long long>(x) +
                             static_cast<unsigned long long>(Nx) * sourceCoord +
                             static_cast<unsigned long long>(Nx) * Ny * z
                       : static_cast<unsigned long long>(x) +
                             static_cast<unsigned long long>(Nx) * y +
                             static_cast<unsigned long long>(Nx) * Ny *
                                 sourceCoord);
        sum += kernel[t + radius] * input[src];
      }
    }
  }
  output[index] = sum;
}

)CU"
R"CU(
extern "C" __device__ double ct_edge_weight(const double* ct,
                                              bool hasCt,
                                              unsigned long long a,
                                              unsigned long long b,
                                              double sigma,
                                              double floorWeight) {
  if (!hasCt) return 1.0;
  const double d = (ct[a] - ct[b]) / sigma;
  return floorWeight + (1.0 - floorWeight) * exp(-0.5 * d * d);
}

extern "C" __global__ void atten_line_integral_3d(
    const double* mu, const double* stepFxByAngle,
    const double* stepFyByAngle, int Nx, int Ny, int Nz, int numAngles,
    double step, int maxSteps, unsigned short* output,
    unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const unsigned long long plane =
      static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny);
  const unsigned long long voxels = plane * static_cast<unsigned long long>(Nz);
  const int angle = static_cast<int>(index / voxels);
  const unsigned long long voxel = index - static_cast<unsigned long long>(angle) * voxels;
  const int z = static_cast<int>(voxel / plane);
  const unsigned long long inPlane = voxel - static_cast<unsigned long long>(z) * plane;
  const int y = static_cast<int>(inPlane / static_cast<unsigned long long>(Nx));
  const int x = static_cast<int>(inPlane - static_cast<unsigned long long>(y) * Nx);
  const double stepFx = stepFxByAngle[angle];
  const double stepFy = stepFyByAngle[angle];
  double fx = static_cast<double>(x) + 0.5 * stepFx;
  double fy = static_cast<double>(y) + 0.5 * stepFy;
  double integral = 0.0;
  const double* slice = mu + static_cast<unsigned long long>(z) * plane;
  for (int k = 0; k < maxSteps; ++k) {
    if (fx <= -1.0 || fy <= -1.0 || fx >= static_cast<double>(Nx) ||
        fy >= static_cast<double>(Ny))
      break;
    const double cx = fmin(fmax(fx, 0.0), static_cast<double>(Nx - 1));
    const double cy = fmin(fmax(fy, 0.0), static_cast<double>(Ny - 1));
    const int x0 = static_cast<int>(floor(cx));
    const int y0 = static_cast<int>(floor(cy));
    const int x1 = x0 + 1 < Nx ? x0 + 1 : Nx - 1;
    const int y1 = y0 + 1 < Ny ? y0 + 1 : Ny - 1;
    const double tx = cx - static_cast<double>(x0);
    const double ty = cy - static_cast<double>(y0);
    const double v00 = slice[static_cast<unsigned long long>(y0) * Nx + x0];
    const double v10 = slice[static_cast<unsigned long long>(y0) * Nx + x1];
    const double v01 = slice[static_cast<unsigned long long>(y1) * Nx + x0];
    const double v11 = slice[static_cast<unsigned long long>(y1) * Nx + x1];
    const double value = (1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) +
                         ty * ((1.0 - tx) * v01 + tx * v11);
    if (value > 0.0) integral += value * step;
    fx += stepFx;
    fy += stepFy;
  }
  const double factor = fmin(fmax(exp(-integral), 0.0), 1.0);
  output[index] = static_cast<unsigned short>(floor(factor * 65535.0 + 0.5));
}

extern "C" __global__ void rdp_grad_3d(const double* input,
                                        const unsigned char* mask,
                                        const double* ct,
                                        bool hasMask,
                                        bool hasCt,
                                        bool skipOutsideMask,
                                        int Nx, int Ny, int Nz,
                                        double eps,
                                        double gamma,
                                        double ctSigma,
                                        double ctWeightFloor,
                                        double* output,
                                        unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  if (hasMask && mask[index] == 0) {
    output[index] = 0.0;
    return;
  }

  const double fj = input[index];
  double g = 0.0;

  const int dx[6] = {-1, 1, 0, 0, 0, 0};
  const int dy[6] = {0, 0, -1, 1, 0, 0};
  const int dz[6] = {0, 0, 0, 0, -1, 1};

  for (int n = 0; n < 6; ++n) {
    const int xx = x + dx[n];
    const int yy = y + dy[n];
    const int zz = z + dz[n];
    if (xx < 0 || xx >= Nx || yy < 0 || yy >= Ny || zz < 0 || zz >= Nz) {
      continue;
    }
    const unsigned long long neighbor =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    if (skipOutsideMask && hasMask && mask[neighbor] == 0) {
      continue;
    }
    const double fk = input[neighbor];
    const double diff = fj - fk;
    const double absDiff = fabs(diff);
    const double signDiff = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
    double denom = fj + fk + gamma * absDiff + 2.0 * eps;
    if (denom < 1.0e-9) {
      denom = 1.0e-9;
    }
    const double term =
        2.0 * diff * denom - diff * diff * (1.0 + gamma * signDiff);
    g += ct_edge_weight(ct, hasCt, index, neighbor, ctSigma,
                        ctWeightFloor) * term / (denom * denom);
  }

  output[index] = g;
}

extern "C" __device__ void tv_compute_p(const double* input,
                                         const unsigned char* mask,
                                         const double* ct,
                                         bool hasMask,
                                         bool hasCt,
                                         bool skipOutsideMask,
                                         int px, int py, int pz,
                                         int Nx, int Ny, int Nz,
                                         double eps,
                                         double ctSigma,
                                         double ctWeightFloor,
                                         double* out_gx, double* out_gy, double* out_gz) {
  const unsigned long long idx =
      static_cast<unsigned long long>(px) +
      static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py) +
      static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
          static_cast<unsigned long long>(pz);

  if (hasMask && mask[idx] == 0) {
    *out_gx = 0.0;
    *out_gy = 0.0;
    *out_gz = 0.0;
    return;
  }

  const double fv = input[idx];
  double gx = 0.0, gy = 0.0, gz = 0.0;
  double mag2 = eps * eps;

  if (px + 1 < Nx) {
    const unsigned long long nidx =
        static_cast<unsigned long long>(px + 1) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(pz);
    if (!skipOutsideMask || !hasMask || mask[nidx] != 0) {
      const double d = input[nidx] - fv;
      const double w = ct_edge_weight(ct, hasCt, idx, nidx, ctSigma,
                                      ctWeightFloor);
      gx = w * d;
      mag2 += w * d * d;
    }
  }
  if (py + 1 < Ny) {
    const unsigned long long nidx =
        static_cast<unsigned long long>(px) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py + 1) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(pz);
    if (!skipOutsideMask || !hasMask || mask[nidx] != 0) {
      const double d = input[nidx] - fv;
      const double w = ct_edge_weight(ct, hasCt, idx, nidx, ctSigma,
                                      ctWeightFloor);
      gy = w * d;
      mag2 += w * d * d;
    }
  }
  if (pz + 1 < Nz) {
    const unsigned long long nidx =
        static_cast<unsigned long long>(px) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(pz + 1);
    if (!skipOutsideMask || !hasMask || mask[nidx] != 0) {
      const double d = input[nidx] - fv;
      const double w = ct_edge_weight(ct, hasCt, idx, nidx, ctSigma,
                                      ctWeightFloor);
      gz = w * d;
      mag2 += w * d * d;
    }
  }

  const double mag = sqrt(mag2);
  *out_gx = gx / mag;
  *out_gy = gy / mag;
  *out_gz = gz / mag;
}

extern "C" __global__ void tv_grad_3d(const double* input,
                                       const unsigned char* mask,
                                       const double* ct,
                                       bool hasMask,
                                       bool hasCt,
                                       bool skipOutsideMask,
                                       int Nx, int Ny, int Nz,
                                       double eps,
                                       double ctSigma,
                                       double ctWeightFloor,
                                       double* output,
                                       unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;

  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  if (hasMask && mask[index] == 0) {
    output[index] = 0.0;
    return;
  }

  /* Compute normalized gradient p(x,y,z) = nabla f / |nabla f| */
  double px, py, pz;
  tv_compute_p(input, mask, ct, hasMask, hasCt, skipOutsideMask,
               x, y, z, Nx, Ny, Nz, eps, ctSigma, ctWeightFloor,
               &px, &py, &pz);

  double diverg = 0.0;

  /* X divergence */
  if (x > 0 && (!skipOutsideMask || !hasMask ||
                mask[index - 1] != 0)) {
    double prev_px, prev_py, prev_pz;
    tv_compute_p(input, mask, ct, hasMask, hasCt, skipOutsideMask,
                 x - 1, y, z, Nx, Ny, Nz, eps, ctSigma, ctWeightFloor,
                 &prev_px, &prev_py, &prev_pz);
    diverg += px - prev_px;
  } else {
    diverg += px;
  }

  /* Y divergence */
  if (y > 0 && (!skipOutsideMask || !hasMask ||
                mask[index - static_cast<unsigned long long>(Nx)] != 0)) {
    double prev_px, prev_py, prev_pz;
    tv_compute_p(input, mask, ct, hasMask, hasCt, skipOutsideMask,
                 x, y - 1, z, Nx, Ny, Nz, eps, ctSigma, ctWeightFloor,
                 &prev_px, &prev_py, &prev_pz);
    diverg += py - prev_py;
  } else {
    diverg += py;
  }

  /* Z divergence */
  const unsigned long long slice =
      static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny);
  if (z > 0 && (!skipOutsideMask || !hasMask ||
                mask[index - slice] != 0)) {
    double prev_px, prev_py, prev_pz;
    tv_compute_p(input, mask, ct, hasMask, hasCt, skipOutsideMask,
                 x, y, z - 1, Nx, Ny, Nz, eps, ctSigma, ctWeightFloor,
                 &prev_px, &prev_py, &prev_pz);
    diverg += pz - prev_pz;
  } else {
    diverg += pz;
  }

  output[index] = -diverg;
}
)CU"

// --- Second source fragment (TV proximal kernels) ---
R"CU(
/* ========== Chambolle-Pock TV Proximal Kernels ========== */

/* Dual update: p = proj_{||·||≤lambda}(p + sigma * grad(u_bar))
 * Uses forward differences for gradient with Neumann BC. */
extern "C" __global__ void tv_prox_dual_update_3d(
    const double* u_bar,
    const double* ct,
    double* px, double* py, double* pz,
    double sigma, double lambda_val,
    bool has_ct, double ct_sigma, double ct_weight_floor,
    int Nx, int Ny, int Nz,
    unsigned long long total)
{
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;

  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  const double u0 = u_bar[index];

  /* Forward difference gradient */
  double gx = 0.0, gy = 0.0, gz = 0.0;
  if (x + 1 < Nx) {
    const unsigned long long j = index + 1;
    double w = 1.0;
    if (has_ct) {
      const double d = (ct[index] - ct[j]) / ct_sigma;
      w = ct_weight_floor + (1.0 - ct_weight_floor) * exp(-0.5 * d * d);
    }
    gx = sqrt(w) * (u_bar[j] - u0);
  }
  if (y + 1 < Ny) {
    const unsigned long long j = index + static_cast<unsigned long long>(Nx);
    double w = 1.0;
    if (has_ct) {
      const double d = (ct[index] - ct[j]) / ct_sigma;
      w = ct_weight_floor + (1.0 - ct_weight_floor) * exp(-0.5 * d * d);
    }
    gy = sqrt(w) * (u_bar[j] - u0);
  }
  if (z + 1 < Nz) {
    const unsigned long long slice =
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny);
    const unsigned long long j = index + slice;
    double w = 1.0;
    if (has_ct) {
      const double d = (ct[index] - ct[j]) / ct_sigma;
      w = ct_weight_floor + (1.0 - ct_weight_floor) * exp(-0.5 * d * d);
    }
    gz = sqrt(w) * (u_bar[j] - u0);
  }

  /* Update dual variables */
  double qx = px[index] + sigma * gx;
  double qy = py[index] + sigma * gy;
  double qz = pz[index] + sigma * gz;

  /* Project onto L2 ball of radius lambda per voxel */
  double norm = sqrt(qx * qx + qy * qy + qz * qz);
  if (norm > lambda_val) {
    double scale = lambda_val / norm;
    qx *= scale;
    qy *= scale;
    qz *= scale;
  }

  px[index] = qx;
  py[index] = qy;
  pz[index] = qz;
}

/* Primal update: u_new = max(0, (tau*v + u + tau*div(p)) / (1+tau))
 * Extrapolation: u_bar = u_new + theta*(u_new - u_old)
 * Divergence uses backward differences (adjoint of forward gradient). */
extern "C" __global__ void tv_prox_primal_update_3d(
    double* u,
    double* u_bar,
    const double* v,
    const double* ct,
    const double* px, const double* py, const double* pz,
    double tau, double theta,
    bool has_ct, double ct_sigma, double ct_weight_floor,
    int Nx, int Ny, int Nz,
    unsigned long long total)
{
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;

  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  /* Divergence: backward differences with Neumann BC (p[-1]=0) */
  double div_val = 0.0;

  /* X */
  if (x + 1 < Nx) {
    double w = 1.0;
    if (has_ct) {
      const double d = (ct[index] - ct[index + 1]) / ct_sigma;
      w = ct_weight_floor + (1.0 - ct_weight_floor) * exp(-0.5 * d * d);
    }
    div_val += sqrt(w) * px[index];
  }
  if (x > 0) {
    const unsigned long long j = index - 1;
    double w = 1.0;
    if (has_ct) {
      const double d = (ct[j] - ct[index]) / ct_sigma;
      w = ct_weight_floor + (1.0 - ct_weight_floor) * exp(-0.5 * d * d);
    }
    div_val -= sqrt(w) * px[j];
  }

  /* Y */
  if (y + 1 < Ny) {
    const unsigned long long j = index + static_cast<unsigned long long>(Nx);
    double w = 1.0;
    if (has_ct) {
      const double d = (ct[index] - ct[j]) / ct_sigma;
      w = ct_weight_floor + (1.0 - ct_weight_floor) * exp(-0.5 * d * d);
    }
    div_val += sqrt(w) * py[index];
  }
  if (y > 0) {
    const unsigned long long j = index - static_cast<unsigned long long>(Nx);
    double w = 1.0;
    if (has_ct) {
      const double d = (ct[j] - ct[index]) / ct_sigma;
      w = ct_weight_floor + (1.0 - ct_weight_floor) * exp(-0.5 * d * d);
    }
    div_val -= sqrt(w) * py[j];
  }

  /* Z */
  if (z + 1 < Nz) {
    const unsigned long long slice =
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny);
    const unsigned long long j = index + slice;
    double w = 1.0;
    if (has_ct) {
      const double d = (ct[index] - ct[j]) / ct_sigma;
      w = ct_weight_floor + (1.0 - ct_weight_floor) * exp(-0.5 * d * d);
    }
    div_val += sqrt(w) * pz[index];
  }
  if (z > 0) {
    const unsigned long long slice =
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny);
    const unsigned long long j = index - slice;
    double w = 1.0;
    if (has_ct) {
      const double d = (ct[j] - ct[index]) / ct_sigma;
      w = ct_weight_floor + (1.0 - ct_weight_floor) * exp(-0.5 * d * d);
    }
    div_val -= sqrt(w) * pz[j];
  }

  /* Primal update */
  const double u_old = u[index];
  double u_new = (tau * v[index] + u_old + tau * div_val) / (1.0 + tau);
  if (u_new < 0.0) u_new = 0.0;

  u[index] = u_new;
  u_bar[index] = u_new + theta * (u_new - u_old);
}
)CU";

  try {
    const std::string arch = nvrtcGpuArchitectureOption(props);
    const char *options[] = {arch.c_str()};
    const std::string ptx = compileNvrtcProgramToPtx(
        source, "volume_ops_cuda.cu", 1, options);

    CUcontext context = nullptr;
    SPECT_CUDRV_CHECK(cuCtxGetCurrent(&context));
    if (context == nullptr) {
      throw std::runtime_error(
          "CUDA context is not initialized for volume helper kernels.");
    }

    if (g_volumeKernelCache.module) {
      SPECT_CUDRV_CHECK(cuModuleUnload(g_volumeKernelCache.module));
      g_volumeKernelCache.module = nullptr;
    }

    loadNvrtcPtxModule(&g_volumeKernelCache.module, ptx);
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convX,
                                          g_volumeKernelCache.module,
                                          "conv_x_mirror_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convY,
                                          g_volumeKernelCache.module,
                                          "conv_y_mirror_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convZ,
                                          g_volumeKernelCache.module,
                                          "conv_z_mirror_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convXF,
                                          g_volumeKernelCache.module,
                                          "conv_x_mirror_3d_f"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convYF,
                                          g_volumeKernelCache.module,
                                          "conv_y_mirror_3d_f"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convZF,
                                          g_volumeKernelCache.module,
                                          "conv_z_mirror_3d_f"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convAdjoint,
                                          g_volumeKernelCache.module,
                                          "conv_mirror_adjoint_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convAdjointF,
                                          g_volumeKernelCache.module,
                                          "conv_mirror_adjoint_3d_f"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convXZero,
                                          g_volumeKernelCache.module,
                                          "conv_x_zero_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convYZero,
                                          g_volumeKernelCache.module,
                                          "conv_y_zero_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convZZero,
                                          g_volumeKernelCache.module,
                                          "conv_z_zero_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.divZnorm,
                                          g_volumeKernelCache.module,
                                          "div_znorm_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convXZeroF,
                                          g_volumeKernelCache.module,
                                          "conv_x_zero_3d_f"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convYZeroF,
                                          g_volumeKernelCache.module,
                                          "conv_y_zero_3d_f"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.convZZeroF,
                                          g_volumeKernelCache.module,
                                          "conv_z_zero_3d_f"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.divZnormF,
                                          g_volumeKernelCache.module,
                                          "div_znorm_3d_f"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(
        &g_volumeKernelCache.attenLineIntegral, g_volumeKernelCache.module,
        "atten_line_integral_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.rdpGrad,
                                          g_volumeKernelCache.module,
                                          "rdp_grad_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.tvGrad,
                                          g_volumeKernelCache.module,
                                          "tv_grad_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.tvProxDual,
                                          g_volumeKernelCache.module,
                                          "tv_prox_dual_update_3d"));
    SPECT_CUDRV_CHECK(cuModuleGetFunction(&g_volumeKernelCache.tvProxPrimal,
                                          g_volumeKernelCache.module,
                                          "tv_prox_primal_update_3d"));
    if (!runVolumeKernelGoldenSelfTest(g_volumeKernelCache.convX,
                                       g_volumeKernelCache.convXF)) {
      throw std::runtime_error("volume convolution golden self-test failed");
    }
    g_volumeKernelCache.device = device;
    g_volumeKernelCache.computeMajor = props.major;
    g_volumeKernelCache.computeMinor = props.minor;
  } catch (const std::exception &ex) {
    if (g_volumeKernelCache.module) {
      cuModuleUnload(g_volumeKernelCache.module);
      g_volumeKernelCache.module = nullptr;
    }
    disableVolumeOps(ex.what());
    throw;
  }
}

template <typename T> class DeviceBuffer {
public:
  ~DeviceBuffer() {
    if (ptr_)
      cudaFree(ptr_);
  }

  void ensure(size_t count) {
    if (count <= capacity_)
      return;
    if (ptr_) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      capacity_ = 0;
    }
    if (count == 0)
      return;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(T)));
    capacity_ = count;
  }

  void upload(const T *host, size_t count) {
    ensure(count);
    if (count > 0)
      CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T),
                            cudaMemcpyHostToDevice));
  }

  void download(T *host, size_t count) const {
    if (count == 0)
      return;
    if (count > capacity_)
      throw std::runtime_error("CUDA device buffer is smaller than requested.");
    CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }

private:
  T *ptr_ = nullptr;
  size_t capacity_ = 0;
};

bool runVolumeKernelGoldenSelfTest(CUfunction convX, CUfunction convXF) {
  constexpr int nx = 3;
  constexpr int ny = 2;
  constexpr int nz = 2;
  constexpr size_t total = static_cast<size_t>(nx) * ny * nz;

  auto run = [&](auto function, auto input, auto kernel, double tolerance) {
    using T = typename std::decay_t<decltype(input)>::value_type;
    T *deviceInput = nullptr;
    T *deviceOutput = nullptr;
    T *deviceKernel = nullptr;
    try {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&deviceInput),
                            total * sizeof(T)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&deviceOutput),
                            total * sizeof(T)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&deviceKernel),
                            kernel.size() * sizeof(T)));
      CUDA_CHECK(cudaMemcpy(deviceInput, input.data(), total * sizeof(T),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(deviceKernel, kernel.data(),
                            kernel.size() * sizeof(T),
                            cudaMemcpyHostToDevice));
      const int radius = 1;
      const unsigned long long totalCount = total;
      CUdeviceptr inputPtr = reinterpret_cast<CUdeviceptr>(deviceInput);
      CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(deviceOutput);
      CUdeviceptr kernelPtr = reinterpret_cast<CUdeviceptr>(deviceKernel);
      void *args[] = {&inputPtr, &outputPtr, const_cast<int *>(&nx),
                      const_cast<int *>(&ny), const_cast<int *>(&nz),
                      &kernelPtr, const_cast<int *>(&radius),
                      const_cast<unsigned long long *>(&totalCount)};
      const unsigned int blockSize = kCudaLaunchBlockSize;
      const unsigned int gridSize =
          static_cast<unsigned int>((total + blockSize - 1) / blockSize);
      SPECT_CUDRV_CHECK(cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1,
                                       1, 0, nullptr, args, nullptr));
      SPECT_CUDRV_CHECK(cuCtxSynchronize());
      std::vector<T> output(total);
      CUDA_CHECK(cudaMemcpy(output.data(), deviceOutput, total * sizeof(T),
                            cudaMemcpyDeviceToHost));
      cudaFree(deviceKernel);
      cudaFree(deviceOutput);
      cudaFree(deviceInput);
      for (const T value : output) {
        if (!std::isfinite(static_cast<double>(value)) ||
            std::abs(static_cast<double>(value) - 1.0) > tolerance)
          return false;
      }
      return true;
    } catch (...) {
      if (deviceKernel)
        cudaFree(deviceKernel);
      if (deviceOutput)
        cudaFree(deviceOutput);
      if (deviceInput)
        cudaFree(deviceInput);
      return false;
    }
  };

  const std::vector<double> inputD(total, 1.0);
  const std::vector<double> kernelD = {0.25, 0.5, 0.25};
  const std::vector<float> inputF(total, 1.0F);
  const std::vector<float> kernelF = {0.25F, 0.5F, 0.25F};
  return run(convX, inputD, kernelD, 1e-12) &&
         run(convXF, inputF, kernelF, 1e-5);
}

struct VolumeOpsState {
  DeviceBuffer<double> primary;
  DeviceBuffer<double> secondary;
  DeviceBuffer<double> kx;
  DeviceBuffer<double> ky;
  DeviceBuffer<double> kz;
  DeviceBuffer<uint8_t> mask;
  DeviceBuffer<double> ct;

  DeviceBuffer<float> primaryF;
  DeviceBuffer<float> secondaryF;
  DeviceBuffer<float> kxF;
  DeviceBuffer<float> kyF;
  DeviceBuffer<float> kzF;
  DeviceBuffer<double> znorm;
  DeviceBuffer<float> znormF;

  // TV proximal (Chambolle-Pock) buffers
  DeviceBuffer<double> tvPx;
  DeviceBuffer<double> tvPy;
  DeviceBuffer<double> tvPz;
  DeviceBuffer<double> tvU;
  DeviceBuffer<double> tvUbar;
  DeviceBuffer<double> tvV;

  std::vector<double> cachedKx;
  std::vector<double> cachedKy;
  std::vector<double> cachedKz;
  std::vector<float> cachedKxF;
  std::vector<float> cachedKyF;
  std::vector<float> cachedKzF;
  std::vector<float> cachedZnormF;
};

thread_local VolumeOpsState g_volumeOpsState;

void uploadKernelIfNeeded(const std::vector<double> &host,
                          std::vector<double> &cached,
                          DeviceBuffer<double> &device) {
  if (cached == host)
    return;
  device.upload(host.data(), host.size());
  cached = host;
}

void uploadKernelIfNeededFloat(const std::vector<double> &host,
                               std::vector<float> &cached,
                               DeviceBuffer<float> &device) {
  std::vector<float> hostF(host.begin(), host.end());
  if (cached == hostF)
    return;
  device.upload(hostF.data(), hostF.size());
  cached = std::move(hostF);
}

void launchVolumeKernel(CUfunction function, void **args,
                        unsigned long long total) {
  const unsigned int blockSize = kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1,
                                   0, nullptr, args, nullptr));
}

void launchSeparableConvKernel(CUfunction function, const double *input,
                               double *output, int Nx, int Ny, int Nz,
                               const double *kernel, int radius,
                               unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr inputPtr = reinterpret_cast<CUdeviceptr>(input);
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  CUdeviceptr kernelPtr = reinterpret_cast<CUdeviceptr>(kernel);
  void *args[] = {&inputPtr, &outputPtr, &Nx, &Ny, &Nz, &kernelPtr, &radius,
                  &total};
  launchVolumeKernel(function, args, total);
}

void launchSeparableConvKernelFloat(CUfunction function, const float *input,
                                    float *output, int Nx, int Ny, int Nz,
                                    const float *kernel, int radius,
                                    unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr inputPtr = reinterpret_cast<CUdeviceptr>(input);
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  CUdeviceptr kernelPtr = reinterpret_cast<CUdeviceptr>(kernel);
  void *args[] = {&inputPtr, &outputPtr, &Nx, &Ny, &Nz, &kernelPtr, &radius,
                  &total};
  launchVolumeKernel(function, args, total);
}

template <typename T>
void launchSeparableConvAdjointKernel(CUfunction function, const T *input,
                                      T *output, int Nx, int Ny, int Nz,
                                      const T *kernel, int radius, int axis,
                                      unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr inputPtr = reinterpret_cast<CUdeviceptr>(input);
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  CUdeviceptr kernelPtr = reinterpret_cast<CUdeviceptr>(kernel);
  void *args[] = {&inputPtr, &outputPtr, &Nx, &Ny, &Nz, &kernelPtr,
                  &radius,   &axis,      &total};
  launchVolumeKernel(function, args, total);
}

void launchTvGradientKernel(const double *input, const uint8_t *mask,
                            const double *ct, bool hasMask, bool hasCt,
                            bool skipOutsideMask, int Nx, int Ny, int Nz,
                            double eps, double ctSigma, double ctWeightFloor,
                            double *output,
                            unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr inputPtr = reinterpret_cast<CUdeviceptr>(input);
  CUdeviceptr maskPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint8_t *>(mask));
  CUdeviceptr ctPtr = reinterpret_cast<CUdeviceptr>(const_cast<double *>(ct));
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  void *args[] = {&inputPtr, &maskPtr, &ctPtr, &hasMask, &hasCt,
                  &skipOutsideMask, &Nx, &Ny, &Nz, &eps, &ctSigma,
                  &ctWeightFloor, &outputPtr, &total};
  launchVolumeKernel(g_volumeKernelCache.tvGrad, args, total);
}

void launchRdpGradientKernel(const double *input, const uint8_t *mask,
                             const double *ct, bool hasMask, bool hasCt,
                             bool skipOutsideMask, int Nx, int Ny, int Nz,
                             double eps, double gamma, double ctSigma,
                             double ctWeightFloor, double *output,
                             unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr inputPtr = reinterpret_cast<CUdeviceptr>(input);
  CUdeviceptr maskPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint8_t *>(mask));
  CUdeviceptr ctPtr = reinterpret_cast<CUdeviceptr>(const_cast<double *>(ct));
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  void *args[] = {&inputPtr, &maskPtr, &ctPtr, &hasMask, &hasCt,
                  &skipOutsideMask, &Nx, &Ny, &Nz, &eps, &gamma, &ctSigma,
                  &ctWeightFloor, &outputPtr, &total};
  launchVolumeKernel(g_volumeKernelCache.rdpGrad, args, total);
}

void launchTvProxDualKernel(const double *u_bar, const double *ct,
                           double *px, double *py,
                           double *pz, double sigma, double lambda_val,
                           bool hasCt, double ctSigma, double ctWeightFloor,
                           int Nx, int Ny, int Nz,
                           unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr ubarPtr = reinterpret_cast<CUdeviceptr>(u_bar);
  CUdeviceptr ctPtr = reinterpret_cast<CUdeviceptr>(const_cast<double *>(ct));
  CUdeviceptr pxPtr = reinterpret_cast<CUdeviceptr>(px);
  CUdeviceptr pyPtr = reinterpret_cast<CUdeviceptr>(py);
  CUdeviceptr pzPtr = reinterpret_cast<CUdeviceptr>(pz);
  void *args[] = {&ubarPtr, &ctPtr, &pxPtr, &pyPtr, &pzPtr, &sigma,
                  &lambda_val, &hasCt, &ctSigma, &ctWeightFloor,
                  &Nx, &Ny, &Nz, &total};
  launchVolumeKernel(g_volumeKernelCache.tvProxDual, args, total);
}

void launchTvProxPrimalKernel(double *u, double *u_bar, const double *v,
                              const double *ct,
                              const double *px, const double *py,
                              const double *pz, double tau, double theta,
                              bool hasCt, double ctSigma,
                              double ctWeightFloor,
                              int Nx, int Ny, int Nz,
                              unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr uPtr = reinterpret_cast<CUdeviceptr>(u);
  CUdeviceptr ubarPtr = reinterpret_cast<CUdeviceptr>(u_bar);
  CUdeviceptr vPtr = reinterpret_cast<CUdeviceptr>(v);
  CUdeviceptr ctPtr = reinterpret_cast<CUdeviceptr>(const_cast<double *>(ct));
  CUdeviceptr pxPtr = reinterpret_cast<CUdeviceptr>(px);
  CUdeviceptr pyPtr = reinterpret_cast<CUdeviceptr>(py);
  CUdeviceptr pzPtr = reinterpret_cast<CUdeviceptr>(pz);
  void *args[] = {&uPtr, &ubarPtr, &vPtr, &ctPtr, &pxPtr, &pyPtr, &pzPtr,
                  &tau, &theta, &hasCt, &ctSigma, &ctWeightFloor,
                  &Nx, &Ny, &Nz, &total};
  launchVolumeKernel(g_volumeKernelCache.tvProxPrimal, args, total);
}

bool shouldUseCudaVolumeOps() {
  return cudaVolumeOpsRequested() &&
         !g_volumeOpsDisabled.load(std::memory_order_relaxed) &&
         isCudaAvailable();
}

} // namespace

bool isCudaVolumeOpsAvailable() { return shouldUseCudaVolumeOps(); }

bool computeAttenFactorsLineIntegralCUDA(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, int numAngles, int rotationSign,
    std::vector<uint16_t> &factors) {
  if (!shouldUseCudaVolumeOps())
    return false;
  const size_t voxels = static_cast<size_t>(nx) * ny * nz;
  if (nx <= 0 || ny <= 0 || nz <= 0 || numAngles <= 0 ||
      muMap.size() != voxels || !(dx > 0.0) || !(dy > 0.0))
    throw std::runtime_error("CUDA CTAC line-integral input mismatch");

  ensureVolumeKernelsLoaded();
  double step = 0.5 * std::min(dx, dy);
  int maxSteps = static_cast<int>(
                     std::ceil(std::hypot(nx * dx, ny * dy) / step)) +
                 2;
  const int sign = rotationSign < 0 ? -1 : 1;
  constexpr double pi = 3.14159265358979323846;
  std::vector<double> stepFx(static_cast<size_t>(numAngles));
  std::vector<double> stepFy(static_cast<size_t>(numAngles));
  for (int angle = 0; angle < numAngles; ++angle) {
    const double theta = sign * static_cast<double>(angle) *
                         (2.0 * pi / static_cast<double>(numAngles));
    stepFx[static_cast<size_t>(angle)] = -std::sin(theta) * step / dx;
    stepFy[static_cast<size_t>(angle)] = std::cos(theta) * step / dy;
  }

  DeviceBuffer<double> dMu;
  DeviceBuffer<double> dStepFx;
  DeviceBuffer<double> dStepFy;
  DeviceBuffer<uint16_t> dFactors;
  dMu.upload(muMap.data(), voxels);
  const size_t count = static_cast<size_t>(numAngles) * voxels;
  factors.resize(count);
  // ponytail: 16-angle batches cap the temporary factor buffer at ~29 MiB
  // for 108x108x80 beds; revisit only if profiling shows launch overhead.
  constexpr int kAngleBatch = 16;
  CUdeviceptr muPtr = reinterpret_cast<CUdeviceptr>(dMu.get());
  for (int firstAngle = 0; firstAngle < numAngles;
       firstAngle += kAngleBatch) {
    int batchAngles = std::min(kAngleBatch, numAngles - firstAngle);
    dStepFx.upload(stepFx.data() + firstAngle,
                   static_cast<size_t>(batchAngles));
    dStepFy.upload(stepFy.data() + firstAngle,
                   static_cast<size_t>(batchAngles));
    const size_t batchCount = static_cast<size_t>(batchAngles) * voxels;
    dFactors.ensure(batchCount);
    unsigned long long total = static_cast<unsigned long long>(batchCount);
    CUdeviceptr fxPtr = reinterpret_cast<CUdeviceptr>(dStepFx.get());
    CUdeviceptr fyPtr = reinterpret_cast<CUdeviceptr>(dStepFy.get());
    CUdeviceptr outPtr = reinterpret_cast<CUdeviceptr>(dFactors.get());
    void *args[] = {&muPtr, &fxPtr, &fyPtr, &nx, &ny, &nz, &batchAngles,
                    &step, &maxSteps, &outPtr, &total};
    launchVolumeKernel(g_volumeKernelCache.attenLineIntegral, args, total);
    SPECT_CUDRV_CHECK(cuCtxSynchronize());
    dFactors.download(factors.data() + static_cast<size_t>(firstAngle) * voxels,
                      batchCount);
  }
  return true;
}

bool applySeparableConv3DInplaceCUDA(std::vector<double> &f, int Nx, int Ny,
                                     int Nz, const std::vector<double> &kx,
                                     const std::vector<double> &ky,
                                     const std::vector<double> &kz,
                                     bool useFloat) {
  if (!shouldUseCudaVolumeOps())
    return false;
  ensureVolumeKernelsLoaded();

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::runtime_error("CUDA PSF input size mismatch");
  if (kx.empty() || ky.empty() || kz.empty())
    throw std::runtime_error("CUDA PSF kernels must be non-empty");

  auto &state = g_volumeOpsState;
  const unsigned long long totalCount =
      static_cast<unsigned long long>(total);
  const int radiusX = static_cast<int>(kx.size()) / 2;
  const int radiusY = static_cast<int>(ky.size()) / 2;
  const int radiusZ = static_cast<int>(kz.size()) / 2;

  if (cudaPsfUseFloat(useFloat)) {
    // Float path: CPU double->float conversion, halved GPU memory bandwidth
    std::vector<float> ff(f.begin(), f.end());
    state.primaryF.upload(ff.data(), total);
    state.secondaryF.ensure(total);
    uploadKernelIfNeededFloat(kx, state.cachedKxF, state.kxF);
    uploadKernelIfNeededFloat(ky, state.cachedKyF, state.kyF);
    uploadKernelIfNeededFloat(kz, state.cachedKzF, state.kzF);

    launchSeparableConvKernelFloat(g_volumeKernelCache.convXF,
                                   state.primaryF.get(),
                                   state.secondaryF.get(), Nx, Ny, Nz,
                                   state.kxF.get(), radiusX, totalCount);
    launchSeparableConvKernelFloat(g_volumeKernelCache.convYF,
                                   state.secondaryF.get(),
                                   state.primaryF.get(), Nx, Ny, Nz,
                                   state.kyF.get(), radiusY, totalCount);
    launchSeparableConvKernelFloat(g_volumeKernelCache.convZF,
                                   state.primaryF.get(),
                                   state.secondaryF.get(), Nx, Ny, Nz,
                                   state.kzF.get(), radiusZ, totalCount);
    SPECT_CUDRV_CHECK(cuCtxSynchronize());
    ff.resize(total);
    state.secondaryF.download(ff.data(), total);
    for (size_t i = 0; i < total; ++i)
      f[i] = static_cast<double>(ff[i]);
  } else {
    // Double path: original precision
    state.primary.upload(f.data(), total);
    state.secondary.ensure(total);
    uploadKernelIfNeeded(kx, state.cachedKx, state.kx);
    uploadKernelIfNeeded(ky, state.cachedKy, state.ky);
    uploadKernelIfNeeded(kz, state.cachedKz, state.kz);

    launchSeparableConvKernel(g_volumeKernelCache.convX, state.primary.get(),
                              state.secondary.get(), Nx, Ny, Nz,
                              state.kx.get(), radiusX, totalCount);
    launchSeparableConvKernel(g_volumeKernelCache.convY, state.secondary.get(),
                              state.primary.get(), Nx, Ny, Nz, state.ky.get(),
                              radiusY, totalCount);
    launchSeparableConvKernel(g_volumeKernelCache.convZ, state.primary.get(),
                              state.secondary.get(), Nx, Ny, Nz, state.kz.get(),
                              radiusZ, totalCount);
    SPECT_CUDRV_CHECK(cuCtxSynchronize());
    state.secondary.download(f.data(), total);
  }
  return true;
}

namespace {

void enqueueCastorConv3DFloatDeviceImpl(
    float *primary, float *secondary, int Nx, int Ny, int Nz,
    const std::vector<double> &kx, const std::vector<double> &ky,
    const std::vector<double> &kz) {
  if (primary == nullptr || secondary == nullptr || primary == secondary)
    throw std::runtime_error(
        "CUDA CASToR PSF requires distinct non-null device buffers");
  if (Nx <= 0 || Ny <= 0 || Nz <= 0 || kx.empty() || ky.empty() || kz.empty())
    throw std::runtime_error("CUDA CASToR PSF device input mismatch");

  ensureVolumeKernelsLoaded();
  auto &state = g_volumeOpsState;
  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  unsigned long long totalCount = static_cast<unsigned long long>(total);
  const int radiusX = static_cast<int>(kx.size()) / 2;
  const int radiusY = static_cast<int>(ky.size()) / 2;
  const int radiusZ = static_cast<int>(kz.size()) / 2;
  uploadKernelIfNeededFloat(kx, state.cachedKxF, state.kxF);
  uploadKernelIfNeededFloat(ky, state.cachedKyF, state.kyF);
  uploadKernelIfNeededFloat(kz, state.cachedKzF, state.kzF);
  const std::vector<double> znormHost = filters::makeCastorZNorm1D(kz, Nz);
  uploadKernelIfNeededFloat(znormHost, state.cachedZnormF, state.znormF);

  launchSeparableConvKernelFloat(g_volumeKernelCache.convXZeroF, primary,
                                 secondary, Nx, Ny, Nz, state.kxF.get(),
                                 radiusX, totalCount);
  launchSeparableConvKernelFloat(g_volumeKernelCache.convYZeroF, secondary,
                                 primary, Nx, Ny, Nz, state.kyF.get(),
                                 radiusY, totalCount);
  launchSeparableConvKernelFloat(g_volumeKernelCache.convZZeroF, primary,
                                 secondary, Nx, Ny, Nz, state.kzF.get(),
                                 radiusZ, totalCount);
  CUdeviceptr dataPtr = reinterpret_cast<CUdeviceptr>(secondary);
  CUdeviceptr normPtr =
      reinterpret_cast<CUdeviceptr>(state.znormF.get());
  void *args[] = {&dataPtr, &normPtr, &Nx, &Ny, &Nz, &totalCount};
  launchVolumeKernel(g_volumeKernelCache.divZnormF, args, totalCount);
}

bool applyCastorConv3DInplaceCUDAImpl(std::vector<double> &f, int Nx, int Ny,
                                      int Nz,
                                      const std::vector<double> &kx,
                                      const std::vector<double> &ky,
                                      const std::vector<double> &kz,
                                      bool useFloat,
                                      bool forceFloat32Staging) {
  if (!shouldUseCudaVolumeOps())
    return false;
  ensureVolumeKernelsLoaded();

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::runtime_error("CUDA CASToR PSF input size mismatch");
  if (kx.empty() || ky.empty() || kz.empty())
    throw std::runtime_error("CUDA CASToR PSF kernels must be non-empty");

  auto &state = g_volumeOpsState;
  unsigned long long totalCount = static_cast<unsigned long long>(total);
  const int radiusX = static_cast<int>(kx.size()) / 2;
  const int radiusY = static_cast<int>(ky.size()) / 2;
  const int radiusZ = static_cast<int>(kz.size()) / 2;
  const std::vector<double> znormHost =
      filters::makeCastorZNorm1D(kz, Nz);

  const bool useFloatStaging =
      forceFloat32Staging || cudaPsfUseFloat(useFloat);
  if (useFloatStaging) {
    std::vector<float> ff(f.begin(), f.end());
    state.primaryF.upload(ff.data(), total);
    state.secondaryF.ensure(total);
    enqueueCastorConv3DFloatDeviceImpl(
        state.primaryF.get(), state.secondaryF.get(), Nx, Ny, Nz, kx, ky, kz);
    SPECT_CUDRV_CHECK(cuCtxSynchronize());
    ff.resize(total);
    state.secondaryF.download(ff.data(), total);
    for (size_t i = 0; i < total; ++i)
      f[i] = static_cast<double>(ff[i]);
  } else {
    state.primary.upload(f.data(), total);
    state.secondary.ensure(total);
    uploadKernelIfNeeded(kx, state.cachedKx, state.kx);
    uploadKernelIfNeeded(ky, state.cachedKy, state.ky);
    uploadKernelIfNeeded(kz, state.cachedKz, state.kz);
    state.znorm.upload(znormHost.data(), znormHost.size());

    launchSeparableConvKernel(g_volumeKernelCache.convXZero,
                              state.primary.get(), state.secondary.get(),
                              Nx, Ny, Nz, state.kx.get(), radiusX, totalCount);
    launchSeparableConvKernel(g_volumeKernelCache.convYZero,
                              state.secondary.get(), state.primary.get(),
                              Nx, Ny, Nz, state.ky.get(), radiusY, totalCount);
    launchSeparableConvKernel(g_volumeKernelCache.convZZero,
                              state.primary.get(), state.secondary.get(),
                              Nx, Ny, Nz, state.kz.get(), radiusZ, totalCount);
    {
      CUdeviceptr dataPtr =
          reinterpret_cast<CUdeviceptr>(state.secondary.get());
      CUdeviceptr normPtr =
          reinterpret_cast<CUdeviceptr>(state.znorm.get());
      void *args[] = {&dataPtr, &normPtr, &Nx, &Ny, &Nz, &totalCount};
      launchVolumeKernel(g_volumeKernelCache.divZnorm, args, totalCount);
    }
    SPECT_CUDRV_CHECK(cuCtxSynchronize());
    state.secondary.download(f.data(), total);
  }
  return true;
}

} // namespace

bool enqueueCastorConv3DFloatDeviceCUDA(
    float *primary, float *secondary, int Nx, int Ny, int Nz,
    const std::vector<double> &kx, const std::vector<double> &ky,
    const std::vector<double> &kz) {
  if (!shouldUseCudaVolumeOps())
    return false;
  enqueueCastorConv3DFloatDeviceImpl(primary, secondary, Nx, Ny, Nz, kx, ky,
                                     kz);
  return true;
}

bool applyCastorConv3DInplaceCUDA(std::vector<double> &f, int Nx, int Ny,
                                  int Nz, const std::vector<double> &kx,
                                  const std::vector<double> &ky,
                                  const std::vector<double> &kz,
                                  bool useFloat) {
  return applyCastorConv3DInplaceCUDAImpl(f, Nx, Ny, Nz, kx, ky, kz,
                                          useFloat, false);
}

bool applyCastorConv3DInplaceCUDAForcedFloat32Staging(
    std::vector<double> &f, int Nx, int Ny, int Nz,
    const std::vector<double> &kx, const std::vector<double> &ky,
    const std::vector<double> &kz) {
  return applyCastorConv3DInplaceCUDAImpl(f, Nx, Ny, Nz, kx, ky, kz, true,
                                          true);
}

bool applySeparableConv3DAdjointInplaceCUDA(
    std::vector<double> &f, int Nx, int Ny, int Nz,
    const std::vector<double> &kx, const std::vector<double> &ky,
    const std::vector<double> &kz, bool useFloat) {
  if (!shouldUseCudaVolumeOps())
    return false;
  ensureVolumeKernelsLoaded();

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::runtime_error("CUDA PSF adjoint input size mismatch");
  if (kx.empty() || ky.empty() || kz.empty())
    throw std::runtime_error("CUDA PSF adjoint kernels must be non-empty");

  auto &state = g_volumeOpsState;
  const auto totalCount = static_cast<unsigned long long>(total);
  const int radiusX = static_cast<int>(kx.size()) / 2;
  const int radiusY = static_cast<int>(ky.size()) / 2;
  const int radiusZ = static_cast<int>(kz.size()) / 2;

  if (cudaPsfUseFloat(useFloat)) {
    std::vector<float> ff(f.begin(), f.end());
    state.primaryF.upload(ff.data(), total);
    state.secondaryF.ensure(total);
    uploadKernelIfNeededFloat(kx, state.cachedKxF, state.kxF);
    uploadKernelIfNeededFloat(ky, state.cachedKyF, state.kyF);
    uploadKernelIfNeededFloat(kz, state.cachedKzF, state.kzF);
    launchSeparableConvAdjointKernel(
        g_volumeKernelCache.convAdjointF, state.primaryF.get(),
        state.secondaryF.get(), Nx, Ny, Nz, state.kzF.get(), radiusZ, 2,
        totalCount);
    launchSeparableConvAdjointKernel(
        g_volumeKernelCache.convAdjointF, state.secondaryF.get(),
        state.primaryF.get(), Nx, Ny, Nz, state.kyF.get(), radiusY, 1,
        totalCount);
    launchSeparableConvAdjointKernel(
        g_volumeKernelCache.convAdjointF, state.primaryF.get(),
        state.secondaryF.get(), Nx, Ny, Nz, state.kxF.get(), radiusX, 0,
        totalCount);
    SPECT_CUDRV_CHECK(cuCtxSynchronize());
    ff.resize(total);
    state.secondaryF.download(ff.data(), total);
    for (size_t i = 0; i < total; ++i)
      f[i] = static_cast<double>(ff[i]);
  } else {
    state.primary.upload(f.data(), total);
    state.secondary.ensure(total);
    uploadKernelIfNeeded(kx, state.cachedKx, state.kx);
    uploadKernelIfNeeded(ky, state.cachedKy, state.ky);
    uploadKernelIfNeeded(kz, state.cachedKz, state.kz);
    launchSeparableConvAdjointKernel(
        g_volumeKernelCache.convAdjoint, state.primary.get(),
        state.secondary.get(), Nx, Ny, Nz, state.kz.get(), radiusZ, 2,
        totalCount);
    launchSeparableConvAdjointKernel(
        g_volumeKernelCache.convAdjoint, state.secondary.get(),
        state.primary.get(), Nx, Ny, Nz, state.ky.get(), radiusY, 1,
        totalCount);
    launchSeparableConvAdjointKernel(
        g_volumeKernelCache.convAdjoint, state.primary.get(),
        state.secondary.get(), Nx, Ny, Nz, state.kx.get(), radiusX, 0,
        totalCount);
    SPECT_CUDRV_CHECK(cuCtxSynchronize());
    state.secondary.download(f.data(), total);
  }
  return true;
}

bool computeRDPGradientCUDA(const std::vector<double> &f, int Nx, int Ny,
                            int Nz, double delta, double gamma,
                            const std::vector<uint8_t> *mask,
                            bool skipOutsideMask,
                            std::vector<double> &gradR,
                            const std::vector<double> *ctImage,
                            double ctSigma, double ctWeightFloor) {
  if (!shouldUseCudaVolumeOps())
    return false;

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::runtime_error("CUDA RDP input size mismatch");
  if (mask && mask->size() != total)
    throw std::runtime_error("CUDA RDP mask size mismatch");
  if (ctImage && ctImage->size() != total)
    throw std::runtime_error("CUDA RDP registered CT size mismatch");
  if (ctImage && !(ctSigma > 0.0))
    throw std::runtime_error("CUDA RDP CT sigma must be positive");
  if (ctWeightFloor < 0.0 || ctWeightFloor > 1.0)
    throw std::runtime_error("CUDA RDP CT weight floor must be in [0,1]");

  const double eps = std::max(delta, 1e-6);
  gradR.resize(total);

  auto &state = g_volumeOpsState;
  state.primary.upload(f.data(), total);
  state.secondary.ensure(total);
  const bool hasMask = mask != nullptr;
  const bool hasCt = ctImage != nullptr;
  if (hasMask)
    state.mask.upload(mask->data(), total);
  if (hasCt)
    state.ct.upload(ctImage->data(), total);

  launchRdpGradientKernel(state.primary.get(),
                          hasMask ? state.mask.get() : nullptr,
                          hasCt ? state.ct.get() : nullptr, hasMask, hasCt,
                          skipOutsideMask, Nx, Ny, Nz, eps, gamma, ctSigma,
                          ctWeightFloor,
                          state.secondary.get(),
                          static_cast<unsigned long long>(total));
  SPECT_CUDRV_CHECK(cuCtxSynchronize());
  state.secondary.download(gradR.data(), total);
  return true;
}

bool computeTVGradientCUDA(const std::vector<double> &f, int Nx, int Ny,
                           int Nz, double eps,
                           const std::vector<uint8_t> *mask,
                           bool skipOutsideMask,
                           std::vector<double> &gradR,
                           const std::vector<double> *ctImage,
                           double ctSigma, double ctWeightFloor) {
  if (!shouldUseCudaVolumeOps())
    return false;

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::runtime_error("CUDA TV input size mismatch");
  if (mask && mask->size() != total)
    throw std::runtime_error("CUDA TV mask size mismatch");
  if (ctImage && ctImage->size() != total)
    throw std::runtime_error("CUDA TV registered CT size mismatch");
  if (ctImage && !(ctSigma > 0.0))
    throw std::runtime_error("CUDA TV CT sigma must be positive");
  if (ctWeightFloor < 0.0 || ctWeightFloor > 1.0)
    throw std::runtime_error("CUDA TV CT weight floor must be in [0,1]");

  const double epsVal = std::max(eps, 1e-12);
  gradR.resize(total);

  auto &state = g_volumeOpsState;
  state.primary.upload(f.data(), total);
  state.secondary.ensure(total);
  const bool hasMask = mask != nullptr;
  const bool hasCt = ctImage != nullptr;
  if (hasMask)
    state.mask.upload(mask->data(), total);
  if (hasCt)
    state.ct.upload(ctImage->data(), total);

  launchTvGradientKernel(state.primary.get(),
                         hasMask ? state.mask.get() : nullptr,
                         hasCt ? state.ct.get() : nullptr, hasMask, hasCt,
                         skipOutsideMask, Nx, Ny, Nz, epsVal, ctSigma,
                         ctWeightFloor,
                         state.secondary.get(),
                         static_cast<unsigned long long>(total));
  SPECT_CUDRV_CHECK(cuCtxSynchronize());
  state.secondary.download(gradR.data(), total);
  return true;
}

bool computeRegGradientCUDA(const std::vector<double> &f, int Nx, int Ny,
                            int Nz,
                            const spect::RegConf &reg,
                            const std::vector<uint8_t> *mask,
                            std::vector<double> &gradR) {
  using spect::RegType;
  switch (reg.type) {
  case RegType::RDP:
    return computeRDPGradientCUDA(f, Nx, Ny, Nz, reg.delta, reg.gamma, mask,
                                  reg.skip_outside_mask, gradR, reg.ct_image,
                                  reg.ct_sigma, reg.ct_weight_floor);
  case RegType::TVIsotropic:
    return computeTVGradientCUDA(f, Nx, Ny, Nz, reg.tv_eps, mask,
                                 reg.skip_outside_mask, gradR, reg.ct_image,
                                 reg.ct_sigma, reg.ct_weight_floor);
  default:
    return false;
  }
}

bool computeTVProxCUDA(const std::vector<double> &v, int Nx, int Ny, int Nz,
                       double lambda, int innerIters,
                       std::vector<double> &u_out,
                       const std::vector<double> *ctImage,
                       double ctSigma, double ctWeightFloor) {
  if (!shouldUseCudaVolumeOps())
    return false;

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (v.size() != total)
    throw std::runtime_error("CUDA TV prox input size mismatch");
  if (ctImage && ctImage->size() != total)
    throw std::runtime_error("CUDA TV prox registered CT size mismatch");
  if (ctImage && !(ctSigma > 0.0))
    throw std::runtime_error("CUDA TV prox CT sigma must be positive");
  if (ctWeightFloor < 0.0 || ctWeightFloor > 1.0)
    throw std::runtime_error("CUDA TV prox CT weight floor must be in [0,1]");
  if (innerIters <= 0)
    innerIters = 1;

  ensureVolumeKernelsLoaded();

  // Chambolle-Pock step sizes: sigma*tau*||grad||^2 < 1, ||grad||^2 <= 12
  const double sigma = 1.0 / std::sqrt(12.0);
  const double tau = 1.0 / std::sqrt(12.0);
  const double theta = 1.0;
  const unsigned long long totalCount =
      static_cast<unsigned long long>(total);

  auto &state = g_volumeOpsState;
  const bool hasCt = ctImage != nullptr;
  if (hasCt)
    state.ct.upload(ctImage->data(), total);

  // Upload v (fixed target)
  state.tvV.upload(v.data(), total);

  // Initialize u = v, u_bar = v
  state.tvU.upload(v.data(), total);
  state.tvUbar.upload(v.data(), total);

  // Initialize dual variables p = 0
  std::vector<double> zeros(total, 0.0);
  state.tvPx.upload(zeros.data(), total);
  state.tvPy.upload(zeros.data(), total);
  state.tvPz.upload(zeros.data(), total);

  // Chambolle-Pock inner iterations (all on GPU)
  for (int k = 0; k < innerIters; ++k) {
    // Dual update: p = proj(p + sigma * grad(u_bar))
    launchTvProxDualKernel(state.tvUbar.get(),
                           hasCt ? state.ct.get() : nullptr, state.tvPx.get(),
                           state.tvPy.get(), state.tvPz.get(), sigma,
                           lambda, hasCt, ctSigma, ctWeightFloor,
                           Nx, Ny, Nz, totalCount);

    // Primal update: u_new, u_bar extrapolation
    launchTvProxPrimalKernel(state.tvU.get(), state.tvUbar.get(),
                             state.tvV.get(),
                             hasCt ? state.ct.get() : nullptr, state.tvPx.get(),
                             state.tvPy.get(), state.tvPz.get(), tau,
                             theta, hasCt, ctSigma, ctWeightFloor,
                             Nx, Ny, Nz, totalCount);
  }

  SPECT_CUDRV_CHECK(cuCtxSynchronize());
  u_out.resize(total);
  state.tvU.download(u_out.data(), total);
  return true;
}

} // namespace spect::cuda

#endif // CUDA_ENABLED
