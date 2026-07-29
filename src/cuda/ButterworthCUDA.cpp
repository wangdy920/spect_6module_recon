#include "filters/ButterworthFilter.h"

#if defined(SPECT_BUTTERWORTH_HAVE_CUFFT)

#include "cuda/CudaConfig.h"
#include "cuda/CudaDriverHelpers.h"
#include "cuda/CudaLaunchConfig.h"
#include "cuda/CudaNvrtcArch.h"
#include "cuda/CudaNvrtcLoader.h"

#if CUDA_ENABLED

#include <cuda.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spect::filters {
namespace {

#define CUFFT_CHECK(call) SPECT_CUFFT_CHECK(call)
#define CUDRV_CHECK(call) SPECT_CUDRV_CHECK(call)

// --- NVRTC kernel: single mask-apply kernel for float R2C/C2R path ---

struct ButterworthKernelCache {
  CUmodule module = nullptr;
  CUfunction maskFunction = nullptr;
  int device = -1;
  int computeMajor = -1;
  int computeMinor = -1;
  std::mutex mutex;

  ~ButterworthKernelCache() {
    if (module)
      cuModuleUnload(module);
  }
};

ButterworthKernelCache g_butterworthKernel;

// --- Float R2C/C2R cache ---

struct CufftCache {
  int Nx = 0;
  int Ny = 0;
  int Nz = 0;
  cufftHandle planFwd = 0;
  cufftHandle planInv = 0;
  cufftComplex *deviceComplex = nullptr; // Nz * Ny * (Nx/2+1) complex floats
  float *deviceReal = nullptr;           // Nz * Ny * Nx floats
  float *deviceMask = nullptr;           // Nz * Ny * (Nx/2+1) floats
  size_t realCount = 0;
  size_t complexCount = 0;
  ButterworthConf bw;
  Spacing3 sp;
  std::vector<float> hostFloat; // reusable double<->float staging

  ~CufftCache() {
    if (planFwd != 0)
      cufftDestroy(planFwd);
    if (planInv != 0)
      cufftDestroy(planInv);
    if (deviceComplex)
      cudaFree(deviceComplex);
    if (deviceReal)
      cudaFree(deviceReal);
    if (deviceMask)
      cudaFree(deviceMask);
  }
};

thread_local CufftCache cache;

inline bool approxEq(double a, double b) {
  return std::abs(a - b) <= 1e-9 * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

inline bool needRebuildPlan(int Nx, int Ny, int Nz) {
  return cache.planFwd == 0 || cache.Nx != Nx || cache.Ny != Ny || cache.Nz != Nz;
}

inline bool needRebuildMask(int Nx, int Ny, int Nz, Spacing3 sp,
                            const ButterworthConf &bw) {
  return needRebuildPlan(Nx, Ny, Nz) || cache.bw.type != bw.type ||
         cache.bw.order != bw.order || !approxEq(cache.bw.cutoff_lo_frac, bw.cutoff_lo_frac) ||
         !approxEq(cache.bw.cutoff_hi_frac, bw.cutoff_hi_frac) ||
         !approxEq(cache.sp.dx, sp.dx) || !approxEq(cache.sp.dy, sp.dy) ||
         !approxEq(cache.sp.dz, sp.dz);
}

// Build the half-spectrum mask for R2C layout: [Nz][Ny][Nx/2+1], float.
// Uses the same formula as make_butterworth_mask_3d but only x=0..Nx/2.
std::vector<float> makeHalfMaskFloat(int Nx, int Ny, int Nz,
                                     const ButterworthConf &bw) {
  const int NxH = Nx / 2 + 1;
  const size_t halfTotal = static_cast<size_t>(Nz) * Ny * NxH;
  std::vector<float> mask(halfTotal, 1.0f);
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
    return 1.0 / (1.0 + std::pow(r / fc, 2 * order));
  };
  auto highpass = [&lowpass](double r, double fc) { return 1.0 - lowpass(r, fc); };

  for (int z = 0; z < Nz; ++z) {
    const int kz = (z <= Nz / 2) ? z : z - Nz;
    const double fz = static_cast<double>(kz) / denomZ;
    for (int y = 0; y < Ny; ++y) {
      const int ky = (y <= Ny / 2) ? y : y - Ny;
      const double fy = static_cast<double>(ky) / denomY;
      for (int x = 0; x < NxH; ++x) {
        // R2C half-spectrum: x covers 0..Nx/2 (non-negative frequencies only)
        const double fx = static_cast<double>(x) / denomX;
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
        mask[static_cast<size_t>(z) * Ny * NxH +
             static_cast<size_t>(y) * NxH + x] = static_cast<float>(value);
      }
    }
  }
  return mask;
}

void ensureButterworthKernelsLoaded() {
  CUDA_CHECK(cudaFree(nullptr));
  CUDRV_CHECK(cuInit(0));

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));

  std::lock_guard<std::mutex> lock(g_butterworthKernel.mutex);
  if (g_butterworthKernel.maskFunction &&
      g_butterworthKernel.device == device &&
      g_butterworthKernel.computeMajor == props.major &&
      g_butterworthKernel.computeMinor == props.minor) {
    return;
  }

  // Single kernel: multiply float2 complex data by float mask
  const char *source = R"CU(
extern "C" __global__ void apply_butterworth_mask_f(
    float2* data, const float* mask, unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const float scale = mask[index];
  data[index].x *= scale;
  data[index].y *= scale;
}
)CU";

  const std::string arch = spect::cuda::nvrtcGpuArchitectureOption(props);
  const char *options[] = {arch.c_str()};
  const std::string ptx = spect::cuda::compileNvrtcProgramToPtx(
      source, "butterworth_mask.cu", 1, options);

  CUcontext context = nullptr;
  CUDRV_CHECK(cuCtxGetCurrent(&context));
  if (context == nullptr) {
    throw std::runtime_error("CUDA context is not initialized for Butterworth mask kernel.");
  }

  if (g_butterworthKernel.module) {
    CUDRV_CHECK(cuModuleUnload(g_butterworthKernel.module));
    g_butterworthKernel.module = nullptr;
    g_butterworthKernel.maskFunction = nullptr;
  }

  spect::cuda::loadNvrtcPtxModule(&g_butterworthKernel.module, ptx);
  CUDRV_CHECK(cuModuleGetFunction(&g_butterworthKernel.maskFunction,
                                  g_butterworthKernel.module,
                                  "apply_butterworth_mask_f"));
  g_butterworthKernel.device = device;
  g_butterworthKernel.computeMajor = props.major;
  g_butterworthKernel.computeMinor = props.minor;
}

void applyMaskOnDevice(cufftComplex *deviceBuffer, float *deviceMask,
                       size_t total) {
  ensureButterworthKernelsLoaded();

  CUdeviceptr dataPtr = reinterpret_cast<CUdeviceptr>(deviceBuffer);
  CUdeviceptr maskPtr = reinterpret_cast<CUdeviceptr>(deviceMask);
  unsigned long long totalCount = static_cast<unsigned long long>(total);
  void *args[] = {&dataPtr, &maskPtr, &totalCount};

  const unsigned int blockSize = spect::cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  CUDRV_CHECK(cuLaunchKernel(g_butterworthKernel.maskFunction, gridSize, 1, 1,
                             blockSize, 1, 1, 0, nullptr, args, nullptr));
}

} // namespace

void apply_butterworth_inplace_cufft(std::vector<double> &f, int Nx, int Ny,
                                     int Nz, Spacing3 sp,
                                     const ButterworthConf &bw, int) {
  if (!bw.enabled)
    return;

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::runtime_error("Butterworth cuFFT input size mismatch");
  if (!spect::cuda::isCudaAvailable())
    throw std::runtime_error("Butterworth cuFFT backend requested without an available CUDA device.");

  const int NxH = Nx / 2 + 1;
  const size_t complexCount = static_cast<size_t>(Nz) * Ny * NxH;

  if (needRebuildPlan(Nx, Ny, Nz)) {
    if (cache.planFwd != 0) {
      cufftDestroy(cache.planFwd);
      cache.planFwd = 0;
    }
    if (cache.planInv != 0) {
      cufftDestroy(cache.planInv);
      cache.planInv = 0;
    }
    if (cache.deviceComplex) {
      CUDA_CHECK(cudaFree(cache.deviceComplex));
      cache.deviceComplex = nullptr;
    }
    if (cache.deviceReal) {
      CUDA_CHECK(cudaFree(cache.deviceReal));
      cache.deviceReal = nullptr;
    }
    if (cache.deviceMask) {
      CUDA_CHECK(cudaFree(cache.deviceMask));
      cache.deviceMask = nullptr;
    }

    cache.Nx = Nx;
    cache.Ny = Ny;
    cache.Nz = Nz;
    cache.realCount = total;
    cache.complexCount = complexCount;

    // cuFFT R2C needs the real buffer padded to 2*(Nx/2+1) stride per row.
    // Use separate real and complex buffers for clarity.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&cache.deviceReal),
                          total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&cache.deviceComplex),
                          complexCount * sizeof(cufftComplex)));
    CUFFT_CHECK(cufftPlan3d(&cache.planFwd, Nz, Ny, Nx, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&cache.planInv, Nz, Ny, Nx, CUFFT_C2R));
  }

  if (needRebuildMask(Nx, Ny, Nz, sp, bw)) {
    auto halfMask = makeHalfMaskFloat(Nx, Ny, Nz, bw);
    if (cache.deviceMask) {
      CUDA_CHECK(cudaFree(cache.deviceMask));
      cache.deviceMask = nullptr;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&cache.deviceMask),
                          complexCount * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(cache.deviceMask, halfMask.data(),
                          complexCount * sizeof(float), cudaMemcpyHostToDevice));
    cache.bw = bw;
    cache.sp = sp;
  }

  // Convert double -> float on CPU (avoids uploading 2x data as double)
  cache.hostFloat.resize(total);
  for (size_t i = 0; i < total; ++i)
    cache.hostFloat[i] = static_cast<float>(f[i]);

  // H2D: upload float real data (half the bytes of double)
  CUDA_CHECK(cudaMemcpy(cache.deviceReal, cache.hostFloat.data(),
                        total * sizeof(float), cudaMemcpyHostToDevice));

  // Forward R2C: real[Nz][Ny][Nx] -> complex[Nz][Ny][Nx/2+1]
  CUFFT_CHECK(cufftExecR2C(cache.planFwd, cache.deviceReal, cache.deviceComplex));

  // Apply frequency-domain mask
  applyMaskOnDevice(cache.deviceComplex, cache.deviceMask, complexCount);

  // Inverse C2R: complex[Nz][Ny][Nx/2+1] -> real[Nz][Ny][Nx]
  CUFFT_CHECK(cufftExecC2R(cache.planInv, cache.deviceComplex, cache.deviceReal));

  // D2H: download float result
  CUDA_CHECK(cudaMemcpy(cache.hostFloat.data(), cache.deviceReal,
                        total * sizeof(float), cudaMemcpyDeviceToHost));

  // Convert float -> double with scale and non-negativity clamp
  const double scale = 1.0 / static_cast<double>(total);
  for (size_t i = 0; i < total; ++i) {
    const double value = static_cast<double>(cache.hostFloat[i]) * scale;
    f[i] = value < 0.0 ? 0.0 : value;
  }
}

} // namespace spect::filters

#endif // CUDA_ENABLED

#endif // defined(SPECT_BUTTERWORTH_HAVE_CUFFT)
