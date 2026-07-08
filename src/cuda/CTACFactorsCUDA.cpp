#include "cuda/CTACFactorsCUDA.h"

#if CUDA_ENABLED

#include "cuda/CudaDriverHelpers.h"
#include "cuda/CudaLaunchConfig.h"
#include "cuda/CudaNvrtcLoader.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace spect::cuda {
namespace {

struct CTACFactorKernelCache {
  CUmodule module = nullptr;
  CUfunction lineIntegral = nullptr;
  int device = -1;
  int computeMajor = -1;
  int computeMinor = -1;
  std::mutex mutex;

  ~CTACFactorKernelCache() {
    if (module != nullptr)
      cuModuleUnload(module);
  }
};

CTACFactorKernelCache g_ctacFactorKernelCache;

template <typename T> class DeviceBuffer {
public:
  ~DeviceBuffer() {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
  }

  void allocate(size_t count) {
    if (count == 0)
      return;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_),
                          count * sizeof(T)));
    count_ = count;
  }

  void upload(const T *host, size_t count) {
    if (count > count_)
      throw std::runtime_error("CUDA CTAC upload exceeds device buffer size.");
    if (count > 0)
      CUDA_CHECK(
          cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void download(T *host, size_t count) const {
    if (count > count_)
      throw std::runtime_error("CUDA CTAC download exceeds device buffer size.");
    if (count > 0)
      CUDA_CHECK(
          cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  T *get() { return ptr_; }

private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

void ensureCTACFactorKernelLoaded() {
  CUDA_CHECK(cudaFree(nullptr));
  SPECT_CUDRV_CHECK(cuInit(0));

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));

  std::lock_guard<std::mutex> lock(g_ctacFactorKernelCache.mutex);
  if (g_ctacFactorKernelCache.lineIntegral != nullptr &&
      g_ctacFactorKernelCache.device == device &&
      g_ctacFactorKernelCache.computeMajor == props.major &&
      g_ctacFactorKernelCache.computeMinor == props.minor) {
    return;
  }

  const char *source = R"CU(
static constexpr double kPi = 3.14159265358979323846;
static constexpr double kAttenUint16Scale = 65535.0;

extern "C" __device__ __forceinline__ double clamp_double(double value,
                                                           double lo,
                                                           double hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

extern "C" __device__ __forceinline__ double sample_mu_bilinear(
    const double* slice, int nx, int ny, double fx, double fy) {
  if (fx <= -1.0 || fy <= -1.0 || fx >= static_cast<double>(nx) ||
      fy >= static_cast<double>(ny)) {
    return 0.0;
  }
  const double cx = clamp_double(fx, 0.0, static_cast<double>(nx - 1));
  const double cy = clamp_double(fy, 0.0, static_cast<double>(ny - 1));
  const int ix0 = static_cast<int>(floor(cx));
  const int iy0 = static_cast<int>(floor(cy));
  const int ix1 = ix0 + 1 < nx ? ix0 + 1 : nx - 1;
  const int iy1 = iy0 + 1 < ny ? iy0 + 1 : ny - 1;
  const double tx = cx - static_cast<double>(ix0);
  const double ty = cy - static_cast<double>(iy0);
  const unsigned long long row0 =
      static_cast<unsigned long long>(iy0) * static_cast<unsigned long long>(nx);
  const unsigned long long row1 =
      static_cast<unsigned long long>(iy1) * static_cast<unsigned long long>(nx);
  const double v00 = slice[row0 + static_cast<unsigned long long>(ix0)];
  const double v10 = slice[row0 + static_cast<unsigned long long>(ix1)];
  const double v01 = slice[row1 + static_cast<unsigned long long>(ix0)];
  const double v11 = slice[row1 + static_cast<unsigned long long>(ix1)];
  return (1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) +
         ty * ((1.0 - tx) * v01 + tx * v11);
}

extern "C" __device__ __forceinline__ unsigned short quantise_factor(
    double factor) {
  const double scaled =
      clamp_double(factor, 0.0, 1.0) * kAttenUint16Scale;
  unsigned int q = static_cast<unsigned int>(floor(scaled + 0.5));
  if (q > 65535u) {
    q = 65535u;
  }
  return static_cast<unsigned short>(q);
}

extern "C" __global__ void compute_ctac_line_integral_factors(
    const double* muMap, int nx, int ny, int nz, double dx, double dy,
    int numAngles, int rotationSign, unsigned long long voxels,
    unsigned short* factors, unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total || nx <= 0 || ny <= 0 || nz <= 0 || numAngles <= 0) {
    return;
  }

  const int angle = static_cast<int>(index / voxels);
  const unsigned long long voxel = index -
      static_cast<unsigned long long>(angle) * voxels;
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const int iz = static_cast<int>(voxel / plane);
  const unsigned long long inPlane =
      voxel - static_cast<unsigned long long>(iz) * plane;
  const int iy =
      static_cast<int>(inPlane / static_cast<unsigned long long>(nx));
  const int ix = static_cast<int>(
      inPlane - static_cast<unsigned long long>(iy) *
                    static_cast<unsigned long long>(nx));

  const int sign = rotationSign < 0 ? -1 : 1;
  const double theta = static_cast<double>(sign) *
                       static_cast<double>(angle) *
                       (2.0 * kPi / static_cast<double>(numAngles));
  const double step = 0.5 * (dx < dy ? dx : dy);
  const double maxX = static_cast<double>(nx) * dx;
  const double maxY = static_cast<double>(ny) * dy;
  const int maxSteps = static_cast<int>(ceil(sqrt(maxX * maxX + maxY * maxY) /
                                             step)) +
                       2;
  const double rayX = -sin(theta);
  const double rayY = cos(theta);
  const double stepFx = rayX * step / dx;
  const double stepFy = rayY * step / dy;

  const double* slice = muMap + static_cast<unsigned long long>(iz) * plane;
  double fx = static_cast<double>(ix) + 0.5 * stepFx;
  double fy = static_cast<double>(iy) + 0.5 * stepFy;
  double integral = 0.0;
  for (int k = 0; k < maxSteps; ++k) {
    if (fx <= -1.0 || fy <= -1.0 || fx >= static_cast<double>(nx) ||
        fy >= static_cast<double>(ny)) {
      break;
    }
    const double mu = sample_mu_bilinear(slice, nx, ny, fx, fy);
    if (mu > 0.0) {
      integral += mu * step;
    }
    fx += stepFx;
    fy += stepFy;
  }
  factors[index] = quantise_factor(exp(-integral));
}
)CU";

  const std::string arch = "--gpu-architecture=compute_" +
                           std::to_string(props.major) +
                           std::to_string(props.minor);
  const char *options[] = {arch.c_str()};
  const std::string ptx =
      compileNvrtcProgramToPtx(source, "ctac_factors_cuda.cu", 1, options);

  CUcontext context = nullptr;
  SPECT_CUDRV_CHECK(cuCtxGetCurrent(&context));
  if (context == nullptr)
    throw std::runtime_error(
        "CUDA context is not initialized for CTAC factor kernels.");

  if (g_ctacFactorKernelCache.module != nullptr) {
    SPECT_CUDRV_CHECK(cuModuleUnload(g_ctacFactorKernelCache.module));
    g_ctacFactorKernelCache.module = nullptr;
    g_ctacFactorKernelCache.lineIntegral = nullptr;
  }

  loadNvrtcPtxModule(&g_ctacFactorKernelCache.module, ptx);
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_ctacFactorKernelCache.lineIntegral, g_ctacFactorKernelCache.module,
      "compute_ctac_line_integral_factors"));
  g_ctacFactorKernelCache.device = device;
  g_ctacFactorKernelCache.computeMajor = props.major;
  g_ctacFactorKernelCache.computeMinor = props.minor;
}

} // namespace

bool computeAttenFactorsLineIntegralCUDA(const std::vector<double> &muMap,
                                         int nx, int ny, int nz, double dx,
                                         double dy, double dz, int numAngles,
                                         int rotationSign,
                                         std::vector<uint16_t> &factors) {
  (void)dz;
  if (numAngles <= 0 || nx <= 0 || ny <= 0 || nz <= 0 || !(dx > 0.0) ||
      !(dy > 0.0)) {
    throw std::runtime_error(
        "CUDA CTAC line-integral factor generation received invalid geometry.");
  }
  const size_t voxels =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  if (muMap.size() != voxels) {
    throw std::runtime_error(
        "CUDA CTAC line-integral factor generation received a muMap size "
        "mismatch.");
  }
  if (voxels > std::numeric_limits<unsigned long long>::max() /
                   static_cast<size_t>(numAngles)) {
    throw std::runtime_error(
        "CUDA CTAC line-integral factor generation output is too large.");
  }

  ensureCTACFactorKernelLoaded();

  const size_t total = voxels * static_cast<size_t>(numAngles);
  factors.assign(total, static_cast<uint16_t>(65535));
  if (total == 0)
    return true;

  DeviceBuffer<double> dMu;
  DeviceBuffer<uint16_t> dFactors;
  dMu.allocate(muMap.size());
  dFactors.allocate(total);
  dMu.upload(muMap.data(), muMap.size());

  CUdeviceptr muPtr = reinterpret_cast<CUdeviceptr>(dMu.get());
  CUdeviceptr outPtr = reinterpret_cast<CUdeviceptr>(dFactors.get());
  unsigned long long voxelsArg = static_cast<unsigned long long>(voxels);
  unsigned long long totalArg = static_cast<unsigned long long>(total);
  void *args[] = {&muPtr,
                  &nx,
                  &ny,
                  &nz,
                  &dx,
                  &dy,
                  &numAngles,
                  &rotationSign,
                  &voxelsArg,
                  &outPtr,
                  &totalArg};

  const unsigned int blockSize = kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(g_ctacFactorKernelCache.lineIntegral,
                                   gridSize, 1, 1, blockSize, 1, 1, 0, nullptr,
                                   args, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  dFactors.download(factors.data(), total);
  return true;
}

} // namespace spect::cuda

#endif // CUDA_ENABLED
