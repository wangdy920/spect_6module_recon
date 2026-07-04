#include "CudaProjKernelLauncher.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include "cuda/CudaDriverHelpers.h"
#include "cuda/SparseOpsCudaKernels.h"

#include <cuda.h>
#include <nvrtc.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace spect {
namespace {

struct ProjDomainKernelCache {
  CUmodule module = nullptr;
  CUfunction ratioFunction = nullptr;
  CUfunction compactRatioFunction = nullptr;
  CUfunction compactTransformBackwardFunction = nullptr;
  CUfunction compactBakeAttenFunction = nullptr;
  CUfunction convYFunction = nullptr;
  CUfunction convXFunction = nullptr;
  int device = -1;
  int computeMajor = -1;
  int computeMinor = -1;
  std::mutex mutex;

  ~ProjDomainKernelCache() {
    if (module != nullptr) {
      cuModuleUnload(module);
    }
  }
};

ProjDomainKernelCache g_projDomainKernelCache;

} // namespace

void ensureProjDomainKernelsLoaded() {
  CUDA_CHECK(cudaFree(nullptr));
  SPECT_CUDRV_CHECK(cuInit(0));

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));

  std::lock_guard<std::mutex> lock(g_projDomainKernelCache.mutex);
  if (g_projDomainKernelCache.ratioFunction != nullptr &&
      g_projDomainKernelCache.compactRatioFunction != nullptr &&
      g_projDomainKernelCache.compactTransformBackwardFunction != nullptr &&
      g_projDomainKernelCache.compactBakeAttenFunction != nullptr &&
      g_projDomainKernelCache.convYFunction != nullptr &&
      g_projDomainKernelCache.convXFunction != nullptr &&
      g_projDomainKernelCache.device == device &&
      g_projDomainKernelCache.computeMajor == props.major &&
      g_projDomainKernelCache.computeMinor == props.minor) {
    return;
  }

  const char *source = cuda::projectionDomainKernelSource();

  nvrtcProgram program;
  SPECT_NVRTC_CHECK(
      nvrtcCreateProgram(&program, source, "projdomain_chunk_ops.cu", 0,
                         nullptr, nullptr));

  std::string arch = "--gpu-architecture=compute_" +
                     std::to_string(props.major) +
                     std::to_string(props.minor);
  const char *options[] = {arch.c_str()};
  const nvrtcResult compileStatus = nvrtcCompileProgram(program, 1, options);

  size_t logSize = 0;
  SPECT_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &logSize));
  std::string log(logSize, '\0');
  if (logSize > 1) {
    SPECT_NVRTC_CHECK(nvrtcGetProgramLog(program, log.data()));
  }
  if (compileStatus != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error(std::string("NVRTC compile failed: ") + log);
  }

  size_t ptxSize = 0;
  SPECT_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptxSize));
  std::string ptx(ptxSize, '\0');
  SPECT_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
  SPECT_NVRTC_CHECK(nvrtcDestroyProgram(&program));

  CUcontext context = nullptr;
  SPECT_CUDRV_CHECK(cuCtxGetCurrent(&context));
  if (context == nullptr) {
    throw std::runtime_error(
        "CUDA context is not initialized for projection-domain helper kernels.");
  }

  if (g_projDomainKernelCache.module != nullptr) {
    SPECT_CUDRV_CHECK(cuModuleUnload(g_projDomainKernelCache.module));
    g_projDomainKernelCache.module = nullptr;
    g_projDomainKernelCache.ratioFunction = nullptr;
    g_projDomainKernelCache.compactRatioFunction = nullptr;
    g_projDomainKernelCache.compactTransformBackwardFunction = nullptr;
    g_projDomainKernelCache.compactBakeAttenFunction = nullptr;
    g_projDomainKernelCache.convYFunction = nullptr;
    g_projDomainKernelCache.convXFunction = nullptr;
  }

  SPECT_CUDRV_CHECK(cuModuleLoadDataEx(&g_projDomainKernelCache.module,
                                       ptx.data(), 0, nullptr, nullptr));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.ratioFunction, g_projDomainKernelCache.module,
      "compute_ratio_chunk"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactRatioFunction,
      g_projDomainKernelCache.module, "compute_compact_ratio_chunk"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactTransformBackwardFunction,
      g_projDomainKernelCache.module, "compact_transform_backward_add"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactBakeAttenFunction,
      g_projDomainKernelCache.module, "compact_bake_atten_values"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.convYFunction, g_projDomainKernelCache.module,
      "conv2d_y_mirror_frames"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.convXFunction, g_projDomainKernelCache.module,
      "conv2d_x_mirror_frames"));
  g_projDomainKernelCache.device = device;
  g_projDomainKernelCache.computeMajor = props.major;
  g_projDomainKernelCache.computeMinor = props.minor;
}

void launchProjDomainRatioKernel(const float *yhat, const float *projections,
                                 size_t projectionOffset, float *ratio,
                                 size_t count, float eps,
                                 cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (count == 0) {
    return;
  }
  CUdeviceptr yhatPtr = reinterpret_cast<CUdeviceptr>(const_cast<float *>(yhat));
  CUdeviceptr projPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(projections));
  CUdeviceptr ratioPtr = reinterpret_cast<CUdeviceptr>(ratio);
  unsigned long long offset = static_cast<unsigned long long>(projectionOffset);
  unsigned long long total = static_cast<unsigned long long>(count);
  void *args[] = {&yhatPtr, &projPtr, &offset, &ratioPtr, &total, &eps};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((count + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(g_projDomainKernelCache.ratioFunction,
                                   gridSize, 1, 1, blockSize, 1, 1, 0,
                                   reinterpret_cast<CUstream>(stream),
                                   args, nullptr));
}

void launchCompactRatioKernel(
    const float *yhat, const float *fullProjection, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &params, float *ratio, size_t count,
    float eps, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (count == 0) {
    return;
  }
  CUdeviceptr yhatPtr = reinterpret_cast<CUdeviceptr>(const_cast<float *>(yhat));
  CUdeviceptr projectionPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(fullProjection));
  CUdeviceptr ratioPtr = reinterpret_cast<CUdeviceptr>(ratio);
  unsigned long long projectionCount =
      static_cast<unsigned long long>(fullProjectionCount);
  unsigned long long total = static_cast<unsigned long long>(count);
  int zMirror = params.zMirror ? 1 : 0;
  void *args[] = {&yhatPtr,
                  &projectionPtr,
                  &projectionCount,
                  const_cast<int *>(&params.symmetryMode),
                  const_cast<int *>(&params.compactRowBase),
                  const_cast<int *>(&params.compactRaysPerAngle),
                  const_cast<int *>(&params.compactNumAngles),
                  const_cast<int *>(&params.detNx),
                  const_cast<int *>(&params.detNz),
                  const_cast<int *>(&params.compactDetZ),
                  const_cast<int *>(&params.fullNumAngles),
                  const_cast<int *>(&params.fullRaysPerAngle),
                  const_cast<int *>(&params.symId),
                  &zMirror,
                  &ratioPtr,
                  &total,
                  &eps};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((count + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactRatioFunction, gridSize, 1, 1, blockSize,
      1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactTransformBackwardAddKernel(
    const float *rotated, double *accum, size_t nvox,
    const cuda::CompactDeviceTransformParams &params, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (nvox == 0) {
    return;
  }
  CUdeviceptr rotatedPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(rotated));
  CUdeviceptr accumPtr = reinterpret_cast<CUdeviceptr>(accum);
  unsigned long long total = static_cast<unsigned long long>(nvox);
  int zMirror = params.zMirror ? 1 : 0;
  void *args[] = {&rotatedPtr,
                  &accumPtr,
                  &total,
                  const_cast<int *>(&params.symmetryMode),
                  const_cast<int *>(&params.nx),
                  const_cast<int *>(&params.ny),
                  const_cast<int *>(&params.nz),
                  const_cast<int *>(&params.symId),
                  &zMirror};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((nvox + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactTransformBackwardFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactBakeAttenKernel(const float *values, const int32_t *rowPtr,
                                  const int32_t *colIdx, const uint16_t *att,
                                  float *baked, int rows,
                                  const cuda::CompactAttenBakeParams &params,
                                  cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (rows <= 0) {
    return;
  }
  CUdeviceptr valuesPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(values));
  CUdeviceptr rowPtrPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<int32_t *>(rowPtr));
  CUdeviceptr colIdxPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<int32_t *>(colIdx));
  CUdeviceptr attPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint16_t *>(att));
  CUdeviceptr bakedPtr = reinterpret_cast<CUdeviceptr>(baked);
  int zMirror = params.zMirror ? 1 : 0;
  void *args[] = {&valuesPtr,
                  &rowPtrPtr,
                  &colIdxPtr,
                  &attPtr,
                  &bakedPtr,
                  &rows,
                  const_cast<int *>(&params.symmetryMode),
                  const_cast<int *>(&params.compactRowBase),
                  const_cast<int *>(&params.compactRaysPerAngle),
                  const_cast<int *>(&params.compactNumAngles),
                  const_cast<int *>(&params.fullNumAngles),
                  const_cast<int *>(&params.symId),
                  &zMirror,
                  const_cast<int *>(&params.nx),
                  const_cast<int *>(&params.ny),
                  const_cast<int *>(&params.nz)};
  const unsigned int blockSize = 256;
  const unsigned int gridSize = static_cast<unsigned int>(rows);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactBakeAttenFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchProjDomainConvYKernel(const float *input, float *output,
                                 const float *kernel, int radius, int detNz,
                                 int detNx, size_t frameCount,
                                 cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  const size_t total = frameCount * static_cast<size_t>(detNz) *
                       static_cast<size_t>(detNx);
  if (total == 0) {
    return;
  }
  CUdeviceptr inputPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(input));
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  CUdeviceptr kernelPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(kernel));
  unsigned long long frames = static_cast<unsigned long long>(frameCount);
  void *args[] = {&inputPtr, &outputPtr, &kernelPtr, &radius, &detNz, &detNx,
                  &frames};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(g_projDomainKernelCache.convYFunction,
                                   gridSize, 1, 1, blockSize, 1, 1, 0,
                                   reinterpret_cast<CUstream>(stream),
                                   args, nullptr));
}

void launchProjDomainConvXKernel(const float *input, float *output,
                                 const float *kernel, int radius, int detNz,
                                 int detNx, size_t frameCount,
                                 cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  const size_t total = frameCount * static_cast<size_t>(detNz) *
                       static_cast<size_t>(detNx);
  if (total == 0) {
    return;
  }
  CUdeviceptr inputPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(input));
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  CUdeviceptr kernelPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(kernel));
  unsigned long long frames = static_cast<unsigned long long>(frameCount);
  void *args[] = {&inputPtr, &outputPtr, &kernelPtr, &radius, &detNz, &detNx,
                  &frames};
  const unsigned int blockSize = 256;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(g_projDomainKernelCache.convXFunction,
                                   gridSize, 1, 1, blockSize, 1, 1, 0,
                                   reinterpret_cast<CUstream>(stream),
                                   args, nullptr));
}

} // namespace spect

#endif