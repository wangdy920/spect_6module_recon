#include "CudaProjKernelLauncher.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include "cuda/CudaDriverHelpers.h"
#include "cuda/CudaLaunchConfig.h"
#include "cuda/CudaNvrtcArch.h"
#include "cuda/CudaNvrtcLoader.h"
#include "cuda/SparseOpsCudaKernels.h"

#include <cuda.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>

namespace spect {
namespace {

struct ProjDomainKernelCache {
  CUmodule module = nullptr;
  CUfunction ratioFunction = nullptr;
  CUfunction compactRatioFunction = nullptr;
  CUfunction compactRatioBatchFunction = nullptr;
  CUfunction compactTransformBackwardFunction = nullptr;
  CUfunction compactTransformBackwardBatchFunction = nullptr;
  CUfunction compactBakeAttenFunction = nullptr;
  CUfunction compactAttenForwardFunction = nullptr;
  CUfunction compactAttenBackwardFunction = nullptr;
  CUfunction convYFunction = nullptr;
  CUfunction convXFunction = nullptr;
  CUfunction compactBranchBatchForwardScaledU16Function = nullptr;
  CUfunction compactBranchBatchBackwardScaledU16Function = nullptr;
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
      g_projDomainKernelCache.compactRatioBatchFunction != nullptr &&
      g_projDomainKernelCache.compactTransformBackwardFunction != nullptr &&
      g_projDomainKernelCache.compactTransformBackwardBatchFunction != nullptr &&
      g_projDomainKernelCache.compactBakeAttenFunction != nullptr &&
      g_projDomainKernelCache.compactAttenForwardFunction != nullptr &&
      g_projDomainKernelCache.compactAttenBackwardFunction != nullptr &&
      g_projDomainKernelCache.convYFunction != nullptr &&
      g_projDomainKernelCache.convXFunction != nullptr &&
      g_projDomainKernelCache.compactBranchBatchForwardScaledU16Function !=
          nullptr &&
      g_projDomainKernelCache.compactBranchBatchBackwardScaledU16Function !=
          nullptr &&
      g_projDomainKernelCache.device == device &&
      g_projDomainKernelCache.computeMajor == props.major &&
      g_projDomainKernelCache.computeMinor == props.minor) {
    return;
  }

  const char *source = cuda::projectionDomainKernelSource();

  const std::string arch = cuda::nvrtcGpuArchitectureOption(props);
  const char *options[] = {arch.c_str()};
  const std::string ptx = cuda::compileNvrtcProgramToPtx(
      source, "projdomain_chunk_ops.cu", 1, options);

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
    g_projDomainKernelCache.compactRatioBatchFunction = nullptr;
    g_projDomainKernelCache.compactTransformBackwardFunction = nullptr;
    g_projDomainKernelCache.compactTransformBackwardBatchFunction = nullptr;
    g_projDomainKernelCache.compactBakeAttenFunction = nullptr;
    g_projDomainKernelCache.compactAttenForwardFunction = nullptr;
    g_projDomainKernelCache.compactAttenBackwardFunction = nullptr;
    g_projDomainKernelCache.convYFunction = nullptr;
    g_projDomainKernelCache.convXFunction = nullptr;
    g_projDomainKernelCache.compactBranchBatchForwardScaledU16Function = nullptr;
    g_projDomainKernelCache.compactBranchBatchBackwardScaledU16Function = nullptr;
  }

  cuda::loadNvrtcPtxModule(&g_projDomainKernelCache.module, ptx);
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.ratioFunction, g_projDomainKernelCache.module,
      "compute_ratio_chunk"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactRatioFunction,
      g_projDomainKernelCache.module, "compute_compact_ratio_chunk"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactRatioBatchFunction,
      g_projDomainKernelCache.module, "compute_compact_ratio_chunk_batch"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactTransformBackwardFunction,
      g_projDomainKernelCache.module, "compact_transform_backward_add"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactTransformBackwardBatchFunction,
      g_projDomainKernelCache.module,
      "compact_transform_backward_add_batch"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactBakeAttenFunction,
      g_projDomainKernelCache.module, "compact_bake_atten_values"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactAttenForwardFunction,
      g_projDomainKernelCache.module, "compact_atten_fused_spmv"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactAttenBackwardFunction,
      g_projDomainKernelCache.module, "compact_atten_fused_spmvt"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.convYFunction, g_projDomainKernelCache.module,
      "conv2d_y_mirror_frames"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.convXFunction, g_projDomainKernelCache.module,
      "conv2d_x_mirror_frames"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactBranchBatchForwardScaledU16Function,
      g_projDomainKernelCache.module,
      "compact_branch_batch_forward_scaled_u16"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactBranchBatchBackwardScaledU16Function,
      g_projDomainKernelCache.module,
      "compact_branch_batch_backward_scaled_u16"));
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
    const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
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
    const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
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
    const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((nvox + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactTransformBackwardFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactRatioBatchKernel(
    const float *yhatBatch, const float *fullProjection,
    size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &commonParams, const int *dSymIds,
    const int *dZMirrors, int batchSize, float *ratioBatch, size_t rows,
    float eps, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (rows == 0 || batchSize <= 0) {
    return;
  }
  CUdeviceptr yhatPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(yhatBatch));
  CUdeviceptr projectionPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(fullProjection));
  CUdeviceptr symIdsPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<int *>(dSymIds));
  CUdeviceptr zMirrorsPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<int *>(dZMirrors));
  CUdeviceptr ratioPtr = reinterpret_cast<CUdeviceptr>(ratioBatch);
  unsigned long long projectionCount =
      static_cast<unsigned long long>(fullProjectionCount);
  unsigned long long rowCount = static_cast<unsigned long long>(rows);
  unsigned long long total =
      rowCount * static_cast<unsigned long long>(batchSize);
  void *args[] = {&yhatPtr,
                  &projectionPtr,
                  &projectionCount,
                  const_cast<int *>(&commonParams.symmetryMode),
                  const_cast<int *>(&commonParams.compactRowBase),
                  const_cast<int *>(&commonParams.compactRaysPerAngle),
                  const_cast<int *>(&commonParams.compactNumAngles),
                  const_cast<int *>(&commonParams.detNx),
                  const_cast<int *>(&commonParams.detNz),
                  const_cast<int *>(&commonParams.compactDetZ),
                  const_cast<int *>(&commonParams.fullNumAngles),
                  const_cast<int *>(&commonParams.fullRaysPerAngle),
                  &symIdsPtr,
                  &zMirrorsPtr,
                  &batchSize,
                  &ratioPtr,
                  &rowCount,
                  &eps};
  const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactRatioBatchFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactTransformBackwardAddBatchKernel(
    const float *rotatedBatch, double *accum, size_t nvox, int batchSize,
    const cuda::CompactDeviceTransformParams &commonParams, const int *dSymIds,
    const int *dZMirrors, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (nvox == 0 || batchSize <= 0) {
    return;
  }
  CUdeviceptr rotatedPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(rotatedBatch));
  CUdeviceptr accumPtr = reinterpret_cast<CUdeviceptr>(accum);
  CUdeviceptr symIdsPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<int *>(dSymIds));
  CUdeviceptr zMirrorsPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<int *>(dZMirrors));
  unsigned long long voxelCount = static_cast<unsigned long long>(nvox);
  unsigned long long total =
      voxelCount * static_cast<unsigned long long>(batchSize);
  void *args[] = {&rotatedPtr,
                  &accumPtr,
                  &voxelCount,
                  &batchSize,
                  const_cast<int *>(&commonParams.symmetryMode),
                  const_cast<int *>(&commonParams.nx),
                  const_cast<int *>(&commonParams.ny),
                  const_cast<int *>(&commonParams.nz),
                  &symIdsPtr,
                  &zMirrorsPtr};
  const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactTransformBackwardBatchFunction, gridSize,
      1, 1, blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args,
      nullptr));
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
    const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize = static_cast<unsigned int>(rows);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactBakeAttenFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactAttenFusedForwardKernel(
    const float *values, const int32_t *rowPtr, const int32_t *colIdx,
    const uint16_t *att, const float *x, float *y, int rows,
    const cuda::CompactAttenBakeParams &params, cudaStream_t stream) {
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
  CUdeviceptr xPtr = reinterpret_cast<CUdeviceptr>(const_cast<float *>(x));
  CUdeviceptr yPtr = reinterpret_cast<CUdeviceptr>(y);
  int zMirror = params.zMirror ? 1 : 0;
  void *args[] = {&valuesPtr,
                  &rowPtrPtr,
                  &colIdxPtr,
                  &attPtr,
                  &xPtr,
                  &yPtr,
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
  const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize = static_cast<unsigned int>(rows);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactAttenForwardFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactAttenFusedBackwardKernel(
    const float *values, const int32_t *rowPtr, const int32_t *colIdx,
    const uint16_t *att, const float *x, float *y, int rows,
    const cuda::CompactAttenBakeParams &params, cudaStream_t stream) {
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
  CUdeviceptr xPtr = reinterpret_cast<CUdeviceptr>(const_cast<float *>(x));
  CUdeviceptr yPtr = reinterpret_cast<CUdeviceptr>(y);
  int zMirror = params.zMirror ? 1 : 0;
  void *args[] = {&valuesPtr,
                  &rowPtrPtr,
                  &colIdxPtr,
                  &attPtr,
                  &xPtr,
                  &yPtr,
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
  const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize = static_cast<unsigned int>(rows);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactAttenBackwardFunction, gridSize, 1, 1,
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
    const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
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
    const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(g_projDomainKernelCache.convXFunction,
                                   gridSize, 1, 1, blockSize, 1, 1, 0,
                                   reinterpret_cast<CUstream>(stream),
                                   args, nullptr));
}

void launchCompactBranchBatchForwardScaledU16(
    const uint32_t *rowPtr, const uint32_t *colIdx, const uint16_t *values,
    const float *rowScales, const float *x, int cols, int batchSize, float *y,
    int rows, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (rows <= 0 || batchSize <= 0) {
    return;
  }
  CUdeviceptr rowPtrPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint32_t *>(rowPtr));
  CUdeviceptr colIdxPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint32_t *>(colIdx));
  CUdeviceptr valuesPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint16_t *>(values));
  CUdeviceptr rowScalesPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(rowScales));
  CUdeviceptr xPtr = reinterpret_cast<CUdeviceptr>(const_cast<float *>(x));
  CUdeviceptr yPtr = reinterpret_cast<CUdeviceptr>(y);
  void *args[] = {&rowPtrPtr, &colIdxPtr, &valuesPtr, &rowScalesPtr, &xPtr,
                  &cols, &batchSize, &yPtr, &rows};
  const unsigned int blockSize =
      static_cast<unsigned int>(std::max(32, std::min(batchSize * 4, 256)));
  const unsigned int gridSize = static_cast<unsigned int>(rows);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactBranchBatchForwardScaledU16Function,
      gridSize, 1, 1, blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream),
      args, nullptr));
}

void launchCompactBranchBatchBackwardScaledU16(
    const uint32_t *rowPtr, const uint32_t *colIdx, const uint16_t *values,
    const float *rowScales, const float *y, int rows, int batchSize, float *x,
    int cols, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (rows <= 0 || batchSize <= 0) {
    return;
  }
  CUdeviceptr rowPtrPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint32_t *>(rowPtr));
  CUdeviceptr colIdxPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint32_t *>(colIdx));
  CUdeviceptr valuesPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint16_t *>(values));
  CUdeviceptr rowScalesPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(rowScales));
  CUdeviceptr yPtr = reinterpret_cast<CUdeviceptr>(const_cast<float *>(y));
  CUdeviceptr xPtr = reinterpret_cast<CUdeviceptr>(x);
  void *args[] = {&rowPtrPtr, &colIdxPtr, &valuesPtr, &rowScalesPtr, &yPtr,
                  &rows, &batchSize, &xPtr, &cols};
  const unsigned int blockSize =
      static_cast<unsigned int>(std::max(32, std::min(batchSize * 4, 256)));
  const unsigned int gridSize = static_cast<unsigned int>(rows);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactBranchBatchBackwardScaledU16Function,
      gridSize, 1, 1, blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream),
      args, nullptr));
}

} // namespace spect

#endif
