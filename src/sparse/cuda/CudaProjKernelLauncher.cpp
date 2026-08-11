#include "CudaProjKernelLauncher.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include "cuda/CudaDriverHelpers.h"
#include "cuda/CudaLaunchConfig.h"
#include "cuda/CudaNvrtcArch.h"
#include "cuda/CudaNvrtcLoader.h"
#include "sparse/cuda/SparseOpsCudaKernels.h"

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
  CUfunction compactGatherBedFunction = nullptr;
  CUfunction compactConvertDoubleToFloatFunction = nullptr;
  CUfunction compactAccumulateBedFunction = nullptr;
  CUfunction compactMlemUpdateFunction = nullptr;
  CUfunction compactTransformForwardBatchFunction = nullptr;
  CUfunction compactTransformBackwardFunction = nullptr;
  CUfunction compactTransformBackwardBatchFunction = nullptr;
  CUfunction compactBakeAttenFunction = nullptr;
  CUfunction compactAttenForwardFunction = nullptr;
  CUfunction compactAttenBackwardFunction = nullptr;
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
      g_projDomainKernelCache.compactGatherBedFunction != nullptr &&
      g_projDomainKernelCache.compactConvertDoubleToFloatFunction != nullptr &&
      g_projDomainKernelCache.compactAccumulateBedFunction != nullptr &&
      g_projDomainKernelCache.compactMlemUpdateFunction != nullptr &&
      g_projDomainKernelCache.compactTransformForwardBatchFunction != nullptr &&
      g_projDomainKernelCache.compactTransformBackwardFunction != nullptr &&
      g_projDomainKernelCache.compactTransformBackwardBatchFunction != nullptr &&
      g_projDomainKernelCache.compactBakeAttenFunction != nullptr &&
      g_projDomainKernelCache.compactAttenForwardFunction != nullptr &&
      g_projDomainKernelCache.compactAttenBackwardFunction != nullptr &&
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
    g_projDomainKernelCache.compactGatherBedFunction = nullptr;
    g_projDomainKernelCache.compactConvertDoubleToFloatFunction = nullptr;
    g_projDomainKernelCache.compactAccumulateBedFunction = nullptr;
    g_projDomainKernelCache.compactMlemUpdateFunction = nullptr;
    g_projDomainKernelCache.compactTransformForwardBatchFunction = nullptr;
    g_projDomainKernelCache.compactTransformBackwardFunction = nullptr;
    g_projDomainKernelCache.compactTransformBackwardBatchFunction = nullptr;
    g_projDomainKernelCache.compactBakeAttenFunction = nullptr;
    g_projDomainKernelCache.compactAttenForwardFunction = nullptr;
    g_projDomainKernelCache.compactAttenBackwardFunction = nullptr;
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
      &g_projDomainKernelCache.compactGatherBedFunction,
      g_projDomainKernelCache.module, "compact_gather_bed_double_to_float"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactConvertDoubleToFloatFunction,
      g_projDomainKernelCache.module, "compact_convert_double_to_float"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactAccumulateBedFunction,
      g_projDomainKernelCache.module,
      "compact_accumulate_bed_float_to_global_double"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactMlemUpdateFunction,
      g_projDomainKernelCache.module, "compact_mlem_update"));
  SPECT_CUDRV_CHECK(cuModuleGetFunction(
      &g_projDomainKernelCache.compactTransformForwardBatchFunction,
      g_projDomainKernelCache.module, "compact_transform_forward_batch"));
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
    const float *yhat, const double *fullProjection, size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &params, float *ratio, size_t count,
    double eps, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (count == 0) {
    return;
  }
  CUdeviceptr yhatPtr = reinterpret_cast<CUdeviceptr>(const_cast<float *>(yhat));
  CUdeviceptr projectionPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<double *>(fullProjection));
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

void launchCompactGatherBedDoubleToFloatKernel(
    const double *globalImage, size_t globalCount, float *bedImage, int nx,
    int ny, int nz, int globalNz, int zStart, bool flipLocalZ,
    cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  const size_t nvox = static_cast<size_t>(nx) * static_cast<size_t>(ny) *
                      static_cast<size_t>(nz);
  if (nvox == 0)
    return;
  CUdeviceptr globalPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<double *>(globalImage));
  CUdeviceptr bedPtr = reinterpret_cast<CUdeviceptr>(bedImage);
  unsigned long long globalSize =
      static_cast<unsigned long long>(globalCount);
  unsigned long long voxelCount = static_cast<unsigned long long>(nvox);
  int flip = flipLocalZ ? 1 : 0;
  void *args[] = {&globalPtr, &globalSize, &bedPtr, &nx, &ny, &nz,
                  &globalNz, &zStart, &flip, &voxelCount};
  const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((nvox + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactGatherBedFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactConvertDoubleToFloatKernel(
    const double *input, float *output, size_t count, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (count == 0)
    return;
  CUdeviceptr inputPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<double *>(input));
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  unsigned long long total = static_cast<unsigned long long>(count);
  void *args[] = {&inputPtr, &outputPtr, &total};
  const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((count + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactConvertDoubleToFloatFunction, gridSize,
      1, 1, blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args,
      nullptr));
}

void launchCompactAccumulateBedFloatToGlobalDoubleKernel(
    const float *bedImage, double *globalImage, size_t globalCount, int nx,
    int ny, int nz, int globalNz, int zStart, bool flipLocalZ,
    cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  const size_t nvox = static_cast<size_t>(nx) * static_cast<size_t>(ny) *
                      static_cast<size_t>(nz);
  if (nvox == 0)
    return;
  CUdeviceptr bedPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(bedImage));
  CUdeviceptr globalPtr = reinterpret_cast<CUdeviceptr>(globalImage);
  unsigned long long globalSize =
      static_cast<unsigned long long>(globalCount);
  unsigned long long voxelCount = static_cast<unsigned long long>(nvox);
  int flip = flipLocalZ ? 1 : 0;
  void *args[] = {&bedPtr, &globalPtr, &globalSize, &nx, &ny, &nz,
                  &globalNz, &zStart, &flip, &voxelCount};
  const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((nvox + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactAccumulateBedFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactMlemUpdateKernel(
    double *image, const double *numer, const double *denom,
    const uint8_t *mask, bool hasMask, size_t count, uint32_t *flags,
    double eps, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (count == 0)
    return;
  CUdeviceptr imagePtr = reinterpret_cast<CUdeviceptr>(image);
  CUdeviceptr numerPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<double *>(numer));
  CUdeviceptr denomPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<double *>(denom));
  CUdeviceptr maskPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint8_t *>(mask));
  CUdeviceptr flagsPtr = reinterpret_cast<CUdeviceptr>(flags);
  int useMask = hasMask ? 1 : 0;
  unsigned long long total = static_cast<unsigned long long>(count);
  void *args[] = {&imagePtr, &numerPtr, &denomPtr, &maskPtr, &useMask,
                  &total, &flagsPtr, &eps};
  const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((count + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactMlemUpdateFunction, gridSize, 1, 1,
      blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launchCompactTransformForwardBatchKernel(
    const float *bedImage, float *rotatedBatch, size_t nvox, int branchCount,
    int zBranchCount, const cuda::CompactDeviceTransformParams &commonParams,
    cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (nvox == 0 || branchCount <= 0 || zBranchCount <= 0)
    return;
  CUdeviceptr bedPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(bedImage));
  CUdeviceptr batchPtr = reinterpret_cast<CUdeviceptr>(rotatedBatch);
  unsigned long long voxelCount = static_cast<unsigned long long>(nvox);
  const size_t total = nvox * static_cast<size_t>(branchCount);
  void *args[] = {
      &bedPtr,
      &batchPtr,
      &voxelCount,
      &branchCount,
      &zBranchCount,
      const_cast<int *>(&commonParams.symmetryMode),
      const_cast<int *>(&commonParams.nx),
      const_cast<int *>(&commonParams.ny),
      const_cast<int *>(&commonParams.nz)};
  const unsigned int blockSize = cuda::kCudaLaunchBlockSize;
  const unsigned int gridSize =
      static_cast<unsigned int>((total + blockSize - 1) / blockSize);
  SPECT_CUDRV_CHECK(cuLaunchKernel(
      g_projDomainKernelCache.compactTransformForwardBatchFunction, gridSize,
      1, 1, blockSize, 1, 1, 0, reinterpret_cast<CUstream>(stream), args,
      nullptr));
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
    const float *yhatBatch, const double *fullProjection,
    size_t fullProjectionCount,
    const cuda::CompactDeviceRatioParams &commonParams, const int *dSymIds,
    const int *dZMirrors, int batchSize, float *ratioBatch, size_t rows,
    double eps, cudaStream_t stream) {
  ensureProjDomainKernelsLoaded();
  if (rows == 0 || batchSize <= 0) {
    return;
  }
  CUdeviceptr yhatPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<float *>(yhatBatch));
  CUdeviceptr projectionPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<double *>(fullProjection));
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


} // namespace spect

#endif
