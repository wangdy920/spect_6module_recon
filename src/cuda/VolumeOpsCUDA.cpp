#include "cuda/VolumeOpsCUDA.h"

#if CUDA_ENABLED

#include "cuda/CudaConfig.h"

#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace spect::cuda {
namespace {

inline void checkCuDriver(CUresult status, const char *expr, const char *file,
                          int line) {
  if (status != CUDA_SUCCESS) {
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    throw std::runtime_error(std::string("CUDA driver error at ") + file + ":" +
                             std::to_string(line) + " for " + expr + " (" +
                             (name ? name : "unknown") + ": " +
                             (message ? message : "unknown") + ")");
  }
}

inline void checkNvrtc(nvrtcResult status, const char *expr, const char *file,
                       int line) {
  if (status != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string("NVRTC error at ") + file + ":" +
                             std::to_string(line) + " for " + expr + " (" +
                             nvrtcGetErrorString(status) + ")");
  }
}

#define SPECT_CUDRV_CHECK(call) checkCuDriver((call), #call, __FILE__, __LINE__)
#define SPECT_NVRTC_CHECK(call) checkNvrtc((call), #call, __FILE__, __LINE__)

struct VolumeKernelCache {
  CUmodule module = nullptr;
  CUfunction convX = nullptr;
  CUfunction convY = nullptr;
  CUfunction convZ = nullptr;
  CUfunction convXF = nullptr;
  CUfunction convYF = nullptr;
  CUfunction convZF = nullptr;
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

void ensureVolumeKernelsLoaded() {
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
  if (idx < 0) {
    idx = -idx;
  }
  if (idx >= dim) {
    idx = 2 * dim - 2 - idx;
  }
  if (idx < 0) {
    idx = 0;
  }
  if (idx >= dim) {
    idx = dim - 1;
  }
  return idx;
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

extern "C" __global__ void rdp_grad_3d(const double* input,
                                        const unsigned char* mask,
                                        bool hasMask,
                                        bool skipOutsideMask,
                                        int Nx, int Ny, int Nz,
                                        double eps,
                                        double gamma,
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
    g += term / (denom * denom);
  }

  output[index] = g;
}

extern "C" __device__ void tv_compute_p(const double* input,
                                         const unsigned char* mask,
                                         bool hasMask,
                                         bool skipOutsideMask,
                                         int px, int py, int pz,
                                         int Nx, int Ny, int Nz,
                                         double eps,
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
  double gx = 0.0;
  double gy = 0.0;
  double gz = 0.0;

  if (px + 1 < Nx) {
    const unsigned long long nidx =
        static_cast<unsigned long long>(px + 1) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(pz);
    if (!skipOutsideMask || !hasMask || mask[nidx] != 0)
      gx = input[nidx] - fv;
  }
  if (py + 1 < Ny) {
    const unsigned long long nidx =
        static_cast<unsigned long long>(px) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py + 1) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(pz);
    if (!skipOutsideMask || !hasMask || mask[nidx] != 0)
      gy = input[nidx] - fv;
  }
  if (pz + 1 < Nz) {
    const unsigned long long nidx =
        static_cast<unsigned long long>(px) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(pz + 1);
    if (!skipOutsideMask || !hasMask || mask[nidx] != 0)
      gz = input[nidx] - fv;
  }

  const double mag = sqrt(gx * gx + gy * gy + gz * gz + eps * eps);
  *out_gx = gx / mag;
  *out_gy = gy / mag;
  *out_gz = gz / mag;
}

extern "C" __global__ void tv_grad_3d(const double* input,
                                       const unsigned char* mask,
                                       bool hasMask,
                                       bool skipOutsideMask,
                                       int Nx, int Ny, int Nz,
                                       double eps,
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
  tv_compute_p(input, mask, hasMask, skipOutsideMask,
               x, y, z, Nx, Ny, Nz, eps, &px, &py, &pz);

  double diverg = 0.0;

  /* X divergence */
  if (x > 0 && (!skipOutsideMask || !hasMask ||
                mask[index - 1] != 0)) {
    double prev_px, prev_py, prev_pz;
    tv_compute_p(input, mask, hasMask, skipOutsideMask,
                 x - 1, y, z, Nx, Ny, Nz, eps,
                 &prev_px, &prev_py, &prev_pz);
    diverg += px - prev_px;
  } else {
    diverg += px;
  }

  /* Y divergence */
  if (y > 0 && (!skipOutsideMask || !hasMask ||
                mask[index - static_cast<unsigned long long>(Nx)] != 0)) {
    double prev_px, prev_py, prev_pz;
    tv_compute_p(input, mask, hasMask, skipOutsideMask,
                 x, y - 1, z, Nx, Ny, Nz, eps,
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
    tv_compute_p(input, mask, hasMask, skipOutsideMask,
                 x, y, z - 1, Nx, Ny, Nz, eps,
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
    double* px, double* py, double* pz,
    double sigma, double lambda_val,
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
  if (x + 1 < Nx) gx = u_bar[index + 1] - u0;
  if (y + 1 < Ny)
    gy = u_bar[index + static_cast<unsigned long long>(Nx)] - u0;
  if (z + 1 < Nz) {
    const unsigned long long slice =
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny);
    gz = u_bar[index + slice] - u0;
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
    const double* px, const double* py, const double* pz,
    double tau, double theta,
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
  div_val += px[index];
  if (x > 0) div_val -= px[index - 1];

  /* Y */
  div_val += py[index];
  if (y > 0)
    div_val -= py[index - static_cast<unsigned long long>(Nx)];

  /* Z */
  div_val += pz[index];
  if (z > 0) {
    const unsigned long long slice =
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny);
    div_val -= pz[index - slice];
  }

  /* Primal update */
  const double u_old = u[index];
  double u_new = (tau * v[index] + u_old + tau * div_val) / (1.0 + tau);
  if (u_new < 0.0) u_new = 0.0;

  u[index] = u_new;
  u_bar[index] = u_new + theta * (u_new - u_old);
}
)CU";

  nvrtcProgram program = nullptr;
  SPECT_NVRTC_CHECK(
      nvrtcCreateProgram(&program, source, "volume_ops_cuda.cu", 0, nullptr,
                         nullptr));

  std::string arch = "--gpu-architecture=compute_" +
                     std::to_string(props.major) +
                     std::to_string(props.minor);
  const char *options[] = {arch.c_str()};
  const nvrtcResult compileStatus =
      nvrtcCompileProgram(program, 1, options);

  size_t logSize = 0;
  SPECT_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &logSize));
  std::string log(logSize, '\0');
  if (logSize > 1)
    SPECT_NVRTC_CHECK(nvrtcGetProgramLog(program, log.data()));

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
        "CUDA context is not initialized for volume helper kernels.");
  }

  if (g_volumeKernelCache.module) {
    SPECT_CUDRV_CHECK(cuModuleUnload(g_volumeKernelCache.module));
    g_volumeKernelCache.module = nullptr;
  }

  SPECT_CUDRV_CHECK(cuModuleLoadDataEx(&g_volumeKernelCache.module, ptx.data(),
                                       0, nullptr, nullptr));
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
  g_volumeKernelCache.device = device;
  g_volumeKernelCache.computeMajor = props.major;
  g_volumeKernelCache.computeMinor = props.minor;
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

struct VolumeOpsState {
  DeviceBuffer<double> primary;
  DeviceBuffer<double> secondary;
  DeviceBuffer<double> kx;
  DeviceBuffer<double> ky;
  DeviceBuffer<double> kz;
  DeviceBuffer<uint8_t> mask;

  DeviceBuffer<float> primaryF;
  DeviceBuffer<float> secondaryF;
  DeviceBuffer<float> kxF;
  DeviceBuffer<float> kyF;
  DeviceBuffer<float> kzF;

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
  const unsigned int blockSize = 256;
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

void launchTvGradientKernel(const double *input, const uint8_t *mask,
                            bool hasMask, bool skipOutsideMask, int Nx, int Ny,
                            int Nz, double eps, double *output,
                            unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr inputPtr = reinterpret_cast<CUdeviceptr>(input);
  CUdeviceptr maskPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint8_t *>(mask));
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  void *args[] = {&inputPtr, &maskPtr, &hasMask, &skipOutsideMask, &Nx,
                  &Ny,      &Nz,      &eps,      &outputPtr,
                  &total};
  launchVolumeKernel(g_volumeKernelCache.tvGrad, args, total);
}

void launchRdpGradientKernel(const double *input, const uint8_t *mask,
                             bool hasMask, bool skipOutsideMask, int Nx, int Ny,
                             int Nz, double eps, double gamma, double *output,
                             unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr inputPtr = reinterpret_cast<CUdeviceptr>(input);
  CUdeviceptr maskPtr =
      reinterpret_cast<CUdeviceptr>(const_cast<uint8_t *>(mask));
  CUdeviceptr outputPtr = reinterpret_cast<CUdeviceptr>(output);
  void *args[] = {&inputPtr, &maskPtr, &hasMask, &skipOutsideMask, &Nx,
                  &Ny,      &Nz,      &eps,      &gamma,            &outputPtr,
                  &total};
  launchVolumeKernel(g_volumeKernelCache.rdpGrad, args, total);
}

void launchTvProxDualKernel(const double *u_bar, double *px, double *py,
                           double *pz, double sigma, double lambda_val,
                           int Nx, int Ny, int Nz,
                           unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr ubarPtr = reinterpret_cast<CUdeviceptr>(u_bar);
  CUdeviceptr pxPtr = reinterpret_cast<CUdeviceptr>(px);
  CUdeviceptr pyPtr = reinterpret_cast<CUdeviceptr>(py);
  CUdeviceptr pzPtr = reinterpret_cast<CUdeviceptr>(pz);
  void *args[] = {&ubarPtr, &pxPtr, &pyPtr, &pzPtr, &sigma, &lambda_val,
                  &Nx,      &Ny,    &Nz,    &total};
  launchVolumeKernel(g_volumeKernelCache.tvProxDual, args, total);
}

void launchTvProxPrimalKernel(double *u, double *u_bar, const double *v,
                              const double *px, const double *py,
                              const double *pz, double tau, double theta,
                              int Nx, int Ny, int Nz,
                              unsigned long long total) {
  ensureVolumeKernelsLoaded();
  CUdeviceptr uPtr = reinterpret_cast<CUdeviceptr>(u);
  CUdeviceptr ubarPtr = reinterpret_cast<CUdeviceptr>(u_bar);
  CUdeviceptr vPtr = reinterpret_cast<CUdeviceptr>(v);
  CUdeviceptr pxPtr = reinterpret_cast<CUdeviceptr>(px);
  CUdeviceptr pyPtr = reinterpret_cast<CUdeviceptr>(py);
  CUdeviceptr pzPtr = reinterpret_cast<CUdeviceptr>(pz);
  void *args[] = {&uPtr,  &ubarPtr, &vPtr,   &pxPtr, &pyPtr,
                  &pzPtr, &tau,     &theta,  &Nx,    &Ny,
                  &Nz,    &total};
  launchVolumeKernel(g_volumeKernelCache.tvProxPrimal, args, total);
}

bool shouldUseCudaVolumeOps() { return isCudaAvailable(); }

} // namespace

bool isCudaVolumeOpsAvailable() { return shouldUseCudaVolumeOps(); }

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

  if (useFloat) {
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

bool computeRDPGradientCUDA(const std::vector<double> &f, int Nx, int Ny,
                            int Nz, double delta, double gamma,
                            const std::vector<uint8_t> *mask,
                            bool skipOutsideMask,
                            std::vector<double> &gradR) {
  if (!shouldUseCudaVolumeOps())
    return false;

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::runtime_error("CUDA RDP input size mismatch");
  if (mask && mask->size() != total)
    throw std::runtime_error("CUDA RDP mask size mismatch");

  const double eps = std::max(delta, 1e-6);
  gradR.resize(total);

  auto &state = g_volumeOpsState;
  state.primary.upload(f.data(), total);
  state.secondary.ensure(total);
  const bool hasMask = mask != nullptr;
  if (hasMask)
    state.mask.upload(mask->data(), total);

  launchRdpGradientKernel(state.primary.get(),
                          hasMask ? state.mask.get() : nullptr, hasMask,
                          skipOutsideMask, Nx, Ny, Nz, eps, gamma,
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
                           std::vector<double> &gradR) {
  if (!shouldUseCudaVolumeOps())
    return false;

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (f.size() != total)
    throw std::runtime_error("CUDA TV input size mismatch");
  if (mask && mask->size() != total)
    throw std::runtime_error("CUDA TV mask size mismatch");

  const double epsVal = std::max(eps, 1e-12);
  gradR.resize(total);

  auto &state = g_volumeOpsState;
  state.primary.upload(f.data(), total);
  state.secondary.ensure(total);
  const bool hasMask = mask != nullptr;
  if (hasMask)
    state.mask.upload(mask->data(), total);

  launchTvGradientKernel(state.primary.get(),
                         hasMask ? state.mask.get() : nullptr, hasMask,
                         skipOutsideMask, Nx, Ny, Nz, epsVal,
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
                                  reg.skip_outside_mask, gradR);
  case RegType::TVIsotropic:
    return computeTVGradientCUDA(f, Nx, Ny, Nz, reg.tv_eps, mask,
                                 reg.skip_outside_mask, gradR);
  default:
    return false;
  }
}

bool computeTVProxCUDA(const std::vector<double> &v, int Nx, int Ny, int Nz,
                       double lambda, int innerIters,
                       std::vector<double> &u_out) {
  if (!shouldUseCudaVolumeOps())
    return false;

  const size_t total = static_cast<size_t>(Nx) * static_cast<size_t>(Ny) *
                       static_cast<size_t>(Nz);
  if (v.size() != total)
    throw std::runtime_error("CUDA TV prox input size mismatch");
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
    launchTvProxDualKernel(state.tvUbar.get(), state.tvPx.get(),
                           state.tvPy.get(), state.tvPz.get(), sigma,
                           lambda, Nx, Ny, Nz, totalCount);

    // Primal update: u_new, u_bar extrapolation
    launchTvProxPrimalKernel(state.tvU.get(), state.tvUbar.get(),
                             state.tvV.get(), state.tvPx.get(),
                             state.tvPy.get(), state.tvPz.get(), tau,
                             theta, Nx, Ny, Nz, totalCount);
  }

  SPECT_CUDRV_CHECK(cuCtxSynchronize());
  u_out.resize(total);
  state.tvU.download(u_out.data(), total);
  return true;
}

} // namespace spect::cuda

#endif // CUDA_ENABLED
