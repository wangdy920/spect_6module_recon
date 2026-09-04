#pragma once

#include "cuda/CudaConfig.h"
#include "filters/PSFConfig.h"
#include "regularizer/RegConfig.h"

#include <cstdint>
#include <vector>

namespace spect::cuda {

#if CUDA_ENABLED

bool isCudaVolumeOpsAvailable();

bool computeAttenFactorsLineIntegralCUDA(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, int numAngles, int rotationSign,
    std::vector<uint16_t> &factors);

bool applySeparableConv3DInplaceCUDA(std::vector<double> &f, int Nx, int Ny,
                                     int Nz,
                                     const std::vector<double> &kx,
                                     const std::vector<double> &ky,
                                     const std::vector<double> &kz,
                                     bool useFloat = true);

// CASToR 前向卷积 C = D_z^-1 * K：零扩展可分离卷积（无镜像补偿），
// 随后逐体素除以 z 边界补偿因子（norm[z] = 落入 [0,Nz) 的 kz 权重和）。
// 反向（伴随侧）按 CASToR 源码复用同一前向调用。
bool applyCastorConv3DInplaceCUDA(std::vector<double> &f, int Nx, int Ny,
                                  int Nz,
                                  const std::vector<double> &kx,
                                  const std::vector<double> &ky,
                                  const std::vector<double> &kz,
                                  bool useFloat = true);

// Enqueue the same float32 CASToR convolution on already-resident device
// buffers. primary contains the input and is used as intermediate storage;
// secondary receives the final result. Work is ordered on the default stream.
bool enqueueCastorConv3DFloatDeviceCUDA(
    float *primary, float *secondary, int Nx, int Ny, int Nz,
    const std::vector<double> &kx, const std::vector<double> &ky,
    const std::vector<double> &kz);

// CASToR forward convolution with an unconditional float32 staging/kernel
// path. This entry intentionally ignores SPECT_CUDA_PSF_DOUBLE.
bool applyCastorConv3DInplaceCUDAForcedFloat32Staging(
    std::vector<double> &f, int Nx, int Ny, int Nz,
    const std::vector<double> &kx, const std::vector<double> &ky,
    const std::vector<double> &kz);

bool applySeparableConv3DAdjointInplaceCUDA(
    std::vector<double> &f, int Nx, int Ny, int Nz,
    const std::vector<double> &kx, const std::vector<double> &ky,
    const std::vector<double> &kz, bool useFloat = true);

bool computeRDPGradientCUDA(const std::vector<double> &f, int Nx, int Ny,
                            int Nz, double delta, double gamma,
                            const std::vector<uint8_t> *mask,
                            bool skipOutsideMask,
                            std::vector<double> &gradR,
                            const std::vector<double> *ctImage = nullptr,
                            double ctSigma = 0.01,
                            double ctWeightFloor = 0.05);

bool computeTVGradientCUDA(const std::vector<double> &f, int Nx, int Ny,
                           int Nz, double eps,
                           const std::vector<uint8_t> *mask,
                           bool skipOutsideMask,
                           std::vector<double> &gradR,
                           const std::vector<double> *ctImage = nullptr,
                           double ctSigma = 0.01,
                           double ctWeightFloor = 0.05);

/// Unified MRF regularizer gradient dispatch.
/// Returns true if the reg type has a CUDA kernel and computation succeeded.
bool computeRegGradientCUDA(const std::vector<double> &f, int Nx, int Ny,
                            int Nz,
                            const spect::RegConf &reg,
                            const std::vector<uint8_t> *mask,
                            std::vector<double> &gradR);

/// Same dispatch on device arrays: dImage (Nx*Ny*Nz doubles, e.g. the
/// resident reconstruction image) -> dGrad, no host round trip. mask and the
/// registered CT are host arrays uploaded once per pointer. Returns false when
/// the reg type has no CUDA kernel.
bool computeRegGradientDeviceCUDA(const double *dImage, int Nx, int Ny, int Nz,
                                  const spect::RegConf &reg,
                                  const std::vector<uint8_t> *mask,
                                  double *dGrad);

/// TV proximal via Chambolle-Pock primal-dual on GPU.
/// Solves: u = argmin_u (1/2)||u-v||^2 + lambda*TV(u)
/// Returns true only when the CUDA kernel ran successfully.
bool computeTVProxCUDA(const std::vector<double> &v, int Nx, int Ny, int Nz,
                       double lambda, int innerIters,
                       std::vector<double> &u_out,
                       const std::vector<double> *ctImage = nullptr,
                       double ctSigma = 0.01,
                       double ctWeightFloor = 0.05);

#else

inline bool isCudaVolumeOpsAvailable() { return false; }

inline bool computeAttenFactorsLineIntegralCUDA(
    const std::vector<double> &, int, int, int, double, double, int, int,
    std::vector<uint16_t> &) {
  return false;
}

inline bool applySeparableConv3DInplaceCUDA(
    std::vector<double> &, int, int, int, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &,
    bool = true) {
  return false;
}

inline bool applyCastorConv3DInplaceCUDA(
    std::vector<double> &, int, int, int, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &,
    bool = true) {
  return false;
}

inline bool enqueueCastorConv3DFloatDeviceCUDA(
    float *, float *, int, int, int, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &) {
  return false;
}

inline bool applyCastorConv3DInplaceCUDAForcedFloat32Staging(
    std::vector<double> &, int, int, int, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &) {
  return false;
}

inline bool applySeparableConv3DAdjointInplaceCUDA(
    std::vector<double> &, int, int, int, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &,
    bool = true) {
  return false;
}

inline bool computeRDPGradientCUDA(const std::vector<double> &, int, int, int,
                                   double, double,
                                   const std::vector<uint8_t> *, bool,
                                   std::vector<double> &,
                                   const std::vector<double> * = nullptr,
                                   double = 0.01, double = 0.05) {
  return false;
}

inline bool computeTVGradientCUDA(const std::vector<double> &, int, int, int,
                                  double,
                                  const std::vector<uint8_t> *, bool,
                                  std::vector<double> &,
                                  const std::vector<double> * = nullptr,
                                  double = 0.01, double = 0.05) {
  return false;
}

inline bool computeRegGradientCUDA(const std::vector<double> &, int, int, int,
                                   const spect::RegConf &,
                                   const std::vector<uint8_t> *,
                                   std::vector<double> &) {
  return false;
}

inline bool computeRegGradientDeviceCUDA(const double *, int, int, int,
                                         const spect::RegConf &,
                                         const std::vector<uint8_t> *,
                                         double *) {
  return false;
}

inline bool computeTVProxCUDA(const std::vector<double> &, int, int, int,
                              double, int, std::vector<double> &,
                              const std::vector<double> * = nullptr,
                              double = 0.01, double = 0.05) {
  return false;
}

#endif

} // namespace spect::cuda
