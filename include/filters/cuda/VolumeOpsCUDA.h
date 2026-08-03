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

inline bool computeTVProxCUDA(const std::vector<double> &, int, int, int,
                              double, int, std::vector<double> &,
                              const std::vector<double> * = nullptr,
                              double = 0.01, double = 0.05) {
  return false;
}

#endif

} // namespace spect::cuda
