#include "main/CTACOps.h"

#include "core/AttenuationConstants.h"
#include "cuda/CudaConfig.h"
#include "cuda/VolumeOpsCUDA.h"

#include <algorithm>
#include <stdexcept>

namespace spect::main {

namespace {

void validateVolume(const std::vector<double> &muMap, int nx, int ny, int nz,
                    int numAngles, const char *fn) {
  if (numAngles <= 0)
    throw std::runtime_error(std::string(fn) + ": numAngles must be positive");
  const size_t voxels =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  if (nx <= 0 || ny <= 0 || nz <= 0 || muMap.size() != voxels)
    throw std::runtime_error(std::string(fn) + ": muMap size mismatch");
}

} // namespace

std::vector<uint16_t> computeAttenFactors(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, double dz, int numAngles, int rotationSign, int threads) {
  (void)dz;
  (void)threads;
  validateVolume(muMap, nx, ny, nz, numAngles, "computeAttenFactors");
  std::vector<uint16_t> factors;
  if (!spect::cuda::computeAttenFactorsLineIntegralCUDA(
          muMap, nx, ny, nz, dx, dy, numAngles, rotationSign, factors))
    throw std::runtime_error("CUDA CTAC factor generation is unavailable");
  return factors;
}

std::vector<std::vector<uint16_t>> computeAttenFactorsMultiBed(
    const std::vector<std::vector<double>> &localMuMaps, int nx, int ny, int nz,
    double dx, double dy, double dz, int numAngles, int rotationSign,
    int threads, int cudaDeviceId) {
  const int bedCount = static_cast<int>(localMuMaps.size());
  std::vector<std::vector<uint16_t>> factors(
      static_cast<size_t>(bedCount));

  const int usedThreads =
      std::max(1, threads > 0 ? threads : 1);
  const int usedBedThreads = std::min(bedCount, usedThreads);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(usedBedThreads)
#endif
  for (int bed = 0; bed < bedCount; ++bed) {
#if defined(SPECT_HAS_CUDA)
    spect::cuda::setDevice(cudaDeviceId);
#endif
    factors[static_cast<size_t>(bed)] = computeAttenFactors(
        localMuMaps[static_cast<size_t>(bed)], nx, ny, nz, dx, dy, dz,
        numAngles, rotationSign, 1);
  }

  return factors;
}

void applyCTACToCSR(spect::io::CSR &csr, const std::vector<double> &muMap,
                     int nx, int ny, int nz, double dx, double dy, double dz,
                     int numAngles, int threads, int rotationSign) {
  if (numAngles <= 0)
    throw std::runtime_error("applyCTACToCSR: numAngles must be positive");

  const std::vector<uint16_t> factors = computeAttenFactors(
      muMap, nx, ny, nz, dx, dy, dz, numAngles, rotationSign, threads);
  const size_t voxels = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  const size_t raysPerAngle = static_cast<size_t>(csr.rows) / static_cast<size_t>(numAngles);
  const double invScale = kAttenUint16InvScale;

#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1)
  for (ptrdiff_t row = 0; row < csr.rows; ++row) {
    const int angle = (raysPerAngle > 0) ? std::min<int>(static_cast<int>(static_cast<size_t>(row) / raysPerAngle), numAngles - 1) : 0;
    const uint16_t *att = factors.data() + static_cast<size_t>(angle) * voxels;
    for (size_t t = csr.row_ptr[static_cast<size_t>(row)]; t < csr.row_ptr[static_cast<size_t>(row) + 1]; ++t) {
      csr.val[t] = static_cast<float>(static_cast<double>(csr.val[t]) * static_cast<double>(att[csr.col_idx[t]]) * invScale);
    }
  }
}

void applyChangsAC(std::vector<double> &image, const std::vector<double> &muMap,
                   int nx, int ny, int nz, double dx, double dy, double dz,
                   int numAngles, int threads, int rotationSign) {
  const std::vector<uint16_t> factors = computeAttenFactors(
      muMap, nx, ny, nz, dx, dy, dz, numAngles, rotationSign, threads);
  const size_t voxels = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  if (image.size() != voxels)
    throw std::runtime_error("applyChangsAC: image size mismatch");

#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1)
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(voxels); ++idx) {
    double avg = 0.0;
    for (int angle = 0; angle < numAngles; ++angle)
      avg += static_cast<double>(factors[static_cast<size_t>(angle) * voxels + static_cast<size_t>(idx)]) * kAttenUint16InvScale;
    avg /= static_cast<double>(numAngles);
    if (avg > 0.0)
      image[static_cast<size_t>(idx)] /= avg;
  }
}

} // namespace spect::main
