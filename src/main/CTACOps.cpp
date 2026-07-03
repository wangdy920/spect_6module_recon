#include "main/CTACOps.h"

#include "core/AttenuationConstants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spect::main {

namespace {

constexpr double kPi = 3.14159265358979323846;

inline uint16_t quantiseFactor(double factor) {
  const double scaled = std::clamp(factor, 0.0, 1.0) * kAttenUint16Scale;
  return static_cast<uint16_t>(std::llround(scaled));
}

/// Bilinear sample of one Z-slice of the mu map at continuous voxel-index
/// coordinates (fx, fy). Outside the volume mu is 0.
inline double sampleMuBilinear(const double *slice, int nx, int ny, double fx,
                               double fy) {
  if (fx <= -1.0 || fy <= -1.0 || fx >= static_cast<double>(nx) ||
      fy >= static_cast<double>(ny))
    return 0.0;
  const double cx = std::clamp(fx, 0.0, static_cast<double>(nx - 1));
  const double cy = std::clamp(fy, 0.0, static_cast<double>(ny - 1));
  const int ix0 = static_cast<int>(std::floor(cx));
  const int iy0 = static_cast<int>(std::floor(cy));
  const int ix1 = std::min(ix0 + 1, nx - 1);
  const int iy1 = std::min(iy0 + 1, ny - 1);
  const double tx = cx - static_cast<double>(ix0);
  const double ty = cy - static_cast<double>(iy0);
  const double v00 = slice[static_cast<size_t>(iy0) * nx + ix0];
  const double v10 = slice[static_cast<size_t>(iy0) * nx + ix1];
  const double v01 = slice[static_cast<size_t>(iy1) * nx + ix0];
  const double v11 = slice[static_cast<size_t>(iy1) * nx + ix1];
  return (1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) +
         ty * ((1.0 - tx) * v01 + tx * v11);
}

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

std::vector<uint16_t> computeAttenFactorsPerVoxel(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, double dz, int numAngles, int threads) {
  (void)dx;
  (void)dy;
  (void)dz;
  (void)threads;
  validateVolume(muMap, nx, ny, nz, numAngles, "computeAttenFactorsPerVoxel");

  const size_t voxels = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  std::vector<uint16_t> factors(static_cast<size_t>(numAngles) * voxels, static_cast<uint16_t>(kAttenUint16Scale));
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1)
  for (ptrdiff_t angle = 0; angle < numAngles; ++angle) {
    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(voxels); ++idx) {
      const double factor = std::exp(-std::max(0.0, muMap[static_cast<size_t>(idx)]));
      factors[static_cast<size_t>(angle) * voxels + static_cast<size_t>(idx)] = quantiseFactor(factor);
    }
  }
  return factors;
}

std::vector<uint16_t> computeAttenFactorsLineIntegral(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, double dz, int numAngles, int rotationSign, int threads) {
  (void)dz; // rays are horizontal: each ray stays inside its own Z slice.
  validateVolume(muMap, nx, ny, nz, numAngles, "computeAttenFactorsLineIntegral");
  if (!(dx > 0.0) || !(dy > 0.0))
    throw std::runtime_error(
        "computeAttenFactorsLineIntegral: voxel spacing must be positive");
  const int sign = (rotationSign < 0) ? -1 : 1;

  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t voxels = plane * static_cast<size_t>(nz);
  std::vector<uint16_t> factors(static_cast<size_t>(numAngles) * voxels,
                                static_cast<uint16_t>(kAttenUint16Scale));

  // Midpoint-rule ray marching with a half-voxel step. The integral starts at
  // the voxel centre, so roughly half of the emitting voxel's own attenuation
  // is included, matching the usual first-order CTAC convention.
  const double step = 0.5 * std::min(dx, dy);
  // Longest possible in-plane path (diagonal of the physical slice).
  const double maxLen = std::hypot(static_cast<double>(nx) * dx,
                                   static_cast<double>(ny) * dy);
  const int maxSteps = static_cast<int>(std::ceil(maxLen / step)) + 2;

#pragma omp parallel for schedule(dynamic) num_threads(threads > 0 ? threads : 1)
  for (ptrdiff_t angle = 0; angle < numAngles; ++angle) {
    const double theta = static_cast<double>(sign) * static_cast<double>(angle) *
                         (2.0 * kPi / static_cast<double>(numAngles));
    // Ray direction toward the detector, expressed in image space. The system
    // matrix generator rotates voxels by theta and places the detector at +Y
    // of the rotated frame; mapping that +Y back to image space gives
    // (-sin(theta), cos(theta)).
    const double rayX = -std::sin(theta);
    const double rayY = std::cos(theta);
    // Per-step advance in fractional voxel-index units.
    const double stepFx = rayX * step / dx;
    const double stepFy = rayY * step / dy;

    uint16_t *out = factors.data() + static_cast<size_t>(angle) * voxels;
    for (int iz = 0; iz < nz; ++iz) {
      const double *slice = muMap.data() + static_cast<size_t>(iz) * plane;
      uint16_t *outSlice = out + static_cast<size_t>(iz) * plane;
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          double fx = static_cast<double>(ix) + 0.5 * stepFx;
          double fy = static_cast<double>(iy) + 0.5 * stepFy;
          double integral = 0.0;
          for (int k = 0; k < maxSteps; ++k) {
            if (fx <= -1.0 || fy <= -1.0 || fx >= static_cast<double>(nx) ||
                fy >= static_cast<double>(ny))
              break;
            const double mu = sampleMuBilinear(slice, nx, ny, fx, fy);
            if (mu > 0.0)
              integral += mu * step;
            fx += stepFx;
            fy += stepFy;
          }
          outSlice[static_cast<size_t>(iy) * nx + ix] =
              quantiseFactor(std::exp(-integral));
        }
      }
    }
  }
  return factors;
}

std::vector<uint16_t> computeAttenFactors(
    CTACModel model, const std::vector<double> &muMap, int nx, int ny, int nz,
    double dx, double dy, double dz, int numAngles, int rotationSign,
    int threads) {
  if (model == CTACModel::LineIntegral)
    return computeAttenFactorsLineIntegral(muMap, nx, ny, nz, dx, dy, dz,
                                           numAngles, rotationSign, threads);
  return computeAttenFactorsPerVoxel(muMap, nx, ny, nz, dx, dy, dz, numAngles,
                                     threads);
}

void applyCTACToCSR(spect::io::CSR &csr, const std::vector<double> &muMap,
                    int nx, int ny, int nz, double dx, double dy, double dz,
                    int numAngles, int threads, CTACModel model,
                    int rotationSign) {
  if (numAngles <= 0)
    throw std::runtime_error("applyCTACToCSR: numAngles must be positive");

  const std::vector<uint16_t> factors = computeAttenFactors(
      model, muMap, nx, ny, nz, dx, dy, dz, numAngles, rotationSign, threads);
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
                   int numAngles, int threads, CTACModel model,
                   int rotationSign) {
  const std::vector<uint16_t> factors = computeAttenFactors(
      model, muMap, nx, ny, nz, dx, dy, dz, numAngles, rotationSign, threads);
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
