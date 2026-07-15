#pragma once

#include "io/CSRTypes.h"

#include <cstdint>
#include <vector>

namespace spect::main {

void applyChangsAC(std::vector<double> &image, const std::vector<double> &muMap,
                   int nx, int ny, int nz, double dx, double dy, double dz,
                   int numAngles, int threads, int rotationSign = 1);

/**
 * @brief Angle-dependent parallel-ray line-integral attenuation factors.
 *
 * Geometry convention matches the 6-module system matrix generator
 * (spect_sysmat_PSC_6module_24pin_YSO v2/v4): the detector sits at +Y of the
 * rotated frame and voxels are rotated by
 *   theta_a = rotationSign * a * 2*pi/numAngles,
 *   (vcx, vcy) = (wx*cos + wy*sin, -wx*sin + wy*cos).
 * Equivalently, in image space the ray from a voxel toward the detector at
 * angle a points along (-sin(theta_a), cos(theta_a), 0).
 *
 * The factor is exp(-Int mu ds) with mu in mm^-1, integrated from the voxel
 * centre (half self-voxel contribution) to the volume boundary. Rays are
 * horizontal in Z (parallel-ray approximation; pinhole convergence is not
 * modelled).
 *
 * Result layout: factors[angle * voxels + voxel], quantised to uint16
 * (65535 == 1.0).
 */
std::vector<uint16_t> computeAttenFactors(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, double dz, int numAngles, int rotationSign, int threads);

void applyCTACToCSR(spect::io::CSR &csr, const std::vector<double> &muMap,
                     int nx, int ny, int nz, double dx, double dy, double dz,
                     int numAngles, int threads, int rotationSign = 1);

} // namespace spect::main
