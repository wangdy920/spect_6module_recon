#pragma once

#include "io/CSRTypes.h"

#include <cstdint>
#include <vector>

namespace spect::main {

/**
 * @brief Attenuation factor model used by CTAC / Chang's AC.
 *
 * - PerVoxelApprox: legacy behaviour, factor = exp(-mu[voxel]) replicated to
 *   every angle. Ignores voxel spacing, angle and ray path. Kept only for
 *   backwards comparison; not quantitative.
 * - LineIntegral: first-order parallel-ray attenuation,
 *   factor(angle, voxel) = exp(-Int mu ds) integrated from the voxel centre
 *   toward the detector along the angle-dependent ray direction. The mu map
 *   must be in mm^-1 and is integrated with the physical voxel spacing.
 */
enum class CTACModel { PerVoxelApprox, LineIntegral };

void applyChangsAC(std::vector<double> &image, const std::vector<double> &muMap,
                   int nx, int ny, int nz, double dx, double dy, double dz,
                   int numAngles, int threads,
                   CTACModel model = CTACModel::LineIntegral,
                   int rotationSign = 1);

/// Legacy per-voxel exp(-mu) factors replicated per angle (see CTACModel).
std::vector<uint16_t> computeAttenFactorsPerVoxel(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, double dz, int numAngles, int threads);

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
 * (65535 == 1.0), identical to computeAttenFactorsPerVoxel.
 */
std::vector<uint16_t> computeAttenFactorsLineIntegral(
    const std::vector<double> &muMap, int nx, int ny, int nz, double dx,
    double dy, double dz, int numAngles, int rotationSign, int threads);

/// Dispatch helper over CTACModel.
std::vector<uint16_t> computeAttenFactors(
    CTACModel model, const std::vector<double> &muMap, int nx, int ny, int nz,
    double dx, double dy, double dz, int numAngles, int rotationSign,
    int threads);

void applyCTACToCSR(spect::io::CSR &csr, const std::vector<double> &muMap,
                    int nx, int ny, int nz, double dx, double dy, double dz,
                    int numAngles, int threads,
                    CTACModel model = CTACModel::LineIntegral,
                    int rotationSign = 1);

} // namespace spect::main
