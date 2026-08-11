/**
 * @file Masks.h
 * @brief Mask generation and manipulation utilities
 * 
 * Migrated from original_source_backup/Masks.h
 */

#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spect {
namespace core {

/**
 * @brief Column-major 3D index
 */
inline size_t idx3(size_t i, size_t j, size_t k, size_t Nx, size_t Ny) {
    return i + Nx * j + Nx * Ny * k;
}

/**
 * @brief Create cylindrical mask with dimensions in mm
 * 
 * @param Nx, Ny, Nz Image dimensions
 * @param voxXmm, voxYmm, voxZmm Voxel sizes in mm
 * @param diameter_mm Cylinder diameter in mm
 * @param height_mm Cylinder height in mm (0 = infinite)
 * @param centerX, centerY, centerZ Center coordinates (1-based, NaN = geometric center)
 * @return Binary mask (uint8_t, 1 = inside)
 */
inline std::vector<uint8_t> makeCylMask_mm(
    int Nx, int Ny, int Nz,
    double voxXmm, double voxYmm, double voxZmm,
    double diameter_mm, double height_mm,
    double centerX = std::numeric_limits<double>::quiet_NaN(),
    double centerY = std::numeric_limits<double>::quiet_NaN(),
    double centerZ = std::numeric_limits<double>::quiet_NaN())
{
    if (Nx <= 0 || Ny <= 0 || Nz <= 0) return {};

    // Use voxel-centered coordinates
    const bool cx_nan = std::isnan(centerX);
    const bool cy_nan = std::isnan(centerY);
    const bool cz_nan = std::isnan(centerZ);
    const double cx_vox = cx_nan ? 0.5 * (static_cast<double>(Nx) - 1.0) : (centerX - 1.0);
    const double cy_vox = cy_nan ? 0.5 * (static_cast<double>(Ny) - 1.0) : (centerY - 1.0);
    const double cz_vox = cz_nan ? 0.5 * (static_cast<double>(Nz) - 1.0) : (centerZ - 1.0);

    std::vector<uint8_t> m(static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz), 0);
    const double radius = std::max(0.0, diameter_mm * 0.5);
    const double halfHeight = (height_mm > 0.0) ? (height_mm * 0.5) : std::numeric_limits<double>::infinity();

    const double r2 = radius * radius + 1e-12;
    const double hz = halfHeight + 1e-12;

    for (int k = 0; k < Nz; ++k) {
        const double z_mm = (static_cast<double>(k) - cz_vox) * voxZmm;
        if (std::abs(z_mm) > hz) continue;

        for (int j = 0; j < Ny; ++j) {
            const double y_mm = (static_cast<double>(j) - cy_vox) * voxYmm;

            for (int i = 0; i < Nx; ++i) {
                const double x_mm = (static_cast<double>(i) - cx_vox) * voxXmm;
                if (x_mm * x_mm + y_mm * y_mm <= r2) {
                    m[idx3(static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(k),
                           static_cast<size_t>(Nx), static_cast<size_t>(Ny))] = 1;
                }
            }
        }
    }
    return m;
}

/**
 * @brief Apply binary mask to volume (in-place)
 * 
 * Sets values to 0 where mask is 0
 */
inline void applyMaskInplace(std::vector<double>& F, const std::vector<uint8_t>& mask) {
    if (F.size() != mask.size()) 
        throw std::runtime_error("applyMaskInplace: size mismatch");
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t t = 0; t < static_cast<ptrdiff_t>(F.size()); ++t) {
        if (!mask[static_cast<size_t>(t)]) 
            F[static_cast<size_t>(t)] = 0.0;
    }
}

/**
 * @brief Create spherical mask
 * 
 * @param Nx, Ny, Nz Image dimensions
 * @param voxXmm, voxYmm, voxZmm Voxel sizes in mm
 * @param diameter_mm Sphere diameter in mm
 * @param centerX, centerY, centerZ Center coordinates (1-based, NaN = geometric center)
 * @return Binary mask
 */
inline std::vector<uint8_t> makeSphereMask_mm(
    int Nx, int Ny, int Nz,
    double voxXmm, double voxYmm, double voxZmm,
    double diameter_mm,
    double centerX = std::numeric_limits<double>::quiet_NaN(),
    double centerY = std::numeric_limits<double>::quiet_NaN(),
    double centerZ = std::numeric_limits<double>::quiet_NaN())
{
    if (Nx <= 0 || Ny <= 0 || Nz <= 0) return {};

    const bool cx_nan = std::isnan(centerX);
    const bool cy_nan = std::isnan(centerY);
    const bool cz_nan = std::isnan(centerZ);
    const double cx_vox = cx_nan ? 0.5 * (static_cast<double>(Nx) - 1.0) : (centerX - 1.0);
    const double cy_vox = cy_nan ? 0.5 * (static_cast<double>(Ny) - 1.0) : (centerY - 1.0);
    const double cz_vox = cz_nan ? 0.5 * (static_cast<double>(Nz) - 1.0) : (centerZ - 1.0);

    std::vector<uint8_t> m(static_cast<size_t>(Nx) * static_cast<size_t>(Ny) * static_cast<size_t>(Nz), 0);
    const double radius = std::max(0.0, diameter_mm * 0.5);
    const double r2 = radius * radius + 1e-12;

    for (int k = 0; k < Nz; ++k) {
        const double z_mm = (static_cast<double>(k) - cz_vox) * voxZmm;
        for (int j = 0; j < Ny; ++j) {
            const double y_mm = (static_cast<double>(j) - cy_vox) * voxYmm;
            for (int i = 0; i < Nx; ++i) {
                const double x_mm = (static_cast<double>(i) - cx_vox) * voxXmm;
                if (x_mm * x_mm + y_mm * y_mm + z_mm * z_mm <= r2) {
                    m[idx3(static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(k),
                           static_cast<size_t>(Nx), static_cast<size_t>(Ny))] = 1;
                }
            }
        }
    }
    return m;
}

/**
 * @brief Create mask from threshold
 * 
 * @param vol Input volume
 * @param threshold Threshold value
 * @param above If true, mask voxels >= threshold; otherwise < threshold
 * @return Binary mask
 */
inline std::vector<uint8_t> makeMaskFromThreshold(
    const std::vector<double>& vol,
    double threshold,
    bool above = true)
{
    std::vector<uint8_t> mask(vol.size(), 0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(vol.size()); ++i) {
        if (above) {
            mask[i] = (vol[i] >= threshold) ? 1 : 0;
        } else {
            mask[i] = (vol[i] < threshold) ? 1 : 0;
        }
    }
    return mask;
}

} // namespace core
} // namespace spect
