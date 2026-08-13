// ProjectionLoader.h - Projection data loading and reordering
// Extracted from SPECT-recon.cpp for modular IO architecture
// 【T2.2】
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace spect {
namespace io {

// Forward declare MatArray (from MatFileReader.h)
struct MatArray;

/**
 * @brief Projection data container with metadata.
 */
struct ProjectionData {
    std::vector<double> data;   ///< Angle-major packed projection data
    int numAngles = 0;          ///< Number of angles
    int detNz = 0;              ///< Detector Z dimension
    int detNx = 0;              ///< Detector X dimension
    size_t raysPerAngle = 0;    ///< detNz * detNx
    bool detectorShapeKnown = true; ///< False when the source only stores rays/angle
    bool binaryFormat = false;      ///< True when loaded from the projection BIN format
    
    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
};

/**
 * @brief Optional stats from projection reordering.
 *
 * Used to preserve original diagnostic logging in callers without duplicating
 * the reordering algorithm.
 */
struct ProjectionReorderStats {
    bool isEventList = false;
    int selectedAngles = 0;
    size_t detectorsPerAngle = 0;
    size_t recordCount = 0;
};

/**
 * @brief Reorder MAT array to angle-major 1D projection vector.
 * 
 * Supports two formats:
 * - 3D array: [detZ x detX x angles] (column-major)
 * - Event list: Nx3 or 3xN with (angle, detector, value) triplets
 * 
 * @param mat Input MAT array.
 * @param requestedAngles Max angles to load (0 = all).
 * @return Angle-major packed projection data.
 * @throws std::runtime_error if format is invalid.
 */
std::vector<double> reorderProjection(const MatArray& mat, int requestedAngles);

/**
 * @brief Reorder MAT array and optionally return event-list stats.
 */
std::vector<double> reorderProjection(const MatArray& mat, int requestedAngles,
                                      ProjectionReorderStats* stats);

/**
 * @brief Return true when a path uses the projection BIN extension.
 */
bool isProjectionBinPath(const std::string& path);

/**
 * @brief Load projection data from the CUDA projection BIN format.
 *
 * The BIN layout matches 6module_bak_src/mat_convert_bin.m:
 * int32 num_angles, int32 rays_per_angle, then float64 angle-major data.
 * The header does not carry detZ/detX, so detectorShapeKnown is false and
 * callers should keep an external -detN/config shape when the product matches.
 */
ProjectionData loadProjectionBin(const std::string& path,
                                 int requestedAngles);

/**
 * @brief Load projection data from MAT file or projection BIN file.
 * 
 * @param path Path to MAT or BIN file.
 * @param varName Variable name for MAT input (empty = first variable).
 * @param requestedAngles Max angles to load (0 = all).
 * @param stats Optional reorder stats (event-list diagnostics).
 * @param outMat Optional MatArray output for info/logging.
 * @return ProjectionData with loaded and reordered data.
 */
ProjectionData loadProjection(const std::string& path, 
                               const std::string& varName,
                               int requestedAngles,
                               ProjectionReorderStats* stats,
                               MatArray* outMat);

/**
 * @brief Load projection data from MAT or projection BIN file (no stats or MatArray).
 */
ProjectionData loadProjection(const std::string& path, 
                               const std::string& varName,
                               int requestedAngles);

/**
 * @brief Load multiple bed projections.
 * 
 * @param paths List of MAT or BIN file paths.
 * @param requestedAngles Max angles per bed.
 * @return Vector of ProjectionData for each bed.
 */
std::vector<ProjectionData> loadMultiBedProjections(
    const std::vector<std::string>& paths,
    int requestedAngles);

} // namespace io
} // namespace spect
