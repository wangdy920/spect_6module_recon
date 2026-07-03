#pragma once

#include <cstdint>
#include <vector>

namespace spect::main {

/**
 * @brief Result of automatic mu-map generation.
 */
struct AutoMuMapResult {
  std::vector<double> muMap; ///< mu in mm^-1, same layout as the input volume
  size_t bodyVoxels = 0;     ///< voxels classified as attenuating body
  double thresholdUsed = 0;  ///< absolute activity threshold applied
  double robustMax = 0;      ///< P99.9 of positive activity values
};

/**
 * @brief Build a water-equivalent mu-map from an early-iteration activity
 *        reconstruction (CTAC bootstrap when no measured mu-map exists).
 *
 * Pipeline:
 *  1. Optional FOV subtraction: voxels where fovMask==0 are ignored.
 *  2. Threshold at thresholdFrac * P99.9(positive activity).
 *  3. Morphological closing (ball radius closingRadiusVoxels) to bridge gaps.
 *  4. Keep the largest 6-connected component (drops speckle).
 *  5. Fill internal holes (background not reachable from the volume border),
 *     so cold cavities inside a water phantom still attenuate.
 *  6. Fill the body with waterMuPerMM.
 *
 * Intended for objects whose activity roughly fills the attenuating volume
 * (water-filled phantoms typical for small-aperture SPECT). For hot-rods-in-
 * cold-body objects (e.g. Derenzo) the threshold underestimates the body;
 * a measured mu-map should be used instead.
 */
AutoMuMapResult buildAutoMuMapFromActivity(
    const std::vector<double> &activity, int nx, int ny, int nz,
    double thresholdFrac, double waterMuPerMM,
    const std::vector<uint8_t> *fovMask = nullptr,
    int closingRadiusVoxels = 2);

} // namespace spect::main
