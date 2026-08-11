#pragma once

namespace spect::core::sixmodule {

inline constexpr int kOutputMaskNx = 100;
inline constexpr int kOutputMaskNy = 100;
inline constexpr int kSingleBedMaskNz = 72;
inline constexpr int kDefaultBlendMarginLayers = 10;

inline constexpr double kVoxelSizeMm = 0.3;
inline constexpr double kMultiBedStepMm = 16.8;
inline constexpr double kSingleBedRoiDiameterMm = 30.0;
inline constexpr double kSingleBedRoiHeightMm = 21.6;
inline constexpr double kMultiBedRoiDiameterMm = 36.0;
inline constexpr double kMultiBedGainRoiDiameterMm = 30.0;

inline constexpr double kMultiBedRoiHeightMm = 66.0;

} // namespace spect::core::sixmodule
