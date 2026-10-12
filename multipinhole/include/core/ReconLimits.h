#pragma once

namespace spect::core {

// Matrix-grid (SC60) scans step the bed in a few mm against a ~17 mm axial
// sensitivity FWHM, so a full scan needs 10-15 positions.
constexpr int kMaxBeds = 16;

// PSC six-module scans keep the fixed Bed1..Bed5 overlap table.
constexpr int kSixModuleMaxBeds = 5;

} // namespace spect::core
