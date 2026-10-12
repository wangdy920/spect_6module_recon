#include "pipeline/ReconSetup.h"
#include "pipeline/VolumeOutput.h"

#include "core/BedPathUtils.h"
#include "core/ReconLimits.h"
#include "io/VolumeLoader.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <sstream>

namespace spect {
namespace main {

namespace {

std::string selectReconDimHintPath(const ReconArgs &args) {
  if (args.dimExplicit)
    return {};

  if (args.doCTAC && !args.attMapPath.empty())
    return args.attMapPath;

  if (args.useReg && !args.regCtPath.empty())
    return args.regCtPath;

  if (!std::isnan(args.maskFromAttThreshold) && !args.attMapPath.empty())
    return args.attMapPath;

  return {};
}

spect::io::VolumeInfo readVolumeInfoForRecon(const std::string &path) {
  const std::string ext = std::filesystem::path(path).extension().string();
  if (ext == ".mhd")
    return spect::io::parseMHD(path);
  if (ext == ".hdr") {
    if (spect::io::isBinaryAnalyze(path))
      return spect::io::parseHDR(path);
    return spect::io::parseInterfile(path);
  }
  throw std::runtime_error("Unsupported volume format for dimension hint: " +
                           path);
}

std::vector<int> inferBedNumbers(const std::vector<std::string> &paths,
                                 int maxBed) {
  std::vector<int> bedNumbers;
  bedNumbers.reserve(paths.size());
  std::vector<bool> seen(static_cast<size_t>(maxBed) + 1u, false);
  for (size_t idx = 0; idx < paths.size(); ++idx) {
    int bed = spect::core::extractBedNumberFromPath(paths[idx]);
    if (bed < 0)
      bed = static_cast<int>(idx) + 1;
    if (bed < 1 || bed > maxBed)
      throw std::runtime_error("Bed number must be in fixed range Bed1..Bed" +
                               std::to_string(maxBed) + ".");
    if (seen[static_cast<size_t>(bed)])
      throw std::runtime_error("Duplicate bed number in -dfList.");
    seen[static_cast<size_t>(bed)] = true;
    bedNumbers.push_back(bed);
  }
  return bedNumbers;
}

std::vector<int> normalizeFixedBedOverlapTable(
    const std::vector<int> &input, std::vector<std::string> &warnings) {
  std::vector<int> fixedTable(
      static_cast<size_t>(spect::core::kSixModuleMaxBeds), 0);
  if (input.empty())
    return fixedTable;
  if (input.size() == 1) {
    std::fill(fixedTable.begin(), fixedTable.end(), input.front());
    return fixedTable;
  }
  if (input.size() > fixedTable.size())
    throw std::runtime_error("-bedOverlap supports at most " +
                             std::to_string(spect::core::kSixModuleMaxBeds) +
                             " fixed bed slots.");
  std::copy(input.begin(), input.end(), fixedTable.begin());
  if (input.size() < fixedTable.size()) {
    warnings.emplace_back(
        "bedOverlap fixed Bed1..Bed5 table shorter than 5 slots. Missing fixed slots use zero.");
  }
  return fixedTable;
}

std::vector<int> selectActualBedOverlaps(
    const std::vector<int> &fixedTable, const std::vector<int> &bedNumbers) {
  std::vector<int> overlaps;
  if (bedNumbers.size() < 2)
    return overlaps;
  overlaps.reserve(bedNumbers.size() - 1);
  for (size_t idx = 0; idx + 1 < bedNumbers.size(); ++idx) {
    const int leftBed = bedNumbers[idx];
    overlaps.push_back(fixedTable[static_cast<size_t>(leftBed - 1)]);
  }
  return overlaps;
}

void applyFixedSixModuleRoiDefaults(ReconArgs &args, int bedCount) {
  if (args.cylinderDimsExplicit)
    return;
  if (bedCount == 1) {
    args.cylinderDiameterMM = static_cast<float>(
        spect::core::sixmodule::kSingleBedRoiDiameterMm);
    args.cylinderHeightMM = static_cast<float>(
        spect::core::sixmodule::kSingleBedRoiHeightMm);
    return;
  }
  args.cylinderDiameterMM = static_cast<float>(
      spect::core::sixmodule::kMultiBedRoiDiameterMm);
  args.cylinderHeightMM = static_cast<float>(
      spect::core::sixmodule::kMultiBedRoiHeightMm);
}

} // namespace

bool adoptReconDimsFromHint(ReconArgs &args, std::string *warningMessage) {
  const std::string hintPath = selectReconDimHintPath(args);
  if (hintPath.empty()) {
    if (warningMessage)
      warningMessage->clear();
    return false;
  }

  const spect::io::VolumeInfo info = readVolumeInfoForRecon(hintPath);
  if (info.Nx <= 0 || info.Ny <= 0 || info.Nz <= 0)
    throw std::runtime_error("Invalid volume dimensions in hint file: " + hintPath);
  int hintNz = info.Nz;
  if (!info.spectBeds.empty()) {
    if (info.Nz % static_cast<int>(info.spectBeds.size()) != 0)
      throw std::runtime_error(
          "Tagged multi-bed map Nz is not divisible by its SPECTBeds count: " +
          hintPath);
    hintNz /= static_cast<int>(info.spectBeds.size());
  }

  if (args.Nx == info.Nx && args.Ny == info.Ny && args.Nz == hintNz) {
    if (warningMessage)
      warningMessage->clear();
    return false;
  }

  args.Nx = info.Nx;
  args.Ny = info.Ny;
  args.Nz = hintNz;

  if (warningMessage) {
    std::ostringstream warning;
    warning << "Reconstruction dimensions were not explicitly set; adopting "
            << info.Nx << 'x' << info.Ny << 'x' << hintNz
            << " from " << hintPath;
    *warningMessage = warning.str();
  }
  return true;
}

bool ensureReconDims(ReconArgs &args, std::string *warningMessage) {
  bool changed = false;
  std::ostringstream warning;
  if (args.Nx <= 0) {
    args.Nx = 200;
    changed = true;
  }
  if (args.Ny <= 0) {
    args.Ny = 200;
    changed = true;
  }
  if (args.Nz <= 0) {
    args.Nz = 200;
    changed = true;
  }

  if (changed) {
    const bool matrixInput = !args.chunkCSRDir.empty();
    if (matrixInput && !args.dimExplicit) {
      throw std::runtime_error(
          "Reconstruction dimensions are required for matrix inputs. Set -dim "
          "or provide compact metadata.");
    }
    warning << "Reconstruction dimensions were missing or invalid; using defaults: "
            << args.Nx << 'x' << args.Ny << 'x' << args.Nz;
    if (warningMessage)
      *warningMessage = warning.str();
  } else if (warningMessage) {
    warningMessage->clear();
  }
  return changed;
}

BedSetupResult prepareBedSetup(ReconArgs &args) {
  BedSetupResult result;
  result.baseNx = args.Nx;
  result.baseNy = args.Ny;
  result.baseNz = args.Nz;
  result.projectionPaths = spect::core::sortPathsByBedOrder(
      args.projPaths.empty() ? std::vector<std::string>{args.projPath}
                             : args.projPaths);
  result.bedCount = std::max(1, static_cast<int>(result.projectionPaths.size()));
  const int maxBeds = args.matrixGridOutput ? spect::core::kMaxBeds
                                            : spect::core::kSixModuleMaxBeds;
  if (result.bedCount > maxBeds)
    throw std::runtime_error("-dfList supports at most " +
                             std::to_string(maxBeds) + " beds" +
                             (args.matrixGridOutput ? "." : " without matrix_grid_output."));
  args.bedCount = result.bedCount;
  if (!args.matrixGridOutput)
    applyFixedSixModuleRoiDefaults(args, result.bedCount);
  result.bedNumbers = inferBedNumbers(result.projectionPaths, maxBeds);
  for (size_t index = 1; index < result.bedNumbers.size(); ++index) {
    if (result.bedNumbers[index] != result.bedNumbers[index - 1] + 1)
      throw std::runtime_error(
          "-dfList beds must be one continuous increasing subset of Bed1..Bed" +
          std::to_string(maxBeds) + ".");
  }

  const int joints = std::max(0, result.bedCount - 1);
  const double voxelSizeZ =
      args.dz > 0.0 ? args.dz : static_cast<double>(args.Vz);
  if (args.matrixGridOutput && joints > 0) {
    if (!std::isfinite(args.bedStepMm) || args.bedStepMm <= 0.0 ||
        !std::isfinite(voxelSizeZ) || voxelSizeZ <= 0.0 ||
        args.bedStepMm / voxelSizeZ > result.baseNz + 0.5)
      throw std::runtime_error("Multi-bed matrix_grid_output requires an explicit finite bed_step_mm in (0, baseNz*voxel_z].");
    const int step = static_cast<int>(std::llround(args.bedStepMm / voxelSizeZ));
    if (step < 1 || step > result.baseNz || std::abs(step * voxelSizeZ - args.bedStepMm) > 1e-4)
      throw std::runtime_error("matrix_grid_output bed_step_mm must align to integer voxel layers (tolerance 1e-4 mm).");
    if (args.seamXyAlign)
      throw std::runtime_error("matrix_grid_output does not apply prototype seam XY alignment.");
    // CTAC and -maskFromAtt read mu maps stacked as one local-matrix-Z block
    // per bed; the joint volume has no such blocks.
    if (args.doCTAC || !std::isnan(args.maskFromAttThreshold) ||
        !args.ctacAutoMuFrom.empty())
      throw std::runtime_error(
          "Multi-bed matrix_grid_output (joint volume) does not support CTAC, "
          "-ctacAutoMu or -maskFromAtt yet.");
    // Joint reconstruction: every bed grid is placed at bed*step in one global
    // volume, so overlapping beds update the same voxels (sensitivities and
    // backprojections add). The saved volume is that global volume; no
    // per-bed output blending.
    result.canonicalGeom = {result.baseNz, step, result.baseNz - step};
    args.outputBedOverlaps.assign(joints, result.baseNz - step);
    args.bedOverlaps.assign(joints, result.baseNz - step);
    args.Nz = result.baseNz + joints * step;
    result.totalOverlap = joints * (result.baseNz - step);
    result.summary = "Multi-bed reconstruction: joint global volume of full matrix grids, explicit voxel-aligned step " +
                     std::to_string(step) + " layers";
    return result;
  }
  const double stepMm =
      args.bedStepMm > 0.0 ? args.bedStepMm
                           : spect::core::sixmodule::kMultiBedStepMm;
  const int expectedStep = voxelSizeZ > 0.0
      ? static_cast<int>(std::llround(stepMm / voxelSizeZ))
      : 0;
  const int expectedOverlap = result.baseNz - expectedStep;
  if (joints == 0) {
    if (args.singleBedStepAligned && !args.matrixGridOutput) {
      result.canonicalGeom = computeCanonicalOutputGeometry(
          result.baseNz, {expectedOverlap}, voxelSizeZ, args.bedStepMm);
    }
    args.bedOverlaps.clear();
    args.outputBedOverlaps.clear();
    result.summary = args.matrixGridOutput
        ? "Single-bed reconstruction: full matrix-grid output"
        : args.singleBedStepAligned
        ? "Single-bed reconstruction: output crop follows bed step"
        : "Single-bed reconstruction: legacy 72-layer output crop";
    return result;
  }

  const std::vector<int> fixedOverlapTable =
      normalizeFixedBedOverlapTable(args.bedOverlaps, result.warnings);
  args.outputBedOverlaps =
      selectActualBedOverlaps(fixedOverlapTable, result.bedNumbers);
  if (static_cast<int>(args.outputBedOverlaps.size()) != joints)
    throw std::runtime_error("Internal bed overlap normalization failed.");

  bool clampedAnyOverlap = false;
  for (int &overlap : args.outputBedOverlaps) {
    const int clamped = std::max(0, std::min(overlap, std::max(0, result.baseNz - 1)));
    if (clamped != overlap) {
      overlap = clamped;
      clampedAnyOverlap = true;
    }
  }
  if (clampedAnyOverlap) {
    result.warnings.emplace_back(
        "bedOverlap values were clamped to the valid range [0, baseNz)."
    );
  }

  // The bed overlap is a derived quantity: baseNz - step, where
  // step = round(bed_step_mm / voxel_z) and bed_step_mm (CLI -bedStepMm /
  // INI [MultiBed] bed_step_mm, default 16.8 mm) is the authoritative
  // scanner property. When the user omits -bedOverlap the fixed table is all
  // zeros; when it is wrong it disagrees with the derived step. In both
  // cases we re-derive the overlap from the scanner step.
  if (expectedOverlap <= 0 || expectedOverlap >= result.baseNz) {
    throw std::runtime_error(
        "Scanner bed step " + std::to_string(stepMm) +
        " mm is incompatible with baseNz=" + std::to_string(result.baseNz) +
        " at voxel_z=" + std::to_string(voxelSizeZ) +
        " mm (derived overlap would be " + std::to_string(expectedOverlap) +
        "). Check -bedStepMm / [MultiBed] bed_step_mm.");
  }
  bool overlapCorrected = false;
  for (size_t i = 0; i < args.outputBedOverlaps.size(); ++i) {
    if (args.outputBedOverlaps[i] != expectedOverlap) {
      args.outputBedOverlaps[i] = expectedOverlap;
      overlapCorrected = true;
    }
  }
  if (overlapCorrected) {
    result.warnings.emplace_back(
        "bed_overlap re-derived from the scanner bed step: corrected to " +
        std::to_string(expectedOverlap) + " (= baseNz " +
        std::to_string(result.baseNz) + " - step " +
        std::to_string(expectedStep) + " layers / " +
        std::to_string(stepMm) + " mm).");
  }

  const auto canonicalGeom = computeCanonicalOutputGeometry(
      result.baseNz, args.outputBedOverlaps, voxelSizeZ, args.bedStepMm);
  result.canonicalGeom = canonicalGeom;
  args.bedOverlaps = std::vector<int>(static_cast<size_t>(joints), 0);
  result.totalOverlap = 0;
  for (int overlap : args.bedOverlaps)
    result.totalOverlap += overlap;

  args.Nz = result.baseNz * result.bedCount - result.totalOverlap;
  std::ostringstream summary;
  summary << "Multi-bed setup: beds=" << result.bedCount
          << ", baseNz=" << result.baseNz
          << ", overlapMode=output-center66_cosine10"
          << ", reconOverlap=" << result.totalOverlap;
  int outputOverlap = 0;
  for (int overlap : args.outputBedOverlaps)
    outputOverlap += overlap;
  summary << ", outputOverlap=" << outputOverlap;
  summary << ", reconNz=" << args.Nz;
  result.summary = summary.str();
  return result;
}

} // namespace main
} // namespace spect
