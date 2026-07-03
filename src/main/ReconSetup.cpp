#include "main/ReconSetup.h"

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

  if (!args.priorPath.empty() && args.priorPath != "none")
    return args.priorPath;

  const bool needsAnatomicalHint =
      !std::isnan(args.maskFromAttThreshold) ||
      (args.useReg &&
       (args.reg.type == RegType::Bowsher || args.reg.type == RegType::PLS));
  if (needsAnatomicalHint && !args.attMapPath.empty())
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

std::vector<int> inferBedNumbers(const std::vector<std::string> &paths) {
  std::vector<int> bedNumbers;
  bedNumbers.reserve(paths.size());
  std::vector<bool> seen(static_cast<size_t>(spect::core::kMaxBeds) + 1u, false);
  for (size_t idx = 0; idx < paths.size(); ++idx) {
    int bed = spect::core::extractBedNumberFromPath(paths[idx]);
    if (bed < 0)
      bed = static_cast<int>(idx) + 1;
    if (bed < 1 || bed > spect::core::kMaxBeds)
      throw std::runtime_error("Bed number must be in fixed range Bed1..Bed" +
                               std::to_string(spect::core::kMaxBeds) + ".");
    if (seen[static_cast<size_t>(bed)])
      throw std::runtime_error("Duplicate bed number in -dfList.");
    seen[static_cast<size_t>(bed)] = true;
    bedNumbers.push_back(bed);
  }
  return bedNumbers;
}

std::vector<int> normalizeFixedBedOverlapTable(
    const std::vector<int> &input, std::vector<std::string> &warnings) {
  std::vector<int> fixedTable(static_cast<size_t>(spect::core::kMaxBeds), 0);
  if (input.empty())
    return fixedTable;
  if (input.size() == 1) {
    std::fill(fixedTable.begin(), fixedTable.end(), input.front());
    return fixedTable;
  }
  if (input.size() > fixedTable.size())
    throw std::runtime_error("-bedOverlap supports at most " +
                             std::to_string(spect::core::kMaxBeds) +
                             " fixed bed slots.");
  std::copy(input.begin(), input.end(), fixedTable.begin());
  if (input.size() < fixedTable.size()) {
    warnings.emplace_back(
        "bedOverlap fixed Bed1..Bed5 table shorter than 5 slots. Missing fixed slots use zero.");
  }
  return fixedTable;
}

std::vector<int> selectActualBedOverlaps(
    const std::vector<int> &fixedTable, const std::vector<int> &bedNumbers,
    std::vector<std::string> &warnings) {
  std::vector<int> overlaps;
  if (bedNumbers.size() < 2)
    return overlaps;
  overlaps.reserve(bedNumbers.size() - 1);
  for (size_t idx = 0; idx + 1 < bedNumbers.size(); ++idx) {
    const int leftBed = bedNumbers[idx];
    const int rightBed = bedNumbers[idx + 1];
    if (rightBed == leftBed + 1) {
      overlaps.push_back(fixedTable[static_cast<size_t>(leftBed - 1)]);
    } else {
      overlaps.push_back(0);
      warnings.emplace_back(
          "Non-adjacent beds in -dfList have no fixed bedOverlap slot. Using zero for that joint.");
    }
  }
  return overlaps;
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

  if (args.Nx == info.Nx && args.Ny == info.Ny && args.Nz == info.Nz) {
    if (warningMessage)
      warningMessage->clear();
    return false;
  }

  args.Nx = info.Nx;
  args.Ny = info.Ny;
  args.Nz = info.Nz;

  if (warningMessage) {
    std::ostringstream warning;
    warning << "Reconstruction dimensions were not explicitly set; adopting "
            << info.Nx << 'x' << info.Ny << 'x' << info.Nz
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
    const bool matrixInput =
        !args.loadCSRPath.empty() || !args.chunkCSRDir.empty() ||
        !args.sysIndexPath.empty() || !args.sysWeightPath.empty();
    if (matrixInput && !args.dimExplicit) {
      throw std::runtime_error(
          "Reconstruction dimensions are required for matrix inputs. Set -dim "
          "or use compact metadata instead of the legacy 200x200x200 default.");
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
  result.projectionPaths = args.projPaths.empty() ? std::vector<std::string>{args.projPath} : args.projPaths;
  result.bedCount = std::max(1, static_cast<int>(result.projectionPaths.size()));
  if (result.bedCount > spect::core::kMaxBeds)
    throw std::runtime_error("-dfList supports at most " +
                             std::to_string(spect::core::kMaxBeds) + " beds.");
  args.bedCount = result.bedCount;

  const int joints = std::max(0, result.bedCount - 1);
  if (joints == 0) {
    result.summary = "Single-bed reconstruction";
    return result;
  }

  const std::vector<int> bedNumbers = inferBedNumbers(result.projectionPaths);
  const std::vector<int> fixedOverlapTable =
      normalizeFixedBedOverlapTable(args.bedOverlaps, result.warnings);
  args.bedOverlaps =
      selectActualBedOverlaps(fixedOverlapTable, bedNumbers, result.warnings);
  if (static_cast<int>(args.bedOverlaps.size()) != joints)
    throw std::runtime_error("Internal bed overlap normalization failed.");

  result.totalOverlap = 0;
  bool clampedAnyOverlap = false;
  for (int &overlap : args.bedOverlaps) {
    const int clamped = std::max(0, std::min(overlap, std::max(0, result.baseNz - 1)));
    if (clamped != overlap) {
      overlap = clamped;
      clampedAnyOverlap = true;
    }
    result.totalOverlap += overlap;
  }
  if (clampedAnyOverlap) {
    result.warnings.emplace_back(
        "bedOverlap values were clamped to the valid range [0, baseNz)."
    );
  }

  args.Nz = result.baseNz * result.bedCount - result.totalOverlap;
  std::ostringstream summary;
  summary << "Multi-bed setup: beds=" << result.bedCount << ", baseNz=" << result.baseNz
          << ", totalOverlap=" << result.totalOverlap << ", Nz=" << args.Nz;
  result.summary = summary.str();
  return result;
}

} // namespace main
} // namespace spect
