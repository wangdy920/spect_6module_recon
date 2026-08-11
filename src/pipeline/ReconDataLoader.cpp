#include "pipeline/ReconDataLoader.h"

#include "io/MatFileReader.h"
#include "io/ProjectionLoader.h"
#include "io/VolumeLoader.h"
#include "io/MatVarPath.h"
#include "pipeline/RunLogger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace spect::main {
namespace {

using BedFileMap = std::map<int, std::filesystem::path>;

const std::regex &timestampPattern() {
  static const std::regex pattern(R"(^\d{8}_\d{6}$)");
  return pattern;
}

const std::regex &bedFilePattern() {
  static const std::regex pattern(R"(^Bed([1-9]\d*)\.bin$)",
                                  std::regex::icase);
  return pattern;
}

bool samePath(const std::filesystem::path &a,
              const std::filesystem::path &b) {
  std::error_code ec;
  const bool equivalent = std::filesystem::equivalent(a, b, ec);
  if (!ec)
    return equivalent;
  return std::filesystem::absolute(a).lexically_normal() ==
         std::filesystem::absolute(b).lexically_normal();
}

BedFileMap listBedFiles(const std::filesystem::path &spectDir) {
  BedFileMap beds;
  for (const auto &entry : std::filesystem::directory_iterator(spectDir)) {
    if (!entry.is_regular_file())
      continue;
    std::smatch match;
    const std::string filename = entry.path().filename().string();
    if (!std::regex_match(filename, match, bedFilePattern()))
      continue;
    const int bed = std::stoi(match[1].str());
    if (!beds.emplace(bed, entry.path()).second)
      throw std::runtime_error("Duplicate Bed" + std::to_string(bed) +
                               " projection in " + spectDir.string());
  }
  return beds;
}

std::string bedSetString(const BedFileMap &beds) {
  std::ostringstream out;
  bool first = true;
  for (const auto &[bed, path] : beds) {
    (void)path;
    if (!first)
      out << ',';
    out << "Bed" << bed;
    first = false;
  }
  return out.str();
}

std::vector<std::vector<std::string>> discoverTimeSegmentSourceGroups(
    const std::vector<std::string> &paths) {
  if (paths.empty())
    throw std::runtime_error("Time-segment merge requires projection paths.");

  const std::filesystem::path selectedSpect =
      std::filesystem::path(paths.front()).parent_path();
  if (selectedSpect.filename().string() != "SPECT")
    throw std::runtime_error(
        "Time-segment merge expects BedN files under a SPECT directory: " +
        selectedSpect.string());
  const std::filesystem::path selectedTimestamp = selectedSpect.parent_path();
  if (!std::regex_match(selectedTimestamp.filename().string(),
                        timestampPattern()))
    throw std::runtime_error(
        "Time-segment merge expects SPECT under YYYYMMDD_HHMMSS: " +
        selectedSpect.string());

  BedFileMap selectedBeds;
  for (const std::string &pathText : paths) {
    const std::filesystem::path path(pathText);
    if (!std::filesystem::is_regular_file(path) ||
        !samePath(path.parent_path(), selectedSpect))
      throw std::runtime_error(
          "All merged projection inputs must be BedN files in one selected "
          "SPECT directory: " + pathText);
    std::smatch match;
    const std::string filename = path.filename().string();
    if (!std::regex_match(filename, match, bedFilePattern()))
      throw std::runtime_error(
          "Time-segment merge supports exact BedN.bin names: " +
          pathText);
    const int bed = std::stoi(match[1].str());
    if (!selectedBeds.emplace(bed, path).second)
      throw std::runtime_error("Duplicate selected Bed" +
                               std::to_string(bed) + " projection.");
  }

  struct Segment {
    std::string timestamp;
    BedFileMap beds;
  };
  std::vector<Segment> segments;
  const std::filesystem::path caseRoot = selectedTimestamp.parent_path();
  for (const auto &entry : std::filesystem::directory_iterator(caseRoot)) {
    if (!entry.is_directory() ||
        !std::regex_match(entry.path().filename().string(), timestampPattern()))
      continue;
    const std::filesystem::path spectDir = entry.path() / "SPECT";
    if (!std::filesystem::is_directory(spectDir))
      continue;
    BedFileMap beds = listBedFiles(spectDir);
    if (beds.empty())
      throw std::runtime_error("No BedN projection files in " +
                               spectDir.string());
    if (bedSetString(beds) != bedSetString(selectedBeds))
      throw std::runtime_error(
          "Time-segment bed set mismatch in " + spectDir.string() +
          ": expected [" + bedSetString(selectedBeds) + "], got [" +
          bedSetString(beds) + "].");
    segments.push_back({entry.path().filename().string(), std::move(beds)});
  }
  std::sort(segments.begin(), segments.end(),
            [](const Segment &a, const Segment &b) {
              return a.timestamp < b.timestamp;
            });
  if (segments.empty())
    throw std::runtime_error("No timestamp/SPECT inputs found under case root: " +
                             caseRoot.string());

  std::vector<std::vector<std::string>> groups;
  groups.reserve(selectedBeds.size());
  for (const auto &[bed, path] : selectedBeds) {
    (void)path;
    std::vector<std::string> sources;
    sources.reserve(segments.size());
    for (const Segment &segment : segments)
      sources.push_back(segment.beds.at(bed).generic_string());
    groups.push_back(std::move(sources));
  }
  logInfo("Time-segment merge: discovered " +
          std::to_string(segments.size()) + " SPECT segment(s) under " +
          caseRoot.string() + "; beds=" + bedSetString(selectedBeds));
  return groups;
}

void requireMatchingGeometry(const spect::io::ProjectionData &projection,
                             std::size_t expectedSize, int expectedAngles,
                             int expectedDetNz, int expectedDetNx,
                             bool expectedShapeKnown,
                             const std::string &context) {
  if (projection.size() != expectedSize ||
      projection.numAngles != expectedAngles ||
      projection.detectorShapeKnown != expectedShapeKnown ||
      (expectedShapeKnown && (projection.detNz != expectedDetNz ||
                              projection.detNx != expectedDetNx)))
    throw std::runtime_error(context + " geometry mismatch.");
}

} // namespace

std::string resolveCompactMetadataPath(const ReconArgs &args) {
  return (std::filesystem::path(args.chunkCSRDir) / "metadata.bin").string();
}

void applyCompactMetadataGeometry(
    ReconArgs &args, const spect::io::CompactSymmetryMetadata &meta) {
  args.Nx = meta.Nx;
  args.Ny = meta.Ny;
  args.Nz = meta.Nz;
  args.dimExplicit = true;
  args.Vx = args.Vy = args.Vz = meta.voxelSize;
  args.dx = args.dy = args.dz = static_cast<double>(meta.voxelSize);
  args.voxelSizeMM = {meta.voxelSize, meta.voxelSize, meta.voxelSize};
  args.detNx = meta.detNx;
  args.detNz = meta.detNz;
  args.numAngles = meta.fullNumAngles;
}

void prepareTimeSegmentProjectionInputs(ReconArgs &args) {
  if (!args.mergeTimeSegments || !args.projPaths.empty() ||
      args.projPath.empty())
    return;
  const std::filesystem::path input(args.projPath);
  if (!std::filesystem::is_directory(input))
    return;
  if (input.filename().string() != "SPECT")
    throw std::runtime_error(
        "merge_time_segments directory input must be a timestamp/SPECT "
        "directory: " + input.string());
  const BedFileMap beds = listBedFiles(input);
  if (beds.empty())
    throw std::runtime_error("No BedN.bin files in " + input.string());
  if (beds.size() == 1) {
    args.projPath = beds.begin()->second.generic_string();
    return;
  }
  args.projPaths.clear();
  for (const auto &[bed, path] : beds) {
    (void)bed;
    args.projPaths.push_back(path.generic_string());
  }
  args.projPath.clear();
}

LoadedProjectionSet loadProjectionSet(const std::vector<std::string> &paths,
                                      int requestedAngles,
                                      const std::string &label,
                                      const std::string &defaultVar,
                                      bool mergeTimeSegments) {
  LoadedProjectionSet loaded;
  loaded.sourceGroups = mergeTimeSegments
                            ? discoverTimeSegmentSourceGroups(paths)
                            : std::vector<std::vector<std::string>>();
  if (!mergeTimeSegments) {
    loaded.sourceGroups.reserve(paths.size());
    for (const std::string &path : paths)
      loaded.sourceGroups.push_back({path});
  }

  for (size_t bed = 0; bed < loaded.sourceGroups.size(); ++bed) {
    std::vector<double> bedData;
    std::size_t bedRays = 0;
    int bedAngles = 0;
    int bedDetNz = 0;
    int bedDetNx = 0;
    bool bedShapeKnown = true;
    std::uintmax_t bedFileSize = 0;
    for (size_t segment = 0; segment < loaded.sourceGroups[bed].size();
         ++segment) {
      std::string path = loaded.sourceGroups[bed][segment];
      const std::string varName = resolveMatVarName(path, defaultVar);
      spect::io::ProjectionReorderStats stats;
      spect::io::MatArray mat;
      auto proj = spect::io::loadProjection(path, varName, requestedAngles,
                                            &stats, &mat);
      const std::size_t projectionRecordCount = proj.size();
      const std::uintmax_t sourceFileSize =
          mergeTimeSegments ? std::filesystem::file_size(path) : 0;
      const std::string segmentSuffix =
          mergeTimeSegments
              ? (", segment " + std::to_string(segment + 1) + "/" +
                 std::to_string(loaded.sourceGroups[bed].size()))
              : std::string();
      const std::string suffix =
          (paths.size() > 1)
              ? (" (bed " + std::to_string(bed + 1) + "/" +
                 std::to_string(paths.size()) + segmentSuffix + ")")
              : mergeTimeSegments
                    ? (" (segment " + std::to_string(segment + 1) + "/" +
                       std::to_string(loaded.sourceGroups[bed].size()) + ")")
                    : std::string();

      if (segment == 0) {
        bedRays = proj.size();
        bedAngles = proj.numAngles;
        bedDetNz = proj.detNz;
        bedDetNx = proj.detNx;
        bedShapeKnown = proj.detectorShapeKnown;
        bedFileSize = sourceFileSize;
        bedData = std::move(proj.data);
        if (!std::all_of(bedData.begin(), bedData.end(),
                         [](double value) { return std::isfinite(value); }))
          throw std::runtime_error(label + suffix +
                                   " contains non-finite values.");
      } else {
        if (sourceFileSize != bedFileSize)
          throw std::runtime_error(label + suffix +
                                   " BIN header/file size mismatch.");
        requireMatchingGeometry(proj, bedRays, bedAngles, bedDetNz, bedDetNx,
                                bedShapeKnown, label + suffix);
        for (size_t i = 0; i < bedData.size(); ++i) {
          bedData[i] += proj.data[i];
          if (!std::isfinite(bedData[i]))
            throw std::runtime_error(label + suffix +
                                     " sum produced a non-finite value.");
        }
      }

      if (proj.binaryFormat) {
        std::ostringstream oss;
        oss << label << suffix << ": BIN projection, selectedAngles="
            << proj.numAngles << ", rays/angle=" << proj.raysPerAngle
            << ", records=" << projectionRecordCount;
        logInfo(oss.str());
      } else {
        spect::io::printMatInfo(mat, label + suffix);
      }

      if (stats.isEventList) {
        std::ostringstream oss;
        oss << label << suffix << ": event-list records=" << stats.recordCount
            << ", selectedAngles=" << stats.selectedAngles
            << ", detectors/angle=" << stats.detectorsPerAngle;
        logInfo(oss.str());
      }
    }

    if (bed == 0) {
      loaded.raysPerBed = bedRays;
      loaded.detNz = bedDetNz;
      loaded.detNx = bedDetNx;
      loaded.numAngles = bedAngles;
      loaded.detectorShapeKnown = bedShapeKnown;
    } else {
      if (bedRays != loaded.raysPerBed || bedAngles != loaded.numAngles ||
          bedShapeKnown != loaded.detectorShapeKnown ||
          (bedShapeKnown &&
           (bedDetNz != loaded.detNz || bedDetNx != loaded.detNx)))
        throw std::runtime_error(label + " geometry mismatch across beds.");
    }
    loaded.data.insert(loaded.data.end(), bedData.begin(), bedData.end());
    if (mergeTimeSegments)
      logInfo("Time-segment merge: bed " + std::to_string(bed + 1) +
              " summed " +
              std::to_string(loaded.sourceGroups[bed].size()) +
              " source(s) in memory.");
  }
  return loaded;
}

std::vector<double> loadVolumeChecked(const std::string &path,
                                      const std::string &label,
                                      int expectedNx, int expectedNy,
                                      int expectedNz) {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  std::vector<double> volume = spect::io::loadVolume(path, nx, ny, nz);
  if (nx != expectedNx || ny != expectedNy || nz != expectedNz) {
    std::ostringstream oss;
    oss << label << " dimensions (" << nx << 'x' << ny << 'x' << nz
        << ") do not match reconstruction dimensions (" << expectedNx << 'x'
        << expectedNy << 'x' << expectedNz << ").";
    throw std::runtime_error(oss.str());
  }
  return volume;
}

} // namespace spect::main
