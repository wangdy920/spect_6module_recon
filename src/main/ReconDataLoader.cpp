#include "main/ReconDataLoader.h"

#include "io/MatFileReader.h"
#include "io/ProjectionLoader.h"
#include "io/VolumeLoader.h"
#include "main/MatVarPath.h"
#include "main/RunLogger.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace spect::main {

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

LoadedProjectionSet loadProjectionSet(const std::vector<std::string> &paths,
                                      int requestedAngles,
                                      const std::string &label,
                                      const std::string &defaultVar) {
  LoadedProjectionSet loaded;
  for (size_t bed = 0; bed < paths.size(); ++bed) {
    std::string path = paths[bed];
    const std::string varName = resolveMatVarName(path, defaultVar);
    spect::io::ProjectionReorderStats stats;
    spect::io::MatArray mat;
    const auto proj = spect::io::loadProjection(path, varName, requestedAngles,
                                                &stats, &mat);
    const std::string suffix = (paths.size() > 1)
                                   ? (" (bed " + std::to_string(bed + 1) + "/" +
                                      std::to_string(paths.size()) + ")")
                                   : std::string();

    if (bed == 0) {
      loaded.raysPerBed = proj.size();
      loaded.detNz = proj.detNz;
      loaded.detNx = proj.detNx;
      loaded.numAngles = proj.numAngles;
      loaded.detectorShapeKnown = proj.detectorShapeKnown;
    } else {
      if (proj.size() != loaded.raysPerBed)
        throw std::runtime_error(label + " size mismatch across beds.");
      if (proj.numAngles != loaded.numAngles ||
          proj.detectorShapeKnown != loaded.detectorShapeKnown)
        throw std::runtime_error(label + " geometry mismatch across beds.");
      if (loaded.detectorShapeKnown &&
          (proj.detNz != loaded.detNz || proj.detNx != loaded.detNx))
        throw std::runtime_error(label + " geometry mismatch across beds.");
    }

    loaded.data.insert(loaded.data.end(), proj.data.begin(), proj.data.end());

    if (proj.binaryFormat) {
      std::ostringstream oss;
      oss << label << suffix << ": BIN projection, selectedAngles="
          << proj.numAngles << ", rays/angle=" << proj.raysPerAngle
          << ", records=" << proj.size();
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
