#include "io/CompactSymmetryMetadata.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace spect::io {
namespace {

template <typename T>
bool readScalar(std::ifstream &in, T &value) {
  return static_cast<bool>(in.read(reinterpret_cast<char *>(&value), sizeof(T)));
}

void requirePositive(int value, const char *name) {
  if (value <= 0) {
    std::ostringstream oss;
    oss << "Compact symmetry metadata has invalid " << name << "=" << value;
    throw std::runtime_error(oss.str());
  }
}

std::size_t metadataFileSize(const std::string &path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    throw std::runtime_error("Cannot stat compact metadata: " + path);
  }
  return static_cast<std::size_t>(size);
}

} // namespace

const char *compactSymmetryModeName() { return "90z"; }

CompactSymmetryMetadata loadCompactSymmetryMetadata(const std::string &path) {
  const std::size_t fileSize = metadataFileSize(path);
  if (fileSize != kCompactMetadataBytes) {
    throw std::runtime_error(
        "Compact metadata size mismatch (" + std::to_string(fileSize) +
        " bytes, canonical v4 SYM_90Z metadata requires exactly " +
        std::to_string(kCompactMetadataBytes) + "): " + path);
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open compact metadata: " + path);
  }

  CompactSymmetryMetadata meta;
  int detNxDuplicate = 0;
  int use90 = 0;
  int useZ = 0;
  int use45 = 0;
  if (!readScalar(in, meta.totalRows) || !readScalar(in, meta.totalVoxels) ||
      !readScalar(in, meta.maxNnz) ||
      !readScalar(in, meta.compactNumAngles) ||
      !readScalar(in, meta.detNx) || !readScalar(in, meta.compactDetZ) ||
      !readScalar(in, meta.compactRaysPerAngle) ||
      !readScalar(in, meta.fullNumAngles) || !readScalar(in, meta.Nx) ||
      !readScalar(in, meta.Ny) || !readScalar(in, meta.Nz) ||
      !readScalar(in, meta.voxelSize) ||
      !readScalar(in, detNxDuplicate) || !readScalar(in, meta.detNz) ||
      !readScalar(in, meta.pixelSize) || !readScalar(in, meta.detectorY) ||
      !readScalar(in, meta.detectorThickness) || !readScalar(in, use90) ||
      !readScalar(in, useZ) || !readScalar(in, use45) ||
      !readScalar(in, meta.symCode) ||
      !readScalar(in, meta.symmetryOrder)) {
    throw std::runtime_error("Compact metadata is truncated: " + path);
  }

  requirePositive(meta.totalRows, "totalRows");
  requirePositive(meta.totalVoxels, "totalVoxels");
  requirePositive(meta.compactNumAngles, "compactNumAngles");
  requirePositive(meta.detNx, "detNx");
  requirePositive(meta.detNz, "detNz");
  requirePositive(meta.compactDetZ, "compactDetZ");
  requirePositive(meta.compactRaysPerAngle, "compactRaysPerAngle");
  requirePositive(meta.fullNumAngles, "fullNumAngles");
  requirePositive(meta.Nx, "Nx");
  requirePositive(meta.Ny, "Ny");
  requirePositive(meta.Nz, "Nz");

  auto reject = [&](const std::string &reason) {
    throw std::runtime_error("Compact metadata is not canonical v4 SYM_90Z (" +
                             reason + "): " + path);
  };
  if (use90 != 1 || useZ != 1 || use45 != 0 || meta.symCode != 90 ||
      meta.symmetryOrder != 4) {
    reject("requires use90=1, useZ=1, use45=0, symCode=90, symmetryOrder=4");
  }
  if (detNxDuplicate != meta.detNx) {
    reject("detNx duplicate does not match detNx");
  }
  if (meta.Nx != meta.Ny) {
    reject("Nx must equal Ny");
  }
  if (meta.fullNumAngles % 4 != 0 ||
      meta.compactNumAngles != meta.fullNumAngles / 4) {
    reject("compactNumAngles must equal fullNumAngles/4");
  }
  if (meta.detNz != 2 * meta.compactDetZ) {
    reject("detNz must equal 2*compactDetZ");
  }
  if (meta.compactRaysPerAngle != meta.detNx * meta.compactDetZ) {
    reject("compactRaysPerAngle must equal detNx*compactDetZ");
  }
  if (meta.totalRows != meta.compactRows()) {
    reject("totalRows does not match compact geometry");
  }
  if (meta.totalVoxels != meta.voxelCount()) {
    reject("totalVoxels does not match image geometry");
  }
  return meta;
}

int fullAngleFromCompactAngle(const CompactSymmetryMetadata &meta,
                              int compactAngle, int symId, bool &valid) {
  valid = compactAngle >= 0 && compactAngle < meta.compactNumAngles &&
          symId >= 0 && symId < meta.xySymmetryCount();
  if (!valid) {
    return 0;
  }
  return compactAngle + symId * meta.compactNumAngles;
}

bool fullProjectionIndexFromCompactRow(const CompactSymmetryMetadata &meta,
                                       int compactRow, int symId,
                                       bool zMirror,
                                       uint64_t &fullProjectionIndex) {
  if (compactRow < 0 || compactRow >= meta.totalRows ||
      meta.compactRaysPerAngle <= 0 || meta.detNx <= 0) {
    return false;
  }

  const int compactAngle = compactRow / meta.compactRaysPerAngle;
  const int rr = compactRow - compactAngle * meta.compactRaysPerAngle;
  const int ix = rr % meta.detNx;
  const int izCompact = rr / meta.detNx;
  bool validAngle = true;
  const int fullAngle =
      fullAngleFromCompactAngle(meta, compactAngle, symId, validAngle);
  if (!validAngle || fullAngle < 0 || fullAngle >= meta.fullNumAngles) {
    return false;
  }

  int izFull = izCompact;
  if (zMirror) {
    izFull = meta.detNz - 1 - izCompact;
    if (izFull < meta.compactDetZ) {
      return false;
    }
  }
  if (izFull < 0 || izFull >= meta.detNz) {
    return false;
  }

  fullProjectionIndex =
      static_cast<uint64_t>(fullAngle) *
          static_cast<uint64_t>(meta.fullRaysPerAngle()) +
      static_cast<uint64_t>(izFull) * static_cast<uint64_t>(meta.detNx) +
      static_cast<uint64_t>(ix);
  return true;
}

void validateCompactSymmetricGeometry(const CompactSymmetryMetadata &meta,
                                      size_t projectionRowsPerBed, int detNz,
                                      int detNx, int numAngles, int Nx, int Ny,
                                      int Nz) {
  auto fail = [](const std::string &message) {
    throw std::runtime_error("Invalid compact-symmetric inputs: " + message);
  };

  if (meta.Nx != meta.Ny) {
    fail("compact XY symmetry requires Nx == Ny.");
  }
  if (meta.fullNumAngles % 4 != 0 ||
      meta.compactNumAngles != meta.fullNumAngles / 4) {
    fail("SYM_90Z requires compactNumAngles == fullNumAngles / 4.");
  }
  if (meta.detNz != 2 * meta.compactDetZ) {
    fail("SYM_90Z requires detNz == 2 * compactDetZ.");
  }
  if (meta.compactRaysPerAngle != meta.detNx * meta.compactDetZ) {
    fail("compactRaysPerAngle must equal detNx * compactDetZ.");
  }
  if (meta.compactDetZ > meta.detNz) {
    fail("compactDetZ must be <= detNz.");
  }
  if (projectionRowsPerBed != static_cast<size_t>(meta.fullRows())) {
    fail("projection rows per bed do not match metadata full rows.");
  }
  if (detNz != meta.detNz || detNx != meta.detNx) {
    fail("detector shape does not match metadata.");
  }
  if (numAngles != meta.fullNumAngles) {
    fail("angle count does not match metadata.");
  }
  if (Nx != meta.Nx || Ny != meta.Ny || Nz != meta.Nz) {
    fail("image dimensions do not match metadata.");
  }
}

void validateCompactSymmetricInputs(const CompactSymmetryMetadata &meta,
                                    const CSRView &matrix,
                                    size_t projectionRowsPerBed, int detNz,
                                    int detNx, int numAngles, int Nx, int Ny,
                                    int Nz) {
  validateCompactSymmetricGeometry(meta, projectionRowsPerBed, detNz, detNx,
                                   numAngles, Nx, Ny, Nz);
  auto fail = [](const std::string &message) {
    throw std::runtime_error("Invalid compact-symmetric inputs: " + message);
  };
  if (matrix.rows != static_cast<int>(meta.compactRows())) {
    fail("matrix rows do not match metadata compact rows.");
  }
  if (matrix.cols != meta.totalVoxels ||
      matrix.cols != static_cast<int>(meta.voxelCount())) {
    fail("matrix columns do not match metadata voxel count.");
  }
}

void validateCompactChunkSymmetricInputs(const CompactSymmetryMetadata &meta,
                                         uint64_t chunkRows, int matrixCols,
                                         size_t projectionRowsPerBed,
                                         int detNz, int detNx, int numAngles,
                                         int Nx, int Ny, int Nz) {
  validateCompactSymmetricGeometry(meta, projectionRowsPerBed, detNz, detNx,
                                   numAngles, Nx, Ny, Nz);
  auto fail = [](const std::string &message) {
    throw std::runtime_error("Invalid compact-symmetric inputs: " + message);
  };
  if (chunkRows != static_cast<uint64_t>(meta.compactRows())) {
    fail("chunk row sum does not match metadata compact rows.");
  }
  if (matrixCols != meta.totalVoxels ||
      matrixCols != static_cast<int>(meta.voxelCount())) {
    fail("chunk matrix columns do not match metadata voxel count.");
  }
}

} // namespace spect::io
