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

int CompactSymmetryMetadata::xySymmetryCount() const {
  switch (mode) {
  case CompactSymmetryMode::Sym45:
  case CompactSymmetryMode::Sym45Z:
    return 8;
  case CompactSymmetryMode::Sym90:
  case CompactSymmetryMode::Sym90Z:
    return 4;
  case CompactSymmetryMode::Sym180:
  case CompactSymmetryMode::SymZ180:
    return 2;
  default:
    return 1;
  }
}

int CompactSymmetryMetadata::zBranchCount() const {
  const bool zMode = mode == CompactSymmetryMode::ZOnly ||
                     mode == CompactSymmetryMode::Sym45Z ||
                     mode == CompactSymmetryMode::Sym90Z ||
                     mode == CompactSymmetryMode::SymZ180;
  return (zMode || useZSym) && compactDetZ < detNz ? 2 : 1;
}

const char *compactSymmetryModeName(CompactSymmetryMode mode) {
  switch (mode) {
  case CompactSymmetryMode::None:
    return "none";
  case CompactSymmetryMode::ZOnly:
    return "zonly";
  case CompactSymmetryMode::Sym45:
    return "45";
  case CompactSymmetryMode::Sym45Z:
    return "45z";
  case CompactSymmetryMode::Sym90:
    return "90";
  case CompactSymmetryMode::Sym90Z:
    return "90z";
  case CompactSymmetryMode::Sym180:
    return "180";
  case CompactSymmetryMode::SymZ180:
    return "z180";
  }
  return "unknown";
}

CompactSymmetryMode inferCompactSymmetryMode(
    const CompactSymmetryMetadata &meta) {
  const bool looks45 = meta.fullNumAngles > 0 &&
                       (meta.fullNumAngles % 8) == 0 &&
                       meta.compactNumAngles == meta.fullNumAngles / 8 + 1;
  const bool looks90 = meta.fullNumAngles > 0 &&
                       (meta.fullNumAngles % 4) == 0 &&
                       meta.compactNumAngles == meta.fullNumAngles / 4;
  const bool looks180 = meta.fullNumAngles > 0 &&
                        (meta.fullNumAngles % 2) == 0 &&
                        meta.compactNumAngles == meta.fullNumAngles / 2;

  if (meta.use45Sym || meta.symCode == 45 || meta.symmetryOrder == 8 ||
      looks45) {
    return meta.useZSym ? CompactSymmetryMode::Sym45Z
                        : CompactSymmetryMode::Sym45;
  }
  if (meta.use90Sym || looks90) {
    return meta.useZSym ? CompactSymmetryMode::Sym90Z
                        : CompactSymmetryMode::Sym90;
  }
  if (looks180) {
    return meta.useZSym ? CompactSymmetryMode::SymZ180
                        : CompactSymmetryMode::Sym180;
  }
  if (meta.useZSym) {
    return CompactSymmetryMode::ZOnly;
  }
  return CompactSymmetryMode::None;
}

CompactSymmetryMetadata loadCompactSymmetryMetadata(const std::string &path) {
  const std::size_t fileSize = metadataFileSize(path);
  if (fileSize < kCompactMetadataLegacyBytes) {
    throw std::runtime_error(
        "Compact metadata file too small (" + std::to_string(fileSize) +
        " bytes, expected at least " + std::to_string(kCompactMetadataLegacyBytes) +
        "): " + path);
  }
  if (fileSize > kCompactMetadataLegacyBytes &&
      fileSize < kCompactMetadataExtendedV1Bytes) {
    throw std::runtime_error(
        "Compact metadata has truncated extended header (" +
        std::to_string(fileSize) + " bytes, expected " +
        std::to_string(kCompactMetadataLegacyBytes) + " or >= " +
        std::to_string(kCompactMetadataExtendedV1Bytes) + "): " + path);
  }
  if (fileSize > kCompactMetadataExtendedV2Bytes) {
    throw std::runtime_error(
        "Compact metadata has unexpected trailing bytes (" +
        std::to_string(fileSize) + " bytes, max supported " +
        std::to_string(kCompactMetadataExtendedV2Bytes) + "): " + path);
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open compact metadata: " + path);
  }

  CompactSymmetryMetadata meta;
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
      !readScalar(in, meta.detNxDuplicate) || !readScalar(in, meta.detNz) ||
      !readScalar(in, meta.pixelSize) || !readScalar(in, meta.detectorY) ||
      !readScalar(in, meta.detectorThickness) || !readScalar(in, use90) ||
      !readScalar(in, useZ)) {
    throw std::runtime_error("Compact metadata is truncated: " + path);
  }

  meta.use90Sym = use90 != 0;
  meta.useZSym = useZ != 0;

  if (fileSize >= kCompactMetadataExtendedV1Bytes) {
    if (!readScalar(in, use45) || !readScalar(in, meta.symCode) ||
        !readScalar(in, meta.symmetryOrder)) {
      throw std::runtime_error(
          "Compact metadata extended header is truncated: " + path);
    }
    meta.use45Sym = use45 != 0;
  }

  if (fileSize >= kCompactMetadataExtendedV2Bytes) {
    if (!readScalar(in, meta.roiRadius) || !readScalar(in, meta.roiSide)) {
      throw std::runtime_error(
          "Compact metadata ROI fields are truncated: " + path);
    }
  }

  if (meta.detNz <= 0) {
    meta.detNz = meta.useZSym ? 2 * meta.compactDetZ : meta.compactDetZ;
  }
  if (meta.compactRaysPerAngle <= 0) {
    meta.compactRaysPerAngle = meta.detNx * meta.compactDetZ;
  }
  if (meta.totalVoxels <= 0) {
    meta.totalVoxels = meta.Nx * meta.Ny * meta.Nz;
  }
  if (meta.symmetryOrder <= 0) {
    meta.symmetryOrder = 1;
  }
  meta.mode = inferCompactSymmetryMode(meta);

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

  return meta;
}

int fullAngleFromCompactAngle(const CompactSymmetryMetadata &meta,
                              int compactAngle, int symId, bool &valid) {
  valid = true;
  if (compactAngle < 0 || compactAngle >= meta.compactNumAngles) {
    valid = false;
    return 0;
  }

  switch (meta.mode) {
  case CompactSymmetryMode::None:
  case CompactSymmetryMode::ZOnly:
    return compactAngle;
  case CompactSymmetryMode::Sym180:
  case CompactSymmetryMode::SymZ180:
    return compactAngle + symId * meta.compactNumAngles;
  case CompactSymmetryMode::Sym90:
  case CompactSymmetryMode::Sym90Z:
    return compactAngle + symId * meta.compactNumAngles;
  case CompactSymmetryMode::Sym45:
  case CompactSymmetryMode::Sym45Z: {
    const int q = meta.fullNumAngles / 8;
    if ((symId & 1) == 0) {
      if (compactAngle >= q) {
        valid = false;
        return 0;
      }
      return symId * q + compactAngle;
    }
    if (compactAngle <= 0) {
      valid = false;
      return 0;
    }
    return symId * q + (q - compactAngle);
  }
  }
  valid = false;
  return 0;
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

  if (meta.mode != CompactSymmetryMode::Sym45Z &&
      meta.mode != CompactSymmetryMode::Sym90Z) {
    fail(std::string("only SYM_45Z and SYM_90Z are supported, metadata mode is ") +
         compactSymmetryModeName(meta.mode));
  }
  if (meta.Nx != meta.Ny) {
    fail("compact XY symmetry requires Nx == Ny.");
  }
  if (meta.mode == CompactSymmetryMode::Sym45Z &&
      (meta.fullNumAngles % 8 != 0 ||
       meta.compactNumAngles != meta.fullNumAngles / 8 + 1)) {
    fail("SYM_45Z requires compactNumAngles == fullNumAngles / 8 + 1.");
  }
  if (meta.mode == CompactSymmetryMode::Sym90Z &&
      (meta.fullNumAngles % 4 != 0 ||
       meta.compactNumAngles != meta.fullNumAngles / 4)) {
    fail("SYM_90Z requires compactNumAngles == fullNumAngles / 4.");
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
