#include "io/CompactSymmetryMetadata.h"

#include <cmath>
#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
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

const char *compactSymmetryModeName(const CompactSymmetryMetadata &meta) {
  switch (meta.profile) {
  case CompactSymmetryProfile::Rot90: return "90";
  case CompactSymmetryProfile::Rot90Z: return "90z";
  }
  throw std::runtime_error("Unsupported compact symmetry profile.");
}

CompactSymmetryMetadata loadCompactSymmetryMetadata(const std::string &path) {
  const std::size_t fileSize = metadataFileSize(path);
  const bool legacyDoubleVoxel = fileSize == kLegacyDoubleVoxelMetadataBytes;
  const bool smmdV3 = fileSize == kSmmdV3MetadataBytes;
  if (fileSize != kCompactMetadataBytes && !legacyDoubleVoxel && !smmdV3) {
    throw std::runtime_error(
        "Compact metadata size mismatch (" + std::to_string(fileSize) +
        " bytes, supported metadata layouts are canonical float32 " +
        std::to_string(kCompactMetadataBytes) +
        " or legacy float64-voxelSize " +
        std::to_string(kLegacyDoubleVoxelMetadataBytes) +
        " or SMMD v3 124): " + path);
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open compact metadata: " + path);
  }

  CompactSymmetryMetadata meta;
  meta.legacyDoubleVoxelSizeLayout = legacyDoubleVoxel;
  if (smmdV3) {
    uint32_t magic = 0, version = 0;
    if (!readScalar(in, magic) || !readScalar(in, version) ||
        magic != 0x444D4D53u || version != 3u) {
      throw std::runtime_error("Invalid SMMD v3 magic/version: " + path);
    }
    meta.metadataVersion = version;
  }
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
      !readScalar(in, meta.Ny) || !readScalar(in, meta.Nz)) {
    throw std::runtime_error("Compact metadata is truncated: " + path);
  }
  if (smmdV3) {
    char decimal[32];
    if (!in.read(decimal, sizeof(decimal)))
      throw std::runtime_error("Truncated SMMD voxel size: " + path);
    const char *end = std::find(decimal, decimal + sizeof(decimal), '\0');
    double voxel = 0.0;
    const auto parsed = std::from_chars(decimal, end, voxel);
    if (end == decimal + sizeof(decimal) || parsed.ec != std::errc{} ||
        parsed.ptr != end || !std::isfinite(voxel) || voxel <= 0.0 ||
        voxel > std::numeric_limits<float>::max())
      throw std::runtime_error("Invalid SMMD decimal voxel size: " + path);
    meta.voxelSize = static_cast<float>(voxel);
  } else if (legacyDoubleVoxel) {
    double voxelSize = 0.0;
    if (!readScalar(in, voxelSize) || !std::isfinite(voxelSize) ||
        voxelSize <= 0.0 ||
        voxelSize > static_cast<double>(std::numeric_limits<float>::max())) {
      throw std::runtime_error(
          "Legacy compact metadata has invalid float64 voxelSize: " + path);
    }
    meta.voxelSize = static_cast<float>(voxelSize);
  } else if (!readScalar(in, meta.voxelSize)) {
    throw std::runtime_error("Compact metadata is truncated: " + path);
  }
  if (!readScalar(in, detNxDuplicate) || !readScalar(in, meta.detNz) ||
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
  if (!std::isfinite(meta.voxelSize) || meta.voxelSize <= 0.0f) {
    throw std::runtime_error(
        "Compact symmetry metadata has invalid voxelSize=" +
        std::to_string(meta.voxelSize));
  }

  auto reject = [&](const std::string &reason) {
    throw std::runtime_error("Unsupported compact metadata (" +
                             reason + "): " + path);
  };
  if (use90 != 1 || (useZ != 0 && useZ != 1) || use45 != 0 || meta.symCode != 90 ||
      meta.symmetryOrder != 4) {
    reject("requires use90=1, useZ=0/1, use45=0, symCode=90, symmetryOrder=4");
  }
  meta.profile = useZ ? CompactSymmetryProfile::Rot90Z
                     : CompactSymmetryProfile::Rot90;
  const int limit = std::numeric_limits<int>::max();
  if (meta.Nx > limit / meta.Ny || meta.Nx * meta.Ny > limit / meta.Nz ||
      meta.detNx > limit / meta.detNz ||
      meta.detNx * meta.detNz > limit / meta.fullNumAngles)
    reject("image/projection dimensions exceed supported int32 indexing");
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
  if (meta.detNz != static_cast<int64_t>(meta.zBranchCount()) * meta.compactDetZ) {
    reject("detNz must equal zBranchCount*compactDetZ");
  }
  if (meta.compactRaysPerAngle != static_cast<int64_t>(meta.detNx) * meta.compactDetZ) {
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

void validateCompactMetadataCompatibility(
    const CompactSymmetryMetadata &reference,
    const CompactSymmetryMetadata &candidate) {
  const auto require = [](bool compatible, const char *field) {
    if (!compatible)
      throw std::runtime_error(std::string("Joint matrix metadata mismatch: ") + field);
  };
  require(reference.Nx == candidate.Nx && reference.Ny == candidate.Ny &&
              reference.Nz == candidate.Nz &&
              reference.totalVoxels == candidate.totalVoxels, "image dimensions");
  require(reference.detNx == candidate.detNx && reference.detNz == candidate.detNz &&
              reference.compactDetZ == candidate.compactDetZ, "detector dimensions");
  require(reference.fullNumAngles == candidate.fullNumAngles &&
              reference.compactNumAngles == candidate.compactNumAngles &&
              reference.compactRaysPerAngle == candidate.compactRaysPerAngle &&
              reference.totalRows == candidate.totalRows, "projection indexing");
  require(reference.profile == candidate.profile &&
              reference.symCode == candidate.symCode &&
              reference.symmetryOrder == candidate.symmetryOrder, "symmetry profile");
  // Exact equality after the reader's float32 normalization: the joint path
  // shares one grid and never resamples or remaps the second channel.
  const auto sameFloat = [](float a, float b) {
    return std::isfinite(a) && std::isfinite(b) && a == b;
  };
  require(sameFloat(reference.voxelSize, candidate.voxelSize), "voxelSize");
  require(sameFloat(reference.pixelSize, candidate.pixelSize), "pixelSize");
  require(sameFloat(reference.detectorY, candidate.detectorY), "detectorY");
  require(sameFloat(reference.detectorThickness, candidate.detectorThickness),
          "detectorThickness");
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
      meta.compactRaysPerAngle <= 0 || meta.detNx <= 0 ||
      (zMirror && meta.zBranchCount() != 2)) {
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

  (void)compactSymmetryModeName(meta);

  if (meta.Nx != meta.Ny) {
    fail("compact XY symmetry requires Nx == Ny.");
  }
  if (meta.fullNumAngles % 4 != 0 ||
      meta.compactNumAngles != meta.fullNumAngles / 4) {
    fail("ROT90 profiles require compactNumAngles == fullNumAngles / 4.");
  }
  if (meta.detNz != static_cast<int64_t>(meta.zBranchCount()) * meta.compactDetZ) {
    fail("profile requires detNz == zBranchCount * compactDetZ.");
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
