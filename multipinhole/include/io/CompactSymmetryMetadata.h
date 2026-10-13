#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace spect::io {

// Canonical v4 metadata.bin layout: 22 int32/float fields, no magic prefix.
// Legacy matrices may store only voxelSize as float64, shifting the remaining
// fields by four bytes. SMMD v3 adds an 8-byte prefix and a 32-byte decimal voxel.
constexpr std::size_t kCompactMetadataBytes = 88;
constexpr std::size_t kLegacyDoubleVoxelMetadataBytes = 92;
constexpr std::size_t kSmmdV3MetadataBytes = 124;

enum class CompactSymmetryProfile : uint32_t { Rot90 = 4, Rot90Z = 5 };

struct CompactSymmetryMetadata {
  int totalRows = 0;
  int totalVoxels = 0;
  int maxNnz = 0;
  int compactNumAngles = 0;
  int detNx = 0;
  int compactDetZ = 0;
  int compactRaysPerAngle = 0;
  int fullNumAngles = 0;
  int Nx = 0;
  int Ny = 0;
  int Nz = 0;
  float voxelSize = 0.0f;
  int detNz = 0;
  float pixelSize = 0.0f;
  float detectorY = 0.0f;
  float detectorThickness = 0.0f;
  int symCode = 0;
  int symmetryOrder = 0;
  bool legacyDoubleVoxelSizeLayout = false;
  uint32_t metadataVersion = 0; // 0 = unversioned legacy, 3 = SMMD v3
  CompactSymmetryProfile profile = CompactSymmetryProfile::Rot90Z;

  int fullRaysPerAngle() const { return detNx * detNz; }
  int64_t fullRows() const {
    return static_cast<int64_t>(fullNumAngles) *
           static_cast<int64_t>(fullRaysPerAngle());
  }
  int64_t compactRows() const {
    return static_cast<int64_t>(compactNumAngles) *
           static_cast<int64_t>(compactRaysPerAngle);
  }
  int64_t voxelCount() const {
    return static_cast<int64_t>(Nx) * static_cast<int64_t>(Ny) *
           static_cast<int64_t>(Nz);
  }
  int xySymmetryCount() const { return 4; }
  int zBranchCount() const {
    return profile == CompactSymmetryProfile::Rot90Z ? 2 : 1;
  }
};

struct CompactVoxelCoord {
  int x = 0;
  int y = 0;
  int z = 0;
};

inline CompactVoxelCoord mapCompactVoxelCoord(int x, int y, int z, int nx,
                                              int nz, int symId,
                                              bool zMirror) {
  CompactVoxelCoord out;
  switch (symId) {
  case 1:
    out.x = nx - 1 - y;
    out.y = x;
    break;
  case 2:
    out.x = nx - 1 - x;
    out.y = nx - 1 - y;
    break;
  case 3:
    out.x = y;
    out.y = nx - 1 - x;
    break;
  default:
    out.x = x;
    out.y = y;
    break;
  }
  out.z = zMirror ? (nz - 1 - z) : z;
  return out;
}

inline size_t compactVoxelLinearIndex(const CompactVoxelCoord &coord,
                                      int nx, int ny) {
  return static_cast<size_t>(coord.z) * static_cast<size_t>(nx) *
             static_cast<size_t>(ny) +
         static_cast<size_t>(coord.y) * static_cast<size_t>(nx) +
         static_cast<size_t>(coord.x);
}

inline size_t mapCompactVoxelIndex(size_t linearIndex, int nx, int ny, int nz,
                                   int symId, bool zMirror) {
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const int z = static_cast<int>(linearIndex / plane);
  const size_t rem = linearIndex - static_cast<size_t>(z) * plane;
  const int y = static_cast<int>(rem / static_cast<size_t>(nx));
  const int x = static_cast<int>(rem - static_cast<size_t>(y) *
                                           static_cast<size_t>(nx));
  return compactVoxelLinearIndex(
      mapCompactVoxelCoord(x, y, z, nx, nz, symId, zMirror), nx, ny);
}

const char *compactSymmetryModeName(const CompactSymmetryMetadata &meta);

CompactSymmetryMetadata loadCompactSymmetryMetadata(const std::string &path);

// Both inputs must have passed the metadata reader. Compare the normalized
// shared coordinate system, not payload identity or unrecorded energy physics.
void validateCompactMetadataCompatibility(
    const CompactSymmetryMetadata &reference,
    const CompactSymmetryMetadata &candidate);

int fullAngleFromCompactAngle(const CompactSymmetryMetadata &meta,
                              int compactAngle, int symId, bool &valid);

bool fullProjectionIndexFromCompactRow(const CompactSymmetryMetadata &meta,
                                       int compactRow, int symId,
                                       bool zMirror,
                                       uint64_t &fullProjectionIndex);

void validateCompactChunkSymmetricInputs(const CompactSymmetryMetadata &meta,
                                         uint64_t chunkRows, int matrixCols,
                                         size_t projectionRowsPerBed,
                                         int detNz, int detNx, int numAngles,
                                         int Nx, int Ny, int Nz);

} // namespace spect::io
