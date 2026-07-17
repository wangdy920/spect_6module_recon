#include "cuda/CompactSymmetricProjector.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)

#include "core/ReconLimits.h"
#include "cuda/CudaLaunchConfig.h"
#include "interfaces/ISparseOps.h"
#include "sparse/SparseOpsBackend.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spect::cuda {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint64_t kCompactChunkResidentReserveBytes =
    kCompactChunkResidentDeviceReserveBytes;

struct RegisteredCompactChunkKey {
  const void *rowPtr = nullptr;
  const void *colIdx = nullptr;
  const void *values = nullptr;
};

int effectiveThreads(int threads) { return threads > 0 ? threads : 1; }

double elapsedSeconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void addTiming(double &target, Clock::time_point start) {
  target += elapsedSeconds(start);
}

bool compactDeviceRatioEnabled() {
  const char *value = std::getenv("SPECT_COMPACT_DEVICE_RATIO");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
         value[0] == 't' || value[0] == 'T';
}

bool compactDeviceBackwardEnabled() {
  const char *value = std::getenv("SPECT_COMPACT_DEVICE_BACKWARD");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
         value[0] == 't' || value[0] == 'T';
}

CompactDeviceRatioParams makeCompactDeviceRatioParams(
    const spect::io::CompactSymmetryMetadata &meta, int compactRowBase,
    int symId, bool zMirror) {
  CompactDeviceRatioParams params;
  params.symmetryMode =
      meta.mode == spect::io::CompactSymmetryMode::Sym45Z ? 45 : 90;
  params.compactRowBase = compactRowBase;
  params.compactRaysPerAngle = meta.compactRaysPerAngle;
  params.compactNumAngles = meta.compactNumAngles;
  params.detNx = meta.detNx;
  params.detNz = meta.detNz;
  params.compactDetZ = meta.compactDetZ;
  params.fullNumAngles = meta.fullNumAngles;
  params.fullRaysPerAngle = meta.fullRaysPerAngle();
  params.symId = symId;
  params.zMirror = zMirror;
  return params;
}

CompactDeviceTransformParams makeCompactDeviceTransformParams(
    const spect::io::CompactSymmetryMetadata &meta, int symId, bool zMirror) {
  CompactDeviceTransformParams params;
  params.symmetryMode =
      meta.mode == spect::io::CompactSymmetryMode::Sym45Z ? 45 : 90;
  params.nx = meta.Nx;
  params.ny = meta.Ny;
  params.nz = meta.Nz;
  params.symId = symId;
  params.zMirror = zMirror;
  return params;
}

CompactAttenBakeParams makeCompactAttenBakeParams(
    const spect::io::CompactSymmetryMetadata &meta, int compactRowBase,
    int symId, bool zMirror) {
  CompactAttenBakeParams params;
  params.symmetryMode =
      meta.mode == spect::io::CompactSymmetryMode::Sym45Z ? 45 : 90;
  params.compactRowBase = compactRowBase;
  params.compactRaysPerAngle = meta.compactRaysPerAngle;
  params.compactNumAngles = meta.compactNumAngles;
  params.fullNumAngles = meta.fullNumAngles;
  params.symId = symId;
  params.zMirror = zMirror;
  params.nx = meta.Nx;
  params.ny = meta.Ny;
  params.nz = meta.Nz;
  return params;
}

size_t compactAttenTableCount(const spect::io::CompactSymmetryMetadata &meta) {
  return static_cast<size_t>(meta.fullNumAngles) *
         static_cast<size_t>(meta.totalVoxels);
}

bool operator==(const RegisteredCompactChunkKey &lhs,
                const RegisteredCompactChunkKey &rhs) {
  return lhs.rowPtr == rhs.rowPtr && lhs.colIdx == rhs.colIdx &&
         lhs.values == rhs.values;
}

uint64_t estimatedCompactChunkResidentBytes(
    const spect::io::CompactChunkCSRDirectory &chunks) {
  uint64_t totalBytes = 0;
  for (const auto &info : chunks.chunks()) {
    if (std::numeric_limits<uint64_t>::max() - totalBytes < info.fileSizeBytes) {
      return std::numeric_limits<uint64_t>::max();
    }
    totalBytes += info.fileSizeBytes;
  }
  return totalBytes;
}

void warmCompactChunkResidentDeviceCache(
    const spect::io::CompactChunkCSRDirectory &chunks) {
  const uint64_t cacheBudget = chunks.cacheBudgetBytes();
  if (cacheBudget != std::numeric_limits<uint64_t>::max() &&
      cacheBudget < estimatedCompactChunkResidentBytes(chunks)) {
    return;
  }
  static std::mutex warmedMutex;
  static const spect::io::CompactChunkCSRDirectory *lastWarmedChunks = nullptr;
  static uint64_t lastWarmedCacheBudget = 0;
  {
    std::lock_guard<std::mutex> lock(warmedMutex);
    if (lastWarmedChunks == &chunks && lastWarmedCacheBudget == cacheBudget) {
      return;
    }
  }

  std::vector<std::shared_ptr<const spect::io::CompactChunkCSRData>> loaded;
  std::vector<spect::io::CSRView> views;
  static std::mutex registrationMutex;
  static std::vector<RegisteredCompactChunkKey> registeredChunks;
  static std::vector<RegisteredCompactChunkKey> failedChunks;
  loaded.reserve(chunks.chunks().size());
  views.reserve(chunks.chunks().size());
  for (size_t chunkIndex = 0; chunkIndex < chunks.chunks().size(); ++chunkIndex) {
    auto chunk = chunks.loadChunk(chunkIndex);
    const RegisteredCompactChunkKey key{chunk->rowPtr.data(),
                                        chunk->colIdxData(),
                                        chunk->valuesData()};
    bool shouldAttemptRegistration = false;
    {
      std::lock_guard<std::mutex> lock(registrationMutex);
      const auto alreadyRegistered =
          std::find(registeredChunks.begin(), registeredChunks.end(), key) !=
          registeredChunks.end();
      const auto alreadyFailed =
          std::find(failedChunks.begin(), failedChunks.end(), key) !=
          failedChunks.end();
      shouldAttemptRegistration = !alreadyRegistered && !alreadyFailed;
    }
    if (shouldAttemptRegistration) {
      const size_t rowBytes = chunk->rowPtr.size() * sizeof(uint32_t);
      const size_t colBytes = chunk->colIdxCount() * sizeof(uint32_t);
      const size_t valBytes = chunk->valuesCount() * sizeof(float);
      bool rowRegistered = false;
      bool colRegistered = false;
      bool valRegistered = false;
      auto tryRegister = [](void *ptr, size_t bytes) {
        if (ptr == nullptr || bytes == 0) {
          return cudaSuccess;
        }
        return cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
      };
      cudaError_t status =
          tryRegister(const_cast<uint32_t *>(chunk->rowPtr.data()), rowBytes);
      if (status == cudaSuccess) {
        rowRegistered = true;
      } else {
        cudaGetLastError();
      }
      if (status == cudaSuccess) {
        status =
            tryRegister(const_cast<uint32_t *>(chunk->colIdxData()), colBytes);
        if (status == cudaSuccess) {
          colRegistered = true;
        } else {
          cudaGetLastError();
        }
      }
      if (status == cudaSuccess) {
        status = tryRegister(const_cast<float *>(chunk->valuesData()), valBytes);
        if (status == cudaSuccess) {
          valRegistered = true;
        } else {
          cudaGetLastError();
        }
      }
      std::lock_guard<std::mutex> lock(registrationMutex);
      if (status == cudaSuccess) {
        registeredChunks.push_back(key);
      } else {
        std::cerr << "[WARN] cudaHostRegister failed for compact chunk "
                  << chunk->info.path.string() << ": "
                  << cudaGetErrorString(status)
                  << "; continuing with pageable host memory.\n";
        if (valRegistered) {
          cudaHostUnregister(const_cast<float *>(chunk->valuesData()));
        }
        if (colRegistered) {
          cudaHostUnregister(const_cast<uint32_t *>(chunk->colIdxData()));
        }
        if (rowRegistered) {
          cudaHostUnregister(const_cast<uint32_t *>(chunk->rowPtr.data()));
        }
        failedChunks.push_back(key);
      }
    }
    views.push_back(chunk->view());
    loaded.push_back(std::move(chunk));
  }
  if (warmupResidentMatrixSetCUDA(views, kCompactChunkResidentReserveBytes)) {
    std::lock_guard<std::mutex> lock(warmedMutex);
    lastWarmedChunks = &chunks;
    lastWarmedCacheBudget = cacheBudget;
  }
}

struct CompactBranchDescriptor {
  int symId = 0;
  bool zMirror = false;
};

std::vector<CompactBranchDescriptor>
enumerateCompactBranches(const spect::io::CompactSymmetryMetadata &meta) {
  std::vector<CompactBranchDescriptor> branches;
  branches.reserve(static_cast<size_t>(meta.xySymmetryCount() *
                                       meta.zBranchCount()));
  for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
    for (int z = 0; z < meta.zBranchCount(); ++z) {
      branches.push_back(CompactBranchDescriptor{sym, z != 0});
    }
  }
  return branches;
}

size_t compactChunkBranchOffset(size_t branchIndex, size_t branchCount,
                                size_t chunkIndex, size_t chunkCount,
                                size_t nvox) {
  (void)branchCount;
  return (branchIndex * chunkCount + chunkIndex) * nvox;
}

void transformForwardInto(const double *input, double *output,
                          const spect::io::CompactSymmetryMetadata &meta,
                          int symId, bool zMirror, int threads) {
  const int nx = meta.Nx;
  const int ny = meta.Ny;
  const int nz = meta.Nz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t nvox = plane * static_cast<size_t>(nz);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(nvox); ++idx) {
    const size_t uidx = static_cast<size_t>(idx);
    const int iz = static_cast<int>(uidx / plane);
    const int rem = static_cast<int>(uidx - static_cast<size_t>(iz) * plane);
    const int iy = rem / nx;
    const int ix = rem - iy * nx;
    const spect::io::CompactVoxelCoord mapped =
        (meta.mode == spect::io::CompactSymmetryMode::Sym90Z)
            ? spect::io::mapSym90ZVoxelCoord(ix, iy, iz, nx, nz, symId,
                                             zMirror)
            : spect::io::mapSym45ZVoxelCoord(ix, iy, iz, nx, nz, symId,
                                             zMirror);
    output[uidx] =
        input[static_cast<size_t>(mapped.z) * plane +
              static_cast<size_t>(mapped.y) * static_cast<size_t>(nx) +
              static_cast<size_t>(mapped.x)];
  }
}

void transformForwardIntoFloat(const double *input, float *output,
                               const spect::io::CompactSymmetryMetadata &meta,
                               int symId, bool zMirror, int threads) {
  const int nx = meta.Nx;
  const int ny = meta.Ny;
  const int nz = meta.Nz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t nvox = plane * static_cast<size_t>(nz);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(nvox); ++idx) {
    const size_t uidx = static_cast<size_t>(idx);
    const int iz = static_cast<int>(uidx / plane);
    const int rem = static_cast<int>(uidx - static_cast<size_t>(iz) * plane);
    const int iy = rem / nx;
    const int ix = rem - iy * nx;
    const spect::io::CompactVoxelCoord mapped =
        (meta.mode == spect::io::CompactSymmetryMode::Sym90Z)
            ? spect::io::mapSym90ZVoxelCoord(ix, iy, iz, nx, nz, symId,
                                             zMirror)
            : spect::io::mapSym45ZVoxelCoord(ix, iy, iz, nx, nz, symId,
                                             zMirror);
    output[uidx] = static_cast<float>(
        input[static_cast<size_t>(mapped.z) * plane +
              static_cast<size_t>(mapped.y) * static_cast<size_t>(nx) +
              static_cast<size_t>(mapped.x)]);
  }
}

void transformForward(const double *input, std::vector<double> &output,
                      const spect::io::CompactSymmetryMetadata &meta,
                      int symId, bool zMirror, int threads) {
  const int nx = meta.Nx;
  const int ny = meta.Ny;
  const int nz = meta.Nz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t nvox = plane * static_cast<size_t>(nz);
  output.resize(nvox);
  transformForwardInto(input, output.data(), meta, symId, zMirror, threads);
}

void transformBackwardAdd(const double *rotated, std::vector<double> &accum,
                          const spect::io::CompactSymmetryMetadata &meta,
                          int symId, bool zMirror, int threads) {
  const int nx = meta.Nx;
  const int ny = meta.Ny;
  const int nz = meta.Nz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t nvox = plane * static_cast<size_t>(nz);
  if (accum.size() != nvox) {
    accum.assign(nvox, 0.0);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(nvox); ++idx) {
    const size_t uidx = static_cast<size_t>(idx);
    const int iz = static_cast<int>(uidx / plane);
    const int rem = static_cast<int>(uidx - static_cast<size_t>(iz) * plane);
    const int iy = rem / nx;
    const int ix = rem - iy * nx;
    const spect::io::CompactVoxelCoord mapped =
        (meta.mode == spect::io::CompactSymmetryMode::Sym90Z)
            ? spect::io::mapSym90ZVoxelCoord(ix, iy, iz, nx, nz, symId,
                                             zMirror)
            : spect::io::mapSym45ZVoxelCoord(ix, iy, iz, nx, nz, symId,
                                             zMirror);
    accum[static_cast<size_t>(mapped.z) * plane +
          static_cast<size_t>(mapped.y) * static_cast<size_t>(nx) +
          static_cast<size_t>(mapped.x)] += rotated[uidx];
  }
}

void transformBackwardAdd(const float *rotated, std::vector<double> &accum,
                          const spect::io::CompactSymmetryMetadata &meta,
                          int symId, bool zMirror, int threads) {
  const int nx = meta.Nx;
  const int ny = meta.Ny;
  const int nz = meta.Nz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t nvox = plane * static_cast<size_t>(nz);
  if (accum.size() != nvox) {
    accum.assign(nvox, 0.0);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(nvox); ++idx) {
    const size_t uidx = static_cast<size_t>(idx);
    const int iz = static_cast<int>(uidx / plane);
    const int rem = static_cast<int>(uidx - static_cast<size_t>(iz) * plane);
    const int iy = rem / nx;
    const int ix = rem - iy * nx;
    const spect::io::CompactVoxelCoord mapped =
        (meta.mode == spect::io::CompactSymmetryMode::Sym90Z)
            ? spect::io::mapSym90ZVoxelCoord(ix, iy, iz, nx, nz, symId,
                                             zMirror)
            : spect::io::mapSym45ZVoxelCoord(ix, iy, iz, nx, nz, symId,
                                             zMirror);
    accum[static_cast<size_t>(mapped.z) * plane +
          static_cast<size_t>(mapped.y) * static_cast<size_t>(nx) +
          static_cast<size_t>(mapped.x)] +=
        static_cast<double>(rotated[uidx]);
  }
}

void transformBackwardAdd(const std::vector<double> &rotated,
                          std::vector<double> &accum,
                          const spect::io::CompactSymmetryMetadata &meta,
                          int symId, bool zMirror, int threads) {
  transformBackwardAdd(rotated.data(), accum, meta, symId, zMirror, threads);
}

void makeUnitRatio(const spect::io::CompactSymmetryMetadata &meta, int symId,
                   bool zMirror, std::vector<double> &ratio, int threads) {
  ratio.resize(static_cast<size_t>(meta.totalRows));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(meta.totalRows);
       ++row) {
    uint64_t fullIndex = 0;
    ratio[static_cast<size_t>(row)] =
        spect::io::fullProjectionIndexFromCompactRow(
            meta, static_cast<int>(row), symId, zMirror, fullIndex)
            ? 1.0
            : 0.0;
  }
}

void makeUnitRatioChunk(const spect::io::CompactSymmetryMetadata &meta,
                        int compactRowBase, int rows, int symId,
                        bool zMirror, std::vector<double> &ratio,
                        int threads) {
  ratio.resize(static_cast<size_t>(rows));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(rows); ++row) {
    uint64_t fullIndex = 0;
    ratio[static_cast<size_t>(row)] =
        spect::io::fullProjectionIndexFromCompactRow(
            meta, compactRowBase + static_cast<int>(row), symId, zMirror,
            fullIndex)
            ? 1.0
            : 0.0;
  }
}

void makeUnitRatioChunkFloat(const spect::io::CompactSymmetryMetadata &meta,
                             int compactRowBase, int rows, int symId,
                             bool zMirror, std::vector<float> &ratio,
                             int threads) {
  ratio.resize(static_cast<size_t>(rows));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(rows); ++row) {
    uint64_t fullIndex = 0;
    ratio[static_cast<size_t>(row)] =
        spect::io::fullProjectionIndexFromCompactRow(
            meta, compactRowBase + static_cast<int>(row), symId, zMirror,
            fullIndex)
            ? 1.0f
            : 0.0f;
  }
}

void makeProjectionRatio(const spect::io::CompactSymmetryMetadata &meta,
                         const double *fullProjection,
                         const double *fullBackground,
                         const std::vector<double> &y, int symId,
                         bool zMirror, std::vector<double> &ratio,
                         int threads) {
  ratio.resize(static_cast<size_t>(meta.totalRows));
  constexpr double kEps = 1e-10;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(meta.totalRows);
       ++row) {
    uint64_t fullIndex = 0;
    if (!spect::io::fullProjectionIndexFromCompactRow(
            meta, static_cast<int>(row), symId, zMirror, fullIndex)) {
      ratio[static_cast<size_t>(row)] = 0.0;
      continue;
    }
    const double background = fullBackground ? fullBackground[fullIndex] : 0.0;
    const double q = std::max(y[static_cast<size_t>(row)] + background, kEps);
    ratio[static_cast<size_t>(row)] = fullProjection[fullIndex] / q;
  }
}

void makeProjectionRatioChunk(const spect::io::CompactSymmetryMetadata &meta,
                              int compactRowBase,
                              const double *fullProjection,
                              const double *fullBackground,
                              const std::vector<double> &y, int symId,
                              bool zMirror, std::vector<double> &ratio,
                              int threads) {
  ratio.resize(y.size());
  constexpr double kEps = 1e-10;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(y.size()); ++row) {
    uint64_t fullIndex = 0;
    if (!spect::io::fullProjectionIndexFromCompactRow(
            meta, compactRowBase + static_cast<int>(row), symId, zMirror,
            fullIndex)) {
      ratio[static_cast<size_t>(row)] = 0.0;
      continue;
    }
    const double background = fullBackground ? fullBackground[fullIndex] : 0.0;
    const double q = std::max(y[static_cast<size_t>(row)] + background, kEps);
    ratio[static_cast<size_t>(row)] = fullProjection[fullIndex] / q;
  }
}

void makeProjectionRatioChunkFloat(const spect::io::CompactSymmetryMetadata &meta,
                                   int compactRowBase,
                                   const double *fullProjection,
                                   const double *fullBackground,
                                   const std::vector<float> &y, int symId,
                                   bool zMirror, std::vector<float> &ratio,
                                   int threads) {
  ratio.resize(y.size());
  constexpr double kEps = 1e-10;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(y.size()); ++row) {
    uint64_t fullIndex = 0;
    if (!spect::io::fullProjectionIndexFromCompactRow(
            meta, compactRowBase + static_cast<int>(row), symId, zMirror,
            fullIndex)) {
      ratio[static_cast<size_t>(row)] = 0.0f;
      continue;
    }
    const double background = fullBackground ? fullBackground[fullIndex] : 0.0;
    const double q = std::max(
        static_cast<double>(y[static_cast<size_t>(row)]) + background, kEps);
    ratio[static_cast<size_t>(row)] =
        static_cast<float>(fullProjection[fullIndex] / q);
  }
}

template <typename ValueType>
void scatterCompactForwardProjection(
    const spect::io::CompactSymmetryMetadata &meta, int compactRowBase,
    const std::vector<ValueType> &compactProjection, int symId, bool zMirror,
    double *fullExpected, std::vector<uint8_t> &coverage) {
  for (size_t row = 0; row < compactProjection.size(); ++row) {
    uint64_t fullIndex = 0;
    if (!spect::io::fullProjectionIndexFromCompactRow(
            meta, compactRowBase + static_cast<int>(row), symId, zMirror,
            fullIndex)) {
      continue;
    }
    const size_t index = static_cast<size_t>(fullIndex);
    if (index >= coverage.size() || coverage[index] != 0) {
      throw std::runtime_error(
          "compact forward projection produced duplicate/out-of-range full row.");
    }
    const double value = static_cast<double>(compactProjection[row]);
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "compact forward projection produced a non-finite value.");
    }
    fullExpected[index] = value;
    coverage[index] = 1;
  }
}

void requireCompleteForwardProjectionCoverage(
    const std::vector<uint8_t> &coverage) {
  if (std::find(coverage.begin(), coverage.end(), uint8_t{0}) !=
      coverage.end()) {
    throw std::runtime_error(
        "compact forward projection did not cover every full projection row.");
  }
}

void checkInputs(const spect::io::CSRView &compactMatrix,
                 const spect::io::CompactSymmetryMetadata &meta) {
  if (!compactMatrix.valid()) {
    throw std::runtime_error("compact-symmetric CUDA helper received invalid CSRView.");
  }
  if (meta.mode != spect::io::CompactSymmetryMode::Sym45Z &&
      meta.mode != spect::io::CompactSymmetryMode::Sym90Z) {
    throw std::runtime_error(
        "compact-symmetric CUDA helper only supports SYM_45Z and SYM_90Z.");
  }
  if (compactMatrix.rows != meta.totalRows ||
      compactMatrix.cols != meta.totalVoxels) {
    throw std::runtime_error(
        "compact-symmetric CUDA helper matrix/metadata mismatch.");
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error("compact-symmetric CUDA helper requires Nx == Ny.");
  }
}

} // namespace

bool computeCompactSymmetricSensitivityCUDA(
    const spect::io::CSRView &compactMatrix,
    const spect::io::CompactSymmetryMetadata &meta, int threads, double *denom,
    CompactSymmetricStageTiming *timing) {
  checkInputs(compactMatrix, meta);
  const ISparseOps &ops = getSparseOpsCUDA();
  if (ops.backend() != BackendType::CUDA) {
    return false;
  }

  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  std::fill(denom, denom + nvox, 0.0);
  std::vector<double> ratio;
  std::vector<double> bpRot(nvox, 0.0);
  std::vector<double> accum(nvox, 0.0);

  for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
    for (int z = 0; z < meta.zBranchCount(); ++z) {
      const bool zMirror = z != 0;
      auto stageStart = Clock::now();
      makeUnitRatio(meta, sym, zMirror, ratio, threads);
      if (timing)
        addTiming(timing->ratioSeconds, stageStart);

      std::fill(bpRot.begin(), bpRot.end(), 0.0);
      stageStart = Clock::now();
      ops.spmvT(compactMatrix, ratio.data(), bpRot.data(), threads);
      if (timing)
        addTiming(timing->backwardSeconds, stageStart);

      stageStart = Clock::now();
      transformBackwardAdd(bpRot, accum, meta, sym, zMirror, threads);
      if (timing)
        addTiming(timing->transformBackwardSeconds, stageStart);
    }
  }

  std::copy(accum.begin(), accum.end(), denom);
  if (timing) {
    timing->sensitivitySeconds += elapsedSeconds(totalStart);
    ++timing->sensitivityCalls;
  }
  return true;
}

bool runCompactSymmetricMlemNumerCUDA(
    const spect::io::CSRView &compactMatrix,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *fullProjection, const double *fullBackground,
    const double *smoothedImage, int threads, double *numer,
    CompactSymmetricStageTiming *timing) {
  checkInputs(compactMatrix, meta);
  const ISparseOps &ops = getSparseOpsCUDA();
  if (ops.backend() != BackendType::CUDA) {
    return false;
  }

  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t rows = static_cast<size_t>(meta.totalRows);
  std::fill(numer, numer + nvox, 0.0);

  std::vector<double> imgRot(nvox, 0.0);
  std::vector<double> y(rows, 0.0);
  std::vector<double> ratio(rows, 0.0);
  std::vector<double> bpRot(nvox, 0.0);
  std::vector<double> accum(nvox, 0.0);

  for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
    for (int z = 0; z < meta.zBranchCount(); ++z) {
      const bool zMirror = z != 0;
      auto stageStart = Clock::now();
      transformForward(smoothedImage, imgRot, meta, sym, zMirror, threads);
      if (timing)
        addTiming(timing->transformForwardSeconds, stageStart);

      std::fill(y.begin(), y.end(), 0.0);
      stageStart = Clock::now();
      ops.spmv(compactMatrix, imgRot.data(), y.data(), threads);
      if (timing)
        addTiming(timing->forwardSeconds, stageStart);

      stageStart = Clock::now();
      makeProjectionRatio(meta, fullProjection, fullBackground, y, sym, zMirror, ratio,
                          threads);
      if (timing)
        addTiming(timing->ratioSeconds, stageStart);

      std::fill(bpRot.begin(), bpRot.end(), 0.0);
      stageStart = Clock::now();
      ops.spmvT(compactMatrix, ratio.data(), bpRot.data(), threads);
      if (timing)
        addTiming(timing->backwardSeconds, stageStart);

      stageStart = Clock::now();
      transformBackwardAdd(bpRot, accum, meta, sym, zMirror, threads);
      if (timing)
        addTiming(timing->transformBackwardSeconds, stageStart);
    }
  }

  std::copy(accum.begin(), accum.end(), numer);
  if (timing) {
    timing->numeratorSeconds += elapsedSeconds(totalStart);
    ++timing->numeratorCalls;
  }
  return true;
}

bool computeCompactSymmetricForwardProjectionCUDA(
    const spect::io::CSRView &compactMatrix,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *smoothedImage, int threads, double *fullExpected) {
  checkInputs(compactMatrix, meta);
  if (smoothedImage == nullptr || fullExpected == nullptr) {
    throw std::runtime_error(
        "compact forward projection received null image/output data.");
  }
  const ISparseOps &ops = getSparseOpsCUDA();
  if (ops.backend() != BackendType::CUDA) {
    return false;
  }

  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());
  std::fill(fullExpected, fullExpected + fullRows, 0.0);
  std::vector<uint8_t> coverage(fullRows, 0);
  std::vector<double> imgRot(nvox, 0.0);
  std::vector<double> y(static_cast<size_t>(meta.totalRows), 0.0);
  for (const auto &branch : enumerateCompactBranches(meta)) {
    transformForward(smoothedImage, imgRot, meta, branch.symId,
                     branch.zMirror, threads);
    std::fill(y.begin(), y.end(), 0.0);
    ops.spmv(compactMatrix, imgRot.data(), y.data(), threads);
    scatterCompactForwardProjection(meta, 0, y, branch.symId, branch.zMirror,
                                    fullExpected, coverage);
  }
  requireCompleteForwardProjectionCoverage(coverage);
  return true;
}

bool computeCompactChunkSensitivityCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta, int threads, double *denom,
    CompactSymmetricStageTiming *timing, const uint16_t *attFactors) {
  if (chunks.totalRows() != static_cast<uint64_t>(meta.compactRows()) ||
      chunks.cols() != meta.totalVoxels) {
    throw std::runtime_error(
        "compact chunk sensitivity matrix/metadata mismatch.");
  }
  if (meta.mode != spect::io::CompactSymmetryMode::Sym45Z &&
      meta.mode != spect::io::CompactSymmetryMode::Sym90Z) {
    throw std::runtime_error(
        "compact chunk CUDA helper only supports SYM_45Z and SYM_90Z.");
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error("compact chunk CUDA helper requires Nx == Ny.");
  }

  const ISparseOps &ops = getSparseOpsCUDA();
  if (ops.backend() != BackendType::CUDA) {
    return false;
  }
  warmCompactChunkResidentDeviceCache(chunks);

  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const auto branches = enumerateCompactBranches(meta);
  const size_t chunkCount = chunks.chunks().size();
  std::fill(denom, denom + nvox, 0.0);
  std::vector<float> ratio;
  std::vector<float> bpRot(nvox, 0.0f);
  std::vector<double> accum(nvox, 0.0);

  for (size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
    const auto chunk = chunks.loadChunk(chunkIndex);
    const spect::io::CSRView view = chunk->view();
    const int compactRowBase = chunk->info.angleOffset * meta.compactRaysPerAngle;
    for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
      const auto &branch = branches[branchIndex];
      auto stageStart = Clock::now();
      makeUnitRatioChunkFloat(meta, compactRowBase, chunk->info.rows,
                              branch.symId, branch.zMirror, ratio, threads);
      if (timing)
        addTiming(timing->ratioSeconds, stageStart);
      stageStart = Clock::now();
      std::fill(bpRot.begin(), bpRot.end(), 0.0f);
      const bool backwardOk =
          attFactors != nullptr
              ? compactAttenSpmvTFloatCUDA(
                    view, attFactors, compactAttenTableCount(meta),
                    makeCompactAttenBakeParams(meta, compactRowBase,
                                               branch.symId, branch.zMirror),
                    ratio.data(), bpRot.data())
              : spmvTFloatCUDA(view, ratio.data(), bpRot.data());
      if (!backwardOk) {
        return false;
      }
      if (timing)
        addTiming(timing->backwardSeconds, stageStart);
      stageStart = Clock::now();
      transformBackwardAdd(bpRot.data(), accum, meta, branch.symId,
                           branch.zMirror, threads);
      if (timing)
        addTiming(timing->transformBackwardSeconds, stageStart);
    }
  }

  std::copy(accum.begin(), accum.end(), denom);
  if (timing) {
    timing->sensitivitySeconds += elapsedSeconds(totalStart);
    ++timing->sensitivityCalls;
  }
  return true;
}

bool prepareCompactChunkExecutionCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks) {
  warmCompactChunkResidentDeviceCache(chunks);
  return true;
}

bool runCompactChunkMlemNumerCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *fullProjection, const double *fullBackground,
    const double *smoothedImage, int threads, double *numer,
    bool reverseChunkOrder, CompactSymmetricStageTiming *timing,
    const uint16_t *attFactors) {
  if (chunks.totalRows() != static_cast<uint64_t>(meta.compactRows()) ||
      chunks.cols() != meta.totalVoxels) {
    throw std::runtime_error("compact chunk numerator matrix/metadata mismatch.");
  }
  if (meta.mode != spect::io::CompactSymmetryMode::Sym45Z &&
      meta.mode != spect::io::CompactSymmetryMode::Sym90Z) {
    throw std::runtime_error(
        "compact chunk CUDA helper only supports SYM_45Z and SYM_90Z.");
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error("compact chunk CUDA helper requires Nx == Ny.");
  }

  const ISparseOps &ops = getSparseOpsCUDA();
  if (ops.backend() != BackendType::CUDA) {
    return false;
  }
  warmCompactChunkResidentDeviceCache(chunks);

  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const auto branches = enumerateCompactBranches(meta);
  const size_t chunkCount = chunks.chunks().size();
  // The fused device-ratio/device-backward kernels have no attenuation
  // semantics; CTAC forces the generic bake + spmv route.
  const bool useDeviceBackward =
      compactDeviceBackwardEnabled() && attFactors == nullptr;
  const bool useDeviceRatio = compactDeviceRatioEnabled() &&
                              fullBackground == nullptr &&
                              attFactors == nullptr;
  const size_t fullProjectionCount = static_cast<size_t>(meta.fullRows());
  std::fill(numer, numer + nvox, 0.0);

  std::vector<float> rotatedImages(branches.size() * nvox, 0.0f);
  std::vector<float> y;
  std::vector<float> ratio;
  std::vector<float> bpRot(nvox, 0.0f);
  std::vector<double> accum(nvox, 0.0);

  if (useDeviceBackward) {
    if (!compactDeviceBackwardBeginCUDA(nvox)) {
      return false;
    }
  }

  for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
    const auto &branch = branches[branchIndex];
    float *imgRot = rotatedImages.data() + branchIndex * nvox;
    const auto stageStart = Clock::now();
    transformForwardIntoFloat(smoothedImage, imgRot, meta, branch.symId,
                              branch.zMirror, threads);
    if (timing)
      addTiming(timing->transformForwardSeconds, stageStart);
  }

  const auto processChunk = [&](size_t chunkIndex) -> bool {
    const auto chunk = chunks.loadChunk(chunkIndex);
    const spect::io::CSRView view = chunk->view();
    const int compactRowBase = chunk->info.angleOffset * meta.compactRaysPerAngle;
    for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
      const auto &branch = branches[branchIndex];
      const float *imgRot = rotatedImages.data() + branchIndex * nvox;
      if (useDeviceRatio) {
        CompactDeviceRatioTiming deviceTiming;
        const CompactDeviceRatioParams params = makeCompactDeviceRatioParams(
            meta, compactRowBase, branch.symId, branch.zMirror);
        if (useDeviceBackward) {
          CompactDeviceBackwardTiming backwardTiming;
          const CompactDeviceTransformParams transformParams =
              makeCompactDeviceTransformParams(meta, branch.symId,
                                               branch.zMirror);
          if (!compactDeviceRatioSpmvTAccumulateFloatCUDA(
                  view, imgRot, fullProjection, fullProjectionCount, params,
                  transformParams, &deviceTiming, &backwardTiming)) {
            return false;
          }
          if (timing) {
            timing->forwardSeconds += deviceTiming.forwardSeconds;
            timing->ratioSeconds += deviceTiming.ratioSeconds;
            timing->backwardSeconds += deviceTiming.backwardSeconds;
            timing->transformBackwardSeconds +=
                backwardTiming.transformBackwardSeconds;
          }
          continue;
        }
        if (!compactDeviceRatioSpmvTFloatCUDA(
                view, imgRot, fullProjection, fullProjectionCount, params,
                bpRot.data(), &deviceTiming)) {
          return false;
        }
        if (timing) {
          timing->forwardSeconds += deviceTiming.forwardSeconds;
          timing->ratioSeconds += deviceTiming.ratioSeconds;
          timing->backwardSeconds += deviceTiming.backwardSeconds;
        }
        const auto stageStart = Clock::now();
        transformBackwardAdd(bpRot.data(), accum, meta, branch.symId,
                             branch.zMirror, threads);
        if (timing)
          addTiming(timing->transformBackwardSeconds, stageStart);
        continue;
      }
      y.assign(static_cast<size_t>(view.rows), 0.0f);
      auto stageStart = Clock::now();
      const bool forwardOk =
          attFactors != nullptr
              ? compactAttenSpmvFloatCUDA(
                    view, attFactors, compactAttenTableCount(meta),
                    makeCompactAttenBakeParams(meta, compactRowBase,
                                               branch.symId, branch.zMirror),
                    imgRot, y.data())
              : spmvFloatCUDA(view, imgRot, y.data());
      if (!forwardOk) {
        return false;
      }
      if (timing)
        addTiming(timing->forwardSeconds, stageStart);

      stageStart = Clock::now();
      makeProjectionRatioChunkFloat(meta, compactRowBase, fullProjection,
                                    fullBackground, y, branch.symId,
                                    branch.zMirror, ratio, threads);
      if (timing)
        addTiming(timing->ratioSeconds, stageStart);
      if (useDeviceBackward) {
        CompactDeviceBackwardTiming deviceTiming;
        const CompactDeviceTransformParams transformParams =
            makeCompactDeviceTransformParams(meta, branch.symId,
                                             branch.zMirror);
        if (!compactDeviceBackwardSpmvTAccumulateFloatCUDA(
                view, ratio.data(), transformParams, &deviceTiming)) {
          return false;
        }
        if (timing) {
          timing->backwardSeconds += deviceTiming.backwardSeconds;
          timing->transformBackwardSeconds +=
              deviceTiming.transformBackwardSeconds;
        }
        continue;
      }
      stageStart = Clock::now();
      std::fill(bpRot.begin(), bpRot.end(), 0.0f);
      const bool backwardOk =
          attFactors != nullptr
              ? compactAttenSpmvTFloatCUDA(
                    view, attFactors, compactAttenTableCount(meta),
                    makeCompactAttenBakeParams(meta, compactRowBase,
                                               branch.symId, branch.zMirror),
                    ratio.data(), bpRot.data())
              : spmvTFloatCUDA(view, ratio.data(), bpRot.data());
      if (!backwardOk) {
        return false;
      }
      if (timing)
        addTiming(timing->backwardSeconds, stageStart);
      stageStart = Clock::now();
      transformBackwardAdd(bpRot.data(), accum, meta, branch.symId,
                           branch.zMirror, threads);
      if (timing)
        addTiming(timing->transformBackwardSeconds, stageStart);
    }
    return true;
  };

  if (reverseChunkOrder) {
    for (size_t chunkIndex = chunkCount; chunkIndex-- > 0;) {
      if (!processChunk(chunkIndex)) {
        return false;
      }
    }
  } else {
    for (size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
      if (!processChunk(chunkIndex)) {
        return false;
      }
    }
  }

  if (useDeviceBackward) {
    if (!compactDeviceBackwardDownloadCUDA(numer, nvox)) {
      return false;
    }
  } else {
    std::copy(accum.begin(), accum.end(), numer);
  }
  if (timing) {
    timing->numeratorSeconds += elapsedSeconds(totalStart);
    ++timing->numeratorCalls;
  }
  return true;
}

bool computeCompactChunkForwardProjectionCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *smoothedImage, int threads, double *fullExpected,
    const uint16_t *attFactors) {
  if (chunks.totalRows() != static_cast<uint64_t>(meta.compactRows()) ||
      chunks.cols() != meta.totalVoxels) {
    throw std::runtime_error(
        "compact chunk forward projection matrix/metadata mismatch.");
  }
  if (meta.mode != spect::io::CompactSymmetryMode::Sym45Z &&
      meta.mode != spect::io::CompactSymmetryMode::Sym90Z) {
    throw std::runtime_error(
        "compact chunk forward projection only supports SYM_45Z and SYM_90Z.");
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error(
        "compact chunk forward projection requires Nx == Ny.");
  }
  if (smoothedImage == nullptr || fullExpected == nullptr) {
    throw std::runtime_error(
        "compact chunk forward projection received null image/output data.");
  }
  const ISparseOps &ops = getSparseOpsCUDA();
  if (ops.backend() != BackendType::CUDA) {
    return false;
  }
  warmCompactChunkResidentDeviceCache(chunks);

  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());
  std::fill(fullExpected, fullExpected + fullRows, 0.0);
  std::vector<uint8_t> coverage(fullRows, 0);
  std::vector<float> imgRot(nvox, 0.0f);
  std::vector<float> y;
  for (const auto &branch : enumerateCompactBranches(meta)) {
    transformForwardIntoFloat(smoothedImage, imgRot.data(), meta, branch.symId,
                              branch.zMirror, threads);
    for (size_t chunkIndex = 0; chunkIndex < chunks.chunks().size();
         ++chunkIndex) {
      const auto chunk = chunks.loadChunk(chunkIndex);
      const spect::io::CSRView view = chunk->view();
      const int compactRowBase =
          chunk->info.angleOffset * meta.compactRaysPerAngle;
      y.assign(static_cast<size_t>(view.rows), 0.0f);
      const bool ok =
          attFactors != nullptr
              ? compactAttenSpmvFloatCUDA(
                    view, attFactors, compactAttenTableCount(meta),
                    makeCompactAttenBakeParams(meta, compactRowBase,
                                               branch.symId, branch.zMirror),
                    imgRot.data(), y.data())
              : spmvFloatCUDA(view, imgRot.data(), y.data());
      if (!ok) {
        return false;
      }
      scatterCompactForwardProjection(meta, compactRowBase, y, branch.symId,
                                      branch.zMirror, fullExpected, coverage);
    }
  }
  requireCompleteForwardProjectionCoverage(coverage);
  return true;
}

bool runCompactChunkMlemNumerMultiBedCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *const *fullProjections,
    const double *const *fullBackgrounds,
    const double *const *smoothedImages, int bedCount, int threads,
    double *const *numers, bool reverseChunkOrder,
    CompactSymmetricStageTiming *timing,
    const uint16_t *const *attFactors) {
  if (bedCount < 1 || bedCount > spect::core::kMaxBeds) {
    throw std::runtime_error(
        "compact chunk multi-bed CUDA helper supports 1.." +
        std::to_string(spect::core::kMaxBeds) + " beds.");
  }
  if (fullProjections == nullptr || smoothedImages == nullptr ||
      numers == nullptr) {
    throw std::runtime_error(
        "compact chunk multi-bed CUDA helper received null bed pointer arrays.");
  }
  for (int bed = 0; bed < bedCount; ++bed) {
    if (fullProjections[bed] == nullptr || smoothedImages[bed] == nullptr ||
        numers[bed] == nullptr) {
      throw std::runtime_error(
          "compact chunk multi-bed CUDA helper received null bed data.");
    }
  }
  if (bedCount == 1) {
    return runCompactChunkMlemNumerCUDA(
        chunks, meta, fullProjections[0],
        fullBackgrounds == nullptr ? nullptr : fullBackgrounds[0],
        smoothedImages[0], threads, numers[0], reverseChunkOrder, timing,
        attFactors == nullptr ? nullptr : attFactors[0]);
  }

  if (chunks.totalRows() != static_cast<uint64_t>(meta.compactRows()) ||
      chunks.cols() != meta.totalVoxels) {
    throw std::runtime_error(
        "compact chunk multi-bed numerator matrix/metadata mismatch.");
  }
  if (meta.mode != spect::io::CompactSymmetryMode::Sym45Z &&
      meta.mode != spect::io::CompactSymmetryMode::Sym90Z) {
    throw std::runtime_error(
        "compact chunk multi-bed CUDA helper only supports SYM_45Z and SYM_90Z.");
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error(
        "compact chunk multi-bed CUDA helper requires Nx == Ny.");
  }

  const ISparseOps &ops = getSparseOpsCUDA();
  if (ops.backend() != BackendType::CUDA) {
    return false;
  }
  warmCompactChunkResidentDeviceCache(chunks);

  bool hasBackground = false;
  if (fullBackgrounds != nullptr) {
    for (int bed = 0; bed < bedCount; ++bed) {
      hasBackground = hasBackground || fullBackgrounds[bed] != nullptr;
    }
  }
  bool hasAtten = false;
  if (attFactors != nullptr) {
    for (int bed = 0; bed < bedCount; ++bed) {
      hasAtten = hasAtten || attFactors[bed] != nullptr;
    }
  }

  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const auto branches = enumerateCompactBranches(meta);
  const size_t branchCount = branches.size();
  const size_t chunkCount = chunks.chunks().size();
  // The fused device-ratio path now supports CTAC. When device-ratio is
  // enabled with CTAC, the device-backward accumulator is still required.
  const bool useDeviceRatio =
      compactDeviceRatioEnabled() && !hasBackground;
  const bool useDeviceBackward =
      compactDeviceBackwardEnabled() && (!hasAtten || useDeviceRatio);
  const size_t fullProjectionCount = static_cast<size_t>(meta.fullRows());

  std::vector<float> rotatedImages(static_cast<size_t>(bedCount) *
                                       branchCount * nvox,
                                   0.0f);
  std::vector<std::vector<double>> accum;
  if (!useDeviceBackward) {
    accum.assign(static_cast<size_t>(bedCount), std::vector<double>(nvox, 0.0));
  }
  std::vector<float> y;
  std::vector<float> ratio;
  std::vector<float> bpRot(nvox, 0.0f);

  for (int bed = 0; bed < bedCount; ++bed) {
    std::fill(numers[bed], numers[bed] + nvox, 0.0);
    if (useDeviceBackward) {
      if (!compactDeviceBackwardBeginCUDA(nvox, bed)) {
        return false;
      }
    }
    for (size_t branchIndex = 0; branchIndex < branchCount; ++branchIndex) {
      const auto &branch = branches[branchIndex];
      float *imgRot = rotatedImages.data() +
                      (static_cast<size_t>(bed) * branchCount + branchIndex) *
                          nvox;
      const auto stageStart = Clock::now();
      transformForwardIntoFloat(smoothedImages[bed], imgRot, meta, branch.symId,
                                branch.zMirror, threads);
      if (timing)
        addTiming(timing->transformForwardSeconds, stageStart);
    }
  }

  const auto processChunk = [&](size_t chunkIndex) -> bool {
    const auto chunk = chunks.loadChunk(chunkIndex);
    const spect::io::CSRView view = chunk->view();
    const int compactRowBase = chunk->info.angleOffset * meta.compactRaysPerAngle;
    for (int bed = 0; bed < bedCount; ++bed) {
      const double *fullProjection = fullProjections[bed];
      const double *fullBackground =
          fullBackgrounds == nullptr ? nullptr : fullBackgrounds[bed];
      for (size_t branchIndex = 0; branchIndex < branchCount; ++branchIndex) {
        const auto &branch = branches[branchIndex];
        const float *imgRot =
            rotatedImages.data() +
            (static_cast<size_t>(bed) * branchCount + branchIndex) * nvox;
        if (useDeviceRatio) {
          CompactDeviceRatioTiming deviceTiming;
          const CompactDeviceRatioParams params = makeCompactDeviceRatioParams(
              meta, compactRowBase, branch.symId, branch.zMirror);
          if (useDeviceBackward) {
            CompactDeviceBackwardTiming backwardTiming;
            const CompactDeviceTransformParams transformParams =
                makeCompactDeviceTransformParams(meta, branch.symId,
                                                 branch.zMirror);
            const uint16_t *bedAtt =
                attFactors == nullptr ? nullptr : attFactors[bed];
            if (bedAtt != nullptr) {
              if (!compactDeviceRatioAttenSpmvTAccumulateFloatCUDA(
                      view, imgRot, bedAtt, compactAttenTableCount(meta),
                      makeCompactAttenBakeParams(meta, compactRowBase,
                                                 branch.symId, branch.zMirror),
                      fullProjection, fullProjectionCount, params,
                      transformParams, &deviceTiming, &backwardTiming, bed,
                      bed)) {
                return false;
              }
            } else {
              if (!compactDeviceRatioSpmvTAccumulateFloatCUDA(
                      view, imgRot, fullProjection, fullProjectionCount, params,
                      transformParams, &deviceTiming, &backwardTiming, bed,
                      bed)) {
                return false;
              }
            }
            if (timing) {
              timing->forwardSeconds += deviceTiming.forwardSeconds;
              timing->ratioSeconds += deviceTiming.ratioSeconds;
              timing->backwardSeconds += deviceTiming.backwardSeconds;
              timing->transformBackwardSeconds +=
                  backwardTiming.transformBackwardSeconds;
            }
            continue;
          }
          if (!compactDeviceRatioSpmvTFloatCUDA(
                  view, imgRot, fullProjection, fullProjectionCount, params,
                  bpRot.data(), &deviceTiming, bed)) {
            return false;
          }
          if (timing) {
            timing->forwardSeconds += deviceTiming.forwardSeconds;
            timing->ratioSeconds += deviceTiming.ratioSeconds;
            timing->backwardSeconds += deviceTiming.backwardSeconds;
          }
          const auto stageStart = Clock::now();
          std::vector<double> &bedAccum = accum[static_cast<size_t>(bed)];
          transformBackwardAdd(bpRot.data(), bedAccum, meta, branch.symId,
                               branch.zMirror, threads);
          if (timing)
            addTiming(timing->transformBackwardSeconds, stageStart);
          continue;
        }

        const uint16_t *bedAtt =
            attFactors == nullptr ? nullptr : attFactors[bed];
        y.assign(static_cast<size_t>(view.rows), 0.0f);
        auto stageStart = Clock::now();
        const bool forwardOk =
            bedAtt != nullptr
                ? compactAttenSpmvFloatCUDA(
                      view, bedAtt, compactAttenTableCount(meta),
                      makeCompactAttenBakeParams(meta, compactRowBase,
                                                 branch.symId, branch.zMirror),
                      imgRot, y.data(), bed)
                : spmvFloatCUDA(view, imgRot, y.data());
        if (!forwardOk) {
          return false;
        }
        if (timing)
          addTiming(timing->forwardSeconds, stageStart);

        stageStart = Clock::now();
        makeProjectionRatioChunkFloat(meta, compactRowBase, fullProjection,
                                      fullBackground, y, branch.symId,
                                      branch.zMirror, ratio, threads);
        if (timing)
          addTiming(timing->ratioSeconds, stageStart);

        if (useDeviceBackward && bedAtt == nullptr) {
          CompactDeviceBackwardTiming deviceTiming;
          const CompactDeviceTransformParams transformParams =
              makeCompactDeviceTransformParams(meta, branch.symId,
                                               branch.zMirror);
          if (!compactDeviceBackwardSpmvTAccumulateFloatCUDA(
                  view, ratio.data(), transformParams, &deviceTiming, bed)) {
            return false;
          }
          if (timing) {
            timing->backwardSeconds += deviceTiming.backwardSeconds;
            timing->transformBackwardSeconds +=
                deviceTiming.transformBackwardSeconds;
          }
          continue;
        }

        stageStart = Clock::now();
        std::fill(bpRot.begin(), bpRot.end(), 0.0f);
        const bool backwardOk =
            bedAtt != nullptr
                ? compactAttenSpmvTFloatCUDA(
                      view, bedAtt, compactAttenTableCount(meta),
                      makeCompactAttenBakeParams(meta, compactRowBase,
                                                 branch.symId, branch.zMirror),
                      ratio.data(), bpRot.data(), bed)
                : spmvTFloatCUDA(view, ratio.data(), bpRot.data());
        if (!backwardOk) {
          return false;
        }
        if (timing)
          addTiming(timing->backwardSeconds, stageStart);

        stageStart = Clock::now();
        std::vector<double> &bedAccum = accum[static_cast<size_t>(bed)];
        transformBackwardAdd(bpRot.data(), bedAccum, meta, branch.symId,
                             branch.zMirror, threads);
        if (timing)
          addTiming(timing->transformBackwardSeconds, stageStart);
      }
    }
    return true;
  };

  if (reverseChunkOrder) {
    for (size_t chunkIndex = chunkCount; chunkIndex-- > 0;) {
      if (!processChunk(chunkIndex)) {
        return false;
      }
    }
  } else {
    for (size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
      if (!processChunk(chunkIndex)) {
        return false;
      }
    }
  }

  for (int bed = 0; bed < bedCount; ++bed) {
    if (useDeviceBackward) {
      if (!compactDeviceBackwardDownloadCUDA(numers[bed], nvox, bed)) {
        return false;
      }
    } else {
      const std::vector<double> &bedAccum = accum[static_cast<size_t>(bed)];
      std::copy(bedAccum.begin(), bedAccum.end(), numers[bed]);
    }
  }
  if (timing) {
    timing->numeratorSeconds += elapsedSeconds(totalStart);
    ++timing->numeratorCalls;
  }
  return true;
}

} // namespace spect::cuda

#endif
