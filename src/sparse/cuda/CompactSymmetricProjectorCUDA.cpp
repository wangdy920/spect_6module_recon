#include "sparse/CompactSymmetricProjector.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)

#include "core/ReconLimits.h"
#include "cuda/CudaConfig.h"
#include "cuda/CudaLaunchConfig.h"
#include "sparse/SparseOpsCUDA.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spect::cuda {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint64_t kCompactChunkResidentReserveBytes =
    kCompactChunkResidentDeviceReserveBytes;

struct CompactChunkResidentCacheLease {
  uint64_t cacheBudget = 0;
  bool residentWarmupComplete = false;
  std::future<void> hostPreload;
};

int effectiveThreads(int threads) { return threads > 0 ? threads : 1; }

double elapsedSeconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void addTiming(double &target, Clock::time_point start) {
  target += elapsedSeconds(start);
}


bool compactBranchBatchEnabled() {
  const char *value = std::getenv("SPECT_COMPACT_BRANCH_BATCH");
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  return value[0] != '0' && value[0] != 'n' && value[0] != 'N' &&
         value[0] != 'f' && value[0] != 'F';
}

CompactDeviceRatioParams makeCompactDeviceRatioParams(
    const spect::io::CompactSymmetryMetadata &meta, int compactRowBase,
    int symId, bool zMirror) {
  CompactDeviceRatioParams params;
  params.symmetryMode = 90;
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
  params.symmetryMode = 90;
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
  params.symmetryMode = 90;
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

  static std::mutex warmupMutex;
  std::lock_guard<std::mutex> lock(warmupMutex);
  auto lease = std::static_pointer_cast<CompactChunkResidentCacheLease>(
      chunks.acceleratorLease());
  if (lease && lease->cacheBudget == cacheBudget &&
      lease->residentWarmupComplete) {
    return;
  }

  chunks.setAcceleratorLease({});
  lease = std::make_shared<CompactChunkResidentCacheLease>();
  lease->cacheBudget = cacheBudget;

  size_t residentPrefixCount = 1;
  try {
    const auto memory = cuda::getMemoryInfo();
    const uint64_t available =
        memory.free > kCompactChunkResidentReserveBytes
            ? static_cast<uint64_t>(memory.free) -
                  kCompactChunkResidentReserveBytes
            : 0;
    uint64_t bytes = 0;
    residentPrefixCount = 0;
    for (const auto &info : chunks.chunks()) {
      if (bytes > available || info.fileSizeBytes > available - bytes)
        break;
      bytes += info.fileSizeBytes;
      ++residentPrefixCount;
    }
    residentPrefixCount = std::max<size_t>(1, residentPrefixCount);
  } catch (...) {
    residentPrefixCount = 1;
  }
  residentPrefixCount =
      std::min(residentPrefixCount, chunks.chunks().size());

  std::vector<std::shared_ptr<const spect::io::CompactChunkCSRData>> loaded;
  std::vector<spect::io::CSRView> views;
  loaded.reserve(residentPrefixCount);
  views.reserve(residentPrefixCount);
  for (size_t chunkIndex = 0; chunkIndex < residentPrefixCount; ++chunkIndex) {
    auto chunk = chunks.loadChunk(chunkIndex);
    views.push_back(chunk->view());
    loaded.push_back(std::move(chunk));
  }

  if (residentPrefixCount < chunks.chunks().size()) {
    const auto *chunkDirectory = &chunks;
    lease->hostPreload = std::async(
        std::launch::async, [chunkDirectory, residentPrefixCount]() {
          try {
            for (size_t chunkIndex = residentPrefixCount;
                 chunkIndex < chunkDirectory->chunks().size(); ++chunkIndex) {
              (void)chunkDirectory->loadChunk(chunkIndex);
            }
          } catch (const std::exception &ex) {
            std::cerr << "[NOTE] compact host preload stopped: " << ex.what()
                      << "; foreground loading will retry.\n";
          }
        });
  }

  lease->residentWarmupComplete =
      views.empty() ||
      warmupResidentMatrixSetCUDA(views, kCompactChunkResidentReserveBytes);
  std::cerr << "[CompactChunkCUDA] Startup overlap: resident prefix "
            << residentPrefixCount << " / " << chunks.chunks().size();
  if (residentPrefixCount < chunks.chunks().size())
    std::cerr << ", remaining host chunks preload in background";
  std::cerr << '\n';
  chunks.setAcceleratorLease(lease);
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
        spect::io::mapCompactVoxelCoord(ix, iy, iz, nx, nz, symId, zMirror);
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
        spect::io::mapCompactVoxelCoord(ix, iy, iz, nx, nz, symId, zMirror);
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
        spect::io::mapCompactVoxelCoord(ix, iy, iz, nx, nz, symId, zMirror);
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
        spect::io::mapCompactVoxelCoord(ix, iy, iz, nx, nz, symId, zMirror);
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

void makeUnitRatioChunkFloat(
    const spect::io::CompactSymmetryMetadata &meta, int compactRowBase,
    int rows, int symId, bool zMirror, std::vector<float> &ratio, int threads) {
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


} // namespace




bool computeCompactChunkSensitivityCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta, int threads, double *denom,
    CompactSymmetricStageTiming *timing, const uint16_t *attFactors) {
  if (chunks.totalRows() != static_cast<uint64_t>(meta.compactRows()) ||
      chunks.cols() != meta.totalVoxels) {
    throw std::runtime_error(
        "compact chunk sensitivity matrix/metadata mismatch.");
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error("compact chunk CUDA helper requires Nx == Ny.");
  }

  warmCompactChunkResidentDeviceCache(chunks);

  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const auto branches = enumerateCompactBranches(meta);
  const size_t chunkCount = chunks.chunks().size();
  std::fill(denom, denom + nvox, 0.0);
  std::vector<float> ratio;
  std::vector<float> bpRot;
  std::vector<double> accum;
  const bool useDeviceBackward = attFactors == nullptr;
  const bool useBranchBatch =
      compactBranchBatchEnabled() && useDeviceBackward && attFactors == nullptr &&
      branches.size() > 1;
  if (!useDeviceBackward) {
    bpRot.assign(nvox, 0.0f);
    accum.assign(nvox, 0.0);
  } else {
    if (!compactDeviceBackwardBeginCUDA(nvox, 0)) {
      return false;
    }
  }

  for (size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
    const auto chunk = chunks.loadChunk(chunkIndex);
    const spect::io::CSRView view = chunk->view();
    const int compactRowBase = chunk->info.angleOffset * meta.compactRaysPerAngle;
    if (useBranchBatch) {
      CompactDeviceBackwardTiming backwardTiming;
      const CompactDeviceTransformParams commonTransform =
          makeCompactDeviceTransformParams(meta, 0, false);
      std::vector<int> symIds(branches.size());
      std::vector<int> zMirrors(branches.size());
      for (size_t b = 0; b < branches.size(); ++b) {
        symIds[b] = branches[b].symId;
        zMirrors[b] = branches[b].zMirror ? 1 : 0;
      }
      const bool ok = compactSensitivityBranchBatchSpmvTAccumulateFloatCUDA(
          view, commonTransform, symIds.data(), zMirrors.data(),
          static_cast<int>(branches.size()), &backwardTiming, 0, 0);
      if (!ok) {
        return false;
      }
      if (timing) {
        timing->backwardSeconds += backwardTiming.backwardSeconds;
        timing->transformBackwardSeconds +=
            backwardTiming.transformBackwardSeconds;
      }
      continue;
    }
    for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
      const auto &branch = branches[branchIndex];
      auto stageStart = Clock::now();
      makeUnitRatioChunkFloat(meta, compactRowBase, chunk->info.rows,
                              branch.symId, branch.zMirror, ratio, threads);
      if (timing)
        addTiming(timing->ratioSeconds, stageStart);
      stageStart = Clock::now();
      if (useDeviceBackward) {
        CompactDeviceBackwardTiming deviceTiming;
        const CompactDeviceTransformParams transformParams =
            makeCompactDeviceTransformParams(meta, branch.symId, branch.zMirror);
        if (!compactDeviceBackwardSpmvTAccumulateFloatCUDA(
                view, ratio.data(), transformParams, &deviceTiming, 0)) {
          return false;
        }
        if (timing) {
          timing->backwardSeconds += deviceTiming.backwardSeconds;
          timing->transformBackwardSeconds +=
              deviceTiming.transformBackwardSeconds;
        }
      } else {
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
  }

  if (useDeviceBackward) {
    if (!compactDeviceBackwardDownloadCUDA(denom, nvox, 0)) {
      return false;
    }
  } else {
    std::copy(accum.begin(), accum.end(), denom);
  }
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

bool prepareCompactBranchBatchInputFromGlobalCUDA(
    const spect::io::CompactSymmetryMetadata &meta,
    const double *globalImage, size_t globalImageCount, int globalNz,
    const int *bedZStarts, int bedCount, bool flipLocalZ,
    const spect::filters::PSFKernels *castorPsf, bool uploadGlobalImage) {
  if (bedCount <= 1 || !compactBranchBatchEnabled())
    return false;
  const CompactDeviceTransformParams commonTransform =
      makeCompactDeviceTransformParams(meta, 0, false);
  return spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
      globalImage, globalImageCount, globalNz, bedZStarts, bedCount,
      flipLocalZ, uploadGlobalImage, commonTransform,
      meta.xySymmetryCount() * meta.zBranchCount(), meta.zBranchCount(),
      castorPsf == nullptr ? nullptr : &castorPsf->kx,
      castorPsf == nullptr ? nullptr : &castorPsf->ky,
      castorPsf == nullptr ? nullptr : &castorPsf->kz);
}

bool finishCompactResidentMlemIterationCUDA(
    const spect::io::CompactSymmetryMetadata &meta,
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ, const double *denom, const uint8_t *mask,
    const spect::filters::PSFKernels *castorPsf,
    bool requirePositiveNumerator, bool refreshStaticInputs,
    bool *imageAllZero) {
  const CompactDeviceTransformParams commonTransform =
      makeCompactDeviceTransformParams(meta, 0, false);
  return spect::cuda::finishCompactResidentMlemIterationCUDA(
      globalImageCount, globalNz, bedZStarts, bedCount, flipLocalZ,
      commonTransform, denom, mask,
      castorPsf == nullptr ? nullptr : &castorPsf->kx,
      castorPsf == nullptr ? nullptr : &castorPsf->ky,
      castorPsf == nullptr ? nullptr : &castorPsf->kz,
      requirePositiveNumerator, refreshStaticInputs, imageAllZero);
}

bool runCompactChunkMlemNumerCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *fullProjection, const double *fullBackground,
    const double *smoothedImage, int threads, double *numer,
    bool reverseChunkOrder, CompactSymmetricStageTiming *timing,
    const uint16_t *attFactors, bool refreshProjectionInput) {
  if (chunks.totalRows() != static_cast<uint64_t>(meta.compactRows()) ||
      chunks.cols() != meta.totalVoxels) {
    throw std::runtime_error("compact chunk numerator matrix/metadata mismatch.");
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error("compact chunk CUDA helper requires Nx == Ny.");
  }

  warmCompactChunkResidentDeviceCache(chunks);

  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const auto branches = enumerateCompactBranches(meta);
  const size_t chunkCount = chunks.chunks().size();
  // The fused device-ratio/device-backward kernels have no attenuation
  // semantics; CTAC forces the generic bake + spmv route.
  const bool useDeviceBackward = attFactors == nullptr;
  const bool useDeviceRatio =
      fullBackground == nullptr && attFactors == nullptr;
  const bool useBranchBatch =
      compactBranchBatchEnabled() && useDeviceRatio && useDeviceBackward &&
      attFactors == nullptr && branches.size() > 1;
  const size_t fullProjectionCount = static_cast<size_t>(meta.fullRows());
  if (useDeviceRatio && refreshProjectionInput &&
      !refreshCompactProjectionCUDA(fullProjection, fullProjectionCount)) {
    return false;
  }
  std::fill(numer, numer + nvox, 0.0);

  std::vector<float> rotatedImages(branches.size() * nvox, 0.0f);
  std::vector<float> y;
  std::vector<float> ratio;
  std::vector<float> bpRot;
  std::vector<double> accum;

  if (useDeviceBackward) {
    if (!compactDeviceBackwardBeginCUDA(nvox)) {
      return false;
    }
  } else {
    bpRot.assign(nvox, 0.0f);
    accum.assign(nvox, 0.0);
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
  if (useBranchBatch &&
      !refreshCompactBranchBatchInputCUDA(rotatedImages.data(),
                                          rotatedImages.size())) {
    return false;
  }
  std::vector<int> branchSymIds(branches.size());
  std::vector<int> branchZMirrors(branches.size());
  for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
    branchSymIds[branchIndex] = branches[branchIndex].symId;
    branchZMirrors[branchIndex] = branches[branchIndex].zMirror ? 1 : 0;
  }
  const double *fullProjections[] = {fullProjection};

  const auto processChunk = [&](size_t chunkIndex) -> bool {
    const auto chunk = chunks.loadChunk(chunkIndex);
    const spect::io::CSRView view = chunk->view();
    const int compactRowBase = chunk->info.angleOffset * meta.compactRaysPerAngle;
    if (useBranchBatch) {
      CompactDeviceRatioTiming deviceTiming;
      CompactDeviceBackwardTiming backwardTiming;
      const CompactDeviceRatioParams commonRatio =
          makeCompactDeviceRatioParams(meta, compactRowBase, 0, false);
      const CompactDeviceTransformParams commonTransform =
          makeCompactDeviceTransformParams(meta, 0, false);
      const bool ok = compactDeviceRatioBedBranchBatchSpmvTAccumulateFloatCUDA(
          view, fullProjections, fullProjectionCount, commonRatio,
          commonTransform, branchSymIds.data(), branchZMirrors.data(),
          static_cast<int>(branches.size()), 1, &deviceTiming,
          &backwardTiming);
      if (!ok) {
        return false;
      }
      if (timing) {
        timing->forwardSeconds += deviceTiming.forwardSeconds;
        timing->ratioSeconds += deviceTiming.ratioSeconds;
        timing->backwardSeconds += deviceTiming.backwardSeconds;
        timing->transformBackwardSeconds +=
            backwardTiming.transformBackwardSeconds;
      }
      return true;
    }
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
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error(
        "compact chunk forward projection requires Nx == Ny.");
  }
  if (smoothedImage == nullptr || fullExpected == nullptr) {
    throw std::runtime_error(
        "compact chunk forward projection received null image/output data.");
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
    const uint16_t *const *attFactors, bool refreshProjectionInputs,
    bool branchBatchInputPrepared, bool downloadBackwardOutputs) {
  if (bedCount < 1 || bedCount > spect::core::kMaxBeds) {
    throw std::runtime_error(
        "compact chunk multi-bed CUDA helper supports 1.." +
        std::to_string(spect::core::kMaxBeds) + " beds.");
  }
  if (fullProjections == nullptr ||
      (smoothedImages == nullptr && !branchBatchInputPrepared) ||
      (numers == nullptr && downloadBackwardOutputs)) {
    throw std::runtime_error(
        "compact chunk multi-bed CUDA helper received null bed pointer arrays.");
  }
  for (int bed = 0; bed < bedCount; ++bed) {
    if (fullProjections[bed] == nullptr ||
        (!branchBatchInputPrepared && smoothedImages[bed] == nullptr) ||
        (downloadBackwardOutputs && numers[bed] == nullptr)) {
      throw std::runtime_error(
          "compact chunk multi-bed CUDA helper received null bed data.");
    }
  }
  if (bedCount == 1) {
    if (branchBatchInputPrepared || !downloadBackwardOutputs) {
      throw std::runtime_error(
          "compact resident branch-batch execution requires multiple beds.");
    }
    return runCompactChunkMlemNumerCUDA(
        chunks, meta, fullProjections[0],
        fullBackgrounds == nullptr ? nullptr : fullBackgrounds[0],
        smoothedImages[0], threads, numers[0], reverseChunkOrder, timing,
        attFactors == nullptr ? nullptr : attFactors[0],
        refreshProjectionInputs);
  }

  if (chunks.totalRows() != static_cast<uint64_t>(meta.compactRows()) ||
      chunks.cols() != meta.totalVoxels) {
    throw std::runtime_error(
        "compact chunk multi-bed numerator matrix/metadata mismatch.");
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error(
        "compact chunk multi-bed CUDA helper requires Nx == Ny.");
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
  // Keep the projection ratio and backward accumulator resident whenever the
  // requested inputs are supported by the exact device path.
  const bool useDeviceRatio = !hasBackground;
  const bool useDeviceBackward = !hasAtten || useDeviceRatio;
  const bool useBranchBatch =
      compactBranchBatchEnabled() && useDeviceRatio && useDeviceBackward &&
      !hasAtten && branchCount > 1;
  if (branchBatchInputPrepared && !useBranchBatch) {
    throw std::runtime_error(
        "compact resident branch-batch input requires the batch CUDA path.");
  }
  const size_t fullProjectionCount = static_cast<size_t>(meta.fullRows());
  if (useDeviceRatio && refreshProjectionInputs) {
    for (int bed = 0; bed < bedCount; ++bed) {
      if (!refreshCompactProjectionCUDA(fullProjections[bed],
                                        fullProjectionCount, bed)) {
        return false;
      }
    }
  }

  std::vector<float> rotatedImages;
  if (!branchBatchInputPrepared) {
    rotatedImages.assign(static_cast<size_t>(bedCount) * branchCount * nvox,
                         0.0f);
  }
  std::vector<std::vector<double>> accum;
  std::vector<float> bpRot;
  if (!useDeviceBackward) {
    accum.assign(static_cast<size_t>(bedCount), std::vector<double>(nvox, 0.0));
    bpRot.assign(nvox, 0.0f);
  }
  std::vector<float> y;
  std::vector<float> ratio;

  for (int bed = 0; bed < bedCount; ++bed) {
    if (downloadBackwardOutputs)
      std::fill(numers[bed], numers[bed] + nvox, 0.0);
    if (useDeviceBackward) {
      if (!compactDeviceBackwardBeginCUDA(nvox, bed)) {
        return false;
      }
    }
    if (branchBatchInputPrepared)
      continue;
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
  if (useBranchBatch && !branchBatchInputPrepared &&
      !refreshCompactBranchBatchInputCUDA(rotatedImages.data(),
                                          rotatedImages.size())) {
    return false;
  }

  std::vector<int> branchSymIds(branchCount);
  std::vector<int> branchZMirrors(branchCount);
  for (size_t branchIndex = 0; branchIndex < branchCount; ++branchIndex) {
    branchSymIds[branchIndex] = branches[branchIndex].symId;
    branchZMirrors[branchIndex] = branches[branchIndex].zMirror ? 1 : 0;
  }

  const auto processChunk = [&](size_t chunkIndex) -> bool {
    const auto chunk = chunks.loadChunk(chunkIndex);
    const spect::io::CSRView view = chunk->view();
    const int compactRowBase = chunk->info.angleOffset * meta.compactRaysPerAngle;
    if (useBranchBatch) {
      CompactDeviceRatioTiming deviceTiming;
      CompactDeviceBackwardTiming backwardTiming;
      const CompactDeviceRatioParams commonRatio =
          makeCompactDeviceRatioParams(meta, compactRowBase, 0, false);
      const CompactDeviceTransformParams commonTransform =
          makeCompactDeviceTransformParams(meta, 0, false);
      const bool ok = compactDeviceRatioBedBranchBatchSpmvTAccumulateFloatCUDA(
          view, fullProjections, fullProjectionCount, commonRatio,
          commonTransform, branchSymIds.data(),
          branchZMirrors.data(), static_cast<int>(branchCount), bedCount,
          &deviceTiming, &backwardTiming);
      if (!ok) {
        return false;
      }
      if (timing) {
        timing->forwardSeconds += deviceTiming.forwardSeconds;
        timing->ratioSeconds += deviceTiming.ratioSeconds;
        timing->backwardSeconds += deviceTiming.backwardSeconds;
        timing->transformBackwardSeconds +=
            backwardTiming.transformBackwardSeconds;
      }
      return true;
    }
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

  if (downloadBackwardOutputs) {
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
  }
  if (timing) {
    timing->numeratorSeconds += elapsedSeconds(totalStart);
    ++timing->numeratorCalls;
  }
  return true;
}

} // namespace spect::cuda

#endif
