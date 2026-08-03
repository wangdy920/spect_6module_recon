#include "recon/CompactSymmetricReconLoop.h"

#include "recon/BedOffsets.h"
#include "recon/BsremUpdate.h"
#include "recon/ProjectionOps.h"
#include "core/Masks.h"
#include "sparse/CompactSymmetricProjector.h"
#include "filters/ButterworthFilter.h"
#include "io/BadDetectorMask.h"
#include "io/MHDWriter.h"
#include "io/DetZRowWeights.h"
#include "io/VolumeLoader.h"
#include "recon/BsremKappaMap.h"
#include "recon/ReconLoopHelpers.h"
#include "pipeline/RunLogger.h"
#include "postprocess/SmoothingOps.h"
#include "pipeline/VolumeOutput.h"
#include "regularizer/RegularizerCompat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
#include <cuda_runtime.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spect::main {
namespace {

using ProfileClock = std::chrono::steady_clock;
struct MemoryUsageSnapshot {
  bool hostAvailable = false;
  uint64_t hostWorkingSetBytes = 0;
  uint64_t hostPeakWorkingSetBytes = 0;
  bool cudaAvailable = false;
  uint64_t cudaUsedBytes = 0;
  uint64_t cudaFreeBytes = 0;
  uint64_t cudaTotalBytes = 0;
};

MemoryUsageSnapshot queryMemoryUsageSnapshot() {
  MemoryUsageSnapshot snapshot;
#ifdef _WIN32
  using K32GetProcessMemoryInfoFn =
      BOOL(WINAPI *)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
  HMODULE kernel32 = GetModuleHandleA("kernel32.dll");
  auto *getProcessMemoryInfo =
      kernel32 ? reinterpret_cast<K32GetProcessMemoryInfoFn>(
                     GetProcAddress(kernel32, "K32GetProcessMemoryInfo"))
               : nullptr;
  if (getProcessMemoryInfo) {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (getProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&counters),
            sizeof(counters))) {
      snapshot.hostAvailable = true;
      snapshot.hostWorkingSetBytes =
          static_cast<uint64_t>(counters.WorkingSetSize);
      snapshot.hostPeakWorkingSetBytes =
          static_cast<uint64_t>(counters.PeakWorkingSetSize);
    }
  }
#endif
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess &&
      totalBytes >= freeBytes) {
    snapshot.cudaAvailable = true;
    snapshot.cudaFreeBytes = static_cast<uint64_t>(freeBytes);
    snapshot.cudaTotalBytes = static_cast<uint64_t>(totalBytes);
    snapshot.cudaUsedBytes = static_cast<uint64_t>(totalBytes - freeBytes);
  }
#endif
  return snapshot;
}

void mergeMemoryPeak(MemoryUsageSnapshot &peak,
                     const MemoryUsageSnapshot &sample) {
  if (sample.hostAvailable) {
    peak.hostAvailable = true;
    peak.hostWorkingSetBytes =
        std::max(peak.hostWorkingSetBytes, sample.hostWorkingSetBytes);
    peak.hostPeakWorkingSetBytes =
        std::max(peak.hostPeakWorkingSetBytes,
                 sample.hostPeakWorkingSetBytes);
  }
  if (sample.cudaAvailable) {
    peak.cudaAvailable = true;
    peak.cudaUsedBytes = std::max(peak.cudaUsedBytes, sample.cudaUsedBytes);
    peak.cudaFreeBytes = sample.cudaFreeBytes;
    peak.cudaTotalBytes = std::max(peak.cudaTotalBytes, sample.cudaTotalBytes);
  }
}

std::string formatMemoryEntry(const std::string &label,
                              const MemoryUsageSnapshot &snapshot) {
  std::ostringstream os;
  os << "[MEM] " << label;
  if (snapshot.hostAvailable) {
    os << " host_working_set_peak_bytes=" << snapshot.hostWorkingSetBytes
       << " host_process_peak_working_set_bytes="
       << snapshot.hostPeakWorkingSetBytes;
  } else {
    os << " host_unavailable=1";
  }
  if (snapshot.cudaAvailable) {
    os << " cuda_used_peak_bytes=" << snapshot.cudaUsedBytes
       << " cuda_total_bytes=" << snapshot.cudaTotalBytes
       << " cuda_free_bytes_last=" << snapshot.cudaFreeBytes;
  } else {
    os << " cuda_unavailable=1";
  }
  return os.str();
}

struct VectorStats {
  size_t count = 0;
  size_t finite = 0;
  size_t nonzero = 0;
  double min = 0.0;
  double max = 0.0;
  double sum = 0.0;
};

struct ProjectionResidualStats {
  size_t count = 0;
  double measuredSum = 0.0;
  double expectedSum = 0.0;
  double poissonDeviance = 0.0;
  double pearsonSquaredSum = 0.0;
};

void addProjectionResidualSample(ProjectionResidualStats &stats,
                                 double measured, double expected) {
  constexpr double kEps = 1e-10;
  if (!std::isfinite(measured) || !std::isfinite(expected) || measured < 0.0) {
    throw std::runtime_error(
        "projection residual diagnostic received invalid projection data.");
  }
  const double q = std::max(expected, kEps);
  ++stats.count;
  stats.measuredSum += measured;
  stats.expectedSum += q;
  stats.poissonDeviance +=
      measured > 0.0
          ? 2.0 * (q - measured + measured * std::log(measured / q))
          : 2.0 * q;
  const double residual = measured - q;
  stats.pearsonSquaredSum += residual * residual / q;
}

double projectionRelativeBias(const ProjectionResidualStats &stats) {
  return (stats.measuredSum - stats.expectedSum) /
         std::max(stats.measuredSum, 1e-10);
}

double projectionPearsonRms(const ProjectionResidualStats &stats) {
  return stats.count > 0
             ? std::sqrt(stats.pearsonSquaredSum /
                         static_cast<double>(stats.count))
             : 0.0;
}

double projectionDeviancePerBin(const ProjectionResidualStats &stats) {
  return stats.count > 0
             ? stats.poissonDeviance / static_cast<double>(stats.count)
             : 0.0;
}

VectorStats computeVectorStats(const std::vector<double> &values) {
  VectorStats stats;
  stats.count = values.size();
  bool initialized = false;
  for (const double value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    ++stats.finite;
    if (value != 0.0) {
      ++stats.nonzero;
    }
    if (!initialized) {
      stats.min = value;
      stats.max = value;
      initialized = true;
    } else {
      stats.min = std::min(stats.min, value);
      stats.max = std::max(stats.max, value);
    }
    stats.sum += value;
  }
  return stats;
}

std::string formatStatsEntry(const std::string &label,
                             const VectorStats &stats) {
  std::ostringstream oss;
  oss << "[STATS] " << label << ": count=" << stats.count
      << ", finite=" << stats.finite << ", nonzero=" << stats.nonzero
      << ", min=" << std::setprecision(12) << stats.min
      << ", max=" << stats.max << ", sum=" << stats.sum;
  return oss.str();
}

std::filesystem::path normalizePathForCache(
    const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path normalized = std::filesystem::weakly_canonical(path, ec);
  if (ec)
    normalized = std::filesystem::absolute(path, ec);
  if (ec)
    normalized = path.lexically_normal();
  return normalized;
}

std::string psfModeKey(filters::PSFMode mode) {
  switch (mode) {
  case filters::PSFMode::None:
    return "None";
  case filters::PSFMode::Operator:
    return "Operator";
  }
  return "Unknown";
}

std::string psfInterpKey(filters::PSFInterp interp) {
  switch (interp) {
  case filters::PSFInterp::Linear:
    return "Linear";
  case filters::PSFInterp::Nearest:
    return "Nearest";
  }
  return "Unknown";
}

void appendFileIdentity(std::ostringstream &oss, const std::string &label,
                        const std::filesystem::path &path) {
  const auto normalized = normalizePathForCache(path);
  oss << label << "=" << normalized.string() << '\n';
  std::error_code ec;
  const auto size = std::filesystem::file_size(normalized, ec);
  if (ec) {
    oss << label << "_size=UNAVAILABLE\n";
  } else {
    oss << label << "_size=" << size << '\n';
  }
  ec.clear();
  const auto writeTime = std::filesystem::last_write_time(normalized, ec);
  if (ec) {
    oss << label << "_mtime=UNAVAILABLE\n";
  } else {
    oss << label << "_mtime=" << writeTime.time_since_epoch().count() << '\n';
  }
}

template <typename T>
void appendVectorLine(std::ostringstream &oss, const std::string &label,
                      const std::vector<T> &values) {
  oss << label << '=';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      oss << ',';
    oss << values[i];
  }
  oss << '\n';
}

std::filesystem::path resolveCompactDenomCacheMhdPath(const ReconArgs &args) {
  if (args.compactDenomCachePath.empty())
    return {};
  std::filesystem::path path(args.compactDenomCachePath);
  if (path.extension() != ".mhd")
    path.replace_extension(".mhd");
  return normalizePathForCache(path);
}

std::filesystem::path compactDenomManifestPath(
    const std::filesystem::path &mhdPath) {
  std::filesystem::path manifest = mhdPath;
  manifest.replace_extension(".manifest.txt");
  return manifest;
}

uint64_t fnv1a64(const std::string &text) {
  uint64_t value = 14695981039346656037ull;
  for (const unsigned char ch : text) {
    value ^= static_cast<uint64_t>(ch);
    value *= 1099511628211ull;
  }
  return value;
}

std::filesystem::path autoCompactDenomCachePath(const std::string &manifest) {
  std::ostringstream name;
  name << "auto_" << std::hex << fnv1a64(manifest) << ".mhd";
  return normalizePathForCache(
      std::filesystem::path("recon_out") / "compact_denom_cache" / name.str());
}

std::string abbreviateManifestLine(const std::string &line) {
  constexpr size_t kMaxManifestLine = 160;
  if (line.size() <= kMaxManifestLine)
    return line;
  return line.substr(0, kMaxManifestLine) + "...";
}

std::string summarizeManifestMismatch(const std::string &expected,
                                      const std::string &actual) {
  std::istringstream expectedStream(expected);
  std::istringstream actualStream(actual);
  std::string expectedLine;
  std::string actualLine;
  size_t lineNo = 1;
  while (true) {
    const bool haveExpected =
        static_cast<bool>(std::getline(expectedStream, expectedLine));
    const bool haveActual =
        static_cast<bool>(std::getline(actualStream, actualLine));
    if (!haveExpected && !haveActual)
      return "cache manifest mismatch";
    if (haveExpected != haveActual || expectedLine != actualLine) {
      std::ostringstream oss;
      oss << "cache manifest mismatch at line " << lineNo << ": expected '"
          << (haveExpected ? abbreviateManifestLine(expectedLine) : "<EOF>")
          << "', actual '"
          << (haveActual ? abbreviateManifestLine(actualLine) : "<EOF>")
          << "'";
      return oss.str();
    }
    ++lineNo;
  }
}

bool isCompactDenomManifestMismatch(const std::string &status) {
  return status.rfind("cache manifest mismatch", 0) == 0;
}

std::string buildCompactDenomCacheManifest(
    const ReconArgs &args, const OutputGeometry &geometry,
    const spect::io::CompactSymmetryMetadata &compactMeta,
    const std::string &compactMetaPath, bool compactChunkLayout,
    const spect::io::CompactChunkCSRDirectory *compactChunks,
    const std::vector<int> &bedZStarts,
    const spect::filters::PSFKernels *psfKernels) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << "cache_schema=compact_global_denom_v1\n";
  oss << "matrix_layout=compact-symmetric\n";
  oss << "matrix_source_kind=chunkCSRDir\n";
  appendFileIdentity(oss, "compact_meta", compactMetaPath);
  oss << "compact_mode="
      << spect::io::compactSymmetryModeName(compactMeta.mode) << '\n';
  oss << "compact_dims=" << compactMeta.Nx << ',' << compactMeta.Ny << ','
      << compactMeta.Nz << '\n';
  oss << "compact_voxel_size=" << compactMeta.voxelSize << '\n';
  oss << "compact_det=" << compactMeta.detNz << ',' << compactMeta.detNx
      << '\n';
  oss << "compact_angles=" << compactMeta.fullNumAngles << '\n';
  oss << "compact_rows=" << compactMeta.compactRows() << '\n';
  oss << "full_rows=" << compactMeta.fullRows() << '\n';
  oss << "matrix_cols=" << compactMeta.totalVoxels << '\n';
  if (compactChunkLayout) {
    oss << "chunk_dir="
        << normalizePathForCache(compactChunks->directory()).string() << '\n';
    oss << "chunk_count=" << compactChunks->chunks().size() << '\n';
    oss << "chunk_total_rows=" << compactChunks->totalRows() << '\n';
    oss << "chunk_total_nnz=" << compactChunks->totalNnz() << '\n';
    for (size_t i = 0; i < compactChunks->chunks().size(); ++i) {
      const auto &chunk = compactChunks->chunks()[i];
      const std::string prefix = "chunk_" + std::to_string(i);
      appendFileIdentity(oss, prefix, chunk.path);
      oss << prefix << "_rows=" << chunk.rows << '\n';
      oss << prefix << "_angle_offset=" << chunk.angleOffset << '\n';
      oss << prefix << "_angle_count=" << chunk.angleCount << '\n';
      oss << prefix << "_nnz=" << chunk.nnz << '\n';
      oss << prefix << "_file_size_bytes=" << chunk.fileSizeBytes << '\n';
    }
  }
  oss << "global_dims=" << geometry.nx << ',' << geometry.ny << ','
      << geometry.nz << '\n';
  oss << "global_spacing=" << geometry.vx << ',' << geometry.vy << ','
      << geometry.vz << '\n';
  // CTAC bakes attenuation into the matrix values, so the sensitivity depends
  // on the mu-map; keep these in the manifest to invalidate stale caches.
  // Lines are emitted only when CTAC is active so pre-existing no-CTAC caches
  // keep matching their stored manifests.
  if (args.doCTAC) {
    oss << "ctac=1\n";
    oss << "ctac_model=" << args.ctacModel << '\n';
    oss << "ctac_rot_sign=" << args.ctacRotSign << '\n';
    if (!args.attMapPath.empty()) {
      appendFileIdentity(oss, "ctac_mu_map", args.attMapPath);
    }
    if (!args.ctacAutoMuFrom.empty()) {
      appendFileIdentity(oss, "ctac_auto_mu_from", args.ctacAutoMuFrom);
      oss << "ctac_auto_mu_thresh=" << args.ctacAutoMuThresh << '\n';
      oss << "ctac_auto_mu_value=" << args.ctacAutoMuValue << '\n';
      oss << "ctac_auto_mu_source=" << args.ctacAutoMuValueSource << '\n';
    }
  }
  // detZ edge censoring masks rows out of the sensitivity, so the denom
  // depends on k; emit only when active so pre-existing caches keep matching
  // their stored manifests (same convention as the CTAC lines above).
  if (args.detZEdgeCensor > 0) {
    oss << "detz_edge_censor=" << args.detZEdgeCensor << '\n';
  }
  // Efficiency weights likewise change the sensitivity; record the file
  // identity and value count only when active (same convention as above).
  if (!args.detZRowWeightsFile.empty()) {
    appendFileIdentity(oss, "detz_row_weights", args.detZRowWeightsFile);
    oss << "detz_row_weights_count="
        << spect::io::loadDetZRowWeights(args.detZRowWeightsFile,
                                         compactMeta.detNz, compactMeta.detNx,
                                         compactMeta.fullNumAngles)
               .size()
        << '\n';
  }
  if (!args.normFactorFile.empty()) {
    appendFileIdentity(oss, "normalization_factor", args.normFactorFile);
    oss << "normalization_factor_count="
        << spect::io::loadDetZRowWeights(
               args.normFactorFile, compactMeta.detNz, compactMeta.detNx,
               compactMeta.fullNumAngles)
               .size()
        << '\n';
    oss << "normalization_factor_semantics="
           "projection_correction_c_to_efficiency_1_over_c\n";
  }
  if (!args.badDetectorFile.empty()) {
    const auto badDetectors =
        spect::io::loadBadDetectorList(args.badDetectorFile);
    if (!badDetectors.empty()) {
      appendFileIdentity(oss, "bad_detector_list", args.badDetectorFile);
      oss << "bad_detector_count=" << badDetectors.size() << '\n';
      for (size_t index = 0; index < badDetectors.size(); ++index)
        oss << "bad_detector_" << index << '='
            << spect::io::badDetectorGlobalNumber(badDetectors[index]) << ':'
            << spect::io::badDetectorName(badDetectors[index]) << '\n';
    }
  }
  oss << "bed_count=" << bedZStarts.size() << '\n';
  appendVectorLine(oss, "bed_z_starts", bedZStarts);
  oss << "bed_overlap_in_recon=0\n";
  appendVectorLine(oss, "output_bed_overlaps", args.outputBedOverlaps);
  if (args.zBedFlip && !args.projPaths.empty() && bedZStarts.size() > 1) {
    oss << "joint_virtual_z_flip=1\n";
    oss << "joint_overlap_accumulation=direct\n";
  }
  // Retired field, kept literal so the 448 MB of existing auto_<hash>.mhd denom
  // caches still key-match; dropping the line would silently rebuild all of them.
  oss << "joseph_passes=0\n";
  oss << "psf_mode=" << psfModeKey(args.psf.mode) << '\n';
  oss << "psf_fwhm=" << args.psf.fwhm_x << ',' << args.psf.fwhm_y << ','
      << args.psf.fwhm_z << ',' << args.psf.fwhm_mm << '\n';
  oss << "psf_z_blur=" << (args.psf.z_blur ? 1 : 0) << '\n';
  oss << "psf_vox=" << args.psf.vox_x << ',' << args.psf.vox_y << ','
      << args.psf.vox_z << '\n';
  oss << "psf_truncation_mm=" << args.psf.truncation_mm << '\n';
  oss << "psf_z_dependent=" << (args.psf.z_dependent ? 1 : 0) << '\n';
  oss << "psf_interp=" << psfInterpKey(args.psf.interp) << '\n';
  oss << "psf_min_sigmas=" << args.psf.min_sigmas << '\n';
  if (args.psf.mode == spect::filters::PSFMode::Operator &&
      bedZStarts.size() > 1) {
    oss << "psf_multibed_scope=per_bed_output_hardcut\n";
  }
  if (psfKernels != nullptr) {
    appendVectorLine(oss, "psf_kernel_x", psfKernels->kx);
    appendVectorLine(oss, "psf_kernel_y", psfKernels->ky);
    appendVectorLine(oss, "psf_kernel_z", psfKernels->kz);
  } else {
    oss << "psf_kernel_x=\npsf_kernel_y=\npsf_kernel_z=\n";
  }
  return oss.str();
}

bool tryLoadCompactDenomCache(const std::filesystem::path &cacheMhdPath,
                              const std::string &expectedManifest,
                              const OutputGeometry &geometry,
                              std::vector<double> &denom,
                              std::string *status) {
  const auto manifestPath = compactDenomManifestPath(cacheMhdPath);
  if (!std::filesystem::exists(cacheMhdPath) ||
      !std::filesystem::exists(manifestPath)) {
    if (status)
      *status = "cache files not found";
    return false;
  }

  std::ifstream manifestIn(manifestPath, std::ios::binary);
  if (!manifestIn) {
    if (status)
      *status = "failed to open cache manifest";
    return false;
  }
  std::ostringstream actualManifest;
  actualManifest << manifestIn.rdbuf();
  if (actualManifest.str() != expectedManifest) {
    if (status)
      *status = summarizeManifestMismatch(expectedManifest,
                                          actualManifest.str());
    return false;
  }

  try {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    std::vector<double> loaded = spect::io::loadVolumeWithSpacing(
        cacheMhdPath.string(), nx, ny, nz, dx, dy, dz);
    const auto closeEnough = [](double a, double b) {
      return std::abs(a - b) <= 1e-9;
    };
    if (nx != geometry.nx || ny != geometry.ny || nz != geometry.nz ||
        !closeEnough(dx, geometry.vx) || !closeEnough(dy, geometry.vy) ||
        !closeEnough(dz, geometry.vz) || loaded.size() != denom.size()) {
      if (status)
        *status = "cache volume geometry mismatch";
      return false;
    }
    const VectorStats stats = computeVectorStats(loaded);
    if (stats.count == 0 || stats.finite != stats.count || stats.nonzero == 0 ||
        !(stats.sum > 0.0)) {
      if (status)
        *status = "cache volume stats invalid";
      return false;
    }
    denom.swap(loaded);
    if (status)
      *status = "cache hit";
    return true;
  } catch (const std::exception &ex) {
    if (status)
      *status = ex.what();
    return false;
  }
}

void saveCompactDenomCache(const std::filesystem::path &cacheMhdPath,
                           const std::string &manifest,
                           const OutputGeometry &geometry,
                           const std::vector<double> &denom) {
  std::error_code ec;
  const auto parent = cacheMhdPath.parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent, ec);
  spect::io::writeMHD(cacheMhdPath.string(), denom, geometry.nx, geometry.ny,
                      geometry.nz, static_cast<float>(geometry.vx),
                      static_cast<float>(geometry.vy),
                      static_cast<float>(geometry.vz), true);
  const auto manifestPath = compactDenomManifestPath(cacheMhdPath);
  std::ofstream manifestOut(manifestPath, std::ios::binary);
  if (!manifestOut)
    throw std::runtime_error("Failed writing compact denom cache manifest: " +
                             manifestPath.string());
  manifestOut << manifest;
  if (!manifestOut)
    throw std::runtime_error("Failed writing compact denom cache manifest: " +
                             manifestPath.string());
}

template <typename MatrixType>
std::vector<double> runCompactSymmetricReconstructionImpl(
    const CompactSymmetricReconLoopContext &ctx, const MatrixType &matrix) {
  const ReconArgs &args = *ctx.args;
  const ReconOpts &opts = *ctx.opts;
  const ReconPreparedParams &prepared = *ctx.prepared;
  const std::vector<double> &projections = *ctx.projections;
  const std::vector<double> &backgroundProj = *ctx.backgroundProj;
  const std::vector<uint8_t> *maskPtr = ctx.mask;
  const filters::PSFKernels *psfPtr = ctx.psfKernels;
  const filters::Spacing3 &spacing = ctx.spacing;
  const OutputGeometry &outputGeometry = ctx.outputGeometry;
  const CropSettings &crop = ctx.crop;
  const OutputTransform *outputTransformPtr = ctx.outputTransform;
  const spect::io::CompactSymmetryMetadata &compactMeta = *ctx.compactMeta;
  const std::string &compactMetaPath = *ctx.compactMetaPath;
  const bool compactChunkLayout = ctx.compactChunkLayout;
  const spect::io::CompactChunkCSRDirectory *compactChunks = ctx.compactChunks;
  CompactExecutionTiming &compactTiming = *ctx.timing;
  const auto loadToIterationStart = ctx.loadToIterationStart;
  const auto fullCommandStart = ctx.fullCommandStart;
  const int bedCount = ctx.bedCount;
  const int baseNx = ctx.baseNx;
  const int baseNy = ctx.baseNy;
  const int baseNz = ctx.baseNz;
  const size_t baseNvox = ctx.baseNvox;
  const size_t globalNvox = ctx.globalNvox;
  const std::vector<int> &bedZStarts = *ctx.bedZStarts;
  const bool bedVirtualZFlip =
      args.zBedFlip && !args.projPaths.empty() && bedCount > 1;
  spect::io::CSRView compactMatrix{};
  if (!compactChunkLayout) {
          if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
            compactMatrix = matrix;
          } else {
            compactMatrix.rows = matrix.rows;
            compactMatrix.cols = matrix.cols;
            compactMatrix.row_ptr = matrix.row_ptr.data();
            compactMatrix.col_idx = matrix.col_idx.data();
            compactMatrix.val = matrix.val.data();
            compactMatrix.nnz = matrix.val.size();
          }
        }

#if !defined(SPECT_ENABLE_CUDA_SPARSEOPS)
        throw std::runtime_error(
            "compact-symmetric layout requires a CUDA-enabled build.");
#else
        const OutputGeometry geometry = outputGeometry;
        const std::string outputPrefix = outputPrefixFromArgs(args);
        std::vector<const uint16_t *> compactAttFactorPtrs;
        if (!opts.bedAttenFactors.empty()) {
          if (!compactChunkLayout) {
            throw std::runtime_error(
                "compact-symmetric CTAC requires the chunk directory layout.");
          }
          if (opts.bedAttenFactors.size() != static_cast<size_t>(bedCount)) {
            throw std::runtime_error(
                "compact-symmetric CTAC attenuation table count does not "
                "match the bed count.");
          }
          compactAttFactorPtrs.reserve(static_cast<size_t>(bedCount));
          for (const auto &factors : opts.bedAttenFactors) {
            compactAttFactorPtrs.push_back(factors.empty() ? nullptr
                                                           : factors.data());
          }
          logInfo("compact CTAC: per-bed attenuation factors fused in native "
                  "CUDA forward/backward/sensitivity kernels.");
        }
        const bool compactHasAttenuation = !compactAttFactorPtrs.empty();
        std::vector<double> denom(globalNvox, 0.0);
        const bool compactPsfActive = needs_psf_blur(opts, psfPtr);
        spect::cuda::CompactSymmetricStageTiming compactCudaTiming;
        MemoryUsageSnapshot compactPeakMemory;
        auto observeCompactMemory = [&]() {
          mergeMemoryPeak(compactPeakMemory, queryMemoryUsageSnapshot());
        };
        observeCompactMemory();
        const auto sensStart = ProfileClock::now();
        std::filesystem::path compactDenomCachePath =
            resolveCompactDenomCacheMhdPath(args);
        std::string compactDenomManifest;
        if (!compactDenomCachePath.empty() ||
            (compactChunkLayout && !compactHasAttenuation)) {
          compactDenomManifest = buildCompactDenomCacheManifest(
              args, geometry, compactMeta, compactMetaPath, compactChunkLayout,
              compactChunks, bedZStarts, psfPtr);
        }
        if (compactDenomCachePath.empty() && compactChunkLayout &&
            !compactHasAttenuation) {
          compactDenomCachePath =
              autoCompactDenomCachePath(compactDenomManifest);
          logDebug("compact denom auto-cache path: " +
                   compactDenomCachePath.string());
        }
        bool loadedCompactDenomCache = false;
        if (!compactDenomCachePath.empty()) {
          std::string cacheStatus;
          loadedCompactDenomCache = tryLoadCompactDenomCache(
              compactDenomCachePath, compactDenomManifest, geometry, denom,
              &cacheStatus);
          if (loadedCompactDenomCache) {
            logInfo("compact denom cache hit: " +
                    compactDenomCachePath.string());
            if (compactChunkLayout) {
              spect::cuda::prepareCompactChunkExecutionCUDA(*compactChunks);
            }
          } else {
            if (isCompactDenomManifestMismatch(cacheStatus)) {
              throw std::runtime_error(
                  "compact denom cache manifest mismatch for " +
                  compactDenomCachePath.string() + ": " + cacheStatus +
                  ". Use a distinct -compactDenomCache path for no-PSF, "
                  "PSF/FWHM, and bed-geometry variants, or delete the "
                  "stale cache intentionally.");
            }
            logInfo("compact denom cache miss: " + cacheStatus);
          }
        }
        if (!loadedCompactDenomCache) {
          std::vector<double> localDenom(baseNvox, 0.0);
          // Weighted sensitivity A^T w (opt-in, compact chunk layout only; CLI
          // validation enforces this). w starts at 1; detZ edge-row censoring
          // zeroes the censored detZ edge rows, and an efficiency-weights file
          // multiplies in per-row, per-detector-bin, or full projection-bin
          // factors. Hardware detector masking then zeroes complete
          // angle/block row sets; all zero masks win.
          // Computed
          // without new CUDA kernels via the numerator identity
          // A^T (y*/(A x*)) with x* = 1 and y* = w . (A 1); rows where
          // A 1 == 0 are zero in y* and contribute nothing via the ratio
          // guard. Measured bins are never scaled here: censoring zeroes them
          // in MainProgram, and the efficiency weights are denominator-only.
          // When the final w is all ones the plain sensitivity path below is
          // kept, so the default stays bit-identical.
          const int detZEdgeCensor = compactChunkLayout ? args.detZEdgeCensor : 0;
          if (!compactChunkLayout &&
              (!args.detZRowWeightsFile.empty() ||
               !args.normFactorFile.empty())) {
            throw std::runtime_error(
                "normalization weights require the compact-symmetric chunk "
                "layout.");
          }
          std::vector<double> detZRowWeights;
          if (!args.normFactorFile.empty()) {
            detZRowWeights = spect::io::loadDetZRowWeights(
                args.normFactorFile, compactMeta.detNz, compactMeta.detNx,
                compactMeta.fullNumAngles);
            if (detZRowWeights.size() !=
                static_cast<size_t>(compactMeta.fullRows())) {
              throw std::runtime_error(
                  "-normFactorFile must contain one correction factor per "
                  "projection bin.");
            }
            for (double &correction : detZRowWeights)
              correction = 1.0 / correction;
            logInfo("normalization correction factors: loaded " +
                    std::to_string(detZRowWeights.size()) + " values from " +
                    args.normFactorFile +
                    "; applying reciprocal efficiency 1/c to the "
                    "sensitivity (denominator) only.");
          } else if (!args.detZRowWeightsFile.empty()) {
            detZRowWeights = spect::io::loadDetZRowWeights(
                args.detZRowWeightsFile, compactMeta.detNz, compactMeta.detNx,
                compactMeta.fullNumAngles);
            logInfo("detZ efficiency weights: loaded " +
                    std::to_string(detZRowWeights.size()) + " values from " +
                    args.detZRowWeightsFile +
                    "; applying to the sensitivity (denominator) only.");
          }
          std::vector<spect::io::BadDetectorId> badDetectors;
          std::vector<uint8_t> badDetectorMask;
          if (!args.badDetectorFile.empty()) {
            badDetectors =
                spect::io::loadBadDetectorList(args.badDetectorFile);
            badDetectorMask = spect::io::buildBadDetectorProjectionMask(
                badDetectors, compactMeta.fullNumAngles, compactMeta.detNz,
                compactMeta.detNx);
          }
          std::vector<double> detZSensWeights;
          if (detZEdgeCensor > 0 || !detZRowWeights.empty() ||
              !badDetectorMask.empty()) {
            detZSensWeights.assign(
                static_cast<size_t>(compactMeta.fullRows()), 1.0);
            const size_t raysPerAngle =
                static_cast<size_t>(compactMeta.fullRaysPerAngle());
            const bool perBinWeights =
                detZRowWeights.size() ==
                static_cast<size_t>(compactMeta.detNz) *
                    static_cast<size_t>(compactMeta.detNx);
            const bool perProjectionWeights =
                detZRowWeights.size() ==
                static_cast<size_t>(compactMeta.fullRows());
            for (int angle = 0; angle < compactMeta.fullNumAngles; ++angle) {
              double *angleBase = detZSensWeights.data() +
                                  static_cast<size_t>(angle) * raysPerAngle;
              for (int detZ = 0; detZ < compactMeta.detNz; ++detZ) {
                double *row = angleBase +
                              static_cast<size_t>(detZ) * compactMeta.detNx;
                if (detZ < detZEdgeCensor ||
                    detZ >= compactMeta.detNz - detZEdgeCensor) {
                  std::fill(row, row + compactMeta.detNx, 0.0);
                  continue;
                }
                if (!detZRowWeights.empty()) {
                  for (int detX = 0; detX < compactMeta.detNx; ++detX) {
                    const size_t wIndex =
                        perProjectionWeights
                            ? static_cast<size_t>(angle) * raysPerAngle +
                                  static_cast<size_t>(detZ) *
                                      compactMeta.detNx +
                                  static_cast<size_t>(detX)
                        : perBinWeights
                            ? static_cast<size_t>(detZ) * compactMeta.detNx +
                                  static_cast<size_t>(detX)
                            : static_cast<size_t>(detZ);
                    row[detX] *= detZRowWeights[wIndex];
                  }
                }
              }
            }
            if (!badDetectorMask.empty()) {
              for (size_t row = 0; row < badDetectorMask.size(); ++row) {
                if (badDetectorMask[row])
                  detZSensWeights[row] = 0.0;
              }
              logInfo("Bad detector mask: removed " +
                      std::to_string(badDetectors.size()) +
                      " detector blocks from the sensitivity denominator.");
            }
          }
          const bool useWeightedSensitivity =
              !detZSensWeights.empty() &&
              !std::all_of(detZSensWeights.begin(), detZSensWeights.end(),
                           [](double w) { return w == 1.0; });
          std::vector<double> detZSensOnes;
          std::vector<double> detZSensFwd;
          if (useWeightedSensitivity) {
            detZSensOnes.assign(baseNvox, 1.0);
            detZSensFwd.assign(
                static_cast<size_t>(compactMeta.fullRows()), 0.0);
            if (detZEdgeCensor > 0) {
              logInfo("detZ edge censoring: using masked sensitivity (k=" +
                      std::to_string(detZEdgeCensor) + ", " +
                      std::to_string(detZEdgeCensor) +
                      " detZ rows censored per side) for the denominator.");
            }
          }
          auto computeLocalSensitivity = [&](const uint16_t *attenuation) {
            std::fill(localDenom.begin(), localDenom.end(), 0.0);
            if (useWeightedSensitivity) {
              if (!spect::cuda::computeCompactChunkForwardProjectionCUDA(
                      *compactChunks, compactMeta, detZSensOnes.data(),
                      opts.threads, detZSensFwd.data(), attenuation)) {
                throw std::runtime_error(
                    "detZ weighted sensitivity requires CUDA sparse backend.");
              }
              for (size_t row = 0; row < detZSensFwd.size(); ++row) {
                detZSensFwd[row] *= detZSensWeights[row];
              }
              if (!spect::cuda::runCompactChunkMlemNumerCUDA(
                      *compactChunks, compactMeta, detZSensFwd.data(),
                      nullptr, detZSensOnes.data(), opts.threads,
                      localDenom.data(), false, &compactCudaTiming,
                      attenuation)) {
                throw std::runtime_error(
                    "detZ weighted sensitivity requires CUDA sparse backend.");
              }
              return;
            }
            const bool ok =
                compactChunkLayout
                    ? spect::cuda::computeCompactChunkSensitivityCUDA(
                          *compactChunks, compactMeta, opts.threads,
                          localDenom.data(), &compactCudaTiming, attenuation)
                    : spect::cuda::computeCompactSymmetricSensitivityCUDA(
                          compactMatrix, compactMeta, opts.threads,
                          localDenom.data(), &compactCudaTiming);
            if (!ok) {
              throw std::runtime_error(
                  "compact-symmetric sensitivity requires CUDA sparse backend.");
            }
          };
          if (compactHasAttenuation) {
            for (int bed = 0; bed < bedCount; ++bed) {
              computeLocalSensitivity(
                  compactAttFactorPtrs[static_cast<size_t>(bed)]);
              apply_psf_backward(localDenom, baseNx, baseNy, baseNz, opts,
                                 psfPtr, opts.threads);
              spect::app::accumulateBedVolumeToGlobal(
                  denom, localDenom, baseNx, baseNy, baseNz,
                  bedZStarts[static_cast<size_t>(bed)], bedVirtualZFlip);
              observeCompactMemory();
            }
          } else {
            computeLocalSensitivity(nullptr);
            apply_psf_backward(localDenom, baseNx, baseNy, baseNz, opts,
                               psfPtr, opts.threads);
            for (int bed = 0; bed < bedCount; ++bed) {
              spect::app::accumulateBedVolumeToGlobal(
                  denom, localDenom, baseNx, baseNy, baseNz,
                  bedZStarts[static_cast<size_t>(bed)], bedVirtualZFlip);
              observeCompactMemory();
            }
          }

          if (!compactDenomCachePath.empty()) {
            try {
              saveCompactDenomCache(compactDenomCachePath, compactDenomManifest,
                                    geometry, denom);
              logInfo("compact denom cache saved: " +
                      compactDenomCachePath.string());
            } catch (const std::exception &ex) {
              logWarn(std::string("Failed saving compact denom cache: ") +
                      ex.what());
            }
          }
        }
        compactTiming.compactSensitivitySeconds +=
            std::chrono::duration<double>(ProfileClock::now() - sensStart)
                .count();
        const VectorStats denomStats = computeVectorStats(denom);
        logDebug(formatStatsEntry("compact_denom", denomStats));
        if (denomStats.count == 0 || denomStats.finite != denomStats.count ||
            denomStats.nonzero == 0 || !(denomStats.sum > 0.0)) {
          throw std::runtime_error(
              "compact-symmetric denominator is empty, non-finite, or zero.");
        }
        const VectorStats projectionStats = computeVectorStats(projections);
        logDebug(formatStatsEntry("compact_projection", projectionStats));
        const bool compactPositiveProjectionInput =
            projectionStats.count > 0 &&
            projectionStats.finite == projectionStats.count &&
            projectionStats.nonzero > 0 && projectionStats.sum > 0.0;

        std::vector<double> image(globalNvox, 1.0);
        std::vector<double> gradReg(globalNvox, 0.0);
        std::vector<double> curvMap;
          std::vector<double> kappaMap =
            (args.algorithm == ReconArgs::Algorithm::BSREM)
              ? computeKappaMapStandard(denom, args, maskPtr)
                : std::vector<double>();
        const bool useRegularization =
            args.algorithm == ReconArgs::Algorithm::BSREM && args.useReg &&
            args.reg.beta > 0.0 && args.reg.type != RegType::None;
        const bool useBsremUpdate =
            args.algorithm == ReconArgs::Algorithm::BSREM;
        const std::vector<uint8_t> *maskForReg =
            (maskPtr && useRegularization && args.reg.skip_outside_mask)
                ? maskPtr
                : nullptr;
        bool compactChunkReverseOrderNext =
            compactChunkLayout && !loadedCompactDenomCache;
        auto lastIterationDone = ProfileClock::now();

        for (int it = 1; it <= args.numIters; ++it) {
          logInfo("[ITER] " + std::to_string(it) + "/" +
                  std::to_string(args.numIters) + " compact");
          std::vector<double> numer(globalNvox, 0.0);
          const std::vector<double> *bedImageSource = &image;

          if (compactChunkLayout) {
            std::vector<std::vector<double>> localImages(
                static_cast<size_t>(bedCount));
            std::vector<std::vector<double>> localUsed(
                static_cast<size_t>(bedCount));
            std::vector<std::vector<double>> localNumers(
                static_cast<size_t>(bedCount),
                std::vector<double>(baseNvox, 0.0));
            std::vector<const double *> projectionPtrs(
                static_cast<size_t>(bedCount), nullptr);
            std::vector<const double *> backgroundPtrs(
                static_cast<size_t>(bedCount), nullptr);
            std::vector<const double *> imagePtrs(static_cast<size_t>(bedCount),
                                                  nullptr);
            std::vector<double *> numerPtrs(static_cast<size_t>(bedCount),
                                            nullptr);

            for (int bed = 0; bed < bedCount; ++bed) {
              const int zStart = bedZStarts[static_cast<size_t>(bed)];
              localImages[static_cast<size_t>(bed)] =
                  spect::app::extractBedVolumeFromGlobal(
                      *bedImageSource, baseNx, baseNy, baseNz, zStart,
                      bedVirtualZFlip);
              imagePtrs[static_cast<size_t>(bed)] =
                  localImages[static_cast<size_t>(bed)].data();
              if (compactPsfActive) {
                logDebug(formatStatsEntry(
                    "compact_local_image.it" + std::to_string(it) + ".bed" +
                        std::to_string(bed),
                    computeVectorStats(localImages[static_cast<size_t>(bed)])));
                apply_psf_forward(localImages[static_cast<size_t>(bed)],
                                  localUsed[static_cast<size_t>(bed)], baseNx,
                                  baseNy, baseNz, opts, psfPtr, opts.threads);
                logDebug(formatStatsEntry(
                    "compact_psf_forward.it" + std::to_string(it) + ".bed" +
                        std::to_string(bed),
                    computeVectorStats(localUsed[static_cast<size_t>(bed)])));
                imagePtrs[static_cast<size_t>(bed)] =
                    localUsed[static_cast<size_t>(bed)].data();
              }

              const size_t projectionOffset =
                  static_cast<size_t>(bed) *
                  static_cast<size_t>(compactMeta.fullRows());
              projectionPtrs[static_cast<size_t>(bed)] =
                  projections.data() + projectionOffset;
              if (!backgroundProj.empty()) {
                backgroundPtrs[static_cast<size_t>(bed)] =
                    backgroundProj.data() + projectionOffset;
              }
              numerPtrs[static_cast<size_t>(bed)] =
                  localNumers[static_cast<size_t>(bed)].data();
            }

            const auto numerStart = ProfileClock::now();
            const bool ok = spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
                *compactChunks, compactMeta, projectionPtrs.data(),
                backgroundProj.empty() ? nullptr : backgroundPtrs.data(),
                imagePtrs.data(), bedCount, opts.threads, numerPtrs.data(),
                compactChunkReverseOrderNext, &compactCudaTiming,
                compactAttFactorPtrs.empty() ? nullptr
                                             : compactAttFactorPtrs.data());
            if (!ok) {
              throw std::runtime_error(
                  "compact-symmetric numerator requires CUDA sparse backend.");
            }
            compactChunkReverseOrderNext = !compactChunkReverseOrderNext;
            compactTiming.compactForwardRatioBackwardSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - numerStart)
                    .count();

            for (int bed = 0; bed < bedCount; ++bed) {
              const int zStart = bedZStarts[static_cast<size_t>(bed)];
              std::vector<double> &localNumer =
                  localNumers[static_cast<size_t>(bed)];
              if (compactPsfActive) {
                logDebug(formatStatsEntry(
                    "compact_local_numer_pre_psf.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localNumer)));
              }
              apply_psf_backward(localNumer, baseNx, baseNy, baseNz, opts,
                                 psfPtr, opts.threads);
              if (compactPsfActive) {
                logDebug(formatStatsEntry(
                    "compact_local_numer_post_psf.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localNumer)));
              }
              spect::app::accumulateBedVolumeToGlobal(
                  numer, localNumer, baseNx, baseNy, baseNz, zStart,
                  bedVirtualZFlip);
            }
          } else {
            for (int bed = 0; bed < bedCount; ++bed) {
              const int zStart = bedZStarts[static_cast<size_t>(bed)];
              std::vector<double> localImage =
                  spect::app::extractBedVolumeFromGlobal(
                      *bedImageSource, baseNx, baseNy, baseNz, zStart,
                      bedVirtualZFlip);
              std::vector<double> localUsed;
              const double *localImageForProjection = localImage.data();
              if (compactPsfActive) {
                logDebug(formatStatsEntry(
                    "compact_local_image.it" + std::to_string(it) + ".bed" +
                        std::to_string(bed),
                    computeVectorStats(localImage)));
                apply_psf_forward(localImage, localUsed, baseNx, baseNy, baseNz,
                                  opts, psfPtr, opts.threads);
                logDebug(formatStatsEntry(
                    "compact_psf_forward.it" + std::to_string(it) + ".bed" +
                        std::to_string(bed),
                    computeVectorStats(localUsed)));
                localImageForProjection = localUsed.data();
              }

              std::vector<double> localNumer(baseNvox, 0.0);
              const size_t projectionOffset =
                  static_cast<size_t>(bed) *
                  static_cast<size_t>(compactMeta.fullRows());
              const double *localBackground =
                  backgroundProj.empty() ? nullptr
                                         : backgroundProj.data() +
                                               projectionOffset;
              const auto numerStart = ProfileClock::now();
              const bool ok = spect::cuda::runCompactSymmetricMlemNumerCUDA(
                  compactMatrix, compactMeta, projections.data() + projectionOffset,
                  localBackground, localImageForProjection, opts.threads,
                  localNumer.data(), &compactCudaTiming);
              if (!ok) {
                throw std::runtime_error(
                    "compact-symmetric numerator requires CUDA sparse backend.");
              }
              compactTiming.compactForwardRatioBackwardSeconds +=
                  std::chrono::duration<double>(ProfileClock::now() -
                                                numerStart)
                      .count();

              if (compactPsfActive) {
                logDebug(formatStatsEntry(
                    "compact_local_numer_pre_psf.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localNumer)));
              }
              apply_psf_backward(localNumer, baseNx, baseNy, baseNz, opts,
                                 psfPtr, opts.threads);
              if (compactPsfActive) {
                logDebug(formatStatsEntry(
                    "compact_local_numer_post_psf.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localNumer)));
              }
              spect::app::accumulateBedVolumeToGlobal(
                  numer, localNumer, baseNx, baseNy, baseNz, zStart,
                  bedVirtualZFlip);
            }
          }
          const VectorStats numerStats = computeVectorStats(numer);
          logDebug(formatStatsEntry("compact_numer.it" + std::to_string(it),
                                   numerStats));
          if (numerStats.count == 0 || numerStats.finite != numerStats.count ||
              (compactPositiveProjectionInput &&
               (numerStats.nonzero == 0 || !(numerStats.sum > 0.0)))) {
            throw std::runtime_error(
                "compact-symmetric numerator is empty, non-finite, or zero "
                "despite positive projection input.");
          }

          const std::vector<double> *gradPtr = nullptr;
          if (useRegularization) {
            const auto regStart = ProfileClock::now();
            if (!tryComputeRegGradientCUDA(image, args, opts, maskForReg,
                                           gradReg)) {
              throw std::runtime_error(
                  "Regularization gradient requires CUDA volume helpers.");
            }
            compactTiming.regGradientSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - regStart)
                    .count();
            ++compactTiming.regGradientCalls;
            gradPtr = &gradReg;
          }

          double bsremAlpha = 1.0;
          double bsremBeta = 0.0;
          const std::vector<double> *curvPtr = nullptr;
          const std::vector<double> *kappaPtr = nullptr;
          if (useBsremUpdate) {
            const int k = it - 1;
            bsremAlpha =
                args.bsrem_alpha0 /
                (1.0 + static_cast<double>(args.bsrem_gamma) * k);
            bsremBeta = useRegularization ? args.reg.beta : 0.0;
            if (args.bsremUseCurvDenom) {
              const bool refreshCurvature =
                  curvMap.empty() || !args.bsremFreezeCurv ||
                  it <= std::max(1, args.bsremCurvWarmupIters);
              if (refreshCurvature) {
                curvMap.assign(globalNvox, 0.0);
                if (gradPtr) {
                  for (size_t i = 0; i < globalNvox; ++i) {
                    const double denomBase =
                        std::max(image[i], args.bsremCurvEps);
                    curvMap[i] = std::abs(gradReg[i]) / denomBase;
                  }
                }
              }
              curvPtr = &curvMap;
            }
            kappaPtr = args.bsremUseKappaMap ? &kappaMap : nullptr;
          }

          const auto updateStart = ProfileClock::now();
          constexpr double eps = 1e-12;
          if (!useBsremUpdate) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(opts.threads)
#endif
            for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(globalNvox); ++j) {
              const size_t idx = static_cast<size_t>(j);
              if (maskPtr && !(*maskPtr)[idx]) {
                image[idx] = 0.0;
                continue;
              }
              const double denomEff = denom[idx];
              if (!(denomEff > 0.0)) {
                image[idx] = 0.0;
                continue;
              }
              double value = image[idx] * (numer[idx] / std::max(denomEff, eps));
              if (opts.nonneg && value < 0.0)
                value = 0.0;
              image[idx] = value;
            }
          } else {
            BsremUpdateParams updateParams;
            updateParams.mask = maskPtr;
            updateParams.gradient = gradPtr;
            updateParams.curvature = curvPtr;
            updateParams.kappa = kappaPtr;
            updateParams.betaScaled = bsremBeta;
            updateParams.alpha = bsremAlpha;
            updateParams.nonneg = opts.nonneg;
            updateParams.threads = opts.threads;
            applyBsremUpdate(image, numer, denom, denom, updateParams);
          }
          compactTiming.mlemUpdateSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - updateStart)
                  .count();

          if (prepared.butterConf.enabled) {
            const auto butterStart = ProfileClock::now();
            filters::apply_butterworth_inplace(image, args.Nx, args.Ny,
                                               args.Nz, spacing,
                                               prepared.butterConf,
                                               opts.threads);
            compactTiming.butterworthSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - butterStart)
                    .count();
          }

          if (prepared.smoothConf.enabled) {
            applySeparableConvRuntime(image, args.Nx, args.Ny, args.Nz,
                                      prepared.smoothConf.kx,
                                      prepared.smoothConf.ky,
                                      prepared.smoothConf.kz, opts,
                                      opts.threads);
          }

          clampNonNegativeInPlace(image, opts.threads);
          if (maskPtr)
            spect::core::applyMaskInplace(image, *maskPtr);
          guardZeroMaxCheckpoint(
              image, "compact iteration " + std::to_string(it));
          lastIterationDone = ProfileClock::now();
          observeCompactMemory();

          if (args.saveInterval > 0 &&
              (it % args.saveInterval == 0 || it == args.numIters)) {
            const auto saveStart = ProfileClock::now();
            saveVolumeOutput(image, geometry, args.outputDir, outputPrefix, it,
                             maskPtr, crop, true, outputTransformPtr);
            compactTiming.outputWriteSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - saveStart)
                    .count();
          }
        }

        if (args.projectionResidual) {
          std::error_code outputDirError;
          std::filesystem::create_directories(args.outputDir, outputDirError);
          if (outputDirError) {
            throw std::runtime_error(
                "Failed creating projection residual output directory: " +
                outputDirError.message());
          }
          const std::filesystem::path residualPath =
              std::filesystem::path(args.outputDir) /
              (outputPrefix + "_projection_residual.csv");
          std::ofstream residualOut(residualPath, std::ios::binary);
          if (!residualOut) {
            throw std::runtime_error(
                "Failed opening projection residual CSV: " +
                residualPath.string());
          }
          residualOut
              << "bed,angle,det_z,count,measured_sum,expected_sum,relative_"
                 "bias,poisson_deviance_per_bin,pearson_rms\n";
          residualOut << std::setprecision(12);

          const std::vector<double> *diagnosticBedSource = &image;

          const size_t fullRows =
              static_cast<size_t>(compactMeta.fullRows());
          for (int bed = 0; bed < bedCount; ++bed) {
            std::vector<double> localImage =
                spect::app::extractBedVolumeFromGlobal(
                    *diagnosticBedSource, baseNx, baseNy, baseNz,
                    bedZStarts[static_cast<size_t>(bed)], bedVirtualZFlip);
            std::vector<double> localUsed;
            const double *localImageForProjection = localImage.data();
            if (compactPsfActive) {
              apply_psf_forward(localImage, localUsed, baseNx, baseNy, baseNz,
                                opts, psfPtr, opts.threads);
              localImageForProjection = localUsed.data();
            }

            std::vector<double> expected(fullRows, 0.0);
            const uint16_t *bedAttenuation =
                compactAttFactorPtrs.empty()
                    ? nullptr
                    : compactAttFactorPtrs[static_cast<size_t>(bed)];
            const bool forwardOk =
                compactChunkLayout
                    ? spect::cuda::computeCompactChunkForwardProjectionCUDA(
                          *compactChunks, compactMeta, localImageForProjection,
                          opts.threads, expected.data(), bedAttenuation)
                    : spect::cuda::computeCompactSymmetricForwardProjectionCUDA(
                          compactMatrix, compactMeta, localImageForProjection,
                          opts.threads, expected.data());
            if (!forwardOk) {
              throw std::runtime_error(
                  "compact projection residual diagnostic requires CUDA "
                  "sparse backend.");
            }

            const size_t projectionOffset =
                static_cast<size_t>(bed) * fullRows;
            const double *measured = projections.data() + projectionOffset;
            const double *background =
                backgroundProj.empty()
                    ? nullptr
                    : backgroundProj.data() + projectionOffset;
            ProjectionResidualStats bedStats;
            for (int angle = 0; angle < compactMeta.fullNumAngles; ++angle) {
              for (int detZ = 0; detZ < compactMeta.detNz; ++detZ) {
                ProjectionResidualStats rowStats;
                const size_t rowBase =
                    static_cast<size_t>(angle) *
                        static_cast<size_t>(compactMeta.fullRaysPerAngle()) +
                    static_cast<size_t>(detZ) *
                        static_cast<size_t>(compactMeta.detNx);
                for (int detX = 0; detX < compactMeta.detNx; ++detX) {
                  const size_t index = rowBase + static_cast<size_t>(detX);
                  const double totalExpected =
                      expected[index] +
                      (background == nullptr ? 0.0 : background[index]);
                  addProjectionResidualSample(rowStats, measured[index],
                                              totalExpected);
                  addProjectionResidualSample(bedStats, measured[index],
                                              totalExpected);
                }
                residualOut << bed << ',' << angle << ',' << detZ << ','
                            << rowStats.count << ',' << rowStats.measuredSum
                            << ',' << rowStats.expectedSum << ','
                            << projectionRelativeBias(rowStats) << ','
                            << projectionDeviancePerBin(rowStats) << ','
                            << projectionPearsonRms(rowStats) << '\n';
              }
            }
            std::ostringstream metric;
            metric << std::setprecision(12)
                   << "[METRIC] compact_projection_residual bed=" << bed
                   << " count=" << bedStats.count
                   << " measured_sum=" << bedStats.measuredSum
                   << " expected_sum=" << bedStats.expectedSum
                   << " relative_bias=" << projectionRelativeBias(bedStats)
                   << " poisson_deviance_per_bin="
                   << projectionDeviancePerBin(bedStats)
                   << " pearson_rms=" << projectionPearsonRms(bedStats);
            logInfo(metric.str());
          }
          residualOut.flush();
          if (!residualOut) {
            throw std::runtime_error(
                "Failed writing projection residual CSV: " +
                residualPath.string());
          }
          logInfo("compact projection residual CSV: " +
                  residualPath.string());
        }

        if (args.saveInterval <= 0 ||
            (args.numIters % args.saveInterval) != 0) {
          const auto saveStart = ProfileClock::now();
          saveVolumeOutput(image, geometry, args.outputDir, outputPrefix,
                           args.numIters, maskPtr, crop, true,
                           outputTransformPtr);
          compactTiming.outputWriteSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - saveStart)
                  .count();
          observeCompactMemory();
        }

        compactTiming.totalLoadToIterationDoneSeconds =
            std::chrono::duration<double>(lastIterationDone -
                                          loadToIterationStart)
                .count();
        compactTiming.fullCommandWallSeconds =
            std::chrono::duration<double>(ProfileClock::now() -
                                          fullCommandStart)
                .count();

        logDebug(formatProfileEntry("projection_load",
                                   compactTiming.projectionLoadSeconds, 1));
        logDebug(formatProfileEntry("csr_map_or_load",
                                   compactTiming.csrMapOrLoadSeconds, 1));
        logDebug(formatProfileEntry(
            "metadata_load_and_validate",
            compactTiming.metadataLoadAndValidateSeconds, 1));
        logDebug(formatProfileEntry("psf_prepare",
                                   compactTiming.psfPrepareSeconds, 1));
        logDebug(formatProfileEntry("compact_sensitivity",
                                   compactTiming.compactSensitivitySeconds,
                                   compactCudaTiming.sensitivityCalls));
        logDebug(formatProfileEntry(
            "compact_forward_ratio_backward",
            compactTiming.compactForwardRatioBackwardSeconds,
            compactCudaTiming.numeratorCalls));
        logDebug(formatProfileEntry("compact.transform_forward",
                                   compactCudaTiming.transformForwardSeconds,
                                   compactCudaTiming.numeratorCalls));
        logDebug(formatProfileEntry("compact.forward",
                                   compactCudaTiming.forwardSeconds,
                                   compactCudaTiming.numeratorCalls));
        logDebug(formatProfileEntry(
            "compact.ratio", compactCudaTiming.ratioSeconds,
            compactCudaTiming.numeratorCalls + compactCudaTiming.sensitivityCalls));
        logDebug(formatProfileEntry(
            "compact.backward", compactCudaTiming.backwardSeconds,
            compactCudaTiming.numeratorCalls + compactCudaTiming.sensitivityCalls));
        logDebug(formatProfileEntry(
            "compact.transform_backward",
            compactCudaTiming.transformBackwardSeconds,
            compactCudaTiming.numeratorCalls + compactCudaTiming.sensitivityCalls));
        if (compactTiming.regGradientCalls > 0) {
          logDebug(formatProfileEntry("compact.reg_gradient",
                                     compactTiming.regGradientSeconds,
                                     compactTiming.regGradientCalls));
        }
        logDebug(formatProfileEntry(useBsremUpdate
                                       ? "bsrem_update"
                                       : "mlem_update",
                                   compactTiming.mlemUpdateSeconds,
                                   static_cast<size_t>(args.numIters)));
        logDebug(formatProfileEntry("butterworth",
                                   compactTiming.butterworthSeconds,
                                   prepared.butterConf.enabled
                                       ? static_cast<size_t>(args.numIters)
                                       : 0));
        logDebug(formatProfileEntry("output_write",
                                   compactTiming.outputWriteSeconds, 1));
        logDebug(formatProfileEntry(
            "total_load_to_iteration_done",
            compactTiming.totalLoadToIterationDoneSeconds, 1));
        logDebug(formatProfileEntry("full_command_wall_time",
                                   compactTiming.fullCommandWallSeconds, 1));
        const spect::cuda::CudaChunkStageTimingSnapshot cudaChunkTiming =
            spect::cuda::getCudaChunkStageTimingSnapshotCUDA();
        logCudaChunkTimingSnapshot(cudaChunkTiming);
        logDebug(formatMemoryEntry("compact_peak_memory", compactPeakMemory));
        return image;
#endif
}

} // namespace

CompactSymmetricReconLoop::CompactSymmetricReconLoop(
    CompactSymmetricReconLoopContext context)
    : context_(context) {}

std::vector<double> CompactSymmetricReconLoop::run(
    const io::CSR &matrix) const {
  return runCompactSymmetricReconstructionImpl(context_, matrix);
}

std::vector<double> CompactSymmetricReconLoop::run(
    const io::CSRView &matrix) const {
  return runCompactSymmetricReconstructionImpl(context_, matrix);
}

} // namespace spect::main
