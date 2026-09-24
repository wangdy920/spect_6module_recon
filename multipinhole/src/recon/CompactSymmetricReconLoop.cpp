#include "recon/CompactSymmetricReconLoop.h"

#include "recon/BedOffsets.h"
#include "recon/BsremUpdate.h"
#include "recon/ProjectionOps.h"
#include "core/AttenuationCoefficients.h"
#include "core/Masks.h"
#include "core/ReconLimits.h"
#include "sparse/CompactSubsets.h"
#include "sparse/CompactSymmetricProjector.h"
#include "sparse/SparseOpsCUDA.h"
#include "filters/ButterworthFilter.h"
#include "filters/cuda/VolumeOpsCUDA.h"
#include "io/BadDetectorMask.h"
#include "io/MHDWriter.h"
#include "io/DetZRowWeights.h"
#include "io/VolumeLoader.h"
#include "recon/BsremDiagnostics.h"
#include "recon/BsremKappaMap.h"
#include "recon/ReconLoopHelpers.h"
#include "pipeline/RunLogger.h"
#include "pipeline/RunManifest.h"
#include "postprocess/SmoothingOps.h"
#include "pipeline/VolumeOutput.h"
#include "regularizer/RegularizerCompat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
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
    return "none";
  case filters::PSFMode::Operator:
    return "operator";
  case filters::PSFMode::Sieve:
    return "sieve";
  }
  return "unknown";
}

std::string psfKernelKey(filters::PSFKernel kernel) {
  return kernel == filters::PSFKernel::Castor ? "castor" : "adjoint";
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

void appendFileStatIdentity(std::ostringstream &oss, const std::string &label,
                            const std::filesystem::path &path) {
  const auto normalized = normalizePathForCache(path);
  // Package roots are deployment details, not denominator inputs.
  oss << label << "_name=" << normalized.filename().string() << '\n';
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

// FNV-1a over 8-byte words (tail bytes folded in), for large numeric tables
// where the byte-wise loop above would take seconds per gigabyte.
uint64_t fnv1a64Words(const void *data, size_t bytes) {
  const auto *ptr = static_cast<const unsigned char *>(data);
  uint64_t value = 14695981039346656037ull;
  size_t offset = 0;
  for (; offset + sizeof(uint64_t) <= bytes; offset += sizeof(uint64_t)) {
    uint64_t word = 0;
    std::memcpy(&word, ptr + offset, sizeof(word));
    value ^= word;
    value *= 1099511628211ull;
  }
  for (; offset < bytes; ++offset) {
    value ^= static_cast<uint64_t>(ptr[offset]);
    value *= 1099511628211ull;
  }
  return value;
}

std::filesystem::path autoCompactDenomCachePath(const ReconArgs &args,
                                                const std::string &manifest) {
  std::ostringstream name;
  name << "auto_" << std::hex << fnv1a64(manifest) << ".mhd";
  const std::filesystem::path dir = args.compactDenomCacheDir.empty()
                                        ? std::filesystem::path("recon_out") /
                                              "compact_denom_cache"
                                        : args.compactDenomCacheDir;
  return normalizePathForCache(dir / name.str());
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

void appendChunkDirectoryIdentity(
    std::ostringstream &oss, const std::string &prefix,
    const spect::io::CompactChunkCSRDirectory &chunks) {
  oss << prefix << "chunk_count=" << chunks.chunks().size() << '\n';
  oss << prefix << "chunk_total_rows=" << chunks.totalRows() << '\n';
  oss << prefix << "chunk_total_nnz=" << chunks.totalNnz() << '\n';
  for (size_t i = 0; i < chunks.chunks().size(); ++i) {
    const auto &chunk = chunks.chunks()[i];
    const std::string chunkPrefix = prefix + "chunk_" + std::to_string(i);
    appendFileStatIdentity(oss, chunkPrefix, chunk.path);
    oss << chunkPrefix << "_rows=" << chunk.rows << '\n';
    oss << chunkPrefix << "_angle_offset=" << chunk.angleOffset << '\n';
    oss << chunkPrefix << "_angle_count=" << chunk.angleCount << '\n';
    oss << chunkPrefix << "_nnz=" << chunk.nnz << '\n';
    oss << chunkPrefix << "_file_size_bytes=" << chunk.fileSizeBytes << '\n';
  }
}

std::string buildCompactDenomCacheManifest(
    const ReconArgs &args, const OutputGeometry &geometry,
    const spect::io::CompactSymmetryMetadata &compactMeta,
    const std::string &compactMetaPath,
    const spect::io::CompactChunkCSRDirectory &compactChunks,
    const std::vector<int> &bedZStarts,
    const spect::filters::PSFKernels *psfKernels,
    const spect::io::CompactChunkCSRDirectory *compactChunks2 = nullptr,
    const std::vector<double> *sensRowWeights = nullptr,
    const std::vector<std::vector<uint16_t>> *bedAttenFactors = nullptr,
    const std::vector<double> *ctacAttenPowers = nullptr) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << "cache_schema=compact_global_denom_v4_portable_stat_identity\n";
  oss << "matrix_layout=compact-symmetric\n";
  oss << "matrix_source_kind=chunkCSRDir\n";
  oss << "matrix_profile="
      << normalizePathForCache(compactChunks.directory()).filename().string()
      << '\n';
  appendFileStatIdentity(oss, "compact_meta", compactMetaPath);
  oss << "compact_mode="
      << spect::io::compactSymmetryModeName(compactMeta) << '\n';
  oss << "compact_dims=" << compactMeta.Nx << ',' << compactMeta.Ny << ','
      << compactMeta.Nz << '\n';
  oss << "compact_voxel_size=" << compactMeta.voxelSize << '\n';
  oss << "compact_det=" << compactMeta.detNz << ',' << compactMeta.detNx
      << '\n';
  oss << "compact_angles=" << compactMeta.fullNumAngles << '\n';
  oss << "compact_rows=" << compactMeta.compactRows() << '\n';
  oss << "full_rows=" << compactMeta.fullRows() << '\n';
  oss << "matrix_cols=" << compactMeta.totalVoxels << '\n';
  appendChunkDirectoryIdentity(oss, "", compactChunks);
  // Joint dual-window: the sensitivity is sum_k w_k A_k^T 1, so the second
  // matrix and both weights are part of the identity. Emitted only when
  // active so single-window caches keep matching their stored manifests.
  if (compactChunks2 != nullptr) {
    oss << "dual_window=1\n";
    oss << "matrix2_profile="
        << normalizePathForCache(compactChunks2->directory()).filename().string()
        << '\n';
    appendChunkDirectoryIdentity(oss, "matrix2_", *compactChunks2);
    oss << "energy_window_weights=";
    for (size_t k = 0; k < args.energyWindowWeights.size(); ++k)
      oss << (k ? "," : "") << args.energyWindowWeights[k];
    oss << '\n';
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
      appendFileStatIdentity(oss, "ctac_mu_map", args.attMapPath);
    }
    if (!args.ctacAutoMuFrom.empty()) {
      appendFileStatIdentity(oss, "ctac_auto_mu_from", args.ctacAutoMuFrom);
      oss << "ctac_auto_mu_thresh=" << args.ctacAutoMuThresh << '\n';
      oss << "ctac_auto_mu_value=" << args.ctacAutoMuValue << '\n';
      oss << "ctac_auto_mu_source=" << args.ctacAutoMuValueSource << '\n';
    }
  }
  // The per-bed attenuation tables are what the sensitivity actually consumes;
  // hashing their content makes the cache safe against any upstream change
  // (registration, bed geometry, mu conversion) that the parameters above do
  // not spell out, so CTAC runs can use the auto cache too.
  if (bedAttenFactors != nullptr && !bedAttenFactors->empty()) {
    oss << "ctac_table_count=" << bedAttenFactors->size() << '\n';
    for (size_t bed = 0; bed < bedAttenFactors->size(); ++bed) {
      const auto &table = (*bedAttenFactors)[bed];
      oss << "ctac_table_" << bed << '=' << table.size() << ':' << std::hex
          << fnv1a64Words(table.data(), table.size() * sizeof(uint16_t))
          << std::dec << '\n';
    }
    // Per-window energy exponent att^power (emitted only when some window is
    // rescaled, so unscaled CTAC caches keep matching).
    if (ctacAttenPowers != nullptr &&
        std::any_of(ctacAttenPowers->begin(), ctacAttenPowers->end(),
                    [](double p) { return p != 1.0; })) {
      oss << "ctac_atten_powers=";
      for (size_t k = 0; k < ctacAttenPowers->size(); ++k)
        oss << (k ? "," : "") << (*ctacAttenPowers)[k];
      oss << '\n';
    }
  }
  // Final sensitivity row weights W (censoring x efficiency x detector mask),
  // hashed as consumed.
  if (sensRowWeights != nullptr && !sensRowWeights->empty()) {
    oss << "sens_row_weights=" << sensRowWeights->size() << ':' << std::hex
        << fnv1a64Words(sensRowWeights->data(),
                        sensRowWeights->size() * sizeof(double))
        << std::dec << '\n';
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
    appendFileStatIdentity(oss, "detz_row_weights",
                           args.detZRowWeightsFile);
    oss << "detz_row_weights_count="
        << spect::io::loadDetZRowWeights(args.detZRowWeightsFile,
                                         compactMeta.detNz, compactMeta.detNx,
                                         compactMeta.fullNumAngles)
               .size()
        << '\n';
  }
  if (!args.normFactorFile.empty()) {
    appendFileStatIdentity(oss, "normalization_factor",
                           args.normFactorFile);
    oss << "normalization_factor_count="
        << spect::io::loadDetZRowWeights(
               args.normFactorFile, compactMeta.detNz, compactMeta.detNx,
               compactMeta.fullNumAngles)
               .size()
        << '\n';
    oss << "normalization_factor_semantics="
           "projection_correction_c_to_efficiency_1_over_c\n";
  }
  if (!args.normFactorFile2.empty()) {
    const auto values = spect::io::loadDetZRowWeights(args.normFactorFile2,
        compactMeta.detNz, compactMeta.detNx, compactMeta.fullNumAngles);
    appendFileStatIdentity(oss, "normalization_factor2", args.normFactorFile2);
    oss << "normalization_factor2_values=" << values.size() << ':' << std::hex
        << fnv1a64Words(values.data(), values.size() * sizeof(double)) << std::dec << '\n';
  }
  if (!args.badDetectorFile.empty()) {
    const auto badDetectors =
        spect::io::loadBadDetectorList(args.badDetectorFile);
    if (!badDetectors.empty()) {
      oss << "bad_detector_count=" << badDetectors.size() << '\n';
      for (size_t index = 0; index < badDetectors.size(); ++index)
        oss << "bad_detector_" << index << '='
            << spect::io::badDetectorGlobalNumber(badDetectors[index]) << ':'
            << spect::io::badDetectorName(badDetectors[index]) << '\n';
    }
  }
  oss << "bed_count=" << bedZStarts.size() << '\n';
  appendVectorLine(oss, "bed_z_starts", bedZStarts);
  oss << "reconstruction_beds_nonoverlapping=1\n";
  appendVectorLine(oss, "output_bed_overlaps", args.outputBedOverlaps);
  if (args.zBedFlip && !args.projPaths.empty() && bedZStarts.size() > 1) {
    oss << "joint_virtual_z_flip=1\n";
    oss << "joint_projection_mapping=local_z_into_bed_storage\n";
  }
  oss << "psf_mode=" << psfModeKey(args.psf.mode) << '\n';
  oss << "psf_kernel=" << psfKernelKey(args.psf.kernel) << '\n';
  oss << "psf_fwhm=" << args.psf.fwhm_x << ',' << args.psf.fwhm_y << ','
      << args.psf.fwhm_z << ',' << args.psf.fwhm_mm << '\n';
  oss << "psf_z_blur=" << (args.psf.z_blur ? 1 : 0) << '\n';
  oss << "psf_vox=" << args.psf.vox_x << ',' << args.psf.vox_y << ','
      << args.psf.vox_z << '\n';
  oss << "psf_truncation_mm=" << args.psf.truncation_mm << '\n';
  oss << "psf_z_dependent=" << (args.psf.z_dependent ? 1 : 0) << '\n';
  oss << "psf_interp=" << psfInterpKey(args.psf.interp) << '\n';
  oss << "psf_min_sigmas=" << args.psf.min_sigmas << '\n';
  if (args.useCylindricalMask && args.psf.mode != spect::filters::PSFMode::None) {
    oss << "psf_matrix_support_mask=cylinder\n";
    oss << "cylinder_mm=" << args.cylinderDiameterMM << ','
        << args.cylinderHeightMM << '\n';
    oss << "cylinder_center=" << args.cylCenterX << ',' << args.cylCenterY
        << ',' << args.cylCenterZ << '\n';
  }
  if (args.psf.mode != spect::filters::PSFMode::None &&
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
  if (ec)
    throw std::runtime_error("Failed creating compact denom cache directory: " +
                             parent.string() + ": " + ec.message());
  const auto manifestPath = compactDenomManifestPath(cacheMhdPath);
  ec.clear();
  const bool manifestExists = std::filesystem::exists(manifestPath, ec);
  if (ec)
    throw std::runtime_error("Failed checking compact denom cache manifest: " +
                             manifestPath.string() + ": " + ec.message());
  if (manifestExists && !std::filesystem::remove(manifestPath, ec)) {
    throw std::runtime_error("Failed invalidating compact denom cache manifest: " +
                             manifestPath.string() + ": " + ec.message());
  }
  spect::io::writeMHD(cacheMhdPath.string(), denom, geometry.nx, geometry.ny,
                      geometry.nz, static_cast<float>(geometry.vx),
                      static_cast<float>(geometry.vy),
                      static_cast<float>(geometry.vz), true);
  std::ofstream manifestOut(manifestPath, std::ios::binary);
  if (!manifestOut)
    throw std::runtime_error("Failed writing compact denom cache manifest: " +
                             manifestPath.string());
  manifestOut << manifest;
  if (!manifestOut)
    throw std::runtime_error("Failed writing compact denom cache manifest: " +
                             manifestPath.string());
}
} // namespace

std::vector<double> runCompactSymmetricReconstruction(
    const CompactSymmetricReconLoopContext &ctx) {
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
  const spect::io::CompactChunkCSRDirectory &compactChunks =
      *ctx.compactChunks;
  // Energy-window channels: one for a plain run, two for joint dual-window
  // MLEM. Projections/backgrounds of window k live in channelProjections[k].
  std::vector<spect::cuda::CompactMatrixChannel> channels;
  std::vector<const std::vector<double> *> channelProjections;
  std::vector<const std::vector<double> *> channelBackgrounds;
  {
    spect::cuda::CompactMatrixChannel first;
    first.chunks = &compactChunks;
    first.weight =
        args.energyWindowWeights.empty() ? 1.0 : args.energyWindowWeights[0];
    channels.push_back(first);
    channelProjections.push_back(ctx.projections);
    channelBackgrounds.push_back(ctx.backgroundProj);
    if (ctx.compactChunks2 != nullptr) {
      if (ctx.projections2 == nullptr || ctx.projections2->empty() ||
          args.energyWindowWeights.size() != 2) {
        throw std::runtime_error(
            "joint dual-window reconstruction needs second-window projections "
            "and two energy weights.");
      }
      spect::cuda::CompactMatrixChannel second;
      second.chunks = ctx.compactChunks2;
      second.weight = args.energyWindowWeights[1];
      channels.push_back(second);
      channelProjections.push_back(ctx.projections2);
      channelBackgrounds.push_back(ctx.backgroundProj2);
    }
  }
  const size_t channelCount = channels.size();
  bool anyBackground = false;
  for (const auto *background : channelBackgrounds)
    anyBackground = anyBackground || (background != nullptr && !background->empty());
  // CTAC energy scaling: the factor table is computed once at the mu map's
  // energy; window k uses att^(mu_w(E_k)/mu_w(E_map)). Without window
  // energies every window uses the table as is (the pre-existing behaviour).
  double ctacMapEnergyKeV = 0.0;
  if (args.doCTAC && !args.ctacWindowEnergiesKeV.empty()) {
    ctacMapEnergyKeV = args.ctacMapEnergyKeV;
    if (ctacMapEnergyKeV <= 0.0) {
      const auto *preset =
          spect::core::waterAttenuationPreset(args.ctacAutoMuNuclide);
      if (preset == nullptr) {
        throw std::runtime_error(
            "CTAC window energies need the mu map's energy (-ctacMapEnergy or "
            "a known [CTAC] auto_mu_nuclide).");
      }
      ctacMapEnergyKeV = preset->energyKeV;
    }
    if (args.ctacWindowEnergiesKeV.size() != channelCount) {
      throw std::runtime_error(
          "CTAC window energies must list one energy per energy window.");
    }
    std::ostringstream powers;
    for (size_t k = 0; k < channelCount; ++k) {
      double ratio = 1.0;
      if (!spect::core::waterMuEnergyRatio(args.ctacWindowEnergiesKeV[k],
                                           ctacMapEnergyKeV, ratio)) {
        throw std::runtime_error(
            "CTAC window/map energy outside the water attenuation table.");
      }
      channels[k].attenPower = ratio;
      powers << (k ? ", " : "") << "window " << k << ": "
             << args.ctacWindowEnergiesKeV[k] << " keV -> att^" << ratio;
    }
    logInfo("CTAC energy scaling: mu map at " +
            std::to_string(ctacMapEnergyKeV) + " keV; " + powers.str() + ".");
  }
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
  spect::cuda::setCsrqForceStreamingCUDA(args.csrqResidency == "streamed");
#endif
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

#if !defined(SPECT_ENABLE_CUDA_SPARSEOPS)
        throw std::runtime_error(
            "compact-symmetric layout requires a CUDA-enabled build.");
#else
        const OutputGeometry geometry = outputGeometry;
        const std::string outputPrefix = outputPrefixFromArgs(args);
        std::vector<const uint16_t *> compactAttFactorPtrs;
        if (!opts.bedAttenFactors.empty()) {
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
          logInfo("compact CTAC: per-bed attenuation factors applied in the "
                  "image domain per compact angle (branch-batch scale before "
                  "the forward, transform-add after the backward); the SpMV "
                  "kernels are the plain ones.");
        }
        const bool compactHasAttenuation = !compactAttFactorPtrs.empty();
        // Ordered subsets over compact angles (design report 7.10). The
        // schedule is non-increasing, so the first stage has the finest
        // subset count; every coarser stage's S_m is an exact sum of the
        // finest ones and stage 1 is the plain full-data update.
        const std::vector<CompactSubsetStage> subsetSchedule =
            parseCompactSubsetSchedule(args.subsetSchedule, args.numIters,
                                       compactMeta.compactNumAngles);
        const int finestSubsetCount = subsetSchedule.front().subsets;
        const bool useSubsets = finestSubsetCount > 1;
        if (useSubsets) {
          if (args.bsremDiagnostics || args.bsremMlemEquivalenceMode) {
            throw std::runtime_error(
                "-subsets > 1 cannot be combined with the BSREM diagnostics or "
                "MLEM-equivalence debug modes (per-iteration objectives are "
                "full-data quantities).");
          }
          std::ostringstream orderText;
          for (const int k : compactSubsetOrder(finestSubsetCount))
            orderText << (orderText.tellp() > 0 ? "," : "") << k;
          logInfo("compact ordered subsets: schedule " +
                  formatCompactSubsetSchedule(subsetSchedule) +
                  " over " + std::to_string(compactMeta.compactNumAngles) +
                  " compact angles (interleaved; visiting order of the finest "
                  "stage: " + orderText.str() + ")");
        }
        std::vector<double> denom(globalNvox, 0.0);
        const bool compactPsfActive = needs_psf_blur(opts, psfPtr);
        // The frozen prototype prunes CSR columns to the cylinder before PSF.
        // Masking after forward and before backward PSF is equivalent here.
        const std::vector<uint8_t> compactLocalSupportMask =
            compactPsfActive && args.useCylindricalMask
                ? spect::core::makeCylMask_mm(
                      baseNx, baseNy, baseNz, spacing.dx, spacing.dy, spacing.dz,
                      args.cylinderDiameterMM, args.cylinderHeightMM,
                      args.cylCenterX, args.cylCenterY, args.cylCenterZ)
                : std::vector<uint8_t>();
        const bool compactRoiPrunesMatrix =
            !compactLocalSupportMask.empty() &&
            // ponytail: full-FOV cylinders rely on the matrix's cylindrical
            // support; add a column-support bitmap if future matrices differ.
            // ROI dimensions are float: compare extents at the same precision.
            // Widened float voxel sizes otherwise make 36/66 mm look pruned.
            (args.cylinderDiameterMM + 1e-6 <
                 static_cast<float>(
                     std::min(baseNx * spacing.dx, baseNy * spacing.dy)) ||
             (args.cylinderHeightMM > 0.0 &&
              args.cylinderHeightMM + 1e-6 <
                  static_cast<float>(baseNz * spacing.dz)) ||
             !std::isnan(args.cylCenterX) || !std::isnan(args.cylCenterY) ||
             !std::isnan(args.cylCenterZ));
        // Sieve: iteration path is identical to Operator; save points write a
        // PSF-convolved copy of the state instead of the raw state.
        const bool sieveSaveCopyActive =
            compactPsfActive && args.psf.mode == filters::PSFMode::Sieve;
        OutputFilterSettings outputFilters;
        outputFilters.smoothEnabled = args.smoothApplyStage == "output";
        outputFilters.smoothFwhmMm = args.outputSmoothFwhmMm;
        outputFilters.smoothSigmas = args.outputSmoothSigmas;
        outputFilters.butterEnabled = args.butterApplyStage == "output";
        outputFilters.butterCutoffCpcm = args.outputButterCutoffCpcm;
        outputFilters.butterOrder = args.outputButterOrder;
        outputFilters.nonnegativeClamp = args.outputNonnegativeClamp;
        outputFilters.overwritePrimary =
            args.outputFiltersOverwritePrimary;
        spect::cuda::CompactSymmetricStageTiming compactCudaTiming;
        MemoryUsageSnapshot compactPeakMemory;
        auto observeCompactMemory = [&]() {
          mergeMemoryPeak(compactPeakMemory, queryMemoryUsageSnapshot());
        };
        observeCompactMemory();
        // Weighted sensitivity A^T w (opt-in, compact chunk layout only; CLI
        // validation enforces this). w starts at 1; detZ edge-row censoring
        // zeroes the censored detZ edge rows, and an efficiency-weights file
        // multiplies in per-row, per-detector-bin, or full projection-bin
        // factors. Hardware detector masking then zeroes complete
        // angle/block row sets; all zero masks win.
        // The weights enter S = A^T W 1 directly through the unit-ratio
        // kernels. Measured bins are never scaled here: censoring zeroes
        // them in MainProgram, and the efficiency weights are
        // denominator-only. When the final w is all ones the unweighted
        // path is kept, so the default stays bit-identical. Computed before
        // the denom cache manifest so its hash is part of the identity.
        const int detZEdgeCensor = args.detZEdgeCensor;
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
        // A calibrated joint model requires a distinct correction for each
        // window. The numerator cancels the efficiency for background-free EM;
        // the sensitivity retains w_k A_k^T (1/c_k), with shared masks.
        std::vector<double> detZSensWeights2;
        if (!args.normFactorFile2.empty()) {
          const auto correction1 = spect::io::loadDetZRowWeights(
              args.normFactorFile, compactMeta.detNz, compactMeta.detNx,
              compactMeta.fullNumAngles);
          const auto correction2 = spect::io::loadDetZRowWeights(
              args.normFactorFile2, compactMeta.detNz, compactMeta.detNx,
              compactMeta.fullNumAngles);
          if (correction1.size() != static_cast<size_t>(compactMeta.fullRows()) ||
              correction2.size() != correction1.size() || channels.size() != 2)
            throw std::runtime_error("Joint normalization requires two full projection correction arrays.");
          detZSensWeights2.resize(correction2.size());
          for (size_t i = 0; i < correction2.size(); ++i)
            detZSensWeights2[i] = detZSensWeights[i] == 0.0 ? 0.0 : 1.0 / correction2[i];
          channels[1].sensitivityRowWeights = detZSensWeights2.data();
        }
        const auto sensStart = ProfileClock::now();
        std::filesystem::path compactDenomCachePath =
            resolveCompactDenomCacheMhdPath(args);
        std::string compactDenomManifest;
        std::vector<double> ctacAttenPowers;
        for (const auto &channel : channels)
          ctacAttenPowers.push_back(channel.attenPower);
        compactDenomManifest = buildCompactDenomCacheManifest(
            args, geometry, compactMeta, compactMetaPath, compactChunks,
            bedZStarts, psfPtr, ctx.compactChunks2,
            useWeightedSensitivity ? &detZSensWeights : nullptr,
            compactHasAttenuation ? &opts.bedAttenFactors : nullptr,
            compactHasAttenuation ? &ctacAttenPowers : nullptr);
        if (useWeightedSensitivity && detZEdgeCensor > 0) {
          logInfo("detZ edge censoring: using masked sensitivity (k=" +
                  std::to_string(detZEdgeCensor) + ", " +
                  std::to_string(detZEdgeCensor) +
                  " detZ rows censored per side) for the denominator.");
        }
        // Assign every window its device matrix slot (aliasing channels that
        // share a directory) before any per-channel call below.
        spect::cuda::prepareChannelExecutionCUDA(channels);
        // Global sensitivity S over all rows into `target` (beds accumulated,
        // PSF^T applied per bed), or S_m over one subset's rows.
        std::vector<double> localDenom(baseNvox, 0.0);
        auto computeGlobalSensitivity = [&](std::vector<double> &target,
                                            const CompactSubsetSpec *subset) {
          std::fill(target.begin(), target.end(), 0.0);
          auto computeLocalSensitivity = [&](const uint16_t *attenuation) {
            std::fill(localDenom.begin(), localDenom.end(), 0.0);
            const bool ok = spect::cuda::computePoissonSensitivityCUDA(
                channels, compactMeta, opts.threads, localDenom.data(),
                &compactCudaTiming, attenuation,
                useWeightedSensitivity ? detZSensWeights.data() : nullptr,
                subset);
            if (!ok) {
              throw std::runtime_error(
                  "compact-symmetric sensitivity requires CUDA sparse backend.");
            }
            if (!compactLocalSupportMask.empty())
              spect::core::applyMaskInplace(localDenom,
                                             compactLocalSupportMask);
            apply_psf_backward(localDenom, baseNx, baseNy, baseNz, opts,
                               psfPtr, opts.threads);
          };
          if (compactHasAttenuation) {
            for (int bed = 0; bed < bedCount; ++bed) {
              computeLocalSensitivity(
                  compactAttFactorPtrs[static_cast<size_t>(bed)]);
              spect::app::accumulateBedVolumeToGlobal(
                  target, localDenom, baseNx, baseNy, baseNz,
                  bedZStarts[static_cast<size_t>(bed)], bedVirtualZFlip);
              observeCompactMemory();
            }
          } else {
            computeLocalSensitivity(nullptr);
            for (int bed = 0; bed < bedCount; ++bed) {
              spect::app::accumulateBedVolumeToGlobal(
                  target, localDenom, baseNx, baseNy, baseNz,
                  bedZStarts[static_cast<size_t>(bed)], bedVirtualZFlip);
              observeCompactMemory();
            }
          }
        };
        // Cache-or-compute one sensitivity volume. Returns true on a cache hit.
        auto loadOrComputeSensitivity =
            [&](const std::filesystem::path &cachePath,
                const std::string &manifest, const char *label,
                std::vector<double> &target, const CompactSubsetSpec *subset) {
              bool loaded = false;
              if (!cachePath.empty()) {
                std::string cacheStatus;
                loaded = tryLoadCompactDenomCache(cachePath, manifest, geometry,
                                                  target, &cacheStatus);
                if (loaded) {
                  logInfo(std::string("compact ") + label + " cache hit: " +
                          cachePath.string());
                } else {
                  if (isCompactDenomManifestMismatch(cacheStatus)) {
                    throw std::runtime_error(
                        std::string("compact ") + label +
                        " cache manifest mismatch for " + cachePath.string() +
                        ": " + cacheStatus +
                        ". Use a distinct -compactDenomCache path for no-PSF, "
                        "PSF/FWHM, and bed-geometry variants, or delete the "
                        "stale cache intentionally.");
                  }
                  logInfo(std::string("compact ") + label + " cache miss: " +
                          cacheStatus);
                }
              }
              if (!loaded) {
                computeGlobalSensitivity(target, subset);
                if (!cachePath.empty()) {
                  try {
                    saveCompactDenomCache(cachePath, manifest, geometry, target);
                    logInfo(std::string("compact ") + label + " cache saved: " +
                            cachePath.string());
                  } catch (const std::exception &ex) {
                    logWarn(std::string("Failed saving compact ") + label +
                            " cache: " + ex.what());
                  }
                }
              }
              return loaded;
            };
        const bool explicitDenomCachePath = !compactDenomCachePath.empty();
        if (compactDenomCachePath.empty()) {
          compactDenomCachePath =
              autoCompactDenomCachePath(args, compactDenomManifest);
          logInfo("compact denom auto-cache path: " +
                  compactDenomCachePath.string());
        }
        const bool loadedCompactDenomCache = loadOrComputeSensitivity(
            compactDenomCachePath, compactDenomManifest, "denom", denom,
            nullptr);
        // Finest-stage subset sensitivities S_m (each its own cache entry:
        // the global manifest plus the subset line). Coarser stages sum them.
        std::vector<std::vector<double>> finestSubsetDenoms;
        if (useSubsets) {
          finestSubsetDenoms.assign(static_cast<size_t>(finestSubsetCount),
                                    std::vector<double>(globalNvox, 0.0));
          for (int k = 0; k < finestSubsetCount; ++k) {
            const CompactSubsetSpec spec{finestSubsetCount, k};
            const std::string subsetManifest =
                compactDenomManifest + "subset=" + std::to_string(k) + "/" +
                std::to_string(finestSubsetCount) + "\n";
            std::filesystem::path subsetCachePath;
            if (explicitDenomCachePath) {
              subsetCachePath = compactDenomCachePath;
              subsetCachePath.replace_extension();
              subsetCachePath += "_s" + std::to_string(k) + "of" +
                                 std::to_string(finestSubsetCount) + ".mhd";
            } else {
              subsetCachePath = autoCompactDenomCachePath(args, subsetManifest);
            }
            const std::string label =
                "subset " + std::to_string(k) + "/" +
                std::to_string(finestSubsetCount) + " denom";
            loadOrComputeSensitivity(subsetCachePath, subsetManifest,
                                     label.c_str(),
                                     finestSubsetDenoms[static_cast<size_t>(k)],
                                     &spec);
          }
          // Consistency gate: the subset rows partition the matrix rows.
          double squaredNorm = 0.0;
          double squaredError = 0.0;
          for (size_t j = 0; j < globalNvox; ++j) {
            double sumSubsets = 0.0;
            for (const auto &subsetDenom : finestSubsetDenoms)
              sumSubsets += subsetDenom[j];
            squaredNorm += denom[j] * denom[j];
            const double error = sumSubsets - denom[j];
            squaredError += error * error;
          }
          const double relDiff =
              std::sqrt(squaredError / std::max(squaredNorm, 1e-300));
          logInfo("compact subset sensitivities: sum_m S_m vs S voxelwise relL2 " +
                  std::to_string(relDiff));
          if (!(relDiff < 1e-4)) {
            throw std::runtime_error(
                "compact subset sensitivities do not add up to the global "
                "sensitivity (relative difference " + std::to_string(relDiff) +
                "); stale cache or subset mapping fault.");
          }
        }
        localDenom = std::vector<double>();
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
        if (maskPtr)
          spect::core::applyMaskInplace(image, *maskPtr);
        std::vector<double> gradReg;
        std::vector<double> curvMap;
        std::vector<double> kappaMap =
            (args.algorithm == ReconArgs::Algorithm::BSREM &&
             args.bsremUseKappaMap)
              ? computeKappaMapStandard(denom, args, maskPtr)
                : std::vector<double>();
        const bool useRegularization =
            args.algorithm == ReconArgs::Algorithm::BSREM && args.useReg &&
            args.reg.beta > 0.0 && args.reg.type != RegType::None;
        const bool useBsremUpdate =
            args.algorithm == ReconArgs::Algorithm::BSREM;
        // C.3: optional relative sensitivity floor. Voxels with
        // s_j < rel * max(s) (low-support bed-edge slices) are frozen to zero
        // instead of diverging under the multiplicative ML-family update.
        double bsremSensFloorAbs = 0.0;
        if (useBsremUpdate && args.bsremSensFloorRel > 0.0f) {
          double sMax = 0.0;
          for (double v : denom)
            sMax = std::max(sMax, v);
          bsremSensFloorAbs =
              static_cast<double>(args.bsremSensFloorRel) * sMax;
          logInfo("[BSREM] sensitivity floor: rel=" +
                  std::to_string(args.bsremSensFloorRel) +
                  " abs=" + std::to_string(bsremSensFloorAbs) +
                  " (max_s=" + std::to_string(sMax) + ")");
        }
        const std::vector<uint8_t> *maskForReg =
            (maskPtr && useRegularization && args.reg.skip_outside_mask)
                ? maskPtr
                : nullptr;
        bool compactChunkReverseOrderNext = !loadedCompactDenomCache;
        const bool debugStats = getVerbosity() >= 2;
        // Both optimizers update the resident image on the device; only the
        // host-side diagnostics, the MLEM-equivalence debug mode and the
        // in-iteration output filters still need the host image every
        // iteration.
        const bool residentUpdateEligible =
            !debugStats && !prepared.butterConf.enabled &&
            !prepared.smoothConf.enabled && !args.bsremMlemEquivalenceMode &&
            !args.bsremDiagnostics;
        const bool residentRegGradient =
            useRegularization && residentUpdateEligible &&
            (args.reg.type == RegType::RDP ||
             args.reg.type == RegType::TVIsotropic);
        // A penalty without a CUDA gradient kernel keeps the host update.
        const bool residentUpdateAllowed =
            residentUpdateEligible && (!useRegularization || residentRegGradient);
        bool deviceCurvatureInitialized = false;
        // Projection storage is immutable throughout the loop. Build the
        // [channel * bedCount + bed] views once, alongside their device slots.
        std::vector<const double *> projectionPtrs(channelCount * bedCount);
        std::vector<const double *> backgroundPtrs(channelCount * bedCount);
        {
          // Pre-upload every window's projections and backgrounds so the
          // first numerator does not pay the H2D inside iteration_core_s;
          // CSRQ and single-window CSRB reuse these slots. Joint CSRB currently
          // refreshes the per-bed slots when switching windows.
          const size_t fullProjectionCount =
              static_cast<size_t>(compactMeta.fullRows());
          for (size_t channel = 0; channel < channelCount; ++channel) {
            for (int bed = 0; bed < bedCount; ++bed) {
              const size_t offset =
                  static_cast<size_t>(bed) * fullProjectionCount;
              const size_t slot = channel * bedCount + bed;
              const auto *background = channelBackgrounds[channel];
              projectionPtrs[slot] = channelProjections[channel]->data() + offset;
              backgroundPtrs[slot] =
                  (background == nullptr || background->empty())
                      ? nullptr : background->data() + offset;
              if (!spect::cuda::refreshCompactProjectionCUDA(
                      projectionPtrs[slot], fullProjectionCount,
                      static_cast<int>(channel) * spect::core::kMaxBeds + bed,
                      backgroundPtrs[slot])) {
                throw std::runtime_error(
                    "compact projection upload requires CUDA sparse backend.");
              }
            }
          }
        }
        auto lastIterationDone = ProfileClock::now();
        bool imageResident = false;
        bool finalOutputSaved = false;
        bool finalManifestWritten = false;
        auto saveCompactOutput = [&](int iteration) {
          const bool isFinalSave = iteration == args.numIters;
          if (isFinalSave && (finalOutputSaved || finalManifestWritten)) {
            throw std::runtime_error(
                "compact final output or run manifest would be written twice");
          }

          const auto saveStart = ProfileClock::now();
          if (imageResident &&
              !spect::cuda::downloadCompactResidentImageCUDA(
                  image.data(), image.size())) {
            throw std::runtime_error(
                "compact resident image download requires CUDA backend.");
          }
          std::vector<double> sieveSaveVolume;
          const std::vector<double> *saveSource = &image;
          if (sieveSaveCopyActive) {
            sieveSaveVolume = build_sieve_save_volume(
                image, baseNx, baseNy, baseNz, bedCount, bedZStarts,
                bedVirtualZFlip, opts, psfPtr, opts.threads);
            saveSource = &sieveSaveVolume;
          }
          saveVolumeOutput(*saveSource, geometry, args.outputDir, outputPrefix,
                           iteration, maskPtr, crop, true, outputTransformPtr,
                           &outputFilters);
          compactTiming.outputWriteSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - saveStart)
                  .count();

          if (isFinalSave) {
            finalOutputSaved = true;
            RunManifestInput manifestInput;
            manifestInput.args = &args;
            manifestInput.compactMeta = &compactMeta;
            manifestInput.compactMetaPath = &compactMetaPath;
            manifestInput.bedZStarts = &bedZStarts;
            manifestInput.outputTransform = outputTransformPtr;
            manifestInput.finalSaveVolume = saveSource;
            manifestInput.geometry = &geometry;
            manifestInput.savedIteration = iteration;
            manifestInput.outputPrefix = outputPrefix;
            writeRunManifest(manifestInput);
            finalManifestWritten = true;
          }
        };

        std::ofstream metricsCsv;
        if (args.bsremDiagnostics) {
          std::error_code metricsDirError;
          std::filesystem::create_directories(args.outputDir, metricsDirError);
          if (metricsDirError) {
            throw std::runtime_error(
                "Failed creating BSREM diagnostics directory: " +
                metricsDirError.message());
          }
          const auto metricsPath =
              std::filesystem::path(args.outputDir) / "iteration_metrics.csv";
          metricsCsv.open(metricsPath, std::ios::binary);
          if (!metricsCsv)
            throw std::runtime_error("Failed opening iteration_metrics.csv");
          metricsCsv << bsremIterationMetricsCsvHeader() << '\n';
        }

        const auto matrixReadStats = [&]() {
          auto reads = compactChunks.readStats();
          if (ctx.compactChunks2 && ctx.compactChunks2 != ctx.compactChunks) {
            const auto second = ctx.compactChunks2->readStats();
            reads.payloadLoads += second.payloadLoads;
            reads.payloadReadBytes += second.payloadReadBytes;
            reads.hostReuseBytes += second.hostReuseBytes;
            reads.emptySlicesSkipped += second.emptySlicesSkipped;
          }
          return reads;
        };
        const auto readsBeforeIterations = matrixReadStats();
        const auto h2dBeforeIterations =
            spect::cuda::getCudaChunkStageTimingSnapshotCUDA().csrbMatrixH2DBytes;
        {
        // Host numerator payloads stay empty on resident iterations. Reuse
        // capacity on host updates, then release it before final output work.
        std::vector<double> numer;
        std::vector<std::vector<double>> localNumers(bedCount);
        std::vector<const double *> imagePtrs(bedCount);
        std::vector<double *> numerPtrs(bedCount);

        // Subset schedule bookkeeping. A stage with M subsets runs M
        // sub-iterations per iteration, each over one interleaved subset of
        // compact angles in the maximal-separation order; the stage's S_m
        // are the finest S_m summed (stage M = 1 uses the global S itself).
        auto stageIndexOfIteration = [&](int iteration) {
          int cumulative = 0;
          for (size_t s = 0; s < subsetSchedule.size(); ++s) {
            cumulative += subsetSchedule[s].iterations;
            if (iteration <= cumulative)
              return s;
          }
          return subsetSchedule.size() - 1;
        };
        size_t currentStage = subsetSchedule.size(); // none entered yet
        int stageSubsetCount = 1;
        std::vector<int> stageOrder{0};
        std::vector<std::vector<double>> stageDenomStorage;
        std::vector<const std::vector<double> *> stageDenoms{&denom};
        auto enterStage = [&](size_t stage) {
          currentStage = stage;
          stageSubsetCount = subsetSchedule[stage].subsets;
          stageOrder = compactSubsetOrder(stageSubsetCount);
          stageDenomStorage.clear();
          stageDenoms.clear();
          if (stageSubsetCount == 1) {
            stageDenoms.push_back(&denom);
          } else if (stageSubsetCount == finestSubsetCount) {
            for (const auto &subsetDenom : finestSubsetDenoms)
              stageDenoms.push_back(&subsetDenom);
          } else {
            // Subset j of M is the union of the finest subsets k == j (mod M).
            stageDenomStorage.assign(static_cast<size_t>(stageSubsetCount),
                                     std::vector<double>(globalNvox, 0.0));
            for (int k = 0; k < finestSubsetCount; ++k) {
              auto &target =
                  stageDenomStorage[static_cast<size_t>(k % stageSubsetCount)];
              const auto &source = finestSubsetDenoms[static_cast<size_t>(k)];
              for (size_t j = 0; j < globalNvox; ++j)
                target[j] += source[j];
            }
            for (const auto &stageDenom : stageDenomStorage)
              stageDenoms.push_back(&stageDenom);
          }
          std::vector<const double *> devicePtrs;
          for (const auto *stageDenom : stageDenoms)
            devicePtrs.push_back(stageDenom->data());
          if (!spect::cuda::setCompactSubsetSensitivitiesCUDA(
                  stageSubsetCount > 1 ? devicePtrs.data() : nullptr,
                  stageSubsetCount > 1 ? stageSubsetCount : 0, globalNvox)) {
            throw std::runtime_error(
                "compact subset sensitivities require the CUDA backend.");
          }
          if (useSubsets) {
            logInfo("compact subset stage " + std::to_string(stage + 1) + "/" +
                    std::to_string(subsetSchedule.size()) + ": " +
                    std::to_string(stageSubsetCount) + " subset(s) for " +
                    std::to_string(subsetSchedule[stage].iterations) +
                    " iteration(s)");
          }
        };
        // Subset of the numerator call after (iteration, sub) in schedule
        // order, for the cross-pass prefetch; nullopt after the last call.
        auto nextSubsetSpec = [&](int iteration, int sub)
            -> std::optional<CompactSubsetSpec> {
          if (sub + 1 < stageSubsetCount) {
            return CompactSubsetSpec{stageSubsetCount,
                                     stageOrder[static_cast<size_t>(sub + 1)]};
          }
          if (iteration >= args.numIters)
            return std::nullopt;
          const int nextCount =
              subsetSchedule[stageIndexOfIteration(iteration + 1)].subsets;
          return CompactSubsetSpec{nextCount, compactSubsetOrder(nextCount)[0]};
        };

        for (int it = 1; it <= args.numIters; ++it) {
          const auto iterationWallStart = ProfileClock::now();
          logInfo("[ITER] " + std::to_string(it) + "/" +
                  std::to_string(args.numIters) + " compact");
          if (stageIndexOfIteration(it) != currentStage)
            enterStage(stageIndexOfIteration(it));
          for (int sub = 0; sub < stageSubsetCount; ++sub) {
          const auto subsetWallStart = ProfileClock::now();
          const CompactSubsetSpec subsetSpec{
              stageSubsetCount, stageOrder[static_cast<size_t>(sub)]};
          const std::optional<CompactSubsetSpec> nextSpec =
              nextSubsetSpec(it, sub);
          // -1 selects the global S on the device (full-data step).
          const int deviceSubsetIndex =
              stageSubsetCount > 1 ? subsetSpec.index : -1;
          const std::vector<double> &subsetDenom =
              *stageDenoms[static_cast<size_t>(
                  stageSubsetCount > 1 ? subsetSpec.index : 0)];
          if (stageSubsetCount > 1) {
            logInfo("[ITER] " + std::to_string(it) + " subset " +
                    std::to_string(subsetSpec.index) + "/" +
                    std::to_string(stageSubsetCount) + " (" +
                    std::to_string(sub + 1) + " of " +
                    std::to_string(stageSubsetCount) + ")");
          }
          // The resident branch-batch input (image rotated on the device)
          // serves any bed count and both formats, backgrounds and CTAC
          // included (the factor is applied on the device per compact angle);
          // ROI-pruned matrices still take the host input path.
          const bool residentBatchInput =
              !debugStats && !compactRoiPrunesMatrix &&
              (!compactPsfActive ||
               opts.psf.kernel == filters::PSFKernel::Castor) &&
              spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
                  compactMeta, image.data(), image.size(),
                  args.Nz, bedZStarts.data(), bedCount, bedVirtualZFlip,
                  compactPsfActive ? psfPtr : nullptr, !imageResident);
          if (imageResident && !residentBatchInput) {
            if (!spect::cuda::downloadCompactResidentImageCUDA(
                    image.data(), image.size())) {
              throw std::runtime_error(
                  "compact resident fallback image download failed.");
            }
            imageResident = false;
          }
          const bool residentIteration =
              residentBatchInput && residentUpdateAllowed;
          if (it == 1 && sub == 0) {
            logInfo(std::string("compact iteration path: input=") +
                    (residentBatchInput ? "device" : "host") + ", update=" +
                    (residentIteration ? "device" : "host") +
                    ", roi_prunes_matrix=" + (compactRoiPrunesMatrix ? "yes" : "no"));
          }
          if (!residentIteration)
            numer.assign(globalNvox, 0.0);

            std::vector<std::vector<double>> localImages(bedCount);
            std::vector<std::vector<double>> localUsed(bedCount);
            if (!residentIteration) {
              for (auto &localNumer : localNumers)
                localNumer.resize(baseNvox);
            }

            for (int bed = 0; bed < bedCount; ++bed) {
              if (!residentBatchInput) {
                const int zStart = bedZStarts[static_cast<size_t>(bed)];
                localImages[static_cast<size_t>(bed)] =
                    spect::app::extractBedVolumeFromGlobal(
                        image, baseNx, baseNy, baseNz, zStart,
                        bedVirtualZFlip);
                imagePtrs[static_cast<size_t>(bed)] =
                    localImages[static_cast<size_t>(bed)].data();
                if (compactPsfActive) {
                  if (debugStats) {
                    logDebug(formatStatsEntry(
                        "compact_local_image.it" + std::to_string(it) +
                            ".bed" + std::to_string(bed),
                        computeVectorStats(
                            localImages[static_cast<size_t>(bed)])));
                  }
                  apply_psf_forward(localImages[static_cast<size_t>(bed)],
                                    localUsed[static_cast<size_t>(bed)], baseNx,
                                    baseNy, baseNz, opts, psfPtr, opts.threads);
                  if (!compactLocalSupportMask.empty())
                    spect::core::applyMaskInplace(
                        localUsed[static_cast<size_t>(bed)],
                        compactLocalSupportMask);
                  if (debugStats) {
                    logDebug(formatStatsEntry(
                        "compact_psf_forward.it" + std::to_string(it) +
                            ".bed" + std::to_string(bed),
                        computeVectorStats(
                            localUsed[static_cast<size_t>(bed)])));
                  }
                  imagePtrs[static_cast<size_t>(bed)] =
                      localUsed[static_cast<size_t>(bed)].data();
                }
              }

              if (!residentIteration) {
                numerPtrs[static_cast<size_t>(bed)] =
                    localNumers[static_cast<size_t>(bed)].data();
              }
            }

            const auto numerStart = ProfileClock::now();
            const bool ok = spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
                channels, compactMeta, projectionPtrs.data(),
                anyBackground ? backgroundPtrs.data() : nullptr,
                residentBatchInput ? nullptr : imagePtrs.data(), bedCount,
                opts.threads,
                residentIteration ? nullptr : numerPtrs.data(),
                compactChunkReverseOrderNext, &compactCudaTiming,
                compactAttFactorPtrs.empty() ? nullptr
                                             : compactAttFactorPtrs.data(),
                /*refreshProjectionInputs=*/false, residentBatchInput,
                !residentIteration, &subsetSpec,
                nextSpec ? &*nextSpec : nullptr);
            if (!ok) {
              throw std::runtime_error(
                  "compact-symmetric numerator requires CUDA sparse backend.");
            }
            compactChunkReverseOrderNext = !compactChunkReverseOrderNext;
            compactTiming.compactForwardRatioBackwardSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - numerStart)
                    .count();

          if (residentIteration) {
            const auto updateStart = ProfileClock::now();
            bool imageAllZero = false;
            if (!useBsremUpdate) {
              if (!spect::cuda::finishCompactResidentMlemIterationCUDA(
                      compactMeta, globalNvox, args.Nz, bedZStarts.data(),
                      bedCount, bedVirtualZFlip, denom.data(),
                      maskPtr == nullptr ? nullptr : maskPtr->data(),
                      compactPsfActive ? psfPtr : nullptr,
                      compactPositiveProjectionInput, !imageResident,
                      &imageAllZero, deviceSubsetIndex)) {
                throw std::runtime_error(
                    "compact resident MLEM update requires CUDA backend.");
              }
            } else {
              // BSREM-II on the resident image: penalty gradient from the
              // device image, then the relative update, no host round trip.
              // alpha is per iteration (constant over its sub-iterations).
              const int k = it - 1;
              spect::cuda::CompactBsremUpdateParams updateParams;
              updateParams.alpha =
                  args.bsrem_alpha0 /
                  (1.0 + static_cast<double>(args.bsrem_gamma) * k);
              updateParams.subsetScale = static_cast<double>(stageSubsetCount);
              updateParams.betaScaled = useRegularization ? args.reg.beta : 0.0;
              updateParams.denomEps = 1e-12;
              updateParams.sMin = std::max(1e-12, bsremSensFloorAbs);
              updateParams.clipRelativeStep = true;
              updateParams.nonneg = opts.nonneg;
              updateParams.zeroLowSensitivity = true;
              updateParams.kappa =
                  args.bsremUseKappaMap ? kappaMap.data() : nullptr;
              if (residentRegGradient) {
                const auto regStart = ProfileClock::now();
                const double *dImage =
                    spect::cuda::compactResidentImageDeviceCUDA(globalNvox);
                double *dGrad =
                    spect::cuda::compactResidentGradientDeviceCUDA(globalNvox);
                if (dImage == nullptr || dGrad == nullptr ||
                    !cuda::computeRegGradientDeviceCUDA(
                        dImage, args.Nx, args.Ny, args.Nz, args.reg,
                        maskForReg, dGrad)) {
                  throw std::runtime_error(
                      "Regularization gradient requires CUDA volume helpers.");
                }
                compactTiming.regGradientSeconds +=
                    std::chrono::duration<double>(ProfileClock::now() -
                                                  regStart)
                        .count();
                ++compactTiming.regGradientCalls;
                updateParams.deviceGradient = dGrad;
                if (args.bsremUseCurvDenom) {
                  updateParams.useCurvature = true;
                  updateParams.refreshCurvature =
                      !deviceCurvatureInitialized || !args.bsremFreezeCurv ||
                      it <= std::max(1, args.bsremCurvWarmupIters);
                  updateParams.curvEps = args.bsremCurvEps;
                  deviceCurvatureInitialized = true;
                }
              }
              if (!spect::cuda::finishCompactResidentBsremIterationCUDA(
                      compactMeta, globalNvox, args.Nz, bedZStarts.data(),
                      bedCount, bedVirtualZFlip, denom.data(),
                      maskPtr == nullptr ? nullptr : maskPtr->data(),
                      compactPsfActive ? psfPtr : nullptr, updateParams,
                      compactPositiveProjectionInput, !imageResident,
                      &imageAllZero, deviceSubsetIndex)) {
                throw std::runtime_error(
                    "compact resident BSREM update requires CUDA backend.");
              }
            }
            compactTiming.mlemUpdateSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - updateStart)
                    .count();
            imageResident = true;
            if (imageAllZero) {
              const std::string message =
                  "checkpoint max==0 at compact iteration " +
                  std::to_string(it) + "; output is all zero.";
              logWarn(message);
              const char *failClosed =
                  std::getenv("SPECT_FAIL_ON_ZERO_CHECKPOINT");
              if (failClosed && failClosed[0] == '1')
                throw std::runtime_error(
                    message +
                    " Set SPECT_FAIL_ON_ZERO_CHECKPOINT=0 to continue.");
            }
          } else {
            for (int bed = 0; bed < bedCount; ++bed) {
              const int zStart = bedZStarts[static_cast<size_t>(bed)];
              std::vector<double> &localNumer =
                  localNumers[static_cast<size_t>(bed)];
              if (compactPsfActive && debugStats) {
                logDebug(formatStatsEntry(
                    "compact_local_numer_pre_psf.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localNumer)));
              }
              if (!compactLocalSupportMask.empty())
                spect::core::applyMaskInplace(localNumer,
                                               compactLocalSupportMask);
              apply_psf_backward(localNumer, baseNx, baseNy, baseNz, opts,
                                 psfPtr, opts.threads);
              if (compactPsfActive && debugStats) {
                logDebug(formatStatsEntry(
                    "compact_local_numer_post_psf.it" + std::to_string(it) +
                        ".bed" + std::to_string(bed),
                    computeVectorStats(localNumer)));
              }
              spect::app::accumulateBedVolumeToGlobal(
                  numer, localNumer, baseNx, baseNy, baseNz, zStart,
                  bedVirtualZFlip);
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
            if (!tryComputeRegGradientCUDA(image, args, maskForReg, gradReg)) {
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
          const bool recordDiag =
              args.bsremDiagnostics &&
              shouldRecordBsremDiagnostics(it, args.numIters,
                                           args.bsremDiagInterval);
          std::vector<double> imagePrev;
          std::vector<double> rawSteps;
          std::vector<double> precondSteps;
          if (recordDiag)
            imagePrev = image;
          if (!useBsremUpdate) {
            applyMlemUpdate(image, numer, subsetDenom, maskPtr, opts.nonneg,
                            opts.threads, &denom);
          } else {
            BsremUpdateParams updateParams;
            updateParams.mask = maskPtr;
            updateParams.gradient =
                args.bsremMlemEquivalenceMode ? nullptr : gradPtr;
            updateParams.curvature =
                args.bsremMlemEquivalenceMode ? nullptr : curvPtr;
            updateParams.kappa =
                args.bsremMlemEquivalenceMode ? nullptr : kappaPtr;
            updateParams.betaScaled =
                args.bsremMlemEquivalenceMode ? 0.0 : bsremBeta;
            updateParams.alpha =
                args.bsremMlemEquivalenceMode ? 1.0 : bsremAlpha;
            updateParams.subsetScale = static_cast<double>(stageSubsetCount);
            updateParams.mlemEquivalenceMode = args.bsremMlemEquivalenceMode;
            updateParams.clipRelativeStep = !args.bsremMlemEquivalenceMode;
            updateParams.denomEps = args.bsremMlemEquivalenceMode ? 0.0 : 1e-12;
            updateParams.sMin = std::max(1e-12, bsremSensFloorAbs);
            updateParams.nonneg = opts.nonneg;
            updateParams.threads = opts.threads;
            if (recordDiag) {
              rawSteps.assign(globalNvox, 0.0);
              precondSteps.assign(globalNvox, 0.0);
              updateParams.rawRelativeSteps = &rawSteps;
              updateParams.preconditionedSteps = &precondSteps;
            }
            applyBsremUpdate(image, numer, denom, subsetDenom, updateParams);
          }
          compactTiming.mlemUpdateSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - updateStart)
                  .count();

          if (recordDiag && metricsCsv) {
            BsremIterationInputs diagIn;
            diagIn.image = &image;
            diagIn.imagePrev = &imagePrev;
            diagIn.numerator = &numer;
            diagIn.globalDenom = &denom;
            diagIn.gradient = args.bsremMlemEquivalenceMode ? nullptr : gradPtr;
            diagIn.curvature =
                args.bsremMlemEquivalenceMode ? nullptr : curvPtr;
            diagIn.kappa = args.bsremMlemEquivalenceMode ? nullptr : kappaPtr;
            diagIn.rawRelativeSteps =
                rawSteps.empty() ? nullptr : &rawSteps;
            diagIn.preconditionedSteps =
                precondSteps.empty() ? nullptr : &precondSteps;
            diagIn.mask = maskPtr;
            diagIn.sMin = std::max(1e-12, bsremSensFloorAbs);
            diagIn.alpha = args.bsremMlemEquivalenceMode ? 1.0 : bsremAlpha;
            diagIn.beta = args.bsremMlemEquivalenceMode ? 0.0 : bsremBeta;
            if (useRegularization && !args.bsremMlemEquivalenceMode) {
              diagIn.regularizerRaw = spect::regularizer::compute_reg_energy_raw(
                  image, args.Nx, args.Ny, args.Nz, args.reg, maskForReg);
            }
            const auto diag = computeBsremIterationMetrics(it, diagIn);
            metricsCsv << formatBsremIterationMetricsCsv(diag) << '\n';
            metricsCsv.flush();
          }

          // In-iteration filters act once per iteration, after its last
          // subset update.
          const bool lastSubIteration = sub + 1 == stageSubsetCount;
          if (lastSubIteration && prepared.butterConf.enabled) {
            const auto butterStart = ProfileClock::now();
            filters::apply_butterworth_inplace(image, args.Nx, args.Ny,
                                               args.Nz, spacing,
                                               prepared.butterConf,
                                               opts.threads);
            compactTiming.butterworthSeconds +=
                std::chrono::duration<double>(ProfileClock::now() - butterStart)
                    .count();
          }

          if (lastSubIteration && prepared.smoothConf.enabled) {
            applySeparableConvRuntime(
                image, args.Nx, args.Ny, args.Nz, prepared.smoothConf.kx,
                prepared.smoothConf.ky, prepared.smoothConf.kz, opts.threads);
          }

          clampNonNegativeInPlace(image, opts.threads);
          if (maskPtr)
            spect::core::applyMaskInplace(image, *maskPtr);
          guardZeroMaxCheckpoint(
              image, "compact iteration " + std::to_string(it) +
                         (stageSubsetCount > 1
                              ? " subset " + std::to_string(subsetSpec.index)
                              : std::string()));
          imageResident = false;
          }
          logInfo("[SUBSET_TIME] iteration=" + std::to_string(it) +
                  " subset=" + std::to_string(subsetSpec.index) + " wall_s=" +
                  std::to_string(std::chrono::duration<double>(
                      ProfileClock::now() - subsetWallStart).count()));
          } // sub-iterations
          lastIterationDone = ProfileClock::now();
          logInfo("[ITER_TIME] iteration=" + std::to_string(it) + " wall_s=" +
                  std::to_string(std::chrono::duration<double>(
                      lastIterationDone - iterationWallStart).count()));
          observeCompactMemory();

          if (args.saveInterval > 0 &&
              it % args.saveInterval == 0 && it != args.numIters) {
            saveCompactOutput(it);
          }
        }
        }

        if (imageResident) {
          if (!spect::cuda::downloadCompactResidentImageCUDA(
                  image.data(), image.size())) {
            throw std::runtime_error(
                "compact final resident image download failed.");
          }
          imageResident = false;
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
              << "window,bed,angle,det_z,count,measured_sum,expected_sum,"
                 "relative_bias,poisson_deviance_per_bin,pearson_rms\n";
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
           for (size_t channel = 0; channel < channelCount; ++channel) {
            // Window mean = w_k * A_k x + b_k.
            std::vector<double> expected(fullRows, 0.0);
            const uint16_t *bedAttenuation =
                compactAttFactorPtrs.empty()
                    ? nullptr
                    : compactAttFactorPtrs[static_cast<size_t>(bed)];
            const bool forwardOk =
                spect::cuda::computeCompactChunkForwardProjectionCUDA(
                    *channels[channel].chunks, compactMeta,
                    localImageForProjection, opts.threads, expected.data(),
                    bedAttenuation, channels[channel].attenPower);
            if (!forwardOk) {
              throw std::runtime_error(
                  "compact projection residual diagnostic requires CUDA "
                  "sparse backend.");
            }
            const double windowWeight = channels[channel].weight;
            for (double &value : expected)
              value *= windowWeight;

            const size_t projectionOffset =
                static_cast<size_t>(bed) * fullRows;
            const double *measured =
                channelProjections[channel]->data() + projectionOffset;
            const auto *channelBackground = channelBackgrounds[channel];
            const double *background =
                (channelBackground == nullptr || channelBackground->empty())
                    ? nullptr
                    : channelBackground->data() + projectionOffset;
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
                residualOut << channel << ',' << bed << ',' << angle << ','
                            << detZ << ',' << rowStats.count << ','
                            << rowStats.measuredSum << ','
                            << rowStats.expectedSum << ','
                            << projectionRelativeBias(rowStats) << ','
                            << projectionDeviancePerBin(rowStats) << ','
                            << projectionPearsonRms(rowStats) << '\n';
              }
            }
            std::ostringstream metric;
            metric << std::setprecision(12)
                   << "[METRIC] compact_projection_residual window=" << channel
                   << " bed=" << bed
                   << " count=" << bedStats.count
                   << " measured_sum=" << bedStats.measuredSum
                   << " expected_sum=" << bedStats.expectedSum
                   << " relative_bias=" << projectionRelativeBias(bedStats)
                   << " poisson_deviance_per_bin="
                   << projectionDeviancePerBin(bedStats)
                   << " pearson_rms=" << projectionPearsonRms(bedStats);
            logDebug(metric.str());
           }
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

        if (!spect::cuda::releaseCompactExecutionCUDA()) {
          throw std::runtime_error(
              "compact final CUDA resource release requires CUDA backend.");
        }

        if (!finalOutputSaved) {
          saveCompactOutput(args.numIters);
          observeCompactMemory();
        }
        if (!finalManifestWritten) {
          throw std::runtime_error(
              "compact final run manifest was not written");
        }

        compactTiming.totalLoadToIterationDoneSeconds =
            std::chrono::duration<double>(lastIterationDone -
                                          loadToIterationStart)
                .count();
        compactTiming.fullCommandWallSeconds =
            std::chrono::duration<double>(ProfileClock::now() -
                                          fullCommandStart)
                .count();

        std::ostringstream perfSummary;
        perfSummary << std::fixed << std::setprecision(3)
                    << "[PERF] denom_cache="
                    << (compactDenomCachePath.empty()
                            ? "disabled"
                            : (loadedCompactDenomCache ? "hit" : "miss"))
                    << " projection_load_s="
                    << compactTiming.projectionLoadSeconds
                    << " csr_open_s="
                    << compactTiming.csrMapOrLoadSeconds
                    << " denom_and_matrix_prepare_s="
                    << compactTiming.compactSensitivitySeconds
                    << " iteration_core_s="
                    << (compactTiming.compactForwardRatioBackwardSeconds +
                        compactTiming.mlemUpdateSeconds +
                        compactTiming.regGradientSeconds +
                        compactTiming.butterworthSeconds)
                    << " load_to_iteration_done_s="
                    << compactTiming.totalLoadToIterationDoneSeconds
                    << " full_command_s="
                    << compactTiming.fullCommandWallSeconds;
        logInfo(perfSummary.str());
        const auto reads = matrixReadStats();
        logInfo("[MATRIX_IO] scope=iterations csrb_matrix_h2d_bytes=" +
                std::to_string(spect::cuda::getCudaChunkStageTimingSnapshotCUDA().csrbMatrixH2DBytes -
                               h2dBeforeIterations));
        logInfo("[MATRIX_IO] scope=iterations application_payload_read_bytes=" +
                std::to_string(reads.payloadReadBytes - readsBeforeIterations.payloadReadBytes) +
                " payload_loads=" + std::to_string(reads.payloadLoads - readsBeforeIterations.payloadLoads) +
                " host_reuse_bytes=" + std::to_string(reads.hostReuseBytes - readsBeforeIterations.hostReuseBytes) +
                " empty_slices_skipped=" + std::to_string(reads.emptySlicesSkipped - readsBeforeIterations.emptySlicesSkipped));
        logInfo("[MATRIX_IO] scope=run application_payload_read_bytes=" +
                std::to_string(reads.payloadReadBytes) +
                " payload_loads=" + std::to_string(reads.payloadLoads));

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


} // namespace spect::main
