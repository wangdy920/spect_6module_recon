#include "pipeline/MainProgram.h"
#include <cstdlib>

#include "recon/CompactSymmetricReconLoop.h"

#include "recon/BedOffsets.h"
#include "cli/ArgumentParser.h"
#include "cli/ExitCodes.h"
#include "cli/Version.h"
#include "core/Masks.h"
#include "core/BedPathUtils.h"
#include "core/ReconLimits.h"
#include "core/SixModuleGeometry.h"
#include "sparse/CompactSymmetricProjector.h"
#include "cuda/CudaConfig.h"
#if defined(SPECT_HAS_CUDA)
#include "cuda/CudaFatbinLoader.h"
#endif
#include "filters/ButterworthFilter.h"
#include "io/BadDetectorMask.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "io/MHDWriter.h"
#include "io/ProjectionLoader.h"
#include "io/VolumeLoader.h"
#include "ctac/AutoMuMap.h"
#include "ctac/CTACFactorCache.h"
#include "ctac/CTACOps.h"
#include "pipeline/ReconDataLoader.h"
#include "pipeline/ReconArgsPrinter.h"
#include "pipeline/ReconPreparation.h"
#include "pipeline/ReconSetup.h"
#include "recon/ReconLoopHelpers.h"
#include "pipeline/RunLogging.h"
#include "pipeline/RunLogger.h"
#include "postprocess/SmoothingOps.h"
#include "pipeline/VolumeOutput.h"
#include "regularizer/RegularizerCompat.h"
#include "sparse/SparseOpsCUDA.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
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

void reportConfigStatus(const ReconArgs &args) {
  if (args.configExplicit) {
    if (!args.configFound) {
      logWarn("Specified config file '" + args.configPath +
              "' was not found. Using internal defaults.");
    } else if (args.configLoaded) {
      logDebug("Loaded configuration from " + args.configPath);
    } else {
      logWarn("Failed to load config file '" + args.configPath +
              "'. Using internal defaults.");
    }
    return;
  }

  if (args.configFound && args.configLoaded) {
    logDebug("Loaded configuration from " + args.configPath);
  } else if (!args.configFound) {
    logInfo("Default config not found in exe-local config/ or cwd fallbacks. Using internal defaults.");
  } else {
    logWarn("Failed to load default config '" + args.configPath +
            "'. Using internal defaults.");
  }
}

int parseRequestedCudaDevice(const ReconArgs &args, int deviceCount) {
  const std::string source = args.cudaDeviceFromCli ? "-cudaDevice" : "SPECT_CUDA_DEVICE";
  const std::string &value = args.cudaDeviceValue;
  bool invalid = value.empty();
  long long parsed = -1;
  if (!invalid) {
    try {
      size_t consumed = 0;
      parsed = std::stoll(value, &consumed, 10);
      invalid = consumed != value.size();
    } catch (const std::exception &) {
      invalid = true;
    }
  }
  if (invalid || parsed < 0 || parsed >= deviceCount) {
    throw std::runtime_error("Invalid CUDA device id '" + value + "' from " +
                             source + "; available CUDA device count is " +
                             std::to_string(deviceCount) + ".");
  }
  return static_cast<int>(parsed);
}

#if defined(SPECT_HAS_CUDA)
void reportCudaRuntimeEnvironment(int deviceId) {
  const auto info = spect::cuda::queryEnvironment(deviceId);
  spect::cuda::ensureDriverSupportsBundledRuntime(info);

  const std::string smTag = "sm_" + std::to_string(info.computeMajor) +
                            std::to_string(info.computeMinor);
  logInfo("CUDA environment: device=" + std::to_string(info.deviceId) + "/" +
          std::to_string(info.deviceCount) + " name=\"" + info.deviceName +
          "\" " + smTag + ", driver_cuda=" +
          spect::cuda::formatCudaApiVersion(info.driverVersion) +
          ", runtime_cuda=" +
          spect::cuda::formatCudaApiVersion(info.runtimeVersion) +
          " (bundled runtime requires driver CUDA >= 13.0 / R580+)");

  if (info.driverVersion < info.runtimeVersion) {
    logWarn("CUDA driver (" +
            spect::cuda::formatCudaApiVersion(info.driverVersion) +
            ") is older than bundled runtime (" +
            spect::cuda::formatCudaApiVersion(info.runtimeVersion) +
            "). Prefer upgrading the display driver to the CUDA " +
            spect::cuda::formatCudaApiVersion(info.runtimeVersion) +
            " toolkit driver floor or newer for maximum stability.");
  }

  // Frozen field/dev GPUs: helper kernels are sm_86/89/120 fatbins.
  if (!spect::cuda::isFrozenConsumerArch(info.computeMajor, info.computeMinor)) {
    throw std::runtime_error(
        "GPU compute capability " + smTag +
        " is outside the frozen field set sm_86/89/120. Helper kernels are "
        "build-time fatbins; runtime NVRTC is disabled.");
  }
}
#endif

int selectCudaDeviceIfRequested(const ReconArgs &args) {
#if defined(SPECT_HAS_CUDA)
  const int deviceCount = spect::cuda::getDeviceCount();
  int deviceId = 0;
  if (args.cudaDeviceRequested) {
    deviceId = parseRequestedCudaDevice(args, deviceCount);
    const std::string source =
        args.cudaDeviceFromCli ? "-cudaDevice" : "SPECT_CUDA_DEVICE";
    logDebug("CUDA device selected from " + source + ": " +
            std::to_string(deviceId) + " (device count=" +
            std::to_string(deviceCount) + ")");
  } else if (deviceCount > 1) {
    logDebug("CUDA device defaulted to 0 (device count=" +
            std::to_string(deviceCount) +
            "); override with -cudaDevice or SPECT_CUDA_DEVICE.");
  }
  spect::cuda::setDevice(deviceId);
  reportCudaRuntimeEnvironment(deviceId);
  return deviceId;
#else
  if (args.cudaDeviceRequested) {
    const int deviceCount = 0;
    (void)parseRequestedCudaDevice(args, deviceCount);
  }
  return 0;
#endif
}

class GpuExecutionLease {
public:
  explicit GpuExecutionLease(int deviceId) {
#ifdef _WIN32
    const std::wstring name =
        L"Global\\SPECT_recon_GPU_" + std::to_wstring(deviceId);
    SetLastError(ERROR_SUCCESS);
    mutex_ = CreateMutexW(nullptr, TRUE, name.c_str());
    if (mutex_ == nullptr) {
      throw std::runtime_error(
          "Failed creating the SPECT_recon GPU process lock (Windows error " +
          std::to_string(GetLastError()) + ").");
    }
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
      CloseHandle(mutex_);
      mutex_ = nullptr;
      throw std::runtime_error(
          "GPU " + std::to_string(deviceId) +
          " is already owned by another SPECT_recon process. Wait for it to "
          "finish or select another GPU with -cudaDevice.");
    }
#else
    (void)deviceId;
#endif
  }

  ~GpuExecutionLease() {
#ifdef _WIN32
    if (mutex_ != nullptr) {
      ReleaseMutex(mutex_);
      CloseHandle(mutex_);
    }
#endif
  }

  GpuExecutionLease(const GpuExecutionLease &) = delete;
  GpuExecutionLease &operator=(const GpuExecutionLease &) = delete;

private:
#ifdef _WIN32
  HANDLE mutex_ = nullptr;
#endif
};

void validateInputMapContract(const std::string &path,
                              const std::string &label,
                              const std::string &role,
                              const std::string &zLayout,
                              const std::string &units,
                              const std::vector<int> &beds) {
  std::string extension = std::filesystem::path(path).extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (extension != ".mhd") {
    if (beds.size() > 1)
      throw std::runtime_error(label +
                               " must be a contract-tagged MHD for multi-bed reconstruction.");
    logWarn(label + " is not MHD; map role and bed identity cannot be checked.");
    return;
  }
  const auto info = spect::io::parseMHD(path);
  if (!spect::io::validateSpectMapContract(info, role, zLayout, units, beds)) {
    if (beds.size() > 1)
      throw std::runtime_error(
          label + " is missing SPECT map role/bed/layout tags; regenerate it "
                  "with scripts/prepare_ct_mu_map.py.");
    logWarn(label +
            " has no SPECT map contract tags; accepting legacy single-bed input.");
  }
}

void reverseZWithinBedBlocks(std::vector<double> &volume, int nx, int ny,
                             int baseNz, int bedCount) {
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t bedSize = plane * static_cast<size_t>(baseNz);
  if (volume.size() != bedSize * static_cast<size_t>(bedCount))
    throw std::runtime_error("Cannot reverse per-bed Z: volume size mismatch.");
  for (int bed = 0; bed < bedCount; ++bed) {
    const size_t bedOffset = static_cast<size_t>(bed) * bedSize;
    for (int z = 0; z < baseNz / 2; ++z) {
      const size_t front = bedOffset + static_cast<size_t>(z) * plane;
      const size_t back = bedOffset +
                          static_cast<size_t>(baseNz - 1 - z) * plane;
      std::swap_ranges(volume.begin() + static_cast<std::ptrdiff_t>(front),
                       volume.begin() + static_cast<std::ptrdiff_t>(front + plane),
                       volume.begin() + static_cast<std::ptrdiff_t>(back));
    }
  }
}

} // namespace

int runProgram(int argc, char **argv) {
  const auto fullCommandStart = ProfileClock::now();
  if (argc <= 1) {
    spect::cli::ArgumentParser::printUsage();
    return spect::cli::kExitSuccess;
  }

  ReconArgs args;
  spect::cli::ArgumentParser parser;
  const auto parseResult = parser.parse(argc, argv, args);
  if (parseResult == spect::cli::ArgumentParser::ParseResult::ExitSuccess)
    return spect::cli::kExitSuccess;
  if (parseResult != spect::cli::ArgumentParser::ParseResult::Success)
    return spect::cli::kExitUserError;

  setVerbosity(std::clamp(args.verbosity, 0, 2));
  args.verbosity = getVerbosity();

  // Log and MHD/RAW share one resolved output directory.  Absolute `-fout`/
  // `-o` (drive letter / UNC) stay on that absolute path; relative dirs are
  // pinned to this process CWD via absolute() so both artifacts agree.
  std::string logDirError;
  auto logDir = resolveLogDirectory(args, logDirError);
  if (logDir.empty())
    logDir = std::filesystem::absolute(".");
  args.outputDir = logDir.string();
  ReconArgs logArgs = args;
  // Build the path once: the failure branch used to call makeDefaultLogPath
  // again and print a different timestamp than the open that actually failed.
  const std::filesystem::path logPath = makeDefaultLogPath(logArgs);
  ConsoleTeeGuard tee(logPath);
  if (tee.enabled()) {
    // Announce via logInfo so -vb 0 still lands in the durable file (file sink)
    // and -vb>=1 reaches console+file through the tee.  Avoid a second raw
    // cerr line that would duplicate the path in the log.
    logInfo("Logging console output to: " + tee.logPath().string());
    if (!logDirError.empty()) {
      logWarn("Failed to create output directory '" + args.outputDir +
              "' for logging (" + logDirError + "); using current directory.");
    }
  } else {
    logWarn("Failed to create log file at: " + logPath.string() +
            (tee.errorMessage().empty() ? "" : " (" + tee.errorMessage() + ")"));
  }

  {
    // Matrix/config resolve vs exe dir; volumes/logs resolve vs CWD. Print the
    // absolute output directory once so a scheduler CWD cannot hide outputs.
    std::error_code absError;
    const auto absOut = std::filesystem::absolute(
        args.outputDir.empty() ? std::filesystem::path("output")
                               : std::filesystem::path(args.outputDir),
        absError);
    const std::string outDir =
        absError ? (args.outputDir.empty() ? "output" : args.outputDir)
                 : absOut.string();
    const std::string outMsg =
        "Output directory: " + outDir + " (name prefix: " +
        (args.outputName.empty() ? "recon" : args.outputName) + ")";
    logInfo(outMsg);
  }

  const std::string commandLine = makeCommandLineString(argc, argv);
  logInfo(std::string("=== SPECT_recon ") + spect::cli::versionString() +
          " ===");
  logInfo("[CLI] Requested command (before config/metadata overrides): " + commandLine);
  reportConfigStatus(args);

  int cudaDeviceId = 0;
  std::unique_ptr<GpuExecutionLease> gpuLease;
  try {
    for (const char *name : {"SPECT_COMPACT_BRANCH_BATCH", "SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES"}) {
      const char *value = std::getenv(name);
      if (value != nullptr && value[0] != '\0')
        throw std::runtime_error(std::string(name) +
            " is retired; unset it. Reproduce legacy experiments with their frozen executable.");
    }
    cudaDeviceId = selectCudaDeviceIfRequested(args);
    gpuLease = std::make_unique<GpuExecutionLease>(cudaDeviceId);
#ifdef _WIN32
    logDebug("GPU process lock acquired: device=" +
            std::to_string(cudaDeviceId) +
            " pid=" + std::to_string(GetCurrentProcessId()));
#else
    logWarn("GPU process lock is unavailable on this non-Windows build.");
#endif
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return spect::cli::kExitUserError;
  }

#ifdef _OPENMP
  omp_set_num_threads(std::max(1, args.numThreads));
#endif

  try {
    CompactExecutionTiming compactTiming;
    const auto loadToIterationStart = ProfileClock::now();
    const std::string compactMetaPath = resolveCompactMetadataPath(args);
    spect::io::CompactSymmetryMetadata compactMeta =
        spect::io::loadCompactSymmetryMetadata(compactMetaPath);
    if (args.dualWindow()) {
      const auto secondMetadataPath =
          (std::filesystem::path(args.chunkCSRDir2) / "metadata.bin").string();
      const auto secondMeta =
          spect::io::loadCompactSymmetryMetadata(secondMetadataPath);
      spect::io::validateCompactMetadataCompatibility(compactMeta, secondMeta);
      logInfo("Joint matrix metadata validated: " + secondMetadataPath);
    }
    if (compactMeta.legacyDoubleVoxelSizeLayout) {
      logWarn("Compatibility: loaded 92-byte legacy compact metadata with "
              "float64 voxelSize; converted to the canonical in-memory "
              "SYM_90Z contract after validation: " + compactMetaPath);
    }
    // Matrix metadata is authoritative for reconstruction dims and voxel
    // size: it always overrides -dim/-vox (the device command line may carry
    // stale or wrong values). Warn when an explicit -dim disagrees so the
    // mismatch is visible on site.
    if (args.dimExplicit &&
        (args.Nx != compactMeta.Nx || args.Ny != compactMeta.Ny ||
         args.Nz != compactMeta.Nz)) {
      logWarn("CLI -dim " + std::to_string(args.Nx) + "," +
              std::to_string(args.Ny) + "," + std::to_string(args.Nz) +
              " disagrees with matrix metadata " +
              std::to_string(compactMeta.Nx) + "," +
              std::to_string(compactMeta.Ny) + "," +
              std::to_string(compactMeta.Nz) +
              "; matrix metadata wins (reconstruction geometry follows the "
              "system matrix).");
    }
    applyCompactMetadataGeometry(args, compactMeta);
    logDebug("compact metadata geometry loaded from " + compactMetaPath +
            " (mode=" + std::string(spect::io::compactSymmetryModeName(compactMeta)) +
            ", compactRows=" + std::to_string(compactMeta.compactRows()) +
            ", fullRows=" + std::to_string(compactMeta.fullRows()) + ")");

    std::string dimHintWarning;
    if (adoptReconDimsFromHint(args, &dimHintWarning) && !dimHintWarning.empty())
      logWarn(dimHintWarning);

    std::string dimsWarning;
    if (ensureReconDims(args, &dimsWarning) && !dimsWarning.empty())
      logWarn(dimsWarning);

    prepareProjectionInputs(args);
    const BedSetupResult bedSetup = prepareBedSetup(args);
    for (const std::string &warning : bedSetup.warnings)
      logWarn(warning);
    if (args.useReg && !args.regCtPath.empty())
      validateInputMapContract(
          args.regCtPath, "Registered CT",
          "regularizer_reconstruction_storage", "reconstruction_storage_z",
          "HU", bedSetup.bedNumbers);
    if ((args.doCTAC || !std::isnan(args.maskFromAttThreshold)) &&
        !args.attMapPath.empty())
      validateInputMapContract(
          args.attMapPath, "Attenuation map", "attenuation_local_matrix_z",
          "local_matrix_z", "mm^-1", bedSetup.bedNumbers);
    logInfo(bedSetup.summary);

    std::filesystem::create_directories(args.outputDir);

    const std::vector<std::string> projPaths = bedSetup.projectionPaths;
    if (projPaths.empty() || projPaths.front().empty())
      throw std::runtime_error("Projection path is required.");

    auto stageStart = ProfileClock::now();
    LoadedProjectionSet projectionSet =
        loadProjectionSet(projPaths, compactMeta, "Projection Data", "matrix",
                          args.mergeTimeSegments);
    args.projectionSourceGroups = projectionSet.sourceGroups;
    compactTiming.projectionLoadSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
    if (projectionSet.data.empty())
      throw std::runtime_error("Projection data is empty after loading.");

    args.numAngles = projectionSet.numAngles;
    if (projectionSet.detectorShapeKnown) {
      args.detNz = projectionSet.detNz;
      args.detNx = projectionSet.detNx;
    } else {
      if (projectionSet.numAngles <= 0 ||
          projectionSet.raysPerBed % static_cast<size_t>(projectionSet.numAngles) != 0) {
        throw std::runtime_error("Projection BIN geometry is inconsistent.");
      }
      const size_t raysPerAngle =
          projectionSet.raysPerBed / static_cast<size_t>(projectionSet.numAngles);
      const size_t configuredRays =
          (args.detNz > 0 && args.detNx > 0)
              ? static_cast<size_t>(args.detNz) * static_cast<size_t>(args.detNx)
              : 0;
      if (configuredRays == raysPerAngle) {
        logDebug("Projection BIN does not store detZ/detX; using configured detector shape " +
                std::to_string(args.detNz) + "x" + std::to_string(args.detNx) + ".");
      } else {
        if (raysPerAngle > static_cast<size_t>(std::numeric_limits<int>::max()))
          throw std::runtime_error("Projection BIN rays_per_angle exceeds int range.");
        args.detNz = 1;
        args.detNx = static_cast<int>(raysPerAngle);
        logWarn("Projection BIN does not store detZ/detX and configured -detN does not match rays_per_angle. Falling back to 1x" +
                std::to_string(args.detNx) +
                "; set -detN <detZ,detX> when projection-domain PSF or detector geometry is required.");
      }
    }
    const int bedCount = std::max(1, bedSetup.bedCount);
    const int baseNx = bedSetup.baseNx;
    const int baseNy = bedSetup.baseNy;
    const int baseNz = bedSetup.baseNz;
    const size_t baseNvox = static_cast<size_t>(baseNx) * static_cast<size_t>(baseNy) *
                            static_cast<size_t>(baseNz);
    const size_t globalNvox = static_cast<size_t>(args.Nx) * static_cast<size_t>(args.Ny) *
                              static_cast<size_t>(args.Nz);
    const bool reverseLocalZInReconStorage = args.zBedFlip && bedCount > 1;

    printReconArgs(args);
    logInfo("Loaded projection rays: total=" +
            std::to_string(projectionSet.data.size()) + ", per bed=" +
            std::to_string(projectionSet.raysPerBed));

    auto loadBackgroundFor = [&](const std::vector<std::string> &listPaths,
                                 const std::string &singlePath,
                                 const char *label) {
      std::vector<double> background;
      std::vector<std::string> backgroundPaths = listPaths;
      if (backgroundPaths.empty() && !singlePath.empty())
        backgroundPaths = {singlePath};
      if (backgroundPaths.empty())
        return background;
      if (bedCount > 1 && backgroundPaths.size() != static_cast<size_t>(bedCount))
        throw std::runtime_error("Background projection count must match bed count.");
      if (bedCount == 1 && backgroundPaths.size() > 1)
        throw std::runtime_error("Multiple background projections were provided for single-bed mode.");

      const LoadedProjectionSet backgroundSet =
          loadProjectionSet(backgroundPaths, compactMeta, label, "b_proj");
      background = backgroundSet.data;
      logInfo(std::string("Loaded ") + label + " term with " +
              std::to_string(background.size()) + " rays.");
      return background;
    };
    std::vector<double> backgroundProj =
        loadBackgroundFor(args.bProjPaths, args.bProjPath, "Background Projection");

    // Joint dual-window: the second window's projections mirror the first
    // bed for bed and go through the same masking below.
    std::vector<double> projections2;
    std::vector<double> backgroundProj2;
    if (args.dualWindow()) {
      const std::vector<std::string> projPaths2 =
          args.projPaths2.empty() ? std::vector<std::string>{args.projPath2}
                                  : args.projPaths2;
      LoadedProjectionSet projectionSet2 =
          loadProjectionSet(projPaths2, compactMeta,
                            "Projection Data (window 2)", "matrix",
                            args.mergeTimeSegments);
      if (projectionSet2.data.size() != projectionSet.data.size()) {
        throw std::runtime_error(
            "Second-window projections do not match the first window's "
            "bed count.");
      }
      projections2 = std::move(projectionSet2.data);
      backgroundProj2 = loadBackgroundFor(args.bProjPaths2, args.bProjPath2,
                                          "Background Projection (window 2)");
      logInfo("Joint dual-window reconstruction: window weights " +
              std::to_string(args.energyWindowWeights[0]) + ", " +
              std::to_string(args.energyWindowWeights[1]) + ".");
    }

    const auto spacingArray = voxelSpacingFromArgs(args);
    const filters::Spacing3 spacing{spacingArray[0], spacingArray[1], spacingArray[2]};
    const OutputGeometry outputGeometry{args.Nx, args.Ny, args.Nz, spacing.dx,
                      spacing.dy, spacing.dz};

    std::vector<double> sharedAnatomical;
    if (args.useReg && !args.regCtPath.empty()) {
      logInfo("Loading registered CT for TV/RDP weights from " + args.regCtPath);
      sharedAnatomical = loadVolumeChecked(args.regCtPath, "Registered CT",
                                          args.Nx, args.Ny, args.Nz, spacing, false);
    }

    std::vector<uint8_t> mask;
    const std::vector<uint8_t> *maskPtr =
        maybeMakeCylindricalMask(mask, args.Nx, args.Ny, args.Nz, spacingArray, args);

    std::vector<double> ctacMuMap;
    if (args.doCTAC || !std::isnan(args.maskFromAttThreshold)) {
      if (!args.attMapPath.empty()) {
        logInfo("Loading attenuation map from " + args.attMapPath);
        ctacMuMap = loadVolumeChecked(args.attMapPath, "Attenuation map",
                                      args.Nx, args.Ny, args.Nz, spacing, true);
      } else if (args.doCTAC && !args.ctacAutoMuFrom.empty()) {
        logInfo("Auto mu-map: segmenting early-iteration reconstruction " +
                args.ctacAutoMuFrom);
        logWarn("Auto mu-map assumes activity roughly fills the attenuating "
                "volume (water-equivalent phantom). For hot-rods-in-cold-body "
                "objects supply a measured -attMap instead.");
        const std::vector<double> earlyRecon = loadAutoMuActivity(
            args.ctacAutoMuFrom, args.Nx, args.Ny, args.Nz, spacing);
        const spect::main::AutoMuMapResult autoMu =
            spect::main::buildAutoMuMapFromActivity(
                earlyRecon, args.Nx, args.Ny, args.Nz, args.ctacAutoMuThresh,
                args.ctacAutoMuValue, maskPtr);
        if (autoMu.bodyVoxels == 0)
          throw std::runtime_error(
              "Auto mu-map segmentation found no body voxels; check "
              "-ctacAutoMu input volume and -ctacAutoMuThresh.");
        logInfo("Auto mu-map: body voxels=" +
                std::to_string(autoMu.bodyVoxels) + ", threshold=" +
                std::to_string(autoMu.thresholdUsed) + " (P99.9=" +
                std::to_string(autoMu.robustMax) + "), mu=" +
                std::to_string(args.ctacAutoMuValue) + " mm^-1 (" +
                args.ctacAutoMuValueSource + ").");
        ctacMuMap = autoMu.muMap;
        saveVolumeOutput(ctacMuMap, outputGeometry, args.outputDir,
                         outputPrefixFromArgs(args) + "_automu", -1, nullptr,
                         sanitizeCropSettings(0, 0));
        if (reverseLocalZInReconStorage) {
          reverseZWithinBedBlocks(ctacMuMap, baseNx, baseNy, baseNz, bedCount);
          logInfo("Auto mu-map converted from reconstruction-storage Z to "
                  "local system-matrix Z for CTAC.");
        }
      } else {
        throw std::runtime_error(
            args.doCTAC
                ? "CTAC requires -attMap <file> or -ctacAutoMu <early recon volume>."
                : "-maskFromAtt requires -attMap <file>.");
      }
    }
    if (args.useReg) {
      if (!sharedAnatomical.empty())
        args.reg.ct_image = &sharedAnatomical;
      if (args.reg.ct_image) {
        logInfo("TV/RDP CT weighting enabled: sigma=" +
                std::to_string(args.reg.ct_sigma) + ", floor=" +
                std::to_string(args.reg.ct_weight_floor) + ".");
      } else {
        logInfo("TV/RDP uses activity-only neighborhood weights.");
      }
    }
    if (!std::isnan(args.maskFromAttThreshold)) {
      const std::vector<double> *maskSource = nullptr;
      std::vector<double> attenuationStorageMap;
      if (!ctacMuMap.empty()) {
        attenuationStorageMap = ctacMuMap;
        if (reverseLocalZInReconStorage)
          reverseZWithinBedBlocks(attenuationStorageMap, baseNx, baseNy,
                                  baseNz, bedCount);
        maskSource = &attenuationStorageMap;
      }
      if (!maskSource)
        throw std::runtime_error("-maskFromAtt requires -attMap.");

      if (mask.empty())
        mask.assign(globalNvox, 1);
      for (size_t i = 0; i < mask.size(); ++i) {
        if ((*maskSource)[i] <= args.maskFromAttThreshold)
          mask[i] = 0;
      }
      maskPtr = &mask;
      logInfo("Applied attenuation-derived body mask.");
    }

    if (maskPtr) {
      const std::vector<double> maskVolume(mask.begin(), mask.end());
      saveVolumeOutput(maskVolume, outputGeometry, args.outputDir,
                       outputPrefixFromArgs(args) + "_mask", -1, nullptr,
                       sanitizeCropSettings(0, 0));
    }

    std::unique_ptr<spect::io::CompactChunkCSRDirectory> compactChunks;

    stageStart = ProfileClock::now();
    logDebug("Opening compact chunk CSR directory from " + args.chunkCSRDir);
    {
      auto opened =
          spect::io::openCompactChunkCSRDirectory(args.chunkCSRDir, compactMeta);
      logDebug("Compact chunk CSR directory opened: chunks=" +
              std::to_string(opened.chunks().size()) +
              ", compactRows=" + std::to_string(opened.totalRows()) +
              ", nnz=" + std::to_string(opened.totalNnz()) +
              ", hostCacheBudgetBytes=" +
              std::to_string(opened.cacheBudgetBytes()));
      compactChunks =
          std::make_unique<spect::io::CompactChunkCSRDirectory>(std::move(opened));
    }
    std::unique_ptr<spect::io::CompactChunkCSRDirectory> compactChunks2;
    if (args.dualWindow()) {
      logDebug("Opening second-window compact chunk directory from " +
               args.chunkCSRDir2);
      auto opened2 = spect::io::openCompactChunkCSRDirectory(args.chunkCSRDir2,
                                                             compactMeta);
      if (opened2.format() != compactChunks->format()) {
        throw std::runtime_error(
            "Joint dual-window reconstruction requires the same matrix format for both windows.");
      }
      logDebug("Second-window chunk directory opened: chunks=" +
               std::to_string(opened2.chunks().size()) +
               ", nnz=" + std::to_string(opened2.totalNnz()));
      compactChunks2 = std::make_unique<spect::io::CompactChunkCSRDirectory>(
          std::move(opened2));
      if (compactChunks->format() != spect::io::CompactChunkFormat::CsrqRunU16) {
        uint64_t bytes[2] = {};
        const spect::io::CompactChunkCSRDirectory *dirs[] = {compactChunks.get(), compactChunks2.get()};
        for (int k = 0; k < 2; ++k)
          for (const auto &info : dirs[k]->chunks()) {
            if (info.fileSizeBytes > std::numeric_limits<uint64_t>::max() - bytes[k])
              throw std::runtime_error("Joint matrix byte count overflow.");
            bytes[k] += info.fileSizeBytes;
          }
        if (bytes[1] > std::numeric_limits<uint64_t>::max() - bytes[0])
          throw std::runtime_error("Joint matrix byte count overflow.");
        const uint64_t total = bytes[0] + bytes[1];
        const uint64_t budget = std::min(total,
            spect::io::defaultCompactChunkHostCacheBudgetBytes(total));
        const uint64_t firstBudget = budget == total ? bytes[0] :
            static_cast<uint64_t>(static_cast<long double>(budget) * bytes[0] / total);
        // Divide one host budget, before either directory allocates payloads.
        compactChunks = std::make_unique<spect::io::CompactChunkCSRDirectory>(
            compactChunks->directory(), compactChunks->chunks(), compactChunks->cols(),
            firstBudget, compactChunks->branchCount());
        compactChunks2 = std::make_unique<spect::io::CompactChunkCSRDirectory>(
            compactChunks2->directory(), compactChunks2->chunks(), compactChunks2->cols(),
            budget - firstBudget, compactChunks2->branchCount());
        logInfo("Joint CSRB host cache budgets: " + std::to_string(firstBudget) +
                " + " + std::to_string(budget - firstBudget) + " bytes (one shared cap).");
      }
    }
    compactTiming.csrMapOrLoadSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();

    const int matrixCols = compactChunks->cols();
    if (matrixCols != static_cast<int>(baseNvox)) {
      throw std::runtime_error("System matrix voxel count does not match the per-bed reconstruction grid.");
    }

    stageStart = ProfileClock::now();
    spect::io::validateCompactChunkSymmetricInputs(
        compactMeta, compactChunks->totalRows(), compactChunks->cols(),
        projectionSet.raysPerBed, args.detNz, args.detNx, args.numAngles,
        baseNx, baseNy, baseNz);
    compactTiming.metadataLoadAndValidateSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
    logDebug("compact-symmetric layout enabled via " + compactMetaPath +
            " (mode=" + std::string(spect::io::compactSymmetryModeName(compactMeta)) +
            ", compactRows=" + std::to_string(compactMeta.totalRows) +
            ", fullRows=" + std::to_string(compactMeta.fullRows()) +
            ", matrix=chunkCSRDir)");
    if (bedCount > spect::core::kMaxBeds) {
      throw std::runtime_error(
          "compact chunk multi-bed CUDA path supports at most " +
          std::to_string(spect::core::kMaxBeds) + " beds.");
    }

    const size_t expectedRays =
        static_cast<size_t>(compactMeta.fullRows()) *
        static_cast<size_t>(bedCount);
    std::vector<double> projections = std::move(projectionSet.data);
    if (projections.size() != expectedRays) {
      throw std::runtime_error(
          "Compact-symmetric layout requires exact full projection length; "
          "expected " +
          std::to_string(expectedRays) + ", got " +
          std::to_string(projections.size()) + ".");
    }
    if (!backgroundProj.empty() && backgroundProj.size() != projections.size()) {
      throw std::runtime_error("Background projection size does not match projection size.");
    }

    std::vector<spect::io::BadDetectorId> badDetectors;
    if (!args.badDetectorFile.empty()) {
      badDetectors = spect::io::loadBadDetectorList(args.badDetectorFile);
      if (!badDetectors.empty()) {
        const std::vector<uint8_t> badDetectorMask =
            spect::io::buildBadDetectorProjectionMask(
                badDetectors, compactMeta.fullNumAngles, compactMeta.detNz,
                compactMeta.detNx);
        const size_t fullRows = static_cast<size_t>(compactMeta.fullRows());
        const size_t maskedRows = static_cast<size_t>(std::count(
            badDetectorMask.begin(), badDetectorMask.end(), uint8_t{1}));
        auto zeroMaskedRows = [&](std::vector<double> &target) {
          if (target.empty())
            return;
          for (int bed = 0; bed < bedCount; ++bed) {
            const size_t bedOffset = static_cast<size_t>(bed) * fullRows;
            for (size_t row = 0; row < fullRows; ++row) {
              if (badDetectorMask[row])
                target[bedOffset + row] = 0.0;
            }
          }
        };
        zeroMaskedRows(projections);
        zeroMaskedRows(backgroundProj);
        zeroMaskedRows(projections2);
        zeroMaskedRows(backgroundProj2);
        std::ostringstream detectorNames;
        for (size_t index = 0; index < badDetectors.size(); ++index) {
          if (index != 0)
            detectorNames << ',';
          detectorNames
              << spect::io::badDetectorGlobalNumber(badDetectors[index]) << '='
              << spect::io::badDetectorName(badDetectors[index]);
        }
        logInfo("Bad detector mask: skipped " +
                std::to_string(badDetectors.size()) + " detector blocks (" +
                std::to_string(maskedRows) + " projection rows per bed): " +
                detectorNames.str());
      }
    }

    if (args.detZEdgeCensor > 0) {
      // CLI validation restricts detZEdgeCensor to the compact-symmetric chunk
      // layout, so compactMeta is loaded and projections are per-bed blocks of
      // fullRows. Censored detZ edge rows are unreliable: zero the measured
      // bins so the MLEM ratio y/(Ax) contributes nothing (numerator
      // censoring); the matching masked sensitivity is computed by the
      // compact reconstruction loop.
      const int censorK = args.detZEdgeCensor;
      if (2 * censorK >= compactMeta.detNz) {
        throw std::runtime_error(
            "-detZEdgeCensor must satisfy 2*k < detNz (k=" +
            std::to_string(censorK) +
            ", detNz=" + std::to_string(compactMeta.detNz) + ").");
      }
      const size_t fullRows =
          static_cast<size_t>(compactMeta.fullRows());
      const size_t raysPerAngle =
          static_cast<size_t>(compactMeta.fullRaysPerAngle());
      for (int bed = 0; bed < bedCount; ++bed) {
        size_t zeroedBins = 0;
        auto censorBed = [&](std::vector<double> &target, bool count) {
          if (target.empty())
            return;
          double *bedProj = target.data() + static_cast<size_t>(bed) * fullRows;
          for (int angle = 0; angle < compactMeta.fullNumAngles; ++angle) {
            double *angleBase =
                bedProj + static_cast<size_t>(angle) * raysPerAngle;
            for (int detZ = 0; detZ < compactMeta.detNz; ++detZ) {
              if (detZ >= censorK && detZ < compactMeta.detNz - censorK)
                continue;
              double *row =
                  angleBase + static_cast<size_t>(detZ) * compactMeta.detNx;
              std::fill(row, row + compactMeta.detNx, 0.0);
              if (count)
                zeroedBins += static_cast<size_t>(compactMeta.detNx);
            }
          }
        };
        censorBed(projections, true);
        censorBed(projections2, false);
        logInfo("detZ edge censoring: zeroed " + std::to_string(zeroedBins) +
                " measured bins in bed " + std::to_string(bed) +
                " (k=" + std::to_string(censorK) + ", detZ rows 0.." +
                std::to_string(censorK - 1) + " and " +
                std::to_string(compactMeta.detNz - censorK) + ".." +
                std::to_string(compactMeta.detNz - 1) + ").");
      }
    }

    const std::vector<int> bedZStarts =
        spect::app::computeBedZOffsets(bedCount, baseNz, args.bedOverlaps);

    logDebug("Bed coordinate contract: Bed1 = min physical Z, bed index "
            "increases toward +Z; configured z_bed_flip=" +
            std::to_string(args.zBedFlip ? 1 : 0) +
            ", effective multi-bed local Z flip=" +
            std::to_string(reverseLocalZInReconStorage ? 1 : 0) + ".");
    if (bedCount > 1) {
      logDebug("Bed physical step policy: canonical stitch uses " +
              std::to_string(bedSetup.canonicalGeom.stepLayers) +
              " layers / " +
              std::to_string(args.bedStepMm > 0.0
                                  ? args.bedStepMm
                                  : spect::core::sixmodule::kMultiBedStepMm) +
              " mm (crop=" +
              std::to_string(bedSetup.canonicalGeom.cropLayers) +
              " layers).");
    }
    double physicalZOffsetMm = 0.0;
    for (int bed = 0; bed < bedCount; ++bed) {
      const int start = bedZStarts[static_cast<size_t>(bed)];
      const std::string &bedFile = projPaths[static_cast<size_t>(bed)];
      const int localZeroStorageZ = reverseLocalZInReconStorage
                                        ? start + baseNz - 1
                                        : start;
      const int localLastStorageZ = reverseLocalZInReconStorage
                                        ? start
                                        : start + baseNz - 1;
      logDebug("Bed " + std::to_string(bedSetup.bedNumbers[static_cast<size_t>(bed)]) +
              " mapping: file=\"" +
              bedFile + "\", physical_z_offset_mm=" +
              std::to_string(physicalZOffsetMm) +
              " (first-selected-bed-relative), local_z_direction=" +
              (reverseLocalZInReconStorage ? "descending_storage_z"
                                           : "ascending_storage_z") +
              " (local z 0->" + std::to_string(baseNz - 1) +
              " maps storage z " + std::to_string(localZeroStorageZ) +
              "->" + std::to_string(localLastStorageZ) +
              "), recon_storage_z=[" + std::to_string(start) + ", " +
              std::to_string(start + baseNz) + ").");
      if (bed + 1 < bedCount)
        physicalZOffsetMm += bedSetup.canonicalGeom.stepLayers * spacing.dz;
    }

    std::vector<std::vector<uint16_t>> dynamicAttenFactors;
    std::unique_ptr<CTACFactorCache> ctacFactorCache;
    if (args.doCTAC) {
      logInfo("CTAC model: angle-dependent parallel-ray line integral "
              "(rotation sign " + std::to_string(args.ctacRotSign) +
              "). Attenuation map must be mu in mm^-1.");
        logInfo("Using dynamic CTAC attenuation factors.");
        const auto attenuationStart = std::chrono::steady_clock::now();
        dynamicAttenFactors.resize(static_cast<size_t>(bedCount));
        const std::filesystem::path cacheDir =
            args.ctacFactorCacheDir.empty()
                ? (std::filesystem::path("recon_out") / "ctac_factor_cache")
                : std::filesystem::path(args.ctacFactorCacheDir);
        ctacFactorCache = std::make_unique<CTACFactorCache>(cacheDir);
        const size_t plane =
            static_cast<size_t>(baseNx) * static_cast<size_t>(baseNy);

        std::vector<std::vector<double>> localMuMaps(
            static_cast<size_t>(bedCount));
        std::vector<std::string> bedManifests(
            static_cast<size_t>(bedCount));
        for (int bed = 0; bed < bedCount; ++bed) {
          auto &localMu = localMuMaps[static_cast<size_t>(bed)];
          localMu.assign(baseNvox, 0.0);
          for (int z = 0; z < baseNz; ++z) {
            const int globalZ = bedZStarts[static_cast<size_t>(bed)] + z;
            const size_t srcOffset = static_cast<size_t>(globalZ) * plane;
            const size_t dstOffset = static_cast<size_t>(z) * plane;
            std::copy_n(ctacMuMap.data() + srcOffset, plane,
                        localMu.data() + dstOffset);
          }
          bedManifests[static_cast<size_t>(bed)] = buildCTACFactorManifest(
              args, baseNx, baseNy, baseNz, spacing,
              bedZStarts[static_cast<size_t>(bed)]);
        }

        const size_t expectedCount =
            static_cast<size_t>(args.numAngles) * baseNvox;
        std::vector<std::vector<double>> muToGenerate;
        std::vector<int> bedsToGenerate;
        for (int bed = 0; bed < bedCount; ++bed) {
          auto cached = ctacFactorCache->tryLoad(
              bedManifests[static_cast<size_t>(bed)]);
          if (cached.has_value() && cached->size() == expectedCount) {
            dynamicAttenFactors[static_cast<size_t>(bed)] =
                std::move(*cached);
          } else {
            muToGenerate.push_back(std::move(localMuMaps[static_cast<size_t>(bed)]));
            bedsToGenerate.push_back(bed);
          }
        }

        if (!muToGenerate.empty()) {
          logInfo("Generating CTAC attenuation factors for " +
                  std::to_string(muToGenerate.size()) + " of " +
                  std::to_string(bedCount) + " beds.");
          auto generated = computeAttenFactorsMultiBed(
              muToGenerate, baseNx, baseNy, baseNz, spacing.dx, spacing.dy,
              spacing.dz, args.numAngles, args.ctacRotSign, args.numThreads,
              cudaDeviceId);
          for (size_t i = 0; i < generated.size(); ++i) {
            const int bed = bedsToGenerate[i];
            dynamicAttenFactors[static_cast<size_t>(bed)] =
                std::move(generated[i]);
            ctacFactorCache->save(bedManifests[static_cast<size_t>(bed)],
                                  dynamicAttenFactors[static_cast<size_t>(bed)],
                                  expectedCount);
          }
        } else {
          logInfo("All CTAC attenuation factors loaded from cache.");
        }

        logDebug(formatProfileEntry(
            "ctac_attenuation_factors",
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          attenuationStart)
                .count(),
            static_cast<size_t>(bedCount)));
    }

    logDebug("Sparse backend: cuda");

    ReconOpts opts;
    opts.threads = std::max(1, args.numThreads);
    opts.nonneg = true;
    opts.w_proj = nullptr;
    opts.b_proj = backgroundProj.empty() ? nullptr : &backgroundProj;
    opts.bedCount = bedCount;
    opts.bedAttenFactors = std::move(dynamicAttenFactors);
    opts.bedOffsetVoxels.resize(static_cast<size_t>(bedCount));
    for (int bed = 0; bed < bedCount; ++bed) {
      opts.bedOffsetVoxels[static_cast<size_t>(bed)] =
          static_cast<long long>(bedZStarts[static_cast<size_t>(bed)]) *
          static_cast<long long>(baseNx) * static_cast<long long>(baseNy);
    }
    opts.num_angles = args.numAngles;

    stageStart = ProfileClock::now();
    auto prepared = prepareReconPreparedParams(args);
    const auto psfKernels = preparePSFForRuntime(opts, args, spacing);
    compactTiming.psfPrepareSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();

    const filters::PSFKernels *psfPtr =
        psfKernels.kx.empty() ? nullptr : &psfKernels;
    const CropSettings crop = sanitizeCropSettings(args.cropEndStart, args.cropEndEnd);
    int outputNzBeforeCrop = args.Nz;
    if (bedCount > 1) {
      outputNzBeforeCrop = bedSetup.canonicalGeom.cropLayers +
          (bedCount - 1) * bedSetup.canonicalGeom.stepLayers;
    }
    if ((crop.start != 0 || crop.end != 0) &&
        !isCropSettingsValid(outputNzBeforeCrop, crop))
      throw std::runtime_error("Invalid output crop range.");
    OutputTransform outputTransform;
    outputTransform.matrixGridOutput = args.matrixGridOutput;
    const OutputTransform *outputTransformPtr = nullptr;
    const bool dfListMode = !args.projPaths.empty();
    const bool enableBedZFlip =
        args.zBedFlip && dfListMode && bedCount <= 1;
    outputTransform.flipZPerBed = enableBedZFlip;
    outputTransform.flipZWholeOutput = args.zOutFlip;
    outputTransform.flipXWholeOutput = args.xOutFlip;
    outputTransform.xyRotationDeg = args.xyOutRotationDeg;
    outputTransform.seamXyAlign = args.seamXyAlign;
    outputTransform.reverseLocalZInReconStorage =
        reverseLocalZInReconStorage;
    outputTransform.centerCropNx = args.matrixGridOutput ? 0 : spect::core::sixmodule::kOutputMaskNx;
    outputTransform.centerCropNy = args.matrixGridOutput ? 0 : spect::core::sixmodule::kOutputMaskNy;
    outputTransform.centerCropNz =
        bedCount == 1 && !args.matrixGridOutput
            ? (args.singleBedStepAligned
                   ? bedSetup.canonicalGeom.cropLayers
                   : spect::core::sixmodule::kSingleBedMaskNz)
            : 0;
    outputTransform.baseNz = baseNz;
    outputTransform.bedZStarts = bedZStarts;
    outputTransform.outputBedOverlaps = args.outputBedOverlaps;
    outputTransform.canonicalGeom = bedSetup.canonicalGeom;

    outputTransformPtr = &outputTransform;
    if (enableBedZFlip)
      logDebug("Output transform: per-bed Z flip enabled for dfList bed segments.");
    if (args.zOutFlip)
      logDebug("Output transform: whole-output Z flip enabled.");
    if (args.xOutFlip)
      logDebug("Output transform: whole-output X flip enabled.");
    if (args.xyOutRotationDeg != 0.0)
      logInfo("Output transform: XY rotation " + std::to_string(args.xyOutRotationDeg) +
              " deg (+X toward +Y), after flips/crop/mask; fixed grid, bilinear, zero outside.");
    if (args.seamXyAlign)
      logDebug("Output transform: seam XY overlap alignment enabled.");
    logInfo(args.matrixGridOutput
                ? (bedCount > 1
                       ? "Output transform: joint multi-bed volume of full matrix grids; no output blending or gain fitting."
                       : "Output transform: full matrix grid.")
                : "Output transform: canonical center66_cosine10 stitch; "
                  "reconstruction beds are non-overlapping.");
    CompactSymmetricReconLoopContext compactContext;
    compactContext.badDetectors = &badDetectors;
    compactContext.args = &args;
    compactContext.opts = &opts;
    compactContext.prepared = &prepared;
    compactContext.projections = &projections;
    compactContext.backgroundProj = &backgroundProj;
    compactContext.mask = maskPtr;
    compactContext.psfKernels = psfPtr;
    compactContext.spacing = spacing;
    compactContext.outputGeometry = outputGeometry;
    compactContext.crop = crop;
    compactContext.outputTransform = outputTransformPtr;
    compactContext.compactMeta = &compactMeta;
    compactContext.compactMetaPath = &compactMetaPath;
    compactContext.compactChunks = compactChunks.get();
    compactContext.compactChunks2 = compactChunks2.get();
    compactContext.projections2 = &projections2;
    compactContext.backgroundProj2 = &backgroundProj2;
    compactContext.timing = &compactTiming;
    compactContext.loadToIterationStart = loadToIterationStart;
    compactContext.fullCommandStart = fullCommandStart;
    compactContext.bedCount = bedCount;
    compactContext.baseNx = baseNx;
    compactContext.baseNy = baseNy;
    compactContext.baseNz = baseNz;
    compactContext.baseNvox = baseNvox;
    compactContext.globalNvox = globalNvox;
    compactContext.bedZStarts = &bedZStarts;
    (void)runCompactSymmetricReconstruction(compactContext);

    logInfo("Reconstruction finished successfully.");
    return spect::cli::kExitSuccess;
  } catch (const std::exception &ex) {
    std::cerr << "[FATAL] " << ex.what() << '\n';
    return spect::cli::kExitRuntimeError;
  }
}

} // namespace spect::main
