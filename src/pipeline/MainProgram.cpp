#include "pipeline/MainProgram.h"

#include "recon/CompactSymmetricReconLoop.h"

#include "recon/BedOffsets.h"
#include "cli/ArgumentParser.h"
#include "cli/ExitCodes.h"
#include "cli/Version.h"
#include "core/Masks.h"
#include "core/ReconLimits.h"
#include "sparse/CompactSymmetricProjector.h"
#include "cuda/CudaConfig.h"
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
#include "sparse/SparseOpsBackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
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
      logInfo("Loaded configuration from " + args.configPath);
    } else {
      logWarn("Failed to load config file '" + args.configPath +
              "'. Using internal defaults.");
    }
    return;
  }

  if (args.configFound && args.configLoaded) {
    logInfo("Loaded configuration from " + args.configPath);
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

  // Consumer GeForce targets for this release package.
  const bool knownConsumerArch =
      (info.computeMajor == 7 && info.computeMinor == 5) || // RTX 20
      (info.computeMajor == 8 && info.computeMinor == 6) || // RTX 30
      (info.computeMajor == 8 && info.computeMinor == 9) || // RTX 40
      (info.computeMajor == 12 && info.computeMinor == 0);  // RTX 50
  if (!knownConsumerArch) {
    logWarn("GPU compute capability " + smTag +
            " is outside the primary consumer RTX set "
            "(sm_75/86/89/120). NVRTC will still JIT for this device if the "
            "bundled toolkit supports it.");
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
    logInfo("CUDA device selected from " + source + ": " +
            std::to_string(deviceId) + " (device count=" +
            std::to_string(deviceCount) + ")");
  } else if (deviceCount > 1) {
    logInfo("CUDA device defaulted to 0 (device count=" +
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

void appendCTACCacheFileIdentity(std::ostringstream &oss,
                                 const std::string &label,
                                 const std::filesystem::path &path) {
  oss << label << "=" << path.string() << '\n';
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    oss << label << "_size=UNAVAILABLE\n";
  } else {
    oss << label << "_size=" << size << '\n';
  }
  ec.clear();
  const auto writeTime = std::filesystem::last_write_time(path, ec);
  if (ec) {
    oss << label << "_mtime=UNAVAILABLE\n";
  } else {
    oss << label << "_mtime=" << writeTime.time_since_epoch().count() << '\n';
  }
}

std::string buildCTACFactorManifest(const ReconArgs &args, int baseNx,
                                    int baseNy, int baseNz,
                                    const filters::Spacing3 &spacing,
                                    int bedZStart) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << "cache_schema=ctac_atten_factors_v1\n";
  oss << "ctac_model=" << args.ctacModel << '\n';
  oss << "nx=" << baseNx << '\n';
  oss << "ny=" << baseNy << '\n';
  oss << "nz=" << baseNz << '\n';
  oss << "dx=" << spacing.dx << '\n';
  oss << "dy=" << spacing.dy << '\n';
  oss << "dz=" << spacing.dz << '\n';
  oss << "num_angles=" << args.numAngles << '\n';
  oss << "rotation_sign=" << args.ctacRotSign << '\n';
  oss << "bed_z_start=" << bedZStart << '\n';
  oss << "ctac_auto_mu_value=" << args.ctacAutoMuValue << '\n';
  oss << "ctac_auto_mu_source=" << args.ctacAutoMuValueSource << '\n';
  if (!args.attMapPath.empty()) {
    appendCTACCacheFileIdentity(oss, "att_map", args.attMapPath);
  }
  if (!args.ctacAutoMuFrom.empty()) {
    appendCTACCacheFileIdentity(oss, "auto_mu_from", args.ctacAutoMuFrom);
    oss << "ctac_auto_mu_thresh=" << args.ctacAutoMuThresh << '\n';
  }
  return oss.str();
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

  std::string logDirError;
  auto logDir = resolveLogDirectory(args, logDirError);
  ReconArgs logArgs = args;
  if (logDir.empty())
    logDir = ".";
  logArgs.outputDir = logDir.string();
  ConsoleTeeGuard tee(makeDefaultLogPath(logArgs));
  if (tee.enabled()) {
    logInfo("Logging console output to: " + tee.logPath().string());
    if (!logDirError.empty()) {
      logWarn("Failed to create output directory '" + args.outputDir +
              "' for logging (" + logDirError + "); using current directory.");
    }
  } else {
    logWarn("Failed to create log file at: " + makeDefaultLogPath(logArgs).string());
  }

  const std::string commandLine = makeCommandLineString(argc, argv);
  logInfo(std::string("=== SPECT_recon ") + spect::cli::versionString() +
          " ===");
  logInfo("Command line: " + commandLine);
  reportConfigStatus(args);

  int cudaDeviceId = 0;
  try {
    cudaDeviceId = selectCudaDeviceIfRequested(args);
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
    const bool compactLayoutRequested = !args.chunkCSRDir.empty();
    const bool compactChunkLayout = compactLayoutRequested;
    spect::io::CompactSymmetryMetadata compactMeta;
    std::string compactMetaPath;

    if (compactLayoutRequested) {
      compactMetaPath = resolveCompactMetadataPath(args);
      compactMeta = spect::io::loadCompactSymmetryMetadata(compactMetaPath);
      applyCompactMetadataGeometry(args, compactMeta);
      logInfo("compact metadata geometry loaded from " + compactMetaPath +
              " (mode=" + spect::io::compactSymmetryModeName(compactMeta.mode) +
              ", compactRows=" + std::to_string(compactMeta.compactRows()) +
              ", fullRows=" + std::to_string(compactMeta.fullRows()) + ")");
    }

    std::string dimHintWarning;
    if (adoptReconDimsFromHint(args, &dimHintWarning) && !dimHintWarning.empty())
      logWarn(dimHintWarning);

    std::string dimsWarning;
    if (ensureReconDims(args, &dimsWarning) && !dimsWarning.empty())
      logWarn(dimsWarning);

    const BedSetupResult bedSetup = prepareBedSetup(args);
    for (const std::string &warning : bedSetup.warnings)
      logWarn(warning);
    logInfo(bedSetup.summary);
    printReconArgs(args, commandLine);

    std::filesystem::create_directories(args.outputDir);

    const std::vector<std::string> projPaths = bedSetup.projectionPaths;
    if (projPaths.empty() || projPaths.front().empty())
      throw std::runtime_error("Projection path is required.");

    auto stageStart = ProfileClock::now();
    const LoadedProjectionSet projectionSet =
        loadProjectionSet(projPaths, args.numAngles, "Projection Data", "matrix");
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
        logInfo("Projection BIN does not store detZ/detX; using configured detector shape " +
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

    logInfo("Loaded projection rays: total=" +
            std::to_string(projectionSet.data.size()) + ", per bed=" +
            std::to_string(projectionSet.raysPerBed));

    std::vector<double> backgroundProj;
    std::vector<std::string> backgroundPaths = args.bProjPaths;
    if (backgroundPaths.empty() && !args.bProjPath.empty())
      backgroundPaths = {args.bProjPath};
    if (!backgroundPaths.empty()) {
      if (bedCount > 1 && backgroundPaths.size() != static_cast<size_t>(bedCount))
        throw std::runtime_error("Background projection count must match bed count.");
      if (bedCount == 1 && backgroundPaths.size() > 1)
        throw std::runtime_error("Multiple background projections were provided for single-bed mode.");

      const LoadedProjectionSet backgroundSet =
          loadProjectionSet(backgroundPaths, args.numAngles,
                            "Background Projection", "b_proj");
      if (backgroundSet.raysPerBed != projectionSet.raysPerBed)
        throw std::runtime_error("Background projection geometry does not match emission projections.");
      backgroundProj = backgroundSet.data;
      logInfo("Loaded background projection term with " +
              std::to_string(backgroundProj.size()) + " rays.");
    }

    const auto spacingArray = voxelSpacingFromArgs(args);
    const filters::Spacing3 spacing{spacingArray[0], spacingArray[1], spacingArray[2]};
    const OutputGeometry outputGeometry{args.Nx, args.Ny, args.Nz, spacing.dx,
                      spacing.dy, spacing.dz};

    std::vector<double> sharedAnatomical;
    if (args.useReg && !args.regCtPath.empty()) {
      logInfo("Loading registered CT for TV/RDP weights from " + args.regCtPath);
      sharedAnatomical = loadVolumeChecked(args.regCtPath, "Registered CT",
                                          args.Nx, args.Ny, args.Nz);
    } else if (!args.doCTAC && !args.attMapPath.empty() &&
               (args.useReg || !std::isnan(args.maskFromAttThreshold))) {
      logInfo("Reusing registered attenuation map for TV/RDP weights.");
      sharedAnatomical = loadVolumeChecked(args.attMapPath, "Attenuation map",
                                          args.Nx, args.Ny, args.Nz);
    }

    std::vector<uint8_t> mask;
    const std::vector<uint8_t> *maskPtr =
        maybeMakeCylindricalMask(mask, args.Nx, args.Ny, args.Nz, spacingArray, args);

    std::vector<double> ctacMuMap;
    if (args.doCTAC) {
      if (!args.attMapPath.empty()) {
        logInfo("Loading attenuation map from " + args.attMapPath);
        ctacMuMap = loadVolumeChecked(args.attMapPath, "Attenuation map",
                                      args.Nx, args.Ny, args.Nz);
      } else if (!args.ctacAutoMuFrom.empty()) {
        logInfo("Auto mu-map: segmenting early-iteration reconstruction " +
                args.ctacAutoMuFrom);
        logWarn("Auto mu-map assumes activity roughly fills the attenuating "
                "volume (water-equivalent phantom). For hot-rods-in-cold-body "
                "objects supply a measured -attMap instead.");
        const std::vector<double> earlyRecon = loadVolumeChecked(
            args.ctacAutoMuFrom, "Auto-mu source reconstruction", args.Nx,
            args.Ny, args.Nz);
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
      } else {
        throw std::runtime_error(
            "CTAC requires -attMap <file> or -ctacAutoMu <early recon volume>.");
      }
    }
    if (args.useReg) {
      if (!sharedAnatomical.empty())
        args.reg.ct_image = &sharedAnatomical;
      else if (!args.attMapPath.empty() && !ctacMuMap.empty())
        args.reg.ct_image = &ctacMuMap;
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
      if (!ctacMuMap.empty())
        maskSource = &ctacMuMap;
      else if (!sharedAnatomical.empty())
        maskSource = &sharedAnatomical;
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
    logInfo("Opening compact chunk CSR directory from " + args.chunkCSRDir);
    {
      auto opened =
          spect::io::openCompactChunkCSRDirectory(args.chunkCSRDir, compactMeta);
      logInfo("Compact chunk CSR directory opened: chunks=" +
              std::to_string(opened.chunks().size()) +
              ", compactRows=" + std::to_string(opened.totalRows()) +
              ", nnz=" + std::to_string(opened.totalNnz()) +
              ", hostCacheBudgetBytes=" +
              std::to_string(opened.cacheBudgetBytes()));
      compactChunks =
          std::make_unique<spect::io::CompactChunkCSRDirectory>(std::move(opened));
    }
    compactTiming.csrMapOrLoadSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();

    const int matrixRows = static_cast<int>(compactChunks->totalRows());
    const int matrixCols = compactChunks->cols();
    if (matrixCols != static_cast<int>(baseNvox)) {
      throw std::runtime_error("System matrix voxel count does not match the per-bed reconstruction grid.");
    }

    const bool compactLayout = compactLayoutRequested;
    if (compactLayout) {
      stageStart = ProfileClock::now();
      if (compactMetaPath.empty()) {
        compactMetaPath = resolveCompactMetadataPath(args);
        compactMeta = spect::io::loadCompactSymmetryMetadata(compactMetaPath);
      }
      if (compactChunkLayout) {
        spect::io::validateCompactChunkSymmetricInputs(
            compactMeta, compactChunks->totalRows(), compactChunks->cols(),
            projectionSet.raysPerBed, args.detNz, args.detNx, args.numAngles,
            baseNx, baseNy, baseNz);
      }
      compactTiming.metadataLoadAndValidateSeconds =
          std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
      logInfo("compact-symmetric layout enabled via " + compactMetaPath +
              " (mode=" + spect::io::compactSymmetryModeName(compactMeta.mode) +
              ", compactRows=" + std::to_string(compactMeta.totalRows) +
              ", fullRows=" + std::to_string(compactMeta.fullRows()) +
              ", matrix=chunkCSRDir)");
      if (compactChunkLayout && bedCount > spect::core::kMaxBeds) {
        throw std::runtime_error(
            "compact chunk multi-bed CUDA path supports at most " +
            std::to_string(spect::core::kMaxBeds) + " beds.");
      }
    }

    const size_t expectedRowsPerBed =
        compactLayout ? static_cast<size_t>(compactMeta.fullRows())
                      : static_cast<size_t>(matrixRows);
    const size_t expectedRays = expectedRowsPerBed * static_cast<size_t>(bedCount);
    std::vector<double> projections = projectionSet.data;
    if (projections.size() < expectedRays) {
      throw std::runtime_error("Projection count is smaller than the required system-matrix rows across all beds.");
    }
    if (projections.size() > expectedRays) {
      if (compactLayout) {
        throw std::runtime_error(
            "Compact-symmetric layout requires exact full projection length; refusing to truncate.");
      }
      logWarn("Projection count exceeds required rays. Truncating to match the system matrix.");
      projections.resize(expectedRays);
    }
    if (!backgroundProj.empty() && backgroundProj.size() != projections.size()) {
      throw std::runtime_error("Background projection size does not match projection size.");
    }

    if (!args.badDetectorFile.empty()) {
      const auto badDetectors =
          spect::io::loadBadDetectorList(args.badDetectorFile);
      if (!badDetectors.empty()) {
        if (!compactChunkLayout) {
          throw std::runtime_error(
              "bad detector masking requires the canonical compact chunk "
              "layout with identity projections.");
        }
        const std::vector<uint8_t> badDetectorMask =
            spect::io::buildBadDetectorProjectionMask(
                badDetectors, compactMeta.fullNumAngles, compactMeta.detNz,
                compactMeta.detNx);
        const size_t fullRows = static_cast<size_t>(compactMeta.fullRows());
        const size_t maskedRows = static_cast<size_t>(std::count(
            badDetectorMask.begin(), badDetectorMask.end(), uint8_t{1}));
        for (int bed = 0; bed < bedCount; ++bed) {
          const size_t bedOffset = static_cast<size_t>(bed) * fullRows;
          for (size_t row = 0; row < fullRows; ++row) {
            if (!badDetectorMask[row])
              continue;
            projections[bedOffset + row] = 0.0;
            if (!backgroundProj.empty())
              backgroundProj[bedOffset + row] = 0.0;
          }
        }
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
      // censoring); the matching masked sensitivity is computed in
      // CompactSymmetricReconLoop.
      if (!compactLayout) {
        throw std::runtime_error(
            "-detZEdgeCensor requires the compact-symmetric chunk layout.");
      }
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
        double *bedProj =
            projections.data() + static_cast<size_t>(bed) * fullRows;
        size_t zeroedBins = 0;
        for (int angle = 0; angle < compactMeta.fullNumAngles; ++angle) {
          double *angleBase =
              bedProj + static_cast<size_t>(angle) * raysPerAngle;
          for (int detZ = 0; detZ < compactMeta.detNz; ++detZ) {
            if (detZ >= censorK && detZ < compactMeta.detNz - censorK)
              continue;
            double *row =
                angleBase + static_cast<size_t>(detZ) * compactMeta.detNx;
            std::fill(row, row + compactMeta.detNx, 0.0);
            zeroedBins += static_cast<size_t>(compactMeta.detNx);
          }
        }
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

    std::vector<std::vector<uint16_t>> dynamicAttenFactors;
    std::unique_ptr<CTACFactorCache> ctacFactorCache;
    if (args.doCTAC) {
      logInfo("CTAC model: angle-dependent parallel-ray line integral "
              "(rotation sign " + std::to_string(args.ctacRotSign) +
              "). Attenuation map must be mu in mm^-1.");
      if (bedCount > 1 || compactChunkLayout) {
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
          const auto cached = ctacFactorCache->tryLoad(
              bedManifests[static_cast<size_t>(bed)]);
          if (cached.has_value() && cached->size() == expectedCount) {
            dynamicAttenFactors[static_cast<size_t>(bed)] =
                std::move(*cached);
          } else {
            muToGenerate.push_back(localMuMaps[static_cast<size_t>(bed)]);
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
    }

    const ISparseOps &sparseOps = getSparseOpsCUDA();
    logInfo("Sparse backend: cuda");

    ReconOpts opts;
    opts.threads = std::max(1, args.numThreads);
    opts.nonneg = true;
    opts.mask = maskPtr;
    opts.w_proj = nullptr;
    opts.w_img = nullptr;
    opts.b_proj = backgroundProj.empty() ? nullptr : &backgroundProj;
    opts.sparseOps = &sparseOps;
    opts.bedCount = bedCount;
    opts.bedOverlaps = args.bedOverlaps;
    opts.bedAttenFactors = dynamicAttenFactors;
    opts.bedOffsetVoxels.resize(static_cast<size_t>(bedCount));
    for (int bed = 0; bed < bedCount; ++bed) {
      opts.bedOffsetVoxels[static_cast<size_t>(bed)] =
          static_cast<long long>(bedZStarts[static_cast<size_t>(bed)]) *
          static_cast<long long>(baseNx) * static_cast<long long>(baseNy);
    }
    opts.det_nz = args.detNz;
    opts.det_nx = args.detNx;
    opts.num_angles = args.numAngles;

    stageStart = ProfileClock::now();
    auto prepared = prepareReconPreparedParams(args, args.psf);
    auto runtimePsf = preparePSFForRuntime(opts, args, spacing, bedCount,
                                           projections.size(), baseNx);
    compactTiming.psfPrepareSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
    for (const std::string &message : runtimePsf.logs.infoMessages)
      logInfo(message);
    for (const std::string &message : runtimePsf.logs.warningMessages)
      logWarn(message);

    const filters::PSFKernels *psfPtr = nullptr;
    if (!runtimePsf.psfKernels.kx.empty()) {
      psfPtr = &runtimePsf.psfKernels;
    } else if (!prepared.psfKernels.kx.empty()) {
      psfPtr = &prepared.psfKernels;
    }
    const CropSettings crop = sanitizeCropSettings(args.cropEndStart, args.cropEndEnd);
    int outputNzBeforeCrop = args.Nz;
    if (bedCount > 1) {
      if (args.bedCropMethod == "center66_cosine10") {
        outputNzBeforeCrop = kCenter66CropLayers +
            (bedCount - 1) * kCenter66BedStepLayers;
      } else {
        for (int overlap : args.outputBedOverlaps)
          outputNzBeforeCrop -= overlap;
      }
    }
    if ((crop.start != 0 || crop.end != 0) &&
        !isCropSettingsValid(outputNzBeforeCrop, crop))
      throw std::runtime_error("Invalid output crop range.");
    OutputTransform outputTransform;
    const OutputTransform *outputTransformPtr = nullptr;
    const bool dfListMode = !args.projPaths.empty();
    const bool enableBedZFlip =
        args.zBedFlip && dfListMode && (bedCount <= 1 || !compactLayout);
    const bool hasOutputOverlapStitch =
        bedCount > 1 &&
        std::any_of(args.outputBedOverlaps.begin(), args.outputBedOverlaps.end(),
                    [](int overlap) { return overlap > 0; });
    // Seam closing only runs when multi-bed output stitching actually uses
    // overlap. Zero-overlap joints or single-bed keep cut/sigmoid only.
    const bool enableSeamPostprocess =
        args.bedSeamPostprocess && hasOutputOverlapStitch &&
        args.bedCropMethod != "direct" &&
        args.bedObjectProfile != "derenzo";
    // The rod passes are gated on the resolved object profile inside
    // prepareOutputVolume, so "auto" is enough to let a Derenzo lattice be
    // recognised from the stitched output itself.
    const bool enableDerenzoPostprocess =
        args.bedDerenzoPostprocess && (args.bedObjectProfile == "derenzo" ||
                                       args.bedObjectProfile == "auto");
    if (args.bedSeamPostprocess && !enableSeamPostprocess) {
      logInfo("Output transform: seam postprocess requested but skipped "
              "(requires multi-bed weighted output stitching).");
    }
    outputTransform.flipZPerBed = enableBedZFlip;
    outputTransform.flipZWholeOutput = args.zOutFlip;
    outputTransform.flipXWholeOutput = args.xOutFlip;
    outputTransform.sigmoidBlendOutput = args.bedCropMethod == "sigmoid";
    outputTransform.center66CosineOutput =
        args.bedCropMethod == "center66_cosine10";
    outputTransform.objectAwareSeamPostprocess = enableSeamPostprocess;
    outputTransform.objectProfile = args.bedObjectProfile;
    outputTransform.derenzoAxialPostprocess = enableDerenzoPostprocess;
    outputTransform.derenzoAxialStrength = args.bedDerenzoAxialStrength;
    outputTransform.derenzoAdhesionPostprocess =
        args.bedDerenzoAdhesion && (args.bedObjectProfile == "derenzo" ||
                                    args.bedObjectProfile == "auto");
    outputTransform.derenzoAdhesionInner = args.bedDerenzoAdhesionInner;
    outputTransform.derenzoAdhesionOuter = args.bedDerenzoAdhesionOuter;
    outputTransform.derenzoCoreDamp = args.bedDerenzoCoreDamp;
    outputTransform.phantomCloseStrength = args.bedPhantomCloseStrength;
    outputTransform.epguanCloseStrength = args.bedEpguanCloseStrength;
    outputTransform.epguanWallMaxPreserve =
        args.bedEpguanWallMaxPreserve;
    outputTransform.epguanSeamEdgeStrength =
        args.bedEpguanSeamEdgeStrength;
    outputTransform.epguanSeamFeatherSlices =
        args.bedEpguanSeamFeatherSlices;
    outputTransform.epguanBackgroundFilter =
        args.bedEpguanBackgroundFilter;
    outputTransform.epguanBackgroundRetain =
        args.bedEpguanBackgroundRetain;
    outputTransform.epguanSupportThreshold =
        args.bedEpguanSupportThreshold;
    outputTransform.epguanSupportMarginMm =
        args.bedEpguanSupportMarginMm;
    outputTransform.zimuCloseStrength = args.bedZimuCloseStrength;
    outputTransform.mouseSeamStrength = args.bedMouseSeamStrength;
    outputTransform.seamHalfWidth = args.bedSeamHalfWidth;
    outputTransform.baseNz = baseNz;
    outputTransform.bedZStarts = bedZStarts;
    outputTransform.outputBedOverlaps = args.outputBedOverlaps;
    outputTransform.fovRadiusMm = args.fovRadiusMm;
    outputTransform.analysisFovRadiusMm = args.analysisFovRadiusMm;
    outputTransformPtr = &outputTransform;
    if (enableBedZFlip)
      logInfo("Output transform: per-bed Z flip enabled for dfList bed segments.");
    if (args.zOutFlip)
      logInfo("Output transform: whole-output Z flip enabled.");
    if (args.xOutFlip)
      logInfo("Output transform: whole-output X flip enabled.");
    logInfo("Output transform: reconstruction beds are non-overlapping; "
            "crop_method=" + args.bedCropMethod);
    if (enableSeamPostprocess)
      logInfo("Output transform: object-aware seam postprocess enabled; "
              "profile=" + args.bedObjectProfile +
              ", phantom_interp_strength=" +
              std::to_string(args.bedPhantomCloseStrength) +
              ", epguan_interp_strength=" +
              std::to_string(args.bedEpguanCloseStrength) +
              ", epguan_wall_max_preserve=" +
              std::to_string(args.bedEpguanWallMaxPreserve ? 1 : 0) +
              ", epguan_seam_edge_strength=" +
              std::to_string(args.bedEpguanSeamEdgeStrength) +
              ", epguan_seam_feather_slices=" +
              std::to_string(args.bedEpguanSeamFeatherSlices) +
              ", epguan_background_filter=" +
              std::to_string(args.bedEpguanBackgroundFilter ? 1 : 0) +
              ", epguan_background_retain=" +
              std::to_string(args.bedEpguanBackgroundRetain) +
              ", epguan_support_threshold=" +
              std::to_string(args.bedEpguanSupportThreshold) +
              ", epguan_support_margin_mm=" +
              std::to_string(args.bedEpguanSupportMarginMm) +
              ", zimu_smooth_strength=" +
              std::to_string(args.bedZimuCloseStrength) +
              ", mouse_seam_strength=" +
              std::to_string(args.bedMouseSeamStrength) +
              ", zimu_sigma_mm=0.3" +
              ", seam_half_width=" +
              std::to_string(args.bedSeamHalfWidth) + ".");
    if (enableDerenzoPostprocess)
      logInfo("Output transform: Derenzo per-rod axial continuity enabled; "
              "positive_only=1, axial_strength=" +
              std::to_string(args.bedDerenzoAxialStrength) +
              ", z_morph_mm=1.5, rod_support_margin=0.");
    if (args.fovRadiusMm > 0.0)
      logInfo("Output transform: circular XY FOV radius=" +
              std::to_string(args.fovRadiusMm) + " mm (4-voxel taper).");
    if (enableSeamPostprocess) {
      const double analysisRadiusMm = args.analysisFovRadiusMm > 0.0
                                          ? args.analysisFovRadiusMm
                                          : args.fovRadiusMm;
      logInfo("Output transform: seam analysis XY FOV radius=" +
              std::to_string(analysisRadiusMm) +
              " mm (0 follows output radius).");
    }

    CompactSymmetricReconLoopContext compactContext;
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
    compactContext.compactChunkLayout = compactChunkLayout;
    compactContext.compactChunks = compactChunks.get();
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
    (void)CompactSymmetricReconLoop(compactContext).run(spect::io::CSRView{});

    logInfo("Reconstruction finished successfully.");
    return spect::cli::kExitSuccess;
  } catch (const std::exception &ex) {
    std::cerr << "[FATAL] " << ex.what() << '\n';
    return spect::cli::kExitRuntimeError;
  }
}

} // namespace spect::main
