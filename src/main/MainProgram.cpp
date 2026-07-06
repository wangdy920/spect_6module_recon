#include "main/MainProgram.h"

#include "main/CompactSymmetricReconLoop.h"

#include "app/ADMMOptimizer.h"
#include "app/BedOffsets.h"
#include "app/BSREMOptimizer.h"
#include "app/BSREMReconstructor.h"
#include "app/IOptimizer.h"
#include "app/MLEMOptimizer.h"
#include "app/OSEMReconstructor.h"
#include "app/ProjectionOps.h"
#include "app/ReconMetrics.h"
#include "cli/ArgumentParser.h"
#include "cli/ExitCodes.h"
#include "core/Masks.h"
#include "core/ReconLimits.h"
#include "cuda/CompactSymmetricProjector.h"
#include "cuda/CudaConfig.h"
#include "cuda/VolumeOpsCUDA.h"
#include "cuda/ProjDomainChunked.h"
#include "filters/ButterworthFilter.h"
#include "filters/Convolution.h"
#include "io/CSRBuilder.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "io/MHDWriter.h"
#include "io/ProjectionLoader.h"
#include "io/VolumeLoader.h"
#include "main/AutoMuMap.h"
#include "main/CTACOps.h"
#include "main/MatVarPath.h"
#include "main/ReconDataLoader.h"
#include "main/ReconArgsPrinter.h"
#include "main/ReconPreparation.h"
#include "main/ReconSetup.h"
#include "main/ReconLoopHelpers.h"
#include "main/RunLogging.h"
#include "main/RunLogger.h"
#include "main/SmoothingOps.h"
#include "main/StandardReconLoop.h"
#include "main/VolumeOutput.h"
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

void selectCudaDeviceIfRequested(const ReconArgs &args) {
  if (!args.cudaDeviceRequested)
    return;

#if defined(SPECT_HAS_CUDA)
  const int deviceCount = spect::cuda::getDeviceCount();
  const int deviceId = parseRequestedCudaDevice(args, deviceCount);
  spect::cuda::setDevice(deviceId);
  const std::string source = args.cudaDeviceFromCli ? "-cudaDevice" : "SPECT_CUDA_DEVICE";
  logInfo("CUDA device selected from " + source + ": " +
          std::to_string(deviceId) + " (device count=" +
          std::to_string(deviceCount) + ")");
#else
  const int deviceCount = 0;
  (void)parseRequestedCudaDevice(args, deviceCount);
#endif
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
  logDebug("Command line: " + makeCommandLineString(argc, argv));
  reportConfigStatus(args);

  try {
    selectCudaDeviceIfRequested(args);
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
    const bool compactLayoutRequested =
        args.matrixLayout == "compact-symmetric";
    const bool compactChunkLayout =
        compactLayoutRequested && !args.chunkCSRDir.empty();
    spect::io::CompactSymmetryMetadata compactMeta;
    std::string compactMetaPath;

    if (compactChunkLayout) {
      compactMetaPath = resolveCompactMetadataPath(args);
      compactMeta = spect::io::loadCompactSymmetryMetadata(compactMetaPath);
      applyCompactMetadataGeometry(args, compactMeta);
      logInfo("compact chunk metadata geometry loaded from " + compactMetaPath +
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
    printReconArgs(args);

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

    if (args.algorithm == ReconArgs::Algorithm::BSREM && args.psfProj) {
      throw std::runtime_error(
          "BSREM does not support projection-domain PSF. Use operator-domain PSF instead.");
    }

    const auto spacingArray = voxelSpacingFromArgs(args);
    const filters::Spacing3 spacing{spacingArray[0], spacingArray[1], spacingArray[2]};
    const OutputGeometry outputGeometry{args.Nx, args.Ny, args.Nz, spacing.dx,
                      spacing.dy, spacing.dz};

    std::vector<double> sharedAnatomical;
    if (!args.priorPath.empty() && args.priorPath != "none") {
      logInfo("Loading anatomical prior from " + args.priorPath);
      sharedAnatomical = loadVolumeChecked(args.priorPath, "Prior volume", args.Nx,
                                          args.Ny, args.Nz);
    } else if ((args.useReg && (args.reg.type == RegType::Bowsher ||
                                args.reg.type == RegType::PLS)) &&
               !args.attMapPath.empty()) {
      logInfo("No explicit prior was provided. Reusing attenuation map as anatomical prior.");
      sharedAnatomical = loadVolumeChecked(args.attMapPath, "Attenuation volume",
                                          args.Nx, args.Ny, args.Nz);
    }
    if (!sharedAnatomical.empty())
      args.reg.anatomical_image = &sharedAnatomical;

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
                std::to_string(args.ctacAutoMuValue) + " mm^-1.");
        ctacMuMap = autoMu.muMap;
        saveVolumeOutput(ctacMuMap, outputGeometry, args.outputDir,
                         outputPrefixFromArgs(args) + "_automu", -1, nullptr,
                         sanitizeCropSettings(0, 0));
      } else {
        throw std::runtime_error(
            "CTAC requires -attMap <file> or -ctacAutoMu <early recon volume>.");
      }
    }
    if (!std::isnan(args.maskFromAttThreshold)) {
      const std::vector<double> *maskSource = nullptr;
      if (!ctacMuMap.empty())
        maskSource = &ctacMuMap;
      else if (!sharedAnatomical.empty())
        maskSource = &sharedAnatomical;
      if (!maskSource)
        throw std::runtime_error("-maskFromAtt requires -attMap or -prior.");

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

    spect::io::CSR sysCSR;
    spect::io::CSRMap csrMap;
    bool useMappedCSR = false;
    std::unique_ptr<spect::io::CompactChunkCSRDirectory> compactChunks;

    stageStart = ProfileClock::now();
    if (compactChunkLayout) {
      logInfo("Opening compact chunk CSR directory from " + args.chunkCSRDir);
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
    } else if (!args.loadCSRPath.empty()) {
      std::error_code csrExistsEc;
      const bool csrExists =
          std::filesystem::exists(args.loadCSRPath, csrExistsEc);
      if (csrExistsEc) {
        throw std::runtime_error("Cannot access CSR cache path " +
                                 args.loadCSRPath + ": " +
                                 csrExistsEc.message());
      }
      if (!csrExists) {
        throw std::runtime_error("CSR cache file does not exist: " +
                                 args.loadCSRPath);
      }
      const uint64_t csrFileSize =
          static_cast<uint64_t>(std::filesystem::file_size(args.loadCSRPath));
      const bool implicitCompactMap =
          compactLayoutRequested &&
          spect::io::shouldUseImplicitCSRMap(csrFileSize);
      if (args.mapCSR || implicitCompactMap) {
        logInfo("Mapping CSR cache from " + args.loadCSRPath +
                (implicitCompactMap && !args.mapCSR
                     ? " (implicit compact single-file large-matrix path)"
                     : ""));
        try {
          csrMap.open(args.loadCSRPath);
          useMappedCSR = true;
        } catch (const std::exception &ex) {
          if (args.mapCSR ||
              spect::io::shouldRefuseFullCopyCSRLoad(
                  csrFileSize, spect::io::availablePhysicalMemoryBytes())) {
            throw;
          }
          logWarn("CSR mmap failed for " + args.loadCSRPath + ": " +
                  ex.what() + "; falling back to full-copy load.");
          sysCSR = spect::io::loadCSRBinary(args.loadCSRPath);
        }
      } else {
        logInfo("Loading CSR cache into memory from " + args.loadCSRPath);
        sysCSR = spect::io::loadCSRBinary(args.loadCSRPath);
      }
    } else {
      if (args.sysIndexPath.empty() || args.sysWeightPath.empty()) {
        throw std::runtime_error(
            "System matrix inputs (-indexMat and -weightMat) are required unless -loadCSR is used.");
      }

      std::string indexPath = args.sysIndexPath;
      std::string weightPath = args.sysWeightPath;
      const std::string indexVar = resolveMatVarName(indexPath, "");
      const std::string weightVar = resolveMatVarName(weightPath, "");
      const std::vector<uint8_t> *maskForCSR =
          (maskPtr && args.ApruneMask && bedCount == 1) ? maskPtr : nullptr;
      if (maskPtr && args.ApruneMask && bedCount > 1)
        logWarn("ApruneMask is ignored for multi-bed CSR build.");

      if (args.Astream) {
        logInfo("Building CSR via streaming MAT reader.");
        sysCSR = spect::io::buildCSR_from_mat_stream(
            indexPath, indexVar, weightPath, weightVar, baseNvox,
            args.weightPruneEps, maskForCSR);
      } else {
        logInfo("Building CSR via in-memory MAT load.");
        const auto indexMat = spect::io::readMat(indexPath, indexVar);
        const auto weightMat = spect::io::readMat(weightPath, weightVar);
        sysCSR = spect::io::buildCSR_from_mat(indexMat, weightMat, baseNvox,
                                              args.weightPruneEps, maskForCSR);
      }
      if (!args.saveCSRPath.empty()) {
        logInfo("Saving CSR cache to " + args.saveCSRPath);
        spect::io::saveCSRBinary(sysCSR, args.saveCSRPath);
      }
    }
    compactTiming.csrMapOrLoadSeconds =
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();

    const int matrixRows = compactChunkLayout
                               ? static_cast<int>(compactChunks->totalRows())
                               : (useMappedCSR ? csrMap.view().rows : sysCSR.rows);
    const int matrixCols =
        compactChunkLayout ? compactChunks->cols()
                           : (useMappedCSR ? csrMap.view().cols : sysCSR.cols);
    spect::io::CSRView matrixViewForValidation{};
    if (compactChunkLayout) {
      matrixViewForValidation.rows = matrixRows;
      matrixViewForValidation.cols = matrixCols;
      matrixViewForValidation.nnz = compactChunks->totalNnz();
    } else if (useMappedCSR) {
      matrixViewForValidation = csrMap.view();
    } else {
      matrixViewForValidation.rows = sysCSR.rows;
      matrixViewForValidation.cols = sysCSR.cols;
      matrixViewForValidation.row_ptr = sysCSR.row_ptr.data();
      matrixViewForValidation.col_idx = sysCSR.col_idx.data();
      matrixViewForValidation.val = sysCSR.val.data();
      matrixViewForValidation.nnz = sysCSR.val.size();
    }
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
      } else {
        spect::io::validateCompactSymmetricInputs(
            compactMeta, matrixViewForValidation, projectionSet.raysPerBed,
            args.detNz, args.detNx, args.numAngles, baseNx, baseNy, baseNz);
      }
      compactTiming.metadataLoadAndValidateSeconds =
          std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
      logInfo("compact-symmetric layout enabled via " + compactMetaPath +
              " (mode=" + spect::io::compactSymmetryModeName(compactMeta.mode) +
              ", compactRows=" + std::to_string(compactMeta.totalRows) +
              ", fullRows=" + std::to_string(compactMeta.fullRows()) +
              (compactChunkLayout ? ", matrix=chunkCSRDir)" : ", matrix=loadCSR)"));
      if (!compactChunkLayout) {
        logWarn("[DEPRECATED] single-file compact-symmetric CSRB "
                "(-loadCSR -compactMeta) is legacy 260525 compatibility. "
                "Prefer -chunkCSRDir with 260601 metadata.");
      }
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

    const std::vector<int> bedZStarts =
        spect::app::computeBedZOffsets(bedCount, baseNz, args.bedOverlaps);

    std::vector<std::vector<uint16_t>> dynamicAttenFactors;
    if (args.doCTAC) {
      const spect::main::CTACModel ctacModel =
          (args.ctacModel == "per-voxel")
              ? spect::main::CTACModel::PerVoxelApprox
              : spect::main::CTACModel::LineIntegral;
      if (ctacModel == spect::main::CTACModel::PerVoxelApprox) {
        logWarn("CTAC model 'per-voxel' is the legacy exp(-mu) approximation "
                "replicated per angle; it is not quantitative line-integral "
                "CTAC. Use -ctacModel line-integral for the physical model.");
      } else {
        logInfo("CTAC model: angle-dependent parallel-ray line integral "
                "(rotation sign " + std::to_string(args.ctacRotSign) +
                "). Attenuation map must be mu in mm^-1.");
      }
      if (useMappedCSR || bedCount > 1 || compactChunkLayout) {
        logInfo("Using dynamic CTAC attenuation factors.");
        dynamicAttenFactors.resize(static_cast<size_t>(bedCount));
        const size_t plane = static_cast<size_t>(baseNx) * static_cast<size_t>(baseNy);
        for (int bed = 0; bed < bedCount; ++bed) {
          std::vector<double> localMu(baseNvox, 0.0);
          for (int z = 0; z < baseNz; ++z) {
            const int globalZ = bedZStarts[static_cast<size_t>(bed)] + z;
            const size_t srcOffset = static_cast<size_t>(globalZ) * plane;
            const size_t dstOffset = static_cast<size_t>(z) * plane;
            std::copy_n(ctacMuMap.data() + srcOffset, plane, localMu.data() + dstOffset);
          }
          dynamicAttenFactors[static_cast<size_t>(bed)] =
              computeAttenFactors(ctacModel, localMu, baseNx, baseNy, baseNz,
                                  spacing.dx, spacing.dy, spacing.dz,
                                  args.numAngles, args.ctacRotSign,
                                  args.numThreads);
        }
      } else {
        logInfo("Applying CTAC directly to in-memory CSR.");
        applyCTACToCSR(sysCSR, ctacMuMap, baseNx, baseNy, baseNz, spacing.dx,
                       spacing.dy, spacing.dz, args.numAngles, args.numThreads,
                       ctacModel, args.ctacRotSign);
      }
    }

    SparseBackendChoice resolvedBackend;
    bool backendParsed = false;
    const ISparseOps &sparseOps =
        selectSparseOps(args.sparseBackend, &resolvedBackend, &backendParsed);
    if (!backendParsed)
      logWarn("Unknown sparse backend '" + args.sparseBackend + "'. Falling back to auto.");
    logInfo(std::string("Sparse backend: ") + sparseBackendChoiceName(resolvedBackend));
    if (compactLayout && resolvedBackend != SparseBackendChoice::CUDA) {
      throw std::runtime_error(
          "compact-symmetric layout requires an enabled CUDA sparse backend.");
    }

    ReconOpts opts;
    opts.threads = std::max(1, args.numThreads);
    opts.n_subsets = std::max(1, args.numSubsets);
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
    if ((crop.start != 0 || crop.end != 0) && !isCropSettingsValid(args.Nz, crop))
      throw std::runtime_error("Invalid output crop range.");
    OutputTransform outputTransform;
    const OutputTransform *outputTransformPtr = nullptr;
    const bool dfListMode = !args.projPaths.empty();
    const bool enableBedZFlip = args.zBedFlip && dfListMode;
    if (enableBedZFlip || args.zOutFlip) {
      outputTransform.flipZPerBed = enableBedZFlip;
      outputTransform.flipZWholeOutput = args.zOutFlip;
      outputTransform.baseNz = baseNz;
      outputTransform.bedZStarts = bedZStarts;
      outputTransformPtr = &outputTransform;
      if (enableBedZFlip)
        logInfo("Output transform: per-bed Z flip enabled for dfList bed segments.");
      if (args.zOutFlip)
        logInfo("Output transform: whole-output Z flip enabled.");
    }

    const auto runReconstruction = [&](const auto &matrix) {
      using MatrixType = std::decay_t<decltype(matrix)>;

      if (compactLayout) {
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
        return CompactSymmetricReconLoop(compactContext).run(matrix);
      }

      StandardReconLoopContext standardContext;
      standardContext.args = &args;
      standardContext.opts = &opts;
      standardContext.prepared = &prepared;
      standardContext.projections = &projections;
      standardContext.mask = maskPtr;
      standardContext.sharedAnatomical = &sharedAnatomical;
      standardContext.psfKernels = psfPtr;
      standardContext.spacing = spacing;
      standardContext.outputGeometry = outputGeometry;
      standardContext.crop = crop;
      standardContext.outputTransform = outputTransformPtr;
      standardContext.resolvedBackend = resolvedBackend;
      standardContext.bedCount = bedCount;
      standardContext.baseNx = baseNx;
      standardContext.baseNy = baseNy;
      standardContext.baseNz = baseNz;
      standardContext.globalNvox = globalNvox;
      standardContext.bedZStarts = &bedZStarts;
      return StandardReconLoop(standardContext).run(matrix);
    };

    if (compactChunkLayout)
      (void)runReconstruction(spect::io::CSRView{});
    else if (useMappedCSR)
      (void)runReconstruction(csrMap.view());
    else
      (void)runReconstruction(sysCSR);

    logInfo("Reconstruction finished successfully.");
    return spect::cli::kExitSuccess;
  } catch (const std::exception &ex) {
    std::cerr << "[FATAL] " << ex.what() << '\n';
    return spect::cli::kExitRuntimeError;
  }
}

} // namespace spect::main
