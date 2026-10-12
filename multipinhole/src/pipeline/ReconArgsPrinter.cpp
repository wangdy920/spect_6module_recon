#include "pipeline/ReconArgsPrinter.h"

#include "core/SixModuleGeometry.h"
#include "pipeline/RunLogger.h"

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace spect {
namespace main {
namespace {

std::string joinPaths(const std::vector<std::string> &paths) {
  if (paths.empty())
    return {};
  std::ostringstream oss;
  for (size_t i = 0; i < paths.size(); ++i) {
    if (i != 0)
      oss << "; ";
    oss << paths[i];
  }
  return oss.str();
}

std::string valueOrUnset(const std::string &value) {
  return value.empty() ? "<unset>" : value;
}

std::string psfModeName(spect::filters::PSFMode mode) {
  switch (mode) {
  case spect::filters::PSFMode::None:
    return "off";
  case spect::filters::PSFMode::Operator:
    return "operator";
  case spect::filters::PSFMode::Sieve:
    return "sieve";
  }
  return "unknown";
}

std::string ctacSummary(const ReconArgs &args) {
  if (!args.doCTAC)
    return "off";
  std::ostringstream oss;
  oss << args.ctacModel << ", factorBackend=cuda"
      << ", rotSign=" << args.ctacRotSign;
  if (!args.attMapPath.empty())
    oss << ", muMap=" << args.attMapPath;
  if (!args.ctacAutoMuFrom.empty())
    oss << ", autoMuFrom=" << args.ctacAutoMuFrom
        << ", nuclide=" << args.ctacAutoMuValueSource
        << ", mu=" << args.ctacAutoMuValue;
  return oss.str();
}

} // namespace

void printReconArgs(const ReconArgs &args) {
  std::ostringstream oss;
  oss << "[CONFIG] Effective reconstruction settings (after config/metadata overrides):\n"
      << "  global: " << (args.configLoaded ? args.configPath : "<internal defaults>") << "\n"
      << "  profile: " << valueOrUnset(args.algoConfPath) << "\n"
      << "  priority: " << (args.reconSectionsOverrideCli
              ? "profile > global > CLI" : "CLI > profile > global")
      << "; matrix metadata owns image geometry; runtime I/O stays caller-owned\n"
      << "  scanner: name=" << valueOrUnset(args.scannerName)
      << ", config=" << valueOrUnset(args.scannerConfig) << "\n"
      << "  acquisition: isotope=" << valueOrUnset(args.isotope)
      << ", projection_model=" << valueOrUnset(args.projectionModel)
      << ", matrix_channels=" << (args.dualWindow() ? 2 : 1) << "\n"
      << "  projection_1: " << (args.projPaths.empty()
              ? valueOrUnset(args.projPath) : joinPaths(args.projPaths)) << "\n"
      << "  matrix_1: " << valueOrUnset(args.chunkCSRDir) << "\n";
  if (args.dualWindow()) {
    oss << "  projection_2: " << (args.projPaths2.empty()
                ? valueOrUnset(args.projPath2) : joinPaths(args.projPaths2)) << "\n"
        << "  matrix_2: " << args.chunkCSRDir2 << "\n"
        << "  window_weights:";
    for (double weight : args.energyWindowWeights)
      oss << ' ' << weight;
    oss << "\n";
  }
  oss << "  recon: algorithm=" << args.getAlgorithmName()
      << ", iterations=" << args.numIters
      << ", subset_schedule=" << args.subsetSchedule
      << ", save_every=" << (args.saveInterval > 0
              ? std::to_string(args.saveInterval) : "final_only") << "\n"
      << "  geometry: recon_grid=" << args.Nx << "x" << args.Ny << "x" << args.Nz
      << ", voxel_mm=" << args.Vx << "," << args.Vy << "," << args.Vz
      << ", angles=" << args.numAngles
      << ", detector_zx=" << args.detNz << "x" << args.detNx << "\n"
      << "  beds: count=" << args.getEffectiveBedCount()
      << ", step_mm=" << (args.bedStepMm > 0.0 || args.matrixGridOutput
              ? args.bedStepMm : spect::core::sixmodule::kMultiBedStepMm)
      << ", z_bed_flip=" << args.zBedFlip
      << ", z_out_flip=" << args.zOutFlip
      << ", x_out_flip=" << args.xOutFlip
      << ", xy_out_rotation_deg=" << args.xyOutRotationDeg << "\n"
      << "  output: dir=" << args.outputDir
      << ", name=" << (args.outputName.empty() ? "recon" : args.outputName)
      << ", matrix_grid_output=" << args.matrixGridOutput
      << "\n"
      << "  psf: " << psfModeName(args.usePSF ? args.psf.mode : PSFMode::None)
      << "/" << (args.psf.kernel == spect::filters::PSFKernel::Castor
                     ? "castor"
                     : "adjoint")
      << " fwhm_mm=" << args.psf.fwhm_mm
      << " z_blur=" << (args.psf.z_blur ? "on" : "off")
      << " trunc_sigma=" << args.psf.min_sigmas << "\n"
      << "  regularization: " << (args.useReg ? regTypeToString(args.reg.type) : "off");
  if (args.useReg) {
    oss << ", beta=" << args.reg.beta << ", gamma=" << args.reg.gamma
        << ", delta=" << args.reg.delta
        << ", CT=" << (args.regCtPath.empty() ? "off" : args.regCtPath)
        << ", ct_sigma=" << args.reg.ct_sigma
        << ", ct_floor=" << args.reg.ct_weight_floor;
  }
  if (args.algorithm == ReconArgs::Algorithm::BSREM) {
    oss << "\n  bsrem: alpha0=" << args.bsrem_alpha0
        << ", gamma=" << args.bsrem_gamma
        << ", kappa=" << args.bsremUseKappaMap
        << ", kappa_clip=" << args.bsremKappaClipMin << "," << args.bsremKappaClipMax
        << ", curvature=" << args.bsremUseCurvDenom;
  }
  oss << "\n  corrections: ctac=" << ctacSummary(args)
      << ", norm_1=" << valueOrUnset(args.normFactorFile)
      << ", norm_2=" << valueOrUnset(args.normFactorFile2)
      << ", background_1=" << (args.bProjPaths.empty()
              ? valueOrUnset(args.bProjPath) : joinPaths(args.bProjPaths))
      << ", background_2=" << (args.bProjPaths2.empty()
              ? valueOrUnset(args.bProjPath2) : joinPaths(args.bProjPaths2))
      << "\n  mask: cylinder=" << args.useCylindricalMask
      << ", diameter_height_mm=" << args.cylinderDiameterMM << "," << args.cylinderHeightMM
      << ", bad_detector_file=" << valueOrUnset(args.badDetectorFile)
      << "\n  filters: smooth:" << args.smoothApplyStage
      << ",butter:" << args.butterApplyStage
      << ", output_smooth_fwhm_mm=" << args.outputSmoothFwhmMm
      << ", output_butter_cutoff_cpcm=" << args.outputButterCutoffCpcm
      << ", output_butter_order=" << args.outputButterOrder
      << ", output_clamp="
      << (args.outputNonnegativeClamp ? "nonnegative" : "linear")
      << ", primary_output="
      << (args.outputFiltersOverwritePrimary ? "filter_chain" : "unfiltered");
  logInfo(oss.str());

  logDebug("Threads: " + std::to_string(args.numThreads) +
           ", sparse backend: cuda");
  logDebug(std::string("CUDA PSF precision: ") +
           ((std::getenv("SPECT_CUDA_PSF_DOUBLE") &&
             std::string(std::getenv("SPECT_CUDA_PSF_DOUBLE")) == "1")
                ? "double" : "float"));
  logDebug(std::string("CUDA volume ops: ") +
           ((std::getenv("SPECT_CUDA_VOLUME_OPS") &&
             std::string(std::getenv("SPECT_CUDA_VOLUME_OPS")) == "0")
                ? "disabled" : "enabled"));
}

} // namespace main
} // namespace spect
