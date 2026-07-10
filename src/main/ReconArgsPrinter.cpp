#include "main/ReconArgsPrinter.h"

#include "cli/Version.h"
#include "main/RunLogger.h"

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

std::string projectionSource(const ReconArgs &args) {
  if (!args.projPaths.empty())
    return joinPaths(args.projPaths);
  return args.projPath.empty() ? std::string("<unset>") : args.projPath;
}

std::string matrixSource(const ReconArgs &args) {
  if (!args.chunkCSRDir.empty())
    return "chunkCSRDir=" + args.chunkCSRDir;
  if (!args.loadCSRPath.empty())
    return "loadCSR=" + args.loadCSRPath;
  if (!args.sysIndexPath.empty() || !args.sysWeightPath.empty())
    return "mat-index=" + args.sysIndexPath + "; mat-weight=" +
           args.sysWeightPath;
  return "<auto/config>";
}

std::string psfModeName(spect::filters::PSFMode mode) {
  switch (mode) {
  case spect::filters::PSFMode::None:
    return "off";
  case spect::filters::PSFMode::Operator:
    return "operator";
  case spect::filters::PSFMode::PostSmooth:
    return "post-smooth";
  case spect::filters::PSFMode::ProjDomain:
    return "projection-domain";
  case spect::filters::PSFMode::Gaussian:
    return "gaussian";
  case spect::filters::PSFMode::File:
    return "file";
  }
  return "unknown";
}

std::string ctacSummary(const ReconArgs &args) {
  if (!args.doCTAC)
    return "off";
  std::ostringstream oss;
  oss << args.ctacModel << ", rotSign=" << args.ctacRotSign;
  if (!args.attMapPath.empty())
    oss << ", muMap=" << args.attMapPath;
  if (!args.ctacAutoMuFrom.empty())
    oss << ", autoMuFrom=" << args.ctacAutoMuFrom
        << ", nuclide=" << args.ctacAutoMuValueSource
        << ", mu=" << args.ctacAutoMuValue;
  return oss.str();
}

} // namespace

void printReconArgs(const ReconArgs &args, const std::string &commandLine) {
  std::ostringstream oss;
  oss << "Run configuration [SPECT_recon " << spect::cli::versionString()
      << "]:\n"
      << "  command: " << (commandLine.empty() ? "<unset>" : commandLine)
      << "\n"
      << "  data: projection=" << projectionSource(args) << "\n"
      << "  matrix: " << matrixSource(args) << "\n"
      << "  output: dir=" << args.outputDir
      << ", name=" << (args.outputName.empty() ? "recon" : args.outputName)
      << "\n"
      << "  recon: opti=" << args.getAlgorithmName()
      << ", iters=" << args.numIters << ", subsets=" << args.numSubsets
      << ", image=" << args.Nx << "x" << args.Ny << "x" << args.Nz
      << ", voxel_mm=" << args.Vx << "," << args.Vy << "," << args.Vz
      << "\n"
      << "  physics: ctac=" << ctacSummary(args)
      << ", psf=" << psfModeName(args.psf.mode)
      << " fwhm_mm=" << args.psf.fwhm_mm
      << " trunc_sigma=" << args.psf.min_sigmas
      << ", butter=" << (args.useButter ? "on" : "off")
      << ", reg=" << (args.useReg ? regTypeToString(args.reg.type) : "off");
  logInfo(oss.str());

  logDebug(std::string("Algorithm: ") + args.getAlgorithmName());
  logDebug("Dims: " + std::to_string(args.Nx) + "x" +
           std::to_string(args.Ny) + "x" + std::to_string(args.Nz));
  logDebug("Voxel: " + std::to_string(args.Vx) + "," +
           std::to_string(args.Vy) + "," + std::to_string(args.Vz) + " mm");
  logDebug("Angles: " + std::to_string(args.numAngles) +
           ", subsets: " + std::to_string(args.numSubsets) +
           ", iterations: " + std::to_string(args.numIters));
  logDebug("Threads: " + std::to_string(args.numThreads) +
           ", sparse backend: cuda");
}

} // namespace main
} // namespace spect
