#include "main/ReconArgsPrinter.h"

#include "main/RunLogger.h"

#include <string>

namespace spect {
namespace main {

void printReconArgs(const ReconArgs &args) {
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
