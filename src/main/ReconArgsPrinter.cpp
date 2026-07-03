#include "main/ReconArgsPrinter.h"

#include <iostream>

namespace spect {
namespace main {

void printReconArgs(const ReconArgs &args) {
  std::cout << "Algorithm: " << args.getAlgorithmName() << '\n';
  std::cout << "Dims: " << args.Nx << 'x' << args.Ny << 'x' << args.Nz << '\n';
  std::cout << "Voxel: " << args.Vx << ',' << args.Vy << ',' << args.Vz << " mm\n";
  std::cout << "Angles: " << args.numAngles << ", subsets: " << args.numSubsets
            << ", iterations: " << args.numIters << '\n';
  std::cout << "Threads: " << args.numThreads << ", sparse backend: "
            << args.sparseBackend << '\n';
}

} // namespace main
} // namespace spect