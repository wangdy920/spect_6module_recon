/**
 * @file test_cuda_scaffold_main.cpp
 * @brief GoogleTest main entry for CUDA scaffold tests with explicit device
 *        synchronization on exit to avoid intermittent Windows driver teardown
 *        crashes.
 */

#include <gtest/gtest.h>

#include <cstdlib>

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
#include <cuda_runtime.h>
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)
  cudaDeviceSynchronize();
#endif
  std::fflush(stdout);
  std::fflush(stderr);
  std::quick_exit(result);
}
