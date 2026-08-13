/**
 * @file test_main.cpp
 * @brief GoogleTest 主入口文件
 * 
 * 所有测试套件共享此入口点
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
    // Drain outstanding GPU work before tearing down the process. On Windows
    // we have observed intermittent access-violation crashes during CRT exit
    // when CUDA driver state is reclaimed while work is still in flight.
    cudaDeviceSynchronize();
#endif
    // Flush output streams so CTest captures any final diagnostics, then use
    // quick_exit to bypass further static destructors that may touch CUDA state.
    std::fflush(stdout);
    std::fflush(stderr);
    std::quick_exit(result);
}
