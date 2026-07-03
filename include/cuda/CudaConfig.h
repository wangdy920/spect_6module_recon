/**
 * @file CudaConfig.h
 * @brief 【T6.1】CUDA配置和条件编译宏
 * 
 * 提供CUDA可用性检测和错误处理宏。
 */

#pragma once

// CUDA可用性检测
#ifdef SPECT_HAS_CUDA
    #define CUDA_ENABLED 1
#else
    #define CUDA_ENABLED 0
#endif

#if CUDA_ENABLED

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace spect {
namespace cuda {

/**
 * @brief CUDA错误检查
 */
inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            ": " + cudaGetErrorString(err));
    }
}

/**
 * @brief CUDA错误检查宏
 */
#define CUDA_CHECK(call) ::spect::cuda::checkCudaError((call), __FILE__, __LINE__)

/**
 * @brief 获取设备数量
 */
inline int getDeviceCount() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

/**
 * @brief 检查CUDA是否可用
 */
inline bool isCudaAvailable() {
    try {
        return getDeviceCount() > 0;
    } catch (...) {
        return false;
    }
}

/**
 * @brief 设置当前设备
 */
inline void setDevice(int deviceId) {
    CUDA_CHECK(cudaSetDevice(deviceId));
}

/**
 * @brief 同步当前设备
 */
inline void synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief GPU内存信息
 */
struct GpuMemoryInfo {
    size_t free;
    size_t total;
    
    double usedPercent() const {
        return total > 0 ? 100.0 * (total - free) / total : 0.0;
    }
};

/**
 * @brief 获取GPU内存信息
 */
inline GpuMemoryInfo getMemoryInfo() {
    GpuMemoryInfo info;
    CUDA_CHECK(cudaMemGetInfo(&info.free, &info.total));
    return info;
}

} // namespace cuda
} // namespace spect

#else // !CUDA_ENABLED

// Stub definitions when CUDA is not available
namespace spect {
namespace cuda {

inline bool isCudaAvailable() { return false; }
inline int getDeviceCount() { return 0; }

} // namespace cuda
} // namespace spect

#endif // CUDA_ENABLED
