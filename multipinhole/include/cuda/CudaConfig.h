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
 * @brief Format CUDA API version integers (e.g. 13020 -> "13.2").
 */
inline std::string formatCudaApiVersion(int version) {
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

/**
 * @brief Snapshot of driver / runtime / device identity for startup logs.
 */
struct CudaEnvironmentInfo {
    int driverVersion = 0;
    int runtimeVersion = 0;
    int deviceId = 0;
    int deviceCount = 0;
    int computeMajor = 0;
    int computeMinor = 0;
    std::string deviceName;
};

inline CudaEnvironmentInfo queryEnvironment(int deviceId = 0) {
    CudaEnvironmentInfo info;
    CUDA_CHECK(cudaDriverGetVersion(&info.driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&info.runtimeVersion));
    info.deviceCount = getDeviceCount();
    if (info.deviceCount <= 0) {
        throw std::runtime_error("No CUDA devices are available.");
    }
    if (deviceId < 0 || deviceId >= info.deviceCount) {
        throw std::runtime_error("CUDA device id " + std::to_string(deviceId) +
                                 " is out of range (device count=" +
                                 std::to_string(info.deviceCount) + ").");
    }
    info.deviceId = deviceId;
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    info.deviceName = props.name ? props.name : "(unknown)";
    info.computeMajor = props.major;
    info.computeMinor = props.minor;
    return info;
}

/**
 * @brief Ensure the installed driver can host the bundled CUDA 13.x runtime
 *        (minor-version compatibility floor: driver CUDA >= 13.0).
 */
inline void ensureDriverSupportsBundledRuntime(const CudaEnvironmentInfo &info) {
    // Bundled cudart from CUDA 13.x require a CUDA 13-capable driver
    // (display driver family >= R580 per NVIDIA minor-version compatibility).
#if defined(SPECT_CUDA_MIN_DRIVER_API_VERSION)
    constexpr int kMinDriverCudaVersion = SPECT_CUDA_MIN_DRIVER_API_VERSION;
#else
    constexpr int kMinDriverCudaVersion = 13000; // 13.0
#endif
    if (info.driverVersion < kMinDriverCudaVersion) {
        throw std::runtime_error(
            "CUDA driver supported version is " +
            formatCudaApiVersion(info.driverVersion) +
            ", but this SPECT_recon build bundles CUDA 13.x runtime "
            "(needed for RTX 50-series sm_120 fatbins). Install an NVIDIA driver with "
            "CUDA 13.x support (R580+ / CUDA 13.0+), then retry.");
    }
    if (info.driverVersion < info.runtimeVersion) {
        // Soft warning path is handled by caller; keep hard fail only for
        // major-family mismatches above.
    }
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

inline uint64_t getL2CacheBytes() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return static_cast<uint64_t>(props.l2CacheSize);
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
