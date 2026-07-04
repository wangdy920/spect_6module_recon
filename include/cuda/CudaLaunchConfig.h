#pragma once

#include <cstddef>
#include <cstdint>

namespace spect::cuda {

constexpr uint64_t kCudaMiB = 1024ull * 1024ull;
constexpr uint64_t kCudaGiB = 1024ull * 1024ull * 1024ull;

constexpr unsigned int kCudaLaunchBlockSize = 256;

constexpr uint64_t kProjDomainChunkMinNnz = 8ull * 1024ull * 1024ull;
constexpr uint64_t kProjDomainChunkDefaultNnz = 48ull * 1024ull * 1024ull;
constexpr uint64_t kProjDomainChunkMaxNnz = 256ull * 1024ull * 1024ull;
constexpr uint64_t kProjDomainChunkBytesPerNnz =
    static_cast<uint64_t>(sizeof(int32_t) + sizeof(float));
constexpr uint64_t kProjDomainChunkFreeMemoryReserveBytes = 768ull * kCudaMiB;
constexpr uint64_t kProjDomainChunkTargetBytesCap = 4ull * kCudaGiB;
constexpr uint64_t kProjDomainChunkHighMemThresholdBytes = 12ull * kCudaGiB;
constexpr uint64_t kProjDomainChunkLowMemTargetBytesCap = 3ull * kCudaGiB;
constexpr uint64_t kProjDomainChunkHighMemDivisor = 2ull;
constexpr uint64_t kProjDomainChunkLowMemDivisor = 3ull;

constexpr uint64_t kPreparedChunkHostCacheMaxBytes = 24ull * kCudaGiB;
constexpr uint64_t kPreparedChunkHostCacheMidBytes = 8ull * kCudaGiB;
constexpr uint64_t kPreparedChunkHostCacheMinBytes = 2ull * kCudaGiB;
constexpr uint64_t kPreparedChunkHostCacheEmergencyBytes = 512ull * kCudaMiB;
constexpr uint64_t kPreparedChunkHostCacheSafetyReserveBytes = 4ull * kCudaGiB;
constexpr uint64_t kPreparedChunkHostCacheHighRamThresholdBytes = 60ull * kCudaGiB;
constexpr uint64_t kPreparedChunkHostCacheMidRamThresholdBytes = 28ull * kCudaGiB;
constexpr uint64_t kPreparedChunkHostCacheEffectiveAvailableNumerator = 2ull;
constexpr uint64_t kPreparedChunkHostCacheEffectiveAvailableDenominator = 3ull;

constexpr uint64_t kCompactChunkHostCacheDefaultBytes = 24ull * kCudaGiB;
constexpr uint64_t kCompactChunkResidentDeviceReserveBytes = 2ull * kCudaGiB;
constexpr uint64_t kDeviceChunkCacheReserveBytes = 2ull * 3ull * kCudaGiB;

constexpr size_t kTransientRangeSlotCount = 2;
constexpr int kCompactDeviceBackwardMaxAccumulators = 5;

} // namespace spect::cuda