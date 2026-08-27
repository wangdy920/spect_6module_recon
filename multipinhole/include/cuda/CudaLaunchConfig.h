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

constexpr uint64_t kCompactChunkHostCacheFallbackBytes = 26ull * kCudaGiB;
constexpr uint64_t kCompactChunkHostCacheMinReserveBytes = 8ull * kCudaGiB;
constexpr uint64_t kCompactChunkResidentDeviceReserveBytes = 6ull * kCudaGiB;
constexpr uint64_t kDeviceChunkCacheReserveBytes = 2ull * 3ull * kCudaGiB;

constexpr size_t kTransientRangeSlotCount = 2;
constexpr int kCompactDeviceBackwardMaxAccumulators = 5;

// Packed-K forward: use ~60% of L2 for ROW-packed X; leave ~40% for CSR
// streaming. K is chosen from {1,2,4}. packed-4 also needs sorted_hint
// (performance confidence, not correctness). Unsorted or K_mem<4 stays
// on packed-2. Transpose is per-RHS CSR_ALG2.
constexpr double kCompactPackedXl2BudgetFraction = 0.60;

inline int compactPackedForwardKMem(int64_t nCols, int batchSize,
                                    uint64_t l2Bytes) {
  if (nCols <= 0 || batchSize < 2) {
    return 1;
  }
  if (l2Bytes == 0) {
    return 2;
  }
  const double vecBytes = 4.0 * static_cast<double>(nCols);
  const double budget =
      kCompactPackedXl2BudgetFraction * static_cast<double>(l2Bytes);
  const int kRaw = static_cast<int>(budget / vecBytes);
  if (kRaw >= 4 && batchSize >= 4) {
    return 4;
  }
  if (kRaw >= 2) {
    return 2;
  }
  return 1;
}

inline int compactPackedForwardTile(int kMem, bool sortedHint) {
  if (kMem < 2) {
    return 1;
  }
  if (kMem >= 4 && sortedHint) {
    return 4;
  }
  return 2;
}

} // namespace spect::cuda
