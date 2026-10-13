#pragma once

#include "core/ReconLimits.h"

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

constexpr uint64_t kCompactChunkHostCacheFallbackBytes = 26ull * kCudaGiB;
constexpr uint64_t kCompactChunkHostCacheMinReserveBytes = 8ull * kCudaGiB;
constexpr uint64_t kCompactChunkResidentDeviceReserveBytes = 6ull * kCudaGiB;
// CSRQ keeps the whole matrix resident and runs custom kernels; its working
// set is the [bed][branch][voxel] batch input plus two [voxel][K] scratch
// vectors, accumulators and volumes (~1 GiB at 120x120x220, 5 beds), so the
// reserve only needs to cover those and the PSF/regularizer volume helpers.
constexpr uint64_t kCsrqResidentDeviceReserveBytes = 2ull * kCudaGiB;
// Energy windows (system matrices) that may be resident at once for joint
// multi-window MLEM (Lu-177: 113 keV + 208 keV).
constexpr int kCsrqMaxMatrices = 2;
// Streamed CSRQ matrices use two device chunk slots; active symmetry K is
// selected from the validated matrix profile (4 or 8), not a device constant.
constexpr int kCsrqStreamSlots = 2;

// One backward accumulator per bed.
constexpr int kCompactDeviceBackwardMaxAccumulators = spect::core::kMaxBeds;

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

// CSRB numerator backward: the ratio is exactly 0 for zero-count bins, so a
// pair-RHS atomic scatter over CSR(A) that skips all-zero rows beats per-RHS
// cuSPARSE CSR_ALG2 transpose once enough bins are empty. Measured on the
// Lu-177 208 keV chunk 0 (730 M nnz, RTX 4070 Ti SUPER, tools/transpose_bench
// scatter mode, i.i.d. zero pattern): zero fraction 0.15 -> 0.84x, 0.30 ->
// 1.00x, 0.40 -> 1.16x, 0.80 -> 3.18x, 0.90 -> 6.35x; dense 0.71x. The
// five-bed Lu-177 input has 74-90 % zero bins. Below the threshold the
// per-RHS cuSPARSE transpose stays.
constexpr double kCompactScatterBackwardZeroFractionMin = 0.40;

inline bool compactScatterBackwardPreferred(double projectionZeroFraction,
                                            int batchSize) {
  return batchSize >= 2 &&
         projectionZeroFraction >= kCompactScatterBackwardZeroFractionMin;
}

} // namespace spect::cuda
