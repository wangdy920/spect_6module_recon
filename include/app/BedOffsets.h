#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace spect {
namespace app {

inline int overlapValueAt(const std::vector<int>& bedOverlaps, int jointIndex) {
    if (jointIndex < 0) {
        return 0;
    }
    const size_t idx = static_cast<size_t>(jointIndex);
    if (idx >= bedOverlaps.size()) {
        return 0;
    }
    return bedOverlaps[idx];
}

inline std::vector<int> computeBedZOffsets(int bedCount, int baseNz,
                                           const std::vector<int>& bedOverlaps) {
    const int actualBeds = std::max(1, bedCount);
    std::vector<int> bedZOffsets(static_cast<size_t>(actualBeds), 0);
    int currentZ = 0;
    for (int b = 0; b < actualBeds; ++b) {
        bedZOffsets[static_cast<size_t>(b)] = currentZ;
        if (b < actualBeds - 1) {
            const int overlap = overlapValueAt(bedOverlaps, b);
            currentZ += (baseNz - overlap);
        }
    }
    return bedZOffsets;
}

inline void accumulateBedVolumeToGlobal(std::vector<double>& global,
                                        const std::vector<double>& local,
                                        int nx, int ny, int baseNz,
                                        int zStart) {
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    for (int z = 0; z < baseNz; ++z) {
        const size_t dstOffset = static_cast<size_t>(zStart + z) * plane;
        const size_t srcOffset = static_cast<size_t>(z) * plane;
        for (size_t xy = 0; xy < plane; ++xy) {
            global[dstOffset + xy] += local[srcOffset + xy];
        }
    }
}

inline std::vector<double> extractBedVolumeFromGlobal(
    const std::vector<double>& global, int nx, int ny, int baseNz,
    int zStart) {
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    std::vector<double> local(plane * static_cast<size_t>(baseNz), 0.0);
    for (int z = 0; z < baseNz; ++z) {
        const size_t srcOffset = static_cast<size_t>(zStart + z) * plane;
        const size_t dstOffset = static_cast<size_t>(z) * plane;
        std::copy_n(global.data() + srcOffset, plane, local.data() + dstOffset);
    }
    return local;
}

inline int computeTotalOverlap(int bedCount, const std::vector<int>& bedOverlaps) {
    const int joints = std::max(0, bedCount - 1);
    int total = 0;
    for (int i = 0; i < joints; ++i) {
        total += overlapValueAt(bedOverlaps, i);
    }
    return total;
}

inline int inferBaseNzFromCombinedNz(int nzCandidate, int bedCount,
                                     const std::vector<int>& bedOverlaps) {
    if (bedCount <= 1) {
        return nzCandidate;
    }
    const int totalOverlap = computeTotalOverlap(bedCount, bedOverlaps);
    const int numerator = nzCandidate + totalOverlap;
    if (numerator > 0 && numerator % bedCount == 0) {
        const int inferred = numerator / bedCount;
        // Heuristic: only reinterpret candidate as "combined Nz" when
        // candidate is clearly larger than a single-bed base Nz.
        // This avoids shrinking user-provided base Nz in common 1/2-bed paths.
        if (inferred > 0 && bedCount > 2 &&
            nzCandidate > inferred * (bedCount - 1)) {
            return inferred;
        }
    }
    return nzCandidate;
}

} // namespace app
} // namespace spect
