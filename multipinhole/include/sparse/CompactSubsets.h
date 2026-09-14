#pragma once

// Ordered subsets over compact angles (design report §7.10).
//
// Subset k of N holds the compact angles {k, k+N, k+2N, ...}; each compact
// angle expands through the xy symmetries and z mirror to full angles spaced
// 90 degrees apart, so an interleaved subset samples all 360 degrees evenly.
// N must divide the compact angle count so every subset carries the same
// number of angles (subset balance). The visiting order maximises the
// circular distance between consecutively processed subsets (0, N/2, ...),
// the classic OSEM ordering, and the schedule "N1:it1,N2:it2,..." lets a run
// start with many subsets and end with fewer (1 = plain MLEM / BSREM).

#include <string>
#include <utility>
#include <vector>

namespace spect {

struct CompactSubsetSpec {
  int count = 1; // N subsets
  int index = 0; // this subset (0..N-1)
};

// Absolute compact angles of subset spec.index out of spec.count.
std::vector<int> compactSubsetAngles(int compactNumAngles,
                                     const CompactSubsetSpec &spec);

// Deterministic processing order of N subsets: greedy maximal circular
// separation from all previously chosen subsets, ties broken by the distance
// to the most recent choices (newest first), then the smallest index.
// N=10 gives 0 5 2 7 1 6 3 8 4 9; N=4 gives 0 2 1 3.
std::vector<int> compactSubsetOrder(int count);

// One stage of a subset schedule: `iterations` full iterations with `subsets`
// subsets each.
struct CompactSubsetStage {
  int subsets = 1;
  int iterations = 0;
};

// "N" -> all iterations with N subsets; "N1:i1,N2:i2,..." -> stages whose
// iterations must sum to numIters. Every subset count must divide
// compactNumAngles and every later count must divide the previous one, so
// the coarser subset sensitivities are exact sums of the finest ones.
std::vector<CompactSubsetStage>
parseCompactSubsetSchedule(const std::string &text, int numIters,
                           int compactNumAngles);

std::string formatCompactSubsetSchedule(
    const std::vector<CompactSubsetStage> &schedule);

} // namespace spect
