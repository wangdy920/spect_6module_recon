// Ordered subsets over compact angles (design report 7.10): the host-side
// partition / schedule, the (chunk, subset) slice, and the CUDA subset
// statistics against the dense oracle restricted to the subset's rows.

#include <gtest/gtest.h>

#include "cuda/CudaConfig.h"
#include "filters/cuda/VolumeOpsCUDA.h"
#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "recon/BsremUpdate.h"
#include "regularizer/RegularizerCompat.h"
#include "sparse/CompactSubsets.h"
#include "sparse/CompactSymmetricProjector.h"
#include "sparse/SparseOpsCUDA.h"
#include "sparse/ProjDomainChunked.h"
#include "tests/common/compact_symmetry_test_helpers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace spect::main {
namespace {

using spect::io::test::ToyCompactCSR;

// --- host partition / schedule ------------------------------------------------

TEST(CompactSubsets, InterleavedAnglesPartitionTheCompactRange) {
  const int angles = 90;
  for (const int count : {1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90}) {
    std::vector<int> seen;
    for (int index = 0; index < count; ++index) {
      const auto subset =
          spect::compactSubsetAngles(angles, spect::CompactSubsetSpec{count, index});
      ASSERT_EQ(subset.size(), static_cast<size_t>(angles / count));
      for (size_t i = 0; i < subset.size(); ++i) {
        EXPECT_EQ(subset[i], index + static_cast<int>(i) * count);
      }
      seen.insert(seen.end(), subset.begin(), subset.end());
    }
    std::sort(seen.begin(), seen.end());
    std::vector<int> all(static_cast<size_t>(angles));
    std::iota(all.begin(), all.end(), 0);
    EXPECT_EQ(seen, all) << "count " << count;
  }
  EXPECT_THROW(spect::compactSubsetAngles(90, spect::CompactSubsetSpec{7, 0}),
               std::runtime_error);
  EXPECT_THROW(spect::compactSubsetAngles(90, spect::CompactSubsetSpec{3, 3}),
               std::runtime_error);
}

TEST(CompactSubsets, ChunkRangesPreserveGlobalSubsetAngles) {
  for (const int count : {10, 5, 2, 1}) {
    for (int index = 0; index < count; ++index) {
      const spect::CompactSubsetSpec spec{count, index};
      std::vector<int> joined;
      // Boundaries deliberately differ from subset boundaries; some chunks
      // contain no selected angle at all.
      const std::vector<int> bounds{0, 1, 4, 9, 15, 20};
      for (size_t c = 1; c < bounds.size(); ++c) {
        const auto angles = spect::compactSubsetAngles(20, spec, bounds[c - 1], bounds[c]);
        for (const int angle : angles) {
          EXPECT_GE(angle, bounds[c - 1]);
          EXPECT_LT(angle, bounds[c]);
        }
        joined.insert(joined.end(), angles.begin(), angles.end());
      }
      EXPECT_EQ(joined, spect::compactSubsetAngles(20, spec));
    }
  }
  EXPECT_EQ(spect::compactSubsetAngles(20, {10, 0}, 4, 15), (std::vector<int>{10}));
  EXPECT_TRUE(spect::compactSubsetAngles(20, {10, 9}, 4, 9).empty());
  EXPECT_TRUE(spect::compactSubsetAngles(20, {1, 0}, 20, 20).empty());
  EXPECT_EQ(spect::compactSubsetAngles(20, {5, 4}, 15), (std::vector<int>{19}));
  EXPECT_THROW(spect::compactSubsetAngles(20, {1, 0}, -1, 2), std::runtime_error);
  EXPECT_THROW(spect::compactSubsetAngles(20, {1, 0}, 5, 4), std::runtime_error);
  EXPECT_THROW(spect::compactSubsetAngles(20, {1, 0}, 0, 21), std::runtime_error);
  EXPECT_THROW(spect::compactSubsetAngles(20, {3, 0}, 0, 6), std::runtime_error);
}

TEST(CompactSubsets, VisitingOrderMaximisesSeparationAndIsAPermutation) {
  EXPECT_EQ(spect::compactSubsetOrder(1), (std::vector<int>{0}));
  EXPECT_EQ(spect::compactSubsetOrder(2), (std::vector<int>{0, 1}));
  EXPECT_EQ(spect::compactSubsetOrder(4), (std::vector<int>{0, 2, 1, 3}));
  EXPECT_EQ(spect::compactSubsetOrder(10),
            (std::vector<int>{0, 5, 2, 7, 1, 6, 3, 8, 4, 9}));
  for (const int count : {3, 6, 9, 15, 18, 30}) {
    auto order = spect::compactSubsetOrder(count);
    ASSERT_EQ(order.size(), static_cast<size_t>(count));
    // Consecutive sub-iterations never visit neighbouring subsets (with 3
    // subsets every pair is adjacent on the circle, so only count >= 4).
    for (size_t i = 1; count >= 4 && i < order.size(); ++i) {
      const int d = std::abs(order[i] - order[i - 1]);
      EXPECT_GT(std::min(d, count - d), 1) << "count " << count << " step " << i;
    }
    std::sort(order.begin(), order.end());
    std::vector<int> all(static_cast<size_t>(count));
    std::iota(all.begin(), all.end(), 0);
    EXPECT_EQ(order, all);
  }
}

TEST(CompactSubsets, ScheduleParsesRefinesCoarselyAndSumsToIterations) {
  const auto single = spect::parseCompactSubsetSchedule("1", 50, 90);
  ASSERT_EQ(single.size(), 1u);
  EXPECT_EQ(single[0].subsets, 1);
  EXPECT_EQ(single[0].iterations, 50);

  const auto flat = spect::parseCompactSubsetSchedule("10", 8, 90);
  ASSERT_EQ(flat.size(), 1u);
  EXPECT_EQ(flat[0].subsets, 10);
  EXPECT_EQ(flat[0].iterations, 8);

  const auto staged = spect::parseCompactSubsetSchedule("10:3,5:2,1:5", 10, 90);
  ASSERT_EQ(staged.size(), 3u);
  EXPECT_EQ(staged[0].subsets, 10);
  EXPECT_EQ(staged[0].iterations, 3);
  EXPECT_EQ(staged[1].subsets, 5);
  EXPECT_EQ(staged[1].iterations, 2);
  EXPECT_EQ(staged[2].subsets, 1);
  EXPECT_EQ(staged[2].iterations, 5);
  EXPECT_EQ(spect::formatCompactSubsetSchedule(staged), "10:3,5:2,1:5");

  // Iterations must sum to -it.
  EXPECT_THROW(spect::parseCompactSubsetSchedule("10:3,1:5", 10, 90),
               std::runtime_error);
  // N must divide the compact angle count.
  EXPECT_THROW(spect::parseCompactSubsetSchedule("7", 10, 90),
               std::runtime_error);
  // Later stages must divide earlier ones (coarser S_m are exact sums).
  EXPECT_THROW(spect::parseCompactSubsetSchedule("10:5,3:5", 10, 90),
               std::runtime_error);
  EXPECT_THROW(spect::parseCompactSubsetSchedule("5:5,10:5", 10, 90),
               std::runtime_error);
  EXPECT_THROW(spect::parseCompactSubsetSchedule("0", 10, 90),
               std::runtime_error);
  EXPECT_THROW(spect::parseCompactSubsetSchedule("10:x", 10, 90),
               std::runtime_error);
  EXPECT_THROW(spect::parseCompactSubsetSchedule("", 10, 90),
               std::runtime_error);
}

// --- toy matrix with 6 compact angles in two chunks -------------------------

spect::io::CompactSymmetryMetadata makeSixAngleMeta() {
  auto meta = spect::io::test::makeToySym90ZMeta();
  meta.compactNumAngles = 6;
  meta.fullNumAngles = 24;
  meta.totalRows = meta.compactNumAngles * meta.compactRaysPerAngle;
  return meta;
}

// Deterministic sparse rows (2..3 nnz each) over the 8 toy voxels.
ToyCompactCSR makeSixAngleCompactCSR(const spect::io::CompactSymmetryMetadata &meta) {
  ToyCompactCSR csr;
  csr.rowPtr.push_back(0);
  uint32_t state = 12345u;
  auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
  };
  for (int row = 0; row < meta.totalRows; ++row) {
    const int count = 2 + static_cast<int>(next() % 2u);
    std::vector<uint32_t> cols;
    while (static_cast<int>(cols.size()) < count) {
      const uint32_t col = next() % static_cast<uint32_t>(meta.totalVoxels);
      if (std::find(cols.begin(), cols.end(), col) == cols.end())
        cols.push_back(col);
    }
    std::sort(cols.begin(), cols.end());
    for (const uint32_t col : cols) {
      csr.colIdx.push_back(col);
      csr.values.push_back(0.2f + 0.1f * static_cast<float>(next() % 12u));
    }
    csr.rowPtr.push_back(static_cast<uint32_t>(csr.colIdx.size()));
  }
  return csr;
}

void writeChunkFile(const std::filesystem::path &path, int32_t rows,
                    int32_t angleOffset, int32_t angleCount,
                    const std::vector<uint32_t> &rowPtr,
                    const std::vector<uint32_t> &colIdx,
                    const std::vector<float> &values) {
  ASSERT_EQ(rowPtr.size(), static_cast<size_t>(rows) + 1u);
  ASSERT_EQ(colIdx.size(), values.size());
  std::vector<int64_t> rowPtr64(rowPtr.begin(), rowPtr.end());
  std::vector<int32_t> colIdx32(colIdx.begin(), colIdx.end());
  std::ofstream out(path, std::ios::binary);
  ASSERT_TRUE(out);
  const uint32_t magic = 0x42525343u;
  const uint32_t version = 1u;
  const uint64_t nnz = static_cast<uint64_t>(values.size());
  out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char *>(&version), sizeof(version));
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&angleOffset), sizeof(angleOffset));
  out.write(reinterpret_cast<const char *>(&angleCount), sizeof(angleCount));
  out.write(reinterpret_cast<const char *>(&nnz), sizeof(nnz));
  out.write(reinterpret_cast<const char *>(rowPtr64.data()),
            static_cast<std::streamsize>(rowPtr64.size() * sizeof(int64_t)));
  out.write(reinterpret_cast<const char *>(colIdx32.data()),
            static_cast<std::streamsize>(colIdx32.size() * sizeof(int32_t)));
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
}

// Writes the compact matrix as chunks of `anglesPerChunk` consecutive angles.
void writeChunkDirectory(const std::filesystem::path &dir,
                         const spect::io::CompactSymmetryMetadata &meta,
                         const ToyCompactCSR &compact, int anglesPerChunk) {
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const int rpa = meta.compactRaysPerAngle;
  int chunkIndex = 0;
  for (int angle0 = 0; angle0 < meta.compactNumAngles;
       angle0 += anglesPerChunk, ++chunkIndex) {
    const int angleCount = std::min(anglesPerChunk, meta.compactNumAngles - angle0);
    const size_t row0 = static_cast<size_t>(angle0) * rpa;
    const size_t row1 = row0 + static_cast<size_t>(angleCount) * rpa;
    const uint32_t nnz0 = compact.rowPtr[row0];
    const uint32_t nnz1 = compact.rowPtr[row1];
    std::vector<uint32_t> rowPtr;
    for (size_t r = row0; r <= row1; ++r)
      rowPtr.push_back(compact.rowPtr[r] - nnz0);
    std::vector<uint32_t> colIdx(compact.colIdx.begin() + nnz0,
                                 compact.colIdx.begin() + nnz1);
    std::vector<float> values(compact.values.begin() + nnz0,
                              compact.values.begin() + nnz1);
    char name[32];
    std::snprintf(name, sizeof(name), "chunk_%03d.csrb", chunkIndex);
    writeChunkFile(dir / name, static_cast<int32_t>(angleCount * rpa), angle0,
                   angleCount, rowPtr, colIdx, values);
  }
}

// Full dense matrix with the rows of full angles outside subset (count, index)
// zeroed: compact angle of full angle f is f % compactNumAngles (SYM_90).
std::vector<double> restrictToSubset(const std::vector<double> &full,
                                     const spect::io::CompactSymmetryMetadata &meta,
                                     int count, int index) {
  std::vector<double> out = full;
  const size_t cols = static_cast<size_t>(meta.totalVoxels);
  const size_t rpaFull = static_cast<size_t>(meta.fullRaysPerAngle());
  for (int fullAngle = 0; fullAngle < meta.fullNumAngles; ++fullAngle) {
    const int compactAngle = fullAngle % meta.compactNumAngles;
    if (compactAngle % count == index)
      continue;
    std::fill(out.begin() + static_cast<ptrdiff_t>(fullAngle * rpaFull * cols),
              out.begin() +
                  static_cast<ptrdiff_t>((fullAngle + 1) * rpaFull * cols),
              0.0);
  }
  return out;
}

double relL2(const std::vector<double> &a, const std::vector<double> &b) {
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    num += (a[i] - b[i]) * (a[i] - b[i]);
    den += b[i] * b[i];
  }
  return std::sqrt(num) / (std::sqrt(den) + 1e-30);
}

TEST(CompactSubsets, ChunkSliceCoalescesAdjacentAnglesAndRebasesRowPointers) {
  const auto meta = makeSixAngleMeta();
  const auto compact = makeSixAngleCompactCSR(meta);
  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_compact_subset_slice_test";
  writeChunkDirectory(dir, meta, compact, /*anglesPerChunk=*/6);
  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunks =
      spect::io::openCompactChunkCSRDirectory(dir.string(), dirMeta, 1 << 20);
  ASSERT_EQ(chunks.chunks().size(), 1u);
  const auto chunk = chunks.loadChunk(0);
  const int rpa = meta.compactRaysPerAngle;

  // Whole chunk: one segment, identical row pointers, tag 0.
  const auto whole = chunk->subsetView(
      spect::compactSubsetAngles(6, spect::CompactSubsetSpec{1, 0}), rpa, 0);
  ASSERT_TRUE(whole.valid());
  EXPECT_EQ(whole.rows, meta.totalRows);
  EXPECT_EQ(whole.blockCount(), 6);
  EXPECT_EQ(whole.segments.size(), 1u);
  EXPECT_EQ(whole.nnz, static_cast<uint64_t>(compact.values.size()));
  EXPECT_EQ(whole.row_ptr, compact.rowPtr);
  EXPECT_EQ(whole.segments[0].col_idx, chunk->view().col_idx);

  // Interleaved subset 1 of 2: angles 1, 3, 5 -> three segments, rows and
  // nnz re-based over the concatenation.
  const auto odd = chunk->subsetView(
      spect::compactSubsetAngles(6, spect::CompactSubsetSpec{2, 1}), rpa, 7);
  ASSERT_TRUE(odd.valid());
  EXPECT_EQ(odd.tag, 7u);
  EXPECT_EQ(odd.rows, 3 * rpa);
  EXPECT_EQ(odd.blockAngles, (std::vector<int>{1, 3, 5}));
  EXPECT_EQ(odd.segments.size(), 3u);
  uint64_t nnz = 0;
  for (size_t b = 0; b < 3; ++b) {
    const int angle = odd.blockAngles[b];
    const uint32_t start = compact.rowPtr[static_cast<size_t>(angle * rpa)];
    const uint32_t end = compact.rowPtr[static_cast<size_t>((angle + 1) * rpa)];
    EXPECT_EQ(odd.segments[b].nnz, end - start);
    EXPECT_EQ(odd.segments[b].col_idx, chunk->view().col_idx + start);
    for (int r = 0; r <= rpa; ++r) {
      EXPECT_EQ(odd.row_ptr[b * rpa + r],
                nnz + compact.rowPtr[static_cast<size_t>(angle * rpa + r)] - start);
    }
    nnz += end - start;
  }
  EXPECT_EQ(odd.nnz, nnz);
  EXPECT_EQ(odd.row_ptr.back(), nnz);

  // Adjacent angles {2, 3} of a run merge into one segment.
  const auto run = chunk->subsetView({2, 3}, rpa, 9);
  EXPECT_EQ(run.blockCount(), 2);
  EXPECT_EQ(run.segments.size(), 1u);
  EXPECT_EQ(run.segments[0].nnz,
            compact.rowPtr[static_cast<size_t>(4 * rpa)] -
                compact.rowPtr[static_cast<size_t>(2 * rpa)]);

  // Angles outside the chunk are skipped; none inside -> empty.
  const auto none = chunk->subsetView({6, 7}, rpa, 3);
  EXPECT_TRUE(none.empty());
  EXPECT_TRUE(none.valid());
  EXPECT_THROW(chunk->subsetView({3, 1}, rpa, 1), std::runtime_error);
  std::filesystem::remove_all(dir);
}

TEST(CompactSubsets, RejectsInvalidWeightsInWholeAndDiskSlices) {
  const auto meta = makeSixAngleMeta();
  auto compact = makeSixAngleCompactCSR(meta);
  const auto dir = std::filesystem::temp_directory_path() / "spect_invalid_csrb_weights";
  for (float value : {0.0f, -0.0f, 1.0f, -1.0f,
                      std::numeric_limits<float>::denorm_min(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::quiet_NaN(),
                      -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::infinity()}) {
    compact.values[0] = value;
    writeChunkDirectory(dir, meta, compact, 6);
    for (uint64_t budget : {0ull, 1048576ull}) {
      auto chunks = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, budget);
      if (std::isfinite(value) && value >= 0) {
        EXPECT_NO_THROW(chunks.loadSubset(0, {0}, meta.compactRaysPerAngle, 1));
        EXPECT_NO_THROW(chunks.loadChunk(0));
      } else {
        EXPECT_THROW(chunks.loadSubset(0, {0}, meta.compactRaysPerAngle, 1), std::runtime_error);
        EXPECT_THROW(chunks.loadChunk(0), std::runtime_error);
      }
    }
  }
  std::filesystem::remove_all(dir);
}

TEST(CompactSubsets, CacheInputsUseCurrentPathAndStatWithoutReadingPayload) {
  const auto meta = makeSixAngleMeta();
  const auto compact = makeSixAngleCompactCSR(meta);
  const auto root = std::filesystem::temp_directory_path() / "spect_cache_inputs";
  writeChunkDirectory(root / "a", meta, compact, 6);
  writeChunkDirectory(root / "b", meta, compact, 6);
  const auto a = spect::io::openCompactChunkCSRDirectory((root / "a").string(), meta, 0);
  const auto b = spect::io::openCompactChunkCSRDirectory((root / "b").string(), meta, 0);
  const auto file = a.chunks()[0].path;
  const auto timestamp = std::filesystem::last_write_time(file);
  std::filesystem::last_write_time(b.chunks()[0].path, timestamp);
  const auto inputs = [](const auto &chunks) {
    std::ostringstream out;
    chunks.appendCacheInputs(out, "");
    return out.str();
  };
  const auto initial = inputs(a);
  EXPECT_NE(initial, inputs(b)); // Same basename and stats, different input path.
  const auto alias = spect::io::openCompactChunkCSRDirectory((root / "a" / ".").string(), meta, 0);
  EXPECT_EQ(initial, inputs(alias));
  std::filesystem::last_write_time(file, timestamp + std::chrono::seconds(1));
  EXPECT_NE(initial, inputs(a));
  EXPECT_EQ(a.readStats().payloadLoads, 0u);
  EXPECT_EQ(b.readStats().payloadReadBytes, 0u);
  std::filesystem::resize_file(file, a.chunks()[0].fileSizeBytes + 4);
  EXPECT_THROW(inputs(a), std::runtime_error);
  std::filesystem::remove(file);
  EXPECT_THROW(inputs(a), std::filesystem::filesystem_error);
  std::filesystem::remove_all(root);
}

TEST(CompactSubsets, CacheInputsDoNotCertifyPayloadHistory) {
  const auto meta = makeSixAngleMeta();
  auto compact = makeSixAngleCompactCSR(meta);
  const auto dir = std::filesystem::temp_directory_path() / "spect_current_matrix_input";
  writeChunkDirectory(dir, meta, compact, 6);
  const auto chunks = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, 1 << 20);
  const auto file = chunks.chunks()[0].path;
  const auto timestamp = std::filesystem::last_write_time(file);
  std::ostringstream initial;
  chunks.appendCacheInputs(initial, "");
  EXPECT_EQ(chunks.readStats().payloadLoads, 0u);
  compact.values[0] *= 2.0f;
  writeChunkDirectory(dir, meta, compact, 6);
  std::filesystem::last_write_time(file, timestamp);
  for (uint64_t budget : {0ull, 1048576ull}) {
    const auto next = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, budget);
    std::ostringstream current;
    next.appendCacheInputs(current, "");
    // Replacing data while preserving stats requires caller cache invalidation.
    EXPECT_EQ(current.str(), initial.str());
    EXPECT_FLOAT_EQ(next.loadChunk(0)->values.front(), compact.values.front());
    EXPECT_EQ(next.readStats().payloadLoads, 1u);
  }
  std::filesystem::remove_all(dir);
}

TEST(CompactSubsets, UncachedSliceOwnsStorageAndEmptySliceDoesNotReadPayload) {
  const auto meta = makeSixAngleMeta();
  const auto compact = makeSixAngleCompactCSR(meta);
  const auto dir = std::filesystem::temp_directory_path() / "spect_subset_lifetime";
  writeChunkDirectory(dir, meta, compact, 3);
  const auto chunks = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, 0);
  EXPECT_TRUE(chunks.loadSubset(1, {0}, meta.compactRaysPerAngle, 1).empty());
  EXPECT_EQ(chunks.readStats().payloadLoads, 0u);
  auto slice = chunks.loadSubset(0, {0}, meta.compactRaysPerAngle, 1);
  std::weak_ptr<const void> owner = slice.sourceOwner;
  ASSERT_FALSE(owner.expired());
  EXPECT_FLOAT_EQ(slice.segments.front().val[0], compact.values[0]);
  {
    const auto reused = chunks.loadSubset(0, {0}, meta.compactRaysPerAngle, 1);
    EXPECT_EQ(chunks.readStats().payloadLoads, 1u);
    EXPECT_EQ(reused.segments.front().val, slice.segments.front().val);
    // A partial read must never be returned as a whole chunk.
    const auto whole = chunks.loadChunk(0);
    EXPECT_EQ(chunks.readStats().payloadLoads, 2u);
    EXPECT_GT(whole->info.nnz, slice.nnz);
  }
  slice = {};
  EXPECT_TRUE(owner.expired());
  std::filesystem::remove_all(dir);
}

TEST(CompactSubsets, DiskSlicesMatchWholeChunkAndAccountOnlySelectedPayload) {
  const auto meta = makeSixAngleMeta();
  auto compact = makeSixAngleCompactCSR(meta);
  const int rpa = meta.compactRaysPerAngle;
  // Include an entirely empty angle and a trailing empty row.
  const uint32_t keep = compact.rowPtr[5 * rpa - 1];
  compact.colIdx.resize(keep);
  compact.values.resize(keep);
  std::fill(compact.rowPtr.begin() + 5 * rpa - 1, compact.rowPtr.end(), keep);
  const auto dir = std::filesystem::temp_directory_path() / "spect_subset_disk_ranges";
  writeChunkDirectory(dir, meta, compact, 6);
  const auto reference = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, 0).loadChunk(0);
  for (const auto &angles : std::vector<std::vector<int>>{{0,1,2,3,4,5}, {0,2,4}, {0,1,4}, {5}}) {
    for (uint64_t budget : {0ull, 1048576ull}) {
      auto chunks = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, budget);
      const auto actual = chunks.loadSubset(0, angles, rpa, 17);
      const auto expected = reference->subsetView(angles, rpa, 17);
      ASSERT_TRUE(actual.valid());
      EXPECT_EQ(actual.blockAngles, expected.blockAngles);
      EXPECT_EQ(actual.row_ptr, expected.row_ptr);
      ASSERT_EQ(actual.segments.size(), expected.segments.size());
      for (size_t i = 0; i < actual.segments.size(); ++i) {
        ASSERT_EQ(actual.segments[i].nnz, expected.segments[i].nnz);
        for (uint64_t j = 0; j < actual.segments[i].nnz; ++j) {
          EXPECT_EQ(actual.segments[i].col_idx[j], expected.segments[i].col_idx[j]);
          EXPECT_EQ(actual.segments[i].val[j], expected.segments[i].val[j]);
        }
      }
      const uint64_t loadedNnz = budget == 0 ? actual.nnz : compact.values.size();
      EXPECT_EQ(chunks.readStats().payloadReadBytes,
                8ull * compact.rowPtr.size() + 8ull * loadedNnz);
      EXPECT_THROW(chunks.loadSubset(0, {2,0}, rpa, 1), std::runtime_error);
      EXPECT_THROW(chunks.loadSubset(0, {0,0}, rpa, 1), std::runtime_error);
      EXPECT_THROW(chunks.loadSubset(0, {0}, rpa + 1, 1), std::runtime_error);
    }
  }
  auto truncated = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, 0);
  std::filesystem::resize_file(truncated.chunks()[0].path, 28 + 8 * compact.rowPtr.size());
  EXPECT_THROW(truncated.loadSubset(0, {0}, rpa, 1), std::runtime_error);
  std::filesystem::remove_all(dir);
}

// --- CUDA subset statistics -------------------------------------------------------

class CompactSubsetCudaTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!spect::cuda::isCudaAvailable()) {
      GTEST_SKIP() << "CUDA is not available";
    }
  }
};

TEST_F(CompactSubsetCudaTest, SynchronousCsrUploadRetainsBackingStorageUntilEviction) {
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  std::weak_ptr<const spect::io::CompactChunkCSRData> previous;
  for (int scale : {1, 2}) {
    std::weak_ptr<const spect::io::CompactChunkCSRData> current;
    {
      auto chunk = std::make_shared<spect::io::CompactChunkCSRData>();
      chunk->info.rows = 2;
      chunk->cols = 2;
      chunk->rowPtr = {0, 1, 2};
      chunk->colIdx = {1, 0};
      chunk->values = {2.0f * scale, 3.0f * scale};
      current = chunk;
      const float x[] = {5.0f, 7.0f};
      float y[2] = {};
      ASSERT_TRUE(spect::cuda::spmvFloatCUDA(chunk->view(), x, y));
      EXPECT_FLOAT_EQ(y[0], 14.0f * scale);
      EXPECT_FLOAT_EQ(y[1], 15.0f * scale);
      ASSERT_TRUE(spect::cuda::spmvTFloatCUDA(chunk->view(), x, y));
      EXPECT_FLOAT_EQ(y[0], 21.0f * scale);
      EXPECT_FLOAT_EQ(y[1], 10.0f * scale);
    }
    // A pointer-keyed device cache must prevent reuse of its source addresses.
    EXPECT_FALSE(current.expired());
    EXPECT_TRUE(previous.expired());
    previous = current;
  }
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  EXPECT_TRUE(previous.expired());
}

TEST_F(CompactSubsetCudaTest, FullForwardReadsEachUncachedChunkOncePerPass) {
  const auto dir = std::filesystem::temp_directory_path() / "spect_forward_one_pass";
  for (const bool useZ : {false, true}) {
    auto meta = makeSixAngleMeta();
    if (!useZ) {
      meta.profile = spect::io::CompactSymmetryProfile::Rot90;
      meta.detNz = meta.compactDetZ;
    }
    const auto compact = makeSixAngleCompactCSR(meta);
    const auto full = spect::io::test::expandCompactToyToFullDense(meta, compact);
    writeChunkDirectory(dir, meta, compact, 2);
    const auto chunks = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, 0);
    std::vector<double> image(meta.totalVoxels), forward(meta.fullRows());
    std::vector<double> reference(meta.fullRows(), 0.0);
    for (int c = 0; c < meta.totalVoxels; ++c) image[c] = 0.4 + 0.13 * c;
    for (size_t r = 0; r < reference.size(); ++r)
      for (int c = 0; c < meta.totalVoxels; ++c)
        reference[r] += full[r * meta.totalVoxels + c] * image[c];
    ASSERT_TRUE(spect::cuda::prepareCompactChunkExecutionCUDA(chunks));
    for (int pass = 0; pass < 2; ++pass) {
      const auto before = chunks.readStats();
      ASSERT_TRUE(spect::cuda::computeCompactChunkForwardProjectionCUDA(
          chunks, meta, image.data(), 1, forward.data()));
      EXPECT_LE(relL2(forward, reference), 1e-5);
      EXPECT_LE(chunks.readStats().payloadLoads - before.payloadLoads,
                chunks.chunks().size());
    }
    ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  }
  std::filesystem::remove_all(dir);
}

TEST_F(CompactSubsetCudaTest, FirstSubsetReleasesWholeChunkDeviceCapacity) {
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  constexpr uint32_t nnz = 8u * 1024u * 1024u;
  const auto makeChunk = [nnz] {
    auto chunk = std::make_shared<spect::io::CompactChunkCSRData>();
    chunk->info.rows = 4;
    chunk->info.angleCount = 2;
    chunk->info.nnz = nnz;
    chunk->cols = nnz / 4;
    chunk->rowPtr = {0, nnz / 4, nnz / 2, 3 * (nnz / 4), nnz};
    chunk->colIdx.resize(nnz);
    for (uint32_t i = 0; i < nnz; ++i) chunk->colIdx[i] = i % chunk->cols;
    chunk->values.assign(nnz, 1.0f);
    return chunk;
  };
  const auto first = makeChunk(), second = makeChunk();
  ASSERT_TRUE(spect::cuda::warmupResidentMatrixSetCUDA({first->view()}));
  const std::vector<float> image(first->cols, 1.0f);
  std::vector<float> forward(4);
  ASSERT_TRUE(spect::cuda::spmvFloatCUDA(
      second->view(), image.data(), forward.data()));
  const auto whole = second->subsetView({0, 1}, 2, 0);
  const auto a = first->subsetView({0}, 2, 1);
  const auto b = first->subsetView({1}, 2, 2);
  const auto before = spect::cuda::getMemoryInfo().free;
  ASSERT_TRUE(spect::cuda::prefetchCompactFloatMatrixCUDA(a, whole));
  ASSERT_TRUE(spect::cuda::prefetchCompactFloatMatrixCUDA(b, a));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  // Each 64 MiB slot becomes 32 MiB; leave 16 MiB for driver/display noise.
  EXPECT_GE(spect::cuda::getMemoryInfo().free, before + 48ull * 1024 * 1024);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
}

// Per subset, S_m and q_m equal the dense oracle restricted to the subset's
// rows; over all subsets they add up to the full-data S and q. Two chunks of
// three angles exercise slices with several segments, slices spanning both
// chunks, and (N = 6) chunks without any row of a subset.
TEST_F(CompactSubsetCudaTest, Rot90N10MatchesIndependentOracleWithPartialCaches) {
  auto meta = makeSixAngleMeta();
  meta.profile = spect::io::CompactSymmetryProfile::Rot90;
  meta.Nz = 3;
  meta.totalVoxels = 12;
  meta.compactNumAngles = 90;
  meta.fullNumAngles = 360;
  meta.compactDetZ = meta.detNz;
  meta.compactRaysPerAngle = meta.detNx * meta.detNz;
  meta.totalRows = meta.compactNumAngles * meta.compactRaysPerAngle;
  auto compact = makeSixAngleCompactCSR(meta);
  // Subset 0 selects an all-zero slice from the first chunk. Exercise the
  // segmented reader's empty payload through CUDA, not just the host view.
  const uint32_t firstAngleNnz = compact.rowPtr[meta.compactRaysPerAngle];
  compact.colIdx.erase(compact.colIdx.begin(), compact.colIdx.begin() + firstAngleNnz);
  compact.values.erase(compact.values.begin(), compact.values.begin() + firstAngleNnz);
  for (size_t row = 0; row < compact.rowPtr.size(); ++row)
    compact.rowPtr[row] = row <= static_cast<size_t>(meta.compactRaysPerAngle)
        ? 0 : compact.rowPtr[row] - firstAngleNnz;
  const int rows = static_cast<int>(meta.fullRows()), cols = meta.totalVoxels;
  // Explicit 2x2 rotations, independent of production coordinate/row helpers.
  const int xy[4][4] = {{0,1,2,3}, {1,3,0,2}, {3,2,1,0}, {2,0,3,1}};
  std::vector<double> full(static_cast<size_t>(rows) * cols, 0.0);
  for (int sym = 0; sym < 4; ++sym)
    for (int r = 0; r < meta.totalRows; ++r)
      for (uint32_t j = compact.rowPtr[r]; j < compact.rowPtr[r+1]; ++j) {
        const int c = static_cast<int>(compact.colIdx[j]);
        full[static_cast<size_t>(sym * meta.totalRows + r) * cols +
             (c / 4) * 4 + xy[sym][c % 4]] += compact.values[j];
      }
  std::vector<double> image(cols), projection(rows), weights(rows);
  for (int c = 0; c < cols; ++c) image[c] = 0.25 + c * 0.19;
  for (int r = 0; r < rows; ++r) {
    weights[r] = r % 7 == 0 ? 0.0 : 0.6 + 0.1 * (r % 3);
    projection[r] = r % 7 == 0 ? 0.0 : 1.0 + 0.02 * (r % 23);
  }
  const auto dir = std::filesystem::temp_directory_path() / "spect_rot90_n10";
  writeChunkDirectory(dir, meta, compact, 5);
  for (const uint64_t budget : {0ull, 1000ull, 1048576ull}) {
    SCOPED_TRACE(budget);
    const auto chunks = spect::io::openCompactChunkCSRDirectory(dir.string(), meta, budget);
    const std::vector<spect::cuda::CompactMatrixChannel> channels{{&chunks, 1.0, 1.0}};
    std::vector<double> forward(rows), reference(rows, 0.0);
    ASSERT_TRUE(spect::cuda::computeCompactChunkForwardProjectionCUDA(
        chunks, meta, image.data(), 1, forward.data()));
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        reference[r] += full[static_cast<size_t>(r) * cols + c] * image[c];
    EXPECT_LE(relL2(forward, reference), 1e-5);
    const double *images[] = {image.data()}, *projections[] = {projection.data()};
    for (const int count : {1, 10}) {
      const auto order = spect::compactSubsetOrder(count);
      for (size_t s = 0; s < order.size(); ++s) {
        const spect::CompactSubsetSpec spec{count, order[s]};
        const spect::CompactSubsetSpec next{count, order[(s+1) % order.size()]};
        auto restricted = restrictToSubset(full, meta, count, spec.index);
        std::vector<double> numer(cols), sensitivity(cols);
        double *numers[] = {numer.data()};
        ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
            channels, meta, projections, nullptr, images, 1, 1, numers,
            s % 2 != 0, nullptr, nullptr, true, false, true, &spec, &next));
        EXPECT_LE(relL2(numer, spect::io::test::denseNumerator(
            restricted, projection, image, rows, cols)), 1e-5);
        ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
            channels, meta, 1, sensitivity.data(), nullptr, nullptr, weights.data(), &spec));
        for (int r = 0; r < rows; ++r)
          for (int c = 0; c < cols; ++c)
            restricted[static_cast<size_t>(r) * cols + c] *= weights[r];
        EXPECT_LE(relL2(sensitivity, spect::io::test::denseDenom(restricted, rows, cols)), 1e-5);
      }
    }
    EXPECT_GT(chunks.readStats().emptySlicesSkipped, 0u);
    if (budget == 0)
      EXPECT_GT(chunks.readStats().payloadLoads, chunks.chunks().size());
  }
  std::filesystem::remove_all(dir);
}

TEST_F(CompactSubsetCudaTest, NativeCsrbWindowsMatchWeightedPoissonOracle) {
  const auto root = std::filesystem::temp_directory_path() / "spect_csrb_windows";
  for (const bool useZ : {false, true}) {
    auto meta = makeSixAngleMeta();
    if (!useZ) {
      meta.profile = spect::io::CompactSymmetryProfile::Rot90;
      meta.detNz = meta.compactDetZ;
    }
    auto a = makeSixAngleCompactCSR(meta), b = a;
    for (size_t j = 0; j < b.values.size(); ++j) {
      b.values[j] *= 0.3f + 0.1f * static_cast<float>(j % 7);
      b.colIdx[j] = (b.colIdx[j] + 3) % meta.totalVoxels;
    }
    writeChunkDirectory(root / "a", meta, a, 3);
    writeChunkDirectory(root / "b", meta, b, 2);
    const int rows = static_cast<int>(meta.fullRows()), cols = meta.totalVoxels;
    const std::vector<double> full[] = {
        spect::io::test::expandCompactToyToFullDense(meta, a),
        spect::io::test::expandCompactToyToFullDense(meta, b)};
    const double weights[] = {0.37, 1.6}, powers[] = {1.0, 1.27};
    std::vector<double> image(2 * cols), p[4], bg[4], rowWeights(rows);
    std::vector<uint16_t> att[2];
    for (int c = 0; c < 2 * cols; ++c) image[c] = 0.4 + 0.13 * c;
    for (int r = 0; r < rows; ++r) rowWeights[r] = r % 4 ? 0.8 + 0.05 * (r % 3) : 0;
    for (int slot = 0; slot < 4; ++slot) {
      p[slot].resize(rows); bg[slot].resize(rows);
      for (int r = 0; r < rows; ++r) {
        p[slot][r] = r % 3 ? 1.0 + slot + 0.02 * r : 0.0;
        bg[slot][r] = 0.2 + 0.03 * (r % 5) + slot * 0.1;
      }
    }
    for (int bed = 0; bed < 2; ++bed) {
      att[bed].resize(meta.fullNumAngles * cols);
      for (size_t i = 0; i < att[bed].size(); ++i)
        att[bed][i] = static_cast<uint16_t>(22000 + (i * 193 + bed * 717) % 30000);
    }
    const double *images[] = {image.data(), image.data() + cols};
    const double *projections[] = {p[0].data(), p[1].data(), p[2].data(), p[3].data()};
    const double *backgrounds[] = {nullptr, bg[1].data(), bg[2].data(), nullptr};
    const uint16_t *attenuation[] = {att[0].data(), att[1].data()};
    const int bedStarts[] = {0, meta.Nz};
    for (const uint64_t budget : {0ull, 1048576ull}) {
      const auto ca = spect::io::openCompactChunkCSRDirectory((root / "a").string(), meta, budget);
      const auto cb = spect::io::openCompactChunkCSRDirectory((root / "b").string(), meta, budget);
      for (const int windows : {1, 2}) {
        std::vector<spect::cuda::CompactMatrixChannel> channels{{&ca, weights[0], powers[0]}};
        if (windows == 2) {
          channels.push_back({&cb, weights[1], powers[1]});
          channels[1].sensitivityRowWeights = rowWeights.data();
        }
        for (const bool ctac : {false, true}) {
          for (const int count : {1, 3}) {
            for (int subset = 0; subset < count; ++subset) {
              SCOPED_TRACE(::testing::Message() << useZ << '/' << budget << '/' << windows
                           << '/' << ctac << '/' << count << '/' << subset);
              const spect::CompactSubsetSpec spec{count, subset};
              std::vector<double> q[2], s[2], expectedQ[2], expectedS[2];
              for (int bed = 0; bed < 2; ++bed) {
                q[bed].resize(cols); s[bed].resize(cols);
                expectedQ[bed].assign(cols, 0); expectedS[bed].assign(cols, 0);
                for (int k = 0; k < windows; ++k) {
                  auto matrix = restrictToSubset(full[k], meta, count, subset);
                  for (int r = 0; r < rows; ++r) {
                    double mean = backgrounds[2*k + bed] ? backgrounds[2*k + bed][r] : 0;
                    for (int c = 0; c < cols; ++c) {
                      auto &value = matrix[static_cast<size_t>(r) * cols + c];
                      value *= weights[k] * (ctac ? std::pow(
                          att[bed][(r / meta.fullRaysPerAngle()) * cols + c] / 65535.0, powers[k]) : 1.0);
                      mean += value * images[bed][c];
                    }
                    for (int c = 0; c < cols; ++c) {
                      const double value = matrix[static_cast<size_t>(r) * cols + c];
                      expectedQ[bed][c] += value * p[2*k + bed][r] / std::max(mean, 1e-10);
                      expectedS[bed][c] += value * (k == 1 ? rowWeights[r] : 1.0);
                    }
                  }
                }
                ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
                    channels, meta, 1, s[bed].data(), nullptr,
                    ctac ? att[bed].data() : nullptr, nullptr, &spec));
                EXPECT_LE(relL2(s[bed], expectedS[bed]), 1e-5);
              }
              double *numers[] = {q[0].data(), q[1].data()};
              for (const bool prepared : {false, true}) {
                if (prepared)
                  ASSERT_TRUE(spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
                      meta, image.data(), image.size(), 2 * meta.Nz, bedStarts, 2, false, nullptr, true));
                const auto before = spect::cuda::getCudaChunkStageTimingSnapshotCUDA();
                ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
                    channels, meta, projections, backgrounds, prepared ? nullptr : images,
                    2, 1, numers, prepared, nullptr, ctac ? attenuation : nullptr,
                    !prepared, prepared, true, &spec, nullptr));
                if (prepared) {
                  const auto after = spect::cuda::getCudaChunkStageTimingSnapshotCUDA();
                  EXPECT_EQ(after.compactProjectionH2DCalls, before.compactProjectionH2DCalls);
                  EXPECT_EQ(after.compactProjectionH2DBytes, before.compactProjectionH2DBytes);
                } else {
                  EXPECT_GT(spect::cuda::getCudaChunkStageTimingSnapshotCUDA().compactProjectionH2DCalls,
                            before.compactProjectionH2DCalls);
                }
                for (int bed = 0; bed < 2; ++bed)
                  EXPECT_LE(relL2(q[bed], expectedQ[bed]), 1e-5);
              }
              // The next explicit refresh must see in-place changes and clear
              // absent backgrounds, even though the host addresses are reused.
              for (double &value : p[2]) value += 0.125;
              backgrounds[1] = backgrounds[1] == nullptr ? bg[1].data() : nullptr;
            }
          }
        }
      }
      ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
    }
  }
  std::filesystem::remove_all(root);
}

TEST_F(CompactSubsetCudaTest, SubsetSensitivityAndNumeratorMatchRestrictedOracle) {
  const auto meta = makeSixAngleMeta();
  const auto compact = makeSixAngleCompactCSR(meta);
  const auto full = spect::io::test::expandCompactToyToFullDense(meta, compact);
  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_compact_subset_cuda_test";
  writeChunkDirectory(dir, meta, compact, /*anglesPerChunk=*/3);
  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunks =
      spect::io::openCompactChunkCSRDirectory(dir.string(), dirMeta, 0);
  ASSERT_EQ(chunks.chunks().size(), 2u);

  const int fullRows = meta.fullRows();
  const int cols = meta.totalVoxels;
  const size_t nvox = static_cast<size_t>(cols);
  constexpr int bedCount = 2;
  std::vector<double> imageBed0 = {1.0, 1.3, 1.7, 2.1, 0.8, 1.2, 1.6, 2.0};
  std::vector<double> imageBed1 = {0.5, 0.8, 1.2, 1.6, 1.0, 1.4, 1.8, 2.2};
  std::vector<double> projBed0(static_cast<size_t>(fullRows));
  std::vector<double> projBed1(static_cast<size_t>(fullRows));
  for (size_t i = 0; i < projBed0.size(); ++i) {
    // ~60 % zero bins so the zero-row scatter backward is exercised too.
    projBed0[i] = (i % 5 < 2) ? 1.0 + 0.01 * static_cast<double>(i) : 0.0;
    projBed1[i] = ((i + 2) % 5 < 2) ? 2.0 + 0.02 * static_cast<double>(i) : 0.0;
  }
  const double *imagePtrs[bedCount] = {imageBed0.data(), imageBed1.data()};
  const double *projPtrs[bedCount] = {projBed0.data(), projBed1.data()};
  const auto channels = std::vector<spect::cuda::CompactMatrixChannel>{
      spect::cuda::CompactMatrixChannel{&chunks, 1.0, 1.0}};

  std::vector<double> sFull(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
      channels, meta, 1, sFull.data(), nullptr, nullptr, nullptr, nullptr));
  EXPECT_LE(relL2(sFull, spect::io::test::denseDenom(full, fullRows, cols)), 1e-5);
  std::vector<std::vector<double>> qFull(bedCount, std::vector<double>(nvox, 0.0));
  {
    double *numerPtrs[bedCount] = {qFull[0].data(), qFull[1].data()};
    ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
        channels, meta, projPtrs, nullptr, imagePtrs, bedCount, 1, numerPtrs,
        false, nullptr, nullptr, true, false, true, nullptr, nullptr));
  }
  EXPECT_LE(relL2(qFull[0], spect::io::test::denseNumerator(
                                full, projBed0, imageBed0, fullRows, cols)),
            1e-5);

  for (const int count : {2, 3, 6}) {
    std::vector<double> sSum(nvox, 0.0);
    std::vector<std::vector<double>> qSum(bedCount, std::vector<double>(nvox, 0.0));
    const auto order = spect::compactSubsetOrder(count);
    for (size_t s = 0; s < order.size(); ++s) {
      const spect::CompactSubsetSpec spec{count, order[s]};
      const auto restricted = restrictToSubset(full, meta, count, spec.index);

      std::vector<double> sSub(nvox, 0.0);
      ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
          channels, meta, 1, sSub.data(), nullptr, nullptr, nullptr, &spec));
      const auto sRef = spect::io::test::denseDenom(restricted, fullRows, cols);
      EXPECT_LE(relL2(sSub, sRef), 1e-5) << "N=" << count << " m=" << spec.index;
      for (size_t j = 0; j < nvox; ++j)
        sSum[j] += sSub[j];

      std::vector<std::vector<double>> qSub(bedCount,
                                            std::vector<double>(nvox, 0.0));
      double *numerPtrs[bedCount] = {qSub[0].data(), qSub[1].data()};
      const std::optional<spect::CompactSubsetSpec> next =
          s + 1 < order.size()
              ? std::optional<spect::CompactSubsetSpec>(
                    spect::CompactSubsetSpec{count, order[s + 1]})
              : std::nullopt;
      ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
          channels, meta, projPtrs, nullptr, imagePtrs, bedCount, 1, numerPtrs,
          (s % 2) == 1, nullptr, nullptr, true, false, true, &spec,
          next ? &*next : nullptr));
      const auto q0Ref = spect::io::test::denseNumerator(
          restricted, projBed0, imageBed0, fullRows, cols);
      const auto q1Ref = spect::io::test::denseNumerator(
          restricted, projBed1, imageBed1, fullRows, cols);
      EXPECT_LE(relL2(qSub[0], q0Ref), 1e-5) << "N=" << count << " m=" << spec.index;
      EXPECT_LE(relL2(qSub[1], q1Ref), 1e-5) << "N=" << count << " m=" << spec.index;
      for (size_t j = 0; j < nvox; ++j) {
        qSum[0][j] += qSub[0][j];
        qSum[1][j] += qSub[1][j];
      }
    }
    EXPECT_LE(relL2(sSum, sFull), 1e-6) << "sum of S_m vs S, N=" << count;
    EXPECT_LE(relL2(qSum[0], qFull[0]), 1e-6) << "sum of q_m vs q, N=" << count;
    EXPECT_LE(relL2(qSum[1], qFull[1]), 1e-6) << "sum of q_m vs q, N=" << count;
  }

  // CTAC: the per-angle factor tables are addressed by the slice's absolute
  // compact angles. Angle-varying tables make a wrong block -> angle mapping
  // visible in the subset sums against the whole-chunk CTAC pass.
  const size_t attCount = static_cast<size_t>(meta.fullNumAngles) * nvox;
  std::vector<uint16_t> attBed0(attCount);
  std::vector<uint16_t> attBed1(attCount);
  for (int fullAngle = 0; fullAngle < meta.fullNumAngles; ++fullAngle) {
    for (size_t v = 0; v < nvox; ++v) {
      const size_t i = static_cast<size_t>(fullAngle) * nvox + v;
      attBed0[i] = static_cast<uint16_t>(
          65535.0 * (0.3 + 0.07 * static_cast<double>((fullAngle * 7 + v) % 11)));
      attBed1[i] = static_cast<uint16_t>(
          65535.0 * (0.4 + 0.05 * static_cast<double>((fullAngle * 5 + 3 * v) % 13)));
    }
  }
  const uint16_t *attPtrs[bedCount] = {attBed0.data(), attBed1.data()};
  std::vector<double> sCtacFull(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
      channels, meta, 1, sCtacFull.data(), nullptr, attBed0.data(), nullptr,
      nullptr));
  std::vector<std::vector<double>> qCtacFull(bedCount,
                                             std::vector<double>(nvox, 0.0));
  {
    double *numerPtrs[bedCount] = {qCtacFull[0].data(), qCtacFull[1].data()};
    ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
        channels, meta, projPtrs, nullptr, imagePtrs, bedCount, 1, numerPtrs,
        false, nullptr, attPtrs, true, false, true, nullptr, nullptr));
  }
  EXPECT_GT(relL2(qCtacFull[0], qFull[0]), 1e-3) << "CTAC must change q";
  for (const int count : {2, 6}) {
    std::vector<double> sSum(nvox, 0.0);
    std::vector<std::vector<double>> qSum(bedCount, std::vector<double>(nvox, 0.0));
    for (int m = 0; m < count; ++m) {
      const spect::CompactSubsetSpec spec{count, m};
      std::vector<double> sSub(nvox, 0.0);
      ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
          channels, meta, 1, sSub.data(), nullptr, attBed0.data(), nullptr,
          &spec));
      std::vector<std::vector<double>> qSub(bedCount,
                                            std::vector<double>(nvox, 0.0));
      double *numerPtrs[bedCount] = {qSub[0].data(), qSub[1].data()};
      ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
          channels, meta, projPtrs, nullptr, imagePtrs, bedCount, 1, numerPtrs,
          false, nullptr, attPtrs, true, false, true, &spec, nullptr));
      for (size_t j = 0; j < nvox; ++j) {
        sSum[j] += sSub[j];
        qSum[0][j] += qSub[0][j];
        qSum[1][j] += qSub[1][j];
      }
    }
    EXPECT_LE(relL2(sSum, sCtacFull), 1e-6) << "CTAC sum of S_m, N=" << count;
    EXPECT_LE(relL2(qSum[0], qCtacFull[0]), 1e-6) << "CTAC sum of q_m, N=" << count;
    EXPECT_LE(relL2(qSum[1], qCtacFull[1]), 1e-6) << "CTAC sum of q_m, N=" << count;
  }
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  std::filesystem::remove_all(dir);
}

// Resident OSEM / BSREM sub-iteration tails: the device update with a
// resident subset sensitivity equals the host update with the same S_m and
// M = subset count; subset index -1 is the unchanged full-data step.
TEST_F(CompactSubsetCudaTest, ResidentSubsetUpdatesMatchHost) {
  for (const bool useZ : {true, false}) {
  SCOPED_TRACE(useZ);
  auto meta = makeSixAngleMeta();
  if (!useZ) {
    meta.profile = spect::io::CompactSymmetryProfile::Rot90;
    meta.detNz = meta.compactDetZ;
  }
  const auto compact = makeSixAngleCompactCSR(meta);
  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_compact_subset_update_test";
  writeChunkDirectory(dir, meta, compact, /*anglesPerChunk=*/3);
  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunks =
      spect::io::openCompactChunkCSRDirectory(dir.string(), dirMeta, 1 << 20);
  const auto channels = std::vector<spect::cuda::CompactMatrixChannel>{
      spect::cuda::CompactMatrixChannel{&chunks, 1.0, 1.0}};

  const int fullRows = meta.fullRows();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  constexpr int bedCount = 2;
  constexpr int count = 3;
  const size_t globalNvox = bedCount * nvox;
  std::vector<double> globalImage(globalNvox);
  for (size_t i = 0; i < globalNvox; ++i)
    globalImage[i] = 0.7 + 0.05 * static_cast<double>(i);
  std::vector<double> projBed0(static_cast<size_t>(fullRows));
  std::vector<double> projBed1(static_cast<size_t>(fullRows));
  for (size_t i = 0; i < projBed0.size(); ++i) {
    projBed0[i] = (i % 3 == 0) ? 0.0 : 1.0 + 0.01 * static_cast<double>(i);
    projBed1[i] = (i % 4 == 0) ? 0.0 : 2.0 + 0.02 * static_cast<double>(i);
  }
  const double *projPtrs[bedCount] = {projBed0.data(), projBed1.data()};
  const int bedZStarts[bedCount] = {0, meta.Nz};

  // Global and subset sensitivities as global (two-bed) volumes.
  auto globalSensitivity = [&](const spect::CompactSubsetSpec *spec) {
    std::vector<double> local(nvox, 0.0);
    EXPECT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
        channels, meta, 1, local.data(), nullptr, nullptr, nullptr, spec));
    std::vector<double> global(globalNvox);
    for (int bed = 0; bed < bedCount; ++bed)
      std::copy(local.begin(), local.end(),
                global.begin() + static_cast<ptrdiff_t>(bed * nvox));
    return global;
  };
  const std::vector<double> sGlobal = globalSensitivity(nullptr);
  std::vector<std::vector<double>> sSubsets;
  std::vector<const double *> sSubsetPtrs;
  for (int m = 0; m < count; ++m) {
    const spect::CompactSubsetSpec spec{count, m};
    sSubsets.push_back(globalSensitivity(&spec));
  }
  for (const auto &s : sSubsets)
    sSubsetPtrs.push_back(s.data());
  ASSERT_TRUE(spect::cuda::setCompactSubsetSensitivitiesCUDA(
      sSubsetPtrs.data(), count, globalNvox));

  // Host reference sub-iteration over subset m from `image` (host path).
  auto hostSubsetNumerator = [&](const std::vector<double> &image, int m) {
    const spect::CompactSubsetSpec spec{count, m};
    std::vector<std::vector<double>> numers(bedCount, std::vector<double>(nvox, 0.0));
    // Literal bound: MSVC does not treat a captured constexpr as constant.
    double *numerPtrs[2] = {numers[0].data(), numers[1].data()};
    const double *imagePtrs[2] = {image.data(), image.data() + nvox};
    EXPECT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
        channels, meta, projPtrs, nullptr, imagePtrs, bedCount, 1, numerPtrs,
        false, nullptr, nullptr, true, false, true, &spec, nullptr));
    std::vector<double> global(globalNvox);
    for (int bed = 0; bed < bedCount; ++bed)
      std::copy(numers[bed].begin(), numers[bed].end(),
                global.begin() + static_cast<ptrdiff_t>(bed * nvox));
    return global;
  };
  // Device sub-iteration over subset m on the resident image.
  auto deviceSubsetNumerator = [&](bool uploadImage, int m) {
    const spect::CompactSubsetSpec spec{count, m};
    ASSERT_TRUE(spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
        meta, globalImage.data(), globalImage.size(), bedCount * meta.Nz,
        bedZStarts, bedCount, false, nullptr, uploadImage));
    ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
        channels, meta, projPtrs, nullptr, nullptr, bedCount, 1, nullptr, false,
        nullptr, nullptr, true, true, false, &spec, nullptr));
  };

  // OSEM: three sub-iterations in visiting order, device vs host.
  std::vector<double> hostImage = globalImage;
  const auto order = spect::compactSubsetOrder(count);
  for (size_t s = 0; s < order.size(); ++s) {
    const int m = order[s];
    const auto q = hostSubsetNumerator(hostImage, m);
    spect::applyMlemUpdate(hostImage, q, sSubsets[static_cast<size_t>(m)],
                           nullptr, true, 1, &sGlobal);
    deviceSubsetNumerator(s == 0, m);
    bool allZero = true;
    ASSERT_TRUE(spect::cuda::finishCompactResidentMlemIterationCUDA(
        meta, globalNvox, bedCount * meta.Nz, bedZStarts, bedCount, false,
        sGlobal.data(), nullptr, nullptr, true, s == 0, &allZero, m));
    EXPECT_FALSE(allZero);
  }
  std::vector<double> deviceImage(globalNvox, 0.0);
  ASSERT_TRUE(spect::cuda::downloadCompactResidentImageCUDA(deviceImage.data(),
                                                             globalNvox));
  EXPECT_LE(relL2(deviceImage, hostImage), 1e-6);
  // The three subset updates moved the image (not a no-op).
  EXPECT_GT(relL2(hostImage, globalImage), 1e-3);

  // BSREM with M = 3 and the global S as scaling: device vs host.
  spect::cuda::CompactBsremUpdateParams deviceParams;
  deviceParams.alpha = 0.8;
  deviceParams.subsetScale = static_cast<double>(count);
  deviceParams.clipRelativeStep = true;
  deviceParams.zeroLowSensitivity = true;
  spect::BsremUpdateParams hostParams;
  hostParams.alpha = 0.8;
  hostParams.subsetScale = static_cast<double>(count);
  hostParams.threads = 1;
  for (const auto type : {spect::RegType::None, spect::RegType::RDP,
                          spect::RegType::TVIsotropic}) {
    SCOPED_TRACE(spect::regTypeToString(type));
    spect::RegConf reg;
    reg.type = type;
    reg.beta = type == spect::RegType::None ? 0.0 : 0.3;
    hostParams.betaScaled = deviceParams.betaScaled = reg.beta;
    std::vector<double> gradient;
    hostImage = globalImage;
    for (size_t s = 0; s < order.size(); ++s) {
      const int m = order[s];
      const auto q = hostSubsetNumerator(hostImage, m);
      spect::regularizer::compute_reg_gradient(
          hostImage, meta.Nx, meta.Ny, bedCount * meta.Nz, reg, nullptr,
          gradient, 1);
      hostParams.gradient = &gradient;
      spect::applyBsremUpdate(hostImage, q, sGlobal, sSubsets[static_cast<size_t>(m)],
                              hostParams);
      deviceSubsetNumerator(s == 0, m);
      deviceParams.deviceGradient = nullptr;
      if (type != spect::RegType::None) {
        auto *dGrad = spect::cuda::compactResidentGradientDeviceCUDA(globalNvox);
        ASSERT_TRUE(spect::cuda::computeRegGradientDeviceCUDA(
            spect::cuda::compactResidentImageDeviceCUDA(globalNvox), meta.Nx,
            meta.Ny, bedCount * meta.Nz, reg, nullptr, dGrad));
        deviceParams.deviceGradient = dGrad;
      }
      bool allZero = true;
      ASSERT_TRUE(spect::cuda::finishCompactResidentBsremIterationCUDA(
          meta, globalNvox, bedCount * meta.Nz, bedZStarts, bedCount, false,
          sGlobal.data(), nullptr, nullptr, deviceParams, true, s == 0, &allZero,
          m));
      EXPECT_FALSE(allZero);
    }
    ASSERT_TRUE(spect::cuda::downloadCompactResidentImageCUDA(deviceImage.data(),
                                                               globalNvox));
    EXPECT_LE(relL2(deviceImage, hostImage), 1e-6);
  }

  // A missing subset index fails loudly instead of using a wrong array.
  deviceSubsetNumerator(true, 0);
  bool allZero = true;
  EXPECT_THROW(spect::cuda::finishCompactResidentMlemIterationCUDA(
                   meta, globalNvox, bedCount * meta.Nz, bedZStarts, bedCount,
                   false, sGlobal.data(), nullptr, nullptr, true, true, &allZero,
                   count),
               std::runtime_error);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  std::filesystem::remove_all(dir);
  }
}

} // namespace
} // namespace spect::main
