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
#include "tests/common/compact_symmetry_test_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
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

// --- CUDA subset statistics -------------------------------------------------------

class CompactSubsetCudaTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!spect::cuda::isCudaAvailable()) {
      GTEST_SKIP() << "CUDA is not available";
    }
  }
};

// Per subset, S_m and q_m equal the dense oracle restricted to the subset's
// rows; over all subsets they add up to the full-data S and q. Two chunks of
// three angles exercise slices with several segments, slices spanning both
// chunks, and (N = 6) chunks without any row of a subset.
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
      spect::io::openCompactChunkCSRDirectory(dir.string(), dirMeta, 1 << 20);
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
  const auto meta = makeSixAngleMeta();
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

} // namespace
} // namespace spect::main
