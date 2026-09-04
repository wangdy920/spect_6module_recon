#include <gtest/gtest.h>

#include "cuda/CudaConfig.h"
#include "io/CompactChunkCSR.h"
#include "sparse/CompactSymmetricProjector.h"
#include "sparse/ProjDomainChunked.h"
#include "sparse/SparseOpsCUDA.h"
#include "tests/common/compact_symmetry_test_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spect::main {
namespace {

using spect::io::test::makeToyCompactCSR;
using spect::io::test::makeToySym90ZMeta;

struct HostCsr {
  int rows = 0;
  int cols = 0;
  std::vector<uint32_t> rowPtr;
  std::vector<uint32_t> colIdx;
  std::vector<float> values;
};

template <typename T>
void writeRaw(std::ofstream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T>
void writeRaw(std::ofstream &out, const std::vector<T> &values) {
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void writeCsrb(const std::filesystem::path &path, const HostCsr &csr,
               int32_t angleOffset, int32_t angleCount) {
  std::vector<int64_t> rowPtr64(csr.rowPtr.begin(), csr.rowPtr.end());
  std::ofstream out(path, std::ios::binary);
  ASSERT_TRUE(out);
  writeRaw(out, uint32_t{0x42525343u});
  writeRaw(out, uint32_t{1u});
  writeRaw(out, static_cast<int32_t>(csr.rows));
  writeRaw(out, angleOffset);
  writeRaw(out, angleCount);
  writeRaw(out, static_cast<uint64_t>(csr.values.size()));
  writeRaw(out, rowPtr64);
  writeRaw(out, csr.colIdx);
  writeRaw(out, csr.values);
}

// Mirrors tools/convert_csrb_to_csrq.py: per-row sorted columns, runs of at
// most 64 consecutive columns, u16 sqrt-companded values with one f32 row
// max. Returns the CSR the CSRQ file actually encodes (sorted columns,
// dequantized fp32 values) so a CSRB reference of the same matrix can be
// written next to it.
HostCsr writeCsrq(const std::filesystem::path &path, const HostCsr &src,
                  int32_t angleOffset, int32_t angleCount) {
  constexpr int kRunLenBits = 6;
  constexpr uint32_t kMaxRun = 1u << kRunLenBits;
  HostCsr decoded;
  decoded.rows = src.rows;
  decoded.cols = src.cols;
  decoded.rowPtr = src.rowPtr;
  std::vector<uint32_t> rowRunPtr(static_cast<size_t>(src.rows) + 1u, 0u);
  std::vector<float> rowMax(static_cast<size_t>(src.rows), 0.0f);
  std::vector<uint16_t> valuesQ;
  std::vector<uint32_t> runs;
  for (int row = 0; row < src.rows; ++row) {
    const size_t begin = src.rowPtr[static_cast<size_t>(row)];
    const size_t end = src.rowPtr[static_cast<size_t>(row) + 1u];
    std::vector<std::pair<uint32_t, float>> entries;
    for (size_t t = begin; t < end; ++t) {
      entries.emplace_back(src.colIdx[t], src.values[t]);
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    float m = 0.0f;
    for (const auto &entry : entries) {
      m = std::max(m, entry.second);
    }
    rowMax[static_cast<size_t>(row)] = m;
    for (size_t i = 0; i < entries.size(); ++i) {
      const double ratio =
          m > 0.0f ? static_cast<double>(entries[i].second) / m : 0.0;
      const double q = std::clamp(std::rint(std::sqrt(ratio) * 65535.0), 0.0,
                                  65535.0);
      const uint16_t qu = static_cast<uint16_t>(q);
      valuesQ.push_back(qu);
      const float t = static_cast<float>(qu) * (1.0f / 65535.0f);
      decoded.colIdx.push_back(entries[i].first);
      decoded.values.push_back(m * (t * t));
      const bool startsRun = i == 0 || entries[i].first != entries[i - 1].first + 1u;
      if (startsRun) {
        runs.push_back((entries[i].first << kRunLenBits) | 0u);
      } else {
        const uint32_t len = (runs.back() & (kMaxRun - 1u)) + 1u;
        if (len == kMaxRun) {
          runs.push_back((entries[i].first << kRunLenBits) | 0u);
        } else {
          runs.back() += 1u;
        }
      }
    }
    rowRunPtr[static_cast<size_t>(row) + 1u] = static_cast<uint32_t>(runs.size());
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot write " + path.string());
  }
  writeRaw(out, uint32_t{0x51525343u});
  writeRaw(out, uint32_t{1u});
  writeRaw(out, static_cast<int32_t>(src.rows));
  writeRaw(out, angleOffset);
  writeRaw(out, angleCount);
  writeRaw(out, static_cast<int32_t>(src.cols));
  writeRaw(out, static_cast<uint64_t>(valuesQ.size()));
  writeRaw(out, static_cast<uint64_t>(runs.size()));
  writeRaw(out, uint32_t{kRunLenBits});
  writeRaw(out, uint32_t{1u});
  writeRaw(out, decoded.rowPtr);
  writeRaw(out, rowRunPtr);
  writeRaw(out, rowMax);
  writeRaw(out, valuesQ);
  if ((valuesQ.size() % 2u) != 0u) {
    writeRaw(out, uint16_t{0u});
  }
  writeRaw(out, runs);
  return decoded;
}

double relL2(const std::vector<double> &got, const std::vector<double> &ref) {
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = got[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return std::sqrt(num) / (std::sqrt(den) + 1e-30);
}

// Chunk directory pair (CSRQ + dequantized CSRB reference) for one matrix.
struct MatrixPair {
  std::filesystem::path csrqDir;
  std::filesystem::path csrbDir;
  spect::io::CompactChunkCSRDirectory csrq;
  spect::io::CompactChunkCSRDirectory csrb;
};

// Rows [begin, end) of a CSR as a standalone CSR.
HostCsr sliceRows(const HostCsr &src, size_t begin, size_t end) {
  HostCsr out;
  out.rows = static_cast<int>(end - begin);
  out.cols = src.cols;
  const uint32_t base = src.rowPtr[begin];
  for (size_t r = begin; r <= end; ++r)
    out.rowPtr.push_back(src.rowPtr[r] - base);
  out.colIdx.assign(src.colIdx.begin() + src.rowPtr[begin],
                    src.colIdx.begin() + src.rowPtr[end]);
  out.values.assign(src.values.begin() + src.rowPtr[begin],
                    src.values.begin() + src.rowPtr[end]);
  return out;
}

// chunkCount chunks of whole compact angles (compactNumAngles must divide).
MatrixPair makeMatrixPair(const std::string &name,
                          const spect::io::CompactSymmetryMetadata &meta,
                          const HostCsr &src, int chunkCount = 1) {
  MatrixPair pair;
  const auto root = std::filesystem::temp_directory_path() / name;
  pair.csrqDir = root / "csrq";
  pair.csrbDir = root / "csrb";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(pair.csrqDir);
  std::filesystem::create_directories(pair.csrbDir);
  const int anglesPerChunk = meta.compactNumAngles / chunkCount;
  for (int chunk = 0; chunk < chunkCount; ++chunk) {
    const int angleOffset = chunk * anglesPerChunk;
    const size_t rowBegin =
        static_cast<size_t>(angleOffset) * meta.compactRaysPerAngle;
    const size_t rowEnd = rowBegin + static_cast<size_t>(anglesPerChunk) *
                                         meta.compactRaysPerAngle;
    const HostCsr part = sliceRows(src, rowBegin, rowEnd);
    char stem[32];
    std::snprintf(stem, sizeof(stem), "chunk_%03d", chunk);
    const HostCsr decoded = writeCsrq(pair.csrqDir / (std::string(stem) + ".csrq"),
                                      part, angleOffset, anglesPerChunk);
    writeCsrb(pair.csrbDir / (std::string(stem) + ".csrb"), decoded,
              angleOffset, anglesPerChunk);
  }
  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  pair.csrq = spect::io::openCompactChunkCSRDirectory(pair.csrqDir.string(),
                                                      dirMeta, 64u << 20);
  pair.csrb = spect::io::openCompactChunkCSRDirectory(pair.csrbDir.string(),
                                                      dirMeta, 64u << 20);
  return pair;
}

// Runs sensitivity, single-bed numerator, forward projection and a two-bed
// numerator (host input and prepared resident batch input) on both
// directories and checks the CSRQ path against the CSRB reference.
void expectCsrqMatchesCsrb(const MatrixPair &pair,
                           const spect::io::CompactSymmetryMetadata &meta,
                           double tol) {
  ASSERT_EQ(pair.csrq.format(), spect::io::CompactChunkFormat::CsrqRunU16);
  ASSERT_EQ(pair.csrb.format(), spect::io::CompactChunkFormat::CsrbFloat32);
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());

  std::vector<double> image0(nvox);
  std::vector<double> image1(nvox);
  std::vector<double> proj0(fullRows);
  std::vector<double> proj1(fullRows);
  for (size_t i = 0; i < nvox; ++i) {
    image0[i] = 0.5 + 0.37 * std::sin(0.11 * static_cast<double>(i));
    image1[i] = 0.8 + 0.25 * std::cos(0.07 * static_cast<double>(i));
  }
  for (size_t i = 0; i < fullRows; ++i) {
    proj0[i] = 1.0 + 0.01 * static_cast<double>(i % 97);
    proj1[i] = 2.0 + 0.02 * static_cast<double>(i % 89);
  }

  auto runAll = [&](const spect::io::CompactChunkCSRDirectory &chunks,
                    std::vector<double> &denom, std::vector<double> &numer,
                    std::vector<double> &forward,
                    std::vector<std::vector<double>> &twoBed,
                    std::vector<std::vector<double>> &twoBedResident) {
    denom.assign(nvox, 0.0);
    ASSERT_TRUE(spect::cuda::computeCompactChunkSensitivityCUDA(
        chunks, meta, 1, denom.data(), nullptr, nullptr));
    numer.assign(nvox, 0.0);
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
        chunks, meta, proj0.data(), nullptr, image0.data(), 1, numer.data()));
    forward.assign(fullRows, 0.0);
    ASSERT_TRUE(spect::cuda::computeCompactChunkForwardProjectionCUDA(
        chunks, meta, image0.data(), 1, forward.data(), nullptr));

    twoBed.assign(2, std::vector<double>(nvox, 0.0));
    const double *imagePtrs[2] = {image0.data(), image1.data()};
    const double *projPtrs[2] = {proj0.data(), proj1.data()};
    double *numerPtrs[2] = {twoBed[0].data(), twoBed[1].data()};
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
        chunks, meta, projPtrs, nullptr, imagePtrs, 2, 1, numerPtrs));

    std::vector<double> globalImage(2 * nvox);
    std::copy(image0.begin(), image0.end(), globalImage.begin());
    std::copy(image1.begin(), image1.end(),
              globalImage.begin() + static_cast<ptrdiff_t>(nvox));
    const int bedZStarts[2] = {0, meta.Nz};
    ASSERT_TRUE(spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
        meta, globalImage.data(), globalImage.size(), 2 * meta.Nz, bedZStarts,
        2, false));
    twoBedResident.assign(2, std::vector<double>(nvox, 0.0));
    double *residentPtrs[2] = {twoBedResident[0].data(),
                               twoBedResident[1].data()};
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
        chunks, meta, projPtrs, nullptr, nullptr, 2, 1, residentPtrs, false,
        nullptr, nullptr, true, true, true));
  };

  std::vector<double> denomRef, numerRef, forwardRef;
  std::vector<std::vector<double>> twoBedRef, twoBedResidentRef;
  runAll(pair.csrb, denomRef, numerRef, forwardRef, twoBedRef,
         twoBedResidentRef);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());

  spect::cuda::resetCudaChunkStageTimingCUDA();
  std::vector<double> denom, numer, forward;
  std::vector<std::vector<double>> twoBed, twoBedResident;
  runAll(pair.csrq, denom, numer, forward, twoBed, twoBedResident);
  const auto timing = spect::cuda::getCudaChunkStageTimingSnapshotCUDA();
  EXPECT_GT(timing.csrqForwardCalls, 0u);
  EXPECT_GT(timing.csrqBackwardCalls, 0u);
  EXPECT_EQ(timing.cusparseForwardCalls, 0u);
  EXPECT_EQ(timing.cusparseBackwardCalls, 0u);
  EXPECT_EQ(timing.compactMatrixUploadCalls, 1u);
  EXPECT_TRUE(spect::cuda::csrqResidentCUDA(0));
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  EXPECT_FALSE(spect::cuda::csrqResidentCUDA(0));

  // Weighted sensitivity S = A^T W 1 (detector masks / efficiency weights):
  // the CSRB direct path against the synthesized route it replaced,
  // A^T((A 1 * W) / (A 1)), then CSRQ against CSRB.
  std::vector<double> rowWeights(fullRows);
  for (size_t i = 0; i < fullRows; ++i) {
    rowWeights[i] =
        (i % 13 == 0) ? 0.0 : 0.5 + 0.01 * static_cast<double>(i % 50);
  }
  auto weightedSensitivity =
      [&](const spect::io::CompactChunkCSRDirectory &chunks,
          std::vector<double> &weighted) {
        weighted.assign(nvox, 0.0);
        const std::vector<spect::cuda::CompactMatrixChannel> single = {
            {&chunks, 1.0}};
        ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
            single, meta, 1, weighted.data(), nullptr, nullptr,
            rowWeights.data()));
      };
  std::vector<double> weightedRef;
  {
    std::vector<double> ones(nvox, 1.0);
    std::vector<double> forwardOnes(fullRows, 0.0);
    ASSERT_TRUE(spect::cuda::computeCompactChunkForwardProjectionCUDA(
        pair.csrb, meta, ones.data(), 1, forwardOnes.data(), nullptr));
    for (size_t i = 0; i < fullRows; ++i)
      forwardOnes[i] *= rowWeights[i];
    weightedRef.assign(nvox, 0.0);
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
        pair.csrb, meta, forwardOnes.data(), nullptr, ones.data(), 1,
        weightedRef.data()));
  }
  std::vector<double> weightedCsrb, weightedCsrq;
  weightedSensitivity(pair.csrb, weightedCsrb);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  weightedSensitivity(pair.csrq, weightedCsrq);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  EXPECT_LE(relL2(weightedCsrb, weightedRef), tol)
      << "CSRB weighted sensitivity vs synthesized route";
  EXPECT_LE(relL2(weightedCsrq, weightedCsrb), tol)
      << "CSRQ weighted sensitivity";
  EXPECT_GT(*std::max_element(weightedCsrb.begin(), weightedCsrb.end()), 0.0);
  EXPECT_GT(relL2(weightedCsrb, denomRef), 1e-3)
      << "weights must change the sensitivity";

  EXPECT_LE(relL2(denom, denomRef), tol) << "sensitivity";
  EXPECT_LE(relL2(numer, numerRef), tol) << "single-bed numerator";
  EXPECT_LE(relL2(forward, forwardRef), tol) << "forward projection";
  EXPECT_LE(relL2(twoBed[0], twoBedRef[0]), tol) << "two-bed numerator 0";
  EXPECT_LE(relL2(twoBed[1], twoBedRef[1]), tol) << "two-bed numerator 1";
  EXPECT_LE(relL2(twoBedResident[0], twoBedResidentRef[0]), tol)
      << "resident two-bed numerator 0";
  EXPECT_LE(relL2(twoBedResident[1], twoBedResidentRef[1]), tol)
      << "resident two-bed numerator 1";
  // Guard against a trivially passing all-zero comparison.
  EXPECT_GT(*std::max_element(denom.begin(), denom.end()), 0.0);
  EXPECT_GT(*std::max_element(numer.begin(), numer.end()), 0.0);
  EXPECT_GT(*std::max_element(forward.begin(), forward.end()), 0.0);
}

class CsrqCompactMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!spect::cuda::isCudaAvailable()) {
      GTEST_SKIP() << "CUDA is not available";
    }
  }
};

TEST_F(CsrqCompactMatrixTest, ToyMatrixMatchesDequantizedCsrb) {
  const auto meta = makeToySym90ZMeta();
  const auto toy = makeToyCompactCSR();
  HostCsr src;
  src.rows = meta.totalRows;
  src.cols = meta.totalVoxels;
  src.rowPtr = toy.rowPtr;
  src.colIdx = toy.colIdx;
  src.values = toy.values;
  const auto pair = makeMatrixPair("spect_csrq_toy_test", meta, src);
  ASSERT_EQ(pair.csrq.chunks().front().runCount, 10u);
  expectCsrqMatchesCsrb(pair, meta, 1e-5);
  std::filesystem::remove_all(pair.csrqDir.parent_path());
}

// 32x32x2 grid, rows mixing long x-runs (split at 64), hundreds of short
// runs (several 256-run tiles per row) and an empty row.
TEST_F(CsrqCompactMatrixTest, MultiTileRunsMatchDequantizedCsrb) {
  spect::io::CompactSymmetryMetadata meta;
  meta.Nx = 32;
  meta.Ny = 32;
  meta.Nz = 2;
  meta.totalVoxels = 32 * 32 * 2;
  meta.fullNumAngles = 12;
  meta.compactNumAngles = 3;
  meta.detNx = 4;
  meta.detNz = 4;
  meta.compactDetZ = 2;
  meta.compactRaysPerAngle = 8;
  meta.totalRows = 24;
  meta.symCode = 90;
  meta.symmetryOrder = 4;

  std::mt19937 rng(20260903u);
  std::uniform_real_distribution<float> valueDist(1e-4f, 1.0f);
  HostCsr src;
  src.rows = meta.totalRows;
  src.cols = meta.totalVoxels;
  src.rowPtr.push_back(0u);
  size_t longRunRows = 0;
  size_t multiTileRows = 0;
  for (int row = 0; row < src.rows; ++row) {
    std::vector<uint32_t> cols;
    if (row == 5) {
      // empty row
    } else if (row % 3 == 0) {
      // Dense long runs: lengths 70..100 force the 64-entry split.
      uint32_t pos = static_cast<uint32_t>(rng() % 20);
      while (pos + 100u < static_cast<uint32_t>(src.cols)) {
        const uint32_t len = 70u + rng() % 31u;
        for (uint32_t k = 0; k < len; ++k) cols.push_back(pos + k);
        pos += len + 1u + rng() % 5u;
      }
      ++longRunRows;
    } else {
      // Short runs (1..2) with gaps of 1: ~700 runs -> 3 tiles of 256.
      uint32_t pos = static_cast<uint32_t>(rng() % 3);
      while (pos + 2u < static_cast<uint32_t>(src.cols)) {
        const uint32_t len = 1u + rng() % 2u;
        for (uint32_t k = 0; k < len; ++k) cols.push_back(pos + k);
        pos += len + 1u;
      }
      ++multiTileRows;
    }
    // CSRB rows are unsorted in production; shuffle to prove the converter
    // sorts and the reference CSRB is written in the sorted order.
    std::shuffle(cols.begin(), cols.end(), rng);
    for (uint32_t col : cols) {
      src.colIdx.push_back(col);
      src.values.push_back(valueDist(rng));
    }
    src.rowPtr.push_back(static_cast<uint32_t>(src.colIdx.size()));
  }
  ASSERT_GT(longRunRows, 0u);
  ASSERT_GT(multiTileRows, 0u);

  const auto pair = makeMatrixPair("spect_csrq_multitile_test", meta, src);
  const auto &info = pair.csrq.chunks().front();
  EXPECT_EQ(info.nnz, src.values.size());
  // Every short-run row alone contributes > 256 runs, so the tile loop must
  // iterate more than once per row.
  EXPECT_GT(info.runCount, 256u * multiTileRows);
  expectCsrqMatchesCsrb(pair, meta, 1e-5);
  std::filesystem::remove_all(pair.csrqDir.parent_path());

  // Same matrix in three chunks, forced through the streamed path (pinned
  // host copies, two device slots, double-buffered uploads). Numerics must
  // match the resident path.
  const auto pair3 = makeMatrixPair("spect_csrq_multitile3_test", meta, src, 3);
  ASSERT_EQ(pair3.csrq.chunks().size(), 3u);
  spect::cuda::setCsrqForceStreamingCUDA(true);
  testing::internal::CaptureStderr();
  expectCsrqMatchesCsrb(pair3, meta, 1e-5);
  const std::string log = testing::internal::GetCapturedStderr();
  spect::cuda::setCsrqForceStreamingCUDA(false);
  EXPECT_NE(log.find("streaming forced"), std::string::npos) << log;
  EXPECT_NE(log.find("streamed (pinned host)"), std::string::npos) << log;
  std::filesystem::remove_all(pair3.csrqDir.parent_path());
}

// Joint two-window MLEM (MB-JR) on the CSRQ path. Both channels use the same
// matrix so the analytic expectations are exact: with weights w1, w2 and no
// background the joint numerator is 2 x the single-window numerator (weights
// cancel in each ratio), the joint sensitivity is (w1 + w2) x A^T 1, and with
// a background b the ratio becomes w p / (w A x + b) per window.
TEST_F(CsrqCompactMatrixTest, JointTwoWindowMatchesAnalyticCombination) {
  spect::io::CompactSymmetryMetadata meta;
  meta.Nx = 16;
  meta.Ny = 16;
  meta.Nz = 2;
  meta.totalVoxels = 16 * 16 * 2;
  meta.fullNumAngles = 12;
  meta.compactNumAngles = 3;
  meta.detNx = 4;
  meta.detNz = 4;
  meta.compactDetZ = 2;
  meta.compactRaysPerAngle = 8;
  meta.totalRows = 24;
  meta.symCode = 90;
  meta.symmetryOrder = 4;

  std::mt19937 rng(7u);
  std::uniform_real_distribution<float> valueDist(0.05f, 1.0f);
  HostCsr src;
  src.rows = meta.totalRows;
  src.cols = meta.totalVoxels;
  src.rowPtr.push_back(0u);
  for (int row = 0; row < src.rows; ++row) {
    uint32_t pos = static_cast<uint32_t>(rng() % 5);
    while (pos + 4u < static_cast<uint32_t>(src.cols)) {
      const uint32_t len = 1u + rng() % 4u;
      for (uint32_t k = 0; k < len; ++k) {
        src.colIdx.push_back(pos + k);
        src.values.push_back(valueDist(rng));
      }
      pos += len + 1u + rng() % 3u;
    }
    src.rowPtr.push_back(static_cast<uint32_t>(src.colIdx.size()));
  }

  const auto root =
      std::filesystem::temp_directory_path() / "spect_csrq_joint_test";
  std::filesystem::remove_all(root);
  const auto dirA = root / "a";
  const auto dirB = root / "b";
  std::filesystem::create_directories(dirA);
  std::filesystem::create_directories(dirB);
  writeCsrq(dirA / "chunk_000.csrq", src, 0, meta.compactNumAngles);
  writeCsrq(dirB / "chunk_000.csrq", src, 0, meta.compactNumAngles);
  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunksA =
      spect::io::openCompactChunkCSRDirectory(dirA.string(), dirMeta, 0);
  const auto chunksB =
      spect::io::openCompactChunkCSRDirectory(dirB.string(), dirMeta, 0);

  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());
  std::vector<double> image(nvox);
  std::vector<double> projA(fullRows);
  std::vector<double> projB(fullRows);
  std::vector<double> backgroundB(fullRows);
  for (size_t i = 0; i < nvox; ++i)
    image[i] = 0.6 + 0.3 * std::sin(0.13 * static_cast<double>(i));
  for (size_t i = 0; i < fullRows; ++i) {
    projA[i] = 1.0 + 0.01 * static_cast<double>(i % 53);
    projB[i] = 3.0 + 0.02 * static_cast<double>(i % 41);
    backgroundB[i] = 0.2 + 0.001 * static_cast<double>(i % 29);
  }
  const double w1 = 0.062;
  const double w2 = 0.104;

  // Single-window references (weight 1, matrix A alone).
  std::vector<double> denomSingle(nvox, 0.0);
  std::vector<double> numerA(nvox, 0.0);
  std::vector<double> numerB(nvox, 0.0);
  std::vector<double> numerBWithBackgroundScaled(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::computeCompactChunkSensitivityCUDA(
      chunksA, meta, 1, denomSingle.data(), nullptr, nullptr));
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      chunksA, meta, projA.data(), nullptr, image.data(), 1, numerA.data()));
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      chunksA, meta, projB.data(), nullptr, image.data(), 1, numerB.data()));
  // w2 * A^T(p / (w2 A x + b)) == A^T(p / (A x + b / w2)).
  {
    std::vector<double> scaledBackground(fullRows);
    for (size_t i = 0; i < fullRows; ++i)
      scaledBackground[i] = backgroundB[i] / w2;
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
        chunksA, meta, projB.data(), scaledBackground.data(), image.data(), 1,
        numerBWithBackgroundScaled.data()));
  }
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());

  std::vector<spect::cuda::CompactMatrixChannel> channels(2);
  channels[0].chunks = &chunksA;
  channels[0].weight = w1;
  channels[1].chunks = &chunksB;
  channels[1].weight = w2;

  std::vector<double> denomJoint(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
      channels, meta, 1, denomJoint.data(), nullptr, nullptr));
  EXPECT_TRUE(spect::cuda::csrqResidentCUDA(0));
  EXPECT_TRUE(spect::cuda::csrqResidentCUDA(1));
  std::vector<double> denomExpected(nvox);
  for (size_t i = 0; i < nvox; ++i)
    denomExpected[i] = (w1 + w2) * denomSingle[i];
  EXPECT_LE(relL2(denomJoint, denomExpected), 1e-5) << "joint sensitivity";

  // Joint numerator without background: sum of the two single-window terms.
  {
    std::vector<double> numerJoint(nvox, 0.0);
    const double *projections[2] = {projA.data(), projB.data()};
    const double *images[1] = {image.data()};
    double *numers[1] = {numerJoint.data()};
    ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
        channels, meta, projections, nullptr, images, 1, 1, numers));
    std::vector<double> expected(nvox);
    for (size_t i = 0; i < nvox; ++i)
      expected[i] = numerA[i] + numerB[i];
    EXPECT_LE(relL2(numerJoint, expected), 1e-5) << "joint numerator";
  }
  // Joint numerator with a background on window 2 only.
  {
    std::vector<double> numerJoint(nvox, 0.0);
    const double *projections[2] = {projA.data(), projB.data()};
    const double *backgrounds[2] = {nullptr, backgroundB.data()};
    const double *images[1] = {image.data()};
    double *numers[1] = {numerJoint.data()};
    ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
        channels, meta, projections, backgrounds, images, 1, 1, numers));
    std::vector<double> expected(nvox);
    for (size_t i = 0; i < nvox; ++i)
      expected[i] = numerA[i] + numerBWithBackgroundScaled[i];
    EXPECT_LE(relL2(numerJoint, expected), 1e-5)
        << "joint numerator with background";
    EXPECT_GT(*std::max_element(numerJoint.begin(), numerJoint.end()), 0.0);
  }
  // Two beds, resident batch input: each bed sees both windows.
  {
    std::vector<double> globalImage(2 * nvox);
    std::copy(image.begin(), image.end(), globalImage.begin());
    for (size_t i = 0; i < nvox; ++i)
      globalImage[nvox + i] = 1.5 * image[i];
    const int bedZStarts[2] = {0, meta.Nz};
    ASSERT_TRUE(spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
        meta, globalImage.data(), globalImage.size(), 2 * meta.Nz, bedZStarts,
        2, false));
    std::vector<std::vector<double>> numersJoint(2, std::vector<double>(nvox));
    // [channel * bedCount + bed]
    const double *projections[4] = {projA.data(), projA.data(), projB.data(),
                                    projB.data()};
    double *numers[2] = {numersJoint[0].data(), numersJoint[1].data()};
    ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
        channels, meta, projections, nullptr, nullptr, 2, 1, numers, false,
        nullptr, nullptr, true, true, true));
    std::vector<double> expected0(nvox);
    for (size_t i = 0; i < nvox; ++i)
      expected0[i] = numerA[i] + numerB[i];
    EXPECT_LE(relL2(numersJoint[0], expected0), 1e-5) << "joint bed 0";
    // Bed 1 holds 1.5 x image: A^T(p / (1.5 A x)) = expected0 / 1.5.
    std::vector<double> expected1(nvox);
    for (size_t i = 0; i < nvox; ++i)
      expected1[i] = expected0[i] / 1.5;
    EXPECT_LE(relL2(numersJoint[1], expected1), 1e-5) << "joint bed 1";
  }
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  EXPECT_FALSE(spect::cuda::csrqResidentCUDA(1));
  std::filesystem::remove_all(root);
}

TEST(CsrqCompactMatrixFormat, RejectsMixedAndTruncatedDirectories) {
  const auto meta = makeToySym90ZMeta();
  const auto toy = makeToyCompactCSR();
  HostCsr src;
  src.rows = meta.totalRows;
  src.cols = meta.totalVoxels;
  src.rowPtr = toy.rowPtr;
  src.colIdx = toy.colIdx;
  src.values = toy.values;
  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();

  const auto root =
      std::filesystem::temp_directory_path() / "spect_csrq_format_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const HostCsr decoded =
      writeCsrq(root / "chunk_000.csrq", src, 0, meta.compactNumAngles);
  {
    const auto chunks = spect::io::openCompactChunkCSRDirectory(
        root.string(), dirMeta, 64u << 20);
    ASSERT_EQ(chunks.format(), spect::io::CompactChunkFormat::CsrqRunU16);
    const auto chunk = chunks.loadChunk(0);
    EXPECT_THROW(chunk->view(), std::runtime_error);
    const auto view = chunk->runView();
    ASSERT_TRUE(view.valid());
    EXPECT_EQ(view.nnz, decoded.values.size());
    // Decoded values from the run view match the reference CSR bit for bit.
    size_t t = 0;
    for (int row = 0; row < view.rows; ++row) {
      for (uint32_t j = view.rowRunPtr[row]; j < view.rowRunPtr[row + 1]; ++j) {
        const uint32_t start = view.runs[j] >> spect::io::kCompactRunLenBits;
        const uint32_t len =
            (view.runs[j] & (spect::io::kCompactRunMaxLen - 1u)) + 1u;
        for (uint32_t k = 0; k < len; ++k, ++t) {
          EXPECT_EQ(start + k, decoded.colIdx[t]);
          const float tq = static_cast<float>(view.valuesQ[t]) * (1.0f / 65535.0f);
          EXPECT_EQ(view.rowMax[row] * (tq * tq), decoded.values[t]);
        }
      }
    }
    EXPECT_EQ(t, decoded.values.size());
  }

  writeCsrb(root / "chunk_001.csrb", decoded, 0, meta.compactNumAngles);
  EXPECT_THROW(spect::io::openCompactChunkCSRDirectory(root.string(), dirMeta,
                                                       64u << 20),
               std::runtime_error);
  std::filesystem::remove(root / "chunk_001.csrb");

  std::filesystem::resize_file(root / "chunk_000.csrq",
                               std::filesystem::file_size(root / "chunk_000.csrq") - 4u);
  EXPECT_THROW(spect::io::openCompactChunkCSRDirectory(root.string(), dirMeta,
                                                       64u << 20),
               std::runtime_error);
  std::filesystem::remove_all(root);
}

} // namespace
} // namespace spect::main
