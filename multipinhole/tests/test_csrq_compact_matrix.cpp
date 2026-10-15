#include <gtest/gtest.h>

#include "core/AttenuationCoefficients.h"
#include "cuda/CudaConfig.h"
#include "filters/cuda/VolumeOpsCUDA.h"
#include "io/CompactChunkCSR.h"
#include "recon/BedOffsets.h"
#include "recon/BsremUpdate.h"
#include "regularizer/RegConfig.h"
#include "sparse/CompactSymmetricProjector.h"
#include "sparse/ProjDomainChunked.h"
#include "sparse/SparseOpsCUDA.h"
#include "tests/common/compact_symmetry_test_helpers.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
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
  EXPECT_GT(timing.csrqPassCalls, 0u);
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

class CsrqProfileTest : public CsrqCompactMatrixTest,
                        public ::testing::WithParamInterface<bool> {
protected:
  void applyProfile(spect::io::CompactSymmetryMetadata &meta) {
    if (GetParam()) {
      meta.profile = spect::io::CompactSymmetryProfile::Rot90;
      meta.detNz = meta.compactDetZ;
    }
  }
};
INSTANTIATE_TEST_SUITE_P(Rot90AndRot90Z, CsrqProfileTest, ::testing::Bool());

TEST_P(CsrqProfileTest, ToyMatrixMatchesDequantizedCsrb) {
  auto meta = makeToySym90ZMeta();
  applyProfile(meta);
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
  // match the resident path. The pin budget is explicit: the automatic one
  // follows the host's free RAM and may legitimately be 0 on a busy machine.
  const auto pair3 = makeMatrixPair("spect_csrq_multitile3_test", meta, src, 3);
  ASSERT_EQ(pair3.csrq.chunks().size(), 3u);
  constexpr int64_t kUnlimitedPin = int64_t{1} << 50;
  spect::cuda::setCsrqForceStreamingCUDA(true);
  spect::cuda::setCsrqHostPlanForTestsCUDA(kUnlimitedPin, -1);
  testing::internal::CaptureStderr();
  expectCsrqMatchesCsrb(pair3, meta, 1e-5);
  const std::string log = testing::internal::GetCapturedStderr();
  spect::cuda::setCsrqHostPlanForTestsCUDA(-1, -1);
  spect::cuda::setCsrqForceStreamingCUDA(false);
  EXPECT_NE(log.find("streaming forced"), std::string::npos) << log;
  EXPECT_NE(log.find("pinned host 3/3 chunks"), std::string::npos) << log;

  // Host page-lock budget for the first chunk only: chunks 1 and 2 are read
  // back from their files every pass through 4 KiB staging pieces (several
  // pieces per span). Numerics must not change.
  {
    const auto &first = pair3.csrq.chunks().front();
    const int64_t firstPayload = static_cast<int64_t>(
        4u * first.runCount + 2u * first.nnz);
    spect::cuda::setCsrqForceStreamingCUDA(true);
    spect::cuda::setCsrqHostPlanForTestsCUDA(firstPayload, 4096);
    testing::internal::CaptureStderr();
    expectCsrqMatchesCsrb(pair3, meta, 1e-5);
    const std::string diskLog = testing::internal::GetCapturedStderr();
    spect::cuda::setCsrqHostPlanForTestsCUDA(-1, -1);
    spect::cuda::setCsrqForceStreamingCUDA(false);
    EXPECT_NE(diskLog.find("pinned host 1/3 chunks"), std::string::npos) << diskLog;
    EXPECT_NE(diskLog.find("re-read from file"), std::string::npos) << diskLog;
  }

  // The driver refuses page-locked memory part way through warmup (at most 11
  // live allocations: 5 per pinned chunk, 3 per file-read chunk, 2 staging
  // buffers): already pinned payloads are handed back so the row pointers and
  // staging buffers still fit, and numerics stay unchanged.
  {
    spect::cuda::setCsrqForceStreamingCUDA(true);
    spect::cuda::setCsrqHostPlanForTestsCUDA(kUnlimitedPin, 4096);
    spect::cuda::setPinnedAllocationLimitForTestsCUDA(11);
    testing::internal::CaptureStderr();
    expectCsrqMatchesCsrb(pair3, meta, 1e-5);
    const std::string refusedLog = testing::internal::GetCapturedStderr();
    spect::cuda::setPinnedAllocationLimitForTestsCUDA(-1);
    spect::cuda::setCsrqHostPlanForTestsCUDA(-1, -1);
    spect::cuda::setCsrqForceStreamingCUDA(false);
    EXPECT_NE(refusedLog.find("page-lock refused"), std::string::npos) << refusedLog;
    EXPECT_NE(refusedLog.find("pinned host 0/3 chunks"), std::string::npos) << refusedLog;
  }

  // Packed [voxel][8] backward (float4 atomics on sm_90+, scalar fallback
  // here): same sums through a different scatter layout and consumer.
  ASSERT_TRUE(spect::cuda::setCsrqPackedBackwardCUDA(true));
  testing::internal::CaptureStderr();
  expectCsrqMatchesCsrb(pair3, meta, 1e-5);
  const std::string packedLog = testing::internal::GetCapturedStderr();
  ASSERT_TRUE(spect::cuda::setCsrqPackedBackwardCUDA(false));
  EXPECT_NE(packedLog.find("packed [voxel][K] backward"), std::string::npos)
      << packedLog;
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
  for (double invalid : {std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    channels[1].weight = invalid;
    EXPECT_THROW(spect::cuda::computePoissonSensitivityCUDA(
                     channels, meta, 1, denomJoint.data(), nullptr, nullptr),
                 std::runtime_error);
  }
  channels[1].weight = w2;
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

// Different window matrices, background, detector weights, two beds, empty
// rows/chunk selections and subset transitions, with and without CTAC. The
// reference uses independently decoded CSRB matrices, never CSRQ's selector.
TEST_P(CsrqProfileTest, JointSubsetsWithAndWithoutCtacMatchDecodedCsrb) {
  spect::io::CompactSymmetryMetadata meta;
  meta.Nx = meta.Ny = 8; meta.Nz = 2; meta.totalVoxels = 128;
  meta.fullNumAngles = 80; meta.compactNumAngles = 20;
  meta.detNx = meta.detNz = 4; meta.compactDetZ = 2;
  meta.compactRaysPerAngle = 8; meta.totalRows = 160;
  meta.symCode = 90; meta.symmetryOrder = 4;
  applyProfile(meta);
  HostCsr src; src.rows = meta.totalRows; src.cols = meta.totalVoxels;
  src.rowPtr.push_back(0);
  for (int row = 0; row < src.rows; ++row) {
    if (row % 17 != 0) for (int col = row % 3; col < src.cols; col += 3) {
      src.colIdx.push_back(col);
      src.values.push_back(.1f + .003f * ((row + col) % 29));
    }
    src.rowPtr.push_back(static_cast<uint32_t>(src.values.size()));
  }
  const auto a = makeMatrixPair("spect_joint_subset_a", meta, src, 4);
  for (size_t i = 0; i < src.values.size(); ++i)
    src.values[i] *= .6f + .02f * (i % 19);
  const auto b = makeMatrixPair("spect_joint_subset_b", meta, src, 5);
  const size_t n = meta.totalVoxels, nr = meta.fullRows();
  std::vector<double> x0(n), x1(n), p0(nr), p1(nr), bg(nr), weights(nr);
  for (size_t j = 0; j < n; ++j) { x0[j] = .7 + .2 * std::sin(j*.13); x1[j] = 1.7*x0[j]; }
  for (size_t j = 0; j < nr; ++j) {
    p0[j] = 1 + .02*(j%23); p1[j] = 2 + .03*(j%31);
    bg[j] = .1 + .001*(j%17); weights[j] = j%7 ? 1.0 : 0.0;
  }
  constexpr double w[2] = {.1036, .0617};
  std::vector<double> scaledBg(nr);
  for (size_t j = 0; j < nr; ++j) scaledBg[j] = bg[j]/w[1];
  const double *images[2] = {x0.data(), x1.data()};
  const double *proj[4] = {p0.data(), p1.data(), p1.data(), p0.data()};
  const double *bgs[4] = {nullptr, nullptr, bg.data(), bg.data()};
  const double *refBgs[4] = {nullptr, nullptr, scaledBg.data(), scaledBg.data()};
  const spect::io::CompactChunkCSRDirectory *refs[2] = {&a.csrb, &b.csrb};
  std::vector<spect::cuda::CompactMatrixChannel> joint{{&a.csrq,w[0]}, {&b.csrq,w[1]}};
  std::vector<double> weights2(weights);
  for (size_t j=0;j<nr;++j) weights2[j] *= 0.4 + 0.03*(j%13);
  joint[1].sensitivityRowWeights = weights2.data();
  std::vector<double> globalX(x0); globalX.insert(globalX.end(),x1.begin(),x1.end());
  const int zStarts[2] = {0, meta.Nz};
  const size_t attCount = static_cast<size_t>(meta.fullNumAngles) * n;
  std::vector<uint16_t> att0(attCount), att1(attCount);
  for (size_t j = 0; j < attCount; ++j) {
    att0[j] = static_cast<uint16_t>(8000 + (j * 37) % 55000);
    att1[j] = static_cast<uint16_t>(11000 + (j * 71) % 52000);
    if (j % 19 == 0) att0[j] = 0;
    if (j % 23 == 0) att1[j] = 65535;
  }
  const uint16_t *att[2] = {att0.data(), att1.data()};
  for (const bool ctac : {false, true}) {
    joint[1].attenPower = ctac ? 1.35 : 1.0;
    // References are retained across device modes so the later modes visit
    // successive subsets/stages without releasing CSRQ's per-subset plans.
    std::map<std::pair<int, int>, std::array<std::vector<double>, 3>> expected;
    std::vector<double> expectedForward(nr);
    ASSERT_TRUE(spect::cuda::computeCompactChunkForwardProjectionCUDA(
        a.csrb, meta, x0.data(), 1, expectedForward.data(), ctac ? att0.data() : nullptr));
    ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
    // Modes: resident, streamed from page-locked host copies, and streamed
    // with every chunk re-read from its file through 64-byte staging pieces.
    for (int mode : {0,1,2}) for (bool packed : {false,true}) {
      const bool streamed = mode > 0;
      spect::cuda::setCsrqForceStreamingCUDA(streamed);
      spect::cuda::setCsrqHostPlanForTestsCUDA(mode == 2 ? 0 : int64_t{1} << 50,
                                               mode == 2 ? 64 : -1);
      ASSERT_TRUE(spect::cuda::setCsrqPackedBackwardCUDA(packed));
      for (int count : {10,5,2,1,10}) {
        std::vector<double> sumS(n,0.0), fullS(n,0.0);
        for (int index : spect::compactSubsetOrder(count)) {
          SCOPED_TRACE(::testing::Message() << "CTAC=" << ctac << " mode=" << mode
              << " packed=" << packed << " N=" << count << " index=" << index);
          const spect::CompactSubsetSpec spec{count,index};
          const auto key = std::make_pair(count, index);
          if (expected.count(key) == 0) {
            auto &ref = expected[key];
            for (auto &volume : ref) volume.assign(n, 0.0);
            for (int k=0;k<2;++k) {
              ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
              std::vector<spect::cuda::CompactMatrixChannel> one{{refs[k],1.0}};
              one[0].attenPower = joint[k].attenPower;
              std::vector<double> sens(n), q0(n), q1(n);
              double *out[2] = {q0.data(),q1.data()};
              ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(one,meta,1,sens.data(),nullptr,
                  ctac ? att0.data() : nullptr,k ? weights2.data() : weights.data(),&spec));
              ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
                  one,meta,proj+2*k,refBgs+2*k,images,2,1,out,false,nullptr,
                  ctac ? att : nullptr,true,false,true,&spec));
              for (size_t j=0;j<n;++j) { ref[0][j]+=w[k]*sens[j]; ref[1][j]+=q0[j]; ref[2][j]+=q1[j]; }
            }
            ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
          }
          const auto &ref = expected.at(key);
          std::vector<double> sens(n), q0(n), q1(n); double *out[2]={q0.data(),q1.data()};
          ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(joint,meta,1,sens.data(),nullptr,
              ctac ? att0.data() : nullptr,weights.data(),&spec));
          ASSERT_TRUE(spect::cuda::csrqResidentCUDA(0)); // ready includes streamed matrices
          const bool residentInput = index%2 == 0;
          if (residentInput) ASSERT_TRUE(spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
              meta,globalX.data(),globalX.size(),2*meta.Nz,zStarts,2,false));
          ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
              joint,meta,proj,bgs,residentInput?nullptr:images,2,1,out,false,nullptr,
              ctac ? att : nullptr,true,residentInput,true,&spec));
          EXPECT_LE(relL2(sens,ref[0]),1e-5);
          EXPECT_LE(relL2(q0,ref[1]),1e-5); EXPECT_LE(relL2(q1,ref[2]),1e-5);
          EXPECT_GT(*std::max_element(q0.begin(),q0.end()),0);
          for (size_t j=0;j<n;++j) sumS[j]+=sens[j];
        }
        ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(joint,meta,1,fullS.data(),nullptr,
            ctac ? att0.data() : nullptr,weights.data()));
        EXPECT_LE(relL2(sumS,fullS),1e-5) << "voxelwise sum of subset sensitivities";
        std::vector<double> forward(nr);
        ASSERT_TRUE(spect::cuda::computeCompactChunkForwardProjectionCUDA(
            a.csrq, meta, x0.data(), 1, forward.data(), ctac ? att0.data() : nullptr));
        EXPECT_LE(relL2(forward,expectedForward),1e-5) << "full forward after subset pass";
      }
      ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
    }
  }
  spect::cuda::setCsrqForceStreamingCUDA(false);
  spect::cuda::setCsrqHostPlanForTestsCUDA(-1, -1);
  ASSERT_TRUE(spect::cuda::setCsrqPackedBackwardCUDA(false));
  std::filesystem::remove_all(a.csrqDir.parent_path());
  std::filesystem::remove_all(b.csrqDir.parent_path());
}

// Residency planner retry: when the second window's matrix reports "does not
// fit even streamed" after the first was made resident, the first is demoted
// to streamed and the second retried; the joint result must be unchanged.
TEST_F(CsrqCompactMatrixTest, DemotesResidentMatrixWhenSecondDoesNotFit) {
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

  std::mt19937 rng(31u);
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
      std::filesystem::temp_directory_path() / "spect_csrq_demote_test";
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
  std::vector<spect::cuda::CompactMatrixChannel> channels(2);
  channels[0].chunks = &chunksA;
  channels[0].weight = 0.4;
  channels[1].chunks = &chunksB;
  channels[1].weight = 0.6;
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);

  // Reference: both resident (nothing fails).
  std::vector<double> denomRef(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(
      channels, meta, 1, denomRef.data(), nullptr, nullptr));
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());

  // The second automatic warmup fails; the planner must demote the first
  // matrix to streamed and retry the second.
  spect::cuda::setCsrqWarmupFitFailuresForTestsCUDA(/*skip=*/1, /*count=*/1);
  testing::internal::CaptureStderr();
  std::vector<double> denom(nvox, 0.0);
  const bool ok = spect::cuda::computePoissonSensitivityCUDA(
      channels, meta, 1, denom.data(), nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();
  spect::cuda::setCsrqWarmupFitFailuresForTestsCUDA(0, 0);
  ASSERT_TRUE(ok) << log;
  EXPECT_NE(log.find("demoting 1 resident matrix"), std::string::npos) << log;
  EXPECT_NE(log.find("streaming forced"), std::string::npos) << log;
  EXPECT_TRUE(spect::cuda::csrqResidentCUDA(0));
  EXPECT_TRUE(spect::cuda::csrqResidentCUDA(1));
  EXPECT_LE(relL2(denom, denomRef), 1e-6) << "joint sensitivity after retry";
  EXPECT_GT(*std::max_element(denom.begin(), denom.end()), 0.0);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());

  // Without an earlier resident matrix to demote the failure is fatal.
  spect::cuda::setCsrqWarmupFitFailuresForTestsCUDA(/*skip=*/0, /*count=*/1);
  std::vector<double> denomFail(nvox, 0.0);
  EXPECT_THROW(spect::cuda::computePoissonSensitivityCUDA(
                   channels, meta, 1, denomFail.data(), nullptr, nullptr),
               spect::cuda::CsrqDeviceMemoryError);
  spect::cuda::setCsrqWarmupFitFailuresForTestsCUDA(0, 0);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  std::filesystem::remove_all(root);
}

// Host oracle of the CTAC model h_ij = a_ij * att[fullAngle(i, k)][M_k(j)]
// on a compact matrix: forward projection H x (full rows), sensitivity H^T 1
// and numerator H^T(p / (H x)), summed over the symmetry branches exactly as
// the projector does (double precision, no quantisation of the products).
struct HostCtacOracle {
  const spect::io::CompactSymmetryMetadata &meta;
  const HostCsr &csr;  // the dequantized compact matrix (all chunks)
  const std::vector<uint16_t> &att;

  double factor(int fullAngle, size_t unrotated) const {
    return static_cast<double>(
               att[static_cast<size_t>(fullAngle) * meta.totalVoxels + unrotated]) /
           65535.0;
  }
  // For every (row, branch) with a full projection row: calls fn(row, fullIndex,
  // fullAngle, symId, zMirror).
  template <typename Fn> void forEachRowBranch(Fn &&fn) const {
    for (int symId = 0; symId < meta.xySymmetryCount(); ++symId) {
      for (int zb = 0; zb < meta.zBranchCount(); ++zb) {
        const bool zMirror = zb != 0;
        for (int row = 0; row < csr.rows; ++row) {
          uint64_t fullIndex = 0;
          if (!spect::io::fullProjectionIndexFromCompactRow(meta, row, symId,
                                                            zMirror, fullIndex))
            continue;
          fn(row, static_cast<size_t>(fullIndex),
             static_cast<int>(fullIndex / meta.fullRaysPerAngle()), symId,
             zMirror);
        }
      }
    }
  }
  size_t unrotated(uint32_t col, int symId, bool zMirror) const {
    return spect::io::mapCompactVoxelIndex(col, meta.Nx, meta.Ny, meta.Nz, symId,
                                           zMirror);
  }
  std::vector<double> forward(const std::vector<double> &image) const {
    std::vector<double> out(static_cast<size_t>(meta.fullRows()), 0.0);
    forEachRowBranch([&](int row, size_t fullIndex, int angle, int symId,
                         bool zMirror) {
      double sum = 0.0;
      for (uint32_t t = csr.rowPtr[row]; t < csr.rowPtr[row + 1]; ++t) {
        const size_t u = unrotated(csr.colIdx[t], symId, zMirror);
        sum += csr.values[t] * factor(angle, u) * image[u];
      }
      out[fullIndex] = sum;
    });
    return out;
  }
  // H^T r for a full-row vector r.
  std::vector<double> backward(const std::vector<double> &r) const {
    std::vector<double> out(static_cast<size_t>(meta.totalVoxels), 0.0);
    forEachRowBranch([&](int row, size_t fullIndex, int angle, int symId,
                         bool zMirror) {
      const double ri = r[fullIndex];
      if (ri == 0.0) return;
      for (uint32_t t = csr.rowPtr[row]; t < csr.rowPtr[row + 1]; ++t) {
        const size_t u = unrotated(csr.colIdx[t], symId, zMirror);
        out[u] += csr.values[t] * factor(angle, u) * ri;
      }
    });
    return out;
  }
  std::vector<double> sensitivity() const {
    return backward(std::vector<double>(static_cast<size_t>(meta.fullRows()), 1.0));
  }
  std::vector<double> numerator(const std::vector<double> &proj,
                                const std::vector<double> &image) const {
    const std::vector<double> yhat = forward(image);
    std::vector<double> ratio(yhat.size(), 0.0);
    for (size_t i = 0; i < yhat.size(); ++i)
      ratio[i] = proj[i] / std::max(yhat[i], 1e-10);
    return backward(ratio);
  }
};

// The dequantized compact matrix a CSRB reference directory encodes, as one
// host CSR over all chunks (rows in compact order).
HostCsr readHostCsr(const spect::io::CompactChunkCSRDirectory &csrb) {
  HostCsr out;
  out.cols = csrb.cols();
  out.rowPtr.push_back(0u);
  for (size_t c = 0; c < csrb.chunks().size(); ++c) {
    const auto chunk = csrb.loadChunk(c);
    const spect::io::CSRView view = chunk->view();
    const uint32_t base = out.rowPtr.back();
    for (int row = 0; row < view.rows; ++row) {
      out.rowPtr.push_back(base + view.row_ptr[row + 1]);
    }
    out.colIdx.insert(out.colIdx.end(), view.col_idx, view.col_idx + view.nnz);
    out.values.insert(out.values.end(), view.val, view.val + view.nnz);
    out.rows += view.rows;
  }
  return out;
}

// CTAC with the u16 full-angle table factorised out of the nnz loop (per
// compact angle: branch batch scaled before the forward, scattered outputs
// scaled after the backward) on both formats against the host oracle of the
// original per-nnz model: sensitivity, single-bed numerator, forward
// projection and a two-bed numerator with per-bed tables (host input and
// resident batch). The adjoint identity <H x, y> = <x, H^T y> is checked on
// the device paths themselves.
TEST_P(CsrqProfileTest, CtacFactorisationMatchesCsrbAndIsAdjoint) {
  spect::io::CompactSymmetryMetadata meta;
  meta.Nx = 16;
  meta.Ny = 16;
  meta.Nz = 4;
  meta.totalVoxels = 16 * 16 * 4;
  meta.fullNumAngles = 24;
  meta.compactNumAngles = 6;
  meta.detNx = 4;
  meta.detNz = 4;
  meta.compactDetZ = 2;
  meta.compactRaysPerAngle = 8;
  meta.totalRows = 48;
  meta.symCode = 90;
  meta.symmetryOrder = 4;

  applyProfile(meta);
  std::mt19937 rng(20260904u);
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
  // Two chunks of three compact angles each so the per-angle walk crosses a
  // chunk boundary and the slice table is rebuilt per chunk.
  const auto pair = makeMatrixPair("spect_csrq_ctac_test", meta, src, 2);
  ASSERT_EQ(pair.csrq.chunks().size(), 2u);

  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());
  const size_t attCount = static_cast<size_t>(meta.fullNumAngles) * nvox;
  // Two distinct tables (one per bed), strictly inside (0, 1) and varying
  // with both the angle and the voxel so a wrong angle or voxel mapping
  // cannot pass.
  std::vector<uint16_t> att0(attCount), att1(attCount);
  std::uniform_int_distribution<int> attDist(9000, 64000);
  for (size_t i = 0; i < attCount; ++i) {
    att0[i] = static_cast<uint16_t>(attDist(rng));
    att1[i] = static_cast<uint16_t>(attDist(rng));
  }
  std::vector<double> image0(nvox), image1(nvox), proj0(fullRows), proj1(fullRows);
  for (size_t i = 0; i < nvox; ++i) {
    image0[i] = 0.5 + 0.37 * std::sin(0.11 * static_cast<double>(i));
    image1[i] = 0.8 + 0.25 * std::cos(0.07 * static_cast<double>(i));
  }
  for (size_t i = 0; i < fullRows; ++i) {
    proj0[i] = 1.0 + 0.01 * static_cast<double>(i % 97);
    proj1[i] = 2.0 + 0.02 * static_cast<double>(i % 89);
  }

  struct Outputs {
    std::vector<double> denom, numer, forward;
    std::vector<std::vector<double>> twoBed, twoBedResident;
  };
  auto runAll = [&](const spect::io::CompactChunkCSRDirectory &chunks,
                    Outputs &out, bool residentBatch) {
    out.denom.assign(nvox, 0.0);
    ASSERT_TRUE(spect::cuda::computeCompactChunkSensitivityCUDA(
        chunks, meta, 1, out.denom.data(), nullptr, att0.data()));
    out.numer.assign(nvox, 0.0);
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
        chunks, meta, proj0.data(), nullptr, image0.data(), 1, out.numer.data(),
        false, nullptr, att0.data()));
    out.forward.assign(fullRows, 0.0);
    ASSERT_TRUE(spect::cuda::computeCompactChunkForwardProjectionCUDA(
        chunks, meta, image0.data(), 1, out.forward.data(), att0.data()));

    const double *imagePtrs[2] = {image0.data(), image1.data()};
    const double *projPtrs[2] = {proj0.data(), proj1.data()};
    const uint16_t *attPtrs[2] = {att0.data(), att1.data()};
    out.twoBed.assign(2, std::vector<double>(nvox, 0.0));
    double *numerPtrs[2] = {out.twoBed[0].data(), out.twoBed[1].data()};
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
        chunks, meta, projPtrs, nullptr, imagePtrs, 2, 1, numerPtrs, false,
        nullptr, attPtrs));
    if (!residentBatch) {
      return;
    }
    std::vector<double> globalImage(2 * nvox);
    std::copy(image0.begin(), image0.end(), globalImage.begin());
    std::copy(image1.begin(), image1.end(),
              globalImage.begin() + static_cast<ptrdiff_t>(nvox));
    const int bedZStarts[2] = {0, meta.Nz};
    ASSERT_TRUE(spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
        meta, globalImage.data(), globalImage.size(), 2 * meta.Nz, bedZStarts,
        2, false));
    out.twoBedResident.assign(2, std::vector<double>(nvox, 0.0));
    double *residentPtrs[2] = {out.twoBedResident[0].data(),
                               out.twoBedResident[1].data()};
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
        chunks, meta, projPtrs, nullptr, nullptr, 2, 1, residentPtrs, false,
        nullptr, attPtrs, true, true, true));
  };

  // Host oracle on the dequantized matrix both directories encode.
  const HostCsr decoded = readHostCsr(pair.csrb);
  ASSERT_EQ(decoded.rows, meta.totalRows);
  const HostCtacOracle oracle0{meta, decoded, att0};
  const HostCtacOracle oracle1{meta, decoded, att1};
  Outputs ref;
  ref.denom = oracle0.sensitivity();
  ref.numer = oracle0.numerator(proj0, image0);
  ref.forward = oracle0.forward(image0);
  ref.twoBed = {oracle0.numerator(proj0, image0), oracle1.numerator(proj1, image1)};

  const double tol = 1e-5;
  auto expectMatches = [&](const Outputs &got, const char *format) {
    EXPECT_LE(relL2(got.denom, ref.denom), tol) << format << " CTAC sensitivity";
    EXPECT_LE(relL2(got.numer, ref.numer), tol)
        << format << " CTAC single-bed numerator";
    EXPECT_LE(relL2(got.forward, ref.forward), tol)
        << format << " CTAC forward projection";
    EXPECT_LE(relL2(got.twoBed[0], ref.twoBed[0]), tol) << format << " two-bed 0";
    EXPECT_LE(relL2(got.twoBed[1], ref.twoBed[1]), tol) << format << " two-bed 1";
    EXPECT_LE(relL2(got.twoBedResident[0], ref.twoBed[0]), tol)
        << format << " resident two-bed 0";
    EXPECT_LE(relL2(got.twoBedResident[1], ref.twoBed[1]), tol)
        << format << " resident two-bed 1";
    EXPECT_GT(*std::max_element(got.denom.begin(), got.denom.end()), 0.0);
  };
  Outputs gotCsrb;
  runAll(pair.csrb, gotCsrb, true);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  expectMatches(gotCsrb, "CSRB");

  // CSRB CTAC numerator with zero-count bins (~60 %): the per-angle backward
  // may take the zero-row pair scatter on the angle's sub-range. Forced on
  // and forced off must both match the oracle and each other.
  {
    std::vector<double> proj0z = proj0, proj1z = proj1;
    for (size_t i = 0; i < fullRows; ++i) {
      if (i % 5 < 3) proj0z[i] = 0.0;
      if ((i + 2) % 5 < 3) proj1z[i] = 0.0;
    }
    const double *imagePtrs[2] = {image0.data(), image1.data()};
    const double *projPtrs[2] = {proj0z.data(), proj1z.data()};
    const uint16_t *attPtrs[2] = {att0.data(), att1.data()};
    auto runTwoBed = [&](const char *env) {
      spect::io::test::setCsrbScatterBackwardEnv(env);
      std::vector<std::vector<double>> numers(2, std::vector<double>(nvox, 0.0));
      double *numerPtrs[2] = {numers[0].data(), numers[1].data()};
      EXPECT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
          pair.csrb, meta, projPtrs, nullptr, imagePtrs, 2, 1, numerPtrs, false,
          nullptr, attPtrs));
      return numers;
    };
    const auto scatter = runTwoBed("1");
    const auto cusparse = runTwoBed("0");
    spect::io::test::setCsrbScatterBackwardEnv("");
    ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
    const std::vector<std::vector<double>> refZ = {
        oracle0.numerator(proj0z, image0), oracle1.numerator(proj1z, image1)};
    for (int bed = 0; bed < 2; ++bed) {
      EXPECT_LE(relL2(scatter[bed], refZ[bed]), tol)
          << "CSRB CTAC scatter backward, zero bins, bed " << bed;
      EXPECT_LE(relL2(cusparse[bed], refZ[bed]), tol)
          << "CSRB CTAC cuSPARSE backward, zero bins, bed " << bed;
      EXPECT_LE(relL2(scatter[bed], cusparse[bed]), 1e-6)
          << "CSRB CTAC scatter vs cuSPARSE, bed " << bed;
    }
  }
  // CSRQ with the packed backward (attenuated packed transform-add), then the
  // default branch-major backward whose outputs feed the adjoint check.
  ASSERT_TRUE(spect::cuda::setCsrqPackedBackwardCUDA(true));
  Outputs gotPacked;
  runAll(pair.csrq, gotPacked, true);
  ASSERT_TRUE(spect::cuda::setCsrqPackedBackwardCUDA(false));
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  expectMatches(gotPacked, "CSRQ packed");
  Outputs got;
  runAll(pair.csrq, got, true);
  expectMatches(got, "CSRQ");

  // The tables must matter: the same runs without attenuation differ.
  std::vector<double> denomPlain(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::computeCompactChunkSensitivityCUDA(
      pair.csrq, meta, 1, denomPlain.data(), nullptr, nullptr));
  EXPECT_GT(relL2(got.denom, denomPlain), 1e-2)
      << "attenuation must change the sensitivity";
  // Per-bed tables must be distinct in the result too.
  EXPECT_GT(relL2(got.twoBed[1], got.twoBed[0]), 1e-2);

  // Adjoint: forward projection gives H x; the numerator with p = y * (H 1)
  // and image 1 gives H^T y (rows with H 1 = 0 have p = 0 and contribute 0).
  std::vector<double> ones(nvox, 1.0);
  std::vector<double> hOnes(fullRows, 0.0);
  ASSERT_TRUE(spect::cuda::computeCompactChunkForwardProjectionCUDA(
      pair.csrq, meta, ones.data(), 1, hOnes.data(), att0.data()));
  std::vector<double> y(fullRows);
  for (size_t i = 0; i < fullRows; ++i)
    y[i] = 0.3 + 0.7 * std::sin(0.05 * static_cast<double>(i)) *
                     std::sin(0.05 * static_cast<double>(i));
  std::vector<double> pAdj(fullRows);
  for (size_t i = 0; i < fullRows; ++i)
    pAdj[i] = y[i] * hOnes[i];
  std::vector<double> hTy(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      pair.csrq, meta, pAdj.data(), nullptr, ones.data(), 1, hTy.data(), false,
      nullptr, att0.data()));
  double lhs = 0.0;
  for (size_t i = 0; i < fullRows; ++i)
    lhs += got.forward[i] * y[i];
  double rhs = 0.0;
  for (size_t j = 0; j < nvox; ++j)
    rhs += image0[j] * hTy[j];
  EXPECT_GT(lhs, 0.0);
  EXPECT_LE(std::abs(lhs - rhs) / std::abs(lhs), 1e-5) << "adjoint identity";
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  std::filesystem::remove_all(pair.csrqDir.parent_path());
}

// Per-window CTAC energy scaling: a channel with attenPower s must equal the
// unscaled channel run on the pre-powered table att^s (quantized to u16), for
// sensitivity and numerator, single window and as the second window of a
// joint two-window model with per-window exponents. Tolerance covers the u16
// quantization of the pre-powered reference table (factor-level ~1e-4 at the
// smallest factors, averaged over the rows).
TEST_F(CsrqCompactMatrixTest, CtacEnergyExponentMatchesPrePoweredTable) {
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

  std::mt19937 rng(4242u);
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
      std::filesystem::temp_directory_path() / "spect_csrq_ctac_power_test";
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
  const size_t attCount = static_cast<size_t>(meta.fullNumAngles) * nvox;
  // Lu-177: table at 208.4 keV, second window 112.9 keV.
  double power = 0.0;
  ASSERT_TRUE(spect::core::waterMuEnergyRatio(112.9, 208.4, power));
  std::vector<uint16_t> att(attCount), attPowered(attCount);
  std::uniform_int_distribution<int> attDist(9000, 64000);
  for (size_t i = 0; i < attCount; ++i) {
    att[i] = static_cast<uint16_t>(attDist(rng));
    const double a = static_cast<double>(att[i]) / 65535.0;
    attPowered[i] = static_cast<uint16_t>(
        std::clamp(std::rint(std::pow(a, power) * 65535.0), 0.0, 65535.0));
  }
  std::vector<double> image(nvox), proj(fullRows);
  for (size_t i = 0; i < nvox; ++i)
    image[i] = 0.6 + 0.3 * std::sin(0.13 * static_cast<double>(i));
  for (size_t i = 0; i < fullRows; ++i)
    proj[i] = 1.0 + 0.01 * static_cast<double>(i % 53);

  auto sensitivity = [&](const std::vector<spect::cuda::CompactMatrixChannel> &ch,
                         const uint16_t *table, std::vector<double> &out) {
    out.assign(nvox, 0.0);
    ASSERT_TRUE(spect::cuda::computePoissonSensitivityCUDA(ch, meta, 1, out.data(),
                                                            nullptr, table));
  };
  auto numerator = [&](const std::vector<spect::cuda::CompactMatrixChannel> &ch,
                       const uint16_t *table, std::vector<double> &out) {
    out.assign(nvox, 0.0);
    std::vector<const double *> projections(ch.size(), proj.data());
    const double *images[1] = {image.data()};
    double *numers[1] = {out.data()};
    const uint16_t *tables[1] = {table};
    ASSERT_TRUE(spect::cuda::accumulatePoissonBackprojectedRatioCUDA(
        ch, meta, projections.data(), nullptr, images, 1, 1, numers, false,
        nullptr, tables));
  };

  const double tol = 2e-4;
  // References on the pre-powered table with power 1.
  std::vector<spect::cuda::CompactMatrixChannel> plain = {{&chunksA, 1.0, 1.0}};
  std::vector<double> denomRef, numerRef, denomUnscaled;
  sensitivity(plain, attPowered.data(), denomRef);
  numerator(plain, attPowered.data(), numerRef);
  sensitivity(plain, att.data(), denomUnscaled);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());

  std::vector<spect::cuda::CompactMatrixChannel> scaled = {{&chunksA, 1.0, power}};
  std::vector<double> denom, numer;
  sensitivity(scaled, att.data(), denom);
  numerator(scaled, att.data(), numer);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  EXPECT_LE(relL2(denom, denomRef), tol) << "att^power sensitivity";
  EXPECT_LE(relL2(numer, numerRef), tol) << "att^power numerator";
  EXPECT_GT(relL2(denom, denomUnscaled), 1e-2) << "exponent must act";

  // Joint two windows: A at the table energy, B at 112.9 keV, weights w1/w2.
  const double w1 = 0.1036;
  const double w2 = 0.0617;
  std::vector<spect::cuda::CompactMatrixChannel> joint = {{&chunksA, w1, 1.0},
                                                          {&chunksB, w2, power}};
  std::vector<double> denomJoint;
  sensitivity(joint, att.data(), denomJoint);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  std::vector<double> expected(nvox);
  for (size_t i = 0; i < nvox; ++i)
    expected[i] = w1 * denomUnscaled[i] + w2 * denomRef[i];
  EXPECT_LE(relL2(denomJoint, expected), tol) << "joint per-window exponents";
  EXPECT_GT(*std::max_element(denomJoint.begin(), denomJoint.end()), 0.0);
  std::filesystem::remove_all(root);
}

// One resident BSREM-II + RDP iteration on the device (two beds, CSRB
// production format) against the host oracle: host numerator / sensitivity
// through the same projector, RDP gradient through the host volume-ops entry,
// applyBsremUpdate with the loop's curvature surrogate. The resident path
// rounds the per-bed numerator through float before the global accumulate, so
// the tolerance is fp32-level; the BSREM step must also visibly differ from
// the plain MLEM step to prove the penalty and relaxation reached the kernel.
TEST_P(CsrqProfileTest, ResidentBsremIterationMatchesHostOracle) {
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

  applyProfile(meta);
  std::mt19937 rng(11u);
  std::uniform_real_distribution<float> valueDist(0.05f, 1.0f);
  HostCsr src;
  src.rows = meta.totalRows;
  src.cols = meta.totalVoxels;
  src.rowPtr.push_back(0u);
  for (int row = 0; row < src.rows; ++row) {
    uint32_t pos = static_cast<uint32_t>(rng() % 5);
    while (pos + 3u < static_cast<uint32_t>(src.cols)) {
      const uint32_t len = 1u + rng() % 3u;
      for (uint32_t k = 0; k < len; ++k) {
        src.colIdx.push_back(pos + k);
        src.values.push_back(valueDist(rng));
      }
      pos += len + 2u + rng() % 4u;
    }
    src.rowPtr.push_back(static_cast<uint32_t>(src.colIdx.size()));
  }
  const auto pair = makeMatrixPair("spect_bsrem_resident_test", meta, src);
  const auto &chunks = pair.csrb;

  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());
  constexpr int kBeds = 2;
  const int globalNz = kBeds * meta.Nz;
  const size_t globalNvox = nvox * kBeds;
  const int bedZStarts[kBeds] = {0, meta.Nz};

  std::vector<double> image(globalNvox);
  for (size_t i = 0; i < globalNvox; ++i)
    image[i] = 0.6 + 0.4 * std::sin(0.13 * static_cast<double>(i)) +
               0.05 * static_cast<double>(i % 7);
  std::vector<double> proj0(fullRows), proj1(fullRows);
  for (size_t i = 0; i < fullRows; ++i) {
    proj0[i] = 1.0 + 0.02 * static_cast<double>(i % 41);
    proj1[i] = 1.5 + 0.03 * static_cast<double>(i % 29);
  }
  const double *projPtrs[kBeds] = {proj0.data(), proj1.data()};

  // Host oracle.
  std::vector<double> localDenom(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::computeCompactChunkSensitivityCUDA(
      chunks, meta, 1, localDenom.data(), nullptr, nullptr));
  std::vector<double> denom(globalNvox, 0.0);
  for (int bed = 0; bed < kBeds; ++bed)
    spect::app::accumulateBedVolumeToGlobal(denom, localDenom, meta.Nx,
                                            meta.Ny, meta.Nz, bedZStarts[bed],
                                            false);
  std::vector<std::vector<double>> localImages(kBeds), localNumers(kBeds);
  const double *imagePtrs[kBeds];
  double *numerPtrs[kBeds];
  for (int bed = 0; bed < kBeds; ++bed) {
    localImages[bed] = spect::app::extractBedVolumeFromGlobal(
        image, meta.Nx, meta.Ny, meta.Nz, bedZStarts[bed], false);
    localNumers[bed].assign(nvox, 0.0);
    imagePtrs[bed] = localImages[bed].data();
    numerPtrs[bed] = localNumers[bed].data();
  }
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
      chunks, meta, projPtrs, nullptr, imagePtrs, kBeds, 1, numerPtrs));
  std::vector<double> numer(globalNvox, 0.0);
  for (int bed = 0; bed < kBeds; ++bed)
    spect::app::accumulateBedVolumeToGlobal(numer, localNumers[bed], meta.Nx,
                                            meta.Ny, meta.Nz, bedZStarts[bed],
                                            false);

  spect::RegConf reg;
  reg.type = spect::RegType::RDP;
  reg.beta = 0.05;
  reg.delta = 0.01;
  reg.gamma = 2.0;
  reg.skip_outside_mask = false;
  std::vector<double> grad;
  ASSERT_TRUE(spect::cuda::computeRegGradientCUDA(image, meta.Nx, meta.Ny,
                                                  globalNz, reg, nullptr,
                                                  grad));
  const double curvEps = 1e-3;
  std::vector<double> curvature(globalNvox);
  for (size_t i = 0; i < globalNvox; ++i)
    curvature[i] = std::abs(grad[i]) / std::max(image[i], curvEps);

  spect::BsremUpdateParams hostParams;
  hostParams.gradient = &grad;
  hostParams.curvature = &curvature;
  hostParams.betaScaled = reg.beta;
  hostParams.alpha = 0.7;
  hostParams.denomEps = 1e-12;
  hostParams.sMin = 1e-12;
  hostParams.clipRelativeStep = true;
  hostParams.nonneg = true;
  hostParams.threads = 1;
  std::vector<double> hostBsrem = image;
  spect::applyBsremUpdate(hostBsrem, numer, denom, denom, hostParams);
  std::vector<double> hostMlem = image;
  spect::applyMlemUpdate(hostMlem, numer, denom, nullptr, true, 1);
  ASSERT_GT(relL2(hostBsrem, hostMlem), 1e-3)
      << "test setup: BSREM step must differ from MLEM";

  // Device: resident batch input, device numerator, device gradient, device
  // BSREM-II update.
  ASSERT_TRUE(spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
      meta, image.data(), image.size(), globalNz, bedZStarts, kBeds, false,
      nullptr, true));
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
      chunks, meta, projPtrs, nullptr, nullptr, kBeds, 1, nullptr, false,
      nullptr, nullptr, true, true, false));
  const double *dImage = spect::cuda::compactResidentImageDeviceCUDA(globalNvox);
  double *dGrad = spect::cuda::compactResidentGradientDeviceCUDA(globalNvox);
  ASSERT_NE(dImage, nullptr);
  ASSERT_NE(dGrad, nullptr);
  ASSERT_TRUE(spect::cuda::computeRegGradientDeviceCUDA(
      dImage, meta.Nx, meta.Ny, globalNz, reg, nullptr, dGrad));
  spect::cuda::CompactBsremUpdateParams deviceParams;
  deviceParams.alpha = hostParams.alpha;
  deviceParams.betaScaled = reg.beta;
  deviceParams.denomEps = 1e-12;
  deviceParams.sMin = 1e-12;
  deviceParams.clipRelativeStep = true;
  deviceParams.nonneg = true;
  deviceParams.deviceGradient = dGrad;
  deviceParams.useCurvature = true;
  deviceParams.refreshCurvature = true;
  deviceParams.curvEps = curvEps;
  bool allZero = true;
  ASSERT_TRUE(spect::cuda::finishCompactResidentBsremIterationCUDA(
      meta, globalNvox, globalNz, bedZStarts, kBeds, false, denom.data(),
      nullptr, nullptr, deviceParams, true, true, &allZero));
  EXPECT_FALSE(allZero);
  std::vector<double> deviceBsrem(globalNvox, 0.0);
  ASSERT_TRUE(spect::cuda::downloadCompactResidentImageCUDA(
      deviceBsrem.data(), globalNvox));
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());

  EXPECT_LE(relL2(deviceBsrem, hostBsrem), 1e-6)
      << "device BSREM-II vs host oracle";
  EXPECT_GT(relL2(deviceBsrem, hostMlem), 1e-3)
      << "device BSREM-II must differ from MLEM";
  std::filesystem::remove_all(pair.csrqDir.parent_path());
}

TEST(CsrqCompactMatrixFormat, CacheInputsDoNotReadPayloadAndPaddingDoesNotChangeValues) {
  auto meta = makeToySym90ZMeta();
  meta.profile = spect::io::CompactSymmetryProfile::Rot90;
  meta.detNz = meta.compactDetZ;
  HostCsr src;
  src.rows = meta.totalRows; src.cols = meta.totalVoxels;
  src.rowPtr.assign(static_cast<size_t>(src.rows) + 1, 1u);
  src.rowPtr[0] = 0;
  src.colIdx = {0u}; src.values = {0.7f};
  const auto pair = makeMatrixPair("spect_csrq_padding", meta, src);
  const auto &info = pair.csrq.chunks()[0];
  std::ostringstream inputs;
  pair.csrq.appendCacheInputs(inputs, "matrix2_");
  EXPECT_FALSE(inputs.str().empty());
  EXPECT_EQ(pair.csrq.readStats().payloadLoads, 0u);
  const auto original = pair.csrq.loadChunk(0);
  EXPECT_EQ(pair.csrq.readStats().payloadLoads, 1u);
  // Odd nnz leaves two padding bytes; they are not matrix values.
  {
    std::fstream file(info.path, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(48 + 12ll * info.rows + 8 + 2);
    file.put('\x31');
  }
  const auto next = spect::io::openCompactChunkCSRDirectory(pair.csrqDir.string(), meta, 1 << 20);
  const auto loaded = next.loadChunk(0);
  EXPECT_EQ(loaded->rowPtr, original->rowPtr);
  EXPECT_EQ(loaded->rowRunPtr, original->rowRunPtr);
  EXPECT_EQ(loaded->rowMax, original->rowMax);
  EXPECT_EQ(loaded->runs, original->runs);
  EXPECT_EQ(loaded->valuesQ, original->valuesQ);
  std::filesystem::remove_all(pair.csrqDir.parent_path());
}

// Streamed chunks beyond the host page-lock budget are read back per pass
// straight from the file: the spans must equal the validated chunk arrays,
// and a file rewritten after validation must be rejected, not decoded.
TEST(CsrqCompactMatrixFormat, PayloadSpanReadsMatchLoadedChunkAndRejectChangedFile) {
  const auto meta = makeToySym90ZMeta();
  const auto toy = makeToyCompactCSR();
  HostCsr src;
  src.rows = meta.totalRows;
  src.cols = meta.totalVoxels;
  src.rowPtr = toy.rowPtr;
  src.colIdx = toy.colIdx;
  src.values = toy.values;
  const auto pair = makeMatrixPair("spect_csrq_payload_span", meta, src);
  const auto &info = pair.csrq.chunks()[0];
  const auto chunk = pair.csrq.loadChunk(0);
  const auto writeTime = std::filesystem::last_write_time(info.path);
  ASSERT_GT(info.nnz, 3u);
  ASSERT_GT(info.runCount, 2u);
  std::vector<uint16_t> values(info.nnz - 1);
  spect::io::readCsrqPayloadSpan(info, writeTime, false, 1, values.size(), values.data());
  EXPECT_TRUE(std::equal(values.begin(), values.end(), chunk->valuesQ.begin() + 1));
  std::vector<uint32_t> runs(info.runCount - 2);
  spect::io::readCsrqPayloadSpan(info, writeTime, true, 2, runs.size(), runs.data());
  EXPECT_TRUE(std::equal(runs.begin(), runs.end(), chunk->runs.begin() + 2));
  EXPECT_THROW(spect::io::readCsrqPayloadSpan(info, writeTime, true, 1, info.runCount,
                                              runs.data()),
               std::runtime_error);
  std::filesystem::last_write_time(info.path, writeTime + std::chrono::seconds(5));
  EXPECT_THROW(spect::io::readCsrqPayloadSpan(info, writeTime, false, 0, 1, values.data()),
               std::runtime_error);
  std::filesystem::remove_all(pair.csrqDir.parent_path());
}

// Spans of 32 MiB and more are read as concurrent ranges; every element must
// land at its own offset (values and runs, unaligned first element).
TEST(CsrqCompactMatrixFormat, LargePayloadSpansReadInParallelRanges) {
  const auto path = std::filesystem::temp_directory_path() / "spect_csrq_parallel_span.csrq";
  spect::io::CompactChunkCSRInfo info;
  info.path = path;
  info.format = spect::io::CompactChunkFormat::CsrqRunU16;
  info.rows = 3;
  info.nnz = 20'000'001;  // odd: two padding bytes before the runs
  info.runCount = 12'000'000;
  {
    std::ofstream out(path, std::ios::binary);
    std::vector<char> head(48 + 12 * info.rows + 8, 0);
    out.write(head.data(), static_cast<std::streamsize>(head.size()));
    std::vector<uint16_t> values(info.nnz);
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<uint16_t>(i * 7u);
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * 2));
    out.write("\0\0", 2);
    std::vector<uint32_t> runs(info.runCount);
    for (size_t i = 0; i < runs.size(); ++i) runs[i] = static_cast<uint32_t>(i * 2654435761u);
    out.write(reinterpret_cast<const char *>(runs.data()),
              static_cast<std::streamsize>(runs.size() * 4));
  }
  info.fileSizeBytes = std::filesystem::file_size(path);
  ASSERT_EQ(info.fileSizeBytes, 48u + 12u * 3u + 8u + 2u * info.nnz + 2u + 4u * info.runCount);
  const auto writeTime = std::filesystem::last_write_time(path);
  std::vector<uint16_t> values(info.nnz - 3);
  spect::io::readCsrqPayloadSpan(info, writeTime, false, 3, values.size(), values.data());
  for (size_t i = 0; i < values.size(); ++i)
    ASSERT_EQ(values[i], static_cast<uint16_t>((i + 3) * 7u)) << i;
  std::vector<uint32_t> runs(info.runCount - 1);
  spect::io::readCsrqPayloadSpan(info, writeTime, true, 1, runs.size(), runs.data());
  for (size_t i = 0; i < runs.size(); ++i)
    ASSERT_EQ(runs[i], static_cast<uint32_t>((i + 1) * 2654435761u)) << i;
  std::filesystem::remove(path);
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

TEST(CsrqCompactMatrixFormat, RejectsHeaderColumnsThatDisagreeWithMetadata) {
  const auto meta = makeToySym90ZMeta();
  const auto toy = makeToyCompactCSR();
  HostCsr src;
  src.rows = meta.totalRows;
  src.cols = meta.totalVoxels + 1;
  src.rowPtr = toy.rowPtr;
  src.colIdx = toy.colIdx;
  src.values = toy.values;
  const auto root = std::filesystem::temp_directory_path() / "spect_csrq_wrong_cols";
  std::filesystem::create_directories(root);
  writeCsrq(root / "chunk_000.csrq", src, 0, meta.compactNumAngles);
  EXPECT_THROW(spect::io::openCompactChunkCSRDirectory(root.string(), meta, 0),
               std::runtime_error);
  EXPECT_THROW(spect::io::openCompactChunkCSRDirectory(root.string(), meta),
               std::runtime_error);
  std::filesystem::remove_all(root);
}

TEST(CsrqCompactMatrixFormat, StreamSlotUsesPerArrayMaxima) {
  const spect::cuda::CsrqChunkShape a{10, 128, 2, 0};
  const spect::cuda::CsrqChunkShape b{6, 40, 40, 10};
  EXPECT_EQ(a.deviceBytes(), 392u);
  EXPECT_EQ(b.deviceBytes(), 320u);
  spect::cuda::CsrqChunkShape slot;
  slot.includeDimensions(a);
  slot.includeDimensions(b);
  EXPECT_EQ(slot.deviceBytes(), 544u);
  EXPECT_GT(slot.deviceBytes(), std::max(a.deviceBytes(), b.deviceBytes()));
}

} // namespace
} // namespace spect::main
