#include <gtest/gtest.h>

#include "sparse/CompactSymmetricProjector.h"
#include "cuda/CudaConfig.h"
#include "io/CompactChunkCSR.h"
#include "sparse/ProjDomainChunked.h"
#include "tests/common/compact_symmetry_test_helpers.h"

using spect::io::test::denseNumerator;
using spect::io::test::expandCompactToyToFullDense;
using spect::io::test::makeToyCompactCSR;
using spect::io::test::makeToySym90ZMeta;

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>

namespace spect::main {
namespace {

class MultiBedCompactNumeratorTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!spect::cuda::isCudaAvailable()) {
      GTEST_SKIP() << "CUDA is not available";
    }
  }

  static void writeChunkFile(const std::filesystem::path &path, int32_t rows,
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
};

TEST_F(MultiBedCompactNumeratorTest, TwoBedsMatchDenseReference) {
  const auto meta = makeToySym90ZMeta();
  const auto compact = makeToyCompactCSR();
  const auto full = expandCompactToyToFullDense(meta, compact);

  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_multibed_compact_num_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  writeChunkFile(dir / "chunk_000.csrb", meta.totalRows, 0,
                 meta.compactNumAngles, compact.rowPtr, compact.colIdx,
                 compact.values);

  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunks =
      spect::io::openCompactChunkCSRDirectory(dir.string(), dirMeta, 1024 * 1024);
  ASSERT_EQ(chunks.chunks().size(), 1u);

  constexpr int bedCount = 2;
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());

  std::vector<double> imageBed0 = {1.0, 1.3, 1.7, 2.1, 0.8, 1.2, 1.6, 2.0};
  std::vector<double> imageBed1 = {0.5, 0.8, 1.2, 1.6, 1.0, 1.4, 1.8, 2.2};
  std::vector<double> projBed0(fullRows, 1.0);
  std::vector<double> projBed1(fullRows, 2.0);

  // Make projections non-uniform so the ratio step is non-trivial.
  for (size_t i = 0; i < fullRows; ++i) {
    projBed0[i] = 1.0 + 0.01 * static_cast<double>(i);
    projBed1[i] = 2.0 + 0.02 * static_cast<double>(i);
  }

  std::vector<std::vector<double>> localImages = {imageBed0, imageBed1};
  std::vector<std::vector<double>> localNumers(
      bedCount, std::vector<double>(nvox, 0.0));

  const double *imagePtrs[bedCount] = {localImages[0].data(),
                                        localImages[1].data()};
  const double *projPtrs[bedCount] = {projBed0.data(), projBed1.data()};
  double *numerPtrs[bedCount] = {localNumers[0].data(),
                                 localNumers[1].data()};
  spect::cuda::resetCudaChunkStageTimingCUDA();

  const bool ok = spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
      chunks, meta, projPtrs, nullptr, imagePtrs, bedCount, 1, numerPtrs, false,
      nullptr, nullptr);
  ASSERT_TRUE(ok);
  const auto cudaTiming = spect::cuda::getCudaChunkStageTimingSnapshotCUDA();
  EXPECT_EQ(cudaTiming.compactForwardOutputD2HCalls, 0u);
  EXPECT_EQ(cudaTiming.compactBackwardOutputD2HCalls,
            static_cast<size_t>(bedCount));
  EXPECT_EQ(cudaTiming.compactOutputD2HCalls,
            static_cast<size_t>(bedCount));
  EXPECT_EQ(cudaTiming.compactForwardOutputD2HBytes, 0u);
  EXPECT_EQ(cudaTiming.compactBackwardOutputD2HBytes,
            static_cast<uint64_t>(bedCount) * nvox * sizeof(double));

  // Compute per-bed reference numerators from the explicit full dense matrix.
  const auto refNumer0 = denseNumerator(full, projBed0, imageBed0,
                                        static_cast<int>(fullRows),
                                        meta.totalVoxels);
  const auto refNumer1 = denseNumerator(full, projBed1, imageBed1,
                                        static_cast<int>(fullRows),
                                        meta.totalVoxels);

  auto expectNear = [](const std::vector<double> &a,
                       const std::vector<double> &b, double tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      EXPECT_NEAR(a[i], b[i], tol)
          << "mismatch at index " << i << " (" << a[i] << " vs " << b[i]
          << ")";
    }
  };

  expectNear(localNumers[0], refNumer0, 1e-4);
  expectNear(localNumers[1], refNumer1, 1e-4);

  std::filesystem::remove_all(dir);
}

TEST_F(MultiBedCompactNumeratorTest, ProjectionBufferMutationRefreshesCache) {
  const auto meta = makeToySym90ZMeta();
  const auto compact = makeToyCompactCSR();
  const auto full = expandCompactToyToFullDense(meta, compact);
  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_compact_low_forward_ratio_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  writeChunkFile(dir / "chunk_000.csrb", meta.totalRows, 0,
                 meta.compactNumAngles, compact.rowPtr, compact.colIdx,
                 compact.values);

  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunks = spect::io::openCompactChunkCSRDirectory(
      dir.string(), dirMeta, 1024 * 1024);
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());
  const std::vector<double> image = {1.0, 1.3, 1.7, 2.1,
                                     0.8, 1.2, 1.6, 2.0};
  std::vector<double> projection(fullRows, 5.0);
  std::vector<double> numer(nvox, 0.0);

  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      chunks, meta, projection.data(), nullptr, image.data(), 1,
      numer.data()));
  for (size_t i = 0; i < fullRows; ++i)
    projection[i] = 1.0 + 0.01 * static_cast<double>(i);
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      chunks, meta, projection.data(), nullptr, image.data(), 1,
      numer.data()));
  const auto expected = denseNumerator(full, projection, image,
                                       static_cast<int>(fullRows),
                                       meta.totalVoxels);
  ASSERT_EQ(numer.size(), expected.size());
  for (size_t i = 0; i < numer.size(); ++i)
    EXPECT_NEAR(numer[i], expected[i], 1e-4) << "mismatch at index " << i;

  std::filesystem::remove_all(dir);
}

TEST_F(MultiBedCompactNumeratorTest, TwoBedsWithCtacMatchSingleBedRuns) {
  const auto meta = makeToySym90ZMeta();
  const auto compact = makeToyCompactCSR();

  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_multibed_compact_ctac_num_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  writeChunkFile(dir / "chunk_000.csrb", meta.totalRows, 0,
                 meta.compactNumAngles, compact.rowPtr, compact.colIdx,
                 compact.values);

  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunks =
      spect::io::openCompactChunkCSRDirectory(dir.string(), dirMeta, 1024 * 1024);
  ASSERT_EQ(chunks.chunks().size(), 1u);

  constexpr int bedCount = 2;
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());
  const size_t attCount =
      static_cast<size_t>(meta.fullNumAngles) * nvox;

  std::vector<double> imageBed0 = {1.0, 1.3, 1.7, 2.1, 0.8, 1.2, 1.6, 2.0};
  std::vector<double> imageBed1 = {0.5, 0.8, 1.2, 1.6, 1.0, 1.4, 1.8, 2.2};
  std::vector<double> projBed0(fullRows, 1.0);
  std::vector<double> projBed1(fullRows, 2.0);
  for (size_t i = 0; i < fullRows; ++i) {
    projBed0[i] = 1.0 + 0.01 * static_cast<double>(i);
    projBed1[i] = 2.0 + 0.02 * static_cast<double>(i);
  }

  // Build two different full-angle attenuation tables so each bed exercises
  // a distinct per-bed cache slot.
  const auto makeAtt = [&attCount](float scale) {
    std::vector<uint16_t> att(attCount);
    const uint16_t value = static_cast<uint16_t>(
        std::clamp(scale * 65535.0f, 0.0f, 65535.0f));
    std::fill(att.begin(), att.end(), value);
    return att;
  };
  const std::vector<uint16_t> attBed0 = makeAtt(1.0f);
  const std::vector<uint16_t> attBed1 = makeAtt(0.5f);

  std::vector<std::vector<double>> localImages = {imageBed0, imageBed1};
  std::vector<std::vector<double>> multiNumers(
      bedCount, std::vector<double>(nvox, 0.0));

  const double *imagePtrs[bedCount] = {localImages[0].data(),
                                        localImages[1].data()};
  const double *projPtrs[bedCount] = {projBed0.data(), projBed1.data()};
  const uint16_t *attPtrs[bedCount] = {attBed0.data(), attBed1.data()};
  double *multiNumerPtrs[bedCount] = {multiNumers[0].data(),
                                      multiNumers[1].data()};

  bool ok = spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
      chunks, meta, projPtrs, nullptr, imagePtrs, bedCount, 1, multiNumerPtrs,
      false, nullptr, attPtrs);
  ASSERT_TRUE(ok);

  // Compute per-bed reference using the single-bed CTAC path.
  std::vector<double> refNumer0(nvox, 0.0);
  std::vector<double> refNumer1(nvox, 0.0);
  ok = spect::cuda::runCompactChunkMlemNumerCUDA(
      chunks, meta, projBed0.data(), nullptr, imageBed0.data(), 1,
      refNumer0.data(), false, nullptr, attBed0.data());
  ASSERT_TRUE(ok);
  ok = spect::cuda::runCompactChunkMlemNumerCUDA(
      chunks, meta, projBed1.data(), nullptr, imageBed1.data(), 1,
      refNumer1.data(), false, nullptr, attBed1.data());
  ASSERT_TRUE(ok);

  auto expectNear = [](const std::vector<double> &a,
                       const std::vector<double> &b, double tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      EXPECT_NEAR(a[i], b[i], tol)
          << "mismatch at index " << i << " (" << a[i] << " vs " << b[i]
          << ")";
    }
  };

  expectNear(multiNumers[0], refNumer0, 1e-4);
  expectNear(multiNumers[1], refNumer1, 1e-4);

  // Also make sure the two beds actually produced different results, otherwise
  // the test would pass even if the CTAC table was ignored or shared.
  double diff = 0.0;
  for (size_t i = 0; i < nvox; ++i) {
    diff = std::max(diff, std::abs(multiNumers[0][i] - multiNumers[1][i]));
  }
  EXPECT_GT(diff, 1e-6) << "beds produced identical numerators unexpectedly";

  std::filesystem::remove_all(dir);
}

} // namespace
} // namespace spect::main
