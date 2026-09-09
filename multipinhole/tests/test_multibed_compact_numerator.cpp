#include <gtest/gtest.h>

#include "recon/BsremUpdate.h"
#include "sparse/CompactSymmetricProjector.h"
#include "sparse/SparseOpsCUDA.h"
#include "cuda/CudaConfig.h"
#include "cuda/CudaLaunchConfig.h"
#include "filters/cuda/VolumeOpsCUDA.h"
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
  // Independent detector-Z anchor: in this 2x2 detector, front and back
  // ix=0 rays are one packed pair. Cover both zero, either zero, and dense;
  // the ix=1 rays stay positive. The expanded dense oracle retains all rows.
  for (int angle = 0; angle < meta.fullNumAngles; ++angle) {
    const size_t front = static_cast<size_t>(angle) * meta.fullRaysPerAngle();
    const size_t back = front + meta.detNx;
    if (angle % 4 < 2) {
      projBed0[front] = 0.0;
      projBed1[back] = 0.0;
    }
    if (angle % 4 == 0 || angle % 4 == 2) {
      projBed0[back] = 0.0;
      projBed1[front] = 0.0;
    }
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

  std::vector<double> background0(fullRows, 0.25);
  std::vector<double> background1(fullRows, 0.5);
  const double *backgroundPtrs[bedCount] = {background0.data(),
                                             background1.data()};
  std::vector<std::vector<double>> backgroundNumers(
      bedCount, std::vector<double>(nvox, 0.0));
  double *backgroundNumerPtrs[bedCount] = {backgroundNumers[0].data(),
                                           backgroundNumers[1].data()};
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
      chunks, meta, projPtrs, backgroundPtrs, imagePtrs, bedCount, 1,
      backgroundNumerPtrs));
  std::vector<double> backgroundRef0(nvox, 0.0);
  std::vector<double> backgroundRef1(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      chunks, meta, projBed0.data(), background0.data(), imageBed0.data(), 1,
      backgroundRef0.data()));
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      chunks, meta, projBed1.data(), background1.data(), imageBed1.data(), 1,
      backgroundRef1.data()));
  expectNear(backgroundNumers[0], backgroundRef0, 1e-4);
  expectNear(backgroundNumers[1], backgroundRef1, 1e-4);

  std::vector<double> globalImage(2 * nvox, 0.0);
  std::copy(imageBed0.begin(), imageBed0.end(), globalImage.begin());
  std::copy(imageBed1.begin(), imageBed1.end(),
            globalImage.begin() + static_cast<ptrdiff_t>(nvox));
  const int bedZStarts[bedCount] = {0, meta.Nz};
  ASSERT_TRUE(spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
      meta, globalImage.data(), globalImage.size(), 2 * meta.Nz, bedZStarts,
      bedCount, false));
  std::vector<std::vector<double>> residentNumers(
      bedCount, std::vector<double>(nvox, 0.0));
  double *residentNumerPtrs[bedCount] = {residentNumers[0].data(),
                                         residentNumers[1].data()};
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
      chunks, meta, projPtrs, nullptr, nullptr, bedCount, 1,
      residentNumerPtrs, false, nullptr, nullptr, true, true));
  expectNear(residentNumers[0], refNumer0, 1e-4);
  expectNear(residentNumers[1], refNumer1, 1e-4);

  spect::filters::PSFKernels castorPsf;
  castorPsf.kx = {0.25, 0.5, 0.25};
  castorPsf.ky = castorPsf.kx;
  castorPsf.kz = castorPsf.kx;
  std::vector<std::vector<double>> blurredImages = {imageBed0, imageBed1};
  for (auto &blurred : blurredImages) {
    ASSERT_TRUE(spect::cuda::applyCastorConv3DInplaceCUDA(
        blurred, meta.Nx, meta.Ny, meta.Nz, castorPsf.kx, castorPsf.ky,
        castorPsf.kz));
  }
  const double *blurredPtrs[bedCount] = {blurredImages[0].data(),
                                         blurredImages[1].data()};
  std::vector<std::vector<double>> hostPsfNumers(
      bedCount, std::vector<double>(nvox, 0.0));
  double *hostPsfNumerPtrs[bedCount] = {hostPsfNumers[0].data(),
                                        hostPsfNumers[1].data()};
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
      chunks, meta, projPtrs, nullptr, blurredPtrs, bedCount, 1,
      hostPsfNumerPtrs));
  ASSERT_TRUE(spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
      meta, globalImage.data(), globalImage.size(), 2 * meta.Nz, bedZStarts,
      bedCount, false, &castorPsf));
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
      chunks, meta, projPtrs, nullptr, nullptr, bedCount, 1,
      residentNumerPtrs, false, nullptr, nullptr, true, true));
  expectNear(residentNumers[0], hostPsfNumers[0], 1e-4);
  expectNear(residentNumers[1], hostPsfNumers[1], 1e-4);

  std::vector<double> expectedUpdated = globalImage;
  for (int bed = 0; bed < bedCount; ++bed) {
    auto backwardPsf = hostPsfNumers[static_cast<size_t>(bed)];
    ASSERT_TRUE(spect::cuda::applyCastorConv3DInplaceCUDA(
        backwardPsf, meta.Nx, meta.Ny, meta.Nz, castorPsf.kx, castorPsf.ky,
        castorPsf.kz));
    const size_t offset = static_cast<size_t>(bed) * nvox;
    for (size_t i = 0; i < nvox; ++i)
      expectedUpdated[offset + i] *= backwardPsf[i];
  }
  std::vector<double> globalDenom(globalImage.size(), 1.0);
  bool imageAllZero = true;
  ASSERT_TRUE(spect::cuda::finishCompactResidentMlemIterationCUDA(
      meta, globalImage.size(), 2 * meta.Nz, bedZStarts, bedCount, false,
      globalDenom.data(), nullptr, &castorPsf, true, true, &imageAllZero));
  EXPECT_FALSE(imageAllZero);
  std::vector<double> residentUpdated(globalImage.size(), 0.0);
  ASSERT_TRUE(spect::cuda::downloadCompactResidentImageCUDA(
      residentUpdated.data(), residentUpdated.size()));
  expectNear(residentUpdated, expectedUpdated, 1e-4);
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());

  std::filesystem::remove_all(dir);
}

TEST_F(MultiBedCompactNumeratorTest,
       LowHostCacheStillRunsGpuResidentWarmup) {
  const auto meta = makeToySym90ZMeta();
  const auto compact = makeToyCompactCSR();
  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_low_host_cache_resident_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  writeChunkFile(dir / "chunk_000.csrb", meta.totalRows, 0,
                 meta.compactNumAngles, compact.rowPtr, compact.colIdx,
                 compact.values);

  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunks =
      spect::io::openCompactChunkCSRDirectory(dir.string(), dirMeta, 1);

  testing::internal::CaptureStderr();
  const bool prepared = spect::cuda::prepareCompactChunkExecutionCUDA(chunks);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(prepared);
  EXPECT_NE(log.find("host cache 0 MiB is below matrix size"),
            std::string::npos);
  EXPECT_NE(log.find("estimated persistent prefix 0 / 1"), std::string::npos);
  EXPECT_NE(log.find("GPU resident warmup remains enabled"), std::string::npos);
  EXPECT_EQ(log.find("GPU resident matrix warmup was unavailable"),
            std::string::npos);

  EXPECT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  std::filesystem::remove_all(dir);
}

TEST_F(MultiBedCompactNumeratorTest, FullHostCachePinsFollowAllocationOwnership) {
  const auto meta = makeToySym90ZMeta();
  const auto compact = makeToyCompactCSR();
  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_compact_host_pin_lifetime_test";
  std::filesystem::create_directories(dir);
  writeChunkFile(dir / "chunk_000.csrb", meta.totalRows, 0,
                 meta.compactNumAngles, compact.rowPtr, compact.colIdx,
                 compact.values);
  auto dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();

  for (int repeat = 0; repeat < 16; ++repeat) {
    auto chunks = spect::io::openCompactChunkCSRDirectory(
        dir.string(), dirMeta, 1024 * 1024);
    auto allocation = chunks.loadChunk(0);
    std::weak_ptr<const spect::io::CompactChunkCSRData> weak = allocation;
    ASSERT_TRUE(spect::cuda::prepareCompactChunkExecutionCUDA(chunks));
    auto lease = chunks.acceleratorLease();
    chunks = {};
    // A lease may outlive its directory. It must retain the registered memory.
    allocation.reset();
    ASSERT_FALSE(weak.expired());
    allocation = weak.lock();
    for (const void *ptr : {static_cast<const void *>(allocation->colIdxData()),
                            static_cast<const void *>(allocation->valuesData())}) {
      cudaPointerAttributes attributes{};
      ASSERT_EQ(cudaPointerGetAttributes(&attributes, ptr), cudaSuccess);
      EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
    }
    ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
    lease.reset();
    for (const void *ptr : {static_cast<const void *>(allocation->colIdxData()),
                            static_cast<const void *>(allocation->valuesData())}) {
      cudaPointerAttributes attributes{};
      ASSERT_EQ(cudaPointerGetAttributes(&attributes, ptr), cudaSuccess);
      EXPECT_EQ(attributes.type, cudaMemoryTypeUnregistered);
    }
    allocation.reset();
    EXPECT_TRUE(weak.expired());
  }
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

  spect::cuda::resetCudaChunkStageTimingCUDA();
  ASSERT_TRUE(spect::cuda::refreshCompactProjectionCUDA(
      projection.data(), projection.size()));
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      chunks, meta, projection.data(), nullptr, image.data(), 1,
      numer.data(), false, nullptr, nullptr, false));
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      chunks, meta, projection.data(), nullptr, image.data(), 1,
      numer.data(), false, nullptr, nullptr, false));
  EXPECT_EQ(spect::cuda::getCudaChunkStageTimingSnapshotCUDA()
                .compactProjectionH2DCalls,
            1u);
  for (size_t i = 0; i < fullRows; ++i)
    projection[i] = 1.0 + 0.01 * static_cast<double>(i);
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerCUDA(
      chunks, meta, projection.data(), nullptr, image.data(), 1,
      numer.data()));
  EXPECT_EQ(spect::cuda::getCudaChunkStageTimingSnapshotCUDA()
                .compactProjectionH2DCalls,
            2u);
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

TEST_F(MultiBedCompactNumeratorTest, ToyAdjointAndSensitivity) {
  const auto meta = makeToySym90ZMeta();
  const auto compact = makeToyCompactCSR();
  const auto full = expandCompactToyToFullDense(meta, compact);
  const int rows = meta.fullRows();
  const int cols = meta.totalVoxels;

  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_compact_adjoint_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  writeChunkFile(dir / "chunk_000.csrb", meta.totalRows, 0,
                 meta.compactNumAngles, compact.rowPtr, compact.colIdx,
                 compact.values);
  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunks =
      spect::io::openCompactChunkCSRDirectory(dir.string(), dirMeta, 1024 * 1024);

  std::vector<double> x(static_cast<size_t>(cols));
  std::vector<double> q(static_cast<size_t>(rows));
  for (int i = 0; i < cols; ++i)
    x[static_cast<size_t>(i)] = 0.4 + 0.05 * static_cast<double>(i);
  for (int i = 0; i < rows; ++i)
    q[static_cast<size_t>(i)] = 0.3 + 0.02 * static_cast<double>(i);

  std::vector<double> Ax(static_cast<size_t>(rows), 0.0);
  ASSERT_TRUE(spect::cuda::computeCompactChunkForwardProjectionCUDA(
      chunks, meta, x.data(), 1, Ax.data(), nullptr));

  double L = 0.0;
  double q2 = 0.0;
  for (int i = 0; i < rows; ++i) {
    L += Ax[static_cast<size_t>(i)] * q[static_cast<size_t>(i)];
    q2 += q[static_cast<size_t>(i)] * q[static_cast<size_t>(i)];
  }

  std::vector<double> y(static_cast<size_t>(rows));
  for (int i = 0; i < rows; ++i)
    y[static_cast<size_t>(i)] =
        q[static_cast<size_t>(i)] * std::max(Ax[static_cast<size_t>(i)], 1e-12);

  std::vector<double> Atq(static_cast<size_t>(cols), 0.0);
  const double *projPtr = y.data();
  const double *xPtr = x.data();
  double *nPtr = Atq.data();
  ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
      chunks, meta, &projPtr, nullptr, &xPtr, 1, 1, &nPtr, false, nullptr,
      nullptr));

  double R = 0.0;
  double x2 = 0.0;
  for (int i = 0; i < cols; ++i) {
    R += x[static_cast<size_t>(i)] * Atq[static_cast<size_t>(i)];
    x2 += x[static_cast<size_t>(i)] * x[static_cast<size_t>(i)];
  }
  const double eAdj = std::abs(L - R) / std::max({std::abs(L), std::abs(R), 1.0});
  EXPECT_LE(eAdj, 1e-4) << "L=" << L << " R=" << R;

  std::vector<double> sCuda(static_cast<size_t>(cols), 0.0);
  ASSERT_TRUE(spect::cuda::computeCompactChunkSensitivityCUDA(
      chunks, meta, 1, sCuda.data(), nullptr, nullptr));
  const auto sRef = spect::io::test::compactToyDenom(meta, compact);
  double ns = 0.0;
  double ds = 0.0;
  for (int i = 0; i < cols; ++i) {
    const double d = sCuda[static_cast<size_t>(i)] - sRef[static_cast<size_t>(i)];
    ns += d * d;
    ds += sRef[static_cast<size_t>(i)] * sRef[static_cast<size_t>(i)];
  }
  EXPECT_LE(std::sqrt(ns) / (std::sqrt(ds) + 1e-30), 1e-5);

  (void)q2;
  (void)x2;
  (void)full;
  std::filesystem::remove_all(dir);
}

TEST_F(MultiBedCompactNumeratorTest, EquivalenceModeMatchesHostMlem) {
  const auto meta = makeToySym90ZMeta();
  const auto compact = makeToyCompactCSR();
  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_compact_bsrem_equiv_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  writeChunkFile(dir / "chunk_000.csrb", meta.totalRows, 0,
                 meta.compactNumAngles, compact.rowPtr, compact.colIdx,
                 compact.values);
  spect::io::CompactSymmetryMetadata dirMeta = meta;
  dirMeta.totalRows = meta.compactRows();
  const auto chunks =
      spect::io::openCompactChunkCSRDirectory(dir.string(), dirMeta, 1024 * 1024);

  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());
  std::vector<double> xMlem(nvox, 1.0);
  std::vector<double> xBsrem(nvox, 1.0);
  std::vector<double> y(fullRows);
  for (size_t i = 0; i < fullRows; ++i)
    y[i] = 1.1 + 0.03 * static_cast<double>(i);
  std::vector<double> s(nvox, 0.0);
  ASSERT_TRUE(spect::cuda::computeCompactChunkSensitivityCUDA(
      chunks, meta, 1, s.data(), nullptr, nullptr));

  spect::BsremUpdateParams p;
  p.mlemEquivalenceMode = true;
  p.threads = 1;
  const double *yPtr = y.data();
  for (int it = 0; it < 5; ++it) {
    std::vector<double> nM(nvox, 0.0);
    std::vector<double> nB(nvox, 0.0);
    const double *xM = xMlem.data();
    const double *xB = xBsrem.data();
    double *nMp = nM.data();
    double *nBp = nB.data();
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
        chunks, meta, &yPtr, nullptr, &xM, 1, 1, &nMp, false, nullptr,
        nullptr));
    ASSERT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
        chunks, meta, &yPtr, nullptr, &xB, 1, 1, &nBp, false, nullptr,
        nullptr));
    spect::applyMlemUpdate(xMlem, nM, s, nullptr, true, 1);
    spect::applyBsremUpdate(xBsrem, nB, s, s, p);
  }
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < nvox; ++i) {
    const double d = xBsrem[i] - xMlem[i];
    num += d * d;
    den += xMlem[i] * xMlem[i];
  }
  EXPECT_LT(std::sqrt(num) / (std::sqrt(den) + 1e-30), 1e-4);
  std::filesystem::remove_all(dir);
}

TEST(CompactPackedForwardK, L2BudgetSelectsSampling4And220As2) {
  const uint64_t l2 = 48ull * 1024ull * 1024ull;
  const int samplingCols = 108 * 108 * 160;
  const int cols220 = 120 * 120 * 220;
  EXPECT_EQ(spect::cuda::compactPackedForwardKMem(samplingCols, 40, l2), 4);
  EXPECT_EQ(spect::cuda::compactPackedForwardKMem(cols220, 40, l2), 2);
  EXPECT_EQ(spect::cuda::compactPackedForwardTile(4, true), 4);
  EXPECT_EQ(spect::cuda::compactPackedForwardTile(4, false), 2);
  EXPECT_EQ(spect::cuda::compactPackedForwardTile(2, true), 2);
  EXPECT_EQ(spect::cuda::compactPackedForwardTile(1, true), 1);
}

TEST(CompactScatterBackward, ZeroFractionThresholdSelectsPath) {
  EXPECT_FALSE(spect::cuda::compactScatterBackwardPreferred(0.15, 40));
  EXPECT_FALSE(spect::cuda::compactScatterBackwardPreferred(0.39, 40));
  EXPECT_TRUE(spect::cuda::compactScatterBackwardPreferred(0.40, 40));
  EXPECT_TRUE(spect::cuda::compactScatterBackwardPreferred(0.85, 16));
  EXPECT_FALSE(spect::cuda::compactScatterBackwardPreferred(0.85, 1));
}

// The zero-row scatter transpose and the per-RHS cuSPARSE transpose must give
// the same numerator on a projection where most bins are zero (the case the
// scatter path is chosen for); both are checked against the dense oracle.
TEST_F(MultiBedCompactNumeratorTest, ScatterBackwardMatchesCusparseAndDense) {
  const auto meta = makeToySym90ZMeta();
  const auto compact = makeToyCompactCSR();
  const auto full = expandCompactToyToFullDense(meta, compact);

  const auto dir = std::filesystem::temp_directory_path() /
                   "spect_scatter_backward_test";
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
  // ~70 % zero bins with a bed-dependent pattern so branch pairs mix
  // both-zero, one-zero and dense rows.
  std::vector<double> projBed0(fullRows, 0.0);
  std::vector<double> projBed1(fullRows, 0.0);
  for (size_t i = 0; i < fullRows; ++i) {
    if (i % 10 < 3) {
      projBed0[i] = 1.0 + 0.01 * static_cast<double>(i);
    }
    if ((i + 4) % 10 < 3) {
      projBed1[i] = 2.0 + 0.02 * static_cast<double>(i);
    }
  }
  const double *imagePtrs[bedCount] = {imageBed0.data(), imageBed1.data()};
  const double *projPtrs[bedCount] = {projBed0.data(), projBed1.data()};

  auto runNumerator = [&](const char *env) {
    spect::io::test::setCsrbScatterBackwardEnv(env);
    std::vector<std::vector<double>> numers(bedCount,
                                            std::vector<double>(nvox, 0.0));
    double *numerPtrs[2] = {numers[0].data(), numers[1].data()};
    EXPECT_TRUE(spect::cuda::runCompactChunkMlemNumerMultiBedCUDA(
        chunks, meta, projPtrs, nullptr, imagePtrs, bedCount, 1, numerPtrs,
        false, nullptr, nullptr));
    return numers;
  };
  const auto scatter = runNumerator("1");
  const auto cusparse = runNumerator("0");
  spect::io::test::setCsrbScatterBackwardEnv("");

  const auto ref0 = denseNumerator(full, projBed0, imageBed0,
                                   static_cast<int>(fullRows), meta.totalVoxels);
  const auto ref1 = denseNumerator(full, projBed1, imageBed1,
                                   static_cast<int>(fullRows), meta.totalVoxels);
  double refMax = 0.0;
  for (size_t i = 0; i < nvox; ++i) {
    refMax = std::max({refMax, std::abs(ref0[i]), std::abs(ref1[i])});
  }
  ASSERT_GT(refMax, 0.0);
  for (size_t i = 0; i < nvox; ++i) {
    EXPECT_NEAR(scatter[0][i], ref0[i], 1e-4) << "bed0 voxel " << i;
    EXPECT_NEAR(scatter[1][i], ref1[i], 1e-4) << "bed1 voxel " << i;
    EXPECT_NEAR(scatter[0][i], cusparse[0][i], 1e-6 * refMax) << "bed0 voxel "
                                                              << i;
    EXPECT_NEAR(scatter[1][i], cusparse[1][i], 1e-6 * refMax) << "bed1 voxel "
                                                              << i;
  }
  ASSERT_TRUE(spect::cuda::releaseCompactExecutionCUDA());
  std::filesystem::remove_all(dir);
}

} // namespace
} // namespace spect::main
