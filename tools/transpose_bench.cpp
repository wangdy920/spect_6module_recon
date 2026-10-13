// Phase B.2 decisive microbenchmark: on one real compact chunk, compare the
// production transpose path CSR(A) op=T against true CSR(A^T) op=N variants
// (per-RHS SpMV and grouped SpMM). Read-only with respect to the matrix; no
// production code path is touched. Evidence consumer:
// reports/2026-08-17-phase-b-transpose-spmm.md (B.2 section).
//
// Usage:
//   spect_transpose_bench <matrixDir> [chunkIndex=0] [batch=40] [reps=5]
//                         [zeroFrac=0] [mode=full|scatter]
//
// 2026-09-07 `scatter` mode (reports/2026-09-04-final-design-poisson-data-term.md
// §7.9): the production numerator ratio is exactly 0 for zero-count bins
// (62 % of K=2 pairs on the five-bed Lu-177 input), so a transpose that
// skips all-zero rows never reads their matrix bytes. zeroFrac draws that
// pattern per (row, RHS); the mode compares per-RHS cuSPARSE CSR_ALG2 with
// the pair / quad scatter kernels of the production fatbin, with and without
// the skip, and needs no CSR(A^T).

#include "io/CompactChunkCSR.h"
#include "io/CompactSymmetryMetadata.h"
#include "CudaProjKernelLauncher.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

#define BENCH_CUDA_CHECK(expr)                                                \
  do {                                                                        \
    const cudaError_t benchErr = (expr);                                      \
    if (benchErr != cudaSuccess) {                                            \
      std::fprintf(stderr, "[FATAL] CUDA error %s at %s:%d\n",                \
                   cudaGetErrorString(benchErr), __FILE__, __LINE__);         \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define BENCH_CUSPARSE_CHECK(expr)                                            \
  do {                                                                        \
    const cusparseStatus_t benchStatus = (expr);                              \
    if (benchStatus != CUSPARSE_STATUS_SUCCESS) {                             \
      std::fprintf(stderr, "[FATAL] cuSPARSE error %s at %s:%d\n",            \
                   cusparseGetErrorString(benchStatus), __FILE__, __LINE__);  \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

double nowSeconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct DeviceMem {
  void *ptr = nullptr;
  ~DeviceMem() {
    if (ptr != nullptr)
      cudaFree(ptr);
  }
  void alloc(size_t bytes) {
    if (ptr != nullptr) {
      BENCH_CUDA_CHECK(cudaFree(ptr));
      ptr = nullptr;
    }
    BENCH_CUDA_CHECK(cudaMalloc(&ptr, bytes));
  }
};

struct BenchResult {
  std::string name;
  double msAvg = 0.0;
  double msMin = 0.0;
  double relL2 = 0.0;
};

double relL2HostVsDevice(const std::vector<float> &ref, const float *dOut,
                         size_t count, std::vector<float> &scratch) {
  scratch.resize(count);
  BENCH_CUDA_CHECK(cudaMemcpy(scratch.data(), dOut, count * sizeof(float),
                              cudaMemcpyDeviceToHost));
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double d =
        static_cast<double>(scratch[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: spect_transpose_bench <matrixDir> [chunkIndex=0] "
                 "[batch=40] [reps=5]\n");
    return 2;
  }
  const std::string matrixDir = argv[1];
  const size_t chunkIndex = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 0;
  const int batch = argc > 3 ? std::atoi(argv[3]) : 40;
  const int reps = argc > 4 ? std::atoi(argv[4]) : 5;
  const double zeroFrac = argc > 5 ? std::atof(argv[5]) : 0.0;
  const std::string mode = argc > 6 ? argv[6] : "full";
  const bool scatterMode = mode == "scatter";
  if (batch <= 0 || reps <= 0 || zeroFrac < 0.0 || zeroFrac >= 1.0 ||
      (mode != "full" && mode != "scatter")) {
    std::fprintf(stderr,
                 "batch and reps must be positive, zeroFrac in [0, 1), mode "
                 "full|scatter\n");
    return 2;
  }

  const auto meta =
      spect::io::loadCompactSymmetryMetadata(matrixDir + "/metadata.bin");
  // Tiny host budget: the bench manages its own copies; do not let the
  // directory cache retain the chunk twice.
  const auto directory =
      spect::io::openCompactChunkCSRDirectory(matrixDir, meta, 1ull << 20);
  if (chunkIndex >= directory.chunks().size()) {
    std::fprintf(stderr, "chunk index %zu out of range (%zu chunks)\n",
                 chunkIndex, directory.chunks().size());
    return 2;
  }

  std::printf("matrix=%s chunk=%zu batch=%d reps=%d zeroFrac=%.3f mode=%s\n",
              matrixDir.c_str(), chunkIndex, batch, reps, zeroFrac,
              mode.c_str());
  const double tLoad0 = nowSeconds();
  const auto chunk = directory.loadChunk(chunkIndex);
  const double tLoad1 = nowSeconds();
  const int rowsA = chunk->info.rows;
  const int colsA = chunk->cols;
  const uint64_t nnz = chunk->info.nnz;
  std::printf("rows=%d cols=%d nnz=%llu load_s=%.2f\n", rowsA, colsA,
              static_cast<unsigned long long>(nnz), tLoad1 - tLoad0);
  if (nnz >= (1ull << 31)) {
    std::fprintf(stderr, "chunk nnz exceeds int32; bench assumes 32-bit\n");
    return 2;
  }
  const uint32_t *hRowPtr = chunk->rowPtr.data();
  const uint32_t *hColIdx = chunk->colIdx.data();
  const float *hValues = chunk->values.data();

  // ---- Host transpose: CSR(A) -> CSR(A^T) (counting sort by column) ----
  // Only the full mode needs CSR(A^T); scatter mode keeps host memory small.
  std::vector<int32_t> tRowPtr;
  std::vector<int32_t> tColIdx;
  std::vector<float> tValues;
  if (!scatterMode) {
    const double tT0 = nowSeconds();
    tRowPtr.assign(static_cast<size_t>(colsA) + 1, 0);
    {
      std::vector<int32_t> counts(static_cast<size_t>(colsA), 0);
      for (uint64_t k = 0; k < nnz; ++k)
        ++counts[hColIdx[k]];
      int64_t running = 0;
      for (int c = 0; c < colsA; ++c) {
        tRowPtr[c] = static_cast<int32_t>(running);
        running += counts[c];
      }
      tRowPtr[colsA] = static_cast<int32_t>(running);
    }
    tColIdx.resize(static_cast<size_t>(nnz));
    tValues.resize(static_cast<size_t>(nnz));
    {
      std::unique_ptr<std::atomic<int32_t>[]> cursors(
          new std::atomic<int32_t>[static_cast<size_t>(colsA)]);
      for (int c = 0; c < colsA; ++c)
        cursors[c].store(tRowPtr[c], std::memory_order_relaxed);
#pragma omp parallel for schedule(dynamic, 1024)
      for (int r = 0; r < rowsA; ++r) {
        for (uint32_t k = hRowPtr[r]; k < hRowPtr[r + 1]; ++k) {
          const uint32_t c = hColIdx[k];
          const int32_t pos =
              cursors[c].fetch_add(1, std::memory_order_relaxed);
          tColIdx[static_cast<size_t>(pos)] = r;
          tValues[static_cast<size_t>(pos)] = hValues[k];
        }
      }
    }
    const double tT1 = nowSeconds();
    std::printf("host_transpose_s=%.2f\n", tT1 - tT0);
  }

  // ---- Device upload ----
  size_t freeB = 0;
  size_t totalB = 0;
  BENCH_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
  std::printf("vram_free_mib=%zu vram_total_mib=%zu\n", freeB >> 20,
              totalB >> 20);

  DeviceMem dARowPtr;
  DeviceMem dAColIdx;
  DeviceMem dAValues;
  dARowPtr.alloc((static_cast<size_t>(rowsA) + 1) * sizeof(int32_t));
  dAColIdx.alloc(static_cast<size_t>(nnz) * sizeof(int32_t));
  dAValues.alloc(static_cast<size_t>(nnz) * sizeof(float));
  BENCH_CUDA_CHECK(cudaMemcpy(dARowPtr.ptr, hRowPtr,
                              (static_cast<size_t>(rowsA) + 1) *
                                  sizeof(int32_t),
                              cudaMemcpyHostToDevice));
  BENCH_CUDA_CHECK(cudaMemcpy(dAColIdx.ptr, hColIdx,
                              static_cast<size_t>(nnz) * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
  BENCH_CUDA_CHECK(cudaMemcpy(dAValues.ptr, hValues,
                              static_cast<size_t>(nnz) * sizeof(float),
                              cudaMemcpyHostToDevice));

  DeviceMem dTRowPtr;
  DeviceMem dTColIdx;
  DeviceMem dTValues;
  if (!scatterMode) {
    dTRowPtr.alloc((static_cast<size_t>(colsA) + 1) * sizeof(int32_t));
    dTColIdx.alloc(static_cast<size_t>(nnz) * sizeof(int32_t));
    dTValues.alloc(static_cast<size_t>(nnz) * sizeof(float));
    BENCH_CUDA_CHECK(cudaMemcpy(dTRowPtr.ptr, tRowPtr.data(),
                                (static_cast<size_t>(colsA) + 1) *
                                    sizeof(int32_t),
                                cudaMemcpyHostToDevice));
    BENCH_CUDA_CHECK(cudaMemcpy(dTColIdx.ptr, tColIdx.data(),
                                static_cast<size_t>(nnz) * sizeof(int32_t),
                                cudaMemcpyHostToDevice));
    BENCH_CUDA_CHECK(cudaMemcpy(dTValues.ptr, tValues.data(),
                                static_cast<size_t>(nnz) * sizeof(float),
                                cudaMemcpyHostToDevice));
  }

  // Backward input: ratio batch (A-row domain). Forward input: image batch
  // (voxel domain). Shared output buffer sized for the larger direction.
  // zeroFrac zeroes each (row, RHS) ratio independently (the production
  // ratio is exactly 0 for zero-count bins); hRatioDense keeps every entry
  // non-zero so the scatter kernels can be timed without any skipping.
  std::vector<float> hRatio(static_cast<size_t>(rowsA) * batch);
  std::vector<float> hRatioDense(hRatio.size());
  std::vector<float> hImage(static_cast<size_t>(colsA) * batch);
  {
    std::mt19937 rng(20260817u);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (size_t i = 0; i < hRatio.size(); ++i) {
      const float v = dist(rng);
      hRatioDense[i] = v;
      hRatio[i] = coin(rng) < zeroFrac ? 0.0f : v;
    }
    for (auto &v : hImage)
      v = dist(rng);
  }
  if (zeroFrac > 0.0) {
    // Fraction of nnz work a K=2 pair (columns b, b+1) / K=4 quad skips.
    uint64_t pairRowsZero = 0;
    uint64_t pairNnzZero = 0;
    uint64_t quadRowsZero = 0;
    uint64_t quadNnzZero = 0;
    uint64_t pairRows = 0;
    uint64_t quadRows = 0;
    for (int b = 0; b + 1 < batch; b += 2) {
      for (int r = 0; r < rowsA; ++r) {
        const uint64_t rowNnz = hRowPtr[r + 1] - hRowPtr[r];
        ++pairRows;
        if (hRatio[static_cast<size_t>(b) * rowsA + r] == 0.0f &&
            hRatio[static_cast<size_t>(b + 1) * rowsA + r] == 0.0f) {
          ++pairRowsZero;
          pairNnzZero += rowNnz;
        }
      }
    }
    for (int b = 0; b + 3 < batch; b += 4) {
      for (int r = 0; r < rowsA; ++r) {
        const uint64_t rowNnz = hRowPtr[r + 1] - hRowPtr[r];
        ++quadRows;
        bool allZero = true;
        for (int q = 0; q < 4; ++q)
          allZero = allZero &&
                    hRatio[static_cast<size_t>(b + q) * rowsA + r] == 0.0f;
        if (allZero) {
          ++quadRowsZero;
          quadNnzZero += rowNnz;
        }
      }
    }
    std::printf("zero_pairs: rows=%.2f%% nnz=%.2f%%   zero_quads: rows=%.2f%% "
                "nnz=%.2f%%\n",
                100.0 * pairRowsZero / std::max<uint64_t>(1, pairRows),
                100.0 * pairNnzZero / std::max<uint64_t>(1, nnz * (pairRows / rowsA)),
                100.0 * quadRowsZero / std::max<uint64_t>(1, quadRows),
                100.0 * quadNnzZero / std::max<uint64_t>(1, nnz * (quadRows / rowsA)));
  }
  DeviceMem dRatio;
  DeviceMem dRatioDense;
  DeviceMem dImage;
  DeviceMem dOut;
  dRatio.alloc(hRatio.size() * sizeof(float));
  dRatioDense.alloc(hRatioDense.size() * sizeof(float));
  dImage.alloc(hImage.size() * sizeof(float));
  dOut.alloc(static_cast<size_t>(colsA) * batch * sizeof(float));
  BENCH_CUDA_CHECK(cudaMemcpy(dRatio.ptr, hRatio.data(),
                              hRatio.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
  BENCH_CUDA_CHECK(cudaMemcpy(dRatioDense.ptr, hRatioDense.data(),
                              hRatioDense.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
  BENCH_CUDA_CHECK(cudaMemcpy(dImage.ptr, hImage.data(),
                              hImage.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

  // H2D bandwidth probe on this exact machine (values array, 3 reps):
  // pageable vs pinned. Feeds the R1 upload-hiding budget.
  if (!scatterMode) {
    const size_t probeBytes = static_cast<size_t>(nnz) * sizeof(float);
    for (int pinned = 0; pinned < 2; ++pinned) {
      if (pinned) {
        BENCH_CUDA_CHECK(cudaHostRegister(
            const_cast<float *>(tValues.data()), probeBytes,
            cudaHostRegisterDefault));
      }
      double bestGiBs = 0.0;
      for (int rep = 0; rep < 3; ++rep) {
        const double t0 = nowSeconds();
        BENCH_CUDA_CHECK(cudaMemcpy(dTValues.ptr, tValues.data(), probeBytes,
                                    cudaMemcpyHostToDevice));
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
        const double dt = nowSeconds() - t0;
        bestGiBs = std::max(
            bestGiBs, static_cast<double>(probeBytes) / dt / (1u << 30));
      }
      std::printf("h2d_%s_gib_s=%.1f\n", pinned ? "pinned" : "pageable",
                  bestGiBs);
      if (pinned) {
        BENCH_CUDA_CHECK(
            cudaHostUnregister(const_cast<float *>(tValues.data())));
      }
    }
  }

  cusparseHandle_t handle = nullptr;
  BENCH_CUSPARSE_CHECK(cusparseCreate(&handle));
  cusparseSpMatDescr_t matA = nullptr;
  cusparseSpMatDescr_t matT = nullptr;
  BENCH_CUSPARSE_CHECK(cusparseCreateCsr(
      &matA, rowsA, colsA, static_cast<int64_t>(nnz), dARowPtr.ptr,
      dAColIdx.ptr, dAValues.ptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  if (!scatterMode) {
    BENCH_CUSPARSE_CHECK(cusparseCreateCsr(
        &matT, colsA, rowsA, static_cast<int64_t>(nnz), dTRowPtr.ptr,
        dTColIdx.ptr, dTValues.ptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  }

  DeviceMem workspace;
  size_t workspaceBytes = 0;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto ensureWorkspace = [&](size_t bytes) {
    if (bytes > workspaceBytes) {
      workspace.alloc(bytes);
      workspaceBytes = bytes;
    }
  };

  // One full pass = batch RHS: inCount-domain inputs -> outCount-domain
  // outputs, per-RHS contiguous slices in both buffers.
  auto runSpmvPass = [&](cusparseSpMatDescr_t mat, cusparseOperation_t op,
                         cusparseSpMVAlg_t alg, const float *dIn,
                         int64_t inCount, int64_t outCount) {
    for (int b = 0; b < batch; ++b) {
      cusparseDnVecDescr_t vecX = nullptr;
      cusparseDnVecDescr_t vecY = nullptr;
      BENCH_CUSPARSE_CHECK(cusparseCreateDnVec(
          &vecX, inCount,
          const_cast<float *>(dIn) + static_cast<size_t>(b) * inCount,
          CUDA_R_32F));
      BENCH_CUSPARSE_CHECK(cusparseCreateDnVec(
          &vecY, outCount,
          static_cast<float *>(dOut.ptr) +
              static_cast<size_t>(b) * outCount,
          CUDA_R_32F));
      size_t bytes = 0;
      BENCH_CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, op, &alpha, mat,
                                                   vecX, &beta, vecY,
                                                   CUDA_R_32F, alg, &bytes));
      ensureWorkspace(bytes);
      BENCH_CUSPARSE_CHECK(cusparseSpMV(handle, op, &alpha, mat, vecX, &beta,
                                        vecY, CUDA_R_32F, alg,
                                        workspace.ptr));
      BENCH_CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
      BENCH_CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    }
  };

  auto runSpmmPass = [&](cusparseSpMatDescr_t mat, cusparseOperation_t op,
                         cusparseSpMMAlg_t alg, int groupK, const float *dIn,
                         int64_t inCount, int64_t outCount) {
    for (int b = 0; b < batch; b += groupK) {
      const int k = std::min(groupK, batch - b);
      cusparseDnMatDescr_t matB = nullptr;
      cusparseDnMatDescr_t matC = nullptr;
      BENCH_CUSPARSE_CHECK(cusparseCreateDnMat(
          &matB, inCount, k, inCount,
          const_cast<float *>(dIn) + static_cast<size_t>(b) * inCount,
          CUDA_R_32F, CUSPARSE_ORDER_COL));
      BENCH_CUSPARSE_CHECK(cusparseCreateDnMat(
          &matC, outCount, k, outCount,
          static_cast<float *>(dOut.ptr) +
              static_cast<size_t>(b) * outCount,
          CUDA_R_32F, CUSPARSE_ORDER_COL));
      size_t bytes = 0;
      BENCH_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
          handle, op, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, matB,
          &beta, matC, CUDA_R_32F, alg, &bytes));
      ensureWorkspace(bytes);
      BENCH_CUSPARSE_CHECK(cusparseSpMM(handle, op,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, mat, matB, &beta, matC,
                                        CUDA_R_32F, alg, workspace.ptr));
      BENCH_CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
      BENCH_CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    }
  };

  cudaEvent_t evStart = nullptr;
  cudaEvent_t evStop = nullptr;
  BENCH_CUDA_CHECK(cudaEventCreate(&evStart));
  BENCH_CUDA_CHECK(cudaEventCreate(&evStop));

  std::vector<float> refBackward;
  std::vector<float> refForward;
  std::vector<float> scratch;
  std::vector<BenchResult> bwdResults;
  std::vector<BenchResult> fwdResults;

  auto benchVariant = [&](const std::string &name, int64_t outCount,
                          std::vector<float> &reference,
                          std::vector<BenchResult> &results, auto &&pass) {
    BENCH_CUDA_CHECK(
        cudaMemset(dOut.ptr, 0, static_cast<size_t>(colsA) * batch *
                                    sizeof(float)));
    pass(); // warmup + workspace sizing
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());
    double msTotal = 0.0;
    double msMin = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
      BENCH_CUDA_CHECK(cudaEventRecord(evStart));
      pass();
      BENCH_CUDA_CHECK(cudaEventRecord(evStop));
      BENCH_CUDA_CHECK(cudaEventSynchronize(evStop));
      float ms = 0.0f;
      BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
      msTotal += ms;
      msMin = std::min(msMin, static_cast<double>(ms));
    }
    BenchResult result;
    result.name = name;
    result.msAvg = msTotal / reps;
    result.msMin = msMin;
    const size_t outTotal = static_cast<size_t>(outCount) * batch;
    if (reference.empty()) {
      reference.resize(outTotal);
      BENCH_CUDA_CHECK(cudaMemcpy(reference.data(), dOut.ptr,
                                  outTotal * sizeof(float),
                                  cudaMemcpyDeviceToHost));
      result.relL2 = 0.0;
    } else {
      result.relL2 = relL2HostVsDevice(
          reference, static_cast<float *>(dOut.ptr), outTotal, scratch);
    }
    results.push_back(result);
    std::printf("%-28s ms_avg=%9.2f ms_min=%9.2f relL2=%.3e\n", name.c_str(),
                result.msAvg, result.msMin, result.relL2);
  };

  const float *ratioIn = static_cast<const float *>(dRatio.ptr);
  const float *ratioDenseIn = static_cast<const float *>(dRatioDense.ptr);
  const float *imageIn = static_cast<const float *>(dImage.ptr);

  // ---- Backward direction: rays -> voxels (the 70% bottleneck) ----
  std::printf("\nbackward (rays -> voxels):\n");
  benchVariant("A_T_spmv_alg2(base)", colsA, refBackward, bwdResults, [&] {
    runSpmvPass(matA, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_SPMV_CSR_ALG2,
                ratioIn, rowsA, colsA);
  });

  if (scatterMode) {
    // Production-fatbin scatter kernels on CSR(A): pairs / quads of RHS,
    // zero-row skipping. Outputs accumulate, so each pass zeroes them first
    // (the same memset the numerator would need instead of beta = 0).
    const float *aValues = static_cast<const float *>(dAValues.ptr);
    const int32_t *aRowPtr = static_cast<const int32_t *>(dARowPtr.ptr);
    const int32_t *aColIdx = static_cast<const int32_t *>(dAColIdx.ptr);
    float *out = static_cast<float *>(dOut.ptr);
    const size_t outBytes = static_cast<size_t>(colsA) * batch * sizeof(float);
    auto pairPass = [&](const float *in) {
      BENCH_CUDA_CHECK(cudaMemsetAsync(out, 0, outBytes));
      for (int b = 0; b + 1 < batch; b += 2) {
        spect::launchCompactCsrSpmvTPairKernel(
            aValues, aRowPtr, aColIdx, in + static_cast<size_t>(b) * rowsA,
            in + static_cast<size_t>(b + 1) * rowsA,
            out + static_cast<size_t>(b) * colsA,
            out + static_cast<size_t>(b + 1) * colsA, rowsA);
      }
    };
    auto quadPass = [&](const float *in) {
      BENCH_CUDA_CHECK(cudaMemsetAsync(out, 0, outBytes));
      for (int b = 0; b + 3 < batch; b += 4) {
        spect::launchCompactCsrSpmvTQuadKernel(
            aValues, aRowPtr, aColIdx, in + static_cast<size_t>(b) * rowsA,
            in + static_cast<size_t>(b + 1) * rowsA,
            in + static_cast<size_t>(b + 2) * rowsA,
            in + static_cast<size_t>(b + 3) * rowsA,
            out + static_cast<size_t>(b) * colsA,
            out + static_cast<size_t>(b + 1) * colsA,
            out + static_cast<size_t>(b + 2) * colsA,
            out + static_cast<size_t>(b + 3) * colsA, rowsA);
      }
    };
    if (batch % 4 != 0) {
      std::fprintf(stderr, "scatter mode needs batch %% 4 == 0\n");
      return 2;
    }
    benchVariant("A_T_scatter_pair_skip", colsA, refBackward, bwdResults,
                 [&] { pairPass(ratioIn); });
    benchVariant("A_T_scatter_quad_skip", colsA, refBackward, bwdResults,
                 [&] { quadPass(ratioIn); });
    // Dense ratios: no row skips; isolates kernel efficiency vs cuSPARSE.
    // Their outputs differ from the reference (different input) so relL2 is
    // reported against a dense cuSPARSE run instead.
    std::vector<float> refDense;
    std::vector<BenchResult> denseResults;
    benchVariant("A_T_spmv_alg2(dense)", colsA, refDense, denseResults, [&] {
      runSpmvPass(matA, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_SPMV_CSR_ALG2,
                  ratioDenseIn, rowsA, colsA);
    });
    benchVariant("A_T_scatter_pair_dense", colsA, refDense, denseResults,
                 [&] { pairPass(ratioDenseIn); });
    benchVariant("A_T_scatter_quad_dense", colsA, refDense, denseResults,
                 [&] { quadPass(ratioDenseIn); });

    auto printSummaryScatter = [](const char *label,
                                  const std::vector<BenchResult> &results) {
      const BenchResult &base = results.front();
      std::printf("\n%s summary (speedup vs %s):\n", label, base.name.c_str());
      for (const auto &r : results) {
        std::printf("%-28s %6.2fx relL2=%.3e\n", r.name.c_str(),
                    base.msMin / r.msMin, r.relL2);
      }
    };
    printSummaryScatter("backward (zeroFrac pattern)", bwdResults);
    printSummaryScatter("backward (dense ratios)", denseResults);
    BENCH_CUDA_CHECK(cudaEventDestroy(evStart));
    BENCH_CUDA_CHECK(cudaEventDestroy(evStop));
    BENCH_CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    BENCH_CUSPARSE_CHECK(cusparseDestroy(handle));
    return 0;
  }

  benchVariant("At_N_spmv_alg2", colsA, refBackward, bwdResults, [&] {
    runSpmvPass(matT, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_SPMV_CSR_ALG2, ratioIn, rowsA, colsA);
  });
  benchVariant("At_N_spmv_alg1", colsA, refBackward, bwdResults, [&] {
    runSpmvPass(matT, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_SPMV_CSR_ALG1, ratioIn, rowsA, colsA);
  });
  benchVariant("At_N_spmm_default_K8", colsA, refBackward, bwdResults, [&] {
    runSpmmPass(matT, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_SPMM_ALG_DEFAULT, 8, ratioIn, rowsA, colsA);
  });
  benchVariant("At_N_spmm_default_Kfull", colsA, refBackward, bwdResults,
               [&] {
                 runSpmmPass(matT, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_SPMM_ALG_DEFAULT, batch, ratioIn, rowsA,
                             colsA);
               });
  benchVariant("At_N_spmm_alg1_Kfull", colsA, refBackward, bwdResults, [&] {
    runSpmmPass(matT, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_SPMM_CSR_ALG1, batch, ratioIn, rowsA, colsA);
  });
  benchVariant("At_N_spmm_alg3_Kfull", colsA, refBackward, bwdResults, [&] {
    runSpmmPass(matT, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_SPMM_CSR_ALG3, batch, ratioIn, rowsA, colsA);
  });

  // ---- Forward direction: voxels -> rays (feasibility of storing only
  // A^T and running forward as op=T scatter into the tiny ray domain) ----
  std::printf("\nforward (voxels -> rays):\n");
  benchVariant("A_N_spmv_alg2(base)", rowsA, refForward, fwdResults, [&] {
    runSpmvPass(matA, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_SPMV_CSR_ALG2, imageIn, colsA, rowsA);
  });
  benchVariant("At_T_spmv_alg2", rowsA, refForward, fwdResults, [&] {
    runSpmvPass(matT, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_SPMV_CSR_ALG2,
                imageIn, colsA, rowsA);
  });
  benchVariant("At_T_spmm_default_K8", rowsA, refForward, fwdResults, [&] {
    runSpmmPass(matT, CUSPARSE_OPERATION_TRANSPOSE,
                CUSPARSE_SPMM_ALG_DEFAULT, 8, imageIn, colsA, rowsA);
  });
  benchVariant("At_T_spmm_default_Kfull", rowsA, refForward, fwdResults,
               [&] {
                 runSpmmPass(matT, CUSPARSE_OPERATION_TRANSPOSE,
                             CUSPARSE_SPMM_ALG_DEFAULT, batch, imageIn, colsA,
                             rowsA);
               });
  benchVariant("A_N_spmm_default_K4", rowsA, refForward, fwdResults, [&] {
    runSpmmPass(matA, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_SPMM_ALG_DEFAULT, 4, imageIn, colsA, rowsA);
  });

  auto printSummary = [](const char *label,
                         const std::vector<BenchResult> &results) {
    const BenchResult &base = results.front();
    std::printf("\n%s summary (speedup vs %s):\n", label, base.name.c_str());
    for (const auto &r : results) {
      std::printf("%-28s %6.2fx relL2=%.3e\n", r.name.c_str(),
                  base.msMin / r.msMin, r.relL2);
    }
  };
  printSummary("backward", bwdResults);
  printSummary("forward", fwdResults);

  BENCH_CUDA_CHECK(cudaEventDestroy(evStart));
  BENCH_CUDA_CHECK(cudaEventDestroy(evStop));
  BENCH_CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  BENCH_CUSPARSE_CHECK(cusparseDestroySpMat(matT));
  BENCH_CUSPARSE_CHECK(cusparseDestroy(handle));
  return 0;
}
