#include "main/StandardReconLoop.h"

#include "app/ADMMOptimizer.h"
#include "app/BSREMOptimizer.h"
#include "app/BSREMReconstructor.h"
#include "app/BedOffsets.h"
#include "app/IOptimizer.h"
#include "app/MLEMOptimizer.h"
#include "app/OSEMReconstructor.h"
#include "app/ProjectionOps.h"
#include "app/ReconMetrics.h"
#include "core/Masks.h"
#include "cuda/ProjDomainChunked.h"
#include "cuda/VolumeOpsCUDA.h"
#include "filters/ButterworthFilter.h"
#include "filters/Convolution.h"
#include "interfaces/ISparseOps.h"
#include "main/CTACOps.h"
#include "main/ReconDataLoader.h"
#include "main/RunLogger.h"
#include "main/SmoothingOps.h"
#include "main/VolumeOutput.h"
#include "regularizer/RegularizerCompat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace spect::main {
namespace {

using ProfileClock = std::chrono::steady_clock;

struct ReconstructionStageTiming {
  double regGradientSeconds = 0.0;
  double subsetSensitivitySeconds = 0.0;
  double cudaChunkWarmupSeconds = 0.0;
  double osemCoreSeconds = 0.0;
  double bsremCoreSeconds = 0.0;
  double butterworthSeconds = 0.0;
  double smoothingSeconds = 0.0;
  double saveSeconds = 0.0;
  size_t regGradientCalls = 0;
  size_t subsetSensitivityCalls = 0;
  size_t cudaChunkWarmupCalls = 0;
  size_t osemCoreCalls = 0;
  size_t bsremCoreCalls = 0;
  size_t butterworthCalls = 0;
  size_t smoothingCalls = 0;
  size_t saveCalls = 0;
};

struct ProjDomainChunkCacheKey {
  const void *rowPtr = nullptr;
  const void *colIdx = nullptr;
  const void *values = nullptr;
  int rows = 0;
  int cols = 0;
  uint64_t nnz = 0;
  uint64_t frameSize = 0;
  uint64_t chunkNnzBudget = 0;

  bool operator==(const ProjDomainChunkCacheKey &other) const {
    return rowPtr == other.rowPtr && colIdx == other.colIdx &&
           values == other.values && rows == other.rows && cols == other.cols &&
           nnz == other.nnz && frameSize == other.frameSize &&
           chunkNnzBudget == other.chunkNnzBudget;
  }
};

std::string formatProfileEntry(const std::string &label, double seconds,
                               size_t calls, uint64_t bytes = 0) {
  std::ostringstream oss;
  oss << "[PROFILE] " << label << ": total=" << std::fixed
      << std::setprecision(3) << seconds << "s";
  if (calls > 0)
    oss << ", calls=" << calls << ", avg="
        << (seconds / static_cast<double>(calls)) << "s";
  if (bytes > 0)
    oss << ", bytes=" << bytes;
  return oss.str();
}

void logCudaChunkTimingSnapshot(
    const spect::cuda::CudaChunkStageTimingSnapshot &cudaChunkTiming) {
  logInfo(formatProfileEntry("cuda.chunk_build",
                             cudaChunkTiming.chunkBuildSeconds,
                             cudaChunkTiming.chunkBuildCalls));
  logInfo(formatProfileEntry("cuda.pinned_stage",
                             cudaChunkTiming.pinnedStageSeconds,
                             cudaChunkTiming.pinnedStageCalls));
  logInfo(formatProfileEntry("cuda.h2d_chunk", cudaChunkTiming.h2dChunkSeconds,
                             cudaChunkTiming.h2dChunkCalls));
  logInfo(formatProfileEntry("cuda.upload_wait",
                             cudaChunkTiming.uploadWaitOnComputeSeconds,
                             cudaChunkTiming.uploadWaitOnComputeCalls));
  logInfo(formatProfileEntry("cuda.fused_bed_cycle",
                             cudaChunkTiming.fusedBedCycleSeconds,
                             cudaChunkTiming.fusedBedCycleCalls));
  logInfo(formatProfileEntry("cuda.cusparse_fwd",
                             cudaChunkTiming.cusparseForwardSeconds,
                             cudaChunkTiming.cusparseForwardCalls));
  logInfo(formatProfileEntry("cuda.cusparse_fwd_host",
                             cudaChunkTiming.cusparseForwardHostSeconds,
                             cudaChunkTiming.cusparseForwardHostCalls));
  logInfo(formatProfileEntry("cuda.ratio", cudaChunkTiming.ratioSeconds,
                             cudaChunkTiming.ratioCalls));
  logInfo(formatProfileEntry("cuda.transpose_chunk",
                             cudaChunkTiming.transposeChunkSeconds,
                             cudaChunkTiming.transposeChunkCalls));
  logInfo(formatProfileEntry("cuda.cusparse_bwd",
                             cudaChunkTiming.cusparseBackwardSeconds,
                             cudaChunkTiming.cusparseBackwardCalls));
  logInfo(formatProfileEntry("cuda.cusparse_bwd_host",
                             cudaChunkTiming.cusparseBackwardHostSeconds,
                             cudaChunkTiming.cusparseBackwardHostCalls));
  if (cudaChunkTiming.compactMatrixUploadCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_matrix_upload",
                               cudaChunkTiming.compactMatrixUploadSeconds,
                               cudaChunkTiming.compactMatrixUploadCalls));
  }
  if (cudaChunkTiming.compactMatrixReuseCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_matrix_reuse_hit", 0.0,
                               cudaChunkTiming.compactMatrixReuseCalls));
  }
  if (cudaChunkTiming.compactInputH2DCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_input_h2d",
                               cudaChunkTiming.compactInputH2DSeconds,
                               cudaChunkTiming.compactInputH2DCalls));
  }
  if (cudaChunkTiming.compactProjectionH2DCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_projection_h2d",
                               cudaChunkTiming.compactProjectionH2DSeconds,
                               cudaChunkTiming.compactProjectionH2DCalls,
                               cudaChunkTiming.compactProjectionH2DBytes));
  }
  if (cudaChunkTiming.compactOutputD2HCalls > 0) {
    logInfo(formatProfileEntry("cuda.compact_output_d2h",
                               cudaChunkTiming.compactOutputD2HSeconds,
                               cudaChunkTiming.compactOutputD2HCalls));
  }
  if (cudaChunkTiming.compactForwardOutputD2HCalls > 0) {
    logInfo(formatProfileEntry(
        "cuda.compact_forward_output_d2h",
        cudaChunkTiming.compactForwardOutputD2HSeconds,
        cudaChunkTiming.compactForwardOutputD2HCalls,
        cudaChunkTiming.compactForwardOutputD2HBytes));
  }
  if (cudaChunkTiming.compactBackwardOutputD2HCalls > 0) {
    logInfo(formatProfileEntry(
        "cuda.compact_backward_output_d2h",
        cudaChunkTiming.compactBackwardOutputD2HSeconds,
        cudaChunkTiming.compactBackwardOutputD2HCalls,
        cudaChunkTiming.compactBackwardOutputD2HBytes));
  }
  if (cudaChunkTiming.deviceCacheHitCalls > 0) {
    logInfo(formatProfileEntry("cuda.device_cache_hit", 0.0,
                               cudaChunkTiming.deviceCacheHitCalls));
  }
}

std::string outputPrefixFromArgs(const ReconArgs &args) {
  return args.outputName.empty() ? std::string("recon") : args.outputName;
}

inline bool canUseFrameAlignedProjDomainChunks(size_t frameSize, int rows) {
  return frameSize > 0 && static_cast<size_t>(rows) % frameSize == 0;
}

template <typename MatrixType>
bool canUseProjDomainCudaChunking(const MatrixType &, const ReconOpts &) {
  return false;
}

inline bool canUseProjDomainCudaChunking(const io::CSRView &matrix,
                                         const ReconOpts &opts) {
  return opts.sparseOps != nullptr &&
         opts.sparseOps->backend() == BackendType::CUDA &&
         matrix.nnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
}

template <typename MatrixType>
std::vector<std::vector<int>> buildProjDomainCudaChunks(
    const MatrixType &matrix, size_t frameSize, uint64_t chunkNnzBudget) {
  const uint64_t kMaxChunkNnz = std::clamp<uint64_t>(
      chunkNnzBudget, 8ull * 1024ull * 1024ull,
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
  const int rows = static_cast<int>(matrix.rows);
  std::vector<std::vector<int>> chunks;
  chunks.reserve(4);

  auto rowOffset = [&](int row) -> uint64_t {
    return static_cast<uint64_t>(matrix.row_ptr[static_cast<size_t>(row)]);
  };

  auto appendChunk = [&](int rowStart, int rowEnd) {
    std::vector<int> chunk(static_cast<size_t>(rowEnd - rowStart));
    std::iota(chunk.begin(), chunk.end(), rowStart);
    chunks.push_back(std::move(chunk));
  };

  if (canUseFrameAlignedProjDomainChunks(frameSize, rows)) {
    const int frameRows = static_cast<int>(frameSize);
    int rowStart = 0;
    while (rowStart < rows) {
      const uint64_t startOffset = rowOffset(rowStart);
      int rowEnd = rowStart;
      while (rowEnd < rows) {
        const int nextEnd = std::min(rows, rowEnd + frameRows);
        const uint64_t endOffset = rowOffset(nextEnd);
        if (endOffset < startOffset) {
          throw std::runtime_error(
              "Proj-domain MLEM CUDA chunking encountered non-monotonic CSR row offsets.");
        }
        if (endOffset - startOffset > kMaxChunkNnz) {
          break;
        }
        rowEnd = nextEnd;
      }

      if (rowEnd == rowStart) {
        throw std::runtime_error(
            "Proj-domain MLEM CUDA chunking found a single detector frame whose nnz exceeds the int32 subset limit.");
      }

      appendChunk(rowStart, rowEnd);
      rowStart = rowEnd;
    }
    return chunks;
  }

  int rowStart = 0;
  while (rowStart < rows) {
    const uint64_t startOffset = rowOffset(rowStart);
    int rowEnd = rowStart;
    while (rowEnd < rows) {
      const uint64_t endOffset = rowOffset(rowEnd + 1);
      if (endOffset < startOffset) {
        throw std::runtime_error(
            "Proj-domain MLEM CUDA chunking encountered non-monotonic CSR row offsets.");
      }
      if (endOffset - startOffset > kMaxChunkNnz) {
        break;
      }
      ++rowEnd;
    }

    if (rowEnd == rowStart) {
      throw std::runtime_error(
          "Proj-domain MLEM CUDA chunking found a single row whose nnz exceeds the int32 subset limit.");
    }

    appendChunk(rowStart, rowEnd);
    rowStart = rowEnd;
  }

  return chunks;
}

inline const std::vector<std::vector<int>> &getProjDomainCudaChunks(
    const io::CSRView &matrix, size_t frameSize, uint64_t chunkNnzBudget) {
  static std::mutex cacheMutex;
  static std::vector<
      std::pair<ProjDomainChunkCacheKey, std::vector<std::vector<int>>>>
      cache;

  const ProjDomainChunkCacheKey key{matrix.row_ptr, matrix.col_idx, matrix.val,
                                    matrix.rows, matrix.cols, matrix.nnz,
                                    static_cast<uint64_t>(frameSize),
                                    chunkNnzBudget};
  std::lock_guard<std::mutex> lock(cacheMutex);
  for (auto &entry : cache) {
    if (entry.first == key) {
      return entry.second;
    }
  }

  if (cache.size() >= 4) {
    cache.erase(cache.begin());
  }
  cache.emplace_back(key,
                     buildProjDomainCudaChunks(matrix, frameSize,
                                               chunkNnzBudget));
  return cache.back().second;
}

inline void runProjDomainChunkedForward(const io::CSRView &matrix,
                                        const ReconOpts &opts,
                                        size_t frameSize,
                                        uint64_t chunkNnzBudget,
                                        const double *x, double *y) {
  const SparseSubsetContext chunkCtx{};
  const auto &chunks =
      getProjDomainCudaChunks(matrix, frameSize, chunkNnzBudget);
  for (const auto &chunkRows : chunks) {
    std::vector<double> local(static_cast<size_t>(chunkRows.size()), 0.0);
    opts.sparseOps->spmvSubset(matrix, x, chunkRows, chunkCtx, local.data(),
                               opts.threads);
    std::copy(local.begin(), local.end(),
              y + static_cast<size_t>(chunkRows.front()));
  }
}

inline void runProjDomainChunkedTranspose(const io::CSRView &matrix,
                                          const ReconOpts &opts,
                                          size_t frameSize,
                                          uint64_t chunkNnzBudget,
                                          const double *x, double *y) {
  const SparseSubsetContext chunkCtx{};
  const auto &chunks =
      getProjDomainCudaChunks(matrix, frameSize, chunkNnzBudget);
  std::vector<double> local(static_cast<size_t>(matrix.cols), 0.0);
  for (const auto &chunkRows : chunks) {
    const size_t chunkSize = static_cast<size_t>(chunkRows.size());
    std::vector<double> localX(chunkSize, 0.0);
    std::copy_n(x + static_cast<size_t>(chunkRows.front()), chunkSize,
                localX.data());
    std::fill(local.begin(), local.end(), 0.0);
    opts.sparseOps->spmvTSubset(matrix, localX.data(), chunkRows, chunkCtx,
                                local.data(), opts.threads);
    for (size_t index = 0; index < local.size(); ++index) {
      y[index] += local[index];
    }
  }
}

std::vector<double> computeKappaMap(const std::vector<double> &globalSensitivity,
                                    const ReconArgs &args,
                                    const std::vector<uint8_t> *mask) {
  std::vector<double> kappa(globalSensitivity.size(), 1.0);
  if (!args.bsremUseKappaMap || globalSensitivity.empty())
    return kappa;

  const double eps = std::max(args.bsremKappaEps, 1e-12);
  double sum = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < globalSensitivity.size(); ++i) {
    if (mask && !(*mask)[i])
      continue;
    const double value = globalSensitivity[i];
    if (value > eps && std::isfinite(value)) {
      kappa[i] = 1.0 / std::sqrt(value + eps);
      sum += kappa[i];
      ++count;
    }
  }
  const double mean = (count > 0) ? (sum / static_cast<double>(count)) : 1.0;
  for (size_t i = 0; i < kappa.size(); ++i) {
    double value = kappa[i] / mean;
    if (args.bsremSoftMask && mask && !(*mask)[i])
      value = std::max(value, 1.0 + args.bsremSoftMaskLambdaOut);
    value =
        std::clamp(value, args.bsremKappaClipMin, args.bsremKappaClipMax);
    kappa[i] = value;
  }

  return kappa;
}

bool tryComputeRegGradientCUDA(const std::vector<double> &image,
                               const ReconArgs &args, const ReconOpts &opts,
                               const std::vector<uint8_t> *maskForReg,
                               std::vector<double> &gradReg) {
  if (!opts.sparseOps || opts.sparseOps->backend() != BackendType::CUDA)
    return false;
  return cuda::computeRegGradientCUDA(image, args.Nx, args.Ny, args.Nz,
                                      args.reg, maskForReg, gradReg);
}

bool tryApplySeparableConvCUDA(std::vector<double> &volume, int nx, int ny,
                               int nz, const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz,
                               const ReconOpts &opts) {
  return opts.sparseOps && opts.sparseOps->backend() == BackendType::CUDA &&
         cuda::isCudaVolumeOpsAvailable() &&
         cuda::applySeparableConv3DInplaceCUDA(volume, nx, ny, nz, kx, ky, kz);
}

void applySeparableConvRuntime(std::vector<double> &volume, int nx, int ny,
                               int nz, const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz,
                               const ReconOpts &opts, int threads) {
  if (!tryApplySeparableConvCUDA(volume, nx, ny, nz, kx, ky, kz, opts)) {
    filters::conv3d_separable_inplace(volume, nx, ny, nz, kx, ky, kz, threads);
  }
}

void applyJosephLike(std::vector<double> &buffer, int nx, int ny, int nz,
                     int passes, int threads, bool clampImage) {
  if (passes <= 0)
    return;
  applyJosephSmoothing(buffer, nx, ny, nz, passes, threads);
  if (clampImage)
    clampNonNegativeInPlace(buffer, threads);
}

template <typename MatrixType>
void runProjDomainMLEMStep(const MatrixType &matrix,
                           const std::vector<double> &projections,
                           std::vector<double> &image,
                           const std::vector<double> &denom,
                           const ReconArgs &args, const ReconOpts &opts,
                           const filters::PSFKernels &psfKernels,
                           const std::vector<double> *gradReg,
                           const std::vector<uint8_t> *maskForReg,
                           const std::vector<uint8_t> *maskPtr,
                           const filters::ButterworthConf &butterConf,
                           const SmoothConf &smoothConf,
                           const filters::Spacing3 &spacing) {
  (void)maskForReg;
  const int rows = matrix.rows;
  const int nvox = static_cast<int>(image.size());
  const double eps = 1e-12;
  const size_t frameSize =
      static_cast<size_t>(opts.det_nz) * static_cast<size_t>(opts.det_nx);
  uint64_t chunkNnzBudget = 8ull * 1024ull * 1024ull;
  bool useChunkedCudaForward = false;
  bool useStreamingChunkedCuda = false;
  bool usedSparseForward = false;
  std::vector<double> numer(static_cast<size_t>(nvox), 0.0);

  if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
    useChunkedCudaForward = canUseProjDomainCudaChunking(matrix, opts);
    if (useChunkedCudaForward) {
      chunkNnzBudget = cuda::getProjDomainChunkNnzBudgetCUDA();
    }
    useStreamingChunkedCuda =
        useChunkedCudaForward && canUseFrameAlignedProjDomainChunks(frameSize, rows);
    if (useStreamingChunkedCuda) {
      const auto &chunks =
          getProjDomainCudaChunks(matrix, frameSize, chunkNnzBudget);
      logInfo("Proj-domain CUDA chunking: chunks=" +
              std::to_string(chunks.size()) +
              ", frameRows=" + std::to_string(frameSize) +
              ", nnzBudget=" + std::to_string(chunkNnzBudget));
      usedSparseForward = cuda::runProjDomainChunkedStreamingCUDA(
          matrix, chunks, projections, psfKernels, frameSize, opts.det_nz,
          opts.det_nx, image.data(), opts.threads, numer.data());
    }
  }

  if (!usedSparseForward) {
    std::vector<double> yhat(static_cast<size_t>(rows), 0.0);

    if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
      if (useChunkedCudaForward) {
        runProjDomainChunkedForward(matrix, opts, frameSize, chunkNnzBudget,
                                    image.data(), yhat.data());
        usedSparseForward = true;
      } else {
        try {
          opts.sparseOps->spmv(matrix, image.data(), yhat.data(), opts.threads);
          usedSparseForward = true;
        } catch (const std::runtime_error &ex) {
          const std::string message = ex.what() ? ex.what() : "";
          if (message.find("row offsets representable as int32") ==
              std::string::npos) {
            throw;
          }
          std::cerr << "[WARN] Proj-domain MLEM forward projection fell back to CPU because the CUDA sparse backend does not support this matrix row-offset range.\n";
        }
      }
    } else {
      spect::io::CSRView view{};
      view.rows = matrix.rows;
      view.cols = matrix.cols;
      view.row_ptr = matrix.row_ptr.data();
      view.col_idx = matrix.col_idx.data();
      view.val = matrix.val.data();
      view.nnz = matrix.val.size();
      try {
        opts.sparseOps->spmv(view, image.data(), yhat.data(), opts.threads);
        usedSparseForward = true;
      } catch (const std::runtime_error &ex) {
        const std::string message = ex.what() ? ex.what() : "";
        if (message.find("row offsets representable as int32") ==
            std::string::npos) {
          throw;
        }
        std::cerr << "[WARN] Proj-domain MLEM forward projection fell back to CPU because the CUDA sparse backend does not support this matrix row-offset range.\n";
      }
    }

    if (!usedSparseForward) {
      yhat = spect::forward_project(matrix, image, opts.threads);
    }

    std::vector<double> ratio = projections;
#pragma omp parallel for schedule(static) num_threads(opts.threads)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(ratio.size()); ++i) {
      ratio[static_cast<size_t>(i)] /=
          std::max(yhat[static_cast<size_t>(i)], eps);
    }

    if (frameSize > 0) {
#pragma omp parallel num_threads(opts.threads) if(opts.threads > 1 && opts.num_angles > 1)
      {
        std::vector<double> frame(frameSize);
#pragma omp for schedule(static)
        for (int angle = 0; angle < opts.num_angles; ++angle) {
          const size_t offset = static_cast<size_t>(angle) * frameSize;
          std::copy_n(ratio.data() + offset, frameSize, frame.data());
          filters::conv2d_separable_inplace(frame, opts.det_nz, opts.det_nx,
                                            psfKernels.ky, psfKernels.kx, 1);
          std::copy_n(frame.data(), frameSize, ratio.data() + offset);
        }
      }
    }

    if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
      if (useChunkedCudaForward) {
        runProjDomainChunkedTranspose(matrix, opts, frameSize, chunkNnzBudget,
                                      ratio.data(), numer.data());
      } else {
        opts.sparseOps->spmvT(matrix, ratio.data(), numer.data(), opts.threads);
      }
    } else {
      spect::io::CSRView view{};
      view.rows = matrix.rows;
      view.cols = matrix.cols;
      view.row_ptr = matrix.row_ptr.data();
      view.col_idx = matrix.col_idx.data();
      view.val = matrix.val.data();
      view.nnz = matrix.val.size();
      opts.sparseOps->spmvT(view, ratio.data(), numer.data(), opts.threads);
    }
  }

#pragma omp parallel for schedule(static) num_threads(opts.threads)
  for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(image.size()); ++j) {
    const size_t index = static_cast<size_t>(j);
    if (maskPtr && !(*maskPtr)[index]) {
      image[index] = 0.0;
      continue;
    }
    double denomEff = denom[index];
    if (!(denomEff > 0.0)) {
      image[index] = 0.0;
      continue;
    }
    if (gradReg) {
      const double fj = std::max(image[index], eps);
      denomEff += args.reg.beta * ((*gradReg)[index] / fj);
      if (!(denomEff > 0.0))
        denomEff =
            denom[index] + args.reg.beta * std::abs((*gradReg)[index]) / fj + eps;
    }
    image[index] *= numer[index] / denomEff;
    if (opts.nonneg && image[index] < 0.0)
      image[index] = 0.0;
  }

  applyJosephLike(image, args.Nx, args.Ny, args.Nz, args.josephPasses,
                  opts.threads, true);
  if (butterConf.enabled) {
    filters::apply_butterworth_inplace(image, args.Nx, args.Ny, args.Nz,
                                       spacing, butterConf, opts.threads);
  }
  if (smoothConf.enabled) {
    applySeparableConvRuntime(image, args.Nx, args.Ny, args.Nz, smoothConf.kx,
                              smoothConf.ky, smoothConf.kz, opts, opts.threads);
    if (maskPtr)
      spect::core::applyMaskInplace(image, *maskPtr);
  }
}

template <typename MatrixType>
std::vector<double> runStandardReconstructionImpl(
    const StandardReconLoopContext &ctx, const MatrixType &matrix) {
  const ReconArgs &args = *ctx.args;
  const ReconOpts &opts = *ctx.opts;
  const ReconPreparedParams &prepared = *ctx.prepared;
  const std::vector<double> &projections = *ctx.projections;
  const std::vector<uint8_t> *maskPtr = ctx.mask;
  const std::vector<double> &sharedAnatomical = *ctx.sharedAnatomical;
  const filters::PSFKernels *psfPtr = ctx.psfKernels;
  const std::vector<int> &bedZStarts = *ctx.bedZStarts;
  const int bedCount = ctx.bedCount;
  const int baseNx = ctx.baseNx;
  const int baseNy = ctx.baseNy;
  const int baseNz = ctx.baseNz;
  const size_t globalNvox = ctx.globalNvox;

  std::vector<double> denom(globalNvox, 0.0);
  // PSF operator model is P = A * G, so sensitivity must be G^T A^T 1
  // (matches numerator G^T A^T r and the compact-symmetric path).
  if (!opts.bedAttenFactors.empty()) {
    for (int bed = 0; bed < bedCount; ++bed) {
      std::vector<double> localDenom = computeNormAT_Atten<MatrixType>(
          matrix, opts.bedAttenFactors[static_cast<size_t>(bed)],
          args.numAngles, opts.threads);
      apply_psf_backward(localDenom, baseNx, baseNy, baseNz, opts, psfPtr,
                         opts.threads);
      spect::app::accumulateBedVolumeToGlobal(
          denom, localDenom, baseNx, baseNy, baseNz,
          bedZStarts[static_cast<size_t>(bed)]);
    }
  } else {
    std::vector<double> localDenom = computeNormAT(matrix, opts.threads);
    apply_psf_backward(localDenom, baseNx, baseNy, baseNz, opts, psfPtr,
                       opts.threads);
    for (int bed = 0; bed < bedCount; ++bed) {
      spect::app::accumulateBedVolumeToGlobal(
          denom, localDenom, baseNx, baseNy, baseNz,
          bedZStarts[static_cast<size_t>(bed)]);
    }
  }

  const std::vector<std::vector<int>> subsets =
      make_simple_subsets(matrix.rows * bedCount, opts.n_subsets);
  if (subsets.empty())
    throw std::runtime_error("Failed to create projection subsets.");
  const auto isOrderedFullSubsetRows = [&](const std::vector<int> &subsetRows) {
    const int64_t totalRows =
        static_cast<int64_t>(matrix.rows) * static_cast<int64_t>(bedCount);
    if (subsetRows.size() != static_cast<size_t>(totalRows))
      return false;
    for (int64_t row = 0; row < totalRows; ++row) {
      if (subsetRows[static_cast<size_t>(row)] != row)
        return false;
    }
    return true;
  };
  const bool reuseFullSubsetBsremDenom =
      args.algorithm == ReconArgs::Algorithm::BSREM && opts.n_subsets == 1 &&
      isOrderedFullSubsetRows(subsets.front());

  std::vector<double> image(globalNvox, 1.0);
  std::vector<double> gradReg(globalNvox, 0.0);
  std::vector<double> subsetDenom;
  std::vector<double> bsremBuffer(globalNvox, 0.0);
  std::vector<double> curvMap;
  std::vector<double> kappaMap = computeKappaMap(denom, args, maskPtr);
  const std::vector<uint8_t> *maskForReg =
      (maskPtr && args.useReg && args.reg.skip_outside_mask) ? maskPtr
                                                             : nullptr;
  const bool useRegularization =
      args.useReg && args.reg.beta > 0.0 && args.reg.type != RegType::None;
  ReconstructionStageTiming stageTiming;
  resetOSEMStageTiming();
  resetBSREMStageTiming();
  spect::cuda::resetCudaChunkStageTimingCUDA();

  std::unique_ptr<IOptimizer> optimizer;
  if (args.algorithm == ReconArgs::Algorithm::BSREM) {
    optimizer = std::make_unique<BSREMOptimizer>();
  } else if (args.algorithm == ReconArgs::Algorithm::ADMM) {
    ADMMConfig admmCfg;
    admmCfg.rho = args.admmRho;
    admmCfg.reg = args.reg;
    admmCfg.beta = useRegularization ? args.reg.beta : 0.0;
    admmCfg.mask = maskForReg;
    admmCfg.threads = opts.threads;
    admmCfg.tvInnerIters = args.admmTvInner;
    admmCfg.adaptive = args.admmAdaptive;
    auto admm = std::make_unique<ADMMOptimizer>(std::move(admmCfg));
    admm->setDimensions(args.Nx, args.Ny, args.Nz);
    optimizer = std::move(admm);
  } else {
    optimizer = std::make_unique<MLEMOptimizer>();
  }

  if (args.josephPasses > 0) {
    applyJosephLike(denom, args.Nx, args.Ny, args.Nz, args.josephPasses,
                    opts.threads, false);
    logInfo("Denom smoothing (Joseph-like) complete.");
  }
  if (reuseFullSubsetBsremDenom) {
    logInfo("BSREM full-subset sensitivity reuses precomputed denom.");
  }

  const OutputGeometry geometry = ctx.outputGeometry;
  const std::string outputPrefix = outputPrefixFromArgs(args);
  const bool canProjPSF =
      opts.psf.mode == filters::PSFMode::ProjDomain && opts.n_subsets == 1 &&
      bedCount == 1 && psfPtr != nullptr && opts.det_nz > 0 &&
      opts.det_nx > 0 &&
      static_cast<size_t>(opts.det_nz) * static_cast<size_t>(opts.det_nx) *
              static_cast<size_t>(opts.num_angles) ==
          projections.size();

  if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
    if (ctx.resolvedBackend == SparseBackendChoice::CUDA && args.numIters > 1 &&
        !canProjPSF) {
      const auto warmupStart = ProfileClock::now();
      const bool warmed = spect::cuda::warmupFullSubsetChunkCacheCUDA(
          matrix, bedCount,
          opts.bedAttenFactors.empty() ? nullptr : &opts.bedAttenFactors,
          args.numAngles, opts.threads);
      if (warmed) {
        stageTiming.cudaChunkWarmupSeconds +=
            std::chrono::duration<double>(ProfileClock::now() - warmupStart)
                .count();
        ++stageTiming.cudaChunkWarmupCalls;
        logInfo("CUDA chunk warmup complete.");
      }
    }
  }

  for (int it = 1; it <= args.numIters; ++it) {
    logInfo("Iteration " + std::to_string(it) + "/" +
            std::to_string(args.numIters) +
            (canProjPSF ? " (Proj-domain PSF path)" : " (standard path)"));

    if (canProjPSF) {
      const std::vector<double> *gradPtr = nullptr;
      if (useRegularization) {
        const auto stageStart = ProfileClock::now();
        if (!tryComputeRegGradientCUDA(image, args, opts, maskForReg, gradReg)) {
          spect::regularizer::compute_reg_gradient(
              image, args.Nx, args.Ny, args.Nz, args.reg, maskForReg, gradReg,
              opts.threads);
        }
        stageTiming.regGradientSeconds +=
            std::chrono::duration<double>(ProfileClock::now() - stageStart)
                .count();
        ++stageTiming.regGradientCalls;
        gradPtr = &gradReg;
      }
      runProjDomainMLEMStep(matrix, projections, image, denom, args, opts,
                            *psfPtr, gradPtr, maskForReg, maskPtr,
                            prepared.butterConf, prepared.smoothConf,
                            ctx.spacing);
    } else {
      for (int subsetIndex = 0; subsetIndex < opts.n_subsets; ++subsetIndex) {
        const std::vector<int> &subsetRows =
            subsets[static_cast<size_t>(subsetIndex)];
        const std::vector<double> *gradPtr = nullptr;
        if (useRegularization) {
          const auto stageStart = ProfileClock::now();
          if (!tryComputeRegGradientCUDA(image, args, opts, maskForReg,
                                         gradReg)) {
            spect::regularizer::compute_reg_gradient(
                image, args.Nx, args.Ny, args.Nz, args.reg, maskForReg,
                gradReg, opts.threads);
          }
          stageTiming.regGradientSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - stageStart)
                  .count();
          ++stageTiming.regGradientCalls;
          gradPtr = &gradReg;
        }

        const std::vector<double> *activeSubsetDenom = &denom;
        double bsremAlpha = 1.0;
        double bsremBetaScaled = 0.0;
        const std::vector<double> *curvPtr = nullptr;
        const std::vector<double> *kappaPtr = nullptr;

        if (args.algorithm == ReconArgs::Algorithm::BSREM) {
          const int k = (it - 1) * opts.n_subsets + subsetIndex;
          bsremAlpha =
              args.bsrem_alpha0 /
              (1.0 + static_cast<double>(args.bsrem_gamma) * k);
          bsremBetaScaled =
              useRegularization ? (args.reg.beta / static_cast<double>(opts.n_subsets))
                                : 0.0;
          if (reuseFullSubsetBsremDenom) {
            activeSubsetDenom = &denom;
          } else {
            const auto subsetSensitivityStart = ProfileClock::now();
            subsetDenom = computeSubsetSensitivity<MatrixType>(
                matrix, subsetRows, static_cast<int>(globalNvox), opts.threads,
                opts);
            stageTiming.subsetSensitivitySeconds +=
                std::chrono::duration<double>(ProfileClock::now() -
                                              subsetSensitivityStart)
                    .count();
            ++stageTiming.subsetSensitivityCalls;
            apply_psf_backward(subsetDenom, args.Nx, args.Ny, args.Nz, opts,
                               psfPtr, opts.threads);
            if (args.josephPasses > 0) {
              applyJosephLike(subsetDenom, args.Nx, args.Ny, args.Nz,
                              args.josephPasses, opts.threads, false);
            }
            activeSubsetDenom = &subsetDenom;
          }

          if (args.bsremUseCurvDenom) {
            const bool refreshCurvature =
                curvMap.empty() || !args.bsremFreezeCurv ||
                it <= std::max(1, args.bsremCurvWarmupIters);
            if (refreshCurvature) {
              curvMap.assign(globalNvox, 0.0);
              for (size_t i = 0; i < globalNvox; ++i) {
                const double denomBase = std::max(image[i], args.bsremCurvEps);
                curvMap[i] = std::abs(gradReg[i]) / denomBase;
              }
            }
            curvPtr = &curvMap;
          }

          kappaPtr = args.bsremUseKappaMap ? &kappaMap : nullptr;
        }

        StepParams sp;
        if constexpr (std::is_same_v<MatrixType, spect::io::CSRView>) {
          sp.matrixView = &matrix;
        } else {
          sp.matrixCSR = &matrix;
        }
        sp.subsetRows = &subsetRows;
        sp.projections = &projections;
        sp.image = &image;
        sp.denom = &denom;
        sp.Nx = args.Nx;
        sp.Ny = args.Ny;
        sp.Nz = args.Nz;
        sp.opts = &opts;
        sp.psfKernels = psfPtr;
        sp.gradReg = gradPtr;
        sp.beta = useRegularization ? args.reg.beta : 0.0;
        sp.subsetDenom = activeSubsetDenom;
        sp.bsremAlpha = bsremAlpha;
        sp.bsremBetaScaled = bsremBetaScaled;
        sp.curvature = curvPtr;
        sp.kappa = kappaPtr;
        sp.bsremBuffer = &bsremBuffer;

        const auto stepStart = ProfileClock::now();
        optimizer->step(sp);
        const double stepSec =
            std::chrono::duration<double>(ProfileClock::now() - stepStart)
                .count();
        if (args.algorithm == ReconArgs::Algorithm::BSREM) {
          stageTiming.bsremCoreSeconds += stepSec;
          ++stageTiming.bsremCoreCalls;
        } else {
          stageTiming.osemCoreSeconds += stepSec;
          ++stageTiming.osemCoreCalls;
        }

        applyJosephLike(image, args.Nx, args.Ny, args.Nz, args.josephPasses,
                        opts.threads, true);

        if (prepared.butterConf.enabled &&
            prepared.butterConf.apply_after_each_subset) {
          const auto stageStart = ProfileClock::now();
          filters::apply_butterworth_inplace(
              image, args.Nx, args.Ny, args.Nz, ctx.spacing,
              prepared.butterConf, opts.threads);
          stageTiming.butterworthSeconds +=
              std::chrono::duration<double>(ProfileClock::now() - stageStart)
                  .count();
          ++stageTiming.butterworthCalls;
        }
      }
    }

    optimizer->postIteration(image, it);

    if (prepared.butterConf.enabled &&
        !prepared.butterConf.apply_after_each_subset) {
      const auto stageStart = ProfileClock::now();
      filters::apply_butterworth_inplace(image, args.Nx, args.Ny, args.Nz,
                                         ctx.spacing, prepared.butterConf,
                                         opts.threads);
      stageTiming.butterworthSeconds +=
          std::chrono::duration<double>(ProfileClock::now() - stageStart)
              .count();
      ++stageTiming.butterworthCalls;
    }

    if (prepared.smoothConf.enabled) {
      const auto stageStart = ProfileClock::now();
      applySeparableConvRuntime(image, args.Nx, args.Ny, args.Nz,
                                prepared.smoothConf.kx, prepared.smoothConf.ky,
                                prepared.smoothConf.kz, opts, opts.threads);
      if (maskPtr)
        spect::core::applyMaskInplace(image, *maskPtr);
      stageTiming.smoothingSeconds +=
          std::chrono::duration<double>(ProfileClock::now() - stageStart)
              .count();
      ++stageTiming.smoothingCalls;
    }

    clampNonNegativeInPlace(image, opts.threads);
    if (maskPtr)
      spect::core::applyMaskInplace(image, *maskPtr);

    if (args.saveInterval > 0 &&
        (it % args.saveInterval == 0 || it == args.numIters)) {
      const auto stageStart = ProfileClock::now();
      saveVolumeOutput(image, geometry, args.outputDir, outputPrefix, it,
                       maskPtr, ctx.crop, true, ctx.outputTransform);
      stageTiming.saveSeconds +=
          std::chrono::duration<double>(ProfileClock::now() - stageStart)
              .count();
      ++stageTiming.saveCalls;
      logDebug("Saved volume at iteration " + std::to_string(it));
    }
  }

  if (args.saveInterval <= 0 || (args.numIters % args.saveInterval) != 0) {
    const auto stageStart = ProfileClock::now();
    saveVolumeOutput(image, geometry, args.outputDir, outputPrefix,
                     args.numIters, maskPtr, ctx.crop, true,
                     ctx.outputTransform);
    stageTiming.saveSeconds +=
        std::chrono::duration<double>(ProfileClock::now() - stageStart).count();
    ++stageTiming.saveCalls;
  }

  if (useRegularization) {
    const double penalty = spect::regularizer::compute_reg_energy(
        image, args.Nx, args.Ny, args.Nz, args.reg, maskForReg);
    const auto metric = app::computePoissonLogLikelihoodRuntime(
        matrix, image, projections, opts, args.Nx, args.Ny, args.Nz, psfPtr,
        opts.threads);
    logInfo(app::formatMetricLine(args.reg.beta, metric.likelihood, penalty,
                                  metric.modelTag));
  }

  logInfo(formatProfileEntry("reg_gradient", stageTiming.regGradientSeconds,
                             stageTiming.regGradientCalls));
  logInfo(formatProfileEntry("subset_sensitivity",
                             stageTiming.subsetSensitivitySeconds,
                             stageTiming.subsetSensitivityCalls));
  logInfo(formatProfileEntry("osem_core", stageTiming.osemCoreSeconds,
                             stageTiming.osemCoreCalls));
  logInfo(formatProfileEntry("bsrem_core", stageTiming.bsremCoreSeconds,
                             stageTiming.bsremCoreCalls));
  logInfo(formatProfileEntry("butterworth", stageTiming.butterworthSeconds,
                             stageTiming.butterworthCalls));
  logInfo(formatProfileEntry("smoothing", stageTiming.smoothingSeconds,
                             stageTiming.smoothingCalls));
  logInfo(formatProfileEntry("save_volume", stageTiming.saveSeconds,
                             stageTiming.saveCalls));
  logInfo(formatProfileEntry("cuda.chunk_warmup",
                             stageTiming.cudaChunkWarmupSeconds,
                             stageTiming.cudaChunkWarmupCalls));

  const BSREMStageTimingSnapshot bsremTiming = getBSREMStageTimingSnapshot();
  const OSEMStageTimingSnapshot osemTiming = getOSEMStageTimingSnapshot();
  logInfo(formatProfileEntry("osem.input_psf", osemTiming.inputPSFSeconds,
                             osemTiming.calls));
  logInfo(formatProfileEntry("osem.forward", osemTiming.forwardSeconds,
                             osemTiming.calls));
  logInfo(formatProfileEntry("osem.backprojection",
                             osemTiming.backprojectionSeconds,
                             osemTiming.calls));
  logInfo(formatProfileEntry("osem.cuda_fused", osemTiming.cudaFusedSeconds,
                             osemTiming.cudaFusedCalls));
  logInfo(formatProfileEntry("osem.buffer_psf", osemTiming.bufferPSFSeconds,
                             osemTiming.calls));
  logInfo(formatProfileEntry("osem.update", osemTiming.updateSeconds,
                             osemTiming.calls));
  logInfo(formatProfileEntry("bsrem.input_psf", bsremTiming.inputPSFSeconds,
                             bsremTiming.calls));
  logInfo(formatProfileEntry("bsrem.forward", bsremTiming.forwardSeconds,
                             bsremTiming.calls));
  logInfo(formatProfileEntry("bsrem.backprojection",
                             bsremTiming.backprojectionSeconds,
                             bsremTiming.calls));
  logInfo(formatProfileEntry("bsrem.cuda_fused", bsremTiming.cudaFusedSeconds,
                             bsremTiming.cudaFusedCalls));
  logInfo(formatProfileEntry("bsrem.buffer_psf", bsremTiming.bufferPSFSeconds,
                             bsremTiming.calls));
  logInfo(formatProfileEntry("bsrem.update", bsremTiming.updateSeconds,
                             bsremTiming.calls));
  const spect::cuda::CudaChunkStageTimingSnapshot cudaChunkTiming =
      spect::cuda::getCudaChunkStageTimingSnapshotCUDA();
  logCudaChunkTimingSnapshot(cudaChunkTiming);

  if (!args.attMapPath.empty() && !args.doCTAC) {
    const spect::main::CTACModel changModel =
        (args.ctacModel == "per-voxel") ? spect::main::CTACModel::PerVoxelApprox
                                        : spect::main::CTACModel::LineIntegral;
    if (changModel == spect::main::CTACModel::PerVoxelApprox) {
      logWarn("Chang's AC output uses the legacy per-voxel exp(-mu) "
              "approximation (-ctacModel per-voxel).");
    } else {
      logInfo("Chang's AC output uses angle-averaged parallel-ray line "
              "integrals of the attenuation map (mu in mm^-1).");
    }
    std::vector<double> muMap;
    if (!sharedAnatomical.empty() && args.attMapPath == args.priorPath) {
      muMap = sharedAnatomical;
    } else {
      muMap = loadVolumeChecked(args.attMapPath, "Attenuation map", args.Nx,
                                args.Ny, args.Nz);
    }
    std::vector<double> acImage = image;
    applyChangsAC(acImage, muMap, args.Nx, args.Ny, args.Nz, ctx.spacing.dx,
                  ctx.spacing.dy, ctx.spacing.dz, args.numAngles, opts.threads,
                  changModel, args.ctacRotSign);
    saveVolumeOutput(acImage, geometry, args.outputDir, outputPrefix + "_AC",
                     -1, maskPtr, ctx.crop, true, ctx.outputTransform);
  }

  return image;
}

} // namespace

StandardReconLoop::StandardReconLoop(StandardReconLoopContext context)
    : context_(context) {}

std::vector<double> StandardReconLoop::run(const io::CSR &matrix) const {
  return runStandardReconstructionImpl(context_, matrix);
}

std::vector<double> StandardReconLoop::run(const io::CSRView &matrix) const {
  return runStandardReconstructionImpl(context_, matrix);
}

} // namespace spect::main
