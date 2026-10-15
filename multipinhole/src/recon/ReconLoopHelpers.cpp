#include "recon/ReconLoopHelpers.h"

#include "filters/cuda/VolumeOpsCUDA.h"
#include "filters/Convolution.h"
#include "pipeline/RunLogger.h"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace spect::main {

std::string formatProfileEntry(const std::string &label, double seconds,
                               size_t calls, uint64_t bytes) {
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
  logDebug(formatProfileEntry("cuda.chunk_build",
                             cudaChunkTiming.chunkBuildSeconds,
                             cudaChunkTiming.chunkBuildCalls));
  logDebug(formatProfileEntry("cuda.pinned_stage",
                             cudaChunkTiming.pinnedStageSeconds,
                             cudaChunkTiming.pinnedStageCalls));
  logDebug(formatProfileEntry("cuda.h2d_chunk", cudaChunkTiming.h2dChunkSeconds,
                             cudaChunkTiming.h2dChunkCalls));
  logDebug(formatProfileEntry("cuda.upload_wait",
                             cudaChunkTiming.uploadWaitOnComputeSeconds,
                             cudaChunkTiming.uploadWaitOnComputeCalls));
  logDebug(formatProfileEntry("cuda.fused_bed_cycle",
                             cudaChunkTiming.fusedBedCycleSeconds,
                             cudaChunkTiming.fusedBedCycleCalls));
  if (cudaChunkTiming.cusparseForwardCalls > 0) {
    logDebug(formatProfileEntry("cuda.cusparse_fwd",
                              cudaChunkTiming.cusparseForwardSeconds,
                              cudaChunkTiming.cusparseForwardCalls));
  }
  if (cudaChunkTiming.csrqDiskReadCalls > 0) {
    logDebug(formatProfileEntry("cuda.csrq_file_reread",
                              cudaChunkTiming.csrqDiskReadSeconds,
                              cudaChunkTiming.csrqDiskReadCalls,
                              cudaChunkTiming.csrqDiskReadBytes));
  }
  logDebug(formatProfileEntry("cuda.cusparse_fwd_host",
                             cudaChunkTiming.cusparseForwardHostSeconds,
                             cudaChunkTiming.cusparseForwardHostCalls));
  if (cudaChunkTiming.ratioCalls > 0) {
    logDebug(formatProfileEntry("cuda.ratio", cudaChunkTiming.ratioSeconds,
                              cudaChunkTiming.ratioCalls));
  }
  logDebug(formatProfileEntry("cuda.transpose_chunk",
                             cudaChunkTiming.transposeChunkSeconds,
                             cudaChunkTiming.transposeChunkCalls));
  if (cudaChunkTiming.cusparseBackwardCalls > 0) {
    logDebug(formatProfileEntry("cuda.cusparse_bwd",
                              cudaChunkTiming.cusparseBackwardSeconds,
                              cudaChunkTiming.cusparseBackwardCalls));
  }
  logDebug(formatProfileEntry("cuda.cusparse_bwd_host",
                             cudaChunkTiming.cusparseBackwardHostSeconds,
                             cudaChunkTiming.cusparseBackwardHostCalls));
  if (cudaChunkTiming.transformBackwardCalls > 0) {
    logDebug(formatProfileEntry("cuda.transform_bwd",
                              cudaChunkTiming.transformBackwardSeconds,
                              cudaChunkTiming.transformBackwardCalls));
  }
  if (cudaChunkTiming.compactMatrixUploadCalls > 0) {
    logDebug(formatProfileEntry("cuda.compact_matrix_upload",
                              cudaChunkTiming.compactMatrixUploadSeconds,
                              cudaChunkTiming.compactMatrixUploadCalls));
  }
  if (cudaChunkTiming.compactMatrixReuseCalls > 0) {
    logDebug(formatProfileEntry("cuda.compact_matrix_reuse_hit", 0.0,
                              cudaChunkTiming.compactMatrixReuseCalls));
  }
  if (cudaChunkTiming.compactInputH2DCalls > 0) {
    logDebug(formatProfileEntry("cuda.compact_input_h2d",
                               cudaChunkTiming.compactInputH2DSeconds,
                               cudaChunkTiming.compactInputH2DCalls));
  }
  if (cudaChunkTiming.compactProjectionH2DCalls > 0) {
    logDebug(formatProfileEntry("cuda.compact_projection_h2d",
                               cudaChunkTiming.compactProjectionH2DSeconds,
                               cudaChunkTiming.compactProjectionH2DCalls,
                               cudaChunkTiming.compactProjectionH2DBytes));
  }
  if (cudaChunkTiming.compactOutputD2HCalls > 0) {
    logDebug(formatProfileEntry("cuda.compact_output_d2h",
                               cudaChunkTiming.compactOutputD2HSeconds,
                               cudaChunkTiming.compactOutputD2HCalls));
  }
  if (cudaChunkTiming.compactForwardOutputD2HCalls > 0) {
    logDebug(formatProfileEntry(
        "cuda.compact_forward_output_d2h",
        cudaChunkTiming.compactForwardOutputD2HSeconds,
        cudaChunkTiming.compactForwardOutputD2HCalls,
        cudaChunkTiming.compactForwardOutputD2HBytes));
  }
  if (cudaChunkTiming.compactBackwardOutputD2HCalls > 0) {
    logDebug(formatProfileEntry(
        "cuda.compact_backward_output_d2h",
        cudaChunkTiming.compactBackwardOutputD2HSeconds,
        cudaChunkTiming.compactBackwardOutputD2HCalls,
        cudaChunkTiming.compactBackwardOutputD2HBytes));
  }
  if (cudaChunkTiming.deviceCacheHitCalls > 0) {
    logDebug(formatProfileEntry("cuda.device_cache_hit", 0.0,
                               cudaChunkTiming.deviceCacheHitCalls));
  }
}

std::string outputPrefixFromArgs(const ReconArgs &args) {
  return args.outputName.empty() ? std::string("recon") : args.outputName;
}

bool tryComputeRegGradientCUDA(const std::vector<double> &image,
                               const ReconArgs &args,
                               const std::vector<uint8_t> *maskForReg,
                               std::vector<double> &gradReg) {
  return cuda::computeRegGradientCUDA(image, args.Nx, args.Ny, args.Nz,
                                      args.reg, maskForReg, gradReg);
}

bool tryApplySeparableConvCUDA(std::vector<double> &volume, int nx, int ny,
                               int nz, const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz) {
  return cuda::isCudaVolumeOpsAvailable() &&
         cuda::applySeparableConv3DInplaceCUDA(volume, nx, ny, nz, kx, ky, kz);
}

void applySeparableConvRuntime(std::vector<double> &volume, int nx, int ny,
                               int nz, const std::vector<double> &kx,
                               const std::vector<double> &ky,
                               const std::vector<double> &kz, int threads) {
  (void)threads;
  if (!tryApplySeparableConvCUDA(volume, nx, ny, nz, kx, ky, kz)) {
    throw std::runtime_error(
        "3D separable convolution requires CUDA volume helpers.");
  }
}

void guardZeroMaxCheckpoint(const std::vector<double> &volume,
                            const std::string &label) {
  bool finite = true;
  double maxValue = 0.0;
  for (const double value : volume) {
    if (!std::isfinite(value)) {
      finite = false;
      break;
    }
    maxValue = std::max(maxValue, value);
  }
  if (finite && maxValue == 0.0) {
    const std::string message =
        "checkpoint max==0 at " + label + "; output is all zero.";
    logWarn(message);
    const char *failClosed = std::getenv("SPECT_FAIL_ON_ZERO_CHECKPOINT");
    if (failClosed && failClosed[0] == '1')
      throw std::runtime_error(message +
                               " Set SPECT_FAIL_ON_ZERO_CHECKPOINT=0 to continue.");
  }
}

} // namespace spect::main
