#pragma once

#include <cuda.h>
#include <nvrtc.h>

#if __has_include(<cufft.h>)
#include <cufft.h>
#define SPECT_CUDA_HELPERS_HAVE_CUFFT 1
#endif

#include <stdexcept>
#include <string>

namespace spect::cuda {

inline void checkCuDriver(CUresult status, const char *expr, const char *file,
                          int line) {
  if (status != CUDA_SUCCESS) {
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    throw std::runtime_error(std::string("CUDA driver error at ") + file +
                             ":" + std::to_string(line) + " for " + expr +
                             " (" + (name ? name : "unknown") + ": " +
                             (message ? message : "unknown") + ")");
  }
}

inline void checkNvrtc(nvrtcResult status, const char *expr, const char *file,
                       int line) {
  if (status != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string("NVRTC error at ") + file + ":" +
                             std::to_string(line) + " for " + expr + " (" +
                             nvrtcGetErrorString(status) + ")");
  }
}

#if defined(SPECT_CUDA_HELPERS_HAVE_CUFFT)
inline void checkCufft(cufftResult status, const char *expr, const char *file,
                       int line) {
  if (status != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string("cuFFT error at ") + file + ":" +
                             std::to_string(line) + " for " + expr +
                             " (status=" +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}
#endif

} // namespace spect::cuda

#define SPECT_CUDRV_CHECK(call)                                               \
  ::spect::cuda::checkCuDriver((call), #call, __FILE__, __LINE__)
#define SPECT_NVRTC_CHECK(call)                                               \
  ::spect::cuda::checkNvrtc((call), #call, __FILE__, __LINE__)
#if defined(SPECT_CUDA_HELPERS_HAVE_CUFFT)
#define SPECT_CUFFT_CHECK(call)                                               \
  ::spect::cuda::checkCufft((call), #call, __FILE__, __LINE__)
#endif
