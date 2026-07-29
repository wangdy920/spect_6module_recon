#pragma once

#include "cuda/CudaDriverHelpers.h"

#include <cuda.h>
#include <nvrtc.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace spect::cuda {

inline std::string compileNvrtcProgramToPtx(const char *source,
                                           const char *programName,
                                           int optionCount,
                                           const char *const *options) {
  int nvrtcMajor = 0;
  int nvrtcMinor = 0;
  SPECT_NVRTC_CHECK(nvrtcVersion(&nvrtcMajor, &nvrtcMinor));

  std::fprintf(stderr, "[INFO] NVRTC compile %s (NVRTC %d.%d",
               programName ? programName : "(unnamed)", nvrtcMajor,
               nvrtcMinor);
  for (int i = 0; i < optionCount; ++i) {
    if (options && options[i]) {
      std::fprintf(stderr, ", %s", options[i]);
    }
  }
  std::fprintf(stderr, ")\n");

  nvrtcProgram program = nullptr;
  SPECT_NVRTC_CHECK(
      nvrtcCreateProgram(&program, source, programName, 0, nullptr, nullptr));

  const nvrtcResult compileStatus =
      nvrtcCompileProgram(program, optionCount, options);

  size_t logSize = 0;
  SPECT_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &logSize));
  std::string log(logSize, '\0');
  if (logSize > 1) {
    SPECT_NVRTC_CHECK(nvrtcGetProgramLog(program, log.data()));
  }

  if (compileStatus != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error(std::string("NVRTC compile failed: ") + log);
  }

  size_t ptxSize = 0;
  SPECT_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptxSize));
  std::string ptx(ptxSize, '\0');
  SPECT_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
  SPECT_NVRTC_CHECK(nvrtcDestroyProgram(&program));
  return ptx;
}

inline void loadNvrtcPtxModule(CUmodule *module, const std::string &ptx) {
  SPECT_CUDRV_CHECK(cuModuleLoadDataEx(module, ptx.data(), 0, nullptr, nullptr));
}

} // namespace spect::cuda