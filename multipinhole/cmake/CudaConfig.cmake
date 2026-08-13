# ----------------------------------
# CUDA options + consumer RTX arch list
# ----------------------------------
option(SPECT_BUILD_TESTS "Build unit tests with GoogleTest" ON)
option(SPECT_DETERMINISTIC "Enable bitwise deterministic mode for multi-threading" OFF)
option(SPECT_ENABLE_CUDA_SPARSEOPS "Enable CUDA sparse backend integration" ON)
option(SPECT_BUILD_CUDA_SCAFFOLD "Build CUDA sparse backend target" ON)
option(SPECT_ENABLE_BUTTERWORTH_CUFFT "Enable cuFFT backend support for Butterworth filtering" ON)
option(SPECT_BUNDLE_CUDA_RUNTIME
  "Copy CUDA redistributable DLLs next to SPECT_recon.exe (required for field GPUs)" ON)

# Consumer GeForce RTX architectures required by every CUDA-enabled build.
# Native CUDA targets receive all three entries. NVRTC kernels still compile
# at runtime for the live device CC, so the package must also bundle CUDA
# 12.8+ / 13.x NVRTC for compute_120 support.
#
#   86  — RTX 30-series (Ampere GeForce)
#   89  — RTX 40-series (Ada, e.g. 4070)
#   120 — RTX 50-series (Blackwell GeForce, e.g. 5060/5070)
#
# Driver floor for bundled CUDA 13.x runtime: CUDA driver API >= 13.0
# (NVIDIA display driver family R580+). Newer drivers remain backward compatible.
set(SPECT_CUDA_ARCHITECTURES "86;89;120" CACHE STRING
  "Required RTX 30/40/50 CUDA architectures (sm_XX)")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "${SPECT_CUDA_ARCHITECTURES}" CACHE STRING
    "CMake CUDA architectures (mirrors SPECT_CUDA_ARCHITECTURES)")
endif()

set(_spect_required_cuda_architectures 86 89 120)
foreach(_spect_arch IN LISTS _spect_required_cuda_architectures)
  list(FIND SPECT_CUDA_ARCHITECTURES "${_spect_arch}" _spect_spect_arch_index)
  list(FIND CMAKE_CUDA_ARCHITECTURES "${_spect_arch}" _spect_cmake_arch_index)
  if(_spect_spect_arch_index EQUAL -1 OR _spect_cmake_arch_index EQUAL -1)
    message(FATAL_ERROR
      "CUDA builds must include RTX 30/40/50 architectures 86;89;120. "
      "SPECT_CUDA_ARCHITECTURES='${SPECT_CUDA_ARCHITECTURES}', "
      "CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'.")
  endif()
endforeach()

# Minimum toolkit that can emit / JIT compute_120 (RTX 50-series).
# Prefer building on CUDA 13.x hosts so redistributed NVRTC matches field drivers ~13.0.
set(SPECT_CUDA_MIN_TOOLKIT_VERSION "12.8" CACHE STRING
  "Minimum CUDA Toolkit version required when CUDA backends are enabled")
set(SPECT_CUDA_MIN_DRIVER_API_VERSION "13000" CACHE STRING
  "Minimum cudaDriverGetVersion (1000*major+10*minor) accepted at runtime")
