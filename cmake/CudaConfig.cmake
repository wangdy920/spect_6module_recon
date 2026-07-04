# ----------------------------------
# 【T0.1】重构选项
# ----------------------------------
option(SPECT_BUILD_TESTS "Build unit tests with GoogleTest" ON)
option(SPECT_DETERMINISTIC "Enable bitwise deterministic mode for multi-threading" OFF)
option(SPECT_CSR_STRICT_VALIDATE "Enable strict CSR validation checks" ON)
option(SPECT_ENABLE_CUDA_SPARSEOPS "Enable CUDA sparse backend integration" OFF)
option(SPECT_BUILD_CUDA_SCAFFOLD "Build the experimental CUDA sparse backend scaffold target" OFF)
option(SPECT_ENABLE_BUTTERWORTH_CUFFT "Enable cuFFT backend support for Butterworth filtering" OFF)
set(SPECT_BUTTERWORTH_DEFAULT_BACKEND "auto" CACHE STRING "Default Butterworth backend: auto|fftw|cufft")
set_property(CACHE SPECT_BUTTERWORTH_DEFAULT_BACKEND PROPERTY STRINGS auto fftw cufft)