# ----------------------------------
# 【T0.1】重构选项
# ----------------------------------
option(SPECT_BUILD_TESTS "Build unit tests with GoogleTest" ON)
option(SPECT_DETERMINISTIC "Enable bitwise deterministic mode for multi-threading" OFF)
option(SPECT_CSR_STRICT_VALIDATE "Enable strict CSR validation checks" ON)
option(SPECT_ENABLE_CUDA_SPARSEOPS "Enable CUDA sparse backend integration" ON)
option(SPECT_BUILD_CUDA_SCAFFOLD "Build CUDA sparse backend target" ON)
option(SPECT_ENABLE_BUTTERWORTH_CUFFT "Enable cuFFT backend support for Butterworth filtering" ON)
