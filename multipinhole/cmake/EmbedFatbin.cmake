# Compile a device-only .cu to a fatbin for SPECT_CUDA_ARCHITECTURES and
# embed it as a C++ byte array on the given target.

if(NOT CUDAToolkit_NVCC_EXECUTABLE)
  find_program(CUDAToolkit_NVCC_EXECUTABLE NAMES nvcc
    HINTS "${CUDAToolkit_BIN_DIR}" "${CUDAToolkit_TARGET_DIR}/bin")
endif()

function(spect_embed_cuda_fatbin target cu_file symbol)
  if(NOT CUDAToolkit_NVCC_EXECUTABLE)
    message(FATAL_ERROR "CUDAToolkit_NVCC_EXECUTABLE is required to build helper fatbins")
  endif()
  set(_dir "${CMAKE_BINARY_DIR}/cuda_fatbin")
  set(_fatbin "${_dir}/${symbol}.fatbin")
  set(_cpp "${_dir}/${symbol}.cpp")
  file(MAKE_DIRECTORY "${_dir}")
  set(_gencodes)
  foreach(_arch IN LISTS SPECT_CUDA_ARCHITECTURES)
    list(APPEND _gencodes "-gencode=arch=compute_${_arch},code=sm_${_arch}")
  endforeach()
  add_custom_command(
    OUTPUT "${_fatbin}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
            --fatbin
            -std=c++17
            -Xcompiler=/utf-8
            ${_gencodes}
            -o "${_fatbin}"
            "${cu_file}"
    DEPENDS "${cu_file}"
    COMMENT "Fatbin ${symbol} for sm_${SPECT_CUDA_ARCHITECTURES}"
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${_cpp}"
    COMMAND "${CMAKE_COMMAND}"
            "-DINPUT=${_fatbin}"
            "-DOUTPUT=${_cpp}"
            "-DSYMBOL=${symbol}"
            -P "${SPECT_RECON_SOURCE_DIR}/cmake/EmbedBin.cmake"
    DEPENDS "${_fatbin}" "${SPECT_RECON_SOURCE_DIR}/cmake/EmbedBin.cmake"
    COMMENT "Embed fatbin ${symbol}"
    VERBATIM
  )
  set_source_files_properties("${_cpp}" PROPERTIES GENERATED TRUE)
  target_sources(${target} PRIVATE "${_cpp}")
endfunction()
