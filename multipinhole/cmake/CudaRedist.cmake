# Collect CUDA redistributable DLLs for Windows field packages.
# Helper kernels are build-time fatbins; field packages need cudart/cusparse/cufft,
# not the NVRTC JIT stack.

function(spect_collect_cuda_runtime_dlls out_var)
  set(_dlls)
  if(NOT WIN32 OR NOT SPECT_BUNDLE_CUDA_RUNTIME)
    set(${out_var} "${_dlls}" PARENT_SCOPE)
    return()
  endif()
  if(NOT SPECT_CUDA_REDIST_DIR)
    set(${out_var} "${_dlls}" PARENT_SCOPE)
    return()
  endif()

  set(_patterns
    "cudart64_*.dll"
    "cusparse64_*.dll"
  )
  if(SPECT_ENABLE_BUTTERWORTH_CUFFT)
    list(APPEND _patterns "cufft64_*.dll")
  endif()

  foreach(_pat IN LISTS _patterns)
    file(GLOB _matches "${SPECT_CUDA_REDIST_DIR}/${_pat}")
    list(APPEND _dlls ${_matches})
  endforeach()

  list(REMOVE_DUPLICATES _dlls)
  set(${out_var} "${_dlls}" PARENT_SCOPE)
endfunction()

function(spect_bundle_cuda_runtime_for_target target_name)
  if(NOT WIN32 OR NOT SPECT_BUNDLE_CUDA_RUNTIME)
    return()
  endif()
  spect_collect_cuda_runtime_dlls(_cuda_dlls)
  if(NOT _cuda_dlls)
    message(WARNING
      "SPECT_BUNDLE_CUDA_RUNTIME is ON but no CUDA DLLs were found under "
      "'${SPECT_CUDA_REDIST_DIR}'")
    return()
  endif()

  set(_copy_cmds)
  foreach(_dll IN LISTS _cuda_dlls)
    list(APPEND _copy_cmds
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${_dll}"
              "$<TARGET_FILE_DIR:${target_name}>")
  endforeach()

  add_custom_command(TARGET ${target_name} POST_BUILD
    ${_copy_cmds}
    COMMENT "Bundling CUDA redistributable DLLs beside ${target_name}")

  list(LENGTH _cuda_dlls _n)
  message(STATUS "Will bundle ${_n} CUDA runtime DLL(s) with ${target_name}")
endfunction()
