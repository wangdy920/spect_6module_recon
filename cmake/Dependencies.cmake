include(GNUInstallDirs)   # 【改动】防止未定义 CMAKE_INSTALL_BINDIR

if((SPECT_BUILD_CUDA_SCAFFOLD AND SPECT_ENABLE_CUDA_SPARSEOPS) OR SPECT_ENABLE_BUTTERWORTH_CUFFT)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")
endif()

# 条件编译定义
if(SPECT_DETERMINISTIC)
  add_compile_definitions(SPECT_DETERMINISTIC=1)
endif()
if(SPECT_CSR_STRICT_VALIDATE)
  add_compile_definitions(SPECT_CSR_STRICT_VALIDATE=1)
endif()
if(SPECT_ENABLE_CUDA_SPARSEOPS)
  add_compile_definitions(SPECT_ENABLE_CUDA_SPARSEOPS=1)
endif()
if(SPECT_ENABLE_BUTTERWORTH_CUFFT)
  add_compile_definitions(SPECT_BUTTERWORTH_HAVE_CUFFT=1)
endif()
if(SPECT_BUTTERWORTH_DEFAULT_BACKEND STREQUAL "cufft")
  add_compile_definitions(SPECT_BUTTERWORTH_DEFAULT_BACKEND_CUFFT=1)
elseif(SPECT_BUTTERWORTH_DEFAULT_BACKEND STREQUAL "fftw")
  add_compile_definitions(SPECT_BUTTERWORTH_DEFAULT_BACKEND_FFTW=1)
else()
  add_compile_definitions(SPECT_BUTTERWORTH_DEFAULT_BACKEND_AUTO=1)
endif()

# 诊断输出
message(STATUS "Generator: ${CMAKE_GENERATOR}")
message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "VCPKG triplet: ${VCPKG_TARGET_TRIPLET}")
message(STATUS "CUDA sparse backend option: ${SPECT_ENABLE_CUDA_SPARSEOPS}")
message(STATUS "CUDA scaffold target option: ${SPECT_BUILD_CUDA_SCAFFOLD}")
message(STATUS "Butterworth cuFFT option: ${SPECT_ENABLE_BUTTERWORTH_CUFFT}")
message(STATUS "Butterworth default backend: ${SPECT_BUTTERWORTH_DEFAULT_BACKEND}")
if(SPECT_ENABLE_CUDA_SPARSEOPS AND NOT SPECT_BUILD_CUDA_SCAFFOLD)
  message(WARNING "SPECT_ENABLE_CUDA_SPARSEOPS is ON, but no external CUDA sparse backend is wired yet; define SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL together with getSparseOpsCUDA().")
endif()

# ----------------------------------
# 1) OpenMP（可选）
# ----------------------------------
find_package(OpenMP QUIET)

# ----------------------------------
# 2) ITK (Removed)
# ----------------------------------
# find_package(ITK CONFIG REQUIRED ...)
# include(${ITK_USE_FILE})

# ----------------------------------
# 3) FFTW3：优先 CONFIG（vcpkg），失败回退 pkg-config / find_library
# ----------------------------------
set(_fftw_targets)
set(_use_fftw_threads OFF)

if(WIN32)  # Windows 仅使用 vcpkg 提供的 CONFIG
  find_package(FFTW3 CONFIG REQUIRED)
  list(APPEND _fftw_targets FFTW3::fftw3)
  
  # Check for threading targets (OpenMP or Threads)
  if(TARGET FFTW3::fftw3_omp)
    message(STATUS "Found FFTW3 OpenMP target")
    list(APPEND _fftw_targets FFTW3::fftw3_omp)
  elseif(TARGET FFTW3::fftw3_threads)
    message(STATUS "Found FFTW3 Threads target")
    list(APPEND _fftw_targets FFTW3::fftw3_threads)
  else()
    # Fallback: find library manually
    find_library(FFTW3_OMP_LIB NAMES fftw3_omp libfftw3_omp)
    if(FFTW3_OMP_LIB)
       message(STATUS "Found fftw3_omp library: ${FFTW3_OMP_LIB}")
       add_library(fftw3_omp UNKNOWN IMPORTED)
       set_target_properties(fftw3_omp PROPERTIES IMPORTED_LOCATION "${FFTW3_OMP_LIB}")
       list(APPEND _fftw_targets fftw3_omp)
    else()
       find_library(FFTW3_THREADS_LIB NAMES fftw3_threads libfftw3_threads)
       if(FFTW3_THREADS_LIB)
           message(STATUS "Found fftw3_threads library: ${FFTW3_THREADS_LIB}")
           add_library(fftw3_threads UNKNOWN IMPORTED)
           set_target_properties(fftw3_threads PROPERTIES IMPORTED_LOCATION "${FFTW3_THREADS_LIB}")
           list(APPEND _fftw_targets fftw3_threads)
       else()
           message(WARNING "FFTW3 threading lib not found. Linking might fail.")
       endif()
    endif()
  endif()
  
  # Define USE_FFTW_THREADS if any threading target/lib was found
  if(TARGET FFTW3::fftw3_omp OR TARGET FFTW3::fftw3_threads OR FFTW3_OMP_LIB OR FFTW3_THREADS_LIB)
      set(_use_fftw_threads ON)
  endif()

  find_package(Threads REQUIRED)
  list(APPEND _fftw_targets Threads::Threads)

else()      # Ubuntu / WSL / 其他 Linux
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3)
  list(APPEND _fftw_targets PkgConfig::FFTW)

  # Ubuntu 没有 fftw3_threads.pc，直接找库
  find_library(FFTW_THREADS_LIB NAMES fftw3_threads
               PATHS /usr/lib /usr/lib/x86_64-linux-gnu)
  if(NOT FFTW_THREADS_LIB)
    message(FATAL_ERROR "fftw3_threads not found. sudo apt install libfftw3-dev libfftw3-threads-dev")
  endif()
  add_library(fftw3_threads UNKNOWN IMPORTED GLOBAL)
  set_target_properties(fftw3_threads PROPERTIES IMPORTED_LOCATION "${FFTW_THREADS_LIB}")
  list(APPEND _fftw_targets fftw3_threads)
  set(_use_fftw_threads ON)

  find_package(Threads REQUIRED)
  list(APPEND _fftw_targets Threads::Threads)
endif()

# ----------------------------------
# Path Hints for vcpkg
# ----------------------------------
# list(APPEND CMAKE_PREFIX_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

# ----------------------------------
# 4) HDF5 (vcpkg / CONFIG 模式)
# ----------------------------------
# 注意：vcpkg 导出的包名通常是小写 hdf5，target 也是 hdf5::hdf5 这种格式
# COMPONENTS 对 vcpkg 的 config 影响不大，可以省略
find_package(hdf5 CONFIG REQUIRED)

set(_hdf5_targets)

# C 接口主库
if(TARGET hdf5::hdf5)
  list(APPEND _hdf5_targets hdf5::hdf5)
endif()
# vcpkg shared naming
if(TARGET hdf5::hdf5-shared)
  list(APPEND _hdf5_targets hdf5::hdf5-shared)
endif()

# High-level C 接口（如果你需要 HL 功能）
if(TARGET hdf5::hdf5_hl)
  list(APPEND _hdf5_targets hdf5::hdf5_hl)
endif()
# vcpkg shared naming
if(TARGET hdf5::hdf5_hl-shared)
  list(APPEND _hdf5_targets hdf5::hdf5_hl-shared)
endif()

# 如果你安装了 cpp feature，还可以顺带连上：
if(TARGET hdf5::hdf5_cpp)
  list(APPEND _hdf5_targets hdf5::hdf5_cpp)
endif()
if(TARGET hdf5::hdf5_hl_cpp)
  list(APPEND _hdf5_targets hdf5::hdf5_hl_cpp)
endif()
# vcpkg shared naming
if(TARGET hdf5::hdf5_cpp-shared)
  list(APPEND _hdf5_targets hdf5::hdf5_cpp-shared)
endif()
if(TARGET hdf5::hdf5_hl_cpp-shared)
  list(APPEND _hdf5_targets hdf5::hdf5_hl_cpp-shared)
endif()

# 调试用：看看到底找到了哪些 target
message(STATUS "HDF5 targets collected: ${_hdf5_targets}")


# ----------------------------------
# 5) ZLIB
# ----------------------------------
find_package(ZLIB REQUIRED) 
set(_zlib_target)
if(TARGET ZLIB::ZLIB)
  set(_zlib_target ZLIB::ZLIB)
elseif(ZLIB_LIBRARIES)
  set(_zlib_target ${ZLIB_LIBRARIES})
endif()

# ----------------------------------
# 6) CURL
# ----------------------------------
find_package(CURL QUIET)
set(_curl_target)
if(TARGET CURL::libcurl)
  set(_curl_target CURL::libcurl)
elseif(CURL_FOUND AND CURL_LIBRARIES)
  set(_curl_target ${CURL_LIBRARIES})
endif()

# ----------------------------------
# 7) matio
# ----------------------------------
find_package(matio CONFIG QUIET)

if(TARGET matio::matio)
    message(STATUS "Found matio via CONFIG: matio::matio")
    set(_matio_target matio::matio)
elseif(TARGET Unofficial::matio::matio)
    message(STATUS "Found matio via CONFIG: Unofficial::matio::matio")
    set(_matio_target Unofficial::matio::matio)
else()
    # Fallback
    set(_matio_hints)
    if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
        list(APPEND _matio_hints "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
    elseif(DEFINED VCPKG_INSTALLED_DIR)
        list(APPEND _matio_hints "${VCPKG_INSTALLED_DIR}")
    endif()

    find_path(MATIO_INCLUDE_DIR 
        NAMES matio.h 
        HINTS ${_matio_hints}
        PATH_SUFFIXES include
    )
    find_library(MATIO_LIBRARY 
        NAMES libmatio matio 
        HINTS ${_matio_hints}
        PATH_SUFFIXES lib
    )
    
    if(MATIO_INCLUDE_DIR AND MATIO_LIBRARY)
        add_library(matio UNKNOWN IMPORTED GLOBAL)
        set_target_properties(matio PROPERTIES IMPORTED_LOCATION "${MATIO_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${MATIO_INCLUDE_DIR}")
        set(_matio_target matio)
        message(STATUS "Found matio (Generic): ${MATIO_LIBRARY}")
    else()
        message(FATAL_ERROR "matio not found. Install via vcpkg (e.g. vcpkg install matio:x64-windows) and ensure vcpkg toolchain is used.")
    endif()
endif()

# ----------------------------------
# 8) 可选 VTK
# ----------------------------------
option(ENABLE_VTK "Enable VTK-based utilities" OFF)
if(ENABLE_VTK)
  find_package(VTK 9 CONFIG REQUIRED
    COMPONENTS CommonCore CommonDataModel IOImage)
endif()

# Force vcpkg include path globally (BEFORE add_executable)
# include_directories("${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")