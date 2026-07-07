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

# 诊断输出
message(STATUS "Generator: ${CMAKE_GENERATOR}")
message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "VCPKG triplet: ${VCPKG_TARGET_TRIPLET}")
message(STATUS "CUDA sparse backend option: ${SPECT_ENABLE_CUDA_SPARSEOPS}")
message(STATUS "CUDA scaffold target option: ${SPECT_BUILD_CUDA_SCAFFOLD}")
message(STATUS "Butterworth cuFFT option: ${SPECT_ENABLE_BUTTERWORTH_CUFFT}")
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
