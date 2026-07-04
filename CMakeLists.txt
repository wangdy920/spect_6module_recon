cmake_minimum_required(VERSION 3.20)

# 更稳的包查找行为
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
      "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# (Duplicate matio check removed)

project(SPECT_recon LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)  # 【T0.1】改为C++17，符合重构规范
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)   # 【改动】防止未定义 CMAKE_INSTALL_BINDIR

# ----------------------------------
# MSVC UTF-8 编码支持 (消除 C4819 警告)
# ----------------------------------
if(MSVC)
  add_compile_options(/utf-8)
endif()

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

# ==================================
# 【T1.3】模块化库定义
# ==================================
# 依赖方向: algo -> core -> interfaces
# 禁止: algo -> app/cli/io, interfaces -> 第三方库

# ----------------------------------
# spect_interfaces: 纯头文件接口库
# ----------------------------------
add_library(spect_interfaces INTERFACE)
target_include_directories(spect_interfaces INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(spect_interfaces INTERFACE cxx_std_17)
# 接口库不得依赖任何第三方库 (matio/fftw/omp等)

# ----------------------------------
# spect_core: 核心类型库 (CSR, Types等)
# ----------------------------------
# 当前为INTERFACE库，后续添加.cpp文件时改为STATIC
add_library(spect_core INTERFACE)
target_link_libraries(spect_core INTERFACE spect_interfaces)
# spect_core将来包含: CSRMap.cpp, CSRBinary.cpp, SparseOpsRef.cpp等

# ----------------------------------
# spect_algo: 算法库 (MLEM/OSEM/正则化器等)
# ----------------------------------
# 当前为INTERFACE库，后续重构时添加.cpp
add_library(spect_algo INTERFACE)
target_link_libraries(spect_algo INTERFACE 
  spect_core
  spect_filters
  spect_regularizer
)
# spect_algo将来包含: MLEMReconstructor.cpp, Regularizers.cpp等
# 约束: 只能依赖spect_core/spect_interfaces，不能依赖io/cli/app

# 便捷alias (用于外部项目)
add_library(spect::interfaces ALIAS spect_interfaces)
add_library(spect::core ALIAS spect_core)
add_library(spect::algo ALIAS spect_algo)

message(STATUS "SPECT modular libraries: spect_interfaces, spect_core, spect_algo")

# ----------------------------------
# spect_sparse: 稀疏矩阵运算库 (SparseOpsRef, SparseOpsCPU等)
# 【T3.1】统一SpMV实现
# ----------------------------------
add_library(spect_sparse INTERFACE)
target_include_directories(spect_sparse INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_sparse INTERFACE spect_interfaces)
add_library(spect::sparse ALIAS spect_sparse)

if(SPECT_BUILD_CUDA_SCAFFOLD)
  add_library(spect_sparse_cuda STATIC
    src/cuda/SparseOpsCUDA.cpp
    src/cuda/CompactSymmetricProjectorCUDA.cpp
  )
  target_include_directories(spect_sparse_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(spect_sparse_cuda PUBLIC spect_interfaces spect_sparse)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(spect_sparse_cuda PUBLIC OpenMP::OpenMP_CXX)
  endif()
  if(SPECT_ENABLE_CUDA_SPARSEOPS)
    target_compile_definitions(spect_sparse_cuda PUBLIC
      SPECT_HAS_CUDA=1
      SPECT_ENABLE_CUDA_SPARSEOPS=1
      SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL=1
    )
    target_link_libraries(spect_sparse_cuda PUBLIC CUDA::cudart CUDA::cusparse CUDA::cuda_driver CUDA::nvrtc)
  endif()
  add_library(spect::sparse_cuda ALIAS spect_sparse_cuda)
endif()

# ----------------------------------
# spect_filters: 滤波器库 (PSF, Butterworth, 卷积等)
# 【NEW】从 original_source_backup/Filters.cpp 迁移
# ----------------------------------
add_library(spect_filters STATIC
  src/filters/Filters.cpp
)
if(TARGET CUDA::cudart)
  target_sources(spect_filters PRIVATE
    src/cuda/VolumeOpsCUDA.cpp
  )
endif()
if(SPECT_ENABLE_BUTTERWORTH_CUFFT)
  target_sources(spect_filters PRIVATE
    src/cuda/ButterworthCUDA.cpp
  )
endif()
target_include_directories(spect_filters PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_filters PUBLIC spect_interfaces)
# 依赖 FFTW3 和 OpenMP - PUBLIC 以传递头文件路径
if(_fftw_targets)
  target_link_libraries(spect_filters PUBLIC ${_fftw_targets})
endif()
if(TARGET CUDA::cudart)
  target_link_libraries(spect_filters PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
  target_compile_definitions(spect_filters PUBLIC
    SPECT_HAS_CUDA=1
  )
endif()
if(SPECT_ENABLE_BUTTERWORTH_CUFFT)
  target_link_libraries(spect_filters PUBLIC CUDA::cufft)
  target_compile_definitions(spect_filters PUBLIC SPECT_BUTTERWORTH_HAVE_CUFFT=1)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(spect_filters PUBLIC OpenMP::OpenMP_CXX)
endif()
add_library(spect::filters ALIAS spect_filters)

# ----------------------------------
# spect_regularizer: 正则化器库 (TV, Tikhonov, LogCosh等)
# 【T4.1】正则化器继承体系 + 从 original_source_backup 迁移实现
# ----------------------------------
add_library(spect_regularizer STATIC
  src/regularizer/Regularizers.cpp
)
target_include_directories(spect_regularizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_regularizer PUBLIC spect_interfaces)
if(OpenMP_CXX_FOUND)
  target_link_libraries(spect_regularizer PUBLIC OpenMP::OpenMP_CXX)
endif()
add_library(spect::regularizer ALIAS spect_regularizer)

# ----------------------------------
# spect_cli: CLI解析库 (ArgumentParser, ConfigParser等)
# 【T2.1】从SPECT-recon.cpp拆分的CLI解析代码
# ----------------------------------
add_library(spect_cli STATIC
  src/cli/ArgumentParser.cpp
)
target_include_directories(spect_cli PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_cli PUBLIC spect_core)
# CLI依赖: Filters.h (PSFConf), Regularizers.h (RegConf)
add_library(spect::cli ALIAS spect_cli)

# ----------------------------------
# spect_io: IO库 (ProjectionLoader, MAT读取, CSR构建, MHD写入等)
# 【T2.2】从SPECT-recon.cpp拆分的数据加载代码 + 从 dataLoader.h 迁移
# ----------------------------------
add_library(spect_io STATIC
  src/io/ProjectionLoader.cpp
  src/io/MatFileReader.cpp
  src/io/CSRBuilder.cpp
  src/io/MHDWriter.cpp
  src/io/CompactSymmetryMetadata.cpp
  src/io/CompactChunkCSR.cpp
)
target_include_directories(spect_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_io PUBLIC spect_core)
# IO依赖matio
target_link_libraries(spect_io PUBLIC ${_matio_target})
add_library(spect::io ALIAS spect_io)

# ----------------------------------
# spect_app: 应用层 (ProjectionOps, OSEM/BSREM步骤等)
# 【T2.3】从MLEMReconstructor.h迁移的重建算法实现
# ----------------------------------
add_library(spect_app STATIC
  src/app/ProjectionOps.cpp
  src/app/OSEMReconstructor.cpp
  src/app/BSREMReconstructor.cpp
)
target_include_directories(spect_app PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_app PUBLIC 
  spect_filters
  spect_regularizer
  spect_io
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(spect_app PUBLIC OpenMP::OpenMP_CXX)
endif()
# MSVC AVX2 optimization for SIMD in reconstructors
if(MSVC)
  target_compile_options(spect_app PRIVATE $<$<CONFIG:Release>:/arch:AVX2>)
endif()
add_library(spect::app ALIAS spect_app)

# ----------------------------------
# 9) 生成可执行文件
# ----------------------------------
# 使用重构后的模块化代码
# 注意: Filters.cpp 和 Regularizers.cpp 已经编译在 spect_filters/spect_regularizer 库中
#       不要在这里重复包含，否则会导致 ODR 违规
add_executable(SPECT_recon
  src/main/AutoMuMap.cpp
  src/main/CTACOps.cpp
  src/main/MainProgram.cpp
  src/main/MatVarPath.cpp
  src/main/ReconDataLoader.cpp
  src/main/ReconPreparation.cpp
  src/main/ReconArgsPrinter.cpp
  src/main/ReconSetup.cpp
  src/main/RunLogging.cpp
  src/main/RunLogger.cpp
  src/main/SmoothingOps.cpp
  src/main/StandardReconLoop.cpp
  src/main/VolumeOutput.cpp
  src/main/main.cpp
)

# 链接模块库 (获取include目录)
target_link_libraries(SPECT_recon PRIVATE 
  spect_algo
  spect_app
  spect_io
  spect_cli
  spect_filters
  spect_regularizer
)

if(SPECT_BUILD_CUDA_SCAFFOLD AND SPECT_ENABLE_CUDA_SPARSEOPS)
  target_link_libraries(SPECT_recon PRIVATE spect_sparse_cuda)
endif()

# 添加include目录
target_include_directories(SPECT_recon PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 【改动】统一输出名（去掉 Windows 的 .exe 后缀感知）
set_target_properties(SPECT_recon PROPERTIES
  OUTPUT_NAME "SPECT_recon"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_custom_command(TARGET SPECT_recon POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:SPECT_recon>/config"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/config/spect_config.ini"
          "$<TARGET_FILE_DIR:SPECT_recon>/config/spect_config.ini"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/config/algo_mlem.conf"
          "$<TARGET_FILE_DIR:SPECT_recon>/config/algo_mlem.conf"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/config/algo_bsrem.conf"
          "$<TARGET_FILE_DIR:SPECT_recon>/config/algo_bsrem.conf"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/config/algo_admm.conf"
          "$<TARGET_FILE_DIR:SPECT_recon>/config/algo_admm.conf"
  COMMENT "Copying runtime config files beside SPECT_recon.exe")

add_executable(spect_convert_6module_sysmat_to_csrb
  tools/convert_6module_sysmat_to_csrb.cpp
)
target_include_directories(spect_convert_6module_sysmat_to_csrb PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(spect_convert_6module_sysmat_to_csrb PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# FFTW threads support
if(_use_fftw_threads)
  target_compile_definitions(SPECT_recon PRIVATE USE_FFTW_THREADS)
endif()

# ----------------------------------
# 10) 统一链接
# ----------------------------------
# OpenMP
if(OpenMP_CXX_FOUND)
  target_link_libraries(SPECT_recon PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(SPECT_recon PRIVATE USE_OPENMP)
endif()

if(NOT _matio_target)
  message(FATAL_ERROR "matio target not resolved; ensure vcpkg/toolchain is configured")
endif()
if(NOT _fftw_targets)
  message(FATAL_ERROR "FFTW3 targets not resolved; ensure vcpkg/toolchain is configured")
endif()
if(NOT _hdf5_targets)
  message(FATAL_ERROR "HDF5 targets not resolved; ensure vcpkg/toolchain is configured")
endif()
if(NOT _zlib_target)
  message(FATAL_ERROR "ZLIB target not resolved; ensure vcpkg/toolchain is configured")
endif()

target_link_libraries(SPECT_recon PRIVATE
  ${_matio_target}
  ${_fftw_targets}
  ${_hdf5_targets}
  ${_zlib_target}
)
if(_curl_target)
  target_link_libraries(SPECT_recon PRIVATE ${_curl_target})
endif()

# ----------------------------------
# 11) Windows MSVC 专用编译与链接优化
# ----------------------------------
if(MSVC)
    # --- 基础选项 (适用于所有模式) ---
    # /MP: 多核编译
    # /EHsc: 标准 C++ 异常处理
    # /bigobj: 允许大型对象文件 (重度模板库如 ITK 需要)
    target_compile_options(SPECT_recon PRIVATE /MP /EHsc /permissive- /bigobj)

    # --- 激进优化选项 (仅限 Release 模式) ---
    # 注意：这里使用生成表达式自动判断配置
    target_compile_options(SPECT_recon PRIVATE
        $<$<CONFIG:Release>:
            /O2         # 最大化速度
            /Oi         # 启用内部函数 (Intrinsics)
            /Ot         # 偏向代码速度优先于大小
            /fp:fast    # 快速浮点模型 (允许编译器为了速度牺牲极微小的精度，这对重建算法通常是可以接受的)
            /arch:AVX2  # 启用 AVX2 指令集 (大大加速矩阵运算)。如果你的 CPU 非常老，请改为 /arch:AVX
            /GL         # 全程序优化 (编译阶段) - 允许跨编译单元内联
        >
    )

    # --- 链接器优化 ---
    target_link_options(SPECT_recon PRIVATE
        $<$<CONFIG:Release>:
            /LTCG       # 链接时代码生成 (配合 /GL 使用)
            /OPT:REF    # 删除未引用的函数和数据
            /OPT:ICF    # 消除相同的 COMDAT (折叠重复代码)
        >
    )
endif()

# ----------------------------------
# 14) 安装规则（可选）
# ----------------------------------
install(TARGETS SPECT_recon
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY "${CMAKE_SOURCE_DIR}/config/"
  DESTINATION "${CMAKE_INSTALL_BINDIR}/config"
  FILES_MATCHING
  PATTERN "*.ini"
  PATTERN "*.conf")

# On Windows, vcpkg typically provides shared libraries (DLLs). The build step
# already places required runtime DLLs next to the executable under
# `${CMAKE_BINARY_DIR}/bin` (via vcpkg applocal). Install them alongside the
# installed executable so `out/install/.../bin/SPECT_recon.exe` is runnable.
if(WIN32)
  install(DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILES_MATCHING
    PATTERN "*.dll")
endif()

# ----------------------------------
# 【T0.1】测试配置 (GoogleTest)
# ----------------------------------
if(SPECT_BUILD_TESTS)
  # 启用CTest
  enable_testing()
  include(CTest)
  
  # 查找GoogleTest
  find_package(GTest CONFIG REQUIRED)
  
  # 测试可执行文件 - 不再依赖 original_source_backup
  add_executable(spect_tests
    src/main/AutoMuMap.cpp
    src/main/CTACOps.cpp
    src/main/MatVarPath.cpp
    src/main/ReconDataLoader.cpp
    src/main/ReconPreparation.cpp
    src/main/ReconSetup.cpp
    src/main/RunLogger.cpp
    src/main/SmoothingOps.cpp
    src/main/VolumeOutput.cpp
    tests/test_main.cpp
    tests/test_placeholder.cpp
    tests/test_csr.cpp
    tests/test_spmv.cpp
    tests/test_bsrem_validation.cpp
    tests/test_bsrem_alignment.cpp
    tests/test_osem_alignment.cpp
    tests/test_psf_modes.cpp
    tests/test_metric_output.cpp
    tests/test_mat_reader.cpp
    tests/test_filters_golden.cpp
    tests/test_regularizers.cpp
    tests/test_golden_recon.cpp
    tests/common/parser_test_helpers.cpp
    tests/test_interfaces.cpp
    tests/test_matrix_io.cpp
    tests/test_compact_symmetry.cpp
    tests/test_parser.cpp
    tests/test_multibed_cli.cpp
    tests/test_bed_offsets.cpp
    tests/test_main_auto_mu_map.cpp
    tests/test_main_ctac_ops.cpp
    tests/test_main_mat_var_path.cpp
    tests/test_main_recon_preparation.cpp
    tests/test_multibed_overlap.cpp
    tests/test_main_recon_setup.cpp
    tests/test_main_smoothing_ops.cpp
    tests/test_main_volume_output.cpp
  )

  target_compile_definitions(spect_tests PRIVATE SPECT_DETERMINISTIC=1)
  
  # 链接GoogleTest
  target_link_libraries(spect_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
  )
  
  # 【T1.3】链接模块库
  target_link_libraries(spect_tests PRIVATE spect_algo)
  
  # 【T2.1】链接CLI模块
  target_link_libraries(spect_tests PRIVATE spect_cli)
  
  # 【T2.2】链接IO模块
  target_link_libraries(spect_tests PRIVATE spect_io)
  
  # 【T2.3】链接App模块
  target_link_libraries(spect_tests PRIVATE spect_app)
  
  # 【T3.1】链接Sparse模块
  target_link_libraries(spect_tests PRIVATE spect_sparse)
  
  # 【T4.1】链接Regularizer模块
  target_link_libraries(spect_tests PRIVATE spect_regularizer)
  
  # 链接第三方依赖 (后续会迁移到spect_io等模块)
  target_link_libraries(spect_tests PRIVATE
    ${_matio_target}
    ${_fftw_targets}
    ${_hdf5_targets}
    ${_zlib_target}
  )
  if(_curl_target)
    target_link_libraries(spect_tests PRIVATE ${_curl_target})
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(spect_tests PRIVATE OpenMP::OpenMP_CXX)
  endif()
  
  if(SPECT_BUILD_CUDA_SCAFFOLD AND SPECT_ENABLE_CUDA_SPARSEOPS)
    target_link_libraries(spect_tests PRIVATE spect_sparse_cuda)
  endif()

  # 包含项目根目录以访问头文件
  target_include_directories(spect_tests PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
  )
  
  # 设置测试输出目录
  set_target_properties(spect_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  
  # 注册测试
  include(GoogleTest)
  gtest_discover_tests(spect_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES
      LABELS "unit"
    DISCOVERY_TIMEOUT 30
  )
  
  message(STATUS "Tests enabled: spect_tests will be built")

  if(SPECT_BUILD_CUDA_SCAFFOLD AND SPECT_ENABLE_CUDA_SPARSEOPS)
    add_executable(spect_cuda_scaffold_tests
      tests/test_cuda_scaffold.cpp
    )
    target_link_libraries(spect_cuda_scaffold_tests PRIVATE
      GTest::gtest
      GTest::gtest_main
      spect_sparse
      spect_app
      spect_sparse_cuda
    )
    target_link_libraries(spect_cuda_scaffold_tests PRIVATE CUDA::cuda_driver CUDA::nvrtc)
    target_include_directories(spect_cuda_scaffold_tests PRIVATE
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/include
    )
    set_target_properties(spect_cuda_scaffold_tests PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    gtest_discover_tests(spect_cuda_scaffold_tests
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      PROPERTIES LABELS "unit;cuda-scaffold"
      DISCOVERY_TIMEOUT 30
    )
    message(STATUS "CUDA scaffold tests enabled: spect_cuda_scaffold_tests will be built")
  endif()

  if(SPECT_BUILD_CUDA_SCAFFOLD AND SPECT_ENABLE_CUDA_SPARSEOPS)
    add_executable(spect_cuda_osem_compare
      tools/cuda_osem_compare.cpp
    )
    target_link_libraries(spect_cuda_osem_compare PRIVATE
      spect_app
      spect_io
      spect_sparse
      spect_sparse_cuda
    )
    target_link_libraries(spect_cuda_osem_compare PRIVATE CUDA::cuda_driver CUDA::nvrtc)
    if(WIN32)
      target_link_libraries(spect_cuda_osem_compare PRIVATE Psapi)
    endif()
    target_include_directories(spect_cuda_osem_compare PRIVATE
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/include
    )
    set_target_properties(spect_cuda_osem_compare PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    message(STATUS "CUDA OSEM compare tool enabled: spect_cuda_osem_compare will be built")

    add_executable(spect_cuda_chunk_probe
      tools/cuda_chunk_probe.cpp
    )
    target_link_libraries(spect_cuda_chunk_probe PRIVATE
      spect_app
      spect_io
      spect_sparse
      spect_sparse_cuda
    )
    target_link_libraries(spect_cuda_chunk_probe PRIVATE CUDA::cuda_driver CUDA::nvrtc)
    if(WIN32)
      target_link_libraries(spect_cuda_chunk_probe PRIVATE Psapi)
    endif()
    target_include_directories(spect_cuda_chunk_probe PRIVATE
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/include
    )
    set_target_properties(spect_cuda_chunk_probe PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    message(STATUS "CUDA chunk probe tool enabled: spect_cuda_chunk_probe will be built")
  endif()
endif()
