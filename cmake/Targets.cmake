# ----------------------------------
# MSVC UTF-8 编码支持 (消除 C4819 警告)
# ----------------------------------
if(MSVC)
  add_compile_options(/utf-8)
endif()

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
# 接口库不得依赖任何第三方库 (matio/cufft/omp等)

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
add_library(spect_sparse STATIC
  src/sparse/SparseOpsDefaults.cpp
)
target_include_directories(spect_sparse PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_sparse PUBLIC spect_interfaces)
if(OpenMP_CXX_FOUND)
  target_link_libraries(spect_sparse PUBLIC OpenMP::OpenMP_CXX)
endif()
add_library(spect::sparse ALIAS spect_sparse)

if(SPECT_BUILD_CUDA_SCAFFOLD)
  add_library(spect_sparse_cuda STATIC
    src/cuda/SparseOpsCUDA.cpp
    src/cuda/CudaSparseOpsFacade.cpp
    src/cuda/CudaChunkCache.cpp
    src/cuda/CudaCsrBuilder.cpp
    src/cuda/CudaProjKernelLauncher.cpp
    src/cuda/SparseOpsCudaKernels.cpp
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
  src/app/BsremUpdate.cpp
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
  src/main/BsremKappaMap.cpp
  src/main/CTACOps.cpp
  src/main/CompactSymmetricReconLoop.cpp
  src/main/MainProgram.cpp
  src/main/MatVarPath.cpp
  src/main/ReconDataLoader.cpp
  src/main/ReconPreparation.cpp
  src/main/ReconArgsPrinter.cpp
  src/main/ReconLoopHelpers.cpp
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
  spect_sparse
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
if(NOT _hdf5_targets)
  message(FATAL_ERROR "HDF5 targets not resolved; ensure vcpkg/toolchain is configured")
endif()
if(NOT _zlib_target)
  message(FATAL_ERROR "ZLIB target not resolved; ensure vcpkg/toolchain is configured")
endif()

target_link_libraries(SPECT_recon PRIVATE
  ${_matio_target}
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
