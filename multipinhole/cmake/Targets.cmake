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
  $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(spect_interfaces INTERFACE cxx_std_17)
# 接口库不得依赖任何第三方库 (matio/cufft/omp等)

# 便捷alias (用于外部项目)
add_library(spect::interfaces ALIAS spect_interfaces)

message(STATUS "SPECT modular libraries: spect_interfaces, spect_filters, spect_regularizer, spect_cli, spect_io, spect_app, spect_pipeline")


if(SPECT_BUILD_CUDA_SCAFFOLD)
  add_library(spect_sparse_cuda STATIC
    src/sparse/cuda/SparseOpsCUDA.cpp
    src/sparse/cuda/CudaChunkCache.cpp
    src/sparse/cuda/CudaCsrBuilder.cpp
    src/sparse/cuda/CudaProjKernelLauncher.cpp
    src/sparse/cuda/CompactSymmetricProjectorCUDA.cpp
  )
  target_include_directories(spect_sparse_cuda PUBLIC
    $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(spect_sparse_cuda PUBLIC spect_interfaces)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(spect_sparse_cuda PUBLIC OpenMP::OpenMP_CXX)
  endif()
  if(SPECT_ENABLE_CUDA_SPARSEOPS)
    target_compile_definitions(spect_sparse_cuda PUBLIC
      SPECT_HAS_CUDA=1
      SPECT_ENABLE_CUDA_SPARSEOPS=1
      SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL=1
    )
    target_link_libraries(spect_sparse_cuda PUBLIC CUDA::cudart CUDA::cusparse CUDA::cuda_driver)
    include(${SPECT_RECON_SOURCE_DIR}/cmake/EmbedFatbin.cmake)
    spect_embed_cuda_fatbin(spect_sparse_cuda
      ${SPECT_RECON_SOURCE_DIR}/src/sparse/cuda/projdomain_chunk_ops.cu
      spect_fatbin_projdomain)
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
    src/filters/cuda/VolumeOpsCUDA.cpp
  )
  if(NOT COMMAND spect_embed_cuda_fatbin)
    include(${SPECT_RECON_SOURCE_DIR}/cmake/EmbedFatbin.cmake)
  endif()
  spect_embed_cuda_fatbin(spect_filters
    ${SPECT_RECON_SOURCE_DIR}/src/filters/cuda/volume_ops_cuda.cu
    spect_fatbin_volume_ops)
endif()
if(SPECT_ENABLE_BUTTERWORTH_CUFFT)
  target_sources(spect_filters PRIVATE
    src/filters/cuda/ButterworthCUDA.cpp
  )
  if(NOT COMMAND spect_embed_cuda_fatbin)
    include(${SPECT_RECON_SOURCE_DIR}/cmake/EmbedFatbin.cmake)
  endif()
  spect_embed_cuda_fatbin(spect_filters
    ${SPECT_RECON_SOURCE_DIR}/src/filters/cuda/butterworth_mask.cu
    spect_fatbin_butterworth)
endif()
target_include_directories(spect_filters PUBLIC
  $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_filters PUBLIC spect_interfaces)
if(TARGET CUDA::cudart)
  target_link_libraries(spect_filters PUBLIC CUDA::cudart CUDA::cuda_driver)
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
# spect_regularizer: 生产正则化器库（仅 TV / RDP）
# ----------------------------------
add_library(spect_regularizer STATIC
  src/regularizer/Regularizers.cpp
)
target_include_directories(spect_regularizer PUBLIC
  $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}/include>
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
  $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}>
  $<BUILD_INTERFACE:${SPECT_GENERATED_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_cli PUBLIC spect_interfaces)
# CLI依赖: Filters.h (PSFConf), Regularizers.h (RegConf)
add_library(spect::cli ALIAS spect_cli)

# ----------------------------------
# spect_io: IO库 (ProjectionLoader, MAT读取, CSR构建, MHD写入等)
# 【T2.2】从SPECT-recon.cpp拆分的数据加载代码 + 从 dataLoader.h 迁移
# ----------------------------------
add_library(spect_io STATIC
  src/io/ProjectionLoader.cpp
  src/io/MatFileReader.cpp
  src/io/MatVarPath.cpp
  src/io/MHDWriter.cpp
  src/io/CompactSymmetryMetadata.cpp
  src/io/CompactChunkCSR.cpp
)
target_include_directories(spect_io PUBLIC
  $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_io PUBLIC spect_interfaces)
# IO依赖matio
target_link_libraries(spect_io PUBLIC ${_matio_target})
add_library(spect::io ALIAS spect_io)

# ----------------------------------
# spect_pipeline: 重建流水线 (recon 算法层 / ctac 物理项 / postprocess / pipeline 编排)
# SPECT_recon、spect_tests 与 tools/ 共用同一份目标文件，避免同源重复编译
# ----------------------------------
add_library(spect_pipeline STATIC
  src/recon/BsremKappaMap.cpp
  src/recon/BsremDiagnostics.cpp
  src/recon/BsremUpdate.cpp
  src/recon/CompactSymmetricReconLoop.cpp
  src/recon/ProjectionOps.cpp
  src/recon/ReconLoopHelpers.cpp

  src/ctac/AutoMuMap.cpp
  src/ctac/CTACFactorCache.cpp
  src/ctac/CTACOps.cpp

  src/postprocess/SmoothingOps.cpp

  src/pipeline/MainProgram.cpp
  src/pipeline/ReconArgsPrinter.cpp
  src/pipeline/ReconDataLoader.cpp
  src/pipeline/ReconPreparation.cpp
  src/pipeline/ReconSetup.cpp
  src/pipeline/RunLogger.cpp
  src/pipeline/RunLogging.cpp
  src/pipeline/RunManifest.cpp
  src/pipeline/VolumeOutput.cpp
)
target_include_directories(spect_pipeline PUBLIC
  $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${SPECT_RECON_SOURCE_DIR}>
  $<BUILD_INTERFACE:${SPECT_GENERATED_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(spect_pipeline PUBLIC
  spect_io
  spect_cli
  spect_filters
  spect_regularizer
)
if(SPECT_BUILD_CUDA_SCAFFOLD AND SPECT_ENABLE_CUDA_SPARSEOPS)
  target_link_libraries(spect_pipeline PUBLIC spect_sparse_cuda)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(spect_pipeline PUBLIC OpenMP::OpenMP_CXX)
endif()
if(MSVC)
  # 与旧 SPECT_recon 目标保持一致，否则重建热循环会失去 AVX2/fp:fast 而改变数值与性能
  target_compile_options(spect_pipeline PRIVATE /MP /EHsc /permissive- /bigobj)
  target_compile_options(spect_pipeline PRIVATE
    $<$<CONFIG:Release>:/O2 /Oi /Ot /fp:fast /arch:AVX2>
  )
  # 这两个文件原属 spect_app，从未启用 /fp:fast；它们是锁定的物理项，
  # 放宽浮点模型会改变生产数值，故显式钉回 /fp:precise（MSVC 取最后一个 /fp 开关）。
  set_source_files_properties(
    src/recon/ProjectionOps.cpp
    src/recon/BsremUpdate.cpp
    src/recon/BsremDiagnostics.cpp
    PROPERTIES COMPILE_OPTIONS "$<$<CONFIG:Release>:/fp:precise>"
  )
endif()
add_library(spect::pipeline ALIAS spect_pipeline)

# ----------------------------------
# 9) 生成可执行文件
# ----------------------------------
# 使用重构后的模块化代码
# 注意: Filters.cpp 和 Regularizers.cpp 已经编译在 spect_filters/spect_regularizer 库中
#       不要在这里重复包含，否则会导致 ODR 违规
add_executable(SPECT_recon
  src/main/main.cpp
)

# 链接模块库 (获取include目录)
target_link_libraries(SPECT_recon PRIVATE
  spect_pipeline
)
target_include_directories(SPECT_recon PRIVATE
  $<BUILD_INTERFACE:${SPECT_GENERATED_INCLUDE_DIR}>
)

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
          "${SPECT_RECON_SOURCE_DIR}/config/spect_config.ini"
          "$<TARGET_FILE_DIR:SPECT_recon>/config/spect_config.ini"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SPECT_RECON_SOURCE_DIR}/config/bad_dector"
          "$<TARGET_FILE_DIR:SPECT_recon>/config/bad_dector"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SPECT_RECON_SOURCE_DIR}/config/algo_mlem.conf"
          "$<TARGET_FILE_DIR:SPECT_recon>/config/algo_mlem.conf"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SPECT_RECON_SOURCE_DIR}/config/algo_bsrem.conf"
          "$<TARGET_FILE_DIR:SPECT_recon>/config/algo_bsrem.conf"
  COMMENT "Copying runtime config files beside SPECT_recon.exe")

# Bundle CUDA 12.8+/13.x redistributables so RTX 30/40/50 field machines get
# matching runtime libraries. Helper kernels ship as sm_86/89/120 fatbins.
if((SPECT_BUILD_CUDA_SCAFFOLD AND SPECT_ENABLE_CUDA_SPARSEOPS) OR SPECT_ENABLE_BUTTERWORTH_CUFFT)
  include(${SPECT_RECON_SOURCE_DIR}/cmake/CudaRedist.cmake)
  spect_bundle_cuda_runtime_for_target(SPECT_recon)
endif()


add_executable(spect_forward_project
  ${CMAKE_SOURCE_DIR}/tools/forward_project_multibed.cpp
)
target_include_directories(spect_forward_project PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(spect_forward_project PRIVATE spect_pipeline)
if(SPECT_BUILD_CUDA_SCAFFOLD AND SPECT_ENABLE_CUDA_SPARSEOPS)
  target_link_libraries(spect_forward_project PRIVATE spect_sparse_cuda)
endif()
set_target_properties(spect_forward_project PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Phase B.2 microbenchmark: CSR(A) op=T vs true CSR(A^T) op=N on one real
# compact chunk. Read-only evidence tool; not part of the reconstruction path.
if(SPECT_BUILD_CUDA_SCAFFOLD AND SPECT_ENABLE_CUDA_SPARSEOPS)
  add_executable(spect_transpose_bench
    ${CMAKE_SOURCE_DIR}/tools/transpose_bench.cpp
  )
  target_include_directories(spect_transpose_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(spect_transpose_bench PRIVATE
    spect_io CUDA::cudart CUDA::cusparse
  )
  if(OpenMP_CXX_FOUND)
    target_link_libraries(spect_transpose_bench PRIVATE OpenMP::OpenMP_CXX)
  endif()
  set_target_properties(spect_transpose_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
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
