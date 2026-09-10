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
    tests/test_main.cpp
    tests/test_placeholder.cpp
    tests/test_psf_modes.cpp
    tests/test_sieve_output_filters.cpp
    tests/test_metric_output.cpp
    tests/test_mat_reader.cpp
    tests/test_filters_golden.cpp
    tests/test_regularizers.cpp
    tests/test_bsrem_update.cpp
    tests/common/parser_test_helpers.cpp
    tests/test_interfaces.cpp
    tests/test_matrix_io.cpp
    tests/test_compact_symmetry.cpp
    tests/test_parser.cpp
    tests/test_multibed_cli.cpp
    tests/test_multibed_compact_numerator.cpp
    tests/test_compact_subsets.cpp
    tests/test_csrq_compact_matrix.cpp
    tests/test_bed_offsets.cpp
    tests/test_main_auto_mu_map.cpp
    tests/test_ctac_factor_cache.cpp
    tests/test_ctac_factor_parallel.cpp
    tests/test_main_ctac_ops.cpp
    tests/test_main_mat_var_path.cpp
    tests/test_main_recon_preparation.cpp
    tests/test_main_recon_setup.cpp
    tests/test_main_smoothing_ops.cpp
    tests/test_main_volume_output.cpp
    tests/test_run_manifest.cpp
  )

  target_compile_definitions(spect_tests PRIVATE SPECT_DETERMINISTIC=1)
  target_compile_definitions(spect_tests PRIVATE
    "SPECT_TEST_BINARY_DIR=\"${CMAKE_BINARY_DIR}\""
  )
  
  # 链接GoogleTest
  target_link_libraries(spect_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
  )
  
  # 【T1.3】链接模块库
  target_link_libraries(spect_tests PRIVATE spect_pipeline)

  # 【T2.1】链接CLI模块
  target_link_libraries(spect_tests PRIVATE spect_cli)
  
  # 【T2.2】链接IO模块
  target_link_libraries(spect_tests PRIVATE spect_io)
  
  
  # 【T4.1】链接Regularizer模块
  target_link_libraries(spect_tests PRIVATE spect_regularizer)
  
  # 链接第三方依赖 (后续会迁移到spect_io等模块)
  target_link_libraries(spect_tests PRIVATE
    ${_matio_target}
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
    ${SPECT_RECON_SOURCE_DIR}
    ${SPECT_RECON_SOURCE_DIR}/include
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
endif()
