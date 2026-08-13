# ----------------------------------
# 14) 安装规则（可选）
# ----------------------------------
install(TARGETS SPECT_recon
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES "${CMAKE_SOURCE_DIR}/CHANGELOG.md"
  DESTINATION ".")

install(DIRECTORY "${SPECT_RECON_SOURCE_DIR}/config/"
  DESTINATION "${CMAKE_INSTALL_BINDIR}/config"
  FILES_MATCHING
  PATTERN "*.ini"
  PATTERN "*.conf"
  PATTERN "bad_dector")

# On Windows, vcpkg places runtime DLLs next to the executable under the
# generator-specific output directory (e.g. bin/Release for VS multi-config).
# Install them alongside the installed executable so `staging/SPECT_recon_rc_dev/bin/`
# is runnable without referencing the build tree.
# CUDA redistributables (cudart/nvrtc/cusparse/cufft) are also copied beside
# the exe at POST_BUILD when SPECT_BUNDLE_CUDA_RUNTIME=ON.
if(WIN32)
  install(DIRECTORY "$<TARGET_FILE_DIR:SPECT_recon>/"
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILES_MATCHING
    PATTERN "*.dll")
endif()
