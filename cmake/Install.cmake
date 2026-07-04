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