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

include(cmake/CudaConfig.cmake)
include(cmake/Dependencies.cmake)
include(cmake/Targets.cmake)
include(cmake/Install.cmake)
include(cmake/Testing.cmake)
