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
enable_testing()
set(CMAKE_CXX_STANDARD 17)  # 【T0.1】改为C++17，符合重构规范
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SPECT_RECON_SOURCE_DIR "${CMAKE_SOURCE_DIR}/multipinhole")

# ---- Build identity (used by -v / start-log banner) ----
string(TIMESTAMP SPECT_RECON_BUILD_DATE "%Y-%m-%d")
string(TIMESTAMP _spect_ver_ymd "%Y%m%d")
set(SPECT_RECON_GIT_HASH "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE SPECT_RECON_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE _spect_git_dirty
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT _spect_git_dirty STREQUAL "")
    set(SPECT_RECON_GIT_HASH "${SPECT_RECON_GIT_HASH}-dirty")
  endif()
endif()
if(SPECT_RECON_GIT_HASH STREQUAL "")
  set(SPECT_RECON_GIT_HASH "unknown")
endif()
set(SPECT_RECON_VERSION_STRING "${_spect_ver_ymd}+${SPECT_RECON_GIT_HASH}")
set(SPECT_RECON_CHANGELOG_PATH "${CMAKE_SOURCE_DIR}/CHANGELOG.md")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${SPECT_RECON_CHANGELOG_PATH}")
file(READ "${SPECT_RECON_CHANGELOG_PATH}" _spect_changelog_text)
# MSVC caps a single string literal at 16380 bytes (C2026); emit the changelog
# as adjacent raw literals split at line boundaries (so multibyte UTF-8 text is
# never cut) and let the compiler concatenate them.
string(LENGTH "${_spect_changelog_text}" _spect_changelog_total)
set(SPECT_RECON_CHANGELOG_LITERALS "")
set(_spect_pos 0)
while(_spect_pos LESS _spect_changelog_total)
  math(EXPR _spect_remaining "${_spect_changelog_total} - ${_spect_pos}")
  if(_spect_remaining GREATER 8000)
    string(SUBSTRING "${_spect_changelog_text}" ${_spect_pos} 8000 _spect_candidate)
    string(FIND "${_spect_candidate}" "\n" _spect_cut REVERSE)
    if(_spect_cut LESS 1)
      set(_spect_take 8000)
    else()
      math(EXPR _spect_take "${_spect_cut} + 1")
    endif()
  else()
    set(_spect_take ${_spect_remaining})
  endif()
  string(SUBSTRING "${_spect_changelog_text}" ${_spect_pos} ${_spect_take} _spect_chunk)
  if(NOT SPECT_RECON_CHANGELOG_LITERALS STREQUAL "")
    string(APPEND SPECT_RECON_CHANGELOG_LITERALS "\n      ")
  endif()
  string(APPEND SPECT_RECON_CHANGELOG_LITERALS
    "R\"SPECT_CHANGELOG(${_spect_chunk})SPECT_CHANGELOG\"")
  math(EXPR _spect_pos "${_spect_pos} + ${_spect_take}")
endwhile()
set(SPECT_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated/include")
configure_file(
  "${SPECT_RECON_SOURCE_DIR}/cmake/Version.h.in"
  "${SPECT_GENERATED_INCLUDE_DIR}/cli/Version.h"
  @ONLY
)
message(STATUS "SPECT_recon version: ${SPECT_RECON_VERSION_STRING}")

add_subdirectory(multipinhole)
