@echo off
setlocal

set "REPO_ROOT=%~dp0.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

set "VCVARS=D:\win_programFiles\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "VCPKG_TOOLCHAIN=D:\win_programFiles\cppDev\vcpkg\scripts\buildsystems\vcpkg.cmake"
set "BUILD_DIR=%REPO_ROOT%\build_butter_cufft"
set "INSTALL_DIR=%REPO_ROOT%\staging\SPECT_recon_rc_dev"

call "%VCVARS%"
if errorlevel 1 exit /b %errorlevel%

cmake -S "%REPO_ROOT%" ^
  -B "%BUILD_DIR%" ^
  -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_TOOLCHAIN%" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DVCPKG_MANIFEST_DIR="%REPO_ROOT%" ^
  -DVCPKG_MANIFEST_MODE=ON ^
  -DVCPKG_APPLOCAL_DEPS=ON ^
  -DSPECT_BUILD_CUDA_SCAFFOLD=ON ^
  -DSPECT_ENABLE_CUDA_SPARSEOPS=ON ^
  -DSPECT_BUILD_TESTS=OFF ^
  -DSPECT_ENABLE_BUTTERWORTH_CUFFT=ON ^
  -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"
if errorlevel 1 exit /b %errorlevel%

cmake --build "%BUILD_DIR%" --target spect_filters SPECT_recon
exit /b %errorlevel%
