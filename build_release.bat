@echo off
setlocal enabledelayedexpansion

REM Release build for consumer GeForce RTX field GPUs.
REM Architectures (NVRTC JIT + any device-code targets):
REM   sm_86  RTX 30-series
REM   sm_89  RTX 40-series (4070 etc.)
REM   sm_120 RTX 50-series (5060/5070 etc.)
REM Requires CUDA Toolkit >= 12.8 (this host: 13.x). Bundles matching
REM cudart/nvrtc/cusparse/nvJitLink(/cufft) DLLs next to SPECT_recon.exe.
REM Field driver floor: CUDA 13.0+ capable (NVIDIA R580+). Newer drivers OK.
REM
REM Directory layout (avoid ambiguous "release" search hits):
REM   build\Release\                    CMake compile tree (objects, libs, logs)
REM   build\Release\bin\Release\        built exe (+ config post-build)
REM   staging\SPECT_recon_rc_dev\       cmake --install dev package (RC layout)
REM   dist\release_candidates\SPECT_recon_rc_YYYYMMDD\  frozen RC archives (not regenerated here)
REM
REM build\ and staging\ are NOT two independent builds: staging copies the same
REM Release artifact out of the build tree for QA and field deployment.

set "PROJECT_DIR=%~dp0"
set "CMAKE_BIN=D:\DevToolchain\CMake\bin"
set "VCPKG_ROOT=D:\DevToolchain\vcpkg"

if exist "!CMAKE_BIN!" set "PATH=!CMAKE_BIN!;!PATH!"

set "VCVARS=D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [ERROR] vcvars64.bat not found: %VCVARS%
    exit /b 1
)
call "%VCVARS%"

set "GIT_BIN=D:\DevToolchain\Git\cmd"
if exist "!GIT_BIN!" set "PATH=!GIT_BIN!;!PATH!"

set "VCPKG_ROOT=D:\DevToolchain\vcpkg"
set "VCPKG_DEFAULT_TRIPLET=x64-windows"
set "CMAKE_TOOLCHAIN_FILE=D:/DevToolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
set "VCPKG_INSTALLED_DIR=D:/DevToolchain/vcpkg/installed"

if not exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
    echo [ERROR] vcpkg not found: %VCPKG_ROOT%
    exit /b 1
)

where nvcc >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvcc not on PATH. Install CUDA Toolkit ^>= 12.8 and open a shell with CUDA_PATH set.
    exit /b 1
)
for /f "tokens=*" %%V in ('nvcc --version ^| findstr /R /C:"release [0-9]"') do echo [INFO] %%V

set "CONFIG=Release"
set "BUILD_DIR=%PROJECT_DIR%build\%CONFIG%"
set "INSTALL_DIR=%PROJECT_DIR%staging\SPECT_recon_rc_dev"
set "CUDA_ARCHS=86;89;120"

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo [INFO] Configure Release + CUDA compact backend ...
echo [INFO] SPECT_CUDA_ARCHITECTURES=%CUDA_ARCHS%
cmake -S "%PROJECT_DIR%." -B "%BUILD_DIR%" -G "Visual Studio 18 2026" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE="%CMAKE_TOOLCHAIN_FILE%" ^
  -DVCPKG_INSTALLED_DIR="%VCPKG_INSTALLED_DIR%" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DVCPKG_APPLOCAL_DEPS=ON ^
  -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
  -DSPECT_CUDA_ARCHITECTURES="%CUDA_ARCHS%" ^
  -DCMAKE_CUDA_ARCHITECTURES="%CUDA_ARCHS%" ^
  -DSPECT_BUNDLE_CUDA_RUNTIME=ON ^
  -DSPECT_BUILD_CUDA_SCAFFOLD=ON ^
  -DSPECT_ENABLE_CUDA_SPARSEOPS=ON ^
  -DSPECT_ENABLE_BUTTERWORTH_CUFFT=ON ^
  -DSPECT_BUILD_TESTS=OFF ^
  -DSPECT_DETERMINISTIC=OFF

if %errorlevel% neq 0 exit /b %errorlevel%

echo [INFO] Build SPECT_recon ...
cmake --build "%BUILD_DIR%" --config %CONFIG% --target SPECT_recon --parallel
if %errorlevel% neq 0 exit /b %errorlevel%

echo [INFO] Reset dev package to prevent retired files from surviving install ...
if exist "%INSTALL_DIR%" rmdir /s /q "%INSTALL_DIR%"

echo [INFO] Install to %INSTALL_DIR% ...
cmake --install "%BUILD_DIR%" --config %CONFIG%
if %errorlevel% neq 0 exit /b %errorlevel%

echo [INFO] Stage RC-layout dev package ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%scripts\stage_dev_package.ps1"
if %errorlevel% neq 0 exit /b %errorlevel%

echo [INFO] Verify bundled CUDA runtime DLLs ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$bin=Join-Path '%INSTALL_DIR%' 'bin'; $need=@('cudart','cusparse','nvrtc'); $missing=@(); foreach($n in $need){ if(-not (Get-ChildItem -LiteralPath $bin -Filter ($n+'*.dll') -ErrorAction SilentlyContinue)){ $missing+=$n } }; if($missing.Count){ Write-Host ('[ERROR] missing CUDA DLLs in package: '+($missing -join ', ')); exit 1 }; Get-ChildItem -LiteralPath $bin -Filter '*.dll' | Where-Object { $_.Name -match 'cudart|cusparse|nvrtc|cufft|nvJitLink' } | ForEach-Object { Write-Host ('[INFO]   '+$_.Name) }"
if %errorlevel% neq 0 exit /b %errorlevel%

echo [INFO] Done.
echo [INFO] Build tree:   %BUILD_DIR%\bin\%CONFIG%\SPECT_recon.exe
echo [INFO] Dev package:  %INSTALL_DIR%\bin\SPECT_recon.exe  ^(cd package root, run .\bin\SPECT_recon.exe^)
echo [INFO] RC archives:  dist\release_candidates\SPECT_recon_rc_YYYYMMDD\
echo [INFO] Target GPUs: RTX 30/40/50 consumer ^(sm_86/89/120^); driver/toolkit ~CUDA 13.x

exit /b 0
