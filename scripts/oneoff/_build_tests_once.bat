@echo off
call "D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat"
cd /d D:\Projects\SPECT-recon\build_codex_cuda
cmake --build . --target spect_tests --clean-first
exit /b %ERRORLEVEL%
