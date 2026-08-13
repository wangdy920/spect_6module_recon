# Multipinhole SPECT reconstruction

This directory contains the current production reconstruction implementation:
source, headers, runtime configuration, tests, maintained documentation, and
CMake target definitions.

Configure and build from the repository root so existing build, staging, test
data, report, and protected-prototype paths remain stable:

```powershell
cmake -S . -B build/Release
cmake --build build/Release --config Release
```

The protected six-module prototype remains at `../6module_bak_src/`, and large
validation inputs remain at `../test_data/`.
