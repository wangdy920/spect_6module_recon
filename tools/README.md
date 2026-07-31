# tools/ 独立工具

本目录存放独立的命令行工具源文件，编译为单独的可执行文件，**不参与** `SPECT_recon.exe` 主程序链接。它们用于离线转换、诊断、基准和后处理。

## 工具清单

| 源文件 | 可执行名 | 用途 | 构建方式 |
|---|---|---|---|
| `seam_postprocess.cpp` | `spect_seam_postprocess.exe` | 离线运行 `SeamPostprocess`：对已写出的 MHD 按 `phantom/zimu/mouse/derenzo/unknown` profile 做接缝后处理 | 主 CMake 构建（链接 `spect_io` + `SeamPostprocess.cpp`） |
| `forward_project_multibed.cpp` | `spect_forward_project.exe` | 匹配模型多床位前投影：把已知连续 phantom 经同一 compact-symmetric + PSF + 虚拟 Z 翻转链前投影为 `BedN.bin`，用于无噪 matched-model 闭环验证 | 主 CMake 构建 |

## 用法示例

### 接缝后处理（离线）

```powershell
.\build\bin\Release\spect_seam_postprocess.exe `
  recon\final_recon_it50.mhd `
  recon\final_recon_seam.mhd `
  mouse
```

`profile` 取值：`phantom` / `zimu` / `mouse` / `derenzo` / `unknown`。`derenzo` 可跟第 4 个参数 `strength`（默认 `0.30`）。该工具复用 `src/main/SeamPostprocess.cpp` 的同一逻辑，便于在不重跑重建的情况下验证后处理参数。

### 匹配模型前投影

```powershell
.\build\bin\Release\spect_forward_project.exe `
  --phantom phantom.json `
  --matrix-dir bin\matrix\260601 `
  --out-dir recon_out\matched_fwd `
  --beds 4 --bed-overlap 16,16,16,0,0
```

产出的 `Bed1.bin … BedN.bin` 可直接喂给 `SPECT_recon -dfList`，形成无噪 matched-model 闭环，用于分离实现缺陷与真实数据的未建模响应。

## 相关文档

- [`doc/Format_Reader_Matrix_CN.md`](../doc/Format_Reader_Matrix_CN.md) — 格式 × 读者能力矩阵
- [`6module_bak_src/README.md`](../6module_bak_src/README.md) — v2/v4 系统矩阵生成端
