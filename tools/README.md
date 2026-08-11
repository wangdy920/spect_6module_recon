# tools/ 独立工具

本目录存放独立的命令行工具源文件，编译为单独的可执行文件，**不参与** `SPECT_recon.exe` 主程序链接。它们用于离线转换、诊断、基准和后处理。

## 工具清单

| 源文件 | 可执行名 | 用途 | 构建方式 |
|---|---|---|---|
| `forward_project_multibed.cpp` | `spect_forward_project.exe` | 匹配模型多床位前投影：把已知连续 phantom 经同一 compact-symmetric + PSF + 虚拟 Z 翻转链前投影为 `BedN.bin`，用于无噪 matched-model 闭环验证 | 主 CMake 构建 |

## 用法示例


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
