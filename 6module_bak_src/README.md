# 6module_bak_src — 系统矩阵生成与 legacy 参考

本目录**不参与**主线 `SPECT_recon` CMake 构建，仅供离线生成系统矩阵与历史数值对齐。

## 文件说明

| 文件 | 状态 | 用途 |
| --- | --- | --- |
| `spect_sysmat_PSC_6module_24pin_YSO_v4.cu` | **推荐生成端** | 输出 `metadata.bin` + `chunk_*.csrb`（与 `-chunkCSRDir` 一致） |
| `spect_sysmat_PSC_6module_24pin_YSO_v2.cu` | legacy | 输出 `chunk_*.bin`（row-wise），仅用于旧转换工具 |
| `spect_recon_mlem.cu` | **deprecated / reference-only** | 独立 CUDA MLEM，只读 `.bin`；与主线逻辑已迁入 `CompactSymmetricProjectorCUDA.cpp`，保留作 legacy 数值对齐 |
| `mat_convert_bin.m` | 工具 | MAT → 投影 BIN（函数 `mat_convert_bin(...)`；示例见 `run_mat_convert_bin_example.m`） |

## Legacy MLEM 编译与 B1/B2 一致性验证

详见 **`doc/Legacy_MLEM_B1B2_Parity_CN.md`**。

```powershell
nvcc -O3 -std=c++14 -arch=sm_86 6module_bak_src\spect_recon_mlem.cu `
  -o build_legacy_6module\spect_recon_mlem_legacy.exe -lcusparse -lcufft

powershell -ExecutionPolicy Bypass -File scripts\run_6module_legacy_b1b2_parity.ps1 `
  -SpectExe build_codex_cuda\bin\SPECT_recon.exe -Iterations 50
```

Legacy 支持 Butterworth（`--butter-cutoff` / `--butter-mode every` 对齐主线每 iter 滤波）。验证口径为 **B1/B1、B2/B2**（legacy 当场生成 reference）；B1 需另备 v2 `chunk_*.bin` 目录，见 `doc/Legacy_MLEM_B1B2_Parity_CN.md`。

历史 50-iter 输出和 sensitivity 参考已移至 `test_data/legacy_ref/`。它们与
`test_data/share/` 同名文件哈希不同，因此保留为 2026-05-23 legacy 对齐参考：
注意：仓库 `.gitignore` 默认忽略 `test_data/`；后续如需新增 legacy reference 文件，需显式 `git add -f test_data/legacy_ref/<file>`。

| 文件 | SHA256 |
| --- | --- |
| `test_data/legacy_ref/recon_iter050.hdr` | `b3a872b4971356846e6eab8af8500326b86e2074d918d8e23316a87e2dd9cec9` |
| `test_data/legacy_ref/recon_iter050.img` | `1b3bd158d27e48e6702e00e92f5e5944ee2f61a8dbab4e3bb9de1cee404c467c` |
| `test_data/legacy_ref/recon_iter050.raw` | `2dff471ec72b91cc0efda6eba4781cc548f2de70c8b9f146433d4177df6b0b6f` |
| `test_data/legacy_ref/sensitivity.raw` | `376cc7eb9dc1a213e4b3f1009ef32cbe08bc90277a4427528b49e8e7789c5ae6` |

## v4 编译示例（Windows）

```powershell
nvcc -O3 -arch=sm_86 spect_sysmat_PSC_6module_24pin_YSO_v4.cu -o spect_sysmat_v4.exe
```

默认 `coll_modules=6`；输出目录含 `metadata.bin`（88B，SYM_90Z）与 `chunk_000.csrb` …

## 与主线关系

- **Canonical**：v4 `.csrb` + 主线 `-chunkCSRDir`
- **Legacy**：v2 `.bin` → `spect_convert_6module_sysmat_to_csrb`（compact 行，不展开对称）

### 格式 × 读取端矩阵（§5.1，canonical 裁决）

| 格式 | 产生端 | 布局 | 主线 `SPECT_recon` | 转换工具 `spect_convert_6module_sysmat_to_csrb` | 独立 `spect_recon_mlem.cu` |
| --- | --- | --- | --- | --- | --- |
| v4 `chunk_*.csrb` + `metadata.bin`(88B, SYM_90Z) | `spect_sysmat_*_v4.cu` | compact CSR chunk（对称压缩，90°Z） | ✅ **canonical**（`-chunkCSRDir`） | ❌ 无需 | ❌ **不支持** |
| v2 `chunk_*.bin` + `metadata.bin`(76B) | `spect_sysmat_*_v2.cu` | row-wise 原始 chunk | ❌ 不直接读（须先转换） | ✅ 唯一消费者 → 输出 `.csrb` | ✅ 唯一直接读取端 |
| 单文件 `matrix_6module.csrb`(260525, 45z) | 转换工具输出 | 合并 CSRB | ✅（`-loadCSR` legacy 路径） | —（自身输出） | ❌ 不支持 |

**已知且接受的限制**：v4 `.csrb` 矩阵**无法直接喂给独立 `spect_recon_mlem.cu`**（后者只解析 v2 row-wise `.bin`）。独立 MLEM 已标 deprecated / reference-only，不再跟进新格式；需要 legacy 数值对齐时，用 v2 生成端重新产出 `.bin`，或直接使用 `test_data/legacy_ref/` 中的既有参考输出。

详见 `doc/SystemMatrix_6module_compact_vs_CSRB_CN.md`。
