# 6module_bak_src — 系统矩阵生成与 legacy 参考

本目录**不参与**主线 `SPECT_recon` CMake 构建，仅供离线生成系统矩阵与历史数值对齐。

## 文件说明

| 文件 | 状态 | 用途 |
| --- | --- | --- |
| `spect_sysmat_PSC_6module_24pin_YSO_v4.cu` | **推荐生成端** | 输出 `metadata.bin` + `chunk_*.csrb`（与 `-chunkCSRDir` 一致） |
| `spect_sysmat_PSC_6module_24pin_YSO_v2.cu` | legacy | 输出 `chunk_*.bin`（row-wise），仅用于旧转换工具 |
| `spect_recon_mlem.cu` | **deprecated / reference-only** | 独立 CUDA MLEM，只读 `.bin`；与主线逻辑已迁入 `CompactSymmetricProjectorCUDA.cpp`，保留作 legacy 数值对齐 |
| `mat_convert_bin.m` | 工具 | MAT → 投影 BIN（函数 `mat_convert_bin(...)`；示例见 `run_mat_convert_bin_example.m`） |

历史 50-iter 输出和 sensitivity 参考已移至 `test_data/legacy_ref/`。它们与
`test_data/share/` 同名文件哈希不同，因此保留为 2026-05-23 legacy 对齐参考：

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

详见 `doc/SystemMatrix_6module_compact_vs_CSRB_CN.md`。
