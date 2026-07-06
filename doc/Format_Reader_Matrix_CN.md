# 6module 格式 × 读者能力矩阵（PR26，2026-07-05）

> 状态：待 Fable 5 / Opus 4.8 review。Canonical 生产路径为 **260601 chunk CSRB + `-chunkCSRDir`**；260525 单文件与 v2 row-wise 工具链保持 legacy/reference-only（§2a 不退场决策）。

| 格式 / 入口 | 生成端 | 磁盘布局 | 主线读者 | 独立 MLEM (`spect_recon_mlem.cu`) | 转换工具 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| v4 chunk CSRB 目录 | `spect_sysmat_PSC_6module_24pin_YSO_v4.cu` | `metadata.bin`(88B+) + `chunk_*.csrb` | ✅ `-chunkCSRDir` | ❌ | — | **Canonical**；260601 |
| v2 row-wise chunk BIN | `spect_sysmat_..._v2.cu` | `metadata.bin` + `chunk_*.bin` | ❌ 直接读 | ✅（legacy） | ✅ `spect_convert_6module_sysmat_to_csrb` | 输出 compact 行 CSRB，不展开对称 |
| 260525 单文件 compact CSRB | 转换工具 / 历史数据 | `matrix_6module.csrb` + 76B metadata | ✅ `-loadCSR -compactMeta` | ❌ | — | Legacy；大文件隐式 mmap（PR3） |
| standard 单文件 CSRB | 无对称压缩 | `.csrb` full rows | ✅ `-loadCSR` (standard layout) | ❌ | — | 260525 对照 / CPU smoke |
| 投影 BIN | `mat_convert_bin.m` | header + float64[] | ✅ `-df` | ✅ | — | 函数化：`mat_convert_bin(...)` |

## metadata 版本

| 大小 | 语义 | 状态 |
| --- | --- | --- |
| 76B | 260525 legacy compact | 保留至 §2a 显式退场 |
| 88B | 260601 ExtendedV1 (Sym90Z 等) | Canonical |
| 96B | ExtendedV2 | 按文件大小强制读取 |

## §2a 决策（2026-07-05）

**不退场，只做兼容保护。** 转换工具、260525 对照脚本、single-file compact CUDA 路径均保留；PR28 删除动作不启动。

## 参考

- `6module_bak_src/README.md`
- `doc/SystemMatrix_6module_compact_vs_CSRB_CN.md`
- `tests/test_convert_6module_sysmat.cpp`（toy chunk → CSRB 回归）
