# 6module 格式 × 读者能力矩阵

> Canonical 生产路径：**260601 chunk CSRB**（`-chunkCSRDir` 或 `bin/matrix` 唯一目录自动发现）。260525 单文件为 legacy（不退场，仅兼容）。

| 格式 / 入口 | 生成端 | 磁盘布局 | 主线读者 | Legacy MLEM (`spect_recon_mlem*.cu`) | 转换工具 | 备注 |
|---|---|---|---|---|---|---|
| v4 chunk CSRB 目录 | `spect_sysmat_*_v4.cu` 等 | `metadata.bin`(88B+) + `chunk_*.csrb` | ✅ `-chunkCSRDir` | ❌ | — | **Canonical**；260601 |
| v2 row-wise chunk BIN | `spect_sysmat_*_v2.cu` | `metadata.bin` + `chunk_*.bin` | ❌ 直接读 | ✅ | ✅ `spect_convert_6module_sysmat_to_csrb` | 输出 compact 行 CSRB，不展开对称 |
| 260525 单文件 compact CSRB | 历史 / 转换 | `matrix_6module.csrb` + 76B metadata | ✅ `-loadCSR -compactMeta` | ❌ | — | Legacy；大文件隐式 mmap |
| standard 单文件 CSRB | 无对称压缩 | `.csrb` full rows | ✅ `-loadCSR` | ❌ | — | standard layout |
| 投影 BIN | `mat_convert_bin.m` | header + float64[] | ✅ `-df` / `-dfList` | ✅ | — | 见 `6module_bak_src` |

## metadata 大小

| 大小 | 语义 | 状态 |
|---|---|---|
| 76B | 260525 legacy compact | 保留兼容 |
| 88B | 260601 ExtendedV1 | Canonical |
| 96B | ExtendedV2 | 按文件大小读取 |

## 参考

- `6module_bak_src/README.md`
- `tests/test_convert_6module_sysmat.cpp`
- `doc/Capability_Matrix_CN.md`
