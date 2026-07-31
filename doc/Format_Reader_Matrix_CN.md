# 系统矩阵格式说明（v4 chunk CSRB × SPECT_recon）

> 唯一生产格式：**v4 compact chunk CSRB 目录**，由 `6module_bak_src/spect_sysmat_PSC_6module_24pin_YSO_v4.cu` 生成，主线 `SPECT_recon` 以 `-chunkCSRDir` 读取。其余历史格式（v2 row-wise `.bin`、260525 单文件 `.csrb`、MAT 矩阵）及对应 CLI 入口已全部退场。

## 磁盘布局

```text
<matrix_dir>/
  metadata.bin        ; 88B（ExtendedV1）或 96B（ExtendedV2）
  metadata.txt        ; 文本版，仅供人工检查
  chunk_000.csrb      ; 按角度分块的 compact CSR（对称压缩，不展开）
  chunk_001.csrb
  ...
```

## metadata.bin

| 大小 | 名称 | 语义 |
|---|---|---|
| 88B | ExtendedV1 | 260601 canonical：几何（dim/vox/detN/ang）+ `use45` / `symCode` / `symmetryOrder` |
| 96B | ExtendedV2 | ExtendedV1 + `roiRadius` / `roiSide` |

当前 6module 矩阵对称码为 **SYM_90Z**（90° 旋转 + Z 对称）。程序按文件实际大小解析，进入 compact 路径后用 metadata 几何覆盖 CLI/INI 的 `dim` / `vox` / `detN` / `ang`（见 `ReconDataLoader.cpp::applyCompactMetadataGeometry`）。

## 读取端

| 读取端 | 入口 | 说明 |
|---|---|---|
| `SPECT_recon` | `-chunkCSRDir <dir>`；或 `[SystemMatrix] default_chunk_csr_dir`；或 exe-local `bin/matrix/*` 唯一候选自动发现 | **唯一消费路径** |

解析与校验见 `src/io/CompactChunkCSR.cpp`（chunk 校验可用 `SPECT_CHUNK_VALIDATE=off` 关闭，会打印 WARN）。

## 参考

- `6module_bak_src/README.md` — 生成端编译与产出
- `include/io/CompactSymmetryMetadata.h` — metadata 布局常量
- `test_data/6module/260601/CompactChunkCSRB_Reconstruction_README_CN.md` — 260601 数据包内说明
- `doc/Capability_Matrix_CN.md` — 能力边界
