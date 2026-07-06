# 260601 compact chunk CSRB 适配规划与执行记录

创建时间：2026-06-01T21:00+08:00

## 1. 目标

把 `test_data/6module/260601` 中的新系统矩阵目录：

```text
metadata.bin
chunk_000.csrb
chunk_001.csrb
chunk_002.csrb
chunk_003.csrb
```

接入当前 `SPECT_recon`，并完成两类重建验证：

1. 单床位圆柱投影：
   `test_data/6module/PSC_6module_24pin_10MBq_600s_1.bin`
2. 三床位 logo 投影：
   `test_data/logo/Bed1.bin`
   `test_data/logo/Bed2.bin`
   `test_data/logo/Bed3.bin`

三床位输出需要按轴向拼接为 Bed1 到 Bed3 的上下顺序，并与 `test_data/6module/260525` 旧矩阵链路的重建结果对比，计算 NCC、relative L2、RMSE/mean、sum relative difference 等指标。

## 2. 当前输入事实

### 2.1 260601 新矩阵

实测 `metadata.bin`：

- `totalRows=103680`
- `totalVoxels=933120`
- `compactNumAngles=90`
- `fullNumAngles=360`
- `detNx=48`
- `compactDetZ=24`
- `compactRaysPerAngle=1152`
- `detNz=48`
- `Nx=108, Ny=108, Nz=80`
- `voxelSize=0.3`
- `use90Sym=1`
- `useZSym=1`
- `use45Sym=0`
- `symCode=90`
- `symmetryOrder=4`

chunk header 实测：

| 文件 | rows | angle_offset | angle_count | nnz |
| --- | ---: | ---: | ---: | ---: |
| `chunk_000.csrb` | 26496 | 0 | 23 | 618747598 |
| `chunk_001.csrb` | 26496 | 23 | 23 | 626836508 |
| `chunk_002.csrb` | 26496 | 46 | 23 | 626384528 |
| `chunk_003.csrb` | 24192 | 69 | 21 | 564900215 |

合计 rows 为 `103680 = 90 * 48 * 24`，合计 nnz 约 `2.4369e9`。full projection 每床位行数仍为 `829440 = 360 * 48 * 48`。

### 2.2 260525 对照矩阵

`test_data/6module/260525` 当前包含：

- `metadata.bin`
- `chunk_000.bin` 到 `chunk_003.bin`
- `matrix_6module.csrb`

该链路是对照基线来源。实现后要用同一投影数据分别跑 `260601` 与 `260525`，并输出可复查的指标 JSON/CSV 和 QA PNG。

### 2.3 参考脚本

参考路径：

`D:\WDY-project\Xiang_Project\SPECT\6Module\release\SPECT_6Module_Windows_20260530_125322\tools`

已确认有用脚本：

- `stitch_recon_beds.py`：读取 MHD/RAW，沿 Z 拼接单床位结果，支持显式 Z start、reverse、flip、crop、overlap mode。
- `render_recon_orthogonal_slices.py`：输出正交 slice 与 MIP QA PNG。
- `make_projection_variants.py`：投影 BIN 读写和 detector grid 变换参考。

## 3. 方案决策

### 3.1 主路径采用 `-chunkCSRDir`

不把 4 个 chunk 合并成一个巨型 CSR 作为重建主入口。

理由：

- 260601 合计 nnz 已约 `2.44B`，单文件 CSR 即使改成 64 位，也会把 IO、内存映射、CUDA descriptor 和转置 SpMV 变成一个超大对象。
- full-row 展开会再放大约 8 倍，nnz 到约 `19.5B`，磁盘和内存不可接受。
- chunk 文件自带 `angle_offset/angle_count`，更适合按 compact angle block 流式读取和缓存。
- 精度不依赖是否合并。精度风险主要来自 `90+Z` symmetry 的 full projection row 映射、体素旋转/反旋转、Z mirror、边界角处理和 numerator/denominator 累加。

### 3.2 64 位只用于 chunk 内部和累计偏移

需要在 host 侧用 64 位表达：

- chunk `row_ptr`
- chunk `nnz`
- 文件偏移
- 总 nnz 汇总

但 GPU 单次执行以单个 chunk 或可控范围上传，避免在现有 `io::CSRView` 的 `uint32_t row_ptr` 上强行表达 2B+ 全局矩阵。

### 3.3 适配范围

本阶段只做：

- `compact-symmetric`
- CUDA sparse backend
- MLEM
- `subsets=1`
- single-bed 和现有 `-dfList/-bedOverlap` 多床位框架
- image-domain Operator PSF 和 Butterworth 保持现有外层逻辑

仍不纳入：

- CTAC
- background/scatter
- projection-domain PSF
- BSREM/ADMM compact chunk 后端
- full-row CSRB 展开作为生产路径

## 4. 实施步骤

### P0. 文档与输入审计

交付物：

- 本文件
- 更新 `.agent/CONTINUITY.md`
- 输入审计 JSON：`recon_out/qa/260601_chunk_input_validation.json`

验收：

- metadata 字段与 chunk headers 可复查。
- 260601 与 260525 的输入路径明确。

### P1. CLI 与 metadata 优先几何

修改点：

- `include/cli/ReconArgs.h` 增加 `chunkCSRDir`。
- `include/cli/ArgumentParser.h` 增加 `-chunkCSRDir`。
- compact 模式允许 `-loadCSR` 或 `-chunkCSRDir` 二选一。
- `-chunkCSRDir` 默认 metadata 为 `<dir>/metadata.bin`，`-compactMeta` 可覆盖。
- compact chunk 模式在读取投影前先读取 metadata，并覆盖/校验 `Nx,Ny,Nz,Vx,Vy,Vz,detN,ang`。

验收：

- parser 回归覆盖 `-chunkCSRDir`。
- help 显示新参数。
- 缺少系统矩阵入口、同时传 `-loadCSR` 和 `-chunkCSRDir` 时明确报错。

### P2. chunk CSRB reader

新增模块建议：

- `include/io/CompactChunkCSR.h`
- `src/io/CompactChunkCSR.cpp`

核心能力：

- 扫描 `chunk_*.csrb`。
- 读取并校验 header：
  `magic=0x42525343`, `version=1`, `rows`, `angle_offset`, `angle_count`, `nnz`。
- 校验 `rows == angle_count * compactRaysPerAngle`。
- 校验 angle ranges 连续覆盖 `0..compactNumAngles-1`。
- 读取单个 chunk 为 host CSR：
  `int64 row_ptr`, `int32 col_idx`, `float values`。
- 校验 `row_ptr[0]==0`, 单调，`row_ptr[rows]==nnz`, `col_idx < totalVoxels`。

验收：

- 260601 目录扫描得到 4 个 chunk，总 rows `103680`，总 nnz 与 header 合计一致。
- toy chunk 单元测试可构造 2-3 个小 `.csrb` 并通过 reader。

### P3. symmetry 泛化到 `90+Z`

修改点：

- `CompactSymmetryMode` 增加 `Sym90Z`，或等价地让 `Sym90 + useZSym` 进入 z branch。
- `inferCompactSymmetryMode()` 对 `use90Sym && useZSym` 返回 `90z`。
- `validateCompactSymmetricInputs()` 支持：
  - `45z`: 旧路径保持不退化。
  - `90z`: `compactNumAngles == fullNumAngles / 4`, `compactDetZ <= detNz`, `Nx == Ny`。
- 新增 `fullAngleFromCompactAngle()` 的 90 度分支边界规则。
- 新增 `mapSym90ZVoxelCoord()`，支持 4 个 XY branch 和 Z mirror。

验收：

- toy metadata 覆盖 `90z` full projection rows：`90 * 48 * 24 * 4 * 2 = 829440`，无重复、无遗漏。
- 旧 `45z` toy 与真实 6module regression 保持通过。

### P4. chunk CUDA numerator/sensitivity

新增 CUDA helper 建议：

- `computeCompactChunkSensitivityCUDA(plan, meta, threads, denom, timing)`
- `runCompactChunkMlemNumerCUDA(plan, meta, projection, image, threads, numer, timing)`

执行口径：

1. 外层遍历 symmetry branch 和 Z branch。
2. 每个 branch 对 local image 做 transform。
3. 逐 chunk 读取或取缓存 host CSR。
4. 上传当前 chunk 到 CUDA sparse ops。
5. forward 得到 `y_chunk`。
6. 按 chunk 的 `angle_offset` 和 local row 生成 full projection ratio。
7. transpose backward 得到 `bpRot`。
8. 反 transform 累加到 local/global numerator 或 denominator。

缓存：

- host chunk cache 默认预算按环境变量 `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES`，无设置时保守取可用内存比例或固定上限。
- device chunk cache 可先做 transient 单 chunk；后续再做 resident cache。

验收：

- 小 toy chunk 和显式 full operator numerator/denominator 对齐。
- 真实 260601 1 iteration 可完成，输出有限、非负、正总量。
- 日志打印 chunk reader/cache/profile。

### P5. 260601 与 260525 重建测试

命令形态：

```powershell
SPECT_recon.exe `
  -config __missing__.ini `
  -df test_data\6module\PSC_6module_24pin_10MBq_600s_1.bin `
  -matrixLayout compact-symmetric `
  -sparseBackend cuda `
  -chunkCSRDir test_data\6module\260601 `
  -alg mlem `
  -subsets 1 `
  -it 50 `
  -th 8 `
  -fout recon_out\260601_psc_it50\recon
```

260525 对照按其当前可运行入口执行，优先使用已有 `matrix_6module.csrb + metadata.bin` compact-symmetric 路径；若需旧 chunk 参考，则明确记录实际命令。

验收：

- 单床位 260601 与 260525 输出 MHD/RAW 均存在。
- QA PNG 输出。
- 指标 JSON 包含 NCC、relative L2、RMSE/mean、sum relative difference、finite ratio、shape、sum、mean、max。

### P6. logo 三床位重建与拼接

命令形态：

```powershell
SPECT_recon.exe `
  -config __missing__.ini `
  -dfList "test_data\logo\Bed1.bin;test_data\logo\Bed2.bin;test_data\logo\Bed3.bin" `
  -bedOverlap 0 `
  -matrixLayout compact-symmetric `
  -sparseBackend cuda `
  -chunkCSRDir test_data\6module\260601 `
  -alg mlem `
  -subsets 1 `
  -it 50 `
  -th 8 `
  -fout recon_out\260601_logo_3bed_it50\recon
```

若主程序直接输出全局多床位体，则用该输出作为拼接结果；若按单床位分别输出，则调用 `stitch_recon_beds.py` 显式按 Bed1、Bed2、Bed3 顺序沿 Z 拼接。

验收：

- Bed1-3 顺序明确，不使用隐式 reverse。
- 拼接 MHD/RAW 存在。
- QA PNG 和指标 JSON 存在。
- 与 260525 三床位对照计算同一组指标。

## 5. 指标脚本

新增脚本建议：

- `scripts/compare_recon_metrics.py`

输入：

- `--candidate <mhd>`
- `--reference <mhd>`
- `--out-json <json>`
- 可选 `--mask-positive-union`
- 可选 `--crop-common-shape`

指标：

- NCC：去均值归一化互相关。
- cosine similarity。
- relative L2。
- RMSE。
- RMSE / reference positive mean。
- sum relative difference。
- finite ratio。
- min/max/mean/sum。

## 6. 执行记录

- 2026-06-01T21:00+08:00：完成输入目录、metadata 和 chunk header 初步审计；确定主路径为 `-chunkCSRDir`，不合并巨型 64 位 CSR。

