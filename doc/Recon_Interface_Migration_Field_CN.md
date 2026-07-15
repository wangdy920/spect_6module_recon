# SPECT 重建接口迁移说明（现场 ThreeModalities → 当前 Release）

**对照来源：** `E:\Xiang_Project\SPECT\6Module\SPECT现场预处理_重建_DICOM流程报告.md`（现场包 `SPECT_6Module_Windows_20260530_125322`）  
**本地副本：** `E:\Xiang_Project\SPECT\6Module\doc\Recon_Interface_Migration_Field_CN.md`（由 `D:\Projects\SPECT-recon\doc\` 同步后按 2026-07-10 实验审阅补充）  
**目标重建：** `SPECT_recon_rc_20260708_ad3c1d4`（及同源 `staging/SPECT_recon_rc_dev`；**以 RC 包 `spect_config.ini` 为准**）  
**目标预处理：** `release/SPECT_6Module_260707/bin/preprocess/Spect_preprocess.exe` + `resources/`（**禁止**用 `eng_to_6module_projection_bin.py` 冒充现场预处理）  
**默认配置依据：** RC `bin/config/spect_config.ini` + `algo_mlem.conf`（`-dumpConf` 已核对）  
**适用范围：** 预处理配对 + 重建接口 + 多床输出交接；DICOM 标签/空间配准仍由上位机负责  
**生成 / 审阅日期：** 2026-07-10（补充：矩阵驱动 projection_variant；保留 `-opti mlem` 与几何四件套默认传入）

---

## 0. 迁移结论（先读）

| 决策 | 口径 |
|---|---|
| 生产主线 | **Spect_preprocess `identity` + 矩阵 `260601` chunk + 多床 `-dfList` + `z_bed_flip=1` + PSF FWHM `1.5,4`（Operator）** |
| 旧矩阵兼容 | **Spect_preprocess `xflip/flipx` + 单文件 `260525` CSRB + `-zbedflip 1` + 同上 PSF** |
| 投影变体选择 | **由系统矩阵版本驱动**（§2.5）：选 260601 → `identity`；选 260525/单文件 CSRB → `xflip`。上位机按矩阵入口自动配对，禁止混用 |
| 算法 CLI | **始终传 `-opti mlem`**（后续可改为 `-opti bsrem` 等）；不要依赖“默认 MLEM 就不写” |
| 几何 CLI | **默认仍传** `-dim` / `-vox` / `-ang` / `-detN`；仅当确认矩阵 `metadata.bin` 含对应字段且程序会明确覆盖时，才可省略（§2.3） |
| 床位内 Z 翻转 | **两条线都保留开启：`z_bed_flip=1` / `-zbedflip 1`**（不要再写 `-zbedflip 0`“纠正”） |
| 预处理工具 | **只用** release `Spect_preprocess.exe`（经包装器或目录模式）；**不要**用 `eng_to_6module_projection_bin.py` 做验收/现场等价 |
| PSF CLI | **默认显式传** `-psf 1.5,4 -psfMode Operator`（FWHM mm[,N] + mode）。旧 `-psf 0.637`（sigma）禁止再传；误传 `0.637` 会被当成窄 FWHM |

S 软管（`20260710_spect_S`）实验摘要见 **§12**。

---

## 1. 范围与活动入口

| 阶段 | 现场旧链 | 当前 Release | 是否本迁移范围 |
|---|---|---|---|
| ENG 预处理 → `BedN.bin` | `run_6module_preprocess.py`，**`PROJECTION_VARIANT=flipx`** | 见 §2.5：chunk 主线改 **`identity`**；单文件 legacy 仍 **`xflip/flipx`** | **配对必改** |
| 预处理可执行文件 | 现场包内 `Spect_preprocess.exe` | `SPECT_6Module_260707\bin\preprocess\Spect_preprocess.exe` + `resources\` | **必换** |
| 单床 / 多床重建 | 每床一次 + 外部拼接 | 一次 `-dfList` 联合重建 | **是** |
| 系统矩阵 | 单文件 `matrix_6module.csrb` + `metadata.bin` | 主线：`bin/matrix/260601`（`-chunkCSRDir` / 自动发现） | **是** |
| 拼接 | `SPECT_Stitcher.exe` → `final_recon.*` | 重建直接出全局 MHD/RAW | **是** |
| DICOM | 读 `final_recon.mhd` | 读重建 `*_itN.mhd`（或复制改名） | 交接约定 |

活动入口：

```text
<RC>/bin/SPECT_recon.exe
<RC>/bin/config/spect_config.ini      ← 自动加载，勿再 -config
<RC>/bin/config/algo_mlem.conf       ← 默认 MLEM 时自动加载
<RC>/bin/matrix/260601/              ← 现场必须部署；唯一矩阵时可省略 CLI

# 预处理（6Module release，与重建 RC 可分开发布）
<6M>/bin/preprocess/Spect_preprocess.exe
<6M>/bin/preprocess/resources/       ← calibration/Tc99m + seg_result LUT
```

原则：包内已固定且与现场协议一致的项（如 overlap、翻转默认）可不重复传；但 **算法（`-opti`）、几何四件套、PSF（`-psf`/`-psfMode`）默认仍显式传入**，便于切换算法与避免依赖隐式默认。矩阵选定后，预处理 `projection_variant` **必须**按 §2.5 自动配对。

---

## 2. 自动搜索与配置优先级（先看这个）

优先级（高 → 低）：

```text
CLI 显式参数  >  algo_*.conf  >  bin/config/spect_config.ini  >  硬编码默认
```

### 2.1 配置文件自动搜索（无需 `-config`）

相对 **exe 所在目录**（即 `<RC>/bin/`）依次找：

1. `config/spect_config.ini`
2. `spect_config.ini`
3. 上一级 `config/spect_config.ini`
4. 当前工作目录下的同名文件

算法配置：传 `-opti mlem|bsrem|admm` 时自动加载对应 `config/algo_<alg>.conf`。未传 `-opti` 时虽默认 MLEM，但**上位机仍应显式传 `-opti mlem`**，以便后续切换 `bsrem` 等时只改这一处。

**注意：** `staging/SPECT_recon_rc_dev` 的 ini 可能与 RC zip **不一致**（例如某次 staging 曾把 `z_bed_flip=0`）。现场集成与文档口径以 **`SPECT_recon_rc_20260708_ad3c1d4`** 为准：`z_bed_flip=1`。

### 2.2 系统矩阵自动搜索（无需 `-chunkCSRDir` 的条件）

1. CLI `-chunkCSRDir`（显式才需要）  
2. `[SystemMatrix] default_chunk_csr_dir`（本 RC **未启用**，仍为注释；6Module `SPECT_6Module_260707` 重建包内已写 `matrix/260601`）  
3. `bin/matrix/` 下**恰好一个**含 `metadata.bin` 的目录 → 自动选用，并切 `compact-symmetric`

因此现场部署约定：

```text
只放一个矩阵目录：bin/matrix/260601/{metadata.bin, chunk_*.csrb}
```

则命令里**不要**再写 `-chunkCSRDir` / `-matrixLayout` / `-loadCSR` / `-compactMeta`。

本 RC zip 默认**不捆绑**矩阵；安装时必须拷入，否则 `chunk_csr_dir=(none)`，重建会失败。

### 2.3 compact 几何：`metadata.bin` 可覆盖 CLI，但上位机默认仍传

进入 `compact-symmetric` 后，当前程序会用 `applyCompactMetadataGeometry()` 从矩阵 `metadata.bin` 写入：

`dim` / `vox` / `detN` / `ang`

包内 `spect_config.ini` 的 `dimensions=200,200,200`、`voxel_size=1`、`channels=40` 只是占位，**会被 metadata 覆盖**。

**上位机集成口径（本迁移要求）：**

1. **默认仍显式传入** `-dim 108,108,80 -vox 0.3,0.3,0.3 -ang 360 -detN 48,48`（与现场旧配置一致），作为契约与兜底。  
2. **仅当同时满足**以下条件时，才允许省略几何四件套：  
   - 所用矩阵目录/`metadata.bin` **已确认**含对应几何字段；且  
   - 运行日志能看到 compact metadata 几何覆盖（约 `108×108×80`、`det 48×48`、`360` 角）；且  
   - 集成方确认“以矩阵为准、CLI 几何可忽略”。  
3. 未完成上述确认前：**不要省略**。传了与 metadata 一致的值是安全的；传了冲突值则以程序实际覆盖行为为准，验收时对照日志。

### 2.4 本 RC `spect_config.ini` / `algo_*.conf` 已固定项与 CLI 建议

| 项 | 包内值（ad3c1d4） | 建议 |
|---|---|---|
| 算法 | `algo_mlem.conf` → MLEM，`subsets=1` | **始终传 `-opti mlem`**（后续切 BSREM 改为 `-opti bsrem`）；`-subsets` 与 algo conf 一致时可省略 |
| 迭代 | `iterations=30:5` | 仅当现场要 **20** 时传 `-it 20:5` |
| 几何 | metadata / ini 占位 | **默认传** `-dim`/`-vox`/`-ang`/`-detN`；省略条件见 §2.3 |
| 多床 overlap | `bed_overlap=16,16,16,0,0` | 当前输出拼接默认 stride=64；其他物理步进需显式传 `-bedOverlap` |
| 裁剪 | `crop_end=0,0` | 可不传 `-cropEnd` |
| 方向 | **`z_bed_flip=1`**，`z_out_flip=0`，`x_out_flip=0` | **保留开启**；**不要**写 `-zbedflip 0`。Legacy 单文件同样 **`-zbedflip 1`**（§2.5） |
| PSF FWHM / N | `fwhm=1.5`，`min_sigmas=4` | **默认传** `-psf 1.5,4`（同时启用 PSF 并写入 FWHM/N） |
| PSF mode | `mode=None`（包内默认关） | **默认传** `-psfMode Operator`（与 `-psf` 一起；勿只传 mode 不传 `-psf`） |
| Butterworth | ini 有 cutoff，但 **仅 `-butter` 才启用** | 旧现场未开滤波 → 不传 `-butter` |
| 稀疏后端 | 固定 CUDA | **禁止** `-sparseBackend` |
| 线程 / 日志 | `threads=16`，`verbosity=1` | 一般不传 |

用包内 exe 自检：

```bat
cd /d <RC>
.\bin\SPECT_recon.exe -dumpConf
```

应看到 `global_ini` / `algo_conf` 指向 `bin\config\...`，且 **`z_bed_flip=1`**。部署矩阵后，实际重建日志里会出现自动解析的 `chunk_csr_dir` 与 compact metadata 几何。

### 2.5 矩阵版本 → 投影变体 → 每床 Z 翻转（上位机按矩阵自动配对）

上位机应先选定系统矩阵，再**自动**决定预处理 `PROJECTION_VARIANT`（不要手填互斥组合）：

```text
if 矩阵入口 == chunk 260601 / -chunkCSRDir .../260601:
      projection_variant = identity
      recon_matrix_cli   = （唯一目录时可省略；否则 -chunkCSRDir）
elif 矩阵入口 == 单文件 CSRB（如 260525 matrix_6module.csrb + metadata.bin）:
      projection_variant = xflip   # 别名 flipx
      recon_matrix_cli   = -loadCSR ... -compactMeta ... -matrixLayout compact-symmetric
else:
      拒绝启动 / 报错：未知矩阵，无法配对投影变体

两条线：z_bed_flip = 1（或显式 -zbedflip 1）
```

| 重建矩阵入口 | 预处理 `PROJECTION_VARIANT` | 每床 `-zbedflip` / `[MultiBed] z_bed_flip` | 说明 |
|---|---|---|---|
| **主线** `-chunkCSRDir` / `bin/matrix/260601` | **`identity`** | **`1`** | 生产推荐；几何吃 metadata（CLI 几何仍建议传，§2.3） |
| **Legacy** `-loadCSR` + `-compactMeta`（如 260525） | **`xflip` / `flipx`** | **`1`** | 仅兼容旧单文件矩阵 |

配置键建议：

| 上位机键 / 状态 | 行为 |
|---|---|
| `spect_matrix_profile=260601`（或检测到 chunk 目录） | 写 `spect_preprocess_projection_variant=identity`，重建走 chunk |
| `spect_matrix_profile=260525`（或检测到单文件 CSRB） | 写 `spect_preprocess_projection_variant=flipx`，重建走 `-loadCSR`/`-compactMeta` |
| 矩阵与 variant 不一致 | **禁止开建**（例如 260601 + flipx BIN，或 260525 + identity BIN） |

要点：

1. 现场报告里的 `--projection-variant flipx` 是给 **旧单文件矩阵** 配的。迁到 **260601 chunk** 后，预处理必须改为 **`identity`**，并**用 Spect_preprocess 重新生成** `BedN.bin`。  
2. 若暂时仍跑单文件 CSRB：预处理继续 **`xflip`**，重建 **`zbedflip=1`**。  
3. **禁止**用 `eng_to_6module_projection_bin.py` 做“快速等价预处理”（§12）。  
4. 已知包装债：`SPECT_6Module_260707\bin\preprocess\config.ini` 里 `[Batch] projection_variant` 仍可能写着 **`flipx`**。目录模式直接调 exe 时务必按矩阵版本覆盖；或用 `run_6module_preprocess.py --projection-variant <由矩阵决定>`。

---

## 3. 端到端流程对照

### 3.1 现场旧链

```text
每床 SPECT_recon -df BedN.bin -loadCSR ... -compactMeta ... -alg ... -sparseBackend ... -dim/-vox/-detN/-psf 0.637 ...
  → BedN_it20/recon_it20.raw → SPECT_Stitcher → final_recon.mhd → DICOM
```

### 3.2 当前 Release 主链（最小面）

```text
Spect_preprocess（identity）→ BedN.bin
一次 SPECT_recon -dfList Bed1;Bed2;... -opti mlem -dim/-vox/-ang/-detN [-it ...] -psf 1.5,4 -psfMode Operator -fout ...\final_recon
  → final_recon_itN.mhd/.raw → DICOM
```

矩阵与几何、overlap、**`z_bed_flip=1`**、MLEM/subsets 均来自 **包内配置 + metadata 自动发现**。

### 3.3 预处理推荐调用（由矩阵版本选择 variant）

```bat
REM 先由矩阵 profile 决定 VARIANT：260601→identity；260525/单文件→xflip
set VARIANT=identity

python tools\run_6module_preprocess.py ^
  --input-root "<ENG_ROOT>\SPECT" ^
  --output-root "<PREDATA>\SPECT" ^
  --exe "<6M>\bin\preprocess\Spect_preprocess.exe" ^
  --lut "<6M>\bin\preprocess\resources" ^
  --scanner-name SPECT_6MODULE ^
  --projection-variant %VARIANT% ^
  --intermediate-format sp6proj ^
  --isotope Tc99m ^
  --merge-eng-per-angle ^
  --jobs auto --max-jobs 8
```

Legacy 260525：`set VARIANT=xflip`（或 `flipx`）。

---

## 4. 文件契约

### 4.1 输入

| 文件 | 说明 |
|---|---|
| `Bed1.bin` … `BedN.bin` | Spect_preprocess 产出；`360` 角 × `2304` rays，float64，8 字节头 `(num_angles, rays)` |

### 4.2 矩阵

| 现场旧 | 当前 |
|---|---|
| `-loadCSR matrix_6module.csrb` + `-compactMeta` + 预处理 **flipx** | 主线：部署 `bin/matrix/260601/` + 预处理 **identity**，**CLI 不传矩阵参数**；**`z_bed_flip=1`** |
| 同上单文件（临时兼容） | 仍可用 `-loadCSR`/`-compactMeta`，但必须保持预处理 **xflip** 且 **`-zbedflip 1`**（§2.5） |

### 4.3 输出 / DICOM 入口

| 现场旧 | 当前 |
|---|---|
| `final_recon.raw/.mhd`（Stitcher） | `-fout ...\recon\final_recon` → `...\recon\final_recon_itN.mhd/.raw` |
| 每床 `recon_it20.raw` | 多床主线不再使用 |

DICOM 仍以 MHD 的 `DimSize` / `ElementSpacing` 为准；取**最终迭代**文件（如 `_it20`），或复制改名为 `final_recon.mhd` 以少改上位机。`-it N:S` 时还会写出中间 `_itS/_it2S/...`，勿误用。

---

## 5. CLI 迁移：旧参数 → 删 / 留 / 改

| 现场参数 | 处理 |
|---|---|
| `-loadCSR` / `-compactMeta` / `-matrixLayout` | 主线唯一 `260601` 时可省略；legacy 单文件仍传 |
| `-alg MLEM` | **删除**（已废弃）；改为 **`-opti mlem`** |
| `-opti mlem` | **保留 / 始终传**；后续切算法改为 `-opti bsrem` 等 |
| `-subsets 1` | 与 algo conf 一致时可省略 |
| `-sparseBackend cuda` | **删除**（再写会报错） |
| `-dim` / `-vox` / `-ang` / `-detN` | **默认仍传**；仅当 §2.3 确认 metadata 覆盖后才可省略 |
| `-psf 0.637`（旧 sigma） | **禁止**；改为 **`-psf 1.5,4`**（FWHM mm[,N]，等价旧 sigma≈0.637） |
| `-psfMode Operator` | **默认与 `-psf` 一起传**。注意：仅 `-psfMode` **不**启用 PSF；必须有 `-psf ...` 才会打开 |
| `-bedOverlap 16,...` | 与 ini 相同则可不传（本 RC 为 `16,16,16,0,0`） |
| `-cropEnd` / `-zoutflip` / `-xoutflip` | 与 ini 相同则可不传 |
| `-zbedflip` | **主线与 legacy 均保持 1**；与 ini 一致时可省略；**禁止**改成 `0` |
| `-it 20` | 覆盖 algo 默认 `30:5` 时保留：`-it 20:5` |
| `-df` 循环 | 改为一次 `-dfList` |
| `-o` 每床目录 | 一次 `-fout` 全局前缀 |

完整废弃表见发布包 `CLI_CHANGES_CN.md`。

### 5.1 推荐：现场多床命令（chunk + identity）

部署前提：`bin/matrix/260601` 已就位；预处理已按矩阵选 **identity** 并用 Spect_preprocess 产出 BIN。

```bat
cd /d "E:\software\ThreeModalitiesBin\SPECT_recon_rc_20260708_ad3c1d4"

.\bin\SPECT_recon.exe ^
  -dfList "D:\data\predata\...\SPECT\Bed1.bin;D:\data\predata\...\SPECT\Bed2.bin" ^
  -opti mlem ^
  -dim 108,108,80 ^
  -vox 0.3,0.3,0.3 ^
  -ang 360 ^
  -detN 48,48 ^
  -it 20:5 ^
  -psf 1.5,4 ^
  -psfMode Operator ^
  -fout "D:\data\predata\...\SPECT\recon\final_recon"
```

含义：

- `-dfList` / `-fout`：每例必变  
- **`-opti mlem`**：显式算法入口（后续可改为 `bsrem`）  
- **`-dim/-vox/-ang/-detN`**：默认传入；若已按 §2.3 确认 metadata 覆盖，可省略这四项  
- `-it 20:5`：覆盖包内默认 `30:5`，对齐现场 `spect_it=20:5`  
- **`-psf 1.5,4 -psfMode Operator`**：默认显式传入（FWHM/N + mode；勿再写旧 sigma `0.637`）  
- 默认含 `bed_overlap=16,...`（stride=64）、**`z_bed_flip=1`**

### 5.2 何时才加回显式参数

| 情况 | 才传 |
|---|---|
| `bin/matrix/` 有多个矩阵目录 | `-chunkCSRDir .\bin\matrix\260601` |
| 矩阵在包外路径 | `-chunkCSRDir <绝对路径>` |
| 临时仍用单文件 `matrix_6module.csrb` | `-loadCSR` + `-compactMeta` + `-matrixLayout compact-symmetric`；预处理 **xflip**；**`-zbedflip 1`** |
| 真实 overlap ≠ `16,16,16,0,0` | `-bedOverlap ...` |
| 调试确认翻转 | 可显式 `-zbedflip 1`（与默认相同，仅防 staging/旧 ini 漂移） |
| 调试单床 | `-df` 单文件 + 同上最小覆盖 |

### 5.3 单床调试命令

```bat
.\bin\SPECT_recon.exe ^
  -df "D:\data\predata\...\SPECT\Bed1.bin" ^
  -opti mlem ^
  -dim 108,108,80 ^
  -vox 0.3,0.3,0.3 ^
  -ang 360 ^
  -detN 48,48 ^
  -it 20:5 ^
  -psf 1.5,4 ^
  -psfMode Operator ^
  -fout "D:\data\predata\...\SPECT\recon\Bed1"
```

### 5.4 Legacy 单文件命令（仅兼容，非默认）

预处理：由矩阵选 **`xflip`**（Spect_preprocess，**非** eng_to）。

```bat
.\bin\SPECT_recon.exe ^
  -dfList "D:\data\predata\...\SPECT\Bed1.bin;D:\data\predata\...\SPECT\Bed2.bin" ^
  -loadCSR "D:\path\matrix_6module.csrb" ^
  -compactMeta "D:\path\metadata.bin" ^
  -matrixLayout compact-symmetric ^
  -opti mlem ^
  -dim 108,108,80 ^
  -vox 0.3,0.3,0.3 ^
  -ang 360 ^
  -detN 48,48 ^
  -zbedflip 1 ^
  -it 20:5 ^
  -psf 1.5,4 ^
  -psfMode Operator ^
  -fout "D:\data\predata\...\SPECT\recon\final_recon"
```

此处 **`-zbedflip 1` 为硬要求**；`-opti`、几何四件套、PSF 参数/mode 同上。

---

## 6. 几何与多床（仍由默认层提供）

| 量 | 来源 |
|---|---|
| 单床约 `108×108×80`、`vox=0.3 mm`、`det 48×48`、`360` 角 | compact `metadata.bin`（覆盖 ini 占位几何） |
| `bed_overlap` | ini `16,16,16,0,0`（`b1=Bed1-Bed2` …） |
| 全局 Z | `Nz * nBeds - sum(所用 overlap)` |
| 方向顺序 | **每床 Z 翻（默认开）** → 整体 Z 翻 → 整体 X 翻 → `cropEnd` |

现场 DICOM 的 `imagePositionOffset*` / `sliceLocationBase_mm` 仍在上位机，不要在重建 CLI 再补偿一遍。

---

## 7. 上位机（ThreeModalities）集成

### 7.1 命令生成策略

1. 固定指向新 RC 的 `bin\SPECT_recon.exe`（工作目录建议 RC 根，或一律绝对路径）。  
2. **先选矩阵 profile（260601 / 260525）→ 自动设 `projection_variant`（§2.5）→ 再跑 Spect_preprocess**；矩阵与 BIN variant 不一致则拒绝。  
3. 每例拼：`-dfList`、`-fout`、**`-opti mlem`**（或未来 `bsrem`）、**默认几何四件套**、**`-psf 1.5,4 -psfMode Operator`**、以及与包默认不同的覆盖（常见：`-it 20:5`）。  
4. 去掉按床循环与 `SPECT_Stitcher` 主线调用；去掉 `-alg` / `-sparseBackend`。  
5. 成功条件：存在最终迭代的 `<prefix>_itN.mhd` + `.raw`。  
6. **`z_bed_flip` 保持 1**；不要在调试脚本里默认写 `-zbedflip 0`。  
7. 几何四件套：未完成 §2.3 确认前不要省略。

### 7.2 现场 config.ini 键

| 键 | 建议 |
|---|---|
| `cmd_spect_predata` | 新 6Module release `...\bin\preprocess\Spect_preprocess.exe` |
| `cmd_spect_recon` | 新 RC `...\bin\SPECT_recon.exe` |
| 矩阵 profile（新建） | 如 `spect_matrix_profile=260601\|260525`，**驱动**预处理 variant 与重建矩阵 CLI |
| `spect_csr` / `spect_compact_meta` | 仅 legacy 260525 使用；主线改为部署 `bin/matrix/260601` |
| `spect_matrix_layout` | legacy 仍可传 `compact-symmetric`；主线可省略 |
| `spect_alg` | 映射为 **`-opti mlem`**（勿再生成 `-alg`）；切 BSREM 时改为 `-opti bsrem` |
| `spect_subsets` / `spect_sparse_backend` | subsets 可省略；sparseBackend **禁止**再生成 |
| `spect_dim` / `spect_vox` / `spect_ang` / `spect_detN` | **默认仍生成对应 CLI**；仅 §2.3 确认后可停 |
| `spect_preprocess_projection_variant` | **由矩阵 profile 自动写入**：260601→`identity`，260525→`flipx` |
| `spect_psf_sigma` / PSF 相关 | 停止按旧 sigma 传参；默认生成 **`-psf 1.5,4 -psfMode Operator`** |
| `spect_it` | 仅当 ≠ 包内 `30:5` 时生成 `-it` |
| `cmd_spect_stitch` | 主线可停用 |

### 7.3 可选：把现场口径写进包内 ini

```ini
; algo_mlem.conf
iterations = 20:5

; spect_config.ini
[PSF]
mode = Operator
fwhm = 1.5
min_sigmas = 4.0

[MultiBed]
z_bed_flip = 1
```

则上位机仍建议保留：

```bat
.\bin\SPECT_recon.exe ^
  -dfList "..." ^
  -opti mlem ^
  -dim ... -vox ... -ang ... -detN ... ^
  -psf 1.5,4 -psfMode Operator ^
  -fout "..."
```

（`-it` 若已写入 ini 可再缩短；**`-opti`、几何、PSF 参数/mode 默认仍建议显式保留**。改包内配置后用 `-dumpConf` 确认。）

---

## 8. 会改变图像的项（验收必盯）

| 项 | 说明 |
|---|---|
| 矩阵换成 260601 | 勿混用旧 `matrix_6module.csrb`；预处理须同步改为 **identity**（Spect_preprocess） |
| 用 eng_to 代替 Spect_preprocess | 投影几乎不相关，螺旋/老鼠形态会崩；**拒收** |
| 单文件 + xflip 却用 `-zbedflip 0` | 头脚方向错；必须 **`-zbedflip 1`** |
| chunk + identity 却用 `-zbedflip 0` | 与 RC 默认及本迁移口径不符；**保留 1** |
| PSF | 旧 `-psf 0.637` 若误当 FWHM 会偏窄；默认传 **`-psf 1.5,4 -psfMode Operator`** |
| overlap | 仅当机械步进与 `16` 不符时才覆盖 |
| `-it` | `20` vs 包默认 `30` 会改收敛与外观 |

---

## 9. 排查顺序

1. `-dumpConf` 是否加载了 RC `bin\config\spect_config.ini`，且 **`z_bed_flip=1`**。  
2. `bin\matrix\` 是否**恰好一个** 260601 目录。  
3. 预处理是否 Spect_preprocess，且 variant 与矩阵配对（identity↔260601 / xflip↔单文件）。  
4. 命令是否仍残留 `-alg` / `-sparseBackend` / `-psf 0.637` / `-zbedflip 0` / 几何四件套。  
5. 日志是否出现 compact metadata 几何与 `psf=operator fwhm_mm=1.5`。  
6. 是否生成约定前缀的 `*_itN.mhd/.raw`。

---

## 10. 验证与未决

### 已核对（2026-07-10）

- 现场旧流程报告  
- RC `bin/config/spect_config.ini`、`algo_mlem.conf` 与同包 `-dumpConf`（新包目标为 `bed_overlap=16,16,16,0,0`、**`z_bed_flip=1`**；旧 RC 需以包内 `-dumpConf` 为准）
- `CLI_CHANGES_CN.md`：`-psf <sigma>` → `-psf <fwhm_mm[,N]>`  
- S 软管实验（§12）：Spect_preprocess 配对、PSF、eng_to 拒收、identity 线 `zbedflip0` vs `zbedflip1` DICOM  
- 6Module release 重建 ini 亦为 **`z_bed_flip=1`**；预处理 ini 仍可能默认 `flipx`（包装债，见 §2.5）

### 现场仍需确认

1. 真实 bed overlap 是否就是包内 `16,16,16,0,0`。  
2. **chunk + identity + `z_bed_flip=1`** 与现场 DICOM/CT 配准是否一致（避免上位机再翻一次）。identity 线已出 `zbedflip1` DICOM 供目视（§12）。  
3. 迭代用包默认 `30:5` 还是继续现场 `20:5`。  
4. PSF：默认 CLI **`-psf 1.5,4 -psfMode Operator`**（勿再传旧 sigma）。  
5. 安装流程是否保证 `bin/matrix/260601` 唯一就位，且预处理已切到 **identity** 后重出 BIN。  
6. 预处理包 `config.ini` 的 `projection_variant` 是否已改为与主线一致的 **identity**（避免目录模式静默 flipx）。

---

## 11. 相关文档

| 文档 | 用途 |
|---|---|
| `SPECT现场预处理_重建_DICOM流程报告.md` | 旧现场对照（部分过时） |
| 发布包 `bin/config/spect_config.ini` | **默认真相**；与 `-dumpConf` 对照 |
| 发布包 `CLI_CHANGES_CN.md` | 废弃参数 / PSF 语义 |
| 发布包 `PARAMETERS_CN.md` | 包说明 |
| `doc/USAGE_CN.md` | 公开 CLI |
| `doc/RELEASE_20260715_CN.md` | 当前 RC |
| `doc/RELEASE_260707_CN.md`（6Module） | 预处理 + 260601 发布说明 |
| `doc/Capability_Matrix_CN.md` | 能力边界 |

---

## 12. 2026-07-10 实验证据（S 软管）

**数据：** `F:\DATA\rawdata\SUPER_PSC_1_0\SPECT\20260710_spect_S`  
**脚本：** `tools/run_S_spect_preprocess_matrix_validate.py` 等  

| 实验 | 结论 |
|---|---|
| release Spect_preprocess **xflip** vs 既有 `predata\...\preprocess_summary.json` bins | 逐床投影 **corr = 1.0**（计数一致）→ 现场预处理可被 release exe 复现 |
| eng_to `--xflip` vs Spect_preprocess xflip | 投影 corr ≈ **0.03** → **eng_to 非等价**，不可用于验收 |
| xflip + 260525 + PSF `1.5,4` + overlap16（Spect_preprocess） | 螺旋 S 形态可恢复；与先前 field 重建 corr ≈ **1.0** |
| identity + 260601 + PSF `1.5,4` + overlap16（Spect_preprocess） | 可出图；与 xflip525 线 outside-overlap corr ≈ **0.01**（不同配对，勿做体素对齐验收） |
| 误用 `-psf 0.637`（新语义当 FWHM） | 核过窄；等价现场旧 sigma 应为 **FWHM 1.5** |
| 验证脚本曾写 `-zbedflip 0` | **偏离 RC 默认**；identity 线已按相反设置重跑 **`-zbedflip 1`** 并导出 DICOM |
| identity `zbedflip0` vs `zbedflip1` | 全卷 corr ≈ **0.05**（翻转生效） |

**identity + 260601 + PSF + zbedflip1 DICOM：**

```text
F:\DATA\rawdata\SUPER_PSC_1_0\SPECT\20260710_spect_S\
  recon_spect_preprocess_validate_20260710e\
    recon_identity_260601_psfFWHM15_zbedflip1\export\dicom
```

**迁移落盘口径（本审阅确认）：** 生产与 legacy **均保留 `-zbedflip 1` / `z_bed_flip=1`**；待 identity 线目视确认配准后，上位机勿再额外翻 Z。
