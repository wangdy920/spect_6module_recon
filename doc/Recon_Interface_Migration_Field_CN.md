# SPECT 重建接口迁移说明（现场 ThreeModalities → 当前 Release）

**目标重建：** 当前 RC（及同源 `staging/SPECT_recon_rc_dev`；**以 RC 包 `bin/config/spect_config.ini` + `-dumpConf` 为准**）  
**目标预处理：** 6Module release `bin/preprocess/Spect_preprocess.exe` + `resources/`（**禁止**用 `eng_to_6module_projection_bin.py` 冒充现场预处理）  
**适用范围：** 预处理配对 + 重建接口 + 多床输出交接；DICOM 标签/空间配准仍由上位机负责  
**对齐日期：** 2026-08-11（CASToR PSF + 固定 6module ROI/mask/床位步进）

---

## 0. 迁移结论（先读）

| 决策 | 口径 |
|---|---|
| 生产主线 | **Spect_preprocess `identity` + 矩阵 `260601` chunk（`-chunkCSRDir` / 自动发现）+ 多床 `-dfList` + `z_bed_flip=1` + `-psf 1.3,3 -psfMode sieve -psfKernel castor`** |
| 投影变体选择 | **由系统矩阵驱动**：当前唯一矩阵入口为 260601 chunk → `identity`。旧 260525 单文件 CSRB + `xflip` 链路已随 `-loadCSR` 一并退场 |
| 算法 CLI | **始终传 `-opti mlem`**（后续可改为 `-opti bsrem`）；compact 布局恒单 subset，无 `-subsets` 参数 |
| 几何 CLI | **默认仍传** `-dim` / `-vox` / `-ang` / `-detN`；仅当确认矩阵 `metadata.bin` 含对应字段且日志可见覆盖时，才可省略（§2.3） |
| 床位内 Z 翻转 | **保留开启：`z_bed_flip=1` / `-zbedflip 1`**（不要再写 `-zbedflip 0`“纠正”） |
| 预处理工具 | **只用** release `Spect_preprocess.exe`；**不要**用 `eng_to_6module_projection_bin.py` 做验收/现场等价 |
| PSF CLI | **默认显式传** `-psf 1.3,3 -psfMode sieve -psfKernel castor`；`castor` 对齐 erf 体素积分、零边界、Z-only norm 与 forward-reuse backward。旧 `-psf 0.637`（sigma）禁止再传 |
| 原型/主线验证 | 主线固定 canonical gain/stitch，无 seam/profile 开关；Smooth 0.30 mm 与 Butterworth 5 cpcm/8 阶须在两侧按同一阶段/参数比较 |

---

## 1. 活动入口

```text
<RC>/bin/SPECT_recon.exe
<RC>/bin/config/spect_config.ini      ← 自动加载，勿再 -config
<RC>/bin/config/algo_mlem.conf       ← 默认 MLEM 时自动加载
<RC>/bin/matrix/260601/              ← 现场必须部署；唯一矩阵时可省略 CLI

# 预处理（6Module release，与重建 RC 可分开发布）
<6M>/bin/preprocess/Spect_preprocess.exe
<6M>/bin/preprocess/resources/       ← calibration/Tc99m + seg_result LUT
```

原则：包内已固定且与现场协议一致的项（如 overlap、翻转默认）可不重复传；但 **算法（`-opti`）、几何四件套、PSF（`-psf`/`-psfMode`/`-psfKernel`）默认仍显式传入**。

---

## 2. 自动搜索与配置优先级（先看这个）

优先级（高 → 低）：

```text
CLI 显式参数  >  algo_*.conf  >  bin/config/spect_config.ini  >  硬编码默认
```

### 2.1 配置文件自动搜索（无需 `-config`）

相对 **exe 所在目录**（即 `<RC>/bin/`）依次找：`config/spect_config.ini` → `spect_config.ini` → 上一级 `config/` → 当前工作目录。传 `-opti mlem|bsrem` 时自动加载对应 `config/algo_<alg>.conf`。

### 2.2 系统矩阵自动搜索（无需 `-chunkCSRDir` 的条件）

1. CLI `-chunkCSRDir`（显式才需要）
2. `[SystemMatrix] default_chunk_csr_dir`
3. `bin/matrix/` 下**恰好一个**含 `metadata.bin` 的目录 → 自动选用

现场部署约定：**只放一个矩阵目录** `bin/matrix/260601/{metadata.bin, chunk_*.csrb}`。本 RC zip 默认**不捆绑**矩阵；安装时必须拷入，否则重建会失败。

### 2.3 compact 几何：`metadata.bin` 可覆盖 CLI，但上位机默认仍传

进入 compact 路径后，程序用 `applyCompactMetadataGeometry()` 从矩阵 `metadata.bin` 写入 `dim` / `vox` / `detN` / `ang`；包内 ini 的几何只是占位。

**上位机集成口径：**

1. **默认仍显式传入** `-dim 108,108,80 -vox 0.3,0.3,0.3 -ang 360 -detN 48,48`，作为契约与兜底。
2. 仅当确认 metadata 覆盖行为（日志可见约 `108×108×80`、`det 48×48`、`360` 角）且集成方确认后，才允许省略。
3. 传了与 metadata 一致的值是安全的。

### 2.4 包内已固定项与 CLI 建议

| 项 | 建议 |
|---|---|
| 算法 | **始终传 `-opti mlem`**（切 BSREM 改为 `-opti bsrem`） |
| 迭代 | 仅当现场要 **20** 时传 `-it 20:5`（覆盖包内 `30:5`） |
| 几何 | **默认传** 四件套；省略条件见 §2.3 |
| 多床 overlap | 与 ini 相同则不传 `-bedOverlap` |
| 方向 | **`z_bed_flip=1` 保留**；**不要**写 `-zbedflip 0` |
| PSF | **默认传** `-psf 1.3,3 -psfMode sieve -psfKernel castor`（只传 mode/kernel 不启用 PSF） |
| Butterworth | 发布默认关闭；`-butter` 仅用于独立的 `in_recon` 能力测试，且与 `sieve` 互斥。原型输出滤波比较须在两侧对主输出离线施加同一参数 |
| 稀疏后端 | 固定 CUDA |

自检：`.\bin\SPECT_recon.exe -dumpConf`，应看到 `global_ini` / `algo_conf` 指向 `bin\config\...` 且 `z_bed_flip=1`。

---

## 3. 端到端流程对照


### 3.1 当前 Release 主链（最小面）

```text
Spect_preprocess（identity）→ BedN.bin
一次 SPECT_recon -dfList Bed1;Bed2;... -opti mlem -dim/-vox/-ang/-detN [-it ...] -psf 1.3,3 -psfMode sieve -psfKernel castor -fout ...\final_recon
  → final_recon_itN.mhd/.raw → DICOM
```

矩阵与几何、overlap、**`z_bed_flip=1`**、算法默认均来自 **包内配置 + metadata 自动发现**。

### 3.2 预处理推荐调用

```bat
REM 260601 chunk 主线固定 VARIANT=identity
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

已知包装债：预处理包 `config.ini` 里 `[Batch] projection_variant` 可能仍写 **`flipx`**（旧单文件矩阵遗留）。目录模式直接调 exe 时务必覆盖为 **identity**。

---

## 4. 文件契约

### 4.1 输入

| 文件 | 说明 |
|---|---|
| `Bed1.bin` … `BedN.bin` | Spect_preprocess 产出；`360` 角 × `2304` rays，float64，8 字节头 `(num_angles, rays)` |

### 4.2 矩阵

| 现场旧（已删） | 当前 |
|---|---|
| `-loadCSR matrix_6module.csrb` + `-compactMeta` + 预处理 flipx | 部署 `bin/matrix/260601/` + 预处理 **identity**，**CLI 不传矩阵参数**；**`z_bed_flip=1`** |

### 4.3 输出 / DICOM 入口

| 现场旧 | 当前 |
|---|---|
| `final_recon.raw/.mhd`（Stitcher） | `-fout ...\recon\final_recon` → `...\recon\final_recon_itN.mhd/.raw` |
| 每床 `recon_it20.raw` | 多床主线不再使用 |

DICOM 仍以 MHD 的 `DimSize` / `ElementSpacing` 为准；取**最终迭代**文件（如 `_it20`），或复制改名为 `final_recon.mhd` 以少改上位机。`-it N:S` 时还会写出中间 `_itS/_it2S/...`，勿误用。

固定输出几何：单床 `100×100×72`；多床 `100×100×(66 + (bed_count-1)×56)`，床位物理步进 `16.8 mm`。单/多床圆柱 ROI 分别为 `30×21.6 mm`、`36×66 mm`。

---

## 5. 推荐命令

### 5.1 现场多床命令（chunk + identity）

前提：`bin/matrix/260601` 已就位；预处理已按 **identity** 用 Spect_preprocess 产出 BIN。

```bat
.\bin\SPECT_recon.exe ^
  -dfList "D:\data\predata\...\SPECT\Bed1.bin;D:\data\predata\...\SPECT\Bed2.bin" ^
  -opti mlem ^
  -dim 108,108,80 ^
  -vox 0.3,0.3,0.3 ^
  -ang 360 ^
  -detN 48,48 ^
  -it 20:5 ^
  -psf 1.3,3 ^
  -psfMode sieve ^
  -psfKernel castor ^
  -fout "D:\data\predata\...\SPECT\recon\final_recon"
```

含义：`-dfList` / `-fout` 每例必变；`-opti mlem` 显式算法入口；几何四件套默认传入（§2.3）；`-it 20:5` 覆盖包内默认；CASToR `sieve` PSF 三件套默认显式传入。

### 5.2 何时才加回显式参数

| 情况 | 才传 |
|---|---|
| `bin/matrix/` 有多个矩阵目录 | `-chunkCSRDir .\bin\matrix\260601` |
| 矩阵在包外路径 | `-chunkCSRDir <绝对路径>` |
| 真实 overlap 与包内默认不符 | `-bedOverlap ...` |
| 调试确认翻转 | 可显式 `-zbedflip 1`（与默认相同，仅防 staging/旧 ini 漂移） |
| 调试单床 | `-df` 单文件 + 同上最小覆盖 |

---

## 6. 上位机（ThreeModalities）集成要点

1. 固定指向新 RC 的 `bin\SPECT_recon.exe`。
2. 预处理 `projection_variant` 固定 **identity**（矩阵唯一入口为 260601 chunk）。
3. 每例拼：`-dfList`、`-fout`、**`-opti mlem`**、**几何四件套**、**`-psf 1.3,3 -psfMode sieve -psfKernel castor`**、以及与包默认不同的覆盖（常见：`-it 20:5`）。
4. 去掉按床循环与 `SPECT_Stitcher` 主线调用；旧 `-alg` / `-sparseBackend` / `-loadCSR` / `-subsets` **禁止再生成**（会报错）。
5. 成功条件：存在最终迭代的 `<prefix>_itN.mhd` + `.raw`。
6. **`z_bed_flip` 保持 1**。

现场 `config.ini` 键映射建议：`spect_alg` → `-opti mlem`；`spect_csr` / `spect_compact_meta` / `spect_matrix_layout` / `spect_subsets` / `spect_sparse_backend` → **停止生成**；`spect_it` ≠ 包默认时生成 `-it`；`cmd_spect_stitch` 主线停用。

---

## 7. 会改变图像的项（验收必盯）

| 项 | 说明 |
|---|---|
| 预处理 variant | 必须 Spect_preprocess **identity**；用 eng_to 代替会在投影上几乎不相关（corr≈0.03，实测拒收） |
| `-zbedflip 0` | 与 RC 默认及本迁移口径不符；**保留 1**，否则头脚方向错 |
| PSF | 旧 `-psf 0.637` 若误当 FWHM 会偏窄；默认传 **`-psf 1.3,3 -psfMode sieve -psfKernel castor`** |
| overlap | 仅当机械步进与包内默认不符时才覆盖 |
| `-it` | `20` vs 包默认 `30` 会改收敛与外观 |

## 8. 排查顺序

1. `-dumpConf` 是否加载了 RC `bin\config\spect_config.ini`，且 **`z_bed_flip=1`**。
2. `bin\matrix\` 是否**恰好一个** 260601 目录。
3. 预处理是否 Spect_preprocess，且 variant = **identity**。
4. 命令是否仍残留 `-alg` / `-sparseBackend` / `-loadCSR` / `-subsets` / `-psf 0.637` / `-zbedflip 0`。
5. 日志是否显示 `psf=sieve`、`fwhm_mm=1.3`、`kernel=castor`、逐床 physical offset、canonical `stitch_method=center66_cosine10`，以及发布滤波阶段均为 `none`。
6. 是否生成约定前缀的 `*_itN.mhd/.raw`。

---

## 9. 相关文档

| 文档 | 用途 |
|---|---|
| 发布包 `bin/config/spect_config.ini` | **默认真相**；与 `-dumpConf` 对照 |
| 发布包 `PARAMETERS_CN.md` | 包说明 |
| `doc/USAGE_CN.md` | 公开 CLI |
| `doc/Capability_Matrix_CN.md` | 能力边界 |
