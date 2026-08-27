# SPECT_recon CLI 使用说明（中文）

> 更新日期：2026-08-13（多床 cosine 沿输出 Z 连续淡入）
> 对齐来源：`src/cli/ArgumentParser.cpp` 与 `SPECT_recon.exe -help`

本页记录当前可执行文件公开接受的 CLI 参数。运行时最终真相仍以当前二进制：

```powershell
.\bin\SPECT_recon.exe -help
.\bin\SPECT_recon.exe -dumpConf
```

为准；`-dumpConf` 会打印配置文件和 CLI 合并后的结果。

---

## 快速开始

```powershell
# RC 包内仅有一个 bin/matrix/<name>/metadata.bin 时，可省略矩阵参数
.\bin\SPECT_recon.exe -df proj.bin -o recon_output -it 50:5

# compact chunk 260601 生产入口（唯一矩阵入口）
.\bin\SPECT_recon.exe -df proj.bin -chunkCSRDir bin\matrix\260601 -it 50

# BSREM
.\bin\SPECT_recon.exe -df proj.bin -chunkCSRDir bin\matrix\260601 -opti bsrem -it 50
```

Windows 路径建议加引号或使用 `/`：`-df "D:\path\file.bin"` 或 `-df D:/path/file.bin`。

---

## 配置文件

未显式传入 `-config` 时，搜索顺序：

1. exe 所在目录的 `config/spect_config.ini`
2. exe 所在目录的 `spect_config.ini`
3. exe 上一级目录的 `config/spect_config.ini`
4. 当前工作目录的 `config/spect_config.ini`
5. 当前工作目录的 `spect_config.ini`

算法默认参数会从同级 `config/algo_<name>.conf`（`algo_mlem.conf` / `algo_bsrem.conf`）加载，也可用 `-algoConf <path>` 指定。CLI 默认覆盖配置文件；当 `[Compatibility] recon_sections_override_cli=true` 时，`spect_config.ini` 仅对 Regularization、PSF、Filtering 与 Postprocess（输出滤波及主输出覆盖策略）具有最终优先级，其他参数仍由 CLI 覆盖。未知或已退役的 CLI 参数只打印 `[NOTE]` 并忽略；已识别参数的非法取值、以及易混淆别名 `-iter` / `-verbose` / `-v <n>` 仍会失败。

compact chunk 矩阵默认识别优先级：

1. `-chunkCSRDir <dir>`
2. `[SystemMatrix] default_chunk_csr_dir`
3. exe-local `bin/matrix/*` 下唯一包含 `metadata.bin` 的目录
4. 多个候选且未指定时直接报错

---

## 数据与矩阵

| 参数 | 说明 |
|---|---|
| `-df <path>` / `-i <path>` / `-input <path>` | 单床位投影 MAT/BIN |
| `[InputOutput] merge_time_segments=true` | `-df` 指向一个 `案例/时间段/SPECT` 目录时，按时间顺序扫描同案例下的 `YYYYMMDD_HHMMSS/SPECT`，在内存中合并各时间段的同名 `BedN.bin`；默认关闭 |
| `[InputOutput] output_filters_overwrite_primary=true` | 将启用的输出阶段后处理按 Smooth→Butterworth 顺序写入无后缀主输出；默认关闭 |
| `-dfList <list>` | 多床位投影列表，逗号/分号分隔；路径含 `BedN` 时按床位号排序，必须是 Bed1..Bed5 的连续子集 |
| `-bProj <path>` | 背景/散射投影 MAT/BIN |
| `-bProjList <list>` | 多床位背景投影列表，排序后必须与床位数量一致 |
| `-chunkCSRDir <dir>` | compact chunk 目录，包含 `metadata.bin + chunk_*.csrb`；**唯一矩阵入口** |
| `-compactDenomCache <mhd>` | 显式复用/保存一个 compact global denominator cache，优先于 `[SystemMatrix]` 的固定路径和自动目录 |
| `-ctacFactorCacheDir <dir>` | 复用/保存 CTAC 逐床位、逐角度衰减系数表缓存；默认 `recon_out/ctac_factor_cache` |
| `-projectionResidual [on\|off]` | 最终迭代后按同一 forward/PSF/CTAC 口径写出逐床、angle×detector-Z 投影残差 CSV；默认关闭，不改变迭代或图像输出 |
| `-badDetectorFile <path>` / `[Detector] bad_detector_file` | 按全局编号或 `link/FPGA/dector` 跳过硬件探测器；默认读取配置同目录的 `bad_dector` 空列表 |
| `-attMap <path>` | 衰减图，mu 单位为 `mm^-1`；多床 MHD 必须标记 exact beds 与 `local_matrix_z` |

`-dfList` 中任一路径含 `BedN` 时，程序会按床号稳定排序；仅把命令文本写成 `Bed5;...;Bed1` **不会**反转实际输入顺序。床号随 patient/physical +Z 增加，Bed1 为最小 Z。少于五床时直接传真实连续子集（如 Bed2,3,4），不补空床；地图 MHD 的 `SPECTBeds` 必须精确相同。

多时间段合并示例：配置 `merge_time_segments=true`，并传入 `-df "D:\data\predata\260810_dedenzo_mdp_6\20260810_210556\SPECT"`。程序只纳入同级、名称满足 `YYYYMMDD_HHMMSS` 且含 `SPECT` 的目录；所有时间段必须具有完全相同的 `BedN.bin` 集合与 BIN 大小/头/几何。每床位逐元素求和后直接进入重建，不写中间投影；manifest 保留每床位的全部来源。任一缺床、几何不一致、非有限值或求和溢出都会失败退出。

无 CTAC 时，程序优先使用显式 `-compactDenomCache`，其次使用 `[SystemMatrix] compact_denom_cache_path`，否则在 `compact_denom_cache_dir` 下自动选择 `auto_<manifest-hash>.mhd`。INI 相对路径按 exe 目录解析，发布默认落在 `bin/cache/compact_denom/`，不再受终端启动目录影响。v4 manifest 包含矩阵 profile/文件统计、metadata、全局维度、床位 Z 起点、PSF、解析后的坏探头列表和 normfactor 口径，但不写 source/package 的绝对根路径，因此同一 profile 随包移动仍可复用；不匹配时不会复用，不同床位数必须分开。旧 v3 cache 会在升级后一次性 miss 并生成新的 v4 文件；显式 v3 文件需改用新路径或有意删除后重建。可手动预热默认配置的 1–5 床（占位 BIN，投影计数不进入 denom；有 CTAC 时不要用）：

```powershell
python tools\prep_compact_denom_cache.py --chunkCSRDir <matrix-dir>
python tools\prep_compact_denom_cache.py --help
```

`bad_dector` 是带方括号的逐行列表，优先使用最简单的全局探测器编号 `0..95`；也兼容 `link-<3|4>-FPGA-<0..5>-dector-<0..7>` 详细格式。编号遵循采集预处理的 `spect_detector_index`：`global = ((link==4 ? 6 : 0) + FPGA)*8 + dector`。读取任一格式后都会补齐另一种标识；跨格式重复项会报错。例如：

受保护原型的 detector 68（`link-4-FPGA-2-dector-4`）使用冻结投影区域 `angle 0..59, z 24..35, x 24..35`；这是原型特例，不按其他编号使用的 ENG identity 映射推导。

```text
[
  68,
  15
]
```

该列表在 BIN 加载后生成投影行 mask：除上述 detector 68 特例外，每个硬件探测器按 ENG identity 映射到一个 FPGA 的 60 个角度和 12x12 detector block。mask 同时将 emission/background 实测行和 sensitivity denominator 中对应行置零，并进入 denom cache identity。无需修改 ENG 预处理；若只在预处理阶段丢事件但不修改 sensitivity，会造成定量偏差。

---

## 几何

| 参数 | 说明 |
|---|---|
| `-dim <Nx,Ny,Nz>` | 重建体素维度 |
| `-vox <v>` / `-vox <Vx,Vy,Vz>` | 体素尺寸 mm；单值表示三轴相同 |
| `-spacing <v>` / `-spacing <Vx,Vy,Vz>` | `-vox` 别名 |
| `-ang <N>` / `-angles <N>` | 投影角度数 |
| `-detN <Nz>` / `-detN <Nz,Nx>` | 探测器 bin 数 |

进入 compact 路径后，矩阵 `metadata.bin` 中的几何会覆盖上述 CLI/INI 值（`applyCompactMetadataGeometry`）。

---

## 算法

| 参数 | 说明 |
|---|---|
| `-opti <mlem\|bsrem>` | 优化算法；自动加载 `config/algo_<name>.conf` |
| `-it <N[:save]>` | 迭代次数与保存间隔 |
| `-th <N>` / `-threads <N>` / `-t <N>` | OpenMP 线程数 |
| `-cudaDevice <id>` | 选择 CUDA device，覆盖环境变量 `SPECT_CUDA_DEVICE` |

compact-symmetric 布局恒为 subset=1（MLEM 式全量更新），无 `-subsets` 参数。稀疏投影后端固定为 CUDA；旧 `-sparseBackend` / `-spBackend` 参数会直接报错。

### compact CUDA 性能环境变量

| 环境变量 | 发布默认 | 说明 |
|---|---|---|
| `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` | `auto` | 覆盖资源感知的主机端矩阵缓存；仅在当前可用内存保留安全余量时完整缓存系统矩阵 |
| `SPECT_COMPACT_BRANCH_BATCH` | 开启 | 常驻 branch×bed 合批（packed-K 正向 + CSR_ALG2 转置）；设为 `0` 会关闭常驻 MLEM，不是产品 A/B |
| `SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES` | 关闭 | 设为 `0` 才对矩阵 payload 使用 mmap；当前低时延路线不依赖 mmap |

实测低时延部署为 16 GB 显存、64 GB 主存和 SSD/NVMe 矩阵；程序为临时 CUDA 工作保留 6 GiB。host cache 默认按已发现矩阵总量与启动时可用内存自动选择，至少为系统保留 8 GiB 且总内存保留不低于 25%；32 GB 主存通常只缓存部分矩阵，不承诺同等吞吐。共享 GPU 或磁盘上的任务可能改变单次耗时，应交错重复基线/候选并用中位数判断。

设备 GPU（例如 RTX 5060）与签收机不同时，空闲可手动搜索上述环境变量，不要改重建路径：

```powershell
python tools\search_compact_perf.py --chunkCSRDir <matrix> --dfList <beds> --compactDenomCache <auto_hash.mhd> --it 3 --repeats 3
```

详见 `tools/README.md`。6 GiB device reserve 与 2 个 transient slot 是编译常量，不在此搜索范围内。

部署后的基线采集优先使用安装包内的 collector，一条命令串行执行三次并只回传一个文本文件：

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\tools\collect_device_recon_diagnostics.ps1 -InputPath "<Bed1.bin;...>" -MultiBedList -OutputPrefix "D:\SPECT_Result\device_test\recon" -Iterations 20:20 -Repeats 3 -ReportPath "D:\SPECT_Result\device_test\spect_device_diagnostics.txt"
```

报告末尾必须同时满足 `all_cache_hits=True`、`all_resident_prefix_nonzero=True`，性能采用 `median_full_command_s`。报告也会记录 free RAM、进程累计 I/O/工作集和 matrix/output 磁盘总线；若首次含 cache miss，建立 cache 后原样重跑。

### BSREM

| 参数 | 说明 |
|---|---|
| `-bsrem <alpha0[,gamma]>` | BSREM 步长参数 |
| `-bsremCurv [on\|off]` | 曲率分母 `D=S+beta*Dreg` |
| `-bsremCurvFreeze [on\|off]` | warmup 后冻结曲率图 |
| `-bsremCurvWarmup <n>` | 曲率 warmup 迭代数，默认 2 |
| `-bsremCurvEps <v>` | 曲率稳定项，默认 `1e-6` |
| `-bsremSensFloor <rel>` | 冻结 `s < rel*max(s)` 的低灵敏度体素（置零），默认 0=关；用于抑制多床边缘切片发散 |
| `-bsremKappa [on\|off]` | kappa 权重图 `beta_j=beta*kappa_j` |
| `-bsremKappaEps <v>` | kappa 稳定项，默认 `1e-6` |
| `-bsremKappaClip <min,max>` | kappa 裁剪范围，默认 `0.1,10` |
| `-bsremSoftMask [on\|off]` | soft-mask 惩罚增强 |
| `-bsremSoftLambda <v>` | mask 外惩罚增强，默认 100 |
| `-bsremEquiv [on\|off]` | MLEM 退化模式：`beta=0`、`alpha=1`、关闭曲率与相对步长钳位 |
| `-bsremDiag [on\|off]` | 写出 `iteration_metrics.csv`（目标函数、投影梯度、钳位） |
| `-bsremDiagInterval <n>` | 第 10 轮之后的诊断间隔，默认 5 |

---

## 正则化

主入口：

```powershell
-reg <type[,beta,...]>
```

| 类型 | 格式 |
|---|---|
| `None` | `-reg None` |
| `TV` / `TVIsotropic` | `-reg TV,0.01,1e-6` |
| `RDP` | `-reg RDP,beta,gamma,delta` |

公开正则化只保留 TV/RDP。`-regBeta` 和 `-regDelta` 可作为简单覆盖；旧 anatomical/NLM/Tikhonov/MRP/LogCosh/Bowsher/PLS 入口与对应细调参数会在 parser 阶段拒绝。

TV/RDP 可选使用已配准到重建存储网格的 HU CT 作为边缘权重：

```powershell
-regCT registered_ct_hu.mhd -regCtSigma 100 -regCtFloor 0.05
```

邻接体素权重为 `floor + (1-floor)*exp(-0.5*(ΔHU/sigma_HU)^2)`。HU 不需要归一化到 `[0,1]`，负值合法；只要 `regCtSigma` 使用同一 HU 单位即可。CT 边缘越强，跨边缘正则耦合越弱，`floor` 防止完全断开。解剖引导只接受显式 `-regCT`；`-attMap` 不再隐式复用，因为它使用逐床 local-matrix-Z，而 `-regCT` 使用 reconstruction-storage-Z。未给 `-regCT` 时严格退化为 activity-only TV/RDP。

---

## 滤波与 PSF

| 参数 | 说明 |
|---|---|
| `-psf [fwhm_mm[,N]]` | 启用 image-domain Gaussian PSF；FWHM 单位 mm；`N` 表示 `N*sigma` 截断（发布默认 `1.3,3`） |
| `-psfKernel <adjoint\|castor>` | `adjoint`：体素中心采样 Gaussian、镜像边界、反向使用真伴随；`castor`：erf 体素积分核、零边界、仅 Z 方向边界归一化，反向严格复用正向算子。发布默认 `castor` |
| `-psfMode <none\|operator\|sieve>` | `sieve` 的迭代轨迹与 `operator` 相同；每个保存点另对副本施加同一 `psfKernel` 正向卷积，不回写迭代状态。`sieve` 拒绝 in_recon smooth/Butterworth |
| `-smooth <sigma>` | 迭代内 Gaussian 平滑 sigma mm；传入即隐含 `smooth_apply_stage=in_recon` |
| `-smoothKernel <n>` | 迭代内 Gaussian kernel half-width |
| `-butter <cutoff[,order]>` | 迭代内 Butterworth 低通（归一化截止），如 `0.35,2`；传入即隐含 `butter_apply_stage=in_recon` |
| `-butterOrder <n>` | 单独设置迭代内 Butterworth order |

`[Filtering]` 以 `smooth_apply_stage` / `butter_apply_stage`（`none|in_recon|output`）控制施加阶段。CLI `-smooth` / `-butter` 始终选择 `in_recon`（改变迭代轨迹）；`output` 仅由 INI 显式配置，不回写迭代状态。发布默认 `[InputOutput] output_filters_overwrite_primary=false`：保留未滤波主输出 `<fout>_it<N>.mhd/.raw`，并从同一份 canonical stitch + mask 输出独立派生 `<fout>_it<N>_smooth` 与 `<fout>_it<N>_butter`。设为 `true` 时，无后缀主输出依次应用已启用的 Smooth、Butterworth；带后缀副本仍独立派生，便于审计。

输出 Smooth 固定使用 CASToR erf 核、`output_smooth_fwhm_mm=0.30`、`output_smooth_sigmas=3`；输出 Butterworth 固定使用 2 倍零填充的径向物理频率响应、`output_butter_cutoff_cpcm=5`、`output_butter_order=8`。`output_nonnegative_clamp=false` 保留原型线性负旁瓣，仅用于同输入数值验证；发布默认 `true`，在每个主/派生副本完成后分别钳为非负。stage 设为 `output` 时对应参数必须为正，否则启动失败。

CASToR PSF 与输出 Smooth 固定走 CUDA CASToR float32 staging；Butterworth 固定 CUDA/cuFFT，不暴露后端选择参数。

注意：`-psf` 单位为 FWHM mm（发布默认 `1.3,3`）。旧现场/冻结包若仍写 `-psf 0.637`，那是历史 sigma 口径，不要与当前 FWHM 混用。

### 原型/主线对比：canonical 输出阶段

主线不包含对象感知 seam/profile 后处理。原型等价比较固定使用同一 Bed 顺序、`z_bed_flip=1`、robust gain-fit、center66/cosine10 拼接；在默认 `output_filters_overwrite_primary=false` 下，主输出与两个派生输出分别对应原型 `stitched`、`stitched_smooth_fwhm0p3` 和 `stitched_butter_fc5cpcm_o8`。带后缀 Smooth/Butterworth 始终从同一未滤波主输出独立计算；覆盖模式是明确的交付输出策略，不替代这项原型等价门槛。

## CTAC / 背景 / 掩码

| 参数 | 说明 |
|---|---|
| `-ctac` | 启用迭代 CTAC |
| `-ctacModel <line-integral>` | CTAC 模型；默认且唯一公开入口为 `line-integral` |
| `-ctacRotSign <1\|-1\|cw\|ccw>` | 角度旋转符号，默认 1 |
| `-ctacAutoMu <vol>` | 从早期重建体自动生成水 mu-map；与 `-attMap` 二选一 |
| `-ctacAutoMuThresh <f>` | auto-mu 阈值，相对 P99.9，范围 `(0,1)`，默认 0.05 |
| `-ctacAutoMuValue <mu>` | 调试覆盖 body fill mu，单位 `mm^-1`；默认由 `[CTAC] auto_mu_nuclide` 查表 |
| `-cyl <diam,height>` / `-cylMask <diam,height>` | 圆柱掩码，单位 mm |
| `-cylc <cx,cy,cz>` / `-cylCenter <cx,cy,cz>` | 圆柱中心，1-based voxel |
| `-maskFromAtt <threshold>` | 从 mu-map 生成 body mask |

默认配置使用 `[CTAC] auto_mu_nuclide = Tc99m`，对应当前历史默认 `0.0152 mm^-1`。支持核素键：`Tc99m`、`I123`、`In111_171`、`In111_245`、`Lu177`、`I131`。`-ctacAutoMuValue` 或 `[CTAC] auto_mu_value` 仅用于调试覆盖。

CTAC line-integral factor 固定由 CUDA 生成，无 CPU fallback 或旧 per-voxel 实现。108×108×80、360 角度时按 16-angle batch 生成，临时 device factor 输出约 29 MiB；compact 系统矩阵 resident cache 仍可能主动使用其余显存，两者不应混作 factor 峰值。

---

## 多床位与输出变换

固定 6module 几何：单床 ROI `30 mm × 21.6 mm`、输出 `100×100×72`；多床 ROI `36 mm × 66 mm`，每床中心 `step+10` 层（0.3 mm 床位为 66），床位步进 `56×0.3=16.8 mm`。

| 参数 | 说明 |
|---|---|
| `-bedOverlap <b1,b2,b3,b4,b5>` | 固定 `Bed1..Bed5` 物理 overlap 表；**派生值** `overlap = baseNz - round(bed_step_mm / voxel_z)`（260601 为 164），可省略（自动推算）或传错（自动纠正 + warning） |
| `-bedStepMm <mm>` / `[MultiBed] bed_step_mm` | 扫描仪机械床步进（默认 `16.8` mm，权威输入）；输出步进 `step = round(bed_step_mm / voxel_z)`，中心裁剪 `step+10` 层 |
| `-detZRowWeightsFile <file>` / `[MultiBed] detz_row_weights_file` | 可选 detZ/per-bin/full-projection 效率权重，仅作用于 compact sensitivity denominator；默认空 |
| `-normFactorFile <file>` / `[MultiBed] normalization_factor_file` | 全投影校正因子 `c`；对 sensitivity 施加 `1/c`；默认空 |
| `-detZEdgeCensor <k>` / `[MultiBed] detz_edge_censor` | compact 分子+分母每侧屏蔽 k 行 detector-Z；默认 0 |
| `-cropEnd <top[,end]>` | 整体输出翻转后将 Z 首尾指定层置零，不改变层数 |
| `-zbedflip [0\|1]` | local system-matrix Z 与 reconstruction-storage Z 的逐床映射；发布默认 1。`-attMap` 必须使用 local-matrix-Z，`-regCT` 不应用此反转 |
| `-zoutflip [0\|1]` | canonical 拼接后仅翻转整体保存输出 Z；默认 0，不得预先作用于 `-attMap/-regCT` |
| `-xoutflip [0\|1]` | canonical 拼接后翻转整体输出 X；默认 0 |
| `-seamXyAlign [0\|1]` / `[MultiBed] seam_xy_align` | 重叠区 NCC 整数 XY 平移后再 cosine 融合；默认 0 |

物理合同固定为 `Bed1=min physical z`，后续 Bed 向较大 physical Z 移动；路径含 `BedN` 时 parser 先按床号排序并拒绝非连续床号。不要逆序文件或整幅翻转地图来补偿 `z_bed_flip=1`。

多床迭代体按非重叠床段存储。保存时从每床取中心 `step+10` 层（`step = round(bed_step_mm / voxel_z)`，默认 `-bedStepMm 16.8`；`-bedOverlap` 须满足 `baseNz - bedOverlap = step`；260601 为 56 层 → 中心 66 层），在相邻的 10 层配准区域按原型 log-ratio median/MAD 估计标量 gain（Bed1 锚定；始终应用原型 log-ratio 链），随后以固定 `step` 层步进和归一化 cosine 权重融合。cosine 沿**输出 Z** 淡入：融合入口继续用上一床，再淡入下一床；`z_bed_flip=1` 不反转这组权重。最终 `Z = crop + (bed_count - 1) * step`，再做整体 Z/X 翻转、`cropEnd` 和固定 XYZ 中心裁剪。

`crop_method`、`sigmoid`、`direct`、seam/profile 后处理、输出 FOV mask 与对象分类均已退场。Mouse/EP/Derenzo 共用相同生产输出路径，不按对象选择算法。

预处理投影与矩阵必须配对（详见 `Recon_Interface_Migration_Field_CN.md`）：260601 `-chunkCSRDir` 对应 `PROJECTION_VARIANT=identity`，`-zbedflip` 保留包默认 1。

## 输出与日志

| 参数 | 说明 |
|---|---|
| `-fout <dir/name>` | 输出路径前缀；传入绝对路径（含盘符/UNC）时，`.log` 与 MHD/RAW 均写入该绝对目录 |
| `-o <dir>` / `-output <dir>` | 输出目录 |
| `-n <name>` / `-name <name>` | 输出文件名前缀 |
| `-vb <0\|1\|2>` | 日志级别；0=安静，1=INFO 默认，2=DEBUG。进度行 `[ITER] n/N compact` 在 `-vb 0` 也会打到控制台 |
| `-v` | 打印当前精确构建号及内嵌的完整 `CHANGELOG.md` 后退出，不依赖当前目录或外部文件。`-v <n>` 会失败（那是 verbosity，请用 `-vb`） |
| `--version` | 只打印单行构建号后退出，保留给脚本解析 |

默认输出主 MHD/RAW、独立 `_smooth`/`_butter` MHD/RAW、`recon_manifest.json` 与自动日志 `<outputDir>/<outputName>_YYYYMMDD_HHMMSS.log`。manifest 记录滤波阶段、覆盖开关、主输出派生顺序、钳制策略及实际六个体文件。无论覆盖开关状态，下游若只取最终交付结果都选择无后缀 `<fout>_it<N>.mhd/.raw`。`-fout D:/data/case/run1`（或 `D:\data\case\run1`）会把 log 与体积都写到绝对目录 `D:\data\case\`，前缀 `run1`；相对 `-fout` 则相对**进程 CWD**（与矩阵/配置按 exe 目录解析不同）。启动时打印 `Logging console output to: ...` 与 `Output directory: ...`（`-vb 0` 也会在控制台打出路径与 `[ITER]`；其余 INFO 仍写入 `.log`）。`-fout path/run.mhd` 会剥掉 `.mhd/.raw/.img`；路径以分隔符结尾或指向已存在目录时，视为输出目录（默认前缀 `recon`）。进度条请匹配 `YYYY-MM-DD HH:MM:SS [ITER] n/N compact`（必须含 `[ITER]`，不要用裸 `n/n`，以免碰到 `device=0/1`）。

---

## 高级选项

| 参数 | 说明 |
|---|---|
| `-config <path>` | 全局配置文件 |
| `-algoConf <path>` | 算法配置文件 |
| `-dumpConf` | 打印合并配置并退出 |
| `-h` / `-help` / `--help` | 打印 help |

---

## 已删除参数的兼容行为

以下入口已退场，parser 只打印 `[NOTE] Ignoring unsupported CLI argument` 并跳过参数及其单个值，不再中止重建。仍应从上位机删除这些字段；已识别的正式参数若取值非法，仍以退出码 1 失败。易与正式入口混淆的别名 `-iter` / `-iterations` / `-verbose` / `-verbosity` 以及 `-v <n>` 会失败，避免 silently 跑错迭代数或把版本查询当成重建成功。

| 已删参数 | 当前行为 |
|---|---|
| `-indexMat` / `-weightMat` / `-loadCSR` / `-mapCSR` / `-saveCSR` / `-compactMeta` / `-matrixLayout` | `[NOTE]` 后忽略（矩阵唯一入口 `-chunkCSRDir`） |
| `-subsets` | `[NOTE]` 后忽略；compact 布局恒单 subset |
| `-alg` | `[NOTE]` 后忽略；正式入口为 `-opti mlem\|bsrem` |
| `-opti admm` | `Unknown optimisation algorithm` |
| `-sparseBackend` / `-spBackend` | `[NOTE]` 后忽略；稀疏后端固定 CUDA |
| `-conv` / `-psfProj` | `[NOTE]` 后忽略；公开入口为 image-domain `-psf` |
| `-butterBackend` / `-bwBackend` | `[NOTE]` 后忽略；Butterworth 固定 cuFFT |
| `-prior` / `-regPower` / `-regSigma` / `-regKernel` 等 anatomical 细调 | `[NOTE]` 后忽略；公开正则仅 TV/RDP |
| `-joe` | `[NOTE]` 后忽略；Joseph-like 平滑退场 |
| `-Aprune` / `-Astream` / `-ApruneMask` | `[NOTE]` 后忽略；compact chunk CSRB 在生成阶段完成剪枝 |

INI 死键（`[SystemMatrix] matrix_layout` / `map_csr` / `prune_epsilon` / `stream_build` / `prune_mask`、`[Reconstruction] subsets` / `joseph_passes`、`[MultiBed] overlap_in_recon` 及 ADMM 相关）已从 `config/spect_config.ini` 移除；未知 INI 键被静默忽略。

---

## 构建与测试入口

```powershell
build_release.bat
scripts\build_cuda_scaffold.ps1 -Build

build\Release\bin\Release\spect_tests.exe --gtest_brief=1
build\Release\bin\Release\spect_cuda_scaffold_tests.exe --gtest_brief=1
```

开发包入口通常为：

```powershell
cd staging\SPECT_recon_rc_dev
.\bin\SPECT_recon.exe ...
```
