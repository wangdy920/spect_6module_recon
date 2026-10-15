# SPECT_recon CLI 使用说明（中文）

> 更新日期：2026-09-21（原生 CSRB 双窗与完整网格多床候选）
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
# 先按方案的matrix_id部署矩阵；默认Tc OSEM50×10
.\bin\SPECT_recon.exe -isotope Tc99m -df proj.bin -o recon_output
# 显式方案替换默认方案，未写字段继承全局
.\bin\SPECT_recon.exe -isotope Tc99m -df proj.bin -algoConf bin/config/profiles/uniformity_Tc99m_BSREM_TV.conf -o recon_tv
# Lu默认合并窗＋单矩阵，原生CSRB OSEM50×10，无正则
.\bin\SPECT_recon.exe -isotope Lu177 -df Bed1.bin -o recon_lu
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

全局 `[Reconstruction] default_config_<核素>` / `default_config` 选择一份方案，`-algoConf` 显式替换；`-config` 也可直接指向含 `[Algorithm]` 且无 `default_config` 的方案。缺失字段继承全局，不叠加另一份默认方案。`-opti`只改优化器，只有无默认方案的旧全局配置才按算法查找 `algo_<name>.conf`。

`[Compatibility] recon_sections_override_cli=false`：CLI > 所选方案 > 全局；true：所选方案 > 全局 > CLI，保护配置已填写的Algorithm/BSREM/Regularization/PSF/Mask/Filtering/SystemMatrix等重建字段。空值 `key =` 在任何一层都等同未配置：继承下层/默认值，不清空全局值，也不压制 CLI。归一化只在全局 `[Normalization] factor_file_<核素>` 绑定，显式 `-normFactorFile` 替换它。输入输出、设备、线程和日志仍由调用方指定。源码默认false，当前6Module部署保留true。

矩阵以合并配置的 `root + matrix_id/matrix_id2` 精确查找 `root/id` 或 `root/isotope/id`，缺失或歧义失败；`root`相对EXE。显式矩阵/缓存CLI也服从优先级。旧配置无ID时才走 `default_chunk_csr_dir` 和唯一目录发现。完整方案表、norm和wrapper说明见 [参数说明](PARAMETERS_CN.template.md)。

固定设备安装可在 `[SystemMatrix]` 设置 `scanner_config=scanners/PSC30-MPH6.ini`，路径相对全局配置目录。非空配置绑定始终优先于 CLI `-scannerConf`，不受 `recon_sections_override_cli` 开关影响；冲突输出 WARNING，配置文件缺失或无效直接失败，不回退到 CLI。未配置时仍接受 `-scannerConf`。这是重建 EXE 的设备配置选择，不会修改或覆盖预处理 wrapper 的设备一致性校验。

PSC30 合并窗 Lu-177 可在对应 scanner INI 的 `[Lu177]` 设置 `input_mode=merged`、`sysmat_high=<208 keV矩阵目录>`、`sysmat_low=`，只传 `BedN.bin`；不要传第二组投影。固定安装的 `[MultiBed] xy_out_rotation_deg=180` 表示 XY 平面旋转半周，不等同于单独 X 镜像；通常配合 `x_out_flip=0`，原有 `z_bed_flip` 和 `z_out_flip` 仍独立生效。此安装配置不自动更改其他设备的默认方向。

---

## 数据与矩阵

| 参数 | 说明 |
|---|---|
| `-df <path>` / `-i <path>` / `-input <path>` | 单床位投影 MAT/BIN |
| `[InputOutput] merge_time_segments=true` | `-df` 指向一个 `案例/时间段/SPECT` 目录时，按时间顺序扫描同案例下的 `YYYYMMDD_HHMMSS/SPECT`，在内存中合并各时间段的同名 `BedN.bin`；默认关闭 |
| `[InputOutput] output_filters_overwrite_primary=true` | 将启用的输出阶段后处理按 Smooth→Butterworth 顺序写入无后缀主输出；默认关闭 |
| `-dfList <list>` | 多床位投影列表，逗号/分号分隔；路径含 `BedN` 时按床位号排序，必须是连续床号；PSC 最多 5 床（Bed1..Bed5），`matrix_grid_output=true` 最多 16 床 |
| `-beds <N\|A:B>` | 在 `-df`（或单项 `-dfList`）给出的目录、或该目录内任一 `BedK.bin` 所在目录中搜索 `Bed1..BedN` / `BedA..BedB`；范围内缺床即报错，范围外的床忽略；`-df2` 同规则 |
| `-bProj <path>` | 背景/散射投影 MAT/BIN |
| `-bProjList <list>` | 多床位背景投影列表，排序后必须与床位数量一致 |
| `-chunkCSRDir <dir>` | compact chunk 目录，包含 `metadata.bin + chunk_*.csrb`；也可用CSRQ及配置matrix_id |
| `-compactDenomCache <mhd>` | 显式复用/保存一个 compact global denominator cache，受配置优先级开关控制 |
| `-ctacFactorCacheDir <dir>` | 复用/保存 CTAC 逐床位、逐角度衰减系数表缓存；默认 `recon_out/ctac_factor_cache` |
| `-projectionResidual [on\|off]` | 最终迭代后按同一 forward/PSF/CTAC 口径写出逐床、angle×detector-Z 投影残差 CSV；默认关闭，不改变迭代或图像输出 |
| `-badDetectorFile <path>` / `[Detector] bad_detector_file` | 按全局编号或 `link/FPGA/dector` 跳过硬件探测器；默认读取配置同目录的 `bad_dector`，当前保护配置包含探测器 68 |
| `-attMap <path>` | 衰减图，mu 单位为 `mm^-1`；多床 MHD 必须标记 exact beds 与 `local_matrix_z` |

`-dfList` 中任一路径含 `BedN` 时，程序会按床号稳定排序；仅把命令文本写成 `Bed5;...;Bed1` **不会**反转实际输入顺序。床号随 patient/physical +Z 增加，Bed1 为最小 Z。少于五床时直接传真实连续子集（如 Bed2,3,4），不补空床；地图 MHD 的 `SPECTBeds` 必须精确相同。

每个能窗、床位和时间段的投影/背景都必须完整匹配矩阵 metadata 的角度数与每角度射线数；MAT 提供探测器形状时还必须匹配 detZ/detX。不会自动截去多余角度，也不以总元素数相同代替布局匹配。所有计数必须有限且 >= 0，零计数合法；负值、NaN、Inf 会报错并标明源文件与从 0 开始的元素/记录索引。MAT 事件表也检查原始计数字段，不允许重复记录聚合掩盖非法值。

多时间段合并示例：配置 `merge_time_segments=true`，并传入 `-df "D:\data\predata\260810_dedenzo_mdp_6\20260810_210556\SPECT"`。程序只纳入同级、名称满足 `YYYYMMDD_HHMMSS` 且含 `SPECT` 的目录；所有时间段必须具有完全相同的 `BedN.bin` 集合与 BIN 大小/头/几何。每份源投影先检查完整几何与计数合法性，再按原顺序逐元素求和，不允许用正负相抵掩盖非法计数。合并后直接进入重建，不写中间投影；manifest 保留每床位的全部来源。任一缺床、几何不一致、非法计数或求和溢出都会失败退出。

有无 CTAC 均可复用分母缓存。按上述 CLI/配置优先级解析后，可使用显式 `-compactDenomCache` 或 `[SystemMatrix] compact_denom_cache_path`，否则在 `compact_denom_cache_dir` 下自动选择 `auto_<manifest-hash>.mhd`。INI 相对路径按 exe 目录解析，发布默认落在 `bin/cache/compact_denom/`，不受终端启动目录影响。

当前 v6 manifest 使用本次矩阵的规范化完整路径、文件大小/修改时间、布局/metadata，以及全局维度、床位 Z 起点、PSF、坏探头、norm、实际行权重、CTAC 因子与能窗指数。**不计算矩阵 SHA，不认证历史矩阵版本。**设备/矩阵选择由调用方负责；同路径原地替换且保留大小和修改时间时，调用方须清除旧分母缓存。正常文件属性或参数变化会产生新自动缓存；矩阵移动路径后也重算。旧 v3/v4/v5 自动缓存升级后一次性 miss，显式旧缓存需改用新路径或有意删除后重建。运行期间不要改写输入。

以下是旧单矩阵预热工具入口，只用于经 dump 确认的单矩阵配置；不替代 joint 两窗缓存准备。可手动预热匹配单矩阵配置的 1–5 床（占位 BIN，投影计数不进入 denom；有 CTAC 时不要用）：

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
| `-opti <mlem\|bsrem>` | 优化器；所选方案不随其自动切换 |
| `-it <A[:B]>` | A次完整迭代、B个角度子集；裸A保留继承的子集 |
| `-save-every <N>` | 每N次保存，0默认仅最后一次 |
| `-subsets <N或N1:i1,N2:i2,...>` | 配合裸 `-it A` 指定子集/阶段，不能与 `-it A:B` 同传 |
| `-th <N>` / `-threads <N>` / `-t <N>` | OpenMP 线程数 |
| `-cudaDevice <id>` | 选择 CUDA device，覆盖环境变量 `SPECT_CUDA_DEVICE` |

MLEM加角度子集即OSEM，BSREM共用划分；单/双窗同格式CSRB和实验joint CSRQ均支持N10。CSRB与CSRQ均已实现CTAC子集路径；CSRQ N>1复用原有逐角度因子分解内核，2026-10-09的K4/K8、驻留/流式、双床双窗GPU数值回归通过，仍实验、未发布。子集数整除compact角度数，递减阶段满足整除关系且迭代数之和为A。当前不提供POSEM像素分支。后端固定CUDA；旧后端选择参数已退役。不能由入口允许或数值回归通过推定生产签收。

### compact CUDA 性能环境变量

| 环境变量 | 发布默认 | 说明 |
|---|---|---|
| `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` | `auto` | 主机端矩阵缓存预算；接受 auto、非负字节数及 K/M/G/T 后缀（1024 倍数），0 关闭永久 host cache；非法/非有限值报错。CSRB auto 按可用内存保留余量；CSRQ 不使用此缓存（设置也不生效，其分块在显存或锁页副本中，见下一行）。仅完整覆盖矩阵的 CSRB host cache 会逐块锁页；驱动拒绝锁页时，已锁页分块保留，其余分块留在 host cache 中走普通内存上传（结果不变、传输变慢），`.log` 记录 `[WARNING] compact host cache page-lock refused ... page-locked k / n chunks`，不再因此报 `out of memory` 退出 |
| `SPECT_CSRQ_HOST_PIN_BYTES` | `auto` | 显存放不下、走流式的 CSRQ 矩阵最多锁页多少主存（auto 或字节数，K/M/G/T 后缀）。auto 与 CSRB host cache 同一余量规则：总内存保留 max(8 GiB, 25%)，启动时可用内存再保留 8 GiB。超出预算或被驱动拒绝锁页的分块只锁页行指针/行最大值（驱动拒绝时先归还已锁页分块，保证行指针与中转缓冲可用），run/值每轮从已校验且未改动的原文件按需读回（两块 256 MiB 锁页中转），结果不变；`.log` 的 `[NOTE] CSRQ matrix ... re-read from file` 与 `[MATRIX_IO] scope=run csrq_file_reread_bytes=` 记录其规模。显存驻留的矩阵不受影响 |
| 已退役实验开关 | 不设置 | `SPECT_COMPACT_BRANCH_BATCH` / `SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES` 不再接受；历史实验使用对应冻结 EXE |

实测低时延部署为 16 GB 显存、64 GB 主存和 SSD/NVMe 矩阵；程序为临时 CUDA 工作保留 6 GiB。host cache 默认按已发现矩阵总量与启动时可用内存自动选择，至少为系统保留 8 GiB 且总内存保留不低于 25%；32 GB 主存通常只缓存部分矩阵，不承诺同等吞吐。共享 GPU 或磁盘上的任务可能改变单次耗时，应交错重复基线/候选并用中位数判断。

设备 GPU（例如 RTX 5060）与签收机不同时，空闲可手动搜索上述环境变量，不要改重建路径：

```powershell
python tools\search_compact_perf.py --chunkCSRDir <matrix> --dfList <beds> --compactDenomCache <auto_hash.mhd> --it 3 --repeats 3
```

详见 `tools/README.md`。6 GiB device reserve 与 2 个 transient slot 是编译常量，不在此搜索范围内。

部署后的基线采集优先使用安装包内的 collector，一条命令串行执行三次并只回传一个文本文件：

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\tools\collect_device_recon_diagnostics.ps1 -InputPath "<Bed1.bin;...>" -MultiBedList -OutputPrefix "D:\SPECT_Result\device_test\recon" -Iterations 20:10 -Repeats 3 -ReportPath "D:\SPECT_Result\device_test\spect_device_diagnostics.txt"
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

公开正则化只保留 TV/RDP，由BSREM执行。每个子更新用当前图像计算正则梯度，数据梯度按子集数缩放，β不再乘子集数；不是完整迭代后才加正则。`-regBeta` 和 `-regDelta` 可作为简单覆盖；旧 anatomical/NLM/Tikhonov/MRP/LogCosh/Bowsher/PLS 入口与对应细调参数会在 parser 阶段拒绝。

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

`[Filtering]` 以 `smooth_apply_stage` / `butter_apply_stage`（`none|in_recon|output`）控制施加阶段。CLI `-smooth` / `-butter` 始终选择 `in_recon`（改变迭代轨迹）；`output` 仅由 INI 显式配置，不回写迭代状态。发布默认 `[InputOutput] output_filters_overwrite_primary=false`：保留未滤波主输出 `<fout>_it<N>.mhd/.raw`，默认Gaussian关闭，仅从同一份canonical保存输出独立派生 `<fout>_it<N>_butter`；显式启用输出Gaussian时才另写 `_smooth`。primary包含既定PSF/sieve及保存处理，不等于内部迭代数组。设为 `true` 时，无后缀主输出依次应用已启用的 Smooth、Butterworth；带后缀副本仍独立派生，便于审计。

输出 Smooth 固定使用 CASToR erf 核、`output_smooth_fwhm_mm=0.30`、`output_smooth_sigmas=3`；输出 Butterworth 固定使用 2 倍零填充的径向物理频率响应、`output_butter_cutoff_cpcm=5`、`output_butter_order=8`。`output_nonnegative_clamp=false` 保留原型线性负旁瓣，仅用于同输入数值验证；发布默认 `true`，在每个主/派生副本完成后分别钳为非负。stage 设为 `output` 时对应参数必须为正，否则启动失败。

CASToR PSF 与输出 Smooth 固定走 CUDA CASToR float32 staging；Butterworth 固定 CUDA/cuFFT，不暴露后端选择参数。

注意：`-psf` 单位为 FWHM mm（发布默认 `1.3,3`）。旧现场/冻结包若仍写 `-psf 0.637`，那是历史 sigma 口径，不要与当前 FWHM 混用。

### 原型/主线对比：canonical 输出阶段

主线不包含对象感知 seam/profile 后处理。原型等价比较固定使用同一 Bed 顺序、`z_bed_flip=1`、robust gain-fit、center66/cosine10 拼接；在默认 `output_filters_overwrite_primary=false` 下，显式启用两种输出滤波进行历史原型对照时，主输出与两个派生输出分别对应原型 `stitched`、`stitched_smooth_fwhm0p3` 和 `stitched_butter_fc5cpcm_o8`。带后缀 Smooth/Butterworth 始终从同一未滤波主输出独立计算；覆盖模式是明确的交付输出策略，不替代这项原型等价门槛。

## CTAC / 背景 / 掩码

| 参数 | 说明 |
|---|---|
| `-ctac` | 启用迭代 CTAC |
| `-ctacModel <line-integral>` | CTAC 模型；默认且唯一公开入口为 `line-integral` |
| `-ctacRotSign <1\|-1\|cw\|ccw>` | 角度旋转符号，默认 1 |
| `-ctacAutoMu <vol>` | 从未变换的全网格早期重建 MHD 生成水 mu-map；要求 `SPECTAutoMuCompatible=1`；与 `-attMap` 二选一 |
| `-ctacAutoMuThresh <f>` | auto-mu 阈值，相对 P99.9，范围 `(0,1)`，默认 0.05 |
| `-ctacAutoMuValue <mu>` | 调试覆盖 body fill mu，单位 `mm^-1`；默认由 `[CTAC] auto_mu_nuclide` 查表 |
| `-ctacMapEnergy <keV>` | μ 图（或 auto-mu 填充值）对应的光子能量；默认取 `auto_mu_nuclide` 预设能量（INI `[CTAC] mu_map_energy_kev`） |
| `-ctacWindowEnergies <keV,...>` | 每个能窗的光电峰能量；衰减表按 `att^(μ_w(E_k)/μ_w(E_map))`（NIST 液态水表 log-log 插值）逐窗缩放。双窗 `-ctac` 必填，如 Lu-177 `208.4,112.9`；CSRB 和 CSRQ 的因子分解路径均支持能量指数（INI `[CTAC] window_energies_kev`）；候选已支持双 CSRB，混合格式拒绝 |
| `-cyl <diam,height>` / `-cylMask <diam,height>` | 圆柱掩码，单位 mm |
| `-cylc <cx,cy,cz>` / `-cylCenter <cx,cy,cz>` | 圆柱中心，1-based voxel |
| `-maskFromAtt <threshold>` | 从 mu-map 生成 body mask |

默认配置使用 `[CTAC] auto_mu_nuclide = Tc99m`，对应当前历史默认 `0.0152 mm^-1`。支持核素键：`Tc99m`、`I123`、`In111_171`、`In111_245`、`Lu177`、`I131`。`-ctacAutoMuValue` 或 `[CTAC] auto_mu_value` 仅用于调试覆盖。

CTAC line-integral factor 固定由 CUDA 生成，无 CPU fallback 或旧 per-voxel 实现。108×108×80、360 角度时按 16-angle batch 生成，临时 device factor 输出约 29 MiB；compact 系统矩阵 resident cache 仍可能主动使用其余显存，两者不应混作 factor 峰值。

CSRQ 多子集 CTAC 不新增调用参数：在已有正确矩阵/投影配置上使用 `-it 50:10 -ctac -attMap D:/data/case/maps/att_local.mhd`。双窗仍需对应的 `-ctacWindowEnergies`，子集数必须整除 compact 角度数；不会自动改为 N=1。CT 或有效 AutoMu 生成的衰减表走同一路径，仅上传当前子集需要的角度切片。此项为待 GPU 回归的工作树候选，不表示已部署的 SPECT_Core 具备该能力。

### 输入检查与缓存

`-attMap`、`-maskFromAtt` 使用的 μ 图必须与内部重建体尺寸和间距一致（间距容差 `1e-6*max(1,预期间距)` mm，仅容纳头部精度舍入），值须有限且非负，单位 `mm^-1`。`-regCT` 同样检查网格和有限值，但允许合法负 HU；它是正则边权输入，不会自行开启 CTAC。多床 μ 图仍按 local-matrix-Z 堆叠，HU 图按 reconstruction-storage-Z 堆叠；不是拼接后的显示体。

AutoMu 保存输出现在带 `SPECTAutoMuCompatible=0/1`。旋转、X/Z 翻转、裁剪/端层置零、拼床或大端保存结果标记为 0；无标记旧文件也拒绝，不能仅靠同尺寸推定坐标一致。使用单床早期 run 的全网格、无输出变换 MHD（`matrix_grid_output=true`、输出翻转/旋转/裁剪均关闭）；多床需调用方提供经过坐标核实的未拼接 storage 体。不要给旧显示图手工补 1 来绕过检查。缺 CT 时不传 `-ctac` 就是不作衰减校正；只有满足活动分布近似水体假设的仿体才考虑 AutoMu，不能把它当作测量 CT。

CTAC 因子缓存 v2 同时记录 MHD/HDR 头和实际 RAW/IMG 的规范化路径、大小、修改时间，并记录 AutoMu 阈值、填充值与圆柱掩码参数。只更新数据文件也会使缓存失效；旧缓存首次 miss。**没有计算矩阵或体数据 SHA。**同路径原地改写且保留大小/时间时由调用方清除旧缓存；运行中输入必须不变。

### SC60 的 CT 准备

`scripts/prepare_ct_mu_map.py --recon-manifest` 从现有运行 manifest 读取完整网格、间距、床号、步进及 Z 映射，不套用 PSC 的 108/120、0.3 mm 或 16.8 mm。参考 SPECT DICOM 必须来自对应无旋转、翻转、裁剪和额外导出变换的全网格 run；不满足时明确报错。单床未知步进可为 0，多床必须已有明确的整数层步进。CT/SPECT 的 FoR/方向/IPP 仍需正确，脚本不会通过网格匹配证明真实配准。

```powershell
python scripts/prepare_ct_mu_map.py `
  --ct-dir D:/data/case/CT --spect-dicom-dir D:/data/case/SPECT_reference `
  --recon-manifest D:/data/case/reference/recon_manifest.json `
  --output D:/data/case/maps/physical_mu.mhd `
  --attenuation-mu-output D:/data/case/maps/att_local.mhd `
  --regct-hu-output D:/data/case/maps/ct_storage.mhd
```

重建沿用 `-ctac -attMap D:/data/case/maps/att_local.mhd`；需要 CT 引导正则时另传 `-regCT .../ct_storage.mhd`。示例沿用脚本现有 Tc 水 μ 默认值，其他能量要显式设置 `--water-mu` 和重建能量参数。`--recon-manifest` 不与 `--profile` 混用，也不自动套用 PSC 的 `mdp_device` 配准预设；原 PSC `--profile device160/recon220` 调用保持可用。

---

## 多床位与输出变换

下面的固定裁剪/gain-fit 是 **PSC 默认合同**。SC60 recipe 使用 `[InputOutput] matrix_grid_output=true`，完整保留每床矩阵网格及 ROI 设置，最多 16 床。多床为联合重建：各床网格按 `床序*step` 放入一个全局体（Z 为 `baseNz+(床数-1)*step`），重叠体素合并灵敏度与反投影后统一更新，直接保存该全局体，不做 cosine 融合，不拟合床间增益；暂不支持 CTAC、`-ctacAutoMu`、`-maskFromAtt`。多床必须给出有限正、整数体素层对齐的 `-bedStepMm`，不能有网格间隙；不支持 seam XY alignment。例如 `-df D:\scan\SPECT -beds 12 -bedStepMm 12`。SC60 设备包的 `spect_config.ini` 写入 `bed_step_mm=12`（2026-10-13 五床线源实测，12/0.4=30 层），不继承 PSC 16.8 mm。详见 [现行格式合同](Format_Reader_Matrix_CN.md#sc60-网格输出)。

固定 6module 几何：单床 ROI `30 mm × 21.6 mm`、默认输出 `100×100×66`（`step+10`）；`-singleBedStepAligned 0` 恢复旧 `100×100×72`；多床 ROI `36 mm × 66 mm`，每床中心 `step+10` 层（0.3 mm 床位为 66），床位步进 `56×0.3=16.8 mm`。

| 参数 | 说明 |
|---|---|
| `-bedStepMm <mm>` / `[MultiBed] bed_step_mm` | 扫描仪机械床步进（PSC 默认 `16.8` mm），多床几何的唯一输入：`step = round(bed_step_mm / voxel_z)`，每个床间 `overlap = baseNz - step`（220 层 0.3 mm 为 164），PSC 中心裁剪 `step+10` 层。旧 `-bedOverlap` / `bed_overlap` 仍被接受但忽略 |
| `-detZRowWeightsFile <file>` / `[MultiBed] detz_row_weights_file` | 可选 detZ/per-bin/full-projection 效率 e；对 sensitivity 施加 e，有背景时内部转换为 b/e；默认空 |
| `-normFactorFile <file>` / 全局 `[Normalization] factor_file_<核素>` | 全投影正校正因子 c，效率 e=1/c；支持同计数空间的背景；默认空。显式 CLI 替换全局绑定 |

归一化与背景均默认关闭；只在已有对应校正/背景数据时配置。归一化是安装标定，只在全局 `spect_config.ini` 的 `[Normalization]` 中按核素绑定：`factor_file_Tc99m`、`factor_file_Lu177`，joint 第二窗为 `factor_file2_Lu177`；recipe/profile、旧 `[MultiBed] normalization_factor_file` 或旧无后缀 `factor_file` 都会被拒绝。相对路径以 EXE 为基准；显式 `-normFactorFile`（相对 CWD）替换该绑定，且不带第二窗。背景按原始探测计数传入，不提前乘 c；完整模型及窗口/时间约定见 [矩阵合同](Format_Reader_Matrix_CN.md)。例如在已有双窗、两床重建命令上增加：

```text
-bProjList "D:/case/Bed1_high_bg.bin;D:/case/Bed2_high_bg.bin" -bProjList2 "D:/case/Bed1_low_bg.bin;D:/case/Bed2_low_bg.bin"
```

程序不会估计散射、自动寻找背景或从另一能窗继承背景。`-projectionResidual 1` 输出有效行的 `eHx+b` 与计数比较；被屏蔽的 detector-Z 行仍保留 CSV 行，但 count=0，不参与统计。
| `-detZEdgeCensor <k>` / `[MultiBed] detz_edge_censor` | compact 分子+分母每侧屏蔽 k 行 detector-Z；默认 0 |
| `-cropEnd <top[,end]>` | 整体输出翻转后将 Z 首尾指定层置零，不改变层数 |
| `-zbedflip [0\|1]` | local system-matrix Z 与 reconstruction-storage Z 的逐床映射；发布默认 1。`-attMap` 必须使用 local-matrix-Z，`-regCT` 不应用此反转 |
| `-zoutflip [0\|1]` | canonical 拼接后仅翻转整体保存输出 Z；默认 0，不得预先作用于 `-attMap/-regCT` |
| `-xoutflip [0\|1]` | canonical 拼接后翻转整体输出 X；默认 0 |
| `-seamXyAlign [0\|1]` / `[MultiBed] seam_xy_align` | 重叠区 NCC 整数 XY 平移后再 cosine 融合；默认 0 |

物理合同固定为 `Bed1=min physical z`，后续 Bed 向较大 physical Z 移动；路径含 `BedN` 时 parser 先按床号排序并拒绝非连续床号。不要逆序文件或整幅翻转地图来补偿 `z_bed_flip=1`。

多床迭代体按非重叠床段存储。保存时从每床取中心 `step+10` 层（`step = round(bed_step_mm / voxel_z)`，默认 `-bedStepMm 16.8`；overlap 恒为 `baseNz - step`；260601 为 56 层 → 中心 66 层），在相邻的 10 层配准区域按原型 log-ratio median/MAD 估计标量 gain（Bed1 锚定；始终应用原型 log-ratio 链），随后以固定 `step` 层步进和归一化 cosine 权重融合。cosine 沿**输出 Z** 淡入：融合入口继续用上一床，再淡入下一床；`z_bed_flip=1` 不反转这组权重。最终 `Z = crop + (bed_count - 1) * step`，再做整体 Z/X 翻转、`cropEnd` 和固定 XYZ 中心裁剪。

`crop_method`、`sigmoid`、`direct`、seam/profile 后处理、输出 FOV mask 与对象分类均已退场。Mouse/EP/Derenzo 共用相同生产输出路径，不按对象选择算法。

预处理投影与矩阵必须配对（详见 `Recon_Interface_Migration_Field_CN.md`）：本批Tc/Lu方案均使用 `PROJECTION_VARIANT=identity`，`-zbedflip` 保留包默认 1。

## 输出与日志

| 参数 | 说明 |
|---|---|
| `-fout <dir/name>` | 输出路径前缀；传入绝对路径（含盘符/UNC）时，`.log` 与 MHD/RAW 均写入该绝对目录 |
| `-o <dir>` / `-output <dir>` | 输出目录 |
| `-n <name>` / `-name <name>` | 输出文件名前缀 |
| `-vb <0\|1\|2>` | 日志级别；0=最小控制台（部署 `spect_config.ini` 默认）：控制台只输出 `YYYY-MM-DD HH:MM:SS [ITER] n/N compact` 进度行与错误行（`Error`/`[ERROR]`/`[FATAL]` 开头），INFO、`[NOTE]`、`[WARNING]` 只写入 `.log`；1=INFO（无配置时的程序默认），2=DEBUG。CLI `-vb` 优先于配置，现场诊断用 `-vb 1` |
| `-v` | 打印当前精确构建号及内嵌的完整 `CHANGELOG.md` 后退出，不依赖当前目录或外部文件。`-v <n>` 会失败（那是 verbosity，请用 `-vb`） |
| `--version` | 只打印单行构建号后退出，保留给脚本解析 |

`[ITER] n/N compact` 只表示完整迭代开始，每轮一条，N 为完整迭代总数，不乘子集数；例如 `50:10` 共 50 条。普通日志与终端使用相同的整轮进度，日志另保留 `[ITER_TIME]` 每轮耗时与末尾 `[PERF]` 汇总。`[SUBSET]`、`[SUBSET_TIME]` 和逐子集缓存明细仅在 `-vb 2` 时写入终端和日志，不参与旧进度识别。最后一条 `[ITER] N/N compact` 不代表输出写入完成；所有输出完成后，终端和日志均打印 `Reconstruction finished successfully.`（包括 `-vb 0`）。上位机仍须以进程退出码 0 和产物状态确认成功。

默认仅末次输出主 MHD/RAW、独立 `_butter` MHD/RAW、`recon_manifest.json` 与自动日志 `<outputDir>/<outputName>_YYYYMMDD_HHMMSS.log`。manifest 记录滤波阶段、覆盖开关、主输出派生顺序、钳制策略及实际输出文件。无论覆盖开关状态，下游若只取最终交付结果都选择无后缀 `<fout>_it<N>.mhd/.raw`。`-fout D:/data/case/run1`（或 `D:\data\case\run1`）会把 log 与体积都写到绝对目录 `D:\data\case\`，前缀 `run1`；相对 `-fout` 则相对**进程 CWD**（与INI中的矩阵/norm/cache路径按EXE解析不同）。`-fout path/run.mhd` 会剥掉 `.mhd/.raw/.img`；路径以分隔符结尾或指向已存在目录时，视为输出目录（默认前缀 `recon`）。进度条请匹配 `YYYY-MM-DD HH:MM:SS [ITER] n/N compact`（必须含 `[ITER]`，不要用裸 `n/n`，以免碰到 `device=0/1`）。

日志中的 `[CLI] Requested command` 仅记录原始调用，不代表实际参数。以 `[CONFIG] Effective reconstruction settings` 为准：该摘要在配置合并、矩阵几何和床位准备后输出，列出选中的 profile、配置优先级、实际算法/完整迭代数/子集调度、投影模型、矩阵路径、PSF、正则及输出变换。`-vb 0` 控制台只显示整轮进度、成功完成、`[WARNING]` 和错误；参数摘要、日志路径和耗时仍写入 `.log`。`-vb 1` 同时在控制台显示这些普通信息；`-vb 2` 才启用子集和底层调试明细。

---

## 高级选项

| 参数 | 说明 |
|---|---|
| `-config <path>` | 全局INI或完整重建方案 |
| `-algoConf <path>` | 显式完整方案，替换默认方案 |
| `-isotope <Tc99m或Lu177>` | 采集身份断言及核素默认方案选择 |
| `-df2/-dfList2`、`-chunkCSRDir2` | joint第二窗投影及同格式矩阵（CSRB/CSRB 或实验 CSRQ/CSRQ），床号及 metadata 共享坐标须与第一窗相同 |
| `-energyWeights <w1,w2>` | 两窗预测计数权重，Lu方案顺序208/113为0.1036/0.0617 |
| `-dumpConf` | 打印合并配置并退出 |
| `-h` / `-help` / `--help` | 打印 help |

---

## 已删除参数的兼容行为

以下入口已退场，parser 只打印 `[NOTE] Ignoring unsupported CLI argument` 并跳过参数及其单个值，不再中止重建。仍应从上位机删除这些字段；已识别的正式参数若取值非法，仍以退出码 1 失败。易与正式入口混淆的别名 `-iter` / `-iterations` / `-verbose` / `-verbosity` 以及 `-v <n>` 会失败，避免 silently 跑错迭代数或把版本查询当成重建成功。

| 已删参数 | 当前行为 |
|---|---|
| `-indexMat` / `-weightMat` / `-loadCSR` / `-mapCSR` / `-saveCSR` / `-compactMeta` / `-matrixLayout` | `[NOTE]` 后忽略（矩阵唯一入口 `-chunkCSRDir`） |
| `-subset-mode` / `-subset-diagnostics` | 已删除的像素分支入口，明确失败；`-subsets`仍是正式调度入口 |
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
