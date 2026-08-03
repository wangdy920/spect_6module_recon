# SPECT_recon CLI 使用说明（中文）

> 更新日期：2026-07-31（收敛重构后）  
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

算法默认参数会从同级 `config/algo_<name>.conf`（`algo_mlem.conf` / `algo_bsrem.conf`）加载，也可用 `-algoConf <path>` 指定。CLI 参数覆盖配置文件。

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
| `-dfList <list>` | 多床位投影列表，逗号/分号分隔；路径含 `BedN` 时按床位号排序，最多 5 床 |
| `-bProj <path>` | 背景/散射投影 MAT/BIN |
| `-bProjList <list>` | 多床位背景投影列表，排序后必须与床位数量一致 |
| `-chunkCSRDir <dir>` | compact chunk 目录，包含 `metadata.bin + chunk_*.csrb`；**唯一矩阵入口** |
| `-compactDenomCache <mhd>` | 复用/保存 compact global denominator cache；PSF/FWHM/床位组合需分文件 |
| `-ctacFactorCacheDir <dir>` | 复用/保存 CTAC 逐床位、逐角度衰减系数表缓存；默认 `recon_out/ctac_factor_cache` |
| `-projectionResidual [on\|off]` | 最终迭代后按同一 forward/PSF/CTAC 口径写出逐床、angle×detector-Z 投影残差 CSV；默认关闭，不改变迭代或图像输出 |
| `-badDetectorFile <path>` / `[Detector] bad_detector_file` | 按全局编号或 `link/FPGA/dector` 跳过硬件探测器；默认读取配置同目录的 `bad_dector` 空列表 |
| `-attMap <path>` | 衰减图，mu 单位为 `mm^-1` |

无 CTAC 的 `-chunkCSRDir` 会自动使用 `recon_out/compact_denom_cache/auto_<manifest-hash>.mhd`；manifest 包含矩阵、metadata、全局维度、床位 Z 起点和 PSF 口径，不匹配时不会复用。

`bad_dector` 是带方括号的逐行列表，优先使用最简单的全局探测器编号 `0..95`；也兼容 `link-<3|4>-FPGA-<0..5>-dector-<0..7>` 详细格式。编号遵循采集预处理的 `spect_detector_index`：`global = ((link==4 ? 6 : 0) + FPGA)*8 + dector`。读取任一格式后都会补齐另一种标识；跨格式重复项会报错。例如：

```text
[
  68,
  15
]
```

该列表在 BIN 加载后生成投影行 mask：每个硬件探测器对应一个 FPGA 的 60 个角度和 12x12 detector block。mask 同时将 emission/background 实测行和 sensitivity denominator 中对应行置零，并进入 denom cache identity。无需修改 ENG 预处理；若只在预处理阶段丢事件但不修改 sensitivity，会造成定量偏差。

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

### BSREM

| 参数 | 说明 |
|---|---|
| `-bsrem <alpha0[,gamma]>` | BSREM 步长参数 |
| `-bsremCurv [on\|off]` | 曲率分母 `D=S+beta*Dreg` |
| `-bsremCurvFreeze [on\|off]` | warmup 后冻结曲率图 |
| `-bsremCurvWarmup <n>` | 曲率 warmup 迭代数，默认 2 |
| `-bsremCurvEps <v>` | 曲率稳定项，默认 `1e-6` |
| `-bsremKappa [on\|off]` | kappa 权重图 `beta_j=beta*kappa_j` |
| `-bsremKappaEps <v>` | kappa 稳定项，默认 `1e-6` |
| `-bsremKappaClip <min,max>` | kappa 裁剪范围，默认 `0.1,10` |
| `-bsremSoftMask [on\|off]` | soft-mask 惩罚增强 |
| `-bsremSoftLambda <v>` | mask 外惩罚增强，默认 100 |

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

TV/RDP 可选使用已配准到重建网格的 CT（或 mu-map）作为边缘权重：

```powershell
-regCT registered_ct.mhd -regCtSigma 0.01 -regCtFloor 0.05
```

邻接体素权重为 `floor + (1-floor)*exp(-0.5*(ΔCT/sigma)^2)`。CT 边缘越强，跨边缘正则耦合越弱；`floor` 防止完全断开。显式 `-regCT` 优先，否则在已提供 `-attMap` 时复用同一配准体；两者均未提供时严格退化为 activity-only TV/RDP。`ct_sigma` 的单位必须与输入 CT 强度一致；HU CT 与 mm^-1 mu-map 不应共用同一数值标度。

---

## 滤波与 PSF

| 参数 | 说明 |
|---|---|
| `-psf [fwhm_mm[,N]]` | 启用 image-domain Operator Gaussian PSF；FWHM 单位 mm，默认 `1.5,4`；`N` 表示 `N*sigma` 截断 |
| `-psfMode <None\|Operator>` | PSF 开关模式；使用 `-psf` 且未显式指定时默认 `Operator` |
| `-smooth <sigma>` | Gaussian 平滑 sigma mm |
| `-smoothKernel <n>` | Gaussian kernel half-width |
| `-butter <cutoff[,order]>` | Butterworth 低通滤波，如 `0.35,2` |
| `-butterOrder <n>` | 单独设置 Butterworth order |

Butterworth 后端固定为 CUDA/cuFFT，不再暴露后端选择参数。

注意：`-psf` 单位为 FWHM mm（默认 `1.5,4`）。旧现场/冻结包若仍写 `-psf 0.637`，那是历史 sigma 口径，不要与当前 FWHM 混用。

---

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

多床位仅支持 **stitch**：各床按零 overlap 独立重建，最终输出前按 `crop_method` 拼接相邻床位的内部接缝。

| 参数 | 说明 |
|---|---|
| `-bedOverlap <b1,b2,b3,b4,b5>` | 固定 `Bed1..Bed5` overlap 表；`b1=Bed1-Bed2`，只传一个值时复制到所有固定槽 |
| `[MultiBed] crop_method` / `-bedCropMethod` | `center66_cosine10`：中心 66 层、固定 stride56、overlap10 归一化 cosine 融合（默认）；`sigmoid`：完整床位 overlap 的 legacy 对照；`direct`：hard-cut 对照 |
| `[MultiBed] seam_postprocess` / `-bedSeamPostprocess` | 请求启用接缝后处理（默认 1）；仅在多床 weighted 输出 overlap 生效时运行 |
| `[MultiBed] fov_radius_mm` | 最终输出的圆形 XY FOV 半径（mm）；`0` 关闭。当前 0.3 mm 数据的验证候选为 `15.0`（R50），边缘固定使用 4 voxel taper |
| `[MultiBed] analysis_fov_radius_mm` | 自动分类和轮廓支持检测使用的固定 XY FOV 半径；`0` 跟随 `fov_radius_mm` |
| `[MultiBed] mouse_seam_strength` | 仅用于显式 `sigmoid` 控制路径的 mouse 双床共同组织置信度权重调整 `[0,1]`；默认 `0` |
| `-detZRowWeightsFile <file>` / `[MultiBed] detz_row_weights_file` | norm W2 探测器效率权重，仅作用于 compact sensitivity denominator；发布默认为空/关闭 |
| `-normFactorFile <file>` / `[MultiBed] normalization_factor_file` | 全投影校正因子 `c`；对 sensitivity 施加倒数 `1/c`；opt-in，默认关闭 |
| `-detZEdgeCensor <k>` / `[MultiBed] detz_edge_censor` | compact 分子+分母每侧屏蔽 k 行探测器 Z 边缘（默认 0） |
| `-cropEnd <top[,end]>` | 将最终输出 Z 轴首尾指定层置零，在输出翻转之后执行；不改变层数 |
| `-zbedflip [0\|1]` | 反转 `-dfList` 的床位局部 Z（输出阶段按床翻转）。配对规则必须与矩阵和预处理方向一致；发布默认 `1` |
| `-zoutflip [0\|1]` | 对最终整体输出再做 Z 翻转，默认 0 |
| `-xoutflip [0\|1]` | 对最终整体输出做 X 翻转，默认 0 |

`-dim` 表示单床位维度。默认 `center66_cosine10` 在输出时从每床取中心 66 层，最终 `Z = 66 + (bed_count - 1) * 56`；`sigmoid/direct` 控制路径仍使用 `Z = Nz * bed_count - sum(overlap)`。迭代体数据为 `Nz * bed_count`，融合只发生在保存前。完整输出顺序为：必要的每床 Z 翻转 -> 中心裁剪与基线 overlap 融合 -> 固定分析 FOV 分类/支持检测 -> 输出 FOV -> profile 后处理 -> 整体 Z/X 翻转 -> `cropEnd`。

`-bedOverlap` 的单位是 Z 体素。默认中心方案要求每个实际接缝满足 `overlap = 单床 Nz - 56`：160 层设备床使用 104，220 层 exact-matrix 床使用 164；两者都产生 56 层物理步进、10 层输出融合区。其他 overlap 只用于显式 `sigmoid/direct` 控制并需单独验证。

Mouse 发布默认使用 `center66_cosine10`，不附加 normalization/detZ 因子。W2 等效率校正会改变局部对比度和组织分布，在组织真实性完成资深阅图确认前仍只作为候选诊断路径。

预处理投影变体必须与矩阵入口配对（详见 `Recon_Interface_Migration_Field_CN.md`）：`-chunkCSRDir`（260601）对应 `PROJECTION_VARIANT=identity`，`-zbedflip` 保留包默认配置。

---

## 输出与日志

| 参数 | 说明 |
|---|---|
| `-fout <dir/name>` | 输出路径前缀 |
| `-o <dir>` / `-output <dir>` | 输出目录 |
| `-n <name>` / `-name <name>` | 输出文件名前缀 |
| `-vb <0\|1\|2>` | 日志级别；0=安静，1=INFO 默认，2=DEBUG |
| `-v` / `--version` | 打印版本字符串并退出（形如 `20260710+ad3c1d4`） |

默认输出 MHD/RAW 体数据与自动日志 `<outputName>_YYYYMMDD_HHMMSS_mmm.log`。重建开始时 INFO 日志会打印 `=== SPECT_recon <version> ===` 与完整 `Command line:`。

---

## 高级选项

| 参数 | 说明 |
|---|---|
| `-config <path>` | 全局配置文件 |
| `-algoConf <path>` | 算法配置文件 |
| `-dumpConf` | 打印合并配置并退出 |
| `-h` / `-help` / `--help` | 打印 help |

---

## 已删除参数的拒绝行为

以下入口已退场。带专用报错的参数在 parser 阶段给出明确迁移提示；其余按 `Unknown argument: <name>` 拒绝（退出码 1）。

| 已删参数 | 当前行为 |
|---|---|
| `-indexMat` / `-weightMat` / `-loadCSR` / `-mapCSR` / `-saveCSR` / `-compactMeta` / `-matrixLayout` | `Unknown argument`（矩阵唯一入口 `-chunkCSRDir`） |
| `-subsets` | 专用报错：compact 布局恒单 subset |
| `-alg` | 专用报错：改用 `-opti mlem\|bsrem` |
| `-opti admm` | `Unknown optimisation algorithm` |
| `-sparseBackend` / `-spBackend` | 专用报错：稀疏后端固定 CUDA |
| `-conv` | 专用报错：改用 image-domain `-psf` |
| `-psfProj` | 专用报错：公开 PSF 仅 Operator |
| `-butterBackend` / `-bwBackend` | 专用报错：Butterworth 固定 cuFFT |
| `-prior` / `-regPower` / `-regSigma` / `-regKernel` 等 anatomical 细调 | 专用报错：公开正则仅 TV/RDP |
| `-joe` | 专用报错：Joseph-like 平滑退场，改用 `-psf` / `-butter` / `-smooth` |
| `-Aprune` / `-Astream` / `-ApruneMask` | 专用报错：compact chunk CSRB 在生成阶段完成剪枝，读取恒为逐 chunk 流式/mmap |

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
