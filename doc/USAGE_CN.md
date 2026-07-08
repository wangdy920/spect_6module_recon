# SPECT_recon CLI 使用说明（中文）

> 更新日期：2026-07-07  
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

# 标准 CSRB
.\bin\SPECT_recon.exe -df proj.bin -loadCSR matrix.csrb -detN 40,40 -it 50

# compact chunk 260601 生产入口
.\bin\SPECT_recon.exe -df proj.bin -chunkCSRDir bin\matrix\260601 -it 50

# BSREM / ADMM
.\bin\SPECT_recon.exe -df proj.mat -loadCSR matrix.csrb -opti bsrem -it 50
.\bin\SPECT_recon.exe -df proj.mat -loadCSR matrix.csrb -opti admm -it 100
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

算法默认参数会从同级 `config/algo_<name>.conf` 加载，也可用 `-algoConf <path>` 指定。CLI 参数覆盖配置文件。

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
| `-indexMat <file>` | MAT 系统矩阵 index 文件 |
| `-weightMat <file>` | MAT 系统矩阵 weight 文件 |
| `-saveCSR <path>` | 从 MAT 构建后保存 CSRB |
| `-loadCSR <path>` | 加载单文件 CSRB |
| `-mapCSR [on\|off]` | 内存映射加载 CSRB |
| `-chunkCSRDir <dir>` | compact chunk 目录，包含 `metadata.bin + chunk_*.csrb` |
| `-matrixLayout <standard\|compact-symmetric>` | 矩阵布局；`compact`/`compact_symmetric` 是别名 |
| `-compactMeta <path>` | compact metadata sidecar；`-loadCSR + -compactMeta` 为 260525 legacy 入口 |
| `-compactDenomCache <mhd>` | 复用/保存 compact global denominator cache；PSF/FWHM/床位组合需分文件 |
| `-attMap <path>` | 衰减图，mu 单位为 `mm^-1` |

无 CTAC 的 `-chunkCSRDir + compact-symmetric` 会自动使用 `recon_out/compact_denom_cache/auto_<manifest-hash>.mhd`；manifest 包含矩阵、metadata、全局维度、床位 Z 起点、Joseph 和 PSF 口径，不匹配时不会复用。

---

## 几何

| 参数 | 说明 |
|---|---|
| `-dim <Nx,Ny,Nz>` | 重建体素维度 |
| `-vox <v>` / `-vox <Vx,Vy,Vz>` | 体素尺寸 mm；单值表示三轴相同 |
| `-spacing <v>` / `-spacing <Vx,Vy,Vz>` | `-vox` 别名 |
| `-ang <N>` / `-angles <N>` | 投影角度数 |
| `-detN <Nz>` / `-detN <Nz,Nx>` | 探测器 bin 数 |

---

## 算法

| 参数 | 说明 |
|---|---|
| `-opti <mlem\|bsrem\|admm>` | 优化算法 |
| `-it <N[:save]>` | 迭代次数与保存间隔 |
| `-subsets <N>` | 子集数；`1=MLEM`，`>1=OSEM` |
| `-th <N>` / `-threads <N>` / `-t <N>` | OpenMP 线程数 |
| `-cudaDevice <id>` | 选择 CUDA device，覆盖环境变量 `SPECT_CUDA_DEVICE` |

稀疏投影后端固定为 CUDA；旧 `-sparseBackend` / `-spBackend` 参数已废弃并会直接报错。

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
| `-bsremSoftBand <vox>` | soft-mask 过渡宽度，默认 2 |
| `-bsremSoftLambda <v>` | mask 外惩罚增强，默认 100 |

### ADMM

| 参数 | 说明 |
|---|---|
| `-admmRho <value>` | ADMM 惩罚参数 rho，默认 1.0 |
| `-admmTvInner <n>` | TV prox Chambolle-Pock 内循环，默认 20 |
| `-admmAdaptive <on\|off>` | 自适应 rho，默认 on |

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

公开正则化只保留 TV/RDP。`-regBeta` 和 `-regDelta` 可作为简单覆盖；旧 anatomical/NLM/Tikhonov/MRP/LogCosh 入口与对应细调参数会在 parser 阶段拒绝。

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
| `-conv <params>` | legacy convolution 参数 |
| `-joe <n>` | Joseph-like smoothing passes，默认 0 |

注意：旧冻结包 `dist/release_candidates/SPECT_recon_rc_20260706_c1ee948` 的 `PARAMETERS_CN.md` 和二进制仍使用 `-psf <sigma>` 口径；当前最新代码和文档已改为 `-psf [fwhm_mm[,N]]`，新发布包应重新生成 `PARAMETERS_CN.md`。

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

---

## 多床位与输出变换

| 参数 | 说明 |
|---|---|
| `-bedOverlap <b1,b2,b3,b4,b5>` | 固定 `Bed1..Bed5` overlap 表；`b1=Bed1-Bed2`，只传一个值时复制到所有固定槽 |
| `-cropEnd <top[,end]>` | 最终输出 Z 轴裁剪，在输出翻转之后执行 |
| `-zbedflip [0\|1]` | 对 `-dfList` 每个床位段单独 Z 翻转，默认 1 |
| `-zoutflip [0\|1]` | 对最终整体输出再做 Z 翻转，默认 0 |
| `-xoutflip [0\|1]` | 对最终整体输出做 X 翻转，默认 0 |

`-dim` 表示单床位维度；最终输出 `Z = Nz * bed_count - sum(overlap)`。保存前处理顺序为：每床 Z 翻转 -> 整体 Z 翻转 -> 整体 X 翻转 -> `cropEnd` 裁剪。

---

## 输出与日志

| 参数 | 说明 |
|---|---|
| `-fout <dir/name>` | 输出路径前缀 |
| `-o <dir>` / `-output <dir>` | 输出目录 |
| `-n <name>` / `-name <name>` | 输出文件名前缀 |
| `-vb <0\|1\|2>` / `-v <0\|1\|2>` | 日志级别；0=安静，1=INFO 默认，2=DEBUG |

默认输出 MHD/RAW 体数据与自动日志 `<outputName>_YYYYMMDD_HHMMSS_mmm.log`。

---

## 高级选项

| 参数 | 说明 |
|---|---|
| `-config <path>` | 全局配置文件 |
| `-algoConf <path>` | 算法配置文件 |
| `-dumpConf` | 打印合并配置并退出 |
| `-Aprune <eps>` | 系统矩阵权重剪枝阈值，默认 `1e-10` |
| `-Astream [on\|off]` | 从磁盘流式构建系统矩阵，默认 off |
| `-ApruneMask [on\|off]` | 剪枝 mask 外列，默认 on |
| `-h` / `-help` / `--help` | 打印 help |

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
