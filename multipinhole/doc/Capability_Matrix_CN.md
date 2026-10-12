# SPECT-recon Capability Matrix

> 基础签收记录：2026-08-14；CTAC、观测模型与回归更新：2026-10-09。以 `src/cli/ArgumentParser.cpp`、`src/pipeline/MainProgram.cpp`、`src/pipeline/VolumeOutput.cpp` 和 compact CUDA 路径为唯一事实来源。

## 证据入口

| 区域 | 代码 |
|---|---|
| CLI 校验 | `src/cli/ArgumentParser.cpp`（`registerHandlers` / `validateArgs`） |
| compact 几何 | `src/io/CompactSymmetryMetadata.cpp`、`ReconDataLoader.cpp::applyCompactMetadataGeometry` |
| 主流程 | `src/pipeline/MainProgram.cpp` |
| compact CUDA | `src/sparse/cuda/CompactSymmetricProjectorCUDA.cpp` |
| 回归 | `tests/`、本地 `recon_out/prototype_sieve_castor_fwhm1p3_final_20260811/current_production_parity_metrics.json`、`reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence.html` |

## 耐久决策

| 决策 | 结论 |
|---|---|
| 生产矩阵输入 | **`-chunkCSRDir`**（或 `bin/matrix` 唯一目录自动发现）；v4 `metadata.bin`（88B 标准或已校验的 92B legacy）+ `chunk_*.csrb`，SYM_90Z compact-symmetric |
| SC60 适配候选 | 同一 CSRB 入口支持 124B SMMD v3、ROT90 无 Z 镜像；`matrix_grid_output=true` 保留完整网格，最多 16 床联合全局体重建（2026-10-11，暂不支持 CTAC/AutoMu/`-maskFromAtt`），多床需显式整数层步进，可用 `-beds N|A:B` 从目录选床；原生双窗 CSRB 已有软件验证。2026-10-09 CSRQ ROT90 共用内核 CUDA 回归通过，非生产签收 |
| 唯一布局 | compact-symmetric；standard 布局与单文件 CSRB/MAT 矩阵入口已整体退场 |
| 算法 | **MLEM / OSEM / BSREM**；`-it A:B` 表示 A 个完整迭代、B 个子集；B=1 为全量更新，保存间隔独立。OSEM 使用 MLEM 数据项的子集更新；ADMM 不支持 |
| 公开正则 | 仅 **TV / RDP**；可选 registered CT 边权，无 CT 时 activity-only |
| 公开 PSF | image-domain **`operator` / `sieve` / `none`**；`-psf` 为 FWHM mm；`-psfKernel adjoint|castor` 选择真伴随 Gaussian 或严格 CASToR erf/Z-norm/forward-reuse 语义；发布默认 `sieve + castor + 1.3,3` |
| 稀疏 / 滤波 | 固定 **CUDA-first / cuFFT**；CLI smooth/Butterworth 为显式 `in_recon`；发布默认在保存副本上独立施加 CASToR Smooth 0.30 mm/3σ 与 2× zero-pad Butterworth 5 cpcm/order 8 |
| CTAC 模型 | **`att[angle, voxel]` 因子分解** line-integral；factor 生成仅 CUDA（16-angle 分批）；AutoMuMap（`-ctacAutoMu` 从早期迭代体生成水 mu-map） |
| PSC 多床位默认 | canonical stitch：`Bed1=min physical z`、连续 Bed1..Bed5 子集、`z_bed_flip=1`、中心 `step+10` 层（260601 为 66）、prototype robust gain-fit、`step` 层步进（260601 为 56）、10 层 cosine 融合；固定多床 ROI `36×66 mm`、输出 XY `100×100`。`-attMap` local-Z 与 `-regCT` storage-Z 分离，`z_out_flip` 仅作用于保存输出 |
| 校正权重 | 坏探测器 mask、normFactor 的效率 e=1/c、直接 detZ 效率；支持加性背景，准备时计算 b/e，诊断使用 eHx+b；不增加 CUDA 内核 |
| GPU | **单 GPU**（`-cudaDevice` / `SPECT_CUDA_DEVICE`）；branch/bed 合批为 packed-K 正向 + CSR_ALG2 转置；非合批和 mmap 实验开关已退役；不规划多 GPU |
| 输出 | 默认仅保存末次primary `<prefix>_it<N>`及独立`_butter` MHD/RAW；Gaussian输出默认关闭；各副本后置非负钳制，manifest记录实际输出清单；退出码 `0` 成功/help/dumpConf、`1` CLI/配置错误、`2` runtime |

## 能力表

输入/物理含义/多窗能力集中维护于 [矩阵读取与多能窗合同](Format_Reader_Matrix_CN.md)。双窗独立读取第二份 metadata 并核对归一坐标，权重要求有限且为正。2026-10-09 CSRB/CSRQ 的 ROT90/ROT90_Z CUDA 回归通过，含多子集 CTAC；CSRQ 仍为实验候选。两者均不补足旧 metadata 未记录的能窗、发射谱和绝对单位。SC60 实采机械步进与 LUT 的签收以实际采集记录为准，不能从合成软件测试推定。

| 路径 | 特性 | 实现 | 生产签收 | 备注 |
|---|---|---:|---:|---|
| compact 260601 | MLEM CUDA | 是 | **是（B1 gate）** | subset=1 全量更新 |
| compact 260601 | BSREM + TV/RDP | 是 | 路径/parity；beta 标定未完 | 可选 CT 边权 |
| compact 260601 | `operator`/`sieve` PSF | 是 | 是 | XY 各向同性 |
| compact 260601 | Butterworth | 是 | 是 | cuFFT；仅显式 `in_recon`，与 `sieve` 互斥 |
| compact 260601 | CTAC 单床 | 是 | 圆柱 true/auto gate | 按角度在图像侧乘因子，旧 fused/baked 核已删除 |
| compact 260601 | multi-bed CTAC | 是 | 数值 parity 通过 | 逐床因子表按 chunk 切片双缓冲上传 |
| compact 260601 | 多床 canonical stitch | 是 | **是（五组数据）** | 两组 EP、两组 mouse 与单床 Derenzo 的对象/数值/输出端点全部通过 |
| compact 260601 | 16 GB GPU、50 次 MLEM 性能 | 是 | 单床通过；四床有界 | 三次中位数：Derenzo 104.59 s，EPguan 189.67 s；64 GB 主存 + NVMe 矩阵口径 |

## CTAC 边界（简）

| 本次输入 | CSRB | CSRQ（实验） |
|---|---|---|
| 无 CT，未启用 CTAC | 普通 MLEM/OSEM/BSREM；N10 路径 | 同类入口，K4/K8 CUDA 回归通过，仍实验 |
| 已配准 CT→μ，或有效 AutoMu | N1 与 N>1 的 CTAC 路径，CUDA 回归通过 | N1 与 N>1 CUDA 数值回归通过；驻留/流式、双床双窗，仍未发布 |
| 无 CT，仅活动正则 | TV/RDP 无 CT 边权 | 同一正则输入边界，不等于已做衰减校正 |

- 上表区分实现与签收；2026-10-09 最终套件 335 项通过、4 项既有可选跳过；原生观测模型 32 组无背景及 32 组有背景通过。真实 SC60 CT 的物理与配准验收不由合成数值测试替代。
- 共享输入检查覆盖尺寸、间距、有限值；μ 非负而 CT HU 可负。AutoMu 只接收带兼容标记的未变换内部体，旧无标记显示体拒绝。
- CTAC 缓存同时跟踪头文件和 RAW/IMG 的路径/大小/时间，不做内容 SHA；SC60 CT 准备使用运行 manifest，入口和限制见 `USAGE_CN.md`。
- 生产模型仍为 `effective = value * att[fullAngle, voxel] / 65535`；实现把因子移到前向输入与反投收口，按角度处理，避免逐 nnz 重复加载因子。
- CSRQ、双窗联合重建与 packed backward 仍为实验能力；CSRB 生产默认不变。
- factor 生成仅保留 CUDA line-integral；16-angle batch，临时显存约 29 MiB。
- Validation-only：`scripts/validate_ctac_pinhole_reference.py` 等抽样对照；不进入生产路径。
- 调试 gate：`doc/CTAC_AutoMu_Validation_CN.md`。
- CT/正则对比：`scripts/run_ct_guided_regularizer_compare.ps1` 固定输出 no-CT MLEM、CTAC MLEM、CTAC BSREM-beta0、CTAC+RDP、CTAC+TV。

## 相关文档

- `../reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence.html` — 当前 `sieve + CASToR + FWHM 1.3 mm` 原型对齐与五组数据决策记录
- `USAGE_CN.md` — CLI
- `Format_Reader_Matrix_CN.md` — 矩阵格式说明
- `Recon_Interface_Migration_Field_CN.md` — 现场迁移
