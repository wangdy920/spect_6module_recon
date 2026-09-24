# SC60 Reference Comparison

## Contract

2026-09-20: compare the multipinhole candidate with the supplied SC60 `img/osem_121314derenzo_141keV` reference, at its earliest saved endpoint: **100 complete OSEM iterations, 6 subsets (600 updates)**. The reference is a control, not ground truth. User approved an independent ROI-derived CSRB directory; original matrix, projection, images and protected Release remain read-only.

Match the recorded reference settings: 90x90x240 at 0.4 mm; centered cylindrical support diameter 32.8 mm / height 12 mm; PSF off (`conv_mode=none`), no prior, no in-loop Gaussian/Butterworth, no normalization or bad detector mask. The reference's stored PSF FWHM 1.5 mm and Gaussian settings are inactive. Compare unfiltered reference `single_bed/Bed1/recon100` against candidate primary center crop `[z=105:135,y=4:86,x=4:86]`, 82x82x30. Do not compare its Butterworth copy with an unfiltered candidate as an algorithm result.

Remaining known differences: reference sequential subset order 0..5 and OMEGA epsilon=1e-5 / float32 state; candidate uses its existing spaced subset order and numerical safeguards. No algorithm fork is introduced to erase these differences. Metadata decimal-to-double versus decimal-to-float is checked for identical discrete ROI membership. Original reference report records source paths but no input hashes, so historical projection byte identity cannot be independently proven.

## Derivation Gate

`tools/prune_sc60_roi.py` retains original row order, all projection rows, global column IDs, and exact retained float32 weights. It removes only columns outside the reference support, preserves metadata bytes, checks rotational invariance, and checks every chunk's masked forward operation with a deterministic nonuniform image at relL2 <=1e-12. This is an algebraic/mapping gate, not a clinical or image-quality acceptance threshold. Every written row pointer, column and weight is reread and compared bitwise. A validated manifest and original chunk SHA256 values are retained in the derived directory. This matrix must not be reused for a larger support or nonzero-PSF experiment without a new support contract.

## Timing And Visuals

Record native steady-clock `[SUBSET_TIME]` and `[ITER_TIME]` wall durations, with iteration timing ending before output save. Also retain existing aggregate `iteration_core_s`, matrix/sensitivity preparation, output write and full command timing. One 100-iteration run describes within-run variability, not independent repeated benchmark variability. Reference timing is unavailable unless a source log is found; no speedup claim will be inferred from filenames.

Freeze rendering before candidate scoring: native index orientation verified against crop code and NIfTI affine; no fitted registration or flips; reference-derived common window [0, reference it100 positive-voxel p99.5], nearest-neighbor display; central orthogonal planes and axial mean; native-intensity signed differences. Report relL2, correlation, total-intensity ratio and separately labeled scale-fitted residual as descriptive diagnostics, not pass/fail quality criteria. Without verified rod ground-truth dimensions, do not claim absolute resolution.

Endpoint: validate derivation, complete 100 iterations, verify finite/nonnegative output, render and inspect paired figures, report parameters and measured times. Stop on an invariant violation; repair it before continuing. Status: completed.

Artifacts: `recon_out/sc60_reference_compare_20260920/`. Pre-timing candidate EXE SHA256 `0A7C28393CA8AD237DA5C2D5A3B5F36CAC511B15D044CFCD9F5A7637EFEB5134` is preserved there as `SPECT_recon.pre-timing.exe`; the only new production-code change for this comparison is timing log instrumentation.

## 对比结果

**在本次相近参数的 100 轮端点，multipinhole 与所给参考具有高度一致的空间分布和强度。** 这比单轮可执行性检查更进一步，支持 SC60 CSRB 适配在此配置下的实际重建一致性；不等同逐位复刻、所有配置兼容或绝对定量/分辨率验证。

| 指标 | 结果 | 含义 |
|---|---:|---|
| 全支持区体素相对 L2 | 0.120116% | 候选与参考的整体数值差异；未配准、未缩放 |
| Pearson 相关系数 | 0.999999257 | 空间强度分布高度一致，但不是单独的质量门 |
| 总强度变化 | -0.001665% | 没有明显全局尺度偏移 |
| 参考 / 候选总强度 | 16,480,262.383 / 16,479,988.016 | 原始重建单位，不能解释为已标定活度 |
| 参考 / 候选峰值 | 12,574.994 / 12,543.242 | 峰值变化约 -0.253% |
| 最大体素绝对差 | 35.1356 | 原始强度单位 |
| 拟合强度系数 | 1.000000336 | 仅诊断；拟合后的 L2 几乎不变，主要差异不是全局尺度 |
| 质心差 XYZ | -0.000063 / +0.000459 / +0.000173 mm | 描述性分布统计，不代表亚体素定位精度认证 |

同色标切面、轴向平均和横向剖面中，三组杆状结构的位置、峰谷和轴向分布基本重合；未见明显翻转、错位、缺失组或额外条带。中心冠状面 y=+0.2 mm 恰好穿过组间空隙，因此图中接近空白，并非重建丢层。横向剖面取参考中心轴面行标准差最大的 y=-2.2 mm，规则只依赖参考，不以候选结果择优。

**不能只看全局小误差而忽略弱信号端层。** z 索引 0、1、28、29 的逐层相对 L2 分别为 51.88%、8.60%、12.79%、31.76%；这四层仅占参考总强度的 0.02457%，贡献全体平方误差的 3.94%。主体 z=2..27 的相对 L2 为 0.11773%。所以主体稳定，但弱信号尾部不属于严格一致区域。子集顺序、OMEGA epsilon 与候选的数值保护/精度差异可能影响这些小量；本次未做隔离 A/B，不将其确认为单一原因。

这是“相近参数比较”，不是以 1e-5 为门限的同实现 parity 测试：参考顺序为 0,1,2,3,4,5，候选为 0,3,1,4,2,5。两者主要物理/重建参数一致，但更新细节并非完全相同。本次没有证据证明哪一方影像质量更优，也未提供真实杆径/活度，故不报告绝对空间分辨率或定量准确度。

## 迭代耗时

RTX 4070 Ti SUPER 16 GB；CUDA device-resident 图像更新；派生矩阵 18 块全部进入主机锁页缓存。100 个完整迭代、每轮 6 个子更新。计时不是纯 CUDA kernel 时间，包含该范围内上传、CPU 调度等；每轮时间截止于保存之前。

| 项目 | 耗时 |
|---|---:|
| 完整迭代平均 / 中位数 | **1.5041 / 1.5399 s** |
| 完整迭代 P95 | 1.5436 s |
| 完整迭代最短 / 最长 | 1.2666 / 1.5732 s |
| 首轮 | 1.5732 s |
| 第 2..100 轮平均 | 1.5034 s |
| 单次子更新平均 / 中位数 | 0.25055 / 0.25583 s |
| 100 轮 wall 合计 | **150.4107 s** |
| 既有 iteration_core_s | 150.298 s |
| 矩阵与灵敏度准备（cache miss） | 36.529 s |
| 程序内部 full_command | 187.284 s |
| 外部进程调用计时（含启动/退出） | **190.746 s** |
| 一次性 ROI 派生及校验 | 684.100 s |

输出保存包含在 full_command 和外部进程计时中，不在上述逐轮 wall 中；当前 vb=1 日志未单独打印纯写盘时间，不能用两个不同起点总时长之差冒充纯 I/O。迭代时间存在两个较低时段，作为同一次运行的波动保留，没有据此推断计算量变化或 GPU 原因。

原矩阵 62.497 GiB，派生矩阵 25.109 GiB，保留 40.1694% nnz。18 块的掩膜前向相对误差最大约 1.2e-16，保留元素逐字节一致。100 轮期间应用层矩阵 payload **重读 0 B**，矩阵 H2D 为 2,695,675,209,600 B，约 25.11 GiB/轮。前者不是“GPU 永久驻留全部矩阵”，后者说明仍在从锁页主机缓存上传。

本次没有原参考程序的运行日志/硬件计时，不能给出两种程序的加速比。也不能与上一轮 N10 / PSF 1.5 mm / 大支持区短测直接相除：子集数、PSF、支持区、缓存覆盖和 device/host 更新路径均不同。

## 图像与数据

![同切面、同色标对照](../recon_out/sc60_reference_compare_20260920/paired_slices.png)

![轴向平均与强度剖面](../recon_out/sc60_reference_compare_20260920/profiles.png)

![放大残差与逐层误差](../recon_out/sc60_reference_compare_20260920/residuals.png)

![逐轮与子更新耗时](../recon_out/sc60_reference_compare_20260920/timing.png)

原始计量：`iterations.csv`（100 行）、`subsets.csv`（600 行）、`comparison.json`、`run.log`、`process-timing.json`。完整候选体积：`osem100_n6_nopsf/sc60_it100.mhd|raw`；对齐裁剪的可视化体积：`candidate_it100_aligned_crop.nii.gz`，保留 float64，并采用参考 affine。精确输入及 EXE SHA256：`inputs.json`；派生矩阵来源及每个原 chunk SHA256：`matrix_roi32p8_h12/derivation.json`。

验证：新计时构建通过，308 项测试通过、4 项既有跳过；运行退出码 0；全图有限/非负/非零；运行 mask 与派生 mask 完全相同；裁剪映射与 NIfTI affine 独立核对；100 轮和 600 个子更新的计时记录完整且加总一致；四张图人工检查，标题重叠已修正。原参考图像、BIN 与受保护 Release 未修改。源码差异留在工作树，未提交或生产替换。

## 复现

派生目录不可覆盖，重做时使用新目录：

```powershell
python tools/prune_sc60_roi.py test_data/SC60_luzhou/matrix/matrix_3module_18pin_YSO_90x90x240_141keV recon_out/sc60_reference_compare_20260920/matrix_roi32p8_h12
```

本次完整运行命令已保存在 `run.log` 的 Command line；关键覆盖为 `-it 100:6 -save-every 100 -psfMode none -cyl 32.8,12 -th 8 -vb 1`，基于 `sc60_osem1000.conf`，显式使用派生矩阵和 Derenzo BIN。重新分析/出图：`python tools/render_sc60_reference_comparison.py`。

当前结论：执行完成；此 100 轮/无 PSF/规定支持区下的相对结果稳定；生产采用未评价。本轮不扩展至 PSF 1.5 mm 长跑、1000 轮、CSRQ 或其他系统族。

## 追加：N10 / PSF 1.5 mm 主机全缓存计时

2026-09-20，按用户后续授权完成。此节扩展上述无 PSF 范围，不改变原比较结果。使用同一 32.8 mm 直径 × 12 mm 高的派生矩阵、Derenzo BIN 和 RTX 4070 Ti SUPER；100 个完整迭代 × 10 个子集 = 1000 次子更新，每 50 轮保存。PSF 为 sieve / CASToR，FWHM 1.5 mm，启用 Z 卷积。无其他滤波、先验或归一化。

**支持区合同：**这里测试的是 ROI 裁剪矩阵限定的投影算子加 PSF，不主张与原始全矩阵加 PSF 等价。实现对前向 PSF 后的局部图像和反向 PSF 前的局部分子施加该支持区。该配置触发 `input=host, update=host, roi_prunes_matrix=yes`，不同于前一组无 PSF 的 device-resident 图像更新路径；投影和 CASToR 卷积仍调用 CUDA。sieve 保存的是卷积后的输出，并非未卷积的迭代内部状态。

测量目标为完整迭代 wall 耗时、缓存覆盖和有效输出，不作优化或生产采用判定。运行不修改源码、原始矩阵或参考图像。EXE 仍为 `build/sc60-adapter-20260920/bin/Release/SPECT_recon.exe`，SHA256 `8C9B004E6286FD3EE814A8D8A60A3A78F73447709B5DFA187BAE87555D15CF2D`，版本标记 `20260920+635dec6-dirty`。输入与派生来源沿用本报告 `inputs.json` 和 `matrix_roi32p8_h12/derivation.json`；有效配置保存在各输出目录 `sc60_manifest.json`。

### 实测耗时

第一次使用 PowerShell `*>` 收集输出。出现 epoch wall 显著大于子更新计时总和的情况，因此保留原结果，并用 `cmd /d /c "... > log 2>&1"` 原生文件重定向复测。两次串行运行，无人为删除慢轮；两次之间灵敏度缓存从 miss 变为 hit，因此准备时间不是同条件缓存 A/B。

| 项目 | 第一次 | 原生重定向复测 |
|---|---:|---:|
| 100 轮完整迭代 wall 合计 | **328.939 s** | **377.784 s** |
| 每轮平均 / 中位数 | 3.289 / 2.717 s | 3.778 / 3.746 s |
| 每轮 P95 | 3.457 s | 4.442 s |
| 每轮最短 / 最长 | 1.960 / 25.980 s | 1.999 / 19.928 s |
| 单次子更新平均 | 0.2640 s | 0.3335 s |
| 1000 次子更新计时合计 | 264.041 s | 333.506 s |
| epoch 内、subset timer 外的 wall 合计 | 64.898 s | 44.278 s |
| 原有 `iteration_core_s` | 187.995 s | 199.638 s |
| 矩阵及灵敏度准备 | 39.775 s（miss） | 85.200 s（hit） |
| 程序内部 full_command | 369.196 s | 463.451 s |
| 外部进程总耗时 | **372.356 s** | **466.676 s** |

**上述两次历史运行观察值是 100 轮约 5.5–6.3 分钟，整个进程约 6.2–7.8 分钟，不是 150 秒。** 后续用户说明当时存在其他 CPU 占用任务；下节低 CPU 干扰复测更新了此配置的耗时参考。历史结果原样保留，不作为稳定吞吐保证或置信区间。准备阶段 hit 仍慢于 miss，也说明不能把当时全部波动解释成固定算法成本。

计时口径：逐轮 wall 包含该轮 CPU 调度、上传、PSF、更新及计时范围内日志调用，不包含轮后保存；`iteration_core_s` 只累计源码中特定 forward/ratio/backward、update 等计时段，在此 host + PSF 路径并未覆盖全部 PSF、复制、掩膜、检查和日志，**不能当作完整 100 轮耗时或纯 GPU 时间**。定位依据见 `multipinhole/src/recon/CompactSymmetricReconLoop.cpp` 中 `iterationWallStart`、`subsetWallStart`、PSF 调用和 `[PERF]` 累计式。

第一次第 29/53/94/96/97 轮在 subset timer 之外分别多出约 9.23/11.27/2.52/16.92/23.19 秒；复测仍存在相同类别的停顿（单轮最大额外 16.20 秒）。源码表明这些间隙可包含计时日志调用和线程被调度出去的等待，现有证据未隔离具体原因，不能认定为 GPU kernel、磁盘重读或仅 PowerShell 所致。本次不继续扩大为性能诊断/优化任务。

两次均确认：18 个块全部进入锁页主机缓存（25711 MiB，约 25.1 GiB）；迭代期间应用层矩阵 payload 重读 **0 B / 0 次**；矩阵 H2D 各为 **2,695,675,209,600 B**，仍约 25.1 GiB/完整迭代。全主机缓存不等于全矩阵 GPU 驻留。以上时间不含已经完成的一次性 ROI 派生。

与前述 N6 / PSF off 的 150.411 s 相比，两次完整迭代 wall 分别为 2.19 / 2.51 倍。但 N、PSF、图像更新路径以及运行波动都不同，不能把比值解释成 PSF 单独开销或适配造成的性能回退。

### 输出检查与观察

两次退出码均为 0；均有完整的 100 条 epoch / 1000 条 subset 计时。第 50、100 轮输出均为 90×90×240、0.4 mm，有限、非负、非零；运行 ROI mask 与派生 mask 完全相同，第 100 轮支持区外全为 0。两次第 100 轮相对 L2 差异为 `9.3849e-8`，最大绝对差 `0.0009765625`，总强度比 `1.00000000399`：支持数值重复性，但不是逐位一致。

同色标图上，sieve 输出的点状结构更宽、峰值更低，细小结构的分离度弱于 N6 / 无 PSF 输出；三组主要位置仍对应。该观察与当前保存卷积后图像的行为一致，不据此宣称 PSF 提高了分辨率，也不把两个参数同时变化的比较当作 PSF 因果实验。首轮全图峰值 3702.774、总强度 16,467,078.684，均为未标定的原始强度。

![N6 无 PSF 与 N10 PSF 输出，同行同色标](../recon_out/sc60_reference_compare_20260920/osem100_n10_psf1p5/images.png)

![N10 PSF 原生重定向复测耗时](../recon_out/sc60_reference_compare_20260920/osem100_n10_psf1p5_repeat/timing.png)

### 复现与证据

两次完整命令分别保存在 `n10-psf1p5.log`、`n10-psf1p5-repeat.log`，关键覆盖 `-it 100:10 -save-every 50 -psfMode sieve -cyl 32.8,12 -th 8 -vb 1`；使用上述相同配置、EXE、BIN 和派生矩阵，设置 `SPECT_FAIL_ON_ZERO_CHECKPOINT=1`。主机缓存使用自动策略，未强制环境变量预算。

输出目录为 `recon_out/sc60_reference_compare_20260920/osem100_n10_psf1p5` 和 `osem100_n10_psf1p5_repeat`，各有原图、manifest、`summary.json`、`iterations.csv`、`subsets.csv`、`timing.png` 和 `images.png`。外部计时保存在上一级 `n10-psf1p5-process.json` / `n10-psf1p5-repeat-process.json`。可重跑分析：

```powershell
python recon_out/sc60_reference_compare_20260920/analyze_n10_psf.py
python recon_out/sc60_reference_compare_20260920/analyze_n10_psf.py --repeat
```

初版分析脚本沿用先前“epoch 与 subset 总和差 <1 s”的假设并触发断言；该上限不是重建正确性条件，已改为校验计时包含关系并完整记录未计入 subset 的间隙，没有删掉异常轮或修改日志。两次最终分析检查均通过；图像与计时图已人工检查。执行与输出检查完成，性能波动原因未隔离，生产采用未评价。

## 低 CPU 干扰复测（当前耗时参考）

用户说明此前另有 CPU 占用任务，并要求重新测试。2026-09-20 13:30 开始，复用上述相同 EXE（运行前 SHA256 一致）、输入、ROI 派生矩阵及全部重建参数；100:10、PSF sieve/CASToR 1.5 mm、8 线程、save50。只新增直接文件日志与每 5 秒的 CPU/GPU 占用采样，不改重建源码，不停止其他用户任务。开始前系统 CPU 3.5%、可用 RAM 46.8 GiB、无其他 SPECT_recon 进程。目标为复核完整 wall 耗时和同设置输出，结束于 100 轮完成及输出检查。

| 项目 | 本次 |
|---|---:|
| **100 轮完整迭代 wall** | **179.969 s（约 3.00 分钟）** |
| 每轮平均 / 中位数 / P95 | 1.7997 / 1.7888 / 1.8588 s |
| 每轮最短 / 最长 | 1.7562 / 2.1079 s |
| 第 2..100 轮平均 / 最长 | 1.7966 / 1.9150 s |
| 单次子更新平均 | 0.17872 s |
| 矩阵及灵敏度准备（hit） | 21.841 s |
| 原有 iteration_core_s（非完整迭代口径） | 144.086 s |
| 程序内部 full_command | 202.442 s |
| **外部进程总耗时** | **204.487 s（约 3.41 分钟）** |

仍为 18 块全部锁页主机缓存，约 25.1 GiB；迭代 payload 重读 0 B / 0 次；矩阵 H2D 2,695,675,209,600 B。输出仍为 host 图像更新、CUDA 投影/卷积路径，没有更换执行算法。epoch 内 subset timer 外的时间总计仅 1.252 s，最大单轮间隙 0.0717 s，之前数秒至二十多秒的停顿不再出现。

迭代期间 35 个占用样本：系统 CPU 平均 38.94%，重建进程 CPU 平均 33.68%（均以全部 20 个逻辑处理器总容量为 100%）；两者差值估计其他 CPU 占用平均 5.27%、P95 9.47%、最大 10.88%。GPU 瞬时利用率采样平均 70.09%，范围 45–100%；可用 RAM 最低 20.67 GiB。CPU 是约 5 秒区间统计，GPU 是离散瞬时采样，首尾样本可能跨阶段，不解释为精确 kernel 忙碌率或完全零干扰运行。

**当前这组配置应参考约 180 秒/100 轮、约 204 秒/完整进程，而非先前的 329–378 秒/100 轮。** 同程序、同参数下明显变快且停顿消失，结合用户提供的后台任务信息，支持先前耗时受到运行环境竞争影响。但此前没有同等占用采样，也未重新施加该任务作受控 A/B，不能量化 CPU 竞争的独立因果贡献。这里只新增一次较低干扰运行，不声称跨运行吞吐已经严格稳定。与历史 N6/无 PSF 的 150.411 s 相比约多 19.65%，仍不等于孤立 PSF 开销。

退出码 0，100 条 epoch / 1000 条 subset 计时齐全，第 50、100 轮图像均有限、非负、非零，ROI mask 完全一致。与第一次同参数 PSF 结果比较：相对 L2 `9.73595e-8`、最大绝对差 `0.000732421875`、总强度比 `1.00000000538`。图像观感未发生变化；CPU 干扰主要体现为本次观察到的耗时差异，没有伴随明显数值结果变化。图像与计时图已检查。

![低 CPU 干扰复测逐轮与子更新耗时](../recon_out/sc60_reference_compare_20260920/osem100_n10_psf1p5_lowcpu/timing.png)

证据目录：`recon_out/sc60_reference_compare_20260920/osem100_n10_psf1p5_lowcpu/`，包含 `summary.json`、`occupancy.csv`、逐轮/子集 CSV、manifest、50/100 轮图像及可视化。上一级保存 `n10-psf1p5-lowcpu.log` 与 `n10-psf1p5-lowcpu-process.json`。运行脚本 `run_lowcpu.py` 拒绝覆盖已存在的输出目录；日志直接由子进程写文件。分析命令：

```powershell
python recon_out/sc60_reference_compare_20260920/analyze_n10_psf.py --lowcpu
```
