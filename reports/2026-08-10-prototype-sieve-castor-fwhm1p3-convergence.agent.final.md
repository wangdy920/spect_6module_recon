# SPECT 重构与受保护原型 `sieve_castor_fwhm1p3` 收敛审查

**执行区间：** 2026-08-10 至 2026-08-11

**最终状态：** 通过冻结五数据集合同；当前重构路线可判定为与受保护原型输出一致。性能优化后，单床 50 次中位数低于 2 分钟，四床中位数为 3.16 分钟并作为当前硬件边界保留。
**数值权威：** `6module_bak_src/spect_recon_PSC_6module_24pin_YSO_bedmerge.cu` 中的 `sieve + CASToR + FWHM 1.3 mm + 3σ` 路线。

## 1. 执行结论与判定合同

本轮不是以旧发布二进制或“看起来相似”为裁决，而是以用户指定的受保护原型为唯一数值基线。重构后的单 GPU compact-v4 路径在修复共享根因后，完成正式 Release 重编译，并对五个冻结数据集逐一执行完整数据、50 次 MLEM。随后又恢复原型在 canonical 输出后的独立 Smooth/Butterworth 派生：五组同输入滤波门、可选钳制门和一迭代真实入口 smoke 全部通过。

| 合同项 | 冻结口径 | 本轮结果 |
|---|---|---|
| 输入 | 指定五数据集、同一 compact 90z 系统矩阵、同一配置与 detector 68 屏蔽 | 42 个文件以路径、大小和 SHA-256 预检，合计 26,342,787,420 字节 |
| 重建 | 单 GPU、完整数据、MLEM 50 次 | 五次正式进程退出码均为 0 |
| PSF | sieve、CASToR erf 体素积分核、FWHM 1.3 mm、3σ | manifest 与命令逐项一致 |
| 多床 | `Bed1=max physical z`、`z_bed_flip=1`、center66/cosine10、原型增益链 | 四/五床增益、方向与对象门通过 |
| 数值门 | 支撑区 NCC ≥ 0.99；缩放匹配 relL2 ≤ 0.10 | 五组全部通过；最弱项为 Derenzo 0.995661 / 0.068587 |
| 对象门 | Derenzo、EP、mouse 各自冻结指标 | 五组全部通过 |
| 输出 | 始终保留 `recon_it<N>`；Smooth 0.30 mm/3σ 与 Butterworth 5 cpcm/order 8 从同一主输出独立派生 | 五组同输入线性/钳制门通过；生产 clamp-on 一迭代入口写出六个体文件且 manifest 一致 |

结论的量词仅覆盖这五个冻结输入和上述合同。它证明当前实现复现指定原型输出；不把原型提升为独立物理真值，也不外推到未验证的矩阵、采集或参数组合。

## 2. 最近重构审查与根因修复

审查先恢复完整构建和可复现入口，再以一迭代结构证书定位数值差异，最后才运行昂贵的 50 次终点。所有最终改动都位于共享数据流，不含按 Derenzo、EP 或 mouse 分支的对象调参。

| 审查项 | 发现 | 最终处置与证据 |
|---|---|---|
| 构建/接口收敛 | 重构后存在 include、命名空间和测试目标残留；输出阶段曾被误删，导致主线不再生成原型 Smooth/Butterworth 派生 | 恢复 `SPECT_recon`/`spect_tests` Release 构建；生产只保留 compact-v4 与一个保存数据流，阶段为 `none|in_recon|output` |
| 输出滤波缓存 | 输出 Butterworth 在相同参数下切换体积尺寸时，plan 重建后 response 指针已释放但参数缓存仍命中 | response/mask 缓存把空 device 指针纳入重建条件；尺寸切换回归和五组连续运行通过 |
| 失败纪律 | PSF CUDA 不可用时存在不可接受的降级风险；复现目录可混入旧产物 | PSF 路径改为 fail-closed；正式运行使用全新输出根并以 manifest/证书闭合 |
| CUDA 投影缓存 | 原实现把 host 指针和数量误当成内容身份，复用/修改缓冲区时可能沿用陈旧投影 | 每个 numerator pass 对每床刷新一次，同时保持分支/块内复用；突变缓存回归测试通过 |
| 多床反向存储 | `z_bed_flip=1` 后，增益拟合读取了错误物理重叠端，且 cosine 融合极性仍按未反向存储解释 | 交换增益读窗并反转 blend 极性；独立修复虽能救 CH mouse 但使 EP 对象门失败，组合修复后五组通过 |
| detector 68 | 通用 ENG identity 映射把 global 68 映射到 `angle 120..179, z 36..47, x 0..11`，而原型明确使用固定块 | 仅对 `link-4/FPGA-2/detector-4` 使用原型固定 `angle 0..59, z 24..35, x 24..35`；其他 detector 仍走通用映射；缓存 schema 升级防止误用旧分母 |
| 首次 PSF 状态 | 当前实现以全体素 1.0 启动，直到首轮末才施加 ROI；CASToR PSF 会把 ROI 外初值扩散入内 | 使用已有 mask 在第一次 PSF forward 之前就约束初始图像，和原型 `k_fill` 后立即施加 local cylinder ROI 的顺序一致 |
| 数值策略候选 | 低预测比值返回中性比、0.01 更新下限、直接 `A^T w` 均曾被怀疑 | 均按预注册门在无改善时拒绝并回退；没有保留第二算法或永久开关 |

### 2.1 一迭代定位证书

结构门固定为尺度匹配 relL2 ≤ `1e-7`；比较的是同一输入、矩阵、屏蔽、ROI 和中心 66 层，方向为 `rot0`。该门用于区分迭代阶段结构，而不是代替最终 50 次对象门。

| 阶段 | 支撑区 NCC | 尺度匹配 relL2 | 判定 |
|---|---:|---:|---|
| 原 detector 68 映射，masked no-PSF | 0.99917756 | 1.2651e-2 | 失败；直接 `A^T w` 候选无改善 |
| 修正 detector 68，no-PSF | 0.99999999999997 | 7.2998e-8 | 通过；屏蔽映射根因闭合 |
| 仅修正 detector 68，sieve | 0.99999545 | 1.1210e-3 | 失败；差异缩小为 PSF 首态问题 |
| 再修正初始 ROI，sieve | 0.99999999999996 | 8.9537e-8 | 通过；首次迭代结构门闭合 |

这条证据链否定了此前“剩余问题一定在加权灵敏度”的临时判断：直接 `A^T w` 实验没有改变失败指标；真正的两个根因分别是 detector 68 坐标合同和 PSF 前初始 ROI 状态。

## 3. 验证方法

### 3.1 输入与可执行文件冻结

`input_preflight.json` 记录 42 个输入文件，包括正式可执行文件、配置、算法配置、坏探测器列表、受保护原型源码、6 个矩阵块及 metadata、18 个床位投影、5 组原型 Analyze 输出及 sidecar。五组 50 次基础重建证据所用二进制 SHA-256 为：

```text
98F4771023E2C9A40C7045CE8F2EB95DDA6E7BD1617EBDDC7F899E857AE03019
```

受保护原型源码 SHA-256 为：

```text
6B332DAD80310B2F0E4F8723B45B2AEB6C1B5DC24A153A6F720A467D45573B0E
```

最终从干净提交 `5baeecc45050a5434fd0c64d6e6cc9c59c35cf3b` 重编译并重置部署包。最新 Release 版本为 `20260811+5baeecc`，构建树与部署包可执行文件 SHA-256 均为：

```text
B927BEDF9E8010ED59718EC1086C696E881DE52936DA169BEF6E4639F75524E0
```

权威配置为 `config/spect_config.ini`，部署副本为 `staging/SPECT_recon_rc_dev/bin/config/spect_config.ini`；两者 SHA-256 均为 `35E950A84C93BA297BF878E480ACC8F48E2E5AC9D3D0717598E3ACDF28ACFB4F`。部署 `-dumpConf` 确认 `smooth_apply_stage=output`、`butter_apply_stage=output`、Smooth `0.30 mm/3σ`、Butterworth `5 cpcm/order 8`、`output_nonnegative_clamp=true`。随后只有本报告元数据更新，不改变该 Release 包。

### 3.2 比较规则

原型 Analyze 输出与当前 MHD 输出按冻结方向比较：Derenzo 为 `rot0`，所有多床数据为 `zflip+rot0`。数值比较报告原始全体积 NCC、原型 1%–P99.9 支撑域 NCC，以及最小二乘尺度匹配后的 relL2。尺度匹配只用于既定数值门，不修改候选输出或 QA 图。

对象门分别为：

- Derenzo：中心保留率 ≥ 95%，rod separability 相对变化绝对值 ≤ 5%，覆盖下降不超过 2 个百分点，最长 gap 增量不超过 1 层。
- EP：轴向 CV 相对变化绝对值 ≤ 5%，覆盖下降不超过 2 个百分点，seam 中位变化 ≥ -0.02。
- Mouse：支撑区 NCC ≥ 0.99，逐切片 NCC 中位数 ≥ 0.95。
- 主输出：体素有限、无负值、间距误差 ≤ `1e-6 mm`。输出滤波先以 clamp-off 对齐原型线性体，再以 clamp-on 对齐 `max(prototype,0)`；manifest 必须声明主、Smooth、Butterworth 共六个体文件。

### 3.3 直接图像质量检查

下图的切片索引只由原型最大平面和确定；每一对原型/当前图使用同一原型正体素 P99.5 窗宽，不插值、不对候选重标度、不逐图归一化。该图用于发现方向、裁剪、接缝或局部结构异常，数值门仍由 JSON 证书裁决。

![五数据集原型与当前输出直接正交切片](assets/2026-08-11-prototype-sieve-castor-fwhm1p3-direct-slices.png)

## 4. 五数据集 50 次正式结果

### 4.1 主数值门

| 数据集 | 旧主线支撑 NCC | 当前支撑 NCC | 旧主线 relL2 | 当前 relL2 | 方向 | 对象/数值门 |
|---|---:|---:|---:|---:|---|---|
| `derenzo_182222` | 0.99416676 | **0.99566121** | 0.07935099 | **0.06858701** | `rot0` | 通过 / 通过 |
| `epguan` | 0.98811926 | **0.9999999999990** | 0.12512118 | **1.2421e-6** | `zflip+rot0` | 通过 / 通过 |
| `epguan_19p2mm` | 0.98978362 | **0.9999999999999** | 0.11587385 | **3.2937e-7** | `zflip+rot0` | 通过 / 通过 |
| `mouse_ch6o6p2` | 0.99810844 | **0.9999999870383** | 0.05630161 | **1.6905e-4** | `zflip+rot0` | 通过 / 通过 |
| `mouse` | 0.99840547 | **0.9999999999764** | 0.05201265 | **5.1246e-5** | `zflip+rot0` | 通过 / 通过 |

Derenzo 是最终最弱项，因此也是残余数值风险的上界：支撑 NCC 仍高于门 0.00566，relL2 低于门 0.03141。其余四组已接近数值重合。这里不声称 bitwise 一致；正式结论是全部通过预先冻结的等价门。

### 4.2 对象端点与多床增益

- Derenzo 13/13 中心保留；separability 相对变化 `+0.0007349`；轴向 CV 相对变化 `-0.0100253`；覆盖变化 0 个百分点；最长 gap 变化 0 层。
- EPguan 轴向 CV 相对变化 `+2.65e-6`，覆盖变化 0，seam 中位变化 `-3.86e-8`；19.2 mm EP 亦通过相同门。
- CH mouse 支撑 NCC `0.9999999870`、逐切片 NCC 中位数 `0.9999999997`；普通 mouse 分别为 `0.99999999998` 和 `0.9999999999998`。

| 数据集 | 正式拟合床增益（Bed1 锚定） |
|---|---|
| EPguan | `[1, 0.99541156, 1.00488482, 1.00712702]` |
| EPguan 19.2 mm | `[1, 1.00559497, 1.01155317, 1.01504480]` |
| CH mouse | `[1, 0.98772624, 1.05752771, 1.06324157, 1.83994881]` |
| Mouse | `[1, 0.93013597, 0.99892396, 1.06802328, 1.44608513]` |

### 4.3 优化前正式运行耗时

| 数据集 | 耗时（秒） | 退出码 |
|---|---:|---:|
| Derenzo | 594.79 | 0 |
| EPguan | 696.66 | 0 |
| EPguan 19.2 mm | 707.68 | 0 |
| CH mouse | 805.77 | 0 |
| Mouse | 741.88 | 0 |

这些是优化前的单次历史记录，只用于量级背景，不作为后续性能采纳统计量。四/五床的同几何分母缓存被复用，但每组投影、50 次 numerator/update 和最终输出均独立执行。

### 4.4 输出级 Smooth / Butterworth 与非负钳制门

输出滤波合同固定为：主输出不滤波；Smooth 和 Butterworth 都从同一份 canonical stitch + mask 体独立派生，不能串联。线性验证关闭 `output_nonnegative_clamp`，生产验证开启并逐体比较 `max(prototype, 0)`。五组冻结同输入采用 `2e-7` relative-L2 门；主输入与 Butterworth 为数值零差异，Smooth 的最大 relL2 为 `1.1563017424662106e-7`，钳制后保持同一误差并通过门。

| 数据集 | 同输入主体 relL2 | Smooth relL2 | Butterworth relL2 | clamp-on 对应门 |
|---|---:|---:|---:|---|
| `derenzo_182222` | 0 | 1.1392794e-7 | 0 | 通过 |
| `epguan` | 0 | 1.1563017e-7 | 0 | 通过 |
| `epguan_19p2mm` | 0 | 1.0843560e-7 | 0 | 通过 |
| `mouse_ch6o6p2` | 0 | 1.1321093e-7 | 0 | 通过 |
| `mouse` | 0 | 1.1213429e-7 | 0 | 通过 |

端到端 smoke 使用正式 Derenzo 投影、已认证 denominator cache、`sieve + CASToR 1.3 mm/3σ` 和一次 MLEM。进程退出码为 0；`recon_it1`、`recon_it1_smooth`、`recon_it1_butter` 均为 `100×100×72`、有限且负体素数为 0，manifest 记录独立派生、生产钳制及六文件清单。该 smoke 只验证真实保存入口，不替代上面的五组同输入数值门，也没有重跑五组 50 次 MLEM。

### 4.5 50 次迭代性能收敛

性能签收按共享设备口径执行：同一冻结 EP 四床输入、同一 SHA-256 矩阵副本、同一 denominator cache，基线与候选按 `A1,B1,B2,A2,A3,B3` 交错运行，各取三次中位数。边界快照中可见其他任务造成的 GPU utilization 为 0%–13%，因此不采用单次最佳值。基线是干净提交 `17bd22c` 且设置 `SPECT_COMPACT_BRANCH_BATCH=0`；候选使用默认合批路径，无性能环境变量覆盖。

| EP 四床短 A/B | 三次耗时（秒） | 中位数（秒） | 相对基线中位数 |
|---|---|---:|---:|
| 逐分支基线 | 80.0837、52.4101、52.0406 | 52.4101 | — |
| branch/bed 合批 | 58.0584、35.9559、35.9731 | **35.9731** | **-31.36%** |

配对 A/B 的主输出最大 relL2 为 `9.7147e-8`，Smooth 最大为 `1.0541e-7`，Butterworth 最大为 `1.3564e-7`；均低于主输出 `1e-5` 和冻结派生体 `2e-7` 门，且全部有限、非负。用户将旧 `1e-7` 性能分支门放宽为 `1e-5` 后，本候选仍多两个数量级余量。

最终 50 次运行使用默认配置、NVMe 上的冻结矩阵副本和命中 denominator cache，各重复三次并取中位数：

| 数据集 | 床数 | 三次耗时（秒） | 中位数（秒） | 120 秒目标 | 与优化前单次记录的背景差值 |
|---|---:|---|---:|---|---:|
| Derenzo | 1 | 99.9661、104.5868、110.4908 | **104.5868** | 三次均通过 | -82.42% |
| EPguan | 4 | 189.6690、191.7576、188.8894 | **189.6690** | 未通过 | -72.78% |

优化后 Derenzo 主输出相对优化前正式体 relL2 为 `2.1353e-7`；EPguan 为 `6.8556e-7`，都低于 `1e-5`。三次重复间，Derenzo 主/Smooth/Butterworth 最大 relL2 分别为 `5.5703e-8`、`5.2646e-8`、`9.9992e-8`；EPguan 分别为 `1.1770e-7`、`1.2516e-7`、`1.5246e-7`，派生体仍通过 `2e-7` 门。

采纳项只有五个共享数据流变化：保留两个 resident CSR chunk 并为临时 CUDA 工作预留 6 GiB；branch/bed 合并为一次 SpMM；每轮 numerator 只上传一次旋转批输入；跨 chunk 复用大块 CUDA CSR allocation；host chunk cache 默认 25 GiB。拒绝并删除的实验包括全量/分段 `cudaHostRegister`、mmap+28 GiB、并行床、SpMM alg2/alg3、256 MiB staging 和三 resident chunk streaming；其中并行床的环境开关、异步 API、per-bed streams 与 scratch buffers 在最终审计中一并删除。它们分别触发显存风险、不支持或持续吞吐下降，没有保留第二生产实现。

在 RTX 4070 Ti SUPER 16 GB、64 GB 主存上，对 EP 三次迭代按 250 ms 采样 121 次：包含其他任务的总显存峰值为 15,519 MiB，最少剩余 543 MiB，运行完成。由此签收的低时延部署是 64 GB 主存 + SSD/NVMe 矩阵；32 GB 主存可降低 host cache 运行，但本轮不承诺相同吞吐。四床 2 分钟目标没有达到，当前证据支持的边界是约 3.16 分钟，而不是继续用高显存风险换取未证实的增益。

### 4.6 日志位置回归

最终 Release 一迭代真实 smoke 的 `recon_20260811_124935_255.log` 位于目标输出目录 `recon_out/perf_120s_20260811/final_release_5baeecc_smoke_1it/`；仓库根目录及 staging 根没有新增 `recon_*.log`。正常路径下旧的本地误写问题已不存在。若目标输出目录本身无法创建，logger 仍会打印明确警告并回退到当前工作目录；该行为是可观察的失败兜底，不应被解释为正常输出位置。

## 5. 决策、限制与停止条件

### 5.1 采纳与拒绝

最终采纳的共享修复为：反向存储的增益读窗、反向存储的 cosine blend 极性、原型 detector 68 固定区域、首次 PSF 前初始 ROI 掩膜，以及防止旧 detector 合同缓存复用的 schema 更新。输出端恢复一个最小保存数据流：未滤波主输出加两个独立派生副本；同时修复 CUDA Butterworth 尺寸切换缓存。性能端只在同一 compact CUDA 数据流内采用 resident cache、branch/bed SpMM 合批、单次批输入上传和 allocation 复用；`SPECT_COMPACT_BRANCH_BATCH=0` 保留可调用的正确性/性能基线。CLI 不新增输出滤波或性能参数，物理单位与钳制策略只由 `[Filtering]` 明确配置。

数值候选 CASToR 低前向中性比、0.01 MLEM 更新下限、直接 `A^T w` 加权灵敏度，以及上述无持续增益的性能候选均明确拒绝并从生产源码移除。保留它们只会制造第二数值路线或不可签收的显存风险。

### 5.2 已知限制

- MSVC 链接仍给出已知 `LNK4098` 默认库冲突警告，但 Release 构建、链接、CTest 233/233 和五次正式运行均成功；未把警告误报为已修复。
- 五组 50 次正式重建先于输出派生恢复；本轮没有重复执行约一小时的五组迭代，而是用冻结的同输入五组体积验证滤波算子，并用一次真实重建验证保存/manifest 接口。
- 原型是用户指定的实现基线，不是独立临床/物理真值。本轮未做新核素、新矩阵、新采集、DICOM 临床阅片或跨 GPU 架构的外推验证。
- Derenzo 通过但不像另外四组那样接近机器精度；若未来收紧 relL2 门，必须重新冻结合同后再定位，不能回写当前阈值。
- 50 次 2 分钟门只在本机单床三次重复中通过；四床 EP 的中位数是 189.67 秒。没有把单床结果外推为所有床数均通过。
- 共享 GPU 的峰值采样只剩 543 MiB，总显存余量偏紧；resident cache warmup 后若其他任务继续显著增长，仍可能触发 OOM。本轮未把这种外部并发变化隐藏为程序保证。

### 5.3 停止条件

在当前五数据集与现有阈值下，数值合同项已全部通过。性能线在两个 resident chunk + branch/bed batch 停止：单床已达到目标，已试的额外并行、pinning、mmap、staging 和算法选择均无持续增益或越过显存边界。只有出现可复现的四床瓶颈定位、新硬件/矩阵、收紧的门或新的安全优化假设时，才重新开启受控实验。

## 6. 复现与验证清单

### 6.1 精确复现命令

```powershell
cmake -S . -B build\Release -DSPECT_BUILD_TESTS=ON
cmake --build build\Release --config Release --target SPECT_recon spect_tests -j 8
ctest --test-dir build\Release -C Release --output-on-failure

python recon_out\canonical_stitch_20260807\preflight.py `
  --out recon_out\prototype_sieve_castor_fwhm1p3_final_20260811 `
  --exe build\Release\bin\Release\SPECT_recon.exe

python recon_out\canonical_stitch_20260807\run_all.py `
  --out recon_out\prototype_sieve_castor_fwhm1p3_final_20260811 `
  --exe build\Release\bin\Release\SPECT_recon.exe `
  --denom-cache-dir recon_out\prototype_sieve_castor_fwhm1p3_final_20260811\denom_cache

python recon_out\canonical_stitch_20260807\analyze_parity.py `
  --out recon_out\prototype_sieve_castor_fwhm1p3_final_20260811

build\Release\bin\Release\spect_tests.exe `
  --gtest_output=json:D:\Projects\SPECT-recon\recon_out\prototype_sieve_castor_fwhm1p3_final_20260811\spect_tests.json

$env:SPECT_FILTER_PARITY_BASE_ROOT = (Resolve-Path recon_out\canonical_stitch_20260807\runs).Path
$env:SPECT_FILTER_PARITY_REFERENCE_ROOT = (Resolve-Path recon_out\canonical_stitch_20260807\prototype_filter_parity).Path
build\Release\bin\Release\spect_tests.exe `
  --gtest_filter=OutputFilterParityTest.FrozenFiveDatasetsWhenRequested

# 清理后先重建同一 SHA-256 矩阵副本和独立基线 worktree。
New-Item -ItemType Directory -Path tmp\matrix_perf_nvme_20260811
Copy-Item -Path F:\matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220\* `
  -Destination tmp\matrix_perf_nvme_20260811
git worktree add --detach tmp\baseline_17bd22c 17bd22c8c22192dbe4ddf3fed57456e41bb81161
Push-Location tmp\baseline_17bd22c
.\build_release.bat
Pop-Location

# 性能 A/B：按 A1,B1,B2,A2,A3,B3 交错；每组取 elapsed_seconds 中位数。
$env:SPECT_COMPACT_BRANCH_BATCH = '0'
recon_out\perf_120s_20260811\run_variant.ps1 `
  -Label median_epguan3_baseline_A1 -Dataset epguan -Iterations 3 `
  -MatrixDir D:\Projects\SPECT-recon\tmp\matrix_perf_nvme_20260811 `
  -DenomCache D:\Projects\SPECT-recon\recon_out\perf_120s_20260811\nvme_denom\epguan.mhd `
  -Executable D:\Projects\SPECT-recon\tmp\baseline_17bd22c\staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe -Verbosity 0
Remove-Item Env:\SPECT_COMPACT_BRANCH_BATCH
recon_out\perf_120s_20260811\run_variant.ps1 `
  -Label median_epguan3_candidate_B1 -Dataset epguan -Iterations 3 `
  -MatrixDir D:\Projects\SPECT-recon\tmp\matrix_perf_nvme_20260811 `
  -DenomCache D:\Projects\SPECT-recon\recon_out\perf_120s_20260811\nvme_denom\epguan.mhd `
  -Executable D:\Projects\SPECT-recon\staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe -Verbosity 0

python recon_out\canonical_stitch_20260807\certify_outputs.py `
  --out recon_out\prototype_sieve_castor_fwhm1p3_final_20260811
```

逐数据集的完整展开命令、绝对输入路径、返回码与耗时保存在 `run_record.json`；不需要从本报告手工重构床位列表。收敛清理已删除 26.1 GB 临时矩阵副本和独立基线 worktree，以上前置命令可按冻结哈希重新创建；最终中位数记录和决策 JSON 保留。

### 6.2 验证状态

| 检查 | 证据 | 状态 |
|---|---|---|
| Release 配置与编译 | `20260811+5baeecc`；构建/部署 exe SHA-256 同为 `B927BE...524E0`；`sm_86/89/120` | 通过；保留 LNK4098 标记 |
| 部署配置 | 源码/部署 `spect_config.ini` SHA-256 同为 `35E950...FB4F`；部署 `-dumpConf` 逐项核对输出滤波与 clamp | 通过 |
| 正式单元/集成测试 | CTest 233/233；外部五集门默认 skip、按冻结环境变量独立执行 1/1 通过 | 通过 |
| detector 68 精确映射 | 新回归测试检查 60×12×12、边界及旧错误区域 | 通过 |
| 一迭代结构门 | no-PSF `7.30e-8`；sieve `8.95e-8` | 通过 |
| 五数据集 50 次 MLEM | 五进程退出码 0；聚合 analyzer 全门为真 | 通过 |
| 方向/间距/有限非负 | 每组 manifest 与指标 JSON | 通过 |
| 输出滤波同输入门 | 五组主输入/Butterworth relL2=0；Smooth 最大 1.1563017e-7；clamp-on 对应门通过 | 通过 |
| 真实保存入口 | 一迭代 Derenzo 写出主/Smooth/Butterworth 六文件；全部有限非负；manifest 一致 | 通过 |
| 直接切片视觉 QA | 共享原型窗宽、无插值、无候选缩放 | 通过 |
| 基础重建输出证书 | 原五组 21 个正式证据文件、90,349,475 字节 SHA-256 列表 | 通过；证书不包含本轮派生 smoke |
| 性能短 A/B | 基线/候选各 3 次交错，中位数 52.4101 / 35.9731 秒；三类输出最大 relL2 1.3564e-7 | 通过，候选 -31.36% |
| 受保护逐分支基线 | 删除并行床残余后以 `SPECT_COMPACT_BRANCH_BATCH=0` 执行一迭代；相对最终合批主/Smooth/Butterworth 最大 relL2 `1.3706e-7` | 通过；基线仍可调用 |
| 50 次性能重复 | Derenzo 3 次中位数 104.5868 秒；EP 四床 3 次中位数 189.6690 秒 | 单床通过；四床未达 120 秒 |
| 显存采样 | EP 3 次、250 ms×121；总峰值 15,519 MiB、最少剩余 543 MiB | 通过但余量偏紧 |
| 日志输出目录 | 最终源码 smoke 日志在目标目录；仓库/staging 根无新增日志 | 正常路径通过；目录创建失败仍有显式 CWD fallback |
| 未验证范围 | 其余三组在优化后二次执行 50 次、32 GB 主存同吞吐、新数据、新矩阵、临床正确性、跨架构性能 | 明确未验证 |

## References

本报告使用的完整本地证据路径见 `reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence_sources.md`。
