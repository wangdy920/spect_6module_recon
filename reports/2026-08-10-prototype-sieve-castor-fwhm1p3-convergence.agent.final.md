# SPECT 重构与受保护原型 `sieve_castor_fwhm1p3` 收敛审查

**执行区间：** 2026-08-10 至 2026-08-11

**最终状态：** 通过冻结五数据集合同；当前重构路线可判定为与受保护原型输出一致。
**数值权威：** `6module_bak_src/spect_recon_PSC_6module_24pin_YSO_bedmerge.cu` 中的 `sieve + CASToR + FWHM 1.3 mm + 3σ` 路线。

## 1. 执行结论与判定合同

本轮不是以旧发布二进制或“看起来相似”为裁决，而是以用户指定的受保护原型为唯一数值基线。重构后的单 GPU compact-v4 路径在修复四类共享根因后，完成正式 Release 重编译，并对五个冻结数据集逐一执行完整数据、50 次 MLEM。最终聚合分析的 `all_five_object_endpoints_passed`、`all_five_numerical_gates_passed`、`single_final_output_contract` 与 `final_endpoint_passed` 均为 `true`。

| 合同项 | 冻结口径 | 本轮结果 |
|---|---|---|
| 输入 | 指定五数据集、同一 compact 90z 系统矩阵、同一配置与 detector 68 屏蔽 | 42 个文件以路径、大小和 SHA-256 预检，合计 26,342,787,420 字节 |
| 重建 | 单 GPU、完整数据、MLEM 50 次 | 五次正式进程退出码均为 0 |
| PSF | sieve、CASToR erf 体素积分核、FWHM 1.3 mm、3σ | manifest 与命令逐项一致 |
| 多床 | `Bed1=max physical z`、`z_bed_flip=1`、center66/cosine10、原型增益链 | 四/五床增益、方向与对象门通过 |
| 数值门 | 支撑区 NCC ≥ 0.99；缩放匹配 relL2 ≤ 0.10 | 五组全部通过；最弱项为 Derenzo 0.995661 / 0.068587 |
| 对象门 | Derenzo、EP、mouse 各自冻结指标 | 五组全部通过 |
| 输出 | 每组只有 `recon_it50.mhd/.raw`；有限、非负、间距/方向正确 | 五组全部通过；21 个正式证据产物已 SHA-256 认证 |

结论的量词仅覆盖这五个冻结输入和上述合同。它证明当前实现复现指定原型输出；不把原型提升为独立物理真值，也不外推到未验证的矩阵、采集或参数组合。

## 2. 最近重构审查与根因修复

审查先恢复完整构建和可复现入口，再以一迭代结构证书定位数值差异，最后才运行昂贵的 50 次终点。所有最终改动都位于共享数据流，不含按 Derenzo、EP 或 mouse 分支的对象调参。

| 审查项 | 发现 | 最终处置与证据 |
|---|---|---|
| 构建/接口收敛 | 重构后存在 include、命名空间和测试目标残留；输出阶段及旧后处理字段仍有失效表面 | 恢复 `SPECT_recon`/`spect_tests` Release 构建；生产只保留 compact-v4、单一输出与 `none|in_recon` 滤波阶段 |
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

`input_preflight.json` 记录 42 个输入文件，包括正式可执行文件、配置、算法配置、坏探测器列表、受保护原型源码、6 个矩阵块及 metadata、18 个床位投影、5 组原型 Analyze 输出及 sidecar。正式二进制 SHA-256 为：

```text
98F4771023E2C9A40C7045CE8F2EB95DDA6E7BD1617EBDDC7F899E857AE03019
```

受保护原型源码 SHA-256 为：

```text
6B332DAD80310B2F0E4F8723B45B2AEB6C1B5DC24A153A6F720A467D45573B0E
```

### 3.2 比较规则

原型 Analyze 输出与当前 MHD 输出按冻结方向比较：Derenzo 为 `rot0`，所有多床数据为 `zflip+rot0`。数值比较报告原始全体积 NCC、原型 1%–P99.9 支撑域 NCC，以及最小二乘尺度匹配后的 relL2。尺度匹配只用于既定数值门，不修改候选输出或 QA 图。

对象门分别为：

- Derenzo：中心保留率 ≥ 95%，rod separability 相对变化绝对值 ≤ 5%，覆盖下降不超过 2 个百分点，最长 gap 增量不超过 1 层。
- EP：轴向 CV 相对变化绝对值 ≤ 5%，覆盖下降不超过 2 个百分点，seam 中位变化 ≥ -0.02。
- Mouse：支撑区 NCC ≥ 0.99，逐切片 NCC 中位数 ≥ 0.95。
- 所有输出：体素有限、无负值、间距误差 ≤ `1e-6 mm`，manifest 仅声明 MHD/RAW 两个最终文件。

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

### 4.3 正式运行耗时

| 数据集 | 耗时（秒） | 退出码 |
|---|---:|---:|
| Derenzo | 594.79 | 0 |
| EPguan | 696.66 | 0 |
| EPguan 19.2 mm | 707.68 | 0 |
| CH mouse | 805.77 | 0 |
| Mouse | 741.88 | 0 |

耗时仅记录本机本轮运行，不作为性能采纳结论。四/五床的同几何分母缓存被复用，但每组投影、50 次 numerator/update 和最终输出均独立执行。

## 5. 决策、限制与停止条件

### 5.1 采纳与拒绝

最终采纳的共享修复为：反向存储的增益读窗、反向存储的 cosine blend 极性、原型 detector 68 固定区域、首次 PSF 前初始 ROI 掩膜，以及防止旧 detector 合同缓存复用的 schema 更新。先前已完成的构建、输出表面、manifest、fail-closed 和 CUDA 投影缓存修复一并保留。

以下候选明确拒绝且实现已回退：CASToR 低前向中性比、0.01 MLEM 更新下限、直接 `A^T w` 加权灵敏度。原因不是“代码复杂”，而是它们在冻结门下没有减少目标误差；保留它们只会制造第二数值路线。

### 5.2 已知限制

- MSVC 链接仍给出已知 `LNK4098` 默认库冲突警告，但 Release 构建、链接、228 项测试和五次正式运行均成功；未把警告误报为已修复。
- 当前工作树包含本轮之前已存在的大范围重构改动且尚未提交。本报告验证的是当前工作树构建出的指定 SHA-256 二进制，不代表已完成 Git 发布收口。
- 原型是用户指定的实现基线，不是独立临床/物理真值。本轮未做新核素、新矩阵、新采集、DICOM 临床阅片或跨 GPU 架构的外推验证。
- Derenzo 通过但不像另外四组那样接近机器精度；若未来收紧 relL2 门，必须重新冻结合同后再定位，不能回写当前阈值。

### 5.3 停止条件

在当前五数据集与现有阈值下，继续引入数值候选不会减少尚未通过的合同项——因为合同项已全部通过。因此本优化线停止在当前单一生产实现。只有出现新的冻结输入、收紧的数值/对象门、不同矩阵或可复现的新失败时，才重新开启受控实验。

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

python recon_out\canonical_stitch_20260807\certify_outputs.py `
  --out recon_out\prototype_sieve_castor_fwhm1p3_final_20260811
```

逐数据集的完整展开命令、绝对输入路径、返回码与耗时保存在 `run_record.json`；不需要从本报告手工重构床位列表。

### 6.2 验证状态

| 检查 | 证据 | 状态 |
|---|---|---|
| Release 配置与编译 | `SPECT_recon`、`spect_tests` 目标成功；正式 exe 哈希已冻结 | 通过；保留 LNK4098 标记 |
| 正式单元/集成测试 | 直接测试 JSON 228/228；CTest 228/228 | 通过 |
| detector 68 精确映射 | 新回归测试检查 60×12×12、边界及旧错误区域 | 通过 |
| 一迭代结构门 | no-PSF `7.30e-8`；sieve `8.95e-8` | 通过 |
| 五数据集 50 次 MLEM | 五进程退出码 0；聚合 analyzer 全门为真 | 通过 |
| 方向/间距/有限非负 | 每组 manifest 与指标 JSON | 通过 |
| 单一输出清单 | 每组仅 MHD/RAW；无派生 smooth/butter 输出 | 通过 |
| 直接切片视觉 QA | 共享原型窗宽、无插值、无候选缩放 | 通过 |
| 输出证书 | 21 个正式证据文件、90,349,475 字节 SHA-256 列表 | 通过 |
| 未验证范围 | 新数据、新矩阵、临床正确性、发布提交/打包、跨架构性能 | 明确未验证 |

## References

本报告使用的完整本地证据路径见 `reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence_sources.md`。
