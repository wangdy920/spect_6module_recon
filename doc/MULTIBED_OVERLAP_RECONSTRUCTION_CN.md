# 多床位协同重建、Overlap 与床间连续性

## 1. 文档范围与当前结论

本文说明当前 `SPECT-recon` 如何从多个床位投影创建一个全局重建体，以及 `bed_overlap` 如何影响：

- 全局 Z 轴尺寸和床位索引；
- forward/backward 使用的局部床位视图；
- sensitivity denominator（以下简称 `denom`）的全局累加；
- numerator 的全局累加和 MLEM 更新；
- Operator PSF 是否跨床位边界传播；
- 输出阶段的 hard-cut、翻转、圆形 FOV 和 `cropEnd`；
- 最终床间暗带、亮线、断裂或形态缺失。

本文以当前主用的 compact-symmetric `260601` 多床位路径为重点。standard CSR 路径会单独指出，不把 compact 路径的 PSF 结论自动外推到 standard 路径。

当前项目结论：

1. 最终推荐配置为 `overlap_in_recon=0, crop_method=sigmoid`：重建床位互不重叠，输出端使用旧版同定义的归一化 `sigmoid(k=12)` 权重融合。
2. 输出只合并相邻床位的内部接缝，保留当前 `Nz * bed_count - sum(overlap)` 几何；不照搬旧 EXE 对首尾 overlap 的裁剪。
3. `sigmoid` 不做 positive-only fill，不修改迭代、denom 或 numerator。
4. `direct` hard-cut 保留为控制/回退方法；旧 `blendfill` 已退出公开配置和运行路径。
5. `overlap_in_recon=1` 仍是受支持的联合重建模式，但不再是默认配置；它会同时改变全局 `denom`、numerator 和 PSF scope。
6. 圆形 XY FOV 只处理外围固定弧形响应，不修复轴向床间接缝。

## 2. 活跃入口与源文件

| 环节 | 活跃入口 | 责任 |
|---|---|---|
| 参数和 ini | [`src/cli/ArgumentParser.cpp`](../src/cli/ArgumentParser.cpp) | 读取 `-dfList`、`-bedOverlap` 和 `[MultiBed]` 配置 |
| 参数状态 | [`include/cli/ReconArgs.h`](../include/cli/ReconArgs.h) | 保存物理 overlap、重建 overlap 和输出 overlap |
| 床位准备 | [`src/main/ReconSetup.cpp`](../src/main/ReconSetup.cpp) | 识别床号、选择实际 joint、决定 overlap 是否进入重建 |
| Z 轴映射 | [`include/app/BedOffsets.h`](../include/app/BedOffsets.h) | 计算 `bedZStarts`，提取和累加床位体数据 |
| 主流程装配 | [`src/main/MainProgram.cpp`](../src/main/MainProgram.cpp) | 加载投影、建立全局几何、设置 `ReconOpts` 和输出变换 |
| compact 联合迭代 | [`src/main/CompactSymmetricReconLoop.cpp`](../src/main/CompactSymmetricReconLoop.cpp) | 计算/缓存 denom，执行多床 numerator 和全局更新 |
| standard 联合迭代 | [`src/main/StandardReconLoop.cpp`](../src/main/StandardReconLoop.cpp) | standard CSR 多床路径 |
| 输出处理 | [`src/main/VolumeOutput.cpp`](../src/main/VolumeOutput.cpp) | hard-cut、方向变换、FOV、`cropEnd`、MHD 写出 |
| 当前默认配置 | [`config/spect_config.ini`](../config/spect_config.ini) | overlap=16（stride=64）、输出 sigmoid(k=12) 拼接、FOV 和方向默认值 |
| 双模式验证 | [`scripts/eval_overlap_recon_switch.py`](../scripts/eval_overlap_recon_switch.py) | mouse/zimu 固定窗和接缝指标比较 |

## 3. 端到端流程

| 阶段 | 输入 | 主要处理 | 输出/状态 |
|---|---|---|---|
| 1. 投影列表 | `-dfList Bed1.bin;...` | 路径标准化，按文件名中的 `BedN` 稳定排序 | `projPaths` |
| 2. 床位准备 | 单床尺寸、床号、`bed_overlap`、`overlap_in_recon` | 选择相邻床位对应的物理 overlap；决定重建 overlap 是否置零 | `baseNz`、`bedCount`、两份 overlap |
| 3. 全局几何 | 重建 overlap | 计算每床 `bedZStarts` 和全局 `Nz` | joint global volume |
| 4. 投影加载 | 每床 BIN/MAT | 每床必须匹配同一系统矩阵完整投影行数 | 连续 projection buffer |
| 5. denom | 系统矩阵、PSF、床位映射 | 计算单床 `A_b^T 1`，再按模式累加/卷积 | 全局 sensitivity denom |
| 6. numerator | 当前全局图像、每床投影 | 提取床位视图，forward、ratio、backward，再累加到全局 | 全局 numerator |
| 7. 更新 | image、numerator、denom | MLEM 或 BSREM 更新；mask/nonnegative 等 | 下一迭代全局图像 |
| 8. 输出 | 最终全局图像 | 可选 hard-cut、全局翻转、FOV、`cropEnd` | MHD/RAW |
| 9. 离线 QA | MHD | 固定窗、seam ±8、主体和非接缝指标 | PNG + manifest JSON |

## 4. 输入和床位编号合同

### 4.1 `-dfList` 和 `-bProjList`

- `-df` 与 `-dfList` 互斥；多床位使用 `-dfList`。
- `-bProj` 与 `-bProjList` 同样互斥。
- 当前最多支持 5 个床位。
- 文件名中可识别 `Bed1`、`BED_2`、`bed-3` 等形式。
- 检测到床号时按床号升序稳定排序；未检测到床号的路径排在已编号路径之后，并保持原相对顺序。
- projection 和 background projection 必须使用一致床位顺序。
- compact-symmetric 路径要求每床投影长度精确等于 `compactMeta.fullRows()`；不会静默截断 compact 输入。

### 4.2 固定 overlap 表

配置示例：

```ini
[MultiBed]
bed_overlap = 16,16,16,0,0
```

固定槽含义：

| 槽位 | 物理 joint |
|---|---|
| 第 1 个值 | Bed1 ↔ Bed2 |
| 第 2 个值 | Bed2 ↔ Bed3 |
| 第 3 个值 | Bed3 ↔ Bed4 |
| 第 4 个值 | Bed4 ↔ Bed5 |
| 第 5 个值 | 保留固定表占位；当前最多 5 床 |

规则：

- 只提供一个值时，复制到所有固定槽。
- 提供的固定表不足 5 项时，缺失槽为 0，并产生 warning。
- 若 `-dfList` 是 Bed2、Bed3、Bed4，则选择固定表中的第 2、3 个 joint 值。
- 非相邻床位没有可定义的固定 joint，例如 Bed1、Bed3，实际 overlap 强制为 0 并产生 warning。
- overlap 会被限制到 `[0, baseNz)`。

## 5. 两份 overlap 状态

床位准备后，程序刻意保留两份不同语义的数据：

| 字段 | 语义 |
|---|---|
| `outputBedOverlaps` | 从配置和床号选择出的物理 overlap；供输出 hard-cut 和 manifest 使用 |
| `bedOverlaps` | 实际参与重建几何的 overlap |

选择逻辑：

```text
overlap_in_recon = 1  => bedOverlaps = outputBedOverlaps
overlap_in_recon = 0  => bedOverlaps = [0, 0, ...]
```

这样可避免同一 overlap 同时在迭代中共享体素、又在输出阶段被二次裁剪。

## 6. 全局 Z 几何

设：

- 床位数为 `B`；
- 单床 Z 像素数为 `L = baseNz`；
- 第 `i` 个相邻床 joint 的重建 overlap 为 `o_i`；
- 第一个床位从 `s_0=0` 开始。

床位起点：

```text
s_0 = 0
s_b = s_(b-1) + L - o_(b-1)
```

全局 Z 尺寸：

```text
globalNz = B * L - sum(o_i)
```

当前四床、`baseNz=80`、物理 overlap=8 的两个模式：

| 模式 | 重建 overlap | `bedZStarts` | 迭代 `globalNz` | 最终输出 Z |
|---|---|---|---:|---:|
| `overlap_in_recon=1` | 8,8,8 | 0,72,144,216 | 296 | 296 |
| `overlap_in_recon=0` | 0,0,0 | 0,80,160,240 | 320 | hard-cut 后 296 |

### 6.1 overlap 参与重建时的共享区

以 Bed1 和 Bed2 为例：

- Bed1 覆盖全局 Z `[0,80)`；
- Bed2 覆盖全局 Z `[72,152)`；
- `[72,80)` 是 8 层共享全局体素。

这里不是先重建两幅独立图再融合。两个床位从迭代开始就读写同一个全局体素区域。

### 6.2 QA seam 参考位置

overlap=8 时，QA 使用 overlap 中点作为接缝参考：

```text
seam = nextBedStart + overlap / 2
```

因此当前三处参考位置是：

```text
76, 148, 220
```

注意：`overlap_in_recon=1` 时这只是共享区的中心参考线，并不是数据结构中真实存在的一条硬边界。

## 7. Z 方向和虚拟索引

当前 compact 多床位配置：

```ini
z_bed_flip = 1
z_out_flip = 0
```

compact joint 模式不会等到输出后再翻转每床。它在每次床位视图提取和回投累加时使用虚拟映射：

```text
globalZ = bedStart + (baseNz - 1 - localZ)
```

这保证：

- forward 提取的床位局部 Z 方向正确；
- denom 和 numerator 回投使用完全相同的方向；
- 最终 joint image 已经处于该虚拟 Z 方向；
- 输出阶段不能再次逐床翻转，否则会发生二次翻转。

`z_out_flip` 是整个最终体的 Z 翻转，语义与床内虚拟映射不同；它在 hard-cut 之后执行。

非 compact joint 路径仍可能在输出阶段执行逐床 Z flip，因此不能把 compact 的方向处理规则直接复制到所有路径。

## 8. 联合重建数学含义

将每床系统矩阵按投影域纵向堆叠：

```text
A = [A_1; A_2; ...; A_B]
y = [y_1; y_2; ...; y_B]
```

无背景项的 MLEM 可写为：

```text
denom = A^T 1 = sum_b A_b^T 1
numer = A^T (y / (A x))
x_next = x * numer / denom
```

有 background/scatter 时，ratio 的分母使用 `A x + b`，但床位累加原则不变。

### 8.1 为什么 overlap 会改变 denom

当相邻床覆盖同一全局体素时，该体素在联合系统矩阵中拥有来自两床的观测行。因此：

- denom 中会累加两床对该体素的 sensitivity；
- numerator 中也会累加两床的 backprojection；
- 额外曝光不会简单把图像乘以 2，因为 numerator 和 denom 同时变化；
- overlap 区由两床共同约束，通常比把两个床边缘直接相邻更连续。

当前累加是直接加法，没有 cosine 权重、gain ramp 或 seam-specific scaling。

### 8.2 不能只修改 denom

如果未来对床位 overlap 使用权重 `w_b(z)`，必须把权重作为统一 operator 合同的一部分，同时进入：

- sensitivity/denom；
- forward model 或对应观测贡献；
- numerator backward；
- denom cache identity。

只改 denom 或只改最终图像都会破坏 `numerator/denom` 的配对含义，可能造成亮带、暗带或定量偏差。

## 9. Compact 路径的 denom 与 PSF

当前 compact Operator PSF 模型可概括为 `P = A G`，因此 sensitivity 是：

```text
P^T 1 = G^T A^T 1
```

### 9.1 `overlap_in_recon=1`

当前执行顺序：

1. CUDA 计算单床 `localDenom = A_b^T 1`。
2. 每床 `localDenom` 按 `bedZStarts` 和虚拟 Z 映射直接累加到全局 denom。
3. 对完整 296 层全局 denom 执行 `G^T`。
4. 每次迭代先对完整全局 image 执行 `G`。
5. 从全局 PSF image 提取各床视图并计算各床 numerator。
6. 各床 numerator 累加到全局后，再对全局 numerator 执行 `G^T`。

结果：PSF 可跨共享 overlap 区连续传播，denom 和 numerator 使用同一全局 PSF 范围。

### 9.2 `overlap_in_recon=0`

当前执行顺序：

1. 单床 `localDenom` 先执行 `G^T`。
2. 各床 PSF denom 放入互不重叠的 80 层段。
3. 每次迭代分别对每床局部 image 执行 `G`。
4. 每床 numerator 分别执行 `G^T`，再放回各自 80 层段。
5. 迭代期间床位之间没有共享体素，也没有跨床 PSF。

这使 320 层迭代体更接近“一个进程内的块分离床位重建”。最终 hard-cut 只能删除边缘层，不能恢复迭代期间缺失的跨床约束。

上述“无跨床耦合”特指当前验证配置中的 likelihood 与 Operator PSF。若另外启用全局 Joseph、Butterworth、Gaussian smoothing 或跨段正则化，它们仍可能在 80/160/240 的段边界传播；本次双模式验证中这些项均关闭。

### 9.3 Standard 路径边界

standard CSR 路径同样按 `bedZStarts` 累加多床 denom，并受 `overlap_in_recon` 的几何影响。但当前源码中 standard sensitivity 是先对 `localDenom` 执行 PSF backward，再累加到全局。

因此：

- “overlap 体素接收多床 sensitivity”仍成立；
- “先全局累加再执行全局 PSF”是当前 compact 多床实现的特定行为；
- 本文 mouse/zimu 运行证据只验证 compact-symmetric `260601`，没有把相同成像结论声明为 standard 路径已验证。

## 10. Denom 缓存合同

compact denom cache manifest 会记录至少以下相关字段：

```text
global_dims
bed_count
bed_z_starts
bed_overlap_in_recon
output_bed_overlaps
joint_virtual_z_flip
psf_mode
psf_fwhm
psf_multibed_scope
```

当前真实 manifest：

| 字段 | overlap 参与重建 | output-only hard-cut |
|---|---|---|
| `global_dims` | 108,108,296 | 108,108,320 |
| `bed_z_starts` | 0,72,144,216 | 0,80,160,240 |
| `bed_overlap_in_recon` | 1 | 0 |
| `output_bed_overlaps` | 8,8,8 | 8,8,8 |
| `joint_virtual_z_flip` | 1 | 1 |
| `psf_multibed_scope` | `global_volume` | `per_bed_output_hardcut` |

两种模式不能共享 denom cache。manifest 不匹配时程序会拒绝旧缓存，而不是静默加载。

切换以下任一项时也应使用不同 cache 或让 manifest 重新生成：

- PSF 开关、FWHM、kernel 或截断范围；
- CTAC 和衰减图；
- `bed_overlap`、床位数、床位顺序；
- `z_bed_flip`；
- Joseph denom smoothing；
- 全局尺寸或 voxel spacing。

## 11. 输出阶段和 hard-cut

统一输出顺序为：

```text
必要的非 joint 每床 Z flip
-> output-only overlap hard-cut
-> whole-output Z flip
-> whole-output X flip
-> circular XY FOV
-> cropEnd
-> MHD/RAW
```

compact joint 的 `z_bed_flip` 已在迭代内完成，因此不会在这里再次逐床 flip。

### 11.1 对称 hard-cut 规则

仅当 `overlap_in_recon=0` 时启用。对于一个物理 overlap `o`：

```text
左床末端裁剪 floor(o / 2)
右床起始裁剪 o - floor(o / 2)
```

偶数 overlap=8 时即两侧各裁 4 层。

四床、单床 80 层的源区间和输出区间：

| 床位 | 320 层源体中保留区间 | 保留层数 | 296 层输出区间 |
|---|---|---:|---|
| Bed1 | `[0,76)` | 76 | `[0,76)` |
| Bed2 | `[84,156)` | 72 | `[76,148)` |
| Bed3 | `[164,236)` | 72 | `[148,220)` |
| Bed4 | `[244,320)` | 76 | `[220,296)` |

图像和 mask 使用完全相同的区间选择。

### 11.2 历史 `blendfill` 规则（已退役）

以下内容仅保留为既有实验解释，不再对应当前公开配置或运行路径。

仅当 `overlap_in_recon=0` 且 `crop_method=blendfill` 时启用：

1. 按物理 overlap 将完整床位映射到最终 Z，重叠层使用 half-sample cosine 权重归一化；
2. 使用 seam 两侧各 4 层建立逐像素线性趋势，XY 上先做 `sigma=1.5`、truncate=4 的 Gaussian；
3. 只修改 seam ±3 层，并只补偿预测值高于原值的部分：`out = original + fill_alpha * max(prediction-original, 0)`；
4. 非接缝层逐体素不变。`fill_alpha=0.25/0.50` 分别对应 fill25/fill50。

当前四组统一验证选择 fill25；fill50 可调但会在 mouse/zimu 增加过度连接或高信号，不能视为统一推荐。

### 11.3 为什么 hard-cut 仍会有缝

hard-cut 不插值也不生成新值，但它会把两床各自靠近 FOV 边缘的切片直接相邻。床位边缘可能存在：

- sensitivity 下降或截断；
- PSF 边界条件差异；
- 投影支持范围减少；
- 床内结构在边缘处发生形态畸变。

因此“不填值”不等于“连续”。当前实测的薄水平线来自边缘响应相邻，而不是 hard-cut 代码做了强度加法。

## 12. 圆形 XY FOV 的边界

`fov_radius_mm` 在输出阶段按真实 `vx/vy` 计算圆形权重：

- 半径内权重严格为 1；
- 半径外使用固定 4 voxel taper；
- 更外侧权重为 0；
- 不改变迭代、denom、numerator 或 Z 轴床位几何。

当前 `0.3 mm` 数据配置为：

```ini
fov_radius_mm = 15.0
```

它可明显去除固定出现的外围弧形/曲面响应，但不会降低轴向 seam jump。XY 弧形清理和 Z 轴接缝必须作为两个独立问题验证。

## 13. 常见不连续来源和排查

| 视觉表现 | 可能根因 | 优先检查 |
|---|---|---|
| overlap 中心出现薄亮线 | 输出填充注入、床边缘高响应、错误的双重融合 | 与同迭代 direct/FOV-only 比较正差分；确认 `blendfill` 未与 joint overlap 同时启用 |
| overlap 中心出现暗带 | 两床 sensitivity 边缘下降、错误 gain、denom/numerator 不配对 | 查看 raw denom 轴向 profile；比较 seam sum ratio |
| 字母/小鼠结构在 seam 断裂 | 床边缘形态缺失、错误 Z 方向、物理 overlap 不准确 | 检查 `z_bed_flip`、床号、`bedZStarts`、固定窗 seam zoom |
| 每床内容上下翻转 | compact 虚拟 Z 与输出 per-bed flip 重复 | compact joint 只允许迭代内虚拟映射，不应输出二次 flip |
| 三处 seam 模式重复 | 固定床位 sensitivity/PSF 边缘模式 | 检查 denom 三个 joint 的轴向模式是否重复 |
| 只在外围出现弧形 | XY FOV 边缘响应 | 对比 FOV off/on；不要用 Z smoothing 处理 |
| 切换参数后图像异常 | 复用了旧 denom cache | 检查 manifest 的 geometry、flip、PSF 和 overlap mode |
| 固定窗看似不同但数值接近 | viewer 自动窗宽或逐体归一化 | 使用共享 P99.5 窗和 raw-value difference |

## 14. 当前双模式验证证据

实验根目录：

```text
F:\DATA\rawdata\SUPER_PSC_1_0\SPECT\dicom_overlap8_seam_polish_20260710\overlap_mode_runtime_v1
```

合同：

- compact-symmetric `260601`；
- 4 床；
- `108x108x80`/床；
- voxel `0.3x0.3x0.3 mm`；
- overlap 8；
- MLEM 1 iteration；
- Operator PSF FWHM 1.5 mm；
- FOV radius 15 mm；
- mouse 和 zimu 使用各自同源投影，两个模式使用同一二进制和同一参数。

结果：

| 指标 | mouse hard-cut | mouse overlap-in-recon | zimu hard-cut | zimu overlap-in-recon |
|---|---:|---:|---:|---:|
| mean seam relative jump | 0.1298 | 0.0175 | 0.1303 | 0.0183 |
| seam sum ratio hard-cut/control | 0.9654 | 1.0000 | 0.9658 | 1.0000 |
| trusted total sum ratio | 0.9924 | 1.0000 | 0.9922 | 1.0000 |
| high-signal sum ratio | 0.9838 | 1.0000 | 0.9831 | 1.0000 |

固定窗显示 output-only hard-cut 在三处床位边界产生薄水平线。结论：

```text
keep switch
reject overlap_in_recon=0 for promotion
recommended default: overlap_in_recon=1
```

证据文件：

```text
overlap_mode_runtime_v1\qa\manifest.json
overlap_mode_runtime_v1\qa\mouse_overlap_mode_coronal.png
overlap_mode_runtime_v1\qa\mouse_overlap_mode_seams.png
overlap_mode_runtime_v1\qa\zimu_overlap_mode_coronal.png
overlap_mode_runtime_v1\qa\zimu_overlap_mode_seams.png
```

## 15. 当前主线与离线参考的边界

### 15.1 当前 C++ 主线

包含：

- `overlap_in_recon=1` 时 overlap 参与 joint geometry 与 denom/numerator direct accumulation；
- 默认 `overlap_in_recon=0`，各床独立重建后以 overlap16、stride64、归一化 sigmoid `k=12` 融合；
- compact global PSF 或 output-only 模式下 per-bed PSF；
- compact joint virtual Z mapping；
- 配置可控的对象感知接缝后处理：mouse no-op、规则仿体轮廓约束插值、zimu 拓扑锁定修正；
- 输出圆形 XY FOV；
- whole Z/X flip 和 `cropEnd`。

不包含：

- joint overlap volume 上的二次 cosine image blending；
- slopeMatchedBias、gain ramp、contour gain；
- XY 注册后 overlap 融合；
- 大范围 Z smoothing。

### 15.2 离线 Python 参考

仅保留当前默认路径的数值参考与 QA：

| 脚本 | 用途/状态 |
|---|---|
| `postprocess_overlap16_auto.py` | 冻结的 overlap16/k12 与对象感知后处理 Python 参考 |
| `compare_seam_postprocess_parity.py` | C++/Python 四类数据数值一致性 gate |
| `export_recon_volume.py` | MHD 到 DICOM/PNG 的下游导出与回读确认 |

### 15.3 DICOM 转换

C++ 重建主线输出 MHD/RAW。后续补齐到 400 层及 DICOM 转换属于下游导出，不参与重建 overlap 或 denom。

不要把：

- 重建 `bed_overlap`；
- 输出 `cropEnd`；
- DICOM 居中补零；

混为同一个裁剪概念。

## 16. 参数建议

当前推荐配置：

```ini
[MultiBed]
bed_overlap = 16,16,16,0,0
overlap_in_recon = 0
crop_method = sigmoid
fov_radius_mm = 15.0
crop_end = 0,0
z_bed_flip = 1
z_out_flip = 0
x_out_flip = 0
```

建议：

1. 常规 compact 多床默认使用 `overlap_in_recon=0, crop_method=sigmoid`。
2. `sigmoid` 固定为归一化 `k=12` 权重；不提供 fill 强度参数，不裁剪首床起点或末床终点。
3. 修改 `bed_overlap` 后先检查 `bedZStarts` 和最终 `Nz`，再生成新 denom cache。
4. 对 seam 算法的任何强度修正必须保留 hard crop/FOV-only 对照，并使用共享窗宽。
5. 如果研究 sensitivity 校正，优先建立物理可解释的床位响应/normalization 模型，并同步 numerator 与 denom。
6. 不使用整体模糊或大范围 Z smoothing 掩盖 seam。

## 17. 验证状态与未确认项

已验证：

- `overlap_in_recon` 配置读取和 dump；
- 物理 overlap 与重建 overlap 分离；
- 四床 overlap=8 的两套 Z 几何；
- output hard-cut 对 image/mask 使用相同切片；
- hard-cut 在 whole Z flip 和 `cropEnd` 之前执行；
- compact denom manifest 区分模式和 PSF scope；
- mouse/zimu 真实 compact MLEM 1-iteration 双模式输出；
- 输出 shape 均为 `108x108x296` 且 finite；
- Release 聚焦测试 33/33、主测试 279/279；
- CUDA scaffold test executable 27/27。

仍未确认或不应外推：

- output-only hard-cut 没有执行 20/50 iteration；1 iteration 已出现明确负向 seam 证据，因此未继续昂贵运行；
- S 数据旧的精确输入合同缺失，本次双模式 gate 未包含 S；
- 当前无绝对 activity/sensitivity 真值，结论属于同源相对证据；
- standard CSR 路径没有完成与 compact 相同范围的双模式成像验证；
- CMake 4.3 的 CUDA scaffold gtest discovery 后处理仍有既知错误，但测试 executable 本身通过。

## 18. 变更或审查清单

修改多床位、overlap 或 seam 相关代码时，至少检查：

1. `-dfList` 排序后的床号是否与投影内容一致。
2. `bed_overlap` 固定槽是否对应实际相邻床位。
3. `outputBedOverlaps` 与 `bedOverlaps` 是否符合开关语义。
4. `bedZStarts`、迭代 `globalNz`、最终输出 `Nz` 是否同时正确。
5. forward 提取、denom 累加、numerator 累加是否使用同一个 Z/flip 映射。
6. PSF 是 global scope 还是 per-bed scope；是否与 manifest 一致。
7. numerator 和 denom 是否成对应用任何新权重或校正。
8. 输出阶段是否发生重复 per-bed flip、重复 FOV 或二次 seam 处理。
9. image 与 mask 是否使用同一 hard-cut/flip。
10. denom cache 是否因 geometry、PSF、flip、CTAC 或 overlap 变化而失效。
11. QA 是否保留共享固定窗、FOV-only/hard baseline、seam zoom、非接缝和主体保持指标。
12. 是否在 mouse、S、zimu 上使用同一参数；若某数据合同缺失，应明确标为未验证。

## 19. 固定 CT 与可变多床 SPECT

配对 DICOM 的物理坐标用于建立初始映射，不等价于已经完成精配准。当前 mouse、S、capillary 的 CT 与对应设备 SPECT 具有相同 `FrameOfReferenceUID`、HFP 和轴方向。`scripts/prepare_ct_mu_map.py` 使用完整 patient-space affine：由 `ImageOrientationPatient` 构造行/列方向与法向量，按 `dot(ImagePositionPatient, normal)` 排序并从实际 IPP 估计层间距；若输入包含 `BedN` DICOM 子目录，则优先使用每床首层 IPP，`bed_step_mm` 只作一致性检查。

CT 保持固定，整段多床 SPECT 只允许一个全局残差变换；当前只验证纯物理映射和全局平移，旋转必须通过留出解剖锚点验证，禁止逐床独立配准，否则会在床间引入人为位移。80 层、0.3 mm、21 mm 步进时：

```text
overlap = 80 - 21/0.3 = 10 voxel
BedN origin = Bed1 origin + (N-1)*21 mm
global Nz = 80*N - 10*(N-1)
```

因此 Bed1、Bed1-2、Bed1-5 分别对应 80、150、360 层；Bed2-4 对应 220 层且世界坐标起点相对 Bed1 前移21 mm。床号必须是1至5的连续子集。

脚本 `scripts/prepare_ct_mu_map.py` 已实现上述初始重采样并记录 sidecar manifest。当前 HU-to-mu 仅为 provisional `0.0152*clip(1+HU/1000,0,3) mm^-1`；在扫描仪/能量标定和全局配准 QA 通过前，不得把生成图用于定量结论。

CT 的主线用途包括配准后的 `-ctac -attMap`，以及可选的 TV/RDP 边缘权重。显式 `-regCT` 优先；未提供时，TV/RDP 可复用同一 `-attMap`；都未提供则保持 activity-only。CT 权重只调节相邻体素的正则耦合，不改变系统矩阵、likelihood 或 sensitivity denom。旧 Bowsher/PLS 等入口继续禁用。验证顺序固定为 MLEM no-CTAC、MLEM+CTAC、BSREM+RDP+CTAC，且三者使用同一 overlap10 几何和同一全局变换。
