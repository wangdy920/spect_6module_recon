# SC60 CSRQ 转换与 W 模体输出角度验证

## 范围与结论边界

用户要求：转换完整 SC60 系统矩阵为 CSRQ，测量存储占用、估算全显存驻留需要，并用 W_logo DICOM 判断 `xy_out_rotation_deg`。

- 原始 CSRB、metadata、DICOM 不改；CSRQ 另存。仅补离线工具的 SMMD v3 读取与必要输入校验，不扩展 CUDA profile、不修改设备默认值、不替换 SPECT_Core。
- 文件转换终点：全部 18 块、量化误差统计、独立结构校验和确定性抽样解码。矩阵量化是有损派生，不以文件误差代替重建图像质量验证。
- 配准终点：同一 W 模体的 Z 轴旋转加三维平移、跨阈值敏感性检查和统一显示参数的叠加图。属于本次扫描的探索性适配建议，不是设备标定或临床配准精度认证。
- 源码基线为 `b206e14` 加当前未提交工作树；本次仅修改转换工具及相关文档。上一轮 CUDA/输出旋转改动均保留。

## 转换支持

源目录：`D:/Projects/SPECT-recon/test_data/SC60_luzhou/matrix/matrix_3module_18pin_YSO_90x90x240_141keV`。

目标目录：同级 `matrix_3module_18pin_YSO_90x90x240_141keV_CSRQ`。

实际 metadata：SMMD v3，124 B，90×90×240、0.4 mm，1,944,000 个体素；90 compact angles、360 full angles、96×96 detector rows，829,440 compact rows；`ROT90`，**没有 Z 对称**。

修改前，`convert_csrb_to_csrq.py` 只接受 88/92 B metadata，不能直接转换该目录。现在可读取 124 B 的体素数和尺寸，核验 magic/version、尺寸乘积和列范围，metadata.bin 原样复制；同时明确拒绝非有限/负权值、错误行指针和越界列。

编码不变：每行最大值 float32，权值 `q=round(sqrt(v/row_max)*65535)` 的 uint16 编码，连续列打包为最长 64 项的 run。不是 FP16，也没有增加 Z-fold 或裁剪 ROI。

**转换支持不等于重建支持。** `CompactChunkCSR.cpp` 当前明确拒绝 CSRQ + ROT90；现有 CSRQ CUDA 计算使用 ROT90_Z 的 8 分支。不能把 `use_z_symmetry` 改成 1 或删除入口检查来运行。当前 SC60 重建仍应使用原生 CSRB。

实际新 EXE 入口探测退出码 2，错误为 `CSRQ currently requires the ROT90_Z profile.`；见实验目录 `runtime_probe.log`。未执行迭代，没有把拒绝误报为可运行。

### 实测存储与量化

| 项目 | 实测 |
|---|---:|
| 原始 18 个 CSRB 文件 | 67,106,070,664 B = 62.4974 GiB |
| 派生 18 个 CSRQ 文件 | 19,680,355,708 B = 18.3288 GiB |
| 减少的磁盘体积 | 70.673%，约 3.410 倍压缩 |
| 非零项记录数（保留原索引项） | 8,387,429,312 |
| 量化为零的原正权值 | 265,839，占 0.0031695% |
| 每块全权值量化 relL2 范围 | 1.352311e-5 .. 1.352896e-5 |
| 转换耗时 | 828.4 s，一次运行，非性能优化对照 |

文件大小不含小型 metadata/验证说明文件。CSRQ 保留项数，但极小值可量化成 0，不能称为“非零值完全保留”。上述 relL2 是矩阵权值误差，不是重建图像误差，不能直接套用图像 relL2 的采用门限。

校验已完成：转换器自检覆盖旧 88/92 B 及 SMMD v3 的体素数读取、非法 metadata magic、非法权值、排序/run 拆分、量化解码；18 块独立验证全部行指针、run 长度和列边界，每块确定性抽取 66 行（总计 1,188 行）确认列索引与源矩阵排序后完全一致，抽样权值误差符合每行最大值的 2e-5 上界。metadata.bin 与源文件逐字节一致。这里没有宣称全部 83.9 亿列项逐项解码比较。

### 显存预算

CSRQ 矩阵设备数组公式：`12*rows + 8 + 2*nnz + 4*runs` B/块（两组 uint32 行指针、float32 行最大值、uint16 权值、uint32 runs）。18 块合计 **19,680,354,824 B = 18.3288 GiB**，不含头和文件对齐填充。

当前调度器采用 `matrixBytes + scratchBytes + 2 GiB reserve <= freeBytes` 的完整驻留判断。按既有 K=8 工作区公式推算为 **20.4693 GiB 可用显存**；将来若 ROT90 正式改为 K=4，对应同样公式约 **20.3991 GiB**。这是规划值，不是未实现内核的显存实测；PSF、其他任务和显示占用还需保留实际可用空间。

因此，**16 GB 无法完整驻留；建议至少 24 GB 级显卡，且运行时可用显存需满足预算**。当前机器 RTX 4070 Ti SUPER 总显存约 16 GiB，不能做全驻留实测。即使换 24 GB 卡，仍须先实现并验证 CSRQ ROT90 执行路径；显存容量不能消除格式/算子限制。当前生产继续使用 CSRB 分块执行。

## W 模体配准

输入：`F:/DATA/SC60_luzhou/W_logo_DICOM/20260924_181915`。用户确认当前没有额外翻转（“应该”），本建议按旧 `x_out_flip=0`、当前旋转为 0、导出不改 XY 索引处理；没有对应原始 MHD/manifest 可独立核验导出路径。

| 项目 | SPECT | CT |
|---|---|---|
| DICOM 数量 | 346 | 1024 |
| 原始尺寸 XYZ | 90×90×346 | 1024×1024×1024 |
| 间距 XYZ | 0.4×0.4×0.4 mm | 0.1×0.1×0.1 mm |
| IOP | 1,0,0,0,1,0 | 1,0,0,0,1,0 |
| 按物理 Z 排序后首体素位置 | 47.48,41.35,158.589 mm | 0.7,-1.2,177.75 mm |

两个序列 FrameOfReferenceUID 相同，但实际 W 结构并未对齐。以 ImagePositionPatient 排序、逐片核验方向和间距，应用 RescaleSlope/Intercept；不按文件名猜 Z 方向。SPECT 存储标签为 PT，但本次仅分析几何，不据此验证定量单位。

### 方法

- CT 每轴取间隔 4 的样本，形成 0.4 mm 网格用于本次快速分析；未将这种采样当作精密标定。
- 根据 CT 实际可见的 W 槽，在 DICOM 坐标 X=42..62、Y=42.4..44.4、Z=211..232 mm 中取 HU<200 的最大连通区域。该 ROI 是本例的人工观察选择，不是通用自动分割规则。
- SPECT 先固定 15% 最大值的 W 连通支持区，再分别取其中 15%、25%、40% 阈值的全部信号，防止高阈值把 W 分成多段后误丢分支。
- 对 Z 轴旋转遍历 -180..178°（步长 2°），再联合细化角度与 XYZ 平移。不引入尺度、X/Y 轴旋转或形变。
- 指标为两点集对称最近邻平方距离均值的平方根，单位 mm。它只描述本次分割和配准拟合，**不是独立地标 TRE，也不是绝对配准精度**。

### 结果

| SPECT 阈值 | 旋转角度 | 拟合点集距离 |
|---|---:|---:|
| 15% | -57.949° | 0.373 mm |
| 25% | -58.568° | 0.267 mm |
| 40% | -58.728° | 0.339 mm |

同一 25% 信号支持下，各角度分别拟合平移：

| 角度 | 拟合点集距离 |
|---|---:|
| 0° | 2.416 mm |
| -60° | 0.269 mm |
| -58.568° | 0.267 mm |
| +120°（反向候选） | 1.443 mm |

建议本次先用 `xy_out_rotation_deg=-60`；自由拟合的 -58.6° 可用于细调对照，但相较 -60° 只改善约 0.002 mm 的分割拟合指标，不足以据此固化更精细的设备角度。三个阈值的范围不是统计置信区间。

方向与新代码一致：正角度把 +X 转向 +Y；当 +X 向右、+Y 向上时，-60° 为 CW 60°。若显示器把递增 Y 行向下显示，屏幕上的顺逆时针相反。

**单改角度不能完成配准。** 以当前 SPECT 输出中心为旋转中心，-60° 后仍需要本例 DICOM 坐标平移约 `(-13.459,-21.330,+2.784) mm`。该偏移包含当前 DICOM 原点约定，不能直接推广为其他扫描的机械偏移或床位步进；应在上游融合/空间映射层单独处理。变换公式为 `p_CT=Rz(theta)*(p_SPECT-c)+c+t`，`c` 为当前 SPECT 网格中心。

![统一窗宽的 CT/SPECT 对照](../recon_out/sc60_csrq_registration_20260928/registration_overlay.png)

三列均已分别拟合平移，目的是隔离旋转差异；不是“仅旋转后已自然重合”。灰度为 CT 切面（-1000..1500 HU），彩色为 SPECT 投影，统一使用原 SPECT 最大值的 50% 为上限，低于 10% 的信号透明。原始 DICOM 没有被重写。

初始诊断曾在每个阈值独立保留最大连通分量，40% 时只留下 W 的一段，产生 +122.5° 的错误局部匹配。原记录保留在 `registration_fit_largest_component.json`，**不作为角度依据**；修正后统一支持区的结果见 `registration_fit.json`。

## 复现与证据

```powershell
python tools/convert_csrb_to_csrq.py --self-check
python tools/convert_csrb_to_csrq.py test_data/SC60_luzhou/matrix/matrix_3module_18pin_YSO_90x90x240_141keV test_data/SC60_luzhou/matrix/matrix_3module_18pin_YSO_90x90x240_141keV_CSRQ --block-rows 128
python recon_out/sc60_csrq_registration_20260928/verify_conversion.py
python recon_out/sc60_csrq_registration_20260928/inspect_registration.py
python recon_out/sc60_csrq_registration_20260928/fit_rotation.py
python recon_out/sc60_csrq_registration_20260928/render_registration.py
```

目标目录已有结果时转换器拒绝覆盖；复现应指定新的目标目录，不覆盖本次证据。验证脚本路径常量对应本次原始及派生矩阵。

本次分析缓存 `volumes.npz` SHA256：`3a2c717a42073597376f1e8c8dd53c6d50ff847fef90472881e0bf2f24da168e`。`geometry.json` 保留序列 UID、逐层 Z 顺序和实际几何；原 DICOM 共 1,370 个文件、2,155,091,270 B。

配准脚本和图像为本次实验资产，非新的生产调用入口。转换器是正式维护工具，CSRQ 仍为实验派生格式；本次没有完成 SC60 CSRQ 图像重建验证，也没有宣称定量等价。
