# SPECT_recon RC 参数说明

本文档面向本发布包根目录执行：

```powershell
cd D:\path\to\<package_root>
```


## 默认识别规则

正常运行不需要传 `-config`。程序会优先按 exe 所在目录自动识别：

1. `.\bin\config\spect_config.ini`（全局配置）。
2. `[Reconstruction] default_config_<核素>` 或 `default_config` 选择的一份 `profiles/*.conf`。
3. 仅未配置默认方案的旧全局 INI 才按优化器选择 `algo_<alg>.conf`。

配置采用全局 INI＋一份重建方案：`[Reconstruction] default_config` 指定默认方案，`-algoConf` 显式替换；`-opti` 不再另换方案。`recon_sections_override_cli=false` 时命令行 > 方案 > 全局；为 `true` 时方案 > 全局 > 命令行，仅保护配置中已填写的重建字段，输入/输出、设备和日志仍由调用方指定。方案可覆盖 Algorithm/BSREM/Regularization/PSF/Mask/Filtering/SystemMatrix/Normalization 等完整参数。`-config` 若直接指向带 `[Algorithm]` 的方案文件，也会内部按显式方案处理并继承部署全局默认值。`-subset-mode`、`-subset-diagnostics`、`-iter` / `-iterations` / `-verbose` / `-verbosity` 及`-v <n>`会失败；其他未知参数打印`[NOTE]`并忽略。单独的`-v` / `--version`只打印版本后退出。

只有在明确要覆盖发布包默认配置时才传：

```powershell
-config D:\custom\spect_config.ini
-algoConf D:\custom\algo_mlem.conf
```

系统矩阵按合并配置解析：`[SystemMatrix] root` 相对 EXE，`matrix_id/matrix_id2` 精确匹配 `root/id` 或 `root/isotope/id`，缺失或歧义均失败。显式 `-chunkCSRDir/-chunkCSRDir2` 服从上述优先级；旧配置无 ID 时才使用 `default_chunk_csr_dir` 或唯一目录发现。不要为多个核素保留一个无身份的默认矩阵。

denom cache 也服从配置优先级。配置的 `compact_denom_cache_path` 为固定文件，否则在 `compact_denom_cache_dir` 下按 manifest hash 选择 `auto_<hash>.mhd`；配置相对路径按 EXE 解析，默认 `.\bin\cache\compact_denom\`。CLI 显式相对缓存路径按 CWD 解析。身份绑定实际两矩阵、能窗权重、norm/行权重、PSF、Mask、床位及子集；改迭代数、β、仅输出滤波不要求清缓存。

## 默认 Tc OSEM 与基础 MLEM

先按所选方案的 matrix_id 放置矩阵。默认 Tc OSEM 为50×10；以下显式演示全数据 MLEM，需确认优先级允许 CLI 覆盖：

```powershell
.\bin\SPECT_recon.exe `
  -isotope Tc99m -df D:\data\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -o recon_output `
  -it 50:1 -save-every 5
```

说明：
- `-opti mlem` 默认使用10个角度子集（OSEM），BSREM共用这套划分。`-it A:1`恢复完整数据更新；子集数必须整除矩阵的compact角度数。同格式双窗CSRB及实验CSRQ均有N10和子集CTAC路径；CSRQ N>1 CTAC为2026-10-09工作树候选，尚待GPU数值回归、未发布。旧POSEM像素划分已退出当前接口。
- `-config` 不需要显式传。
- 多矩阵用所选方案的精确 `matrix_id/matrix_id2`；CLI覆盖受优先级开关控制。
- 投影路径仍需要按实际数据传入。

## 多床位

下述固定裁剪/ROI 是 PSC 默认。源码另支持 `[InputOutput] matrix_grid_output=true`：完整每床网格，最多 16 床，2026-10-11 起多床为联合全局体重建（重叠体素共同更新，不做 cosine 融合、不做床间 gain-fit，暂不支持 CTAC/AutoMu/`-maskFromAtt`）；多床必须显式给出正且整数体素层对齐的床步进。床多时可用 `-df <目录> -beds N`（或 `A:B`）代替逐个列出 `-dfList`。SC60 recipe 的 `bed_step_mm=0` 为未知，不能当作 PSC 16.8 mm；当前只有合成多床验证，部署须确认包内 EXE 是否包含该候选。完整边界见 `Format_Reader_Matrix_CN.md`，不变更已有发布默认。

```powershell
.\bin\SPECT_recon.exe `
  -dfList "D:\scan\Bed1.bin;D:\scan\Bed2.bin;D:\scan\Bed3.bin;D:\scan\Bed4.bin" `
  -o recon_multibed `
  -it 50:1 -save-every 50
```

固定合同：
- PSC 最多 5 个床位（`matrix_grid_output=true` 最多 16）；路径含 `BedN` 时按床号排序，且必须是连续子集。`Bed1` 是最小 physical-Z 床，床号每增加 1 向较大 physical Z 移动 `16.8 mm`。少床不补虚拟床。
- `-bedOverlap` 表达重建空间的重叠层数；`-bedStepMm <mm>`（INI `[MultiBed] bed_step_mm`，默认 `16.8`）声明扫描仪机械步进。输出步进由 `step = round(bed_step_mm / voxel_z)` 自动确定，**overlap 是派生值**：`overlap = baseNz - step`。不传 `-bedOverlap` 时自动推算；传错时自动纠正并打印 `[NOTE]`（`bed_step_mm` 是权威输入）。220 层 0.3 mm 床位为 `overlap = 220 - 56 = 164`（16.8 mm 步进）；其他步进/几何按同一关系换算。
- `[MultiBed] z_bed_flip=1` 是发布默认，只定义每床 local system-matrix Z 与 reconstruction-storage Z 的关系；`-attMap` 使用前者，`-regCT` 使用后者。不要通过逆序床号或整幅 Z 翻转二次补偿。
- 保存阶段只保留一条路径：每床中心 `step + 10` 层（260601 为 66 层）-> prototype log-ratio median/MAD gain-fit（Bed1 锚定，始终应用原型 log-ratio 链（无可靠性门））-> `step` 层步进 -> 10 层归一化 cosine 融合（沿输出 Z，入口保持上一床；`z_bed_flip` 只反转存储与增益窗）。
- 输出尺寸为 `100×100×[crop + (床数 - 1)×step]`（260601 为 `[66 + (床数 - 1)×56]`）。单床ROI为 `30 mm × 21.6 mm`，多床ROI为 `36 mm × 66 mm`；最终裁剪还受 `single_bed_step_aligned` 影响，尺寸以实际MHD为准，不固定假设72层。
- `sigmoid`、`direct`、对象感知 seam/profile 修补、输出 FOV mask 和离线自动拼接均已删除。

## 输出方向

默认配置：

```ini
[MultiBed]
bed_overlap = 164,164,164,164,0
crop_end    = 0,0
z_bed_flip  = 1
z_out_flip  = 0
x_out_flip  = 0
xy_out_rotation_deg = 0
seam_xy_align = 0
```

命令行只在确有坐标合同要求时覆盖。沿用现有优先级：若 `recon_sections_override_cli=true`，配置中已写出的同名设置优先，应修改选中配置；需 CLI 优先时设为 `false`：

```powershell
-zbedflip 1
-zoutflip 0
-xoutflip 0
-xyoutrot 90
-cropEnd top,end
```

保存阶段顺序为：每床 Z 翻转（适用时）-> 整体 Z/X 翻转 -> 拼床 -> 中心裁剪 -> `cropEnd` 端层置零及掩膜 -> XY 旋转 -> 输出滤波。新增 XY 旋转及整体 Z/X 翻转只影响保存输出，不改变迭代状态、系统矩阵或两类 CT 输入地图；`z_bed_flip` 仍遵循前述 local-Z/storage-Z 合同。

`xy_out_rotation_deg` / `-xyoutrot` 支持任意有限角度（度），默认 `0` 不旋转；正值将 `+X` 转向 `+Y`，在 `+X` 向右、`+Y` 向上、从 `+Z` 看向原点的视图中为逆时针（CCW），负值为顺时针（CW）。若查看器把递增 Y 行向下显示，视觉方向相反，不能仅凭屏幕判断。示例：`90` 为 CCW 90°，`-90` 为 CW 90°。

旋转围绕最终输出 XY 网格中心，按毫米坐标逐层双线性重采样，尺寸和体素间距不变，越界源采样补零，旋出视野的内容会丢失；任意角度可能改变峰值及总和。旧 `x_out_flip=0/1` 和 `-xoutflip` 完全保留，含义仍是 X 镜像，不是旋转；二者同时启用时先镜像后旋转。primary 和输出滤波副本使用同一旋转结果，run manifest 记录角度。此功能实际旋转像素，不自动修改 MHD/DICOM 患者方向元数据；CT 融合前需保持双方坐标一致。


## 常用覆盖参数

- `-it A[:B]`：A次完整迭代、B个子集，例如`100:10`；保存频率独立为`-save-every N`，默认0仅末次。分阶段使用`-it A -subsets N1:i1,N2:i2,...`，不能同时传`-it A:B`和`-subsets`。
- `-o <dir>` 或 `-fout <dir/name>`：输出目录或输出前缀。
- `-chunkCSRDir <dir>`：显式指定 compact chunk 矩阵目录；只有多矩阵或外部矩阵目录时通常需要。
- `-compactDenomCache <mhd>`：显式指定 denom cache，受配置优先级控制。
- `-psf [fwhm_mm[,N]]`：启用 image-domain PSF；发布默认 `1.3,3`、模式 `sieve`、`psfKernel=castor`。`-psfKernel adjoint` 使用采样 Gaussian + 镜像边界 + 真伴随；`-psfKernel castor` 使用 erf 体素积分核 + 零边界 + Z-only 归一化，反向复用正向。
- 发布 `[Filtering]` 默认 `smooth_apply_stage=none`、`butter_apply_stage=output`：保留primary，并从同一primary独立写出 `_butter`（5 cycles/cm、order 8）。显式启用输出Gaussian时才另写`_smooth`（CASToR 0.30 mm/3σ）。`output_nonnegative_clamp=true` 在每个副本滤波后钳为非负；设为 `false` 仅用于原型线性旁瓣对齐。
- `-detZRowWeightsFile <file>`：可选探测器效率 e，直接进入 compact sensitivity，有背景时内部使用 b/e。不是校正因子 c；不与 `-normFactorFile` 共用。发布默认为空，只有独立验证对应效率后才传入。
- `-opti mlem|bsrem`：切换优化器，已有方案时不另换方案；`-it A`仅改次数，保留继承的子集。
- 正则化由所选方案决定，不能以为`-opti bsrem`会自动开启RDP。TV方案为02、Lu合并RDP为03、Lu joint RDP为05；用实际调用参数加`-dumpConf`确认。每个子更新使用当前图像重新计算正则梯度，数据梯度乘当前子集数，β不乘子集数。输出Butterworth仅作用于保存副本。
- `-bProj <path>` / `-bProjList <list>`：加性背景/散射投影项。

性能相关环境变量不属于重建数值合同：`SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` 默认按矩阵总量和启动时可用内存自动选择，也可显式覆盖；branch/bed 合批（packed-K 正向 + CSR_ALG2 转置）为正式路径。`SPECT_COMPACT_BRANCH_BATCH` 和 `SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES` 已退役，设置会报错；旧实验由冻结 EXE 复现。16 GB 显存的低时延实测使用 64 GB 主存与 SSD/NVMe 矩阵，32 GB 主存通常只能缓存部分矩阵，性能需重新以交错重复运行的中位数签收。

## 查看合并配置

```powershell
.\bin\SPECT_recon.exe -dumpConf
```

不带核素默认Tc；检查Lu使用 `-isotope Lu177 -dumpConf`。输出来源应显示类似：

```text
# global_ini = ...\bin\config\spect_config.ini (loaded)
# algo_conf  = ...\bin\config\profiles\standard_Tc99m_OSEM.conf
```

## 核素、矩阵及归一化配置（2026-09-21）

`-isotope Tc99m|Lu177` 是可选的采集身份断言；部署 wrapper 自动从预处理记录传入。全局配置用 `default_config_Tc99m` / `default_config_Lu177` 选择核素默认方案。已有显式 `-algoConf` 始终替换默认方案，方案核素与采集不符时失败。

`[SystemMatrix] root` 相对 EXE，`matrix_id` / `matrix_id2` 精确查找 `root/id` 或 `root/isotope/id`；找不到或同时存在两份均失败。旧 `-chunkCSRDir` / `-chunkCSRDir2` 继续支持；配置优先模式下以配置为准。几何仍由矩阵 metadata 检查。

Lu-177 joint 固定本批验证的通道顺序 **208、113 keV**，`energy_window_weights=0.1036,0.0617`。权重进入预测计数一次，并非人为给某张图加权平均，也不能凭权重大小认定某窗的信息量占比。

`[Normalization] factor_file` / `factor_file2` 分别绑定这两个通道的投影校正因子 c；程序使用 1/c 构造灵敏度。Tc-99m 单窗只用第一个文件。joint 要么两份都指定，要么两份均关闭，不继承另一个核素或合并窗的 norm。相对路径以 EXE 为基准。当前方案 norm 留空，因为还没有验证并交付本批分窗标定文件；不代表两核素 norm 相同。CSRB/CSRQ 均允许同时提供 `-bProj/-bProjList`（第二窗用对应的 `2` 参数）：b 必须是同床/能窗/采集时间下的预计探测计数，不提前乘 c 或减出投影。程序内部一次性计算 b/e，残差仍使用原始 b，模型与数值下限见 [矩阵合同](Format_Reader_Matrix_CN.md)。

已有 denom 缓存同时绑定两矩阵、能窗权重、norm 数值、Mask/PSF 和子集；改迭代数、β 或仅输出滤波，不另造缓存层。

部署 wrapper 原来的 `--preprocess-root`、`--output-root`、`--config` 调用不变；`--config` 可传全局 INI 或完整方案，后者由内部转换为 `-algoConf`。可在 `--` 后追加 `-it 40:10` 等参数，其优先级遵守上述开关。日志中的配置来自实际重建 argv 加 `-dumpConf`。

Lu-177 正式重建方案读取合并 `BedN.bin` 与单份系统矩阵；预处理仍可保存 `BedN_high.bin` / `BedN_low.bin` 辅助数据。历史joint方案不能用合并BIN冒充分窗输入。未来加权合并矩阵是模型更新，不能仅按相同profile名称继承旧正则参数的验证结论。

## 方案表与部署接口

| 方案文件（config/profiles） | 模型 | 算法/正则化 | 迭代×子集 | BSREM α₀/γ |
|---|---|---|---|---|
| resolution_Tc99m_BSREM.conf | Tc单141 SCALE2 CSRB | BSREM无正则 | 100×10 | 1.5/0 |
| uniformity_Tc99m_BSREM_TV.conf | Tc单141 SCALE2 CSRB | BSREM TV β=0.0003 | 75×10 | 1/0.05 |
| mouse_Lu177_BSREM_RDP.conf | Lu合并窗＋单208 CSRB | BSREM RDP β=0.1 | 100×10 | 1/0.05 |
| standard_Tc99m_OSEM.conf | Tc单141 SCALE2 CSRB；Tc默认 | OSEM无正则 | 50×10 | — |
| standard_Lu177_OSEM.conf | Lu合并窗＋单208 CSRB；Lu默认 | OSEM无正则 | 50×10 | — |
| sc60_osem1000.conf | SC60单窗CSRB、完整矩阵网格；显式兼容验证方案 | OSEM无正则，PSF1.5 mm | 1000×10，每50轮保存 | — |

正式命名为 `用途[_应用][_核素]_算法[_正则].conf`，不含编号、迭代次数、joint或detail。核素专用配方保留核素；mouse省略应用时指全身取向，bone/tumor仅用于独立验证的专用参数。通用名称仅用于不绑定核素的配方。历史分窗配方位于 `config/historical/05_mouse_joint_rdp30.conf`；SC60长迭代验证方案不随SPECT_Core发布。

这些方案适用于原实验范围，不是所有系统或IQ标定下的全局最优。两种Lu模型的β不直接互换。源配置默认false；2026-09-14的E盘6Module部署保留true。6Module部署EXE位于 `bin/recon/`，独立重建RC位于 `bin/`，两者的相对配置规则相同。详细上位机迁移见 `Recon_Interface_Migration_Field_CN.md`（源码文档），部署包见其wrapper指南。
