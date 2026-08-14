# SPECT_recon RC 参数说明

本文档面向本发布包根目录执行：

```powershell
cd D:\path\to\<package_root>
```


## 默认识别规则

正常运行不需要传 `-config`。程序会优先按 exe 所在目录自动识别：

1. `.\bin\config\spect_config.ini`
2. `.\bin\config\algo_mlem.conf`
3. `.\bin\config\algo_bsrem.conf`

发布配置默认启用 `[Compatibility] recon_sections_override_cli=true`：
`spect_config.ini` 对 Regularization、PSF、Filtering 和 Postprocess（输出滤波及主输出覆盖策略）具有最终优先级；算法、迭代、几何、输入输出等其他正式 CLI 参数仍按 CLI 优先。未知或已退役 CLI 参数只打印 `[NOTE]` 并忽略，已识别参数的非法取值仍会失败。

只有在明确要覆盖发布包默认配置时才传：

```powershell
-config D:\custom\spect_config.ini
-algoConf D:\custom\algo_mlem.conf
```

系统矩阵默认识别规则：

1. 显式 `-chunkCSRDir <dir>` 优先。
2. 否则读取 `.\bin\config\spect_config.ini` 中 `[SystemMatrix] default_chunk_csr_dir`，相对路径按 `.\bin\` 解析。
3. 否则如果 `.\bin\matrix\` 下只有一个包含 `metadata.bin` 的矩阵目录，自动使用该目录。
4. 如果有多个矩阵目录且未指定，程序直接报错并列出候选目录。

denom cache 识别规则：显式 `-compactDenomCache <mhd>` 优先（CLI 相对路径按终端 CWD 解析）；否则使用 `[SystemMatrix] compact_denom_cache_path` 的固定文件；再否则在 `compact_denom_cache_dir` 下按 manifest hash 自动选择 `auto_<hash>.mhd`。两个 INI 路径均相对 `SPECT_recon.exe` 目录解析，发布默认目录为 `.\bin\cache\compact_denom\`；不同矩阵、PSF、床位几何或 normfactor 不会误复用。

## 推荐基础 MLEM

如果发布包中只有一个 `.\bin\matrix\<matrix_name>\metadata.bin`：

```powershell
.\bin\SPECT_recon.exe `
  -df D:\data\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -o recon_output `
  -it 50:5
```

说明：
- `-opti mlem` 可由 `bin/config/algo_mlem.conf` 提供，通常不用显式传；compact 布局恒单 subset，无 `-subsets` 参数。
- `-config` 不需要显式传。
- 多个矩阵目录时，在 `bin/config/spect_config.ini` 设置 `default_chunk_csr_dir = matrix/<matrix_name>`，或在命令行传 `-chunkCSRDir .\bin\matrix\<matrix_name>`。
- 投影路径仍需要按实际数据传入。

## 多床位

```powershell
.\bin\SPECT_recon.exe `
  -dfList "D:\scan\Bed1.bin;D:\scan\Bed2.bin;D:\scan\Bed3.bin;D:\scan\Bed4.bin" `
  -bedOverlap 164,164,164,0,0 `
  -o recon_multibed `
  -it 50:50
```

固定合同：
- 最多 5 个床位；路径含 `BedN` 时按床号排序，且必须是连续子集。`Bed1` 是最小 physical-Z 床，床号每增加 1 向较大 physical Z 移动 `16.8 mm`。少床不补虚拟床。
- `-bedOverlap` 表达重建空间的重叠层数；`-bedStepMm <mm>`（INI `[MultiBed] bed_step_mm`，默认 `16.8`）声明扫描仪机械步进。输出步进由 `step = round(bed_step_mm / voxel_z)` 自动确定，**overlap 是派生值**：`overlap = baseNz - step`。不传 `-bedOverlap` 时自动推算；传错时自动纠正并打印 `[NOTE]`（`bed_step_mm` 是权威输入）。220 层 0.3 mm 床位为 `overlap = 220 - 56 = 164`（16.8 mm 步进）；其他步进/几何按同一关系换算。
- `[MultiBed] z_bed_flip=1` 是发布默认，只定义每床 local system-matrix Z 与 reconstruction-storage Z 的关系；`-attMap` 使用前者，`-regCT` 使用后者。不要通过逆序床号或整幅 Z 翻转二次补偿。
- 保存阶段只保留一条路径：每床中心 `step + 10` 层（260601 为 66 层）-> prototype log-ratio median/MAD gain-fit（Bed1 锚定，始终应用原型 log-ratio 链（无可靠性门））-> `step` 层步进 -> 10 层归一化 cosine 融合（沿输出 Z，入口保持上一床；`z_bed_flip` 只反转存储与增益窗）。
- 输出尺寸为 `100×100×[crop + (床数 - 1)×step]`（260601 为 `[66 + (床数 - 1)×56]`）。单床固定输出为 `100×100×72`，ROI `30 mm × 21.6 mm`；多床 ROI 为 `36 mm × 66 mm`。
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
seam_xy_align = 0
```

命令行只在确有坐标合同要求时覆盖：

```powershell
-zbedflip 1
-zoutflip 0
-xoutflip 0
-cropEnd top,end
```

处理顺序为：`-attMap` local-Z / `-regCT` storage-Z 输入 -> compact 每床映射 -> canonical gain/stitch -> 整体输出 Z 翻转 -> 整体输出 X 翻转 -> `cropEnd top,end`。`z_out_flip` 只影响保存输出，不影响两类 CT 输入地图。


## 常用覆盖参数

- `-it N[:save]`：迭代次数与保存间隔，例如 `50:5`。
- `-o <dir>` 或 `-fout <dir/name>`：输出目录或输出前缀。
- `-chunkCSRDir <dir>`：显式指定 compact chunk 矩阵目录；只有多矩阵或外部矩阵目录时通常需要。
- `-compactDenomCache <mhd>`：显式指定一个 denom cache 文件，覆盖 INI 的固定路径和自动 hash 目录。
- `-psf [fwhm_mm[,N]]`：启用 image-domain PSF；发布默认 `1.3,3`、模式 `sieve`、`psfKernel=castor`。`-psfKernel adjoint` 使用采样 Gaussian + 镜像边界 + 真伴随；`-psfKernel castor` 使用 erf 体素积分核 + 零边界 + Z-only 归一化，反向复用正向。
- 发布 `[Filtering]` 默认 `smooth_apply_stage=output`、`butter_apply_stage=output`：保留未滤波主输出，并从同一主输出独立写出 `_smooth`（CASToR 0.30 mm/3σ）与 `_butter`（5 cycles/cm、order 8）。`output_nonnegative_clamp=true` 在每个副本滤波后钳为非负；设为 `false` 仅用于原型线性旁瓣对齐。
- `-detZRowWeightsFile <file>`：可选 norm W2 探测器效率权重，仅作用于 compact sensitivity denominator。发布默认为空/关闭；Mouse 默认保持 default 配置。只有明确验证 W2 候选时才传入。
- `-opti mlem|bsrem`：切换算法，并自动搜索 exe 同级 `config/algo_<alg>.conf`。
- `-bProj <path>` / `-bProjList <list>`：加性背景/散射投影项。

性能相关环境变量不属于重建数值合同：`SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` 默认按矩阵总量和启动时可用内存自动选择，也可显式覆盖；`SPECT_COMPACT_BRANCH_BATCH` 默认开启，设为 `0` 可运行逐分支基线；mmap 默认关闭。16 GB 显存的低时延实测使用 64 GB 主存与 SSD/NVMe 矩阵，32 GB 主存通常只能缓存部分矩阵，性能需重新以交错重复运行的中位数签收。

## 查看合并配置

```powershell
.\bin\SPECT_recon.exe -dumpConf
```

输出应显示：

```text
# global_ini = ...\bin\config\spect_config.ini (loaded)
# algo_conf  = ...\bin\config\algo_mlem.conf
```
