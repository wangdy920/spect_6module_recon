# SPECT_recon RC 参数说明

本文档面向本发布包根目录执行：

```powershell
cd D:\path\to\<package_root>
```

已废弃或改名的参数见 `CLI_CHANGES_CN.md`。迁移旧脚本时先看那份短表。

## 默认识别规则

正常运行不需要传 `-config`。程序会优先按 exe 所在目录自动识别：

1. `.\bin\config\spect_config.ini`
2. `.\bin\config\algo_mlem.conf`
3. `.\bin\config\algo_bsrem.conf`
4. `.\bin\config\algo_admm.conf`

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

## 推荐基础 MLEM

如果发布包中只有一个 `.\bin\matrix\<matrix_name>\metadata.bin`：

```powershell
.\bin\SPECT_recon.exe `
  -df D:\data\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -o recon_output `
  -it 50:5
```

说明：
- `-opti mlem`、`-subsets 1` 可由 `bin/config/algo_mlem.conf` 提供，通常不用显式传。
- `-config` 不需要显式传。
- 多个矩阵目录时，在 `bin/config/spect_config.ini` 设置 `default_chunk_csr_dir = matrix/<matrix_name>`，或在命令行传 `-chunkCSRDir .\bin\matrix\<matrix_name>`。
- 投影路径仍需要按实际数据传入。

## 多床位

```powershell
.\bin\SPECT_recon.exe `
  -dfList "D:\scan\Bed1.bin;D:\scan\Bed2.bin;D:\scan\Bed3.bin;D:\scan\Bed4.bin" `
  -bedOverlap 16,16,16,0,0 `
  -o recon_multibed `
  -it 50:5
```

多床位规则：
- 最多 5 个床位。
- `-dfList` 默认按路径中的 `BedN` 排序。
- `-bProjList` 使用同样排序规则。
- `-bedOverlap b1,b2,b3,b4,b5` 是固定五槽表：
  - `b1 = Bed1-Bed2`
  - `b2 = Bed2-Bed3`
  - `b3 = Bed3-Bed4`
  - `b4 = Bed4-Bed5`
  - `b5` 保留占位
- 如果实际传入 `Bed2,Bed3,Bed4`，使用 `b2,b3`。
- 如果只传 `-bedOverlap 16`，会复制到所有固定槽位。

## 输出方向

默认配置：

```ini
[MultiBed]
bed_overlap = 16,16,16,0,0
crop_end    = 0,0
z_bed_flip  = 0
z_out_flip  = 0
x_out_flip  = 0
```

命令行覆盖：

```powershell
-zbedflip 0
-zoutflip 1
-xoutflip 1
-cropEnd top,end
```

处理顺序固定为：每床 Z 翻转 -> 整体输出 Z 翻转 -> 整体输出 X 翻转 -> `cropEnd top,end`。

矩阵入口与预处理投影变体必须配对：

| 矩阵入口 | `PROJECTION_VARIANT` | `z_bed_flip` / `-zbedflip` |
|---|---|---|
| `-chunkCSRDir`（推荐） | `identity` | 保留本包 `[MultiBed]` 默认，勿擅自改 |
| `-loadCSR` + `-compactMeta`（单文件 legacy） | `xflip` / `flipx` | **必须为 `1`（每床翻转）** |

## 常用覆盖参数

- `-it N[:save]`：迭代次数与保存间隔，例如 `50:5`。
- `-o <dir>` 或 `-fout <dir/name>`：输出目录或输出前缀。
- `-chunkCSRDir <dir>`：显式指定 compact chunk 矩阵目录；只有多矩阵或外部矩阵目录时通常需要。
- `-psf [fwhm_mm[,N]]`：启用 image-domain Operator PSF；FWHM 单位为 mm，默认 `1.5,4`。
- `-opti mlem|bsrem|admm`：切换算法，并自动搜索 exe 同级 `config/algo_<alg>.conf`。
- `-bProj <path>` / `-bProjList <list>`：加性背景/散射投影项。

## 查看合并配置

```powershell
.\bin\SPECT_recon.exe -dumpConf
```

输出应显示：

```text
# global_ini = ...\bin\config\spect_config.ini (loaded)
# algo_conf  = ...\bin\config\algo_mlem.conf
```
