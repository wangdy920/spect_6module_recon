# tools/ 独立工具

本目录存放独立命令行工具，**不参与** `SPECT_recon.exe` 主程序链接。C++ 工具随主 CMake 构建；Python 脚本手动调用。

## 工具清单

| 源文件 | 可执行名 | 用途 | 构建方式 |
|---|---|---|---|
| `forward_project_multibed.cpp` | `spect_forward_project.exe` | 匹配模型多床位前投影：把已知连续 phantom 经同一 compact-symmetric + PSF + 虚拟 Z 翻转链前投影为 `BedN.bin`，用于无噪 matched-model 闭环验证 | 主 CMake 构建 |
| `collect_device_recon_diagnostics.ps1` | `bin/tools/collect_device_recon_diagnostics.ps1` | 串行启动 1–9 次设备重建，把进程树/工作集/I/O、GPU、可用内存、磁盘总线及实时负载、cache、逐次计时/中位数和完整日志汇总为一个文本文件 | 随安装包复制 |
| `prep_compact_denom_cache.py` | `python tools/prep_compact_denom_cache.py` | 按默认 `spect_config.ini` 手动预热无 CTAC 的 1–5 床 `auto_<hash>.mhd` denom cache | 终端手动调用，不编译 |
| `search_compact_perf.py` | `python tools/search_compact_perf.py` | 设备空闲时手动搜索 host cache / mmap / 线程 / branch-batch；中位数墙钟 + 同输入 relL2 `1e-5` | 终端手动调用，不编译 |
| `run_220_mlem_perf_gate.py` | `python tools/run_220_mlem_perf_gate.py` | Lu-177 208 keV CSRB、正常小鼠五床 MLEM50：核心目标 120 s / 下限 180 s，denom hit，relL2 `1e-5`；2026-09-05 替代 Sampling 门，既有双窗合并计数仅用于性能／一致性 | 终端手动调用，不编译 |
| `convert_csrb_to_csrq.py` | `python tools/convert_csrb_to_csrq.py` | 离线把 CSRB 矩阵目录转成 GPU 常驻的 CSRQ 目录（列排序 + run-length 索引 + u16 sqrt 压扩值，约 2.2 B/nnz）；`--self-check` 自检写入/解码 | 终端手动调用，不编译 |
| `convert_csrb_values.py` | `python tools/convert_csrb_values.py` | 实验用：把 CSRB 目录做 quantize→dequantize 复制（`u16_sqrt`/`u16_row`/`fp16`/`fp24`/`identity`，可选 `--sort-cols`），用未改动的 exe 做端到端值格式验证 | 终端手动调用，不编译 |
| `analyze_csrb_structure.py` | `python tools/analyze_csrb_structure.py` | 采样 CSRB 行统计 run 长度、列增量、值动态范围与各量化方案对 PSF 形状（RMS 半径/质心）的影响 | 终端手动调用，不编译 |
| `analyze_csrq_footprint_groups.py` | `python tools/analyze_csrq_footprint_groups.py <csrq-dir> --angle-slices 3` | 在 CSRQ 块上量化 backward 内核分组方案：相邻探测器行分组 vs (angle, z 切片) 分组各自能减少多少全局原子加、tile 的 smem 需求 | 终端手动调用，不编译 |

## 用法示例

### CSRB → CSRQ 离线转换

```powershell
python tools\convert_csrb_to_csrq.py --self-check
python tools\convert_csrb_to_csrq.py <csrb-dir> <csrb-dir>_csrq
```

逐块（默认 1024 行一批，峰值内存约 2 GB）读源 CSRB，输出 `chunk_NNN.csrq` 并复制 `metadata.bin`/`metadata.txt`，在 `metadata.txt` 追加每块 run 数、拆分数、被冲零 nnz 数与值 relL2。Tc99m 120×120×220 六块约 6 min，24.3 GiB → 6.74 GiB。得到的目录直接作为 `-chunkCSRDir`；CSRB 仍是规范源，CSRQ 是派生品。


### 匹配模型前投影

```powershell
.\build\bin\Release\spect_forward_project.exe `
  --phantom phantom.json `
  --matrix-dir bin\matrix\260601 `
  --out-dir recon_out\matched_fwd `
  --beds 4 --bed-overlap 16,16,16,0,0
```

产出的 `Bed1.bin … BedN.bin` 可直接喂给 `SPECT_recon -dfList`，形成无噪 matched-model 闭环，用于分离实现缺陷与真实数据的未建模响应。

### 无 CTAC denom 预热

矩阵或默认 PSF/圆柱/坏探头变更后，在仓库根目录手动跑一次（每种床数会预载矩阵，耗时；不要加 `-ctac`）：

```powershell
python tools\prep_compact_denom_cache.py --chunkCSRDir <matrix-dir>
```

脚本写入占位 `Bed1.bin`…`Bed5.bin`（几何匹配、填充 1），以 `-it 1` 调用当前 `SPECT_recon.exe` 和 `multipinhole/config/spect_config.ini`。程序按配置把 v4 portable cache 写到 exe 相对的 `cache/compact_denom/auto_<hash>.mhd`；同一 matrix profile 的 source/package 根目录变化不会改变 identity。`--dry-run` 只打印命令；`--self-check` 只检查占位 BIN 与 metadata。

### 设备诊断与三次中位数

从部署包根目录启动；固定部署结构下，exe、配置、唯一的 `bin/matrix/<profile>` 和 exe 相对 denom cache 会自动解析：

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\tools\collect_device_recon_diagnostics.ps1 `
  -InputPath "D:\data\case\SPECT\Bed1.bin;D:\data\case\SPECT\Bed2.bin" -MultiBedList `
  -OutputPrefix "D:\SPECT_Result\device_test\recon" -Iterations 20:20 -Repeats 3 `
  -ReportPath "D:\SPECT_Result\device_test\spect_device_diagnostics.txt"
```

参数拼写是 `-Repeats 3`（不是 `-Repeates`）。三个重建按顺序执行，不会并发占用 GPU；每次输出加 `_run1`…`_run3`，原生重建日志仍留在目标输出目录。只需回传一个 `spect_device_diagnostics.txt`，其末尾 `REPEAT SUMMARY` 给出 `median_full_command_s`、逐次 `denom_cache`、resident prefix，以及 `all_cache_hits`；前文同时记录 free RAM、SPECT 进程工作集/累计读写量，以及 matrix/denom/output 所在磁盘的总线和可用空间。默认直接使用 `spect_config.ini` 的 `compact_denom_cache_path` / `compact_denom_cache_dir`，不再重复传 `-compactDenomCache`；仅隔离实验才显式使用 `-DenomCachePath`。`-Iterations N:save` 和 `-OutputPrefix` 会转换为 CLI `-it` / `-fout`，其迭代数、保存间隔和输出路径继续覆盖 INI。首次运行若包含 cache miss，待其建立 cache 后原样再执行一次，以第二份报告作为稳定 cache-hit 性能证据。必要时显式传 `-MatrixDir`；CLI 路径按调用终端的 CWD 解析。

### 设备空闲性能搜索

RTX 5060 等与签收机（16 GB / 64 GB / NVMe）不同的设备，在空闲时对**现有**环境变量做一次交错重复搜索。不改重建算法，不搜索已编译的 6 GiB 显存预留。先预热 denom cache，再用真实投影：

```powershell
python tools\search_compact_perf.py `
  --chunkCSRDir <matrix-dir> `
  --dfList Bed1.bin;Bed2.bin;Bed3.bin;Bed4.bin;Bed5.bin `
  --compactDenomCache <exe-dir>\cache\compact_denom\auto_<hash>.mhd `
  --it 3 --repeats 3
```

默认套件：`baseline`（unset 环境）、`host_cache_8g/16g/32g`、`mmap_0`、`threads_8`、`branch_batch_0`。决策量是进程墙钟中位数；同输入主体积 relL2 门为 `1e-5`。必须用 `-vb 1`（脚本已固定）：`-vb 2` 会关掉 GPU 驻留 MLEM，计时无效。3 次迭代只用于排序；设备上 20 次约 10 min 时，对胜出项再跑 `--it 20 --repeats 3 --variants baseline,<winner>`。结果在 `recon_out/perf_search_<时间戳>/experiment.json` 与 `decision.md`。

## 相关文档

- [`multipinhole/doc/Format_Reader_Matrix_CN.md`](../multipinhole/doc/Format_Reader_Matrix_CN.md) — 格式 × 读者能力矩阵
- [`6module_bak_src/README.md`](../6module_bak_src/README.md) — v2/v4 系统矩阵生成端
