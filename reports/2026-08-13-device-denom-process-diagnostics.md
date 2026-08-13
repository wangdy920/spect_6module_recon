# 设备端 denom / 重复进程修复与采集交付

日期：2026-08-13
状态：已收到一轮 RTX 5060 诊断样本并修复矩阵 host-cache 默认策略；无磁盘拷贝条件下的三次 cache-hit 中位数仍待设备复测。

## 冻结合同

- 固定部署目录下，denom 自动缓存不得依赖终端 CWD。
- 支持 INI 的 exe 相对自动目录/固定文件，以及 CLI 显式文件覆盖。
- 同一 GPU 不允许两个 `SPECT_recon.exe` 同时加载矩阵。
- 低 host-cache 配置不能顺带关闭 GPU resident matrix warmup。
- 设备测试将 1–9 次串行重建的进程树/工作集/I/O、GPU、free RAM、matrix/denom/output 磁盘总线、配置、cache、逐次计时和重建日志汇总为一个文件；三次基线直接给出中位数。
- 数值门沿用用户接受的主体积 `relative_l2 <= 1e-5`；性能采用重复运行中位数，单次本机数据不作为 RTX 5060 签收结论。

## 根因与修改

1. 原自动 denom 路径是 CWD 相对的 `recon_out/compact_denom_cache`；调度器改变启动目录就会反复 miss。现默认 `[SystemMatrix] compact_denom_cache_dir=cache/compact_denom`，INI 相对 exe 解析；`compact_denom_cache_path` 可固定文件，CLI `-compactDenomCache` 仍最高优先。进一步检查发现 v3 manifest 会嵌入 matrix/config 辅助文件的绝对路径，包根目录变化即使内容相同也会 miss/显式 mismatch；v4 改用 matrix profile、文件名/size/mtime、解析后的坏探头策略和数值参数，不再把绝对根路径作为数值 identity。
2. 主程序没有创建第二个重建进程；两个 PID 来自外部重复启动。现用 Windows named mutex `Global\\SPECT_recon_GPU_<id>` 做每 GPU 单实例保护，第二个实例在矩阵 metadata/CSR 加载前退出。
3. 原代码在 host cache 小于矩阵总量时直接跳过整个 GPU resident warmup。现仍预热显存可容纳的矩阵前缀，同时跳过无法保留的后台 host preload，避免无效磁盘争用。
4. denom 写入先移除旧 manifest，再写 MHD/RAW，最后发布 manifest；中断写不会留下可被误判为有效的新缓存。
5. INFO 日志新增 `[PERF]` 汇总；日志名加入 PID。安装包的 `bin/tools/collect_device_recon_diagnostics.ps1` 支持 `-Repeats 1..9`，每次使用独立输出前缀并在单一回传文件末尾给出 cache/resident 状态和 `median_full_command_s`；同一文件还给出 CPU、free RAM、进程工作集/累计读写量、matrix metadata SHA-256，以及各路径的 volume/disk/bus 信息。
6. 开发 staging 不再以 junction 指向受保护的源矩阵；只写 `SOURCE_MATRIX_DIRECTORY.txt`。自包含设备包继续由 `package_rc.ps1 -IncludeMatrix` 复制矩阵，清理 staging 不会牵连源数据。

## 本机数值与时延证据

环境：RTX 4070 Ti SUPER 16 GB；260601 四个 chunk，总计约 18.16 GiB；单床 108×108×80；1 次迭代。以下仅是路径冒烟，不是设备性能签收。

| 路径 | denom+matrix prepare | iteration core | full command | 备注 |
|---|---:|---:|---:|---|
| 自动 cache miss | 32.103 s | 0.962 s | 33.388 s | 建立 `auto_f8fcaf893a7aa8be` |
| 自动 cache hit | 4.425 s | 12.128 s | 16.871 s | 自动发现同一 hash；首迭代与后台 preload 重叠 |

- prepare 单次下降 86.22%，full command 单次下降 49.47%。这两个比例受 OS cache/JIT/后台任务影响，不作为中位数性能结论。
- miss 与 hit 的最终主体积：`relative_l2=3.304095619226407e-08`，`NCC=0.9999999999999956`，通过 `1e-5` 门。
- 预占 GPU 0 mutex 后启动 exe：第二实例退出码 1，日志明确写出 `GPU 0 is already owned...`，且文件名带 `_pid58504.log`。
- 增强后的 collector 真实 cache-hit 冒烟生成单个 88,151-byte UTF-8 报告，`all_cache_hits=True`、`all_resident_prefix_nonzero=True`；同时采到 NVMe 总线、free RAM、进程工作集峰值 `19,758,362,624` bytes 和累计读取 `19,521,592,662` bytes，可直接判断设备是否在整矩阵读入阶段发生内存/磁盘退化。
- 重复 collector 真实 cache-hit 冒烟串行执行两次，`full_command_s=34.344/28.156`、中位数 `31.250 s`，两次 resident prefix 均为 1；本机同时存在其他 GPU 任务，因此这些秒数只验证采集合同。两次主体积 relL2 `3.3679e-8`、NCC `1.0000000000000033`，通过 `1e-5` 门。
- v4 便携清单以源码 config/真实 matrix 路径建立显式 cache，再以 staging config/不同 junction 路径复用：第二次明确 `denom_cache=hit`；主体积 `relative_l2=3.377098333824908e-08`、`NCC=0.9999999999999994`，且 manifest 绝对路径 identity 行数为 0。

## RTX 5060 首轮回传（诊断样本）

回传文件已原样归档到忽略目录 `test-output/device_diagnostics/20260813-rtx5060-old-collector.txt`，SHA-256 为 `C8B4CEA97F27C7CA9B583CA0FD463FB92F0B9C7C8919CB118DB0E4715F619C91`。它由旧版 collector 单次运行产生，没有 `collector_schema`、重复中位数、free RAM、进程 I/O 或磁盘实时负载，因此结合用户说明的同期数据拷贝，只能作为诊断样本。

| 观测 | 结果 | 判定 |
|---|---:|---|
| EXE | `20260813+00801d3`，SHA-256 与 clean Release 相同 | 二进制正确 |
| SPECT 进程 | 仅 PID 48604 | 本次未复现重复启动 |
| denom | miss；prepare `79.535 s` | 首轮包含一次性建缓存 |
| 20 次 iteration core | `710.891 s`（约 `35.54 s/iter`） | 明显慢；不是 denom 单独造成 |
| full command | `791.573 s` | 单次、受拷贝污染，不作签收 |
| GPU iteration 采样 | 中位利用率 100%；约 81.6% 样本 ≥95%，约 9.6% 样本 ≤5% | 主体仍在 GPU；存在可疑停顿 |
| host cache / 矩阵 | `25.000 / 25.813 GiB` | 固定默认不足 |
| resident prefix | 2 / 6 | 16 GB 显存策略正常 |

固定 25 GiB cache 会永久保留前五块而让末块约 4.30 GiB 每个矩阵 pass 重新打开；20 次迭代最多对应约 86.0 GiB 的额外矩阵读取。旧 collector 没有进程 I/O，不能证明这些字节都落到物理盘，但同期数据拷贝会降低 Windows 文件缓存命中，足以放大这一确定的代码缺口。

设备 INI 还是 FWHM `1.2 mm`、Butterworth order `4`、旧 cache 目录；它不满足冻结的 FWHM `1.3 mm / 3 sigma`、Butterworth order `8` 合同。`x_out_flip=1` 是设备 CT 对齐所需的有意差异，继续保留。

## 矩阵可变后的缓存策略

- 删除固定 25/26 GiB 默认；启动时先发现 `chunk_*.csrb`，再按矩阵总量、系统总内存和当前可用内存选预算。
- 完整缓存的条件是当前可用内存至少覆盖“矩阵 + 8 GiB”，同时总内存至少保留 25%；否则采用安全的部分缓存。
- `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` 仍可显式覆盖。启动日志打印 `auto|override`、budget、matrix、system total/available；部分缓存还打印预计常驻块数和每个 pass 最多重读量。
- 设备旧 sampling 矩阵为 `25.813 GiB`；本机候选 `F:\matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220` 为 `24.331 GiB`。二者都不再依赖 profile 名或固定阈值。
- collector schema 2 同时记录自身路径/hash和整机各物理盘 read/write B/s、active time、queue，后续可直接标记数据拷贝污染。
- 本机以 220 矩阵真实打开验证：`matrix=24915 MiB`、启动可用内存 `33772 MiB`，auto 选择 `budget=24915 MiB`，完整缓存；新的 v4 denominator 在 `177 s` 后保存成功。随后单次迭代因本机同时存在另一 PET GPU 任务超过 10 分钟测试上限而主动终止，不构成 220 矩阵性能证据。

## 设备执行

先部署本轮 clean package，然后在设备 EXE 所在目录运行。第一次若是 cache miss，让它完成以建立显式 v4 cache；停止数据拷贝后原样再运行一次，正式采用第二次报告中的三次中位数：

```powershell
Set-Location "E:\software\ThreeModalitiesBin\SPECT_6Module_260707\bin\recon"
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\collect_device_recon_diagnostics.ps1 `
  -InputPath "D:\data\predata\20260812_nude_MDP_spect\20260812_170822\SPECT\Bed1.bin;D:\data\predata\20260812_nude_MDP_spect\20260812_170822\SPECT\Bed2.bin;D:\data\predata\20260812_nude_MDP_spect\20260812_170822\SPECT\Bed3.bin;D:\data\predata\20260812_nude_MDP_spect\20260812_170822\SPECT\Bed4.bin;D:\data\predata\20260812_nude_MDP_spect\20260812_170822\SPECT\Bed5.bin" `
  -MultiBedList -MatrixDir .\matrix\260601 `
  -DenomCachePath .\cache\compact_denom\sampling_260601_v4.mhd `
  -OutputPrefix "D:\data\recondata\20260812_nude_MDP_spect\20260812_170822\SPECT\device_test\recon" `
  -Iterations 20:20 -Repeats 3 -SampleSeconds 5 `
  -ReportPath "D:\data\recondata\20260812_nude_MDP_spect\20260812_170822\SPECT\device_test\spect_device_diagnostics_v2.txt"
```

更换为 220 矩阵时，仅把 `-MatrixDir` 改成 `F:\matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220`，并换一个独立的 `-DenomCachePath` 文件名。只回传 schema 2 报告。末尾应为 `all_cache_hits=True`、`all_resident_prefix_nonzero=True`；磁盘采样应无持续拷贝负载，再采用 `median_full_command_s`。

## 复现命令

```powershell
.\build_release.bat
cmake --fresh -S . -B build\verify-vs -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=D:/DevToolchain/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_INSTALLED_DIR=D:/DevToolchain/vcpkg/installed `
  -DVCPKG_TARGET_TRIPLET=x64-windows -DSPECT_CUDA_ARCHITECTURES="86;89;120" `
  -DSPECT_BUILD_TESTS=ON
cmake --build build\verify-vs --config Release --target SPECT_recon spect_tests --parallel 8
ctest --test-dir build\verify-vs -C Release --output-on-failure
python tools\search_compact_perf.py --self-check
python tools\prep_compact_denom_cache.py --dry-run --beds 1 --exe build\Release\bin\Release\SPECT_recon.exe --config multipinhole\config\spect_config.ini --chunkCSRDir test_data\6module\260601
python scripts\compare_recon_metrics.py --candidate test-output\auto_cache_hit\recon_hit_it1.mhd --reference test-output\auto_cache_hit\recon_it1.mhd --out-json test-output\auto_cache_hit\miss_vs_hit.json
$base = Join-Path (Get-Location) ("test-output\portable_denom_v4_" + (Get-Date -Format yyyyMMdd_HHmmss))
$matrix = (Resolve-Path test_data\6module\260601).Path
$alias = Join-Path $base matrix\260601
New-Item -ItemType Directory -Force -Path @((Split-Path $alias), (Join-Path $base cache), (Join-Path $base source), (Join-Path $base package)) | Out-Null
New-Item -ItemType Junction -Path $alias -Target $matrix | Out-Null
& build\Release\bin\Release\SPECT_recon.exe -config multipinhole\config\spect_config.ini -df test_data\logo\Bed1.bin -chunkCSRDir $matrix -it 1:1 -fout (Join-Path $base source\recon) -compactDenomCache (Join-Path $base cache\portable.mhd)
& build\Release\bin\Release\SPECT_recon.exe -config staging\SPECT_recon_rc_dev\bin\config\spect_config.ini -df test_data\logo\Bed1.bin -chunkCSRDir $alias -it 1:1 -fout (Join-Path $base package\recon) -compactDenomCache (Join-Path $base cache\portable.mhd)
python scripts\compare_recon_metrics.py --candidate (Join-Path $base package\recon_it1.mhd) --reference (Join-Path $base source\recon_it1.mhd) --out-json (Join-Path $base source_vs_package.json)
[IO.Directory]::Delete($alias)
powershell -ExecutionPolicy Bypass -File tools\collect_device_recon_diagnostics.ps1 -InputPath test_data\logo\Bed1.bin -OutputPrefix test-output\collector_repeat_smoke\recon -MatrixDir test_data\6module\260601 -Iterations 1:1 -Repeats 2 -SampleSeconds 1 -ReportPath test-output\collector_repeat_smoke\diagnostics.txt
cmake --install build\Release --config Release --prefix staging\SPECT_recon_device_diag
```

## 验证清单

- [x] fresh Release 构建与 staging 安装成功；仅保留既有 OpenMP/C4244/LNK4098 warning。
- [x] 顶层 CTest 旧索引已修复；246 个测试：245 通过，1 个外部原型目录 parity 测试按既有条件跳过。
- [x] 新增低 host-cache resident warmup 回归通过。
- [x] Windows 同 GPU 重复实例实测被拒绝。
- [x] 自动 denom miss→save→hit 真实矩阵链路通过，输出数值通过 `1e-5` 门。
- [x] v4 显式 denom 经源码与 staging 两个根路径 miss→hit；清单不含绝对根路径，输出数值通过 `1e-5` 门。
- [x] PowerShell AST、Python `py_compile`/self-check、两次真实 cache-hit 单文件 collector 冒烟通过。
- [x] 开发 staging 的旧 matrix junction 已被无递归地替换为普通占位目录；源矩阵 6 个文件保持完整，并通过真实 miss→save 的 1 次重建打开验证。
- [x] clean Release 每次从当前 Git HEAD 重建并把 collector、CHANGELOG 与 exe 一并安装到 `staging/SPECT_recon_rc_dev`；旧的 dirty 诊断 bundle 不再是交付入口。
- [x] RTX 5060 首轮单次诊断完成：一个 PID、resident prefix 2/6、denom miss、20 次 core `710.891 s`；因旧 collector 和同期数据拷贝，不作性能签收。
- [x] resource-aware host cache、部分缓存重读提示、collector schema 2 与磁盘实时采样已实现并通过构建/单元检查。
- [ ] RTX 5060 上停止数据拷贝后的三次 cache-hit 中位数尚未执行。

本轮首个 clean 构建检查点为 `20260813+b0226b8`，EXE SHA-256 `AC033850EA904E797379EC81C5E9A2C93001939FF9E7947338610E9F842B8244`；报告落盘后的最终 commit/EXE identity 见 continuity 最新 outcome。交付路径：`staging/SPECT_recon_rc_dev/`。开发 staging 不内置大型 matrix；设备部署继续保留固定 `bin/matrix/`，或使用 `package_rc.ps1 -IncludeMatrix` 生成独立副本。
