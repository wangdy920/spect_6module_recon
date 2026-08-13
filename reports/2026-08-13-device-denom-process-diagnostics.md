# 设备端 denom / 重复进程修复与采集交付

日期：2026-08-13
状态：本机实现与验证完成；RTX 5060 的 20 次迭代结论等待设备报告。

## 冻结合同

- 固定部署目录下，denom 自动缓存不得依赖终端 CWD。
- 支持 INI 的 exe 相对自动目录/固定文件，以及 CLI 显式文件覆盖。
- 同一 GPU 不允许两个 `SPECT_recon.exe` 同时加载矩阵。
- 低 host-cache 配置不能顺带关闭 GPU resident matrix warmup。
- 设备测试将进程树、GPU、配置、cache 与重建日志汇总为一个文件。
- 数值门沿用用户接受的主体积 `relative_l2 <= 1e-5`；性能采用重复运行中位数，单次本机数据不作为 RTX 5060 签收结论。

## 根因与修改

1. 原自动 denom 路径是 CWD 相对的 `recon_out/compact_denom_cache`；调度器改变启动目录就会反复 miss。现默认 `[SystemMatrix] compact_denom_cache_dir=cache/compact_denom`，INI 相对 exe 解析；`compact_denom_cache_path` 可固定文件，CLI `-compactDenomCache` 仍最高优先。
2. 主程序没有创建第二个重建进程；两个 PID 来自外部重复启动。现用 Windows named mutex `Global\\SPECT_recon_GPU_<id>` 做每 GPU 单实例保护，第二个实例在矩阵 metadata/CSR 加载前退出。
3. 原代码在 host cache 小于矩阵总量时直接跳过整个 GPU resident warmup。现仍预热显存可容纳的矩阵前缀，同时跳过无法保留的后台 host preload，避免无效磁盘争用。
4. denom 写入先移除旧 manifest，再写 MHD/RAW，最后发布 manifest；中断写不会留下可被误判为有效的新缓存。
5. INFO 日志新增 `[PERF]` 汇总；日志名加入 PID。安装包新增 `bin/tools/collect_device_recon_diagnostics.ps1`。

## 本机数值与时延证据

环境：RTX 4070 Ti SUPER 16 GB；260601 四个 chunk，总计约 18.16 GiB；单床 108×108×80；1 次迭代。以下仅是路径冒烟，不是设备性能签收。

| 路径 | denom+matrix prepare | iteration core | full command | 备注 |
|---|---:|---:|---:|---|
| 自动 cache miss | 32.103 s | 0.962 s | 33.388 s | 建立 `auto_f8fcaf893a7aa8be` |
| 自动 cache hit | 4.425 s | 12.128 s | 16.871 s | 自动发现同一 hash；首迭代与后台 preload 重叠 |

- prepare 单次下降 86.22%，full command 单次下降 49.47%。这两个比例受 OS cache/JIT/后台任务影响，不作为中位数性能结论。
- miss 与 hit 的最终主体积：`relative_l2=3.304095619226407e-08`，`NCC=0.9999999999999956`，通过 `1e-5` 门。
- 预占 GPU 0 mutex 后启动 exe：第二实例退出码 1，日志明确写出 `GPU 0 is already owned...`，且文件名带 `_pid58504.log`。
- 诊断脚本预期失败冒烟生成单个 22,090-byte UTF-8 报告，包含 native recon log、PID/父 PID/命令行、GPU 样本、配置和 cache 清单。

## 设备执行

先保留设备既有 `bin/matrix/<profile>/`，部署诊断 bundle 中的 `bin` 内容。第一次若是 cache miss，让它完成以建立自动 cache；随后运行采集命令测稳定 cache-hit：

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\tools\collect_device_recon_diagnostics.ps1 `
  -InputPath "D:\data\case\SPECT\Bed1.bin;D:\data\case\SPECT\Bed2.bin" -MultiBedList `
  -OutputPrefix "D:\SPECT_Result\device_test\recon" -Iterations 20:20 `
  -ReportPath "D:\SPECT_Result\device_test\spect_device_diagnostics.txt"
```

如需排除自动目录差异，追加绝对 `-DenomCachePath "D:\...\cache.mhd"`。回传 `spect_device_diagnostics.txt`。稳定性能比较用相同输入重复三次，采用 full-command 中位数，并同时检查每次都是 `denom_cache=hit`、只有一个 SPECT PID、resident prefix 非零。

## 复现命令

```powershell
cmake --build build\Release --config Release --target SPECT_recon spect_tests --parallel 8
.\build\Release\bin\Release\spect_tests.exe --gtest_brief=1
python tools\search_compact_perf.py --self-check
python tools\prep_compact_denom_cache.py --dry-run --beds 1 --exe build\Release\bin\Release\SPECT_recon.exe --config multipinhole\config\spect_config.ini --chunkCSRDir test_data\6module\260601
python scripts\compare_recon_metrics.py --candidate test-output\auto_cache_hit\recon_hit_it1.mhd --reference test-output\auto_cache_hit\recon_it1.mhd --out-json test-output\auto_cache_hit\miss_vs_hit.json
cmake --install build\Release --config Release --prefix staging\SPECT_recon_device_diag
```

## 验证清单

- [x] Release 构建成功；仅保留既有 `LNK4098` warning。
- [x] 245 个测试：244 通过，1 个外部原型目录 parity 测试按既有条件跳过。
- [x] 新增低 host-cache resident warmup 回归通过。
- [x] Windows 同 GPU 重复实例实测被拒绝。
- [x] 自动 denom miss→save→hit 真实矩阵链路通过，输出数值通过 `1e-5` 门。
- [x] PowerShell AST、Python `py_compile`/self-check、单文件 collector 冒烟通过。
- [x] 诊断 bundle 安装并验证 `-v` 内嵌 CHANGELOG；exe SHA-256：`A5C1702484F7137D19EE5CF8BF26047946CDC7EB878D98BB2AE31D4A0394FBF6`。
- [ ] RTX 5060 上 20 次迭代的三次 cache-hit 中位数尚未执行。
- [ ] 当前 bundle 身份为 `20260813+a696994-dirty`，用于设备诊断；在现有 CT 工作树收敛前不宣称 clean release。

交付路径：`staging/SPECT_recon_device_diag/`。该诊断 bundle 不内置大型 matrix 文件，部署时保留设备原有固定 `bin/matrix/`。
