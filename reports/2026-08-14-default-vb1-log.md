# 新版默认日志（`-vb 1`）完整样本

日期：2026-08-14  
Exe：`20260814+7d5e813-dirty`（`build\verify-vs\bin\Release\SPECT_recon.exe`）  
原始 `.log`：[`assets/2026-08-14-default-vb1/recon_20260814_224000_000.log`](assets/2026-08-14-default-vb1/recon_20260814_224000_000.log)

这是当前默认 Info 级别下，一次**无 CTAC、五床、MLEM 50:50、denom 已命中**重建的完整控制台 / `.log` 内容。行集合按源码 `logInfo` 顺序排出；路径、GPU 名、秒数是标本，不是这次实跑抓取（工作区没有采样矩阵）。

文件名：`<outputName>_YYYYMMDD_HHMMSS_mmm.log`（无 PID）。  
每行格式：`YYYY-MM-DD HH:MM:SS <message>`（Info 没有级别标签）。

## 完整默认输出

```text
2026-08-14 22:40:00 Logging console output to: D:\data\recondata\case\SPECT\recon_20260814_224000_000.log
2026-08-14 22:40:00 Output directory: D:\data\recondata\case\SPECT (name prefix: recon)
2026-08-14 22:40:00 === SPECT_recon 20260814+7d5e813-dirty ===
2026-08-14 22:40:00 Command line: D:\pkg\bin\SPECT_recon.exe -config D:\pkg\bin\config\spect_config.ini -dfList Bed1.bin;Bed2.bin;Bed3.bin;Bed4.bin;Bed5.bin
2026-08-14 22:40:00 Loaded configuration from D:\pkg\bin\config\spect_config.ini
2026-08-14 22:40:01 CUDA environment: device=0/1 name="NVIDIA GeForce RTX 4070 Ti SUPER" sm_89, driver_cuda=13.0, runtime_cuda=13.0 (bundled runtime requires driver CUDA >= 13.0 / R580+)
2026-08-14 22:40:02 Multi-bed setup: beds=5, baseNz=220, overlapMode=output-center66_cosine10, reconOverlap=0, outputOverlap=656, reconNz=1100
2026-08-14 22:40:04 Loaded projection rays: total=4147200, per bed=829440
2026-08-14 22:40:04 Bad detector mask: skipped 1 detector blocks (8640 projection rows per bed): 68=link-4-FPGA-2-dector-4
2026-08-14 22:40:04 Output transform: canonical center66_cosine10 stitch; reconstruction beds are non-overlapping.
2026-08-14 22:40:12 compact denom auto-cache path: D:\pkg\bin\cache\compact_denom\auto_<fnv1a64>.mhd
2026-08-14 22:40:12 compact denom cache hit: D:\pkg\bin\cache\compact_denom\auto_<fnv1a64>.mhd
2026-08-14 22:40:15 [ITER] 1/50 compact
2026-08-14 22:40:18 [ITER] 2/50 compact
2026-08-14 22:40:21 [ITER] 3/50 compact
2026-08-14 22:40:24 [ITER] 4/50 compact
2026-08-14 22:40:27 [ITER] 5/50 compact
2026-08-14 22:40:30 [ITER] 6/50 compact
2026-08-14 22:40:33 [ITER] 7/50 compact
2026-08-14 22:40:36 [ITER] 8/50 compact
2026-08-14 22:40:39 [ITER] 9/50 compact
2026-08-14 22:40:42 [ITER] 10/50 compact
2026-08-14 22:40:45 [ITER] 11/50 compact
2026-08-14 22:40:48 [ITER] 12/50 compact
2026-08-14 22:40:51 [ITER] 13/50 compact
2026-08-14 22:40:54 [ITER] 14/50 compact
2026-08-14 22:40:57 [ITER] 15/50 compact
2026-08-14 22:41:00 [ITER] 16/50 compact
2026-08-14 22:41:03 [ITER] 17/50 compact
2026-08-14 22:41:06 [ITER] 18/50 compact
2026-08-14 22:41:09 [ITER] 19/50 compact
2026-08-14 22:41:12 [ITER] 20/50 compact
2026-08-14 22:41:15 [ITER] 21/50 compact
2026-08-14 22:41:18 [ITER] 22/50 compact
2026-08-14 22:41:21 [ITER] 23/50 compact
2026-08-14 22:41:24 [ITER] 24/50 compact
2026-08-14 22:41:27 [ITER] 25/50 compact
2026-08-14 22:41:30 [ITER] 26/50 compact
2026-08-14 22:41:33 [ITER] 27/50 compact
2026-08-14 22:41:36 [ITER] 28/50 compact
2026-08-14 22:41:39 [ITER] 29/50 compact
2026-08-14 22:41:42 [ITER] 30/50 compact
2026-08-14 22:41:45 [ITER] 31/50 compact
2026-08-14 22:41:48 [ITER] 32/50 compact
2026-08-14 22:41:51 [ITER] 33/50 compact
2026-08-14 22:41:54 [ITER] 34/50 compact
2026-08-14 22:41:57 [ITER] 35/50 compact
2026-08-14 22:42:00 [ITER] 36/50 compact
2026-08-14 22:42:03 [ITER] 37/50 compact
2026-08-14 22:42:06 [ITER] 38/50 compact
2026-08-14 22:42:09 [ITER] 39/50 compact
2026-08-14 22:42:12 [ITER] 40/50 compact
2026-08-14 22:42:15 [ITER] 41/50 compact
2026-08-14 22:42:18 [ITER] 42/50 compact
2026-08-14 22:42:21 [ITER] 43/50 compact
2026-08-14 22:42:24 [ITER] 44/50 compact
2026-08-14 22:42:27 [ITER] 45/50 compact
2026-08-14 22:42:30 [ITER] 46/50 compact
2026-08-14 22:42:33 [ITER] 47/50 compact
2026-08-14 22:42:36 [ITER] 48/50 compact
2026-08-14 22:42:39 [ITER] 49/50 compact
2026-08-14 22:42:42 [ITER] 50/50 compact
2026-08-14 22:42:45 Run manifest: D:\data\recondata\case\SPECT\recon_manifest.json
2026-08-14 22:42:45 [PERF] denom_cache=hit projection_load_s=2.041 csr_open_s=1.882 denom_and_matrix_prepare_s=8.310 iteration_core_s=148.002 load_to_iteration_done_s=160.417 full_command_s=165.880
2026-08-14 22:42:47 Reconstruction finished successfully.
```

共 65 行，与 assets 里的 `.log` 相同。

固定数字（不是示意）：

| 字段 | 值 | 来源 |
|---|---|---|
| `[ITER] n/50` | `algo_mlem.conf` `iterations = 50:50` | 未传 `-it` |
| `per bed=829440` | `360×48×48` | 矩阵 metadata |
| `total=4147200` | `5×829440` | 五床 |
| `8640 projection rows` | `60×12×12` | 探头 68 块 |
| `outputOverlap=656` | `4×164` | `baseNz 220 − step 56` |
| `reconNz=1100` | `5×220`，重建床不重叠 | `reconOverlap=0` |

现场 RTX 5060 只换 CUDA 那一行的 `name=` / `sm_120`。`-it 20:20` 则 `[ITER]` 为 `n/20`，共 20 行，其余行不变。

## 同级别下会多出来的行（非这条默认命令）

首次无缓存（替换 hit 那一行）：

```text
compact denom cache miss: cache files not found
compact denom cache saved: D:\pkg\bin\cache\compact_denom\auto_<fnv1a64>.mhd
```

`[PERF]` 此时是 `denom_cache=miss`。

| 条件 | 额外 Info |
|---|---|
| 单床 | `Single-bed reconstruction`（没有 Multi-bed 那行） |
| `-opti bsrem` 无 `-regCT` | `TV/RDP uses activity-only neighborhood weights.` |
| `-regCT <mhd>` | `Loading registered CT...` 然后 `TV/RDP CT weighting enabled: sigma=..., floor=....` |
| `-ctac` | `CTAC model: ...`、`Using dynamic CTAC...`、`Generating...` 或 `All CTAC attenuation factors loaded from cache.`、`compact CTAC: per-bed attenuation factors fused...`；无 auto denom 行，`[PERF] denom_cache=disabled` |
| `-bProj` | `Loaded background projection term with N rays.` |
| denom miss 且坏探头 | 另有 `Bad detector mask: removed 1 detector blocks from the sensitivity denominator.` |
| `-projectionResidual` | `compact projection residual CSV: ...` |
| 矩阵 host cache 装不下 | 无时间戳的 `[NOTE] compact host cache ... below matrix size ...` |
| 保存失败 | `[NOTE] Failed saving compact denom cache: ...` |

## 默认不再出现

`-vb 2` 才有：run-configuration dump、逐床 mapping、center66 gain/seam、fatbin/CSR 内部、`GPU process lock acquired`、每床 `[METRIC]`。  
成功路径也不再刷 `host cache auto:`、fatbin loaded、resident warmup 成功行。

## Verification

- 行集合对照 `MainProgram.cpp` / `CompactSymmetricReconLoop.cpp` 的 `logInfo`（`-vb 1`）。
- 几何/探头计数对照 INI、`algo_mlem.conf`、`bad_dector`、compact metadata 合同。
- 未实跑重建；秒数、盘符、`auto_<fnv1a64>` 是标本。
