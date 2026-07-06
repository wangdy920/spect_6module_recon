# Share MLEM 50 Iter Regression Test

生成时间：2026-06-05T21:27+08:00

## 1. 目的

使用 `D:\WDY-project\SPECT-recon\test_data\share` 中的投影和参考重建结果，对当前 compact-symmetric 后端做基础 MLEM 回归测试。

参考结果的原始指令：

```bash
./spect_recon \
  --matrix-dir /mnt/sharedisk2/recon/matrix_PSC_6module_24pin/matrix_PSC_6modules_24pin_YSO_855x855_DOI_PEN \
  --proj /home/bayimaging_recon4/recon/PSC_6module_24pin_10MBq_600s_test_1.bin \
  --out recon_output \
  --iters 50 \
  --gpus 2 \
  --save-every 5
```

该指令为基础 MLEM，无 PSF、无后滤波、无正则化。

## 2. 输入和参考

- 投影输入：`test_data/share/PSC_6module_24pin_10MBq_600s_test_1.bin`
- 当前矩阵目录：`test_data/6module/260601`
- 当前 denominator cache：`recon_out/compact_denom_cache/psc_260601_global_denom.mhd`
- 参考输出：
  - `test_data/share/recon_iter050.img`
  - `test_data/share/recon_iter050.raw`
  - `test_data/share/sensitivity.raw`

格式确认：

- `recon_iter050.img` 是 headerless little-endian float64，体素数 `933120`。
- `recon_iter050.raw` 是 20-byte header + little-endian float32 payload，payload 与 `.img` 数值一致。
- `sensitivity.raw` 同样是 20-byte header + float32 payload。

注意：`share` 中的投影 BIN 与此前常用 `test_data/6module/PSC_6module_24pin_10MBq_600s_1.bin` SHA256 不同，因此本轮使用 `share` 投影单独运行。

## 3. 当前运行命令

```powershell
out\build\chunk-csrb-agent\bin\SPECT_recon.exe `
  -df test_data\share\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -chunkCSRDir test_data\6module\260601 `
  -fout recon_out\share_regression_mlem50_20260605\recon `
  -compactDenomCache recon_out\compact_denom_cache\psc_260601_global_denom.mhd `
  -config __missing_share_regression_config__.ini `
  -matrixLayout compact-symmetric `
  -sparseBackend cuda `
  -alg MLEM `
  -subsets 1 `
  -it 50:5 `
  -th 8
```

输出目录：`recon_out/share_regression_mlem50_20260605`

保存间隔：`-it 50:5`，已生成 `recon_it5` 到 `recon_it50`，每 5 iter 一个输出。

## 4. Sensitivity 对比

当前 denominator cache 对参考 `sensitivity.raw`：

- compare: `recon_out/share_regression_sensitivity_vs_current_cache.json`
- `relative_l2=2.8732813379821047e-07`
- `rmse_over_reference_mean=3.0024714619973516e-07`
- `sum_rel_diff=2.0869436392332782e-07`
- `candidate_sum=68987253.4364087`
- `reference_sum=68987239.03916073`

结论：sensitivity 口径对齐，差异为 float/double 与累加舍入量级。

## 5. 重建耗时和资源

本轮直接运行，无 system-load precheck。

- `total_load_to_iteration_done=137.306s`
- `full_command_wall_time=137.314s`
- `compact_sensitivity=23.067s`
- `compact_forward_ratio_backward=113.410s/50 calls`
- `compact.forward=80.182s/50 calls`
- `compact.backward=30.708s/50 calls`
- `mlem_update=0.032s/50 calls`
- `cuda.compact_matrix_upload=52.124s/151 calls`
- `cuda.compact_output_d2h=57.048s/3200 calls`
- `cuda.compact_forward_output_d2h=27.121s/1600 calls/165,888,000 bytes`
- `cuda.compact_backward_output_d2h=29.927s/1600 calls/5,971,968,000 bytes`
- `compact_peak_memory host_working_set_peak_bytes=5064089600`
- `compact_peak_memory host_process_peak_working_set_bytes=19663564800`
- `compact_peak_memory cuda_used_peak_bytes=8546484224`

注：本轮耗时高于此前空闲设备上的正式 50 iter benchmark，不作为性能签收，只作为回归对比运行记录。

## 6. 重建对比结果

当前 `recon_it50.mhd` 对参考 `recon_iter050.img`：

- compare: `recon_out/share_regression_mlem50_20260605/metrics_vs_share_recon_iter050_img.json`
- `passed=true`
- `relative_l2=9.522275479052328e-06`
- `rmse=5.116314841516955e-07`
- `rmse_over_reference_mean=1.0632945644628913e-05`
- `sum_rel_diff=1.1546361206897968e-09`
- `max_abs_diff=5.304748889684019e-06`
- `p95_abs_diff=1.080608726456225e-06`
- `p99_abs_diff=1.5583434346762275e-06`
- `candidate_sum=44899.46491747093`
- `reference_sum=44899.46496931347`

当前 `recon_it50.mhd` 对参考 `recon_iter050.raw`：

- compare: `recon_out/share_regression_mlem50_20260605/metrics_vs_share_recon_iter050_raw.json`
- 指标与 `.img` 对比一致。

当前输出统计：

- `count=933120`
- `finite=933120`
- `min=0.001250171608777362`
- `max=0.22206817084246988`
- `mean=0.04811756785565729`
- `sum=44899.46491747093`

## 7. 结论

本轮基础 MLEM 回归测试通过。

- 当前 compact-symmetric 后端在 `share` 投影输入上，与参考 `recon_iter050` 数值高度一致。
- sensitivity 也与参考 `sensitivity.raw` 对齐。
- 回归差异量级为 `relative_l2=9.52e-06`、`rmse/reference_mean=1.06e-05`，远小于当前 compare 脚本默认阈值。
- 没有证据表明最近的 multi-bed 或 BSREM/RDP 适配破坏基础单床位 MLEM 结果。
