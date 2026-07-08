# CTAC pinhole attenuation model decision

## Decision

当前生产路径继续保留 `att[angle, voxel]` fused CTAC。

不继续实现生产级 dense `att[row, voxel]`、production on-the-fly Siddon 或 subray CTAC。下一步若要推进，只做 validation-only reference：用抽样 Siddon / pinhole-center ray 量化当前近似误差，再决定是否升级。

本决策不包含 q16 / CSRBQ CTAC；该方向已暂缓。

## Current Implementation

当前 compact float CTAC 在 CUDA nnz loop 中融合 attenuation factor：

```text
effective[t] = value[t] *
               att[fullAngle(row(t)) * nvox + transformedVoxel(col(t))] /
               65535
```

同一个 attenuation lookup 用于 forward、backward 和 sensitivity。`att` 表是 full-angle table，布局为：

```text
att[fullAngle, voxel]
```

当前表生成是 angle-dependent parallel-ray line integral：

```text
att[angle, voxel] = exp(- integral mu(s) ds)
```

这应视为 simple attenuation model，不是物理错误，但它不表达同一 angle 下不同 detector bin / subray 的路径差异。

## Local Evidence

当前 fused CTAC 批次已通过：

- `spect_tests.exe --gtest_brief=1`: `271/271 passed`
- `spect_cuda_scaffold_tests.exe --gtest_brief=1`: `28/28 passed` in the original fused CTAC run; after scope split, direct scaffold run was `27/27 passed`
- scaffold 覆盖 fused vs baked forward/backward/sensitivity、adjoint dot-product、unit attenuation parity、Sym90Z branch/zMirror
- 260601 圆柱 true-mu CTAC vs auto-mu CTAC：
  - iter 20: `relative_l2=3.472e-05`, `NCC=0.999999963095`
  - iter 50: `relative_l2=6.061e-05`, `NCC=0.999999941147`
  - iter 100: `relative_l2=1.065e-04`, `NCC=0.999999900582`
- 20-iter true-mu CTAC performance:
  - fused default: about `41 s` outer wall
  - baked reference: stopped after `>300 s`, only reached iter `12/20`

This proves implementation correctness and pipeline stability. It does not prove that `att[angle, voxel]` is always physically sufficient for pinhole geometry, because true-mu and auto-mu runs both use the same attenuation approximation.

## Validation-Only Pinhole Reference Result

Added tool:

```text
scripts/validate_ctac_pinhole_reference.py
```

Scope:

- reads compact `metadata.bin + chunk_*.csrb` by memmap
- samples high-weight nnz entries
- compares current parallel-ray `att[angle, voxel]` integral with pinhole-center Siddon integral
- writes `summary.json` and `samples.csv`
- does not change production reconstruction or attenuation table generation

Self-test:

```powershell
python scripts\validate_ctac_pinhole_reference.py --self-test
```

260601 true-mu active-voxel gate:

```powershell
python scripts\validate_ctac_pinhole_reference.py `
  --mu-map recon_out\ctac_automu_cylinder_debug_260601_fused_ctac_20260708\maps\sys260601_ref_true_mu.mhd `
  --out-dir recon_out\ctac_pinhole_reference_validation_260601_active_4096 `
  --rows-per-chunk 16 `
  --nnz-per-row 8 `
  --active-voxel-only
```

Result:

| Metric | Value |
| --- | ---: |
| samples | `4096` |
| active samples | `3734` |
| active H-weighted RMS `abs(delta I)` | `6.693e-04` |
| active H-weighted P95 `abs(delta I)` | `1.515e-03` |
| max `abs(delta I)` | `2.963e-03` |
| decision | `keep` |

This is far below the current upgrade trigger (`RMS_H > 0.02` or `P95_H > 0.03-0.05`) for this uniform-cylinder true-mu gate. The default unfiltered sample is allowed to return `blocked_insufficient_active_samples` when high-weight samples do not cross nonzero mu; use `--active-voxel-only` for this cylinder-specific validation gate.

## GPT 5.5 PRO Consultation Summary

GPT 5.5 PRO recommends:

1. Keep `att[angle, voxel]` fused CTAC as production.
2. Add validation-only Siddon / pinhole-center reference.
3. Do not implement production dense `att[row, voxel]` or on-the-fly Siddon unless residual gates fail.
4. If upgrade is needed, prefer same-layout pinhole-center table first:

```text
att[fullAngle, voxel] =
    exp(- integral from voxel center to pinhole center mu(s) ds)
```

This keeps the current table layout and fused CUDA kernel. It only changes how the attenuation table is generated.

## Why Not Dense Row/Voxel CTAC Now

Current volume size:

```text
nvox = 108 * 108 * 80 = 933120
```

Full projection rows:

```text
360 * 48 * 48 = 829440
```

A dense `att[row, voxel]` table would require:

```text
829440 * 933120 = 773967052800 entries
```

At uint16 this is about `1.55 TB` decimal storage. Even compact rows:

```text
90 * 24 * 48 * 933120 = 96745881600 entries
```

would still be about `193 GB` uint16. Dense row/voxel CTAC is not a practical production target.

Production on-the-fly Siddon avoids the table size, but adds heavy divergent ray traversal inside the reconstruction loop and risks using a CTAC ray model that does not exactly match the system matrix generator.

## Required Validation Before Any Upgrade

Only add a more complex CTAC model if validation-only reference shows material residuals.

Minimum gates:

| Gate | Purpose | Upgrade trigger |
| --- | --- | --- |
| sampled nnz `angle` vs pinhole-center vs row/subray Siddon | Direct model residual | `RMS_H(abs(delta I)) > 0.02` or `P95_H(abs(delta I)) > 0.03-0.05` |
| sensitivity map vs reference | Detect coherent physics drift | relative L2 `>1%` or P95 voxel error `>3%` |
| uniform water cylinder radial/sector profile | CTAC artifact gate | center-edge outside `0.98-1.02`, sector peak-to-peak `>2%` |
| off-center slab / half-space mu-map | `ctacRotSign` and divergent-ray stress | correct-sign sector residual `>2-3%` |
| reference-generated projection reconstructed with current CTAC | End-to-end mismatch | ROI mean bias `>2%`, active ROI relative L2 `>1%`, or iter 20 to 100 residual growth |
| mu=0 -> water -> mu=0 cache gate | Denom/cache invalidation | stale or reused sensitivity after mu-map switch |

For sub-2% absolute quantification targets, these thresholds should be tightened.

## Upgrade Path If Gates Fail

Use this order:

1. Add validation-only sampled Siddon / pinhole-center reference.
2. If parallel-ray approximation is the main error, upgrade table generation to same-layout pinhole-center `att[fullAngle, voxel]`.
3. If detector-bin residual remains material, test coarse detector-bin groups.
4. Only if coarse groups still fail, consider sparse per-nnz attenuation reference.
5. Avoid production on-the-fly Siddon unless a measured requirement justifies the runtime and geometry maintenance cost.

## Implementation Decision

Do not continue implementing production row/bin/subray/Siddon CTAC now.

Recommended next implementation is a small validation-only tool/test that samples high-weight nnz entries and reports:

- `delta I` between current `att[angle, voxel]` and pinhole-center Siddon
- optional `delta I` between pinhole-center and row/subray reference if row geometry is available
- H-weighted RMS/P95 residual
- per-module/per-branch residual summaries

If that report stays below the thresholds above, keep current fused CTAC and spend engineering time elsewhere.
