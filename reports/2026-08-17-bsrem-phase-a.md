# Phase A: freeze full-scan BSREM-II correctness

Date: 2026-08-17. Provenance: user Phase A spec.  
Status: **toy-scale GO**; Sampling-scale adjoint/sensitivity and compact `data_objective` from μ are **blocked / deferred**, not silent passes.

## Contract

| Item | Value |
|---|---|
| Protected outcome | β=0, α=1 BSREM reproduces host MLEM; RDP ∇R matches energy; λ=βκ in both num and den; clamp is after α; NaN/Inf/illegal μ = 0 in tests |
| Not in this step | PNCG, subsets, CSR(Aᵀ), SpMM T, BF16, new PSF, beta sweep, voxel size, matrix generator |
| Operator keep | Sampling 50-iter resident MLEM path **not modified**. Host MLEM extracted to `applyMlemUpdate` (used at `-vb 2` / diagnostics). 50-iter `iteration_core_s` **not remeasured**. |

## Formula (now the unique host BSREM-II update)

```text
lambda_j = beta * kappa_j
u_j = alpha * (n_j - s_m,j - lambda_j r_j) / (s_j + lambda_j max(c_j,0) + eps_d)
ū_j = clip(u_j, -0.9, 1)     # skipped in -bsremEquiv
x <- x * (1 + ū)
s_j <= sMin -> zero (production); equivalence uses MLEM's s>0 then max(s,1e-12)
```

`D_reg` in production remains the surrogate `|∇R|/max(x,eps)` (always ≥0). Negative-curvature STOP does not apply to this surrogate.

## GO / STOP (this session)

| Gate | Result | Evidence class | Maturity |
|---|---|---|---|
| A0/A1 host + toy CUDA, e_x < 1e-4 | **pass** (host 1e-12; CUDA 5-iter < 1e-4) | machine-checked | provisional (toy A, not Sampling) |
| A2 clip ≠ A0 | **pass** (documented, not an equivalence proof) | machine-checked | diagnostic |
| Toy adjoint e_adj ≤ 1e-4 | **pass** | CUDA vs dense, 1 chunk / 1 bed | provisional |
| Toy s_cache vs Aᵀ1 e_s ≤ 1e-5 | **pass** vs dense compact denom | machine-checked | provisional |
| Sampling 6-chunk adjoint / 5-bed | **not run** | matrix not in this workspace glob | blocked |
| RDP constant interior ∇R≈0 | **pass** (~0 within 1e-12) | CPU 8³ | provisional |
| RDP directional FD e_R < 1e-2 | **pass** | CPU 8³ | provisional |
| λ in num and den | **pass** | unit: v=0 when λ matches | diagnostic |
| Clamp after α | **pass** | α=2, u=6 → ū=1, x=2 | diagnostic |
| C0–C3 finite, not all-clamped | **pass** | host synthetic | diagnostic |
| Compact Φ_data from μ | **not in loop** | would extra-scan A or change ratio kernel | deferred |
| NaN/Inf/illegal μ | **pass** in poisson unit test | machine-checked | diagnostic |

## How to run A0/A1 on a real matrix (not executed here)

A0 (host MLEM, resident off):

```text
SPECT_recon ... -opti mlem -it 20:1 -vb 2
```

A1 (degeneration; forces β=0, α=1, curv off, strips algo_bsrem RDP):

```text
SPECT_recon ... -opti bsrem -bsremEquiv on -bsremDiag on -it 20:1
```

A2 (production clip, still β=0):

```text
SPECT_recon ... -opti bsrem -bsrem 1,0 -reg None -it 20:1 -bsremDiag on
```

`-bsremEquiv` overrides `algo_bsrem.conf` RDP. In-recon Butterworth/Smooth still fail closed.

## Remaining gap

1. Export μ from the fused ratio stage (or opt-in extra forward) so `iteration_metrics.csv` fills `data_objective` without a second matrix philosophy fight.
2. Sampling 6-chunk + 5-bed adjoint/sensitivity when that matrix is mounted.
3. True RDP Hessian diagonal vs `|∇R|/x` surrogate, if C2 is ever used as a curvature claim.

Phase B (2-RHS T SpMM) must not start from a claim that Sampling-scale Aᵀ is adjoint-certified; only the toy operator is.

## Verification

| Check | Evidence |
|---|---|
| Focused gtests | 17+1 passed (see command below) |
| `SPECT_recon.exe` relinked | `build\verify-vs\bin\Release\SPECT_recon.exe` 2026-08-17 |
| 50-iter Sampling gate | **not run** |
| Derenzo 500 | **not run** |
