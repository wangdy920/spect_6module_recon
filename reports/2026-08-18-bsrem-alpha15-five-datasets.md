# BSREM α=1.5 five-dataset generalization + RDP interaction (Phase C.1 close-out)

Date: 2026-08-18. Provenance: overnight batch `tools/run_bsrem_rdp_convergence.py`
(exit 0, 2026-08-18 05:24 local), frozen 2026-08-10 prototype commands, evidence in
`recon_out/bsrem_rdp_convergence_20260817/` (curves/slices PNG per dataset,
`convergence_summary.json`, per-arm `iteration_metrics.csv`, per-run
`recon_manifest.json` with regularization asserted by the driver).

## Question

Does the derenzo finding — constant α=1.5, γ=0 is the clean single-subset BSREM
acceleration point (~1.45×) — generalize to the four multi-bed datasets, and how does
it interact with the historical RDP calibration?

## Arms (all 100 iter, ckpt/10; reference MLEM 500 iter, ckpt/50)

| arm | schedule | reg | manifest asserted |
|---|---|---|---|
| `mlem500` | — | none | — |
| `bsrem_b0` | α₀=1, γ=0.05 (historical) | none | enabled=false |
| `bsrem_rdp` | α₀=1, γ=0.05 | RDP β=0.3, δ=0.01, κ on | enabled=true, β=0.3 |
| `bsrem_a15g0` | **α=1.5, γ=0** | none | enabled=false |
| `bsrem_a15_rdp` | **α=1.5, γ=0** | RDP β=0.3, δ=0.01, κ on | enabled=true, β=0.3 |

relL2 measured on the saved (stitched, masked) MET_DOUBLE volumes against MLEM it500
of the same dataset.

## Result 1 — α=1.5 acceleration is uniform across all five datasets

MLEM-equivalent iteration of `bsrem_a15g0` (linear interpolation on the MLEM ckpt
curve):

| dataset | at it100 | at it50 | verdict |
|---|---|---|---|
| derenzo_182222 | MLEM it142 (1.42×) | it68 (1.36×) | consistent |
| epguan | MLEM it147 (1.47×) | it75 (1.51×) | consistent |
| epguan_19p2mm | MLEM it145 (1.45×) | it73 (1.46×) | consistent |
| mouse_ch6o6p2 | MLEM it148 (1.48×) | it76 (1.52×) | consistent |
| mouse | MLEM it146 (1.46×) | it74 (1.48×) | consistent |

**1.42–1.48× at matched iteration count, no dataset-dependent tuning.** The factor is
flat from it50 to it100 (not a transient). The historical α₀=1/γ=0.05 schedule
decelerates on every dataset (e.g. epguan `bsrem_b0` it100 = 0.166 relL2, worse than
MLEM it50 = 0.149). Since bare-BSREM per-iteration cost is at parity with MLEM
(plan §C.0), the iteration ratio is the wall-clock ratio.

## Result 2 — multi-bed low-sensitivity blowup: ML-family behavior, RDP immune

Diagnostic `roi_sum` (internal masked image) on the mouse datasets grows without
bound in **all unpenalized arms** — this is not an α=1.5 defect:

| arm (mouse) | roiSum it10 | it100 | image_max it100 |
|---|---|---|---|
| bsrem_b0 (α decaying) | 2.2e7 | 4.6e11 | 1.5e11 |
| bsrem_a15g0 | 2.2e7 | 2.1e13 | 1.8e12 |
| bsrem_rdp | 1.4e7 | **1.2e7 (stable)** | **4.4e2** |

Interpretation (from_source + diag): voxels in low-sensitivity bed-edge slices have
ML solutions that diverge; every ML-family update grows them (α=1.5 ~1.5× faster in
log-space; the clamp-negative counts, 23–60k voxels, are 100% inside the nominal ROI
mask but in these edge slices). The center66-cosine stitch excludes these slices, so
the **saved volumes are unaffected** — relL2-to-reference decays smoothly (this is
why Result 1 holds). derenzo/epguan (single-bed geometry margin) show zero clamps.

Consequences:
1. Not a blocker at ≤100 iter (float64 internal), but production hardening for long
   accelerated runs should freeze `s_j < s_min` voxels (same floor the PNCG plan
   already requires). Tracked as C.3.
2. RDP's neighborhood penalty suppresses the blowup entirely — a real engineering
   benefit of the penalized route, independent of the image-quality β debate.
3. `roi_sum` as logged is polluted by edge slices on multi-bed runs; read it only
   as a divergence detector, not as activity.

## Result 3 — α=1.5 does NOT transfer to penalized (RDP) objectives — REJECTED

The combined production candidate (α=1.5 + RDP) was tested on derenzo through a
full factor matrix. Every α≥1.5 penalized variant either enters a persistent limit
cycle or loses to the historical schedule:

| arm (all RDP δ=0.01, derenzo) | schedule | curv | β | it100 relChange | persistent clamps | verdict |
|---|---|---|---|---|---|---|
| bsrem_rdp (historical) | α₀=1, γ=0.05 | off | 0.3 | 5.9e-3 | ~1k, decaying | stable ref |
| bsrem_a15_rdp | α=1.5, γ=0 | frozen@it2 | 0.3 | 0.63 | 271k both-clamp | limit cycle |
| a15g002_rdp | α₀=1.5, γ=0.02 | off | 0.3 | ~0.5 | large | limit cycle |
| a15g005_rdp | α₀=1.5, γ=0.05 | off | 0.3 | declining | — | stable but relL2 0.661 > historical 0.619: overshoot damage never repaid |
| a15g0_rdp_nokappa | α=1.5, γ=0 | off | 0.3 | ~0.7 | large | limit cycle (κ not the driver) |
| a15g0_b01 | α=1.5, γ=0 | off | 0.1 | 0.15 | 50k at +1 | limit cycle |
| **a15_b01_livecurv** | α=1.5, γ=0 | **live** | 0.1 | 0.15 | 8k | limit cycle (curvature helps 6×, does not fix) |
| a15g0_b003 | α=1.5, γ=0 | off | 0.03 | 7.5e-3 | ~1k persistent | marginal pollution |
| rdp_b01 (control) | α₀=1, γ=0.05 | off | 0.1 | **8.4e-4** | **0** | converged by ~it100 |

Notes:
- The flat relL2-to-x* of `a15g0_b01` (~0.673) is *oscillation around a polluted
  state*, not convergence: it drifts away from the converged `rdp_b01` point
  (cross-arm relL2 0.34→0.35 rising) while `rdp_b01` reaches relChange 8.4e-4
  with zero clamps.
- Mechanism: the Nuyts RDP gradient is bounded per pair (∈[−3,1]) but at formed
  edges ψ''→0, so the curvature denominator has nothing to damp with exactly
  where the instability lives; a sub-population (cold voxels adjacent to hot
  structure, where β∇R competes with weak data support) enters a clamp cycle once
  α ≥ 1.5. Frozen-at-warmup curvature (default `curv_freeze` after 2 iters) is
  additionally useless because it snapshots a flat image.
- Instability onset is at image formation (~it20), so a "fast warmup then decay"
  hybrid would not dodge it.

**Penalized recipe stays α₀=1, γ=0.05.** RDP runs reach their own penalized
optimum in ~50–100 iterations anyway (rdp_b01 converged, clamps cold) — their
iteration economics are penalty-driven (high-frequency suppression), which must be
sold as penalized-likelihood, not optimizer acceleration.

Quality caveat unchanged from §C.1: β=0.3/δ=0.01 is the 2026-08-14 MDP calibration;
on derenzo it over-smooths (rod separation worse than MLEM x*). β task-quality
recalibration is C.2 and is a per-task decision, not part of the α adoption.

## Adoption

- **Adopt α=1.5, γ=0 for UNPENALIZED BSREM only** (β=0): 1.42–1.48× across all five
  datasets, no per-dataset tuning. **Penalized (RDP) runs keep α₀=1, γ=0.05** —
  every accelerated penalized variant tested is rejected (Result 3).
- Stable single-subset ceiling stands at ≈1.5× unpenalized / ≈1.0× penalized.
  Beyond that requires cross-iteration information with a line search that kills
  the clamp-cycle mode → this is now hard evidence for the P1 projected-PNCG GO
  criteria, for both objective classes.
- Remaining Phase C items: C.2 β task-quality calibration (needs task metrics /
  user criteria; the β ladder runs derenzo b0.3/b0.1/b0.03 are ready as visual
  input); C.3 low-sensitivity voxel freeze — **implemented and validated same day,
  see below**.

## C.2 — β task-quality calibration on user metrics (2026-08-18)

User metrics (2026-08-18): **1) rod separation, 2) per-rod axial uniformity.**
Implemented in `recon_out/bsrem_rdp_convergence_20260817/rod_metrics.py`: rod
centers detected once on the MLEM x* central-slab image (16 rods, 16 adjacent
pairs after rim-artifact and cross-sector exclusion), all volumes sampled at the
same coordinates. Separation = mean peak-valley contrast (P−V)/(P+V) over
adjacent pairs at spacings 10–15 vox; uniformity = CV of the 3×3-averaged axial
profile over the reference z-span (37–63).

| arm (derenzo, it100 unless noted) | separation | axial CV | note |
|---|---|---|---|
| MLEM it500 (x*) | 0.645 | 0.145 | reference; axCV worst (ML noise builds axially) |
| MLEM it150 | 0.395 | 0.136 | |
| **α=1.5 β=0** | **0.379** | **0.136** | = MLEM it~145 on both metrics — task-level confirmation of the 1.45× |
| MLEM it100 | 0.298 | 0.136 | |
| α=1.5 β=0.03 | 0.247 | 0.130 | mild limit-cycle pollution (~1k clamps) |
| hist β=0 | 0.106 | 0.128 | schedule alone destroys separation |
| hist β=0.03 | 0.097 | 0.127 | |
| hist β=0.05 | 0.092 | 0.126 | |
| hist β=0.1 | 0.083 | 0.123 | |
| hist β=0.3 | 0.071 | 0.112 | best uniformity, worst separation |

Factor decomposition (the surprise): **at a fixed 100-iteration budget the
historical γ=0.05 schedule — not β — is the binding constraint on rod
separation.** β=0 → β=0.3 on the historical schedule moves separation only
0.106→0.071, because by it100 α_k ≈ 0.17 and the image is quasi-frozen near its
early state. Axial uniformity improves weakly and monotonically with β
(0.128→0.112, and even x* is only 0.145): it is a noise metric, spanning ~1.3×
while separation spans ~9×.

**Recommendation (derenzo-class rod tasks, fixed iteration budget):**
1. **α=1.5, β=0 dominates**: separation 127% of MLEM-it100 (96% of it150),
   uniformity equal to MLEM. No penalized arm comes close on the primary metric.
2. Every measured β buys at most +18% uniformity for −35…−81% separation —
   a bad trade whenever rods are the task. If more axial uniformity is needed,
   it should come from a post-filter or from β re-tested on the task class where
   uniformity actually matters (organ uptake), not on derenzo.
3. The historical (α₀=1, γ=0.05) configuration should be retired for rod tasks
   at any β.

Evidence: `derenzo_182222_rod_metrics.json`, `derenzo_182222_rod_detection.png`
(geometry sanity), `derenzo_182222_metric_tradeoff.png` (trade-off scatter),
new stable-schedule arms `rdp_b005`, `rdp_b003`.

## C.3 — sensitivity floor freeze (implemented 2026-08-18)

Opt-in `-bsremSensFloor <rel>` (INI `[BSREM] sens_floor_rel`, manifest
`sens_floor_rel`; default 0 = off, frozen baselines untouched): voxels with
`s_j < rel·max(s)` are frozen to zero instead of updated. Wired through
`BsremUpdateParams.sMin` (mechanism already existed with an ineffective 1e-12
absolute default) and the diagnostics ROI gate.

Validation (mouse, `a15g0_floor` = α=1.5 + floor 1e-4, vs `bsrem_a15g0`):

| metric at it100 | no floor | floor rel=1e-4 |
|---|---|---|
| internal image_max | 1.8e12 | 3.6e4 |
| internal roi_sum | 2.1e13 | 3.7e7 |
| clamp-negative voxels | 23,220 | 4,253 |
| saved it100 volume delta | — | relL2 8.4e-5 |
| relL2-to-x* curve | — | identical to 4 decimals at every ckpt |

Containment is 6–8 orders of magnitude with no effect on the stitched output.
Voxels just above the floor still grow slowly (~1.04×/iter); for runs ≫100
iterations raise the floor (e.g. 1e-3). Tests: update-kernel freeze unit test,
parser accept/reject tests; suite 270 passed / 4 skipped. This also satisfies
the PNCG precondition "ROI clamp fraction ≲ 1e-3 after the first scans"
(4,253 voxels ≈ 5e-4 of the mask).

## Same-session true-RDP pipeline cost (Sampling, 50 iter)

Same binary, same session, back-to-back (`tools/run_sampling_rdp_cost.ps1`,
experiment INI, manifest asserted `regularization_enabled=true, RDP β=0.3`):

| arm | iteration_core_s (50 it) |
|---|---|
| MLEM | 208.285 |
| BSREM + RDP (α=1.5, β=0.3, κ on) | 205.090 |

**True-RDP pipeline cost is at parity with MLEM** (−1.5%, inside session noise):
the host OpenMP RDP gradient (1.86M voxels) hides fully under GPU work. This
closes the §C.0 correction debt — the earlier "parity" measured bare BSREM; the
statement now holds for genuinely-enabled RDP too. Note the pair runs ~50% slower
than yesterday's session (MLEM 208 vs 138 s core) — cross-session drift, which is
exactly why cost claims use same-session pairs; the protected keep (126.261 s)
stays untouched.

## Evidence index

- `recon_out/bsrem_rdp_convergence_20260817/convergence_summary.json` — all arms,
  all datasets, relL2 curves.
- `<dataset>_curves.png`, `<dataset>_slices.png` per dataset; 
  `derenzo_182222_beta_ladder_slices.png` — C.2 visual input (β=0.3 washes out
  derenzo rods; β≈0.03–0.1 is the plausible search range).
- Per-arm `iteration_metrics.csv` (relChange, clamp counts = the convergence
  discriminators), `recon_manifest.json` (regularization asserted).
- `recon_out/perf_rdpcost_{mlem,bsremrdp}_20260818_122024/` — cost pair.
- Driver: `tools/run_bsrem_rdp_convergence.py` (arms registry incl. rejected
  pilots); `tools/run_sampling_rdp_cost.ps1`.
