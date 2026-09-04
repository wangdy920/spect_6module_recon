"""Sample compact chunk CSRB rows and report structure statistics that drive
matrix-compression / kernel-design decisions (run lengths, column deltas,
value dynamic range, per-row Z span).

Usage: python tools/analyze_csrb_structure.py <chunk.csrb> --nx 120 --ny 120 --nz 220 [--rows 512]
"""
import argparse
import sys
import numpy as np

HEADER = 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk")
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--rows", type=int, default=512, help="rows to sample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    hdr = np.fromfile(a.chunk, dtype=np.uint8, count=HEADER)
    magic, version = (int(x) for x in np.frombuffer(hdr[:8].tobytes(), dtype=np.uint32))
    rows, ang_off, ang_cnt = (int(x) for x in np.frombuffer(hdr[8:20].tobytes(), dtype=np.int32))
    nnz = int(np.frombuffer(hdr[20:28].tobytes(), dtype=np.uint64)[0])
    print(f"magic=0x{magic:08x} version={version} rows={rows} angle_offset={ang_off} "
          f"angle_count={ang_cnt} nnz={nnz} bytes/nnz(on disk)={8.0:.1f}")

    row_ptr = np.memmap(a.chunk, dtype=np.int64, mode="r", offset=HEADER, shape=(rows + 1,))
    col_off = HEADER + 8 * (rows + 1)
    val_off = col_off + 4 * nnz
    col_idx = np.memmap(a.chunk, dtype=np.uint32, mode="r", offset=col_off, shape=(nnz,))
    values = np.memmap(a.chunk, dtype=np.float32, mode="r", offset=val_off, shape=(nnz,))

    rng = np.random.default_rng(a.seed)
    sample = np.sort(rng.choice(rows, size=min(a.rows, rows), replace=False))

    plane = a.nx * a.ny
    run_lens = []
    runs_per_row = []
    nnz_per_row = []
    deltas = []
    zspan_frac = []
    lines_per_row = []
    vmin_all = np.inf
    vmax_all = 0.0
    n_lt_fp16_norm = 0
    n_lt_fp16_sub = 0
    n_total = 0
    row_dyn = []       # rowmax / rowmin
    sorted_rows = 0
    rel_err_fp16 = []
    rel_err_bfp16 = []  # per-row block-scaled uint16
    rel_err_fp24 = []   # float32 with 8 mantissa LSB dropped
    prune_thr = [1e-6, 1e-5, 1e-4, 1e-3]
    prune_frac = [[] for _ in prune_thr]
    prune_err = [[] for _ in prune_thr]
    # PSF-shape impact of value quantization: per-row footprint RMS radius
    # (value-weighted, voxel units), centroid shift, row-sum (sensitivity)
    # change, and L1 share of nnz that quantize to exactly zero.
    quant_names = ["fp16", "u16_row", "u16_sqrt", "fp24"]
    q_rms_rel = {k: [] for k in quant_names}
    q_cent_shift = {k: [] for k in quant_names}
    q_sum_rel = {k: [] for k in quant_names}
    q_zero_l1 = {k: [] for k in quant_names}
    q_zero_frac = {k: [] for k in quant_names}
    q_rel_l2 = {k: [] for k in quant_names}

    def quantize(name, v):
        m = v.max()
        if name == "fp16":
            return v.astype(np.float16).astype(np.float64)
        if name == "u16_row":
            s = m / 65535.0
            return np.round(v / s) * s
        if name == "u16_sqrt":
            q = np.round(np.sqrt(v / m) * 65535.0)
            return m * (q / 65535.0) ** 2
        if name == "fp24":
            u = v.astype(np.float32).view(np.uint32)
            return ((u + 0x80) & 0xFFFFFF00).astype(np.uint32).view(np.float32).astype(np.float64)
        raise ValueError(name)

    def footprint_moments(c, w):
        z = c // plane
        rem = c - z * plane
        y = rem // a.nx
        x = rem - y * a.nx
        pts = np.stack([x, y, z], axis=1).astype(np.float64)
        ws = w.sum()
        cen = (pts * w[:, None]).sum(axis=0) / ws
        rms = np.sqrt((((pts - cen) ** 2).sum(axis=1) * w).sum() / ws)
        return cen, rms

    for r in sample:
        s, e = int(row_ptr[r]), int(row_ptr[r + 1])
        if e <= s:
            continue
        c = np.asarray(col_idx[s:e], dtype=np.int64)
        v = np.asarray(values[s:e], dtype=np.float64)
        nnz_per_row.append(e - s)
        if np.all(np.diff(c) > 0):
            sorted_rows += 1
        else:
            order = np.argsort(c, kind="stable")
            c = c[order]
            v = v[order]
        d = np.diff(c)
        for k, thr in enumerate(prune_thr):
            m = v < thr * v.max()
            prune_frac[k].append(m.mean())
            prune_err[k].append(np.linalg.norm(v[m]) / np.linalg.norm(v))
        deltas.append(d)
        # runs of consecutive columns (delta == 1)
        breaks = np.flatnonzero(d != 1)
        rl = np.diff(np.concatenate(([-1], breaks, [len(c) - 1])))
        run_lens.append(rl)
        runs_per_row.append(len(rl))
        z = c // plane
        zspan_frac.append((z.max() - z.min() + 1) / a.nz)
        lines_per_row.append(len(np.unique(c // a.nx)))
        vmin_all = min(vmin_all, v[v > 0].min() if np.any(v > 0) else vmin_all)
        vmax_all = max(vmax_all, v.max())
        n_total += len(v)
        n_lt_fp16_norm += int(np.sum(v < 6.1035e-5))
        n_lt_fp16_sub += int(np.sum(v < 5.96e-8))
        row_dyn.append(v.max() / v[v > 0].min())
        # quantization experiments (relative L2 error of the row vector)
        q16 = v.astype(np.float16).astype(np.float64)
        rel_err_fp16.append(np.linalg.norm(q16 - v) / np.linalg.norm(v))
        scale = v.max() / 65535.0
        qb = np.round(v / scale) * scale
        rel_err_bfp16.append(np.linalg.norm(qb - v) / np.linalg.norm(v))
        u = v.astype(np.float32).view(np.uint32)
        u24 = ((u + 0x80) & 0xFFFFFF00).astype(np.uint32)  # round mantissa to 15 bits
        q24 = u24.view(np.float32).astype(np.float64)
        rel_err_fp24.append(np.linalg.norm(q24 - v) / np.linalg.norm(v))
        cen0, rms0 = footprint_moments(c, v)
        for name in quant_names:
            vq = quantize(name, v)
            q_rel_l2[name].append(np.linalg.norm(vq - v) / np.linalg.norm(v))
            zero = vq == 0.0
            q_zero_frac[name].append(zero.mean())
            q_zero_l1[name].append(v[zero].sum() / v.sum())
            q_sum_rel[name].append((vq.sum() - v.sum()) / v.sum())
            cen1, rms1 = footprint_moments(c, vq)
            q_rms_rel[name].append((rms1 - rms0) / rms0)
            q_cent_shift[name].append(np.linalg.norm(cen1 - cen0))

    run_lens = np.concatenate(run_lens)
    deltas = np.concatenate(deltas)
    nnz_per_row = np.array(nnz_per_row)
    runs_per_row = np.array(runs_per_row)

    def pct(x, q):
        return np.percentile(x, q)

    print(f"\nsampled rows={len(nnz_per_row)}  sorted_cols_rows={sorted_rows}/{len(nnz_per_row)}")
    print(f"nnz/row: mean={nnz_per_row.mean():.0f} p50={pct(nnz_per_row,50):.0f} "
          f"p95={pct(nnz_per_row,95):.0f} max={nnz_per_row.max()}")
    print(f"consecutive-column runs: runs/row mean={runs_per_row.mean():.0f}, "
          f"run length mean={run_lens.mean():.2f} p50={pct(run_lens,50):.0f} "
          f"p90={pct(run_lens,90):.0f} p99={pct(run_lens,99):.0f} max={run_lens.max()}")
    print(f"  fraction of nnz in runs>=4: {run_lens[run_lens>=4].sum()/run_lens.sum():.3f}; "
          f">=8: {run_lens[run_lens>=8].sum()/run_lens.sum():.3f}; "
          f">=16: {run_lens[run_lens>=16].sum()/run_lens.sum():.3f}")
    print(f"distinct (z,y) lines/row mean={np.mean(lines_per_row):.0f}  "
          f"z-span fraction: p50={pct(zspan_frac,50):.3f} p90={pct(zspan_frac,90):.3f}")
    print(f"column deltas (sorted): ==1 {np.mean(deltas==1):.3f}  <=255 {np.mean(deltas<=255):.3f}  "
          f"<=65535 {np.mean(deltas<=65535):.3f}  max={deltas.max()}  (min={deltas.min()})")
    for k, thr in enumerate(prune_thr):
        print(f"prune |a| < {thr:.0e}*rowmax: removes {np.mean(prune_frac[k]):.3f} of nnz, "
              f"row relL2 of removed p50={np.percentile(prune_err[k],50):.2e} max={np.max(prune_err[k]):.2e}")
    # index-encoding cost estimates (bytes per nnz)
    n = run_lens.sum()
    seg_cost = (len(run_lens) * 6) / n           # int32 start + uint16 len per run
    seg_cost_delta = (len(run_lens) * 4) / n     # uint16 delta-start + uint16 len per run (approx)
    d8 = np.mean(deltas <= 254)
    vbyte = (d8 * 1 + (1 - d8) * 5)
    print(f"index bytes/nnz: int32=4.00  run(int32+u16)={seg_cost:.2f}  "
          f"run(u16+u16)={seg_cost_delta:.2f}  varbyte-delta~{vbyte:.2f}")
    print(f"\nvalues: min>0={vmin_all:.3e} max={vmax_all:.3e}  "
          f"frac<fp16_normal_min(6.1e-5)={n_lt_fp16_norm/n_total:.4f}  "
          f"frac<fp16_subnormal_min(6e-8)={n_lt_fp16_sub/n_total:.5f}")
    print(f"per-row max/min: p50={pct(row_dyn,50):.1f} p90={pct(row_dyn,90):.1f} max={np.max(row_dyn):.1f}")
    print(f"per-row rel L2 quantization error: fp16 p50={pct(rel_err_fp16,50):.2e} max={np.max(rel_err_fp16):.2e} | "
          f"row-scaled u16 p50={pct(rel_err_bfp16,50):.2e} max={np.max(rel_err_bfp16):.2e} | "
          f"fp24(15b mant) p50={pct(rel_err_fp24,50):.2e} max={np.max(rel_err_fp24):.2e}")
    print("\nPSF-shape impact per quantizer (per-row footprint, voxel units):")
    print(f"{'quant':9s} {'relL2 p50/max':>17s} {'zeroed nnz':>11s} {'zeroed L1':>10s} {'rowsum rel (p50/max|.|)':>26s} "
          f"{'RMS radius rel (p50/max|.|)':>29s} {'centroid shift vox (max)':>25s}")
    for name in quant_names:
        zs = np.array(q_sum_rel[name]); zr = np.array(q_rms_rel[name]); rl = np.array(q_rel_l2[name])
        print(f"{name:9s} {pct(rl,50):8.2e}/{np.max(rl):.2e} {np.mean(q_zero_frac[name]):11.4f} {np.mean(q_zero_l1[name]):10.2e} "
              f"{pct(zs,50):+12.2e}/{np.max(np.abs(zs)):.2e} "
              f"{pct(zr,50):+15.2e}/{np.max(np.abs(zr)):.2e} "
              f"{np.max(q_cent_shift[name]):25.2e}")
    # coverage of the current chunk's column space
    print(f"\nchunk on-disk payload GiB={(8*nnz)/2**30:.2f}; with run-index + fp32 values: "
          f"{(nnz*(4+seg_cost))/2**30:.2f} GiB; run-index + 16-bit values: {(nnz*(2+seg_cost))/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
