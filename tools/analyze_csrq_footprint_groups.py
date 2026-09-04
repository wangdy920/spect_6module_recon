"""Measure what a row-group (P3) backward kernel would gain on a CSRQ chunk.

Rows of one compact angle are ordered (detZ, detX); the G consecutive rows of
one (angle, detZ) group are adjacent detector pixels whose footprints overlap.
For sampled groups and every z slice this script reports the union bounding
box of the G footprints, the nnz inside it, the distinct voxels touched and
hence the factor by which per-voxel shared-memory accumulation would cut the
global atomics of the current per-nnz scatter.

Usage:
  python tools/analyze_csrq_footprint_groups.py <csrq_dir> [--chunk 0]
         [--group-rows 16] [--groups 24] [--nx 120 --ny 120 --nz 220]
         [--rays-per-angle 1152 --det-nx 48]
"""
import argparse
from pathlib import Path

import numpy as np

CSRQ_HEADER = 48
RUN_LEN_BITS = 6
MAX_RUN_LEN = 1 << RUN_LEN_BITS


def load_csrq(path: Path):
    hdr = path.read_bytes()[:CSRQ_HEADER]
    rows, ang_off, ang_cnt, cols = (int(x) for x in np.frombuffer(hdr[8:24], dtype=np.int32))
    nnz, run_count = (int(x) for x in np.frombuffer(hdr[24:40], dtype=np.uint64))
    off = CSRQ_HEADER
    row_nnz_ptr = np.fromfile(path, dtype=np.uint32, count=rows + 1, offset=off); off += 4 * (rows + 1)
    row_run_ptr = np.fromfile(path, dtype=np.uint32, count=rows + 1, offset=off); off += 4 * (rows + 1)
    off += 4 * rows  # row_max
    off += 2 * nnz + (4 - (2 * nnz) % 4) % 4  # values (+pad)
    runs = np.memmap(path, dtype=np.uint32, mode="r", offset=off, shape=(run_count,))
    return rows, ang_off, ang_cnt, cols, row_nnz_ptr, row_run_ptr, runs


def group_stats(runs, row_run_ptr, rows_in_group, nx, ny, nz):
    """Per z slice: union bbox (dy, dx), nnz, distinct voxels, smem bytes."""
    plane = nx * ny
    starts = []
    lens = []
    for r in rows_in_group:
        w = np.asarray(runs[int(row_run_ptr[r]):int(row_run_ptr[r + 1])], dtype=np.int64)
        starts.append(w >> RUN_LEN_BITS)
        lens.append((w & (MAX_RUN_LEN - 1)) + 1)
    start = np.concatenate(starts)
    length = np.concatenate(lens)
    if start.size == 0:
        return []
    z = start // plane
    rem = start - z * plane
    y = rem // nx
    x0 = rem - y * nx
    x1 = x0 + length - 1
    out = []
    for zz in np.unique(z):
        m = z == zz
        ys = y[m]
        xs0 = x0[m]
        xs1 = x1[m]
        ls = length[m]
        dy = int(ys.max() - ys.min() + 1)
        dx = int(xs1.max() - xs0.min() + 1)
        nnz = int(ls.sum())
        # distinct voxels via a dense mask of the bbox
        mask = np.zeros((dy, dx), dtype=bool)
        yy = ys - ys.min()
        for a, b, c in zip(yy, xs0 - xs0.min(), xs1 - xs0.min()):
            mask[a, b:c + 1] = True
        distinct = int(mask.sum())
        out.append((int(zz), dy, dx, nnz, distinct))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csrq_dir")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--group-rows", type=int, default=16)
    ap.add_argument("--groups", type=int, default=24)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--ny", type=int, default=120)
    ap.add_argument("--nz", type=int, default=220)
    ap.add_argument("--rays-per-angle", type=int, default=1152)
    ap.add_argument("--det-nx", type=int, default=48)
    ap.add_argument("--angle-slices", type=int, default=0,
                    help="also evaluate the (angle, z-slice) decomposition on this many angles")
    a = ap.parse_args()

    path = Path(a.csrq_dir) / f"chunk_{a.chunk:03d}.csrq"
    rows, ang_off, ang_cnt, cols, row_nnz_ptr, row_run_ptr, runs = load_csrq(path)
    assert cols == a.nx * a.ny * a.nz, (cols, a.nx * a.ny * a.nz)
    det_nz = a.rays_per_angle // a.det_nx
    rng = np.random.default_rng(3)
    G = a.group_rows
    groups_per_detrow = a.det_nx // G
    print(f"{path.name}: rows={rows} angles={ang_off}..{ang_off + ang_cnt - 1} "
          f"G={G} ({groups_per_detrow} groups per detector row)")
    print(f"{'angle':>5s} {'detZ':>4s} {'g':>2s} | {'zslices':>7s} {'nnz':>9s} {'distinct':>9s} "
          f"{'atomic_x':>8s} | {'bbox_max dy x dx':>16s} {'smem_max_KB(2br)':>16s} {'smem_p95_KB':>11s}")
    tot_nnz = 0
    tot_distinct = 0
    smem_all = []
    for _ in range(a.groups):
        angle = int(rng.integers(0, ang_cnt))
        det_z = int(rng.integers(0, det_nz))
        g = int(rng.integers(0, groups_per_detrow))
        base = angle * a.rays_per_angle + det_z * a.det_nx + g * G
        rows_in_group = list(range(base, base + G))
        stats = group_stats(runs, row_run_ptr, rows_in_group, a.nx, a.ny, a.nz)
        if not stats:
            continue
        nnz = sum(s[3] for s in stats)
        distinct = sum(s[4] for s in stats)
        dy_max = max(s[1] for s in stats)
        dx_max = max(s[2] for s in stats)
        smem = np.array([s[1] * s[2] * 2 * 4 / 1024.0 for s in stats])
        smem_all.extend(smem.tolist())
        tot_nnz += nnz
        tot_distinct += distinct
        print(f"{angle + ang_off:5d} {det_z:4d} {g:2d} | {len(stats):7d} {nnz:9d} {distinct:9d} "
              f"{nnz / max(distinct, 1):8.2f} | {dy_max:7d} x {dx_max:<6d} {smem.max():16.1f} "
              f"{np.percentile(smem, 95):11.1f}")
    smem_all = np.array(smem_all)
    print(f"\noverall: nnz/distinct (global atomics reduction) = {tot_nnz / max(tot_distinct, 1):.2f}x; "
          f"smem per (group, z) tile: max {smem_all.max():.1f} KB, p99 {np.percentile(smem_all, 99):.1f} KB, "
          f"median {np.median(smem_all):.1f} KB (2 branches, fp32)")

    if a.angle_slices > 0:
        angle_slice_stats(runs, row_run_ptr, ang_cnt, a)
    return 0


def angle_slice_stats(runs, row_run_ptr, ang_cnt, a):
    """Alternative decomposition: one block per (angle, z slice) over all rows
    of the angle, accumulating into a whole-slice tile (nx*ny fp32)."""
    plane = a.nx * a.ny
    rng = np.random.default_rng(11)
    print(f"\n(angle, z-slice) decomposition, tile = whole slice = {plane * 4 / 1024:.1f} KB per branch")
    print(f"{'angle':>5s} | {'rows/slice':>10s} {'runs/slice':>10s} {'nnz/slice':>10s} {'distinct':>9s} "
          f"{'atomic_x':>8s} | {'max nnz':>8s} {'max runs':>8s} {'slices>0':>8s}")
    for _ in range(a.angle_slices):
        angle = int(rng.integers(0, ang_cnt))
        base = angle * a.rays_per_angle
        # gather every run of the angle with its row id
        starts, lens, rows_of = [], [], []
        for r in range(base, base + a.rays_per_angle):
            w = np.asarray(runs[int(row_run_ptr[r]):int(row_run_ptr[r + 1])], dtype=np.int64)
            starts.append(w >> RUN_LEN_BITS)
            lens.append((w & (MAX_RUN_LEN - 1)) + 1)
            rows_of.append(np.full(w.size, r, dtype=np.int64))
        start = np.concatenate(starts)
        length = np.concatenate(lens)
        row = np.concatenate(rows_of)
        z = start // plane
        rem = start - z * plane
        per_z_nnz, per_z_runs, per_z_rows, per_z_distinct = [], [], [], []
        for zz in np.unique(z):
            m = z == zz
            per_z_nnz.append(int(length[m].sum()))
            per_z_runs.append(int(m.sum()))
            per_z_rows.append(int(np.unique(row[m]).size))
            mask = np.zeros(plane, dtype=bool)
            for s0, ln in zip(rem[m], length[m]):
                mask[s0:s0 + ln] = True
            per_z_distinct.append(int(mask.sum()))
        n = len(per_z_nnz)
        print(f"{angle:5d} | {np.mean(per_z_rows):10.1f} {np.mean(per_z_runs):10.1f} {np.mean(per_z_nnz):10.1f} "
              f"{np.mean(per_z_distinct):9.1f} {sum(per_z_nnz) / max(sum(per_z_distinct), 1):8.2f} | "
              f"{max(per_z_nnz):8d} {max(per_z_runs):8d} {n:8d}")


if __name__ == "__main__":
    raise SystemExit(main())
