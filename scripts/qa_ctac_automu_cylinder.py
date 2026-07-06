#!/usr/bin/env python3
"""Measure a uniform cylinder and compare fitted true-mu vs auto-mu masks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_mhd_stats import read_mhd


def parse_triplet(raw: str, cast=float) -> tuple:
    parts = [p.strip() for p in raw.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"expected three comma-separated values, got: {raw}")
    return tuple(cast(p) for p in parts)


def read_activity(path: Path, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    if path.suffix.lower() == ".mhd":
        return read_mhd(path).astype(np.float64, copy=False)
    if path.suffix.lower() == ".img":
        nx, ny, nz = shape_xyz
        data = np.fromfile(path, dtype="<f8")
        expected = nx * ny * nz
        if data.size != expected:
            raise ValueError(f"{path} has {data.size} voxels, expected {expected}")
        return data.reshape((nz, ny, nx)).astype(np.float64, copy=False)
    raise ValueError(f"unsupported activity format: {path}")


def write_mhd(path: Path, image: np.ndarray, spacing_xyz: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_suffix(".raw")
    image.astype("<f4", copy=False).tofile(raw_path)
    nx, ny, nz = image.shape[2], image.shape[1], image.shape[0]
    sx, sy, sz = spacing_xyz
    path.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "NDims = 3",
                f"DimSize = {nx} {ny} {nz}",
                f"ElementSpacing = {sx} {sy} {sz}",
                "ElementType = MET_FLOAT",
                "ElementByteOrderMSB = False",
                "HeaderSize = 0",
                f"ElementDataFile = {raw_path.name}",
                "",
            ]
        ),
        encoding="ascii",
    )


def ball_offsets(radius: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    r2 = radius * radius
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy + dz * dz <= r2:
                    out.append((dz, dy, dx))
    return out


def shifted_slices(shape: tuple[int, int, int], dz: int, dy: int, dx: int):
    nz, ny, nx = shape
    src_z = slice(max(0, -dz), min(nz, nz - dz))
    src_y = slice(max(0, -dy), min(ny, ny - dy))
    src_x = slice(max(0, -dx), min(nx, nx - dx))
    dst_z = slice(max(0, dz), min(nz, nz + dz))
    dst_y = slice(max(0, dy), min(ny, ny + dy))
    dst_x = slice(max(0, dx), min(nx, nx + dx))
    return (src_z, src_y, src_x), (dst_z, dst_y, dst_x)


def morph(mask: np.ndarray, radius: int, dilate: bool) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool) if dilate else np.ones_like(mask, dtype=bool)
    for dz, dy, dx in ball_offsets(radius):
        src, dst = shifted_slices(mask.shape, dz, dy, dx)
        if dilate:
            out[dst] |= mask[src]
        else:
            out[dst] &= mask[src]
    return out


def largest_component(mask: np.ndarray) -> np.ndarray:
    # ponytail: no scipy dependency; this debug volume is small enough for BFS.
    flat = mask.ravel()
    seen = np.zeros(flat.size, dtype=bool)
    nz, ny, nx = mask.shape
    plane = ny * nx
    best: list[int] = []
    for seed in np.flatnonzero(flat):
        seed = int(seed)
        if seen[seed]:
            continue
        q: deque[int] = deque([seed])
        seen[seed] = True
        comp: list[int] = []
        while q:
            idx = q.popleft()
            comp.append(idx)
            z = idx // plane
            rem = idx - z * plane
            y = rem // nx
            x = rem - y * nx
            neighbors = []
            if x > 0:
                neighbors.append(idx - 1)
            if x + 1 < nx:
                neighbors.append(idx + 1)
            if y > 0:
                neighbors.append(idx - nx)
            if y + 1 < ny:
                neighbors.append(idx + nx)
            if z > 0:
                neighbors.append(idx - plane)
            if z + 1 < nz:
                neighbors.append(idx + plane)
            for nb in neighbors:
                if not seen[nb] and flat[nb]:
                    seen[nb] = True
                    q.append(nb)
        if len(comp) > len(best):
            best = comp
    out = np.zeros(flat.size, dtype=bool)
    if best:
        out[np.array(best, dtype=np.intp)] = True
    return out.reshape(mask.shape)


def fill_holes(body: np.ndarray) -> np.ndarray:
    background = ~body
    flat_bg = background.ravel()
    reached = np.zeros(flat_bg.size, dtype=bool)
    nz, ny, nx = body.shape
    plane = ny * nx
    seeds: list[int] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if x not in (0, nx - 1) and y not in (0, ny - 1) and z not in (0, nz - 1):
                    continue
                idx = z * plane + y * nx + x
                if flat_bg[idx] and not reached[idx]:
                    seeds.append(idx)
    for seed in seeds:
        if reached[seed]:
            continue
        q: deque[int] = deque([seed])
        reached[seed] = True
        while q:
            idx = q.popleft()
            z = idx // plane
            rem = idx - z * plane
            y = rem // nx
            x = rem - y * nx
            neighbors = []
            if x > 0:
                neighbors.append(idx - 1)
            if x + 1 < nx:
                neighbors.append(idx + 1)
            if y > 0:
                neighbors.append(idx - nx)
            if y + 1 < ny:
                neighbors.append(idx + nx)
            if z > 0:
                neighbors.append(idx - plane)
            if z + 1 < nz:
                neighbors.append(idx + plane)
            for nb in neighbors:
                if flat_bg[nb] and not reached[nb]:
                    reached[nb] = True
                    q.append(nb)
    holes = background & ~reached.reshape(body.shape)
    return body | holes


def robust_p999(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    rank = min(values.size - 1, int(0.999 * values.size))
    return float(np.partition(values, rank)[rank])


def fov_mask(shape: tuple[int, int, int], spacing_xyz: tuple[float, float, float], radius_mm: float) -> np.ndarray | None:
    if radius_mm <= 0:
        return None
    _, ny, nx = shape
    sx, sy, _ = spacing_xyz
    yy, xx = np.ogrid[:ny, :nx]
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    rr_mm = np.sqrt(((xx - cx) * sx) ** 2 + ((yy - cy) * sy) ** 2)
    return np.broadcast_to(rr_mm <= radius_mm, shape).copy()


def auto_body_mask(
    activity: np.ndarray,
    threshold_frac: float,
    closing_radius: int,
    fov: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    finite = np.isfinite(activity)
    allowed = finite & (activity > 0)
    if fov is not None:
        allowed &= fov
    positives = activity[allowed]
    p999 = robust_p999(positives)
    threshold = threshold_frac * p999
    body = allowed & (activity >= threshold)
    body = morph(body, closing_radius, dilate=True)
    body = morph(body, closing_radius, dilate=False)
    body = largest_component(body)
    body = fill_holes(body)
    return body, {"robust_p999": p999, "threshold": float(threshold)}


def fit_cylinder(mask: np.ndarray, spacing_xyz: tuple[float, float, float], area_frac: float) -> dict:
    z_area = mask.sum(axis=(1, 2)).astype(np.float64)
    max_area = float(z_area.max()) if z_area.size else 0.0
    if max_area <= 0:
        raise ValueError("empty body mask")
    active_z = np.flatnonzero(z_area >= max(1.0, area_frac * max_area))
    if active_z.size == 0:
        active_z = np.flatnonzero(z_area > 0)
    z0, z1 = int(active_z[0]), int(active_z[-1])
    core = mask[z0 : z1 + 1]
    zz, yy, xx = np.nonzero(core)
    if xx.size == 0:
        raise ValueError("empty fitted cylinder core")
    cx = float(xx.mean())
    cy = float(yy.mean())
    median_area = float(np.median(z_area[active_z]))
    radius_vox = float(np.sqrt(median_area / np.pi))
    sx, sy, sz = spacing_xyz
    return {
        "center_x_vox": cx,
        "center_y_vox": cy,
        "center_z_vox": 0.5 * (z0 + z1),
        "z_min": z0,
        "z_max": z1,
        "radius_vox": radius_vox,
        "diameter_mm": 2.0 * radius_vox * float((sx + sy) / 2.0),
        "height_mm": (z1 - z0 + 1) * sz,
        "median_area_vox": median_area,
        "active_z_count": int(active_z.size),
    }


def cylinder_mask(shape: tuple[int, int, int], fit: dict) -> np.ndarray:
    nz, ny, nx = shape
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.sqrt((xx - fit["center_x_vox"]) ** 2 + (yy - fit["center_y_vox"]) ** 2)
    xy = rr <= fit["radius_vox"]
    out = np.zeros(shape, dtype=bool)
    out[int(fit["z_min"]) : int(fit["z_max"]) + 1] = xy
    return out


def mask_metrics(auto: np.ndarray, true: np.ndarray, mu: float) -> dict:
    inter = int(np.count_nonzero(auto & true))
    auto_count = int(np.count_nonzero(auto))
    true_count = int(np.count_nonzero(true))
    fp = int(np.count_nonzero(auto & ~true))
    fn = int(np.count_nonzero(~auto & true))
    union = int(np.count_nonzero(auto | true))
    dice = 2.0 * inter / max(auto_count + true_count, 1)
    auto_mu = auto.astype(np.float64) * mu
    true_mu = true.astype(np.float64) * mu
    diff = auto_mu - true_mu
    return {
        "true_body_voxels": true_count,
        "auto_body_voxels": auto_count,
        "intersection_voxels": inter,
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "false_positive_fraction_of_true": fp / max(true_count, 1),
        "false_negative_fraction_of_true": fn / max(true_count, 1),
        "dice": dice,
        "jaccard": inter / max(union, 1),
        "mu_rmse": float(np.sqrt(np.mean(diff * diff))),
        "mu_mean_abs_diff": float(np.mean(np.abs(diff))),
        "mu_sum_rel_diff": float(abs(auto_mu.sum() - true_mu.sum()) / max(abs(true_mu.sum()), 1e-12)),
    }


def profiles(activity: np.ndarray, auto: np.ndarray, true: np.ndarray, fit: dict, spacing_xyz: tuple[float, float, float]):
    sx, sy, sz = spacing_xyz
    yy, xx = np.indices(activity.shape[1:])
    rr_mm = np.sqrt(((xx - fit["center_x_vox"]) * sx) ** 2 + ((yy - fit["center_y_vox"]) * sy) ** 2)
    z0, z1 = int(fit["z_min"]), int(fit["z_max"])
    radial_rows = []
    max_r = float(rr_mm.max())
    edges = np.arange(0.0, max_r + 0.5, 0.5)
    core_act = activity[z0 : z1 + 1]
    core_auto = auto[z0 : z1 + 1]
    core_true = true[z0 : z1 + 1]
    core_rr = np.broadcast_to(rr_mm, core_act.shape)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (core_rr >= lo) & (core_rr < hi)
        if not np.any(m):
            continue
        radial_rows.append(
            {
                "radius_mm": float((lo + hi) / 2.0),
                "activity_mean": float(core_act[m].mean()),
                "auto_fraction": float(core_auto[m].mean()),
                "true_fraction": float(core_true[m].mean()),
            }
        )
    z_rows = []
    xy_true = true.any(axis=0)
    for z in range(activity.shape[0]):
        m = xy_true
        z_rows.append(
            {
                "z": int(z),
                "z_mm_from_first_slice": float(z * sz),
                "activity_mean_in_true_xy": float(activity[z][m].mean()) if np.any(m) else 0.0,
                "auto_area_voxels": int(auto[z].sum()),
                "true_area_voxels": int(true[z].sum()),
            }
        )
    return radial_rows, z_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="ascii")
        return
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_views(path: Path, activity: np.ndarray, auto: np.ndarray, true: np.ndarray, fit: dict, radial_rows: list[dict], z_rows: list[dict]) -> None:
    zc = (int(fit["z_min"]) + int(fit["z_max"])) // 2
    diff = auto.astype(np.int8) - true.astype(np.int8)
    vmin = float(np.percentile(activity[np.isfinite(activity)], 1.0))
    vmax = float(np.percentile(activity[np.isfinite(activity)], 99.0))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].imshow(activity[zc], origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("activity XY")
    axes[0, 1].imshow(true[zc], origin="lower", cmap="gray")
    axes[0, 1].set_title("fitted true mu mask")
    axes[0, 2].imshow(auto[zc], origin="lower", cmap="gray")
    axes[0, 2].set_title("auto mu mask")
    axes[1, 0].imshow(diff[zc], origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 0].set_title("auto - true")
    for ax in axes.ravel()[:4]:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 1].plot([r["radius_mm"] for r in radial_rows], [r["activity_mean"] for r in radial_rows])
    axes[1, 1].set_title("radial activity profile")
    axes[1, 1].set_xlabel("radius mm")
    axes[1, 2].plot([r["z_mm_from_first_slice"] for r in z_rows], [r["auto_area_voxels"] for r in z_rows], label="auto")
    axes[1, 2].plot([r["z_mm_from_first_slice"] for r in z_rows], [r["true_area_voxels"] for r in z_rows], label="true")
    axes[1, 2].set_title("z area profile")
    axes[1, 2].set_xlabel("z mm")
    axes[1, 2].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyse(args: argparse.Namespace) -> dict:
    shape_xyz = parse_triplet(args.shape, int)
    spacing_xyz = parse_triplet(args.voxel_size, float)
    activity_path = Path(args.activity)
    activity = read_activity(activity_path, shape_xyz)
    fov = fov_mask(activity.shape, spacing_xyz, args.fov_radius_mm)
    measured, measured_meta = auto_body_mask(
        activity, args.measure_threshold_frac, args.closing_radius, fov
    )
    fit = fit_cylinder(measured, spacing_xyz, args.fit_area_frac)
    true = cylinder_mask(activity.shape, fit)
    auto, auto_meta = auto_body_mask(activity, args.threshold_frac, args.closing_radius, fov)
    cyl_auto, cyl_auto_meta = auto_body_mask(
        activity, args.threshold_frac, args.closing_radius, true
    )
    metrics = {
        "activity": str(activity_path),
        "shape_zyx": list(activity.shape),
        "voxel_size_xyz_mm": list(spacing_xyz),
        "threshold_frac": float(args.threshold_frac),
        "measure_threshold_frac": float(args.measure_threshold_frac),
        "closing_radius_voxels": int(args.closing_radius),
        "water_mu_per_mm": float(args.water_mu),
        "auto_threshold": auto_meta,
        "cylinder_masked_auto_threshold": cyl_auto_meta,
        "measurement_threshold": measured_meta,
        "measured_cylinder": fit,
        "mask_match": {
            "default_auto_vs_true": mask_metrics(auto, true, args.water_mu),
            "cylinder_masked_auto_vs_true": mask_metrics(cyl_auto, true, args.water_mu),
        },
    }

    out_dir = Path(args.out_dir)
    prefix = args.prefix
    true_mu = true.astype(np.float32) * float(args.water_mu)
    auto_mu = auto.astype(np.float32) * float(args.water_mu)
    cyl_auto_mu = cyl_auto.astype(np.float32) * float(args.water_mu)
    paths = {
        "activity_mhd": out_dir / f"{prefix}_activity.mhd",
        "true_mu_mhd": out_dir / f"{prefix}_true_mu.mhd",
        "auto_mu_mhd": out_dir / f"{prefix}_auto_mu.mhd",
        "auto_mu_cylmasked_mhd": out_dir / f"{prefix}_auto_mu_cylmasked.mhd",
        "metrics_json": out_dir / f"{prefix}_metrics.json",
        "radial_profile_csv": out_dir / f"{prefix}_radial_profile.csv",
        "z_profile_csv": out_dir / f"{prefix}_z_profile.csv",
        "views_png": out_dir / f"{prefix}_views.png",
    }
    write_mhd(paths["activity_mhd"], activity, spacing_xyz)
    write_mhd(paths["true_mu_mhd"], true_mu, spacing_xyz)
    write_mhd(paths["auto_mu_mhd"], auto_mu, spacing_xyz)
    write_mhd(paths["auto_mu_cylmasked_mhd"], cyl_auto_mu, spacing_xyz)
    radial_rows, z_rows = profiles(activity, auto, true, fit, spacing_xyz)
    write_csv(paths["radial_profile_csv"], radial_rows)
    write_csv(paths["z_profile_csv"], z_rows)
    save_views(paths["views_png"], activity, auto, true, fit, radial_rows, z_rows)
    metrics["outputs"] = {k: str(v) for k, v in paths.items()}
    paths["metrics_json"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def self_test() -> int:
    nz, ny, nx = 16, 32, 32
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.sqrt((xx - 15.5) ** 2 + (yy - 15.5) ** 2)
    true = np.zeros((nz, ny, nx), dtype=bool)
    true[2:14] = rr <= 9.0
    activity = true.astype(np.float64)
    activity[6:10, rr <= 3.0] = 0.0
    activity[:, 1, 1] = 3.0
    auto, _ = auto_body_mask(activity, 0.05, 1, None)
    m = mask_metrics(auto, true, 0.0152)
    assert m["dice"] > 0.92, m
    assert auto[:, 1, 1].sum() == 0, "disconnected speckle was not removed"
    assert auto[8, 15, 15], "cold cavity was not filled"
    print("self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity", default="test_data/sys_260601_recon_ref/recon_iter050.img")
    parser.add_argument("--shape", default="108,108,80", help="x,y,z for headerless .img")
    parser.add_argument("--voxel-size", default="0.3,0.3,0.3", help="x,y,z mm")
    parser.add_argument("--threshold-frac", type=float, default=0.05)
    parser.add_argument(
        "--measure-threshold-frac",
        type=float,
        default=0.35,
        help="Higher threshold used only to fit the ideal cylinder from noisy FOV data",
    )
    parser.add_argument("--water-mu", type=float, default=0.0152)
    parser.add_argument("--closing-radius", type=int, default=2)
    parser.add_argument("--fit-area-frac", type=float, default=0.25)
    parser.add_argument("--fov-radius-mm", type=float, default=0.0)
    parser.add_argument("--out-dir", default="recon_out/ctac_automu_cylinder_debug/maps")
    parser.add_argument("--prefix", default="sys260601_ref")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    metrics = analyse(args)
    print(json.dumps(metrics, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
