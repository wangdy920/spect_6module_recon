#!/usr/bin/env python3
"""Validation-only CTAC pinhole-center reference sampler."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval_mhd_stats import read_mhd


K_CHUNK_MAGIC = 0x42525343
K_CHUNK_VERSION = 1


@dataclass(frozen=True)
class CompactMeta:
    total_rows: int
    total_voxels: int
    compact_num_angles: int
    det_nx: int
    compact_det_z: int
    compact_rays_per_angle: int
    full_num_angles: int
    nx: int
    ny: int
    nz: int
    voxel_size: float
    det_nz: int
    pixel_size: float
    detector_y: float
    use_90: bool
    use_z: bool
    use_45: bool
    sym_code: int
    symmetry_order: int

    @property
    def mode(self) -> str:
        # load_metadata validates the explicit profile; dimensions can be
        # ambiguous for tiny fixtures and must not override that declaration.
        return "90z" if self.use_z else "90"


@dataclass(frozen=True)
class ChunkInfo:
    path: Path
    rows: int
    angle_offset: int
    angle_count: int
    nnz: int


def read_mhd_spacing(path: Path) -> tuple[float, float, float] | None:
    header = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        header[key.strip()] = value.strip()
    raw = header.get("ElementSpacing") or header.get("ElementSize")
    if not raw:
        return None
    vals = tuple(float(v) for v in raw.split())
    if len(vals) != 3:
        return None
    return vals


def load_metadata(path: Path) -> CompactMeta:
    data = path.read_bytes()
    if len(data) in (88, 92):
        values = struct.unpack("<11i" + ("f" if len(data) == 88 else "d") + "2i3f5i", data)
    elif len(data) == 124 and struct.unpack_from("<II", data) == (0x444D4D53, 3):
        raw_voxel = data[52:84]
        if b"\0" not in raw_voxel:
            raise ValueError("SMMD voxel text is not NUL terminated")
        voxel = float(raw_voxel.split(b"\0", 1)[0].decode("ascii"))
        values = (*struct.unpack_from("<11i", data, 8), voxel,
                  *struct.unpack_from("<2i3f5i", data, 84))
    else:
        raise ValueError(f"unsupported compact metadata layout: {path}")
    (total_rows, total_voxels, _max_nnz, compact_num_angles, det_nx,
     compact_det_z, compact_rays_per_angle, full_num_angles, nx, ny, nz,
     voxel_size, det_nx_duplicate, det_nz, pixel_size, detector_y,
     _thickness, use_90, use_z, use_45, sym_code, symmetry_order) = values
    voxel_size = float(np.float32(voxel_size))
    if (not math.isfinite(voxel_size) or voxel_size <= 0 or nx != ny
            or min(nx, ny, nz, det_nx, compact_det_z, full_num_angles) <= 0
            or det_nx != det_nx_duplicate or total_voxels != nx * ny * nz
            or use_90 != 1 or use_z not in (0, 1) or use_45 != 0
            or sym_code != 90 or symmetry_order != 4
            or full_num_angles != 4 * compact_num_angles
            or det_nz != compact_det_z * (2 if use_z else 1)
            or compact_rays_per_angle != det_nx * compact_det_z
            or total_rows != compact_num_angles * compact_rays_per_angle):
        raise ValueError(f"invalid ROT90/ROT90_Z geometry: {path}")
    return CompactMeta(
        total_rows,
        total_voxels,
        compact_num_angles,
        det_nx,
        compact_det_z,
        compact_rays_per_angle,
        full_num_angles,
        nx,
        ny,
        nz,
        voxel_size,
        det_nz,
        pixel_size,
        detector_y,
        use_90,
        use_z,
        use_45,
        sym_code,
        symmetry_order,
    )


def read_chunk_header(path: Path) -> ChunkInfo:
    with path.open("rb") as f:
        magic, version, rows, angle_offset, angle_count, nnz = struct.unpack(
            "<IIiiiQ", f.read(28)
        )
    if magic != K_CHUNK_MAGIC or version != K_CHUNK_VERSION:
        raise ValueError(f"unsupported chunk header: {path}")
    return ChunkInfo(path, rows, angle_offset, angle_count, int(nnz))


def discover_chunks(chunk_dir: Path) -> list[ChunkInfo]:
    chunks = [read_chunk_header(p) for p in chunk_dir.glob("chunk_*.csrb")]
    chunks.sort(key=lambda c: (c.angle_offset, c.path.name))
    if not chunks:
        raise ValueError(f"no chunk_*.csrb files under {chunk_dir}")
    return chunks


def branches(meta: CompactMeta) -> list[tuple[int, bool]]:
    sym_count = 8 if meta.mode.startswith("45") else 4 if meta.mode.startswith("90") else 1
    z_count = 2 if ("z" in meta.mode and meta.compact_det_z < meta.det_nz) else 1
    return [(s, bool(z)) for s in range(sym_count) for z in range(z_count)]


def full_angle(meta: CompactMeta, compact_angle: int, sym_id: int) -> int | None:
    if compact_angle < 0 or compact_angle >= meta.compact_num_angles:
        return None
    if meta.mode.startswith("45"):
        q = meta.full_num_angles // 8
        if sym_id & 1:
            return None if compact_angle <= 0 else sym_id * q + (q - compact_angle)
        return None if compact_angle >= q else sym_id * q + compact_angle
    if meta.mode.startswith("90"):
        return compact_angle + sym_id * meta.compact_num_angles
    return compact_angle


def map_voxel(meta: CompactMeta, col: int, sym_id: int, z_mirror: bool) -> int:
    plane = meta.nx * meta.ny
    z = col // plane
    rem = col - z * plane
    y = rem // meta.nx
    x = rem - y * meta.nx
    if meta.mode.startswith("45"):
        maps = {
            1: (y, x),
            2: (meta.nx - 1 - y, x),
            3: (meta.nx - 1 - x, y),
            4: (meta.nx - 1 - x, meta.ny - 1 - y),
            5: (meta.nx - 1 - y, meta.ny - 1 - x),
            6: (y, meta.ny - 1 - x),
            7: (x, meta.ny - 1 - y),
        }
    else:
        maps = {
            1: (meta.nx - 1 - y, x),
            2: (meta.nx - 1 - x, meta.ny - 1 - y),
            3: (y, meta.ny - 1 - x),
        }
    x, y = maps.get(sym_id, (x, y))
    if z_mirror:
        z = meta.nz - 1 - z
    return z * plane + y * meta.nx + x


def voxel_zyx(index: int, nx: int, ny: int) -> tuple[int, int, int]:
    plane = nx * ny
    z = index // plane
    rem = index - z * plane
    y = rem // nx
    x = rem - y * nx
    return z, y, x


def bilinear(slice_: np.ndarray, fx: float, fy: float) -> float:
    ny, nx = slice_.shape
    if fx <= -1.0 or fy <= -1.0 or fx >= nx or fy >= ny:
        return 0.0
    cx = min(max(fx, 0.0), nx - 1.0)
    cy = min(max(fy, 0.0), ny - 1.0)
    x0 = int(math.floor(cx))
    y0 = int(math.floor(cy))
    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)
    tx = cx - x0
    ty = cy - y0
    return float(
        (1 - ty) * ((1 - tx) * slice_[y0, x0] + tx * slice_[y0, x1])
        + ty * ((1 - tx) * slice_[y1, x0] + tx * slice_[y1, x1])
    )


def current_parallel_integral(
    mu: np.ndarray,
    x: int,
    y: int,
    z: int,
    angle: int,
    num_angles: int,
    spacing: tuple[float, float, float],
    rotation_sign: int,
) -> float:
    dx, dy, _dz = spacing
    theta = rotation_sign * angle * (2.0 * math.pi / num_angles)
    ray_x = -math.sin(theta)
    ray_y = math.cos(theta)
    step = 0.5 * min(dx, dy)
    step_fx = ray_x * step / dx
    step_fy = ray_y * step / dy
    max_len = math.hypot(mu.shape[2] * dx, mu.shape[1] * dy)
    max_steps = int(math.ceil(max_len / step)) + 2
    fx = x + 0.5 * step_fx
    fy = y + 0.5 * step_fy
    total = 0.0
    for _ in range(max_steps):
        if fx <= -1.0 or fy <= -1.0 or fx >= mu.shape[2] or fy >= mu.shape[1]:
            break
        v = bilinear(mu[z], fx, fy)
        if v > 0.0:
            total += v * step
        fx += step_fx
        fy += step_fy
    return total


def voxel_center_mm(
    x: int, y: int, z: int, shape: tuple[int, int, int], spacing: tuple[float, float, float]
) -> np.ndarray:
    nz, ny, nx = shape
    dx, dy, dz = spacing
    return np.array(
        [
            (x - 0.5 * (nx - 1)) * dx,
            (y - 0.5 * (ny - 1)) * dy,
            (z - 0.5 * (nz - 1)) * dz,
        ],
        dtype=np.float64,
    )


def pinhole_center_mm(
    angle: int,
    num_angles: int,
    detector_y: float,
    rotation_sign: int,
    pinhole_x: float,
    pinhole_z: float,
) -> np.ndarray:
    theta = rotation_sign * angle * (2.0 * math.pi / num_angles)
    return np.array(
        [
            pinhole_x * math.cos(theta) - detector_y * math.sin(theta),
            pinhole_x * math.sin(theta) + detector_y * math.cos(theta),
            pinhole_z,
        ],
        dtype=np.float64,
    )


def siddon_integral(
    mu: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    nz, ny, nx = mu.shape
    dx, dy, dz = spacing
    mins = np.array([-0.5 * nx * dx, -0.5 * ny * dy, -0.5 * nz * dz])
    maxs = np.array([0.5 * nx * dx, 0.5 * ny * dy, 0.5 * nz * dz])
    delta = p1 - p0
    t0 = 0.0
    t1 = 1.0
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            if p0[axis] < mins[axis] or p0[axis] > maxs[axis]:
                return 0.0
            continue
        a = (mins[axis] - p0[axis]) / delta[axis]
        b = (maxs[axis] - p0[axis]) / delta[axis]
        t0 = max(t0, min(a, b))
        t1 = min(t1, max(a, b))
    if t1 <= t0:
        return 0.0

    ts = [t0, t1]
    for axis, (count, step) in enumerate(((nx, dx), (ny, dy), (nz, dz))):
        if abs(delta[axis]) < 1e-12:
            continue
        lo = mins[axis]
        for k in range(1, count):
            t = (lo + k * step - p0[axis]) / delta[axis]
            if t0 < t < t1:
                ts.append(float(t))
    ts = sorted(set(round(t, 15) for t in ts))
    length = float(np.linalg.norm(delta))
    total = 0.0
    for a, b in zip(ts, ts[1:]):
        if b <= a:
            continue
        mid = p0 + ((a + b) * 0.5) * delta
        ix = int(math.floor((mid[0] - mins[0]) / dx))
        iy = int(math.floor((mid[1] - mins[1]) / dy))
        iz = int(math.floor((mid[2] - mins[2]) / dz))
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            v = float(mu[iz, iy, ix])
            if v > 0.0:
                total += v * length * (b - a)
    return total


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = q * float(np.sum(weights))
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def summarize(
    rows: list[dict],
    rms_threshold: float,
    p95_threshold: float,
    min_active_samples: int,
) -> dict:
    delta = np.array([abs(r["delta_i"]) for r in rows], dtype=np.float64)
    weights = np.array([abs(r["weight"]) for r in rows], dtype=np.float64)
    weights = np.maximum(weights, 1e-30)
    active = np.array(
        [max(abs(r["current_i"]), abs(r["pinhole_i"])) > 1e-12 for r in rows],
        dtype=bool,
    )
    rms = math.sqrt(float(np.sum(weights * delta * delta) / np.sum(weights)))
    p95 = weighted_quantile(delta, weights, 0.95)
    if int(np.count_nonzero(active)) >= min_active_samples:
        active_delta = delta[active]
        active_weights = weights[active]
        active_rms = math.sqrt(
            float(np.sum(active_weights * active_delta * active_delta) / np.sum(active_weights))
        )
        active_p95 = weighted_quantile(active_delta, active_weights, 0.95)
        decision = (
            "keep"
            if active_rms <= rms_threshold and active_p95 <= p95_threshold
            else "investigate"
        )
    else:
        active_rms = 0.0
        active_p95 = 0.0
        decision = "blocked_insufficient_active_samples"
    out = {
        "sample_count": len(rows),
        "active_sample_count": int(np.count_nonzero(active)),
        "weighted_abs_delta_i_mean": float(np.sum(weights * delta) / np.sum(weights)),
        "weighted_abs_delta_i_rms": rms,
        "weighted_abs_delta_i_p95": p95,
        "active_weighted_abs_delta_i_rms": active_rms,
        "active_weighted_abs_delta_i_p95": active_p95,
        "max_abs_delta_i": float(np.max(delta)) if len(delta) else 0.0,
        "decision": decision,
        "thresholds": {"rms": rms_threshold, "p95": p95_threshold},
        "min_active_samples": min_active_samples,
        "branch": {},
    }
    for branch in sorted({r["branch"] for r in rows}):
        part = [r for r in rows if r["branch"] == branch]
        d = np.array([abs(r["delta_i"]) for r in part], dtype=np.float64)
        w = np.maximum(np.array([abs(r["weight"]) for r in part], dtype=np.float64), 1e-30)
        out["branch"][branch] = {
            "sample_count": len(part),
            "weighted_abs_delta_i_rms": math.sqrt(float(np.sum(w * d * d) / np.sum(w))),
            "weighted_abs_delta_i_p95": weighted_quantile(d, w, 0.95),
        }
    return out


def select_rows(rows: int, count: int) -> np.ndarray:
    if count <= 0 or rows <= count:
        return np.arange(rows, dtype=np.int64)
    return np.unique(np.linspace(0, rows - 1, count, dtype=np.int64))


def col_has_active_branch(
    meta: CompactMeta,
    col: int,
    compact_angle: int,
    branch_list: list[tuple[int, bool]],
    active_flat: np.ndarray,
) -> bool:
    for sym_id, z_mirror in branch_list:
        if full_angle(meta, compact_angle, sym_id) is None:
            continue
        if active_flat[map_voxel(meta, col, sym_id, z_mirror)]:
            return True
    return False


def run(args: argparse.Namespace) -> dict:
    meta = load_metadata(args.metadata)
    mu = read_mhd(args.mu_map).astype(np.float64, copy=False)
    spacing = read_mhd_spacing(args.mu_map) or (
        float(meta.voxel_size),
        float(meta.voxel_size),
        float(meta.voxel_size),
    )
    if mu.shape != (meta.nz, meta.ny, meta.nx):
        raise ValueError(f"mu-map shape {mu.shape} != metadata {(meta.nz, meta.ny, meta.nx)}")
    active_flat = (mu.ravel() > args.active_mu_min)

    chunks = discover_chunks(args.chunk_dir)
    out_rows: list[dict] = []
    branch_list = branches(meta)
    detector_y = float(args.detector_y)

    for chunk in chunks:
        row_ptr = np.memmap(chunk.path, dtype="<i8", mode="r", offset=28, shape=(chunk.rows + 1,))
        col_off = 28 + (chunk.rows + 1) * 8
        val_off = col_off + chunk.nnz * 4
        col_idx = np.memmap(chunk.path, dtype="<u4", mode="r", offset=col_off, shape=(chunk.nnz,))
        values = np.memmap(chunk.path, dtype="<f4", mode="r", offset=val_off, shape=(chunk.nnz,))
        for local_row in select_rows(chunk.rows, args.rows_per_chunk):
            start = int(row_ptr[local_row])
            end = int(row_ptr[local_row + 1])
            if end <= start:
                continue
            compact_row = chunk.angle_offset * meta.compact_rays_per_angle + int(local_row)
            compact_angle = compact_row // meta.compact_rays_per_angle
            if args.active_voxel_only:
                rel_candidates = [
                    rel
                    for rel, col in enumerate(col_idx[start:end])
                    if col_has_active_branch(
                        meta, int(col), compact_angle, branch_list, active_flat
                    )
                ]
                if not rel_candidates:
                    continue
                rel_candidates = np.array(rel_candidates, dtype=np.int64)
                row_values = values[start + rel_candidates]
                k = min(args.nnz_per_row, row_values.size)
                pick = rel_candidates[np.argpartition(np.abs(row_values), -k)[-k:]]
                pick = pick[np.argsort(-np.abs(values[start + pick]))]
            else:
                k = min(args.nnz_per_row, end - start)
                row_values = values[start:end]
                pick = np.argpartition(np.abs(row_values), -k)[-k:]
                pick = pick[np.argsort(-np.abs(row_values[pick]))]
            for rel in pick:
                weight = float(values[start + int(rel)])
                col = int(col_idx[start + int(rel)])
                for sym_id, z_mirror in branch_list:
                    angle = full_angle(meta, compact_angle, sym_id)
                    if angle is None:
                        continue
                    mapped = map_voxel(meta, col, sym_id, z_mirror)
                    z, y, x = voxel_zyx(mapped, meta.nx, meta.ny)
                    current_i = current_parallel_integral(
                        mu, x, y, z, angle, meta.full_num_angles, spacing, args.rotation_sign
                    )
                    p0 = voxel_center_mm(x, y, z, mu.shape, spacing)
                    p1 = pinhole_center_mm(
                        angle,
                        meta.full_num_angles,
                        detector_y,
                        args.rotation_sign,
                        args.pinhole_x,
                        args.pinhole_z,
                    )
                    pinhole_i = siddon_integral(mu, p0, p1, spacing)
                    out_rows.append(
                        {
                            "chunk": chunk.path.name,
                            "local_row": int(local_row),
                            "compact_row": compact_row,
                            "compact_angle": compact_angle,
                            "full_angle": angle,
                            "branch": f"sym{sym_id}_z{1 if z_mirror else 0}",
                            "col": col,
                            "mapped_col": mapped,
                            "x": x,
                            "y": y,
                            "z": z,
                            "weight": weight,
                            "current_i": current_i,
                            "pinhole_i": pinhole_i,
                            "delta_i": pinhole_i - current_i,
                        }
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "samples.csv"
    if out_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
    summary = summarize(
        out_rows, args.rms_threshold, args.p95_threshold, args.min_active_samples
    )
    summary.update(
        {
            "mu_map": str(args.mu_map),
            "chunk_dir": str(args.chunk_dir),
            "metadata": str(args.metadata),
            "mode": meta.mode,
            "detector_y_mm": detector_y,
            "pinhole_x_mm": args.pinhole_x,
            "pinhole_z_mm": args.pinhole_z,
            "rows_per_chunk": args.rows_per_chunk,
            "nnz_per_row": args.nnz_per_row,
            "active_voxel_only": args.active_voxel_only,
            "active_mu_min": args.active_mu_min,
            "samples_csv": str(csv_path),
        }
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def geometry_probe(args: argparse.Namespace) -> dict:
    """Geometry-only comparison: no inferred per-pinhole matrix weights."""
    meta = load_metadata(args.metadata)
    audit = args.geometry_audit.read_text(encoding="utf-8")
    pins = [(int(i), np.array([float(v) for v in xyz.split(",")]))
            for i, xyz in re.findall(r"pinhole=(\d+) .*?pm_center_mm=\(([^)]+)\)", audit)]
    if (not pins or len({i for i, _ in pins}) != len(pins)
            or any(p.shape != (3,) or not np.isfinite(p).all() for _, p in pins)):
        raise ValueError("geometry audit must contain finite pinhole pm centers")
    shape = (meta.nz, meta.ny, meta.nx)
    spacing = (meta.voxel_size,) * 3
    z, y, x = np.indices(shape, dtype=np.float64)
    x = (x - (meta.nx - 1) / 2) * spacing[0]
    y = (y - (meta.ny - 1) / 2) * spacing[1]
    z = (z - (meta.nz - 1) / 2) * spacing[2]
    radius = min(meta.nx, meta.ny) * spacing[0] * 0.45
    half_height = meta.nz * spacing[2] * 0.45
    support = (x*x + y*y <= radius*radius) & (abs(z) <= half_height)
    water = support * 0.0152
    heterogeneous = water.copy()
    heterogeneous[support & ((x - radius * 0.25)**2 + y*y < (radius * 0.2)**2)] = 0.03
    rows = []
    for phantom, mu in (("water_cylinder", water), ("dense_insert", heterogeneous)):
        for target_z in (-half_height * 0.9, 0.0, half_height * 0.9):
            for target_x, target_y in ((0, 0), (radius * .75, 0), (-radius * .75, 0),
                                     (0, radius * .75), (0, -radius * .75)):
                ix, iy, iz = (int(round(t / s + (n - 1) / 2))
                              for t, s, n in zip((target_x, target_y, target_z), spacing,
                                                  (meta.nx, meta.ny, meta.nz)))
                p0 = voxel_center_mm(ix, iy, iz, shape, spacing)
                for angle in range(0, meta.full_num_angles, max(1, meta.full_num_angles // 8)):
                    current = current_parallel_integral(mu, ix, iy, iz, int(angle),
                                                        meta.full_num_angles, spacing, args.rotation_sign)
                    theta = args.rotation_sign * int(angle) * 2 * math.pi / meta.full_num_angles
                    direction = np.array([-math.sin(theta), math.cos(theta), 0.0])
                    parallel = siddon_integral(mu, p0, p0 + direction * 1000, spacing)
                    for pin_id, pin in pins:
                        p1 = pinhole_center_mm(int(angle), meta.full_num_angles, pin[1],
                                               args.rotation_sign, pin[0], pin[2])
                        pinhole = siddon_integral(mu, p0, p1, spacing)
                        rows.append(dict(phantom=phantom, pinhole=pin_id, angle=int(angle),
                                         x_mm=p0[0], y_mm=p0[1], z_mm=p0[2],
                                         current_i=current, parallel_siddon_i=parallel, pinhole_i=pinhole,
                                         current_factor=math.exp(-current), pinhole_factor=math.exp(-pinhole),
                                         geometry_relative_factor_error=math.expm1(pinhole-parallel),
                                         total_relative_factor_error=math.expm1(pinhole-current)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "geometry_samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"evidence": "actual supplied pinhole centers; synthetic mu; unweighted path comparison",
               "decision": "descriptive_only_not_image_quality_or_adoption",
               "metadata": str(args.metadata), "geometry_audit": str(args.geometry_audit),
               "shape_zyx": shape, "spacing_mm": spacing, "radius_mm": radius,
               "half_height_mm": half_height, "mu_per_mm": [0.0152, 0.03], "samples": len(rows),
               "limitations": "No per-pinhole contribution weights, real CT, or reconstructed-image error.",
               "phantoms": {}}
    for phantom in ("water_cylinder", "dense_insert"):
        group = [r for r in rows if r["phantom"] == phantom]
        result = {}
        for field in ("geometry_relative_factor_error", "total_relative_factor_error"):
            errors = np.array([r[field] for r in group])
            result[field] = {"median_abs": float(np.median(abs(errors))),
                             "p95_abs": float(np.quantile(abs(errors), .95)),
                             "max_abs": float(max(abs(errors)))}
        summary["phantoms"][phantom] = result
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def self_test() -> None:
    mu = np.ones((3, 3, 3), dtype=np.float64)
    spacing = (1.0, 1.0, 1.0)
    assert abs(siddon_integral(mu, np.array([0.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]), spacing) - 1.5) < 1e-12
    assert abs(current_parallel_integral(mu, 1, 1, 1, 0, 4, spacing, 1) - 2.0) < 1e-12
    class M:
        nx = 3
        ny = 3
        nz = 3
        mode = "90z"
    assert map_voxel(M, 13, 1, False) == 13
    # Physical half-voxel boundary anchor, independent of the sampled integrator.
    assert abs(siddon_integral(mu, np.zeros(3), np.array([2., 2., 0.]), spacing)
               - 1.5 * math.sqrt(2)) < 1e-12
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "metadata.bin"
        prefix = (16, 64, 64, 2, 2, 4, 8, 8, 4, 4, 4)
        suffix = (2, 4, 1., 29., 9., 1, 0, 0, 90, 4)
        layouts = [struct.pack("<11if2i3f5i", *prefix, .4, *suffix),
                   struct.pack("<11id2i3f5i", *prefix, .4, *suffix),
                   struct.pack("<II11i32s2i3f5i", 0x444D4D53, 3,
                               *prefix, b"0.4", *suffix)]
        for data in layouts:
            path.write_bytes(data)
            meta = load_metadata(path)
            assert meta.mode == "90" and meta.total_rows == 16
            assert meta.voxel_size == float(np.float32(.4))
        path.write_bytes(layouts[-1][:-1])
        try:
            load_metadata(path)
        except ValueError:
            pass
        else:
            raise AssertionError("truncated metadata accepted")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--chunk-dir", type=Path, default=Path("test_data/6module/260601"))
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--geometry-audit", type=Path,
                   help="Probe supplied pm centers on synthetic mu phantoms; no matrix payload read")
    p.add_argument("--mu-map", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("recon_out/ctac_pinhole_reference_validation"))
    p.add_argument("--rows-per-chunk", type=int, default=8)
    p.add_argument("--nnz-per-row", type=int, default=4)
    p.add_argument("--active-voxel-only", action="store_true")
    p.add_argument("--active-mu-min", type=float, default=1e-12)
    p.add_argument("--detector-y", type=float, default=None,
                   help="Legacy option name: explicit PINHOLE-plane Y in mm, not detector-plane Y")
    p.add_argument("--pinhole-x", type=float, default=0.0)
    p.add_argument("--pinhole-z", type=float, default=0.0)
    p.add_argument("--rotation-sign", type=int, choices=(-1, 1), default=1)
    p.add_argument("--rms-threshold", type=float, default=0.02)
    p.add_argument("--p95-threshold", type=float, default=0.05)
    p.add_argument("--min-active-samples", type=int, default=16)
    args = p.parse_args()
    if args.self_test:
        return args
    if args.metadata is None:
        args.metadata = args.chunk_dir / "metadata.bin"
    if args.mu_map is None and args.geometry_audit is None:
        p.error("--mu-map or --geometry-audit is required")
    if args.geometry_audit is None and (args.detector_y is None or not math.isfinite(args.detector_y)):
        p.error("--detector-y must explicitly supply finite pinhole-plane Y; metadata detectorY is not a pinhole center")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return 0
    if args.geometry_audit is not None:
        print(json.dumps(geometry_probe(args), indent=2))
        return 0
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["decision"] == "keep" else 2


if __name__ == "__main__":
    raise SystemExit(main())
