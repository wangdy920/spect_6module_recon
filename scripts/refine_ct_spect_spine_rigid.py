#!/usr/bin/env python3
"""Diagnose residual CT/SPECT pose on existing SPECT-grid MHD pairs.

Production refine lives in prepare_ct_mu_map.py (CT-anchored local COM,
3-DOF only). This script compares that local pairing with independent
midline peaks and a bounded 6-DOF fit. The 1-voxel / 0.3 mm figure is a
diagnostic target, not a production adoption gate. Occupancy cannot pass
the method.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_mhd_stats import read_mhd  # noqa: E402
from prepare_ct_mu_map import (  # noqa: E402
    _caudal_is_low_z,
    _knee_deltas,
    _midline_z_lag_vox,
    _occupied_z,
    _slice_ridge_peak,
    bone_hot_masks,
    occupancy_translation_mm,
)

VOXEL_MM = 0.3
TARGET_VOX = 1.0


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def transform_points(points: np.ndarray, params: np.ndarray,
                     center: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(params[:3])
    return rotation.apply(points - center) + center + params[3:]


def inverse_points(points: np.ndarray, params: np.ndarray,
                   center: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(params[:3])
    return rotation.inv().apply(points - center - params[3:]) + center


def resample_volume(volume: np.ndarray, params: np.ndarray,
                    center_xyz_vox: np.ndarray, cval: float) -> np.ndarray:
    """Put CT onto the SPECT grid: output[p] = CT[T(p)] for SPECT→CT T.

    If T moves SPECT onto CT, sampling CT at T(p) shifts the CT image by
    T^{-1} so both sit on the unchanged SPECT indices.
    """
    nz, ny, nx = volume.shape
    z, y, x = np.indices(volume.shape, dtype=np.float64)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    src = transform_points(pts, params, center_xyz_vox)
    coords = np.vstack([
        src[:, 2],
        src[:, 1],
        src[:, 0],
    ]).reshape(3, nz, ny, nx)
    return map_coordinates(volume, coords, order=1, mode="constant", cval=cval)


def _xy_stats(dx: np.ndarray, dy: np.ndarray, z: np.ndarray) -> dict:
    if dx.size == 0:
        return {
            "count": 0,
            "dx_median_vox": None,
            "dy_median_vox": None,
            "dx_mad_vox": None,
            "dy_mad_vox": None,
            "dx_p90_vox": None,
            "dy_p90_vox": None,
            "dx_within_1vox_frac": None,
            "dy_within_1vox_frac": None,
            "xy_within_1vox_frac": None,
            "dx_vs_z_slope_vox_per_vox": None,
            "dy_vs_z_slope_vox_per_vox": None,
            "dx_linear_span_vox": None,
            "dy_linear_span_vox": None,
        }
    dx_med = float(np.median(dx))
    dy_med = float(np.median(dy))
    zc = z - np.median(z)
    dx_slope = float(np.polyfit(zc, dx, 1)[0]) if dx.size >= 8 else 0.0
    dy_slope = float(np.polyfit(zc, dy, 1)[0]) if dy.size >= 8 else 0.0
    z_span = float(z.max() - z.min()) if dx.size else 0.0
    xy = np.hypot(dx, dy)
    return {
        "count": int(dx.size),
        "dx_median_vox": dx_med,
        "dy_median_vox": dy_med,
        "dx_mad_vox": float(np.median(np.abs(dx - dx_med))),
        "dy_mad_vox": float(np.median(np.abs(dy - dy_med))),
        "dx_p90_vox": float(np.percentile(np.abs(dx - dx_med), 90)),
        "dy_p90_vox": float(np.percentile(np.abs(dy - dy_med), 90)),
        "dx_within_1vox_frac": float(np.mean(np.abs(dx) <= TARGET_VOX)),
        "dy_within_1vox_frac": float(np.mean(np.abs(dy) <= TARGET_VOX)),
        "xy_within_1vox_frac": float(np.mean(xy <= TARGET_VOX)),
        "dx_vs_z_slope_vox_per_vox": dx_slope,
        "dy_vs_z_slope_vox_per_vox": dy_slope,
        "dx_linear_span_vox": dx_slope * z_span,
        "dy_linear_span_vox": dy_slope * z_span,
    }


def _window_com(image2d: np.ndarray, x: float, y: float, half: int,
                min_mass: float) -> tuple[float, float] | None:
    ny, nx = image2d.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - half), min(nx, xi + half + 1)
    y0, y1 = max(0, yi - half), min(ny, yi + half + 1)
    patch = image2d[y0:y1, x0:x1].astype(float)
    mass = float(patch.sum())
    if mass < min_mass:
        return None
    yy, xx = np.indices(patch.shape)
    return (
        x0 + float((patch * xx).sum() / mass),
        y0 + float((patch * yy).sum() / mass),
    )


def _rolling_median(values: np.ndarray, width: int = 11) -> np.ndarray:
    if values.size == 0:
        return values
    half = width // 2
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        out[i] = float(np.median(values[max(0, i - half):min(len(values), i + half + 1)]))
    return out


def spine_correspondences(hu: np.ndarray, activity: np.ndarray,
                          spacing_xyz: np.ndarray, pairing: str = "local",
                          local_half_vox: int = 12) -> dict:
    """Pair spine landmarks.

    `independent` is the production midline-band peak vs peak extractor. It
    treats bladder/rib hot voxels in the same X band as spine and inflates MAD.
    `local` first tracks the CT bone ridge, then takes the SPECT activity COM
    in a window around that ridge. Valid only after a coarse 3-DOF alignment.
    """
    bone, hot = bone_hot_masks(hu, activity, hot_frac=0.28)
    occupied = _occupied_z(bone, 20.0)
    if occupied.size < 30:
        raise RuntimeError("Insufficient CT bone occupancy for spine correspondences")
    caudal_low = _caudal_is_low_z(bone, occupied)
    lag_z, (x0, x1) = _midline_z_lag_vox(bone, hot)
    n = len(occupied)
    if caudal_low:
        z0, z1 = occupied[int(0.22 * n)], occupied[int(0.82 * n)]
    else:
        z0, z1 = occupied[int(0.18 * n)], occupied[int(0.78 * n)]
    ct_raw = []
    for z in range(int(z0), int(z1) + 1):
        ct = _slice_ridge_peak(bone[z], x0, x1, 4)
        if ct is None:
            continue
        ct_raw.append((ct[0], ct[1], float(z)))
    if len(ct_raw) < 20:
        raise RuntimeError(f"Insufficient CT spine ridge ({len(ct_raw)} slices)")
    ct_raw = np.asarray(ct_raw, dtype=float)
    ct_smooth_x = _rolling_median(ct_raw[:, 0])
    ct_smooth_y = _rolling_median(ct_raw[:, 1])
    ct_keep = (
        (np.abs(ct_raw[:, 0] - ct_smooth_x) <= 2.5) &
        (np.abs(ct_raw[:, 1] - ct_smooth_y) <= 2.5)
    )
    ct_pts_all = np.column_stack([
        ct_smooth_x[ct_keep], ct_smooth_y[ct_keep], ct_raw[ct_keep, 2]])
    spect_pts = []
    ct_pts = []
    edge_hits = 0
    for x_ct, y_ct, z in ct_pts_all:
        zs = int(round(z - lag_z))
        if zs < 0 or zs >= activity.shape[0]:
            continue
        if pairing == "independent":
            sp = _slice_ridge_peak(hot[zs], x0, x1, 3)
            if sp is None:
                continue
            x_sp, y_sp = sp
        else:
            sp = _window_com(activity[zs] * hot[zs], x_ct, y_ct, local_half_vox, 3.0)
            if sp is None:
                continue
            x_sp, y_sp = sp
            if max(abs(x_sp - x_ct), abs(y_sp - y_ct)) >= local_half_vox - 0.5:
                edge_hits += 1
                continue
        ct_pts.append((x_ct, y_ct, z))
        spect_pts.append((x_sp, y_sp, float(zs)))
    ct_pts = np.asarray(ct_pts, dtype=float)
    spect_pts = np.asarray(spect_pts, dtype=float)
    if len(ct_pts) < 20:
        raise RuntimeError(f"Insufficient paired spine slices ({len(ct_pts)})")
    knees, knee_z = _knee_deltas(
        bone, hot, occupied, caudal_low, lag_z,
        float(np.median(ct_pts[:, 0] - spect_pts[:, 0])),
        float(np.median(ct_pts[:, 1] - spect_pts[:, 1])),
        spacing_xyz)
    dx = ct_pts[:, 0] - spect_pts[:, 0]
    dy = ct_pts[:, 1] - spect_pts[:, 1]
    return {
        "pairing": pairing,
        "local_half_vox": local_half_vox,
        "ct_xyz_vox": ct_pts,
        "spect_xyz_vox": spect_pts,
        "dx_vox": dx,
        "dy_vox": dy,
        "z_ct_vox": ct_pts[:, 2],
        "lag_z_vox": lag_z,
        "midline_x_band_vox": [x0, x1],
        "z_band_vox": [int(z0), int(z1)],
        "caudal_is_low_z": bool(caudal_low),
        "knees": knees,
        "knee_z_band_vox": knee_z,
        "bone": bone,
        "hot": hot,
        "ct_ridge_kept_frac": float(ct_keep.mean()),
        "pair_count": int(len(ct_pts)),
        "ct_candidates": int(len(ct_pts_all)),
        "window_edge_rejects": int(edge_hits),
        "xy_stats": _xy_stats(dx, dy, ct_pts[:, 2]),
    }


def _band_stats(corr: dict, lo_frac: float, hi_frac: float) -> dict:
    z = corr["z_ct_vox"]
    z0, z1 = np.quantile(z, [lo_frac, hi_frac])
    selected = (z >= z0) & (z <= z1)
    return _xy_stats(corr["dx_vox"][selected], corr["dy_vox"][selected], z[selected])


def correspondence_report(corr: dict, spacing_xyz: np.ndarray) -> dict:
    knees = []
    for knee in corr["knees"]:
        delta = np.asarray(knee["delta_xyz_mm"], dtype=float)
        knees.append({
            "name": knee["name"],
            "delta_xyz_mm": delta.tolist(),
            "delta_xyz_vox": (delta / spacing_xyz).tolist(),
            "within_1vox": bool(np.all(np.abs(delta / spacing_xyz) <= TARGET_VOX)),
        })
    occupancy = occupancy_translation_mm(corr["bone"], corr["hot"], spacing_xyz)
    return {
        "pairing": corr.get("pairing"),
        "pair_count": corr.get("pair_count"),
        "ct_ridge_kept_frac": corr.get("ct_ridge_kept_frac"),
        "window_edge_rejects": corr.get("window_edge_rejects"),
        "spine": corr["xy_stats"],
        "spine_caudal_third": _band_stats(corr, 0.00, 0.33),
        "spine_mid_third": _band_stats(corr, 0.33, 0.67),
        "spine_cranial_third": _band_stats(corr, 0.67, 1.00),
        "knees": knees,
        "lag_z_vox": corr["lag_z_vox"],
        "occupancy_diagnostic": occupancy,
    }


def inlier_mask(dx: np.ndarray, dy: np.ndarray, max_abs_vox: float = 4.0) -> np.ndarray:
    return (np.abs(dx) <= max_abs_vox) & (np.abs(dy) <= max_abs_vox)


def fit_translation(corr: dict, spacing_xyz: np.ndarray) -> dict:
    dx, dy = corr["dx_vox"], corr["dy_vox"]
    keep = inlier_mask(dx, dy)
    if int(keep.sum()) < 20:
        raise RuntimeError(f"Too few local-spine inliers ({int(keep.sum())})")
    t_vox = np.array([
        float(np.median(dx)),
        float(np.median(dy)),
        float(corr["lag_z_vox"]),
    ])
    params = np.r_[np.zeros(3), t_vox * spacing_xyz]
    center = 0.5 * (
        corr["spect_xyz_vox"].mean(0) + corr["ct_xyz_vox"].mean(0)) * spacing_xyz
    return {
        "params": params,
        "center_mm": center,
        "angles_deg": [0.0, 0.0, 0.0],
        "translation_mm": params[3:].tolist(),
        "inlier_count": int(keep.sum()),
        "inlier_frac": float(keep.mean()),
        "rotation_or_translation_at_bound": bool(np.any(np.abs(params[3:]) >= 1.96)),
        "success": True,
        "message": "median of all CT-anchored local pairs; inliers |dxy|<=4 vox are the quality metric",
        "after_point_xy": None,
        "after_dz_mad_vox": None,
        "after_dz_median_vox": None,
    }


def fit_rigid(corr: dict, spacing_xyz: np.ndarray, max_angle_deg: float,
              max_trans_mm: float) -> dict:
    moving = corr["spect_xyz_vox"] * spacing_xyz
    fixed = corr["ct_xyz_vox"] * spacing_xyz
    center = 0.5 * (moving.mean(0) + fixed.mean(0))
    weights = np.ones(len(moving))
    knee_moving = []
    knee_fixed = []
    for knee in corr["knees"]:
        knee_moving.append(np.asarray(knee["spect_xyz_vox"], dtype=float) * spacing_xyz)
        knee_fixed.append(np.asarray(knee["ct_xyz_vox"], dtype=float) * spacing_xyz)
        weights = np.concatenate([weights, np.full(1, 12.0)])
    if knee_moving:
        moving = np.vstack([moving, np.stack(knee_moving)])
        fixed = np.vstack([fixed, np.stack(knee_fixed)])

    def residual(params: np.ndarray) -> np.ndarray:
        delta = transform_points(moving, params, center) - fixed
        return (delta * weights[:, None]).ravel()

    t0 = np.median(fixed[:len(corr["spect_xyz_vox"])] - moving[:len(corr["spect_xyz_vox"])],
                   axis=0)
    start = np.r_[np.zeros(3), t0]
    angle = math.radians(max_angle_deg)
    result = least_squares(
        residual, start,
        bounds=(np.r_[[-angle] * 3, [-max_trans_mm] * 3],
                np.r_[[angle] * 3, [max_trans_mm] * 3]),
        loss="soft_l1", f_scale=0.3, max_nfev=400)
    params = result.x
    spine_n = len(corr["spect_xyz_vox"])
    moved_spine = transform_points(moving[:spine_n], params, center)
    after_dx = (fixed[:spine_n, 0] - moved_spine[:, 0]) / spacing_xyz[0]
    after_dy = (fixed[:spine_n, 1] - moved_spine[:, 1]) / spacing_xyz[1]
    after_dz = (fixed[:spine_n, 2] - moved_spine[:, 2]) / spacing_xyz[2]
    at_bound = bool(
        np.any(np.abs(params[:3]) >= 0.98 * angle) or
        np.any(np.abs(params[3:]) >= 0.98 * max_trans_mm))
    return {
        "params": params,
        "center_mm": center,
        "angles_deg": np.degrees(params[:3]).tolist(),
        "translation_mm": params[3:].tolist(),
        "rotation_or_translation_at_bound": at_bound,
        "cost_before": float(np.mean(residual(start) ** 2)),
        "cost_after": float(np.mean(residual(params) ** 2)),
        "success": bool(result.success),
        "message": str(result.message),
        "after_point_xy": _xy_stats(after_dx, after_dy, corr["z_ct_vox"]),
        "after_dz_mad_vox": float(np.median(np.abs(after_dz - np.median(after_dz)))),
        "after_dz_median_vox": float(np.median(after_dz)),
    }


def score_resampled(hu: np.ndarray, activity: np.ndarray, fit: dict,
                    spacing_xyz: np.ndarray, pairing: str) -> tuple:
    params_vox = np.asarray(fit["params"], dtype=float).copy()
    params_vox[3:] = params_vox[3:] / spacing_xyz
    center_vox = np.asarray(fit["center_mm"], dtype=float) / spacing_xyz
    moved = resample_volume(hu, params_vox, center_vox, cval=-1000.0)
    corr = spine_correspondences(moved, activity, spacing_xyz, pairing=pairing)
    report = correspondence_report(corr, spacing_xyz)
    report["resampled_hu_finite"] = bool(np.isfinite(moved).all())
    report["pairing"] = pairing
    return report, moved, corr


def write_slice_plot(path: Path, before: dict, after: dict, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(before["z_ct_vox"], before["dx_vox"], ".", ms=3, alpha=0.7,
                 label="current 3-DOF maps")
    axes[0].plot(after["z_ct_vox"], after["dx_vox"], ".", ms=3, alpha=0.7,
                 label="local translation")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].axhline(1, color="0.6", ls="--", lw=0.6)
    axes[0].axhline(-1, color="0.6", ls="--", lw=0.6)
    axes[0].set_ylabel("dx vox (CT − SPECT)")
    axes[0].legend(loc="upper right")
    axes[1].plot(before["z_ct_vox"], before["dy_vox"], ".", ms=3, alpha=0.7)
    axes[1].plot(after["z_ct_vox"], after["dy_vox"], ".", ms=3, alpha=0.7)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].axhline(1, color="0.6", ls="--", lw=0.6)
    axes[1].axhline(-1, color="0.6", ls="--", lw=0.6)
    axes[1].set_ylabel("dy vox (CT − SPECT)")
    axes[1].set_xlabel("CT z voxel")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def write_ridge_plot(path: Path, corr: dict, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(corr["z_ct_vox"], corr["ct_xyz_vox"][:, 0], ".", ms=3, label="CT")
    axes[0].plot(corr["z_ct_vox"], corr["spect_xyz_vox"][:, 0], ".", ms=3, label="SPECT")
    axes[0].set_ylabel("ridge x vox")
    axes[0].legend(loc="upper right")
    axes[1].plot(corr["z_ct_vox"], corr["ct_xyz_vox"][:, 1], ".", ms=3)
    axes[1].plot(corr["z_ct_vox"], corr["spect_xyz_vox"][:, 1], ".", ms=3)
    axes[1].set_ylabel("ridge y vox")
    axes[1].set_xlabel("CT z voxel")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def write_overlay(path: Path, hu: np.ndarray, moved: np.ndarray,
                  activity: np.ndarray, before: dict, after: dict) -> None:
    bone0, hot = bone_hot_masks(hu, activity, hot_frac=0.28)
    bone1, _ = bone_hot_masks(moved, activity, hot_frac=0.28)
    zs = [
        int(np.median(before["z_ct_vox"])),
        int(np.percentile(before["z_ct_vox"], 20)),
        int(np.percentile(before["z_ct_vox"], 80)),
    ]
    if after["knees"]:
        zs.append(int(round(after["knees"][0]["ct_xyz_vox"][2])))
    fig, axes = plt.subplots(len(zs), 2, figsize=(7, 3.1 * len(zs)))
    vmax = float(np.percentile(activity[activity > 0], 99.5))
    for row, z in enumerate(zs):
        z = int(np.clip(z, 0, activity.shape[0] - 1))
        for col, bone in enumerate((bone0, bone1)):
            rgb = np.zeros((hot.shape[1], hot.shape[2], 3), dtype=float)
            rgb[..., 0] = np.clip(activity[z] / max(vmax, 1e-12), 0, 1)
            rgb[..., 1] = bone[z].astype(float)
            axes[row, col].imshow(rgb, origin="lower")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"z={z}")
        if row == 0:
            axes[row, 0].set_title("current 3-DOF")
            axes[row, 1].set_title("local translation")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def default_cases() -> list[tuple[str, Path]]:
    root = HERE.parent / "recon_out" / "ct_spect_reg_20260813"
    return [
        ("mdp1", root / "20260729_mdp_fixedrefine"),
        ("mdp2", root / "20260729_mdp2_fixedrefine"),
        ("mdp2_4h", root / "20260729_mdp2_4h_fixedrefine"),
    ]


def run_case(name: str, folder: Path, output: Path, max_angle_deg: float,
             max_trans_mm: float) -> dict:
    hu = read_mhd(folder / "ct_mu_recon220_spectgrid_hu.mhd").astype(np.float64)
    activity = read_mhd(folder / "spect_activity.mhd").astype(np.float64)
    spacing = np.array([VOXEL_MM, VOXEL_MM, VOXEL_MM])
    independent = spine_correspondences(hu, activity, spacing, pairing="independent")
    before_corr = spine_correspondences(hu, activity, spacing, pairing="local")
    before = correspondence_report(before_corr, spacing)
    trans = fit_translation(before_corr, spacing)
    trans_report, trans_moved, trans_corr = score_resampled(
        hu, activity, trans, spacing, pairing="local")
    fit = fit_rigid(before_corr, spacing, max_angle_deg, max_trans_mm)
    rigid_report, _moved, _rigid_corr = score_resampled(
        hu, activity, fit, spacing, pairing="local")
    output.mkdir(parents=True, exist_ok=True)
    slice_plot = output / f"{name}_spine_xy_vs_z.png"
    ridge_plot = output / f"{name}_ridge_xy.png"
    overlay = output / f"{name}_axial_overlay.png"
    write_slice_plot(
        slice_plot, before_corr, trans_corr,
        f"{name}: CT-anchored local residual | blue=current, orange=local translation")
    write_ridge_plot(
        ridge_plot, trans_corr,
        f"{name}: CT-anchored ridges after local translation")
    write_overlay(overlay, hu, trans_moved, activity, before_corr, trans_corr)
    spine_before = before["spine"]
    spine_trans = trans_report["spine"]
    spine_rigid = rigid_report["spine"]
    keep = inlier_mask(before_corr["dx_vox"], before_corr["dy_vox"])
    return {
        "case": name,
        "folder": str(folder),
        "legacy_independent_spine": independent["xy_stats"],
        "local_inliers_before": {
            "count": int(keep.sum()),
            "frac": float(keep.mean()),
            "xy": _xy_stats(before_corr["dx_vox"][keep], before_corr["dy_vox"][keep],
                            before_corr["z_ct_vox"][keep]),
        },
        "translation": {
            "translation_mm": trans["translation_mm"],
            "inlier_count": trans["inlier_count"],
            "inlier_frac": trans["inlier_frac"],
        },
        "rigid": {
            "angles_deg": fit["angles_deg"],
            "translation_mm": fit["translation_mm"],
            "at_bound": fit["rotation_or_translation_at_bound"],
        },
        "before": before,
        "after_translation": trans_report,
        "after_rigid": rigid_report,
        "target": {
            "voxel_mm": VOXEL_MM,
            "ideal_abs_vox": TARGET_VOX,
            "spine_xy_mad_within_1vox_before": bool(
                spine_before["dx_mad_vox"] <= TARGET_VOX and
                spine_before["dy_mad_vox"] <= TARGET_VOX),
            "spine_xy_mad_within_1vox_after_translation": bool(
                spine_trans["dx_mad_vox"] <= TARGET_VOX and
                spine_trans["dy_mad_vox"] <= TARGET_VOX),
            "spine_xy_mad_within_1vox_after_rigid": bool(
                spine_rigid["dx_mad_vox"] <= TARGET_VOX and
                spine_rigid["dy_mad_vox"] <= TARGET_VOX),
        },
        "figures": [str(slice_plot), str(ridge_plot), str(overlay)],
    }


def self_check() -> None:
    rng = np.random.default_rng(0)
    z = np.arange(40, 160, dtype=float)
    ct = np.column_stack([50 + 0.02 * z, 40 + 0.01 * z, z])
    true = np.array([math.radians(1.2), math.radians(-0.8), math.radians(0.4),
                     0.21, -0.18, 0.09])
    spect_clean = inverse_points(ct * VOXEL_MM, true, 0.5 * (ct * VOXEL_MM).mean(0))
    # Rebuild with the same center convention the fitter uses.
    center = 0.5 * ((spect_clean / VOXEL_MM).mean(0) * VOXEL_MM + (ct * VOXEL_MM).mean(0))
    spect = inverse_points(ct * VOXEL_MM, true, center) / VOXEL_MM
    spect = spect + rng.normal(0, 0.12, spect.shape)
    dummy_hu = np.full((200, 80, 80), -1000.0)
    dummy_act = np.zeros_like(dummy_hu)
    corr = {
        "ct_xyz_vox": ct,
        "spect_xyz_vox": spect,
        "dx_vox": ct[:, 0] - spect[:, 0],
        "dy_vox": ct[:, 1] - spect[:, 1],
        "z_ct_vox": ct[:, 2],
        "knees": [],
        "bone": dummy_hu > 0,
        "hot": dummy_act > 0,
    }
    fit = fit_rigid(corr, np.array([VOXEL_MM] * 3), 3.0, 2.0)
    recovered = transform_points(spect * VOXEL_MM, fit["params"], fit["center_mm"])
    err = np.linalg.norm(recovered - ct * VOXEL_MM, axis=1)
    # Euler parameters depend on the rotation center; the protected check is
    # point mapping, not a unique (angle, translation) tuple.
    assert float(np.median(err)) < 0.5 * VOXEL_MM
    assert float(np.percentile(err, 90)) < VOXEL_MM
    image = np.zeros((40, 40))
    image[18:23, 10:15] = 1.0
    com = _window_com(image, 12.0, 20.0, 8, 1.0)
    assert com is not None
    assert abs(com[0] - 12.0) < 0.2 and abs(com[1] - 20.0) < 0.2
    shifted = {
        "dx_vox": np.full(40, 1.5),
        "dy_vox": np.full(40, -2.0),
        "lag_z_vox": 0,
        "ct_xyz_vox": np.column_stack([np.full(40, 50.0), np.full(40, 40.0), np.arange(40)]),
        "spect_xyz_vox": np.column_stack([np.full(40, 48.5), np.full(40, 42.0), np.arange(40)]),
    }
    trans = fit_translation(shifted, np.array([VOXEL_MM] * 3))
    assert np.allclose(trans["translation_mm"], [0.45, -0.60, 0.0], atol=1e-9)
    blob = np.zeros((12, 12, 12))
    blob[5, 5, 7] = 1.0
    params = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    moved = resample_volume(blob, params, np.array([6.0, 6.0, 6.0]), 0.0)
    assert moved[5, 5, 5] > 0.5 and moved[5, 5, 7] < 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=HERE.parent / "reports" / "assets" /
                        "2026-08-14-ct-spect-rigid6")
    parser.add_argument("--max-angle-deg", type=float, default=1.5)
    parser.add_argument("--max-trans-mm", type=float, default=2.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_test:
        return 0
    rows = []
    for name, folder in default_cases():
        if not (folder / "ct_mu_recon220_spectgrid_hu.mhd").is_file():
            raise RuntimeError(f"missing spectgrid HU for {name}: {folder}")
        rows.append(run_case(name, folder, args.output_dir, args.max_angle_deg,
                             args.max_trans_mm))
    summary = {
        "contract": {
            "candidate": "CT-anchored local spine COM inliers, 3-DOF translation; 6-DOF reported as rejected comparator",
            "baseline": "frozen device translation + 3-DOF spine/knee refine",
            "grid": "SPECT DICOM 0.3 mm, existing *_fixedrefine spectgrid MHD",
            "allowed": "one global rigid residual; no deformation; no per-bed",
            "pairing": "local SPECT COM in ±12 voxel window around smoothed CT ridge",
            "occupancy": "diagnostic only; cannot pass the method",
            "legacy_independent_peaks": "reported but not used for the 6-DOF fit",
            "ideal_target": "slice-wise spine |dx| and |dy| MAD <= 1 voxel = 0.3 mm",
            "gate_maturity": "diagnostic",
        },
        "cases": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable({
        "output": str(args.output_dir / "metrics.json"),
        "cases": [{
            "case": row["case"],
            "translation_mm": row["translation"]["translation_mm"],
            "inlier_frac": row["translation"]["inlier_frac"],
            "before_dx_mad": row["before"]["spine"]["dx_mad_vox"],
            "after_dx_mad": row["after_translation"]["spine"]["dx_mad_vox"],
            "before_dy_mad": row["before"]["spine"]["dy_mad_vox"],
            "after_dy_mad": row["after_translation"]["spine"]["dy_mad_vox"],
            "before_dy_median": row["before"]["spine"]["dy_median_vox"],
            "after_dy_median": row["after_translation"]["spine"]["dy_median_vox"],
            "rigid_angles_deg": row["rigid"]["angles_deg"],
            "within_1vox_translation": row["target"]["spine_xy_mad_within_1vox_after_translation"],
            "within_1vox_rigid": row["target"]["spine_xy_mad_within_1vox_after_rigid"],
        } for row in rows],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
