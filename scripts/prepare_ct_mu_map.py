#!/usr/bin/env python3
"""Resample a paired CT DICOM series onto the SPECT reconstruction grid."""

import argparse
import configparser
import json
import re
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    PositronEmissionTomographyImageStorage,
    SecondaryCaptureImageStorage,
    generate_uid,
)
from scipy.ndimage import center_of_mass, gaussian_filter1d, map_coordinates
from scipy.signal import correlate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_mhd_stats import read_mhd  # noqa: E402

DEVICE_BED_NZ = 160
DEVICE_BED_OVERLAP = 104
DEVICE_BED_STEP_MM = 16.8
DEVICE_TARGET_SPACING_XYZ_MM = np.array([0.3, 0.3, 0.3])
DEVICE_IMAGE_SHAPE_XY = (108, 108)
DEVICE_PHYSICAL_BEDS = range(1, 6)
# Frozen CT←SPECT device translation (2026-08-13 MDP spine median).
# INI is the load/save format; do not treat this vector as landmark TRE.
# The reconstruction executable does not load this registration INI.
MDP_DEVICE_REGISTRATION_INI = (
    HERE.parent / "multipinhole" / "config" / "ct_spect_registration.ini"
)
REGISTRATION_SECTION = "Registration"
# 220-layer production matrix is the same 0.3 mm / 16.8 mm bed step, centered
# on the device 160/108 FOV: +6 voxels XY (1.8 mm) and +30 voxels Z (9 mm)
# on each side of every bed.
RECON220_BED_NZ = 220
RECON220_BED_OVERLAP = 164
RECON220_IMAGE_SHAPE_XY = (120, 120)
RECON220_XY_PAD_VOX = 6
RECON220_Z_PAD_VOX = 30

PROFILES = {
    "device160": {
        "bed_nz": DEVICE_BED_NZ,
        "overlap": DEVICE_BED_OVERLAP,
        "xy": DEVICE_IMAGE_SHAPE_XY,
        "xy_pad_vox": 0,
        "z_pad_vox": 0,
    },
    "recon220": {
        "bed_nz": RECON220_BED_NZ,
        "overlap": RECON220_BED_OVERLAP,
        "xy": RECON220_IMAGE_SHAPE_XY,
        "xy_pad_vox": RECON220_XY_PAD_VOX,
        "z_pad_vox": RECON220_Z_PAD_VOX,
    },
}


def _parse_xyz(value: str) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected three finite comma-separated values")
    return values


def _parse_fixed_translation(value: str):
    key = value.strip().lower()
    if key in ("mdp_device", "default"):
        return key
    return _parse_xyz(value)


def _read_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if not path.is_file():
        raise RuntimeError(f"missing registration INI {path}")
    if not parser.read(path, encoding="utf-8"):
        raise RuntimeError(f"failed to read registration INI {path}")
    return parser


def load_frozen_translation(path: Path) -> np.ndarray:
    cfg = _read_ini(path)
    if not cfg.has_section(REGISTRATION_SECTION):
        raise RuntimeError(f"{path} missing [{REGISTRATION_SECTION}]")
    values = _parse_xyz(cfg.get(REGISTRATION_SECTION, "translation_xyz_mm"))
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"{path} [Registration] translation_xyz_mm must be three finite values")
    return values


def _xyz_ini(values) -> str:
    return ",".join(f"{float(v):.6g}" for v in np.asarray(values, dtype=float).reshape(3))


def write_registration_ini(path: Path, fields: dict[str, str]) -> None:
    lines = [
        "; SPECT CT registration parameters. Key=value INI for ConfigParser.",
        "; xyz triples are comma-separated with no spaces (C++ split like voxel_size).",
        f"[{REGISTRATION_SECTION}]",
    ]
    for key, value in fields.items():
        lines.append(f"{key} = {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixed_bed_origins(bed1_origin: np.ndarray, normal: np.ndarray,
                       bed_step_mm: float, beds=DEVICE_PHYSICAL_BEDS) -> dict[int, np.ndarray]:
    return {
        bed: bed1_origin + (bed - 1) * bed_step_mm * normal
        for bed in beds
    }


def load_recon_grid(path: Path) -> dict:
    """Use the effective grid, not scanner-specific constants or matrix identity."""
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema") != "spect_recon_run_manifest_v1":
        raise RuntimeError("Unsupported reconstruction manifest schema")
    matrix, multi, output, projection = (
        run[key] for key in ("matrix", "multibed", "output", "projection"))
    dims = matrix["dims"]
    spacing = np.asarray(output["voxel_mm"], dtype=float)
    beds = sorted(int(key) for key in projection if key.isdigit())
    if (len(dims) != 3 or any(type(v) is not int or v <= 0 for v in dims) or
            spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or
            np.any(spacing <= 0) or not np.isfinite(matrix["voxel_size_mm"]) or
            not np.allclose(spacing, matrix["voxel_size_mm"], rtol=0, atol=1e-6)):
        raise RuntimeError("Invalid or inconsistent reconstruction manifest grid")
    if (not beds or beds != list(range(beds[0], beds[0] + len(beds))) or
            beds[0] <= 0 or multi["bed_count"] != len(beds) or
            multi["bed_z_starts"] != [i * dims[2] for i in range(len(beds))]):
        raise RuntimeError("Manifest requires consecutive beds in separate storage blocks")
    angle = float(output.get("xy_out_rotation_deg", 0.0))
    per_bed_flip = output.get("z_per_bed_flip", projection["z_bed_flip"] and len(beds) == 1)
    if (not multi["matrix_grid_output"] or output["x_out_flip"] or
            output["z_out_flip"] or per_bed_flip or output["seam_xy_align"] or
            output["crop_end_start"] or output["crop_end_end"] or
            not np.isfinite(angle) or angle % 360.0 != 0):
        raise RuntimeError(
            "--recon-manifest requires full matrix-grid SPECT DICOM without output "
            "rotation, flips, cropping or seam alignment; regenerate an untransformed reference")
    step_mm = float(multi["bed_step_mm"])
    if not np.isfinite(step_mm) or step_mm < 0:
        raise RuntimeError("Manifest bed_step_mm must be finite and nonnegative")
    step = dims[2]
    if len(beds) > 1:
        step = round(step_mm / spacing[2])
        if (not 1 <= step <= dims[2] or abs(step * spacing[2] - step_mm) > 1e-4 or
                multi["bed_step_layers"] != step or
                multi["output_bed_overlaps"] != [dims[2] - step] * (len(beds) - 1)):
            raise RuntimeError("Manifest multi-bed step must be explicit and voxel-aligned")
    expected = [dims[0], dims[1], dims[2] + (len(beds) - 1) * step]
    if output["dimensions"] != expected:
        raise RuntimeError("Manifest saved dimensions are not the full matrix grid")
    return {"bed_nz": dims[2], "overlap": dims[2] - step,
            "xy": (dims[1], dims[0]), "xy_pad_vox": 0, "z_pad_vox": 0,
            "spacing": spacing, "beds": beds, "bed_step_mm": step_mm,
            "z_bed_flip": bool(projection["z_bed_flip"]) and len(beds) > 1}


def _coverage_summary(coords: np.ndarray, ct_shape: tuple[int, int, int],
                      beds: list[int], bed_nz: int, overlap: int) -> tuple[np.ndarray, dict]:
    tolerance = 1e-4
    inside = np.ones(coords.shape[1:], dtype=bool)
    for axis, size in enumerate(ct_shape):
        inside &= (coords[axis] >= -tolerance) & (coords[axis] <= size - 1 + tolerance)
    stride = bed_nz - overlap
    per_bed = {}
    for index, bed in enumerate(beds):
        block = inside[index * stride:index * stride + bed_nz]
        inside_count = int(np.count_nonzero(block))
        per_bed[str(bed)] = {
            "inside_voxels": inside_count,
            "outside_voxels": int(block.size - inside_count),
            "coverage_fraction": inside_count / block.size,
        }
    return inside, per_bed


def resolve_ct_dir(root: Path) -> Path:
    """Use cleaned_dicom when present so cabin-removal is not mixed with raw CT."""
    cleaned = root / "cleaned_dicom"
    if cleaned.is_dir() and any(cleaned.glob("*.dcm")):
        return cleaned
    return root


def dicoms(root: Path) -> list:
    """Load one series. Prefer *.dcm in `root`; recurse only if the folder is empty."""
    paths = sorted(root.glob("*.dcm"))
    if not paths:
        paths = [path for path in sorted(root.rglob("*")) if path.is_file()]
    result = []
    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            if hasattr(ds, "PixelData") and hasattr(ds, "ImagePositionPatient"):
                result.append(ds)
        except Exception:
            pass
    if not result:
        raise RuntimeError(f"No image DICOM found under {root}")
    return result


def _resample_hu(hu: np.ndarray, coords: np.ndarray, ct_shape: tuple[int, int, int],
                 allow_padding: bool) -> tuple[np.ndarray, int]:
    tolerance = 1e-4
    inside = np.ones(coords.shape[1:], dtype=bool)
    for axis, size in enumerate(ct_shape):
        inside &= (coords[axis] >= -tolerance) & (coords[axis] <= size - 1 + tolerance)
    outside = int(inside.size - np.count_nonzero(inside))
    if outside and not allow_padding:
        raise RuntimeError(
            f"Fixed SPECT grid extends outside CT by {outside} voxels; "
            "refuse silent air padding (use --allow-ct-padding only after coverage review)")
    if not outside:
        for axis, size in enumerate(ct_shape):
            coords[axis] = np.clip(coords[axis], 0, size - 1)
    target_hu = map_coordinates(hu, coords, order=1, mode="constant", cval=-1000.0)
    return target_hu, outside


def _subvoxel_peak(corr: np.ndarray, peak: tuple[int, int, int]) -> np.ndarray:
    """Parabolic peak offset in voxel units, wrapped for FFT phase correlation."""
    offset = np.zeros(3, dtype=float)
    shape = np.asarray(corr.shape, dtype=float)
    for axis, index in enumerate(peak):
        n = corr.shape[axis]
        prev_i = (index - 1) % n
        next_i = (index + 1) % n
        slicer = [peak[0], peak[1], peak[2]]
        slicer[axis] = prev_i
        left = float(corr[tuple(slicer)])
        slicer[axis] = index
        center = float(corr[tuple(slicer)])
        slicer[axis] = next_i
        right = float(corr[tuple(slicer)])
        denom = 2.0 * (2.0 * center - left - right)
        delta = 0.0 if abs(denom) < 1e-12 else (left - right) / denom
        wrapped = index + delta
        if wrapped > n / 2.0:
            wrapped -= n
        offset[axis] = wrapped
    return offset


def translation_residual_mm(fixed: np.ndarray, moving: np.ndarray,
                            spacing_zyx: np.ndarray, mask: np.ndarray) -> dict:
    """FFT phase-correlation translation of moving onto fixed, in mm.

    Arrays are zyx. Positive offset means moving must shift toward +axis to
    match fixed (i.e. SPECT is that far from the DICOM-tag CT grid).
    """
    work_fixed = np.where(mask, fixed, 0.0)
    work_moving = np.where(mask, moving, 0.0)
    fa = np.fft.fftn(work_fixed)
    fb = np.fft.fftn(work_moving)
    cross = fa * np.conj(fb)
    denom = np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifftn(cross / denom).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    voxel = _subvoxel_peak(corr, peak)
    # voxel[0]=z, [1]=y, [2]=x in array order; spacing_zyx matches that.
    shift_zyx_mm = voxel * spacing_zyx
    shift_xyz_mm = np.array([shift_zyx_mm[2], shift_zyx_mm[1], shift_zyx_mm[0]])
    magnitude = float(np.linalg.norm(shift_xyz_mm))
    return {
        "translation_xyz_mm": shift_xyz_mm.tolist(),
        "translation_zyx_vox": voxel.tolist(),
        "magnitude_mm": magnitude,
        "peak_correlation": float(corr[peak]),
        "mask_voxels": int(np.count_nonzero(mask)),
    }


def occupancy_translation_mm(fixed_mask: np.ndarray, moving_mask: np.ndarray,
                             spacing_xyz: np.ndarray) -> dict:
    """Translation that moves `moving` onto `fixed` from Z occupancy + slice XY COM.

    spacing_xyz is (dx, dy, dz) in mm. Arrays are zyx.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import correlate

    fixed_z = gaussian_filter1d(fixed_mask.sum(axis=(1, 2)).astype(float), 2)
    moving_z = gaussian_filter1d(moving_mask.sum(axis=(1, 2)).astype(float), 2)
    corr = correlate(fixed_z, moving_z, mode="full")
    lags = np.arange(-len(moving_z) + 1, len(fixed_z))
    lag_z = int(lags[int(np.argmax(corr))])

    def slice_com(mask):
        points = []
        for z in range(mask.shape[0]):
            sl = mask[z]
            if sl.sum() < 8:
                continue
            yy, xx = np.nonzero(sl)
            points.append((z, yy.mean(), xx.mean()))
        return np.array(points) if points else np.zeros((0, 3))

    fixed_pts = slice_com(fixed_mask)
    moving_pts = slice_com(moving_mask)
    dxs, dys = [], []
    if len(fixed_pts) and len(moving_pts):
        moving_z_shifted = moving_pts[:, 0] + lag_z
        for z, y, x in zip(moving_z_shifted, moving_pts[:, 1], moving_pts[:, 2]):
            match = np.where(np.abs(fixed_pts[:, 0] - z) <= 1)[0]
            if match.size:
                i = match[np.argmin(np.abs(fixed_pts[match, 0] - z))]
                dxs.append(fixed_pts[i, 2] - x)
                dys.append(fixed_pts[i, 1] - y)
    dx = float(np.median(dxs)) if dxs else 0.0
    dy = float(np.median(dys)) if dys else 0.0
    t_xyz = np.array([dx * spacing_xyz[0], dy * spacing_xyz[1],
                      lag_z * spacing_xyz[2]])
    return {
        "translation_xyz_mm": t_xyz.tolist(),
        "z_lag_vox": lag_z,
        "xy_slice_count": int(len(dxs)),
        "magnitude_mm": float(np.linalg.norm(t_xyz)),
        "method": "z_occupancy_xcorr_plus_slice_xy_com",
    }


def _occupied_z(mask: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    zsum = mask.sum(axis=(1, 2))
    positive = zsum[zsum > 0]
    if positive.size == 0:
        return np.arange(0, dtype=int)
    return np.flatnonzero(zsum >= max(1.0, np.percentile(positive, percentile)))


def _x_span(mask: np.ndarray, zs: np.ndarray) -> float:
    if zs.size == 0:
        return 0.0
    xs = np.nonzero(mask[int(zs[0]):int(zs[-1]) + 1].max(0))[1]
    return float(xs.max() - xs.min()) if xs.size else 0.0


def _caudal_is_low_z(bone: np.ndarray, occupied: np.ndarray) -> bool:
    """Hindlimbs are the occupied-Z end with the larger left-right bone span."""
    n = max(8, int(0.18 * len(occupied)))
    return _x_span(bone, occupied[:n]) >= _x_span(bone, occupied[-n:])


def _parabolic_offset(profile: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(profile) - 1:
        return 0.0
    left, center, right = (float(profile[index - 1]), float(profile[index]),
                           float(profile[index + 1]))
    denom = 2.0 * (2.0 * center - left - right)
    return 0.0 if abs(denom) < 1e-12 else (left - right) / denom


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


def _slice_ridge_peak(mask2d: np.ndarray, x0: int, x1: int, min_pix: int = 4):
    """Midline ridge as X/Y profile peaks, not blob COM (ribs/bladder pull COM)."""
    band = mask2d[:, x0:x1]
    if int(band.sum()) < min_pix:
        return None
    y_prof = band.sum(axis=1).astype(float)
    x_prof = band.sum(axis=0).astype(float)
    if y_prof.max() <= 0 or x_prof.max() <= 0:
        return None
    iy = int(np.argmax(y_prof))
    ix = int(np.argmax(x_prof))
    return float(ix + x0 + _parabolic_offset(x_prof, ix)), float(iy + _parabolic_offset(y_prof, iy))


def _midline_z_lag_vox(bone: np.ndarray, hot: np.ndarray) -> tuple[int, tuple[int, int]]:
    nx = bone.shape[2]
    x0, x1 = int(nx * 0.38), int(nx * 0.62)
    bone_z = gaussian_filter1d(bone[:, :, x0:x1].sum(axis=(1, 2)).astype(float), 2)
    hot_z = gaussian_filter1d(hot[:, :, x0:x1].sum(axis=(1, 2)).astype(float), 2)
    corr = correlate(bone_z, hot_z, mode="full")
    lags = np.arange(-len(hot_z) + 1, len(bone_z))
    return int(lags[int(np.argmax(corr))]), (x0, x1)


def _spine_xy_median_vox(bone: np.ndarray, hot: np.ndarray, activity: np.ndarray,
                         occupied: np.ndarray, caudal_low: bool, lag_z: int,
                         x0: int, x1: int, local_half_vox: int = 12) -> dict:
    """CT bone ridge vs SPECT activity COM in a window around that ridge.

    Independent midline-band peaks are not used: SPECT bladder/rib hot voxels
    in the same X band are not spine correspondences. Valid after the frozen
    device translation has removed the ~10 mm Z offset.
    """
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
    empty = {
        "z_band_vox": [int(z0), int(z1)],
        "slice_count": 0,
        "dx_vox": 0.0,
        "dy_vox": 0.0,
        "dx_mad_vox": None,
        "dy_mad_vox": None,
        "pairing": "ct_anchored_local_com",
        "local_half_vox": int(local_half_vox),
        "window_edge_rejects": 0,
    }
    if len(ct_raw) < 20:
        return empty
    ct_raw = np.asarray(ct_raw, dtype=float)
    ct_x = _rolling_median(ct_raw[:, 0])
    ct_y = _rolling_median(ct_raw[:, 1])
    keep = (np.abs(ct_raw[:, 0] - ct_x) <= 2.5) & (np.abs(ct_raw[:, 1] - ct_y) <= 2.5)
    dxs, dys = [], []
    edge_rejects = 0
    for x_ct, y_ct, z in zip(ct_x[keep], ct_y[keep], ct_raw[keep, 2]):
        zs = int(round(z - lag_z))
        if zs < 0 or zs >= activity.shape[0]:
            continue
        sp = _window_com(activity[zs] * hot[zs], x_ct, y_ct, local_half_vox, 3.0)
        if sp is None:
            continue
        if max(abs(sp[0] - x_ct), abs(sp[1] - y_ct)) >= local_half_vox - 0.5:
            edge_rejects += 1
            continue
        dxs.append(x_ct - sp[0])
        dys.append(y_ct - sp[1])
    if len(dxs) < 20:
        empty["window_edge_rejects"] = int(edge_rejects)
        return empty
    dxs = np.asarray(dxs, dtype=float)
    dys = np.asarray(dys, dtype=float)
    return {
        "z_band_vox": [int(z0), int(z1)],
        "slice_count": int(len(dxs)),
        "dx_vox": float(np.median(dxs)),
        "dy_vox": float(np.median(dys)),
        "dx_mad_vox": float(np.median(np.abs(dxs - np.median(dxs)))),
        "dy_mad_vox": float(np.median(np.abs(dys - np.median(dys)))),
        "xy_within_1vox_frac": float(np.mean(np.hypot(dxs, dys) <= 1.0)),
        "pairing": "ct_anchored_local_com",
        "local_half_vox": int(local_half_vox),
        "window_edge_rejects": int(edge_rejects),
    }


def _knee_deltas(bone: np.ndarray, hot: np.ndarray, occupied: np.ndarray,
                 caudal_low: bool, lag_z: int, dx_vox: float, dy_vox: float,
                 spacing_xyz: np.ndarray) -> tuple[list[dict], list[int]]:
    """CT knee windows; SPECT windows shifted by the spine lag so Z offset is visible."""
    nz, ny, nx = bone.shape
    n = len(occupied)
    if caudal_low:
        z0, z1 = occupied[int(0.04 * n)], occupied[int(0.26 * n)]
    else:
        z0, z1 = occupied[int(0.74 * n)], occupied[int(0.96 * n)]
    z0, z1 = int(z0), int(z1)
    ys, xs = np.nonzero(bone[z0:z1 + 1].max(0))
    if xs.size < 20:
        return [], [z0, z1]
    xmid = float(np.median(xs))
    knees = []
    for name, selected in (("knee_left", xs < xmid - 3), ("knee_right", xs >= xmid + 3)):
        if selected.sum() < 8:
            continue
        cx = float(xs[selected].mean())
        cy = float(ys[selected].mean())
        x0w, x1w = max(0, int(cx) - 8), min(nx, int(cx) + 9)
        y0w, y1w = max(0, int(cy) - 8), min(ny, int(cy) + 9)
        win_bone = bone[z0:z1 + 1, y0w:y1w, x0w:x1w]
        if win_bone.sum() < 4:
            continue
        zb, yb, xb = center_of_mass(win_bone.astype(float))
        ct = np.array([xb + x0w, yb + y0w, zb + z0], dtype=float)
        z0s, z1s = np.clip([int(round(z0 - lag_z)), int(round(z1 - lag_z))], 0, nz - 1)
        x0s, x1s = np.clip([int(round(x0w - dx_vox)), int(round(x1w - dx_vox))], 0, nx)
        y0s, y1s = np.clip([int(round(y0w - dy_vox)), int(round(y1w - dy_vox))], 0, ny)
        if z1s <= z0s or x1s <= x0s or y1s <= y0s:
            continue
        win_hot = hot[z0s:z1s + 1, y0s:y1s, x0s:x1s]
        if win_hot.sum() < 4:
            continue
        zh, yh, xh = center_of_mass(win_hot.astype(float))
        spect = np.array([xh + x0s, yh + y0s, zh + z0s], dtype=float)
        delta_mm = (ct - spect) * spacing_xyz
        knees.append({
            "name": name,
            "ct_xyz_vox": ct.tolist(),
            "spect_xyz_vox": spect.tolist(),
            "delta_xyz_mm": delta_mm.tolist(),
        })
    return knees, [z0, z1]


def spine_knee_translation_mm(hu: np.ndarray, activity: np.ndarray,
                              spacing_xyz: np.ndarray) -> dict:
    """Translation moving SPECT onto CT from spine ridge + left/right knees.

    Z comes from midline occupancy xcorr (handles ~10 mm device offset).
    Spine XY is the CT bone ridge vs SPECT activity COM in a ±12 voxel window
    around that ridge. Independent midline-band peaks are not used. Combined
    vector is the median of that local spine XY and available knees.
    Whole-body bone/hot COM is diagnostic only.
    """
    bone, hot = bone_hot_masks(hu, activity, hot_frac=0.28)
    occupied = _occupied_z(bone, 20.0)
    if occupied.size < 30:
        raise RuntimeError("Insufficient CT bone occupancy for spine/knee landmarks")
    caudal_low = _caudal_is_low_z(bone, occupied)
    lag_z, (x0, x1) = _midline_z_lag_vox(bone, hot)
    spine_xy = _spine_xy_median_vox(
        bone, hot, activity, occupied, caudal_low, lag_z, x0, x1)
    if spine_xy["slice_count"] < 20:
        raise RuntimeError(
            f"Insufficient spine ridge overlap ({spine_xy['slice_count']} slices); "
            "refuse occupancy fallback")
    t_spine = np.array([
        spine_xy["dx_vox"] * spacing_xyz[0],
        spine_xy["dy_vox"] * spacing_xyz[1],
        lag_z * spacing_xyz[2],
    ], dtype=float)
    knees, knee_z = _knee_deltas(
        bone, hot, occupied, caudal_low, lag_z,
        spine_xy["dx_vox"], spine_xy["dy_vox"], spacing_xyz)
    samples_xy = [t_spine[:2]]
    accepted = []
    rejected = []
    for knee in knees:
        delta = np.asarray(knee["delta_xyz_mm"], dtype=float)
        if np.any(np.abs(delta - t_spine) > 2.0):
            knee["rejected"] = "disagrees_with_spine_gt_2p0mm"
            rejected.append(knee)
        else:
            accepted.append(knee)
            samples_xy.append(delta[:2])
    t_xy = np.median(np.stack(samples_xy), axis=0)
    combined = np.array([t_xy[0], t_xy[1], t_spine[2]], dtype=float)
    return {
        "method": "spine_ridge_plus_knee_com",
        "translation_xyz_mm": combined.tolist(),
        "magnitude_mm": float(np.linalg.norm(combined)),
        "spine_translation_xyz_mm": t_spine.tolist(),
        "spine": {
            **spine_xy,
            "z_lag_vox": lag_z,
            "midline_x_band_vox": [x0, x1],
        },
        "knees": accepted,
        "knees_rejected": rejected,
        "knee_z_band_vox": knee_z,
        "caudal_is_low_z": bool(caudal_low),
        "landmark_count": int(1 + len(accepted)),
        "bone_voxels": int(bone.sum()),
        "hot_voxels": int(hot.sum()),
    }


def clip_translation_mm(translation_xyz_mm, limit_mm: float) -> np.ndarray:
    vec = np.asarray(translation_xyz_mm, dtype=float)
    if limit_mm < 0:
        raise RuntimeError("refine translation limit must be >= 0")
    return np.clip(vec, -limit_mm, limit_mm)


def _patient_z_span_mm(origin, normal, n_slices: int, dz_mm: float) -> list[float]:
    z0 = float(np.dot(np.asarray(origin, dtype=float), np.asarray(normal, dtype=float)))
    z1 = z0 + (int(n_slices) - 1) * float(dz_mm)
    return [min(z0, z1), max(z0, z1)]


def _fov_coverage(ct_span, spect_span) -> dict:
    overlap = [max(ct_span[0], spect_span[0]), min(ct_span[1], spect_span[1])]
    if overlap[1] < overlap[0]:
        overlap = [None, None]
    return {
        "ct_patient_z_mm": [float(ct_span[0]), float(ct_span[1])],
        "spect_dicom_patient_z_mm": [float(spect_span[0]), float(spect_span[1])],
        "overlap_patient_z_mm": overlap,
        "ct_beyond_spect_low_z_mm": float(max(0.0, spect_span[0] - ct_span[0])),
        "ct_beyond_spect_high_z_mm": float(max(0.0, ct_span[1] - spect_span[1])),
        "spect_beyond_ct_low_z_mm": float(max(0.0, ct_span[0] - spect_span[0])),
        "spect_beyond_ct_high_z_mm": float(max(0.0, spect_span[1] - ct_span[1])),
        "note": (
            "SPECT multi-bed FOV can start after and/or end before CT. "
            "CT-only anatomy at the SPECT start (often the head vertex) is coverage, "
            "not a Z pose residual; do not use FOV endpoints as landmarks."
        ),
    }


def bone_hot_masks(hu: np.ndarray, activity: np.ndarray,
                   hot_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    positive = activity[activity > 0]
    scale = max(float(np.percentile(positive, 99.0)), 1e-12) if positive.size else 1.0
    return hu > 400, activity > hot_frac * scale


def spect_activity_volume(series: list) -> np.ndarray:
    return np.stack([
        item.pixel_array.astype(np.float32) * float(getattr(item, "RescaleSlope", 1.0))
        + float(getattr(item, "RescaleIntercept", 0.0))
        for item in series
    ])


def _geometry(ds) -> dict:
    """Return the DICOM patient-space affine basis for one slice."""
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    if iop.size != 6 or not np.all(np.isfinite(iop)):
        raise RuntimeError("DICOM ImageOrientationPatient must contain six finite values")
    col_dir = iop[:3]
    row_dir = iop[3:]
    if not np.isclose(np.linalg.norm(col_dir), 1.0, atol=1e-4):
        raise RuntimeError("DICOM column direction is not unit length")
    if not np.isclose(np.linalg.norm(row_dir), 1.0, atol=1e-4):
        raise RuntimeError("DICOM row direction is not unit length")
    if not np.isclose(np.dot(col_dir, row_dir), 0.0, atol=1e-4):
        raise RuntimeError("DICOM row/column directions are not orthogonal")
    normal = np.cross(col_dir, row_dir)
    normal /= np.linalg.norm(normal)
    return {
        "col_dir": col_dir,
        "row_dir": row_dir,
        "normal": normal,
        "origin": np.asarray(ds.ImagePositionPatient, dtype=float),
        "pixel_spacing": np.asarray(ds.PixelSpacing, dtype=float),
    }


def _sorted_series(items: list) -> tuple[list, dict, np.ndarray]:
    if not items:
        raise RuntimeError("Empty DICOM series")
    geom = _geometry(items[0])
    for item in items[1:]:
        other = _geometry(item)
        if not np.allclose(other["col_dir"], geom["col_dir"], atol=1e-5):
            raise RuntimeError("DICOM series has inconsistent column directions")
        if not np.allclose(other["row_dir"], geom["row_dir"], atol=1e-5):
            raise RuntimeError("DICOM series has inconsistent row directions")
        if not np.allclose(other["pixel_spacing"], geom["pixel_spacing"], atol=1e-5):
            raise RuntimeError("DICOM series has inconsistent pixel spacing")
    ordered = sorted(
        items,
        key=lambda item: float(np.dot(np.asarray(item.ImagePositionPatient, dtype=float),
                                      geom["normal"])),
    )
    positions = np.asarray([
        np.dot(np.asarray(item.ImagePositionPatient, dtype=float), geom["normal"])
        for item in ordered
    ], dtype=float)
    if len(positions) > 1:
        diffs = np.diff(positions)
        spacing = float(np.median(diffs))
        if not np.allclose(diffs, spacing, rtol=0.02, atol=1e-3):
            raise RuntimeError("DICOM slice spacing is not regular in patient space")
        if spacing <= 0:
            raise RuntimeError("DICOM slice order is not increasing in patient space")
    else:
        spacing = float(getattr(ordered[0], "SliceThickness", 0.0))
    geom = dict(geom)
    geom["origin"] = np.asarray(ordered[0].ImagePositionPatient, dtype=float)
    geom["slice_spacing"] = spacing
    geom["iop"] = np.concatenate([geom["col_dir"], geom["row_dir"]])
    return ordered, geom, positions


def _discover_bed_dirs(root: Path) -> dict[int, Path]:
    """Find optional BedN DICOM directories without changing the old CLI."""
    result = {}
    for child in root.iterdir() if root.is_dir() else []:
        if not child.is_dir():
            continue
        match = re.fullmatch(r"Bed(\d+)", child.name, flags=re.IGNORECASE)
        if match:
            result[int(match.group(1))] = child
    return result


def _patient_affine_coords(target_origin, target_geom, target_spacing, shape,
                           ct_geom, ct_spacing):
    z, y, x = np.indices(shape, dtype=np.float32)
    world = (
        target_origin[:, None, None, None]
        + target_geom["normal"][:, None, None, None] * (z * target_spacing[2])[None]
        + target_geom["row_dir"][:, None, None, None] * (y * target_spacing[1])[None]
        + target_geom["col_dir"][:, None, None, None] * (x * target_spacing[0])[None]
    )
    delta = world - ct_geom["origin"][:, None, None, None]
    return np.array([
        np.sum(delta * ct_geom["normal"][:, None, None, None], axis=0) / ct_spacing[2],
        np.sum(delta * ct_geom["row_dir"][:, None, None, None], axis=0) / ct_spacing[1],
        np.sum(delta * ct_geom["col_dir"][:, None, None, None], axis=0) / ct_spacing[0],
    ])


def hu_to_mu(hu: np.ndarray, water_mu: float) -> np.ndarray:
    return water_mu * np.clip(1.0 + hu / 1000.0, 0.0, 3.0)


def stack_bed_blocks(volume: np.ndarray, bed_count: int, bed_nz: int,
                     overlap: int, reverse_local_z: bool) -> np.ndarray:
    stride = bed_nz - overlap
    blocks = [volume[index * stride:index * stride + bed_nz]
              for index in range(bed_count)]
    if any(block.shape[0] != bed_nz for block in blocks):
        raise RuntimeError("Registered CT cannot supply every reconstruction bed")
    if reverse_local_z:
        blocks = [block[::-1] for block in blocks]
    return np.concatenate(blocks, axis=0)


def write_mhd(path: Path, volume: np.ndarray, spacing: tuple[float, float, float],
              metadata: dict[str, str] | None = None) -> None:
    raw = path.with_suffix(".raw")
    np.asarray(volume, dtype="<f4").tofile(raw)
    z, y, x = volume.shape
    header = (
        "ObjectType = Image\nNDims = 3\n"
        f"DimSize = {x} {y} {z}\nElementSpacing = {spacing[0]} {spacing[1]} {spacing[2]}\n"
        "ElementType = MET_FLOAT\nElementByteOrderMSB = False\n")
    if metadata:
        header += "".join(f"{key} = {value}\n" for key, value in metadata.items())
    header += f"ElementDataFile = {raw.name}\n"
    path.write_text(header, encoding="ascii")


def _hot_colormap(t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    rgb = np.empty(t.shape + (3,), dtype=np.float32)
    rgb[..., 0] = np.clip(t / 0.375, 0.0, 1.0)
    rgb[..., 1] = np.clip((t - 0.375) / 0.375, 0.0, 1.0)
    rgb[..., 2] = np.clip((t - 0.75) / 0.25, 0.0, 1.0)
    return rgb


def _window01(volume: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros(volume.shape, dtype=np.float32)
    return np.clip((volume.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def fusion_rgb_volume(ct_hu: np.ndarray, activity: np.ndarray) -> np.ndarray:
    """CT gray + SPECT hot overlay, masked to CT coverage (HU > -900)."""
    gray = _window01(ct_hu, -200.0, 800.0)
    positive = activity[activity > 0]
    scale = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    scale = max(scale, 1e-12)
    hot = np.clip(activity.astype(np.float32) / scale, 0.0, 1.0)
    alpha = np.where(ct_hu > -900.0, np.clip(hot * 1.35, 0.0, 0.85), 0.0)
    rgb = _hot_colormap(hot)
    fused = gray[..., None] * (1.0 - alpha[..., None]) + rgb * alpha[..., None]
    return np.clip(np.rint(fused * 255.0), 0, 255).astype(np.uint8)


_PET_ONLY_TAGS = (
    "Units", "DecayCorrection", "CountsSource", "DoseCalibrationFactor",
    "FrameReferenceTime", "ImageIndex", "ImagesInAcquisition", "NumberOfFrames",
    "NumberOfTimeSlices", "RadiopharmaceuticalInformationSequence",
    "ReconstructionMethod", "SUVType", "SeriesType", "AcquisitionDateTime",
    "AcquisitionNumber", "CorrectedImage",
)


def _copy_geometry_dataset(template, sop_class, series_uid: str, instance_uid: str,
                           series_number: int, instance_number: int,
                           series_description: str):
    ds = deepcopy(template)
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = sop_class
    ds.file_meta.MediaStorageSOPInstanceUID = instance_uid
    ds.file_meta.ImplementationClassUID = generate_uid()
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = instance_uid
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number
    ds.SeriesDescription = series_description
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def _drop_tags(ds, names) -> None:
    for name in names:
        if name in ds:
            delattr(ds, name)


def _dicom_ds(value: float) -> str:
    text = f"{float(value):.8g}"
    if len(text) > 16:
        text = f"{float(value):.6e}"
    return text[:16]


def _save_mono_slice(ds, pixels, out_path: Path, signed: bool, slope: float,
                     intercept: float) -> None:
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = int(pixels.shape[0])
    ds.Columns = int(pixels.shape[1])
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1 if signed else 0
    ds.RescaleSlope = _dicom_ds(slope)
    ds.RescaleIntercept = _dicom_ds(intercept)
    ds.PixelData = np.ascontiguousarray(pixels).tobytes()
    for tag in ("PixelPaddingValue", "SmallestImagePixelValue", "LargestImagePixelValue",
                "PlanarConfiguration"):
        if tag in ds:
            delattr(ds, tag)
    ds.save_as(out_path, write_like_original=False)


def write_aligned_dicom_overlay(out_dir: Path, templates: list, ct_hu: np.ndarray,
                                spect_activity: np.ndarray) -> dict:
    """Write CT / SPECT / fusion series that share SPECT FoR and per-slice IPP.

    Registration lives in the CT voxels. IPP is copied from the original SPECT
    slices so a viewer overlay is index-aligned without a second origin shift.
    """
    if ct_hu.shape != spect_activity.shape:
        raise RuntimeError(
            f"Aligned DICOM CT {ct_hu.shape} does not match SPECT {spect_activity.shape}")
    if len(templates) != ct_hu.shape[0]:
        raise RuntimeError(
            f"SPECT template count {len(templates)} does not match volume Z {ct_hu.shape[0]}")
    out_dir = Path(out_dir)
    ct_dir = out_dir / "ct"
    spect_dir = out_dir / "spect"
    fusion_dir = out_dir / "fusion"
    for folder in (ct_dir, spect_dir, fusion_dir):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.dcm"):
            old.unlink()

    study_uid = str(templates[0].StudyInstanceUID)
    frame_uid = str(templates[0].FrameOfReferenceUID)
    ct_series = generate_uid()
    spect_series = generate_uid()
    fusion_series = generate_uid()
    fusion = fusion_rgb_volume(ct_hu, spect_activity)
    # Device CT is unsigned stored = HU + 1000. SPECT on this scanner is
    # Modality=PT / PET SOP Class so BIViewer fusion accepts PET/CT.
    ct_stored = np.clip(np.rint(ct_hu + 1000.0), 0, 65535).astype(np.uint16)
    spect_sop = str(getattr(
        templates[0], "SOPClassUID", PositronEmissionTomographyImageStorage))
    spect_modality = str(getattr(templates[0], "Modality", "PT") or "PT")
    spect_desc = str(getattr(templates[0], "SeriesDescription", None) or "SPECT Reconstruction")

    for z, template in enumerate(templates):
        instance = z + 1
        name = f"image_{z:04d}.dcm"
        ct_ds = _copy_geometry_dataset(
            template, CTImageStorage, ct_series, generate_uid(), 101, instance, "CT")
        ct_ds.StudyInstanceUID = study_uid
        ct_ds.FrameOfReferenceUID = frame_uid
        ct_ds.Modality = "CT"
        ct_ds.ImageType = ["DERIVED", "PRIMARY", "AXIAL"]
        ct_ds.Manufacturer = "BAY IMAGIMG"
        ct_ds.WindowCenter = "300"
        ct_ds.WindowWidth = "1500"
        ct_ds.RescaleType = "HU"
        _drop_tags(ct_ds, _PET_ONLY_TAGS + ("ManufacturerModelName",))
        _save_mono_slice(ct_ds, ct_stored[z], ct_dir / name, False, 1.0, -1000.0)

        spect_ds = _copy_geometry_dataset(
            template, spect_sop, spect_series, generate_uid(), 102, instance, spect_desc)
        spect_ds.StudyInstanceUID = study_uid
        spect_ds.FrameOfReferenceUID = frame_uid
        spect_ds.Modality = spect_modality
        spect_ds.save_as(spect_dir / name, write_like_original=False)

        fusion_ds = _copy_geometry_dataset(
            template, SecondaryCaptureImageStorage, fusion_series, generate_uid(),
            103, instance, "overlay RGB")
        fusion_ds.StudyInstanceUID = study_uid
        fusion_ds.FrameOfReferenceUID = frame_uid
        fusion_ds.Modality = "OT"
        fusion_ds.SamplesPerPixel = 3
        fusion_ds.PhotometricInterpretation = "RGB"
        fusion_ds.PlanarConfiguration = 0
        fusion_ds.Rows = int(fusion.shape[1])
        fusion_ds.Columns = int(fusion.shape[2])
        fusion_ds.BitsAllocated = 8
        fusion_ds.BitsStored = 8
        fusion_ds.HighBit = 7
        fusion_ds.PixelRepresentation = 0
        fusion_ds.ImageType = ["DERIVED", "SECONDARY"]
        _drop_tags(fusion_ds, _PET_ONLY_TAGS + (
            "RescaleSlope", "RescaleIntercept", "RescaleType", "WindowCenter",
            "WindowWidth", "PixelPaddingValue"))
        fusion_ds.PixelData = np.ascontiguousarray(fusion[z]).tobytes()
        fusion_ds.save_as(fusion_dir / name, write_like_original=False)

    readme = out_dir / "README.txt"
    readme.write_text(
        "Device fusion viewer (PET/CT, PET/MR, SPECT/CT) needs one CT series\n"
        "and one PT/NM series. Import only ct/ and spect/. Do not import fusion/\n"
        "(RGB OT) and do not import the parent folder as a whole.\n"
        "\n"
        "ct/     registered CT (Modality=CT), HU stored as unsigned + intercept -1000\n"
        "spect/  SPECT copied as Modality=PT (device original tagging)\n"
        "fusion/ RGB overlay for generic viewers; not for BIViewer fusion\n"
        "\n"
        "Both series share FrameOfReferenceUID and ImagePositionPatient.\n"
        "The rigid shift is in the CT voxels. Head-end CT without SPECT is FOV.\n",
        encoding="utf-8",
    )
    return {
        "dir": str(out_dir),
        "slice_count": int(ct_hu.shape[0]),
        "shape_zyx": [int(v) for v in ct_hu.shape],
        "study_instance_uid": study_uid,
        "frame_of_reference_uid": frame_uid,
        "series": {
            "ct": {"dir": str(ct_dir), "series_instance_uid": ct_series, "series_number": 101},
            "spect": {"dir": str(spect_dir), "series_instance_uid": spect_series,
                      "series_number": 102},
            "fusion": {"dir": str(fusion_dir), "series_instance_uid": fusion_series,
                       "series_number": 103},
        },
        "ipp_source": "original_spect_dicom",
        "registration_in": "ct_voxels",
        "readme": str(readme),
    }


def _overlay_rgb(bone_view: np.ndarray, hot_view: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*bone_view.shape, 3), dtype=float)
    rgb[..., 1] = bone_view.astype(float)
    rgb[..., 0] = hot_view.astype(float)
    return rgb


def _crop_xy(volume: np.ndarray, z: int, x: float, y: float, half: int = 18):
    z = int(np.clip(z, 0, volume.shape[0] - 1))
    x0 = int(np.clip(round(x) - half, 0, volume.shape[2] - 2 * half))
    y0 = int(np.clip(round(y) - half, 0, volume.shape[1] - 2 * half))
    return volume[z, y0:y0 + 2 * half, x0:x0 + 2 * half]


def write_registration_qa(path: Path, hu: np.ndarray, activity: np.ndarray,
                          residual=None) -> None:
    bone, hot = bone_hot_masks(hu, activity, hot_frac=0.28)
    # SPECT DICOM Z is longer than CT; padded air is HU=-1000. Overlay MIPs
    # must not treat extra SPECT slices as a cranial shift.
    hot_in_ct = hot & (hu > -900.0)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for row, axis, name in ((0, 1, "coronal"), (1, 2, "sagittal")):
        ct_view = bone.max(axis)
        axes[row, 0].imshow(ct_view, origin="lower", cmap="gray", aspect="auto")
        axes[row, 0].set_title(f"CT bone HU>400 {name}")
        axes[row, 1].imshow(hot.max(axis), origin="lower", cmap="hot", aspect="auto")
        axes[row, 1].set_title(f"SPECT hot {name}")
        axes[row, 2].imshow(_overlay_rgb(ct_view, hot_in_ct.max(axis)),
                            origin="lower", aspect="auto")
        axes[row, 2].set_title("overlay in CT coverage")
        if residual and axis == 1:
            spine = residual.get("spine") or {}
            band = spine.get("z_band_vox")
            if band:
                axes[row, 0].axhline(band[0], color="cyan", lw=0.8)
                axes[row, 0].axhline(band[1], color="cyan", lw=0.8)
            knee_band = residual.get("knee_z_band_vox")
            if knee_band:
                axes[row, 0].axhline(knee_band[0], color="orange", lw=0.8)
                axes[row, 0].axhline(knee_band[1], color="orange", lw=0.8)
    zooms = [("spine", None)]
    for knee in (residual or {}).get("knees") or []:
        zooms.append((knee["name"], knee))
    while len(zooms) < 3:
        zooms.append((f"empty{len(zooms)}", None))
    spine = (residual or {}).get("spine") or {}
    nx, ny = hu.shape[2], hu.shape[1]
    band = spine.get("midline_x_band_vox") or [int(nx * 0.38), int(nx * 0.62)]
    x0b, x1b = int(band[0]), int(band[1])
    spine_z = int(np.mean(spine["z_band_vox"])) if spine.get("z_band_vox") else hu.shape[0] // 2
    for delta in range(0, 16):
        hit = False
        for cand in (spine_z + delta, spine_z - delta):
            if 0 <= cand < bone.shape[0] and int(bone[cand, :, x0b:x1b].sum()) >= 4:
                spine_z = cand
                hit = True
                break
        if hit:
            break
    y_prof = bone[spine_z, :, x0b:x1b].sum(axis=1)
    spine_x = 0.5 * (float(x0b) + float(x1b))
    spine_y = float(np.argmax(y_prof)) if y_prof.max() > 0 else ny / 2.0
    for col, (title, knee) in enumerate(zooms[:3]):
        if knee is not None:
            xyz = knee["ct_xyz_vox"]
            ct_p = _crop_xy(bone, int(round(xyz[2])), xyz[0], xyz[1])
            sp_p = _crop_xy(hot, int(round(xyz[2])), xyz[0], xyz[1])
            axes[2, col].set_title(title)
        else:
            ct_p = _crop_xy(bone, spine_z, spine_x, spine_y, half=24)
            sp_p = _crop_xy(hot, spine_z, spine_x, spine_y, half=24)
            axes[2, col].set_title(f"spine z={spine_z}")
        axes[2, col].imshow(_overlay_rgb(ct_p, sp_p), origin="lower")
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _measure_grid_residual(method: str, hu_grid: np.ndarray, activity: np.ndarray,
                           spacing_xyz: np.ndarray) -> dict:
    if method == "spine_knee":
        metrics = spine_knee_translation_mm(hu_grid, activity, spacing_xyz)
    else:
        bone, hot = bone_hot_masks(hu_grid, activity)
        metrics = occupancy_translation_mm(bone, hot, spacing_xyz)
        metrics["bone_voxels"] = int(bone.sum())
        metrics["hot_voxels"] = int(hot.sum())
    bone, hot = bone_hot_masks(hu_grid, activity)
    metrics["occupancy_diagnostic"] = occupancy_translation_mm(bone, hot, spacing_xyz)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--spect-dicom-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default=None,
        help="device160: 108x108x160 stitch (SPECT DICOM grid). "
             "recon220: 120x120x220 per-bed stack for SPECT_recon -attMap/-regCT")
    parser.add_argument(
        "--recon-manifest", type=Path,
        help="Effective full matrix-grid geometry (e.g. SC60); replaces the PSC profile. "
             "SPECT DICOM must be exported from this untransformed reference run")
    parser.add_argument(
        "--hu-output", type=Path,
        help="Optional HU volume on the same grid as --output")
    parser.add_argument(
        "--spect-activity-output", type=Path,
        help="Write the SPECT DICOM activity (sorted patient-Z) as MHD")
    parser.add_argument(
        "--reconstruction-mu-output", type=Path,
        help="Optional per-bed stacked mu-map for output-overlap reconstruction")
    parser.add_argument(
        "--attenuation-mu-output", type=Path,
        help="Optional per-bed CTAC mu-map in increasing bed-number order; "
             "each bed uses local system-matrix Z order")
    parser.add_argument(
        "--regct-hu-output", type=Path,
        help="Optional HU volume for -regCT in reconstruction-storage Z order")
    parser.add_argument(
        "--z-bed-flip", action="store_true", default=None,
        help="Reverse each -attMap bed block to match reconstruction -zbedflip 1; "
             "does not change -regCT or whole-output orientation")
    parser.add_argument("--beds", default=None,
                        help="Consecutive physical bed numbers, e.g. 2,3,4")
    parser.add_argument("--bed-nz", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--bed-step-mm", type=float, default=None)
    parser.add_argument("--water-mu", type=float, default=.0152)
    parser.add_argument("--activity", type=Path)
    parser.add_argument(
        "--fixed-translation-xyz", type=_parse_fixed_translation,
        help="Frozen device translation (mm) applied before per-scan refine. "
             "Use mdp_device to load multipinhole/config/ct_spect_registration.ini, or x,y,z")
    parser.add_argument(
        "--registration-ini", type=Path, default=MDP_DEVICE_REGISTRATION_INI,
        help="INI with [Registration] translation_xyz_mm; also the format written next to --output")
    parser.add_argument(
        "--refine-translation-limit-mm", type=float, default=2.0,
        help="Clamp per-scan 3-DOF refine after the frozen translation; ignored without --fixed-translation-xyz")
    parser.add_argument(
        "--residual-method", choices=("spine_knee", "occupancy"), default="spine_knee",
        help="spine_knee: spine ridge + knees (MDP). occupancy: whole-body bone/hot (diagnostic only)")
    parser.add_argument("--measure-residual", action="store_true",
                        help="Measure SPECT→CT translation residual on the SPECT DICOM grid")
    parser.add_argument("--apply-residual", action="store_true",
                        help="Apply the measured per-scan residual (clamped when a frozen translation is set)")
    parser.add_argument("--residual-gate-mm", type=float, default=0.3,
                        help="Provisional spine/knee translation-magnitude gate (mm); diagnostic until TRE is calibrated")
    parser.add_argument("--bed-step-tolerance-mm", type=float, default=0.75,
                        help="Diagnostic tolerance for per-bed DICOM origin checks")
    parser.add_argument(
        "--spect-reference-bed", type=int, default=None,
        help="Physical bed represented by a flat --spect-dicom-dir [1]")
    parser.add_argument(
        "--bed1-origin-xyz", type=_parse_xyz,
        help="Calibrated Bed1 grid origin in CT patient space; otherwise infer it once from SPECT DICOM")
    parser.add_argument(
        "--allow-ct-padding", action="store_true",
        help="Allow target voxels outside the CT volume to be filled as air")
    parser.add_argument(
        "--aligned-dicom-dir", type=Path,
        help="Write CT/SPECT/fusion DICOM series on the SPECT DICOM grid "
             "(shared FoR and IPP; registration is in the CT voxels)")
    args = parser.parse_args()
    grid = load_recon_grid(args.recon_manifest) if args.recon_manifest else None
    if grid:
        if args.profile is not None:
            raise RuntimeError("--recon-manifest and --profile are mutually exclusive")
        if isinstance(args.fixed_translation_xyz, str) and args.registration_ini == MDP_DEVICE_REGISTRATION_INI:
            raise RuntimeError("Manifest mode requires device-specific registration; do not reuse the PSC MDP preset")
        args.profile = "recon_manifest"
        profile = grid
        for key in ("bed_nz", "overlap", "bed_step_mm", "z_bed_flip"):
            value = getattr(args, key)
            if value is not None and value != grid[key]:
                raise RuntimeError(f"--{key.replace('_', '-')} conflicts with reconstruction manifest")
            setattr(args, key, grid[key])
        if args.beds is not None and [int(v) for v in args.beds.split(",")] != grid["beds"]:
            raise RuntimeError("--beds conflicts with reconstruction manifest")
        args.beds = ",".join(str(bed) for bed in grid["beds"])
    else:
        args.profile = args.profile or "device160"
        profile = PROFILES[args.profile]
        args.beds = args.beds or "1,2,3,4"
        args.bed_step_mm = DEVICE_BED_STEP_MM if args.bed_step_mm is None else args.bed_step_mm
        args.z_bed_flip = bool(args.z_bed_flip)
    if isinstance(args.fixed_translation_xyz, str):
        try:
            args.fixed_translation_xyz = load_frozen_translation(args.registration_ini)
        except (RuntimeError, configparser.Error, argparse.ArgumentTypeError) as exc:
            raise SystemExit(f"--fixed-translation-xyz mdp_device: {exc}") from exc
    if args.bed_nz is None:
        args.bed_nz = int(profile["bed_nz"])
    if args.overlap is None:
        args.overlap = int(profile["overlap"])

    beds = [int(value) for value in args.beds.split(",")]
    if args.spect_reference_bed is None:
        args.spect_reference_bed = beds[0] if grid else 1
    if (not beds or beds != list(range(beds[0], beds[0] + len(beds))) or
            not all(bed in (grid["beds"] if grid else DEVICE_PHYSICAL_BEDS) for bed in beds)):
        raise RuntimeError("--beds must be consecutive and match the selected geometry")
    if not grid and not np.isclose(args.bed_step_mm, DEVICE_BED_STEP_MM, atol=1e-9):
        raise RuntimeError("6Module bed_step_mm is fixed at 16.8")
    if args.bed_nz != int(profile["bed_nz"]) or args.overlap != int(profile["overlap"]):
        raise RuntimeError(
            f"profile {args.profile} requires bed_nz={profile['bed_nz']}, "
            f"overlap={profile['overlap']}; got bed_nz={args.bed_nz}, overlap={args.overlap}")
    if args.spect_reference_bed not in (beds if grid else DEVICE_PHYSICAL_BEDS):
        raise RuntimeError("--spect-reference-bed must belong to the selected geometry")
    if not np.isfinite(args.water_mu) or args.water_mu <= 0:
        raise RuntimeError("--water-mu must be finite and positive (mm^-1)")
    if args.apply_residual and not args.measure_residual:
        raise RuntimeError("--apply-residual requires --measure-residual")
    nz = args.bed_nz + (len(beds) - 1) * (args.bed_nz - args.overlap)
    args.ct_dir = resolve_ct_dir(args.ct_dir)
    ct, ct_geom, _ = _sorted_series(dicoms(args.ct_dir))
    bed_dirs = _discover_bed_dirs(args.spect_dicom_dir)
    bed_series = {}
    bed_geometries = {}
    if bed_dirs:
        missing = [bed for bed in beds if bed not in bed_dirs]
        if missing:
            raise RuntimeError(f"Missing DICOM Bed directories: {missing}")
        for bed in beds:
            ordered, geometry, _ = _sorted_series(dicoms(bed_dirs[bed]))
            bed_series[bed] = ordered
            bed_geometries[bed] = geometry
        spect = bed_series[beds[0]]
        spect_geom = bed_geometries[beds[0]]
        first_for = str(getattr(spect[0], "FrameOfReferenceUID", ""))
        for bed in beds[1:]:
            if str(getattr(bed_series[bed][0], "FrameOfReferenceUID", "")) != first_for:
                raise RuntimeError("SPECT Bed DICOM FrameOfReferenceUID values differ")
    else:
        spect, spect_geom, _ = _sorted_series(dicoms(args.spect_dicom_dir))
    c0, s0 = ct[0], spect[0]
    if c0.FrameOfReferenceUID != s0.FrameOfReferenceUID:
        raise RuntimeError("CT and SPECT FrameOfReferenceUID differ")
    if not np.allclose(ct_geom["iop"], spect_geom["iop"], atol=1e-5):
        raise RuntimeError("CT and SPECT orientations differ; residual rigid registration is required")
    ct_patient_position = str(getattr(c0, "PatientPosition", ""))
    spect_patient_position = str(getattr(s0, "PatientPosition", ""))
    if ct_patient_position and spect_patient_position and ct_patient_position != spect_patient_position:
        raise RuntimeError("CT and SPECT PatientPosition differ")

    hu = np.stack([d.pixel_array.astype(np.float32) * float(d.RescaleSlope) +
                   float(d.RescaleIntercept) for d in ct])
    ct_origin = ct_geom["origin"]
    ct_spacing = np.array([ct_geom["pixel_spacing"][1],
                           ct_geom["pixel_spacing"][0],
                           ct_geom["slice_spacing"]])
    target_spacing = np.array([spect_geom["pixel_spacing"][1],
                               spect_geom["pixel_spacing"][0],
                               spect_geom["slice_spacing"]])
    required_spacing = grid["spacing"] if grid else DEVICE_TARGET_SPACING_XYZ_MM
    if not np.allclose(target_spacing, required_spacing, rtol=0, atol=1e-5):
        raise RuntimeError(
            f"SPECT DICOM spacing {target_spacing.tolist()} does not match target {required_spacing.tolist()}")
    spect_xy = (int(s0.Rows), int(s0.Columns))
    required_xy = grid["xy"] if grid else DEVICE_IMAGE_SHAPE_XY
    if spect_xy != required_xy:
        raise RuntimeError(
            f"SPECT DICOM XY grid {spect_xy} does not match target {required_xy}")
    if grid:
        if not bed_series and args.spect_reference_bed != beds[0]:
            raise RuntimeError("Flat manifest DICOM must start at the first selected bed")
        for series, geometry in ([(bed_series[b], bed_geometries[b]) for b in beds]
                                 if bed_series else [(spect, spect_geom)]):
            spacing = [geometry["pixel_spacing"][1], geometry["pixel_spacing"][0],
                       geometry["slice_spacing"]]
            expected_nz = args.bed_nz if bed_series else nz
            if (len(series) != expected_nz or
                    (int(series[0].Rows), int(series[0].Columns)) != required_xy or
                    not np.allclose(spacing, required_spacing, rtol=0, atol=1e-5) or
                    not np.allclose(geometry["iop"], spect_geom["iop"], rtol=0, atol=1e-5)):
                raise RuntimeError("Every SPECT DICOM bed must match the manifest full-grid geometry")
    target_xy = tuple(int(v) for v in profile["xy"])
    target_grid_stride = (args.bed_nz - args.overlap) * target_spacing[2]
    if (not grid or len(beds) > 1) and not np.isclose(
            target_grid_stride, args.bed_step_mm, rtol=0, atol=1e-4 if grid else 1e-6):
        raise RuntimeError(
            f"bed step {args.bed_step_mm:.6f} mm does not match grid stride "
            f"{target_grid_stride:.6f} mm")

    measured_bed_origins = {}
    bed_step_measurements = []
    bed_origin_residuals_mm = {}
    normal = spect_geom["normal"]
    if args.bed1_origin_xyz is not None:
        bed1_origin = args.bed1_origin_xyz
        bed_origin_source = "explicit_device_calibration"
    elif bed_series:
        reference_bed = beds[0]
        reference_origin = bed_geometries[reference_bed]["origin"]
        bed1_origin = reference_origin - (reference_bed - 1) * args.bed_step_mm * normal
        bed_origin_source = f"inferred_from_Bed{reference_bed}_dicom_ipp"
    else:
        reference_bed = args.spect_reference_bed
        bed1_origin = (spect_geom["origin"] -
                       (reference_bed - 1) * args.bed_step_mm * normal)
        bed_origin_source = f"inferred_from_flat_Bed{reference_bed}_dicom_ipp"
    fixed_bed_origins = _fixed_bed_origins(
        bed1_origin, normal, args.bed_step_mm, beds if grid else DEVICE_PHYSICAL_BEDS)
    origin_shift = (
        float(profile["xy_pad_vox"]) * target_spacing[0] * spect_geom["col_dir"]
        + float(profile["xy_pad_vox"]) * target_spacing[1] * spect_geom["row_dir"]
        + float(profile["z_pad_vox"]) * target_spacing[2] * normal
    )
    target_origin = fixed_bed_origins[beds[0]] - origin_shift
    dicom_target_origin = np.asarray(target_origin, dtype=float).copy()
    dicom_spect_origin = np.asarray(spect_geom["origin"], dtype=float).copy()
    fixed_translation = (
        np.asarray(args.fixed_translation_xyz, dtype=float)
        if args.fixed_translation_xyz is not None else None)
    if fixed_translation is not None:
        spect_geom = dict(spect_geom)
        spect_geom["origin"] = dicom_spect_origin + fixed_translation
        target_origin = dicom_target_origin + fixed_translation

    if bed_series:
        measured_bed_origins = {
            bed: bed_geometries[bed]["origin"].tolist() for bed in beds
        }
        for bed in beds:
            residual = bed_geometries[bed]["origin"] - fixed_bed_origins[bed]
            bed_origin_residuals_mm[str(bed)] = residual.tolist()
            if np.linalg.norm(residual) > args.bed_step_tolerance_mm:
                warnings.warn(
                    f"Bed{bed} DICOM origin differs from fixed device mapping by "
                    f"{np.linalg.norm(residual):.4f} mm")
        for previous, current in zip(beds, beds[1:]):
            measured = float(np.dot(bed_geometries[current]["origin"] -
                                    bed_geometries[previous]["origin"], normal))
            bed_step_measurements.append(measured)
            if abs(measured - args.bed_step_mm) > args.bed_step_tolerance_mm:
                warnings.warn(
                    f"Bed{current}-Bed{previous} DICOM step {measured:.4f} mm "
                    f"differs from fixed {args.bed_step_mm:.4f} mm")
    coords = _patient_affine_coords(
        target_origin, spect_geom, target_spacing,
        (nz, target_xy[0], target_xy[1]), ct_geom, ct_spacing)
    inside_ct, ct_coverage_by_bed = _coverage_summary(
        coords, hu.shape, beds, args.bed_nz, args.overlap)
    target_hu, outside_voxels = _resample_hu(
        hu, coords, hu.shape, args.allow_ct_padding)
    mu = hu_to_mu(target_hu, args.water_mu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_mhd(args.output, mu, tuple(target_spacing))
    if args.hu_output:
        args.hu_output.parent.mkdir(parents=True, exist_ok=True)
        write_mhd(args.hu_output, target_hu, tuple(target_spacing))

    spect_activity = spect_activity_volume(spect)
    if args.spect_activity_output:
        args.spect_activity_output.parent.mkdir(parents=True, exist_ok=True)
        write_mhd(args.spect_activity_output, spect_activity, tuple(target_spacing[:2].tolist() + [float(spect_geom["slice_spacing"])]))

    residual_metrics = None
    spect_dicom_hu = None
    spect_spacing = np.array([
        spect_geom["pixel_spacing"][1],
        spect_geom["pixel_spacing"][0],
        spect_geom["slice_spacing"],
    ])
    spect_shape = (len(spect), int(s0.Rows), int(s0.Columns))
    if args.measure_residual or args.apply_residual:
        spect_coords = _patient_affine_coords(
            spect_geom["origin"], spect_geom, spect_spacing, spect_shape,
            ct_geom, ct_spacing)
        spect_dicom_hu, spect_dicom_outside = _resample_hu(
            hu, spect_coords, hu.shape, args.allow_ct_padding)
        if spect_activity.shape != spect_dicom_hu.shape:
            raise RuntimeError(
                f"SPECT activity {spect_activity.shape} does not match "
                f"DICOM-grid CT {spect_dicom_hu.shape}")
        residual_metrics = _measure_grid_residual(
            args.residual_method, spect_dicom_hu, spect_activity, spect_spacing)
        residual_metrics["ct_outside_voxels_on_spect_dicom"] = spect_dicom_outside
        residual_metrics["gate_mm"] = args.residual_gate_mm
        residual_metrics["gate_maturity"] = "diagnostic"
        residual_metrics["residual_method"] = args.residual_method
        residual_metrics["within_gate"] = bool(
            residual_metrics["magnitude_mm"] <= args.residual_gate_mm)
        if args.apply_residual:
            requested = np.array(residual_metrics["translation_xyz_mm"], dtype=float)
            if residual_metrics["magnitude_mm"] <= args.residual_gate_mm:
                applied = np.zeros(3, dtype=float)
                residual_metrics["refine_skipped"] = "already_within_gate"
            elif fixed_translation is not None:
                applied = clip_translation_mm(requested, args.refine_translation_limit_mm)
                residual_metrics["refine_requested_xyz_mm"] = requested.tolist()
                residual_metrics["refine_limit_mm"] = args.refine_translation_limit_mm
                residual_metrics["refine_clipped"] = bool(
                    not np.allclose(applied, requested, atol=1e-9))
            else:
                applied = requested
            residual_metrics["applied_translation_xyz_mm"] = applied.tolist()
            residual_metrics["applied"] = bool(not np.allclose(applied, 0.0, atol=1e-9))
            if residual_metrics["applied"]:
                target_origin = target_origin + applied
                spect_geom = dict(spect_geom)
                spect_geom["origin"] = np.asarray(spect_geom["origin"], dtype=float) + applied
                coords = _patient_affine_coords(
                    target_origin, spect_geom, target_spacing,
                    (nz, target_xy[0], target_xy[1]), ct_geom, ct_spacing)
                inside_ct, ct_coverage_by_bed = _coverage_summary(
                    coords, hu.shape, beds, args.bed_nz, args.overlap)
                target_hu, outside_voxels = _resample_hu(
                    hu, coords, hu.shape, args.allow_ct_padding)
                mu = hu_to_mu(target_hu, args.water_mu)
                write_mhd(args.output, mu, tuple(target_spacing))
                if args.hu_output:
                    write_mhd(args.hu_output, target_hu, tuple(target_spacing))
                spect_coords = _patient_affine_coords(
                    spect_geom["origin"], spect_geom, spect_spacing, spect_shape,
                    ct_geom, ct_spacing)
                spect_dicom_hu, spect_dicom_outside = _resample_hu(
                    hu, spect_coords, hu.shape, args.allow_ct_padding)
                after = _measure_grid_residual(
                    args.residual_method, spect_dicom_hu, spect_activity, spect_spacing)
                residual_metrics["after_apply"] = after
                residual_metrics["magnitude_mm_after_apply"] = after["magnitude_mm"]
                residual_metrics["translation_xyz_mm_after_apply"] = after["translation_xyz_mm"]
                residual_metrics["within_gate"] = bool(
                    after["magnitude_mm"] <= args.residual_gate_mm)
            else:
                residual_metrics["magnitude_mm_after_apply"] = residual_metrics["magnitude_mm"]
                residual_metrics["translation_xyz_mm_after_apply"] = requested.tolist()
                residual_metrics["within_gate"] = bool(
                    residual_metrics["magnitude_mm"] <= args.residual_gate_mm)
        if spect_dicom_hu is not None:
            spect_hu_path = args.output.with_name(args.output.stem + "_spectgrid_hu.mhd")
            write_mhd(spect_hu_path, spect_dicom_hu, tuple(spect_spacing.tolist()))
            residual_metrics["spect_grid_hu"] = str(spect_hu_path)

    reconstruction_mu = None
    attenuation_mu = None
    regct_hu = None
    bed_metadata = {
        "SPECTBedOrder": "increasing_bed_number",
        "SPECTBeds": ",".join(str(bed) for bed in beds),
    }
    if (args.reconstruction_mu_output or args.attenuation_mu_output or
            args.regct_hu_output):
        reconstruction_mu = stack_bed_blocks(
            mu, len(beds), args.bed_nz, args.overlap, False)
        if args.reconstruction_mu_output:
            args.reconstruction_mu_output.parent.mkdir(parents=True, exist_ok=True)
            write_mhd(args.reconstruction_mu_output, reconstruction_mu,
                      tuple(target_spacing), bed_metadata | {
                          "SPECTMapRole": "reconstruction_mu_storage",
                          "SPECTZLayout": "reconstruction_storage_z",
                          "SPECTIntensityUnits": "mm^-1",
                      })
        if args.attenuation_mu_output:
            attenuation_mu = stack_bed_blocks(
                mu, len(beds), args.bed_nz, args.overlap, args.z_bed_flip)
            args.attenuation_mu_output.parent.mkdir(parents=True, exist_ok=True)
            write_mhd(args.attenuation_mu_output, attenuation_mu,
                      tuple(target_spacing), bed_metadata | {
                          "SPECTMapRole": "attenuation_local_matrix_z",
                          "SPECTZLayout": "local_matrix_z",
                          "SPECTIntensityUnits": "mm^-1",
                      })
        if args.regct_hu_output:
            regct_hu = stack_bed_blocks(
                target_hu, len(beds), args.bed_nz, args.overlap, False)
            args.regct_hu_output.parent.mkdir(parents=True, exist_ok=True)
            write_mhd(args.regct_hu_output, regct_hu, tuple(target_spacing),
                      bed_metadata | {
                          "SPECTMapRole": "regularizer_reconstruction_storage",
                          "SPECTZLayout": "reconstruction_storage_z",
                          "SPECTIntensityUnits": "HU",
                      })

    manifest = {
        "decision": "geometry_ready_mu_calibration_provisional",
        "device_geometry_profile": args.profile,
        "device_geometry": {
            "physical_beds": beds if grid else list(DEVICE_PHYSICAL_BEDS),
            "bed_nz": args.bed_nz,
            "voxel_xyz_mm": target_spacing.tolist(),
            "bed_overlap": args.overlap,
            "bed_step_mm": args.bed_step_mm,
            "image_shape_xy": list(target_xy),
            "spect_dicom_shape_xy": list(spect_xy),
            "xy_pad_vox": int(profile["xy_pad_vox"]),
            "z_pad_vox": int(profile["z_pad_vox"]),
        },
        "frame_of_reference_uid": str(c0.FrameOfReferenceUID),
        "ct_dir_resolved": str(args.ct_dir),
        "ct_shape_zyx": list(hu.shape), "target_shape_zyx": list(mu.shape),
        "beds": beds, "bed_nz": args.bed_nz, "overlap": args.overlap,
        "bed_step_mm": args.bed_step_mm,
        "bed_number_direction": "increasing_bed_number=+patient_normal",
        "ct_origin_xyz": ct_origin.tolist(), "target_origin_xyz": target_origin.tolist(),
        "dicom_target_origin_xyz": dicom_target_origin.tolist(),
        "fixed_translation_xyz_mm": (
            None if fixed_translation is None else fixed_translation.tolist()),
        "origin_shift_xyz": origin_shift.tolist(),
        "bed1_origin_xyz": bed1_origin.tolist(),
        "fixed_bed_origins_xyz": {
            str(bed): origin.tolist() for bed, origin in fixed_bed_origins.items()
        },
        "ct_spacing_xyz": ct_spacing.tolist(), "target_spacing_xyz": target_spacing.tolist(),
        "ct_image_orientation_patient": ct_geom["iop"].tolist(),
        "spect_image_orientation_patient": spect_geom["iop"].tolist(),
        "patient_space_normal": normal.tolist(),
        "slice_sort": "ascending dot(ImagePositionPatient, normal)",
        "bed_origin_source": bed_origin_source,
        "measured_bed_origins_xyz": measured_bed_origins,
        "bed_origin_residuals_mm": bed_origin_residuals_mm,
        "bed_step_measurements_mm": bed_step_measurements,
        "target_grid_stride_mm": target_grid_stride,
        "bed_step_tolerance_mm": args.bed_step_tolerance_mm,
        "ct_coverage_by_bed": ct_coverage_by_bed,
        "ct_outside_voxels": outside_voxels,
        "ct_padding_allowed": bool(args.allow_ct_padding),
        "fov_coverage": _fov_coverage(
            _patient_z_span_mm(ct_origin, normal, hu.shape[0], ct_spacing[2]),
            _patient_z_span_mm(
                dicom_spect_origin, normal, len(spect), float(spect_geom["slice_spacing"])),
        ),
        "patient_position": ct_patient_position or spect_patient_position,
        "hu_to_mu": "mu=water_mu*clip(1+HU/1000,0,3)", "water_mu_mm-1": args.water_mu,
        "registration_assumption": (
            "shared FrameOfReferenceUID + ImagePosition/Orientation/PixelSpacing; "
            "explicit bed grid mapping; optional frozen device translation; "
            "per-scan clamped 3-DOF spine+knee refine using CT-anchored local "
            "SPECT COM; no rotation; no per-bed registration"),
        "mu_map": str(args.output),
    }
    if args.recon_manifest:
        manifest["reconstruction_manifest"] = str(args.recon_manifest.resolve())
        manifest["reference_requirement"] = "untransformed full matrix-grid DICOM; no extra export flips"
    if residual_metrics is not None:
        manifest["dicom_translation_residual"] = residual_metrics
        before = float(residual_metrics["magnitude_mm"])
        after = float(residual_metrics.get("magnitude_mm_after_apply", before))
        residual_metrics["within_gate_before"] = before <= args.residual_gate_mm
        residual_metrics["within_gate_after"] = after <= args.residual_gate_mm
        if before <= args.residual_gate_mm:
            manifest["decision"] = "fixed_or_dicom_within_provisional_0p3mm_gate"
        elif args.apply_residual and after <= args.residual_gate_mm:
            if fixed_translation is not None:
                manifest["decision"] = "fixed_plus_clamped_refine_within_provisional_0p3mm_gate"
            else:
                manifest["decision"] = "landmark_translation_within_provisional_0p3mm_gate"
        else:
            manifest["decision"] = "spine_knee_residual_exceeds_provisional_0p3mm_gate"
    if args.hu_output:
        manifest["hu_map"] = str(args.hu_output)
    if args.spect_activity_output:
        manifest["spect_activity"] = str(args.spect_activity_output)
    if args.reconstruction_mu_output:
        manifest["reconstruction_mu_map"] = str(args.reconstruction_mu_output)
        manifest["reconstruction_shape_zyx"] = list(reconstruction_mu.shape)
        manifest["reconstruction_layout"] = (
            "increasing bed-number blocks; reconstruction-storage Z; overlap duplicated")
    if args.attenuation_mu_output:
        manifest["attenuation_mu_map"] = str(args.attenuation_mu_output)
        manifest["attenuation_shape_zyx"] = list(attenuation_mu.shape)
        manifest["attenuation_layout"] = (
            "increasing bed-number blocks; local system-matrix Z; overlap duplicated")
        manifest["attenuation_z_bed_flip"] = bool(args.z_bed_flip)
    if args.regct_hu_output:
        manifest["regct_hu_map"] = str(args.regct_hu_output)
        manifest["regct_layout"] = (
            "increasing bed-number blocks; reconstruction-storage Z; overlap duplicated")
        manifest["regct_note"] = (
            "HU is not normalized; -regCtSigma is in HU, not mu. "
            "Whole-output zOutFlip is never applied to input maps")
    qa_hu = None
    qa_activity = None
    if args.activity:
        qa_activity = read_mhd(args.activity)
        qa_hu = target_hu
        if qa_activity.shape != mu.shape:
            raise RuntimeError(f"Activity {qa_activity.shape} does not match mu-map {mu.shape}")
    elif spect_dicom_hu is not None:
        qa_hu = spect_dicom_hu
        qa_activity = spect_activity
    if qa_hu is not None and qa_activity is not None:
        qa = args.output.with_name(args.output.stem + "_registration_qa.png")
        qa_residual = residual_metrics if args.residual_method == "spine_knee" else None
        if residual_metrics is not None and residual_metrics.get("after_apply"):
            qa_residual = residual_metrics["after_apply"]
        write_registration_qa(qa, qa_hu, qa_activity, qa_residual)
        manifest["registration_qa"] = str(qa)
    if args.aligned_dicom_dir:
        if spect_dicom_hu is None:
            spect_coords = _patient_affine_coords(
                spect_geom["origin"], spect_geom, spect_spacing, spect_shape,
                ct_geom, ct_spacing)
            spect_dicom_hu, _ = _resample_hu(
                hu, spect_coords, hu.shape, args.allow_ct_padding)
        if spect_activity.shape != spect_dicom_hu.shape:
            raise RuntimeError(
                f"SPECT activity {spect_activity.shape} does not match "
                f"DICOM-grid CT {spect_dicom_hu.shape}")
        manifest["aligned_dicom"] = write_aligned_dicom_overlay(
            args.aligned_dicom_dir, spect, spect_dicom_hu, spect_activity)
    refine_xyz = np.zeros(3, dtype=float)
    if residual_metrics is not None and residual_metrics.get("applied_translation_xyz_mm") is not None:
        refine_xyz = np.asarray(residual_metrics["applied_translation_xyz_mm"], dtype=float)
    frozen_xyz = (
        np.zeros(3, dtype=float) if fixed_translation is None
        else np.asarray(fixed_translation, dtype=float))
    registration_ini = args.output.with_name(args.output.stem + "_registration.ini")
    write_registration_ini(registration_ini, {
        "profile": str(args.profile),
        "fixed_translation_xyz_mm": _xyz_ini(frozen_xyz),
        "refine_translation_xyz_mm": _xyz_ini(refine_xyz),
        "total_translation_xyz_mm": _xyz_ini(frozen_xyz + refine_xyz),
        "origin_shift_xyz_mm": _xyz_ini(origin_shift),
        "target_origin_xyz_mm": _xyz_ini(target_origin),
        "dicom_target_origin_xyz_mm": _xyz_ini(dicom_target_origin),
        "beds": ",".join(str(bed) for bed in beds),
        "bed_number_direction": "increasing_bed_number=+patient_normal",
        "attenuation_layout": "local_matrix_z",
        "regct_layout": "reconstruction_storage_z",
        "z_bed_flip": "1" if args.z_bed_flip else "0",
        "frozen_ini": str(args.registration_ini),
    })
    manifest["registration_ini"] = str(registration_ini)
    args.output.with_name(args.output.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def self_check() -> None:
    assert np.allclose(hu_to_mu(np.array([-1000., 0., 1000.]), .0152), [0, .0152, .0304])
    identity = {
        "col_dir": np.array([1., 0., 0.]),
        "row_dir": np.array([0., 1., 0.]),
        "normal": np.array([0., 0., 1.]),
        "origin": np.zeros(3),
    }
    coords = _patient_affine_coords(
        np.zeros(3), identity, np.ones(3), (2, 2, 2), identity, np.ones(3))
    assert np.allclose(coords, np.indices((2, 2, 2), dtype=np.float32))
    rotated = {
        "col_dir": np.array([0., 1., 0.]),
        "row_dir": np.array([-1., 0., 0.]),
        "normal": np.array([0., 0., 1.]),
        "origin": np.zeros(3),
    }
    coords = _patient_affine_coords(
        np.zeros(3), rotated, np.ones(3), (2, 2, 2), rotated, np.ones(3))
    assert np.allclose(coords, np.indices((2, 2, 2), dtype=np.float32))
    bed_origins = _fixed_bed_origins(np.zeros(3), np.array([0., 0., 1.]), 16.8)
    assert np.allclose(bed_origins[1], [0., 0., 0.])
    assert np.allclose(bed_origins[5], [0., 0., 67.2])
    for start in DEVICE_PHYSICAL_BEDS:
        for count in range(1, 6 - start + 1):
            beds = list(range(start, start + count))
            nz = DEVICE_BED_NZ + (count - 1) * (DEVICE_BED_NZ - DEVICE_BED_OVERLAP)
            assert nz == 160 + (count - 1) * 56
            assert np.allclose(bed_origins[beds[0]][2], (start - 1) * 16.8)
    assert PROFILES["recon220"]["bed_nz"] == 220
    assert 220 + 4 * 56 == 444
    assert 220 * 5 == 1100
    bed_ramp = np.arange(6, dtype=np.float32)[:, None, None]
    storage_stack = stack_bed_blocks(bed_ramp, 2, 4, 2, False)
    local_stack = stack_bed_blocks(bed_ramp, 2, 4, 2, True)
    assert np.array_equal(storage_stack[:, 0, 0], [0, 1, 2, 3, 2, 3, 4, 5])
    assert np.array_equal(local_stack[:, 0, 0], [3, 2, 1, 0, 5, 4, 3, 2])
    blob = np.zeros((32, 16, 16), dtype=bool)
    blob[10:14, 6:10, 6:10] = True
    moved = np.roll(blob, 4, axis=0)
    occ = occupancy_translation_mm(blob, moved, np.array([0.3, 0.3, 0.3]))
    assert occ["z_lag_vox"] == -4
    assert abs(occ["translation_xyz_mm"][2] + 1.2) < 1e-6
    ct_span = _patient_z_span_mm([0.7, -1.2, 27.7], [0.0, 0.0, 1.0], 512, 0.2)
    spect_span = _patient_z_span_mm([46.06, 39.51, 37.05], [0.0, 0.0, 1.0], 384, 0.3)
    cov = _fov_coverage(ct_span, spect_span)
    assert abs(ct_span[0] - 27.7) < 1e-9
    assert abs(spect_span[0] - 37.05) < 1e-9
    assert abs(cov["ct_beyond_spect_low_z_mm"] - 9.35) < 1e-6
    assert np.allclose(clip_translation_mm([3.0, -4.0, 0.5], 2.0), [2.0, -2.0, 0.5])
    hu_ph = np.full((80, 48, 48), -1000.0, dtype=np.float32)
    hu_ph[10:16, 22:26, 22:26] = 850.0
    hu_ph[18:62, 20:26, 21:27] = 800.0
    hu_ph[58:68, 18:28, 6:14] = 900.0
    hu_ph[58:68, 18:28, 34:42] = 900.0
    act_ph = np.zeros_like(hu_ph)
    act_ph[hu_ph > 0] = 100.0
    shifted = np.roll(np.roll(np.roll(act_ph, 5, 0), -2, 1), 3, 2)
    rec = spine_knee_translation_mm(hu_ph, shifted, np.array([0.3, 0.3, 0.3]))
    t = np.asarray(rec["translation_xyz_mm"])
    assert rec["landmark_count"] >= 1
    assert rec["spine"]["pairing"] == "ct_anchored_local_com"
    assert abs(t[0] + 0.9) < 0.45
    assert abs(t[1] - 0.6) < 0.45
    assert abs(t[2] + 1.5) < 0.45
    blob = np.zeros((40, 40))
    blob[18:23, 10:15] = 1.0
    com = _window_com(blob, 12.0, 20.0, 8, 1.0)
    assert com is not None
    assert abs(com[0] - 12.0) < 0.2 and abs(com[1] - 20.0) < 0.2
    hot = _hot_colormap(np.array([0.0, 0.375, 1.0], dtype=np.float32))
    assert np.allclose(hot[0], [0.0, 0.0, 0.0], atol=1e-6)
    assert hot[1, 0] == 1.0 and hot[1, 1] <= 1e-6
    assert np.allclose(hot[2], [1.0, 1.0, 1.0], atol=1e-6)
    fused = fusion_rgb_volume(
        np.array([[[-1000.0, 400.0]]], dtype=np.float32),
        np.array([[[10.0, 10.0]]], dtype=np.float32))
    assert fused.shape == (1, 1, 2, 3)
    assert fused[0, 0, 0].sum() < fused[0, 0, 1].sum()
    frozen = load_frozen_translation(MDP_DEVICE_REGISTRATION_INI)
    assert np.allclose(frozen, [-0.225, 1.6, -11.1])
    frozen_cfg = _read_ini(MDP_DEVICE_REGISTRATION_INI)
    assert frozen_cfg.getfloat(REGISTRATION_SECTION, "refine_limit_mm") == 2.0
    assert frozen_cfg.getfloat(REGISTRATION_SECTION, "residual_gate_mm") == 0.3
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder) / "applied.ini"
        write_registration_ini(tmp, {
            "fixed_translation_xyz_mm": _xyz_ini(frozen),
            "refine_translation_xyz_mm": "0,0,0.6",
            "total_translation_xyz_mm": _xyz_ini(frozen + np.array([0.0, 0.0, 0.6])),
            "z_bed_flip": "1",
        })
        applied = _read_ini(tmp)
        loaded = _parse_xyz(applied.get(REGISTRATION_SECTION, "total_translation_xyz_mm"))
        assert np.allclose(loaded, frozen + np.array([0.0, 0.0, 0.6]))
        assert applied.get(REGISTRATION_SECTION, "z_bed_flip") == "1"


if __name__ == "__main__":
    self_check()
    raise SystemExit(main())
