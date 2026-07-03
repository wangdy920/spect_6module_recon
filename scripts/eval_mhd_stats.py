#!/usr/bin/env python3
"""Evaluate image statistics for a .mhd/.raw volume."""

import argparse
import json
from pathlib import Path

import numpy as np


def _read_mhd_header(path: Path) -> dict:
    header = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        header[key.strip()] = value.strip()
    return header


def _dtype_from_header(header: dict) -> np.dtype:
    element_type = header.get("ElementType", "")
    byte_order_msb = header.get("ElementByteOrderMSB", "False").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )
    if element_type == "MET_DOUBLE":
        dtype = np.dtype("f8")
    elif element_type == "MET_FLOAT":
        dtype = np.dtype("f4")
    else:
        raise ValueError(f"Unsupported ElementType: {element_type}")
    dtype = dtype.newbyteorder(">" if byte_order_msb else "<")
    return dtype


def read_mhd(path: Path) -> np.ndarray:
    header = _read_mhd_header(path)
    dims_raw = header.get("DimSize", "")
    if not dims_raw:
        raise ValueError("DimSize missing in MHD header")
    dims = tuple(int(v) for v in dims_raw.split())
    if len(dims) != 3:
        raise ValueError(f"Expected 3D volume, got DimSize={dims_raw}")

    data_file = header.get("ElementDataFile", "")
    if not data_file:
        raise ValueError("ElementDataFile missing in MHD header")
    raw_path = (path.parent / data_file).resolve()
    dtype = _dtype_from_header(header)

    arr = np.fromfile(raw_path, dtype=dtype)
    expected = dims[0] * dims[1] * dims[2]
    if arr.size != expected:
        raise ValueError(
            f"Raw voxel count mismatch: got {arr.size}, expected {expected} ({dims})"
        )

    # MHD DimSize uses x y z, reshape as z y x for array operations.
    arr = arr.reshape((dims[2], dims[1], dims[0]))
    return arr


def compute_stats(image: np.ndarray, mask: np.ndarray | None) -> dict:
    flat = image.ravel()
    out = {
        "voxels": int(flat.size),
        "global_mean": float(np.mean(flat)),
        "global_std": float(np.std(flat)),
        "global_p95": float(np.percentile(flat, 95.0)),
        "global_p99": float(np.percentile(flat, 99.0)),
        "global_sum": float(np.sum(flat)),
    }
    if mask is None:
        return out

    mask = mask.astype(bool)
    inside = image[mask]
    outside = image[~mask]
    out["mask_inside_voxels"] = int(inside.size)
    out["mask_outside_voxels"] = int(outside.size)
    if inside.size > 0:
        out["inside_mean"] = float(np.mean(inside))
        out["inside_std"] = float(np.std(inside))
        out["inside_p95"] = float(np.percentile(inside, 95.0))
        out["inside_p99"] = float(np.percentile(inside, 99.0))
    else:
        out["inside_mean"] = 0.0
        out["inside_std"] = 0.0
        out["inside_p95"] = 0.0
        out["inside_p99"] = 0.0
    if outside.size > 0:
        out["outside_mean"] = float(np.mean(outside))
        out["outside_std"] = float(np.std(outside))
        out["outside_p95"] = float(np.percentile(outside, 95.0))
        out["outside_p99"] = float(np.percentile(outside, 99.0))
        out["outside_sum"] = float(np.sum(outside))
    else:
        out["outside_mean"] = 0.0
        out["outside_std"] = 0.0
        out["outside_p95"] = 0.0
        out["outside_p99"] = 0.0
        out["outside_sum"] = 0.0

    out["inside_outside_contrast"] = float(
        out["inside_mean"] / max(out["outside_mean"], 1e-12)
    )
    out["outside_sum_ratio"] = float(out["outside_sum"] / max(out["global_sum"], 1e-12))
    if abs(out["inside_mean"]) > 1e-12:
        out["inside_cv_pct"] = float(100.0 * out["inside_std"] / abs(out["inside_mean"]))
    else:
        out["inside_cv_pct"] = float("nan")
    if abs(out["outside_mean"]) > 1e-12:
        out["outside_cv_pct"] = float(
            100.0 * out["outside_std"] / abs(out["outside_mean"])
        )
    else:
        out["outside_cv_pct"] = float("nan")
    return out


def compute_reference_stats(
    image: np.ndarray, ref: np.ndarray, mask: np.ndarray | None
) -> dict:
    if image.shape != ref.shape:
        raise ValueError(f"Reference shape {ref.shape} != image shape {image.shape}")

    diff = image - ref
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    ref_mean = float(np.mean(ref))
    ref_std = float(np.std(ref))
    ref_range = float(np.max(ref) - np.min(ref))

    out = {
        "rmse_ref": rmse,
        "mae_ref": mae,
        "nrmse_ref_mean": float(rmse / max(abs(ref_mean), 1e-12)),
        "nrmse_ref_std": float(rmse / max(ref_std, 1e-12)),
        "nrmse_ref_range": float(rmse / max(ref_range, 1e-12)),
    }
    if mask is not None:
        mask = mask.astype(bool)
        inside = diff[mask]
        outside = diff[~mask]
        if inside.size > 0:
            out["rmse_ref_inside"] = float(np.sqrt(np.mean(inside * inside)))
        else:
            out["rmse_ref_inside"] = float("nan")
        if outside.size > 0:
            out["rmse_ref_outside"] = float(np.sqrt(np.mean(outside * outside)))
        else:
            out["rmse_ref_outside"] = float("nan")
    return out


def compute_seam_stats(image: np.ndarray, seam_z_list: list[int], seam_window: int) -> dict:
    if not seam_z_list:
        return {}
    if seam_window < 0:
        raise ValueError("seam_window must be >= 0")

    z_profile = np.mean(image, axis=(1, 2))
    grad = np.abs(np.diff(z_profile))
    if grad.size == 0:
        return {}

    seam_mask = np.zeros(grad.shape[0], dtype=bool)
    seam_peaks: list[float] = []
    valid_seams: list[int] = []

    for seam_z in seam_z_list:
        grad_idx = seam_z - 1
        if grad_idx < 0 or grad_idx >= grad.shape[0]:
            continue
        begin = max(0, grad_idx - seam_window)
        end = min(grad.shape[0], grad_idx + seam_window + 1)
        seam_mask[begin:end] = True
        seam_peaks.append(float(np.max(grad[begin:end])))
        valid_seams.append(int(seam_z))

    if not seam_peaks:
        return {}

    baseline = grad[~seam_mask]
    seam_peak_max = float(max(seam_peaks))
    seam_baseline_mean = (
        float(np.mean(baseline)) if baseline.size > 0 else float("nan")
    )
    seam_score = (
        float(seam_peak_max / max(seam_baseline_mean, 1e-12))
        if np.isfinite(seam_baseline_mean)
        else float("nan")
    )

    return {
        "seam_count": int(len(valid_seams)),
        "seam_window": int(seam_window),
        "seam_indices_z": valid_seams,
        "seam_peaks": seam_peaks,
        "seam_peak_max": seam_peak_max,
        "seam_baseline_mean": seam_baseline_mean,
        "seam_score": seam_score,
    }


def _parse_seam_z_list(raw: str) -> list[int]:
    if not raw.strip():
        return []
    out: list[int] = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Target reconstruction MHD")
    parser.add_argument(
        "--mask-source",
        default="",
        help="Optional MHD used to derive mask (same shape as --image)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=None,
        help="Threshold for mask-source (mask = source > threshold)",
    )
    parser.add_argument(
        "--ref-image",
        default="",
        help="Optional reference MHD for RMSE/NRMSE evaluation",
    )
    parser.add_argument(
        "--seam-z-list",
        default="",
        help="Optional seam z-index list (comma-separated, start index of next bed)",
    )
    parser.add_argument(
        "--seam-window",
        type=int,
        default=2,
        help="Half window size for seam gradient pickup",
    )
    args = parser.parse_args()

    image = read_mhd(Path(args.image))
    mask = None
    if args.mask_source and args.mask_threshold is not None:
        src = read_mhd(Path(args.mask_source))
        if src.shape != image.shape:
            raise ValueError(
                f"Mask source shape {src.shape} != image shape {image.shape}"
            )
        mask = src > float(args.mask_threshold)

    stats = compute_stats(image, mask)
    if args.ref_image:
        ref = read_mhd(Path(args.ref_image))
        stats.update(compute_reference_stats(image, ref, mask))
    seam_z_list = _parse_seam_z_list(args.seam_z_list)
    if seam_z_list:
        stats.update(compute_seam_stats(image, seam_z_list, args.seam_window))
    print(json.dumps(stats, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
