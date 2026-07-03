#!/usr/bin/env python3
"""Run compact-symmetric cylinder smoke QA on an MHD/RAW reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_mhd_stats import read_mhd


def _disk_masks(shape: tuple[int, int, int], inside_radius: float, outside_radius: float):
    _, ny, nx = shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = rr <= inside_radius
    outside = rr >= outside_radius
    return inside, outside, xx, yy, cx, cy


def _safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def compute_qa(image: np.ndarray, inside_radius: float, outside_radius: float) -> dict:
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={image.shape}")

    finite = np.isfinite(image)
    zc = image.shape[0] // 2
    yc = image.shape[1] // 2
    xc = image.shape[2] // 2
    xy = image[zc, :, :]
    xz = image[:, yc, :]
    yz = image[:, :, xc]
    inside_mask, outside_mask, xx, yy, cx, cy = _disk_masks(
        image.shape, inside_radius, outside_radius
    )

    inside = xy[inside_mask]
    outside = xy[outside_mask]
    left_inside = xy[inside_mask & (xx < cx)]
    right_inside = xy[inside_mask & (xx > cx)]
    top_inside = xy[inside_mask & (yy < cy)]
    bottom_inside = xy[inside_mask & (yy > cy)]

    inside_mean = float(np.mean(inside)) if inside.size else float("nan")
    outside_mean = float(np.mean(outside)) if outside.size else float("nan")
    inside_std = float(np.std(inside)) if inside.size else float("nan")

    left_right = _safe_ratio(
        abs(float(np.mean(left_inside)) - float(np.mean(right_inside))), inside_mean
    )
    top_bottom = _safe_ratio(
        abs(float(np.mean(top_inside)) - float(np.mean(bottom_inside))), inside_mean
    )
    central_disk_cv = _safe_ratio(inside_std, inside_mean)
    outside_inside = _safe_ratio(outside_mean, inside_mean)

    z_profile = image.mean(axis=(1, 2))
    z_half_diff = _safe_ratio(
        abs(float(image[:zc].mean()) - float(image[zc:].mean())), float(image.mean())
    )

    checks = {
        "finite_ratio_is_1": bool(float(finite.mean()) == 1.0),
        "no_nan_inf": bool(np.all(finite)),
        "nonnegative": bool(float(np.nanmin(image)) >= -1e-9),
        "positive_sum": bool(float(np.nansum(image)) > 0.0),
        "inside_mean_gt_outside_mean": bool(inside_mean > outside_mean),
        "outside_inside_lt_0p35": bool(outside_inside < 0.35),
        "left_right_diff_lt_0p20": bool(left_right < 0.20),
        "top_bottom_diff_lt_0p20": bool(top_bottom < 0.20),
        "central_disk_cv_lt_0p35": bool(central_disk_cv < 0.35),
        "z_half_diff_lt_0p20": bool(z_half_diff < 0.20),
    }
    calibrated_1iter_checks = {
        key: checks[key]
        for key in (
            "finite_ratio_is_1",
            "no_nan_inf",
            "nonnegative",
            "positive_sum",
            "left_right_diff_lt_0p20",
            "top_bottom_diff_lt_0p20",
            "central_disk_cv_lt_0p35",
            "z_half_diff_lt_0p20",
        )
    }

    return {
        "shape_zyx": list(image.shape),
        "dim_xyz": [int(image.shape[2]), int(image.shape[1]), int(image.shape[0])],
        "finite_ratio": float(finite.mean()),
        "min": float(np.nanmin(image)),
        "max": float(np.nanmax(image)),
        "sum": float(np.nansum(image)),
        "global_mean": float(np.nanmean(image)),
        "global_std": float(np.nanstd(image)),
        "inside_radius_px": float(inside_radius),
        "outside_radius_px": float(outside_radius),
        "center_xy_inside_mean": inside_mean,
        "center_xy_outside_mean": outside_mean,
        "outside_inside_mean_ratio": outside_inside,
        "left_right_mean_diff_over_inside": left_right,
        "top_bottom_mean_diff_over_inside": top_bottom,
        "central_disk_cv": central_disk_cv,
        "z_half_mean_diff_over_global": z_half_diff,
        "z_profile_min": float(np.min(z_profile)),
        "z_profile_max": float(np.max(z_profile)),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "calibrated_1iter_checks": calibrated_1iter_checks,
        "calibrated_1iter_passed": bool(all(calibrated_1iter_checks.values())),
        "calibrated_1iter_note": (
            "The 6module 1-iteration legacy CUDA baseline has a ring-like "
            "boundary enhancement, so inside/outside absolute contrast is "
            "reported but excluded from this calibrated smoke gate."
        ),
    }


def save_views(image: np.ndarray, out_png: Path, title: str) -> None:
    zc = image.shape[0] // 2
    yc = image.shape[1] // 2
    xc = image.shape[2] // 2
    views = [
        ("XY z=center", image[zc, :, :]),
        ("XZ y=center", image[:, yc, :]),
        ("YZ x=center", image[:, :, xc]),
    ]
    vmin = float(np.percentile(image, 1.0))
    vmax = float(np.percentile(image, 99.0))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (label, data) in zip(axes, views):
        im = ax.imshow(data, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input reconstruction .mhd")
    parser.add_argument("--out-json", required=True, help="Output QA JSON")
    parser.add_argument("--out-png", required=True, help="Output center-view PNG")
    parser.add_argument(
        "--inside-radius",
        type=float,
        default=25.0,
        help="Central XY disk radius in pixels for inside metrics",
    )
    parser.add_argument(
        "--outside-radius",
        type=float,
        default=45.0,
        help="Central XY radius in pixels for outside/background metrics",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    image = read_mhd(image_path)
    metrics = compute_qa(image, args.inside_radius, args.outside_radius)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_views(image, Path(args.out_png), image_path.name)
    print(json.dumps(metrics, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
