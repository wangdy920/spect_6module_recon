#!/usr/bin/env python3
"""Register one multi-bed SPECT volume to CT from bone-center ridges."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
from scipy.optimize import least_squares, minimize

from refine_ct_spect_bone_distance import (
    affine_from_params,
    fov_weight,
    inverse_points,
    overlay,
    qa_levels,
    robust_distance,
    sample_distance,
    thin_points,
    transform_points,
)


def center_ridges(mask: np.ndarray, score: np.ndarray, size_xy: int,
                  minimum: float) -> np.ndarray:
    local_max = maximum_filter(score, size=(1, size_xy, size_xy), mode="nearest")
    return mask & (score >= local_max - 1e-7) & (score >= minimum)


def optimize_translation(objective, start: np.ndarray) -> object:
    def translation_objective(offset: np.ndarray) -> float:
        params = np.r_[np.zeros(3), offset]
        return objective(params)

    return minimize(
        translation_objective, start, method="Powell", bounds=[(-2, 2)] * 3,
        options={"maxiter": 120, "xtol": 1e-4, "ftol": 1e-5})


def distances(distance: np.ndarray, points: np.ndarray, params: np.ndarray,
              center: np.ndarray, spacing: float) -> np.ndarray:
    return sample_distance(
        distance, transform_points(points, params, center), spacing)


def distance_summary(distance: np.ndarray, points: np.ndarray, params: np.ndarray,
                     center: np.ndarray, spacing: float) -> dict:
    values = distances(distance, points, params, center, spacing)
    return {
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.percentile(values, 90)),
        "mean_mm": float(np.mean(values)),
    }


def band_summaries(distance: np.ndarray, points: np.ndarray, params: np.ndarray,
                   center: np.ndarray, spacing: float, nz: int) -> list[dict]:
    moved = transform_points(points, params, center)
    values = sample_distance(distance, moved, spacing)
    edges = np.linspace(0, nz * spacing, 5)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (moved[:, 2] >= lo) & (moved[:, 2] < hi)
        if not np.any(selected):
            continue
        rows.append({
            "z_mm": [float(lo), float(hi)],
            "points": int(selected.sum()),
            "median_mm": float(np.median(values[selected])),
            "p90_mm": float(np.percentile(values[selected], 90)),
        })
    return rows


def resample_activity(activity: sitk.Image, reference: sitk.Image,
                      affine: sitk.Transform) -> np.ndarray:
    moved = sitk.Resample(
        activity, reference, affine.GetInverse(), sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(moved)


def write_image_qa(path: Path, fixed: np.ndarray, before: np.ndarray,
                   translated: np.ndarray, rigid: np.ndarray) -> None:
    views = [
        ("coronal", 1),
        ("sagittal", 2),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 10))
    for row, (name, axis) in enumerate(views):
        ct = fixed.max(axis)
        images = [ct, before.max(axis), translated.max(axis), rigid.max(axis)]
        axes[row, 0].imshow(ct, origin="lower", cmap="gray", vmin=0, vmax=1,
                            aspect="auto")
        axes[row, 0].set_title(f"CT bone {name}")
        for col, title in enumerate(
                ["", "DICOM geometry", "center translation", "center small-rigid"]):
            if col:
                axes[row, col].imshow(
                    overlay(ct, images[col]), origin="lower", aspect="auto")
                axes[row, col].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_surface_comparison(path: Path, fixed: np.ndarray, before: np.ndarray,
                             surface: np.ndarray, translated: np.ndarray,
                             rigid: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17, 10))
    columns = [
        (None, "CT bone"),
        (before, "DICOM geometry"),
        (surface, "bone-surface rigid"),
        (translated, "bone-center translation"),
        (rigid, "bone-center small-rigid"),
    ]
    for row, (name, axis) in enumerate([("coronal", 1), ("sagittal", 2)]):
        ct = fixed.max(axis)
        for col, (volume, title) in enumerate(columns):
            if volume is None:
                axes[row, col].imshow(ct, origin="lower", cmap="gray",
                                      vmin=0, vmax=1, aspect="auto")
            else:
                axes[row, col].imshow(
                    overlay(ct, volume.max(axis)), origin="lower", aspect="auto")
            axes[row, col].set_title(f"{title} {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_center_qa(path: Path, fixed_points: np.ndarray,
                    moving_points: np.ndarray, translation: np.ndarray,
                    rigid: np.ndarray, center: np.ndarray) -> None:
    candidates = [
        ("DICOM geometry", moving_points),
        ("center translation", transform_points(moving_points, translation, center)),
        ("center small-rigid", transform_points(moving_points, rigid, center)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for col, (title, moving) in enumerate(candidates):
        axes[0, col].scatter(fixed_points[:, 0], fixed_points[:, 2], s=1,
                             c="lime", alpha=.35)
        axes[0, col].scatter(moving[:, 0], moving[:, 2], s=2,
                             c="red", alpha=.45)
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("x mm"); axes[0, col].set_ylabel("z mm")
        axes[1, col].scatter(fixed_points[:, 1], fixed_points[:, 2], s=1,
                             c="lime", alpha=.35)
        axes[1, col].scatter(moving[:, 1], moving[:, 2], s=2,
                             c="red", alpha=.45)
        axes[1, col].set_xlabel("y mm"); axes[1, col].set_ylabel("z mm")
        for row in range(2):
            axes[row, col].set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_axial_qa(path: Path, fixed: np.ndarray, before: np.ndarray,
                   translated: np.ndarray, rigid: np.ndarray,
                   levels: list[int], labels: tuple[str, str] =
                   ("center translation", "center small-rigid")) -> None:
    fig, axes = plt.subplots(len(levels), 4, figsize=(12, 3 * len(levels)))
    for row, z in enumerate(levels):
        axes[row, 0].imshow(fixed[z], origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"CT bone z={z}")
        for col, (volume, title) in enumerate([
                (before, "DICOM geometry"),
                (translated, labels[0]),
                (rigid, labels[1])], start=1):
            axes[row, col].imshow(overlay(fixed[z], volume[z]), origin="lower")
            axes[row, col].set_title(title)
        for axis in axes[row]:
            axis.set_xticks([]); axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def anatomical_anchors(fixed_points: np.ndarray,
                       moving_points: np.ndarray,
                       extent: np.ndarray, fixed_score: np.ndarray,
                       moving_score: np.ndarray, spacing: float) -> list[dict]:
    rois = {
        "hip_left": (.20, .45, .10, .27),
        "hip_right": (.55, .80, .10, .27),
        "spine_lower": (.40, .60, .28, .40),
        "spine_middle": (.40, .60, .40, .52),
        "spine_upper": (.40, .60, .52, .62),
        "shoulder_left": (.20, .45, .55, .68),
        "shoulder_right": (.55, .80, .55, .68),
    }
    moving_index = np.rint(moving_points / spacing).astype(int)[:, [2, 1, 0]]
    moving_values = moving_score[tuple(moving_index.T)]
    fixed_index = np.rint(fixed_points / spacing).astype(int)[:, [2, 1, 0]]
    fixed_values = fixed_score[tuple(fixed_index.T)]
    anchors = []
    for name, (x0, x1, z0, z1) in rois.items():
        fixed_selected = (
            (fixed_points[:, 0] >= x0 * extent[0]) &
            (fixed_points[:, 0] < x1 * extent[0]) &
            (fixed_points[:, 2] >= z0 * extent[2]) &
            (fixed_points[:, 2] < z1 * extent[2]))
        moving_selected = (
            (moving_points[:, 0] >= x0 * extent[0]) &
            (moving_points[:, 0] < x1 * extent[0]) &
            (moving_points[:, 2] >= z0 * extent[2]) &
            (moving_points[:, 2] < z1 * extent[2]))
        if fixed_selected.sum() < 10 or moving_selected.sum() < 3:
            continue
        moving_candidates = np.flatnonzero(moving_selected)
        peak = moving_points[
            moving_candidates[np.argmax(moving_values[moving_candidates])]]
        moving_local = moving_selected & (
            np.linalg.norm(moving_points - peak, axis=1) <= 1.5)
        moving_weights = moving_values[moving_local]
        moving_center = np.average(
            moving_points[moving_local], axis=0, weights=moving_weights)
        fixed_local = fixed_selected & (
            np.linalg.norm(fixed_points - moving_center, axis=1) <= 2.5)
        if fixed_local.sum() < 3:
            continue
        fixed_weights = fixed_values[fixed_local]
        fixed_center = np.average(
            fixed_points[fixed_local], axis=0, weights=fixed_weights)
        offset = fixed_center - moving_center
        anchors.append({
            "name": name,
            "fixed_points": int(fixed_local.sum()),
            "moving_points": int(moving_local.sum()),
            "moving_xyz_mm": moving_center,
            "fixed_xyz_mm": fixed_center,
            "translation_mm": offset,
        })
    return anchors


def anchor_residuals(anchors: list[dict], params: np.ndarray,
                     center: np.ndarray) -> dict:
    moving = np.asarray([item["moving_xyz_mm"] for item in anchors])
    fixed = np.asarray([item["fixed_xyz_mm"] for item in anchors])
    values = np.linalg.norm(transform_points(moving, params, center) - fixed, axis=1)
    return {item["name"]: float(value) for item, value in zip(anchors, values)}


def anchor_group_holdout(anchors: list[dict], center: np.ndarray) -> list[dict]:
    """Fit translation on two anatomical groups and score the held-out group."""
    moving = np.asarray([item["moving_xyz_mm"] for item in anchors])
    fixed = np.asarray([item["fixed_xyz_mm"] for item in anchors])
    groups = {
        "hips": np.array([item["name"].startswith("hip_") for item in anchors]),
        "spine": np.array([item["name"].startswith("spine_") for item in anchors]),
        "shoulders": np.array([item["name"].startswith("shoulder_") for item in anchors]),
    }
    rows = []
    for held_out, test_mask in groups.items():
        train_mask = ~test_mask
        if train_mask.sum() < 3 or not test_mask.any():
            continue
        offset = np.median(fixed[train_mask] - moving[train_mask], axis=0)
        params = np.r_[np.zeros(3), offset]
        before = np.linalg.norm(moving[test_mask] - fixed[test_mask], axis=1)
        after = np.linalg.norm(
            transform_points(moving[test_mask], params, center) - fixed[test_mask], axis=1)
        rows.append({
            "held_out": held_out,
            "fit_groups": [name for name in groups if name != held_out],
            "fit_translation_mm": offset.tolist(),
            "held_out_anchor_names": [
                item["name"] for item, selected in zip(anchors, test_mask) if selected
            ],
            "before_median_mm": float(np.median(before)),
            "after_median_mm": float(np.median(after)),
            "before_mean_mm": float(np.mean(before)),
            "after_mean_mm": float(np.mean(after)),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu-map", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--bed-nz", type=int, default=80)
    parser.add_argument("--water-mu", type=float, default=.0152)
    parser.add_argument("--max-angle-deg", type=float, default=1.0)
    parser.add_argument("--surface-transform", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mu_image = sitk.ReadImage(str(args.mu_map), sitk.sitkFloat32)
    activity_image = sitk.ReadImage(str(args.activity), sitk.sitkFloat32)
    mu = sitk.GetArrayFromImage(mu_image)
    activity = sitk.GetArrayFromImage(activity_image)
    if mu.shape != activity.shape:
        raise RuntimeError(f"mu-map {mu.shape} != activity {activity.shape}")
    spacing = float(activity_image.GetSpacing()[0])
    activity *= fov_weight(activity.shape[1:])[None]
    activity_fov_image = sitk.GetImageFromArray(activity.astype(np.float32))
    activity_fov_image.CopyInformation(activity_image)

    hu = 1000 * (mu / args.water_mu - 1)
    fixed = gaussian_filter(np.clip((hu - 80) / 920, 0, 1), 1)
    scale = max(float(np.percentile(activity[activity > 0], 99.5)), 1e-12)
    moving = gaussian_filter(
        np.log1p(20 * np.clip(activity / scale, 0, 2)) / np.log(41), 1)

    z, y, x = np.indices(activity.shape)
    valid = np.hypot(x - (activity.shape[2] - 1) / 2,
                     y - (activity.shape[1] - 1) / 2) <= 50
    valid[:8] = False
    valid[-8:] = False
    step = args.bed_nz - args.overlap
    bed_count = int(round((activity.shape[0] - args.overlap) / step))
    seams = [i * step + args.overlap // 2 for i in range(1, bed_count)]
    for seam in seams:
        valid[max(0, seam - 8):min(len(valid), seam + 8)] = False

    ct_bone = valid & (fixed > .10)
    ct_inside = distance_transform_edt(ct_bone, sampling=(spacing,) * 3)
    ct_centers = center_ridges(ct_bone, ct_inside, 5, spacing)
    ct_surface_distance = distance_transform_edt(~ct_bone, sampling=(spacing,) * 3)
    spect_candidate = valid & (moving > .25) & (ct_surface_distance <= 3.0)
    spect_centers = center_ridges(spect_candidate, moving, 7, .25)

    ct_center_distance = distance_transform_edt(~ct_centers, sampling=(spacing,) * 3)
    spect_center_distance = distance_transform_edt(
        ~spect_centers, sampling=(spacing,) * 3)
    spect_support = spect_centers & (ct_center_distance <= 3.0)
    ct_support = ct_centers & (spect_center_distance <= 3.0)
    moving_points = thin_points(spect_support, 40000) * spacing
    fixed_points = thin_points(ct_support, 40000) * spacing
    if len(moving_points) < 300 or len(fixed_points) < 300:
        raise RuntimeError("Insufficient paired bone-center support")

    center_index = (np.asarray(activity.shape[::-1]) - 1) / 2
    center = center_index * spacing
    center_physical = np.asarray(
        activity_image.TransformContinuousIndexToPhysicalPoint(center_index.tolist()))

    def objective(params: np.ndarray) -> float:
        forward = sample_distance(
            ct_center_distance,
            transform_points(moving_points, params, center), spacing)
        reverse = sample_distance(
            spect_center_distance,
            inverse_points(fixed_points, params, center), spacing)
        return .65 * robust_distance(forward) + .35 * robust_distance(reverse)

    translation_result = optimize_translation(objective, np.zeros(3))
    translation_params = np.r_[np.zeros(3), translation_result.x]
    angle = math.radians(args.max_angle_deg)
    rigid_result = minimize(
        objective, translation_params, method="Powell",
        bounds=[(-angle, angle)] * 3 + [(-2, 2)] * 3,
        options={"maxiter": 180, "xtol": 1e-4, "ftol": 1e-5})
    rigid_params = rigid_result.x

    extent = (np.asarray(activity.shape[::-1]) - 1) * spacing
    anchors = anatomical_anchors(
        fixed_points, moving_points, extent, ct_inside, moving, spacing)
    if len(anchors) < 4:
        raise RuntimeError(f"Only {len(anchors)} anatomical anchors found")
    anchor_moving = np.asarray([item["moving_xyz_mm"] for item in anchors])
    anchor_fixed = np.asarray([item["fixed_xyz_mm"] for item in anchors])
    anchor_translation_params = np.r_[
        np.zeros(3),
        np.median(anchor_fixed - anchor_moving, axis=0),
    ]
    if np.any(np.abs(anchor_translation_params[3:]) > 2):
        offsets = {item["name"]: item["translation_mm"].tolist()
                   for item in anchors}
        raise RuntimeError(
            "Anatomical ROI centers exceed the +/-2 mm residual-motion contract: "
            f"{offsets}")

    def anchor_residual_vector(params: np.ndarray) -> np.ndarray:
        return (transform_points(anchor_moving, params, center) -
                anchor_fixed).ravel()

    anchor_rigid_result = least_squares(
        anchor_residual_vector, anchor_translation_params,
        bounds=(np.r_[[-angle] * 3, [-2] * 3],
                np.r_[[angle] * 3, [2] * 3]),
        loss="soft_l1", f_scale=.3, max_nfev=300)
    anchor_rigid_params = anchor_rigid_result.x
    anchor_holdout = anchor_group_holdout(anchors, center)

    translation_affine = affine_from_params(translation_params, center_physical)
    rigid_affine = affine_from_params(rigid_params, center_physical)
    anchor_translation_affine = affine_from_params(
        anchor_translation_params, center_physical)
    anchor_rigid_affine = affine_from_params(anchor_rigid_params, center_physical)
    translated = resample_activity(activity_fov_image, mu_image, translation_affine)
    rigid = resample_activity(activity_fov_image, mu_image, rigid_affine)
    anchor_translated = resample_activity(
        activity_fov_image, mu_image, anchor_translation_affine)
    anchor_rigid = resample_activity(
        activity_fov_image, mu_image, anchor_rigid_affine)
    for volume in (translated, rigid, anchor_translated, anchor_rigid):
        local_scale = max(float(np.percentile(volume[volume > 0], 99.5)), 1e-12)
        volume[:] = gaussian_filter(
            np.log1p(20 * np.clip(volume / local_scale, 0, 2)) / np.log(41), 1)

    translation_mu = sitk.Resample(
        mu_image, activity_image, translation_affine, sitk.sitkLinear, 0.0)
    rigid_mu = sitk.Resample(
        mu_image, activity_image, rigid_affine, sitk.sitkLinear, 0.0)
    anchor_translation_mu = sitk.Resample(
        mu_image, activity_image, anchor_translation_affine,
        sitk.sitkLinear, 0.0)
    anchor_rigid_mu = sitk.Resample(
        mu_image, activity_image, anchor_rigid_affine,
        sitk.sitkLinear, 0.0)
    translation_mu_path = args.output_dir / "mouse_ct_mu_ov10_center_translation.mhd"
    rigid_mu_path = args.output_dir / "mouse_ct_mu_ov10_center_rigid.mhd"
    sitk.WriteImage(translation_mu, str(translation_mu_path), False)
    sitk.WriteImage(rigid_mu, str(rigid_mu_path), False)
    anchor_translation_mu_path = (
        args.output_dir / "mouse_ct_mu_ov10_anchor_translation.mhd")
    anchor_rigid_mu_path = args.output_dir / "mouse_ct_mu_ov10_anchor_rigid.mhd"
    sitk.WriteImage(anchor_translation_mu, str(anchor_translation_mu_path), False)
    sitk.WriteImage(anchor_rigid_mu, str(anchor_rigid_mu_path), False)
    sitk.WriteTransform(
        translation_affine,
        str(args.output_dir / "mouse_spect_to_ct_center_translation.tfm"))
    sitk.WriteTransform(
        rigid_affine,
        str(args.output_dir / "mouse_spect_to_ct_center_rigid.tfm"))
    sitk.WriteTransform(
        anchor_translation_affine,
        str(args.output_dir / "mouse_spect_to_ct_anchor_translation.tfm"))
    sitk.WriteTransform(
        anchor_rigid_affine,
        str(args.output_dir / "mouse_spect_to_ct_anchor_rigid.tfm"))

    image_qa = args.output_dir / "mouse_bone_center_image_qa.png"
    center_qa = args.output_dir / "mouse_bone_center_points_qa.png"
    axial_qa = args.output_dir / "mouse_bone_center_axial_qa.png"
    write_image_qa(image_qa, fixed, moving, translated, rigid)
    write_center_qa(
        center_qa, fixed_points, moving_points, translation_params,
        rigid_params, center)
    levels = qa_levels(len(fixed))
    write_axial_qa(axial_qa, fixed, moving, translated, rigid, levels)
    anchor_axial_qa = args.output_dir / "mouse_anatomical_anchor_axial_qa.png"
    write_axial_qa(
        anchor_axial_qa, fixed, moving, anchor_translated, anchor_rigid,
        levels, ("anchor translation", "anchor small-rigid"))
    comparison_qa = None
    if args.surface_transform:
        surface_affine = sitk.ReadTransform(str(args.surface_transform))
        surface = resample_activity(activity_fov_image, mu_image, surface_affine)
        surface_scale = max(
            float(np.percentile(surface[surface > 0], 99.5)), 1e-12)
        surface = gaussian_filter(
            np.log1p(20 * np.clip(surface / surface_scale, 0, 2)) / np.log(41), 1)
        comparison_qa = args.output_dir / "mouse_bone_surface_vs_center_qa.png"
        write_surface_comparison(
            comparison_qa, fixed, moving, surface, translated, rigid)
    summaries = {
        "before": distance_summary(
            ct_center_distance, moving_points, np.zeros(6), center, spacing),
        "translation": distance_summary(
            ct_center_distance, moving_points, translation_params, center, spacing),
        "small_rigid": distance_summary(
            ct_center_distance, moving_points, rigid_params, center, spacing),
    }
    at_bound = bool(np.any(
        np.abs(rigid_params[:3]) >= angle * .98)) if angle else False
    anchor_at_bound = bool(np.any(
        np.abs(anchor_rigid_params[:3]) >= angle * .98)) if angle else False
    manifest = {
        "decision": "retain_center_translation_as_scoped_candidate_reject_center_rigid",
        "evidence_regime": "paired_relative_no_manual_landmark_ground_truth",
        "gate_maturity": "diagnostic",
        "protected_outcome": "spine and paired-joint center alignment without unstable rotation",
        "method": "axial bone-center ridges; symmetric robust center distance",
        "spect_fov_before_registration": {"radius_vox": 50, "taper_vox": 4},
        "seams_excluded": seams,
        "end_slices_excluded": {"leading": 8, "trailing": 8},
        "one_transform_for_all_beds": True,
        "support_points": {"spect": len(moving_points), "ct": len(fixed_points)},
        "distance_mm": summaries,
        "distance_by_z_quarter": {
            "before": band_summaries(
                ct_center_distance, moving_points, np.zeros(6), center, spacing,
                activity.shape[0]),
            "translation": band_summaries(
                ct_center_distance, moving_points, translation_params, center,
                spacing, activity.shape[0]),
            "small_rigid": band_summaries(
                ct_center_distance, moving_points, rigid_params, center, spacing,
                activity.shape[0]),
        },
        "objective": {
            "before": objective(np.zeros(6)),
            "translation": float(translation_result.fun),
            "small_rigid": float(rigid_result.fun),
        },
        "translation_spect_to_ct": {
            "translation_mm": translation_params[3:].tolist()},
        "small_rigid_spect_to_ct": {
            "angles_deg": np.degrees(rigid_params[:3]).tolist(),
            "translation_mm": rigid_params[3:].tolist(),
            "rotation_at_bound": at_bound,
            "max_angle_deg": args.max_angle_deg,
        },
        "anatomical_anchors": [{
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in item.items()
        } for item in anchors],
        "anchor_translation_spect_to_ct": {
            "translation_mm": anchor_translation_params[3:].tolist(),
            "residual_mm": anchor_residuals(
                anchors, anchor_translation_params, center),
            "delta_from_center_translation_mm": float(np.linalg.norm(
                anchor_translation_params[3:] - translation_params[3:])),
        },
        "anchor_residual_comparison_mm": {
            "before": anchor_residuals(anchors, np.zeros(6), center),
            "center_translation": anchor_residuals(
                anchors, translation_params, center),
            "anatomical_anchor_translation": anchor_residuals(
                anchors, anchor_translation_params, center),
        },
        "anchor_group_holdout": anchor_holdout,
        "anchor_small_rigid_spect_to_ct": {
            "angles_deg": np.degrees(anchor_rigid_params[:3]).tolist(),
            "translation_mm": anchor_rigid_params[3:].tolist(),
            "residual_mm": anchor_residuals(
                anchors, anchor_rigid_params, center),
            "rotation_at_bound": anchor_at_bound,
            "max_angle_deg": args.max_angle_deg,
        },
        "center_physical_mm": center_physical.tolist(),
        "translation_mu_map": str(translation_mu_path),
        "rigid_mu_map": str(rigid_mu_path),
        "anchor_translation_mu_map": str(anchor_translation_mu_path),
        "anchor_rigid_mu_map": str(anchor_rigid_mu_path),
        "image_qa": str(image_qa),
        "center_points_qa": str(center_qa),
        "axial_qa": str(axial_qa),
        "anatomical_anchor_axial_qa": str(anchor_axial_qa),
        "surface_comparison_qa": str(comparison_qa) if comparison_qa else None,
    }
    (args.output_dir / "bone_center_registration_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
