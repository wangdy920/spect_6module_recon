#!/usr/bin/env python3
"""Refine one global SPECT-to-CT rigid transform from strong bone uptake."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def fov_weight(shape: tuple[int, int], radius: float = 50, taper: float = 4) -> np.ndarray:
    y, x = np.ogrid[:shape[0], :shape[1]]
    r = np.hypot(x - (shape[1] - 1) / 2, y - (shape[0] - 1) / 2)
    return np.clip((radius + taper - r) / taper, 0, 1)


def transform_points(points: np.ndarray, params: np.ndarray,
                     center: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(params[:3])
    return rotation.apply(points - center) + center + params[3:]


def inverse_points(points: np.ndarray, params: np.ndarray,
                   center: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(params[:3])
    return rotation.inv().apply(points - center - params[3:]) + center


def sample_distance(distance: np.ndarray, points_xyz: np.ndarray,
                    spacing: float) -> np.ndarray:
    indices = (points_xyz / spacing)[:, [2, 1, 0]].T
    return map_coordinates(distance, indices, order=1, mode="constant", cval=5.0)


def thin_points(mask: np.ndarray, max_points: int = 60000) -> np.ndarray:
    points = np.argwhere(mask)
    stride = max(1, math.ceil(len(points) / max_points))
    return points[::stride][:, [2, 1, 0]].astype(float)


def robust_distance(values: np.ndarray) -> float:
    return float(np.mean(np.minimum(values, 3.0) ** 2))


def affine_from_params(params: np.ndarray, center: np.ndarray) -> sitk.AffineTransform:
    affine = sitk.AffineTransform(3)
    affine.SetCenter(center.tolist())
    affine.SetMatrix(Rotation.from_rotvec(params[:3]).as_matrix().ravel().tolist())
    affine.SetTranslation(params[3:].tolist())
    return affine


def overlay(ct: np.ndarray, spect: np.ndarray) -> np.ndarray:
    image = np.zeros((*ct.shape, 3), dtype=float)
    image[..., 1] = np.clip(ct, 0, 1)
    image[..., 0] = np.clip(spect, 0, 1)
    return image


def write_slice_qa(path: Path, fixed: np.ndarray, before: np.ndarray,
                   after: np.ndarray, levels: list[int]) -> None:
    fig, axes = plt.subplots(len(levels), 3, figsize=(9, 3 * len(levels)))
    for row, z in enumerate(levels):
        axes[row, 0].imshow(fixed[z], origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"CT bone z={z}")
        axes[row, 1].imshow(overlay(fixed[z], before[z]), origin="lower")
        axes[row, 1].set_title("DICOM geometry")
        axes[row, 2].imshow(overlay(fixed[z], after[z]), origin="lower")
        axes[row, 2].set_title("bone-distance rigid")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu-map", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--bed-nz", type=int, default=80)
    parser.add_argument("--water-mu", type=float, default=.0152)
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
    moving = gaussian_filter(np.log1p(20 * np.clip(activity / scale, 0, 2)) / np.log(41), 1)

    z, y, x = np.indices(activity.shape)
    radial = np.hypot(x - (activity.shape[2] - 1) / 2,
                      y - (activity.shape[1] - 1) / 2)
    valid = radial <= 50
    step = args.bed_nz - args.overlap
    bed_count = int(round((activity.shape[0] - args.overlap) / step))
    seams = [i * step + args.overlap // 2 for i in range(1, bed_count)]
    for seam in seams:
        valid[max(0, seam - 8):min(len(valid), seam + 8)] = False
    ct_bone = valid & (fixed > .10)
    spect_bone = valid & (moving > .25)
    ct_distance = distance_transform_edt(~ct_bone, sampling=(spacing,) * 3)
    spect_distance = distance_transform_edt(~spect_bone, sampling=(spacing,) * 3)
    spect_support = spect_bone & (ct_distance <= 3.0)
    ct_support = ct_bone & (spect_distance <= 3.0)
    moving_points = thin_points(spect_support) * spacing
    fixed_points = thin_points(ct_support) * spacing
    if len(moving_points) < 1000 or len(fixed_points) < 1000:
        raise RuntimeError("Insufficient paired bone support")
    center_index = (np.asarray(activity.shape[::-1]) - 1) / 2
    center = center_index * spacing
    center_physical = np.asarray(
        activity_image.TransformContinuousIndexToPhysicalPoint(center_index.tolist()))

    def objective(params: np.ndarray) -> float:
        moving_to_fixed = sample_distance(
            ct_distance, transform_points(moving_points, params, center), spacing)
        fixed_to_moving = sample_distance(
            spect_distance, inverse_points(fixed_points, params, center), spacing)
        return .65 * robust_distance(moving_to_fixed) + .35 * robust_distance(fixed_to_moving)

    angle = math.radians(3)
    bounds = [(-angle, angle)] * 3 + [(-3, 3)] * 3
    starts = [np.zeros(6), np.array([0, 0, 0, .4, -.7, 1.1])]
    results = [minimize(objective, start, method="Powell", bounds=bounds,
                        options={"maxiter": 180, "xtol": 1e-4, "ftol": 1e-5})
               for start in starts]
    result = min(results, key=lambda item: item.fun)
    params = result.x
    affine = affine_from_params(params, center_physical)
    moved_image = sitk.Resample(activity_fov_image, mu_image, affine.GetInverse(),
                                sitk.sitkLinear, 0.0)
    moved = sitk.GetArrayFromImage(moved_image)
    moved *= fov_weight(moved.shape[1:])[None]
    moved_scale = max(float(np.percentile(moved[moved > 0], 99.5)), 1e-12)
    moved_display = gaussian_filter(
        np.log1p(20 * np.clip(moved / moved_scale, 0, 2)) / np.log(41), 1)
    registered_mu = sitk.Resample(mu_image, activity_image, affine,
                                  sitk.sitkLinear, 0.0)
    mu_path = args.output_dir / "mouse_ct_mu_ov10_bone_registered.mhd"
    transform_path = args.output_dir / "mouse_spect_to_ct_bone_global.tfm"
    sitk.WriteImage(registered_mu, str(mu_path), True)
    sitk.WriteTransform(affine, str(transform_path))

    before_distance = sample_distance(ct_distance, moving_points, spacing)
    after_distance = sample_distance(
        ct_distance, transform_points(moving_points, params, center), spacing)
    qa_path = args.output_dir / "mouse_bone_registration_axial_qa.png"
    write_slice_qa(qa_path, fixed, moving, moved_display, [55, 115, 175, 225])
    manifest = {
        "decision": "pending_visual_gate",
        "evidence_regime": "paired_relative_no_manual_landmark_ground_truth",
        "method": "symmetric robust distance between CT bone and strong SPECT uptake",
        "spect_fov_before_registration": {"radius_vox": 50, "taper_vox": 4},
        "seams_excluded": seams, "one_transform_for_all_beds": True,
        "support_points": {"spect": len(moving_points), "ct": len(fixed_points)},
        "distance_mm": {
            "before_median": float(np.median(before_distance)),
            "before_p90": float(np.percentile(before_distance, 90)),
            "after_median": float(np.median(after_distance)),
            "after_p90": float(np.percentile(after_distance, 90))},
        "objective": {"before": objective(np.zeros(6)), "after": float(result.fun),
                      "success": bool(result.success), "message": str(result.message)},
        "spect_to_ct": {"angles_deg": np.degrees(params[:3]).tolist(),
                        "translation_mm": params[3:].tolist(),
                        "center_local_mm": center.tolist(),
                        "center_physical_mm": center_physical.tolist()},
        "registered_mu_map": str(mu_path), "transform": str(transform_path),
        "axial_qa": str(qa_path),
    }
    (args.output_dir / "bone_registration_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def self_check() -> None:
    points = np.array([[1., 2., 3.]])
    params = np.array([0., 0., 0., .5, -.5, 1.])
    center = np.zeros(3)
    assert np.allclose(inverse_points(transform_points(points, params, center),
                                      params, center), points)


if __name__ == "__main__":
    self_check()
    raise SystemExit(main())
