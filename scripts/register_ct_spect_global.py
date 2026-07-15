#!/usr/bin/env python3
"""Estimate one global rigid CT/SPECT residual transform for all beds."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter


def normalized_mi(a: np.ndarray, b: np.ndarray, mask: np.ndarray, bins: int = 48) -> float:
    av, bv = a[mask], b[mask]
    hist, _, _ = np.histogram2d(av, bv, bins=bins)
    p = hist / max(hist.sum(), 1)
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    hxy = -np.sum(p[nz] * np.log(p[nz]))
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    return float((hx + hy) / max(hxy, 1e-12))


def image_like(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.CopyInformation(reference)
    return image


def registration_method(fixed_mask: sitk.Image, moving_mask: sitk.Image,
                        initial: sitk.Transform, iterations: int) -> sitk.ImageRegistrationMethod:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    method.SetMetricSamplingStrategy(method.REGULAR)
    method.SetMetricSamplingPercentage(.25)
    method.SetMetricFixedMask(fixed_mask)
    method.SetMetricMovingMask(moving_mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=.6, minStep=.005, numberOfIterations=iterations,
        gradientMagnitudeTolerance=1e-7)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([1.2, .6, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    return method


def overlay(ct: np.ndarray, spect: np.ndarray) -> np.ndarray:
    result = np.zeros((*ct.shape, 3), dtype=float)
    result[..., 1] = np.clip(ct, 0, 1)
    result[..., 0] = np.clip(spect, 0, 1)
    return result


def write_qa(path: Path, fixed: np.ndarray, before: np.ndarray,
             translated: np.ndarray, after: np.ndarray) -> None:
    views = [
        ("coronal", fixed.max(1), before.max(1), after.max(1)),
        ("sagittal", fixed.max(2), before.max(2), after.max(2)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 10))
    for row, (name, ct, old, new) in enumerate(views):
        axes[row, 0].imshow(ct, origin="lower", cmap="gray", vmin=0, vmax=1,
                            aspect="auto")
        axes[row, 0].set_title(f"CT bone {name}")
        axes[row, 1].imshow(overlay(ct, old), origin="lower", aspect="auto")
        axes[row, 1].set_title("DICOM geometry")
        mid = translated.max(1 if row == 0 else 2)
        axes[row, 2].imshow(overlay(ct, mid), origin="lower", aspect="auto")
        axes[row, 2].set_title("translation residual")
        axes[row, 3].imshow(overlay(ct, new), origin="lower", aspect="auto")
        axes[row, 3].set_title("global rigid residual")
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
    parser.add_argument("--fixed-bone-threshold", type=float, default=.10)
    parser.add_argument("--moving-strong-threshold", type=float, default=.25)
    parser.add_argument("--seam-half-width", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mu_image = sitk.ReadImage(str(args.mu_map), sitk.sitkFloat32)
    activity_image = sitk.ReadImage(str(args.activity), sitk.sitkFloat32)
    if mu_image.GetSize() != activity_image.GetSize():
        raise RuntimeError(f"mu-map {mu_image.GetSize()} != activity {activity_image.GetSize()}")
    mu = sitk.GetArrayFromImage(mu_image)
    activity = sitk.GetArrayFromImage(activity_image)
    yy, xx = np.ogrid[:activity.shape[1], :activity.shape[2]]
    radius = np.hypot(xx - (activity.shape[2] - 1) / 2,
                      yy - (activity.shape[1] - 1) / 2)
    fov_weight = np.clip((54.0 - radius) / 4.0, 0, 1)
    activity = activity * fov_weight[None]
    hu = 1000.0 * (mu / args.water_mu - 1.0)
    fixed = gaussian_filter(np.clip((hu - 80.0) / 920.0, 0, 1), 1.0)
    scale = max(float(np.percentile(activity[activity > 0], 99.5)), 1e-12)
    moving = gaussian_filter(np.log1p(20 * np.clip(activity / scale, 0, 2)) / np.log(41), 1.0)

    z, y, x = np.indices(activity.shape)
    radial = np.hypot(x - (activity.shape[2] - 1) / 2,
                      y - (activity.shape[1] - 1) / 2)
    valid = radial <= 50
    step = args.bed_nz - args.overlap
    bed_count = int(round((activity.shape[0] - args.overlap) / step))
    seams = [i * step + args.overlap // 2 for i in range(1, bed_count)]
    for seam in seams:
        valid[max(0, seam - args.seam_half_width):
              min(len(valid), seam + args.seam_half_width)] = False
    fixed_mask_array = valid & (fixed > args.fixed_bone_threshold)
    moving_mask_array = valid & (moving > args.moving_strong_threshold)
    if fixed_mask_array.sum() < 1000 or moving_mask_array.sum() < 1000:
        raise RuntimeError("Registration mask has insufficient support")

    fixed_image = image_like(fixed, mu_image)
    moving_image = image_like(moving, activity_image)
    fixed_mask = image_like(fixed_mask_array, mu_image) > 0
    moving_mask = image_like(moving_mask_array, activity_image) > 0

    translation = sitk.TranslationTransform(3)
    stage1 = registration_method(fixed_mask, moving_mask, translation, 80)
    translated = stage1.Execute(fixed_image, moving_image)
    if translated.GetName() == "CompositeTransform":
        translated = translated.GetBackTransform()
    translated = sitk.TranslationTransform(translated)

    center = fixed_image.TransformContinuousIndexToPhysicalPoint(
        [(size - 1) / 2 for size in fixed_image.GetSize()])
    rigid = sitk.Euler3DTransform()
    rigid.SetCenter(center)
    rigid.SetTranslation(translated.GetOffset())
    stage2 = registration_method(fixed_mask, moving_mask, rigid, 140)
    final = stage2.Execute(fixed_image, moving_image)
    if final.GetName() == "CompositeTransform":
        final = final.GetBackTransform()
    final = sitk.Euler3DTransform(final)

    translated_moved = sitk.Resample(moving_image, fixed_image, translated,
                                      sitk.sitkLinear, 0.0)
    moved = sitk.Resample(moving_image, fixed_image, final, sitk.sitkLinear, 0.0)
    translated_array = sitk.GetArrayFromImage(translated_moved)
    moved_array = sitk.GetArrayFromImage(moved)
    registered_mu = sitk.Resample(mu_image, activity_image, final.GetInverse(),
                                  sitk.sitkLinear, 0.0)
    translated_mu = sitk.Resample(mu_image, activity_image, translated.GetInverse(),
                                  sitk.sitkLinear, 0.0)
    registered_mu_path = args.output_dir / "mouse_ct_mu_ov10_registered.mhd"
    translated_mu_path = args.output_dir / "mouse_ct_mu_ov10_translation.mhd"
    sitk.WriteImage(registered_mu, str(registered_mu_path), True)
    sitk.WriteImage(translated_mu, str(translated_mu_path), True)
    transform_path = args.output_dir / "mouse_spect_to_ct_global.tfm"
    translation_path = args.output_dir / "mouse_spect_to_ct_translation.tfm"
    sitk.WriteTransform(final.GetInverse(), str(transform_path))
    sitk.WriteTransform(translated.GetInverse(), str(translation_path))

    before_nmi = normalized_mi(fixed, moving, fixed_mask_array)
    translated_nmi = normalized_mi(fixed, translated_array, fixed_mask_array)
    after_nmi = normalized_mi(fixed, moved_array, fixed_mask_array)
    angles_deg = [math.degrees(value) for value in
                  (final.GetAngleX(), final.GetAngleY(), final.GetAngleZ())]
    translation_mm = list(final.GetTranslation())
    qa_path = args.output_dir / "mouse_global_rigid_registration_qa.png"
    write_qa(qa_path, fixed, moving, translated_array, moved_array)
    manifest = {
        "decision": "pending_visual_gate",
        "evidence_regime": "paired_relative_no_landmark_ground_truth",
        "fixed": "CT bone-weighted image", "moving": "SPECT log activity",
        "spect_fov_before_registration": {"radius_vox": 50, "taper_vox": 4},
        "strong_calibration_mask": {
            "ct_bone_threshold": args.fixed_bone_threshold,
            "spect_log_activity_threshold": args.moving_strong_threshold,
            "seam_half_width_vox": args.seam_half_width},
        "one_transform_for_all_beds": True, "seams_excluded": seams,
        "metric": {"normalized_mi_before": before_nmi,
                   "normalized_mi_translation": translated_nmi,
                   "normalized_mi_after": after_nmi,
                   "mattes_final": float(stage2.GetMetricValue()),
                   "optimizer_stop": stage2.GetOptimizerStopConditionDescription()},
        "transform_fixed_ct_to_moving_spect": {
            "angles_deg": angles_deg, "translation_mm": translation_mm,
            "center_mm": list(final.GetCenter())},
        "translation_only_fixed_ct_to_moving_spect_mm": list(translated.GetOffset()),
        "registered_mu_map": str(registered_mu_path),
        "translation_mu_map": str(translated_mu_path),
        "spect_to_ct_transform": str(transform_path), "qa": str(qa_path),
        "spect_to_ct_translation": str(translation_path),
    }
    (args.output_dir / "registration_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def self_check() -> None:
    a = np.eye(8)
    mask = np.ones_like(a, dtype=bool)
    assert normalized_mi(a, a, mask) >= normalized_mi(a, np.roll(a, 1, axis=0), mask)


if __name__ == "__main__":
    self_check()
    raise SystemExit(main())
