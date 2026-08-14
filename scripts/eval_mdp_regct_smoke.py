#!/usr/bin/env python3
"""Paired QA for plain-RDP versus CT-weighted RDP on the five-bed MDP case."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_mhd_stats import read_mhd  # noqa: E402

DIFFERENCE_DISPLAY_FRACTION = 0.01
ACTIVITY_DISPLAY_PERCENTILE = 99.9


def relative_l2(candidate: np.ndarray, control: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm((candidate - control)[mask]) /
                 max(np.linalg.norm(control[mask]), 1e-12))


def ncc(candidate: np.ndarray, control: np.ndarray, mask: np.ndarray) -> float:
    a = candidate[mask] - candidate[mask].mean()
    b = control[mask] - control[mask].mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    result = np.zeros_like(volume, dtype=np.float64)
    for axis in range(3):
        delta = np.diff(volume, axis=axis)
        target = [slice(None)] * 3
        target[axis] = slice(0, -1)
        result[tuple(target)] += delta * delta
    return np.sqrt(result)


def smooth_profile(profile: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(profile, kernel, mode="same")


def profile_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.log1p(np.maximum(a, 0.0))
    b = np.log1p(np.maximum(b, 0.0))
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) /
                 max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def best_profile_window(short: np.ndarray, long: np.ndarray) -> dict:
    if len(short) > len(long):
        raise ValueError("profile reference must not be shorter than output")
    scores = [(profile_ncc(short, long[start:start + len(short)]), start)
              for start in range(len(long) - len(short) + 1)]
    score, start = max(scores)
    return {"ncc": score, "reference_start_z": start}


def center_crop(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if any(source < target for source, target in zip(volume.shape, shape)):
        raise ValueError(f"cannot crop {volume.shape} to {shape}")
    starts = tuple((source - target) // 2
                   for source, target in zip(volume.shape, shape))
    return volume[tuple(slice(start, start + size)
                        for start, size in zip(starts, shape))]


def volume_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(np.dot(a, b) /
                 max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def orientation_diagnostic(activity: np.ndarray, stitched_ct: np.ndarray,
                           spect_reference: np.ndarray,
                           physical_ct: np.ndarray) -> dict:
    cropped_spect = center_crop(spect_reference, activity.shape)
    cropped_ct = center_crop(physical_ct, stitched_ct.shape)
    output_profile = smooth_profile(activity.sum(axis=(1, 2)))
    spect_profile = smooth_profile(spect_reference.sum(axis=(1, 2)))
    regct_profile = smooth_profile(
        np.clip(stitched_ct - 300.0, 0.0, None).sum(axis=(1, 2)))
    physical_ct_profile = smooth_profile(
        np.clip(physical_ct - 300.0, 0.0, None).sum(axis=(1, 2)))
    return {
        "interpretation": (
            "diagnostic only; compares same-modality axial profiles before "
            "cross-modality bone-tracer interpretation"
        ),
        "output_vs_dicom_spect": {
            "center_crop_3d_ncc_same_z": volume_ncc(activity, cropped_spect),
            "center_crop_3d_ncc_reversed_z": volume_ncc(
                activity, cropped_spect[::-1]),
            "same_z_direction": best_profile_window(output_profile, spect_profile),
            "reversed_z_direction": best_profile_window(
                output_profile, spect_profile[::-1]),
        },
        "stitched_regct_vs_physical_ct": {
            "center_crop_max_abs_hu_same_z": float(
                np.max(np.abs(stitched_ct - cropped_ct))),
            "center_crop_max_abs_hu_reversed_z": float(
                np.max(np.abs(stitched_ct - cropped_ct[::-1]))),
            "same_z_direction": best_profile_window(
                regct_profile, physical_ct_profile),
            "reversed_z_direction": best_profile_window(
                regct_profile, physical_ct_profile[::-1]),
        },
        "bone_tracer_profile_ncc": {
            "regct_as_used": profile_ncc(output_profile, regct_profile),
            "diagnostic_regct_z_flip": profile_ncc(
                output_profile, regct_profile[::-1]),
        },
        "display_origin": "lower; changes screen direction only, not relative array order",
        "verdict": "registered CT is globally Z-reversed relative to reconstruction output",
    }


def ct_coupling_summary(ct: np.ndarray, mask: np.ndarray,
                        sigma: float = 100.0, floor: float = 0.05) -> dict:
    weights = []
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        pair_mask = mask[tuple(left)] & mask[tuple(right)]
        delta = (ct[tuple(left)] - ct[tuple(right)])[pair_mask]
        weights.append(floor + (1.0 - floor) * np.exp(-0.5 * (delta / sigma) ** 2))
    joined = np.concatenate(weights)
    quantiles = np.percentile(joined, [1, 5, 25, 50, 75, 95, 99])
    return {
        "formula": "floor + (1-floor) * exp(-0.5 * (delta_HU / sigma)^2)",
        "sigma_hu": sigma,
        "floor": floor,
        "pair_population": "3-axis neighbors inside paired nonzero activity support",
        "pair_count": int(joined.size),
        "weight_percentiles_p1_p5_p25_p50_p75_p95_p99": quantiles.tolist(),
        "fraction_le_0p10": float(np.mean(joined <= 0.10)),
        "fraction_le_0p50": float(np.mean(joined <= 0.50)),
        "fraction_ge_0p90": float(np.mean(joined >= 0.90)),
    }


def stitch_regct(regct: np.ndarray) -> np.ndarray:
    """Match center66_cosine10; NumPy is ZYX while MHD DimSize is XYZ."""
    beds, bed_nz, crop_layers, step_layers, blend = 5, 220, 66, 56, 10
    if regct.shape != (beds * bed_nz, 120, 120):
        raise ValueError(f"expected regCT shape (1100, 120, 120), got {regct.shape}")
    crop_start = (bed_nz - crop_layers) // 2
    out = np.zeros((crop_layers + (beds - 1) * step_layers, 120, 120), dtype=np.float64)
    weights = np.zeros(out.shape[0], dtype=np.float64)
    for bed in range(beds):
        for local_z in range(crop_layers):
            weight = 1.0
            if bed > 0 and local_z < blend:
                weight *= np.sin(0.5 * np.pi * local_z / (blend - 1)) ** 2
            if bed + 1 < beds and local_z >= step_layers:
                weight *= 1.0 - np.sin(
                    0.5 * np.pi * (local_z - step_layers) / (blend - 1)) ** 2
            src_z = bed * bed_nz + crop_start + local_z
            dst_z = bed * step_layers + local_z
            out[dst_z] += weight * regct[src_z]
            weights[dst_z] += weight
    out /= weights[:, None, None]
    return out[:, 10:110, 10:110]


def evaluate(control: np.ndarray, candidate: np.ndarray, ct: np.ndarray) -> dict:
    if control.shape != candidate.shape or control.shape != ct.shape:
        raise ValueError(
            f"paired geometry mismatch: control={control.shape}, "
            f"candidate={candidate.shape}, CT={ct.shape}")
    if not np.isfinite(control).all() or not np.isfinite(candidate).all():
        raise ValueError("reconstruction outputs must be finite")
    if control.min() < 0 or candidate.min() < 0:
        raise ValueError("reconstruction outputs must be nonnegative")
    trusted = (control > 0) | (candidate > 0)
    if not np.any(trusted):
        raise ValueError("reconstruction outputs are empty")

    control_grad = gradient_magnitude(control)
    candidate_grad = gradient_magnitude(candidate)
    ct_grad = gradient_magnitude(ct)
    positive_ct_grad = ct_grad[trusted & (ct_grad > 0)]
    low, high = np.percentile(positive_ct_grad, [25, 75])
    ct_flat = trusted & (ct_grad <= low)
    ct_edge = trusted & (ct_grad >= high)
    edge_ratio = float(candidate_grad[ct_edge].mean() /
                       max(control_grad[ct_edge].mean(), 1e-12))
    flat_ratio = float(candidate_grad[ct_flat].mean() /
                       max(control_grad[ct_flat].mean(), 1e-12))
    landmark_slices = {}
    for z, name in ((93, "knees"), (117, "spine"), (228, "shoulders")):
        slice_mask = trusted[z]
        landmark_slices[name] = {
            "z_index": z,
            "relative_l2": relative_l2(candidate[z], control[z], slice_mask),
            "ncc": ncc(candidate[z], control[z], slice_mask),
            "sum_ratio": float(candidate[z][slice_mask].sum() /
                               control[z][slice_mask].sum()),
        }
    return {
        "shape_zyx": list(control.shape),
        "mhd_dimsize_xyz": list(reversed(control.shape)),
        "finite_nonnegative_nonzero": True,
        "relative_l2": relative_l2(candidate, control, trusted),
        "ncc": ncc(candidate, control, trusted),
        "sum_ratio": float(candidate[trusted].sum() / control[trusted].sum()),
        "max_ratio": float(candidate.max() / control.max()),
        "ct_gradient_rule": {
            "population": "positive CT gradients inside paired nonzero activity support",
            "flat": "at or below P25",
            "edge": "at or above P75",
            "p25_hu_per_voxel": float(low),
            "p75_hu_per_voxel": float(high),
        },
        "candidate_control_gradient_ratio": {
            "ct_flat": flat_ratio,
            "ct_edge": edge_ratio,
            "edge_over_flat": edge_ratio / max(flat_ratio, 1e-12),
        },
        "landmark_slices": landmark_slices,
        "ct_neighbor_coupling": ct_coupling_summary(ct, trusted),
        "interpretation_limit": (
            "Diagnostic 20-iteration morphology check; thresholds are distribution-based "
            "and are not calibrated adoption gates."
        ),
    }


def trajectory(control_path: Path, candidate_path: Path, ct: np.ndarray) -> dict:
    result = {}
    for iteration in (5, 10, 15, 20):
        control_file = control_path.with_name(f"recon_it{iteration}.mhd")
        candidate_file = candidate_path.with_name(f"recon_it{iteration}.mhd")
        if not control_file.exists() or not candidate_file.exists():
            continue
        values = evaluate(
            read_mhd(control_file).astype(np.float64, copy=False),
            read_mhd(candidate_file).astype(np.float64, copy=False), ct)
        result[str(iteration)] = {
            "relative_l2": values["relative_l2"],
            "ncc": values["ncc"],
            "sum_ratio": values["sum_ratio"],
            "ct_flat_gradient_ratio": values["candidate_control_gradient_ratio"]["ct_flat"],
            "ct_edge_gradient_ratio": values["candidate_control_gradient_ratio"]["ct_edge"],
            "edge_over_flat": values["candidate_control_gradient_ratio"]["edge_over_flat"],
        }
    return result


def render(control: np.ndarray, candidate: np.ndarray, ct: np.ndarray,
           output_dir: Path) -> list[str]:
    positive = control[control > 0]
    vmax = float(np.percentile(positive, ACTIVITY_DISPLAY_PERCENTILE))
    diff = candidate - control
    dmax = vmax * DIFFERENCE_DISPLAY_FRACTION

    fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharex=True, sharey=True)
    panels = (
        (control.max(1), "plain RDP", "jet", 0, vmax),
        (candidate.max(1), "CT-weighted RDP", "jet", 0, vmax),
        (diff.mean(1), "candidate - control", "coolwarm", -dmax, dmax),
        (ct.mean(1), "regCT as used (Z-reversed)", "gray", -1000, 1500),
    )
    for axis, (image, title, cmap, vmin, panel_vmax) in zip(axes, panels):
        axis.imshow(image, origin="lower", cmap=cmap, vmin=vmin,
                    vmax=panel_vmax, aspect="auto")
        axis.set_title(title)
        for seam in (56, 112, 168, 224):
            axis.axhline(seam, color="white", lw=0.5, alpha=0.65)
    fig.suptitle("MDP it20 | shared control P99.9 window | RDP beta=0.3")
    fig.tight_layout()
    coronal = output_dir / "mdp_regct_it20_coronal.png"
    fig.savefig(coronal, dpi=180, facecolor="white")
    plt.close(fig)

    slices = ((93, "knees"), (117, "spine"), (228, "shoulders"))
    fig, axes = plt.subplots(len(slices), 4, figsize=(14, 10))
    for row, (z, label) in enumerate(slices):
        axial = (
            (control[z], f"{label} z={z} | plain RDP", "jet", 0, vmax),
            (candidate[z], "CT-weighted RDP", "jet", 0, vmax),
            (diff[z], "candidate - control", "coolwarm", -dmax, dmax),
            (ct[z], "regCT as used (Z-reversed)", "gray", -1000, 1500),
        )
        for axis, (image, title, cmap, vmin, panel_vmax) in zip(axes[row], axial):
            axis.imshow(image, origin="lower", cmap=cmap, vmin=vmin,
                        vmax=panel_vmax)
            axis.set_title(title)
            axis.set_axis_off()
    fig.tight_layout()
    landmarks = output_dir / "mdp_regct_it20_axial_landmarks.png"
    fig.savefig(landmarks, dpi=180, facecolor="white")
    plt.close(fig)
    return [str(coronal), str(landmarks)]


def render_ct_overlay(control: np.ndarray, candidate: np.ndarray,
                      ct: np.ndarray, activity_vmax: float,
                      output_dir: Path) -> str:
    difference = 100.0 * (candidate - control) / activity_vmax
    limit = 100.0 * DIFFERENCE_DISPLAY_FRACTION
    slices = ((93, "knees"), (117, "spine"), (228, "shoulders"))
    fig, axes = plt.subplots(len(slices), 3, figsize=(11, 10))
    for row, (z, label) in enumerate(slices):
        activity_panels = (
            (100.0 * control[z] / activity_vmax, "plain RDP + CT"),
            (100.0 * candidate[z] / activity_vmax, "CT-weighted RDP + CT"),
        )
        for column, (activity, title) in enumerate(activity_panels):
            axis = axes[row, column]
            axis.imshow(ct[z], origin="lower", cmap="gray",
                        vmin=-1000, vmax=1500)
            normalized = np.clip(activity / 100.0, 0.0, 1.0)
            rgba = plt.get_cmap("hot")(normalized)
            rgba[..., 3] = 0.85 * np.sqrt(normalized)
            axis.imshow(rgba, origin="lower")
            if row == 0:
                axis.set_title(title)

        axis = axes[row, 2]
        axis.imshow(ct[z], origin="lower", cmap="gray",
                    vmin=-1000, vmax=1500)
        diff_panel = difference[z]
        normalized = np.clip((diff_panel + limit) / (2.0 * limit), 0.0, 1.0)
        rgba = plt.get_cmap("coolwarm")(normalized)
        rgba[..., 3] = 0.9 * np.sqrt(np.clip(np.abs(diff_panel) / limit, 0.0, 1.0))
        axis.imshow(rgba, origin="lower")
        if row == 0:
            axis.set_title("candidate - control + CT")

        for column, axis in enumerate(axes[row]):
            if column == 0:
                axis.set_ylabel(f"{label} z={z}")
            axis.set_xticks([])
            axis.set_yticks([])

    activity_map = plt.cm.ScalarMappable(
        norm=plt.Normalize(0.0, 100.0), cmap="hot")
    difference_map = plt.cm.ScalarMappable(
        norm=plt.Normalize(-limit, limit), cmap="coolwarm")
    activity_axis = fig.add_axes((0.25, 0.045, 0.25, 0.015))
    difference_axis = fig.add_axes((0.62, 0.045, 0.25, 0.015))
    fig.colorbar(activity_map, cax=activity_axis, orientation="horizontal",
                 label="activity (% of control P99.9)")
    fig.colorbar(difference_map, cax=difference_axis, orientation="horizontal",
                 label="candidate - control (% of control P99.9)")
    fig.suptitle(
        "it20 fusion with CT exactly as used (Z-reversed) | CT [-1000, 1500] HU")
    fig.subplots_adjust(top=0.94, bottom=0.1, wspace=0.08, hspace=0.12)
    path = output_dir / "mdp_regct_it20_ct_overlay.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(path)


def render_orientation_diagnostic(activity: np.ndarray, stitched_ct: np.ndarray,
                                  activity_vmax: float,
                                  output_dir: Path) -> list[str]:
    flipped_ct = stitched_ct[::-1]
    y = activity.shape[1] // 2
    x = activity.shape[2] // 2
    planes = (
        (activity[:, y, :], stitched_ct[:, y, :], flipped_ct[:, y, :],
         f"coronal y={y}", "X index"),
        (activity[:, :, x], stitched_ct[:, :, x], flipped_ct[:, :, x],
         f"sagittal x={x}", "Y index"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 10), sharey=True)
    for row, (spect, current_ct, diagnostic_ct, label, horizontal) in enumerate(planes):
        axes[row, 0].imshow(spect, origin="lower", cmap="hot",
                            vmin=0.0, vmax=activity_vmax, aspect="auto")
        for column, ct_panel in ((1, current_ct), (2, diagnostic_ct)):
            axis = axes[row, column]
            axis.imshow(ct_panel, origin="lower", cmap="gray",
                        vmin=-1000, vmax=1500, aspect="auto")
            normalized = np.clip(spect / activity_vmax, 0.0, 1.0)
            rgba = plt.get_cmap("hot")(normalized)
            rgba[..., 3] = 0.8 * np.sqrt(normalized)
            axis.imshow(rgba, origin="lower", aspect="auto")
        axes[row, 0].set_ylabel(f"{label}\noutput Z index (low at bottom)")
        for axis in axes[row]:
            axis.set_xlabel(horizontal)
    axes[0, 0].set_title("SPECT control")
    axes[0, 1].set_title("regCT as used + SPECT")
    axes[0, 2].set_title("diagnostic CT Z-flip + SPECT")
    activity_map = plt.cm.ScalarMappable(
        norm=plt.Normalize(0.0, 100.0), cmap="hot")
    colorbar_axis = fig.add_axes((0.35, 0.035, 0.3, 0.015))
    fig.colorbar(activity_map, cax=colorbar_axis, orientation="horizontal",
                 label="SPECT activity (% of control P99.9)")
    fig.suptitle(
        "Multiplanar Z-orientation diagnostic | CT [-1000, 1500] HU")
    fig.subplots_adjust(top=0.93, bottom=0.1, wspace=0.12, hspace=0.2)
    multiplanar = output_dir / "mdp_regct_orientation_multiplanar.png"
    fig.savefig(multiplanar, dpi=180, facecolor="white")
    plt.close(fig)

    spect_profile = smooth_profile(activity.sum(axis=(1, 2)))
    ct_profile = smooth_profile(
        np.clip(stitched_ct - 300.0, 0.0, None).sum(axis=(1, 2)))
    spect_profile /= max(spect_profile.max(), 1e-12)
    ct_profile /= max(ct_profile.max(), 1e-12)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    z = np.arange(activity.shape[0])
    axis.plot(z, spect_profile, label="SPECT output", color="#d62728", lw=2)
    axis.plot(z, ct_profile, label="regCT as used (bone proxy)",
              color="#1f77b4", lw=1.5)
    axis.plot(z, ct_profile[::-1], label="diagnostic regCT Z-flip",
              color="#2ca02c", lw=1.5)
    for seam in (56, 112, 168, 224):
        axis.axvline(seam, color="0.75", lw=0.7)
    axis.set(xlabel="output Z index", ylabel="normalized axial profile",
             title="Axial profile orientation check")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    profiles = output_dir / "mdp_regct_orientation_z_profiles.png"
    fig.savefig(profiles, dpi=180, facecolor="white")
    plt.close(fig)
    return [str(multiplanar), str(profiles)]


def render_trajectory(control_path: Path, candidate_path: Path,
                      ct: np.ndarray, activity_vmax: float,
                      output_dir: Path) -> str | None:
    checkpoints = []
    for iteration in (5, 10, 15, 20):
        control_file = control_path.with_name(f"recon_it{iteration}.mhd")
        candidate_file = candidate_path.with_name(f"recon_it{iteration}.mhd")
        if control_file.exists() and candidate_file.exists():
            checkpoints.append((
                iteration,
                read_mhd(candidate_file).astype(np.float64, copy=False) -
                read_mhd(control_file).astype(np.float64, copy=False),
            ))
    if not checkpoints:
        return None

    scale = 100.0 / activity_vmax
    limit = 100.0 * DIFFERENCE_DISPLAY_FRACTION
    slices = ((93, "knees z=93"), (117, "spine z=117"),
              (228, "shoulders z=228"))
    fig, axes = plt.subplots(len(checkpoints), len(slices),
                             figsize=(13, 3 * len(checkpoints)))
    axes = np.atleast_2d(axes)
    for row, (iteration, difference) in enumerate(checkpoints):
        for column, (z, title) in enumerate(slices):
            axis = axes[row, column]
            axis.imshow(ct[z], origin="lower", cmap="gray",
                        vmin=-1000, vmax=1500)
            panel = difference[z] * scale
            normalized = np.clip((panel + limit) / (2.0 * limit), 0.0, 1.0)
            rgba = plt.get_cmap("coolwarm")(normalized)
            rgba[..., 3] = 0.9 * np.sqrt(np.clip(np.abs(panel) / limit, 0.0, 1.0))
            axis.imshow(rgba, origin="lower")
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"it{iteration}")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "As-used Z-reversed CT overlay | weighted RDP - plain RDP | shared +/-1.0%")
    fig.subplots_adjust(top=0.94, right=0.87, wspace=0.08, hspace=0.12)
    colorbar_axis = fig.add_axes((0.89, 0.25, 0.018, 0.5))
    difference_map = plt.cm.ScalarMappable(
        norm=plt.Normalize(-limit, limit), cmap="coolwarm")
    fig.colorbar(difference_map, cax=colorbar_axis,
                 label="difference (% of control P99.9)")
    path = output_dir / "mdp_regct_iteration_difference.png"
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return str(path)


def self_test() -> None:
    regct = np.ones((1100, 120, 120), dtype=np.float32) * 42
    stitched = stitch_regct(regct)
    assert stitched.shape == (290, 100, 100)
    assert np.allclose(stitched, 42)
    toy = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    assert gradient_magnitude(np.ones_like(toy)).max() == 0
    assert relative_l2(toy, toy, np.ones_like(toy, dtype=bool)) == 0
    profile = np.array([0.0, 1.0, 4.0, 1.0, 0.0])
    reference = np.concatenate((np.zeros(2), profile, np.zeros(3)))
    match = best_profile_window(profile, reference)
    assert match["reference_start_z"] == 2
    assert np.isclose(match["ncc"], 1.0)
    cropped = center_crop(np.arange(7 * 5 * 3).reshape(7, 5, 3), (3, 3, 3))
    assert cropped.shape == (3, 3, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--regct", type=Path)
    parser.add_argument("--spect-reference", type=Path)
    parser.add_argument("--physical-ct", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return 0
    if not all((args.control, args.candidate, args.regct, args.output_dir)):
        parser.error("--control, --candidate, --regct and --output-dir are required")
    if bool(args.spect_reference) != bool(args.physical_ct):
        parser.error("--spect-reference and --physical-ct must be provided together")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    control = read_mhd(args.control).astype(np.float64, copy=False)
    candidate = read_mhd(args.candidate).astype(np.float64, copy=False)
    ct = stitch_regct(read_mhd(args.regct).astype(np.float64, copy=False))
    activity_vmax = float(np.percentile(
        control[control > 0], ACTIVITY_DISPLAY_PERCENTILE))
    difference_percent = 100.0 * np.abs(candidate - control) / activity_vmax
    metrics = {
        "contract": {
            "control": "BSREM RDP beta=0.3 without CT weights",
            "candidate": "same control plus regCT HU weights, sigma=100, floor=0.05",
            "ctac": False,
            "iterations": 20,
        },
        "inputs": {
            "control": str(args.control),
            "candidate": str(args.candidate),
            "regct": str(args.regct),
        },
        "metrics": evaluate(control, candidate, ct),
        "checkpoint_trajectory": trajectory(args.control, args.candidate, ct),
        "visualization_contract": {
            "activity_window": f"shared control P99.9: 0..{activity_vmax}",
            "difference_data_abs_p99_5_percent_of_activity_window":
                float(np.percentile(difference_percent, 99.5)),
            "difference_data_abs_p99_95_percent_of_activity_window":
                float(np.percentile(difference_percent, 99.95)),
            "difference_window": "fixed +/-1.0% of control P99.9",
            "difference_window_over_activity_window":
                DIFFERENCE_DISPLAY_FRACTION,
            "difference_fraction_clipped_by_display_window":
                float(np.mean(difference_percent >
                              100.0 * DIFFERENCE_DISPLAY_FRACTION)),
            "ct_window_hu": [-1000, 1500],
            "overlay_alpha": "0.9*sqrt(min(abs(diff_percent)/1.0,1))",
            "interpolation": "none; array-index matched",
        },
        "figures": render(control, candidate, ct, args.output_dir),
        "decision": {
            "execution": "completed",
            "quality": "undetermined",
            "adoption": "keep baseline",
            "reason": "one MDP case at 20 iterations; diagnostic morphology evidence only",
        },
    }
    metrics["figures"].append(
        render_ct_overlay(control, candidate, ct, activity_vmax,
                          args.output_dir))
    trajectory_figure = render_trajectory(
        args.control, args.candidate, ct, activity_vmax,
        args.output_dir)
    if trajectory_figure:
        metrics["figures"].append(trajectory_figure)
    if args.spect_reference:
        spect_reference = read_mhd(args.spect_reference).astype(
            np.float64, copy=False)
        physical_ct = read_mhd(args.physical_ct).astype(np.float64, copy=False)
        metrics["orientation_diagnostic"] = orientation_diagnostic(
            control, ct, spect_reference, physical_ct)
        metrics["decision"] = {
            "execution": "completed",
            "quality": "reject current run as anatomical evidence",
            "method_quality": "undetermined until corrected-map rerun",
            "adoption": "keep baseline",
            "reason": "regCT used by the candidate is globally Z-reversed",
        }
        metrics["figures"].extend(render_orientation_diagnostic(
            control, ct, activity_vmax, args.output_dir))
    path = args.output_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
