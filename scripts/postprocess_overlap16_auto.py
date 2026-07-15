#!/usr/bin/env python3
"""Fixed overlap-16/k12 fusion with conservative object-aware seam closing."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_mhd_stats import read_mhd  # noqa: E402

BED_NZ = 80
OVERLAP = 16
STRIDE = BED_NZ - OVERLAP
SIGMOID_K = 12


def write_mhd(path: Path, volume: np.ndarray) -> None:
    raw = path.with_suffix(".raw")
    np.asarray(volume, dtype="<f8").tofile(raw)
    z, y, x = volume.shape
    path.write_text(
        "ObjectType = Image\nNDims = 3\n"
        f"DimSize = {x} {y} {z}\nElementSpacing = 0.3 0.3 0.3\n"
        "ElementType = MET_DOUBLE\nElementByteOrderMSB = False\n"
        f"ElementDataFile = {raw.name}\n",
        encoding="ascii",
    )


def split_beds(volume: np.ndarray, bed_nz: int) -> np.ndarray:
    if volume.ndim != 3 or volume.shape[0] % bed_nz:
        raise ValueError(f"expected Z divisible by {bed_nz}, got {volume.shape}")
    return volume.reshape(-1, bed_nz, *volume.shape[1:]).astype(np.float64)


def fuse_sigmoid(beds: np.ndarray, stride: int, k: float) -> np.ndarray:
    bed_count, bed_nz, ny, nx = beds.shape
    overlap = bed_nz - stride
    nz = bed_nz + stride * (bed_count - 1)
    numerator = np.zeros((nz, ny, nx), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    phase = np.arange(overlap, dtype=np.float64) / overlap
    ramp_up = 1.0 / (1.0 + np.exp(-k * (phase - 0.5)))
    for bed in range(bed_count):
        weights = np.ones(bed_nz, dtype=np.float64)
        if bed > 0:
            weights[:overlap] *= ramp_up
        if bed + 1 < bed_count:
            weights[-overlap:] *= 1.0 - ramp_up
        target = slice(bed * stride, bed * stride + bed_nz)
        weight = weights[:, None, None]
        numerator[target] += beds[bed] * weight
        denominator[target] += weight
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > 1e-12)


def _centerline_features(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    smooth = ndimage.gaussian_filter(volume, (0, 1, 1))
    profile = np.percentile(smooth, 99.8, axis=(1, 2))
    scale = max(float(np.percentile(volume[volume > 0], 99.5)), 1e-12)
    width = (smooth > 0.25 * scale).sum(axis=(1, 2)).astype(np.float64)
    return profile, width


def metrics(candidate: np.ndarray, baseline: np.ndarray,
            stride: int, bed_nz: int) -> dict:
    overlap = bed_nz - stride
    bed_count = 1 + (candidate.shape[0] - bed_nz) // stride
    seams = [bed * stride + overlap / 2 for bed in range(1, bed_count)]
    profile, width = _centerline_features(candidate)
    profile /= max(float(np.median(profile[profile > 0])), 1e-12)
    width /= max(float(np.median(width[width > 0])), 1e-12)
    rows = []
    changed = np.zeros(candidate.shape[0], dtype=bool)
    for seam_value in seams:
        seam = int(round(seam_value))
        lo, hi = max(2, seam - 6), min(len(profile) - 2, seam + 7)
        local = profile[lo:hi]
        local_width = width[lo:hi]
        shoulder = np.r_[profile[lo:lo + 3], profile[hi - 3:hi]]
        center = profile[max(lo, seam - 1):min(hi, seam + 2)]
        rows.append({
            "z": seam_value,
            "profile_jump": float(abs(profile[seam] - profile[seam - 1])),
            "profile_curvature": float(np.mean(np.abs(np.diff(local, n=2)))),
            "profile_band_to_shoulders": float(
                np.mean(center) / max(float(np.mean(shoulder)), 1e-12)),
            "profile_cv": float(np.std(local) / max(float(np.mean(local)), 1e-12)),
            "width_cv": float(
                np.std(local_width) / max(float(np.mean(local_width)), 1e-12)),
        })
        changed[max(0, seam - overlap):min(len(changed), seam + overlap + 1)] = True
    mask = ~changed
    delta = candidate[mask] - baseline[mask]
    ref = baseline[mask]
    positive = baseline[baseline > 0]
    threshold = float(np.percentile(positive, 90)) if positive.size else 0.0
    high = baseline >= threshold
    return {
        "seams": rows,
        "mean_profile_jump": float(np.mean([row["profile_jump"] for row in rows])),
        "mean_profile_curvature": float(
            np.mean([row["profile_curvature"] for row in rows])),
        "mean_profile_band_error": float(np.mean([
            abs(row["profile_band_to_shoulders"] - 1) for row in rows])),
        "mean_width_cv": float(np.mean([row["width_cv"] for row in rows])),
        "nonseam_relative_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-12)),
        "total_sum_ratio": float(candidate.sum() / max(baseline.sum(), 1e-12)),
        "baseline_high_signal_sum_ratio": float(
            candidate[high].sum() / max(baseline[high].sum(), 1e-12)),
    }


def _component_sizes(mask: np.ndarray, min_area: int = 8) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    sizes = np.bincount(labels.ravel())[1:]
    return sizes[sizes >= min_area]


def classify(volume: np.ndarray) -> tuple[str, dict]:
    positive = volume[volume > 0]
    if not positive.size:
        return "unknown", {"reason": "no_positive_signal"}
    p995 = float(np.percentile(positive, 99.5))
    mask = volume >= 0.10 * p995
    component_counts = []
    main_fractions = []
    for plane in mask:
        sizes = _component_sizes(plane)
        component_counts.append(int(sizes.size))
        if sizes.size:
            main_fractions.append(float(sizes.max() / sizes.sum()))

    # Coronal support separates the sparse/irregular mouse from calibrated
    # capillary/tube and the multi-glyph zimu phantom on the current evidence set.
    projection_sizes = _component_sizes(volume.max(axis=1) >= 0.10 * p995)
    projection_components = int(projection_sizes.size)
    high_fraction = float(np.mean(volume[mask] >= 0.50 * p995))
    median_components = float(np.median(component_counts))
    p90_components = float(np.percentile(component_counts, 90))
    median_main_fraction = float(np.median(main_fractions)) if main_fractions else 0.0

    # ponytail: provisional four-dataset gate; keep explicit --profile and
    # unknown/no-op until a broader labelled set justifies a trained classifier.
    if projection_components >= 12 and high_fraction <= 0.25 and p90_components >= 4:
        profile, reason = "mouse", "sparse_irregular_multicomponent_support"
    elif 4 <= projection_components <= 12 and high_fraction >= 0.30 and p90_components <= 3:
        profile, reason = "zimu", "regular_multiglyph_support"
    elif projection_components <= 3 and high_fraction >= 0.30 and p90_components <= 3:
        profile, reason = "phantom", "compact_regular_support"
    else:
        profile, reason = "unknown", "classification_gray_zone"
    return profile, {
        "reason": reason,
        "p995": p995,
        "projection_components": projection_components,
        "high_fraction": high_fraction,
        "median_components": median_components,
        "p90_components": p90_components,
        "median_main_fraction": median_main_fraction,
        "gate_maturity": "provisional_four_dataset",
    }


def _hole_pixels(mask: np.ndarray) -> int:
    return int(ndimage.binary_fill_holes(mask).sum() - mask.sum())


def seam_close(volume: np.ndarray, profile: str, close_z: int,
               half_width: int, strength: float | None) -> tuple[np.ndarray, dict]:
    default_strength = {"mouse": 0.0, "phantom": 1.0, "zimu": 1.0, "unknown": 0.0}
    amount = default_strength[profile] if strength is None else float(strength)
    if close_z < 3 or close_z % 2 == 0:
        raise ValueError("--close-z must be an odd integer >= 3")
    if not 0.0 <= amount <= 1.0:
        raise ValueError("--strength must be in [0,1]")

    seams = [bed * STRIDE + OVERLAP // 2 for bed in range(1, 4)]
    output = volume.copy()
    if amount == 0.0:
        return output, {"strength": amount, "seams": seams, "changed_voxels": 0,
                        "reverted_zimu_seams": []}

    positive = volume[volume > 0]
    threshold = 0.10 * float(np.percentile(positive, 99.5))
    radius = close_z // 2
    supported_gap = np.zeros(volume.shape, dtype=bool)
    supported_gap[radius:-radius] = (
        (volume[:-2 * radius] >= threshold) & (volume[2 * radius:] >= threshold)
    )
    closed = ndimage.grey_closing(volume, size=(close_z, 1, 1), mode="nearest")
    delta = np.maximum(closed - volume, 0.0)
    changed = np.zeros(volume.shape, dtype=bool)
    reverted = []
    for seam in seams:
        z0, z1 = max(0, seam - half_width), min(volume.shape[0], seam + half_width + 1)
        local = supported_gap[z0:z1] & (delta[z0:z1] > 0)
        output[z0:z1][local] += amount * delta[z0:z1][local]
        if profile == "zimu":
            before = volume[z0:z1]
            after = output[z0:z1]
            before_holes = (
                _hole_pixels(before.max(axis=1) >= threshold)
                + _hole_pixels(before.max(axis=2) >= threshold)
            )
            after_holes = (
                _hole_pixels(after.max(axis=1) >= threshold)
                + _hole_pixels(after.max(axis=2) >= threshold)
            )
            if after_holes < before_holes:
                output[z0:z1] = before
                reverted.append(seam)
                continue
        changed[z0:z1] |= local
    return output, {
        "strength": amount,
        "close_z": close_z,
        "seam_half_width": half_width,
        "seams": seams,
        "changed_voxels": int(changed.sum()),
        "reverted_zimu_seams": reverted,
    }


def _close_open_z(volume: np.ndarray, length: int) -> np.ndarray:
    closed = ndimage.grey_closing(volume, size=(length, 1, 1), mode="nearest")
    return ndimage.grey_opening(closed, size=(length, 1, 1), mode="nearest")


def seam_dual_scale(volume: np.ndarray, profile: str, half_width: int,
                    outer_z: int, core_z: int, spacing_z: float,
                    core_sigma_mm: float, core_strength: float) -> tuple[np.ndarray, dict]:
    seams = [bed * STRIDE + OVERLAP // 2 for bed in range(1, 4)]
    if profile not in {"phantom", "zimu"}:
        return volume.copy(), {"mode": "dual_scale", "seams": seams,
                               "changed_voxels": 0, "component_scales": [],
                               "reverted_zimu_seams": []}
    if outer_z < 3 or outer_z % 2 == 0 or core_z < outer_z or core_z % 2 == 0:
        raise ValueError("dual-scale Z lengths must be odd with 3 <= outer <= core")
    if spacing_z <= 0.0 or core_sigma_mm < 0.0 or not 0.0 <= core_strength <= 1.0:
        raise ValueError("invalid dual-scale spacing/sigma/strength")

    positive = volume[volume > 0]
    p995 = float(np.percentile(positive, 99.5))
    threshold = 0.10 * p995
    core_threshold = 0.40 * p995

    def endpoint_support(radius: int, level: float) -> np.ndarray:
        support = np.zeros(volume.shape, dtype=bool)
        support[radius:-radius] = (
            (volume[:-2 * radius] >= level) & (volume[2 * radius:] >= level)
        )
        return support

    outer_support = endpoint_support(outer_z // 2, threshold)
    core_support = endpoint_support(core_z // 2, core_threshold)
    outer = _close_open_z(volume, outer_z)
    core = _close_open_z(volume, core_z)
    if core_sigma_mm > 0.0 and core_strength > 0.0:
        smooth = ndimage.gaussian_filter1d(
            core, core_sigma_mm / spacing_z, axis=0, mode="nearest"
        )
        core = (1.0 - core_strength) * core + core_strength * smooth
    core_weight = ndimage.gaussian_filter(
        core_support.astype(np.float64), sigma=(0.0, 0.75, 0.75), mode="nearest"
    )

    output = volume.copy()
    component_scales = []
    reverted = []
    topology_locked_voxels = 0
    for seam in seams:
        z0, z1 = max(0, seam - half_width), min(volume.shape[0], seam + half_width + 1)
        if z0 >= z1:
            continue
        before = volume[z0:z1]
        candidate = before.copy()
        valid = outer_support[z0:z1] | core_support[z0:z1]
        candidate[valid] = outer[z0:z1][valid]
        weight = core_weight[z0:z1] * core_support[z0:z1]
        candidate = candidate * (1.0 - weight) + core[z0:z1] * weight

        labels, count = ndimage.label(
            before.max(axis=0) >= threshold, structure=np.ones((3, 3), dtype=bool)
        )
        seam_scales = []
        for label in range(1, count + 1):
            component = labels == label
            if int(component.sum()) < 8:
                continue
            before_sum = float(before[:, component].sum())
            after_sum = float(candidate[:, component].sum())
            if before_sum > 0.0 and after_sum > 0.0:
                scale = before_sum / after_sum
                candidate[:, component] *= scale
                seam_scales.append(scale)
        candidate = np.maximum(candidate, 0.0)

        if profile == "zimu":
            before_mask = before >= threshold
            crossed = before_mask != (candidate >= threshold)
            topology_locked_voxels += int(crossed.sum())
            candidate[before_mask] = np.maximum(candidate[before_mask], threshold)
            candidate[~before_mask] = np.minimum(
                candidate[~before_mask], np.nextafter(threshold, 0.0)
            )
            before_holes = (
                _hole_pixels(before.max(axis=1) >= threshold)
                + _hole_pixels(before.max(axis=2) >= threshold)
            )
            after_holes = (
                _hole_pixels(candidate.max(axis=1) >= threshold)
                + _hole_pixels(candidate.max(axis=2) >= threshold)
            )
            if after_holes < before_holes:
                reverted.append(seam)
                continue
        output[z0:z1] = candidate
        component_scales.append({"seam": seam, "scales": seam_scales})

    changed = output != volume
    return output, {
        "mode": "dual_scale",
        "outer_close_open_z": outer_z,
        "core_close_open_z": core_z,
        "core_sigma_mm": core_sigma_mm,
        "core_strength": core_strength,
        "spacing_z_mm": spacing_z,
        "seam_half_width": half_width,
        "seams": seams,
        "changed_voxels": int(changed.sum()),
        "component_scales": component_scales,
        "reverted_zimu_seams": reverted,
        "topology_locked_voxels": topology_locked_voxels,
    }


def seam_level(volume: np.ndarray, profile: str, half_width: int,
               anchor_width: int, max_gain: float) -> tuple[np.ndarray, dict]:
    seams = [bed * STRIDE + OVERLAP // 2 for bed in range(1, 4)]
    if profile not in {"phantom", "zimu"}:
        return volume.copy(), {"mode": "axial_level", "seams": seams,
                               "changed_slices": 0, "gains": []}
    if half_width < 2 or anchor_width < 1 or max_gain <= 1.0:
        raise ValueError("level mode requires half-width >= 2, anchor-width >= 1, max-gain > 1")

    smooth = ndimage.gaussian_filter(volume, sigma=(0.0, 1.0, 1.0), mode="nearest")
    axial = np.percentile(smooth, 99.8, axis=(1, 2))
    output = volume.copy()
    gain_records = []
    for seam in seams:
        z0, z1 = seam - half_width, seam + half_width + 1
        if z0 - anchor_width < 0 or z1 + anchor_width > volume.shape[0]:
            continue
        left = float(np.median(axial[z0 - anchor_width:z0]))
        right = float(np.median(axial[z1:z1 + anchor_width]))
        # Do not invent a continuation where the object actually ends.
        if min(left, right) < 0.15 * max(left, right, 1e-12):
            gain_records.append({"seam": seam, "skipped": "missing_endpoint"})
            continue
        target = np.linspace(left, right, z1 - z0)
        raw_gain = np.divide(target, axial[z0:z1], out=np.ones_like(target),
                             where=axial[z0:z1] > 1e-12)
        raw_gain = np.clip(raw_gain, 1.0 / max_gain, max_gain)
        gain = ndimage.gaussian_filter1d(raw_gain, 0.5, mode="nearest")
        output[z0:z1] *= gain[:, None, None]
        gain_records.append({"seam": seam, "min": float(gain.min()),
                             "max": float(gain.max())})
    return output, {"mode": "axial_level", "seams": seams,
                    "seam_half_width": half_width, "anchor_width": anchor_width,
                    "max_gain": max_gain,
                    "changed_slices": int(np.any(output != volume, axis=(1, 2)).sum()),
                    "gains": gain_records}


def seam_interpolate(volume: np.ndarray, profile: str, half_width: int,
                     anchor_width: int, contour_margin: float = 1.0,
                     spacing_z: float = 0.3, zimu_sigma_mm: float = 0.3,
                     zimu_strength: float = 0.5) -> tuple[np.ndarray, dict]:
    seams = [bed * STRIDE + OVERLAP // 2 for bed in range(1, 4)]
    if profile not in {"phantom", "zimu"}:
        return volume.copy(), {"mode": "anchor_interpolation", "seams": seams,
                               "changed_slices": 0, "records": []}
    if (half_width < 2 or anchor_width < 1 or contour_margin < 0.0
            or spacing_z <= 0.0 or zimu_sigma_mm < 0.0
            or not 0.0 <= zimu_strength <= 1.0):
        raise ValueError("invalid interp parameters")

    positive = volume[volume > 0]
    p995 = float(np.percentile(positive, 99.5))
    threshold = 0.10 * p995
    output = volume.copy()
    records = []

    if profile == "zimu":
        half_width = min(half_width, 4)
        smooth = ndimage.gaussian_filter1d(
            volume, zimu_sigma_mm / spacing_z, axis=0, mode="nearest"
        )
        locked_voxels = 0
        for seam in seams:
            z0, z1 = seam - half_width, seam + half_width + 1
            before = volume[z0:z1]
            mask = before >= threshold
            candidate = before.copy()
            candidate[mask] = ((1.0 - zimu_strength) * before[mask]
                               + zimu_strength * smooth[z0:z1][mask])
            before_sum = float(before[mask].sum())
            after_sum = float(candidate[mask].sum())
            if before_sum > 0.0 and after_sum > 0.0:
                candidate[mask] *= before_sum / after_sum
            crossed = mask != (candidate >= threshold)
            locked_voxels += int(crossed.sum())
            candidate[mask] = np.maximum(candidate[mask], threshold)
            candidate[~mask] = np.minimum(candidate[~mask],
                                           np.nextafter(threshold, 0.0))
            output[z0:z1] = candidate
            records.append({"seam": seam, "support_voxels": int(mask.sum())})
        return output, {
            "mode": "topology_locked_intensity",
            "seams": seams,
            "seam_half_width": half_width,
            "sigma_mm": zimu_sigma_mm,
            "strength": zimu_strength,
            "topology_locked_voxels": locked_voxels,
            "changed_slices": int(np.any(output != volume, axis=(1, 2)).sum()),
            "records": records,
        }

    yy, xx = np.indices(volume.shape[1:])

    def center(plane: np.ndarray, mask: np.ndarray) -> np.ndarray:
        weights = plane * mask
        total = float(weights.sum())
        if total <= 0.0:
            return np.array([(plane.shape[0] - 1) / 2.0,
                             (plane.shape[1] - 1) / 2.0])
        return np.array([(weights * yy).sum() / total,
                         (weights * xx).sum() / total])

    def components(plane: np.ndarray) -> list[dict]:
        labels, count = ndimage.label(plane >= threshold)
        result = []
        for label in range(1, count + 1):
            mask = labels == label
            if int(mask.sum()) < 8:
                continue
            soft = ndimage.binary_dilation(mask, iterations=3)
            result.append({"mask": mask, "soft": soft,
                           "center": center(plane, mask)})
        return result

    def signed_distance(mask: np.ndarray) -> np.ndarray:
        return (ndimage.distance_transform_edt(mask)
                - ndimage.distance_transform_edt(~mask))

    for seam in seams:
        z0, z1 = seam - half_width, seam + half_width + 1
        if z0 - anchor_width < 0 or z1 + anchor_width > volume.shape[0]:
            continue
        left = volume[z0 - anchor_width:z0].mean(axis=0)
        right = volume[z1:z1 + anchor_width].mean(axis=0)
        left_components, right_components = components(left), components(right)
        if (not left_components or len(left_components) != len(right_components)
                or len(left_components) > 3):
            records.append({"seam": seam, "skipped": "missing_endpoint"})
            continue

        costs = np.array([
            [np.linalg.norm(a["center"] - b["center"])
             for b in right_components] for a in left_components
        ])
        rows, cols = linear_sum_assignment(costs)
        pairs = [(left_components[i], right_components[j])
                 for i, j in zip(rows, cols)]
        candidate = np.zeros((z1 - z0, *volume.shape[1:]), dtype=volume.dtype)
        for index, z in enumerate(range(z0, z1)):
            fraction = (index + 1.0) / (z1 - z0 + 1.0)
            plane = np.zeros(volume.shape[1:], dtype=volume.dtype)
            for a, b in pairs:
                target = ((1.0 - fraction) * a["center"]
                          + fraction * b["center"])
                shift_a, shift_b = target - a["center"], target - b["center"]
                image_a = ndimage.shift(left * a["soft"], shift_a,
                                        order=1, mode="constant")
                image_b = ndimage.shift(right * b["soft"], shift_b,
                                        order=1, mode="constant")
                sdf_a = ndimage.shift(signed_distance(a["mask"]), shift_a,
                                      order=1, mode="constant", cval=-100.0)
                sdf_b = ndimage.shift(signed_distance(b["mask"]), shift_b,
                                      order=1, mode="constant", cval=-100.0)
                sdf = (1.0 - fraction) * sdf_a + fraction * sdf_b
                gate = np.clip(sdf + contour_margin, 0.0, 1.0)
                plane += ((1.0 - fraction) * image_a
                          + fraction * image_b) * gate
            candidate[index] = plane

        valid = True
        for plane in candidate:
            labels, count = ndimage.label(plane >= threshold)
            sizes = np.bincount(labels.ravel())[1:]
            if int(np.sum(sizes >= 8)) != len(pairs):
                valid = False
                break
        if not valid:
            records.append({"seam": seam, "skipped": "topology_gate"})
            continue

        for index, z in enumerate(range(z0, z1)):
            original_support = ndimage.binary_dilation(
                volume[z] >= 0.05 * p995, iterations=3
            )
            replace = original_support | (candidate[index] > 0.0)
            output[z][replace] = candidate[index][replace]
        records.append({
            "seam": seam,
            "components": len(pairs),
            "max_match_distance": float(costs[rows, cols].max()),
        })
    return output, {"mode": "anchor_interpolation", "seams": seams,
                    "seam_half_width": half_width, "anchor_width": anchor_width,
                    "contour_margin": contour_margin,
                    "changed_slices": int(np.any(output != volume, axis=(1, 2)).sum()),
                    "records": records}


def self_check() -> None:
    volume = np.zeros((80, 5, 5), dtype=np.float64)
    volume[71, 1:4, 1:4] = volume[73, 1:4, 1:4] = 1.0
    phantom, _ = seam_close(volume, "phantom", 3, 1, None)
    mouse, _ = seam_close(volume, "mouse", 3, 1, None)
    assert phantom[72, 2, 2] == 1.0
    assert np.array_equal(mouse, volume)
    assert np.array_equal(phantom[:71], volume[:71])
    dual, _ = seam_dual_scale(volume, "phantom", 1, 3, 5, 0.3, 0.25, 0.2)
    assert dual[72, 2, 2] > 0.0
    assert np.isclose(dual[71:74, 1:4, 1:4].sum(), 18.0)
    assert np.array_equal(dual[:71], volume[:71])
    hump = volume.copy()
    hump[68:77, 1:4, 1:4] = 1.0
    hump[72, 1:4, 1:4] = 2.0
    level, _ = seam_level(hump, "phantom", 2, 1, 2.0)
    assert level[72, 2, 2] < hump[72, 2, 2]
    assert np.array_equal(seam_level(hump, "mouse", 2, 1, 2.0)[0], hump)
    interp, _ = seam_interpolate(hump, "phantom", 2, 1)
    assert interp[72, 2, 2] == 1.0
    zimu, _ = seam_interpolate(hump, "zimu", 2, 1)
    threshold = 0.10 * np.percentile(hump[hump > 0], 99.5)
    assert np.array_equal(zimu >= threshold, hump >= threshold)


def main() -> int:
    self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Four concatenated 80-slice beds or fused 272-slice MHD.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("auto", "mouse", "phantom", "zimu", "unknown"),
                        default="auto")
    parser.add_argument("--mode", choices=("close", "dual", "level", "interp"),
                        default="close")
    parser.add_argument("--close-z", type=int, default=5)
    parser.add_argument("--seam-half-width", type=int, default=4)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--outer-close-z", type=int, default=3)
    parser.add_argument("--core-close-z", type=int, default=5)
    parser.add_argument("--spacing-z", type=float, default=0.3)
    parser.add_argument("--core-sigma-mm", type=float, default=0.25)
    parser.add_argument("--core-strength", type=float, default=0.20)
    parser.add_argument("--level-anchor-width", type=int, default=3)
    parser.add_argument("--level-max-gain", type=float, default=2.0)
    parser.add_argument("--contour-margin", type=float, default=1.0)
    parser.add_argument("--zimu-sigma-mm", type=float, default=0.3)
    parser.add_argument("--zimu-strength", type=float, default=0.5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    loaded = read_mhd(args.input)
    if loaded.shape[0] == 4 * BED_NZ:
        beds = split_beds(loaded, BED_NZ)
        base = fuse_sigmoid(beds, STRIDE, SIGMOID_K)
    elif loaded.shape[0] == BED_NZ + 3 * STRIDE:
        base = loaded.astype(np.float64)
    else:
        raise ValueError(f"expected Z=320 full beds or Z=272 fused volume, got {loaded.shape}")
    auto_profile, features = classify(base)
    profile = auto_profile if args.profile == "auto" else args.profile
    if args.mode == "dual":
        processed, operation = seam_dual_scale(
            base, profile, args.seam_half_width, args.outer_close_z,
            args.core_close_z, args.spacing_z, args.core_sigma_mm,
            args.core_strength)
    elif args.mode == "level":
        processed, operation = seam_level(
            base, profile, args.seam_half_width, args.level_anchor_width,
            args.level_max_gain)
    elif args.mode == "interp":
        processed, operation = seam_interpolate(
            base, profile, args.seam_half_width, args.level_anchor_width,
            args.contour_margin, args.spacing_z, args.zimu_sigma_mm,
            args.zimu_strength)
    else:
        processed, operation = seam_close(
            base, profile, args.close_z, args.seam_half_width, args.strength
        )

    write_mhd(args.output / "overlap16_sigmoid_k12.mhd", base)
    suffix = profile if args.mode == "close" else f"{profile}_{args.mode}"
    write_mhd(args.output / f"overlap16_sigmoid_k12_{suffix}.mhd", processed)
    seam_mask = np.zeros(base.shape[0], dtype=bool)
    for seam in operation["seams"]:
        seam_mask[max(0, seam - args.seam_half_width):seam + args.seam_half_width + 1] = True
    report = {
        "decision": "diagnostic_object_aware_postprocess",
        "input": str(args.input),
        "fusion": {"overlap": OVERLAP, "stride": STRIDE, "method": "sigmoid", "k": SIGMOID_K},
        "requested_profile": args.profile,
        "selected_profile": profile,
        "mode": args.mode,
        "classification": features,
        "operation": operation,
        "metrics_before": metrics(base, base, STRIDE, BED_NZ),
        "metrics_after": metrics(processed, base, STRIDE, BED_NZ),
        "nonseam_exact": bool(np.array_equal(processed[~seam_mask], base[~seam_mask])),
        "sum_ratio": float(processed.sum() / max(base.sum(), 1e-12)),
    }
    report_name = "report.json" if args.mode == "close" else f"report_{args.mode}.json"
    (args.output / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
