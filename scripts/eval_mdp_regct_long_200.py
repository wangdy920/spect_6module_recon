#!/usr/bin/env python3
"""Evaluate the five-arm 200-iteration MDP BSREM regularizer comparison."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_mhd_stats import read_mhd  # noqa: E402
from eval_mdp_regct_smoke import gradient_magnitude, stitch_regct  # noqa: E402
from prepare_ct_mu_map import (bone_hot_masks, occupancy_translation_mm,
                               spine_knee_translation_mm)  # noqa: E402

CASES = ("bare_bsrem", "tv_plain", "tv_ct", "rdp_plain", "rdp_ct")
LABELS = {
    "bare_bsrem": "bare BSREM", "tv_plain": "TV", "tv_ct": "TV + CT",
    "rdp_plain": "RDP", "rdp_ct": "RDP + CT",
}
COLORS = {
    "bare_bsrem": "#111111", "tv_plain": "#3b82f6", "tv_ct": "#06b6d4",
    "rdp_plain": "#ef4444", "rdp_ct": "#f59e0b",
}


def rel_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm((a - b)[mask]) /
                 max(np.linalg.norm(b[mask]), 1e-12))


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x, y = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


def load_runs(root: Path, checkpoints: list[int]) -> dict[str, dict[int, np.ndarray]]:
    runs = {}
    for case in CASES:
        runs[case] = {}
        for iteration in checkpoints:
            path = root / case / f"recon_it{iteration}.mhd"
            if not path.exists():
                raise FileNotFoundError(path)
            volume = read_mhd(path).astype(np.float64, copy=False)
            if volume.shape != (290, 100, 100) or not np.isfinite(volume).all() or volume.min() < 0:
                raise ValueError(f"invalid reconstruction {path}: shape={volume.shape}")
            runs[case][iteration] = volume
    return runs


def edge_masks(ct: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    gradient = gradient_magnitude(ct)
    values = gradient[support & (gradient > 0)]
    low, high = np.percentile(values, (25, 75))
    flat = support & (gradient <= low)
    edge = support & (gradient >= high)
    return flat, edge, {"flat_max_hu_per_voxel": float(low),
                        "edge_min_hu_per_voxel": float(high)}


def summarize(runs: dict[str, dict[int, np.ndarray]], ct: np.ndarray,
              checkpoints: list[int]) -> dict:
    metrics = {"checkpoints": {}, "pairs": {}}
    previous = {}
    for iteration in checkpoints:
        volumes = {case: runs[case][iteration] for case in CASES}
        support = np.logical_or.reduce([volume > 0 for volume in volumes.values()])
        flat, edge, thresholds = edge_masks(ct, support)
        bare = volumes["bare_bsrem"]
        entry = {"ct_gradient_rule": thresholds, "arms": {}}
        for case, volume in volumes.items():
            gradient = gradient_magnitude(volume)
            entry["arms"][case] = {
                "sum_ratio_to_bare": float(volume[support].sum() /
                                           max(bare[support].sum(), 1e-12)),
                "relative_l2_to_bare": rel_l2(volume, bare, support),
                "ncc_to_bare": ncc(volume, bare, support),
                "mean_gradient_over_mean_activity": float(
                    gradient[support].mean() / max(volume[support].mean(), 1e-12)),
                "change_from_previous_checkpoint_rel_l2": (
                    None if case not in previous else rel_l2(volume, previous[case], support)),
            }
            previous[case] = volume
        for name, candidate, control in (
            ("tv_ct_vs_plain", "tv_ct", "tv_plain"),
            ("rdp_ct_vs_plain", "rdp_ct", "rdp_plain"),
            ("tv_plain_vs_bare", "tv_plain", "bare_bsrem"),
            ("rdp_plain_vs_bare", "rdp_plain", "bare_bsrem"),
            ("tv_ct_vs_bare", "tv_ct", "bare_bsrem"),
            ("rdp_ct_vs_bare", "rdp_ct", "bare_bsrem"),
            ("tv_plain_vs_rdp_plain", "tv_plain", "rdp_plain"),
            ("tv_ct_vs_rdp_ct", "tv_ct", "rdp_ct"),
        ):
            a, b = volumes[candidate], volumes[control]
            ga, gb = gradient_magnitude(a), gradient_magnitude(b)
            pair = {
                "relative_l2": rel_l2(a, b, support),
                "ncc": ncc(a, b, support),
                "sum_ratio": float(a[support].sum() / max(b[support].sum(), 1e-12)),
                "ct_flat_gradient_ratio": float(ga[flat].mean() / max(gb[flat].mean(), 1e-12)),
                "ct_edge_gradient_ratio": float(ga[edge].mean() / max(gb[edge].mean(), 1e-12)),
            }
            pair["edge_over_flat_gradient_ratio"] = (
                pair["ct_edge_gradient_ratio"] / max(pair["ct_flat_gradient_ratio"], 1e-12))
            entry.setdefault("pairs", {})[name] = pair
        metrics["checkpoints"][str(iteration)] = entry
    return metrics


def plane(volume: np.ndarray, kind: str, index: int) -> np.ndarray:
    if kind == "axial":
        return volume[index]
    if kind == "coronal":
        return volume[:, index, :]
    if kind == "sagittal":
        return volume[:, :, index]
    raise ValueError(kind)


def activity_overlay(axis, ct_panel: np.ndarray, activity: np.ndarray,
                     vmax: float) -> None:
    axis.imshow(ct_panel, origin="lower", cmap="gray", vmin=-1000, vmax=1500,
                aspect="auto" if ct_panel.shape[0] != ct_panel.shape[1] else "equal")
    normalized = np.clip(activity / vmax, 0, 1)
    rgba = plt.get_cmap("hot")(normalized)
    rgba[..., 3] = 0.82 * np.sqrt(normalized)
    axis.imshow(rgba, origin="lower",
                aspect="auto" if ct_panel.shape[0] != ct_panel.shape[1] else "equal")


def difference_overlay(axis, ct_panel: np.ndarray, difference: np.ndarray,
                       limit: float) -> None:
    axis.imshow(ct_panel, origin="lower", cmap="gray", vmin=-1000, vmax=1500,
                aspect="auto" if ct_panel.shape[0] != ct_panel.shape[1] else "equal")
    normalized = np.clip((difference + limit) / (2 * limit), 0, 1)
    rgba = plt.get_cmap("coolwarm")(normalized)
    rgba[..., 3] = 0.9 * np.sqrt(np.clip(np.abs(difference) / limit, 0, 1))
    axis.imshow(rgba, origin="lower",
                aspect="auto" if ct_panel.shape[0] != ct_panel.shape[1] else "equal")


def render_multiplanar(final: dict[str, np.ndarray], ct: np.ndarray,
                       vmax: float, output: Path) -> str:
    views = (("axial", 93, "knees z=93"), ("axial", 117, "spine z=117"),
             ("axial", 228, "shoulders z=228"), ("coronal", 50, "coronal y=50"),
             ("sagittal", 50, "sagittal x=50"))
    fig, axes = plt.subplots(len(views), len(CASES), figsize=(16, 17))
    for row, (kind, index, view_label) in enumerate(views):
        ct_panel = plane(ct, kind, index)
        for column, case in enumerate(CASES):
            activity_overlay(axes[row, column], ct_panel, plane(final[case], kind, index), vmax)
            if row == 0:
                axes[row, column].set_title(LABELS[case])
            if column == 0:
                axes[row, column].set_ylabel(view_label)
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    fig.suptitle("MDP it200 | corrected CT overlay | shared bare-BSREM P99.9 activity window")
    fig.subplots_adjust(top=0.965, bottom=0.055, wspace=0.04, hspace=0.1)
    bar = fig.add_axes((0.34, 0.025, 0.32, 0.012))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap="hot"),
                 cax=bar, orientation="horizontal", label="activity (shared absolute scale)")
    path = output / "it200_five_arm_ct_overlay_multiplanar.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(path)


def render_differences(final: dict[str, np.ndarray], ct: np.ndarray,
                       pairs: tuple[tuple[str, str, str], ...], title: str,
                       filename: str, output: Path) -> tuple[str, dict]:
    views = (("axial", 93, "knees z=93"), ("axial", 117, "spine z=117"),
             ("axial", 228, "shoulders z=228"), ("coronal", 50, "coronal y=50"),
             ("sagittal", 50, "sagittal x=50"))
    differences = [final[a] - final[b] for _, a, b in pairs]
    pooled = np.concatenate([np.abs(diff).ravel() for diff in differences])
    limit = float(np.percentile(pooled, 99.9))
    fig, axes = plt.subplots(len(views), len(pairs), figsize=(4 * len(pairs), 17))
    axes = np.atleast_2d(axes)
    for row, (kind, index, view_label) in enumerate(views):
        ct_panel = plane(ct, kind, index)
        for column, ((label, _, _), difference) in enumerate(zip(pairs, differences)):
            difference_overlay(axes[row, column], ct_panel, plane(difference, kind, index), limit)
            if row == 0:
                axes[row, column].set_title(label)
            if column == 0:
                axes[row, column].set_ylabel(view_label)
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    fig.suptitle(title + " | corrected CT | pooled P99.9 difference scale",
                 y=0.99, fontsize=14)
    fig.subplots_adjust(top=0.94, bottom=0.055, wspace=0.05, hspace=0.1)
    bar = fig.add_axes((0.34, 0.025, 0.32, 0.012))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-limit, limit), cmap="coolwarm"),
                 cax=bar, orientation="horizontal", label="candidate - control")
    path = output / filename
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(path), {"difference_limit": limit,
                       "clipped_fraction": float(np.mean(pooled > limit))}


def render_plain_differences(final: dict[str, np.ndarray],
                             pairs: tuple[tuple[str, str, str], ...],
                             title: str, filename: str, limit: float,
                             output: Path) -> str:
    views = (("axial", 93, "knees z=93"), ("axial", 117, "spine z=117"),
             ("axial", 228, "shoulders z=228"), ("coronal", 50, "coronal y=50"),
             ("sagittal", 50, "sagittal x=50"))
    fig, axes = plt.subplots(len(views), len(pairs), figsize=(4 * len(pairs), 17))
    axes = np.atleast_2d(axes)
    for row, (kind, index, view_label) in enumerate(views):
        for column, (label, candidate, control) in enumerate(pairs):
            panel = plane(final[candidate] - final[control], kind, index)
            axes[row, column].imshow(
                panel, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit,
                aspect="auto" if panel.shape[0] != panel.shape[1] else "equal")
            if row == 0:
                axes[row, column].set_title(label)
            if column == 0:
                axes[row, column].set_ylabel(view_label)
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    fig.suptitle(title + " | SPECT difference only | same P99.9 scale",
                 y=0.99, fontsize=14)
    fig.subplots_adjust(top=0.94, bottom=0.055, wspace=0.05, hspace=0.1)
    bar = fig.add_axes((0.34, 0.025, 0.32, 0.012))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-limit, limit), cmap="coolwarm"),
                 cax=bar, orientation="horizontal", label="candidate - control")
    path = output / filename
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(path)


def render_trajectory(runs: dict[str, dict[int, np.ndarray]], ct: np.ndarray,
                      checkpoints: list[int], output: Path) -> tuple[str, dict]:
    pairs = (("TV + CT - TV", "tv_ct", "tv_plain"),
             ("RDP + CT - RDP", "rdp_ct", "rdp_plain"))
    all_diffs = [runs[a][iteration] - runs[b][iteration]
                 for _, a, b in pairs for iteration in checkpoints]
    pooled = np.concatenate([np.abs(diff).ravel() for diff in all_diffs])
    limit = float(np.percentile(pooled, 99.9))
    fig, axes = plt.subplots(len(pairs), len(checkpoints), figsize=(20, 5.2))
    for row, (label, candidate, control) in enumerate(pairs):
        for column, iteration in enumerate(checkpoints):
            difference_overlay(axes[row, column], ct[117],
                               runs[candidate][iteration][117] - runs[control][iteration][117], limit)
            axes[row, column].set_title(f"it{iteration}" if row == 0 else "")
            if column == 0:
                axes[row, column].set_ylabel(label + "\nspine z=117")
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    fig.suptitle("CT-guidance difference trajectory | fixed pooled P99.9 scale")
    fig.subplots_adjust(top=0.87, bottom=0.15, wspace=0.04, hspace=0.12)
    bar = fig.add_axes((0.38, 0.07, 0.25, 0.025))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-limit, limit), cmap="coolwarm"),
                 cax=bar, orientation="horizontal", label="CT-guided - plain")
    path = output / "ct_guidance_axial_trajectory.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(path), {"difference_limit": limit,
                       "clipped_fraction": float(np.mean(pooled > limit))}


def render_plain_trajectory(runs: dict[str, dict[int, np.ndarray]],
                            checkpoints: list[int], limit: float,
                            output: Path) -> str:
    pairs = (("TV + CT - TV", "tv_ct", "tv_plain"),
             ("RDP + CT - RDP", "rdp_ct", "rdp_plain"))
    fig, axes = plt.subplots(len(pairs), len(checkpoints), figsize=(20, 5.2))
    for row, (label, candidate, control) in enumerate(pairs):
        for column, iteration in enumerate(checkpoints):
            panel = runs[candidate][iteration][117] - runs[control][iteration][117]
            axes[row, column].imshow(panel, origin="lower", cmap="coolwarm",
                                     vmin=-limit, vmax=limit)
            axes[row, column].set_title(f"it{iteration}" if row == 0 else "")
            if column == 0:
                axes[row, column].set_ylabel(label + "\nspine z=117")
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    fig.suptitle("CT-guidance difference trajectory | SPECT difference only | fixed P99.9 scale")
    fig.subplots_adjust(top=0.87, bottom=0.15, wspace=0.04, hspace=0.12)
    bar = fig.add_axes((0.38, 0.07, 0.25, 0.025))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-limit, limit), cmap="coolwarm"),
                 cax=bar, orientation="horizontal", label="CT-guided - plain")
    path = output / "ct_guidance_axial_trajectory_no_ct.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(path)


def registration_audit(ct: np.ndarray, activity: np.ndarray) -> dict:
    spacing = np.array([0.3, 0.3, 0.3])
    landmarks = spine_knee_translation_mm(ct, activity, spacing)
    bone, hot = bone_hot_masks(ct, activity, hot_frac=0.28)
    occupancy = occupancy_translation_mm(bone, hot, spacing)
    return {
        "status": "orientation_correct_strict_registration_not_validated",
        "activity": "bare BSREM it200; avoids circular CT-guided candidate selection",
        "landmark_residual": landmarks,
        "occupancy_residual": occupancy,
        "interpretation": (
            "The landmark aggregate and occupancy estimates disagree, while slice-wise "
            "spine MAD remains multi-voxel. No single translation is certified as a fix."
        ),
    }


def render_registration_audit(ct: np.ndarray, activity: np.ndarray,
                              output: Path) -> str:
    views = (("axial", 93, "z=93"), ("axial", 117, "z=117"),
             ("axial", 228, "z=228"), ("coronal", 50, "y=50"),
             ("sagittal", 50, "x=50"))
    vmax = float(np.percentile(activity[activity > 0], 99.9))
    fig, axes = plt.subplots(1, len(views), figsize=(18, 5))
    for axis, (kind, index, label) in zip(axes, views):
        spect = plane(activity, kind, index)
        ct_panel = plane(ct, kind, index)
        axis.imshow(spect, origin="lower", cmap="hot", vmin=0, vmax=vmax,
                    aspect="auto" if spect.shape[0] != spect.shape[1] else "equal")
        axis.contour(ct_panel, levels=[400], colors=["#00ff66"], linewidths=0.7,
                     origin="lower")
        axis.set_title(label); axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle("Registration residual diagnostic | bare BSREM activity + CT HU=400 contour\n"
                 "orientation is corrected; local spatial agreement is not certified")
    fig.tight_layout()
    path = output / "registration_residual_diagnostic.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(path)


def render_metrics(metrics: dict, checkpoints: list[int], output: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    entries = [metrics["checkpoints"][str(i)] for i in checkpoints]
    for name, label in (("tv_ct_vs_plain", "TV + CT vs TV"),
                        ("rdp_ct_vs_plain", "RDP + CT vs RDP"),
                        ("tv_plain_vs_bare", "TV vs bare"),
                        ("rdp_plain_vs_bare", "RDP vs bare")):
        axes[0, 0].plot(checkpoints, [e["pairs"][name]["relative_l2"] for e in entries],
                        marker="o", label=label)
        axes[0, 1].plot(checkpoints, [e["pairs"][name]["sum_ratio"] for e in entries],
                        marker="o", label=label)
    for case in CASES:
        changes = [e["arms"][case]["change_from_previous_checkpoint_rel_l2"] for e in entries]
        axes[1, 0].plot(checkpoints[1:], changes[1:], marker="o", color=COLORS[case], label=LABELS[case])
        axes[1, 1].plot(checkpoints,
                        [e["arms"][case]["mean_gradient_over_mean_activity"] for e in entries],
                        marker="o", color=COLORS[case], label=LABELS[case])
    titles = ("paired relative L2", "paired sum ratio", "change from previous 20 iterations",
              "mean gradient / mean activity")
    for axis, title in zip(axes.ravel(), titles):
        axis.set_title(title); axis.set_xlabel("iteration"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
    fig.suptitle("200-iteration trajectory metrics | unfiltered primary outputs")
    fig.tight_layout()
    path = output / "iteration_metrics.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(path)


def compare_previous_map(final: dict[str, np.ndarray], ct: np.ndarray,
                         previous_root: Path, previous_regct: Path,
                         iteration: int) -> dict:
    previous = load_runs(previous_root, [iteration])
    previous_ct = stitch_regct(read_mhd(previous_regct).astype(np.float64, copy=False))
    if previous_ct.shape != ct.shape:
        raise ValueError(f"previous CT shape {previous_ct.shape} != {ct.shape}")
    support = np.logical_or.reduce([volume > 0 for volume in final.values()])
    ct_support = np.isfinite(ct) & np.isfinite(previous_ct)
    pairs = {}
    for case in ("tv_ct", "rdp_ct"):
        a, b = final[case], previous[case][iteration]
        pairs[case] = {
            "relative_l2": rel_l2(a, b, support),
            "ncc": ncc(a, b, support),
            "sum_ratio": float(a[support].sum() / max(b[support].sum(), 1e-12)),
        }
    return {
        "previous_root": str(previous_root),
        "previous_regct": str(previous_regct),
        "stitched_ct_relative_l2": rel_l2(ct, previous_ct, ct_support),
        "reconstruction_it200": pairs,
    }


def self_test() -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    a = np.arange(8, dtype=float).reshape(2, 2, 2)
    assert rel_l2(a, a, mask) == 0
    assert np.isclose(ncc(a, a, mask), 1)
    assert np.array_equal(plane(a, "axial", 1), a[1])
    assert np.array_equal(plane(a, "coronal", 1), a[:, 1, :])
    assert stitch_regct(np.ones((1100, 120, 120))).shape == (290, 100, 100)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--regct", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-root", type=Path)
    parser.add_argument("--compare-regct", type=Path)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); print("self-test passed"); return 0
    if not all((args.root, args.regct, args.output_dir)):
        parser.error("--root, --regct and --output-dir are required")
    checkpoints = list(range(args.save_every, args.iterations + 1, args.save_every))
    runs = load_runs(args.root, checkpoints)
    ct = stitch_regct(read_mhd(args.regct).astype(np.float64, copy=False))
    if ct.shape != (290, 100, 100) or not np.isfinite(ct).all():
        raise ValueError(f"invalid stitched CT: {ct.shape}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = summarize(runs, ct, checkpoints)
    final = {case: runs[case][args.iterations] for case in CASES}
    positive = final["bare_bsrem"][final["bare_bsrem"] > 0]
    vmax = float(np.percentile(positive, 99.9))
    figures = [render_multiplanar(final, ct, vmax, args.output_dir)]
    path, ct_scale = render_differences(
        final, ct, (("TV + CT - TV", "tv_ct", "tv_plain"),
                    ("RDP + CT - RDP", "rdp_ct", "rdp_plain")),
        "it200 CT-guidance paired differences", "it200_ct_guidance_differences.png", args.output_dir)
    figures.append(path)
    figures.append(render_plain_differences(
        final, (("TV + CT - TV", "tv_ct", "tv_plain"),
                ("RDP + CT - RDP", "rdp_ct", "rdp_plain")),
        "it200 CT-guidance paired differences", "it200_ct_guidance_differences_no_ct.png",
        ct_scale["difference_limit"], args.output_dir))
    path, reg_scale = render_differences(
        final, ct, (("TV - bare", "tv_plain", "bare_bsrem"),
                    ("RDP - bare", "rdp_plain", "bare_bsrem"),
                    ("TV + CT - bare", "tv_ct", "bare_bsrem"),
                    ("RDP + CT - bare", "rdp_ct", "bare_bsrem")),
        "it200 regularizer-versus-bare differences", "it200_regularizer_differences.png", args.output_dir)
    figures.append(path)
    figures.append(render_plain_differences(
        final, (("TV - bare", "tv_plain", "bare_bsrem"),
                ("RDP - bare", "rdp_plain", "bare_bsrem"),
                ("TV + CT - bare", "tv_ct", "bare_bsrem"),
                ("RDP + CT - bare", "rdp_ct", "bare_bsrem")),
        "it200 regularizer-versus-bare differences",
        "it200_regularizer_differences_no_ct.png",
        reg_scale["difference_limit"], args.output_dir))
    path, trajectory_scale = render_trajectory(runs, ct, checkpoints, args.output_dir)
    figures += [path,
                render_plain_trajectory(runs, checkpoints,
                                        trajectory_scale["difference_limit"], args.output_dir),
                render_metrics(metrics, checkpoints, args.output_dir),
                render_registration_audit(ct, final["bare_bsrem"], args.output_dir)]
    result = {
        "contract": {"ctac": False, "iterations": args.iterations,
                     "save_every": args.save_every, "activity_source": "unfiltered primary",
                     "activity_window": f"bare BSREM P99.9 = {vmax}",
                     "ct_window_hu": [-1000, 1500],
                     "difference_window_rule": "pooled absolute P99.9 per comparison family"},
        "inputs": {"root": str(args.root), "regct": str(args.regct)},
        "metrics": metrics,
        "visualization": {"ct_guidance": ct_scale, "regularizer": reg_scale,
                          "trajectory": trajectory_scale},
        "registration_audit": registration_audit(ct, final["bare_bsrem"]),
        "figures": figures,
        "interpretation_limit": "single subject, no independent truth or calibrated acceptance threshold",
    }
    if args.compare_root or args.compare_regct:
        if not (args.compare_root and args.compare_regct):
            parser.error("--compare-root and --compare-regct must be set together")
        result["previous_map_compare"] = compare_previous_map(
            final, ct, args.compare_root, args.compare_regct, args.iterations)
    path = args.output_dir / "metrics.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
