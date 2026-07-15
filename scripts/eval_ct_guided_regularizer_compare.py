#!/usr/bin/env python3
"""Fixed-window QA for CTAC and CT-weighted RDP/TV multi-bed runs."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_mhd_stats import read_mhd  # noqa: E402
from eval_overlap10_vs11_joint import metrics as seam_metrics  # noqa: E402

CASES = ("mlem_noct", "mlem_ctac", "bsrem0_ctac", "rdp_ctac", "tv_ctac")


def rel_l2(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm((candidate - reference)[mask]) /
                 max(np.linalg.norm(reference[mask]), 1e-12))


def gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    result = np.zeros_like(volume, dtype=np.float64)
    for axis in range(3):
        delta = np.diff(volume, axis=axis)
        target = [slice(None)] * 3
        target[axis] = slice(0, -1)
        result[tuple(target)] += delta * delta
    return np.sqrt(result)


def evaluate_dataset(root: Path, name: str, mu_path: Path, iterations: int) -> dict:
    volumes = {
        case: read_mhd(root / name / case / f"recon_it{iterations}.mhd").astype(np.float64)
        for case in CASES
    }
    reference = volumes["mlem_noct"]
    if any(v.shape != reference.shape for v in volumes.values()):
        raise ValueError(f"{name}: output geometry mismatch")
    if any(not np.isfinite(v).all() or np.min(v) < 0 for v in volumes.values()):
        raise ValueError(f"{name}: output must be finite and nonnegative")
    mu = read_mhd(mu_path).astype(np.float64)
    if mu.shape != reference.shape:
        raise ValueError(f"{name}: mu-map geometry mismatch")

    yy, xx = np.indices(reference.shape[1:])
    trusted = np.broadcast_to(np.hypot(xx - 53.5, yy - 53.5) <= 46, reference.shape)
    seam_z = np.zeros(reference.shape[0], dtype=bool)
    for centre in (75, 145, 215):
        seam_z |= np.abs(np.arange(reference.shape[0]) - centre) <= 8
    nonseam = trusted & ~np.broadcast_to(seam_z[:, None, None], reference.shape)
    body = trusted & (mu > 1e-5)
    mu_grad = gradient_magnitude(mu)
    positive_mu_grad = mu_grad[body & (mu_grad > 0)]
    edge_threshold = (float(np.percentile(positive_mu_grad, 75))
                      if positive_mu_grad.size else np.inf)
    ct_edge = body & (mu_grad >= edge_threshold)
    ct_flat = body & (mu_grad < edge_threshold)

    result = {"mu_map": str(mu_path), "shape_zyx": list(reference.shape)}
    for case, volume in volumes.items():
        comparison = volumes["bsrem0_ctac"] if case in ("rdp_ctac", "tv_ctac") else reference
        positive = comparison[trusted & (comparison > 0)]
        high_threshold = float(np.percentile(positive, 90)) if positive.size else np.inf
        high = trusted & (comparison >= high_threshold)
        image_grad = gradient_magnitude(volume)
        comparison_grad = gradient_magnitude(comparison)
        result[case] = {
            "reference": "bsrem0_ctac" if case in ("rdp_ctac", "tv_ctac") else "mlem_noct",
            "trusted_relative_l2": rel_l2(volume, comparison, trusted),
            "nonseam_relative_l2": rel_l2(volume, comparison, nonseam),
            "trusted_sum_ratio": float(volume[trusted].sum() /
                                       max(comparison[trusted].sum(), 1e-12)),
            "reference_top10_sum_ratio": float(volume[high].sum() /
                                               max(comparison[high].sum(), 1e-12)),
            "ct_edge_gradient_ratio": float(image_grad[ct_edge].mean() /
                                            max(comparison_grad[ct_edge].mean(), 1e-12))
                                      if np.any(ct_edge) else None,
            "ct_flat_gradient_ratio": float(image_grad[ct_flat].mean() /
                                            max(comparison_grad[ct_flat].mean(), 1e-12))
                                      if np.any(ct_flat) else None,
            "seam": seam_metrics(volume, 10),
        }

    positive = volumes["mlem_ctac"][volumes["mlem_ctac"] > 0]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    fig, axes = plt.subplots(1, len(CASES), figsize=(15, 9), sharex=True, sharey=True)
    for axis, case in zip(axes, CASES):
        axis.imshow(volumes[case].max(1), origin="lower", cmap="jet",
                    vmin=0, vmax=vmax, aspect="auto")
        for seam in (75, 145, 215):
            axis.axhline(seam, color="white", lw=0.5, alpha=0.7)
        axis.set_title(case)
    fig.suptitle(f"{name} | fixed CTAC P99.5 window | overlap10 FOV15 it{iterations}")
    fig.tight_layout()
    figure = root / f"{name}_ct_guided_regularizer_fixed_window.png"
    fig.savefig(figure, dpi=180, facecolor="white")
    plt.close(fig)
    result["fixed_window_png"] = str(figure)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        toy = np.arange(27, dtype=float).reshape(3, 3, 3)
        mask = np.ones_like(toy, dtype=bool)
        assert rel_l2(toy, toy, mask) == 0.0
        assert gradient_magnitude(np.ones_like(toy)).max() == 0.0
        assert gradient_magnitude(toy).max() > 0.0
        print("self-test passed")
        return 0
    if args.root is None:
        parser.error("--root is required unless --self-test is used")
    manifest = json.loads((args.root / "run_manifest.json").read_text(encoding="utf-8-sig"))
    mu_by_dataset = {}
    for run in manifest["runs"]:
        mu_by_dataset[run["dataset"]] = Path(run["mu"])
    metrics = {
        "contract": manifest["contract"],
        "comparisons": {
            name: evaluate_dataset(args.root, name, mu, args.iterations)
            for name, mu in mu_by_dataset.items()
        },
        "decision": "pending_visual_and_quantitative_gate",
    }
    path = args.root / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
