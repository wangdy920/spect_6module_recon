#!/usr/bin/env python3
"""Minimal paired QA for one CTAC/no-CTAC reconstruction smoke."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval_mhd_stats import compute_seam_stats, read_mhd


def normalized(volume: np.ndarray) -> np.ndarray:
    scale = max(float(np.percentile(volume[volume > 0], 99.5)), 1e-12)
    return np.clip(volume / scale, 0, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ctac", type=Path, required=True)
    parser.add_argument("--ctac", type=Path, required=True)
    parser.add_argument("--mu-map", type=Path, required=True)
    parser.add_argument("--no-ctac-denom", type=Path, required=True)
    parser.add_argument("--ctac-denom", type=Path, required=True)
    parser.add_argument("--ctac-control", type=Path)
    parser.add_argument("--ctac-control-denom", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    no_ctac = read_mhd(args.no_ctac).astype(np.float64)
    ctac = read_mhd(args.ctac).astype(np.float64)
    mu = read_mhd(args.mu_map).astype(np.float64)
    no_denom = read_mhd(args.no_ctac_denom).astype(np.float64)
    ctac_denom = read_mhd(args.ctac_denom).astype(np.float64)
    if not (no_ctac.shape == ctac.shape == mu.shape == no_denom.shape ==
            ctac_denom.shape):
        raise RuntimeError("CTAC smoke volumes do not share one geometry")
    if not np.isfinite(ctac).all() or np.min(ctac) < 0 or not np.any(ctac > 0):
        raise RuntimeError("CTAC output is empty, negative, or non-finite")

    no_norm, ctac_norm = normalized(no_ctac), normalized(ctac)
    body = mu > .001
    diff = ctac_norm - no_norm
    seams = [75, 145, 215]
    metrics = {
        "decision": "runtime_keep_image_quality_pending",
        "shape_zyx": list(ctac.shape),
        "ctac_finite_nonnegative_nonzero": True,
        "sum_ratio_ctac_to_no_ctac": float(ctac.sum() / max(no_ctac.sum(), 1e-12)),
        "body_sum_ratio_ctac_to_no_ctac": float(
            ctac[body].sum() / max(no_ctac[body].sum(), 1e-12)),
        "normalized_relative_l2": float(
            np.linalg.norm(diff) / max(np.linalg.norm(no_norm), 1e-12)),
        "denom_sum_ratio_ctac_to_no_ctac": float(
            ctac_denom.sum() / max(no_denom.sum(), 1e-12)),
        "denom_relative_l2": float(
            np.linalg.norm(ctac_denom - no_denom) /
            max(np.linalg.norm(no_denom), 1e-12)),
        "no_ctac_seam": compute_seam_stats(no_ctac, seams, 2),
        "ctac_seam": compute_seam_stats(ctac, seams, 2),
    }
    control = None
    control_norm = None
    if args.ctac_control:
        control = read_mhd(args.ctac_control).astype(np.float64)
        if control.shape != ctac.shape:
            raise RuntimeError("CTAC control does not share candidate geometry")
        control_norm = normalized(control)
        metrics["candidate_vs_control"] = {
            "sum_ratio": float(ctac.sum() / max(control.sum(), 1e-12)),
            "normalized_relative_l2": float(
                np.linalg.norm(ctac_norm - control_norm) /
                max(np.linalg.norm(control_norm), 1e-12)),
            "control_seam": compute_seam_stats(control, seams, 2),
        }
        if args.ctac_control_denom:
            control_denom = read_mhd(args.ctac_control_denom).astype(np.float64)
            metrics["candidate_vs_control"]["denom_relative_l2"] = float(
                np.linalg.norm(ctac_denom - control_denom) /
                max(np.linalg.norm(control_denom), 1e-12))
    metrics_path = args.output_dir / "ctac_smoke_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    no_mip, ctac_mip = no_norm.max(1), ctac_norm.max(1)
    columns = 4 if control_norm is not None else 3
    fig, axes = plt.subplots(1, columns, figsize=(3.4 * columns, 8))
    axes[0].imshow(no_mip, origin="lower", cmap="jet", vmin=0, vmax=1,
                   aspect="auto")
    axes[0].set_title("no CTAC | normalized P99.5")
    if control_norm is not None:
        axes[1].imshow(control_norm.max(1), origin="lower", cmap="jet",
                       vmin=0, vmax=1, aspect="auto")
        axes[1].set_title("CTAC DICOM geometry")
        candidate_axis, diff_axis = 2, 3
    else:
        candidate_axis, diff_axis = 1, 2
    axes[candidate_axis].imshow(ctac_mip, origin="lower", cmap="jet",
                                vmin=0, vmax=1, aspect="auto")
    axes[candidate_axis].set_title("CTAC center translation")
    display_diff = (ctac_norm - control_norm if control_norm is not None
                    else diff)
    limit = max(float(np.percentile(np.abs(display_diff), 99)), 1e-6)
    axes[diff_axis].imshow(display_diff.max(1), origin="lower",
                           cmap="coolwarm", vmin=-limit, vmax=limit,
                           aspect="auto")
    axes[diff_axis].set_title("center translation - control")
    for axis in axes:
        for seam in seams:
            axis.axhline(seam, color="white", linewidth=.5, alpha=.5)
    fig.tight_layout()
    qa_path = args.output_dir / "ctac_vs_no_ctac_coronal_qa.png"
    fig.savefig(qa_path, dpi=180, facecolor="white")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
