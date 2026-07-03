#!/usr/bin/env python3
"""Compare compact-symmetric reconstruction output against legacy CUDA output."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from eval_mhd_stats import read_mhd


def _read_legacy_raw(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"Legacy raw is too small: {path}")
    nx, ny, nz, _magic = struct.unpack("<iiii", data[:16])
    _voxel_size = struct.unpack("<f", data[16:20])[0]
    expected = nx * ny * nz
    arr = np.frombuffer(data, dtype="<f4", offset=20)
    if arr.size != expected:
        raise ValueError(
            f"Legacy raw voxel count mismatch: got {arr.size}, expected {expected}"
        )
    return arr.astype(np.float64, copy=False).reshape((nz, ny, nx))


def _read_headerless_img_f64(path: Path, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape_xyz
    expected = nx * ny * nz
    arr = np.fromfile(path, dtype="<f8")
    if arr.size != expected:
        raise ValueError(
            f"Headerless IMG voxel count mismatch: got {arr.size}, expected {expected}"
        )
    return arr.astype(np.float64, copy=False).reshape((nz, ny, nx))


def read_volume(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".mhd":
        return read_mhd(path).astype(np.float64, copy=False)
    if suffix == ".raw":
        return _read_legacy_raw(path)
    if suffix == ".img":
        return _read_headerless_img_f64(path, (108, 108, 80))
    raise ValueError(f"Unsupported volume format: {path}")


def compare(candidate: np.ndarray, reference: np.ndarray) -> dict:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: candidate={candidate.shape}, reference={reference.shape}"
        )
    if not np.all(np.isfinite(candidate)):
        raise ValueError("Candidate contains NaN or Inf")
    if not np.all(np.isfinite(reference)):
        raise ValueError("Reference contains NaN or Inf")

    diff = candidate - reference
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    relative_l2 = float(
        np.linalg.norm(diff.ravel()) / max(float(np.linalg.norm(reference.ravel())), 1e-12)
    )
    reference_mean = float(np.mean(reference))
    reference_sum = float(np.sum(reference))
    candidate_sum = float(np.sum(candidate))
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    p95_abs = float(np.percentile(abs_diff, 95.0))
    p99_abs = float(np.percentile(abs_diff, 99.0))
    rel_rmse_mean = float(rmse / max(abs(reference_mean), 1e-12))
    sum_rel_diff = float(abs(candidate_sum - reference_sum) / max(abs(reference_sum), 1e-12))
    return {
        "shape_zyx": list(candidate.shape),
        "candidate_sum": candidate_sum,
        "reference_sum": reference_sum,
        "reference_mean": reference_mean,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "p95_abs_diff": p95_abs,
        "p99_abs_diff": p99_abs,
        "rmse": rmse,
        "relative_l2": relative_l2,
        "rmse_over_reference_mean": rel_rmse_mean,
        "sum_rel_diff": sum_rel_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="New compact backend MHD")
    parser.add_argument(
        "--reference",
        required=True,
        help="Legacy CUDA reference MHD or legacy raw with 20-byte header",
    )
    parser.add_argument("--out-json", required=True, help="Output JSON report")
    parser.add_argument(
        "--max-rmse-over-mean",
        type=float,
        default=0.02,
        help="Acceptance threshold for rmse / reference mean",
    )
    parser.add_argument(
        "--max-sum-rel-diff",
        type=float,
        default=0.005,
        help="Acceptance threshold for absolute sum relative difference",
    )
    args = parser.parse_args()

    candidate = read_volume(Path(args.candidate))
    reference = read_volume(Path(args.reference))
    metrics = compare(candidate, reference)
    metrics["thresholds"] = {
        "max_rmse_over_reference_mean": float(args.max_rmse_over_mean),
        "max_sum_rel_diff": float(args.max_sum_rel_diff),
    }
    metrics["checks"] = {
        "rmse_over_reference_mean_ok": bool(
            metrics["rmse_over_reference_mean"] <= args.max_rmse_over_mean
        ),
        "sum_rel_diff_ok": bool(metrics["sum_rel_diff"] <= args.max_sum_rel_diff),
    }
    metrics["passed"] = bool(all(metrics["checks"].values()))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=True))
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
