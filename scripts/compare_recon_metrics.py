#!/usr/bin/env python3
"""Compare two SPECT MHD/RAW reconstructions with scalar similarity metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TYPE_TO_DTYPE = {
    "MET_DOUBLE": np.dtype("<f8"),
    "MET_FLOAT": np.dtype("<f4"),
}


def read_mhd(path: Path) -> tuple[dict[str, str], np.ndarray]:
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    nx, ny, nz = [int(v) for v in meta["DimSize"].split()]
    dtype = TYPE_TO_DTYPE[meta["ElementType"]]
    raw_path = path.parent / meta["ElementDataFile"]
    data = np.fromfile(raw_path, dtype=dtype)
    expected = nx * ny * nz
    if data.size != expected:
        raise ValueError(f"{raw_path} has {data.size} voxels, expected {expected}")
    return meta, data.reshape((nz, ny, nx)).astype(np.float64, copy=False)


def center_crop_to_shape(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    slices = []
    for axis, target in enumerate(shape):
        if target > image.shape[axis]:
            raise ValueError(f"target shape {shape} exceeds image shape {image.shape}")
        start = (image.shape[axis] - target) // 2
        slices.append(slice(start, start + target))
    return image[tuple(slices)]


def stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return {
            "count": int(values.size),
            "finite": 0,
            "finite_ratio": 0.0,
        }
    positive = finite_values[finite_values > 0]
    return {
        "count": int(values.size),
        "finite": int(finite.sum()),
        "finite_ratio": float(finite.sum() / values.size) if values.size else 0.0,
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
        "sum": float(finite_values.sum()),
        "positive_mean": float(positive.mean()) if positive.size else 0.0,
        "positive_sum": float(positive.sum()) if positive.size else 0.0,
    }


def compute_metrics(candidate: np.ndarray, reference: np.ndarray, positive_union: bool) -> dict:
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: candidate={candidate.shape}, reference={reference.shape}")
    mask = np.isfinite(candidate) & np.isfinite(reference)
    if positive_union:
        mask &= (candidate > 0) | (reference > 0)
    a = candidate[mask].astype(np.float64, copy=False)
    b = reference[mask].astype(np.float64, copy=False)
    if a.size == 0:
        raise ValueError("no valid voxels after masking")

    diff = a - b
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_norm = float(np.linalg.norm(b))
    rel_l2 = float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else None
    ref_mean = float(np.mean(b))
    ref_pos = b[b > 0]
    ref_pos_mean = float(ref_pos.mean()) if ref_pos.size else 0.0
    rmse_over_ref_positive_mean = (
        float(rmse / ref_pos_mean) if ref_pos_mean > 0 else None
    )
    sum_ref = float(np.sum(b))
    sum_rel_diff = float(abs(np.sum(a) - sum_ref) / abs(sum_ref)) if sum_ref else None

    a0 = a - float(a.mean())
    b0 = b - float(b.mean())
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    ncc = float(np.dot(a0, b0) / denom) if denom > 0 else None
    cosine_denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / cosine_denom) if cosine_denom > 0 else None

    return {
        "shape_zyx": list(candidate.shape),
        "mask_voxels": int(a.size),
        "ncc": ncc,
        "cosine": cosine,
        "relative_l2": rel_l2,
        "rmse": rmse,
        "reference_mean": ref_mean,
        "reference_positive_mean": ref_pos_mean,
        "rmse_over_reference_positive_mean": rmse_over_ref_positive_mean,
        "sum_relative_difference": sum_rel_diff,
        "candidate": stats(candidate),
        "reference": stats(reference),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--positive-union", action="store_true")
    parser.add_argument("--crop-common-shape", action="store_true")
    args = parser.parse_args()

    cand_meta, candidate = read_mhd(args.candidate)
    ref_meta, reference = read_mhd(args.reference)
    if candidate.shape != reference.shape:
        if not args.crop_common_shape:
            raise ValueError(
                f"shape mismatch: candidate={candidate.shape}, reference={reference.shape}; "
                "use --crop-common-shape to center-crop both"
            )
        common = tuple(min(a, b) for a, b in zip(candidate.shape, reference.shape))
        candidate = center_crop_to_shape(candidate, common)
        reference = center_crop_to_shape(reference, common)

    result = {
        "candidate_path": str(args.candidate),
        "reference_path": str(args.reference),
        "candidate_dim_size": cand_meta.get("DimSize"),
        "reference_dim_size": ref_meta.get("DimSize"),
        "positive_union": bool(args.positive_union),
        "metrics": compute_metrics(candidate, reference, args.positive_union),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"wrote {args.out_json} "
        f"ncc={result['metrics']['ncc']} "
        f"relative_l2={result['metrics']['relative_l2']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

