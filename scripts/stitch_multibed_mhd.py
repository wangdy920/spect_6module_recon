#!/usr/bin/env python3
"""Register and blend independently reconstructed SPECT bed volumes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from eval_mhd_stats import _read_mhd_header, read_mhd


def _spacing(path: Path) -> tuple[float, float, float]:
    header = _read_mhd_header(path)
    values = (header.get("ElementSpacing") or "1 1 1").split()
    if len(values) != 3:
        raise ValueError(f"Invalid ElementSpacing in {path}")
    return tuple(float(value) for value in values)


def _write_mhd(path: Path, image: np.ndarray, spacing: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_suffix(".raw")
    np.asarray(image, dtype="<f8").tofile(raw_path)
    nz, ny, nx = image.shape
    path.write_text(
        "\n".join(
            (
                "ObjectType = Image",
                "NDims = 3",
                f"DimSize = {nx} {ny} {nz}",
                f"ElementSpacing = {spacing[0]:g} {spacing[1]:g} {spacing[2]:g}",
                "ElementType = MET_DOUBLE",
                "ElementByteOrderMSB = False",
                "HeaderSize = 0",
                f"ElementDataFile = {raw_path.name}",
                "",
            )
        ),
        encoding="ascii",
    )


def _mask_xy(image: np.ndarray, spacing: tuple[float, float, float], diameter_mm: float) -> None:
    if diameter_mm <= 0:
        return
    _, ny, nx = image.shape
    yy, xx = np.ogrid[:ny, :nx]
    radius2 = (diameter_mm * 0.5) ** 2
    outside = (
        ((xx - (nx - 1) * 0.5) * spacing[0]) ** 2
        + ((yy - (ny - 1) * 0.5) * spacing[1]) ** 2
    ) > radius2
    image[:, outside] = 0.0


def _pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=np.float64).ravel()
    b = np.asarray(rhs, dtype=np.float64).ravel()
    active = (a > 0) | (b > 0)
    if np.count_nonzero(active) < 32:
        return -1.0
    a = a[active]
    b = b[active]
    a -= np.mean(a)
    b -= np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


def score_overlap(lhs: np.ndarray, rhs: np.ndarray, overlap: int, edge_ignore: int) -> float:
    if overlap <= 2 * edge_ignore or overlap >= min(lhs.shape[0], rhs.shape[0]):
        return -1.0
    a = lhs[-overlap + edge_ignore : -edge_ignore or None]
    b = rhs[edge_ignore : overlap - edge_ignore]
    return _pearson(a, b)


def find_overlaps(
    beds: list[np.ndarray], minimum: int, maximum: int, edge_ignore: int
) -> tuple[list[int], list[dict[str, float | int]]]:
    overlaps: list[int] = []
    details: list[dict[str, float | int]] = []
    for joint, (lhs, rhs) in enumerate(zip(beds, beds[1:]), start=1):
        scores = [
            (overlap, score_overlap(lhs, rhs, overlap, edge_ignore))
            for overlap in range(minimum, maximum + 1)
        ]
        overlap, score = max(scores, key=lambda item: item[1])
        overlaps.append(overlap)
        details.append({"joint": joint, "overlap": overlap, "ncc": score})
    return overlaps, details


def find_shared_overlap(
    beds: list[np.ndarray], minimum: int, maximum: int, edge_ignore: int
) -> tuple[int, list[dict[str, float | int]], list[dict[str, float | int]]]:
    candidates: list[tuple[float, int, list[float]]] = []
    for overlap in range(minimum, maximum + 1):
        scores = [
            score_overlap(lhs, rhs, overlap, edge_ignore)
            for lhs, rhs in zip(beds, beds[1:])
        ]
        candidates.append((float(np.mean(scores)), overlap, scores))
    mean_score, overlap, scores = max(candidates, key=lambda item: item[0])
    details = [
        {"joint": joint, "overlap": overlap, "ncc": score}
        for joint, score in enumerate(scores, start=1)
    ]
    ranking = [
        {"overlap": candidate_overlap, "mean_ncc": candidate_mean}
        for candidate_mean, candidate_overlap, _ in sorted(candidates, reverse=True)[:5]
    ]
    return overlap, details, ranking


def stitch(beds: list[np.ndarray], overlaps: list[int]) -> np.ndarray:
    out = beds[0].copy()
    for bed, overlap in zip(beds[1:], overlaps):
        if overlap <= 0:
            out = np.concatenate((out, bed), axis=0)
            continue
        if overlap >= min(out.shape[0], bed.shape[0]):
            raise ValueError(f"Invalid overlap {overlap}")
        guard = min(overlap // 4, max(0, (overlap - 1) // 2))
        next_weight = np.zeros(overlap, dtype=np.float64)
        transition = overlap - 2 * guard
        phase = np.arange(1, transition + 1, dtype=np.float64) / (transition + 1)
        next_weight[guard : overlap - guard] = 0.5 - 0.5 * np.cos(math.pi * phase)
        if guard:
            next_weight[overlap - guard :] = 1.0
        shape = (overlap,) + (1,) * (out.ndim - 1)
        next_weight = next_weight.reshape(shape)
        blended = out[-overlap:] * (1.0 - next_weight) + bed[:overlap] * next_weight
        out = np.concatenate((out[:-overlap], blended, bed[overlap:]), axis=0)
    return out


def _parse_overlap_range(raw: str) -> tuple[int, int]:
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError("--overlap-range expects min:max")
    minimum, maximum = (int(value) for value in parts)
    if minimum < 1 or maximum < minimum:
        raise ValueError("Invalid --overlap-range")
    return minimum, maximum


def _run_self_test() -> None:
    rng = np.random.default_rng(7)
    source = rng.random((42, 8, 8))
    beds = [source[:24], source[18:]]
    overlap, _, _ = find_shared_overlap(beds, 4, 10, 1)
    overlaps = [overlap]
    assert overlap == 6, overlap
    merged = stitch(beds, overlaps)
    assert merged.shape == source.shape
    assert np.allclose(merged, source)
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--overlap-range", default="8:28")
    parser.add_argument("--overlap", default="auto", help="Shared overlap in voxels, or auto")
    parser.add_argument("--edge-ignore", type=int, default=4)
    parser.add_argument("--z-flip", choices=("0", "1", "auto"), default="auto")
    parser.add_argument("--fov-diameter-mm", type=float, default=30.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return 0
    if not args.inputs or not args.output:
        parser.error("--inputs and --output are required")

    spacing = _spacing(args.inputs[0])
    beds = [read_mhd(path) for path in args.inputs]
    if len({bed.shape for bed in beds}) != 1:
        raise ValueError("All bed volumes must have the same shape")
    for bed in beds:
        _mask_xy(bed, spacing, args.fov_diameter_mm)

    minimum, maximum = _parse_overlap_range(args.overlap_range)
    orientations = (False, True) if args.z_flip == "auto" else (args.z_flip == "1",)
    candidates = []
    for flip in orientations:
        oriented = [bed[::-1].copy() if flip else bed for bed in beds]
        if args.overlap == "auto":
            overlap, details, ranking = find_shared_overlap(
                oriented, minimum, maximum, args.edge_ignore
            )
        else:
            overlap = int(args.overlap)
            details = [
                {
                    "joint": joint,
                    "overlap": overlap,
                    "ncc": score_overlap(lhs, rhs, overlap, args.edge_ignore),
                }
                for joint, (lhs, rhs) in enumerate(zip(oriented, oriented[1:]), start=1)
            ]
            ranking = []
        mean_ncc = float(np.mean([item["ncc"] for item in details]))
        candidates.append((mean_ncc, flip, oriented, [overlap] * (len(oriented) - 1), details, ranking))
    mean_ncc, flip, beds, overlaps, details, ranking = max(candidates, key=lambda item: item[0])

    merged = stitch(beds, overlaps)
    reference = np.concatenate(beds, axis=0)
    _write_mhd(args.output, merged, spacing)
    if args.reference_output:
        _write_mhd(args.reference_output, reference, spacing)

    report = {
        "inputs": [str(path) for path in args.inputs],
        "output": str(args.output),
        "reference_output": str(args.reference_output) if args.reference_output else "",
        "z_flip": int(flip),
        "overlaps": overlaps,
        "overlap_mm": [value * spacing[2] for value in overlaps],
        "joint_scores": details,
        "overlap_ranking": ranking,
        "mean_ncc": mean_ncc,
        "input_shape_zyx": list(beds[0].shape),
        "reference_shape_zyx": list(reference.shape),
        "output_shape_zyx": list(merged.shape),
        "spacing_xyz_mm": list(spacing),
        "fov_diameter_mm": args.fov_diameter_mm,
        "blend": "guarded-raised-cosine",
        "blend_edge_guard_voxels": overlaps[0] // 4 if overlaps else 0,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
