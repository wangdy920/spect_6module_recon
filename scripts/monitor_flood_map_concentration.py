#!/usr/bin/env python3
"""Monitor 6-module SPECT flood-map concentration risk across acquisitions.

The tool scans visible acquisition roots, reads ``Bed*/Angle*/*.eng`` files,
reconstructs per-detector flood maps from the 12 ADC energy channels, and
writes metrics/figures that can be used to tune threshold ranges and guard
criteria.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


HEADER_BYTES = 80
RECORD_UINT32 = 16
RECORD_BYTES = RECORD_UINT32 * 4
EXPECTED_MAGIC = b"SPENG1\r\n"
EXPECTED_LINKS = (3, 4)
EXPECTED_FPGAS = range(6)
EXPECTED_DETECTORS = range(8)
BED_RE = re.compile(r"^Bed(\d+)$", re.IGNORECASE)
ANGLE_RE = re.compile(r"^Angle(\d+)$", re.IGNORECASE)
DATA_RE = re.compile(r"^data(\d+)$", re.IGNORECASE)


@dataclass
class Acquisition:
    label: str
    path: str
    top_root: str
    bed_count: int
    eng_file_count: int


@dataclass
class FileSummary:
    acquisition: str
    bed: str
    path: str
    bytes: int
    records: int
    invalid_rows: int
    trailing_bytes: int
    status: str
    error: str = ""


def numeric_suffix(path: Path, pattern: re.Pattern[str]) -> int:
    text = path.stem if path.is_file() else path.name
    match = pattern.match(text)
    return int(match.group(1)) if match else -1


def sanitize_label(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = [part for part in rel.parts if part not in {":", "\\"}]
    label = "__".join(parts)
    label = re.sub(r"[^0-9A-Za-z._-]+", "_", label)
    return label.strip("_") or re.sub(r"[^0-9A-Za-z._-]+", "_", path.name)


def detector_index(link: np.ndarray, fpga: np.ndarray, detector: np.ndarray) -> np.ndarray:
    return np.where(link == 4, 48, 0) + fpga * 8 + detector


def channel_key_from_index(index: int) -> tuple[int, int, int]:
    link = 4 if index >= 48 else 3
    local = index - 48 if link == 4 else index
    fpga = local // 8
    detector = local % 8
    return link, fpga, detector


def channel_key_string(index: int) -> str:
    link, fpga, detector = channel_key_from_index(index)
    return f"link{link}_fpga{fpga}_det{detector}"


def sorted_bed_dirs(acq: Path) -> list[Path]:
    return sorted(
        (p for p in acq.iterdir() if p.is_dir() and BED_RE.match(p.name)),
        key=lambda p: numeric_suffix(p, BED_RE),
    )


def sorted_angle_dirs(bed: Path) -> list[Path]:
    return sorted(
        (p for p in bed.iterdir() if p.is_dir() and ANGLE_RE.match(p.name)),
        key=lambda p: numeric_suffix(p, ANGLE_RE),
    )


def eng_files_for_bed(bed: Path) -> list[Path]:
    out: list[Path] = []
    for angle in sorted_angle_dirs(bed):
        out.extend(sorted(angle.glob("*.eng"), key=lambda p: numeric_suffix(p, DATA_RE)))
    return out


def count_eng_in_acquisition(acq: Path) -> tuple[int, int]:
    beds = sorted_bed_dirs(acq)
    count = sum(len(eng_files_for_bed(bed)) for bed in beds)
    return len(beds), count


def discover_acquisitions(root: Path, explicit: list[Path]) -> list[Acquisition]:
    candidates: list[Path] = []
    if explicit:
        candidates.extend(path for path in explicit if path.exists())
    else:
        skip = {
            "$RECYCLE.BIN",
            "System Volume Information",
            "WeGameApps",
            "Program Files (x86)",
            "Android",
            "HOMEPC",
            "LeiGod_Acc",
            "PET_agent_outputs",
            "三模态阅片软件更新部署",
            "团建照片",
            "小米手机照片",
            "scripts",
        }
        for top in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            if top.name in skip:
                continue
            if sorted_bed_dirs(top):
                candidates.append(top)
            for spect_dir in top.rglob("SPECT"):
                if spect_dir.is_dir():
                    candidates.append(spect_dir)

    acquisitions: list[Acquisition] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve()).lower()
        if resolved in seen:
            continue
        seen.add(resolved)
        bed_count, eng_count = count_eng_in_acquisition(candidate)
        if bed_count == 0 or eng_count == 0:
            continue
        acquisitions.append(
            Acquisition(
                label=sanitize_label(candidate, root),
                path=str(candidate),
                top_root=str(candidate.anchor or root),
                bed_count=bed_count,
                eng_file_count=eng_count,
            )
        )
    acquisitions.sort(key=lambda item: item.path.lower())
    return acquisitions


def read_eng(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    size = path.stat().st_size
    if size < HEADER_BYTES:
        return np.empty((0, RECORD_UINT32), dtype=np.uint32), {
            "status": "too_small",
            "records": 0,
            "trailing_bytes": 0,
            "error": "file smaller than ENG header",
        }
    with path.open("rb") as handle:
        header = handle.read(HEADER_BYTES)
        if header[:8] != EXPECTED_MAGIC:
            return np.empty((0, RECORD_UINT32), dtype=np.uint32), {
                "status": "bad_magic",
                "records": 0,
                "trailing_bytes": max(0, size - HEADER_BYTES),
                "error": "ENG magic mismatch",
            }
        field_count = struct.unpack_from("<I", header, 16)[0]
        record_size = struct.unpack_from("<I", header, 20)[0]
        header_records = struct.unpack_from("<Q", header, 24)[0]
        payload_bytes = max(0, size - HEADER_BYTES)
        full_records = payload_bytes // RECORD_BYTES
        trailing_bytes = payload_bytes % RECORD_BYTES
        if field_count != RECORD_UINT32 or record_size != RECORD_BYTES:
            return np.empty((0, RECORD_UINT32), dtype=np.uint32), {
                "status": "unsupported_layout",
                "records": 0,
                "trailing_bytes": int(trailing_bytes),
                "error": f"field_count={field_count}, record_size={record_size}",
            }
        count = int(min(header_records, full_records))
        if count == 0:
            return np.empty((0, RECORD_UINT32), dtype=np.uint32), {
                "status": "empty",
                "records": 0,
                "trailing_bytes": int(trailing_bytes),
                "error": "",
            }
        data = np.fromfile(handle, dtype="<u4", count=count * RECORD_UINT32)
    if data.size != count * RECORD_UINT32:
        count = data.size // RECORD_UINT32
        data = data[: count * RECORD_UINT32]
    return data.reshape(count, RECORD_UINT32), {
        "status": "ok",
        "records": int(count),
        "trailing_bytes": int(trailing_bytes),
        "error": "",
    }


class BedAccumulator:
    def __init__(self, thresholds: list[int], bins: int, flood_size: int) -> None:
        self.thresholds = thresholds
        self.bins = bins
        self.flood_size = flood_size
        self.hist = np.zeros((len(thresholds), 96, bins, bins), dtype=np.uint32)
        self.raw_counts = np.zeros(96, dtype=np.uint64)
        self.valid_counts = np.zeros((len(thresholds), 96), dtype=np.uint64)
        self.invalid_rows = 0
        self.records = 0

    def add_records(self, records: np.ndarray) -> int:
        if records.size == 0:
            return 0
        self.records += int(records.shape[0])
        fpgas = records[:, 12].astype(np.int16, copy=False)
        detectors = records[:, 13].astype(np.int16, copy=False)
        links = records[:, 15].astype(np.int16, copy=False)
        valid = (
            np.isin(links, EXPECTED_LINKS)
            & (fpgas >= 0)
            & (fpgas <= 5)
            & (detectors >= 0)
            & (detectors <= 7)
        )
        invalid = int((~valid).sum())
        self.invalid_rows += invalid
        if not np.any(valid):
            return invalid
        energies = records[valid, :12].astype(np.float32, copy=False)
        indices = detector_index(links[valid], fpgas[valid], detectors[valid]).astype(np.int16, copy=False)
        for idx in np.unique(indices):
            channel_events = energies[indices == idx]
            channel = int(idx)
            self.raw_counts[channel] += int(channel_events.shape[0])
            for t_index, threshold in enumerate(self.thresholds):
                y_bins, x_bins = flood_bins(channel_events, threshold, self.flood_size, self.bins)
                if x_bins.size == 0:
                    continue
                self.valid_counts[t_index, channel] += int(x_bins.size)
                np.add.at(self.hist[t_index, channel], (y_bins, x_bins), 1)
        return invalid


def flood_bins(events: np.ndarray, threshold: int, flood_size: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if events.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    data = events - float(threshold)
    np.maximum(data, 0, out=data)
    row = data[:, :6]
    col = data[:, 6:12]
    row_total = row.sum(axis=1)
    col_total = col.sum(axis=1)
    valid = np.isfinite(row_total) & np.isfinite(col_total) & (row_total > 0) & (col_total > 0)
    if not np.any(valid):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    weights = np.arange(6, dtype=np.float32)
    x = np.floor((col[valid] @ weights) / col_total[valid] / 5.0 * flood_size + 0.5).astype(np.int64)
    y = np.floor((row[valid] @ weights) / row_total[valid] / 5.0 * flood_size + 0.5).astype(np.int64)
    in_range = (x > 0) & (x <= flood_size) & (y > 0) & (y <= flood_size)
    if not np.any(in_range):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    x_bin = np.floor((x[in_range] - 1) / flood_size * bins).astype(np.int64)
    y_bin = np.floor((y[in_range] - 1) / flood_size * bins).astype(np.int64)
    return np.clip(y_bin, 0, bins - 1), np.clip(x_bin, 0, bins - 1)


def weighted_quantile_from_hist(weights: np.ndarray, q: float) -> int:
    total = int(weights.sum())
    if total <= 0:
        return -1
    target = total * q
    cdf = np.cumsum(weights)
    return int(np.searchsorted(cdf, target, side="left"))


def hist_metrics(
    hist: np.ndarray,
    raw_count: int,
    valid_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    total = int(hist.sum())
    if total == 0:
        return {
            "raw_events": int(raw_count),
            "valid_events": int(valid_count),
            "valid_fraction": 0.0,
            "center_fraction": 0.0,
            "edge_fraction": 0.0,
            "peak_bin_fraction": 0.0,
            "entropy_norm": 0.0,
            "effective_area_fraction": 0.0,
            "bbox_area_p01_p99": 0.0,
            "radial_mean": 0.0,
            "radial_p90": 0.0,
            "com_distance": 0.0,
            "quadrant_imbalance": 0.0,
            "risk_score": 0,
            "risk_level": "LOW_COUNT" if raw_count < args.min_valid_events else "NO_VALID",
            "risk_reasons": "no_valid_flood_positions",
        }

    bins = hist.shape[0]
    centers = (np.arange(bins, dtype=np.float64) + 0.5) / bins
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    center = (np.abs(xx - 0.5) <= args.center_halfwidth) & (np.abs(yy - 0.5) <= args.center_halfwidth)
    edge = (xx <= args.edge_margin) | (xx >= 1 - args.edge_margin) | (yy <= args.edge_margin) | (yy >= 1 - args.edge_margin)
    center_fraction = float(hist[center].sum() / total)
    edge_fraction = float(hist[edge].sum() / total)
    peak_bin_fraction = float(hist.max() / total)

    probabilities = hist[hist > 0].astype(np.float64) / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    entropy_norm = float(entropy / math.log(bins * bins))
    effective_area_fraction = float(math.exp(entropy) / (bins * bins))

    x_marg = hist.sum(axis=0)
    y_marg = hist.sum(axis=1)
    x01 = weighted_quantile_from_hist(x_marg, 0.01)
    x99 = weighted_quantile_from_hist(x_marg, 0.99)
    y01 = weighted_quantile_from_hist(y_marg, 0.01)
    y99 = weighted_quantile_from_hist(y_marg, 0.99)
    bbox_area = float(max(0, x99 - x01 + 1) * max(0, y99 - y01 + 1) / (bins * bins))

    radial = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    radial_mean = float((hist * radial).sum() / total)
    order = np.argsort(radial.ravel())
    radial_sorted = radial.ravel()[order]
    weights_sorted = hist.ravel()[order]
    cdf = np.cumsum(weights_sorted)
    radial_p90 = float(radial_sorted[int(np.searchsorted(cdf, total * 0.90, side="left"))])
    x_mean = float((hist * xx).sum() / total)
    y_mean = float((hist * yy).sum() / total)
    com_distance = float(math.sqrt((x_mean - 0.5) ** 2 + (y_mean - 0.5) ** 2))
    quadrants = [
        hist[(yy < 0.5) & (xx < 0.5)].sum(),
        hist[(yy < 0.5) & (xx >= 0.5)].sum(),
        hist[(yy >= 0.5) & (xx < 0.5)].sum(),
        hist[(yy >= 0.5) & (xx >= 0.5)].sum(),
    ]
    quadrant_fractions = [float(v / total) for v in quadrants]
    quadrant_imbalance = max(quadrant_fractions) - min(quadrant_fractions)

    reasons: list[str] = []
    risk_score = 0
    if valid_count < args.min_valid_events:
        reasons.append(f"valid_events<{args.min_valid_events}")
    if center_fraction >= args.center_fraction_warn:
        risk_score += 2
        reasons.append(f"center_fraction>={args.center_fraction_warn}")
    if edge_fraction <= args.edge_fraction_low:
        risk_score += 1
        reasons.append(f"edge_fraction<={args.edge_fraction_low}")
    if effective_area_fraction <= args.effective_area_low:
        risk_score += 1
        reasons.append(f"effective_area<={args.effective_area_low}")
    if bbox_area <= args.bbox_area_low:
        risk_score += 1
        reasons.append(f"bbox_area<={args.bbox_area_low}")
    if peak_bin_fraction >= args.peak_bin_fraction_high:
        risk_score += 1
        reasons.append(f"peak_bin_fraction>={args.peak_bin_fraction_high}")
    if radial_p90 <= args.radial_p90_low:
        risk_score += 1
        reasons.append(f"radial_p90<={args.radial_p90_low}")

    if valid_count < args.min_valid_events:
        risk_level = "LOW_COUNT"
    elif risk_score >= args.high_risk_score:
        risk_level = "HIGH"
    elif risk_score >= args.warn_risk_score:
        risk_level = "WARN"
    else:
        risk_level = "OK"

    return {
        "raw_events": int(raw_count),
        "valid_events": int(valid_count),
        "valid_fraction": float(valid_count / max(raw_count, 1)),
        "center_fraction": center_fraction,
        "edge_fraction": edge_fraction,
        "peak_bin_fraction": peak_bin_fraction,
        "entropy_norm": entropy_norm,
        "effective_area_fraction": effective_area_fraction,
        "bbox_area_p01_p99": bbox_area,
        "radial_mean": radial_mean,
        "radial_p90": radial_p90,
        "com_distance": com_distance,
        "quadrant_imbalance": quadrant_imbalance,
        "risk_score": int(risk_score),
        "risk_level": risk_level,
        "risk_reasons": ";".join(reasons) if reasons else "",
    }


def analyze_bed(
    acquisition: Acquisition,
    bed: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[FileSummary], BedAccumulator]:
    files = eng_files_for_bed(bed)
    if args.max_files_per_bed is not None:
        files = files[: args.max_files_per_bed]
    acc = BedAccumulator(args.thresholds, args.hist_bins, args.flood_size)
    file_rows: list[FileSummary] = []
    for path in files:
        try:
            records, info = read_eng(path)
            invalid = acc.add_records(records)
            file_rows.append(
                FileSummary(
                    acquisition=acquisition.label,
                    bed=bed.name,
                    path=str(path),
                    bytes=path.stat().st_size,
                    records=int(info["records"]),
                    invalid_rows=int(invalid),
                    trailing_bytes=int(info["trailing_bytes"]),
                    status=str(info["status"]),
                    error=str(info["error"]),
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve batch audit.
            file_rows.append(
                FileSummary(
                    acquisition=acquisition.label,
                    bed=bed.name,
                    path=str(path),
                    bytes=path.stat().st_size if path.exists() else 0,
                    records=0,
                    invalid_rows=0,
                    trailing_bytes=0,
                    status="error",
                    error=str(exc),
                )
            )

    metric_rows: list[dict[str, Any]] = []
    for t_index, threshold in enumerate(args.thresholds):
        for channel in range(96):
            link, fpga, detector = channel_key_from_index(channel)
            metrics = hist_metrics(
                acc.hist[t_index, channel],
                int(acc.raw_counts[channel]),
                int(acc.valid_counts[t_index, channel]),
                args,
            )
            metric_rows.append(
                {
                    "acquisition": acquisition.label,
                    "acquisition_path": acquisition.path,
                    "bed": bed.name,
                    "threshold": threshold,
                    "detector_index": channel,
                    "channel": channel_key_string(channel),
                    "link": link,
                    "fpga": fpga,
                    "detector": detector,
                    **metrics,
                }
            )
    return metric_rows, file_rows, acc


def summarize_beds(metric_rows: list[dict[str, Any]], primary_threshold: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in metric_rows:
        key = (row["acquisition"], row["bed"], int(row["threshold"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (acq, bed, threshold), rows in sorted(groups.items()):
        levels = {level: sum(1 for row in rows if row["risk_level"] == level) for level in ["HIGH", "WARN", "OK", "LOW_COUNT", "NO_VALID"]}
        valid_rows = [row for row in rows if row["risk_level"] not in {"LOW_COUNT", "NO_VALID"}]
        out.append(
            {
                "acquisition": acq,
                "bed": bed,
                "threshold": threshold,
                "is_primary_threshold": threshold == primary_threshold,
                "channels": len(rows),
                **{f"{key.lower()}_channels": value for key, value in levels.items()},
                "median_center_fraction": median_or_blank(row["center_fraction"] for row in valid_rows),
                "median_edge_fraction": median_or_blank(row["edge_fraction"] for row in valid_rows),
                "median_effective_area": median_or_blank(row["effective_area_fraction"] for row in valid_rows),
                "max_risk_score": max((int(row["risk_score"]) for row in rows), default=0),
                "total_raw_events": int(sum(int(row["raw_events"]) for row in rows)),
                "total_valid_events": int(sum(int(row["valid_events"]) for row in rows)),
            }
        )
    return out


def median_or_blank(values: Any) -> float | str:
    data = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not data:
        return ""
    mid = len(data) // 2
    if len(data) % 2:
        return data[mid]
    return (data[mid - 1] + data[mid]) / 2.0


def summarize_acquisitions(bed_summary: list[dict[str, Any]], primary_threshold: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in bed_summary:
        if int(row["threshold"]) == primary_threshold:
            grouped.setdefault(row["acquisition"], []).append(row)
    out: list[dict[str, Any]] = []
    for acq, rows in sorted(grouped.items()):
        out.append(
            {
                "acquisition": acq,
                "beds": len(rows),
                "high_channels": int(sum(int(row["high_channels"]) for row in rows)),
                "warn_channels": int(sum(int(row["warn_channels"]) for row in rows)),
                "low_count_channels": int(sum(int(row["low_count_channels"]) for row in rows)),
                "max_risk_score": max((int(row["max_risk_score"]) for row in rows), default=0),
                "total_raw_events": int(sum(int(row["total_raw_events"]) for row in rows)),
                "total_valid_events": int(sum(int(row["total_valid_events"]) for row in rows)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def risk_heatmap(acc: BedAccumulator, acquisition: str, bed: str, metric_rows: list[dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> None:
    rows = [row for row in metric_rows if row["acquisition"] == acquisition and row["bed"] == bed and int(row["threshold"]) == args.primary_threshold]
    matrix = np.full((12, 8), np.nan, dtype=np.float64)
    for row in rows:
        link_row = 0 if int(row["link"]) == 3 else 6
        matrix[link_row + int(row["fpga"]), int(row["detector"])] = float(row["risk_score"])
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(matrix, vmin=0, vmax=max(args.high_risk_score + 2, 6), cmap="magma")
    ax.set_title(f"{acquisition} {bed} risk score @ threshold {args.primary_threshold}")
    ax.set_xlabel("detector")
    ax.set_ylabel("link/fpga row: link3 fpga0-5, link4 fpga0-5")
    ax.set_xticks(range(8))
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"L3F{i}" for i in range(6)] + [f"L4F{i}" for i in range(6)])
    fig.colorbar(im, ax=ax, label="risk score")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{acquisition}__{bed}__risk_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def flood_montage(acc: BedAccumulator, acquisition: str, bed: str, args: argparse.Namespace, out_dir: Path) -> None:
    if args.primary_threshold not in args.thresholds:
        return
    t_index = args.thresholds.index(args.primary_threshold)
    fig, axes = plt.subplots(12, 8, figsize=(12, 16), dpi=140)
    for channel in range(96):
        link, fpga, detector = channel_key_from_index(channel)
        row = (0 if link == 3 else 6) + fpga
        ax = axes[row, detector]
        image = np.log1p(acc.hist[t_index, channel].astype(np.float64))
        ax.imshow(image, cmap="gray", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"L{link}F{fpga}D{detector}", fontsize=5)
    fig.suptitle(f"{acquisition} {bed} flood maps @ threshold {args.primary_threshold}", fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{acquisition}__{bed}__flood_montage.png", bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    acquisitions: list[Acquisition],
    args: argparse.Namespace,
    dataset_summary: list[dict[str, Any]],
    bed_summary: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# F-drive SPECT Flood Map Concentration Monitor",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- input_root: `{args.root}`",
        f"- output_dir: `{args.output_dir}`",
        f"- acquisitions: `{len(acquisitions)}`",
        f"- thresholds: `{','.join(str(v) for v in args.thresholds)}`",
        f"- primary_threshold: `{args.primary_threshold}`",
        f"- flood_size: `{args.flood_size}`; histogram bins: `{args.hist_bins}`",
        "",
        "## Method",
        "",
        "- ENG records are read as a 80-byte header followed by 16 uint32 fields.",
        "- Fields 0..11 are interpreted as the 12 energy channels.",
        "- Flood position follows the existing MATLAB/Python centroid rule: subtract threshold, clamp negatives to zero, use channels 0..5 as row and 6..11 as column.",
        "- Concentration risk is not a production decision by itself; it is a monitoring signal for threshold-range and guard tuning.",
        "",
        "## Guard Metrics",
        "",
        f"- center_fraction_warn: `{args.center_fraction_warn}` inside half-width `{args.center_halfwidth}`.",
        f"- edge_fraction_low: `{args.edge_fraction_low}` with edge margin `{args.edge_margin}`.",
        f"- effective_area_low: `{args.effective_area_low}`.",
        f"- bbox_area_low: `{args.bbox_area_low}`.",
        f"- peak_bin_fraction_high: `{args.peak_bin_fraction_high}`.",
        f"- radial_p90_low: `{args.radial_p90_low}`.",
        f"- WARN score >= `{args.warn_risk_score}`, HIGH score >= `{args.high_risk_score}`.",
        "",
        "## Acquisition Summary",
        "",
        "| acquisition | beds | high | warn | low_count | raw_events | valid_events | max_score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_summary:
        lines.append(
            f"| {row['acquisition']} | {row['beds']} | {row['high_channels']} | {row['warn_channels']} | {row['low_count_channels']} | {row['total_raw_events']} | {row['total_valid_events']} | {row['max_risk_score']} |"
        )
    lines.extend(
        [
            "",
            "## Highest-Risk Channels",
            "",
            "| acquisition | bed | threshold | detector | channel | risk | score | center | edge | eff_area | peak_bin | reasons |",
            "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in top_rows[:30]:
        lines.append(
            "| {acquisition} | {bed} | {threshold} | {detector_index} | {channel} | {risk_level} | {risk_score} | {center_fraction:.4g} | {edge_fraction:.4g} | {effective_area_fraction:.4g} | {peak_bin_fraction:.4g} | {risk_reasons} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Tuning Plan",
            "",
            "1. Treat `HIGH` channels as candidates for closer visual review in `figures/*flood_montage.png`.",
            "2. Compare the same channel across thresholds in `channel_metrics.csv`; if concentration grows monotonically with threshold, tighten the allowed threshold upper range for that detector.",
            "3. If `LOW_COUNT` dominates, collect more events or lower `--min-valid-events` only for exploratory review; do not promote those rows to production guards.",
            "4. For default switching, join these flood-collapse metrics with peak/FWHM/acceptance guards; do not use object-derived response evidence alone as a production response map.",
            "",
            "## Files",
            "",
            "- `manifest_acquisitions.csv`",
            "- `file_summary.csv`",
            "- `channel_metrics.csv`",
            "- `bed_summary.csv`",
            "- `dataset_summary.csv`",
            "- `top_risk_channels.csv`",
            "- `figures/*risk_heatmap.png`",
            "- `figures/*flood_montage.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_thresholds(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one threshold is required")
    return sorted(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("F:/"))
    parser.add_argument("--acquisition", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("recon_out/flood_map_monitor_F_20260608"))
    parser.add_argument("--thresholds", default="0,20,40,60,80,100,120")
    parser.add_argument("--primary-threshold", type=int, default=20)
    parser.add_argument("--flood-size", type=int, default=512)
    parser.add_argument("--hist-bins", type=int, default=64)
    parser.add_argument("--max-files-per-bed", type=int, default=None)
    parser.add_argument("--min-valid-events", type=int, default=1000)
    parser.add_argument("--center-halfwidth", type=float, default=0.18)
    parser.add_argument("--edge-margin", type=float, default=0.10)
    parser.add_argument("--center-fraction-warn", type=float, default=0.55)
    parser.add_argument("--edge-fraction-low", type=float, default=0.04)
    parser.add_argument("--effective-area-low", type=float, default=0.08)
    parser.add_argument("--bbox-area-low", type=float, default=0.10)
    parser.add_argument("--peak-bin-fraction-high", type=float, default=0.12)
    parser.add_argument("--radial-p90-low", type=float, default=0.28)
    parser.add_argument("--warn-risk-score", type=int, default=2)
    parser.add_argument("--high-risk-score", type=int, default=4)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    args.thresholds = parse_thresholds(args.thresholds)
    if args.primary_threshold not in args.thresholds:
        args.thresholds = sorted(args.thresholds + [args.primary_threshold])
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    acquisitions = discover_acquisitions(args.root, args.acquisition)
    write_csv(args.output_dir / "manifest_acquisitions.csv", [asdict(item) for item in acquisitions])
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    all_metrics: list[dict[str, Any]] = []
    all_files: list[dict[str, Any]] = []
    for acquisition in acquisitions:
        acq_path = Path(acquisition.path)
        for bed in sorted_bed_dirs(acq_path):
            print(f"[flood-monitor] {acquisition.label} {bed.name}", flush=True)
            metric_rows, file_rows, acc = analyze_bed(acquisition, bed, args)
            all_metrics.extend(metric_rows)
            all_files.extend(asdict(row) for row in file_rows)
            if not args.skip_figures:
                risk_heatmap(acc, acquisition.label, bed.name, metric_rows, args, figures_dir)
                flood_montage(acc, acquisition.label, bed.name, args, figures_dir)

    bed_summary = summarize_beds(all_metrics, args.primary_threshold)
    dataset_summary = summarize_acquisitions(bed_summary, args.primary_threshold)
    top_rows = sorted(
        (
            row
            for row in all_metrics
            if int(row["threshold"]) == args.primary_threshold and row["risk_level"] in {"HIGH", "WARN"}
        ),
        key=lambda row: (
            int(row["risk_score"]),
            float(row["center_fraction"]),
            float(row["peak_bin_fraction"]),
            int(row["valid_events"]),
        ),
        reverse=True,
    )
    write_csv(args.output_dir / "file_summary.csv", all_files)
    write_csv(args.output_dir / "channel_metrics.csv", all_metrics)
    write_csv(args.output_dir / "bed_summary.csv", bed_summary)
    write_csv(args.output_dir / "dataset_summary.csv", dataset_summary)
    write_csv(args.output_dir / "top_risk_channels.csv", top_rows)
    write_report(args.output_dir / "REPORT.md", acquisitions, args, dataset_summary, bed_summary, top_rows)
    print(f"[flood-monitor] wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
