#!/usr/bin/env python3
"""Convert Bed*/Angle*/data*.eng files to 6-module projection BIN."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

HEADER_BYTES = 80
RECORD_UINT32 = 16
RECORD_BYTES = RECORD_UINT32 * 4
EXPECTED_MAGIC = b"SPENG1\r\n"
ANGLE_RE = re.compile(r"^Angle(\d+)$", re.IGNORECASE)
DATA_RE = re.compile(r"^data(\d+)$", re.IGNORECASE)


def numeric_suffix(path: Path, pattern: re.Pattern[str]) -> int:
    text = path.stem if path.is_file() else path.name
    match = pattern.match(text)
    return int(match.group(1)) if match else -1


def read_eng(path: Path) -> tuple[np.ndarray, dict]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES or header[:8] != EXPECTED_MAGIC:
            return np.empty((0, RECORD_UINT32), dtype=np.uint32), {
                "status": "bad_magic",
                "records": 0,
            }
        field_count = struct.unpack_from("<I", header, 16)[0]
        record_size = struct.unpack_from("<I", header, 20)[0]
        header_records = struct.unpack_from("<Q", header, 24)[0]
        if field_count != RECORD_UINT32 or record_size != RECORD_BYTES:
            return np.empty((0, RECORD_UINT32), dtype=np.uint32), {
                "status": "unsupported_layout",
                "records": 0,
            }
        full_records = max(0, size - HEADER_BYTES) // RECORD_BYTES
        count = int(min(header_records, full_records))
        data = np.fromfile(handle, dtype="<u4", count=count * RECORD_UINT32)
    return data.reshape((-1, RECORD_UINT32)), {"status": "ok", "records": int(count)}


def add_records(proj: np.ndarray, physical_angle: int, records: np.ndarray, threshold: int, xflip: bool) -> int:
    if records.size == 0:
        return 0
    fpgas = records[:, 12].astype(np.int16, copy=False)
    detectors = records[:, 13].astype(np.int16, copy=False)
    links = records[:, 15].astype(np.int16, copy=False)
    valid = (
        ((links == 3) | (links == 4))
        & (fpgas >= 0)
        & (fpgas < 6)
        & (detectors >= 0)
        & (detectors < 8)
    )
    if not np.any(valid):
        return 0

    energies = records[valid, :12].astype(np.float32, copy=False) - float(threshold)
    energies = np.maximum(energies, 0.0)
    row_total = energies[:, :6].sum(axis=1)
    col_total = energies[:, 6:12].sum(axis=1)
    good = (row_total > 0.0) & (col_total > 0.0)
    if not np.any(good):
        return 0

    weights = np.arange(6, dtype=np.float32)
    local_y = np.floor((energies[good, :6] @ weights) / row_total[good] / 5.0 * 12.0).astype(np.int16)
    local_x = np.floor((energies[good, 6:12] @ weights) / col_total[good] / 5.0 * 12.0).astype(np.int16)
    good2 = (local_x >= 0) & (local_x < 12) & (local_y >= 0) & (local_y < 12)
    if not np.any(good2):
        return 0

    module = fpgas[valid][good][good2].astype(np.int16, copy=False)
    local_detector = (np.where(links[valid][good][good2] == 4, 8, 0) + detectors[valid][good][good2]).astype(
        np.int16, copy=False
    )
    block_x = local_detector % 4
    block_y = local_detector // 4
    x = block_x * 12 + local_x[good2]
    y = block_y * 12 + local_y[good2]
    if xflip:
        x = 47 - x
    full_angle = physical_angle + module * 60
    ray = y * 48 + x
    np.add.at(proj, (full_angle, ray), 1.0)
    return int(ray.size)


def convert_bed(bed_dir: Path, out_bin: Path, threshold: int, xflip: bool) -> dict:
    proj = np.zeros((360, 2304), dtype=np.float64)
    angle_dirs = sorted((p for p in bed_dir.iterdir() if p.is_dir() and ANGLE_RE.match(p.name)), key=lambda p: numeric_suffix(p, ANGLE_RE))
    selected = 0
    accepted = 0
    bad_files = 0
    for angle_dir in angle_dirs:
        angle = numeric_suffix(angle_dir, ANGLE_RE)
        if angle < 0 or angle >= 60:
            continue
        for path in sorted(angle_dir.glob("*.eng"), key=lambda p: numeric_suffix(p, DATA_RE)):
            records, info = read_eng(path)
            selected += 1
            if info["status"] != "ok":
                bad_files += 1
                continue
            accepted += add_records(proj, angle, records, threshold, xflip)

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with out_bin.open("wb") as f:
        f.write(struct.pack("<ii", 360, 2304))
        proj.tofile(f)
    return {
        "bed_dir": str(bed_dir),
        "output_bin": str(out_bin),
        "num_angles": 360,
        "rays_per_angle": 2304,
        "source_eng_files_selected": selected,
        "bad_files": bad_files,
        "accepted_events": accepted,
        "total_counts": float(proj.sum()),
        "nonzero_bins": int(np.count_nonzero(proj)),
        "threshold": threshold,
        "projection_variant": "xflip" if xflip else "none",
        "angle_mapping": "full_angle = physical_angle + fpga * 60",
        "ray_mapping": "16 detector blocks of 12x12 flood bins -> 48x48",
    }


def self_test() -> int:
    proj = np.zeros((360, 2304), dtype=np.float64)
    rec = np.zeros((1, RECORD_UINT32), dtype=np.uint32)
    rec[0, :6] = [0, 0, 100, 0, 0, 0]
    rec[0, 6:12] = [0, 0, 0, 100, 0, 0]
    rec[0, 12] = 2
    rec[0, 13] = 5
    rec[0, 15] = 3
    added = add_records(proj, 7, rec, threshold=0, xflip=True)
    assert added == 1
    assert proj.sum() == 1.0
    assert np.count_nonzero(proj) == 1
    assert np.nonzero(proj)[0][0] == 127
    print("self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bed-dir", type=Path, required=False)
    parser.add_argument("--out-bin", type=Path, required=False)
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--no-xflip", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.bed_dir is None or args.out_bin is None:
        parser.error("--bed-dir and --out-bin are required unless --self-test is used")
    summary = convert_bed(args.bed_dir, args.out_bin, args.threshold, not args.no_xflip)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
