#!/usr/bin/env python3
"""Validate the fixed 6module compact-symmetric input files."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


PROFILES: dict[str, dict[str, Any]] = {
    "260525": {
        "matrix_rows": 52992,
        "matrix_cols": 933120,
        "matrix_nnz": 470220768,
        "projection_num_angles": 360,
        "projection_rays_per_angle": 2304,
        "projection_full_rows": 829440,
        "compact_num_angles": 46,
        "det_nx": 48,
        "det_nz": 48,
        "compact_det_z": 24,
        "compact_rays_per_angle": 1152,
        "image_nx": 108,
        "image_ny": 108,
        "image_nz": 80,
        "total_voxels": 933120,
        "metadata_mode": "45z",
    },
    "260601": {
        "matrix_rows": 103680,
        "matrix_cols": 933120,
        # TODO: 260601 ships as metadata.bin + chunk_*.csrb (4 chunks); directory total nnz=2436868849.
        # No single merged CSRB baseline exists for this profile, so matrix_nnz is not checked.
        "matrix_nnz": None,
        "projection_num_angles": 360,
        "projection_rays_per_angle": 2304,
        "projection_full_rows": 829440,
        "compact_num_angles": 90,
        "det_nx": 48,
        "det_nz": 48,
        "compact_det_z": 24,
        "compact_rays_per_angle": 1152,
        "image_nx": 108,
        "image_ny": 108,
        "image_nz": 80,
        "total_voxels": 933120,
        "metadata_mode": "90z",
    },
}


def resolve_expected(profile: str, overrides: dict[str, Any | None]) -> dict[str, Any]:
    expected = dict(PROFILES[profile])
    for key, value in overrides.items():
        if value is not None:
            expected[key] = value
    return expected


def read_csrb(path: Path) -> dict:
    file_size = path.stat().st_size
    if file_size < 24:
        raise ValueError(f"CSRB file is too small: {path}")
    with path.open("rb") as f:
        header = f.read(24)
        magic, version, rows, cols, nnz = struct.unpack("<IIIIQ", header)
        row_ptr_offset = 24
        f.seek(row_ptr_offset + rows * 4)
        row_ptr_last = struct.unpack("<I", f.read(4))[0]
    expected_size = 24 + (rows + 1) * 4 + nnz * 4 + nnz * 4
    return {
        "magic": magic,
        "magic_text": magic.to_bytes(4, "big").decode("ascii", errors="replace"),
        "version": version,
        "rows": rows,
        "cols": cols,
        "nnz": nnz,
        "row_ptr_last": row_ptr_last,
        "file_size": file_size,
        "expected_file_size": expected_size,
    }


def read_projection_bin(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"Projection BIN file is too small: {path}")
    num_angles, rays_per_angle = struct.unpack_from("<ii", data, 0)
    payload_bytes = len(data) - 8
    if payload_bytes % 8 != 0:
        raise ValueError("Projection BIN payload is not float64 aligned")
    records = payload_bytes // 8
    return {
        "num_angles": num_angles,
        "rays_per_angle": rays_per_angle,
        "records": records,
        "file_size": len(data),
        "expected_file_size": 8 + num_angles * rays_per_angle * 8,
    }


def read_metadata(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 44:
        raise ValueError(f"metadata.bin is too small: {path}")
    ints = struct.unpack_from("<11i", data, 0)
    (
        total_rows,
        total_voxels,
        max_nnz,
        compact_num_angles,
        det_nx,
        compact_det_z,
        compact_rays_per_angle,
        full_num_angles,
        nx,
        ny,
        nz,
    ) = ints
    offset = 44
    voxel_size = struct.unpack_from("<f", data, offset)[0]
    offset += 4
    det_nx_dup, det_nz = struct.unpack_from("<ii", data, offset)
    offset += 8
    pixel_size, detector_y, thickness = struct.unpack_from("<fff", data, offset)
    offset += 12
    use_90_sym, use_z_sym = struct.unpack_from("<ii", data, offset)
    offset += 8
    use_45_sym = 0
    sym_code = 0
    symmetry_order = 1
    if len(data) >= offset + 12:
        use_45_sym, sym_code, symmetry_order = struct.unpack_from("<iii", data, offset)
    looks_45 = compact_num_angles == full_num_angles // 8 + 1
    looks_90 = compact_num_angles == full_num_angles // 4
    if use_z_sym and (use_45_sym or looks_45):
        mode = "45z"
    elif use_z_sym and (use_90_sym or looks_90):
        mode = "90z"
    elif use_90_sym or looks_90:
        mode = "90"
    elif use_45_sym or looks_45:
        mode = "45"
    else:
        mode = "unknown"
    return {
        "compact_num_angles": compact_num_angles,
        "det_nx": det_nx,
        "det_nx_duplicate": det_nx_dup,
        "det_nz": det_nz,
        "compact_det_z": compact_det_z,
        "compact_rays_per_angle": compact_rays_per_angle,
        "total_rows": total_rows,
        "total_voxels": total_voxels,
        "max_nnz": max_nnz,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "full_num_angles": full_num_angles,
        "use_45_sym": use_45_sym,
        "use_90_sym": use_90_sym,
        "use_z_sym": use_z_sym,
        "sym_code": sym_code,
        "symmetry_order": symmetry_order,
        "mode": mode,
        "voxel_size": voxel_size,
        "pixel_size": pixel_size,
        "detector_y": detector_y,
        "thickness": thickness,
    }


def add_check(checks: dict[str, bool], name: str, actual, expected) -> None:
    checks[name] = actual == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="260525",
        help="Geometry profile (default: 260525)",
    )
    parser.add_argument(
        "--matrix",
        default="gatedata/6module/matrix_6module.csrb",
        help="Compact CSRB path",
    )
    parser.add_argument(
        "--projection",
        default="gatedata/6module/PSC_6module_24pin_10MBq_600s_1.bin",
        help="Full projection BIN path",
    )
    parser.add_argument(
        "--metadata",
        default="gatedata/6module/metadata.bin",
        help="Compact symmetry metadata path",
    )
    parser.add_argument(
        "--out-json",
        default="recon_out/qa/6module_compact_input_validation.json",
        help="Output JSON report",
    )
    parser.add_argument("--rows", type=int, default=None, help="Override compact matrix rows")
    parser.add_argument("--cols", type=int, default=None, help="Override matrix cols / total voxels")
    parser.add_argument("--nnz", type=int, default=None, help="Override single-file CSRB nnz")
    parser.add_argument("--angles", type=int, default=None, help="Override compact_num_angles")
    parser.add_argument(
        "--full-angles",
        type=int,
        default=None,
        help="Override full projection num_angles",
    )
    parser.add_argument("--detz", type=int, default=None, help="Override compact_det_z")
    parser.add_argument("--detnx", type=int, default=None, help="Override det_nx")
    parser.add_argument("--detnz", type=int, default=None, help="Override det_nz")
    parser.add_argument(
        "--rays-per-angle",
        type=int,
        default=None,
        help="Override compact_rays_per_angle",
    )
    parser.add_argument("--nx", type=int, default=None, help="Override image Nx")
    parser.add_argument("--ny", type=int, default=None, help="Override image Ny")
    parser.add_argument("--nz", type=int, default=None, help="Override image Nz")
    args = parser.parse_args()

    expected = resolve_expected(
        args.profile,
        {
            "matrix_rows": args.rows,
            "matrix_cols": args.cols,
            "matrix_nnz": args.nnz,
            "compact_num_angles": args.angles,
            "projection_num_angles": args.full_angles,
            "compact_det_z": args.detz,
            "det_nx": args.detnx,
            "det_nz": args.detnz,
            "compact_rays_per_angle": args.rays_per_angle,
            "image_nx": args.nx,
            "image_ny": args.ny,
            "image_nz": args.nz,
        },
    )
    if args.full_angles is not None:
        expected["projection_full_rows"] = (
            expected["projection_num_angles"]
            * expected["det_nx"]
            * expected["det_nz"]
        )
    if args.angles is not None or args.rays_per_angle is not None:
        expected["matrix_rows"] = expected["compact_num_angles"] * expected["compact_rays_per_angle"]
    if args.nx is not None or args.ny is not None or args.nz is not None:
        expected["total_voxels"] = expected["image_nx"] * expected["image_ny"] * expected["image_nz"]
        expected["matrix_cols"] = expected["total_voxels"]

    skip_matrix = expected["matrix_nnz"] is None
    matrix = None if skip_matrix else read_csrb(Path(args.matrix))
    projection = read_projection_bin(Path(args.projection))
    metadata = read_metadata(Path(args.metadata))

    checks: dict[str, bool] = {}
    if matrix is not None:
        add_check(checks, "matrix_magic_is_csrb", matrix["magic_text"], "CSRB")
        add_check(checks, "matrix_version_is_1", matrix["version"], 1)
        add_check(checks, "matrix_rows", matrix["rows"], expected["matrix_rows"])
        add_check(checks, "matrix_cols", matrix["cols"], expected["matrix_cols"])
        add_check(checks, "matrix_nnz", matrix["nnz"], expected["matrix_nnz"])
        add_check(checks, "matrix_row_ptr_last_matches_nnz", matrix["row_ptr_last"], matrix["nnz"])
        add_check(checks, "matrix_file_size_matches", matrix["file_size"], matrix["expected_file_size"])
    add_check(checks, "projection_num_angles", projection["num_angles"], expected["projection_num_angles"])
    add_check(checks, "projection_rays_per_angle", projection["rays_per_angle"], expected["projection_rays_per_angle"])
    add_check(checks, "projection_records", projection["records"], expected["projection_full_rows"])
    add_check(checks, "projection_file_size_matches", projection["file_size"], projection["expected_file_size"])
    add_check(checks, "metadata_mode", metadata["mode"], expected["metadata_mode"])
    add_check(checks, "metadata_compact_num_angles", metadata["compact_num_angles"], expected["compact_num_angles"])
    add_check(checks, "metadata_det_nx", metadata["det_nx"], expected["det_nx"])
    add_check(checks, "metadata_det_nz", metadata["det_nz"], expected["det_nz"])
    add_check(checks, "metadata_compact_det_z", metadata["compact_det_z"], expected["compact_det_z"])
    add_check(
        checks,
        "metadata_compact_rays_per_angle",
        metadata["compact_rays_per_angle"],
        expected["compact_rays_per_angle"],
    )
    add_check(checks, "metadata_total_rows", metadata["total_rows"], expected["matrix_rows"])
    add_check(checks, "metadata_total_voxels", metadata["total_voxels"], expected["total_voxels"])
    add_check(checks, "metadata_full_num_angles", metadata["full_num_angles"], expected["projection_num_angles"])
    add_check(checks, "metadata_nx", metadata["nx"], expected["image_nx"])
    add_check(checks, "metadata_ny", metadata["ny"], expected["image_ny"])
    add_check(checks, "metadata_nz", metadata["nz"], expected["image_nz"])
    add_check(
        checks,
        "compact_rows_formula",
        metadata["compact_num_angles"] * metadata["compact_rays_per_angle"],
        expected["matrix_rows"],
    )
    add_check(
        checks,
        "full_rows_formula",
        metadata["full_num_angles"] * metadata["det_nx"] * metadata["det_nz"],
        expected["projection_full_rows"],
    )
    add_check(
        checks,
        "total_voxels_formula",
        metadata["nx"] * metadata["ny"] * metadata["nz"],
        expected["total_voxels"],
    )

    report = {
        "profile": args.profile,
        "expected": expected,
        "matrix": matrix,
        "matrix_skipped": skip_matrix,
        "projection": projection,
        "metadata": metadata,
        "checks": checks,
        "passed": all(checks.values()),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
