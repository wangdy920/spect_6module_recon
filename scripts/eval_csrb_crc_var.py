#!/usr/bin/env python3
"""Approximate local CRC-variance curves from a standard CSRB system matrix."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


CSRB_MAGIC = 0x43535242
EPS = 1e-12


def read_csrb(path: Path):
    with path.open("rb") as stream:
        magic, version, rows, cols, nnz = struct.unpack("<IIIIQ", stream.read(24))
    if magic != CSRB_MAGIC or version != 1:
        raise ValueError(f"Expected standard CSRB v1, got magic=0x{magic:08x}, version={version}")
    expected = 24 + (rows + 1) * 4 + nnz * 8
    if path.stat().st_size != expected:
        raise ValueError(f"CSRB size mismatch: expected {expected}, got {path.stat().st_size}")
    row_ptr = np.memmap(path, "<u4", "r", 24, (rows + 1,))
    col_offset = 24 + (rows + 1) * 4
    col_idx = np.memmap(path, "<u4", "r", col_offset, (nnz,))
    values = np.memmap(path, "<f4", "r", col_offset + nnz * 4, (nnz,))
    return rows, cols, nnz, row_ptr, col_idx, values


def centered_indices(dims, roi_dims):
    nx, ny, nz = dims
    rx, ry, rz = roi_dims
    if min(dims) <= 0 or min(roi_dims) <= 0 or rx > nx or ry > ny or rz > nz:
        raise ValueError("ROI dimensions must be positive and fit inside --dims")
    x0, y0, z0 = (nx - rx) // 2, (ny - ry) // 2, (nz - rz) // 2
    indices = [((z0 + z) * ny + (y0 + y)) * nx + x0 + x
               for z in range(rz) for y in range(ry) for x in range(rx)]
    return np.asarray(indices, dtype=np.int64)


def extract_local_matrix(path, dims, roi_dims, max_rows):
    rows, cols, nnz, row_ptr, col_idx, values = read_csrb(path)
    if np.prod(dims, dtype=np.int64) != cols:
        raise ValueError(f"--dims product {np.prod(dims)} does not match CSRB cols {cols}")
    take = min(rows, max_rows)
    row_ids = np.unique(np.linspace(0, rows - 1, take, dtype=np.int64))
    roi = centered_indices(dims, roi_dims)
    lookup = np.full(cols, -1, dtype=np.int32)
    lookup[roi] = np.arange(roi.size, dtype=np.int32)
    matrix = np.zeros((row_ids.size, roi.size), dtype=np.float64)
    expected = np.zeros(row_ids.size, dtype=np.float64)
    for out_row, row in enumerate(row_ids):
        start, end = int(row_ptr[row]), int(row_ptr[row + 1])
        row_values = np.asarray(values[start:end], dtype=np.float64)
        expected[out_row] = row_values.sum()
        local = lookup[np.asarray(col_idx[start:end], dtype=np.int64)]
        keep = local >= 0
        matrix[out_row, local[keep]] = row_values[keep]
    valid = expected > EPS
    if not np.any(valid):
        raise ValueError("Sampled rows have zero expected response")
    return matrix[valid], expected[valid], row_ids[valid], roi, {
        "rows": rows, "cols": cols, "nnz": nnz, "sample_scale": rows / float(row_ids.size)
    }


def regularizer(x, shape, spacing):
    rx, ry, rz = shape
    sx, sy, sz = spacing
    volume = x.T.reshape((-1, rz, ry, rx))
    out = np.zeros_like(volume)
    for axis, weight in ((3, 1 / sx**2), (2, 1 / sy**2), (1, 1 / sz**2)):
        left = [slice(None)] * 4
        right = [slice(None)] * 4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        diff = volume[tuple(right)] - volume[tuple(left)]
        out[tuple(right)] += weight * diff
        out[tuple(left)] -= weight * diff
    return out.reshape((x.shape[1], x.shape[0])).T


def compute_curves(matrix, expected, roi_shape, spacing, lambdas, sample_scale=1.0):
    weighted = matrix / expected[:, None]
    fisher = sample_scale * matrix.T @ weighted
    regularizer_matrix = regularizer(np.eye(matrix.shape[1]), roi_shape, spacing)
    diag_f = np.diag(fisher)
    diag_r = np.diag(regularizer_matrix)
    positive_f = diag_f[diag_f > EPS]
    positive_r = diag_r[diag_r > EPS]
    if positive_f.size == 0 or positive_r.size == 0:
        raise ValueError("Fisher or regularizer diagonal is empty")
    beta_scale = float(np.mean(positive_f) / np.mean(positive_r))

    identity = np.eye(matrix.shape[1], dtype=np.float64)
    results = []
    for lambda_value in lambdas:
        beta = float(lambda_value) * beta_scale
        system = fisher + beta * regularizer_matrix
        sol_var = np.linalg.solve(system, identity)
        sol_crc = np.linalg.solve(system, fisher)
        variance = np.sum(sol_var * (fisher @ sol_var), axis=0)
        crc = np.diag(sol_crc)
        results.append({
            "lambda": float(lambda_value), "beta": beta,
            "crc_mean": float(np.mean(crc)), "variance_mean": float(np.mean(variance)),
            "crc_min": float(np.min(crc)), "crc_max": float(np.max(crc)),
            "condition_number": float(np.linalg.cond(system)),
        })
    return beta_scale, results


def self_test():
    matrix = np.eye(9, dtype=np.float64)
    beta_scale, curves = compute_curves(matrix, np.ones(9), (3, 3, 1), (1, 1, 1), [0, 1])
    assert beta_scale > 0
    assert abs(curves[0]["crc_mean"] - 1.0) < 1e-10
    assert abs(curves[0]["variance_mean"] - 1.0) < 1e-10
    assert curves[1]["variance_mean"] < curves[0]["variance_mean"]
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--dims", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=(108, 108, 80))
    parser.add_argument("--roi-dims", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=(7, 7, 3))
    parser.add_argument("--spacing", type=float, nargs=3, metavar=("SX", "SY", "SZ"), default=(0.3, 0.3, 0.3))
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--lambdas", type=float, nargs="+", default=(1e-4, 1e-3, 1e-2, 1e-1, 1))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.matrix or not args.output:
        parser.error("--matrix and --output are required unless --self-test is used")
    matrix, expected, row_ids, roi, info = extract_local_matrix(
        args.matrix, tuple(args.dims), tuple(args.roi_dims), args.max_rows)
    beta_scale, curves = compute_curves(
        matrix, expected, tuple(args.roi_dims), tuple(args.spacing), args.lambdas, info["sample_scale"])
    report = {
        "status": "diagnostic_smoke_only", "matrix": str(args.matrix.resolve()),
        "matrix_info": info, "dims_xyz": args.dims, "roi_dims_xyz": args.roi_dims,
        "spacing_xyz_mm": args.spacing, "sampled_rows": int(row_ids.size),
        "roi_voxels": int(roi.size), "solver": "dense_float64",
        "beta_scale": beta_scale, "curves": curves,
        "limitations": [
            "uniform full-FOV unit object",
            "central ROI submatrix with outside voxels held fixed", "no CTAC, PSF, background, or count normalization",
        ] + (["uniformly sampled projection rows"] if args.max_rows < info["rows"] else []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
