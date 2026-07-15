#!/usr/bin/env python3
"""Render a reconstruction volume as a true 3D isosurface PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from eval_mhd_stats import read_mhd


def read_spacing(path: Path) -> tuple[float, float, float]:
    header = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    spacing = tuple(float(v) for v in header["ElementSpacing"].split())
    if len(spacing) != 3:
        raise ValueError("ElementSpacing must contain x y z")
    return spacing


def apply_fov_mask(
    image: np.ndarray, spacing_xyz: tuple[float, float, float], diameter_mm: float
) -> np.ndarray:
    if diameter_mm <= 0:
        return image
    sx, sy, _ = spacing_xyz
    _, ny, nx = image.shape
    cy = (ny - 1) * 0.5
    cx = (nx - 1) * 0.5
    yy, xx = np.ogrid[:ny, :nx]
    inside = ((yy - cy) * sy) ** 2 + ((xx - cx) * sx) ** 2 <= (
        diameter_mm * 0.5
    ) ** 2
    out = image.copy()
    out[:, ~inside] = 0.0
    return out


def render_isosurface(
    image_path: Path,
    output_path: Path,
    threshold: float | None,
    percentile: float,
    fov_diameter_mm: float,
    downsample: int,
    elev: float,
    azim: float,
) -> dict:
    image = read_mhd(image_path).astype(np.float32, copy=False)
    spacing_xyz = read_spacing(image_path)
    image = apply_fov_mask(image, spacing_xyz, fov_diameter_mm)
    positive = image[image > 0]
    if positive.size == 0:
        raise ValueError("volume has no positive voxels after FOV masking")
    level = float(np.percentile(positive, percentile) if threshold is None else threshold)
    if not (float(np.min(image)) < level < float(np.max(image))):
        raise ValueError(
            f"isosurface threshold {level:g} is outside volume range "
            f"[{float(np.min(image)):g}, {float(np.max(image)):g}]"
        )

    step = max(1, int(downsample))
    sampled = image[::step, ::step, ::step]
    sampled_spacing = tuple(v * step for v in spacing_xyz)
    vertices_zyx, faces, normals, values = marching_cubes(
        sampled, level=level, spacing=(sampled_spacing[2], sampled_spacing[1], sampled_spacing[0])
    )
    vertices_xyz = vertices_zyx[:, [2, 1, 0]]
    face_values = values[faces].mean(axis=1)
    norm = plt.Normalize(vmin=level, vmax=max(float(np.max(face_values)), level))
    colors = cm.magma(norm(face_values))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(vertices_xyz[faces], facecolors=colors, linewidths=0.0)
    mesh.set_alpha(0.9)
    ax.add_collection3d(mesh)
    extent = np.array(sampled.shape)[[2, 1, 0]] * np.array(sampled_spacing)
    ax.set_xlim(0, extent[0])
    ax.set_ylim(0, extent[1])
    ax.set_zlim(0, extent[2])
    ax.set_box_aspect(extent)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{image_path.name}\n3D isosurface, threshold={level:g}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "source_mhd": str(image_path),
        "output_png": str(output_path),
        "shape_zyx": list(image.shape),
        "spacing_xyz_mm": list(spacing_xyz),
        "fov_diameter_mm": fov_diameter_mm,
        "threshold": level,
        "percentile": percentile if threshold is None else None,
        "downsample": step,
        "vertices": int(vertices_xyz.shape[0]),
        "faces": int(faces.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Input reconstruction .mhd")
    parser.add_argument("--output", help="Output 3D PNG")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--fov-diameter-mm", type=float, default=0.0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        grid = np.zeros((24, 24, 24), dtype=np.float32)
        grid[6:18, 6:18, 6:18] = 1.0
        vertices, faces, _, _ = marching_cubes(grid, level=0.5)
        assert vertices.size and faces.size
        print("render_volume_3d self-test passed")
        return 0
    if not args.image or not args.output:
        parser.error("--image and --output are required unless --self-test is used")
    result = render_isosurface(
        Path(args.image),
        Path(args.output),
        args.threshold,
        args.percentile,
        args.fov_diameter_mm,
        args.downsample,
        args.elev,
        args.azim,
    )
    import json

    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
