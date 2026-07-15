#!/usr/bin/env python3
"""Export SPECT_recon MHD/RAW volumes to DICOM series and/or PNG slice stacks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from eval_mhd_stats import _read_mhd_header, read_mhd

DEFAULT_SPECT_TO_DICOM_CANDIDATES = (
    Path(r"D:\Projects\spect_to_dicom\build\spect_to_dicom.exe"),
    Path(r"D:\Projects\spect_to_dicom\out\install\Release\bin\spect_to_dicom.exe"),
    Path(r"D:\Projects\spect_to_dicom\build\Release\spect_to_dicom.exe"),
)


def resolve_spect_to_dicom_exe(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    env = os.environ.get("SPECT_TO_DICOM_EXE", "").strip()
    if env:
        path = Path(env)
        if path.is_file():
            return path
    for candidate in DEFAULT_SPECT_TO_DICOM_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def read_mhd_spacing(path: Path) -> tuple[float, float, float]:
    header = _read_mhd_header(path)
    raw = header.get("ElementSpacing") or header.get("ElementSize")
    if not raw:
        return (1.0, 1.0, 1.0)
    vals = tuple(float(v) for v in raw.split())
    if len(vals) != 3:
        return (1.0, 1.0, 1.0)
    return (vals[0], vals[1], vals[2])


def save_center_views(
    image: np.ndarray,
    out_png: Path,
    title: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    zc = image.shape[0] // 2
    yc = image.shape[1] // 2
    xc = image.shape[2] // 2
    views = [
        ("Axial Z=center", image[zc, :, :]),
        ("Coronal Y=center", image[:, yc, :]),
        ("Sagittal X=center", image[:, :, xc]),
    ]
    finite = image[np.isfinite(image)]
    lo = float(np.percentile(finite, 1.0)) if vmin is None else float(vmin)
    hi = float(np.percentile(finite, 99.0)) if vmax is None else float(vmax)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (label, data) in zip(axes, views):
        im = ax.imshow(data, cmap="magma", origin="lower", vmin=lo, vmax=hi)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_slice_stack(
    image: np.ndarray,
    out_dir: Path,
    plane: str,
    prefix: str,
    vmin: float | None,
    vmax: float | None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    finite = image[np.isfinite(image)]
    lo = float(np.percentile(finite, 1.0)) if vmin is None else float(vmin)
    hi = float(np.percentile(finite, 99.0)) if vmax is None else float(vmax)

    count = 0
    if plane == "axial":
        for z in range(image.shape[0]):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(image[z, :, :], cmap="magma", origin="lower", vmin=lo, vmax=hi)
            ax.set_title(f"Z={z}")
            ax.axis("off")
            fig.savefig(out_dir / f"{prefix}_z{z:04d}.png", dpi=120, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            count += 1
    elif plane == "coronal":
        for y in range(image.shape[1]):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(image[:, y, :], cmap="magma", origin="lower", vmin=lo, vmax=hi)
            ax.set_title(f"Y={y}")
            ax.axis("off")
            fig.savefig(out_dir / f"{prefix}_y{y:04d}.png", dpi=120, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            count += 1
    elif plane == "sagittal":
        for x in range(image.shape[2]):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(image[:, :, x], cmap="magma", origin="lower", vmin=lo, vmax=hi)
            ax.set_title(f"X={x}")
            ax.axis("off")
            fig.savefig(out_dir / f"{prefix}_x{x:04d}.png", dpi=120, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            count += 1
    else:
        raise ValueError(f"Unsupported plane: {plane}")
    return count


def _scale_to_uint16(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint16), 1.0, 0.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint16), 1.0, lo
    slope = (hi - lo) / 65535.0
    stored = np.clip(np.rint((image - lo) / slope), 0, 65535).astype(np.uint16)
    return stored, slope, lo


def write_dicom_series(
    image: np.ndarray,
    out_dir: Path,
    spacing_xyz: tuple[float, float, float],
    series_description: str,
    patient_name: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = spacing_xyz
    stored, slope, intercept = _scale_to_uint16(image)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref = generate_uid()
    sop_class = pydicom.uid.SecondaryCaptureImageStorage

    now = datetime.now()
    study_date = now.strftime("%Y%m%d")
    study_time = now.strftime("%H%M%S.%f")[:10]

    files: list[str] = []
    for z in range(image.shape[0]):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = sop_class
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.ImplementationClassUID = generate_uid()

        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "OT"
        ds.SeriesDescription = series_description
        ds.StudyDescription = series_description
        ds.PatientName = patient_name
        ds.PatientID = re.sub(r"[^A-Za-z0-9_-]", "_", patient_name)[:64]
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_of_ref
        ds.StudyDate = study_date
        ds.StudyTime = study_time
        ds.SeriesNumber = 1
        ds.InstanceNumber = z + 1
        ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = image.shape[1]
        ds.Columns = image.shape[2]
        ds.PixelSpacing = [sy, sx]
        ds.SliceThickness = str(sz)
        ds.SpacingBetweenSlices = str(sz)
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(z * sz)]
        ds.RescaleIntercept = intercept
        ds.RescaleSlope = slope
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.PixelData = stored[z, :, :].tobytes()

        out_path = out_dir / f"slice_{z:04d}.dcm"
        ds.save_as(out_path, write_like_original=False)
        files.append(str(out_path))

    return {
        "backend": "pydicom",
        "dicom_dir": str(out_dir),
        "slice_count": len(files),
        "study_instance_uid": study_uid,
        "series_instance_uid": series_uid,
        "modality": "OT",
        "sop_class": str(sop_class),
        "note": "Fallback writer uses Secondary Capture; prefer spect_to_dicom backend for 3D viewers.",
    }


def apply_xy_fov(
    image: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    diameter_mm: float,
    crop_xy: bool,
) -> np.ndarray:
    """Circular XY FOV mask (and optional crop) matching spect_to_dicom -rad logic."""
    if diameter_mm <= 0:
        return image
    sx, sy, _ = spacing_xyz
    _, ny, nx = image.shape
    target_w = max(1, min(int(round(diameter_mm / sx)), nx))
    target_h = max(1, min(int(round(diameter_mm / sy)), ny))
    cx = (nx - 1) * 0.5
    cy = (ny - 1) * 0.5
    radius_mm = diameter_mm * 0.5

    out = image.copy()
    yy, xx = np.ogrid[:ny, :nx]
    dy_mm = (yy - cy) * sy
    dx_mm = (xx - cx) * sx
    inside = (dx_mm * dx_mm + dy_mm * dy_mm) <= radius_mm * radius_mm
    for z in range(out.shape[0]):
        sl = out[z]
        sl[~inside] = 0.0

    if not crop_xy:
        return out

    left = (nx - target_w) // 2
    top = (ny - target_h) // 2
    return out[:, top : top + target_h, left : left + target_w]


def export_dicom_via_spect_to_dicom(
    image_path: Path,
    out_dir: Path,
    exe_path: Path,
    patient_name: str,
    study_id: str,
    xflip: bool = False,
    zflip: bool = False,
    fov_diameter_mm: float = 0.0,
    fov_crop_xy: bool = False,
    crop_z_mm: tuple[float, float] = (0.0, 0.0),
) -> dict:
    top_mm, bottom_mm = crop_z_mm
    if out_dir.exists():
        for old in out_dir.glob("*.dcm"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe_path),
        "--patient-name",
        patient_name,
        "--patient-id",
        re.sub(r"[^A-Za-z0-9_-]", "_", patient_name)[:64] or "SPECT_RECON",
        "--study-id",
        study_id,
    ]
    if xflip:
        cmd.extend(["-xflip", "1"])
    if zflip:
        cmd.extend(["-zflip", "1"])
    if fov_diameter_mm > 0.0:
        cmd.extend(["-rad", str(fov_diameter_mm)])
        if fov_crop_xy:
            cmd.extend(["-rad-crop", "1"])
    top_mm, bottom_mm = crop_z_mm
    if top_mm > 0.0 or bottom_mm > 0.0:
        cmd.extend(["-cropZ", f"{top_mm},{bottom_mm}"])
    cmd.extend([str(image_path), str(out_dir)])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"spect_to_dicom failed (exit {proc.returncode}): {detail or cmd}"
        )

    files = sorted(out_dir.glob("*.dcm"))
    if not files:
        raise RuntimeError(f"spect_to_dicom produced no .dcm files in {out_dir}")

    return {
        "backend": "spect_to_dicom",
        "exe": str(exe_path),
        "dicom_dir": str(out_dir),
        "slice_count": len(files),
        "file_pattern": "image_*.dcm",
        "modality": "PT",
        "sop_class": "1.2.840.10008.5.1.4.1.1.128",
        "rescale_type": "PERCENT",
        "units": "PCNT",
        "fov_diameter_mm": fov_diameter_mm,
        "fov_crop_xy": fov_crop_xy,
        "crop_z_mm": [top_mm, bottom_mm],
        "stdout_tail": (proc.stdout or "").splitlines()[-3:],
    }


def export_volume(
    image_path: Path,
    out_dir: Path,
    fmt: str,
    png_mode: str,
    png_planes: list[str],
    series_description: str,
    patient_name: str,
    vmin: float | None,
    vmax: float | None,
    dicom_backend: str = "auto",
    spect_to_dicom_exe: str | None = None,
    dicom_xflip: bool = False,
    dicom_zflip: bool = False,
    fov_diameter_mm: float = 0.0,
    fov_crop_xy: bool = False,
    crop_z_mm: tuple[float, float] = (0.0, 0.0),
) -> dict:
    image = read_mhd(image_path).astype(np.float64, copy=False)
    spacing = read_mhd_spacing(image_path)
    display_image = image
    if fov_diameter_mm > 0.0:
        display_image = apply_xy_fov(image, spacing, fov_diameter_mm, fov_crop_xy)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "source_mhd": str(image_path),
        "out_dir": str(out_dir),
        "shape_zyx": list(image.shape),
        "spacing_xyz_mm": list(spacing),
        "format": fmt,
        "fov_diameter_mm": fov_diameter_mm,
        "fov_crop_xy": fov_crop_xy,
        "crop_z_mm": list(crop_z_mm),
    }

    if fmt in {"png", "both"}:
        png_root = out_dir / "png"
        png_summary: dict = {"root": str(png_root)}
        if png_mode in {"center", "both"}:
            center_png = png_root / "views_center.png"
            save_center_views(display_image, center_png, image_path.name, vmin, vmax)
            png_summary["center_views"] = str(center_png)
        if png_mode in {"slices", "both"}:
            stacks: dict[str, dict] = {}
            for plane in png_planes:
                plane_dir = png_root / plane
                count = save_slice_stack(
                    display_image, plane_dir, plane, image_path.stem, vmin, vmax
                )
                stacks[plane] = {"dir": str(plane_dir), "count": count}
            png_summary["slice_stacks"] = stacks
        summary["png"] = png_summary

    if fmt in {"dicom", "both"}:
        dicom_dir = out_dir / "dicom"
        backend = dicom_backend
        exe = resolve_spect_to_dicom_exe(spect_to_dicom_exe)
        if backend == "auto":
            backend = "spect_to_dicom" if exe is not None else "pydicom"
        if backend == "spect_to_dicom":
            if exe is None:
                raise RuntimeError(
                    "spect_to_dicom backend requested but executable not found. "
                    "Set --spect-to-dicom-exe or SPECT_TO_DICOM_EXE."
                )
            summary["dicom"] = export_dicom_via_spect_to_dicom(
                image_path,
                dicom_dir,
                exe,
                patient_name,
                study_id=series_description,
                xflip=dicom_xflip,
                zflip=dicom_zflip,
                fov_diameter_mm=fov_diameter_mm,
                fov_crop_xy=fov_crop_xy,
                crop_z_mm=crop_z_mm,
            )
        elif backend == "pydicom":
            summary["dicom"] = write_dicom_series(
                image, dicom_dir, spacing, series_description, patient_name
            )
        else:
            raise ValueError(f"Unsupported dicom backend: {backend}")

    summary_path = out_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input reconstruction .mhd")
    parser.add_argument("--out-dir", required=True, help="Export output directory")
    parser.add_argument(
        "--format",
        choices=("dicom", "png", "both"),
        default="both",
        help="Export format (default: both)",
    )
    parser.add_argument(
        "--png-mode",
        choices=("slices", "center", "both"),
        default="slices",
        help="PNG: slice stacks (default), center views, or both",
    )
    parser.add_argument(
        "--png-planes",
        default="axial",
        help="PNG slice planes: axial,coronal,sagittal",
    )
    parser.add_argument("--series-description", default="SPECT_recon")
    parser.add_argument("--patient-name", default="SPECT_RECON")
    parser.add_argument(
        "--dicom-backend",
        choices=("auto", "spect_to_dicom", "pydicom"),
        default="auto",
        help="DICOM writer: spect_to_dicom (DCMTK, default when exe found) or pydicom fallback",
    )
    parser.add_argument(
        "--spect-to-dicom-exe",
        default=None,
        help="Path to spect_to_dicom.exe (or set SPECT_TO_DICOM_EXE)",
    )
    parser.add_argument(
        "--dicom-xflip",
        action="store_true",
        help="Pass -xflip 1 to spect_to_dicom when orientation looks mirrored",
    )
    parser.add_argument(
        "--dicom-zflip",
        action="store_true",
        help="Pass -zflip 1 to spect_to_dicom to reverse slice order",
    )
    parser.add_argument(
        "--fov-diameter-mm",
        type=float,
        default=0.0,
        help="XY FOV diameter in mm (spect_to_dicom -rad); e.g. 30 for mouse",
    )
    parser.add_argument(
        "--fov-crop-xy",
        action="store_true",
        help="Crop output columns/rows to FOV box (spect_to_dicom -rad-crop 1)",
    )
    parser.add_argument(
        "--crop-z-mm",
        default="",
        help="Zero axial end ranges in mm at export, e.g. 3.6,3.6 (spect_to_dicom -cropZ)",
    )
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    crop_z_mm = (0.0, 0.0)
    if args.crop_z_mm.strip():
        parts = [p.strip() for p in args.crop_z_mm.split(",") if p.strip()]
        if len(parts) != 2:
            raise SystemExit("--crop-z-mm expects top,bottom in mm")
        crop_z_mm = (float(parts[0]), float(parts[1]))

    planes = [p.strip().lower() for p in args.png_planes.split(",") if p.strip()]
    summary = export_volume(
        Path(args.image),
        Path(args.out_dir),
        args.format,
        args.png_mode,
        planes,
        args.series_description,
        args.patient_name,
        args.vmin,
        args.vmax,
        args.dicom_backend,
        args.spect_to_dicom_exe,
        args.dicom_xflip,
        args.dicom_zflip,
        args.fov_diameter_mm,
        args.fov_crop_xy,
        crop_z_mm,
    )
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
