#!/usr/bin/env python3
"""Resample a paired CT DICOM series onto the SPECT reconstruction grid."""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_mhd_stats import read_mhd  # noqa: E402


def dicoms(root: Path) -> list:
    result = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), force=True)
            if hasattr(ds, "PixelData") and hasattr(ds, "ImagePositionPatient"):
                result.append(ds)
        except Exception:
            pass
    if not result:
        raise RuntimeError(f"No image DICOM found under {root}")
    return result


def _geometry(ds) -> dict:
    """Return the DICOM patient-space affine basis for one slice."""
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    if iop.size != 6 or not np.all(np.isfinite(iop)):
        raise RuntimeError("DICOM ImageOrientationPatient must contain six finite values")
    col_dir = iop[:3]
    row_dir = iop[3:]
    if not np.isclose(np.linalg.norm(col_dir), 1.0, atol=1e-4):
        raise RuntimeError("DICOM column direction is not unit length")
    if not np.isclose(np.linalg.norm(row_dir), 1.0, atol=1e-4):
        raise RuntimeError("DICOM row direction is not unit length")
    if not np.isclose(np.dot(col_dir, row_dir), 0.0, atol=1e-4):
        raise RuntimeError("DICOM row/column directions are not orthogonal")
    normal = np.cross(col_dir, row_dir)
    normal /= np.linalg.norm(normal)
    return {
        "col_dir": col_dir,
        "row_dir": row_dir,
        "normal": normal,
        "origin": np.asarray(ds.ImagePositionPatient, dtype=float),
        "pixel_spacing": np.asarray(ds.PixelSpacing, dtype=float),
    }


def _sorted_series(items: list) -> tuple[list, dict, np.ndarray]:
    if not items:
        raise RuntimeError("Empty DICOM series")
    geom = _geometry(items[0])
    for item in items[1:]:
        other = _geometry(item)
        if not np.allclose(other["col_dir"], geom["col_dir"], atol=1e-5):
            raise RuntimeError("DICOM series has inconsistent column directions")
        if not np.allclose(other["row_dir"], geom["row_dir"], atol=1e-5):
            raise RuntimeError("DICOM series has inconsistent row directions")
        if not np.allclose(other["pixel_spacing"], geom["pixel_spacing"], atol=1e-5):
            raise RuntimeError("DICOM series has inconsistent pixel spacing")
    ordered = sorted(
        items,
        key=lambda item: float(np.dot(np.asarray(item.ImagePositionPatient, dtype=float),
                                      geom["normal"])),
    )
    positions = np.asarray([
        np.dot(np.asarray(item.ImagePositionPatient, dtype=float), geom["normal"])
        for item in ordered
    ], dtype=float)
    if len(positions) > 1:
        diffs = np.diff(positions)
        spacing = float(np.median(diffs))
        if not np.allclose(diffs, spacing, rtol=0.02, atol=1e-3):
            raise RuntimeError("DICOM slice spacing is not regular in patient space")
        if spacing <= 0:
            raise RuntimeError("DICOM slice order is not increasing in patient space")
    else:
        spacing = float(getattr(ordered[0], "SliceThickness", 0.0))
    geom = dict(geom)
    geom["origin"] = np.asarray(ordered[0].ImagePositionPatient, dtype=float)
    geom["slice_spacing"] = spacing
    geom["iop"] = np.concatenate([geom["col_dir"], geom["row_dir"]])
    return ordered, geom, positions


def _discover_bed_dirs(root: Path) -> dict[int, Path]:
    """Find optional BedN DICOM directories without changing the old CLI."""
    result = {}
    for child in root.iterdir() if root.is_dir() else []:
        if not child.is_dir():
            continue
        match = re.fullmatch(r"Bed(\d+)", child.name, flags=re.IGNORECASE)
        if match:
            result[int(match.group(1))] = child
    return result


def _patient_affine_coords(target_origin, target_geom, target_spacing, shape,
                           ct_geom, ct_spacing):
    z, y, x = np.indices(shape, dtype=np.float32)
    world = (
        target_origin[:, None, None, None]
        + target_geom["normal"][:, None, None, None] * (z * target_spacing[2])[None]
        + target_geom["row_dir"][:, None, None, None] * (y * target_spacing[1])[None]
        + target_geom["col_dir"][:, None, None, None] * (x * target_spacing[0])[None]
    )
    delta = world - ct_geom["origin"][:, None, None, None]
    return np.array([
        np.sum(delta * ct_geom["normal"][:, None, None, None], axis=0) / ct_spacing[2],
        np.sum(delta * ct_geom["row_dir"][:, None, None, None], axis=0) / ct_spacing[1],
        np.sum(delta * ct_geom["col_dir"][:, None, None, None], axis=0) / ct_spacing[0],
    ])


def hu_to_mu(hu: np.ndarray, water_mu: float) -> np.ndarray:
    return water_mu * np.clip(1.0 + hu / 1000.0, 0.0, 3.0)


def write_mhd(path: Path, volume: np.ndarray, spacing: tuple[float, float, float]) -> None:
    raw = path.with_suffix(".raw")
    np.asarray(volume, dtype="<f4").tofile(raw)
    z, y, x = volume.shape
    path.write_text(
        "ObjectType = Image\nNDims = 3\n"
        f"DimSize = {x} {y} {z}\nElementSpacing = {spacing[0]} {spacing[1]} {spacing[2]}\n"
        "ElementType = MET_FLOAT\nElementByteOrderMSB = False\n"
        f"ElementDataFile = {raw.name}\n",
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--spect-dicom-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beds", default="1,2,3,4",
                        help="Consecutive physical bed numbers, e.g. 2,3,4")
    parser.add_argument("--bed-nz", type=int, default=80)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--bed-step-mm", type=float, default=21.0)
    parser.add_argument("--water-mu", type=float, default=.0152)
    parser.add_argument("--activity", type=Path)
    parser.add_argument("--bed-step-tolerance-mm", type=float, default=0.75,
                        help="Diagnostic tolerance for per-bed DICOM origin checks")
    args = parser.parse_args()

    beds = [int(value) for value in args.beds.split(",")]
    if not beds or beds != list(range(beds[0], beds[0] + len(beds))) or not all(1 <= b <= 5 for b in beds):
        raise RuntimeError("--beds must be a consecutive subset of 1..5")
    nz = args.bed_nz * len(beds) - args.overlap * (len(beds) - 1)
    ct, ct_geom, _ = _sorted_series(dicoms(args.ct_dir))
    bed_dirs = _discover_bed_dirs(args.spect_dicom_dir)
    bed_series = {}
    bed_geometries = {}
    if bed_dirs:
        missing = [bed for bed in beds if bed not in bed_dirs]
        if missing:
            raise RuntimeError(f"Missing DICOM Bed directories: {missing}")
        for bed in beds:
            ordered, geometry, _ = _sorted_series(dicoms(bed_dirs[bed]))
            bed_series[bed] = ordered
            bed_geometries[bed] = geometry
        spect = bed_series[beds[0]]
        spect_geom = bed_geometries[beds[0]]
        first_for = str(getattr(spect[0], "FrameOfReferenceUID", ""))
        for bed in beds[1:]:
            if str(getattr(bed_series[bed][0], "FrameOfReferenceUID", "")) != first_for:
                raise RuntimeError("SPECT Bed DICOM FrameOfReferenceUID values differ")
    else:
        spect, spect_geom, _ = _sorted_series(dicoms(args.spect_dicom_dir))
    c0, s0 = ct[0], spect[0]
    if c0.FrameOfReferenceUID != s0.FrameOfReferenceUID:
        raise RuntimeError("CT and SPECT FrameOfReferenceUID differ")
    if not np.allclose(ct_geom["iop"], spect_geom["iop"], atol=1e-5):
        raise RuntimeError("CT and SPECT orientations differ; residual rigid registration is required")
    ct_patient_position = str(getattr(c0, "PatientPosition", ""))
    spect_patient_position = str(getattr(s0, "PatientPosition", ""))
    if ct_patient_position and spect_patient_position and ct_patient_position != spect_patient_position:
        raise RuntimeError("CT and SPECT PatientPosition differ")

    hu = np.stack([d.pixel_array.astype(np.float32) * float(d.RescaleSlope) +
                   float(d.RescaleIntercept) for d in ct])
    ct_origin = ct_geom["origin"]
    ct_spacing = np.array([ct_geom["pixel_spacing"][1],
                           ct_geom["pixel_spacing"][0],
                           ct_geom["slice_spacing"]])
    target_spacing = np.array([spect_geom["pixel_spacing"][1],
                               spect_geom["pixel_spacing"][0],
                               spect_geom["slice_spacing"]])
    bed_origin_source = "bed_step_fallback"
    bed_origins = {}
    bed_step_measurements = []
    if bed_series:
        bed_origins = {bed: bed_geometries[bed]["origin"].tolist() for bed in beds}
        first_origin = bed_geometries[beds[0]]["origin"]
        normal = spect_geom["normal"]
        target_grid_stride = (args.bed_nz - args.overlap) * target_spacing[2]
        for previous, current in zip(beds, beds[1:]):
            measured = float(np.dot(bed_geometries[current]["origin"] -
                                    bed_geometries[previous]["origin"], normal))
            bed_step_measurements.append(measured)
            if abs(measured - args.bed_step_mm) > args.bed_step_tolerance_mm:
                warnings.warn(
                    f"Bed{current}-Bed{previous} DICOM step {measured:.4f} mm "
                    f"differs from configured {args.bed_step_mm:.4f} mm")
            if abs(measured - target_grid_stride) > args.bed_step_tolerance_mm:
                raise RuntimeError(
                    f"DICOM bed step {measured:.4f} mm is incompatible with "
                    f"regular target grid stride {target_grid_stride:.4f} mm")
        target_origin = first_origin
        bed_origin_source = "per_bed_dicom_ipp"
    else:
        target_origin = spect_geom["origin"] + spect_geom["normal"] * ((beds[0] - 1) * args.bed_step_mm)
        target_grid_stride = (args.bed_nz - args.overlap) * target_spacing[2]
    coords = _patient_affine_coords(
        target_origin, spect_geom, target_spacing,
        (nz, int(s0.Rows), int(s0.Columns)), ct_geom, ct_spacing)
    target_hu = map_coordinates(hu, coords, order=1, mode="constant", cval=-1000.0)
    mu = hu_to_mu(target_hu, args.water_mu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_mhd(args.output, mu, tuple(target_spacing))

    manifest = {
        "decision": "geometry_ready_mu_calibration_provisional",
        "frame_of_reference_uid": str(c0.FrameOfReferenceUID),
        "ct_shape_zyx": list(hu.shape), "target_shape_zyx": list(mu.shape),
        "beds": beds, "bed_nz": args.bed_nz, "overlap": args.overlap,
        "bed_step_mm": args.bed_step_mm,
        "ct_origin_xyz": ct_origin.tolist(), "target_origin_xyz": target_origin.tolist(),
        "ct_spacing_xyz": ct_spacing.tolist(), "target_spacing_xyz": target_spacing.tolist(),
        "ct_image_orientation_patient": ct_geom["iop"].tolist(),
        "spect_image_orientation_patient": spect_geom["iop"].tolist(),
        "patient_space_normal": spect_geom["normal"].tolist(),
        "slice_sort": "ascending dot(ImagePositionPatient, normal)",
        "bed_origin_source": bed_origin_source,
        "bed_origins_xyz": bed_origins,
        "bed_step_measurements_mm": bed_step_measurements,
        "target_grid_stride_mm": target_grid_stride,
        "bed_step_tolerance_mm": args.bed_step_tolerance_mm,
        "patient_position": ct_patient_position or spect_patient_position,
        "hu_to_mu": "mu=water_mu*clip(1+HU/1000,0,3)", "water_mu_mm-1": args.water_mu,
        "registration_assumption": "DICOM patient-space affine first; at most one global residual transform; no per-bed registration",
        "mu_map": str(args.output),
    }
    if args.activity:
        activity = read_mhd(args.activity)
        if activity.shape != mu.shape:
            raise RuntimeError(f"Activity {activity.shape} does not match mu-map {mu.shape}")
        ct_coronal = np.percentile(target_hu, 90, axis=1)
        act_coronal = activity.max(1)
        ct_norm = np.clip((ct_coronal + 200) / 1200, 0, 1)
        act_norm = np.clip(act_coronal / max(np.percentile(act_coronal[act_coronal > 0], 99.5), 1e-12), 0, 1)
        fig, axes = plt.subplots(1, 3, figsize=(10, 7), sharex=True, sharey=True)
        axes[0].imshow(ct_coronal, origin="lower", cmap="gray", vmin=-200, vmax=1000, aspect="auto")
        axes[0].set_title("CT P90")
        axes[1].imshow(act_coronal, origin="lower", cmap="hot", vmin=0,
                       vmax=np.percentile(act_coronal[act_coronal > 0], 99.5), aspect="auto")
        axes[1].set_title("SPECT MIP")
        overlay = np.zeros((*ct_norm.shape, 3), dtype=float)
        overlay[..., 1] = ct_norm
        overlay[..., 0] = act_norm
        axes[2].imshow(overlay, origin="lower", aspect="auto")
        axes[2].set_title("overlay: CT green / SPECT red")
        fig.tight_layout()
        qa = args.output.with_name(args.output.stem + "_registration_qa.png")
        fig.savefig(qa, dpi=180, facecolor="white")
        plt.close(fig)
        manifest["registration_qa"] = str(qa)
    args.output.with_name(args.output.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def self_check() -> None:
    assert np.allclose(hu_to_mu(np.array([-1000., 0., 1000.]), .0152), [0, .0152, .0304])
    identity = {
        "col_dir": np.array([1., 0., 0.]),
        "row_dir": np.array([0., 1., 0.]),
        "normal": np.array([0., 0., 1.]),
        "origin": np.zeros(3),
    }
    coords = _patient_affine_coords(
        np.zeros(3), identity, np.ones(3), (2, 2, 2), identity, np.ones(3))
    assert np.allclose(coords, np.indices((2, 2, 2), dtype=np.float32))
    rotated = {
        "col_dir": np.array([0., 1., 0.]),
        "row_dir": np.array([-1., 0., 0.]),
        "normal": np.array([0., 0., 1.]),
        "origin": np.zeros(3),
    }
    coords = _patient_affine_coords(
        np.zeros(3), rotated, np.ones(3), (2, 2, 2), rotated, np.ones(3))
    assert np.allclose(coords, np.indices((2, 2, 2), dtype=np.float32))


if __name__ == "__main__":
    self_check()
    raise SystemExit(main())
