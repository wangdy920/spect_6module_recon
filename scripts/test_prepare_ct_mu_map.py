"""CPU-only CT preparation regression; synthetic data, no calibrated device claim.

Run: python scripts/test_prepare_ct_mu_map.py [--output <evidence directory>]
"""
import argparse
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

import prepare_ct_mu_map as prep


def write_series(root, volume, spacing, frame_uid):
    root.mkdir(parents=True)
    series_uid = generate_uid()
    for z, plane in enumerate(volume):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.Rows, ds.Columns = plane.shape
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [10, -20, 30 + z * spacing]
        ds.PixelSpacing = [spacing, spacing]
        ds.SliceThickness = spacing
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = np.asarray(plane, dtype="<i2").tobytes()
        ds.save_as(root / f"{z:04}.dcm", enforce_file_format=True)


def check(root):
    prep.self_check()
    summaries = []
    cases = [("sc60_single", (240, 90, 90), 0.4, [1], 0.0, False, True),
             ("multibed", (12, 8, 10), 0.4, [2, 3], 2.4, True, True),
             ("legacy_psc", (160, 108, 108), 0.3, [1], 16.8, False, False)]
    for name, (nz, ny, nx), mm, beds, step_mm, flip, use_manifest in cases:
        directory = root / name
        directory.mkdir(parents=True)
        step = round(step_mm / mm) if len(beds) > 1 else nz
        shape = (nz + (len(beds) - 1) * step, ny, nx)
        z, y, x = np.indices(shape)
        hu = (-500 + 3 * z + 2 * y + x).astype(np.int16)
        frame = generate_uid()
        write_series(directory / "ct", hu, mm, frame)
        write_series(directory / "spect", np.ones(shape, dtype=np.int16), mm, frame)
        manifest = {
            "schema": "spect_recon_run_manifest_v1",
            "matrix": {"dims": [nx, ny, nz], "voxel_size_mm": mm},
            "projection": {str(b): f"bed{b}.bin" for b in beds} | {"z_bed_flip": flip},
            "multibed": {"bed_count": len(beds), "bed_z_starts": [i * nz for i in range(len(beds))],
                         "bed_step_layers": step, "bed_step_mm": step_mm,
                         "output_bed_overlaps": [nz - step] * (len(beds) - 1),
                         "matrix_grid_output": True},
            "output": {"dimensions": [nx, ny, shape[0]], "voxel_mm": [mm] * 3,
                       "x_out_flip": False, "z_out_flip": False, "z_per_bed_flip": False,
                       "xy_out_rotation_deg": 0, "seam_xy_align": False,
                       "crop_end_start": 0, "crop_end_end": 0}}
        manifest_path = directory / "recon_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        argv = ["prepare_ct_mu_map.py", "--ct-dir", str(directory / "ct"),
                "--spect-dicom-dir", str(directory / "spect"),
                "--output", str(directory / "mu.mhd"),
                "--hu-output", str(directory / "hu.mhd"),
                "--attenuation-mu-output", str(directory / "att.mhd"),
                "--regct-hu-output", str(directory / "regct.mhd")]
        argv += (["--recon-manifest", str(manifest_path)] if use_manifest else ["--beds", "1"])
        with patch.object(sys, "argv", argv):
            assert prep.main() == 0
        actual = prep.read_mhd(directory / "hu.mhd")
        # Float32 resampling coordinates allow < 0.001 HU of roundoff, not a spatial tolerance.
        hu_error = float(np.max(np.abs(actual - hu)))
        assert actual.shape == hu.shape and hu_error < 1e-3
        expected_mu = 0.0152 * np.clip(1.0 + hu.astype(float) / 1000.0, 0, 3)
        expected_stack = np.concatenate([hu[i * step:i * step + nz] for i in range(len(beds))])
        expected_att = np.concatenate([
            expected_mu[i * step:i * step + nz][::-1 if flip else 1]
            for i in range(len(beds))])
        assert np.allclose(prep.read_mhd(directory / "regct.mhd"), expected_stack, rtol=0, atol=1e-3)
        assert np.allclose(prep.read_mhd(directory / "att.mhd"), expected_att, rtol=0, atol=2e-7)
        generated = json.loads((directory / "mu_manifest.json").read_text(encoding="utf-8"))
        assert np.allclose(generated["device_geometry"]["voxel_xyz_mm"], mm)
        assert generated["bed_step_mm"] == step_mm
        summaries.append({"case": name, "shape_zyx": list(shape), "max_hu_error": hu_error})
        if use_manifest:
            if len(beds) == 1:
                direct_single = deepcopy(manifest)
                direct_single["projection"]["z_bed_flip"] = True
                direct_path = directory / "single_direct_manifest.json"
                direct_path.write_text(json.dumps(direct_single), encoding="utf-8")
                assert not prep.load_recon_grid(direct_path)["z_bed_flip"]
            for section, key, value in [
                ("output", "xy_out_rotation_deg", 90), ("output", "x_out_flip", True),
                ("output", "z_out_flip", True), ("output", "z_per_bed_flip", True),
                ("output", "crop_end_start", 1), ("output", "dimensions", [nx, ny, 1]),
                ("output", "voxel_mm", [0, mm, mm]), ("multibed", "bed_step_mm", float("nan")),
                ("multibed", "matrix_grid_output", False)]:
                invalid = deepcopy(manifest)
                invalid[section][key] = value
                bad = directory / "invalid_manifest.json"
                bad.write_text(json.dumps(invalid), encoding="utf-8")
                try:
                    prep.load_recon_grid(bad)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f"accepted invalid {section}.{key}")
            mismatch = deepcopy(manifest)
            mismatch["output"]["voxel_mm"] = [0.8] * 3
            mismatch["matrix"]["voxel_size_mm"] = 0.8
            # Valid manifest, wrong DICOM spacing: fail before producing any map.
            if len(beds) == 1:
                bad.write_text(json.dumps(mismatch), encoding="utf-8")
                with patch.object(sys, "argv", argv[:-1] + [str(bad)]):
                    try:
                        prep.main()
                    except RuntimeError as exc:
                        assert "spacing" in str(exc)
                    else:
                        raise AssertionError("accepted DICOM/manifest spacing mismatch")
    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        check(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="spect_ct_grid_") as tmp:
            check(Path(tmp))
