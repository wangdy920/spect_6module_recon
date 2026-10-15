"""Native CLI regression of measured-count normalization and residuals.

Uses tiny all-one matrices, whose EM update has an independent scalar oracle.
This checks pipeline composition, not scanner calibration or image quality.
Requires CUDA and the existing numpy/converter dependencies.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from convert_csrb_to_csrq import convert_chunk


def projection(path, values):
    path.write_bytes(struct.pack("<ii", 8, 8) + np.asarray(values, "<f8").tobytes())


def matrix(path, zfold, csrq):
    path.mkdir(parents=True, exist_ok=True)
    compact_z = 2 if zfold else 4
    rows = 2 * 2 * compact_z
    metadata = struct.pack("<11if2i3f5i", rows, 64, 64, 2, 2, compact_z,
                           2 * compact_z, 8, 4, 4, 4, 1.0, 2, 4,
                           1.0, 50.0, 2.0, 1, int(zfold), 0, 90, 4)
    (path / "metadata.bin").write_bytes(metadata)
    chunk = path / "chunk_000.csrb"
    chunk.write_bytes(struct.pack("<IIiiiQ", 0x42525343, 1, rows, 0, 2, rows * 64)
                      + np.arange(0, (rows + 1) * 64, 64, dtype="<i8").tobytes()
                      + np.tile(np.arange(64, dtype="<i4"), rows).tobytes()
                      + np.ones(rows * 64, dtype="<f4").tobytes())
    if csrq:
        target = path.parent / (path.name + "q")
        target.mkdir(exist_ok=True)
        (target / "metadata.bin").write_bytes(metadata)
        convert_chunk(chunk, target / "chunk_000.csrq", 64, 1024)
        return target
    return path


def volume(path):
    fields = dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
                  if "=" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    dtype = {"MET_FLOAT": "<f4", "MET_DOUBLE": "<f8"}[fields["ElementType"]]
    return np.fromfile(path.parent / fields["ElementDataFile"], dtype=dtype)


def run_case(exe, directory, mat, joint, censor, subsets, background, weighted, beds):
    directory.mkdir(parents=True, exist_ok=True)
    index = np.arange(64)
    valid = ((index % 8 // 2 >= censor) & (index % 8 // 2 < 4 - censor))
    weights = [0.7, 0.3] if joint else [1.0]
    efficiencies, observations, backgrounds = [], [], []
    config = ("[InputOutput]\nmatrix_grid_output=true\n[Mask]\nuse_cylinder=false\n"
              "[PSF]\nmode=none\n[Filtering]\nbutter_apply_stage=none\n"
              "smooth_apply_stage=none\n[MultiBed]\nz_bed_flip=0\n"
              "z_out_flip=0\nx_out_flip=0\nbed_step_mm=4\n")
    norms = ""
    for channel, weight in enumerate(weights):
        efficiency = 0.4 + (index % (7 + channel)) * 0.13 if weighted else np.ones(64)
        b = np.array([2.0 + index % 5 + bed for bed in range(beds)]) if background else np.zeros((beds, 64))
        y = np.array([efficiency * weight * 64 * (1.5 + index % 3 * 0.2 + bed * 0.3)
                      for bed in range(beds)]) + b
        efficiencies.append(efficiency)
        observations.append(y)
        backgrounds.append(b)
        for bed in range(beds):
            projection(directory / f"Bed{bed + 1}_{channel}.bin", y[bed])
            projection(directory / f"Bed{bed + 1}_bg{channel}.bin", b[bed])
        if weighted:
            norm = directory / f"norm{channel}.bin"
            projection(norm, 1.0 / efficiency)
            norms += f"factor_file{'2' if channel else ''}_Lu177={norm.as_posix()}\n"
    # Normalization is bound only by the global config, never by the recipe.
    conf = directory / "input.ini"
    conf.write_text(config + ("[Normalization]\n" + norms if norms else ""), encoding="utf-8")
    recipe = directory / "recipe.ini"
    recipe.write_text(config, encoding="utf-8")
    command = [str(exe), "-config", str(conf), "-algoConf", str(recipe), "-isotope", "Lu177",
               "-chunkCSRDir", str(mat), "-dfList",
               ";".join(str(directory / f"Bed{b + 1}_0.bin") for b in range(beds)),
               "-opti", "mlem", "-it", f"2:{subsets}", "-dim", "4,4,4",
               "-vox", "1", "-ang", "8", "-detN", "4,2", "-psfMode", "none", "-vb", "1",
               "-detZEdgeCensor", str(censor), "-projectionResidual", "1",
               "-fout", str(directory / "recon")]
    if joint:
        command += ["-chunkCSRDir2", str(mat), "-dfList2",
                    ";".join(str(directory / f"Bed{b + 1}_1.bin") for b in range(beds)),
                    "-energyWeights", "0.7,0.3"]
    if background:
        command += ["-bProjList", ";".join(str(directory / f"Bed{b + 1}_bg0.bin") for b in range(beds))]
        if joint:
            command += ["-bProjList2", ";".join(str(directory / f"Bed{b + 1}_bg1.bin") for b in range(beds))]
    (directory / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (directory / "console.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if completed.returncode:
        raise AssertionError(f"native exit {completed.returncode}: {directory / 'console.log'}")
    x = np.ones(beds)
    for _ in range(2):
        for subset in range(subsets):
            selected = valid & ((index // 8) % subsets == subset)
            q, s = 0.0, 0.0
            for e, y, b, w in zip(efficiencies, observations, backgrounds, weights):
                h = w * e[selected]
                q += np.sum(h * y[:, selected] / (h * 64 * x[:, None] + b[:, selected]), axis=1)
                s += np.sum(h)
            x *= q / s
    actual = volume(directory / "recon_it2.mhd").reshape(beds, 64)
    np.testing.assert_allclose(actual, np.repeat(x[:, None], 64, axis=1), rtol=1e-5, atol=1e-7)
    max_error = 0.0
    with (directory / "recon_projection_residual.csv").open(encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    assert len(records) == 32 * len(weights) * beds
    for row in records:
        k, a, z = (int(row[key]) for key in ("window", "angle", "det_z"))
        bed = int(row["bed"])
        sl = slice(a * 8 + z * 2, a * 8 + z * 2 + 2)
        keep = valid[sl]
        y = observations[k][bed, sl][keep]
        mean = (efficiencies[k][sl] * weights[k] * 64 * x[bed] + backgrounds[k][bed, sl])[keep]
        assert int(row["count"]) == len(y), row
        expected = float(mean.sum())
        max_error = max(max_error, abs(float(row["expected_sum"]) - expected))
        np.testing.assert_allclose(float(row["expected_sum"]), expected, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(float(row["measured_sum"]), y.sum(), rtol=1e-10, atol=1e-8)
        deviance = np.mean(2 * (mean - y + y * np.log(y / mean))) if len(y) else 0.0
        np.testing.assert_allclose(float(row["poisson_deviance_per_bin"]), deviance,
                                   rtol=1e-4, atol=2e-5)
    return {"case": directory.name, "max_residual_sum_abs_error": max_error,
            "image_relative_error": float(np.linalg.norm(actual - x[:, None]) / (np.linalg.norm(x) * 8))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for zfold in (False, True):
        for csrq in (False, True):
            mat = matrix(output / f"matrix_{int(zfold)}", zfold, csrq)
            for joint, censor, subsets, weighted in ((False, 0, 1, False),
                    (False, 1, 2, True), (True, 0, 2, True), (True, 1, 1, True)):
                for beds in (1, 2):
                    name = f"k{8 if zfold else 4}_{'csrq' if csrq else 'csrb'}_j{int(joint)}_m{censor}_n{subsets}_beds{beds}"
                    results.append(run_case(args.exe.resolve(), output / name, mat,
                                            joint, censor, subsets, args.background, weighted, beds))
                    print(f"PASS {name}", flush=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"PASS {len(results)} native observation-model cases")
    # Reuse one native case's inputs, but verify the PSF/ROI diagnostic against
    # an independent separable convolution of the saved operator-mode state.
    base = output / "k4_csrb_j0_m0_n1_beds1"
    psf_dir = output / "psf_mask"
    psf_dir.mkdir(exist_ok=True)
    command = json.loads((base / "command.json").read_text(encoding="utf-8"))
    command[command.index("-psfMode") + 1] = "operator"
    command[command.index("-fout") + 1] = str(psf_dir / "recon")
    command += ["-psfKernel", "adjoint", "-psf", "1.5,3", "-cyl", "2.5,2.5"]
    (psf_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (psf_dir / "console.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
    blurred = volume(psf_dir / "recon_it2.mhd").reshape(4, 4, 4)
    sigma = 1.5 / 2.354820045
    radius = int(np.ceil(3 * sigma))
    kernel = np.exp(-.5 * (np.arange(-radius, radius + 1) / sigma)**2)
    kernel /= kernel.sum()
    for axis in range(3):
        blurred = np.apply_along_axis(lambda row: np.convolve(
            np.pad(row, (radius, radius), mode="reflect"), kernel, mode="valid"), axis, blurred)
    mask = volume(psf_dir / "recon_mask.mhd").reshape(4, 4, 4) > 0
    predicted = float(blurred[mask].sum())
    assert blurred[~mask].sum() > .01 * predicted, "fixture must expose missing post-PSF mask"
    with (psf_dir / "recon_projection_residual.csv").open(encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    b = 2.0 + np.arange(64) % 5 if args.background else np.zeros(64)
    expected = predicted * 2 + b.reshape(32, 2).sum(axis=1)
    np.testing.assert_allclose([float(r["expected_sum"]) for r in records], expected, rtol=1e-5)
    print("PASS operator PSF + ROI residual closure")


if __name__ == "__main__":
    main()
