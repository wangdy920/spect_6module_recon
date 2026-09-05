#!/usr/bin/env python3
"""Three-arm BSREM+RDP convergence verification on the five frozen
prototype-parity datasets (2026-08-10 canonical_stitch commands).

Arms per dataset:
  mlem500   : -opti mlem,  500 iters, checkpoints every 50 (reference x*)
  bsrem_b0  : -opti bsrem, -reg none, 100 iters, ckpt every 10 (+diag)
              -> isolates BSREM step/clamp machinery at beta=0
  bsrem_rdp : -opti bsrem, algo_bsrem.conf calibration (RDP beta=0.3,
              delta=0.01, alpha0=1, gamma=0.05, kappa on), 100 iters (+diag)

Frozen geometry/PSF/flip arguments are reused verbatim from
recon_out/canonical_stitch_20260807/run_record.json with only path
rewrites (matrix moved off F:, config moved under multipinhole/).

Usage:
  python tools/run_bsrem_rdp_convergence.py --datasets derenzo_182222
  python tools/run_bsrem_rdp_convergence.py            # all five
  python tools/run_bsrem_rdp_convergence.py --analyze  # metrics only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_RECORD = os.path.join(ROOT, "recon_out", "canonical_stitch_20260807",
                          "run_record.json")
FROZEN_RUNS = os.path.join(ROOT, "recon_out", "canonical_stitch_20260807",
                           "runs")
OUT_ROOT = os.path.join(ROOT, "recon_out", "bsrem_rdp_convergence_20260817")
EXE = os.path.join(ROOT, "build", "verify-vs", "bin", "Release",
                   "SPECT_recon.exe")
ALGO_BSREM = os.path.join(ROOT, "multipinhole", "config", "algo_bsrem.conf")
ALGO_MLEM = os.path.join(ROOT, "multipinhole", "config", "algo_mlem.conf")

# The derenzo frozen command points at the production spect_config.ini,
# which sets recon_sections_override_cli=true and would wipe any CLI/algoConf
# regularization. Route it to an experiment copy with the protection off.
EXPERIMENT_INI = os.path.join(OUT_ROOT, "spect_config.experiment.ini")

PATH_REWRITES = [
    ("D:\\Projects\\SPECT-recon\\config\\spect_config.ini", EXPERIMENT_INI),
    ("D:\\Projects\\SPECT-recon\\multipinhole\\config\\spect_config.ini",
     EXPERIMENT_INI),
    ("D:\\Projects\\SPECT-recon\\config\\algo_mlem.conf", ALGO_MLEM),
    ("F:\\matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220",
     os.path.join(ROOT, "test_data", "6module",
                  "matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220")),
]


def ensure_experiment_ini():
    src = os.path.join(ROOT, "multipinhole", "config", "spect_config.ini")
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()
    patched = text.replace("recon_sections_override_cli = true",
                           "recon_sections_override_cli = false")
    if patched == text:
        raise RuntimeError("recon_sections_override_cli=true not found in "
                           f"{src}; refusing to guess")
    with open(EXPERIMENT_INI, "w", encoding="utf-8") as f:
        f.write(patched)

# Historical calibration from the accepted 2026-08-14 MDP five-arm run:
# RDP beta=0.3, rdp_gamma=0, delta=0.01; alpha0=1, gamma=0.05, kappa on.
# -reg is passed explicitly on the CLI because the production INI carries
# recon_sections_override_cli=true, which silently restores the INI's
# [Regularization] (type=None) over algo_bsrem.conf after parsing.
ARMS = {
    "mlem500": {
        "opti": "mlem", "algoConf": ALGO_MLEM, "it": "500:50",
        "extra": [], "expect_reg": None,
    },
    "bsrem_b0": {
        "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
        "extra": ["-reg", "none", "-bsremDiag", "-bsremDiagInterval", "1"],
        "expect_reg": {"enabled": False, "type": "None", "beta": 0.0},
    },
    "bsrem_rdp": {
        "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
        "extra": ["-reg", "rdp,0.3,0,0.01",
                  "-bsremDiag", "-bsremDiagInterval", "1"],
        "expect_reg": {"enabled": True, "type": "RDP", "beta": 0.3},
    },
}

# Acceleration sweep (user 2026-08-17: BSREM's purpose is faster convergence
# at matched iteration count). beta=0 arms with alpha0 > 1; the historical
# alpha0=1/gamma=0.05 schedule was measured to DECELERATE vs MLEM on derenzo.
# Not part of the default batch; select with --arms.
for _name, _a0, _gamma in [("bsrem_a15g0", 1.5, 0.0), ("bsrem_a2g0", 2.0, 0.0),
                           ("bsrem_a3g0", 3.0, 0.0), ("bsrem_a2g001", 2.0, 0.01)]:
    ARMS[_name] = {
        "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
        "extra": ["-reg", "none", "-bsrem", f"{_a0},{_gamma}",
                  "-bsremDiag", "-bsremDiagInterval", "1"],
        "expect_reg": {"enabled": False, "type": "None", "beta": 0.0},
    }

# Long-run counterpart of bsrem_a15g0: same alpha=1.5 / gamma=0 / beta=0,
# but 500 iters with ckpt/50 to sit on the same ladder as mlem500.
ARMS["bsrem_a15g0_500"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "500:50",
    "extra": ["-reg", "none", "-bsrem", "1.5,0",
              "-bsremDiag", "-bsremDiagInterval", "5"],
    "expect_reg": {"enabled": False, "type": "None", "beta": 0.0},
}

# Production candidate: clean acceleration point (alpha=1.5, verified
# 1.42-1.48x across all five datasets) combined with the historical RDP
# calibration. RDP additionally suppresses the low-sensitivity bed-edge
# voxel blowup seen in all unpenalized arms.
#
# 2026-08-18: without the curvature denominator this arm is UNSTABLE on
# derenzo (relChange stuck at 0.7-0.9, 2e5 voxels ping-ponging between both
# clamps): denom stays s while the numerator gains beta*grad_R, and alpha=1.5
# amplifies the overshoot around the *penalized* fixed point. The beta=0
# stability result does not transfer. Evidence preserved in
# derenzo_182222/bsrem_a15_rdp_nocurv. -bsremCurv on adds beta*D_reg^+ to the
# denominator (Ahn-Fessler), which is the mechanistically matched damping.
ARMS["bsrem_a15_rdp"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
    "extra": ["-reg", "rdp,0.3,0,0.01", "-bsrem", "1.5,0", "-bsremCurv", "on",
              "-bsremDiag", "-bsremDiagInterval", "1"],
    "expect_reg": {"enabled": True, "type": "RDP", "beta": 0.3},
}

# 2026-08-18 pilots: curvature cannot damp this RDP calibration (delta=0.01 is
# far below the image scale, so psi'' ~ 0 exactly where the gradient
# saturates). Factor split for the alpha=1.5 penalized instability:
# light decay / kappa off / historical decay with the faster alpha0.
_RDP = ["-reg", "rdp,0.3,0,0.01", "-bsremDiag", "-bsremDiagInterval", "1"]
_EXPECT_RDP = {"enabled": True, "type": "RDP", "beta": 0.3}
ARMS["a15g002_rdp"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
    "extra": _RDP + ["-bsrem", "1.5,0.02"], "expect_reg": _EXPECT_RDP,
}
ARMS["a15g0_rdp_nokappa"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
    "extra": _RDP + ["-bsrem", "1.5,0", "-bsremKappa", "off"],
    "expect_reg": _EXPECT_RDP,
}
ARMS["a15g005_rdp"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
    "extra": _RDP + ["-bsrem", "1.5,0.05"], "expect_reg": _EXPECT_RDP,
}

# 2026-08-18 beta ladder: all alpha>=1.5 schedules stay polluted at beta=0.3
# (which over-smooths derenzo anyway, C.2). Test whether a task-scale beta is
# stable under the adopted acceleration schedule.
for _bname, _beta in [("a15g0_b01", 0.1), ("a15g0_b003", 0.03)]:
    ARMS[_bname] = {
        "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
        "extra": ["-reg", f"rdp,{_beta},0,0.01", "-bsrem", "1.5,0",
                  "-bsremDiag", "-bsremDiagInterval", "1"],
        "expect_reg": {"enabled": True, "type": "RDP", "beta": _beta},
    }
ARMS["rdp_b01"] = {  # control: historical schedule at beta=0.1
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
    "extra": ["-reg", "rdp,0.1,0,0.01", "-bsrem", "1,0.05",
              "-bsremDiag", "-bsremDiagInterval", "1"],
    "expect_reg": {"enabled": True, "type": "RDP", "beta": 0.1},
}
# C.2 beta ladder on the STABLE schedule (user metrics 2026-08-18: rod
# separation + per-rod axial uniformity). beta=0.3 over-smooths derenzo;
# these give clean quality points at 0.05 / 0.03.
for _bname, _beta in [("rdp_b005", 0.05), ("rdp_b003", 0.03)]:
    ARMS[_bname] = {
        "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
        "extra": ["-reg", f"rdp,{_beta},0,0.01", "-bsrem", "1,0.05",
                  "-bsremDiag", "-bsremDiagInterval", "1"],
        "expect_reg": {"enabled": True, "type": "RDP", "beta": _beta},
    }
# C.3 validation: sensitivity floor freeze on the adopted acceleration arm.
# Expect: internal bed-edge blowup contained (roiSum/image_max sane, clamp-
# cold), saved stitched volume trajectory unchanged vs bsrem_a15g0.
ARMS["a15g0_floor"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
    "extra": ["-reg", "none", "-bsrem", "1.5,0", "-bsremSensFloor", "1e-4",
              "-bsremDiag", "-bsremDiagInterval", "1"],
    "expect_reg": {"enabled": False, "type": "None", "beta": 0.0},
}

# Last mechanistic rescue for alpha=1.5+RDP: LIVE curvature (the earlier curv
# arm froze D_reg at it2 on a flat image, a confound).
ARMS["a15_b01_livecurv"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "100:10",
    "extra": ["-reg", "rdp,0.1,0,0.01", "-bsrem", "1.5,0",
              "-bsremCurv", "on", "-bsremCurvFreeze", "off",
              "-bsremDiag", "-bsremDiagInterval", "1"],
    "expect_reg": {"enabled": True, "type": "RDP", "beta": 0.1},
}

# Phase C.4 (2026-08-19): MLEM 1000 baseline + three 1000-iter BSREM recipes.
# New directories; do not overwrite mlem500 / bsrem_a15g0_500.
ARMS["mlem1000"] = {
    "opti": "mlem", "algoConf": ALGO_MLEM, "it": "1000:50",
    "extra": [], "expect_reg": None,
}
ARMS["a15g0_1000"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "1000:50",
    "extra": ["-reg", "none", "-bsrem", "1.5,0",
              "-bsremDiag", "-bsremDiagInterval", "5"],
    "expect_reg": {"enabled": False, "type": "None", "beta": 0.0},
}
ARMS["rdp_b03_1000"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "1000:50",
    "extra": ["-reg", "rdp,0.3,0,0.01", "-bsrem", "1,0.05",
              "-bsremDiag", "-bsremDiagInterval", "5"],
    "expect_reg": {"enabled": True, "type": "RDP", "beta": 0.3},
}
ARMS["rdp_b003_1000"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "1000:50",
    "extra": ["-reg", "rdp,0.03,0,0.01", "-bsrem", "1,0.05",
              "-bsremDiag", "-bsremDiagInterval", "5"],
    "expect_reg": {"enabled": True, "type": "RDP", "beta": 0.03},
}
ARMS["rdp_b01_1000"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "1000:50",
    "extra": ["-reg", "rdp,0.1,0,0.01", "-bsrem", "1,0.05",
              "-bsremDiag", "-bsremDiagInterval", "5"],
    "expect_reg": {"enabled": True, "type": "RDP", "beta": 0.1},
}
ARMS["rdp_b005_1000"] = {
    "opti": "bsrem", "algoConf": ALGO_BSREM, "it": "1000:50",
    "extra": ["-reg", "rdp,0.05,0,0.01", "-bsrem", "1,0.05",
              "-bsremDiag", "-bsremDiagInterval", "5"],
    "expect_reg": {"enabled": True, "type": "RDP", "beta": 0.05},
}

MLEM1000_ARMS = ["mlem1000"]
BSREM1000_ARMS = ["a15g0_1000", "rdp_b03_1000", "rdp_b003_1000"]
FALLBACK1000_ARMS = ["rdp_b01_1000", "rdp_b005_1000"]
DEFAULT_ARMS = [name for name in ARMS
                if name not in (MLEM1000_ARMS + BSREM1000_ARMS
                                + FALLBACK1000_ARMS)]

# Args owned by the arm definition; strip from the frozen command.
# -reg is always stripped: the frozen commands carry "-reg none", which
# would override the RDP settings loaded from algo_bsrem.conf.
STRIP_WITH_VALUE = {"-fout", "-it", "-opti", "-algoConf", "-reg"}


def load_frozen_commands():
    with open(RUN_RECORD, "r", encoding="utf-8") as f:
        record = json.load(f)
    commands = {}
    for run in record["runs"]:
        commands[run["dataset"]] = run["command"]
    return commands


def build_command(frozen, dataset, arm_name):
    arm = ARMS[arm_name]
    out_dir = os.path.join(OUT_ROOT, dataset, arm_name)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [EXE]
    i = 1
    while i < len(frozen):
        tok = frozen[i]
        if tok in STRIP_WITH_VALUE:
            i += 2
            continue
        rewritten = tok
        for old, new in PATH_REWRITES:
            rewritten = rewritten.replace(old, new)
        cmd.append(rewritten)
        i += 1
    cmd += ["-algoConf", arm["algoConf"], "-opti", arm["opti"],
            "-it", arm["it"], "-fout", os.path.join(out_dir, "recon"),
            "-vb", "1"]
    cmd += arm["extra"]
    return cmd, out_dir


def run_arm(frozen, dataset, arm_name):
    cmd, out_dir = build_command(frozen, dataset, arm_name)
    log_path = os.path.join(out_dir, "console.log")
    final_it = int(ARMS[arm_name]["it"].split(":")[0])
    final_vol = os.path.join(out_dir, f"recon_it{final_it}.mhd")
    if os.path.exists(final_vol):
        print(f"[skip] {dataset}/{arm_name}: {final_vol} exists", flush=True)
        return verify_manifest(dataset, arm_name, out_dir)
    print(f"[run ] {dataset}/{arm_name}", flush=True)
    env = dict(os.environ)
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(subprocess.list2cmdline(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=ROOT, env=env)
    elapsed = time.time() - start
    perf = ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as log:
        for line in log:
            if "[PERF]" in line:
                perf = line.strip()
    print(f"[done] {dataset}/{arm_name}: exit={proc.returncode} "
          f"wall={elapsed:.0f}s", flush=True)
    if perf:
        print("       " + perf, flush=True)
    if proc.returncode != 0:
        return False
    return verify_manifest(dataset, arm_name, out_dir)


def verify_manifest(dataset, arm_name, out_dir):
    """Fail fast if the run's effective regularization does not match the
    arm definition (guards against INI-side silent overrides)."""
    expect = ARMS[arm_name]["expect_reg"]
    if expect is None:
        return True
    manifest_path = os.path.join(out_dir, "recon_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    bsrem = manifest["algorithm"]["bsrem"]
    ok = (bsrem["regularization_enabled"] == expect["enabled"]
          and bsrem["regularizer_type"] == expect["type"])
    # beta is only meaningful when regularization is enabled; disabled runs
    # can carry a leftover algo-conf beta in the manifest.
    if ok and expect["enabled"]:
        ok = abs(bsrem["regularizer_beta"] - expect["beta"]) < 1e-12
    if not ok:
        print(f"[FAIL] {dataset}/{arm_name}: manifest regularization "
              f"mismatch: enabled={bsrem['regularization_enabled']} "
              f"type={bsrem['regularizer_type']} "
              f"beta={bsrem['regularizer_beta']}, expected {expect}",
              flush=True)
    return ok


def read_mhd_volume(mhd_path):
    dims = None
    dtype = np.float64
    data_file = None
    with open(mhd_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "DimSize":
                dims = tuple(int(v) for v in value.split())
            elif key == "ElementType":
                dtype = {"MET_DOUBLE": np.float64,
                         "MET_FLOAT": np.float32}[value]
            elif key == "ElementDataFile":
                data_file = value
    raw = os.path.join(os.path.dirname(mhd_path), data_file)
    vol = np.fromfile(raw, dtype=dtype)
    expected = int(np.prod(dims))
    if vol.size != expected:
        raise ValueError(f"{raw}: {vol.size} elements, expected {expected}")
    return vol.astype(np.float64)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def ncc(a, b):
    """Pearson correlation of two flattened volumes (same voxels as relL2)."""
    aa = np.ravel(a)
    bb = np.ravel(b)
    a0 = aa - aa.mean()
    b0 = bb - bb.mean()
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    return float(np.dot(a0, b0) / max(denom, 1e-300))


def list_checkpoints(out_dir):
    pat = re.compile(r"recon_it(\d+)\.mhd$")
    found = []
    for name in os.listdir(out_dir):
        m = pat.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(out_dir, name)))
    return sorted(found)


def analyze(datasets):
    summary = {}
    for dataset in datasets:
        entry = {}
        ds_dir = os.path.join(OUT_ROOT, dataset)
        ref500_path = os.path.join(ds_dir, "mlem500", "recon_it500.mhd")
        ref1000_path = os.path.join(ds_dir, "mlem1000", "recon_it1000.mhd")
        ref500 = read_mhd_volume(ref500_path) if os.path.exists(ref500_path) else None
        ref1000 = (read_mhd_volume(ref1000_path)
                   if os.path.exists(ref1000_path) else None)
        if ref500 is None and ref1000 is None:
            print(f"[analyze] {dataset}: no mlem500 or mlem1000 reference yet")
            continue

        frozen50 = os.path.join(FROZEN_RUNS, dataset, "recon_it50.mhd")
        fresh50 = os.path.join(ds_dir, "mlem500", "recon_it50.mhd")
        if os.path.exists(frozen50) and os.path.exists(fresh50):
            entry["parity_it50_relL2_vs_frozen"] = rel_l2(
                read_mhd_volume(fresh50), read_mhd_volume(frozen50))
        mlem1000_it500 = os.path.join(ds_dir, "mlem1000", "recon_it500.mhd")
        if ref500 is not None and os.path.exists(mlem1000_it500):
            entry["parity_it500_relL2_mlem1000_vs_mlem500"] = rel_l2(
                read_mhd_volume(mlem1000_it500), ref500)

        for arm in ARMS:
            arm_dir = os.path.join(ds_dir, arm)
            if not os.path.isdir(arm_dir):
                continue
            curve = {}
            prev = None
            for it, path in list_checkpoints(arm_dir):
                vol = read_mhd_volume(path)
                point = {}
                if ref500 is not None:
                    point["relL2_to_mlem500"] = rel_l2(vol, ref500)
                    point["ncc_to_mlem500"] = ncc(vol, ref500)
                if ref1000 is not None:
                    point["relL2_to_mlem1000"] = rel_l2(vol, ref1000)
                    point["ncc_to_mlem1000"] = ncc(vol, ref1000)
                if prev is not None:
                    point["relL2_to_prev_ckpt"] = rel_l2(vol, prev[1])
                    point["prev_ckpt_it"] = prev[0]
                curve[it] = point
                prev = (it, vol)
            entry[arm] = curve
        summary[dataset] = entry
    out_path = os.path.join(OUT_ROOT, "convergence_summary.json")
    os.makedirs(OUT_ROOT, exist_ok=True)
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(summary)
        summary = existing
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"[analyze] wrote {out_path}")

    for dataset, entry in summary.items():
        print(f"\n=== {dataset} ===")
        if "parity_it50_relL2_vs_frozen" in entry:
            print(f"  parity it50 vs frozen: "
                  f"{entry['parity_it50_relL2_vs_frozen']:.3e}")
        if "parity_it500_relL2_mlem1000_vs_mlem500" in entry:
            print(f"  parity it500 mlem1000 vs mlem500: "
                  f"{entry['parity_it500_relL2_mlem1000_vs_mlem500']:.3e}")
        for arm in ARMS:
            if arm not in entry or not entry[arm]:
                continue
            pts = sorted(entry[arm].items())
            if pts and "relL2_to_mlem1000" in pts[0][1]:
                rel = ", ".join(
                    f"it{it}={vals['relL2_to_mlem1000']:.4f}"
                    for it, vals in pts)
                print(f"  {arm}: relL2 to mlem1000: {rel}")
            elif pts and "relL2_to_mlem500" in pts[0][1]:
                rel = ", ".join(
                    f"it{it}={vals['relL2_to_mlem500']:.4f}"
                    for it, vals in pts)
                print(f"  {arm}: relL2 to mlem500: {rel}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    frozen_commands = load_frozen_commands()
    datasets = args.datasets or list(frozen_commands.keys())
    for dataset in datasets:
        if dataset not in frozen_commands:
            sys.exit(f"unknown dataset '{dataset}'; "
                     f"have {sorted(frozen_commands)}")

    if args.analyze:
        analyze(datasets)
        return

    ensure_experiment_ini()
    for dataset in datasets:
        for arm_name in args.arms:
            if not run_arm(frozen_commands[dataset], dataset, arm_name):
                sys.exit(f"{dataset}/{arm_name} failed; stopping")
    analyze(datasets)


if __name__ == "__main__":
    main()
