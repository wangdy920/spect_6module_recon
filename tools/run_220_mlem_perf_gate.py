#!/usr/bin/env python3
"""Frozen Lu-177 208 keV CSRB five-bed MLEM50 performance gate (2026-09-05).

Drives the existing SPECT_recon.exe. Does not add a second reconstruction
path. Primary metric is log [PERF] iteration_core_s (target 120 s, floor
180 s). Same-input primary relL2 gate is 1e-5. Use -vb 1: debug verbosity
disables GPU-resident MLEM and invalidates the timing.

Refuse to start if SPECT_recon.exe is already running.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
import prep_compact_denom_cache as denom  # noqa: E402
import search_compact_perf as perf  # noqa: E402

REPO_ROOT = denom.REPO_ROOT
CANONICAL_MATRIX = (
    REPO_ROOT / "test_data" / "6module" / "Lu177_sysmat" /
    "matrix_PSC_PIN26x28x20_208keV_120x120x220_ZFOLD")
# Existing five-bed count vectors include both Lu windows. This is a frozen
# performance/parity workload, not a single-photopeak quantitative endpoint.
DEFAULT_BEDS = Path(
    "E:/Xiang_Project/SPECT/6Module/experiments/recon_param_study/pre/mouse_normal_lu")
EXPECTED_XYZ = (120, 120, 220)
EXPECTED_CHUNKS = 6
EXPECTED_NNZ = 4399851643
TARGET_ITERATION_CORE_S = 120.0
FLOOR_ITERATION_CORE_S = 180.0

PERF_RE = re.compile(
    r"\[PERF\] denom_cache=(hit|miss|disabled)\s+"
    r"projection_load_s=([0-9.]+)\s+"
    r"csr_open_s=([0-9.]+)\s+"
    r"denom_and_matrix_prepare_s=([0-9.]+)\s+"
    r"iteration_core_s=([0-9.]+)\s+"
    r"load_to_iteration_done_s=([0-9.]+)\s+"
    r"full_command_s=([0-9.]+)")
IDENTITY_RE = re.compile(r"=== SPECT_recon ([^\s=]+) ===")
NVRTC_RE = re.compile(r"NVRTC compile", re.IGNORECASE)


def spect_recon_running() -> bool:
    if os.name == "nt":
        completed = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq SPECT_recon.exe"],
            check=False, capture_output=True, text=True, encoding="utf-8",
            errors="replace")
        return "SPECT_recon.exe" in completed.stdout
    completed = subprocess.run(
        ["pgrep", "-f", "SPECT_recon"], check=False, capture_output=True,
        text=True)
    return bool(completed.stdout.strip())


def default_bed_list() -> list[Path]:
    paths = [DEFAULT_BEDS / f"Bed{i}.bin" for i in range(1, 6)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "five-bed Lu-177 mouse projections missing (1-bed cannot close "
            f"this gate): {names}")
    return paths


def inspect_matrix(matrix: Path) -> dict[str, Any]:
    meta_txt = (matrix / "metadata.txt").read_text(
        encoding="utf-8", errors="replace")
    chunks = sorted(matrix.glob("chunk_*.csrb"))
    payload = sum(path.stat().st_size for path in chunks)
    nnz = None
    stats_txt = matrix / "correction_stats.txt"
    extra_lines = meta_txt.splitlines()
    if stats_txt.is_file():
        extra_lines.extend(
            stats_txt.read_text(encoding="utf-8", errors="replace").splitlines())
    for line in extra_lines:
        if line.startswith("voxel_z_expansion_output_nnz=") or line.startswith(
                "total_nnz="):
            nnz = int(line.split("=", 1)[1])
    bin_meta = denom.read_compact_metadata(matrix / "metadata.bin")
    info = {
        "path": str(matrix),
        "nx": int(bin_meta["Nx"]),
        "ny": int(bin_meta["Ny"]),
        "nz": int(bin_meta["Nz"]),
        "chunk_count": len(chunks),
        "nnz": nnz,
        "payload_bytes": payload,
    }
    if (info["nx"], info["ny"], info["nz"]) != EXPECTED_XYZ:
        raise RuntimeError(
            f"matrix geometry {info['nx']}x{info['ny']}x{info['nz']} != "
            f"{EXPECTED_XYZ[0]}x{EXPECTED_XYZ[1]}x{EXPECTED_XYZ[2]}")
    if info["chunk_count"] != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"matrix chunks {info['chunk_count']} != {EXPECTED_CHUNKS}")
    if nnz != EXPECTED_NNZ:
        raise RuntimeError(f"matrix nnz {nnz} != {EXPECTED_NNZ}")
    return info


def parse_perf(text: str) -> dict[str, Any]:
    metrics = perf.parse_log_metrics(text)
    match = PERF_RE.search(text)
    if match:
        metrics["denom_cache"] = match.group(1)
        metrics["projection_load_s"] = float(match.group(2))
        metrics["csr_open_s"] = float(match.group(3))
        metrics["denom_and_matrix_prepare_s"] = float(match.group(4))
        metrics["iteration_core_s"] = float(match.group(5))
        metrics["load_to_iteration_done_s"] = float(match.group(6))
        metrics["full_command_s"] = float(match.group(7))
    identity = IDENTITY_RE.search(text)
    if identity:
        metrics["exe_identity"] = identity.group(1)
    metrics["nvrtc_compile"] = bool(NVRTC_RE.search(text))
    return metrics


def classify_gate(iteration_core_s: float | None,
                  denom_cache: str | None) -> str:
    if iteration_core_s is None:
        return "unmeasured"
    if denom_cache != "hit":
        return "warmup_not_gate"
    if iteration_core_s <= TARGET_ITERATION_CORE_S:
        return "target_met"
    if iteration_core_s <= FLOOR_ITERATION_CORE_S:
        return "floor_met"
    return "below_floor"


def build_command(*, exe: Path, config: Path, matrix: Path, beds: list[Path],
                  fout: Path, iterations: int, extra: list[str]) -> list[str]:
    return [
        str(exe),
        "-config", str(config),
        "-opti", "mlem",
        "-it", f"{iterations}:{iterations}",
        "-vb", "1",
        "-chunkCSRDir", str(matrix),
        "-dfList", ";".join(str(path) for path in beds),
        "-fout", str(fout),
        *extra,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Repository root examples:

  python tools/run_220_mlem_perf_gate.py --dry-run
  python tools/run_220_mlem_perf_gate.py --it 3 --stage warmup
  python tools/run_220_mlem_perf_gate.py --stage baseline
  python tools/run_220_mlem_perf_gate.py --stage candidate --baseline-mhd <frozen_it50.mhd>
""")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=denom.DEFAULT_CONFIG)
    parser.add_argument("--chunkCSRDir", type=Path, default=CANONICAL_MATRIX)
    parser.add_argument("--dfList", default=None)
    parser.add_argument("--it", type=int, default=50)
    parser.add_argument("--stage", default="baseline")
    parser.add_argument("--baseline-mhd", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.it <= 0:
        raise RuntimeError("--it must be positive")
    work_dir = args.work_dir.expanduser().resolve()
    config = args.config.expanduser().resolve()
    matrix = args.chunkCSRDir.expanduser().resolve()
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if matrix != CANONICAL_MATRIX.resolve():
        raise RuntimeError(
            "performance gate matrix must be "
            f"{CANONICAL_MATRIX}; got {matrix}")
    matrix_info = inspect_matrix(matrix)
    exe = denom.discover_exe(args.exe)
    if args.dfList:
        beds = [Path(part.strip()).expanduser().resolve()
                for part in re.split(r"[;,]", args.dfList) if part.strip()]
        if len(beds) != 5:
            raise RuntimeError(
                f"this gate requires five beds, got {len(beds)}")
        missing = [path for path in beds if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "projection missing: " + ", ".join(str(p) for p in missing))
    else:
        beds = [path.resolve() for path in default_bed_list()]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir
               else work_dir / "recon_out" / f"perf_220_mlem_{args.stage}_{stamp}")
    fout = out_dir / "recon"
    cmd = build_command(
        exe=exe, config=config, matrix=matrix, beds=beds, fout=fout,
        iterations=args.it, extra=extra)

    record: dict[str, Any] = {
        "stage": args.stage,
        "gate_id": "lu177_208_csrb_mouse_normal_5bed_20260905",
        "projection_scope": "existing merged 113+208 count vectors; performance and parity only",
        "question": (
            "50 compact MLEM iterations, Lu-177 208 keV CSRB, five mouse beds: "
            "iteration_core_s target <=120 s, floor <=180 s"),
        "exe": str(exe),
        "config": str(config),
        "matrix": matrix_info,
        "beds": [str(path) for path in beds],
        "bed_count": len(beds),
        "iterations": args.it,
        "command": cmd,
        "target_iteration_core_s": TARGET_ITERATION_CORE_S,
        "floor_iteration_core_s": FLOOR_ITERATION_CORE_S,
        "rel_l2_gate": perf.PRIMARY_REL_L2_GATE,
    }

    print(f"exe    = {exe}")
    print(f"matrix = {matrix}")
    print(f"beds   = {len(beds)}")
    print(f"it     = {args.it}:{args.it}")
    print(f"out    = {out_dir}")
    print("+ " + subprocess.list2cmdline(cmd), flush=True)
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dry_run.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        print("dry-run only")
        return 0

    if spect_recon_running():
        raise RuntimeError(
            "SPECT_recon.exe is already running; this gate is single-GPU")

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    code = proc.wait()
    wall_s = time.perf_counter() - started
    text = "".join(lines)
    (out_dir / "console.log").write_text(text, encoding="utf-8")
    metrics = parse_perf(text)
    metrics["exit"] = code
    metrics["wall_s"] = wall_s
    metrics["ok"] = (
        code == 0 and "Reconstruction finished successfully." in text)
    primary = perf.find_primary_mhd(out_dir, args.it)
    rel_l2 = None
    if args.baseline_mhd is not None and primary is not None:
        _, candidate = perf.load_mhd_values(primary)
        _, reference = perf.load_mhd_values(args.baseline_mhd)
        rel_l2 = perf.rel_l2(candidate, reference)
        metrics["rel_l2"] = rel_l2
        metrics["numerical_ok"] = rel_l2 <= perf.PRIMARY_REL_L2_GATE
    metrics["primary_mhd"] = str(primary) if primary else None
    metrics["gate"] = classify_gate(
        metrics.get("iteration_core_s"), metrics.get("denom_cache"))
    record["metrics"] = metrics
    (out_dir / "result.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(
        f"gate={metrics['gate']} denom={metrics.get('denom_cache')} "
        f"iteration_core_s={metrics.get('iteration_core_s')} "
        f"full_command_s={metrics.get('full_command_s')} "
        f"rel_l2={rel_l2}",
        flush=True)
    if not metrics["ok"]:
        return 1
    if rel_l2 is not None and rel_l2 > perf.PRIMARY_REL_L2_GATE:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
