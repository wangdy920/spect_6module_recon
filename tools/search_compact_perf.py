#!/usr/bin/env python3
"""Manually search compact CUDA performance knobs on an idle device GPU.

Drives the existing SPECT_recon.exe. Does not add a second reconstruction
path. Default search is one-factor-at-a-time against the protected unset
environment (host cache auto, branch batch on, mmap off, config threads).

Use -vb 1 (not 2): debug verbosity disables GPU-resident MLEM and would
invalidate timings. Wall time is the process clock. Same-input primary
relL2 gate is 1e-5. Interleave variants each repeat and rank by median.

Compiled constants (6 GiB device reserve, two transient slots) are not
searchable here.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import re
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
import prep_compact_denom_cache as denom  # noqa: E402

REPO_ROOT = denom.REPO_ROOT
DEFAULT_CONFIG = denom.DEFAULT_CONFIG
PRIMARY_REL_L2_GATE = 1e-5
RESIDENT_RE = re.compile(r"Startup overlap: resident prefix (\d+) / (\d+)")
PROFILE_WALL_RE = re.compile(
    r"\[PROFILE\] full_command_wall_time: total=([0-9.]+)s")
PERF_WALL_RE = re.compile(r"\[PERF\].*?full_command_s=([0-9.]+)")
DENOM_STATE_RE = re.compile(r"\[PERF\] denom_cache=(hit|miss|disabled)")
MMAP_FAIL_RE = re.compile(r"compact-chunk mmap failed", re.IGNORECASE)

# id -> (env overrides, extra CLI after the common command)
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "env": {},
        "cli": [],
        "purpose": "protected unset environment (auto host cache, batch on, mmap off)",
    },
    "host_cache_8g": {
        "env": {"SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES": "8g"},
        "cli": [],
        "purpose": "smaller host matrix cache (32 GB-class RAM / contention)",
    },
    "host_cache_16g": {
        "env": {"SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES": "16g"},
        "cli": [],
        "purpose": "mid host matrix cache",
    },
    "host_cache_32g": {
        "env": {"SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES": "32g"},
        "cli": [],
        "purpose": "explicit 32g host cache if device RAM allows",
    },
    "mmap_0": {
        "env": {"SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES": "0"},
        "cli": [],
        "purpose": "mmap chunk payloads; previously rejected on 64 GB+NVMe",
    },
    "threads_8": {
        "env": {},
        "cli": ["-th", "8"],
        "purpose": "OpenMP threads vs config default 16",
    },
    "branch_batch_0": {
        "env": {"SPECT_COMPACT_BRANCH_BATCH": "0"},
        "cli": [],
        "purpose": "protected scalar-branch baseline; expect slower",
    },
}

DEFAULT_SUITE = [
    "baseline",
    "host_cache_8g",
    "host_cache_16g",
    "host_cache_32g",
    "mmap_0",
    "threads_8",
    "branch_batch_0",
]


def parse_variant_list(text: str) -> list[str]:
    names = [part.strip() for part in re.split(r"[,;]", text) if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("variant list is empty")
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown variants: " + ", ".join(unknown) +
            "; known: " + ", ".join(VARIANTS))
    if "baseline" not in names:
        names = ["baseline", *names]
    # Keep baseline first, preserve remaining order and uniqueness.
    ordered = ["baseline"]
    for name in names:
        if name not in ordered:
            ordered.append(name)
    return ordered


def parse_mhd_header(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "nx": 0, "ny": 0, "nz": 0, "is_float": True, "raw": None,
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("DimSize"):
            parts = line.split("=", 1)[1].split()
            info["nx"], info["ny"], info["nz"] = (int(parts[0]), int(parts[1]),
                                                  int(parts[2]))
        elif line.startswith("ElementType"):
            info["is_float"] = "MET_DOUBLE" not in line
        elif line.startswith("ElementDataFile"):
            raw_name = line.split("=", 1)[1].strip()
            info["raw"] = path.parent / raw_name
    if info["nx"] <= 0 or info["raw"] is None or not info["raw"].is_file():
        raise RuntimeError(f"invalid MHD: {path}")
    return info


def load_mhd_values(path: Path) -> tuple[tuple[int, int, int], list[float]]:
    info = parse_mhd_header(path)
    count = info["nx"] * info["ny"] * info["nz"]
    width = 4 if info["is_float"] else 8
    data = info["raw"].read_bytes()
    expected = count * width
    if len(data) < expected:
        raise RuntimeError(
            f"{info['raw']} is {len(data)} bytes, expected {expected}")
    arr = array.array("f" if info["is_float"] else "d")
    arr.frombytes(data[:expected])
    if sys.byteorder != "little":
        arr.byteswap()
    return (info["nx"], info["ny"], info["nz"]), [float(value) for value in arr]


def rel_l2(candidate: list[float], reference: list[float]) -> float:
    if len(candidate) != len(reference):
        raise ValueError(
            f"volume length {len(candidate)} != reference {len(reference)}")
    num = 0.0
    den = 0.0
    for left, right in zip(candidate, reference):
        delta = left - right
        num += delta * delta
        den += right * right
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return math.sqrt(num / den)


def find_primary_mhd(run_dir: Path, iteration: int) -> Path | None:
    suffix = f"_it{iteration}.mhd"
    matches = [
        path for path in run_dir.glob(f"*{suffix}")
        if "_smooth" not in path.name and "_butter" not in path.name
    ]
    return matches[0] if matches else None


def parse_log_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "resident_prefix": None,
        "resident_total": None,
        "profile_wall_s": None,
        "denom_cache": None,
        "mmap_fallback": bool(MMAP_FAIL_RE.search(text)),
    }
    resident = RESIDENT_RE.search(text)
    if resident:
        metrics["resident_prefix"] = int(resident.group(1))
        metrics["resident_total"] = int(resident.group(2))
    wall = PROFILE_WALL_RE.search(text) or PERF_WALL_RE.search(text)
    if wall:
        metrics["profile_wall_s"] = float(wall.group(1))
    denom_state = DENOM_STATE_RE.search(text)
    if denom_state:
        metrics["denom_cache"] = denom_state.group(1)
    return metrics


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def build_command(*, exe: Path, config: Path, matrix: Path, proj_args: list[str],
                  fout: Path, iteration: int, denom_cache: Path | None,
                  variant_cli: list[str], extra: list[str]) -> list[str]:
    cmd = [
        str(exe),
        "-config", str(config),
        "-opti", "mlem",
        "-it", str(iteration),
        "-chunkCSRDir", str(matrix),
        *proj_args,
        "-fout", str(fout),
        "-vb", "1",
    ]
    if denom_cache is not None:
        cmd.extend(["-compactDenomCache", str(denom_cache)])
    cmd.extend(variant_cli)
    cmd.extend(extra)
    return cmd


def run_recon(cmd: list[str], work_dir: Path, env_overlay: dict[str, str]) -> dict[str, Any]:
    env = os.environ.copy()
    for key in (
        "SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES",
        "SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES",
        "SPECT_COMPACT_BRANCH_BATCH",
    ):
        env.pop(key, None)
    env.update(env_overlay)
    env["SPECT_FAIL_ON_ZERO_CHECKPOINT"] = "0"
    print("+ " + subprocess.list2cmdline(cmd), flush=True)
    if env_overlay:
        print("  env " + " ".join(f"{k}={v}" for k, v in env_overlay.items()),
              flush=True)
    started = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(work_dir),
        env=env,
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
    metrics = parse_log_metrics(text)
    metrics["exit"] = code
    metrics["wall_s"] = wall_s
    metrics["ok"] = code == 0 and "Reconstruction finished successfully." in text
    return metrics


def summarize(variant_ids: list[str], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_id in variant_ids:
        subset = [row for row in runs if row["variant"] == variant_id]
        walls = [row["wall_s"] for row in subset if row.get("ok")]
        rels = [row["rel_l2"] for row in subset
                if row.get("rel_l2") is not None]
        ok = bool(subset) and all(row.get("ok") for row in subset)
        max_rel = max(rels) if rels else None
        numerical_ok = max_rel is not None and max_rel <= PRIMARY_REL_L2_GATE
        rows.append({
            "id": variant_id,
            "n": len(subset),
            "n_ok": sum(1 for row in subset if row.get("ok")),
            "median_wall_s": median(walls),
            "max_rel_l2": max_rel,
            "ok": ok,
            "numerical_ok": numerical_ok,
            "eligible": ok and numerical_ok,
            "mmap_fallback": any(row.get("mmap_fallback") for row in subset),
        })
    baseline = next((row for row in rows if row["id"] == "baseline"), None)
    base_wall = baseline["median_wall_s"] if baseline else None
    for row in rows:
        if base_wall and row["median_wall_s"] is not None:
            row["delta_vs_baseline_s"] = row["median_wall_s"] - base_wall
            row["delta_vs_baseline_pct"] = (
                100.0 * row["delta_vs_baseline_s"] / base_wall)
        else:
            row["delta_vs_baseline_s"] = None
            row["delta_vs_baseline_pct"] = None
        if row["id"] == "baseline":
            row["verdict"] = "baseline"
        elif not row["ok"]:
            row["verdict"] = "rejected_failed"
        elif not row["numerical_ok"]:
            row["verdict"] = "rejected_relL2"
        else:
            row["verdict"] = "eligible"
    eligible = [row for row in rows if row["eligible"] and row["median_wall_s"] is not None]
    if eligible:
        winner = min(eligible, key=lambda row: row["median_wall_s"])
        for row in rows:
            if row["id"] == winner["id"] and row["id"] != "baseline":
                row["verdict"] = "winner"
            elif row["id"] == winner["id"]:
                row["verdict"] = "baseline_fastest"
    return rows


def write_decision(path: Path, rows: list[dict[str, Any]], gate: float) -> None:
    lines = [
        "# compact CUDA performance search",
        "",
        f"Primary same-input relL2 gate: `{gate:g}`.",
        "Metric: process wall seconds, median of interleaved repeats.",
        "",
        "| variant | median_s | vs baseline | max relL2 | verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        wall = (f"{row['median_wall_s']:.3f}" if row["median_wall_s"] is not None
                else "n/a")
        if row["delta_vs_baseline_pct"] is None:
            delta = "n/a"
        else:
            delta = f"{row['delta_vs_baseline_pct']:+.2f}%"
        rel = (f"{row['max_rel_l2']:.3e}" if row["max_rel_l2"] is not None
               else "n/a")
        lines.append(
            f"| `{row['id']}` | {wall} | {delta} | {rel} | {row['verdict']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check() -> int:
    assert parse_variant_list("host_cache_8g,mmap_0")[0] == "baseline"
    ident = [1.0, 2.0, 3.0]
    assert rel_l2(ident, ident) == 0.0
    shifted = [1.0, 2.0, 4.0]
    got = rel_l2(shifted, ident)
    expected = math.sqrt(1.0 / (1.0 + 4.0 + 9.0))
    if abs(got - expected) > 1e-12:
        raise RuntimeError(f"relL2 self-check failed: {got} vs {expected}")
    text = (
        "[CompactChunkCUDA] Startup overlap: resident prefix 2 / 6\n"
        "[PERF] denom_cache=hit full_command_s=12.345\n"
        "[NOTE] compact-chunk mmap failed, falling back to full copy load\n"
    )
    metrics = parse_log_metrics(text)
    assert metrics["resident_prefix"] == 2
    assert metrics["resident_total"] == 6
    assert abs(metrics["profile_wall_s"] - 12.345) < 1e-9
    assert metrics["denom_cache"] == "hit"
    assert metrics["mmap_fallback"] is True
    tmp = REPO_ROOT / "recon_out" / "compact_denom_cache" / "_perf_selfcheck.mhd"
    raw = tmp.with_suffix(".raw")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    values = [1.0, 2.0, 3.0, 4.0]
    raw.write_bytes(struct.pack("<4d", *values))
    tmp.write_text(
        "ObjectType = Image\nNDims = 3\nDimSize = 2 2 1\n"
        "ElementType = MET_DOUBLE\nElementDataFile = _perf_selfcheck.raw\n",
        encoding="utf-8",
    )
    shape, loaded = load_mhd_values(tmp)
    tmp.unlink()
    raw.unlink()
    if shape != (2, 2, 1) or loaded != values:
        raise RuntimeError("MHD round-trip self-check failed")
    print("self-check passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Idle-device search of compact CUDA performance knobs. "
            "Writes recon_out/perf_search_<stamp>/experiment.json."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (repository root; prep denom cache first):

  python tools/search_compact_perf.py --chunkCSRDir <matrix> --dfList Bed1.bin;Bed2.bin;Bed3.bin;Bed4.bin;Bed5.bin --compactDenomCache recon_out/compact_denom_cache/auto_<hash>.mhd --it 3 --repeats 3

  python tools/search_compact_perf.py --dry-run --chunkCSRDir <matrix> --df proj.bin

Do not pass -vb 2. Ranking uses 3 iterations by default; confirm a winner with --it 20 on the device.
""",
    )
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunkCSRDir", type=Path, default=None)
    parser.add_argument("--df", type=Path, default=None,
                        help="single-bed projection")
    parser.add_argument("--dfList", default=None,
                        help="multi-bed list, comma or semicolon separated")
    parser.add_argument("--compactDenomCache", type=Path, default=None,
                        help="existing no-CTAC denom cache MHD (strongly recommended)")
    parser.add_argument("--it", type=int, default=3,
                        help="iterations per run (default 3; use 20 to match device wall-clock)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", type=parse_variant_list,
                        default=parse_variant_list(",".join(DEFAULT_SUITE)),
                        help="comma-separated variant ids (baseline is always included)")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="extra SPECT_recon args after --extra")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return self_check()

    if args.it <= 0 or args.repeats <= 0:
        raise RuntimeError("--it and --repeats must be positive")
    args.work_dir = args.work_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    if not args.config.is_file():
        raise FileNotFoundError(f"config not found: {args.config}")
    if args.df is not None and args.dfList:
        raise RuntimeError("use either --df or --dfList")
    if args.df is None and not args.dfList:
        raise RuntimeError("pass --df or --dfList (real device projections)")

    exe = denom.discover_exe(args.exe)
    matrix = denom.discover_matrix(args.chunkCSRDir, exe)
    if args.dfList:
        paths = [part.strip() for part in re.split(r"[;,]", args.dfList)
                 if part.strip()]
        proj_args = ["-dfList", ";".join(str(Path(path).expanduser().resolve())
                                         for path in paths)]
        bed_count = len(paths)
    else:
        proj_args = ["-df", str(args.df.expanduser().resolve())]
        bed_count = 1
    denom_cache = (args.compactDenomCache.expanduser().resolve()
                   if args.compactDenomCache else None)
    if denom_cache is None:
        print("[NOTE] no --compactDenomCache; each run recomputes sensitivity")
    elif not denom_cache.is_file():
        raise FileNotFoundError(f"denom cache not found: {denom_cache}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir
               else args.work_dir / "recon_out" / f"perf_search_{stamp}")
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    print(f"exe      = {exe}")
    print(f"config   = {args.config}")
    print(f"matrix   = {matrix}")
    print(f"beds     = {bed_count}")
    print(f"it       = {args.it}")
    print(f"repeats  = {args.repeats}")
    print(f"variants = {','.join(args.variants)}")
    print(f"out      = {out_dir}")
    print(f"gate     = relL2 <= {PRIMARY_REL_L2_GATE:g} vs first baseline volume")
    print("metric   = process wall seconds (median); -vb 1 keeps resident MLEM")

    jobs = [(variant, repeat)
            for repeat in range(1, args.repeats + 1)
            for variant in args.variants]
    if args.dry_run:
        for variant, repeat in jobs:
            fout = out_dir / "runs" / f"{variant}_r{repeat}" / "perf"
            cmd = build_command(
                exe=exe, config=args.config, matrix=matrix, proj_args=proj_args,
                fout=fout, iteration=args.it, denom_cache=denom_cache,
                variant_cli=list(VARIANTS[variant]["cli"]), extra=extra)
            overlay = dict(VARIANTS[variant]["env"])
            print(subprocess.list2cmdline(cmd))
            if overlay:
                print("  env " + " ".join(f"{k}={v}" for k, v in overlay.items()))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    reference: list[float] | None = None
    reference_shape: tuple[int, int, int] | None = None
    runs: list[dict[str, Any]] = []
    for variant, repeat in jobs:
        spec = VARIANTS[variant]
        run_dir = out_dir / "runs" / f"{variant}_r{repeat}"
        fout = run_dir / "perf"
        cmd = build_command(
            exe=exe, config=args.config, matrix=matrix, proj_args=proj_args,
            fout=fout, iteration=args.it, denom_cache=denom_cache,
            variant_cli=list(spec["cli"]), extra=extra)
        metrics = run_recon(cmd, args.work_dir, dict(spec["env"]))
        record: dict[str, Any] = {
            "variant": variant,
            "repeat": repeat,
            "purpose": spec["purpose"],
            "env": spec["env"],
            "cli": spec["cli"],
            "fout": str(fout),
            **metrics,
            "rel_l2": None,
            "primary_mhd": None,
        }
        if record["ok"]:
            mhd = find_primary_mhd(run_dir, args.it)
            if mhd is None:
                record["ok"] = False
                record["error"] = f"missing primary *_it{args.it}.mhd"
            else:
                record["primary_mhd"] = str(mhd)
                shape, volume = load_mhd_values(mhd)
                if reference is None:
                    if variant != "baseline":
                        raise RuntimeError("internal: baseline must run first")
                    reference = volume
                    reference_shape = shape
                    record["rel_l2"] = 0.0
                else:
                    if shape != reference_shape:
                        record["ok"] = False
                        record["error"] = (
                            f"shape {shape} != baseline {reference_shape}")
                    else:
                        record["rel_l2"] = rel_l2(volume, reference)
        runs.append(record)
        print(
            f"{variant} r{repeat}: wall={record['wall_s']:.3f}s ok={record['ok']} "
            f"relL2={record['rel_l2'] if record['rel_l2'] is not None else 'n/a'}",
            flush=True)

    rows = summarize(args.variants, runs)
    payload = {
        "schema": "spect.performance-search.v1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "gate_rel_l2": PRIMARY_REL_L2_GATE,
        "it": args.it,
        "repeats": args.repeats,
        "exe": str(exe),
        "config": str(args.config),
        "matrix": str(matrix),
        "denom_cache": str(denom_cache) if denom_cache else None,
        "variants": {
            name: {"env": VARIANTS[name]["env"], "cli": VARIANTS[name]["cli"],
                   "purpose": VARIANTS[name]["purpose"]}
            for name in args.variants
        },
        "runs": runs,
        "summary": rows,
        "note": (
            "Does not search compiled 6 GiB device reserve or transient slot "
            "count. Do not adopt a winner that fails the relL2 gate. "
            "Confirm on-device with --it 20 if ranking used fewer iterations."
        ),
    }
    experiment_path = out_dir / "experiment.json"
    experiment_path.write_text(json.dumps(payload, indent=2) + "\n",
                               encoding="utf-8")
    decision_path = out_dir / "decision.md"
    write_decision(decision_path, rows, PRIMARY_REL_L2_GATE)
    print("\n" + decision_path.read_text(encoding="utf-8"))
    print(f"wrote {experiment_path}")
    winner = next((row for row in rows if row["verdict"] in
                   ("winner", "baseline_fastest")), None)
    if winner:
        print(f"ranked winner: {winner['id']} "
              f"(median {winner['median_wall_s']:.3f}s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, argparse.ArgumentTypeError,
            ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
