#!/usr/bin/env python3
"""Manually precompute no-CTAC compact denom caches for 1-5 beds.

Drives SPECT_recon.exe with the shipped spect_config.ini. Dummy BedN.bin
files supply geometry only; measured counts are not part of the cache
identity. SPECT_recon writes auto_<hash>.mhd under the INI-configured cache
directory when -compactDenomCache is omitted and CTAC is off.

Run from a terminal when a matrix/PSF change requires a refresh. This is
intentionally slow (matrix preload per bed count) and is not a second
sensitivity implementation.
"""

from __future__ import annotations

import argparse
import configparser
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "multipinhole" / "config" / "spect_config.ini"
CACHE_DIRNAME = Path("recon_out") / "compact_denom_cache"
DUMMY_DIRNAME = CACHE_DIRNAME / "_dummy_proj"
SCRATCH_DIRNAME = CACHE_DIRNAME / "_prep_scratch"
MAX_BEDS = 5
META_FMT = struct.Struct("<11if2i3f5i")

CACHE_SAVED_RE = re.compile(r"compact denom cache saved:\s*(.+)$", re.MULTILINE)
CACHE_HIT_RE = re.compile(r"compact denom cache hit:\s*(.+)$", re.MULTILINE)
CACHE_MISS_RE = re.compile(r"compact denom cache miss:")
CTAC_RE = re.compile(r"compact CTAC:", re.IGNORECASE)


def configured_cache_dir(config: Path, exe: Path | None,
                         work_dir: Path) -> Path:
    parser = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=(";", "#"))
    parser.read(config, encoding="utf-8-sig")
    value = parser.get("SystemMatrix", "compact_denom_cache_dir", fallback="").strip()
    if not value:
        return (work_dir / CACHE_DIRNAME).resolve()
    path = Path(value).expanduser()
    base = exe.parent if exe is not None else work_dir
    return (path if path.is_absolute() else base / path).resolve()


def parse_beds(text: str) -> list[int]:
    raw = text.strip()
    if not raw:
        raise argparse.ArgumentTypeError("bed list is empty")
    try:
        if "-" in raw and "," not in raw and ";" not in raw:
            lo_s, hi_s = raw.split("-", 1)
            beds = list(range(int(lo_s), int(hi_s) + 1))
        else:
            beds = [int(part) for part in re.split(r"[,;]", raw) if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid bed list '{text}'") from exc
    if not beds:
        raise argparse.ArgumentTypeError("bed list is empty")
    unique = sorted(set(beds))
    for bed in unique:
        if bed < 1 or bed > MAX_BEDS:
            raise argparse.ArgumentTypeError(
                f"bed count must be in 1..{MAX_BEDS}, got {bed}")
    return unique


def write_dummy_projection_bin(path: Path, num_angles: int, rays_per_angle: int,
                               fill: float = 1.0) -> int:
    if num_angles <= 0 or rays_per_angle <= 0:
        raise ValueError("dummy BIN geometry must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = num_angles * rays_per_angle
    payload = struct.pack("<d", float(fill)) * count
    with path.open("wb") as handle:
        handle.write(struct.pack("<ii", num_angles, rays_per_angle))
        handle.write(payload)
    return 8 + count * 8


def read_compact_metadata(path: Path) -> dict[str, int | float]:
    data = path.read_bytes()
    if len(data) != META_FMT.size:
        raise RuntimeError(
            f"Compact metadata size mismatch ({len(data)} bytes, "
            f"canonical v4 requires {META_FMT.size}): {path}")
    fields = META_FMT.unpack(data)
    meta = {
        "fullNumAngles": int(fields[7]),
        "Nx": int(fields[8]),
        "Ny": int(fields[9]),
        "Nz": int(fields[10]),
        "voxelSize": float(fields[11]),
        "detNx": int(fields[4]),
        "detNz": int(fields[13]),
    }
    rays = meta["detNx"] * meta["detNz"]
    if meta["fullNumAngles"] <= 0 or rays <= 0:
        raise RuntimeError(f"Invalid detector geometry in {path}")
    meta["raysPerAngle"] = rays
    return meta


def discover_exe(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SPECT_recon not found: {path}")
        return path
    candidates = [
        REPO_ROOT / "build" / "Release" / "bin" / "Release" / "SPECT_recon.exe",
        REPO_ROOT / "build" / "bin" / "Release" / "SPECT_recon.exe",
        REPO_ROOT / "staging" / "SPECT_recon_rc_dev" / "bin" / "SPECT_recon.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "SPECT_recon.exe not found. Pass --exe or build Release first.")


def discover_matrix(explicit: Path | None, exe: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not (path / "metadata.bin").is_file():
            raise FileNotFoundError(
                f"matrix directory has no metadata.bin: {path}")
        return path
    if exe is not None:
        matrix_root = exe.parent / "matrix"
        if matrix_root.is_dir():
            found = [
                child for child in matrix_root.iterdir()
                if child.is_dir() and (child / "metadata.bin").is_file()
            ]
            if len(found) == 1:
                return found[0].resolve()
            if len(found) > 1:
                names = ", ".join(child.name for child in found)
                raise RuntimeError(
                    "multiple exe-local matrices (" + names +
                    "); pass --chunkCSRDir")
    raise FileNotFoundError(
        "No compact matrix found. Pass --chunkCSRDir <dir> with metadata.bin.")


def parse_manifest_fields(manifest_path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def existing_auto_caches(cache_dir: Path, bed: int, matrix_dir: Path) -> list[Path]:
    wanted = str(matrix_dir.resolve())
    matches: list[Path] = []
    for manifest in sorted(cache_dir.glob("auto_*.manifest.txt")):
        fields = parse_manifest_fields(manifest)
        if fields.get("bed_count") != str(bed):
            continue
        chunk_dir = fields.get("chunk_dir", "")
        if chunk_dir and Path(chunk_dir).resolve() != Path(wanted):
            continue
        matches.append(manifest.with_suffix(".mhd"))
    return matches


def delete_cache_set(mhd_path: Path) -> None:
    for extra in (mhd_path, mhd_path.with_suffix(".raw"),
                  mhd_path.with_suffix(".manifest.txt")):
        if extra.exists():
            extra.unlink()


def build_command(*, exe: Path, config: Path, matrix: Path, meta: dict[str, int | float],
                  beds: int, dummy_dir: Path, scratch_prefix: Path) -> list[str]:
    dummy_paths = [dummy_dir / f"Bed{idx}.bin" for idx in range(1, beds + 1)]
    if beds == 1:
        proj_args = ["-df", str(dummy_paths[0])]
    else:
        proj_args = ["-dfList", ";".join(str(path) for path in dummy_paths)]
    return [
        str(exe),
        "-config", str(config),
        "-opti", "mlem",
        "-it", "1",
        "-chunkCSRDir", str(matrix),
        "-dim", f"{meta['Nx']},{meta['Ny']},{meta['Nz']}",
        "-vox", ",".join(f"{float(meta['voxelSize']):.6g}" for _ in range(3)),
        "-ang", str(meta["fullNumAngles"]),
        "-detN", f"{meta['detNz']},{meta['detNx']}",
        *proj_args,
        "-fout", str(scratch_prefix),
        "-vb", "1",
    ]


def run_recon(cmd: list[str], work_dir: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env["SPECT_FAIL_ON_ZERO_CHECKPOINT"] = "0"
    print("+ " + subprocess.list2cmdline(cmd), flush=True)
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
    return code, "".join(lines)


def interpret_log(text: str) -> tuple[str, str]:
    if CTAC_RE.search(text):
        raise RuntimeError(
            "SPECT_recon enabled CTAC; this script is no-CTAC only. "
            "Do not pass -ctac / -attMap.")
    saved = CACHE_SAVED_RE.findall(text)
    hits = CACHE_HIT_RE.findall(text)
    if saved:
        return "saved", saved[-1].strip()
    if hits:
        return "hit", hits[-1].strip()
    if CACHE_MISS_RE.search(text):
        raise RuntimeError(
            "denom cache miss without a following save; check the log above")
    raise RuntimeError(
        "SPECT_recon did not report a compact denom cache save or hit")


def cleanup_scratch(scratch_dir: Path, keep: bool) -> None:
    if keep or not scratch_dir.is_dir():
        return
    for path in scratch_dir.rglob("*"):
        if path.suffix.lower() in {".mhd", ".raw", ".json"}:
            path.unlink(missing_ok=True)


def self_check(args: argparse.Namespace) -> int:
    assert parse_beds("1-5") == [1, 2, 3, 4, 5]
    assert parse_beds("1,3,5") == [1, 3, 5]
    assert interpret_log(
        "compact denom cache miss: cache files not found\n"
        "compact denom cache saved: C:\\cache\\auto_test.mhd\n"
        "[ITER] 1/1 compact\n") == ("saved", "C:\\cache\\auto_test.mhd")
    assert interpret_log(
        "compact denom cache hit: C:\\cache\\auto_test.mhd\n"
        "[PERF] denom_cache=hit full_command_s=1\n") == (
            "hit", "C:\\cache\\auto_test.mhd")
    print("self-check: denom log parser ok")
    dummy = Path(args.work_dir) / DUMMY_DIRNAME / "self_check_Bed1.bin"
    size = write_dummy_projection_bin(dummy, 4, 6, fill=1.0)
    payload = dummy.read_bytes()
    dummy.unlink()
    if size != 8 + 4 * 6 * 8:
        raise RuntimeError("dummy BIN size mismatch")
    angles, rays = struct.unpack_from("<ii", payload)
    fill = struct.unpack_from("<d", payload, 8)[0]
    if angles != 4 or rays != 6 or fill != 1.0:
        raise RuntimeError("dummy BIN header/payload mismatch")
    print("self-check: dummy BIN writer ok")
    exe = None
    if args.exe is not None:
        exe = discover_exe(args.exe)
        print(f"self-check: exe {exe}")
    try:
        matrix = discover_matrix(args.chunkCSRDir, exe)
        meta = read_compact_metadata(matrix / "metadata.bin")
        print(
            f"self-check: matrix {matrix} "
            f"{meta['Nx']}x{meta['Ny']}x{meta['Nz']} "
            f"ang={meta['fullNumAngles']} det={meta['detNz']}x{meta['detNx']}")
    except FileNotFoundError as exc:
        print(f"self-check: matrix not resolved ({exc})")
    print("self-check passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute no-CTAC compact denom caches for default spect_config "
            "and 1-5 beds. Writes INI cache_dir/auto_<hash>.mhd."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (from the repository root):

  python tools/prep_compact_denom_cache.py --chunkCSRDir F:\\matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220

  python tools/prep_compact_denom_cache.py --chunkCSRDir bin\\matrix\\260601 --beds 1-5

  python tools/prep_compact_denom_cache.py --dry-run --chunkCSRDir <matrix>
""",
    )
    parser.add_argument("--exe", type=Path, default=None,
                        help="SPECT_recon.exe (default: Release build or staging package)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="spect_config.ini (default: repo multipinhole/config/spect_config.ini)")
    parser.add_argument("--chunkCSRDir", type=Path, default=None,
                        help="compact matrix directory containing metadata.bin")
    parser.add_argument("--beds", type=parse_beds, default=parse_beds("1-5"),
                        help="bed counts to prepare, e.g. 1-5 or 1,3,5 (default: 1-5)")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT,
                        help="CWD for temporary projections and reconstruction outputs")
    parser.add_argument("--force", action="store_true",
                        help="delete matching auto_* caches for these bed counts before running")
    parser.add_argument("--keep-scratch", action="store_true",
                        help="keep dummy recon MHD/RAW under _prep_scratch")
    parser.add_argument("--dry-run", action="store_true",
                        help="print SPECT_recon commands without running them")
    parser.add_argument("--self-check", action="store_true",
                        help="verify dummy BIN I/O and optional metadata parse, then exit")
    args = parser.parse_args()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()

    if args.self_check:
        return self_check(args)

    if not args.config.is_file():
        raise FileNotFoundError(f"config not found: {args.config}")

    exe: Path | None = None
    try:
        exe = discover_exe(args.exe)
    except FileNotFoundError:
        if not args.dry_run:
            raise
        print("warning: SPECT_recon.exe not found; dry-run uses SPECT_recon.exe")
    matrix = discover_matrix(args.chunkCSRDir, exe)
    exe_path = exe if exe is not None else Path("SPECT_recon.exe")
    meta = read_compact_metadata(matrix / "metadata.bin")
    work_dir = args.work_dir
    cache_dir = configured_cache_dir(args.config, exe, work_dir)
    dummy_dir = work_dir / DUMMY_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"exe     = {exe_path}")
    print(f"config  = {args.config}")
    print(f"matrix  = {matrix}")
    print(f"geometry= {meta['Nx']}x{meta['Ny']}x{meta['Nz']}  "
          f"ang={meta['fullNumAngles']} det={meta['detNz']}x{meta['detNx']}")
    print(f"cache   = {cache_dir / 'auto_<hash>.mhd'}")
    print(f"beds    = {','.join(str(bed) for bed in args.beds)}")
    print("mode    = no-CTAC (do not pass -ctac / -attMap / -compactDenomCache)")

    dummy_size = 8 + int(meta["fullNumAngles"]) * int(meta["raysPerAngle"]) * 8
    for bed in range(1, max(args.beds) + 1):
        dummy_path = dummy_dir / f"Bed{bed}.bin"
        if args.dry_run:
            print(f"dummy   = {dummy_path} ({dummy_size} bytes, fill=1)")
            continue
        write_dummy_projection_bin(
            dummy_path,
            int(meta["fullNumAngles"]),
            int(meta["raysPerAngle"]),
            fill=1.0,
        )
        print(f"wrote dummy {dummy_path.name} ({dummy_path.stat().st_size} bytes)")

    results: list[tuple[int, str, str]] = []
    for bed in args.beds:
        scratch_prefix = work_dir / SCRATCH_DIRNAME / f"beds{bed}" / "denom_prep"
        cmd = build_command(
            exe=exe_path,
            config=args.config,
            matrix=matrix,
            meta=meta,
            beds=bed,
            dummy_dir=dummy_dir,
            scratch_prefix=scratch_prefix,
        )
        if args.force and not args.dry_run:
            for stale in existing_auto_caches(cache_dir, bed, matrix):
                print(f"force-delete {stale}")
                delete_cache_set(stale)
        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
            continue
        code, log_text = run_recon(cmd, work_dir)
        status, cache_path = interpret_log(log_text)
        if code != 0 and status != "saved":
            raise RuntimeError(
                f"SPECT_recon exited {code} for {bed}-bed without saving a denom cache")
        if code != 0:
            print(
                f"[NOTE] SPECT_recon exited {code} after denom cache {status}; "
                f"keeping {cache_path}")
        print(f"{bed}-bed denom cache {status}: {cache_path}")
        cleanup_scratch(scratch_prefix.parent, args.keep_scratch)
        results.append((bed, status, cache_path))

    if args.dry_run:
        return 0

    print("\nPrepared no-CTAC denom caches:")
    for bed, status, cache_path in results:
        print(f"  beds={bed}  {status:5s}  {cache_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, argparse.ArgumentTypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
