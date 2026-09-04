"""Offline CSRB value converter: writes a new compact chunk CSRB directory whose
values went through quantize -> dequantize (fp32 arithmetic, as a GPU kernel
would decode), so the unchanged SPECT_recon.exe can reconstruct with the
candidate value format for end-to-end validation.

Modes:
  u16_sqrt  q = rint(sqrt(v/rowmax)*65535); v' = rowmax*(q/65535)^2
  u16_row   q = rint(v/rowmax*65535);       v' = rowmax*q/65535
  fp16      v' = float16(v)
  fp24      drop 8 mantissa LSB (round-to-nearest)
  identity  copy (optionally with --sort-cols only)

--sort-cols sorts col_idx ascending inside every row (lossless reorder).

Usage:
  python tools/convert_csrb_values.py <src_dir> <dst_dir> --mode u16_sqrt [--sort-cols]
         [--chunks 0,1] [--block-rows 1024] [--overwrite]
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HEADER = 28
MAGIC = 0x42525343
ONE_OVER_65535_F32 = np.float32(1.0 / 65535.0)


def read_header(path: Path):
    hdr = np.fromfile(path, dtype=np.uint8, count=HEADER)
    magic, version = (int(x) for x in np.frombuffer(hdr[:8].tobytes(), dtype=np.uint32))
    rows, ang_off, ang_cnt = (int(x) for x in np.frombuffer(hdr[8:20].tobytes(), dtype=np.int32))
    nnz = int(np.frombuffer(hdr[20:28].tobytes(), dtype=np.uint64)[0])
    if magic != MAGIC or version != 1:
        raise RuntimeError(f"{path}: unexpected magic/version {magic:#x}/{version}")
    expected = HEADER + 8 * (rows + 1) + 8 * nnz
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"{path}: size {actual} != expected {expected}")
    return hdr.tobytes(), rows, ang_off, ang_cnt, nnz


def row_max(values: np.ndarray, starts_local: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-row max for a block; rows with zero nnz get 1.0 (never used)."""
    m = np.ones(len(counts), dtype=np.float64)
    nonempty = counts > 0
    if np.any(nonempty):
        red = np.maximum.reduceat(values.astype(np.float64), starts_local[nonempty])
        m[nonempty] = red
    return m


def dequant_block(mode: str, v32: np.ndarray, starts_local: np.ndarray, counts: np.ndarray):
    if mode == "identity":
        return v32
    if mode == "fp16":
        return v32.astype(np.float16).astype(np.float32)
    if mode == "fp24":
        u = v32.view(np.uint32)
        return ((u + np.uint32(0x80)) & np.uint32(0xFFFFFF00)).view(np.float32)
    m = row_max(v32, starts_local, counts)
    m32 = np.repeat(m.astype(np.float32), counts)
    if mode == "u16_row":
        q = np.rint(v32.astype(np.float64) / m32.astype(np.float64) * 65535.0)
        q = np.clip(q, 0, 65535).astype(np.float32)
        return m32 * (q * ONE_OVER_65535_F32)
    if mode == "u16_sqrt":
        q = np.rint(np.sqrt(v32.astype(np.float64) / m32.astype(np.float64)) * 65535.0)
        q = np.clip(q, 0, 65535).astype(np.float32)
        t = q * ONE_OVER_65535_F32
        return m32 * (t * t)
    raise ValueError(mode)


def convert_chunk(src: Path, dst: Path, mode: str, sort_cols: bool, block_rows: int):
    hdr, rows, ang_off, ang_cnt, nnz = read_header(src)
    row_ptr = np.fromfile(src, dtype=np.int64, count=rows + 1, offset=HEADER)
    col_off = HEADER + 8 * (rows + 1)
    val_off = col_off + 4 * nnz
    src_col = np.memmap(src, dtype=np.uint32, mode="r", offset=col_off, shape=(nnz,))
    src_val = np.memmap(src, dtype=np.float32, mode="r", offset=val_off, shape=(nnz,))

    with open(dst, "wb") as f:
        f.write(hdr)
        f.write(row_ptr.tobytes())
        f.truncate(val_off + 4 * nnz)
    dst_col = np.memmap(dst, dtype=np.uint32, mode="r+", offset=col_off, shape=(nnz,))
    dst_val = np.memmap(dst, dtype=np.float32, mode="r+", offset=val_off, shape=(nnz,))

    t0 = time.time()
    zeroed = 0
    sq_err = 0.0
    sq_ref = 0.0
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        s, e = int(row_ptr[r0]), int(row_ptr[r1])
        if e <= s:
            continue
        starts_local = (row_ptr[r0:r1] - s).astype(np.int64)
        counts = (row_ptr[r0 + 1:r1 + 1] - row_ptr[r0:r1]).astype(np.int64)
        c = np.asarray(src_col[s:e])
        v = np.asarray(src_val[s:e])
        if sort_cols:
            row_id = np.repeat(np.arange(r1 - r0, dtype=np.int64), counts)
            key = row_id * np.int64(1 << 32) + c.astype(np.int64)
            order = np.argsort(key, kind="stable")
            c = c[order]
            v = v[order]
        vq = dequant_block(mode, v, starts_local, counts)
        zeroed += int(np.count_nonzero((vq == 0) & (v != 0)))
        d = vq.astype(np.float64) - v.astype(np.float64)
        sq_err += float(np.dot(d, d))
        sq_ref += float(np.dot(v.astype(np.float64), v.astype(np.float64)))
        dst_col[s:e] = c
        dst_val[s:e] = vq
    dst_col.flush()
    dst_val.flush()
    del dst_col, dst_val, src_col, src_val
    rel_l2 = (sq_err / sq_ref) ** 0.5 if sq_ref > 0 else 0.0
    print(f"  {src.name}: rows={rows} nnz={nnz} zeroed={zeroed} ({zeroed / nnz:.4%}) "
          f"chunk relL2={rel_l2:.3e}  {time.time() - t0:.1f}s", flush=True)
    return zeroed, rel_l2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", default="u16_sqrt",
                    choices=["u16_sqrt", "u16_row", "fp16", "fp24", "identity"])
    ap.add_argument("--sort-cols", action="store_true")
    ap.add_argument("--chunks", default="", help="comma list of chunk indices (default all)")
    ap.add_argument("--block-rows", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    src = Path(a.src)
    dst = Path(a.dst)
    chunks = sorted(p for p in src.iterdir() if p.name.lower().startswith("chunk_") and p.suffix.lower() == ".csrb")
    if not chunks:
        sys.exit(f"no chunk_*.csrb in {src}")
    if a.chunks:
        keep = {int(x) for x in a.chunks.split(",")}
        chunks = [p for i, p in enumerate(chunks) if i in keep]
    if dst.exists() and any(dst.iterdir()) and not a.overwrite:
        sys.exit(f"{dst} exists and is not empty (use --overwrite)")
    dst.mkdir(parents=True, exist_ok=True)

    for name in ("metadata.bin", "metadata.txt"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    total_bytes = sum(p.stat().st_size for p in chunks)
    free = shutil.disk_usage(dst).free
    if free < total_bytes + (1 << 30):
        sys.exit(f"insufficient free space at {dst}: need {total_bytes / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB")

    print(f"converting {len(chunks)} chunks ({total_bytes / 2**30:.1f} GiB) mode={a.mode} sort_cols={a.sort_cols}\n"
          f"  {src} -> {dst}", flush=True)
    t0 = time.time()
    stats = []
    for p in chunks:
        stats.append(convert_chunk(p, dst / p.name, a.mode, a.sort_cols, a.block_rows))
    with open(dst / "metadata.txt", "a", encoding="utf-8") as f:
        f.write("\n# convert_csrb_values postprocess\n")
        f.write(f"value_convert_mode={a.mode}\n")
        f.write(f"value_convert_dequant_arith=fp32\n")
        f.write(f"value_convert_sort_cols={int(a.sort_cols)}\n")
        f.write(f"value_convert_source={src}\n")
        for p, (z, rl) in zip(chunks, stats):
            f.write(f"value_convert_{p.stem}_zeroed={z}\n")
            f.write(f"value_convert_{p.stem}_rel_l2={rl:.6e}\n")
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
