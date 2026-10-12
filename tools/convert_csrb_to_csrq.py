"""Offline CSRB -> CSRQ converter (compact chunk matrix, run-indexed columns +
u16 sqrt-companded values). See multipinhole/include/io/CompactChunkCSR.h for
the frozen on-disk layout; this script is the only producer.

CSRQ v1 chunk file (little endian):
  0   u32 magic = 0x51525343 ("CSRQ")
  4   u32 version = 1
  8   i32 rows
  12  i32 angle_offset
  16  i32 angle_count
  20  i32 cols
  24  u64 nnz
  32  u64 run_count
  40  u32 run_len_bits = 6      run = (start_col << 6) | (len - 1), len in 1..64
  44  u32 value_encoding = 1    v = row_max * (q / 65535)^2, q = u16
  48  u32 row_nnz_ptr[rows + 1]
      u32 row_run_ptr[rows + 1]
      f32 row_max[rows]
      u16 values[nnz]           (+2 pad bytes when nnz is odd)
      u32 runs[run_count]

Within a row, columns are sorted ascending and the values follow run order, so
the k-th value of run j sits at row_nnz_ptr[r] + (sum of earlier run lengths).
Runs never cross a row and never exceed 64 entries (longer x-runs are split).

Usage:
  python tools/convert_csrb_to_csrq.py <src_csrb_dir> <dst_csrq_dir>
         [--chunks 0,1] [--block-rows 1024] [--overwrite]
  python tools/convert_csrb_to_csrq.py --self-check
"""
import argparse
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

CSRB_HEADER = 28
CSRB_MAGIC = 0x42525343
CSRQ_HEADER = 48
CSRQ_MAGIC = 0x51525343
CSRQ_VERSION = 1
RUN_LEN_BITS = 6
MAX_RUN_LEN = 1 << RUN_LEN_BITS
MAX_START_COL = 1 << (32 - RUN_LEN_BITS)
VALUE_ENCODING_U16_SQRT = 1
Q_SCALE = 65535.0
ONE_OVER_Q_SCALE_F32 = np.float32(1.0 / Q_SCALE)


def read_csrb_header(path: Path):
    hdr = np.fromfile(path, dtype=np.uint8, count=CSRB_HEADER)
    magic, version = (int(x) for x in np.frombuffer(hdr[:8].tobytes(), dtype=np.uint32))
    rows, ang_off, ang_cnt = (int(x) for x in np.frombuffer(hdr[8:20].tobytes(), dtype=np.int32))
    nnz = int(np.frombuffer(hdr[20:28].tobytes(), dtype=np.uint64)[0])
    if magic != CSRB_MAGIC or version != 1:
        raise RuntimeError(f"{path}: unexpected CSRB magic/version {magic:#x}/{version}")
    expected = CSRB_HEADER + 8 * (rows + 1) + 8 * nnz
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"{path}: size {actual} != expected {expected}")
    if nnz >= 1 << 32:
        raise RuntimeError(f"{path}: nnz {nnz} exceeds the per-chunk u32 row pointer range")
    return rows, ang_off, ang_cnt, nnz


def dequantize_u16_sqrt(q: np.ndarray, row_max_per_nnz: np.ndarray) -> np.ndarray:
    """Reference decode in fp32, identical to the CUDA kernel arithmetic."""
    t = q.astype(np.float32) * ONE_OVER_Q_SCALE_F32
    return row_max_per_nnz.astype(np.float32) * (t * t)


def encode_block(row_ptr_block: np.ndarray, col: np.ndarray, val: np.ndarray):
    """Sort one block of rows by column, quantize, run-encode.

    row_ptr_block: local nnz offsets, len = nrows + 1 (starts at 0).
    Returns (q u16, runs u32, run_counts_per_row i64, row_max f32, stats dict).
    """
    nrows = len(row_ptr_block) - 1
    counts = np.diff(row_ptr_block).astype(np.int64)
    n = int(row_ptr_block[-1])
    if n == 0:
        return (np.zeros(0, np.uint16), np.zeros(0, np.uint32),
                np.zeros(nrows, np.int64), np.zeros(nrows, np.float32),
                {"split_runs": 0, "zeroed": 0, "sorted_values": np.zeros(0, np.float32)})
    if not np.isfinite(val).all() or np.any(val < 0):
        raise RuntimeError("invalid matrix value encountered; CSRQ requires finite v >= 0")
    row_id = np.repeat(np.arange(nrows, dtype=np.int64), counts)
    key = (row_id << 32) | col.astype(np.int64)
    order = np.argsort(key, kind="stable")
    col = col[order].astype(np.int64)
    val = val[order].astype(np.float32)
    row_id = row_id[order]

    same_row = row_id[1:] == row_id[:-1]
    d = np.diff(col)
    if np.any(same_row & (d == 0)):
        raise RuntimeError("duplicate column inside a row; CSRQ requires unique columns")
    is_break = ~(same_row & (d == 1))
    run_first = np.concatenate(([0], np.flatnonzero(is_break) + 1)).astype(np.int64)
    run_len = np.diff(np.concatenate((run_first, [n]))).astype(np.int64)

    # Split runs longer than MAX_RUN_LEN into consecutive pieces.
    n_parts = (run_len + MAX_RUN_LEN - 1) // MAX_RUN_LEN
    split_runs = int(np.count_nonzero(n_parts > 1))
    if split_runs:
        part_idx = np.repeat(run_first, n_parts)
        within = np.arange(int(n_parts.sum()), dtype=np.int64) - np.repeat(
            np.cumsum(n_parts) - n_parts, n_parts)
        run_first = part_idx + within * MAX_RUN_LEN
        full_len = np.repeat(run_len, n_parts)
        run_len = np.minimum(MAX_RUN_LEN, full_len - within * MAX_RUN_LEN)
    start_col = col[run_first]
    if np.any(start_col >= MAX_START_COL):
        raise RuntimeError(f"column index >= {MAX_START_COL} cannot be run-packed")
    runs = ((start_col << RUN_LEN_BITS) | (run_len - 1)).astype(np.uint32)
    run_counts = np.bincount(row_id[run_first], minlength=nrows).astype(np.int64)

    row_max = np.zeros(nrows, dtype=np.float32)
    nonempty = counts > 0
    row_max[nonempty] = np.maximum.reduceat(val, row_ptr_block[:-1][nonempty])
    m = np.repeat(row_max, counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(m > 0, val.astype(np.float64) / m.astype(np.float64), 0.0)
    q = np.rint(np.sqrt(ratio) * Q_SCALE)
    q = np.clip(q, 0, Q_SCALE).astype(np.uint16)
    zeroed = int(np.count_nonzero((q == 0) & (val != 0)))
    return q, runs, run_counts, row_max, {"split_runs": split_runs, "zeroed": zeroed,
                                          "sorted_values": val}


def write_csrq_header(f, rows, ang_off, ang_cnt, cols, nnz, run_count):
    hdr = np.zeros(CSRQ_HEADER, dtype=np.uint8)
    hdr[0:8] = np.frombuffer(np.array([CSRQ_MAGIC, CSRQ_VERSION], dtype=np.uint32).tobytes(), np.uint8)
    hdr[8:24] = np.frombuffer(np.array([rows, ang_off, ang_cnt, cols], dtype=np.int32).tobytes(), np.uint8)
    hdr[24:40] = np.frombuffer(np.array([nnz, run_count], dtype=np.uint64).tobytes(), np.uint8)
    hdr[40:48] = np.frombuffer(np.array([RUN_LEN_BITS, VALUE_ENCODING_U16_SQRT], dtype=np.uint32).tobytes(), np.uint8)
    f.seek(0)
    f.write(hdr.tobytes())


def convert_chunk(src: Path, dst: Path, cols: int, block_rows: int):
    rows, ang_off, ang_cnt, nnz = read_csrb_header(src)
    row_ptr = np.fromfile(src, dtype=np.int64, count=rows + 1, offset=CSRB_HEADER)
    if row_ptr[0] != 0 or row_ptr[-1] != nnz or np.any(np.diff(row_ptr) < 0):
        raise RuntimeError(f"{src}: invalid row pointers")
    col_off = CSRB_HEADER + 8 * (rows + 1)
    val_off = col_off + 4 * nnz
    src_col = np.memmap(src, dtype=np.uint32, mode="r", offset=col_off, shape=(nnz,))
    src_val = np.memmap(src, dtype=np.float32, mode="r", offset=val_off, shape=(nnz,))

    ptr_bytes = 4 * (rows + 1)
    values_off = CSRQ_HEADER + 2 * ptr_bytes + 4 * rows
    values_bytes = 2 * nnz
    pad = (4 - values_bytes % 4) % 4
    runs_off = values_off + values_bytes + pad

    with open(dst, "wb") as f:
        f.truncate(runs_off)
    dst_q = np.memmap(dst, dtype=np.uint16, mode="r+", offset=values_off, shape=(nnz,)) if nnz else None

    row_run_counts = np.zeros(rows, dtype=np.int64)
    row_max_all = np.zeros(rows, dtype=np.float32)
    run_chunks = []
    split_total = 0
    zeroed_total = 0
    sq_err = 0.0
    sq_ref = 0.0
    t0 = time.time()
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        s, e = int(row_ptr[r0]), int(row_ptr[r1])
        local_ptr = (row_ptr[r0:r1 + 1] - s).astype(np.int64)
        c = np.asarray(src_col[s:e])
        v = np.asarray(src_val[s:e])
        if np.any(c >= cols):
            raise RuntimeError(f"{src}: column outside metadata grid")
        q, runs, run_counts, row_max, st = encode_block(local_ptr, c, v)
        if e > s:
            dst_q[s:e] = q
            m = np.repeat(row_max, np.diff(local_ptr))
            dq = dequantize_u16_sqrt(q, m).astype(np.float64)
            v_sorted = st["sorted_values"].astype(np.float64)
            d = dq - v_sorted
            sq_err += float(np.dot(d, d))
            sq_ref += float(np.dot(v_sorted, v_sorted))
        row_run_counts[r0:r1] = run_counts
        row_max_all[r0:r1] = row_max
        run_chunks.append(runs)
        split_total += st["split_runs"]
        zeroed_total += st["zeroed"]
    if dst_q is not None:
        dst_q.flush()
        del dst_q
    runs_all = np.concatenate(run_chunks) if run_chunks else np.zeros(0, np.uint32)
    run_count = int(runs_all.size)
    row_run_ptr = np.concatenate(([0], np.cumsum(row_run_counts))).astype(np.uint32)
    with open(dst, "r+b") as f:
        write_csrq_header(f, rows, ang_off, ang_cnt, cols, nnz, run_count)
        f.seek(CSRQ_HEADER)
        f.write(row_ptr.astype(np.uint32).tobytes())
        f.write(row_run_ptr.tobytes())
        f.write(row_max_all.tobytes())
        f.seek(runs_off)
        f.write(runs_all.tobytes())
    rel_l2 = (sq_err / sq_ref) ** 0.5 if sq_ref > 0 else 0.0
    out_bytes = dst.stat().st_size
    print(f"  {src.name}: rows={rows} nnz={nnz} runs={run_count} "
          f"({nnz / max(run_count, 1):.2f} nnz/run, split={split_total}) zeroed={zeroed_total} "
          f"relL2={rel_l2:.3e} bytes/nnz={out_bytes / max(nnz, 1):.3f} "
          f"{out_bytes / 2**30:.2f} GiB {time.time() - t0:.1f}s", flush=True)
    return {"rows": rows, "nnz": nnz, "runs": run_count, "split": split_total,
            "zeroed": zeroed_total, "rel_l2": rel_l2, "bytes": out_bytes}


def read_total_voxels(metadata_bin: Path) -> int:
    """Read the voxel count from legacy 88/92 B or SMMD v3 124 B metadata.

    Metadata is preserved, not converted; CSRQ runtime profile support is separate.
    """
    data = metadata_bin.read_bytes()
    if len(data) == 124:
        if struct.unpack_from("<II", data) != (0x444D4D53, 3):
            raise RuntimeError(f"{metadata_bin}: invalid SMMD v3 magic/version")
        offset = 12
        dims_offset = 40
    elif len(data) in (88, 92):
        offset = 4
        dims_offset = 32
    else:
        raise RuntimeError(f"{metadata_bin}: unsupported metadata size {len(data)}")
    cols = struct.unpack_from("<i", data, offset)[0]
    dims = struct.unpack_from("<iii", data, dims_offset)
    if cols <= 0 or min(dims) <= 0 or cols != dims[0] * dims[1] * dims[2]:
        raise RuntimeError(f"{metadata_bin}: invalid voxel count/dimensions")
    if cols > MAX_START_COL:
        raise RuntimeError(f"{metadata_bin}: grid exceeds CSRQ column range")
    return cols


def decode_csrq_chunk(path: Path):
    """Independent reader used by --self-check: returns (rows, ang_off, ang_cnt,
    cols, row_nnz_ptr, list of (col, value) arrays per row in stored order)."""
    hdr = path.read_bytes()[:CSRQ_HEADER]
    magic, version = np.frombuffer(hdr[0:8], dtype=np.uint32)
    rows, ang_off, ang_cnt, cols = (int(x) for x in np.frombuffer(hdr[8:24], dtype=np.int32))
    nnz, run_count = (int(x) for x in np.frombuffer(hdr[24:40], dtype=np.uint64))
    run_len_bits, value_encoding = (int(x) for x in np.frombuffer(hdr[40:48], dtype=np.uint32))
    if magic != CSRQ_MAGIC or version != CSRQ_VERSION:
        raise RuntimeError("bad CSRQ magic/version")
    if run_len_bits != RUN_LEN_BITS or value_encoding != VALUE_ENCODING_U16_SQRT:
        raise RuntimeError("unexpected CSRQ encoding fields")
    off = CSRQ_HEADER
    row_nnz_ptr = np.fromfile(path, dtype=np.uint32, count=rows + 1, offset=off); off += 4 * (rows + 1)
    row_run_ptr = np.fromfile(path, dtype=np.uint32, count=rows + 1, offset=off); off += 4 * (rows + 1)
    row_max = np.fromfile(path, dtype=np.float32, count=rows, offset=off); off += 4 * rows
    q = np.fromfile(path, dtype=np.uint16, count=nnz, offset=off); off += 2 * nnz + (4 - (2 * nnz) % 4) % 4
    runs = np.fromfile(path, dtype=np.uint32, count=run_count, offset=off); off += 4 * run_count
    if off != path.stat().st_size:
        raise RuntimeError(f"CSRQ size mismatch: parsed {off}, file {path.stat().st_size}")
    per_row = []
    for r in range(rows):
        cols_r = []
        p = int(row_nnz_ptr[r])
        for j in range(int(row_run_ptr[r]), int(row_run_ptr[r + 1])):
            length = int(runs[j] & (MAX_RUN_LEN - 1)) + 1
            start = int(runs[j] >> RUN_LEN_BITS)
            cols_r.append(np.arange(start, start + length, dtype=np.int64))
            p += length
        if p != int(row_nnz_ptr[r + 1]):
            raise RuntimeError(f"row {r}: run lengths sum to {p - int(row_nnz_ptr[r])}, nnz {int(row_nnz_ptr[r + 1]) - int(row_nnz_ptr[r])}")
        c = np.concatenate(cols_r) if cols_r else np.zeros(0, np.int64)
        qr = q[int(row_nnz_ptr[r]):int(row_nnz_ptr[r + 1])]
        v = dequantize_u16_sqrt(qr, np.full(len(qr), row_max[r], np.float32))
        per_row.append((c, v))
    return rows, ang_off, ang_cnt, cols, row_nnz_ptr, per_row


def write_csrb(path: Path, rows: int, ang_off: int, ang_cnt: int,
               row_ptr: np.ndarray, col: np.ndarray, val: np.ndarray):
    with open(path, "wb") as f:
        f.write(np.array([CSRB_MAGIC, 1], dtype=np.uint32).tobytes())
        f.write(np.array([rows, ang_off, ang_cnt], dtype=np.int32).tobytes())
        f.write(np.array([len(val)], dtype=np.uint64).tobytes())
        f.write(row_ptr.astype(np.int64).tobytes())
        f.write(col.astype(np.uint32).tobytes())
        f.write(val.astype(np.float32).tobytes())


def self_check() -> int:
    for invalid in (-1.0, float("nan"), float("inf")):
        try:
            encode_block(np.array([0, 1]), np.array([0], np.uint32),
                         np.array([invalid], np.float32))
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid matrix value accepted")
    rng = np.random.default_rng(7)
    cols = 5000
    rows = 12
    row_ptr = [0]
    all_c, all_v = [], []
    for r in range(rows):
        if r == 3:
            row_ptr.append(row_ptr[-1])  # empty row
            continue
        # random runs incl. lengths 1, 64 and >64 (to exercise splitting)
        c = []
        pos = int(rng.integers(0, 10))
        while pos < cols - 200 and len(c) < 1500:
            length = int(rng.choice([1, 2, 5, 15, 33, 64, 70, 130]))
            c.extend(range(pos, min(pos + length, cols)))
            pos += length + int(rng.integers(0, 4))
        c = np.array(c, dtype=np.int64)
        rng.shuffle(c)  # CSRB rows are unsorted in the wild
        v = rng.random(len(c)).astype(np.float32) * float(rng.random() * 3 + 0.01)
        v[rng.integers(0, len(v))] = 0.0  # explicit zero value survives
        all_c.append(c); all_v.append(v)
        row_ptr.append(row_ptr[-1] + len(c))
    col = np.concatenate(all_c); val = np.concatenate(all_v)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for size in (88, 92, 124):
            meta = bytearray(size)
            if size == 124:
                struct.pack_into("<II", meta, 0, 0x444D4D53, 3)
            struct.pack_into("<i", meta, 12 if size == 124 else 4, cols)
            struct.pack_into("<iii", meta, 40 if size == 124 else 32, 10, 10, 50)
            path = tmp / "metadata.bin"
            path.write_bytes(meta)
            assert read_total_voxels(path) == cols
        meta[0] = 0
        path.write_bytes(meta)
        try:
            read_total_voxels(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid SMMD magic accepted")
        src = tmp / "chunk_000.csrb"
        dst = tmp / "chunk_000.csrq"
        write_csrb(src, rows, 0, 1, np.array(row_ptr), col, val)
        stats = convert_chunk(src, dst, cols, block_rows=5)
        _, _, _, cols_hdr, row_nnz_ptr, per_row = decode_csrq_chunk(dst)
        assert cols_hdr == cols
        assert stats["split"] > 0, "self-check matrix must contain runs > 64"
        assert stats["zeroed"] == 0
        assert (row_nnz_ptr == np.array(row_ptr, dtype=np.uint32)).all()
        max_rel = 0.0
        for r in range(rows):
            s, e = row_ptr[r], row_ptr[r + 1]
            c_ref = col[s:e]; v_ref = val[s:e].astype(np.float64)
            order = np.argsort(c_ref, kind="stable")
            c_got, v_got = per_row[r]
            assert (c_got == c_ref[order]).all(), f"row {r}: column mismatch"
            if e > s and v_ref.max() > 0:
                rel = np.abs(v_got.astype(np.float64) - v_ref[order]).max() / v_ref.max()
                max_rel = max(max_rel, rel)
                # sqrt-companding: max abs error <= m * (2*sqrt(t)*0.5/65535) ~ 1.6e-5 * m
                assert rel < 2e-5, f"row {r}: rel err {rel}"
        assert 0 < stats["rel_l2"] < 3e-5
    print(f"self-check passed: {stats['nnz']} nnz, {stats['runs']} runs, "
          f"split={stats['split']}, max value rel err {max_rel:.2e}, relL2 {stats['rel_l2']:.2e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--chunks", default="", help="comma list of chunk indices (default all)")
    ap.add_argument("--block-rows", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    if not a.src or not a.dst:
        ap.error("src and dst are required")
    if a.block_rows <= 0:
        ap.error("--block-rows must be positive")
    src = Path(a.src)
    dst = Path(a.dst)
    chunks = sorted(p for p in src.iterdir()
                    if p.name.lower().startswith("chunk_") and p.suffix.lower() == ".csrb")
    if not chunks:
        sys.exit(f"no chunk_*.csrb in {src}")
    if a.chunks:
        keep = {int(x) for x in a.chunks.split(",")}
        chunks = [p for i, p in enumerate(chunks) if i in keep]
    if dst.exists() and any(dst.iterdir()) and not a.overwrite:
        sys.exit(f"{dst} exists and is not empty (use --overwrite)")
    cols = read_total_voxels(src / "metadata.bin")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "metadata.bin", dst / "metadata.bin")
    if (src / "metadata.txt").exists():
        shutil.copy2(src / "metadata.txt", dst / "metadata.txt")

    total_in = sum(p.stat().st_size for p in chunks)
    free = shutil.disk_usage(dst).free
    if free < total_in // 3 + (1 << 30):
        sys.exit(f"insufficient free space at {dst}: ~{total_in / 3 / 2**30:.1f} GiB needed, free {free / 2**30:.1f} GiB")
    print(f"converting {len(chunks)} CSRB chunks ({total_in / 2**30:.1f} GiB) -> CSRQ cols={cols}\n"
          f"  {src} -> {dst}", flush=True)
    t0 = time.time()
    stats = []
    for p in chunks:
        stats.append((p.stem, convert_chunk(p, dst / (p.stem + ".csrq"), cols, a.block_rows)))
    total_nnz = sum(s["nnz"] for _, s in stats)
    total_out = sum(s["bytes"] for _, s in stats)
    with open(dst / "metadata.txt", "a", encoding="utf-8") as f:
        f.write("\n# convert_csrb_to_csrq postprocess\n")
        f.write("chunk_format_derived=csrq\n")
        f.write(f"csrq_value_encoding=u16_sqrt_rowmax\ncsrq_run_len_bits={RUN_LEN_BITS}\n")
        f.write(f"csrq_source={src}\ncsrq_total_nnz={total_nnz}\ncsrq_total_bytes={total_out}\n")
        for name, s in stats:
            f.write(f"csrq_{name}_runs={s['runs']}\ncsrq_{name}_split_runs={s['split']}\n"
                    f"csrq_{name}_zeroed={s['zeroed']}\ncsrq_{name}_rel_l2={s['rel_l2']:.6e}\n")
    print(f"done in {time.time() - t0:.1f}s: {total_out / 2**30:.2f} GiB "
          f"({total_out / max(total_nnz, 1):.3f} B/nnz) vs {total_in / 2**30:.2f} GiB CSRB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
