"""Offline SC60 reference ROI derivation; preserves rows, column IDs and f32 bits."""
import argparse
import hashlib
import json
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from convert_csrb_to_csrq import read_csrb_header


def cylinder(voxel, diameter):
    z, y, x = np.ogrid[:240, :90, :90]
    return ((abs((z - 119.5) * voxel) <= 6.0) &
            (((x - 44.5) * voxel) ** 2 + ((y - 44.5) * voxel) ** 2 <= (diameter / 2) ** 2))


def rowsum(values, rowptr):
    result = np.zeros(len(rowptr) - 1, dtype=np.float64)
    active = np.flatnonzero(np.diff(rowptr))
    if active.size:
        result[active] = np.add.reduceat(values, rowptr[active], dtype=np.float64)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    assert not args.destination.exists(), "Destination must be new"
    metadata = (args.source / "metadata.bin").read_bytes()
    assert len(metadata) == 124 and struct.unpack_from("<II", metadata) == (0x444D4D53, 3)
    assert struct.unpack_from("<iii", metadata, 40) == (90, 90, 240)
    assert float(metadata[52:84].split(b"\0")[0]) == 0.4
    mask = cylinder(0.4, 32.8)
    assert np.array_equal(mask, cylinder(float(np.float32(0.4)), float(np.float32(32.8))))
    assert all(np.array_equal(mask, np.rot90(mask, k, axes=(1, 2))) for k in range(4))
    assert np.array_equal(np.flatnonzero(mask.any(axis=(1, 2))), np.arange(105, 135))
    mask = mask.ravel()
    image = (0.25 + (np.arange(mask.size) % 997) / 997.0) * mask
    args.destination.mkdir(parents=True)
    np.save(args.destination / "roi_mask.npy", mask.reshape(240, 90, 90))
    manifest = dict(status="incomplete", source=str(args.source.resolve()),
                    roi_mm=[32.8, 12.0], grid_xyz=[90, 90, 240], crop_start_xyz=[4, 4, 105],
                    active_voxels=int(mask.sum()), column_remap=False,
                    validation="retained IDs and weights bitwise; masked forward relL2 <= 1e-12",
                    chunks=[])
    start = time.perf_counter()
    expected_angle = 0
    for path in sorted(args.source.glob("chunk_*.csrb")):
        rows, offset, angles, nnz = read_csrb_header(path)
        assert offset == expected_angle and rows == angles * 9216
        expected_angle += angles
        with path.open("rb") as stream:
            header = stream.read(28)
            row = np.fromfile(stream, dtype="<i8", count=rows + 1)
            col = np.fromfile(stream, dtype="<i4", count=nnz)
            val = np.fromfile(stream, dtype="<f4", count=nnz)
        assert row[0] == 0 and row[-1] == nnz and np.all(np.diff(row) >= 0)
        assert len(val) == nnz and col.min() >= 0 and col.max() < mask.size
        assert np.isfinite(val).all() and (val >= 0).all()
        source_hash = hashlib.sha256(header)
        for array in (row, col, val):
            source_hash.update(memoryview(array))
        keep = mask[col]
        counts = rowsum(keep, row).astype(np.int64)
        newrow = np.r_[np.int64(0), np.cumsum(counts)]
        newcol, newval = col[keep], val[keep]
        squared_error = squared_norm = 0.0
        for first in range(0, rows, 256):
            last = min(rows, first + 256)
            a, b = row[first], row[last]
            c, d = newrow[first], newrow[last]
            ref = rowsum(val[a:b].astype(np.float64) * image[col[a:b]], row[first:last+1] - a)
            got = rowsum(newval[c:d].astype(np.float64) * image[newcol[c:d]], newrow[first:last+1] - c)
            squared_error += float(np.sum((ref - got) ** 2))
            squared_norm += float(np.sum(ref ** 2))
        rel = (squared_error / max(squared_norm, 1e-300)) ** 0.5
        assert rel <= 1e-12, (path, rel)
        output = args.destination / path.name
        with output.open("xb") as stream:
            stream.write(struct.pack("<IIiiiQ", 0x42525343, 1, rows, offset, angles, len(newval)))
            for array in (newrow, newcol, newval):
                array.tofile(stream)
        assert read_csrb_header(output) == (rows, offset, angles, len(newval))
        with output.open("rb") as stream:
            stream.seek(28)
            for array in (newrow, newcol, newval):
                reread = np.fromfile(stream, dtype=array.dtype, count=array.size)
                assert np.array_equal(reread.view(np.uint8), array.view(np.uint8))
        manifest["chunks"].append(dict(name=path.name, source_sha256=source_hash.hexdigest(),
            source_nnz=nnz, retained_nnz=len(newval), bytes=output.stat().st_size,
            masked_forward_rel_l2=rel))
        (args.destination / "derivation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{path.name}: {nnz} -> {len(newval)} nnz; relL2={rel:.3g}", flush=True)
        del row, col, val, keep, newrow, newcol, newval, reread
    assert expected_angle == 90
    shutil.copyfile(args.source / "metadata.bin", args.destination / "metadata.bin")
    manifest.update(status="validated", elapsed_s=time.perf_counter() - start,
                    metadata_sha256=hashlib.sha256(metadata).hexdigest())
    (args.destination / "derivation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "chunks"}), flush=True)


if __name__ == "__main__":
    main()
