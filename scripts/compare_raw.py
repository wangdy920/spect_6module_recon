"""Compare two .raw float32 volumes for numerical accuracy."""
import sys
import numpy as np

def compare(path_a, path_b, label=""):
    a = np.fromfile(path_a, dtype=np.float32).astype(np.float64)
    b = np.fromfile(path_b, dtype=np.float32).astype(np.float64)
    print(f"=== {label} ===")
    print(f"  shapes: {a.shape} vs {b.shape}")
    print(f"  NaN count: A={np.isnan(a).sum()} D={np.isnan(b).sum()}")

    # Focus on positive reconstruction-relevant voxels
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) & (a < 1e6) & (b < 1e6)
    n = mask.sum()
    print(f"  Active voxels (0<v<1e6, both finite): {n}/{len(a)}")
    if n == 0:
        print("  No active voxels to compare!")
        return
    da = a[mask]
    db = b[mask]
    diff = np.abs(da - db)
    denom = np.maximum(np.abs(da), 1e-30)
    rel = diff / denom

    print(f"  max_abs_diff = {diff.max():.6e}")
    print(f"  rmse         = {np.sqrt(np.mean(diff**2)):.6e}")
    print(f"  rel_l2       = {np.linalg.norm(diff) / np.linalg.norm(da):.6e}")
    print(f"  max_rel      = {rel.max():.6e}")
    print(f"  mean_rel     = {rel.mean():.6e}")
    print(f"  median_rel   = {np.median(rel):.6e}")
    pctiles = np.percentile(diff, [50, 90, 95, 99, 99.9, 100])
    labels = ["50", "90", "95", "99", "99.9", "100"]
    for lbl, val in zip(labels, pctiles):
        print(f"  abs_diff p{lbl:>5s} = {val:.6e}")
    exact = int(np.sum(da == db))
    print(f"  Exact match: {exact}/{n} ({100.0*exact/n:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: compare_raw.py <file_a.raw> <file_b.raw> [label]")
        sys.exit(1)
    label = sys.argv[3] if len(sys.argv) > 3 else "Comparison"
    compare(sys.argv[1], sys.argv[2], label)
