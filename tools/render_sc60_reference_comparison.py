"""Render the frozen SC60 100-iteration comparison and native timing records."""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "recon_out/sc60_reference_compare_20260920"
REFERENCE = REPO / "test_data/SC60_luzhou/img/osem_121314derenzo_141keV"
CROP = (slice(105, 135), slice(4, 86), slice(4, 86))


def main():
    ni = nib.load(REFERENCE / "single_bed/Bed1/recon100.hdr")
    assert ni.shape == (82, 82, 30)
    expected_affine = np.diag([0.4, 0.4, 0.4, 1.0])
    expected_affine[:3, 3] = [-16.2, -16.2, -5.8]
    assert np.allclose(ni.affine, expected_affine, atol=2e-6, rtol=0)
    ref = ni.get_fdata().transpose(2, 1, 0)
    stitched = nib.load(REFERENCE / "stitched/recon100.hdr").get_fdata().transpose(2, 1, 0)
    assert np.array_equal(ref, stitched)
    mhd = OUT / "osem100_n6_nopsf/sc60_it100.mhd"
    fields = dict(line.split("=", 1) for line in mhd.read_text().splitlines() if "=" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    assert fields["DimSize"] == "90 90 240" and fields["ElementType"] == "MET_DOUBLE"
    assert fields["ElementSpacing"] == "0.4 0.4 0.4" and fields["ElementByteOrderMSB"] == "False"
    full = np.fromfile(mhd.parent / fields["ElementDataFile"], dtype="<f8").reshape(240, 90, 90)
    assert np.isfinite(full).all() and full.min() >= 0 and full.max() > 0
    candidate = full[CROP]
    nib.save(nib.Nifti1Image(candidate.transpose(2, 1, 0), ni.affine),
             OUT / "candidate_it100_aligned_crop.nii.gz")
    mask_full = np.load(OUT / "matrix_roi32p8_h12/roi_mask.npy")
    runtime_mask = np.fromfile(mhd.parent / "sc60_mask.raw", dtype="<f8").reshape(full.shape) > 0
    assert np.array_equal(runtime_mask, mask_full), "Runtime/derivation ROI mismatch"
    assert np.count_nonzero(full[~mask_full]) == 0
    mask = mask_full[CROP]
    a, b = ref[mask], candidate[mask]
    scale = float(np.dot(a, b) / np.dot(b, b))
    z, y, x = np.indices(ref.shape)
    def centroid(v):
        return [float(np.sum(v * (p - center) * 0.4) / v.sum())
                for p, center in ((x, 40.5), (y, 40.5), (z, 14.5))]
    limit = float(np.percentile(ref[ref > 0], 99.5))
    metrics = dict(iterations=100, subsets=6, updates=600, psf="off", crop_start_xyz=[4, 4, 105],
        roi_voxels=int(mask.sum()), display_window=[0, limit],
        relative_l2=float(np.linalg.norm(b-a) / np.linalg.norm(a)),
        pearson_r=float(np.corrcoef(a, b)[0, 1]), intensity_sum_ratio=float(b.sum()/a.sum()),
        scale_fitted_to_reference=scale,
        scale_fitted_relative_l2=float(np.linalg.norm(scale*b-a)/np.linalg.norm(a)),
        reference_total=float(ref.sum()), candidate_total=float(candidate.sum()),
        reference_max=float(ref.max()), candidate_max=float(candidate.max()),
        maximum_absolute_difference=float(np.max(np.abs(candidate-ref))),
        reference_centroid_xyz_mm=centroid(ref), candidate_centroid_xyz_mm=centroid(candidate),
        no_registration=True, single_equals_stitched=True, runtime_roi_matches_derived=True)
    edge = [0, 1, 28, 29]
    metrics["edge_diagnostic"] = dict(z_indices=edge,
        reference_intensity_fraction=float(ref[edge].sum()/ref.sum()),
        total_squared_error_fraction=float(np.sum((candidate[edge]-ref[edge])**2)/np.sum((candidate-ref)**2)),
        relative_l2_by_plane=[float(np.linalg.norm(candidate[k]-ref[k])/np.linalg.norm(ref[k])) for k in edge],
        interior_relative_l2=float(np.linalg.norm(candidate[2:28]-ref[2:28])/np.linalg.norm(ref[2:28])))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.facecolor": "white"})
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout="constrained",
                             gridspec_kw={"height_ratios": [2.7, 1, 1]})
    planes = [(lambda v: v[15], [-16.4,16.4,-16.4,16.4], "Axial: z = +0.2 mm", "x (mm)", "y (mm)"),
              (lambda v: v[:,41,:], [-16.4,16.4,-6,6], "Coronal: y = +0.2 mm", "x (mm)", "z (mm)"),
              (lambda v: v[:,:,41], [-16.4,16.4,-6,6], "Sagittal: x = +0.2 mm", "y (mm)", "z (mm)")]
    for row, (plane, extent, title, xlabel, ylabel) in enumerate(planes):
        for column, (v, label) in enumerate(((ref,"Reference"),(candidate,"multipinhole"),(candidate-ref,"Candidate - reference"))):
            ax = axes[row, column]
            im = ax.imshow(plane(v), origin="lower", extent=extent, interpolation="nearest",
                           cmap="RdBu_r" if column == 2 else "magma",
                           vmin=-0.1*limit if column == 2 else 0,
                           vmax=0.1*limit if column == 2 else limit)
            ax.set(title=f"{label}\n{title}", xlabel=xlabel, ylabel=ylabel)
            if row == 0:
                fig.colorbar(im, ax=ax, shrink=0.72, label="Raw intensity difference" if column == 2 else "Raw intensity")
    fig.suptitle("SC60 | 100 OSEM epochs x 6 subsets | PSF off\nNative coordinates; shared reference window; no registration or intensity fitting", fontsize=15)
    fig.savefig(OUT / "paired_slices.png", dpi=170)
    plt.close(fig)

    profile_y = int(np.argmax(np.std(ref[15], axis=1)))
    metrics["reference_selected_profile_y_index"] = profile_y
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    mean_limit = float(np.percentile(ref.mean(axis=0)[mask.any(axis=0)], 99.5))
    for ax, data, title in zip(axes[0], (ref, candidate), ("Reference", "multipinhole")):
        im = ax.imshow(data.mean(axis=0), origin="lower", extent=[-16.4,16.4,-16.4,16.4],
                       cmap="magma", vmin=0, vmax=mean_limit, interpolation="nearest")
        ax.set(title=title + " | axial mean over 30 layers", xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="Mean raw intensity")
    for data, label, color in ((ref,"Reference","#262626"),(candidate,"multipinhole","#008a91")):
        axes[1,0].plot((np.arange(82)-40.5)*0.4, data[15,profile_y], label=label, color=color)
        axes[1,1].plot((np.arange(30)-14.5)*0.4, data.sum(axis=(1,2)), label=label, color=color)
    axes[1,0].set(title=f"Axial profile: z=+0.2, y={(profile_y-40.5)*0.4:.1f} mm", xlabel="x (mm)", ylabel="Raw intensity")
    axes[1,1].set(title="Axial intensity distribution", xlabel="z (mm)", ylabel="Sum per plane")
    for ax in axes[1]:
        ax.legend(); ax.grid(alpha=0.2)
    fig.suptitle("Matched endpoint morphology | raw amplitudes, no fitted normalization", fontsize=15)
    fig.savefig(OUT / "profiles.png", dpi=170)
    plt.close(fig)

    # Supplementary amplified diagnostic; the primary fixed window is unchanged.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    im = axes[0].imshow((candidate-ref)[15], origin="lower", extent=[-16.4,16.4,-16.4,16.4],
                        cmap="RdBu_r", vmin=-0.005*limit, vmax=0.005*limit, interpolation="nearest")
    axes[0].set(title="Axial difference | amplified 20x vs main plot", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(im,ax=axes[0],label="Candidate - reference (raw units)")
    axes[1].plot((np.arange(30)-14.5)*0.4,
                 np.sqrt(np.sum((candidate-ref)**2,axis=(1,2))/np.maximum(np.sum(ref**2,axis=(1,2)),1e-300))*100,
                 color="#008a91")
    axes[1].set(title="Relative L2 by plane (low-signal ends amplify ratios)", xlabel="z (mm)", ylabel="Difference (%)", yscale="log")
    axes[1].grid(alpha=0.2)
    fig.savefig(OUT/"residuals.png",dpi=170); plt.close(fig)

    log = (OUT / "run.log").read_text(encoding="utf-8-sig")
    assert "Reconstruction finished successfully." in log
    iterations = [(int(i),float(t)) for i,t in re.findall(r"\[ITER_TIME\] iteration=(\d+) wall_s=([\d.]+)",log)]
    subsets = [(int(i),int(s),float(t)) for i,s,t in re.findall(r"\[SUBSET_TIME\] iteration=(\d+) subset=(\d+) wall_s=([\d.]+)",log)]
    assert [i for i,_ in iterations] == list(range(1,101)) and len(subsets) == 600
    assert all(-1e-5 <= t - sum(st for si,_,st in subsets if si == i) < 1.0 for i,t in iterations)
    for name, header, rows in (("iterations.csv",["iteration","wall_s"],iterations),
                                ("subsets.csv",["iteration","subset","wall_s"],subsets)):
        with (OUT/name).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream); writer.writerow(header); writer.writerows(rows)
    times = np.array([t for _,t in iterations])
    def stats(v):
        return dict(count=len(v), mean_s=float(np.mean(v)), median_s=float(np.median(v)),
                    minimum_s=float(np.min(v)), maximum_s=float(np.max(v)), p95_s=float(np.percentile(v,95)))
    perf = re.search(r"\[PERF\] (.+)",log).group(1)
    metrics["timing"] = dict(all_iterations=stats(times), iterations_2_to_100=stats(times[1:]),
                             subsets=stats([t for _,_,t in subsets]), first_iteration_s=float(times[0]),
                             iteration_wall_sum_s=float(times.sum()),
                             native_perf={k:v for k,v in re.findall(r"(\w+)=([\w.]+)",perf)})
    fig, axes = plt.subplots(1,2,figsize=(13,4.4),layout="constrained")
    axes[0].plot(np.arange(1,101),times,color="#008a91",linewidth=1.3)
    axes[0].axhline(np.median(times[1:]),color="#ba4945",linestyle="--",label="Median epochs 2-100")
    axes[0].set(xlabel="Complete iteration (6 subset updates)",ylabel="Wall seconds, excluding saves",title="Native steady-clock iteration timing")
    axes[0].legend()
    axes[1].boxplot([[t for _,s,t in subsets if s == k] for k in range(6)],tick_labels=range(6),showfliers=True)
    axes[1].set(xlabel="Subset ID",ylabel="Wall seconds",title="100 samples per subset; within-run variability")
    for ax in axes: ax.grid(axis="y",alpha=0.2)
    fig.savefig(OUT/"timing.png",dpi=170); plt.close(fig)
    (OUT/"comparison.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
