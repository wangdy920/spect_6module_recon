"""Audit frozen subset outputs and render the primary-output report figures.

Run: python tools/audit_subset_convergence_20260910.py
Writes report assets and copies updated PNGs to the original analysis directory.
Reconstruction inputs/volumes are read-only; spatial ROIs are descriptive only.
"""
import hashlib
import io
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.colors import ListedColormap
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'recon_out/subset_convergence_20260910'
OUT = ROOT / 'reports/assets/osem-posem-convergence-20260910/audit'
ARMS = dict(mlem1000=1000, a5=200, a10=100, p5=200, p10=100,
            asched=100, psched=100)
LABELS = dict(mlem1000='MLEM', a5='OSEM N=5', a10='OSEM N=10',
              p5='POSEM N=5', p10='POSEM N=10', asched='OSEM staged', psched='POSEM staged')
COLORS = dict(mlem1000='#303030', a5='#2774AE', a10='#2774AE', p5='#CD731C',
              p10='#CD731C', asched='#66834A', psched='#A04C82')


def save_figure(fig, name):
    fig.savefig(OUT.parent/name, dpi=160)
    plt.close(fig)
    shutil.copyfile(OUT.parent/name, RUN/name)


def subset_strategy_figures():
    """Code-derived N5/N10 versus the exact published P16 (PMB 55, p2027).

    All plots are assignments, not image reconstructions or performance tests.
    Blue selects subset 0; gray excludes it. Other subset IDs stay explicit in
    the zoom, so viewers can verify the mapping without a 16-color legend.
    """
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    z,x=np.indices((48,48))
    def current(a,n):
        return (a%90+x+48*np.minimum(z,47-z))%n
    # Transcribed from the rendered original paper, section 2.4, page 2027.
    p16=np.array([[9,13,1,5],[0,4,8,12],[6,10,14,2],[15,3,7,11]])
    assert sorted(p16.ravel())==list(range(16))
    paper=p16[z%4,x%4]
    for n in (5,10):
        assert np.array_equal(current(0,n)[23],current(0,n)[24])
        assert np.array_equal(current(1,n),(current(0,n)+1)%n)
        labels=np.stack([current(a,n) for a in range(360)])
        assert np.all(np.bincount(labels.ravel(),minlength=n)==829440//n)
    assert np.all(paper[23]!=paper[24])
    assert np.all(np.bincount(paper.ravel())==144)
    cmap=ListedColormap(['#E8EDF0','#2774AE'])
    def detector(ax,labels,title):
        ax.imshow(labels==0,origin='upper',cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        ax.axhline(23.5,color='#C87722',ls='--',lw=1.3)
        ax.set(xticks=[0,12,24,36,47],yticks=[0,12,23,24,36,47],xlabel='探测器 x',ylabel='探测器 z')
        ax.set_title(title,fontsize=11,pad=10)
        # Avoid overlapping labels on the two adjacent seam rows.
        ax.set_yticks([0,12,24,36,47])
    with plt.rc_context({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False}):
        fig,axes=plt.subplots(2,3,figsize=(12,8.5),layout='constrained')
        for row,n in enumerate((5,10)):
            ax=axes[row,0]
            deg=np.arange(360)
            theta=np.deg2rad(deg)
            sel=deg%n==0
            ax.scatter(np.cos(theta),np.sin(theta),s=8,color='#C7CED4')
            ax.scatter(np.cos(theta[sel]),np.sin(theta[sel]),s=19,color='#2774AE')
            ax.set_aspect('equal')
            ax.set(xlim=(-1.25,1.25),ylim=(-1.25,1.25))
            for d in (0,90,180,270):
                t=np.deg2rad(d)
                ax.text(1.13*np.cos(t),1.13*np.sin(t),f'{d}°',ha='center',va='center')
            ax.text(0,0,f'每 {n}° 一个视角\n共 {360//n} 个视角\n每个选中视角取完整\n48×48 探测器平面',ha='center',va='center',fontsize=11)
            ax.axis('off')
            ax.set_title(f'当前 OSEM，N={n}：子集 0',fontsize=12)
            detector(axes[row,1],current(0,n),f'当前 POSEM，N={n}：子集 0\n完整视角 a=0，选择 {(current(0,n)==0).sum()} 像素')
            detector(axes[row,2],current(1,n),f'当前 POSEM，N={n}：子集 0\n完整视角 a=1，选择 {(current(1,n)==0).sum()} 像素')
        fig.suptitle('当前子集策略：整视角交错 vs 每个视角内的像素交错\n蓝色属于子集 0；灰色属于其他子集；橙线为 z=23/24 镜像接缝',fontsize=14)
        save_figure(fig,'subset_current_strategies.png')

        fig,axes=plt.subplots(2,3,figsize=(12.5,9),layout='constrained')
        variants=[(current(0,5),'当前 POSEM，N=5'),(current(0,10),'当前 POSEM，N=10'),
                  (paper,'原论文 P16，N=16')]
        for col,(labels,title) in enumerate(variants):
            detector(axes[0,col],labels,title+'\n子集 0 在 48×48 平面上的分布')
            ax=axes[1,col]
            crop=labels[20:28,:8]
            ax.imshow(crop==0,cmap=cmap,vmin=0,vmax=1,origin='upper',interpolation='nearest')
            for rz in range(8):
                for cx in range(8):
                    ax.text(cx,rz,str(crop[rz,cx]),ha='center',va='center',fontsize=11,
                            color='white' if crop[rz,cx]==0 else '#24323B')
            ax.axhline(3.5,color='#C87722',ls='--',lw=2)
            ax.set(xticks=range(8),yticks=range(8),yticklabels=range(20,28),
                   xlabel='探测器 x',ylabel='探测器 z')
            ax.set_title('接缝放大：数字为所属子集 ID\nz=23/24 的同 x 像素同组' if col<2 else
                         '4×4 原文图案，横纵各平铺两次\nz=23/24 的同 x 像素不同组',fontsize=11)
        fig.suptitle('当前 POSEM 与原论文图案：分配规则比较\n均展示视角 a=0；N 不相同，只比较规则，不作性能或质量比较',fontsize=14)
        save_figure(fig,'subset_current_vs_paper.png')
    return dict(paper_P16=p16.tolist(),paper_source='Branderhorst et al. 2010, section 2.4, printed page 2027',
                current_formula='(a%90 + x + 48*min(z,47-z))%N',checks='PASS')


def read_volume(path):
    h = dict(line.split(' = ', 1) for line in path.read_text().splitlines()
             if ' = ' in line)
    dims = tuple(map(int, h['DimSize'].split()))
    assert h['ElementByteOrderMSB'] == 'False'
    data = np.fromfile(path.parent / h['ElementDataFile'],
                       dtype={'MET_DOUBLE': '<f8', 'MET_FLOAT': '<f4'}[h['ElementType']])
    assert data.size == np.prod(dims) and np.isfinite(data).all()
    return data.reshape(dims[::-1])  # x + Nx*(y + Ny*z): NumPy [z,y,x]


def volume(arm, it, suffix=''):
    return read_volume(RUN / arm / f'recon_it{it}{suffix}.mhd')


def rel(a, b):
    return float(np.linalg.norm(a-b) / np.linalg.norm(b))


def spatial(v):
    yy, xx = np.mgrid[:100, :100]
    r = np.hypot(xx-50, yy-50)
    zmean = v[:, r < 38].mean(axis=1)
    return dict(axial_cv=float(zmean.std()/zmean.mean()),
                annulus_std=float(v[:, (r > 42) & (r < 48)].std(axis=1).mean()),
                ring_mean=float(v[:, (r > 26) & (r < 36)].mean()),
                disk_mean=float(zmean.mean()))


def flat(j, prefix=''):
    if isinstance(j, dict):
        return {key: val for k, v in j.items() for key, val in flat(v, prefix+'.'+k).items()}
    return {prefix: j}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planes(v):
    return [v[36], v[:, 50, :], v[:, :, 50]]


def main():
    # Independent non-cubic coordinate anchor, including every axis and boundary.
    nx, ny, nz = 3, 4, 5
    a = np.arange(nx*ny*nz).reshape(nz, ny, nx)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                assert a[z,y,x] == x + nx*(y + ny*z)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'saved outputs; no reconstruction rerun', 'coordinate_check': 'PASS'}
    result['subset_strategy_figures']=subset_strategy_figures()
    manifests = {arm: flat(json.loads((RUN/arm/'recon_manifest.json').read_text())) for arm in ARMS}
    result['manifest_differences'] = {
        k: {a: j[k] for a,j in manifests.items()} for k in manifests['mlem1000']
        if len({json.dumps(j[k]) for j in manifests.values()}) > 1}
    result['endpoint'] = {}
    for suffix in ('',):
        ref = volume('mlem1000', 1000, suffix)
        result['endpoint'][suffix or 'primary'] = {
            arm: dict(relL2_to_mlem1000=rel(volume(arm,it,suffix), ref),
                      **spatial(volume(arm,it,suffix))) for arm,it in ARMS.items()}
    ref = volume('mlem1000', 1000)
    result['subset_balance'] = {}
    for arm in ('a5', 'a10', 'p5', 'p10'):
        log = next((RUN/arm).glob('recon_*.log')).read_text(encoding='utf-8-sig')
        global_path = Path(re.search(r'compact denom cache hit: (.*\.mhd)', log)[1])
        paths = re.findall(r'compact subset \d+/\d+ denom cache (?:hit|saved): (.*\.mhd)', log)
        s = read_volume(global_path).ravel()
        sm = np.array([read_volume(Path(p)).ravel() for p in paths])
        n = len(sm)
        assert n == (5 if arm.endswith('5') else 10)
        valid = s > 0
        result['subset_balance'][arm] = {
            'N': n, 'sum_relL2': rel(sm.sum(axis=0), s),
            'weighted_rms_NSm_minus_S': float(np.linalg.norm(n*sm-s)/np.sqrt(n)/np.linalg.norm(s)),
            'max_abs_NSm_over_S_minus_1': float(np.max(np.abs(n*sm[:,valid]/s[valid]-1))),
            'supported_voxels': int(valid.sum()),
            'missing_subset_support': int((sm[:,valid] == 0).sum())}
    # Full detector coordinates independently anchor the canonical-row mapping.
    aa, zz, xx = np.indices((360,48,48))
    p = np.fromfile(Path(manifests['mlem1000']['.projection.1']), dtype='<f8', offset=8).reshape(360,48,48)
    result['projection_subsets'] = {}
    for n in (5,10):
        for mode in ('angles','pixels'):
            labels = (aa % 90) % n if mode == 'angles' else ((aa % 90)+48*np.minimum(zz,47-zz)+xx) % n
            bins = np.bincount(labels.ravel(),minlength=n)
            counts = np.bincount(labels.ravel(),weights=p.ravel(),minlength=n)
            assert bins.sum() == p.size and np.isclose(counts.sum(),p.sum())
            result['projection_subsets'][f'{mode}{n}'] = dict(bins=bins.tolist(),counts=counts.tolist(),
                count_cv=float(counts.std()/counts.mean()))
    result['timing'] = {}
    curves = json.loads((RUN/'curves.json').read_text())
    complete_times = {}
    for arm in ARMS:
        log = next((RUN/arm).glob('recon_*.log')).read_text(encoding='utf-8-sig')
        ts = {int(it): datetime.strptime(t,'%Y-%m-%d %H:%M:%S') for t,it in
              re.findall(r'(?m)^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[ITER\] (\d+)/',log)}
        last = datetime.strptime(re.search(r'(?m)^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[PERF\]',log)[1],
                                 '%Y-%m-%d %H:%M:%S')
        complete_times[arm] = {it:(ts.get(it+1,last)-ts[1]).total_seconds() for it in ts}
        hits = {}
        for threshold in (0.40,0.23,0.135,0.05):
            row = next((r for r in curves if r['arm']==arm and r['relL2_to_ref'] <= threshold),None)
            if row is not None and row['iter']+1 in ts:
                it = row['iter']
                hits[str(threshold)] = dict(iteration=it,updates=row['updates'],
                    start_s=(ts[it]-ts[1]).total_seconds(),
                    completed_by_s=(ts[it+1]-ts[1]).total_seconds())
        result['timing'][arm] = dict(version=re.search(r'=== SPECT_recon (.*?) ===',log)[1],
            path=re.search(r'compact iteration path: (.*)',log)[1],threshold_bounds_since_iter1=hits)
    paths = [ROOT/'multipinhole/config/spect_config.ini', ROOT/'multipinhole/config/algo_mlem.conf',
             ROOT/'multipinhole/config/bad_dector', ROOT/'build/Release/bin/Release/SPECT_recon.exe']
    pre = Path(manifests['mlem1000']['.projection.1'])
    paths += [pre, pre.parent/'preprocess_summary.json']
    paths += [Path(manifests['mlem1000']['.matrix.metadata_path'])]
    matrix = Path(manifests['mlem1000']['.matrix.chunk_csr_dir'])
    result['matrix_current_inventory'] = {p.name: dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                                           for p in matrix.glob('*.csrb')}
    paths += list(RUN.glob('*/recon_manifest.json'))
    paths += [RUN/a/f'recon_it{it}.raw' for a,it in ARMS.items()]
    result['current_sha256_not_historical_freeze'] = {str(p): sha(p) for p in paths}
    vmax = float(np.percentile(ref, 99.8))
    # Numeric line chart: 20 paired checkpoints per arm, shared zero-based axis.
    # Blue=angle, orange=pixel; dashed/open markers=N5, solid markers=N10.
    fig, ax = plt.subplots(figsize=(8,4.8),layout='constrained')
    for arm in ('a5','a10','p5','p10'):
        points = sorted((int(r['updates']),float(r['relL2_to_mlem_matched']))
                        for r in curves if r['arm']==arm)
        x,y = np.array(points,dtype=float).T
        assert len(x)==20 and np.isfinite(y).all()
        n5 = arm.endswith('5')
        ax.plot(x,y,color=COLORS[arm],label=LABELS[arm],ls='--' if n5 else '-',
                marker='o',ms=3,mfc='white' if n5 else COLORS[arm])
    ax.set(xlabel='Subset updates u',ylabel='relL2 to MLEM at the same u',ylim=(0,0.0042),
           title='Deviation from the MLEM trajectory\nPrimary output; lower means closer, not faster')
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=2))
    ax.grid(alpha=.2)
    ax.legend(ncol=2,fontsize=9,loc='upper left')
    assert ax.yaxis.get_converter() is None
    save_figure(fig,'mlem_equivalence_gap.png')

    # Matched comparisons only. Flat and staged pairs use separate shared
    # scales; the MLEM iteration change is a third scale, explicitly labelled.
    cases=[('a10',50,500),('p10',50,500),('a5',100,500),
           ('asched',90,350),('psched',90,350),('mlem1000',500,1000)]
    diffs=[np.abs(volume(a,it)-volume('mlem1000',u))[36] for a,it,u in cases]
    assert all(float(d.max())>0 for d in diffs)
    flat_limit=max(float(d.max()) for d in diffs[:3])
    staged_limit=max(float(d.max()) for d in diffs[3:5])
    fig, axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    result['difference_panels']=[]
    for i,(ax,d,(arm,it,u)) in enumerate(zip(axes.flat,diffs,cases)):
        cap=flat_limit if i<3 else staged_limit if i<5 else float(d.max())
        im=ax.imshow(d,origin='lower',cmap='magma',vmin=0,vmax=cap,interpolation='nearest')
        title=f'{LABELS[arm]} it{it} vs MLEM it{u}'
        scale_label='Flat shared scale' if i<3 else 'Staged shared scale' if i<5 else 'Iteration change; separate scale'
        ax.set_title(title+'\n'+scale_label,fontsize=10)
        ax.set_axis_off()
        bar=fig.colorbar(im,ax=ax,shrink=.82,pad=.025)
        bar.locator=MaxNLocator(5)
        bar.update_ticks()
        result['difference_panels'].append(dict(arm=arm,iteration=it,mlem_iteration=u,
                                                slice_max=float(d.max()),vmax=cap))
    fig.suptitle('Absolute primary-output differences |candidate - reference|\nAxial z=36; shared scales within flat / staged groups; native image units',fontsize=13)
    save_figure(fig,'difference_images.png')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for arm in ARMS:
        pts=sorted((r for r in curves if r['arm']==arm),key=lambda r:r['iter'])
        # Exact zero at the frozen reference is omitted on the logarithmic axis.
        pts=[r for r in pts if r['relL2_to_ref']>0]
        for ax,x in zip(axes,[[r['updates'] for r in pts],
                              [complete_times[arm][r['iter']]/60 for r in pts]]):
            ax.plot(x,[r['relL2_to_ref'] for r in pts],color=COLORS[arm],
                    ls='--' if arm.endswith('5') else '-',marker='o',ms=2,label=LABELS[arm])
    for ax in axes:
        ax.set_yscale('log')
        ax.set_ylabel('relL2 to MLEM it1000')
        ax.grid(alpha=.2)
    axes[0].set(xlabel='Subset updates u',title='Convergence by updates')
    axes[1].set(xlabel='Minutes since ITER 1 (completion upper bounds)',title='Convergence by elapsed time')
    axes[0].legend(fontsize=8,ncol=2)
    fig.suptitle('Fixed workload, primary output; 20 saved checkpoints per arm',fontsize=13)
    save_figure(fig,'relL2_curves.png')

    fig,axes=plt.subplots(3,4,figsize=(11,7.8),layout='constrained')
    for ri,arm in enumerate(('mlem1000','a10','p10')):
        for ci,u in enumerate((50,100,300,1000)):
            it=u if arm=='mlem1000' else u//10
            axes[ri,ci].imshow(volume(arm,it)[36],origin='lower',cmap='gray',vmin=0,vmax=vmax,interpolation='nearest')
            axes[ri,ci].set_title(f'{LABELS[arm]} it{it}; u={u}',fontsize=10)
            axes[ri,ci].axis('off')
    fig.suptitle('Primary output, axial z=36; shared MLEM1000 p99.8 window',fontsize=13)
    save_figure(fig,'slice_montage.png')
    frames=[]
    for k in range(1,21):
        fig,axes=plt.subplots(3,3,figsize=(9,8))
        for ci,(arm,it) in enumerate([('mlem1000',1000),('a10',5*k),('p10',5*k)]):
            for ri,img in enumerate(planes(volume(arm,it))):
                axes[ri,ci].imshow(img,origin='lower',cmap='gray',vmin=0,vmax=vmax,interpolation='nearest')
                axes[ri,ci].axis('off')
                if ri==0: axes[ri,ci].set_title(f'{arm} it{it}')
        fig.suptitle(f'Primary output; subset updates u={50*k}; rows axial/coronal/sagittal')
        fig.tight_layout()
        buf=io.BytesIO()
        fig.savefig(buf,format='png',dpi=85)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))
    frames[0].save(OUT/'compare_flatN10_corrected.gif',save_all=True,append_images=frames[1:],duration=550,loop=0)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS: coordinates, numeric curves, nonzero matched differences, primary PNG/GIF; '+str(OUT/'audit.json'))


if __name__ == '__main__':
    main()
