"""Bounded subset experiment; contract is report section 8. Never overwrites runs.

python tools/run_subset_followup_20260910.py init|diagnostic|analyze|paired|summarize|smoke
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'recon_out/subset_followup_20260910'
OLD = ROOT / 'recon_out/subset_convergence_20260910'
EXE = ROOT / 'build/verify-vs/bin/Release/SPECT_recon.exe'
SOURCE_CONFIG = ROOT / 'multipinhole/config'
MATRIX = ROOT / 'test_data/6module/Lu177_sysmat/matrix_PSC_PIN26x28x20_208keV_120x120x220_ZFOLD'
INPUT = Path('E:/Xiang_Project/SPECT/6Module/experiments/recon_param_study/pre/derenzo_lu_6s/Bed1.bin')
THRESHOLDS = [.135, .05, .005, .002]
MODES = ('angles', 'pixels', 'pixels2d')
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    h = dict(line.split(' = ', 1) for line in path.read_text(encoding='utf-8').splitlines() if ' = ' in line)
    assert h['ElementByteOrderMSB'] == 'False'
    dims = tuple(map(int, h['DimSize'].split()))
    a = np.fromfile(path.parent / h['ElementDataFile'], dtype={'MET_DOUBLE': '<f8', 'MET_FLOAT': '<f4'}[h['ElementType']])
    assert a.size == np.prod(dims) and np.isfinite(a).all(), str(path)
    return a.reshape(dims[::-1])


def rel(a, b):
    return float(np.linalg.norm((a-b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-300))


def identity():
    files = [EXE, INPUT, MATRIX/'metadata.bin', OLD/'mlem1000/recon_it1000.raw',
             OUT/'config/spect_config.ini', OUT/'config/algo_mlem.conf', OUT/'config/bad_dector']
    return {'sha256': {str(p): sha(p) for p in files}, 'matrix_inventory': {
        p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(MATRIX.glob('chunk_*.csrb'))}}


def initialize():
    if (OUT/'identity.json').exists():
        assert identity() == json.loads((OUT/'identity.json').read_text()), 'Frozen identity changed'
        return
    (OUT/'config').mkdir(parents=True, exist_ok=True)
    s = (SOURCE_CONFIG/'spect_config.ini').read_text(encoding='utf-8-sig')
    cache = ROOT/'build/Release/bin/Release/cache/compact_denom'
    s = re.sub(r'^compact_denom_cache_dir\s*=.*$', 'compact_denom_cache_dir = '+cache.as_posix(), s, flags=re.M)
    (OUT/'config/spect_config.ini').write_text(s, encoding='utf-8')
    for name in ('algo_mlem.conf', 'bad_dector'):
        (OUT/'config'/name).write_bytes((SOURCE_CONFIG/name).read_bytes())
    patch = subprocess.check_output(['git', 'diff', '--binary', '--', 'multipinhole'], cwd=ROOT)
    (OUT/'source.patch').write_bytes(patch)
    save(OUT/'identity.json', identity())
    save(OUT/'contract.json', {'thresholds': THRESHOLDS, 'numerical_closure_relL2': 1e-5,
        'diagnostic_iterations': [0,10,30,80,100], 'candidate': '(a+x+3*z_compact)%N',
        'repeats': 2, 'run_timeout_s': 1200, 'reference': 'finite MLEM1000 primary, not truth',
        'code_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'source_patch_sha256': sha(OUT/'source.patch')})


def resource_sample(pid=None):
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, creationflags=FLAGS).strip().split(',')
    foreign = []
    for p in psutil.process_iter(['pid','name','cpu_times']):
        name = (p.info['name'] or '').lower()
        if p.pid != pid and (name.startswith('castor-recon') or name in {
                'spect_recon.exe','gpu_px_direct.exe','super_cuda.exe','nvcc.exe','cl.exe','link.exe'}):
            foreign.append({'pid':p.pid,'name':name})
    return {'available_gib':psutil.virtual_memory().available/2**30,
            'gpu_mib':int(gpu[0]),'gpu_util':int(gpu[1]),'foreign':foreign,
            'rss_gib':psutil.Process(pid).memory_info().rss/2**30 if pid else None}


def wait_ready():
    started = time.monotonic()
    while True:
        sample = resource_sample()
        if sample['available_gib'] >= 42 and sample['gpu_mib'] < 3000 and sample['gpu_util'] < 25 and not sample['foreign']:
            return sample
        print('Waiting for uncontaminated full-cache resources:', sample, flush=True)
        if time.monotonic()-started > 900:
            raise RuntimeError('Resource precondition unavailable for 15 minutes; outputs preserved')
        time.sleep(30)


def run(name, iterations, mode='angle', schedule=None, diagnostic=False, save_every=5,
        timing=True):
    initialize()
    dest = OUT/name
    if (dest/'result.json').exists():
        result = json.loads((dest/'result.json').read_text())
        assert result['exit_code'] == 0 and result['complete'], name
        print('Already complete:', name, flush=True)
        return
    if dest.exists():
        raise RuntimeError(f'Incomplete run preserved: {dest}; use a distinct attempt name')
    preflight = wait_ready() if timing else resource_sample()
    dest.mkdir()
    cmd = [str(EXE), '-config', str(OUT/'config/spect_config.ini'), '-opti','mlem',
           '-chunkCSRDir',str(MATRIX),'-df',str(INPUT),'-fout',str(dest/'recon'),
           '-vb','1','-it',str(iterations) if schedule else f'{iterations}:10',
           '-subset-mode',mode]
    if save_every is not None: cmd += ['-save-every',str(save_every)]
    if schedule: cmd += ['-subsets',schedule]
    if diagnostic: cmd += ['-subset-diagnostics','0,10,30,80,100']
    save(dest/'command.json', {'argv':cmd,'preflight':preflight,'diagnostic':diagnostic,'timing':timing})
    start_epoch, start = time.time(), time.monotonic()
    samples = []
    timed_out = False
    with (dest/'stdout.log').open('wb') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)
        try:
            while proc.poll() is None:
                try:
                    samples.append({'seconds':time.monotonic()-start, **resource_sample(proc.pid)})
                except psutil.NoSuchProcess:
                    break
                if time.monotonic()-start > 1200:
                    proc.kill()
                    timed_out = True
                    break
                time.sleep(2)
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            save(dest/'resources.json', samples)
    log_text = (dest/'stdout.log').read_text(encoding='utf-8-sig')
    core = re.search(r'iteration_core_s=([\d.]+)',log_text)
    final = dest/f'recon_it{iterations}.mhd'
    complete = code == 0 and final.exists() and core is not None
    result = {'exit_code':code,'complete':complete,'wall_s':time.monotonic()-start,
              'start_epoch':start_epoch,'iteration_core_s':float(core[1]) if core else None,
              'max_rss_gib':max(s['rss_gib'] for s in samples),
              'max_gpu_mib':max(s['gpu_mib'] for s in samples),
              'foreign_samples':sum(bool(s['foreign']) for s in samples),
              'full_cache':bool(re.search(r'page-locked: 6 chunks',log_text)),
              'diagnostic':diagnostic,'timing':timing,'timed_out':timed_out}
    save(dest/'result.json', result)
    assert complete, f'Failed run: {name}; see stdout.log'
    assert identity() == json.loads((OUT/'identity.json').read_text()), 'Input identity changed'
    print(name, result, flush=True)


def summarize():
    reference = read(OLD/'mlem1000/recon_it1000.mhd')
    summary = {}
    for receipt in sorted(OUT.glob('*/result.json')):
        result = json.loads(receipt.read_text())
        if not result['complete']: continue
        result['timing_valid'] = (result.get('timing',True) and not result['diagnostic'] and result['full_cache']
                                  and result['foreign_samples'] == 0)
        log = (receipt.parent/'stdout.log').read_text(encoding='utf-8-sig')
        prepare = re.search(r'denom_and_matrix_prepare_s=([\d.]+)',log)
        result['prepare_s'] = float(prepare[1]) if prepare else None
        records = []
        for p in receipt.parent.glob('recon_it*.mhd'):
            match = re.fullmatch(r'recon_it(\d+)\.mhd',p.name)
            if match:
                records.append({'iteration':int(match[1]),'relL2':rel(read(p),reference),
                    'wall_s':p.stat().st_mtime-result['start_epoch']})
        records.sort(key=lambda r:r['iteration'])
        result['checkpoints'] = records
        result['first_threshold'] = {str(t):next((r for r in records if r['relL2'] <= t),None) for t in THRESHOLDS}
        summary[receipt.parent.name] = result
    save(OUT/'summary.json',summary)
    comparison = {'selected':{}, 'missing_clean_repeats':[], 'arms':{}, 'manifest_equal':None}
    common = None
    for arm in ('a10','schedule','p10','p2d'):
        chosen = []
        for repeat in (1,2):
            base = f'{arm}_r{repeat}'
            name = next((n for n in (base,base+'_retry1')
                         if n in summary and summary[n]['timing_valid']), None)
            comparison['selected'][base] = name
            if name is None:
                comparison['missing_clean_repeats'].append(base)
                continue
            chosen.append(name)
            manifest = json.loads((OUT/name/'recon_manifest.json').read_text())
            del manifest['timestamp_utc']
            for field in ('iterations','subsets','subset_mode'):
                del manifest['algorithm'][field]
            for field in ('dir','saved_iteration','files'):
                del manifest['output'][field]
            if common is None: common = manifest
            assert manifest == common, f'Uncontrolled effective configuration difference: {name}'
            comparison['manifest_equal'] = True
        if len(chosen) != 2: continue
        records = [summary[n] for n in chosen]
        comparison['arms'][arm] = {
            'runs':chosen,
            'core_s':[r['iteration_core_s'] for r in records],
            'wall_s':[r['wall_s'] for r in records],
            'prepare_s':[r['prepare_s'] for r in records],
            'endpoint_relL2':[r['checkpoints'][-1]['relL2'] for r in records],
            'threshold_wall_s':{str(t):[r['first_threshold'][str(t)]['wall_s']
                if r['first_threshold'][str(t)] else None for r in records] for t in THRESHOLDS}}
        endpoints = [read(OUT/n/f"recon_it{summary[n]['checkpoints'][-1]['iteration']}.mhd") for n in chosen]
        comparison['arms'][arm]['repeat_endpoint_relL2'] = rel(*endpoints)
    assert identity() == json.loads((OUT/'identity.json').read_text()), 'Frozen identity changed'
    save(OUT/'comparison.json',comparison)
    if not comparison['missing_clean_repeats']:
        plot_comparison(summary,comparison,reference)
    print(json.dumps(comparison,indent=2))


def plot_comparison(summary,comparison,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assets = ROOT/'reports/assets/osem-posem-convergence-20260910'
    labels = {'a10':'OSEM N10','schedule':'OSEM 10:80,5:30,1:50',
              'p10':'POSEM N10','p2d':'POSEM 2D N10'}
    colors = ('#2774AE','#8A4CAA','#CD731C','#328A62')
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    endpoints = []
    for (arm,label),color in zip(labels.items(),colors):
        for repeat,name in enumerate(comparison['arms'][arm]['runs']):
            checkpoints = summary[name]['checkpoints']
            updates = [min(r['iteration'],80)*10+max(min(r['iteration']-80,30),0)*5+
                       max(r['iteration']-110,0) if arm=='schedule' else r['iteration']*10
                       for r in checkpoints]
            for ax,x in zip(axes,(updates,[r['wall_s'] for r in checkpoints])):
                ax.plot(x,[r['relL2'] for r in checkpoints],color=color,
                        ls='-' if repeat==0 else '--',alpha=1 if repeat==0 else .55,
                        label=label if repeat==0 else None)
        first = comparison['arms'][arm]['runs'][0]
        iteration = summary[first]['checkpoints'][-1]['iteration']
        endpoints.append(read(OUT/first/f'recon_it{iteration}.mhd'))
    for ax,xlabel in zip(axes,('sub-updates','seconds since command launch')):
        ax.set(xlabel=xlabel,ylabel='relative L2 to finite MLEM1000',yscale='log')
        ax.grid(alpha=.2)
        for threshold in THRESHOLDS: ax.axhline(threshold,color='gray',lw=.6,ls=':')
    axes[0].legend(fontsize=8)
    axes[1].set_title('Solid: repeat 1; dashed: repeat 2')
    fig.savefig(assets/'subset_followup_curves.png',dpi=160)
    plt.close(fig)
    assert reference.shape == (66,100,100) and all(x.shape==reference.shape for x in endpoints)
    z = 36  # Same output-grid anchor as the original audited report.
    differences = [np.abs(x-reference) for x in endpoints]
    assert all(np.max(x[z]) > 0 for x in differences)
    vmax = float(np.quantile(reference,.998))
    dmax = float(np.quantile(np.stack([x[z] for x in differences]),.995))
    fig,axes = plt.subplots(2,5,figsize=(15,6.5),layout='constrained')
    for ax,volume,label in zip(axes[0],[reference]+endpoints,['MLEM1000']+list(labels.values())):
        im = ax.imshow(volume[z],origin='lower',cmap='gray',vmin=0,vmax=vmax)
        ax.set_title(label,fontsize=9)
        ax.axis('off')
    fig.colorbar(im,ax=list(axes[0]),shrink=.75,label='primary intensity')
    axes[1,0].axis('off')
    axes[1,0].text(.5,.5,'Absolute difference\nfrom MLEM1000\n(shared scale)',ha='center',va='center')
    for ax,volume in zip(axes[1,1:],differences):
        im = ax.imshow(volume[z],origin='lower',cmap='magma',vmin=0,vmax=dmax)
        ax.axis('off')
    fig.colorbar(im,ax=list(axes[1,1:]),shrink=.75,label='absolute primary difference')
    fig.suptitle('1000 sub-updates; axial output z=36; first clean repeat; upper scales p99.8 / p99.5')
    fig.savefig(assets/'subset_followup_primary.png',dpi=160)
    plt.close(fig)


def smoke_defaults():
    run('default_final_smoke',3,schedule='1',save_every=None,timing=False)
    folder = OUT/'default_final_smoke'
    files = sorted(p.name for p in folder.glob('recon_it*'))
    assert files == ['recon_it3.mhd','recon_it3.raw','recon_it3_butter.mhd','recon_it3_butter.raw'], files
    manifest = json.loads((folder/'recon_manifest.json').read_text())
    assert manifest['algorithm']['save_interval'] == 0
    assert manifest['output']['filters']['smooth_apply_stage'] == 'none'
    assert manifest['output']['filters']['butter_apply_stage'] == 'output'
    for name in ('recon_it3.mhd','recon_it3_butter.mhd'): read(folder/name)
    print('Default final-only primary + Butterworth smoke: PASS',flush=True)


def analyze_diagnostics():
    from scipy.ndimage import binary_erosion
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = OUT/'diagnostic/subset_diagnostics'
    full_s = read(d/'S_full.mhd')
    initial = read(d/'it0_state.mhd')
    support = (full_s > 0) & (initial > 0)
    regions = {'support':support,
               'low_sensitivity':support & (full_s <= np.quantile(full_s[support],.1)),
               'edge_2vox':support & ~binary_erosion(support,iterations=2)}
    result = {'regions':{k:int(v.sum()) for k,v in regions.items()},
              'closure':{},'balance':{},'updates':[],
              'endpoint_replay_relL2':rel(read(OUT/'diagnostic/recon_it100.mhd'),read(OLD/'a10/recon_it100.mhd'))}
    assert result['endpoint_replay_relL2'] <= 1e-5, 'Diagnostic changed protected angle trajectory'
    z,x = np.indices((24,48))
    old5 = np.stack([(a+x+48*z)%5 for a in range(90)])
    new5 = np.stack([(a+x+3*z)%5 for a in range(90)])
    new10 = np.stack([(a+x+3*z)%10 for a in range(90)])
    assert np.array_equal(old5,new5) and np.array_equal(new10%5,new5)
    assert np.all(np.bincount(new10.ravel()) == 103680//10)
    result['mapping_checks'] = {'n5_same_canonical_partition':True,
        'n10_to_n5_nested':True,'n10_full_bins_per_subset':82944,
        'all_eight_symmetry_branches_stay_with_canonical_row':True}
    sensitivity = {}
    for mode in MODES:
        s = np.stack([read(d/f'S_{mode}_{k}.mhd') for k in range(10)])
        closure = rel(s.sum(axis=0),full_s)
        assert closure <= 1e-5, (mode,'sensitivity closure',closure)
        result['closure'][mode] = {'sensitivity':closure,'numerator':{}}
        for count in (5,10):
            sm = s if count == 10 else s[:5]+s[5:]
            sensitivity[mode,count] = sm[:,support]
            result['balance'][f'{mode}_{count}'] = float(np.linalg.norm(
                (count*sm-full_s)[:,support])/np.sqrt(count)/np.linalg.norm(full_s[support]))
        del s
    fig, axes = plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    for iteration in (0,10,30,80,100):
        state = read(d/f'it{iteration}_state.mhd')
        full_q = read(d/f'it{iteration}_q_full.mhd')
        r_full = full_q[support]/np.maximum(full_s[support],1e-12)
        for mode in MODES:
            q = np.stack([read(d/f'it{iteration}_q_{mode}_{k}.mhd') for k in range(10)])
            closure = rel(q.sum(axis=0), full_q)
            assert closure <= 1e-5, (iteration,mode,'numerator closure',closure)
            result['closure'][mode]['numerator'][iteration] = closure
            for count in (5,10):
                qm = q if count == 10 else q[:5]+q[5:]
                sm = sensitivity[mode,count]
                # Match applyMlemUpdate: positive S is floored at 1e-12;
                # zero subset S with positive global S leaves the voxel intact.
                r = np.divide(qm[:,support],np.maximum(sm,1e-12),out=np.ones_like(sm),where=sm>0)
                delta = state[support][None,:]*(r-r_full)
                entry = {'iteration':iteration,'subupdates':10*iteration,'mode':mode,'count':count,'regions':{}}
                for key, mask in regions.items():
                    selected = mask[support]
                    disagreement = np.linalg.norm(delta[:,selected])/np.sqrt(count)
                    full_step = np.linalg.norm(state[mask]*(r_full[selected]-1))
                    norm = np.linalg.norm(state[mask])
                    entry['regions'][key] = {
                        'update_disagreement_rel_image':float(disagreement/norm),
                        'full_update_rel_image':float(full_step/norm),
                        'disagreement_over_full_step':float(disagreement/max(full_step,1e-300)),
                        'zero_subset_sensitivity':int((sm[:,selected]<=0).sum()),
                        'zero_subset_update':int((qm[:,support][:,selected]==0).sum())}
                result['updates'].append(entry)
            del q
    for ax, region in zip(axes,regions):
        for mode,color in zip(MODES,('#2774AE','#CD731C','#328A62')):
            for count,style in ((5,'--'),(10,'-')):
                records = [r for r in result['updates'] if r['mode']==mode and r['count']==count and r['iteration']>0]
                ax.plot([r['subupdates'] for r in records],
                        [r['regions'][region]['disagreement_over_full_step'] for r in records],
                        style, marker='o', color=color,label=f'{mode} N={count}')
        ax.axhline(1,color='gray',lw=.8)
        ax.set(xlabel='baseline sub-updates',ylabel='subset disagreement / full-data step',title=region)
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    asset=ROOT/'reports/assets/osem-posem-convergence-20260910/subset_update_diagnostics.png'
    fig.savefig(asset,dpi=160)
    plt.close(fig)
    save(OUT/'diagnostic_summary.json',result)
    print(json.dumps({'replay':result['endpoint_replay_relL2'],'balance':result['balance'],
        'ratios':{f"{r['iteration']}_{r['mode']}_{r['count']}":r['regions']['support']['disagreement_over_full_step']
                  for r in result['updates']}},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['init','diagnostic','analyze','paired','summarize','smoke'])
    args = parser.parse_args()
    if args.stage == 'init': initialize()
    elif args.stage == 'diagnostic': run('diagnostic',100,diagnostic=True)
    elif args.stage == 'summarize': summarize()
    elif args.stage == 'analyze':
        analyze_diagnostics()
    elif args.stage == 'smoke': smoke_defaults()
    elif args.stage == 'paired':
        schedule = json.loads((OUT/'contract.json').read_text())['schedule_candidate']
        iterations = sum(int(s.split(':')[1]) for s in schedule.split(','))
        for name in ['a10_r1','schedule_r1','p10_r1','p2d_r1','p2d_r2','p10_r2','schedule_r2','a10_r2']:
            params = dict(iterations=iterations if name.startswith('schedule') else 100,
                          mode='pixels2d' if name.startswith('p2d') else 'pixels' if name.startswith('p10') else 'angle',
                          schedule=schedule if name.startswith('schedule') else None)
            receipt = OUT/name/'result.json'
            result = json.loads(receipt.read_text()) if receipt.exists() else None
            if result is None or result['complete']:
                run(name, **params)
                result = json.loads(receipt.read_text())
            if not result['complete'] or result['foreign_samples'] or not result['full_cache']:
                # Keep contaminated evidence and take the first clean retry;
                # never select a repeat because it happened to be faster.
                run(name+'_retry1', **params)
        summarize()
