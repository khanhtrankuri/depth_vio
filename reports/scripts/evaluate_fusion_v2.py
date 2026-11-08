"""Evaluation-only registration and fixed-ROI metrics; never imported by mapper.

Development selection is restricted to V1_01_easy. Freeze selection.json before
running final V1_02/V1_03 metrics. Baseline and new clouds use the same reference,
ROI, voxel sampling and thresholds. No ICP or GT-derived rotation corrections.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pipeline'))
from fusion_v2 import write_json, sha, ROOT
from audit_cached import ply, voxel

DEST=ROOT/'reports/artifacts/fusion_v2'
ROI_LO=np.array([-4.8,-4.8,-.6]); ROI_HI=np.array([4.8,4.8,3.6])


def crop(p): return p[((p>=ROI_LO)&(p<=ROI_HI)).all(1)]


def score(points,ref):
    if len(points)==0: raise RuntimeError('Cannot score an empty map')
    p=voxel(crop(points),.02); tree=cKDTree(p)
    acc=cKDTree(ref).query(p,workers=2)[0]; comp=tree.query(ref,workers=2)[0]
    metrics=dict(n_prediction=len(p),n_reference=len(ref),accuracy_mean_m=float(acc.mean()),
                 accuracy_rmse_m=float(np.sqrt(np.mean(acc**2))),accuracy_median_m=float(np.median(acc)),
                 accuracy_p95_m=float(np.quantile(acc,.95)),completeness_mean_m=float(comp.mean()),
                 chamfer_m=float((acc.mean()+comp.mean())/2))
    for cm in [2,5,10]:
        precision=float(np.mean(acc<cm/100)); recall=float(np.mean(comp<cm/100))
        metrics[f'precision_{cm}cm']=precision; metrics[f'recall_{cm}cm']=recall
        metrics[f'f1_{cm}cm']=2*precision*recall/max(precision+recall,1e-12)
    return metrics


def choose_balanced(results):
    uniform=results.get('uniform_support0',results['legacy_tsdf'])['metrics']
    eligible={k:a for k,a in results.items() if k not in ['legacy_tsdf','legacy_naive'] and
              a['metrics']['recall_5cm']>=uniform['recall_5cm']-.02}
    best=max(a['metrics']['f1_5cm'] for a in eligible.values())
    near={k:a for k,a in eligible.items() if a['metrics']['f1_5cm']>=best-.005}
    # User priority: balance. Avoid a 4x integration cost for a negligible F1
    # difference; tie within the same backend by the better F1.
    selected=min(near,key=lambda k:(near[k]['integration']['integrate_ms_mean'],-near[k]['metrics']['f1_5cm']))
    return selected,eligible,best


def evaluate(seq,variants):
    scan_path=ROOT/'data'/seq/'mav0/pointcloud0/data.ply'
    digest=sha(scan_path)
    reference_cache=DEST/('reference_roi_'+digest+'.npy')
    if reference_cache.exists():
        ref=np.load(reference_cache)
    else:
        scan=ply(scan_path); ref=crop(voxel(scan,.02)); del scan
        np.save(reference_cache,ref)
    out=ROOT/'outputs'/seq
    alignment=json.loads((out/'stage9_eval/vio_metrics.json').read_text())
    R,t=np.array(alignment['align_R']),np.array(alignment['align_t'])
    results={}
    baseline=out/'stage8_fusion/tsdf_cloud_openvins.ply'
    results['legacy_tsdf']=dict(metrics=score(ply(baseline),ref),source_sha256=sha(baseline))
    naive=out/'stage8_fusion/naive_accum_openvins.ply'
    results['legacy_naive']=dict(metrics=score(ply(naive),ref),source_sha256=sha(naive))
    for variant in variants:
        for path in sorted((out/'stage8_fusion_v2'/variant).glob('*support*_cloud.ply')):
            support=path.stem.removeprefix('mesh_').removeprefix('surface_').removesuffix('_cloud')
            key=variant+'_'+support
            p=ply(path)@R.T+t
            results[key]=dict(metrics=score(p,ref),source_sha256=sha(path),
                              integration=json.loads((path.parent/'integration.json').read_text()),
                              extraction=json.loads((path.parent/(path.stem.replace('_cloud','')+'.json')).read_text()))
    return results


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--sequence',required=True)
    parser.add_argument('--variants',nargs='+',default=['uniform','spatial','temporal'])
    parser.add_argument('--select',action='store_true')
    args=parser.parse_args(); DEST.mkdir(parents=True,exist_ok=True)
    if args.select and args.sequence!='V1_01_easy': raise ValueError('Selection is development-only V1_01')
    start=time.perf_counter(); results=evaluate(args.sequence,args.variants)
    protocol=dict(reference='original Leica scan',registration='one existing SE3 trajectory alignment, evaluation only; no ICP',
                  roi_lo=ROI_LO.tolist(),roi_hi=ROI_HI.tolist(),voxel_m=.02,
                  limitations='point-to-point on sampled surfaces; fixed ROI not independent visibility-certified reference',
                  audit='no gtrotcorr; no reference enters inference/preparation/integration')
    write_json(DEST/(args.sequence+'.json'),dict(sequence=args.sequence,protocol=protocol,results=results,wall_s=time.perf_counter()-start))
    for name,a in results.items():
        m=a['metrics']; print(name,'mean',round(m['accuracy_mean_m'],4),'P/R/F1@5',*[round(m[k],4) for k in ['precision_5cm','recall_5cm','f1_5cm']],flush=True)
    if args.select:
        selected,eligible,best=choose_balanced(results)
        choice=dict(development_sequence='V1_01_easy',selected=selected,
                    variant=selected.rsplit('_support',1)[0],
                    min_surface_support=eligible[selected]['extraction'].get('min_surface_support',int(re.search(r'_support(\d+)',selected).group(1))),
                    min_component_points=eligible[selected]['extraction'].get('min_component_points',0),
                    min_component_faces=eligible[selected]['extraction'].get('min_component_faces',30),
                    criterion='fastest integration within 0.5pp of best development F1@5cm; recall no more than 2pp below native uniform/support0; ties use higher F1',
                    best_development_f1=best,
                    heldout_sequences=['V1_02_medium','V1_03_difficult'],
                    result_sha256=sha(DEST/(args.sequence+'.json')),
                    code_sha256={p.name:sha(p) for p in (ROOT/'pipeline').glob('fusion_*.py')})
        selection=DEST/'selection.json'
        if selection.exists(): raise RuntimeError('Selection already frozen; do not tune using held-out results')
        write_json(selection,choice); print('FROZEN',selected,flush=True)


if __name__=='__main__': main()
