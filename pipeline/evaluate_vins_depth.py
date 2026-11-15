"""Paired ATE/RPE evaluation: common timestamps, raw EuRoC GT, SE3 alignment, scale=1.

RPE uses relative SE3 transforms at exactly 1s and 5s (tolerance 1ms), all valid
start timestamps. No GT bias/orientation corrections and no training/tuning.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tum(path):
    rows=[line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith('#')]
    ns=np.array([int(Decimal(row[0])*Decimal(1000000000)) for row in rows],dtype=np.int64)
    a=np.array([[float(v) for v in row[1:8]] for row in rows])
    if len(ns)<2 or not np.isfinite(a).all() or (np.diff(ns)<=0).any():raise ValueError('Invalid TUM trajectory')
    return ns,a[:,:3],Rotation.from_quat(a[:,3:7]).as_matrix()


def align(P,G):
    U,_,Vt=np.linalg.svd((P-P.mean(0)).T@(G-G.mean(0)))
    D=np.eye(3);D[-1,-1]=np.linalg.det(U@Vt)
    R=(U@D@Vt).T
    translation=G.mean(0)-R@P.mean(0)
    return P@R.T+translation,R,translation


def scores(times,P,orientations,G,gt_orientations,verify_evo=False):
    P,R,t=align(P,G);Q=np.einsum('ij,njk->nik',R,orientations)
    errors=np.linalg.norm(P-G,axis=1)
    result=dict(n=len(times),ATE_RMSE_m=float(np.sqrt(np.mean(errors**2))),
                ATE_max_m=float(errors.max()),alignment_R=R.tolist(),alignment_t=t.tolist())
    if verify_evo:
        from evo.core import metrics
        from evo.core.trajectory import PosePath3D
        E=np.tile(np.eye(4),(len(P),1,1));E[:,:3,:3]=Q;E[:,:3,3]=P
        GT=np.tile(np.eye(4),(len(G),1,1));GT[:,:3,:3]=gt_orientations;GT[:,:3,3]=G
        ape=metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((PosePath3D(poses_se3=GT),PosePath3D(poses_se3=E)))
        diff=float(np.max(np.abs(ape.error-errors)))
        if diff>1e-9:raise AssertionError('ATE disagrees with evo')
        result['evo_ATE_max_abs_difference_m']=diff
    for seconds in [1,5]:
        candidates=np.searchsorted(times,times+seconds*1000000000)
        valid=candidates<len(times);i=np.flatnonzero(valid);j=candidates[valid]
        valid=np.abs(times[j]-times[i]-seconds*1000000000)<=1000000;i=i[valid];j=j[valid]
        if len(i)==0:raise ValueError('No RPE pairs at requested delta')
        estimate_R=np.einsum('nji,njk->nik',Q[i],Q[j])
        reference_R=np.einsum('nji,njk->nik',gt_orientations[i],gt_orientations[j])
        estimate_t=np.einsum('nji,nj->ni',Q[i],P[j]-P[i])
        reference_t=np.einsum('nji,nj->ni',gt_orientations[i],G[j]-G[i])
        trans=np.linalg.norm(estimate_t-reference_t,axis=1)
        rotations=np.einsum('nji,njk->nik',reference_R,estimate_R)
        angles=np.rad2deg(Rotation.from_matrix(rotations).magnitude())
        result['RPE_'+str(seconds)+'s']=dict(pairs=len(i),translation_RMSE_m=float(np.sqrt(np.mean(trans**2))),
                                           rotation_RMSE_deg=float(np.sqrt(np.mean(angles**2))))
        if verify_evo:
            matrices=np.array([metrics.RPE.rpe_base(GT[a],GT[b],E[a],E[b]) for a,b in zip(i,j)])
            dt=float(np.max(np.abs(np.linalg.norm(matrices[:,:3,3],axis=1)-trans)))
            dr=float(np.max(np.abs(np.rad2deg(Rotation.from_matrix(matrices[:,:3,:3]).magnitude())-angles)))
            if dt>1e-9 or dr>1e-7:raise AssertionError('RPE disagrees with evo')
            result['RPE_'+str(seconds)+'s']['evo_max_abs_difference_m']=dt
            result['RPE_'+str(seconds)+'s']['evo_max_abs_difference_deg']=dr
    return result,errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline');parser.add_argument('--depth')
    parser.add_argument('--gt',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--historical',help='Optional original baseline; scored on same timestamps')
    parser.add_argument('--verify-evo',action='store_true',help='Independently cross-check every ATE/RPE error with installed evo')
    parser.add_argument('--methods-json',help='JSON object of method name to TUM path for a common-timestamp multi-method comparison')
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args()
    if args.methods_json:
        paths=json.loads(Path(args.methods_json).read_text())
        if 'baseline' not in paths:parser.error('Multi-method comparison requires a baseline key')
    else:
        if not args.baseline or not args.depth:parser.error('Provide --baseline and --depth, or --methods-json')
        paths=dict(baseline=args.baseline,depth=args.depth)
    if args.historical:paths['historical']=args.historical
    trajectories={name:read_tum(path) for name,path in paths.items()}
    # Round epoch timestamps to microseconds only for association keys (stored originals are retained).
    lookup={name:{int((t+500)//1000):i for i,t in enumerate(traj[0])} for name,traj in trajectories.items()}
    for name,table in lookup.items():
        if len(table)!=len(trajectories[name][0]):raise ValueError('Duplicate timestamp association key')
    common=sorted(set.intersection(*(set(table) for table in lookup.values())))
    with open(args.gt) as handle:rows=list(csv.reader(line for line in handle if not line.startswith('#')))
    gt_times=np.array([int(row[0]) for row in rows],dtype=np.int64)
    gt=np.array([[float(v) for v in row[1:]] for row in rows])
    times=np.array(common,dtype=np.int64)*1000
    right=np.clip(np.searchsorted(gt_times,times),0,len(gt_times)-1);left=np.maximum(0,right-1)
    idx=np.where(abs(gt_times[left]-times)<=abs(gt_times[right]-times),left,right)
    valid=abs(gt_times[idx]-times)<=10000000
    common=[key for key,keep in zip(common,valid) if keep];times=times[valid];idx=idx[valid]
    if len(common)<100 or len(np.unique(idx))!=len(idx):raise ValueError('Insufficient/non-unique GT associations')
    G=gt[idx,:3];GQ=Rotation.from_quat(gt[idx][:,[4,5,6,3]]).as_matrix()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    results={};errors={}
    for name,(all_t,P,Q) in trajectories.items():
        selected=np.array([lookup[name][key] for key in common])
        results[name],errors[name]=scores(times,P[selected],Q[selected],G,GQ,args.verify_evo)
        results[name]['input_poses']=len(all_t)
    summary=dict(protocol='common exposure timestamps across all methods; GT nearest <=10ms; SE3 fit per method on all common positions; scale fixed to 1; all valid 1s/5s pairs',
                 matched=len(times),duration_s=float((times[-1]-times[0])/1e9),
                 max_gt_association_ms=float(np.max(np.abs(gt_times[idx]-times))/1e6),
                 evaluator_sha256=sha(__file__),evo_crosscheck=args.verify_evo,
                 reference='raw state_groundtruth_estimate0; known V1_01 orientation caveat, no method-derived correction',
                 results=results,source_sha256={k:sha(p) for k,p in dict(paths,gt=args.gt).items()})
    if 'depth' in results:
        summary['ATE_RMSE_m_change_percent']=100*(results['depth']['ATE_RMSE_m']/results['baseline']['ATE_RMSE_m']-1)
    (output/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for name in results:ax.plot((times-times[0])/1e9,errors[name]*100,label=name,lw=1)
        ax.set(xlabel='Time from first common pose [s]',ylabel='Position error [cm]',title='VINS-Fusion: baseline vs IGEV depth factors (one SE3 fit, scale=1)')
        ax.legend();ax.grid(alpha=.2);fig.savefig(output/'ate_comparison.png',dpi=140);plt.close(fig)
    print(json.dumps({k:{'ATE_cm':v['ATE_RMSE_m']*100,'RPE_1s':v['RPE_1s'],'RPE_5s':v['RPE_5s']} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
