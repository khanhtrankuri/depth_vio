"""Sequential full-sequence depth-weight/frequency screening and confirmation.

Screening uses 1x playback with a fresh depth-off baseline. The candidate with
lowest ATE is then compared to a fresh baseline at 0.5x. This is development-set
tuning on V1_01, not held-out validation or a guarantee of improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TUM='vins_fusion_traj_tum.txt'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def validate_run(path):
    metadata=json.loads((path/'run.json').read_text())
    if metadata['status']!='completed' or metadata.get('solver_evaluation_failures')!=0:
        raise ValueError('Estimator did not complete cleanly: '+str(path))
    summary=metadata['trajectory']
    if summary['poses']<1430 or summary['max_pose_gap_s']>.151:
        raise ValueError('Trajectory dropout/insufficient coverage: '+str(path))
    return metadata


def run_estimator(args,path,rate,config=None):
    path=Path(path)
    if (path/TUM).exists():
        meta=validate_run(path)
        if meta['playback_rate']!=rate:raise ValueError('Existing run has different playback rate')
        if bool(config)!=meta['igev_depth_enabled']:raise ValueError('Existing run has wrong depth mode')
        if config and meta['depth_manifest']['sweep']['identifier']!=config['identifier']:
            raise ValueError('Existing run has different depth settings')
        print('Verified/reused',path.name,flush=True);return
    if (path/'vio.csv').exists():raise ValueError('Incomplete output exists; use a new work directory: '+str(path))
    command=[sys.executable,str(ROOT/'pipeline/run_vins_fusion.py'),'--sequence',args.sequence,
             '--image',args.image,'--bag',str(Path(args.bag).resolve()),'--work',str(path),
             '--rate',str(rate),'--drain-seconds','15']
    if config:command+=['--depth-dir',config['depth_dir']]
    print('Running',path.name,'rate',rate,'settings',None if config is None else config['identifier'],flush=True)
    subprocess.run(command,cwd=ROOT,check=True)
    validate_run(path)


def evaluate(args,methods,folder):
    folder.mkdir(parents=True,exist_ok=True)
    write(folder/'methods.json',methods)
    with (folder/'evaluation.log').open('w') as handle:
        subprocess.run([sys.executable,str(ROOT/'pipeline/evaluate_vins_depth.py'),
                        '--methods-json',str(folder/'methods.json'),
                        '--gt',str(ROOT/'data'/args.sequence/'mav0/state_groundtruth_estimate0/data.csv'),
                        '--output',str(folder),'--verify-evo','--no-plot'],
                       cwd=ROOT,check=True,stdout=handle,stderr=subprocess.STDOUT)
    metrics=json.loads((folder/'metrics.json').read_text())
    if metrics['matched']<1420:raise ValueError('Too few common GT matches for a fair comparison')
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',default='V1_01_easy',choices=['V1_01_easy'])
    parser.add_argument('--configs',required=True);parser.add_argument('--bag',required=True)
    parser.add_argument('--work',required=True);parser.add_argument('--image',default='depth-vio-vins-igev:depth-v2')
    parser.add_argument('--screen-rate',type=float,default=1.);parser.add_argument('--confirmation-rate',type=float,default=.5)
    parser.add_argument('--confirmation-pairs',type=int,default=1)
    args=parser.parse_args();work=Path(args.work).resolve();work.mkdir(parents=True,exist_ok=True)
    configs=json.loads(Path(args.configs).read_text());ordered=list(configs);random.Random(42).shuffle(ordered)
    plan=dict(sequence=args.sequence,selection='lowest position ATE RMSE on development sequence; all RPE values retained',
              configs=configs,order=[c['identifier'] for c in ordered],screen_rate=args.screen_rate,
              confirmation_rate=args.confirmation_rate,confirmation_pairs=args.confirmation_pairs,
              image=args.image,phase='development tuning; no held-out sequence',
              source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [Path(__file__),ROOT/'pipeline/run_vins_fusion.py',ROOT/'pipeline/evaluate_vins_depth.py',ROOT/'vins_fusion/entrypoint.sh']})
    if (work/'plan.json').exists():
        if json.loads((work/'plan.json').read_text())!=plan:raise ValueError('Sweep plan/source changed; use a new work directory')
    else:write(work/'plan.json',plan)
    state=dict(status='running',stage='screening',completed=[],failures={},started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(work/'progress.json',state)
    baseline=work/'screening/baseline'
    run_estimator(args,baseline,args.screen_rate)
    methods={'baseline':str(baseline/TUM)}
    baseline_meta=validate_run(baseline)
    for number,config in enumerate(ordered,1):
        name=config['identifier'];path=work/'screening'/name
        state['current']=name;write(work/'progress.json',state)
        try:
            run_estimator(args,path,args.screen_rate,config)
            meta=validate_run(path)
            if meta['image_id']!=baseline_meta['image_id']:raise ValueError('Image changed during sweep')
            for calibration in ['estimator_config.yaml','cam0_mei.yaml','cam1_mei.yaml']:
                if (baseline/calibration).read_bytes()!=(path/calibration).read_bytes():raise ValueError('Estimator calibration/config changed')
            methods[name]=str(path/TUM)
            metrics=evaluate(args,methods,work/'screening/comparison')
            result=metrics['results'][name]
            print('SCREEN',number,'/',len(ordered),name,'ATE_cm',round(result['ATE_RMSE_m']*100,5),
                  'RPE1_cm',round(result['RPE_1s']['translation_RMSE_m']*100,5),flush=True)
            state['completed'].append(name)
        except (subprocess.CalledProcessError,ValueError) as error:
            state['failures'][name]=str(error);print('Rejected config',name,str(error),flush=True)
        write(work/'progress.json',state)
    metrics=evaluate(args,methods,work/'screening/comparison')
    eligible=[c for c in configs if c['identifier'] in metrics['results']]
    if not eligible:raise RuntimeError('No valid candidate runs')
    candidate=min(eligible,key=lambda c:metrics['results'][c['identifier']]['ATE_RMSE_m'])
    write(work/'selected.json',dict(candidate=candidate,screening_metrics=metrics))
    state['stage']='confirmation';state['selected']=candidate['identifier'];write(work/'progress.json',state)
    confirmations=[]
    for repeat in range(args.confirmation_pairs):
        folder=work/'confirmation'/('pair_'+str(repeat+1))
        run_estimator(args,folder/'baseline',args.confirmation_rate)
        run_estimator(args,folder/'candidate',args.confirmation_rate,candidate)
        comparison=evaluate(args,{'baseline':str(folder/'baseline'/TUM),'candidate':str(folder/'candidate'/TUM)},folder/'comparison')
        confirmations.append(comparison)
    summary=dict(plan=plan,selected=candidate,screening=metrics,confirmation=confirmations,failures=state['failures'])
    write(work/'summary.json',summary)
    state['status']='completed';state['stage']='done';write(work/'progress.json',state)
    print('COMPLETED',candidate['identifier'],'summary:',work/'summary.json',flush=True)


if __name__=='__main__':main()
