"""Run paired depth-off/depth-on VINS on the same bag and evaluate common poses.

The same patched executable is used with its extension disabled/enabled. Runs
are sequential to avoid CPU contention; original baseline is also scored if present.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',default='V1_01_easy')
    parser.add_argument('--depth-dir',required=True)
    parser.add_argument('--bag',required=True)
    parser.add_argument('--image',default='depth-vio-vins-igev:depth-v2')
    parser.add_argument('--work',required=True)
    parser.add_argument('--rate',type=float,default=.5)
    args=parser.parse_args()
    work=Path(args.work).resolve();work.mkdir(parents=True,exist_ok=True)
    depth=Path(args.depth_dir).resolve();bag=Path(args.bag).resolve()
    files=[ROOT/'vins_fusion/apply_depth_patch.py',*sorted((ROOT/'vins_fusion/depth').glob('*')),
           ROOT/'vins_fusion/entrypoint.sh',ROOT/'pipeline/evaluate_vins_depth.py',ROOT/'pipeline/run_vins_fusion.py']
    (work/'sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    for name in ['baseline','depth']:
        command=[sys.executable,str(ROOT/'pipeline/run_vins_fusion.py'),'--sequence',args.sequence,
                 '--image',args.image,'--work',str(work/name),'--bag',str(bag),'--rate',str(args.rate)]
        if name=='depth':command+=['--depth-dir',str(depth)]
        print('Running paired estimator:',name,flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    command=[sys.executable,str(ROOT/'pipeline/evaluate_vins_depth.py'),
             '--baseline',str(work/'baseline/vins_fusion_traj_tum.txt'),
             '--depth',str(work/'depth/vins_fusion_traj_tum.txt'),
             '--gt',str(ROOT/'data'/args.sequence/'mav0/state_groundtruth_estimate0/data.csv'),
             '--output',str(work/'comparison')]
    historical=ROOT/'outputs'/args.sequence/'stage7_vins_fusion/vins_fusion_traj_tum.txt'
    if historical.is_file():command+=['--historical',str(historical)]
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
