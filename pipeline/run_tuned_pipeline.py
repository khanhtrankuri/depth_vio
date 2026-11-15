"""Run the selected depth-assisted VINS configuration and mapping into one latest result.

Inputs are existing IGEV++ disparity caches and EuRoC calibration/images. This
does not run model inference or use GT inside estimation/mapping. Evaluation
compares against a frozen baseline reference retained as an input.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def prepare_depth(sequence,settings,target):
    from common import stereo_rectify,cam_calib
    from export_igev_vins_depth import raw_inverse_map
    from prepare_depth_sweep import select_exposures
    cache=ROOT/'outputs'/sequence/'stage8_fusion/disp'
    files=sorted(cache.glob('*.npz'))
    if not files:raise FileNotFoundError('Missing disparity inputs: '+str(cache))
    data=ROOT/'data'/sequence/'mav0'
    with (data/'cam0/data.csv').open() as handle:
        timestamps=[int(row[0]) for row in csv.reader(line for line in handle if not line.startswith('#')) if row]
    chosen=select_exposures([(int(p.stem),p.name) for p in files],timestamps,settings['depth_hz'])
    calibrations={camera:sha(data/camera/'sensor.yaml') for camera in ['cam0','cam1']}
    sources={name:sha(cache/name) for _,name in chosen}
    signature=dict(settings=settings,calibration_sha256=calibrations,source_sha256=sources,
                   exporter_sha256=sha(ROOT/'pipeline/export_igev_vins_depth.py'),
                   selector_sha256=sha(ROOT/'pipeline/prepare_depth_sweep.py'))
    manifest=target/'manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if previous.get('signature')!=signature:raise ValueError('Tuned depth inputs/config changed; choose a new depth directory')
        if not (target/'index.txt').exists() or any(not (target/(str(t)+'.bin')).exists() for t,_ in chosen):
            raise ValueError('Incomplete tuned depth export')
        print('Verified/reused',len(chosen),'selected depth exposures',flush=True)
        return
    target.mkdir(parents=True,exist_ok=True)
    rect=stereo_rectify();K,D,_,_=cam_calib('cam0')
    index=[]
    for i,(timestamp,name) in enumerate(chosen):
        with np.load(cache/name) as item:
            if item['disp'].shape!=rect['map0'][0].shape:raise ValueError('Disparity/calibration shape mismatch')
            observations=raw_inverse_map(item['disp'],item['valid'],rect,K,D,
                zmin=settings['zmin_m'],zmax=settings['zmax_m'],
                sigma_px=settings['sigma_px'],relative_sigma=settings['relative_sigma'])
        h,w=observations.shape[:2]
        path=target/(str(timestamp)+'.bin')
        with path.open('wb') as handle:
            handle.write(struct.pack('<II',w,h));handle.write(observations.tobytes())
        index.append(str(timestamp)+' '+path.name)
        if i%50==0:print('prepare tuned depth',i,'/',len(chosen),flush=True)
    save(manifest,dict(sequence=sequence,frames=len(chosen),sigma_px=settings['sigma_px'],
         relative_sigma=settings['relative_sigma'],calibration_sha256=calibrations,signature=signature,
         depth_hz=settings['depth_hz'],nominal_information_weight=settings['information_weight'],
         coordinate_frame='raw cam0 optical Z; inverse metres',
         uncertainty='uncalibrated sigma proxy; same stereo inputs as VINS',
         sampling='same exposure only; valid smooth raw 3x3 neighbourhood in C++'))
    (target/'index.txt').write_text('\n'.join(index)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',default='V1_01_easy',choices=['V1_01_easy'])
    parser.add_argument('--work',help='Default outputs/SEQ/latest; existing result is refused')
    parser.add_argument('--bag',help='Default data/SEQ/input.bag')
    parser.add_argument('--reference',help='Frozen baseline TUM; default outputs/SEQ/inputs/baseline_reference.tum')
    args=parser.parse_args()
    os.environ['EUROC_SEQ']=args.sequence
    settings=json.loads((ROOT/'vins_fusion/tuned_config.json').read_text())
    output=Path(args.work).resolve() if args.work else ROOT/'outputs'/args.sequence/'latest'
    if (output/'vio/vio.csv').exists() or (output/'run.json').exists():
        parser.error('Output already contains a run; select a fresh --work')
    inputs=ROOT/'outputs'/args.sequence/'inputs'
    bag=Path(args.bag).resolve() if args.bag else ROOT/'data'/args.sequence/'input.bag'
    reference=Path(args.reference).resolve() if args.reference else inputs/'baseline_reference.tum'
    for path in [bag,reference]:
        if not path.is_file():raise FileNotFoundError(path)
    depth=inputs/'igev_depth_w4_hz1'
    prepare_depth(args.sequence,settings,depth)
    output.mkdir(parents=True,exist_ok=True)
    common=[sys.executable]
    commands=[
        common+[str(ROOT/'pipeline/run_vins_fusion.py'),'--sequence',args.sequence,
                '--depth-dir',str(depth),'--bag',str(bag),'--work',str(output/'vio'),
                '--rate','0.5','--drain-seconds','15'],
        common+[str(ROOT/'pipeline/run_balanced_fusion.py'),'--sequence',args.sequence,
                '--poses',str(output/'vio/vins_fusion_traj_tum.txt'),'--work',str(output/'map')],
        common+[str(ROOT/'pipeline/evaluate_vins_depth.py'),'--baseline',str(reference),
                '--depth',str(output/'vio/vins_fusion_traj_tum.txt'),
                '--gt',str(ROOT/'data'/args.sequence/'mav0/state_groundtruth_estimate0/data.csv'),
                '--output',str(output/'evaluation'),'--verify-evo']]
    for command in commands:
        print('Running',Path(command[1]).name,flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    metadata=json.loads((output/'vio/run.json').read_text())
    cloud=output/'map/surfel_uniform/surface_support6_cc30_cloud.ply'
    surface=json.loads((cloud.parent/'surface_support6_cc30.json').read_text())
    metrics=json.loads((output/'evaluation/metrics.json').read_text())
    if metadata['status']!='completed' or metadata['solver_evaluation_failures']!=0:
        raise RuntimeError('Estimator failed validation')
    if metadata['trajectory']['max_pose_gap_s']>.151 or not cloud.is_file() or surface['vertices']<=0:
        raise RuntimeError('Trajectory/map validation failed')
    result=dict(status='completed',sequence=args.sequence,settings=settings,
        trajectory=metadata['trajectory'],surface=surface,metrics=metrics,
        reference_note='Frozen earlier depth-OFF run; not a fresh paired baseline',
        input_bag_sha256=sha(bag),reference_sha256=sha(reference),
        trajectory_sha256=sha(output/'vio/vins_fusion_traj_tum.txt'),
        cloud_sha256=sha(cloud),image_id=metadata['image_id'],
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in
            [Path(__file__),ROOT/'pipeline/run_vins_fusion.py',ROOT/'pipeline/fusion_v2.py',
             ROOT/'pipeline/fusion_tsdf.py',ROOT/'vins_fusion/tuned_config.json']})
    save(output/'run.json',result)
    print('Validated latest output:',output,flush=True)
    print('Surfels:',surface['vertices'],'ATE cm:',metrics['results']['depth']['ATE_RMSE_m']*100,flush=True)


if __name__=='__main__':main()
