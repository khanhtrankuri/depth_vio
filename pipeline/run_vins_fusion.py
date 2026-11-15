"""Build/run official VINS-Fusion on EuRoC ASL and export native IMU TUM poses."""
import argparse
import csv
from decimal import Decimal
import json
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ['V1_01_easy', 'V1_02_medium', 'V1_03_difficult']


def trajectory_summary(trajectory, camera_csv):
    with open(trajectory) as handle:
        times = [Decimal(line.split()[0]) for line in handle
                 if line.strip() and not line.startswith('#')]
    with open(camera_csv) as handle:
        image_times = [Decimal(row[0])/Decimal(1000000000)
                       for row in csv.reader(line for line in handle if not line.startswith('#')) if row]
    if len(times) < 2 or not image_times:
        raise ValueError('Missing trajectory or image timestamps')
    lag = float(image_times[-1]-times[-1])
    if abs(lag) > 1:
        raise ValueError('Trajectory does not cover end of input (lag %.3fs); inspect logs, lower --rate or increase --drain-seconds' % lag)
    return dict(poses=len(times), duration_s=float(times[-1]-times[0]),
                initialization_delay_s=float(times[0]-image_times[0]), last_image_lag_s=lag,
                max_pose_gap_s=float(max(b-a for a,b in zip(times,times[1:]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True, choices=SEQUENCES)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--image',help='Default: official baseline, or depth-v2 when --depth-dir is supplied')
    parser.add_argument('--rate', type=float, default=.5)
    parser.add_argument('--drain-seconds', type=int, default=15)
    parser.add_argument('--work',help='Isolated estimator output directory')
    parser.add_argument('--bag',help='Reuse an existing ROS bag, mounted read-only')
    parser.add_argument('--depth-dir',help='Raw-cam0 IGEV inverse-depth export (enables optimizer factors)')
    args = parser.parse_args()
    if args.image is None:
        args.image='depth-vio-vins-igev:depth-v2' if args.depth_dir else 'depth-vio-vins-fusion:be55a937'
    if not 0 < args.rate <= 1 or args.drain_seconds < 1:
        parser.error('Use 0 < --rate <= 1 and positive --drain-seconds')
    data = ROOT/'data'/args.sequence
    for sensor in ['cam0', 'cam1', 'imu0']:
        if not (data/'mav0'/sensor/'data.csv').is_file():
            parser.error('Missing EuRoC ASL input: ' + str(data/'mav0'/sensor/'data.csv'))
    output = Path(args.work).resolve() if args.work else ROOT/'outputs'/args.sequence/('stage7_vins_igev' if args.depth_dir else 'stage7_vins_fusion')
    if (output/'vio.csv').exists() or (output/'vins_fusion_traj_tum.txt').exists():
        parser.error('Existing run at ' + str(output) + '; move it before rerunning')
    subprocess.run(['docker', 'info'], check=True, stdout=subprocess.DEVNULL)
    if args.build:
        dockerfile='Dockerfile.depth' if args.depth_dir else 'Dockerfile'
        subprocess.run(['docker', 'build', '-f', str(ROOT/'vins_fusion'/dockerfile),
                        '-t', args.image, str(ROOT)], check=True)
    revision = subprocess.check_output(['docker', 'run', '--rm', '--entrypoint', 'git',
        args.image, '-C', '/catkin_ws/src/VINS-Fusion', 'rev-parse', 'HEAD'], text=True).strip()
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                       args.image], text=True).strip()
    output.mkdir(parents=True, exist_ok=True)
    run = ['docker', 'run', '--rm', '--mount', 'type=bind,source='+str(data)+',target=/data,readonly',
           '--mount', 'type=bind,source='+str(output)+',target=/out',
           '--mount', 'type=bind,source='+str(ROOT/'vins_fusion/entrypoint.sh')+',target=/integration/entrypoint.sh,readonly']
    if args.bag:
        bag=Path(args.bag).resolve()
        if not bag.is_file(): parser.error('Missing bag: '+str(bag))
        run+=['--mount','type=bind,source='+str(bag)+',target=/input/euroc.bag,readonly','-e','BAG_PATH=/input/euroc.bag']
    elif not (output/'euroc.bag').exists():
        subprocess.run(run+[args.image, 'bag'], check=True)
    if args.depth_dir:
        depth=Path(args.depth_dir).resolve()
        depth_metadata=json.loads((depth/'manifest.json').read_text())
        if depth_metadata['sequence']!=args.sequence: parser.error('Depth sequence mismatch')
        for camera in ['cam0','cam1']:
            digest=hashlib.sha256((data/'mav0'/camera/'sensor.yaml').read_bytes()).hexdigest()
            if digest!=depth_metadata['calibration_sha256'][camera]:
                parser.error('Depth export calibration changed: '+camera)
        run+=['--mount','type=bind,source='+str(depth)+',target=/depth,readonly','-e','IGEV_DEPTH_DIR=/depth']
    metadata = dict(estimator='VINS-Fusion', image=args.image,
                    source_commit=revision, image_id=image_id,
                    sequence=args.sequence, pose_frame='T_WI', quaternion='Hamilton xyzw',
                    loop_closure=False, estimate_extrinsic=False, estimate_td=False,
                    playback_rate=args.rate, drain_seconds=args.drain_seconds, status='running')
    metadata['igev_depth_enabled']=bool(args.depth_dir)
    metadata['entrypoint_sha256']=hashlib.sha256((ROOT/'vins_fusion/entrypoint.sh').read_bytes()).hexdigest()
    if args.depth_dir: metadata['depth_manifest']=depth_metadata
    manifest = output/'run.json'
    manifest.write_text(json.dumps(metadata, indent=2))
    try:
        subprocess.run(run+['-e', 'PLAYBACK_RATE='+str(args.rate), '-e',
                           'DRAIN_SECONDS='+str(args.drain_seconds), args.image, 'run'], check=True)
        metadata['trajectory'] = trajectory_summary(output/'vins_fusion_traj_tum.txt',
                                                     data/'mav0/cam0/data.csv')
        log=(output/'estimator.log').read_text(errors='replace')
        metadata['solver_evaluation_failures']=log.count('Terminating: Residual and Jacobian evaluation failed')
        if metadata['solver_evaluation_failures']:
            raise ValueError('Ceres reported failed residual/Jacobian evaluation; run rejected')
        if args.depth_dir:
            import re
            counts=[int(n) for n in re.findall(r'IGEV_FACTORS timestamp=[\d.]+ count=(\d+)',log)]
            if not counts or max(counts)==0: raise ValueError('No IGEV optimizer factors were activated')
            metadata['depth_factor_diagnostics']=dict(optimizations=len(counts),with_depth=sum(n>0 for n in counts),
                                                      max_factors=max(counts),mean_factors=sum(counts)/len(counts))
    except (subprocess.CalledProcessError, ValueError, OSError):
        metadata['status'] = 'failed'
        manifest.write_text(json.dumps(metadata, indent=2))
        raise
    metadata['status'] = 'completed'
    manifest.write_text(json.dumps(metadata, indent=2))
    print('TUM trajectory:', output/'vins_fusion_traj_tum.txt')
    print('Next:', sys.executable, 'pipeline/run_balanced_fusion.py --sequence', args.sequence)


if __name__ == '__main__':
    main()
