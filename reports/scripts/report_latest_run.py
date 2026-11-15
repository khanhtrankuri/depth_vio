"""Publish the validated latest local run as compact, reviewable Git artifacts."""
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]

def main():
    work=ROOT/'outputs/V1_01_easy/latest'
    result=json.loads((work/'run.json').read_text())
    assert result['status']=='completed'
    metadata=json.loads((work/'vio/run.json').read_text())
    result['estimator_validation']={key:metadata[key] for key in ['solver_evaluation_failures','depth_factor_diagnostics','entrypoint_sha256','source_commit']}
    result['published_utc']=datetime.now(timezone.utc).isoformat()
    target=ROOT/'reports/artifacts/latest'
    target.mkdir(parents=True,exist_ok=True)
    (target/'run.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    shutil.copyfile(work/'vio/vins_fusion_traj_tum.txt',target/'trajectory.tum')
    cloud=work/'map/surfel_uniform/surface_support6_cc30_cloud.ply'
    if cloud.stat().st_size>50*1024**2:raise ValueError('Cloud too large for ordinary Git')
    shutil.copyfile(cloud,target/'point_cloud.ply')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with cloud.open('rb') as handle:
        while handle.readline().strip()!=b'end_header':pass
        points=np.frombuffer(handle.read(),dtype=[('x','<f8'),('y','<f8'),('z','<f8'),('r','u1'),('g','u1'),('b','u1')])
    sample=points[::max(1,len(points)//25000)]
    fig=plt.figure(figsize=(9,7),layout='constrained');ax=fig.add_subplot(projection='3d')
    ax.scatter(sample['x'],sample['y'],sample['z'],c=sample['r']/255.,cmap='gray',s=.5,vmin=0,vmax=1)
    ax.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]',title='Latest VINS-Fusion + IGEV++ surfel point cloud')
    ranges=np.array([np.ptp(sample[k]) for k in ['x','y','z']]);ax.set_box_aspect(ranges)
    ax.view_init(elev=25,azim=-65)
    fig.savefig(ROOT/'reports/figures/latest_point_cloud.png',dpi=150);plt.close(fig)
    shutil.copyfile(work/'evaluation/ate_comparison.png',ROOT/'reports/figures/latest_ate.png')
    metrics=result['metrics']['results'];base=metrics['baseline'];depth=metrics['depth']
    rows=[('ATE position (cm)',base['ATE_RMSE_m']*100,depth['ATE_RMSE_m']*100)]
    for lag in [1,5]:
        rows.extend([(f'RPE translation {lag}s (cm)',base[f'RPE_{lag}s']['translation_RMSE_m']*100,depth[f'RPE_{lag}s']['translation_RMSE_m']*100),
                     (f'RPE rotation {lag}s (deg)',base[f'RPE_{lag}s']['rotation_RMSE_deg'],depth[f'RPE_{lag}s']['rotation_RMSE_deg'])])
    lines=['# Latest VINS-Fusion + IGEV++ run','',
        'Sequence: V1_01_easy. Depth information weight: 4; depth exposure rate: 1 Hz. Sigma proxies: 1 px and 2.5%; optical Z range: 0.2–4 m. Playback: 0.5x; fixed extrinsic/time offset; loop closure disabled.','',
        'The baseline is a frozen earlier depth-OFF run, not a freshly replayed paired control. The settings were selected using GT on this same sequence; this run is not independent validation. RPE translation tradeoffs remain visible. GT is used only for evaluation, never for pose estimation or mapping.','',
        '| Metric | Frozen baseline | New depth run | Change |','|---|---:|---:|---:|']
    for name,b,d in rows:lines.append(f'| {name} | {b:.6f} | {d:.6f} | {100*(d/b-1):+.3f}% |')
    lines+=['',f"Matched poses: {result['metrics']['matched']}; independently cross-checked with evo. Native trajectory: {result['trajectory']['poses']} poses; max gap {result['trajectory']['max_pose_gap_s']:.3f}s. Ceres evaluation failures: {metadata['solver_evaluation_failures']}.",'',
        f"Point cloud: {result['surface']['vertices']:,} surfels; 3 cm voxels, support from at least six distinct frames, components smaller than 30 points removed. Native VIO coordinates; a point surface, not a mesh. No new Leica/reference-cloud mapping score is claimed.",'',
        '[Run metadata and metrics](artifacts/latest/run.json) · [Native T_WI trajectory](artifacts/latest/trajectory.tum) · [3D point cloud](artifacts/latest/point_cloud.ply)','',
        '![ATE comparison](figures/latest_ate.png)','','![Point cloud preview](figures/latest_point_cloud.png)','',
        '## Reproduce','',
        'Requires Docker Linux containers, a complete NumPy/SciPy/OpenCV/PyYAML/Pillow/PyTorch/evo environment, EuRoC images/calibration, existing IGEV++ disparity inputs, `data/V1_01_easy/input.bag`, and the frozen baseline in `outputs/V1_01_easy/inputs/baseline_reference.tum`. Build the depth image once using `docker build -f vins_fusion/Dockerfile.depth -t depth-vio-vins-igev:depth-v2 .` after building the baseline image as documented in `vins_fusion/README.md`.','',
        '```sh','python pipeline/run_tuned_pipeline.py','python reports/scripts/report_latest_run.py','```','',
        'Default output: `outputs/V1_01_easy/latest/`. A populated output is refused; select a fresh `--work` when repeating. The preserved disparity and rectification caches are required mapping inputs. Old run directories and generated HTML were removed after validating this result. Reports 01–10 remain historical documentation; their original local run folders may no longer exist.','']
    (ROOT/'reports/latest_run.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Published latest artifacts:',target)

if __name__=='__main__':main()