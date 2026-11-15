"""Publish compact measured depth-in-VINS comparison without copying trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',required=True)
    args=parser.parse_args();work=Path(args.benchmark).resolve()
    metrics=json.loads((work/'comparison/metrics.json').read_text())
    baseline=json.loads((work/'baseline/run.json').read_text())
    depth=json.loads((work/'depth/run.json').read_text())
    assert baseline['status']==depth['status']=='completed'
    assert baseline['image_id']==depth['image_id'] and not baseline['igev_depth_enabled'] and depth['igev_depth_enabled']
    assert baseline['solver_evaluation_failures']==depth['solver_evaluation_failures']==0
    assert depth['depth_factor_diagnostics']['max_factors']>0
    # Identical configuration and calibration are required for the paired comparison.
    for name in ['estimator_config.yaml','cam0_mei.yaml','cam1_mei.yaml']:
        assert (work/'baseline'/name).read_bytes()==(work/'depth'/name).read_bytes()
    evidence=dict(metrics=metrics,baseline_run=baseline,depth_run=depth,
                  cpp_validation='All local Jacobians agree with central finite differences within 1e-5; pose translation solve, finite behind-camera penalty and timestamp tests pass',
                  python_validation='16 tests passed',
                  final_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [ROOT/'pipeline/evaluate_vins_depth.py',ROOT/'pipeline/run_vins_fusion.py',ROOT/'vins_fusion/apply_depth_patch.py',
                       *sorted((ROOT/'vins_fusion/depth').glob('*'))]})
    artifacts=ROOT/'reports/artifacts';artifacts.mkdir(exist_ok=True)
    (artifacts/'vins_depth_v1.json').write_text(json.dumps(evidence,indent=2))
    figure=ROOT/'reports/figures/vins_depth_ate_v1.png'
    shutil.copyfile(work/'comparison/ate_comparison.png',figure)
    rows=[];names=dict(historical='Original saved VINS-Fusion',baseline='Same executable, depth OFF',depth='Same executable, IGEV depth ON')
    for name in ['historical','baseline','depth']:
        if name not in metrics['results']:continue
        r=metrics['results'][name];a=r['RPE_1s'];b=r['RPE_5s']
        rows.append('| '+names[name]+f" | {r['ATE_RMSE_m']*100:.3f} | {a['translation_RMSE_m']*100:.3f} | {a['rotation_RMSE_deg']:.3f} | {b['translation_RMSE_m']*100:.3f} | {b['rotation_RMSE_deg']:.3f} |")
    change=metrics['ATE_RMSE_m_change_percent']
    diagnostics=depth['depth_factor_diagnostics']
    report=f'''# 09 — IGEV++ depth inside VINS-Fusion

Measured on V1_01_easy, 2026-11-13 (Asia/Bangkok). Integration and benchmark
completed; this report does not assume that depth improves accuracy.

The extension adds optical-depth factors to Ceres pose/landmark optimisation and
the oldest-frame marginalisation prior. Accepted depth replaces the correlated
right-stereo factor at that observation. The original baseline is preserved.
See [implementation and commands](../vins_fusion/DEPTH_INTEGRATION.md).

| Method | ATE RMSE cm | RPE trans 1s cm | RPE rot 1s deg | RPE trans 5s cm | RPE rot 5s deg |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

Depth-on ATE changes by **{change:+.2f}%** relative to the paired depth-off run
(negative is better). This is one run per mode on one sequence, not a statistical
or cross-dataset claim. The saved historical baseline also shows run variability.

![ATE comparison](figures/vins_depth_ate_v1.png)

## Protocol and evidence

- {metrics['matched']} shared GT-matched exposure timestamps over {metrics['duration_s']:.1f} seconds.
- One SE3 alignment per method fitted to all common positions, scale fixed at 1.
- RPE uses all available pairs separated by 1s/5s within 1ms, identical across methods.
- Raw EuRoC GT, nearest association <=10ms; no orientation offset fitted or applied.
- Same image, calibration and solver settings; sequential 0.5x replay and 15s drain.
- Both trajectories pass the end-of-input coverage check. Initialisation/missing
  observations are documented in the run metadata.
- Both runs have zero failed residual/Jacobian evaluation windows. Prototype v1
  was rejected for 3 such windows; its diagnostic artifacts are kept separately.
- Depth active in {diagnostics['with_depth']}/{diagnostics['optimizations']} optimisations;
  mean {diagnostics['mean_factors']:.2f}, maximum {diagnostics['max_factors']} factors per window.
- C++ Jacobian, pose-solve and sampler controls pass; all 16 Python tests pass.
- Every ATE/RPE error was cross-checked with evo: {metrics['evo_crosscheck']}.

Compact metrics, run metadata, image ID, depth input hashes and code fingerprints:
[vins_depth_v1.json](artifacts/vins_depth_v1.json).
Large trajectories/logs stay under `outputs/V1_01_easy/vins_depth_benchmark_v2/`.

## Limits

Depth comes from the existing IGEV++ cache rather than a fresh inference run.
The export validates L-R masks, range, calibration and neighbourhood consistency;
the historical cache does not establish fresh model-weight provenance.
Only matching cached exposures are used; depth is not propagated between frames.
Sigma is an uncalibrated 5%/2px proxy fixed before scoring, with no GT tuning.
The learned depths are correlated with stereo observations; replacing one factor
does not make all measurements independent. V1_01's documented reference
orientation caveat also affects interpretation of raw-GT RPE. Solver time limits
and asynchronous ROS execution introduce run variability. V1_02/V1_03 have not
been benchmarked with this extension. The baseline stays the default; depth is opt-in.
'''
    (ROOT/'reports/09_vins_depth_integration.md').write_text(report,encoding='utf-8')
    print('Wrote reports/09_vins_depth_integration.md')


if __name__=='__main__':main()
