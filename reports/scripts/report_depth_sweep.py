"""Report measured weight/frequency sweep and a separately replayed confirmation pair."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[2]


def compare(base,candidate):
    return dict(ATE_percent=100*(candidate['ATE_RMSE_m']/base['ATE_RMSE_m']-1),
                RPE_trans_1s_percent=100*(candidate['RPE_1s']['translation_RMSE_m']/base['RPE_1s']['translation_RMSE_m']-1),
                RPE_rot_1s_percent=100*(candidate['RPE_1s']['rotation_RMSE_deg']/base['RPE_1s']['rotation_RMSE_deg']-1),
                RPE_trans_5s_percent=100*(candidate['RPE_5s']['translation_RMSE_m']/base['RPE_5s']['translation_RMSE_m']-1),
                RPE_rot_5s_percent=100*(candidate['RPE_5s']['rotation_RMSE_deg']/base['RPE_5s']['rotation_RMSE_deg']-1))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',required=True)
    args=parser.parse_args();work=Path(args.work).resolve()
    summary=json.loads((work/'summary.json').read_text());screen=summary['screening'];results=screen['results']
    selected=summary['selected'];base=results['baseline'];configs=summary['plan']['configs']
    confirmations=[]
    for i,metric in enumerate(summary['confirmation'],1):
        bm=metric['results']['baseline'];cm=metric['results']['candidate']
        folder=work/'confirmation'/('pair_'+str(i))
        for name in ['baseline','candidate']:
            metadata=json.loads((folder/name/'run.json').read_text())
            assert metadata['status']=='completed' and metadata['solver_evaluation_failures']==0
            assert metadata['trajectory']['max_pose_gap_s']<=.151
        assert (folder/'baseline/estimator_config.yaml').read_bytes()==(folder/'candidate/estimator_config.yaml').read_bytes()
        confirmations.append(dict(pair=i,baseline=bm,candidate=cm,changes=compare(bm,cm)))
    confirmed_ate_better=all(p['candidate']['ATE_RMSE_m']<p['baseline']['ATE_RMSE_m'] for p in confirmations)
    confirmed_all_1s_better=confirmed_ate_better and all(
        p['candidate']['RPE_1s']['translation_RMSE_m']<=p['baseline']['RPE_1s']['translation_RMSE_m'] and
        p['candidate']['RPE_1s']['rotation_RMSE_deg']<=p['baseline']['RPE_1s']['rotation_RMSE_deg'] for p in confirmations)
    confirmed_all_metrics_better=confirmed_all_1s_better and all(
        p['candidate']['RPE_5s']['translation_RMSE_m']<=p['baseline']['RPE_5s']['translation_RMSE_m'] and
        p['candidate']['RPE_5s']['rotation_RMSE_deg']<=p['baseline']['RPE_5s']['rotation_RMSE_deg'] for p in confirmations)
    evidence=dict(summary=summary,confirmation_summary=confirmations,confirmed_ATE_better=confirmed_ate_better,
                  confirmed_ATE_and_both_1s_RPE_better=confirmed_all_1s_better,
                  confirmed_all_five_metrics_nonworse=confirmed_all_metrics_better,
                  qualification='GT-guided tuning and repeated replay on V1_01 only; not independent generalisation validation',
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [ROOT/'pipeline/prepare_depth_sweep.py',ROOT/'pipeline/sweep_vins_depth.py',ROOT/'pipeline/evaluate_vins_depth.py']})
    (ROOT/'reports/artifacts/vins_depth_sweep_v1.json').write_text(json.dumps(evidence,indent=2,allow_nan=False))
    weight_values=[.25,1.,4.];hz_values=[.5,1.,2.]
    names={c['identifier']:c for c in configs};matrix=[np.full((3,3),np.nan) for _ in range(3)]
    rows=[]
    for config in configs:
        name=config['identifier']
        if name not in results:
            rows.append(f"| {config['weight']:g} | {config['hz']:g} | rejected | — | — | — | — | — |");continue
        r=results[name];delta=compare(base,r);row=weight_values.index(config['weight']);column=hz_values.index(config['hz'])
        for m,key in zip(matrix,['ATE_percent','RPE_trans_1s_percent','RPE_rot_1s_percent']):m[row,column]=delta[key]
        rows.append(f"| {config['weight']:g} | {config['hz']:g} | {r['ATE_RMSE_m']*100:.4f} | {delta['ATE_percent']:+.3f}% | {r['RPE_1s']['translation_RMSE_m']*100:.4f} | {r['RPE_1s']['rotation_RMSE_deg']:.4f} | {r['RPE_5s']['translation_RMSE_m']*100:.4f} | {r['RPE_5s']['rotation_RMSE_deg']:.4f} |")
    fig,axes=plt.subplots(1,3,figsize=(11,4.4),layout='constrained')
    for ax,m,title in zip(axes,matrix,['ATE change [%]','RPE translation, 1s [%]','RPE rotation, 1s [%]']):
        scale=max(.01,float(np.nanmax(np.abs(m))));norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale)
        im=ax.imshow(m,cmap='RdBu_r',norm=norm)
        ax.set(xticks=range(3),xticklabels=['0.5','1','2'],yticks=range(3),yticklabels=['0.25','1','4'],xlabel='Depth frequency [Hz]',ylabel='Depth weight',title=title)
        for i in range(3):
            for j in range(3):
                value=m[i,j]
                text='failed' if not np.isfinite(value) else f'{value:+.2f}'
                rgba=im.cmap(norm(value)) if np.isfinite(value) else (1,1,1,1)
                luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,text,ha='center',va='center',color='black' if luminance>.55 else 'white')
    fig.suptitle('Development sweep on V1_01: change vs fresh baseline at 1x playback (negative is better)')
    fig.savefig(ROOT/'reports/figures/vins_depth_sweep_v1.png',dpi=150,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    confirmation_rows=[]
    for p in confirmations:
        for key,label in [('baseline','Depth OFF'),('candidate','Selected depth ON')]:
            r=p[key]
            confirmation_rows.append(f"| {p['pair']} | {label} | {r['ATE_RMSE_m']*100:.4f} | {r['RPE_1s']['translation_RMSE_m']*100:.4f} | {r['RPE_1s']['rotation_RMSE_deg']:.4f} | {r['RPE_5s']['translation_RMSE_m']*100:.4f} | {r['RPE_5s']['rotation_RMSE_deg']:.4f} |")
    report=f'''# 10 — Depth weight and frequency sweep

V1_01_easy, 2026-11-13 (Asia/Bangkok). This is development-set tuning using GT.
The selected setting is confirmed by another replay of the same sequence, not
by an independent dataset. Default baseline behaviour is preserved.

## Experiment design

- Nine fixed combinations: nominal Gaussian information multipliers 0.25, 1, 4;
  matching depth-exposure frequencies approximately 0.5, 1, 2 Hz, phase fixed at 0.
- Weight changes sigma by `1/sqrt(weight)`. Huber loss means the complete robust
  objective is not globally scaled by that weight. Measured depth values stay identical.
- Frequency selection respects VINS's existing every-other-image backend phase.
  Backend remains about 10Hz; no stale depth or interpolation between exposures.
- Same depth-v2 executable/image, input bag, calibration, tracking, IMU and solver.
  Sequential full-sequence screening at 1x playback, then confirmation at 0.5x.
- Order shuffled once with seed 42 before scoring; candidate selected by minimum ATE.
- {screen['matched']} shared GT timestamps, {screen['duration_s']:.1f}s for the screening table.
  SE3 alignment, fixed scale=1, all valid 1s/5s RPE pairs; every error checked with evo.
- Incomplete trajectories, gaps >151ms, solver-evaluation failures and zero depth
  factor activation are rejected; failures are retained in the evidence JSON.

## Screening results (1x playback)

Fresh baseline: ATE {base['ATE_RMSE_m']*100:.4f}cm; RPE@1s translation
{base['RPE_1s']['translation_RMSE_m']*100:.4f}cm, rotation {base['RPE_1s']['rotation_RMSE_deg']:.4f}deg.

| Weight | Depth Hz | ATE cm | ATE change | RPE trans 1s cm | RPE rot 1s deg | RPE trans 5s cm | RPE rot 5s deg |
|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

![Weight/frequency sweep](figures/vins_depth_sweep_v1.png)

Selected: **weight {selected['weight']:g}, frequency {selected['hz']:g}Hz**,
sigma multiplier {selected['sigma_multiplier']:g}. This selection uses V1_01 GT.

## Confirmation results (0.5x playback)

| Pair | Mode | ATE cm | RPE trans 1s cm | RPE rot 1s deg | RPE trans 5s cm | RPE rot 5s deg |
|---:|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(confirmation_rows)+f'''

Lower ATE confirmed in every pair: **{confirmed_ate_better}**.
Lower ATE and non-worse translation/rotation RPE@1s in every pair: **{confirmed_all_1s_better}**.
Non-worse results across all five ATE/RPE metrics in every pair: **{confirmed_all_metrics_better}**.
These flags describe measured comparisons, not statistical significance.

## Reproduction and limitations

[vins_depth_sweep_v1.json](artifacts/vins_depth_sweep_v1.json) contains all scores,
the predeclared plan, selection, rejected runs, fingerprints and confirmation.
Large logs/trajectories: `outputs/V1_01_easy/vins_depth_sweep_v1/`.
Depth view definitions: `outputs/V1_01_easy/depth_sweep_views_v1/configs.json`.
Use `pipeline/prepare_depth_sweep.py` and `pipeline/sweep_vins_depth.py`.

GT-guided selection can overfit this sequence. More repetitions and a held-out
sequence are required before a general improvement claim. Raw V1_01 reference
orientation retains the caveat documented in report 09. The model uses cached
depth, fixed 5%/2px base sigma proxies, a 0.2–4m range and the same quality gates.
No default estimator settings are automatically changed by this experiment.
'''
    (ROOT/'reports/10_depth_weight_frequency_sweep.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(selected=selected,confirmation=confirmations,
                         confirmed_ATE_better=confirmed_ate_better,confirmed_all_1s_better=confirmed_all_1s_better),indent=2))


if __name__=='__main__':main()
