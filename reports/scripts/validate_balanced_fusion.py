"""Validate the fast preset against the frozen development/held-out outputs."""
import json
from pathlib import Path
import numpy as np
from audit_cached import ROOT, SEQS, ply, sha, J

dest=ROOT/'reports/artifacts/fusion_v2'
selection=J(dest/'selection.json')
checks={}
for name,digest in selection['code_sha256'].items():
    assert sha(dest/'frozen_sources'/name)==digest, 'Frozen validation source not preserved: '+name
for seq in SEQS:
    old=ROOT/'outputs'/seq/'stage8_fusion_v2'
    new=ROOT/'outputs'/seq/'stage8_fusion_balanced'
    om,nm=J(old/'prepared/manifest.json'),J(new/'prepared/manifest.json')
    assert om['timestamps_ns']==nm['timestamps_ns']
    np.testing.assert_array_equal(np.load(old/'prepared/poses_native.npy'),np.load(new/'prepared/poses_native.npy'))
    for i in np.linspace(0,len(om['timestamps_ns'])-1,10,dtype=int):
        name=str(om['timestamps_ns'][i])+'.npz'
        with np.load(old/'prepared'/name) as a,np.load(new/'prepared'/name) as b:
            np.testing.assert_array_equal(a['depth'],b['depth'])
            np.testing.assert_array_equal(a['gray'],b['gray'])
    file=f"surface_support{selection['min_surface_support']}_cc{selection['min_component_points']}_cloud.ply"
    a=ply(old/selection['variant']/file); b=ply(new/selection['variant']/file)
    assert a.shape==b.shape
    diff=float(np.max(abs(a-b)))
    assert diff<1e-4, 'Performance optimization changed the surface geometry'
    checks[seq]=dict(frames=len(nm['timestamps_ns']),points=len(b),max_surface_coordinate_difference_m=diff,
                     pose_arrays_exactly_equal=True,ten_depth_and_color_frames_exactly_equal=True,
                     confidence_mode=nm['config']['confidence_mode'],prepare_ms_mean=nm['prepare_ms_mean'],
                     prepare_ms_p90=nm['prepare_ms_p90'],prepare_wall_s=nm['wall_s'],
                     integration=J(new/selection['variant']/'integration.json'))
# Historical sources/checkpoints/results must remain unchanged. The comparison
# template gets only a navigation link and is intentionally excluded here.
baseline=J(ROOT/'reports/artifacts/input_manifest.json')
for path,record in baseline.items():
    if path.endswith('compare_template.html'): continue
    digest=record['sha256'] if isinstance(record,dict) else record
    assert sha(ROOT/path)==digest, 'Historical baseline changed: '+path
summary=dict(status='passed',frozen_config_unchanged=True,
             note='Only unused confidence calculation and repeated color remap setup were skipped. Point-coordinate tolerance allows CUDA scatter floating-point reduction order.',
             sequences=checks,historical_source_checkpoint_output_hashes_unchanged=True)
(dest/'fast_path_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({s:{k:a[k] for k in ['frames','points','max_surface_coordinate_difference_m','prepare_ms_mean']} for s,a in checks.items()},indent=2))
