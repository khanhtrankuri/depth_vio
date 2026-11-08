"""Same-input synchronized GPU timing of original and new fusion backends."""
import ast
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'pipeline'))
from fusion_geometry import Camera
from fusion_tsdf import FrustumTSDF, VoxelSurfels

seq='V1_01_easy'; source=ROOT/'outputs'/seq/'stage8_fusion_v2/prepared'
manifest=json.loads((source/'manifest.json').read_text()); cfg=manifest['config']; cam=Camera(**cfg['camera'])
poses=np.load(source/'poses_native.npy')
chosen=np.linspace(0,len(poses)-1,20,dtype=int)
data=[]
for i in chosen:
    t=manifest['timestamps_ns'][i]
    with np.load(source/(str(t)+'.npz')) as a:
        data.append((a['depth'],a['gray'],(a['depth']>0).astype(np.float32),
                     a['spatial'].astype(np.float32)*a['temporal'].astype(np.float32),poses[i]))
tree=ast.parse((ROOT/'pipeline/stage8_fusion.py').read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TorchTSDF')
namespace=dict(np=np,fx=cam.fx,fy=cam.fy,cx=cam.cx,cy=cam.cy)
exec(compile(ast.Module(body=[node],type_ignores=[]),'legacy_TorchTSDF', 'exec'),namespace)
results=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,sequence=seq,
             warmup=5,repeats=30,sample_frame_indices=chosen.tolist(),
             scope='integration including H2D copies; excludes I/O, stereo, confidence preparation, extraction',
             native_bounds=[manifest['volume_lo'],manifest['volume_hi']],backends={})
for name,backend in [('legacy',namespace['TorchTSDF']),('frustum_uniform',FrustumTSDF),('surfel_uniform',VoxelSurfels)]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if name=='legacy':
        volume=backend(manifest['volume_lo'],manifest['volume_hi'],cfg['voxel'],cfg['trunc'])
        call=lambda d,g,w,robust,T: volume.integrate(d,g,np.linalg.inv(T))
    else:
        volume=backend(manifest['volume_lo'],manifest['volume_hi'],cam,cfg['voxel'],cfg['trunc'])
        call=lambda d,g,w,robust,T: volume.integrate(d,w,g,T)
    for a in data[:5]: call(*a)
    torch.cuda.synchronize(); timing=[]
    for i in range(30):
        a=data[i%len(data)]
        torch.cuda.synchronize(); tick=time.perf_counter(); call(*a); torch.cuda.synchronize()
        timing.append((time.perf_counter()-tick)*1000)
    results['backends'][name]=dict(mean_ms=float(np.mean(timing)),p50_ms=float(np.median(timing)),
                                  p90_ms=float(np.quantile(timing,.9)),dims=volume.dims.tolist(),
                                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    print(name,results['backends'][name],flush=True)
    del call,volume; gc.collect(); torch.cuda.empty_cache()
(ROOT/'reports/artifacts/fusion_v2/backend_timing.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
