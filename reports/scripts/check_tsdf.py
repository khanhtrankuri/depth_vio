"""Numerical controls on the unchanged production TSDF integrator, isolated via AST."""
import ast
import json
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
tree = ast.parse((root/'pipeline/stage8_fusion.py').read_text(encoding='utf-8'))
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TorchTSDF')
env = dict(np=np, fx=436.244295647102, fy=436.244295647102, cx=364.44123458862305, cy=256.95167541503906)
exec(compile(ast.Module(body=[node], type_ignores=[]), 'production_TorchTSDF', 'exec'), env)
volume = env['TorchTSDF']([-.15, -.15, 1.8], [.15, .15, 2.22], .03, .12)
depth = np.full((480, 752), 2., dtype=np.float32)
gray = np.full((480, 752), 128, dtype=np.uint8)
volume.integrate(depth, gray, np.eye(4))
p = volume.P.cpu().numpy(); tsdf = volume.tsdf.cpu().numpy().ravel(); w = volume.w.cpu().numpy().ravel()
valid = 2.-p[:, 2] > -.12
expected = np.minimum((2.-p[:, 2])/.12, 1.)
error = float(np.max(abs(tsdf[valid]-expected[valid])))
assert np.array_equal(w, valid.astype(float)) and error < 1e-5
volume.integrate(depth, gray, np.eye(4))
assert np.array_equal(volume.w.cpu().numpy().ravel(), valid.astype(float)*2)
before = volume.tsdf.clone(); before_w = volume.w.clone()
volume.integrate(np.zeros_like(depth), gray, np.eye(4))
assert torch.equal(before, volume.tsdf) and torch.equal(before_w, volume.w)
result = dict(status='VERIFIED: synthetic integration controls only', cuda_device=torch.cuda.get_device_name(),
              plane_depth_m=2., n_voxels=len(w), max_tsdf_error=error,
              unit_weight_increment='passed', behind_truncation_excluded='passed',
              invalid_zero_depth_does_not_update='passed',
              extraction_and_real_scene_accuracy='NOT EVALUATED by this control')
(root/'reports/artifacts/tsdf_controls.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
