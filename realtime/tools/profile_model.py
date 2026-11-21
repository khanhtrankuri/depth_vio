#!/usr/bin/env python3
"""Latency profile of one stereo model at EuRoC resolution (752x480 -> padded 768x480)."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'depth_node'))
import stereo_models  # noqa: E402

torch.backends.cudnn.benchmark = True
spec = sys.argv[1] if len(sys.argv) > 1 else 'rt:sceneflow:6'
m = stereo_models.build(spec)
l = torch.rand(1, 1, 480, 752, device='cuda') * 255
r = torch.rand(1, 1, 480, 752, device='cuda') * 255
for _ in range(10):
    m(l, r)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    m(l, r)
torch.cuda.synchronize()
print(spec, 'ms/frame', (time.perf_counter() - t0) / 50 * 1e3)
if '--prof' in sys.argv:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            m(l, r)
        torch.cuda.synchronize()
    print(p.key_averages().table(sort_by='cuda_time_total', row_limit=15))
