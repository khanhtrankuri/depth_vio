#!/usr/bin/env python3
"""Where do two runs diverge? Per-second relative rotation/translation error vs GT for run A and run B."""
import sys
import numpy as np
from evo.core import sync
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from eval_traj import load_tum, load_euroc_gt  # noqa: E402


def rel_err(ref, est, step):
    out = []
    for i in range(0, est.num_poses - step, step):
        Ea = np.linalg.inv(est.poses_se3[i]) @ est.poses_se3[i + step]
        Ga = np.linalg.inv(ref.poses_se3[i]) @ ref.poses_se3[i + step]
        E = np.linalg.inv(Ga) @ Ea
        ang = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
        out.append((est.timestamps[i], ang, np.linalg.norm(E[:3, 3])))
    return np.array(out)


gt = load_euroc_gt(sys.argv[3])
res = []
for p in sys.argv[1:3]:
    r, e = sync.associate_trajectories(gt, load_tum(p), max_diff=0.01)
    res.append(rel_err(r, e, 10))
a, b = res
n = min(len(a), len(b))
t0 = a[0, 0]
print('t(s)  rotA  rotB  trA(cm) trB(cm)   (1 s windows; showing where B is much worse)')
d = b[:n, 1] - a[:n, 1]
for i in np.argsort(-d)[:15]:
    print(f'{a[i,0]-t0:6.1f} {a[i,1]:5.2f} {b[i,1]:5.2f} {a[i,2]*100:6.2f} {b[i,2]*100:6.2f}')
print('mean rot A/B', a[:n, 1].mean(), b[:n, 1].mean())
