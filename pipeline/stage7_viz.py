"""Stage 7: visualise OpenVINS output (real run in Docker) — trajectory, camera frames, sparse landmarks.

Landmarks come from /ov_msckf/points_slam (persistent SLAM features) and /ov_msckf/points_msckf
(features used in MSCKF updates) recorded during the run, in OpenVINS' global frame. They are
transformed into the GT frame with the same SE(3) alignment evo computed for the trajectory.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path
from common import *

d7 = outdir("stage7_vio")
d9 = os.path.join(OUT, "stage9_eval")
al = json.load(open(os.path.join(d9, "vio_metrics.json")))
Ra, ta = np.array(al["align_R"]), np.array(al["align_t"]).ravel()

pts = {"/ov_msckf/points_slam": [], "/ov_msckf/points_msckf": []}
with AnyReader([Path(d7) / "openvins_points.bag"]) as rd:
    conns = [c for c in rd.connections if c.topic in pts]
    for c, ts, raw in rd.messages(connections=conns):
        m = rd.deserialize(raw, c.msgtype)
        n = m.width * m.height
        if n == 0:
            continue
        a = np.frombuffer(m.data, dtype=np.float32).reshape(n, m.point_step // 4)[:, :3]
        pts[c.topic].append(a)
slam = np.unique(np.round(np.vstack(pts["/ov_msckf/points_slam"]), 2), axis=0) if pts["/ov_msckf/points_slam"] else np.zeros((0, 3))
msckf = np.unique(np.round(np.vstack(pts["/ov_msckf/points_msckf"]), 2), axis=0) if pts["/ov_msckf/points_msckf"] else np.zeros((0, 3))
slam_w = slam @ Ra.T + ta; msckf_w = msckf @ Ra.T + ta
np.save(os.path.join(d7, "landmarks_slam_gtframe.npy"), slam_w.astype(np.float32))
np.save(os.path.join(d7, "landmarks_msckf_gtframe.npy"), msckf_w.astype(np.float32))

e = np.loadtxt(os.path.join(d9, "vio_aligned_tum.txt"))
from scipy.spatial.transform import Rotation as Rot
rect = stereo_rectify()
fig = plt.figure(figsize=(13, 6))
a = fig.add_subplot(1, 2, 1, projection="3d")
k = np.random.default_rng(0).choice(len(msckf_w), min(15000, len(msckf_w)), replace=False)
a.scatter(*msckf_w[k].T, s=0.2, c="0.6", label=f"MSCKF features ({len(msckf_w)})")
a.scatter(*slam_w.T, s=2, c="C3", label=f"SLAM landmarks ({len(slam_w)})")
a.plot(*e[:, 1:4].T, "C0-", lw=1.2, label="OpenVINS trajectory")
for i in range(0, len(e), 120):  # camera axes every ~6 s
    T_WB = np.eye(4); T_WB[:3, :3] = Rot.from_quat(e[i, 4:8]).as_matrix(); T_WB[:3, 3] = e[i, 1:4]
    T = T_WB @ rect["T_B0"]
    for j, col in enumerate("rgb"):
        a.plot(*np.stack([T[:3, 3], T[:3, 3] + 0.3 * T[:3, j]]).T, col, lw=1)
a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_zlim(-1, 3)
a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_zlabel("z [m]"); a.legend(fontsize=7, loc="upper left")
a.set_title("OpenVINS sparse map + camera frames (x=r, y=g, z=b)")
a = fig.add_subplot(1, 2, 2)
a.scatter(msckf_w[k, 0], msckf_w[k, 1], s=0.2, c="0.6"); a.scatter(slam_w[:, 0], slam_w[:, 1], s=2, c="C3")
a.plot(e[:, 1], e[:, 2], "C0-", lw=1); a.set_aspect("equal"); a.set_xlim(-4, 4); a.set_ylim(-4, 4)
a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_title("top view")
plt.tight_layout(); plt.savefig(os.path.join(d7, "sparse_map.png"), dpi=110); plt.close()
tm = np.genfromtxt(os.path.join(d7, "openvins_timing.txt"), delimiter=",", comments="#")
info = dict(n_slam=len(slam_w), n_msckf=len(msckf_w), n_poses=len(e),
            timing_ms_mean={n: float(tm[:, i + 1].mean() * 1e3) for i, n in enumerate(
                ["tracking", "propagation", "msckf_update", "slam_update", "slam_delayed", "retri_marg", "total"])})
json.dump(info, open(os.path.join(d7, "stage7_info.json"), "w"), indent=2)
print(info)
