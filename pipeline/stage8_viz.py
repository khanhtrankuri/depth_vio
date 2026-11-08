"""Stage 8/9 figures: fused map top/side views with trajectory, depth-consistency over time,
naive accumulation vs TSDF, and reconstruction error vs Leica scan."""
import json, os
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

d8 = os.path.join(OUT, "stage8_fusion"); d9 = os.path.join(OUT, "stage9_eval")
ref = o3d.io.read_point_cloud(os.path.join(MAV, "pointcloud0", "data.ply")).voxel_down_sample(0.03)
R = np.asarray(ref.points)


def load(kind, tag):
    pc = o3d.io.read_point_cloud(os.path.join(d8, f"{kind}_{tag}.ply"))
    return np.asarray(pc.points), np.asarray(pc.colors)[:, 0] if pc.has_colors() else None


cams = {tag: np.array([np.array(c["T_WC"]) for c in json.load(open(os.path.join(d8, f"cams_{tag}.json")))]) for tag in ("openvins", "gt", "gtrotcorr")}
P, G = load("tsdf_cloud", "openvins")
N, NG = load("naive_accum", "openvins")

# --- fused map views ---
fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
for a, (P_, G_, ttl) in zip(ax[:2], ((N, NG, "naive accumulation (no fusion)"), (P, G, "TSDF fusion (3 cm voxels)"))):
    k = (P_[:, 2] < 1.6)  # cut ceiling for a readable top view
    s = np.random.default_rng(0).choice(np.where(k)[0], min(250000, k.sum()), replace=False)
    a.scatter(P_[s, 0], P_[s, 1], c=G_[s], cmap="gray", s=0.05, vmin=0, vmax=1)
    a.plot(cams["openvins"][:, 0, 3], cams["openvins"][:, 1, 3], "C1-", lw=1, label="OpenVINS camera path")
    a.set_aspect("equal"); a.set_xlim(-4.2, 4.2); a.set_ylim(-4.2, 4.2); a.set_title(ttl + ", z<1.6 m"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    a.legend(fontsize=8, loc="lower left")
a = ax[2]
k = np.abs(P[:, 1] - 0.0) < 0.25  # vertical slice y≈0
a.scatter(P[k, 0], P[k, 2], s=0.3, c="C0", label="TSDF (OpenVINS poses)")
kr = np.abs(R[:, 1]) < 0.25
a.scatter(R[kr, 0], R[kr, 2], s=0.3, c="C3", alpha=.4, label="Leica scan (reference)")
a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("z [m]"); a.set_title("vertical slice |y|<0.25 m"); a.legend(fontsize=8, markerscale=10)
plt.tight_layout(); plt.savefig(os.path.join(d8, "fused_views.png"), dpi=110); plt.close()

# --- consistency over time ---
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
for tag, c in (("openvins", "C1"), ("gt", "k"), ("gtrotcorr", "C2")):
    cons = json.load(open(os.path.join(d8, f"fusion_{tag}.json")))["consistency"]
    t = (np.array([x["t"] for x in cons]) - cons[0]["t"]) / 1e9
    ax[0].plot(t, [x["median_rel"] * 100 for x in cons], c, lw=.7, label=f"{tag} poses")
    ax[1].hist([x["median_abs_m"] * 100 for x in cons], bins=60, range=(0, 15), histtype="step", color=c, label=f"{tag} poses")
ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("median |Δz|/z [%]"); ax[0].set_title("depth consistency: keyframe k−1 warped into k"); ax[0].legend(fontsize=8)
ax[0].set_ylim(0, 10)
ax[1].set_xlabel("median |Δz| per keyframe pair [cm]"); ax[1].set_ylabel("# pairs"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(d8, "consistency.png"), dpi=110); plt.close()

# --- reconstruction error coloured ---
rec = json.load(open(os.path.join(d9, "recon_metrics.json")))
pc = o3d.io.read_point_cloud(os.path.join(d8, "tsdf_cloud_openvins.ply")).voxel_down_sample(0.02)
Q = np.asarray(pc.points); acc = np.load(os.path.join(d9, "acc_tsdf_cloud_openvins.npy"))
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
k = Q[:, 2] < 1.6
sc = ax[0].scatter(Q[k, 0], Q[k, 1], c=acc[k] * 100, cmap="inferno_r", s=0.1, vmin=0, vmax=15)
ax[0].set_aspect("equal"); ax[0].set_xlim(-4.2, 4.2); ax[0].set_ylim(-4.2, 4.2)
ax[0].set_title("TSDF (OpenVINS) → nearest Leica point distance, z<1.6 m"); ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
plt.colorbar(sc, ax=ax[0], label="distance [cm]")
for key, lab, c in (("naive_accum_openvins", "naive, OpenVINS", "C1"), ("tsdf_cloud_openvins", "TSDF, OpenVINS", "C0"),
                    ("naive_accum_gt", "naive, GT poses", "0.6"), ("tsdf_cloud_gt", "TSDF, GT poses", "k"), ("tsdf_cloud_gtrotcorr", "TSDF, GT + rot. offset fix (diag.)", "C2")):
    a = np.load(os.path.join(d9, f"acc_{key}.npy"))
    xs = np.sort(a) * 100
    ax[1].plot(xs, np.linspace(0, 1, len(xs)), color=c, label=f"{lab} (median {np.median(a)*100:.1f} cm)")
ax[1].set_xlim(0, 30); ax[1].set_xlabel("distance to Leica scan [cm]"); ax[1].set_ylabel("cumulative fraction of points")
ax[1].set_title("accuracy CDF"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(d9, "recon_error.png"), dpi=110); plt.close()

# --- viewer payloads ---
s = np.random.default_rng(1).choice(len(P), min(140000, len(P)), replace=False)
np.save(os.path.join(d8, "viewer_global.npy"), np.hstack([P[s], G[s, None]]).astype(np.float32))
print("ok")
