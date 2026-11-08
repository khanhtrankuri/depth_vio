"""Stage 4 (IGEV++ disparity, + StereoSGBM baseline), Stage 5 (metric depth),
Stage 6 (local point cloud), and the single-frame depth evaluation against the
Leica laser scan (pointcloud0) projected with the Vicon ground-truth pose."""
import json, os, time
import numpy as np
import cv2
import torch
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from igev import IGEV, lr_check

FRAME = int(os.environ.get("FRAME", 700))
ZMAX = 10.0
pairs, _, _ = stereo_pairs()
t_ns, pL, pR = pairs[FRAME]
rect = stereo_rectify()
L, R = rectify_pair(rect, read_gray(pL), read_gray(pR))
fx, B = rect["fx"], rect["baseline"]
timing = {}

# ---------------- Stage 4 ----------------
d4 = outdir("stage4_disparity")
model = IGEV()
model(L, R); torch.cuda.synchronize()  # warm-up
t0 = time.perf_counter(); dL = model(L, R); torch.cuda.synchronize(); timing["igev_ms"] = (time.perf_counter() - t0) * 1e3
dR = model.right_disp(L, R)
lr = lr_check(dL, dR, 1.0)
border = cv2.remap(np.full_like(read_gray(pL), 255), rect["map0"][0], rect["map0"][1], cv2.INTER_NEAREST) > 0
valid_igev = lr & border & (dL > 0.5)

sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=5, P1=8 * 25, P2=32 * 25,
                             disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
t0 = time.perf_counter(); dS = sgbm.compute(L, R).astype(np.float32) / 16.0; timing["sgbm_ms"] = (time.perf_counter() - t0) * 1e3
valid_sgbm = (dS > 0.5) & border

np.save(f"{d4}/disp_igev.npy", dL); np.save(f"{d4}/disp_igev_right.npy", dR); np.save(f"{d4}/disp_sgbm.npy", dS)
vmax = np.percentile(dL[valid_igev], 99)
fig, ax = plt.subplots(2, 2, figsize=(13, 8.4))
ax[0, 0].imshow(L, cmap="gray"); ax[0, 0].set_title(f"rectified cam0, t={t_ns}")
im = ax[0, 1].imshow(dL, cmap="turbo", vmin=0, vmax=vmax); ax[0, 1].set_title("IGEV++ (sceneflow.pth) raw disparity")
plt.colorbar(im, ax=ax[0, 1], label="disparity [px]")
im = ax[1, 0].imshow(np.ma.masked_where(~valid_igev, dL), cmap="turbo", vmin=0, vmax=vmax); ax[1, 0].set_facecolor("k")
ax[1, 0].set_title(f"IGEV++ after L-R check (<1px): {valid_igev.mean()*100:.1f}% valid")
plt.colorbar(im, ax=ax[1, 0], label="disparity [px]")
im = ax[1, 1].imshow(np.ma.masked_where(~valid_sgbm, dS), cmap="turbo", vmin=0, vmax=vmax); ax[1, 1].set_facecolor("k")
ax[1, 1].set_title(f"BASELINE OpenCV StereoSGBM: {valid_sgbm.mean()*100:.1f}% valid")
plt.colorbar(im, ax=ax[1, 1], label="disparity [px]")
for a in ax.ravel(): a.set_xlabel("u [px]"); a.set_ylabel("v [px]")
plt.tight_layout(); plt.savefig(f"{d4}/disparity.png", dpi=110); plt.close()
plt.imsave(f"{d4}/disp_igev_color.png", dL, cmap="turbo", vmin=0, vmax=vmax)
inv = ~valid_igev
dstats = dict(checkpoint=os.path.relpath(model.ckpt, ROOT), iters=model.iters, frame=FRAME, t_ns=t_ns,
              igev_disp_min=float(dL[valid_igev].min()), igev_disp_max=float(dL[valid_igev].max()),
              igev_disp_median=float(np.median(dL[valid_igev])), igev_valid_frac=float(valid_igev.mean()),
              igev_lr_fail_frac=float((~lr & border).mean()), sgbm_valid_frac=float(valid_sgbm.mean()),
              invalid_left_band_frac=float(inv[:, :60].mean()), timing=timing)
cv2.imwrite(f"{d4}/invalid_mask.png", (inv * 255).astype(np.uint8))
print("Stage4", dstats)

# ---------------- Stage 5 ----------------
d5 = outdir("stage5_depth")


def to_depth(d, valid):
    z = np.full(d.shape, np.nan, np.float32)
    z[valid] = fx * B / d[valid]
    z[z > ZMAX] = np.nan
    return z


Z = to_depth(dL, valid_igev); ZS = to_depth(dS, valid_sgbm)
np.save(f"{d5}/depth_igev_m.npy", Z)

# Reference depth: Leica scan → rectified cam0 using Vicon/GT pose (geometric reference, not perfect)
pcd_ref = o3d.io.read_point_cloud(os.path.join(MAV, "pointcloud0", "data.ply"))
Pw = np.asarray(pcd_ref.points)
tg, Tg = load_gt()
if os.environ.get("GT_TUM"):  # diagnostic: alternative pose file (e.g. offset-corrected GT)
    from stage8_fusion import load_tum
    tg, Tg = load_tum(os.environ["GT_TUM"])
    d5 = outdir("stage5_depth_diag_" + os.environ.get("DIAG_TAG", "alt"))
T_WB = interp_pose(t_ns, tg, Tg)
Zgt = np.full(L.shape, np.nan, np.float32)
if T_WB is not None:
    T_CW = np.linalg.inv(T_W_cam0rect(T_WB, rect))
    Pc = Pw @ T_CW[:3, :3].T + T_CW[:3, 3]
    Pc = Pc[(Pc[:, 2] > 0.1) & (Pc[:, 2] < 20)]
    u = fx * Pc[:, 0] / Pc[:, 2] + rect["cx"]; v = rect["fy"] * Pc[:, 1] / Pc[:, 2] + rect["cy"]
    h, w = L.shape
    zb = np.full(h * w, np.inf, np.float32)
    for du in (0, 1):  # 2x2 splat z-buffer to close gaps in the scan
        for dv in (0, 1):
            ui = (u + du - 0.5).astype(int); vi = (v + dv - 0.5).astype(int)
            k = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            np.minimum.at(zb, vi[k] * w + ui[k], Pc[k, 2].astype(np.float32))
    zb[np.isinf(zb)] = np.nan
    Zgt = zb.reshape(h, w)
np.save(f"{d5}/depth_leica_ref_m.npy", Zgt)


def metrics(z, zr):
    k = np.isfinite(z) & np.isfinite(zr)
    e = z[k] - zr[k]
    ratio = np.maximum(z[k] / zr[k], zr[k] / z[k])
    return dict(n_px=int(k.sum()), MAE_m=float(np.abs(e).mean()), RMSE_m=float(np.sqrt((e ** 2).mean())),
                AbsRel=float((np.abs(e) / zr[k]).mean()), median_err_m=float(np.median(e)),
                median_abs_err_m=float(np.median(np.abs(e))), delta_1_25=float((ratio < 1.25).mean()))


m_igev = metrics(Z, Zgt); m_sgbm = metrics(ZS, Zgt)
zv = Z[np.isfinite(Z)]
fig, ax = plt.subplots(2, 2, figsize=(13, 8.4))
for a, im_, ttl in ((ax[0, 0], Z, "IGEV++ metric depth Z = fx·B/d"), (ax[0, 1], ZS, "BASELINE StereoSGBM depth"),
                    (ax[1, 0], Zgt, "Reference: Leica scan projected with GT pose")):
    h_ = a.imshow(im_, cmap="turbo_r", vmin=0.5, vmax=6); a.set_facecolor("k"); a.set_title(ttl)
    plt.colorbar(h_, ax=a, label="depth [m]")
h_ = ax[1, 1].imshow(np.abs(Z - Zgt), cmap="magma", vmin=0, vmax=0.5); ax[1, 1].set_facecolor("k")
ax[1, 1].set_title(f"|IGEV++ − ref|  MAE={m_igev['MAE_m']:.3f} m"); plt.colorbar(h_, ax=ax[1, 1], label="abs error [m]")
for a in ax.ravel(): a.set_xlabel("u [px]"); a.set_ylabel("v [px]")
plt.tight_layout(); plt.savefig(f"{d5}/depth.png", dpi=110); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ax[0].hist(zv, bins=100, color="C0"); ax[0].set_xlabel("depth [m]"); ax[0].set_ylabel("# pixels"); ax[0].set_title("IGEV++ valid depth distribution")
k = np.isfinite(Z) & np.isfinite(Zgt)
ax[1].hexbin(Zgt[k], Z[k], gridsize=80, bins="log", cmap="viridis", extent=(0, 7, 0, 7))
ax[1].plot([0, 7], [0, 7], "r--", lw=1); ax[1].set_xlabel("reference depth [m]"); ax[1].set_ylabel("IGEV++ depth [m]")
ax[1].set_title("per-pixel depth vs reference")
plt.tight_layout(); plt.savefig(f"{d5}/depth_stats.png", dpi=110); plt.close()
zs = np.linspace(0.5, 6, 12)
binned = [np.abs(Z - Zgt)[k & (Zgt >= a) & (Zgt < b)] for a, b in zip(zs[:-1], zs[1:])]
depth_info = dict(fx_px=fx, baseline_m=B, fxB=fx * B, z_max_clip=ZMAX,
                  depth_min=float(zv.min()), depth_max=float(zv.max()), depth_median=float(np.median(zv)),
                  valid_frac=float(np.isfinite(Z).mean()), ref_coverage=float(np.isfinite(Zgt).mean()),
                  metrics_igev_vs_leica=m_igev, metrics_sgbm_vs_leica=m_sgbm,
                  err_by_depth=[dict(z0=float(a), z1=float(b), n=int(len(e)), median_abs=float(np.median(e)) if len(e) else None)
                                for a, b, e in zip(zs[:-1], zs[1:], binned)])
print("Stage5", {k_: v for k_, v in depth_info.items() if k_ != "err_by_depth"})

# ---------------- Stage 6 ----------------
d6 = outdir("stage6_pointcloud")
t0 = time.perf_counter()
vv, uu = np.nonzero(np.isfinite(Z))
z = Z[vv, uu]
X = np.stack([(uu - rect["cx"]) * z / fx, (vv - rect["cy"]) * z / rect["fy"], z], 1)
g = L[vv, uu].astype(np.float64) / 255.0
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
pc.colors = o3d.utility.Vector3dVector(np.repeat(g[:, None], 3, 1))
timing["backproject_ms"] = (time.perf_counter() - t0) * 1e3
o3d.io.write_point_cloud(f"{d6}/local_cloud_cam0rect.ply", pc)
pcd_ds = pc.voxel_down_sample(0.02)
o3d.io.write_point_cloud(f"{d6}/local_cloud_ds2cm.ply", pcd_ds)
sel = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
np.save(f"{d6}/viewer_xyzg.npy", np.hstack([X[sel], g[sel, None]]).astype(np.float32))
fig = plt.figure(figsize=(13, 4.3))
for i, (el, az, ttl) in enumerate(((0, -90, "front (camera view)"), (90, -90, "top-down (X–Z)"), (20, -50, "oblique"))):
    a = fig.add_subplot(1, 3, i + 1, projection="3d")
    s = sel[:20000]
    a.scatter(X[s, 0], X[s, 2], -X[s, 1], c=g[s], cmap="gray", s=0.3)
    a.set_xlabel("X [m]"); a.set_ylabel("Z [m]"); a.set_zlabel("−Y [m]"); a.set_title(ttl)
    a.view_init(elev=el, azim=az)
plt.tight_layout(); plt.savefig(f"{d6}/pointcloud_views.png", dpi=110); plt.close()
pc_info = dict(n_points=len(X), n_points_ds2cm=len(pcd_ds.points), frame="rectified cam0",
               bbox_min=X.min(0).tolist(), bbox_max=X.max(0).tolist())
print("Stage6", pc_info)
json.dump(dict(stage4=dstats, stage5=depth_info, stage6=pc_info, timing=timing,
               gt_pose_T_W_B=None if T_WB is None else T_WB.tolist()),
          open(os.path.join(d5 if os.environ.get("GT_TUM") else OUT, "single_frame_summary.json"), "w"), indent=2)
