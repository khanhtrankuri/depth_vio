"""Stage 8: IGEV++ depth over the sequence + pose → world-frame fusion.

Usage:  python stage8_fusion.py infer               # IGEV++ on every STEP-th stereo pair (cached)
        python stage8_fusion.py fuse <pose_tum> <tag> # fuse with a TUM pose file of the IMU/body
Pose file convention: TUM 'ts tx ty tz qx qy qz qw', T_W_B (body = IMU = EuRoC 'B').
"""
import json, os, sys, time, glob
import numpy as np
import cv2
import open3d as o3d
from common import *

STEP = int(os.environ.get("STEP", 5))
ZMAX_FUSE = float(os.environ.get("ZMAX_FUSE", 4.0))
VOXEL = float(os.environ.get("VOXEL", 0.03))
d8 = outdir("stage8_fusion")
ddir = os.path.join(d8, "disp"); os.makedirs(ddir, exist_ok=True)
rect = stereo_rectify()
fx, fy, cx, cy, B = rect["fx"], rect["fy"], rect["cx"], rect["cy"], rect["baseline"]
pairs, _, _ = stereo_pairs()


def infer():
    import torch
    from igev import IGEV, lr_check
    model = IGEV()
    times = []
    sel = pairs[::STEP]
    for i, (t, pl, pr) in enumerate(sel):
        f = os.path.join(ddir, f"{t}.npz")
        if os.path.exists(f):
            continue
        L, R = rectify_pair(rect, read_gray(pl), read_gray(pr))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        dL = model(L, R); dR = model.right_disp(L, R)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1e3)
        valid = lr_check(dL, dR, 1.0) & (dL > 0.5)
        np.savez_compressed(f, disp=dL.astype(np.float16), valid=valid)
        if i % 50 == 0:
            print(f"{i}/{len(sel)}  {np.mean(times):.0f} ms/pair (L+R passes)", flush=True)
    if times:
        json.dump(dict(n=len(times), igev_LR_ms_mean=float(np.mean(times)), igev_LR_ms_p90=float(np.percentile(times, 90))),
                  open(os.path.join(d8, "infer_timing.json"), "w"), indent=2)


def load_tum(path):
    d = np.loadtxt(path, comments="#")
    t = (d[:, 0] * 1e9).round().astype(np.int64) if d[0, 0] < 1e12 else d[:, 0].astype(np.int64)
    T = np.tile(np.eye(4), (len(d), 1, 1))
    for i, r in enumerate(d):
        T[i, :3, :3] = quat_to_R(r[7], r[4], r[5], r[6]); T[i, :3, 3] = r[1:4]
    return t, T


def depth_of(t):
    z = np.load(os.path.join(ddir, f"{t}.npz"))
    d = z["disp"].astype(np.float32); v = z["valid"]
    Z = np.zeros_like(d); Z[v] = fx * B / d[v]
    Z[(Z > ZMAX_FUSE)] = 0
    return Z


class TorchTSDF:
    """Dense TSDF volume on the GPU (Curless & Levoy weighted running average).

    Replaces Open3D's ScalableTSDFVolume, which returns empty volumes in this
    environment (Open3D 0.20 + NumPy 2.4 on Windows; verified on a synthetic plane).
    """

    def __init__(self, lo, hi, voxel, trunc):
        import torch
        self.torch = torch
        self.lo = np.asarray(lo, np.float32); self.voxel = voxel; self.trunc = trunc
        self.dims = np.ceil((np.asarray(hi) - self.lo) / voxel).astype(int)
        dev = "cuda"
        self.tsdf = torch.ones(tuple(self.dims), device=dev)
        self.w = torch.zeros(tuple(self.dims), device=dev)
        self.col = torch.zeros(tuple(self.dims), device=dev)
        g = [torch.arange(n, device=dev, dtype=torch.float32) for n in self.dims]
        X, Y, Zg = torch.meshgrid(*g, indexing="ij")
        self.P = torch.stack([X, Y, Zg], -1).reshape(-1, 3) * voxel + torch.tensor(self.lo, device=dev) + voxel / 2

    def integrate(self, depth, gray, T_CW):
        torch = self.torch
        dev = self.P.device
        Rm = torch.tensor(T_CW[:3, :3], device=dev, dtype=torch.float32); tv = torch.tensor(T_CW[:3, 3], device=dev, dtype=torch.float32)
        Pc = self.P @ Rm.T + tv
        z = Pc[:, 2]
        u = torch.round(fx * Pc[:, 0] / z.clamp(min=1e-3) + cx).long(); v = torch.round(fy * Pc[:, 1] / z.clamp(min=1e-3) + cy).long()
        ok = (z > 0.2) & (u >= 0) & (u < 752) & (v >= 0) & (v < 480)
        D = torch.from_numpy(depth).to(dev); Gimg = torch.from_numpy(gray.astype(np.float32) / 255).to(dev)
        d = torch.zeros_like(z); d[ok] = D[v[ok], u[ok]]
        sdf = d - z
        upd = ok & (d > 0) & (sdf > -self.trunc)
        s = torch.clamp(sdf / self.trunc, max=1.0)
        T = self.tsdf.view(-1); W = self.w.view(-1); C = self.col.view(-1)
        c = torch.zeros_like(z); c[ok] = Gimg[v[ok], u[ok]]
        T[upd] = (T[upd] * W[upd] + s[upd]) / (W[upd] + 1)
        C[upd] = (C[upd] * W[upd] + c[upd]) / (W[upd] + 1)
        W[upd] += 1

    def extract(self, min_w=2):
        from skimage import measure
        vol = self.tsdf.cpu().numpy(); w = self.w.cpu().numpy(); col = self.col.cpu().numpy()
        mask = w >= min_w
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.0, mask=mask)
        # drop vertices interpolated against an unobserved voxel (spurious surface at the back of the
        # truncation band, where observed negative TSDF meets the +1 initial value)
        lo_i = np.clip(np.floor(verts).astype(int), 0, self.dims - 1); hi_i = np.clip(np.ceil(verts).astype(int), 0, self.dims - 1)
        good = (w[lo_i[:, 0], lo_i[:, 1], lo_i[:, 2]] >= min_w) & (w[hi_i[:, 0], hi_i[:, 1], hi_i[:, 2]] >= min_w)
        keep_f = good[faces].all(1)
        remap = -np.ones(len(verts), int); remap[good] = np.arange(good.sum())
        verts, faces = verts[good], remap[faces[keep_f]]
        idx = np.clip(np.round(verts).astype(int), 0, self.dims - 1)
        g = col[idx[:, 0], idx[:, 1], idx[:, 2]]
        verts = verts * self.voxel + self.lo + self.voxel / 2
        return verts, faces, g


def fuse(pose_file, tag):
    tp, Tp = load_tum(pose_file)
    frames = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(ddir, "*.npz")))
    img_of = {t: pl for t, pl, _ in pairs}
    tsdf = TorchTSDF(lo=(-4.8, -4.8, -0.6), hi=(4.8, 4.8, 3.6), voxel=VOXEL, trunc=4 * VOXEL)
    naive, cams, cons, t_int = [], [], [], []
    prev = None
    for t in frames:
        T_WB = interp_pose(t, tp, Tp, max_gap_ns=30_000_000)
        if T_WB is None:
            continue
        T_WC = T_W_cam0rect(T_WB, rect)
        Z = depth_of(t)
        L = cv2.remap(read_gray(img_of[t]), rect["map0"][0], rect["map0"][1], cv2.INTER_LINEAR)
        t0 = time.perf_counter()
        tsdf.integrate(Z.astype(np.float32), L, np.linalg.inv(T_WC))
        tsdf.torch.cuda.synchronize()
        t_int.append((time.perf_counter() - t0) * 1e3)
        vv, uu = np.nonzero(Z > 0)
        s = np.random.default_rng(t % 2**32).choice(len(vv), min(3000, len(vv)), replace=False)
        z = Z[vv[s], uu[s]]
        Pc = np.stack([(uu[s] - cx) * z / fx, (vv[s] - cy) * z / fy, z], 1)
        naive.append(np.hstack([Pc @ T_WC[:3, :3].T + T_WC[:3, 3], L[vv[s], uu[s], None] / 255.0]))
        cams.append(dict(t=int(t), T_WC=T_WC.tolist()))
        # depth consistency: warp previous keyframe depth into the current view
        if prev is not None:
            _, Zp, Tp_WC = prev
            vv2, uu2 = np.nonzero(Zp > 0)
            z2 = Zp[vv2, uu2]
            P = np.stack([(uu2 - cx) * z2 / fx, (vv2 - cy) * z2 / fy, z2], 1)
            T_rel = np.linalg.inv(T_WC) @ Tp_WC
            Q = P @ T_rel[:3, :3].T + T_rel[:3, 3]
            ok = Q[:, 2] > 0.2
            u = np.round(fx * Q[ok, 0] / Q[ok, 2] + cx).astype(int); v = np.round(fy * Q[ok, 1] / Q[ok, 2] + cy).astype(int)
            inb = (u >= 0) & (u < 752) & (v >= 0) & (v < 480)
            zc = Z[v[inb], u[inb]]; zw = Q[ok, 2][inb]
            m = zc > 0
            if m.sum() > 500:
                rel = np.abs(zc[m] - zw[m]) / zc[m]
                cons.append(dict(t=int(t), n=int(m.sum()), median_abs_m=float(np.median(np.abs(zc[m] - zw[m]))),
                                 median_rel=float(np.median(rel)), frac_rel_lt_5pct=float((rel < 0.05).mean())))
        prev = (t, Z, T_WC)
    verts, faces, g = tsdf.extract()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces[:, ::-1].copy()))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(g[:, None], 3, 1))
    mesh.compute_vertex_normals()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
    cloud.colors = o3d.utility.Vector3dVector(np.repeat(g[:, None], 3, 1))
    o3d.io.write_triangle_mesh(os.path.join(d8, f"tsdf_mesh_{tag}.ply"), mesh)
    o3d.io.write_point_cloud(os.path.join(d8, f"tsdf_cloud_{tag}.ply"), cloud)
    naive = np.vstack(naive)
    pn = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(naive[:, :3]))
    pn.colors = o3d.utility.Vector3dVector(np.repeat(naive[:, 3:4], 3, 1))
    o3d.io.write_point_cloud(os.path.join(d8, f"naive_accum_{tag}.ply"), pn)
    json.dump(cams, open(os.path.join(d8, f"cams_{tag}.json"), "w"))
    summ = dict(tag=tag, pose_file=pose_file, n_frames=len(cams), voxel_m=VOXEL, zmax_m=ZMAX_FUSE,
                tsdf_points=len(cloud.points), mesh_vertices=len(mesh.vertices), naive_points=len(naive),
                integrate_ms_mean=float(np.mean(t_int)), consistency=cons)
    json.dump(summ, open(os.path.join(d8, f"fusion_{tag}.json"), "w"), indent=1)
    print({k: v for k, v in summ.items() if k != "consistency"},
          "median consistency rel", np.median([c["median_rel"] for c in cons]))


if __name__ == "__main__":
    if sys.argv[1] == "infer":
        infer()
    else:
        fuse(sys.argv[2], sys.argv[3])
