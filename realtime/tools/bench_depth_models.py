#!/usr/bin/env python3
"""Rank live stereo models for VINS by (1) GPU latency and (2) depth error at VINS-like corner features.

Reference depth: the Leica scan of Vicon Room 1 projected into raw cam0 with the batch GT pose and a z-buffer
(geometric reference, registration not perfect, so relative ranking is what matters). Only pixels with depth
0.2-4 m and a smooth reference are scored. GT is used here only for scoring.

    python bench_depth_models.py --data <repo>/data --out bench.json --models rt:sceneflow:6 pp:eth3d:8 ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'depth_node'))
from stereo_geometry import StereoRig, DepthConverter  # noqa: E402
import stereo_models  # noqa: E402


def read_ply_xyz(path):
    with open(path, 'rb') as f:
        n, props, fmt = 0, [], 'ascii'
        while True:
            line = f.readline().decode().strip()
            if line.startswith('format'):
                fmt = line.split()[1]
            elif line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                props.append(line.split()[1:])
            elif line == 'end_header':
                break
        types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1', 'int': 'i4'}
        if fmt == 'ascii':
            data = np.loadtxt(f, max_rows=n)[:, :3]
            return data
        dt = np.dtype([(p[-1], ('<' if 'little' in fmt else '>') + types[p[0]]) for p in props])
        a = np.frombuffer(f.read(n * dt.itemsize), dt, n)
        return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)


def quat_R(w, x, y, z):
    q = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z]); w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def leica_depth_raw(P, T_WB, rig):
    T_WC = T_WB @ rig.T_B0
    R, t = T_WC[:3, :3].T, -T_WC[:3, :3].T @ T_WC[:3, 3]
    Pc = P @ R.T + t
    Pc = Pc[(Pc[:, 2] > 0.1) & (Pc[:, 2] < 8)]
    uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), rig.K0, rig.D0)
    uv = uv.reshape(-1, 2)
    w, h = rig.size
    zb = np.full(h * w, np.inf, np.float32)
    for du in (0, 1):
        for dv in (0, 1):
            ui = np.floor(uv[:, 0] + du - 0.5).astype(int); vi = np.floor(uv[:, 1] + dv - 0.5).astype(int)
            k = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            np.minimum.at(zb, vi[k] * w + ui[k], Pc[k, 2].astype(np.float32))
    zb[np.isinf(zb)] = np.nan
    return zb.reshape(h, w)


def corners(img):
    # Same detector family/parameters as the VINS-Fusion feature tracker (max_cnt 150, min_dist 30).
    p = cv2.goodFeaturesToTrack(img, 150, 0.01, 30)
    return np.zeros((0, 2)) if p is None else p.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seqs', nargs='+', default=['V1_02_medium', 'V1_03_difficult'])
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--models', nargs='+', required=True)
    args = ap.parse_args()
    data = Path(args.data)
    samples = []
    for seq in args.seqs:
        mav = data / seq / 'mav0'
        rig = StereoRig.from_euroc(mav / 'cam0/sensor.yaml', mav / 'cam1/sensor.yaml')
        gt = np.loadtxt(mav / 'state_groundtruth_estimate0/data.csv', delimiter=',', comments='#')
        P = read_ply_xyz(mav / 'pointcloud0/data.ply')
        rows = [l.split(',') for l in open(mav / 'cam0/data.csv') if not l.startswith('#')]
        stamps = [int(r[0]) for r in rows]
        sel = np.linspace(len(stamps) * 0.1, len(stamps) * 0.95, args.frames).astype(int)
        for i in sel:
            t = stamps[i]
            j = np.argmin(np.abs(gt[:, 0] - t))
            if abs(gt[j, 0] - t) > 2.5e6:
                continue
            T = np.eye(4); T[:3, :3] = quat_R(*gt[j, 4:8]); T[:3, 3] = gt[j, 1:4]
            L = cv2.imread(str(mav / 'cam0/data' / f'{t}.png'), 0)
            Rr = cv2.imread(str(mav / 'cam1/data' / f'{t}.png'), 0)
            if L is None or Rr is None:
                continue
            samples.append(dict(seq=seq, t=t, L=L, R=Rr, ref=leica_depth_raw(P, T, rig), rig=rig, uv=corners(L)))
        print(seq, 'frames', sum(s['seq'] == seq for s in samples), flush=True)
    conv = {}
    results = {}
    for spec in args.models:
        model = stereo_models.build(spec)
        errs_feat, errs_dense, cov_feat, lat = [], [], [], []
        for s in samples:
            c = conv.setdefault(s['seq'], DepthConverter(s['rig'], zmin=0.2, zmax=4.0))
            torch.cuda.synchronize(); t0 = time.perf_counter()
            l, r = c.rectify_pair(s['L'], s['R'])
            disp = model(l, r)
            out = c.raw_inverse_depth(disp)
            torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
            inv, ref = out[..., 0], s['ref']
            z = np.where(inv > 0, 1 / np.maximum(inv, 1e-9), np.nan)
            ok = np.isfinite(z) & np.isfinite(ref) & (ref > 0.2) & (ref < 4)
            errs_dense.append(np.abs(z[ok] - ref[ok]) / ref[ok])
            uv = np.round(s['uv']).astype(int)
            vv, uu = uv[:, 1], uv[:, 0]
            zr, zp = ref[vv, uu], z[vv, uu]
            k = np.isfinite(zr) & (zr > 0.2) & (zr < 4)
            have = k & np.isfinite(zp)
            cov_feat.append((have.sum(), k.sum()))
            errs_feat.append(np.abs(zp[have] - zr[have]) / zr[have])
        # steady-state latency on one frame
        l, r = conv[samples[0]['seq']].rectify_pair(samples[0]['L'], samples[0]['R'])
        for _ in range(5):
            model(l, r)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(30):
            d = model(l, r)
        torch.cuda.synchronize(); infer = (time.perf_counter() - t0) / 30
        ef, ed = np.concatenate(errs_feat), np.concatenate(errs_dense)
        a, b = np.sum(cov_feat, 0)
        results[spec] = dict(infer_ms=infer * 1e3, pipeline_ms_median=float(np.median(lat) * 1e3),
                             feat_absrel_median=float(np.median(ef)), feat_absrel_mean=float(np.mean(ef)),
                             feat_within_5pct=float(np.mean(ef < 0.05)), feat_outlier_15pct=float(np.mean(ef > 0.15)),
                             feat_coverage=float(a / max(b, 1)), dense_absrel_median=float(np.median(ed)),
                             dense_within_5pct=float(np.mean(ed < 0.05)), gpu_mem_mb=torch.cuda.max_memory_allocated() / 2**20)
        print(spec, json.dumps({k: round(v, 4) for k, v in results[spec].items()}), flush=True)
        del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    Path(args.out).write_text(json.dumps(dict(frames=len(samples), seqs=args.seqs, results=results), indent=2))


if __name__ == '__main__':
    main()
