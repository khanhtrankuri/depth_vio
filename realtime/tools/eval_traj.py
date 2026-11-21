#!/usr/bin/env python3
"""ATE / RPE of a VINS trajectory (TUM, T_WI) against EuRoC GT, computed with evo.

Association: nearest GT within 10 ms. Alignment: one SE3 (Umeyama, scale fixed to 1) over all matched poses.
RPE: translation (m) and rotation (deg) over 1 s and 5 s windows (all pairs). GT is used only for scoring.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D


def load_tum(path):
    a = np.loadtxt(path, comments='#')
    a = a[np.argsort(a[:, 0])]
    _, idx = np.unique(a[:, 0], return_index=True)
    if len(idx) != len(a) or np.any(np.diff(np.loadtxt(path, comments='#')[:, 0]) <= 0):
        # two estimators publishing on the same graph (or a restart): the run is invalid
        raise SystemExit(f'non-monotonic/duplicate odometry stamps in {path}: run invalid')
    a = a[idx]
    return PoseTrajectory3D(a[:, 1:4], a[:, [7, 4, 5, 6]], a[:, 0])  # evo quats are wxyz


def load_euroc_gt(path):
    a = np.loadtxt(path, delimiter=',', comments='#')
    return PoseTrajectory3D(a[:, 1:4], a[:, 4:8], a[:, 0] * 1e-9)


def rpe(ref, est, delta_s, kind):
    rel = metrics.PoseRelation.translation_part if kind == 't' else metrics.PoseRelation.rotation_angle_deg
    dt = float(np.median(np.diff(est.timestamps)))  # evo only supports frame/distance deltas
    m = metrics.RPE(rel, delta=max(1, int(round(delta_s / dt))), delta_unit=metrics.Unit.frames, all_pairs=True)
    m.process_data((ref, est))
    return float(m.get_statistic(metrics.StatisticsType.rmse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('est'); ap.add_argument('gt')
    ap.add_argument('--out'); ap.add_argument('--vins-log'); ap.add_argument('--depth-stats')
    a = ap.parse_args()
    est_full = load_tum(a.est)
    gt = load_euroc_gt(a.gt)
    ref, est = sync.associate_trajectories(gt, est_full, max_diff=0.01)
    est.align(ref, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))
    gt_dur = gt.timestamps[-1] - gt.timestamps[0]
    r = dict(ate_rmse_m=float(ape.get_statistic(metrics.StatisticsType.rmse)),
             ate_max_m=float(ape.get_statistic(metrics.StatisticsType.max)),
             rpe1_t_m=rpe(ref, est, 1.0, 't'), rpe1_r_deg=rpe(ref, est, 1.0, 'r'),
             rpe5_t_m=rpe(ref, est, 5.0, 't'), rpe5_r_deg=rpe(ref, est, 5.0, 'r'),
             poses=int(est_full.num_poses), matched=int(est.num_poses),
             coverage=float((est_full.timestamps[-1] - est_full.timestamps[0]) / gt_dur),
             max_gap_s=float(np.max(np.diff(est_full.timestamps))))
    if a.vins_log and Path(a.vins_log).exists():
        log = Path(a.vins_log).read_text(errors='ignore')
        obs = [int(x) for x in re.findall(r'LIVE_DEPTH_OBS t=\S+ accepted=(\d+)', log)]
        last = re.findall(r'received=(\d+) used=(\d+) missed=(\d+)', log)
        r['depth_frames_used'] = len(obs)
        r['depth_obs_mean'] = float(np.mean(obs)) if obs else 0.0
        if last:
            r['depth_received'], r['depth_used'], r['depth_missed'] = map(int, last[-1])
        r['vins_failures'] = len(re.findall(r'failure detection|system reboot', log))
    if a.depth_stats and Path(a.depth_stats).exists():
        s = np.loadtxt(a.depth_stats, delimiter=',', skiprows=1, comments='#', ndmin=2)
        if len(s):
            r['depth_compute_ms_median'] = float(np.median(s[:, 1]))
            r['depth_compute_ms_p95'] = float(np.percentile(s[:, 1], 95))
            r['depth_latency_ms_median'] = float(np.median(s[:, 2]))
            r['depth_latency_ms_p95'] = float(np.percentile(s[:, 2], 95))
            r['depth_rate_hz'] = float(len(s) / max((s[-1, 0] - s[0, 0]) * 1e-9, 1e-9))
            r['depth_valid_frac'] = float(np.mean(s[:, 3]))
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
