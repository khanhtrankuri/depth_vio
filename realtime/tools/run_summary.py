#!/usr/bin/env python3
"""Summary of one run_live_trt.sh run directory (numpy only, runs on the Jetson).

    python3 run_summary.py <run_dir> [<run_dir> ...]

Reports, per run:
  - depth usage in VINS (LIVE_DEPTH_OBS lines: frames that got depth, features accepted, used/missed counters)
  - VINS solver time per window
  - depth node timing (depth_stats.csv)
  - pose output lag above its minimum (est.txt.recv): growing values mean the backend falls behind real time
  - closed-loop drift: start-end distance / path length (meaningful only if the run starts and ends at the
    same physical spot), with path length summed over steps >= 2 cm
"""
import re
import sys
from pathlib import Path

import numpy as np


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float('nan')


def summarize(run):
    run = Path(run)
    print(f'=== {run}')
    if not run.is_dir():
        print('  directory does not exist (the run did not start?)')
        return
    vins = (run / 'vins.log').read_text(errors='replace') if (run / 'vins.log').exists() else ''
    for bad in ('terminate called', 'what():', 'numerical unstable', 'failure detection', 'system reboot'):
        n = vins.count(bad)
        if n:
            print(f'  !! vins.log contains "{bad}" x{n}')
    obs = re.findall(r'LIVE_DEPTH_OBS t=([\d.]+) accepted=(\d+) received=(\d+) used=(\d+) missed=(\d+)(?: skipped=(\d+))?', vins)
    if 'LIVE_DEPTH topic' in vins:
        if obs:
            acc = np.array([int(o[1]) for o in obs])
            last = obs[-1]
            span = float(obs[-1][0]) - float(obs[0][0])
            print(f'  depth in VINS: {len(obs)} frames with depth over {span:.1f} s (~{len(obs) / max(span, 1e-9):.1f} Hz), '
                  f'features accepted per frame median {np.median(acc):.0f} (p5 {pct(acc, 5):.0f})')
            print(f'                 depth images received={last[2]} used={last[3]} missed={last[4]} '
                  f'{"skipped=" + last[5] + " " if last[5] else ""}(missed = waited, depth never arrived; skipped = not computed, no wait)')
        else:
            print('  depth in VINS: depth enabled but NO frame used depth (no LIVE_DEPTH_OBS lines)')
    else:
        print('  depth in VINS: disabled (no depth_topic)')
    solver = np.array([float(x) for x in re.findall(r'solver costs: ([\d.]+)', vins)])
    if len(solver):
        print(f'  VINS solver: {len(solver)} windows, median {np.median(solver):.1f} ms, p95 {pct(solver, 95):.1f} ms, '
              f'max {solver.max():.1f} ms')

    st = run / 'depth_stats.csv'
    if st.exists():
        rows = [r.split(',') for r in st.read_text().splitlines()[1:] if r and not r.startswith('#')]
        if rows:
            a = np.array(rows, float)
            tail = [r for r in st.read_text().splitlines() if r.startswith('#')]
            print(f'  depth node: {len(a)} frames, compute median {np.median(a[:, 4]):.1f} ms (p95 {pct(a[:, 4], 95):.1f}), '
                  f'net {np.median(a[:, 2]):.1f} ms, latency median {np.median(a[:, 5]):.1f} ms (p95 {pct(a[:, 5], 95):.1f}), '
                  f'valid {a[:, 6].mean():.2f}  {tail[0][2:] if tail else ""}')

    est = run / 'est.txt'
    if not est.exists():
        print('  no est.txt')
        return
    e = np.loadtxt(est, comments='#', ndmin=2)
    if len(e) < 2:
        print(f'  est.txt has {len(e)} poses')
        return
    t, p = e[:, 0], e[:, 1:4]
    recv = run / 'est.txt.recv'
    if recv.exists():
        r = np.loadtxt(recv, ndmin=2)
        if len(r) > 10:
            lag = r[:, 1] - r[:, 0]
            lag -= lag.min()
            n = len(lag)
            print(f'  pose lag above minimum: median {np.median(lag) * 1e3:.0f} ms, p95 {pct(lag, 95) * 1e3:.0f} ms, '
                  f'first 10% {np.median(lag[:n // 10]) * 1e3:.0f} ms -> last 10% {np.median(lag[-n // 10:]) * 1e3:.0f} ms')
    L, last = 0.0, p[0]
    for x in p[1:]:
        d = np.linalg.norm(x - last)
        if d >= 0.02:
            L += d
            last = x
    d = p[-1] - p[0]
    print(f'  trajectory: {len(e)} poses, {t[-1] - t[0]:.1f} s, path {L:.2f} m, '
          f'start-end {np.linalg.norm(d):.3f} m (XY {np.linalg.norm(d[:2]):.3f}, Z {d[2]:+.3f}) '
          f'= {100 * np.linalg.norm(d) / L if L > 0 else float("nan"):.2f} % of path')


if __name__ == '__main__':
    for a in sys.argv[1:]:
        summarize(a)
