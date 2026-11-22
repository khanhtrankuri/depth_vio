#!/usr/bin/env python3
"""Compare repeated runs (numpy only, runs on the Jetson).

    python3 compare_runs.py <out_dir>            # run dirs <variant>_r<N> inside out_dir
    python3 compare_runs.py <run_dir> <run_dir> ...

Closed-loop drift is measured over a COMMON time window for all runs: from the latest first pose to the earliest
last pose. Runs that initialise at different moments otherwise start their "loop" at different physical places.
Drift = |p(t_end) - p(t_start)| / path length over the window (path summed over steps >= 2 cm). Meaningful only if
the recording starts and ends at the same spot.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_run(d):
    d = Path(d)
    est = d / 'est.txt'
    if not est.exists():
        return None
    e = np.loadtxt(est, comments='#', ndmin=2)
    if len(e) < 10:
        return None
    r = dict(name=d.name, t=e[:, 0], p=e[:, 1:4])
    vins = (d / 'vins.log').read_text(errors='replace') if (d / 'vins.log').exists() else ''
    obs = re.findall(r'LIVE_DEPTH_OBS t=([\d.]+) accepted=\d+ received=\d+ used=(\d+) missed=(\d+)', vins)
    if obs:
        span = float(obs[-1][0]) - float(obs[0][0])
        r['depth_hz'] = len(obs) / max(span, 1e-9)
        r['missed'] = int(obs[-1][2])
    solver = [float(x) for x in re.findall(r'solver costs: ([\d.]+)', vins)]
    r['solver'] = float(np.median(solver)) if solver else float('nan')
    recv = d / 'est.txt.recv'
    if recv.exists():
        rc = np.loadtxt(recv, ndmin=2)
        if len(rc) > 10:
            lag = rc[:, 1] - rc[:, 0]
            r['lag'] = float(np.median(lag - lag.min()) * 1e3)
    r['fail'] = sum(vins.count(k) for k in ('failure detection', 'system reboot', 'numerical unstable'))
    return r


def drift(r, t0, t1):
    m = (r['t'] >= t0) & (r['t'] <= t1)
    p = r['p'][m]
    L, last = 0.0, p[0]
    for x in p[1:]:
        s = np.linalg.norm(x - last)
        if s >= 0.02:
            L += s
            last = x
    d = p[-1] - p[0]
    return dict(L=L, d=np.linalg.norm(d), xy=np.linalg.norm(d[:2]), z=d[2], pct=100 * np.linalg.norm(d) / L)


def main():
    args = [Path(a) for a in sys.argv[1:]]
    dirs = sorted(x for x in args[0].iterdir() if x.is_dir()) if len(args) == 1 and not (args[0] / 'est.txt').exists() \
        else args
    runs = [r for r in (load_run(d) for d in dirs) if r]
    if not runs:
        raise SystemExit('no runs with est.txt found')
    t0 = max(r['t'][0] for r in runs)
    t1 = min(r['t'][-1] for r in runs)
    tmin = min(r['t'][0] for r in runs)
    starts = ', '.join('%s +%.1f s' % (r['name'], r['t'][0] - tmin) for r in runs)
    print(f'common window: {t1 - t0:.1f} s (latest start {t0:.2f}, earliest end {t1:.2f})')
    print(f'first pose per run: {starts}\n')
    hdr = f'{"run":<18}{"path m":>8}{"end-start m":>12}{"XY m":>7}{"Z m":>8}{"drift %":>9}{"depth Hz":>9}{"missed":>7}' \
          f'{"lag ms":>8}{"solver ms":>10}{"fail":>5}'
    print(hdr)
    groups = defaultdict(list)
    for r in runs:
        x = drift(r, t0, t1)
        groups[re.sub(r'_r\d+$', '', r['name'])].append(x)
        print('%-18s%8.2f%12.3f%7.3f%+8.3f%9.2f%9.1f%7d%8.0f%10.1f%5d' % (
            r['name'], x['L'], x['d'], x['xy'], x['z'], x['pct'], r.get('depth_hz', float('nan')),
            r.get('missed', 0), r.get('lag', float('nan')), r['solver'], r['fail']))
    print('\nmean +- std per variant (n runs)')
    for g, xs in groups.items():
        a = {k: np.array([x[k] for x in xs]) for k in ('pct', 'xy', 'z', 'L')}
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else float('nan')  # noqa: E731
        print('  %-14s n=%d  drift %.2f +- %.2f %%   XY %.3f +- %.3f m   Z %+.3f +- %.3f m   path %.2f m' % (
            g, len(xs), a['pct'].mean(), sd(a['pct']), a['xy'].mean(), sd(a['xy']), a['z'].mean(), sd(a['z']),
            a['L'].mean()))


if __name__ == '__main__':
    main()
