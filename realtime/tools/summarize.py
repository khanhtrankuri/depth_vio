#!/usr/bin/env python3
"""Summarise runs under ~/vio_data/rt/runs/<seq>/<tag>_<name>_r<rep>/metrics.json into a table + JSON."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path.home() / 'vio_data/rt/runs'
tag = sys.argv[1] if len(sys.argv) > 1 else ''
rows = defaultdict(lambda: defaultdict(list))
for m in root.glob('*/*/metrics.json'):
    seq, run = m.parent.parent.name, m.parent.name
    if tag and not run.startswith(tag + '_'):
        continue
    name = re.sub(r'_r\d+$', '', run)
    rows[name][seq].append(json.loads(m.read_text()))
seqs = sorted({s for v in rows.values() for s in v})
keys = ['ate_rmse_m', 'rpe1_t_m', 'rpe1_r_deg']
print(f"{'config':40s} " + ' '.join(f'{s[:6]:>18s}' for s in seqs) + '   mean ATE cm')
out = {}
for name in sorted(rows):
    cells, ates = [], []
    out[name] = {}
    for s in seqs:
        rs = rows[name].get(s, [])
        if not rs:
            cells.append(f"{'-':>18s}"); continue
        ate = np.array([r['ate_rmse_m'] for r in rs]) * 100
        ates.append(ate.mean())
        cells.append(f"{ate.mean():6.2f}±{ate.std():4.2f} (n={len(rs)})")
        out[name][s] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if isinstance(rs[0][k], (int, float))}
        out[name][s]['runs'] = len(rs)
        out[name][s]['ate_cm_runs'] = ate.round(3).tolist()
    mean = f'{np.mean(ates):6.2f}' if len(ates) == len(seqs) else '   -'
    print(f'{name:40s} ' + ' '.join(cells) + '   ' + mean)
(Path.home() / 'vio_data/rt' / f'summary_{tag or "all"}.json').write_text(json.dumps(out, indent=2))
