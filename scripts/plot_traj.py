#!/usr/bin/env python3
"""Plot estimate vs ground truth (SE3-aligned on positions) and per-axis error. Saves a PNG and prints per-axis stats.
Usage: plot_traj.py <gt.txt> <est.txt> <out.png> [title]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

gt = np.loadtxt(sys.argv[1], comments="#")
es = np.loadtxt(sys.argv[2], comments="#")
title = sys.argv[4] if len(sys.argv) > 4 else "OpenVINS vs ground truth"

tg, pg = gt[:, 0], gt[:, 1:4]
te, pe = es[:, 0], es[:, 1:4]
idx = np.clip(np.searchsorted(tg, te), 1, len(tg) - 1)
idx = np.where(np.abs(tg[idx] - te) < np.abs(tg[idx - 1] - te), idx, idx - 1)
ok = np.abs(tg[idx] - te) < 0.01
pg, pe, te = pg[idx][ok], pe[ok], te[ok]

mg, me = pg.mean(0), pe.mean(0)
U, S, Vt = np.linalg.svd((pg - mg).T @ (pe - me))
Rg = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
pa = (Rg @ (pe - me).T).T + mg
e = pa - pg
t = te - te[0]
norm = np.linalg.norm(e, axis=1)
length = np.linalg.norm(np.diff(pg, axis=0), axis=1).sum()

print(f"poses: {len(te)}  duration: {t[-1]:.1f}s  GT path length: {length:.2f} m")
for i, a in enumerate("XYZ"):
    print(f"axis {a}: rmse {np.sqrt((e[:, i]**2).mean())*100:.2f} cm  mean {e[:, i].mean()*100:+.2f} cm  max|e| {np.abs(e[:, i]).max()*100:.2f} cm")
print(f"3D: rmse {np.sqrt((norm**2).mean())*100:.2f} cm  max {norm.max()*100:.2f} cm   rmse/length = {np.sqrt((norm**2).mean())/length*100:.2f} %")
print(f"final-pose error (aligned): {norm[-1]*100:.2f} cm")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].plot(pg[:, 0], pg[:, 1], "k-", lw=2, label="ground truth")
ax[0, 0].plot(pa[:, 0], pa[:, 1], "r-", lw=1, label="estimate (SE3-aligned)")
ax[0, 0].set(xlabel="x [m]", ylabel="y [m]", title="Top view (XY)", aspect="equal"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
ax[0, 1].plot(t, pg[:, 2], "k-", label="GT z"); ax[0, 1].plot(t, pa[:, 2], "r-", label="est z")
ax[0, 1].set(xlabel="time [s]", ylabel="z [m]", title="Height"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
for i, (a, c) in enumerate(zip("XYZ", ["tab:red", "tab:green", "tab:blue"])):
    ax[1, 0].plot(t, e[:, i] * 100, c, lw=1, label=f"{a} error")
ax[1, 0].set(xlabel="time [s]", ylabel="error [cm]", title="Per-axis position error"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
ax[1, 1].plot(t, norm * 100, "k-"); ax[1, 1].set(xlabel="time [s]", ylabel="3D error [cm]", title="Absolute position error"); ax[1, 1].grid(alpha=.3)
fig.suptitle(title)
fig.tight_layout()
fig.savefig(sys.argv[3], dpi=110)
print("saved", sys.argv[3])
