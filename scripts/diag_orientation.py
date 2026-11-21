#!/usr/bin/env python3
"""Independent check of ATE and of a possible constant body-frame rotation offset between estimate and GT.
Usage: diag_orientation.py <gt.txt> <est.txt> [reference_gt2.txt]
Convention: files hold `t x y z qx qy qz qw`; the estimate is OpenVINS' raw poseimu (JPL q_GtoI numerics == Hamilton q_ItoG).
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def load(p):
    a = np.loadtxt(p, comments="#")
    return a[:, 0], a[:, 1:4], R.from_quat(a[:, 4:8])


def match(t_ref, t_q, tol=0.01):
    idx = np.clip(np.searchsorted(t_ref, t_q), 1, len(t_ref) - 1)
    idx = np.where(np.abs(t_ref[idx] - t_q) < np.abs(t_ref[idx - 1] - t_q), idx, idx - 1)
    ok = np.abs(t_ref[idx] - t_q) < tol
    return idx, ok


tg, pg, qg = load(sys.argv[1])
te, pe, qe = load(sys.argv[2])
idx, ok = match(tg, te)
pg, qg, pe, qe = pg[idx][ok], qg[idx[ok]], pe[ok], qe[ok]
print(f"matched {ok.sum()} of {len(te)} estimate poses (|dt|<10ms)")

# Umeyama SE3 on positions
mg, me = pg.mean(0), pe.mean(0)
U, S, Vt = np.linalg.svd((pg - mg).T @ (pe - me))
D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
Rg = U @ D @ Vt  # maps est -> gt
pe_a = (Rg @ (pe - me).T).T + mg
err = np.linalg.norm(pg - pe_a, axis=1)
print(f"[independent] SE3-aligned position ATE rmse = {np.sqrt((err**2).mean()):.4f} m  max = {err.max():.4f} m")

# Orientation: R_gt_i = Rg * R_est_i * R_off  (constant body-frame offset)
R_al = R.from_matrix(Rg) * qe
rel = (qg.inv() * R_al)  # per-pose residual rotation (body frame)
ang = rel.magnitude() * 180 / np.pi
print(f"orientation error after SE3 align: mean {ang.mean():.2f} deg  (min {ang.min():.2f}, max {ang.max():.2f})")
# Is the residual a constant rotation in the body frame?
mean_rel = rel.mean()
resid = (mean_rel.inv() * rel).magnitude() * 180 / np.pi
print(f"constant body offset (rotvec deg): {np.round(mean_rel.as_rotvec()*180/np.pi, 2)}  -> magnitude {mean_rel.magnitude()*180/np.pi:.2f} deg")
print(f"orientation error after removing that constant offset: mean {resid.mean():.2f} deg  max {resid.max():.2f} deg")
