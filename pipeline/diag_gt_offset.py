"""Diagnostic: estimate a constant body-frame rotation offset between EuRoC GT orientation and the
OpenVINS (aligned) IMU orientation, then write GT positions with offset-corrected orientation.
R_gt(t) ≈ R_vio(t) · R_off  →  R_off = mean(R_vio⁻¹ · R_gt). Output is a DIAGNOSTIC, not ground truth."""
import json, os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from common import OUT
d9 = os.path.join(OUT, "stage9_eval")
g = np.loadtxt(os.path.join(d9, "gt_body_tum.txt")); e = np.loadtxt(os.path.join(d9, "vio_aligned_tum.txt"))
i = np.clip(np.searchsorted(g[:, 0], e[:, 0]), 0, len(g) - 1)
Rg, Re = Rot.from_quat(g[i, 4:8]), Rot.from_quat(e[:, 4:8])
off = (Re.inv() * Rg).mean()
before = (Re.inv() * Rg).magnitude() * 180 / np.pi; after = ((Re * off).inv() * Rg).magnitude() * 180 / np.pi
Rc = Rot.from_quat(g[:, 4:8]) * off.inv()
np.savetxt(os.path.join(d9, "gt_rotcorr_tum.txt"), np.column_stack([g[:, :4], Rc.as_quat()]), fmt="%.9f")
res = dict(offset_deg=float(off.magnitude() * 180 / np.pi), offset_rotvec_deg=(off.as_rotvec() * 180 / np.pi).tolist(),
           rot_diff_median_before_deg=float(np.median(before)), rot_diff_median_after_deg=float(np.median(after)))
json.dump(res, open(os.path.join(d9, "gt_rot_offset_diag.json"), "w"), indent=2); print(res)
