"""Shared helpers: EuRoC ASL loading, calibration, stereo rectification, GT poses."""
import os
import numpy as np
import cv2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.environ.get("EUROC_SEQ", "V1_01_easy")
MAV = os.path.join(ROOT, "data", SEQ, "mav0")
OUT = os.path.join(ROOT, "outputs", SEQ)


def outdir(stage):
    d = os.path.join(OUT, stage)
    os.makedirs(d, exist_ok=True)
    return d


def load_sensor(name):
    with open(os.path.join(MAV, name, "sensor.yaml")) as f:
        y = yaml.safe_load(f)
    out = dict(y)
    if "T_BS" in y:
        out["T_BS"] = np.array(y["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    return out


def cam_calib(name):
    s = load_sensor(name)
    fx, fy, cx, cy = s["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array(s["distortion_coefficients"], dtype=np.float64)  # k1 k2 p1 p2
    return K, D, s["T_BS"], tuple(s["resolution"])


def stereo_rectify(alpha=0.0):
    """Returns dict with rectification maps and rectified intrinsics.

    OpenCV convention: R, T map a point from cam0 coords into cam1 coords.
    T_c1_c0 = inv(T_B_c1) @ T_B_c0.
    """
    K0, D0, T_B0, size = cam_calib("cam0")
    K1, D1, T_B1, _ = cam_calib("cam1")
    T_10 = np.linalg.inv(T_B1) @ T_B0
    R, T = T_10[:3, :3].copy(), T_10[:3, 3].reshape(3, 1).copy()
    R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
        K0, D0, K1, D1, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    m0 = cv2.initUndistortRectifyMap(K0, D0, R0, P0, size, cv2.CV_32FC1)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    fx = P0[0, 0]
    baseline = -P1[0, 3] / P1[0, 0]
    return dict(K0=K0, D0=D0, K1=K1, D1=D1, T_B0=T_B0, T_B1=T_B1, T_10=T_10,
                R0=R0, R1=R1, P0=P0, P1=P1, Q=Q, roi0=roi0, roi1=roi1,
                map0=m0, map1=m1, fx=fx, fy=P0[1, 1], cx=P0[0, 2], cy=P0[1, 2],
                baseline=baseline, size=size)


def rectify_pair(rect, left, right):
    l = cv2.remap(left, rect["map0"][0], rect["map0"][1], cv2.INTER_LINEAR)
    r = cv2.remap(right, rect["map1"][0], rect["map1"][1], cv2.INTER_LINEAR)
    return l, r


def list_frames(cam="cam0"):
    rows = np.genfromtxt(os.path.join(MAV, cam, "data.csv"), delimiter=",",
                         dtype=None, encoding=None, comments="#")
    return [(int(t), os.path.join(MAV, cam, "data", f)) for t, f in rows]


def stereo_pairs():
    """Pairs cam0/cam1 frames by identical timestamp."""
    c0 = dict(list_frames("cam0"))
    c1 = dict(list_frames("cam1"))
    ts = sorted(set(c0) & set(c1))
    return [(t, c0[t], c1[t]) for t in ts], len(c0), len(c1)


def read_gray(p):
    return cv2.imread(p, cv2.IMREAD_UNCHANGED)


def quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_gt():
    """Returns (t_ns [N], T_W_B [N,4,4]) from state_groundtruth_estimate0 (body = IMU)."""
    d = np.loadtxt(os.path.join(MAV, "state_groundtruth_estimate0", "data.csv"),
                   delimiter=",", comments="#")
    t = d[:, 0].astype(np.int64)
    T = np.tile(np.eye(4), (len(d), 1, 1))
    for i, r in enumerate(d):
        T[i, :3, :3] = quat_to_R(*r[4:8])
        T[i, :3, 3] = r[1:4]
    return t, T


def interp_pose(t_query, t, T, max_gap_ns=20_000_000):
    """Nearest-neighbour pose lookup (GT is 200 Hz → ≤2.5 ms error). None if outside."""
    i = np.searchsorted(t, t_query)
    cand = [j for j in (i - 1, i) if 0 <= j < len(t)]
    if not cand:
        return None
    j = min(cand, key=lambda k: abs(t[k] - t_query))
    if abs(t[j] - t_query) > max_gap_ns:
        return None
    return T[j]


def T_W_cam0rect(T_W_B, rect):
    """World pose of the *rectified* cam0 frame: T_W_B · T_B_c0 · R0ᵀ."""
    T_c0_rect = np.eye(4)
    T_c0_rect[:3, :3] = rect["R0"].T
    return T_W_B @ rect["T_B0"] @ T_c0_rect
