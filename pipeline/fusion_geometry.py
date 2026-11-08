"""Geometry contracts for depth/VIO fusion. No GT, OpenCV, or import-time I/O."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float

    def rays(self):
        v, u = np.indices((self.height, self.width), dtype=np.float32)
        return np.stack(((u-self.cx)/self.fx, (v-self.cy)/self.fy, np.ones_like(u)), -1)


class PoseStream:
    """T_WI Hamilton poses. Linear position + shortest-arc SLERP, no extrapolation.

    Query relative seconds to avoid subtracting floating-point epoch nanoseconds.
    OpenVINS text covariance blocks are retained, never silently interpreted as
    calibrated camera-frame uncertainty or as independent relative-pose covariance.
    """
    def __init__(self, path):
        a = np.atleast_2d(np.loadtxt(path, comments='#'))
        if a.shape[1] < 8 or len(a) < 2 or not np.isfinite(a[:, :8]).all():
            raise ValueError('Need at least two finite timestamped TUM poses')
        self.epoch_s = float(a[0, 0])
        self.t = a[:, 0]-self.epoch_s
        if (np.diff(self.t) <= 0).any():
            raise ValueError('Pose timestamps must increase strictly')
        self.p = a[:, 1:4]
        self.rot = Rotation.from_quat(a[:, 4:8])
        self.slerp = Slerp(self.t, self.rot)
        self.covariance_blocks = a[:, 8:20] if a.shape[1] >= 20 else None

    def at_ns(self, timestamp_ns, max_gap_s=.151):
        # Integer epoch decomposition retains sub-microsecond precision.
        sec, nano = divmod(int(timestamp_ns), 1_000_000_000)
        q = (sec-self.epoch_s) + nano*1e-9
        if q < self.t[0] or q > self.t[-1]:
            return None
        hi = int(np.searchsorted(self.t, q, side='right'))
        hi = min(max(hi, 1), len(self.t)-1); lo = hi-1
        gap = float(self.t[hi]-self.t[lo])
        if gap > max_gap_s:
            return None
        alpha = (q-self.t[lo])/gap
        T = np.eye(4)
        T[:3, :3] = self.slerp([q]).as_matrix()[0]
        T[:3, 3] = (1-alpha)*self.p[lo]+alpha*self.p[hi]
        return T, dict(bracket_s=gap, nearest_dt_s=float(min(q-self.t[lo], self.t[hi]-q)))


def calibration(root, seq):
    """Reuse saved rectified intrinsics; derive rectification rotation from sensors.

    The JSON contains calibration only; no trajectory/ground truth is consulted.
    R0 is OpenCV stereoRectify's horizontal-rig half-rotation/baseline construction.
    """
    import json
    root = Path(root)
    rect = json.loads((root/'outputs'/seq/'stage2_rectify/rectify.json').read_text())
    sensors = [yaml.safe_load((root/'data'/seq/'mav0'/c/'sensor.yaml').read_text()) for c in ['cam0','cam1']]
    Ts = [np.array(s['T_BS']['data'], dtype=float).reshape(4,4) for s in sensors]
    rel = np.linalg.inv(Ts[1])@Ts[0]
    half = Rotation.from_rotvec(-.5*Rotation.from_matrix(rel[:3,:3]).as_rotvec()).as_matrix()
    t = half@rel[:3,3]
    if abs(t[0]) < abs(t[1]):
        raise ValueError('Only horizontal stereo disparity is supported')
    axis = np.cross(t, [np.sign(t[0]),0,0])
    angle = np.arccos(np.clip(abs(t[0])/np.linalg.norm(t),-1,1))
    wr = Rotation.from_rotvec(axis/max(np.linalg.norm(axis),1e-15)*angle).as_matrix()
    R0 = wr@half.T
    ext = Ts[0].copy(); ext[:3,:3] = ext[:3,:3]@R0.T
    w,h = sensors[0]['resolution']
    cam = Camera(w,h,rect['fx_rect'],rect['fy_rect'],rect['cx_rect'],rect['cy_rect'],rect['baseline_m'])
    return cam, ext, R0, sensors[0]


def depth_from_disparity(disparity, mask, camera, zmin=.2, zmax=4.):
    d = np.asarray(disparity,dtype=np.float32)
    good = np.asarray(mask,bool)&np.isfinite(d)&(d>.5)
    z = np.zeros_like(d)
    np.divide(camera.fx*camera.baseline,d,out=z,where=good)
    z[(~np.isfinite(z))|(z<zmin)|(z>zmax)] = 0
    return z


def transform(points, T):
    return points@T[:3,:3].T+T[:3,3]


def frustum_bounds(T_WC, cam, zmin, zmax):
    uv = np.array([[0,0],[cam.width-1,0],[0,cam.height-1],[cam.width-1,cam.height-1]])
    rays = np.column_stack(((uv[:,0]-cam.cx)/cam.fx,(uv[:,1]-cam.cy)/cam.fy,np.ones(4)))
    p = transform(np.vstack((rays*zmin,rays*zmax)),T_WC)
    return p.min(0),p.max(0)
