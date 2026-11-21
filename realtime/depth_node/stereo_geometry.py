"""Stereo calibration, rectification and rectified-disparity -> raw-cam0 inverse-depth conversion.

Calibration comes from the EuRoC/Kalibr style `sensor.yaml` (pinhole + radtan, T_BS).
Output depth lives on the *raw* (distorted) cam0 pixel grid because VINS-Fusion samples
features at raw pixel coordinates; the value is inverse optical Z of the raw cam0 frame.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml


def load_sensor_yaml(path):
    with open(path) as f:
        y = yaml.safe_load(f)
    fx, fy, cx, cy = y['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    D = np.array(y['distortion_coefficients'], np.float64)
    T = np.array(y['T_BS']['data'], np.float64).reshape(4, 4)
    return K, D, T, tuple(y['resolution'])


@dataclass
class StereoRig:
    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    T_B0: np.ndarray
    T_B1: np.ndarray
    size: tuple

    @classmethod
    def from_euroc(cls, cam0_yaml, cam1_yaml):
        K0, D0, T0, size = load_sensor_yaml(cam0_yaml)
        K1, D1, T1, _ = load_sensor_yaml(cam1_yaml)
        return cls(K0, D0, K1, D1, T0, T1, size)

    def rectify(self):
        T10 = np.linalg.inv(self.T_B1) @ self.T_B0
        R0, R1, P0, P1, *_ = cv2.stereoRectify(self.K0, self.D0, self.K1, self.D1, self.size,
                                               T10[:3, :3].copy(), T10[:3, 3].reshape(3, 1).copy(),
                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        m0 = cv2.initUndistortRectifyMap(self.K0, self.D0, R0, P0, self.size, cv2.CV_32FC1)
        m1 = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1, self.size, cv2.CV_32FC1)
        return dict(R0=R0, P0=P0, P1=P1, map0=m0, map1=m1, fx=P0[0, 0], fy=P0[1, 1], cx=P0[0, 2], cy=P0[1, 2],
                    baseline=-P1[0, 3] / P1[0, 0])


class DepthConverter:
    """GPU rectification of the input pair and disparity -> raw inverse depth + sigma proxy."""

    def __init__(self, rig: StereoRig, device='cuda', zmin=0.2, zmax=4.0, sigma_px=1.0, relative_sigma=0.025):
        self.rect = r = rig.rectify()
        w, h = rig.size
        self.size = (h, w)
        self.device = device
        self.zmin, self.zmax, self.sigma_px, self.relative_sigma = zmin, zmax, sigma_px, relative_sigma

        def grid(mx, my):
            g = np.stack([2 * mx / (w - 1) - 1, 2 * my / (h - 1) - 1], -1)[None]
            return torch.from_numpy(g.astype(np.float32)).to(device)
        self.grid0 = grid(*r['map0'])
        self.grid1 = grid(*r['map1'])
        # raw pixel -> rectified pixel (for sampling the rectified disparity on the raw grid)
        v, u = np.indices((h, w), dtype=np.float32)
        rays = cv2.undistortPoints(np.stack([u, v], -1).reshape(-1, 1, 2), rig.K0, rig.D0).reshape(h, w, 2)
        rays = np.concatenate([rays, np.ones((h, w, 1))], 2) @ r['R0'].T
        scale = rays[..., 2]
        mx = r['fx'] * rays[..., 0] / scale + r['cx']
        my = r['fy'] * rays[..., 1] / scale + r['cy']
        self.grid_raw = grid(mx, my)
        self.scale = torch.from_numpy(scale.astype(np.float32)).to(device)
        self.fb = float(r['fx'] * r['baseline'])

    def rectify_pair(self, left_u8, right_u8):
        """uint8 HxW numpy -> rectified float tensors (1,1,H,W) in [0,255] on the GPU."""
        def go(img, g):
            t = torch.from_numpy(img).to(self.device, non_blocking=True).float()[None, None]
            return F.grid_sample(t, g, mode='bilinear', padding_mode='zeros', align_corners=True)
        return go(left_u8, self.grid0), go(right_u8, self.grid1)

    @staticmethod
    def _warp_right(right, disp):
        """Sample the rectified right image at x - d (left view). right: (1,1,H,W), disp: (H,W)."""
        h, w = disp.shape
        xs = torch.arange(w, device=disp.device, dtype=torch.float32).view(1, w).expand(h, w) - disp
        ys = torch.arange(h, device=disp.device, dtype=torch.float32).view(h, 1).expand(h, w)
        g = torch.stack([2 * xs / (w - 1) - 1, 2 * ys / (h - 1) - 1], -1)[None]
        return F.grid_sample(right, g, mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0], \
            (xs >= 0) & (xs <= w - 1)

    def photometric_valid(self, left, right, disp, thresh=10.0, win=5):
        """Mean absolute intensity difference (win x win) between left and the disparity-warped right image.
        Large residuals flag occlusions, specularities and wrong matches."""
        warped, inside = self._warp_right(right, disp)
        err = (left[0, 0] - warped).abs()[None, None]
        err = F.avg_pool2d(err, win, stride=1, padding=win // 2)[0, 0]
        return inside & (err < thresh)

    @staticmethod
    def lr_valid(disp_left, disp_right, thresh=1.0):
        """Left-right consistency: |dL(x) - dR(x - dL(x))| < thresh. disp_right is the right-view disparity."""
        h, w = disp_left.shape
        xs = torch.arange(w, device=disp_left.device, dtype=torch.float32).view(1, w).expand(h, w) - disp_left
        ys = torch.arange(h, device=disp_left.device, dtype=torch.float32).view(h, 1).expand(h, w)
        g = torch.stack([2 * xs / (w - 1) - 1, 2 * ys / (h - 1) - 1], -1)[None]
        dr = F.grid_sample(disp_right[None, None], g, mode='nearest', align_corners=True)[0, 0]
        return (xs >= 0) & ((disp_left - dr).abs() < thresh)

    def raw_inverse_depth(self, disp_rect, valid=None):
        """disp_rect: (H,W) tensor of rectified left disparity [px]. Returns (H,W,2) float32 numpy."""
        d = disp_rect[None, None].float()
        if valid is not None:
            d = torch.where(valid[None, None], d, torch.zeros_like(d))
        d = F.grid_sample(d, self.grid_raw, mode='nearest', padding_mode='zeros', align_corners=True)[0, 0]
        inv = self.scale * d / self.fb
        sigma = torch.maximum(self.relative_sigma * inv, self.scale.abs() * self.sigma_px / self.fb)
        good = (d > 0) & (inv >= 1 / self.zmax) & (inv <= 1 / self.zmin) & (self.scale > 0)
        out = torch.stack([torch.where(good, inv, torch.zeros_like(inv)),
                           torch.where(good, sigma, torch.zeros_like(sigma))], -1)
        return out.cpu().numpy()
