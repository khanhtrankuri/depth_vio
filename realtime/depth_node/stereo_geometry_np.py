"""Torch-free twin of stereo_geometry.DepthConverter (numpy + OpenCV, CPU).

Same maths as the GPU version: rectify the raw pair with cv2.stereoRectify (CALIB_ZERO_DISPARITY, alpha=0),
then sample the rectified left disparity at every raw cam0 pixel (nearest) and convert it to inverse optical Z
of the raw cam0 frame plus a 1-sigma proxy. Used by the TensorRT depth node on the Jetson, where no PyTorch exists.
"""
import cv2
import numpy as np


class DepthConverterNP:
    def __init__(self, rig, zmin=0.2, zmax=4.0, sigma_px=1.0, relative_sigma=0.025):
        self.rect = r = rig.rectify()
        w, h = rig.size
        self.size = (h, w)
        self.zmin, self.zmax, self.sigma_px, self.relative_sigma = zmin, zmax, sigma_px, relative_sigma
        if r['baseline'] <= 0:
            raise ValueError(f'cam1 is not to the right of cam0 after rectification (baseline {r["baseline"]:.4f} m); '
                             'swap cam0/cam1')
        # float maps on float images: matches the GPU grid_sample rectification (fixed-point maps / uint8 output
        # differ by ~1 gray level on average, which the network sees)
        self.map0, self.map1 = r['map0'], r['map1']
        # raw cam0 pixel -> rectified pixel (where to read the rectified disparity)
        v, u = np.indices((h, w), dtype=np.float32)
        rays = cv2.undistortPoints(np.stack([u, v], -1).reshape(-1, 1, 2), rig.K0, rig.D0).reshape(h, w, 2)
        rays = np.concatenate([rays, np.ones((h, w, 1))], 2) @ r['R0'].T
        scale = rays[..., 2]
        mx = r['fx'] * rays[..., 0] / scale + r['cx']
        my = r['fy'] * rays[..., 1] / scale + r['cy']
        # nearest sampling, same as grid_sample(mode='nearest', align_corners=True); outside -> invalid
        ix, iy = np.rint(mx).astype(np.int64), np.rint(my).astype(np.int64)
        self.inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & (scale > 0)
        self.flat_idx = (np.clip(iy, 0, h - 1) * w + np.clip(ix, 0, w - 1)).astype(np.int64)
        self.scale = scale.astype(np.float32)
        self.fb = float(r['fx'] * r['baseline'])
        self.sigma_floor = (np.abs(self.scale) * self.sigma_px / self.fb).astype(np.float32)

    def rectify_pair(self, left_u8, right_u8):
        """uint8 HxW -> rectified float32 HxW in [0,255] (bilinear, zero border)."""
        l = cv2.remap(left_u8.astype(np.float32), *self.map0, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        r = cv2.remap(right_u8.astype(np.float32), *self.map1, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return l, r

    def raw_inverse_depth(self, disp_rect):
        """disp_rect: HxW float32 rectified left disparity [px]. Returns HxWx2 float32 (inv depth, sigma)."""
        d = disp_rect.reshape(-1)[self.flat_idx]
        d = np.where(self.inside, d, 0.0).astype(np.float32)
        inv = self.scale * d / self.fb
        sigma = np.maximum(self.relative_sigma * inv, self.sigma_floor)
        good = (d > 0) & (inv >= 1 / self.zmax) & (inv <= 1 / self.zmin)
        out = np.zeros(self.size + (2,), np.float32)
        out[..., 0] = np.where(good, inv, 0.0)
        out[..., 1] = np.where(good, sigma, 0.0)
        return out
