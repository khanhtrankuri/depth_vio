"""Stereo calibration from EuRoC/Kalibr `sensor.yaml` files and OpenCV rectification (no torch)."""
from dataclasses import dataclass

import cv2
import numpy as np
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
