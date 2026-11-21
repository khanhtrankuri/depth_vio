#!/usr/bin/env python3
"""Live learned-stereo depth node for VINS-Fusion (ROS2 Humble).

Subscribes to the raw stereo pair, rectifies on the GPU, runs a learned stereo network (RT-IGEV by default,
CUDA-graph replay) and publishes a 32FC2 image on the *raw cam0 pixel grid*:
    channel 0: inverse optical Z of raw cam0 [1/m] (0 = invalid)
    channel 1: 1-sigma of the inverse depth (proxy: max(relative, disparity-noise))
The header stamp is the exact cam0 exposure stamp, so the estimator can match it without interpolation.

Real-time policy: always process the newest complete stereo pair and drop older ones (never queue), so latency
stays bounded when the GPU is slower than the camera.

    ros2 run is not needed:  python depth_node.py --ros-args -p cam0_yaml:=... -p cam1_yaml:=...
"""
import array
import csv
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

from stereo_geometry import StereoRig, DepthConverter
import stereo_models


def to_gray(msg):
    if msg.encoding not in ('mono8', '8UC1'):
        raise ValueError(f'expected mono8 images, got {msg.encoding}')
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    return np.ascontiguousarray(a[:, :msg.width])


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


class DepthNode(Node):
    def __init__(self):
        super().__init__('learned_depth')
        dyn = ParameterDescriptor(dynamic_typing=True)
        p = lambda n, v: self.declare_parameter(n, v, dyn).value  # noqa: E731
        self.model_spec = p('model', 'rt:sceneflow:6')
        cam0, cam1 = p('cam0_yaml', ''), p('cam1_yaml', '')
        self.max_rate = float(p('max_rate_hz', 0.0))
        self.stats_csv = p('stats_csv', '')
        self.photo_thresh = float(p('photo_thresh', 0.0))  # gray levels, 0 = off
        self.lr_thresh = float(p('lr_thresh', 0.0))        # px, 0 = off (costs a second inference)
        out_topic = p('depth_topic', '/depth/inverse_depth')
        if not cam0 or not cam1:
            raise SystemExit('cam0_yaml and cam1_yaml (EuRoC/Kalibr sensor.yaml) are required')
        torch.backends.cudnn.benchmark = True
        self.conv = DepthConverter(StereoRig.from_euroc(cam0, cam1), zmin=float(p('zmin', 0.2)),
                                   zmax=float(p('zmax', 4.0)), sigma_px=float(p('sigma_px', 1.0)),
                                   relative_sigma=float(p('relative_sigma', 0.025)))
        self.model = stereo_models.build(self.model_spec)
        h, w = self.conv.size
        dummy = np.zeros((h, w), np.uint8)
        for _ in range(3):  # warm-up + CUDA-graph capture before data arrives
            self.model(*self.conv.rectify_pair(dummy, dummy))
        torch.cuda.synchronize()

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub = self.create_publisher(Image, out_topic, qos)
        self.create_subscription(Image, p('image0_topic', '/cam0/image_raw'), lambda m: self.on_image(0, m), qos)
        self.create_subscription(Image, p('image1_topic', '/cam1/image_raw'), lambda m: self.on_image(1, m), qos)
        self.buf = ({}, {})
        self.latest = None
        self.cv = threading.Condition()
        self.stats = []
        self.received = self.dropped = 0
        self.last_proc = 0.0
        self.running = True
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()
        self.get_logger().info(f'learned depth ready: model={self.model_spec} -> {out_topic}')

    def on_image(self, cam, msg):
        t = stamp_ns(msg)
        with self.cv:
            self.buf[cam][t] = (msg, time.monotonic())
            other = self.buf[1 - cam]
            if t in other:
                m0, m1 = (msg, other[t][0]) if cam == 0 else (other[t][0], msg)
                if self.latest is not None:
                    self.dropped += 1
                self.latest = (t, m0, m1, time.monotonic())
                self.received += 1
                for b in self.buf:  # discard everything up to this exposure
                    for k in [k for k in b if k <= t]:
                        del b[k]
                self.cv.notify()
            for b in self.buf:
                while len(b) > 20:
                    del b[min(b)]

    def loop(self):
        while self.running:
            with self.cv:
                while self.latest is None and self.running:
                    self.cv.wait(0.1)
                if not self.running:
                    return
                t, m0, m1, t_rx = self.latest
                self.latest = None
            if self.max_rate > 0 and time.monotonic() - self.last_proc < 1.0 / self.max_rate:
                continue
            self.last_proc = time.monotonic()
            t0 = time.monotonic()
            l, r = self.conv.rectify_pair(to_gray(m0), to_gray(m1))
            disp = self.model(l, r)
            valid = None
            if self.photo_thresh > 0:
                valid = self.conv.photometric_valid(l, r, disp, self.photo_thresh)
            if self.lr_thresh > 0:
                # right-view disparity with the horizontal-flip trick (second pass of the same graph)
                disp_r = self.model(torch.flip(r, [-1]), torch.flip(l, [-1])).flip(-1)
                lr = self.conv.lr_valid(disp, disp_r, self.lr_thresh)
                valid = lr if valid is None else valid & lr
            depth = self.conv.raw_inverse_depth(disp, valid)
            t1 = time.monotonic()
            out = Image()
            out.header = m0.header
            out.height, out.width = depth.shape[:2]
            out.encoding = '32FC2'
            out.is_bigendian = 0
            out.step = out.width * 8
            out.data = array.array("B", depth.tobytes())  # bytes assignment is ~300x slower in rclpy
            self.pub.publish(out)
            valid = float((depth[..., 0] > 0).mean())
            self.stats.append((t, (t1 - t0) * 1e3, (time.monotonic() - t_rx) * 1e3, valid))
            if len(self.stats) % 100 == 0:
                s = np.array(self.stats[-100:])
                self.get_logger().info(f'depth frames={len(self.stats)} compute={np.median(s[:, 1]):.1f}ms '
                                       f'latency={np.median(s[:, 2]):.1f}ms valid={s[:, 3].mean():.2f} '
                                       f'dropped={self.dropped}/{self.received}')

    def shutdown(self):
        self.running = False
        if self.stats_csv:
            Path(self.stats_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['stamp_ns', 'compute_ms', 'latency_ms', 'valid_frac'])
                w.writerows(self.stats)
                f.write(f'# received={self.received} dropped={self.dropped} model={self.model_spec}\n')


def main():
    rclpy.init()
    node = DepthNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
