#!/usr/bin/env python3
"""Live learned-stereo depth node for VINS-Fusion without PyTorch (TensorRT engine, Jetson).

Same output contract as depth_node.py, so the patched VINS-Fusion needs no change:
    /depth/inverse_depth  sensor_msgs/Image 32FC2 on the *raw cam0 pixel grid*
        channel 0: inverse optical Z of raw cam0 [1/m] (0 = invalid)
        channel 1: 1-sigma of the inverse depth
    header = the cam0 image header (exact exposure stamp).

Request/pending protocol with the patched VINS-Fusion (default on, follow_requests:=true):
    <depth_topic>/request  std_msgs/Header  VINS announces each frame its backend will process (before tracking)
    <depth_topic>/pending  std_msgs/Header  this node announces the frame it has started computing
When idle, the node starts on the next *requested* frame as soon as it arrives (never on a stale queued one), so
every depth image it produces is for a frame the backend will use, and VINS waits only for frames that are being
computed. If no request arrives for 2 s (older VINS or none running), it falls back to "newest pair".

Pipeline per stereo pair: decode (mono8/8UC1/rgb8/bgr8/mono16) -> OpenCV rectification (CPU) -> TensorRT engine
(GPU, optional CUDA-graph replay) -> raw-grid inverse depth (numpy). Always processes the newest pair and drops older
ones, so latency stays bounded when the network is slower than the camera.

    python3 depth_node_trt.py --ros-args -p engine:=rtigev_i6_512x640_fp16.engine \
        -p cam0_yaml:=cam0_sensor.yaml -p cam1_yaml:=cam1_sensor.yaml \
        -p image0_topic:=/cam0/image_thermal -p image1_topic:=/cam1/image_thermal
Use -p onnx:=model.onnx instead of engine to run with onnxruntime (testing on a machine without TensorRT).
"""
import array
import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from stereo_rig import StereoRig
from stereo_geometry_np import DepthConverterNP


def to_gray(msg):
    enc = msg.encoding
    if enc in ('mono8', '8UC1'):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
    elif enc in ('rgb8', 'bgr8'):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        a = cv2.cvtColor(a.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2GRAY if enc == 'rgb8' else cv2.COLOR_BGR2GRAY)
    elif enc in ('mono16', '16UC1'):
        dt = np.dtype('>u2' if msg.is_bigendian else '<u2')
        a = np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 2)[:, :msg.width]
        a = (a >> 8).astype(np.uint8)  # same scaling as cv_bridge mono16 -> mono8 (what VINS sees)
    else:
        raise ValueError(f'unsupported image encoding {enc}')
    return np.ascontiguousarray(a)


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


class DepthNodeTRT(Node):
    def __init__(self):
        super().__init__('learned_depth_trt')
        dyn = ParameterDescriptor(dynamic_typing=True)
        p = lambda n, v: self.declare_parameter(n, v, dyn).value  # noqa: E731
        engine, onnx = p('engine', ''), p('onnx', '')
        cam0, cam1 = p('cam0_yaml', ''), p('cam1_yaml', '')
        self.max_rate = float(p('max_rate_hz', 0.0))
        self.sync_tol_ns = int(float(p('sync_tol_ms', 3.0)) * 1e6)
        self.follow = bool(p('follow_requests', True))
        self.stats_csv = p('stats_csv', '')
        out_topic = p('depth_topic', '/depth/inverse_depth')
        if not cam0 or not cam1:
            raise SystemExit('cam0_yaml and cam1_yaml (EuRoC/Kalibr sensor.yaml) are required')
        if bool(engine) == bool(onnx):
            raise SystemExit('set exactly one of engine:=<file.engine> or onnx:=<file.onnx>')
        self.conv = DepthConverterNP(StereoRig.from_euroc(cam0, cam1), zmin=float(p('zmin', 0.2)),
                                     zmax=float(p('zmax', 4.0)), sigma_px=float(p('sigma_px', 1.0)),
                                     relative_sigma=float(p('relative_sigma', 0.025)))
        if engine:
            from trt_runner import TrtStereo
            self.model = TrtStereo(engine, cuda_graph=bool(p('cuda_graph', True)), logger=self.get_logger().warn)
            self.model_name = f'{Path(engine).name} (TensorRT, cuda_graph={self.model.graph_exec is not None})'
        else:
            from trt_runner import OnnxStereo
            self.model = OnnxStereo(onnx)
            self.model_name = f'{Path(onnx).name} (onnxruntime)'
        if tuple(self.model.size) != tuple(self.conv.size):
            raise SystemExit(f'network input {self.model.size} != camera size {self.conv.size} (HxW)')
        h, w = self.conv.size
        dummy = np.zeros((h, w), np.float32)
        for _ in range(2):  # warm-up before data arrives
            self.model(dummy, dummy)

        r = self.conv.rect
        self.get_logger().info(f'rectified fx={r["fx"]:.1f} baseline={r["baseline"]*100:.2f} cm '
                               f'-> depth {self.conv.zmin}..{self.conv.zmax} m = disparity '
                               f'{self.conv.fb/self.conv.zmax:.1f}..{self.conv.fb/self.conv.zmin:.1f} px')
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub = self.create_publisher(Image, out_topic, qos)
        self.create_subscription(Image, p('image0_topic', '/cam0/image_raw'), lambda m: self.on_image(0, m), qos)
        self.create_subscription(Image, p('image1_topic', '/cam1/image_raw'), lambda m: self.on_image(1, m), qos)
        self.pub_pending = self.create_publisher(Header, out_topic + '/pending', qos)
        self.create_subscription(Header, out_topic + '/request', self.on_request, qos)
        self.buf = ({}, {})
        self.pairs = {}         # stamp_ns -> (m0, m1, t_rx): complete stereo pairs, newest few
        self.requested = set()  # stamps VINS asked for since the worker became idle
        self.busy = False
        self.last_request = -1e9
        self.n_requests = self.served = 0
        self.latest = None
        self.cv = threading.Condition()
        self.stats = []
        self.received = self.dropped = 0
        self.last_proc = 0.0
        self.running = True
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()
        self.get_logger().info(f'learned depth ready: model={self.model_name} -> {out_topic}')

    def on_image(self, cam, msg):
        t = stamp_ns(msg)
        with self.cv:
            self.buf[cam][t] = (msg, time.monotonic())
            other = self.buf[1 - cam]
            if other:
                t_other = min(other, key=lambda k: abs(k - t))
                if abs(t_other - t) <= self.sync_tol_ns:
                    m0, m1 = (msg, other[t_other][0]) if cam == 0 else (other[t_other][0], msg)
                    self.received += 1
                    self._offer_pair(stamp_ns(m0), m0, m1, time.monotonic())
                    t_cut = max(t, t_other)
                    for b in self.buf:  # discard everything up to this exposure
                        for k in [k for k in b if k <= t_cut]:
                            del b[k]
                    self.cv.notify()
            for b in self.buf:
                while len(b) > 20:
                    del b[min(b)]

    def _following(self):
        return self.follow and time.monotonic() - self.last_request < 2.0

    def _start(self, t, m0, m1, t_rx):  # caller holds self.cv
        if self.latest is not None:
            self.dropped += 1
        self.latest = (t, m0, m1, t_rx)
        self.cv.notify()

    def _offer_pair(self, t, m0, m1, t_rx):  # caller holds self.cv
        self.pairs[t] = (m0, m1, t_rx)
        while len(self.pairs) > 30:
            del self.pairs[min(self.pairs)]
        if not self._following():
            self._start(t, m0, m1, t_rx)  # fallback: newest pair, older ones dropped
        elif not self.busy and self.latest is None and any(abs(r - t) <= 2000 for r in self.requested):
            self._start(t, m0, m1, t_rx)
        else:
            self.dropped += 1

    def on_request(self, msg):
        t = msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec
        with self.cv:
            self.n_requests += 1
            if not self._following():
                self.get_logger().info('VINS depth requests received: computing only requested frames')
            self.last_request = time.monotonic()
            self.requested.add(t)
            while len(self.requested) > 30:
                self.requested.discard(min(self.requested))
            if not self.busy and self.latest is None:
                hit = [k for k in self.pairs if abs(k - t) <= 2000]
                if hit:
                    self._start(hit[0], *self.pairs[hit[0]])

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
                self.busy = True
            pend = Header()
            pend.stamp = m0.header.stamp
            pend.frame_id = m0.header.frame_id
            self.pub_pending.publish(pend)
            self.last_proc = t0 = time.monotonic()
            l, r = self.conv.rectify_pair(to_gray(m0), to_gray(m1))
            t1 = time.monotonic()
            disp = self.model(l, r)
            t2 = time.monotonic()
            depth = self.conv.raw_inverse_depth(disp)
            t3 = time.monotonic()
            out = Image()
            out.header = m0.header
            out.height, out.width = depth.shape[:2]
            out.encoding = '32FC2'
            out.is_bigendian = 0
            out.step = out.width * 8
            out.data = array.array('B', depth.tobytes())  # bytes assignment is ~300x slower in rclpy
            self.pub.publish(out)
            with self.cv:
                self.busy = False
                self.served += 1
                self.requested.clear()  # only requests that arrive from now on are fresh
            valid = float((depth[..., 0] > 0).mean())
            self.stats.append((t, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3,
                               (time.monotonic() - t_rx) * 1e3, valid))
            if len(self.stats) % 50 == 0:
                s = np.array(self.stats[-50:])
                self.get_logger().info(
                    f'depth frames={len(self.stats)} rectify={np.median(s[:, 1]):.1f} net={np.median(s[:, 2]):.1f} '
                    f'convert={np.median(s[:, 3]):.1f} total={np.median(s[:, 4]):.1f}ms '
                    f'latency={np.median(s[:, 5]):.1f}ms valid={s[:, 6].mean():.2f} dropped={self.dropped}/{self.received} '
                    f'requests={self.n_requests} mode={"requested" if self._following() else "newest"}')

    def shutdown(self):
        self.running = False
        if self.stats_csv:
            Path(self.stats_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['stamp_ns', 'rectify_ms', 'net_ms', 'convert_ms', 'compute_ms', 'latency_ms', 'valid_frac'])
                w.writerows(self.stats)
                f.write(f'# received={self.received} dropped={self.dropped} model={self.model_name}\n')


def main():
    rclpy.init()
    node = DepthNodeTRT()
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
