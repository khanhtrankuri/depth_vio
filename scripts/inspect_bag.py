#!/usr/bin/env python3
"""Inspect a ROS1 .bag or ROS2 bag directory for VIO suitability.

Reports per-topic: type, count, rate, image encoding/size, frame_id, header-time monotonicity,
header-vs-bag-time offset; stereo pair sync (cam0 vs cam1 header stamps); IMU presence/rate.
Nothing is estimated or invented: every number comes from the messages read.

Usage: ~/venv_bag/bin/python inspect_bag.py <bag> [--cam0 /cam0/image_thermal --cam1 /cam1/image_thermal --imu /imu]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

IMG = "sensor_msgs/msg/Image"
CIMG = "sensor_msgs/msg/CompressedImage"
IMU = "sensor_msgs/msg/Imu"


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--cam0", default=None)
    ap.add_argument("--cam1", default=None)
    ap.add_argument("--imu", default=None)
    ap.add_argument("--max-msgs", type=int, default=0, help="stop after N msgs per topic (0 = all)")
    a = ap.parse_args()

    path = Path(a.bag)
    stamps = defaultdict(list)       # topic -> header stamps
    bagts = defaultdict(list)        # topic -> bag receive times
    info = {}                        # topic -> dict of first-message facts
    with AnyReader([path]) as r:
        print(f"Bag: {path}   duration: {r.duration*1e-9:.2f}s   messages: {r.message_count}")
        print(f"{'topic':32s} {'type':32s} {'count':>7s}")
        for c in r.connections:
            print(f"{c.topic:32s} {c.msgtype:32s} {c.msgcount:7d}")
        conns = [c for c in r.connections if c.msgtype in (IMG, CIMG, IMU)]
        counts = defaultdict(int)
        for conn, t, raw in r.messages(connections=conns):
            if a.max_msgs and counts[conn.topic] >= a.max_msgs:
                continue
            counts[conn.topic] += 1
            m = r.deserialize(raw, conn.msgtype)
            stamps[conn.topic].append(stamp(m.header))
            bagts[conn.topic].append(t * 1e-9)
            if conn.topic not in info:
                if conn.msgtype == IMG:
                    info[conn.topic] = dict(kind="image", enc=m.encoding, w=m.width, h=m.height,
                                            step=m.step, frame_id=m.header.frame_id,
                                            bigendian=m.is_bigendian)
                elif conn.msgtype == CIMG:
                    info[conn.topic] = dict(kind="compressed", fmt=m.format, frame_id=m.header.frame_id)
                else:
                    info[conn.topic] = dict(kind="imu", frame_id=m.header.frame_id,
                                            has_orientation=bool(m.orientation_covariance[0] != -1))

    print("\n== Sensor topics ==")
    for tp, ts in stamps.items():
        ts = np.array(ts)
        d = np.diff(ts)
        bt = np.array(bagts[tp])
        rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else float("nan")
        mono = bool(np.all(d > 0)) if len(d) else True
        print(f"{tp}: {info[tp]}")
        print(f"   n={len(ts)} rate={rate:.2f}Hz  dt[min/med/max]={d.min()*1e3:.2f}/{np.median(d)*1e3:.2f}/{d.max()*1e3:.2f} ms"
              f"  strictly-increasing={mono}  header-bag offset median={np.median(bt-ts)*1e3:.1f} ms")
        if len(d) and d.max() > 3 * np.median(d):
            print(f"   WARNING: gap of {d.max()*1e3:.1f} ms vs median {np.median(d)*1e3:.1f} ms (dropped frames/samples?)")

    cam0 = a.cam0 or next((t for t in stamps if "cam0" in t), None)
    cam1 = a.cam1 or next((t for t in stamps if "cam1" in t), None)
    if cam0 in stamps and cam1 in stamps:
        s0, s1 = np.array(stamps[cam0]), np.array(stamps[cam1])
        idx = np.clip(np.searchsorted(s1, s0), 1, len(s1) - 1)
        nearest = np.where(np.abs(s1[idx] - s0) < np.abs(s1[idx - 1] - s0), s1[idx], s1[idx - 1])
        dt = (nearest - s0) * 1e3
        print(f"\n== Stereo sync {cam0} vs {cam1} ==")
        print(f"   n0={len(s0)} n1={len(s1)}  dt(cam1-cam0) ms: mean={dt.mean():.3f} std={dt.std():.3f} "
              f"max|dt|={np.abs(dt).max():.3f}")
    else:
        print("\n== Stereo sync == SKIPPED (need two camera topics)")

    imu = a.imu or next((t for t, i in info.items() if i["kind"] == "imu"), None)
    print("\n== IMU ==")
    if imu in stamps:
        print(f"   PRESENT on {imu}: {len(stamps[imu])} samples. -> VIO possible (needs valid calibration).")
    else:
        print("   NO sensor_msgs/Imu topic found. -> Only visual odometry (VO) is possible; do NOT call it VIO.")


if __name__ == "__main__":
    sys.exit(main())
