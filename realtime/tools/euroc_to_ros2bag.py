#!/usr/bin/env python3
"""Convert an EuRoC ASL folder (mav0/{cam0,cam1,imu0}) into a ROS2 sqlite3 bag.

Topics match the official VINS-Fusion EuRoC config: /cam0/image_raw, /cam1/image_raw (mono8), /imu0.
Only stereo pairs with identical timestamps are written. Sensor timestamps are kept exactly (ns).

    python euroc_to_ros2bag.py <mav0_dir> <output_bag_dir>
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_HUMBLE)
Image = TS.types['sensor_msgs/msg/Image']
Imu = TS.types['sensor_msgs/msg/Imu']
Header = TS.types['std_msgs/msg/Header']
Time = TS.types['builtin_interfaces/msg/Time']
Vec3 = TS.types['geometry_msgs/msg/Vector3']
Quat = TS.types['geometry_msgs/msg/Quaternion']


def rows(path):
    with open(path) as f:
        return [r for r in csv.reader(f) if r and not r[0].startswith('#')]


def header(ns, frame):
    return Header(stamp=Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000), frame_id=frame)


def main(mav, out):
    mav, out = Path(mav), Path(out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    c0 = {int(r[0]): r[1].strip() for r in rows(mav / 'cam0/data.csv')}
    c1 = {int(r[0]): r[1].strip() for r in rows(mav / 'cam1/data.csv')}
    stamps = sorted(set(c0) & set(c1))
    imu = [(int(r[0]), [float(x) for x in r[1:7]]) for r in rows(mav / 'imu0/data.csv')]
    events = [(t, 'imu', v) for t, v in imu] + [(t, 'img', None) for t in stamps]
    events.sort(key=lambda e: (e[0], e[1] != 'imu'))
    cov = np.zeros(9)
    with Writer(out, version=8) as w:
        cimu = w.add_connection('/imu0', Imu.__msgtype__, typestore=TS)
        cc0 = w.add_connection('/cam0/image_raw', Image.__msgtype__, typestore=TS)
        cc1 = w.add_connection('/cam1/image_raw', Image.__msgtype__, typestore=TS)
        for n, (t, kind, v) in enumerate(events):
            if kind == 'imu':
                msg = Imu(header=header(t, 'imu4'), orientation=Quat(x=0., y=0., z=0., w=1.),
                          orientation_covariance=np.array([-1.] + [0.] * 8),
                          angular_velocity=Vec3(x=v[0], y=v[1], z=v[2]), angular_velocity_covariance=cov,
                          linear_acceleration=Vec3(x=v[3], y=v[4], z=v[5]), linear_acceleration_covariance=cov)
                w.write(cimu, t, TS.serialize_cdr(msg, Imu.__msgtype__))
                continue
            for conn, cam, files in ((cc0, 'cam0', c0), (cc1, 'cam1', c1)):
                img = cv2.imread(str(mav / cam / 'data' / files[t]), cv2.IMREAD_GRAYSCALE)
                h, wd = img.shape
                msg = Image(header=header(t, cam), height=h, width=wd, encoding='mono8', is_bigendian=0,
                            step=wd, data=img.reshape(-1))
                w.write(conn, t, TS.serialize_cdr(msg, Image.__msgtype__))
            if n % 2000 == 0:
                print(f'{n}/{len(events)}', flush=True)
    print(f'wrote {len(stamps)} stereo pairs, {len(imu)} imu samples -> {out}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
