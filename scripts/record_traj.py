#!/usr/bin/env python3
"""Record a pose topic to a text file: `timestamp tx ty tz qx qy qz qw`.
Values are written exactly as published; nothing is modified.
Supports geometry_msgs/PoseWithCovarianceStamped (OpenVINS poseimu) and nav_msgs/Odometry (VINS-Fusion odometry).
Usage: record_traj.py <out.txt> [topic=/ov_msckf/poseimu] [type=pose|odom]
"""
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class Rec(Node):
    def __init__(self, path, topic, kind):
        super().__init__("traj_recorder")
        self.f = open(path, "w")
        self.f.write("# timestamp(s) tx ty tz qx qy qz qw\n")
        self.recv = open(path + ".recv", "w")  # sidecar: header stamp, wall-clock receive time (latency check)
        self.n = 0
        qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE)
        msg_t = Odometry if kind == "odom" else PoseWithCovarianceStamped
        self.create_subscription(msg_t, topic, self.cb, qos)

    def cb(self, m):
        p, q, h = m.pose.pose.position, m.pose.pose.orientation, m.header.stamp
        self.f.write(f"{h.sec + h.nanosec * 1e-9:.9f} {p.x:.9f} {p.y:.9f} {p.z:.9f} {q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n")
        self.recv.write(f"{h.sec + h.nanosec * 1e-9:.9f} {time.time():.6f}\n")
        self.n += 1

    def close(self):
        self.f.close()
        self.recv.close()
        print(f"recorded {self.n} poses")


def main():
    rclpy.init()
    node = Rec(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/ov_msckf/poseimu",
               sys.argv[3] if len(sys.argv) > 3 else "pose")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
