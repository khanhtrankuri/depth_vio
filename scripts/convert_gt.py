#!/usr/bin/env python3
"""EuRoC ASL ground truth (state_groundtruth_estimate0/data.csv) -> `timestamp(s) tx ty tz qx qy qz qw`.
EuRoC columns: ts[ns], px py pz, qw qx qy qz, ... ; only reordered/rescaled, no other change.
Usage: convert_gt.py <data.csv> <out.txt>
"""
import sys

with open(sys.argv[1]) as fi, open(sys.argv[2], "w") as fo:
    fo.write("# timestamp(s) tx ty tz qx qy qz qw\n")
    n = 0
    for line in fi:
        if line.startswith("#") or not line.strip():
            continue
        c = line.strip().split(",")
        ts = int(c[0]) * 1e-9
        px, py, pz, qw, qx, qy, qz = (float(v) for v in c[1:8])
        fo.write(f"{ts:.9f} {px:.9f} {py:.9f} {pz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        n += 1
print(f"wrote {n} GT poses -> {sys.argv[2]}")
