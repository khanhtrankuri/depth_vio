#!/usr/bin/env python3
"""Convert loop_fusion's vio_loop.csv (t_ns, x, y, z, qw, qx, qy, qz) to TUM (t x y z qx qy qz qw)."""
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f, open(dst, "w") as g:
    g.write("# timestamp(s) tx ty tz qx qy qz qw\n")
    for line in f:
        v = [x for x in line.strip().split(",") if x]
        if len(v) < 8:
            continue
        t = int(v[0]) * 1e-9
        x, y, z, qw, qx, qy, qz = map(float, v[1:8])
        g.write(f"{t:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
