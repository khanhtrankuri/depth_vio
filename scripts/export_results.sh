#!/usr/bin/env bash
# Copy a results dir from WSL (~/vio_data/results/<name>) into the project's results/ folder (small files only).
# Usage: export_results.sh <name>
N=${1:?name}
SRC=$HOME/vio_data/results/$N
DST=$(cd "$(dirname "$0")/.." && pwd)/results/$N
mkdir -p "$DST"
for f in est.txt ate_posyaw.txt ate_se3.txt trajectory_ate.png vins.log openvins.log; do [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/"; done
cp "$HOME/vio_data/configs/vins_euroc/euroc_stereo_imu_config.yaml" "$DST/" 2>/dev/null
ls -la "$DST"
echo "--- warnings in log:"; grep -i -m5 warn "$SRC/vins.log" 2>/dev/null
