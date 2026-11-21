#!/usr/bin/env bash
# While OpenVINS runs: confirm it publishes real output (odometry, path, features) and report rates.
source "$(dirname "$0")/common.sh"
NS=${1:-/ov_msckf}
echo "== topics under $NS =="; ros2 topic list -t | grep "$NS" || echo "(none: node not running?)"
for t in odomimu pathimu poseimu points_msckf points_slam points_sim trackhist; do
  if ros2 topic list | grep -qx "$NS/$t"; then
    printf '%-28s ' "$NS/$t"; timeout 6 ros2 topic hz "$NS/$t" 2>/dev/null | grep -m1 'average rate' || echo "no messages in 6s"
  fi
done
echo "== one odometry sample =="; timeout 6 ros2 topic echo --once "$NS/odomimu" 2>/dev/null | head -20 || echo "no odometry received"
