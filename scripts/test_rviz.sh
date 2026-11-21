#!/usr/bin/env bash
# Launch rviz2 for a few seconds to confirm it starts under WSLg; falls back to software GL on failure.
source /opt/ros/humble/setup.bash
run() { timeout "${2:-12}" rviz2 >"$1" 2>&1 & local pid=$!; sleep 8; if kill -0 $pid 2>/dev/null; then echo "ALIVE"; else echo "DIED"; fi; wait $pid 2>/dev/null; }
echo "== GPU (D3D12) =="; r=$(run /tmp/rviz_gpu.log); echo "$r"; tail -5 /tmp/rviz_gpu.log
if [[ "$r" != "ALIVE" ]]; then
  echo "== software GL =="; export LIBGL_ALWAYS_SOFTWARE=1; r=$(run /tmp/rviz_sw.log); echo "$r"; tail -5 /tmp/rviz_sw.log
fi
