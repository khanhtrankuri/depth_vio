#!/usr/bin/env bash
# Summarise the latest OpenVINS log: initialization, warnings/errors, timing lines, plus running processes.
source "$(dirname "$0")/common.sh"
L=$(ls -t "$LOG_DIR"/openvins_*.log 2>/dev/null | head -1)
echo "log: $L"
sed 's/\x1b\[[0-9;]*m//g' "$L" > /tmp/ov_clean.log
echo "== initialization =="; grep -i -E 'success|initial' /tmp/ov_clean.log | grep -v 'not enough' | head -5
echo "(first init msg / count of 'not enough feats' while waiting:)"; grep -c 'not enough feats' /tmp/ov_clean.log
echo "== processing time lines =="; grep -E '\[TIME\]' /tmp/ov_clean.log | tail -3
echo "== warnings/errors (count) =="; grep -c -i -E 'warn|error' /tmp/ov_clean.log
grep -i -E 'warn|error' /tmp/ov_clean.log | sort | uniq -c | sort -rn | head -5
echo "== last lines =="; tail -8 /tmp/ov_clean.log
echo "== processes =="; pgrep -a -f 'run_subscribe_msckf|rviz2|ros2 bag play' | cut -c1-110
