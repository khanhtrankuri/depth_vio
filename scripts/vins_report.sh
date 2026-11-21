#!/usr/bin/env bash
# Post-run report for a results dir: independent ATE/orientation checks (user GT and OpenVINS reference GT), plot, VINS log stats.
# Usage: vins_report.sh <results_dir> [reference_gt.txt]
R=${1:?results dir}
REF=${2:-$HOME/ws_ov/src/open_vins/ov_data/euroc_mav/V1_01_easy.txt}
D=$(cd "$(dirname "$0")" && pwd)
PY=$HOME/venv_bag/bin/python
cd "$R" || exit 1
echo "== independent check vs provided GT =="; $PY "$D/diag_orientation.py" gt.txt est.txt
echo; echo "== vs OpenVINS reference GT =="; $PY "$D/diag_orientation.py" "$REF" est.txt
echo; echo "== per-axis + plot =="; $PY "$D/plot_traj.py" gt.txt est.txt trajectory_ate.png "$(basename "$R")"
echo; echo "== log stats =="
LOG=vins.log; [ -f "$LOG" ] || LOG=openvins.log
echo "lines: $(wc -l < $LOG)"
for k in 'failure' 'reset' 'restart' 'wait for imu' 'large' 'Warn' 'warn' 'error' 'Error' 'unstable'; do
  c=$(grep -c -i -- "$k" $LOG); [ "$c" -gt 0 ] && echo "'$k': $c"
done
echo "solver costs (ms):"; grep -o 'solver costs: [0-9.]*' $LOG | awk '{s+=$3; n++; if($3>m)m=$3} END{if(n) printf "  mean %.1f  max %.1f  n=%d\n", s/n, m, n}'
