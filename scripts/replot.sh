#!/usr/bin/env bash
# Regenerate trajectory_ate.png for results dirs and export them to the project. Usage: replot.sh "<name>|<title>" ...
D=$(cd "$(dirname "$0")" && pwd)
for spec in "$@"; do
  n=${spec%%|*}; t=${spec#*|}
  ( cd "$HOME/vio_data/results/$n" && "$HOME/venv_bag/bin/python" "$D/plot_traj.py" gt.txt est.txt trajectory_ate.png "$t" | tail -1 )
  bash "$D/export_results.sh" "$n" >/dev/null
done
