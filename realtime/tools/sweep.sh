#!/usr/bin/env bash
# Run a list of configurations on several sequences, sequentially (inside WSL).
# Usage: sweep.sh <tag> <reps> "<seq list>" "<name>|<depth>|<k=v k=v ...>" ...
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${1:?tag}; REPS=${2:?reps}; SEQS=${3:?seqs}; shift 3
for cfg in "$@"; do
  IFS='|' read -r name depth kv <<< "$cfg"
  for rep in $(seq "$REPS"); do
    for s in $SEQS; do
      run="${TAG}_${name}_r${rep}"
      [ -f "$HOME/vio_data/rt/runs/$s/$run/metrics.json" ] && continue
      echo "=== $s $run ($kv)"
      # shellcheck disable=SC2086
      bash "$HERE/run_vio.sh" "$s" "$run" "$depth" $kv 2>&1 | tail -1
    done
  done
done
