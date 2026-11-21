#!/usr/bin/env bash
# Latency of candidate models (run inside WSL).
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/venv_depth/bin/python}
for m in "$@"; do "$PY" realtime/tools/profile_model.py "$m" 2>&1 | grep -E "ms/frame|Error|error" ; done
