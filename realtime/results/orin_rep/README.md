# Orin NX, thermal bag MotvongQuangLabLowContrast_11_11, real-time 1.0x playback

Baseline (`base_r*`, patched VINS without depth) vs live TensorRT depth (`depth_r*`, depth_node_trt.py + request/pending
sync), 3 runs each. Configs: `*/config.yaml` (td=0, max_cnt=150, show_track=0; depth_period=0.12, weight=16).
`base_r0`/`depth_r0` were run by hand, the others with `tools/repeat_runs.sh`. Logs are not committed.

- `comparison.txt`: closed-loop drift over a common 110 s window (`tools/compare_runs.py`).
  Baseline 2.05 +- 0.01 %, depth 1.89 +- 0.22 % (1.74 / 1.79 / 2.14); not significant with n=3.
- `run_summaries.txt`: depth usage (5.9 Hz, 0 missed), solver time, pose lag (~137 ms with depth), per run.
- `divergence_40-53s.png`: all depth runs leave the baseline in 45-51 s (window time), while the camera faces a near,
  almost textureless surface: depth valid fraction 0.85 -> 0.39, features with depth ~100 -> ~54.
  depth_r2 gains 0.35 m there. Cause under investigation.
