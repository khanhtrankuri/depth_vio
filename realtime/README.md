# Real-time learned depth inside VINS-Fusion (ROS2 Humble)

A learned stereo network runs **live on the GPU** next to VINS-Fusion and feeds metric depth factors into the
sliding-window optimiser. Everything here runs at sensor rate during real-time bag playback (1.0x):
there is no offline cache.

```
 /cam0,/cam1 ──► depth_node (RT-IGEV, CUDA graph, 20 Hz) ──► /depth/inverse_depth (32FC2, raw cam0 grid)
      │                                                               │ exact-exposure match
      └────────► VINS-Fusion ROS2 (patched) ◄── /imu0                 ▼
                 feature tracker ─► sliding window + IMU + reprojection + depth factors ─► /odometry
```

## Results (EuRoC Vicon Room 1, real-time 1.0x playback, RTX 4060 Laptop, WSL2)

ATE position RMSE: SE3 alignment, scale fixed to 1, GT matched within 10 ms, evo. Mean ± std over repeated runs.

| Sequence | VINS-Fusion stereo+IMU | + live learned depth | Change |
|---|---:|---:|---:|
| V1_01_easy | 11.30 ± 0.02 cm | **9.85 ± 0.06 cm** | −12.8% |
| V1_02_medium | 10.77 ± 0.00 cm | **8.70 ± 0.01 cm** | −19.2% |
| V1_03_difficult | 11.20 ± 0.00 cm | **9.56 ± 0.40 cm** | −14.6% |
| **Mean** | 11.09 cm | **9.37 cm** | **−15.5%** |

Depth was better than the baseline on every sequence in every repeat (baseline n=2, depth n=3).
RPE over 1 s: translation improved on all three sequences (4.84→4.74, 5.19→4.47, 4.37→4.03 cm). Rotation
improved on V1_02/V1_03 but got worse on V1_01 (0.70→1.13°/s); V1_01's GT orientation is known to be
unreliable (see reports/02), but take that number at face value.

Real-time budget (measured inside the live runs):

| | median | p95 |
|---|---:|---:|
| Depth node compute (rectify + network + raw-grid conversion) | 34.3 ms | 35.7 ms |
| Depth latency, stereo pair complete → depth published | 37.5 ms | 39.0 ms |
| Depth output rate | 19.9 Hz (= camera rate) | |
| Backend frames that wanted depth but did not get it | 0 | |
| VINS solver time per window, depth on / off (V1_01) | 30.0 / 22.6 ms | 50.4 / 34.4 ms |

Evidence: [results/](results/) (per-run metrics, configs, trajectories, depth timing CSVs, full sweep table).

**Caveat.** The settings were picked with GT on these same three sequences (there are only three Vicon
Room 1 sequences locally); no held-out sequence was used. All runs are on one machine.

## Design choices (and what was measured)

**Depth model: RT-IGEV (SceneFlow weights, 6 GRU iterations).** Compared against the Leica scan projected
into raw cam0 with GT poses, at the same Shi-Tomasi corners VINS tracks (60 frames, V1_02 + V1_03,
[benchmark](results/depth_model_benchmark.json)):

| Model | ms/frame | median rel. error | within 5% |
|---|---:|---:|---:|
| RT-IGEV sceneflow, 4 iters | 32 | 1.99% | 76.2% |
| **RT-IGEV sceneflow, 6 iters** | **39** | **1.99%** | **76.6%** |
| RT-IGEV sceneflow, 12 iters | 58 | 1.96% | 77.2% |
| RT-IGEV kitti, 6 iters | 38 | 2.78% | 72.5% |
| IGEV++ eth3d, 8 iters | 200 | 1.84% | 78.8% |
| IGEV++ middlebury, 8 iters | 197 | 1.79% | 78.7% |

IGEV++ is only slightly more accurate at 5x the cost, so the real-time model wins.

**Making it real-time.**
- The network launches ~1600 small kernels per frame; under WSL2 launch overhead dominated (72 ms). The
  whole forward pass is captured once as a **CUDA graph** and replayed (33 ms). Two host-side tensor
  creations in RT-IGEV's geometry volume were moved onto the GPU so capture is legal (monkeypatched, the
  upstream submodule is untouched).
- rclpy copies `bytes` into `uint8[]` element by element (190 ms for a 2.9 MB image); `array.array` takes
  0.6 ms. That one change took latency from 300 ms to 38 ms.
- The node always processes the newest stereo pair and drops older ones, so latency stays bounded if the
  GPU is ever slower than the camera.

**Estimator integration** ([vins_patch/](vins_patch/)):
- Depth is published on the raw (distorted) cam0 grid as inverse optical Z plus sigma, so VINS samples it at
  its own feature pixels; it is matched to the exact image exposure (±1 µs), never interpolated or reused.
- Factor: predicted optical Z of the landmark in the observing camera vs. measured depth (the existing
  `IgevDepthFactor`), including marginalization. At accepted observations it replaces the correlated
  right-camera reprojection factor.
- Depth factors on one backend frame per second (`depth_period`), weight 16 on the 2.5% / 1 px sigma proxy.
- **Start delay (3 s after initialisation).** Without it, depth factors during take-off right after
  initialisation corrupted rotation on V1_02 (12.90 cm vs 10.77 cm baseline); with it, 9.28 cm.
- **Consistency gate (10%).** A depth factor is added only if the current estimate agrees with the
  measurement within 10%; otherwise the stereo factor is kept. Without it, weight 32 diverged on V1_01
  (kilometres); with it, the same weight gives 9.71 cm.
- Tried and not adopted: left-right consistency check (doubles GPU cost, 94 ms latency, worse on V1_03),
  photometric check (no mean gain), depth range 3 m, 2 Hz or 0.5 Hz depth, weights 1/4/64.
  See [results/sweep_table.txt](results/sweep_table.txt).

## Reproduce (inside WSL Ubuntu 22.04 with ROS2 Humble, Ceres 2.1, CUDA GPU)

```bash
# 1. Python env: PyTorch CUDA + deps (system site packages give rclpy)
python3 -m venv --system-site-packages ~/venv_depth
~/venv_depth/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
~/venv_depth/bin/pip install "numpy<2" opencv-python-headless scipy timm==0.5.4 evo rosbags pyyaml
# 2. Patched VINS-Fusion in a separate workspace (~/ws_vins_depth); needs ~/ws_vins from scripts/03_build_vins_fusion.sh
bash realtime/vins_patch/build_vins_depth.sh
# 3. EuRoC ASL folders -> ROS2 bags (~/vio_data/bags/<seq>_r2) and GT copies
PY=~/venv_depth/bin/python bash realtime/tools/make_bags.sh
# 4. One live run (depth node + VINS + bag at 1.0x) and its ATE/RPE
bash realtime/tools/run_vio.sh V1_02_medium mytest 1 weight=16 period=1.0 delay=3 gate=0.1
bash realtime/tools/run_vio.sh V1_02_medium mybase 0          # baseline
```

Weights: `IGEV-plusplus/pretrained_models/igev_rt/sceneflow.pth` (from the official IGEV++ release).

Live use with a real stereo camera + IMU: `bash realtime/run_live.sh cam0_sensor.yaml cam1_sensor.yaml [vins.yaml]`
(Kalibr/EuRoC `sensor.yaml` calibration; set topics and extrinsics in the VINS yaml; extrinsics and td must be
fixed). Removing `depth_topic` from the yaml gives unmodified VINS-Fusion.

## Files

| Path | What |
|---|---|
| `depth_node/depth_node.py` | ROS2 node: sync, GPU rectification, inference, 32FC2 publishing, timing stats |
| `depth_node/stereo_models.py` | RT-IGEV / IGEV++ wrappers, CUDA-graph replay |
| `depth_node/stereo_geometry.py` | Rectification, rectified disparity → raw-cam0 inverse depth + sigma, confidence checks |
| `vins_patch/apply_ros2_depth_patch.py` | Source patch for VINS-Fusion-ROS2-Humble (fails on upstream drift) |
| `vins_patch/depth_topic_provider.h` | Thread-safe exposure-matched depth buffer (+ `test_depth_topic_provider.cpp`) |
| `config/euroc_stereo_imu_depth.yaml` | Final configuration |
| `tools/` | Bag conversion, model benchmark, run/sweep/evaluate scripts |
