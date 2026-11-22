# Real-time learned depth inside VINS-Fusion (ROS2 Humble)

A learned stereo network runs **live on the GPU** next to VINS-Fusion and feeds metric depth factors into the
sliding-window optimiser. Everything here runs at sensor rate during real-time bag playback (1.0x):
there is no offline cache.

[Sơ đồ kiến trúc, mã nguồn và thuật toán tích hợp (tiếng Việt)](RT_IGEV_VINS_INTEGRATION.md)

```
 /cam0,/cam1 ──► depth_node (RT-IGEV, CUDA graph, 20 Hz) ──► /depth/inverse_depth (32FC2, raw cam0 grid)
      │                                                               │ exact-exposure match
      └────────► VINS-Fusion ROS2 (patched) ◄── /imu0                 ▼
                 feature tracker ─► sliding window + IMU + reprojection + depth factors ─► /odometry
```

## Results (EuRoC Vicon Room 1, real-time 1.0x playback, RTX 4060 Laptop, WSL2)

ATE position RMSE: SE3 alignment, scale fixed to 1, GT matched within 10 ms, evo. Mean ± std over repeated runs
(baselines n=2, depth n=3). All columns use the same binary (including the initialisation fix below).

| Sequence | VINS-Fusion stereo+IMU (150 features) | VINS-Fusion, 200 features | **+ live learned depth** (200 features) | vs. 150 / vs. 200 |
|---|---:|---:|---:|---:|
| V1_01_easy | 11.32 ± 0.00 cm | 10.27 ± 0.00 cm | **7.51 ± 0.00 cm** | −34% / −27% |
| V1_02_medium | 10.77 ± 0.00 cm | 8.82 ± 0.00 cm | **5.64 ± 0.14 cm** | −48% / −36% |
| V1_03_difficult | 11.20 ± 0.00 cm | 11.01 ± 0.02 cm | **7.77 ± 0.03 cm** | −31% / −29% |
| **Mean** | 11.10 cm | 10.03 cm | **6.97 cm** | **−37% / −30%** |

The first published version (commit 94f9124: depth on 1 frame/s, 150 features, 3 s start delay) reached 9.37 cm.
RPE over 1 s, baseline (150) → depth: translation 4.84→4.74, 5.19→4.46, 4.37→3.93 cm (better on all three);
rotation 0.69→0.65, 0.74→0.81, 1.52→1.72 °/s (better on V1_01, worse on V1_02/V1_03).

Real-time budget (measured inside the live depth runs, all three sequences):

| | median | p95 |
|---|---:|---:|
| Depth node compute (rectify + network + raw-grid conversion) | 36.7–37.4 ms | 39.3–40.8 ms |
| Depth latency, stereo pair complete → depth published | 39.9–40.6 ms | 43.1–53.8 ms |
| Depth output rate | 20.0 Hz (= camera rate) | |
| Backend frames that wanted depth but did not get it | 1 in 9 runs | |
| VINS solver time per 10 Hz window (V1_03 / V1_02 / V1_01) | 22.7 / 45.8 / 54.7 ms | 56.5 / 59.5 / 68.8 ms |
| Pose output lag above its minimum (backlog check) | 44–75 ms | 84–123 ms |

The backend keeps up with the 10 Hz window rate (no backlog growth; worst single window 157 ms on V1_01), but
with 200 features and 5 Hz depth it uses most of the 100 ms budget on V1_01.

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
- Depth factors on one backend frame every 0.2 s (`depth_period`, 5 Hz), weight 16 on the 2.5% / 1 px sigma
  proxy; feature tracker at 200 features / 25 px spacing (upstream 150 / 30).
- **Consistency gate (10%).** A depth factor is added only if the current estimate agrees with the
  measurement within 10%; otherwise the stereo factor is kept.
- **Take-off guard.** Depth factors start only after the initialised estimator has moved 0.5 m
  (`depth_start_dist`), then 1 s more (`depth_start_delay`). Depth during take-off locks in the rotation error
  VINS has at that moment (the baseline itself is ~7° off in that second of V1_02): with dense depth V1_02 went to
  12–30 cm with only a time delay of 3 s, and to 5.4–5.9 cm with the motion guard. A motion threshold does not
  depend on how long the platform sits still before it moves.
- **Initialisation fix (VINS-Fusion bug, also upstream)** ([patch_vins_init.py](vins_patch/patch_vins_init.py)).
  The stereo+IMU initialisation copied window poses `Rs[i]` onto *every* entry of `all_image_frame`, which while
  the platform is static holds more frames than the window: poses go to the wrong frames and `Rs[]` is read past
  its end. The gyro bias was then random and occasionally exploded ("numerical unstable in preintegration",
  kilometre-scale divergence in 2 of ~45 V1_01 runs, with and without depth). It is now solved from the window
  itself; on V1_01 it gives (−0.002, 0.020, 0.078) rad/s, matching the dataset's bias, and an implausible
  correction (> 0.1 rad/s) is rejected and retried. One divergence previously blamed on depth weight 32 was this bug.
- Tried and not adopted (see [results/sweep_table.txt](results/sweep_table.txt); rows `s1_`–`s8_` and
  `final_` were run before the initialisation fix, `s9_` onward after it):
  - **Loop closure** (`loop_fusion`, now buildable and runnable with remaps, see
    [patch_loop_fusion.py](vins_patch/patch_loop_fusion.py)): V1_02 8.70 → 11.4 cm for the real-time
    loop-corrected pose, 9.4 cm for the final smoothed path. The sequences are too short for loops to pay off.
  - Longer solver budget (0.08 s, 12 iterations): no gain. Keeping the right-camera factor next to depth
    (`depth_keep_stereo: 1`): worse. 250 features: same mean, more solver time.
  - Left-right consistency check (doubles GPU cost), photometric check, depth range 3 m, weights 1/4/64.

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
bash realtime/tools/run_vio.sh V1_02_medium mytest 1 weight=16 gate=0.1 period=0.2 maxcnt=200 mindist=25 delay=1 dist=0.5
bash realtime/tools/run_vio.sh V1_02_medium mybase 0 maxcnt=200 mindist=25   # baseline (drop maxcnt/mindist for 150)
# add loop=1 to also run loop_fusion (metrics.json = loop-corrected pose, metrics_vio.json = VIO)
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
| `vins_patch/patch_vins_init.py` | Stereo+IMU initialisation fix (gyro bias from the window) |
| `vins_patch/patch_loop_fusion.py` | loop_fusion fixes: ROS remap arguments, exit crash, optional skip params |
| `config/euroc_stereo_imu_depth.yaml` | Final configuration |
| `tools/` | Bag conversion, model benchmark, run/sweep/evaluate scripts |
