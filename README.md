# VIO on Windows 11: WSL2 + Ubuntu 22.04 + ROS2 Humble + OpenVINS

## Real-time learned depth in VINS-Fusion
A GPU stereo network (RT-IGEV, CUDA-graph replay, 20 Hz, 38 ms latency) feeds live depth factors into
VINS-Fusion ROS2. On EuRoC V1_01/02/03 at real-time playback it lowers mean ATE from 11.09 cm to 9.37 cm
(−15.5%), better on every sequence. Design, measurements and how to reproduce: [realtime/README.md](realtime/README.md).

## Layout
| Where | What |
|---|---|
| WSL distro `Ubuntu-22.04` (user `vio`, passwordless sudo) | ROS2 Humble, OpenVINS workspace `~/ws_ov` (RelWithDebInfo), bag converter venv `~/venv_bag` |
| `~/vio_data/{bags,configs,results,logs}` (inside WSL) | Data and logs. Keep bags on the WSL ext4 disk, not `/mnt/c` (much faster). |
| `scripts/` (this repo) | All helper scripts. The `docker-desktop` distro is never touched. |

## After a reboot
Nothing needs starting by hand: WSL boots on first use. From PowerShell in this folder:

```powershell
.\scripts\vio.ps1 check                 # PASS/FAIL table of the environment
.\scripts\vio.ps1 rviz                  # RViz2 appears on the Windows desktop (WSLg)
.\scripts\vio.ps1 run euroc_mav         # OpenVINS (config name or full path to estimator_config.yaml)
.\scripts\vio.ps1 play ~/vio_data/bags/<bag_dir> 1.0   # play a ROS2 bag
.\scripts\vio.ps1 stop                  # graceful stop (SIGINT, then SIGKILL stragglers)
```
Start order: `rviz` -> `run` -> `play` (each in its own terminal), so no early IMU/camera data is lost.
Inside WSL the same scripts are `scripts/run_rviz.sh`, `run_openvins.sh`, `play_bag.sh`, `check_output.sh`, `stop_all.sh`.
If RViz2 misbehaves on the GPU path, set `VIO_SOFTWARE_GL=1`.

## Bags
1. ROS1 `.bag` -> ROS2: `scripts/convert_bag.sh in.bag` (uses `rosbags`, no ROS1 needed).
2. Inspect before running: `~/venv_bag/bin/python scripts/inspect_bag.py <bag> [--cam0 .. --cam1 .. --imu ..]`.
   Reports encoding, size, rate, timestamp monotonicity, stereo sync and whether an IMU topic exists.
3. Without an IMU topic only stereo **VO** is possible. Do not call it VIO.

## ATE / drift evaluation (EuRoC with ground truth)
```bash
# headless run + record poseimu + ov_eval ATE/RPE (inside WSL)
bash scripts/eval_ate.sh ~/vio_data/bags/V1_01_easy_ros2 euroc_mav <.../mav0/state_groundtruth_estimate0/data.csv> <run_name>
bash scripts/ate_only.sh ~/vio_data/results/<run_name>     # recompute metrics without re-running
~/venv_bag/bin/python scripts/plot_traj.py gt.txt est.txt out.png   # per-axis error + trajectory plot
~/venv_bag/bin/python scripts/diag_orientation.py gt.txt est.txt    # independent ATE + constant body-offset check
```
`error_singlerun` opens matplotlib windows that block the shell; `ate_only.sh` sets `MPLBACKEND=Agg`.
First result: `results/v1_01_stereo_vio/` (V1_01_easy, stereo VIO, position ATE RMSE 6.5 cm).

## VINS-Fusion on ROS2 Humble
Port used: [fanhong-li/VINS-Fusion-ROS2-Humble](https://github.com/fanhong-li/VINS-Fusion-ROS2-Humble) (CPU mode, `GPU_MODE 0`).
It needs Ceres >= 2.1 (`ceres::CUDA` enum), so Ceres 2.1.0 is built privately in `~/opt/ceres-2.1.0` (system Ceres 2.0 untouched).
```bash
bash scripts/03a_build_ceres_2_1.sh && bash scripts/03_build_vins_fusion.sh      # workspace ~/ws_vins
bash scripts/prep_vins_euroc_cfg.sh <run_name>                                   # config copy with output paths
bash scripts/eval_vins.sh <bag_dir> ~/vio_data/configs/vins_euroc/euroc_stereo_imu_config.yaml <gt.csv> <run_name>
bash scripts/vins_report.sh ~/vio_data/results/<run_name>                        # independent checks + plot + log stats
```
Note: the port's own `vio.csv` truncates timestamps to whole seconds, so trajectories are recorded from the `/odometry` topic.

## Jetson thermal stereo bags (`/cam0/image_thermal`, `/cam1/image_thermal`)
Create `~/vio_data/configs/jetson/` with `estimator_config.yaml`, `kalibr_imu_chain.yaml`, `kalibr_imucam_chain.yaml`
(copy the layout of `~/ws_ov/src/open_vins/config/euroc_mav`). Set the topics to the real ones. The camera
intrinsics/extrinsics must come from your own calibration, not guessed values. The 8-bit mono thermal images are
handled as `mono8`; check `inspect_bag.py` output for the true encoding.

## Debugging C++ in VS Code
Built with `-DCMAKE_BUILD_TYPE=RelWithDebInfo` and `compile_commands.json`. Open the WSL folder `~/ws_ov` with the
VS Code "WSL" extension and attach gdb to `run_subscribe_msckf`. For stepping through code use a Debug build:
`colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug --packages-select ov_core ov_init ov_msckf`.

## Rebuild from scratch
`scripts/01_install_ros2_humble.sh` then `scripts/02_build_openvins.sh` (both idempotent).
