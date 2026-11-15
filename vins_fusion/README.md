# VINS-Fusion integration

For optional IGEV++ depth factors inside the pose optimizer and paired ATE/RPE
evaluation, see [DEPTH_INTEGRATION.md](DEPTH_INTEGRATION.md). The baseline below
remains available and is the default when `--depth-dir` is omitted.

Source: https://github.com/HKUST-Aerial-Robotics/VINS-Fusion
Pinned upstream commit: `be55a937a57436548ddfb1bd324bc1e9a9e828e0`.
The Docker image builds the upstream `vins` target and its dependencies under ROS
Melodic / Ubuntu 18.04, without modifying the estimator. Upstream retains GPL-3.0.

From the repository root, with Docker Linux containers available:

```sh
python pipeline/run_vins_fusion.py --sequence V1_01_easy --build
python pipeline/run_balanced_fusion.py --sequence V1_01_easy
```

The first command needs only Python's standard library on the host. The mapper
still requires its documented NumPy/SciPy/PyTorch environment. Use `--cpu-python`
and `--gpu-python` on the balanced runner if necessary.

Input is the existing EuRoC ASL layout `data/SEQ/mav0/{cam0,cam1,imu0}/`.
The runner converts it to a chronologically ordered ROS1 bag inside Docker,
publishes stereo images and IMU, and records the estimator's native `vio.csv`.
Outputs and logs live in `outputs/SEQ/stage7_vins_fusion/`. Existing trajectory
outputs are refused; move the previous run directory before a fresh run.

`convert_trajectory.py` converts upstream CSV timestamps **nanoseconds -> seconds**
and quaternions **wxyz -> xyzw**, with normalization and strict timestamp checks.
The resulting TUM file describes **T_WI**, the IMU/body pose in native VINS world
coordinates. The mapper applies the existing camera extrinsic and stereo
rectification exactly once. Camera pose topics must not be substituted for this
IMU trajectory. No covariance is invented for the exported poses.

Calibration uses pinhole intrinsics, radial-tangential distortion and `T_BS`
from this sequence's `sensor.yaml` files. Fixed camera–IMU extrinsics and `td=0`
keep the estimator and mapper calibration consistent. Other solver/noise
settings follow the official EuRoC stereo/IMU configuration. Input images are
raw; rectified IGEV++ depth is produced separately by the depth pipeline.

The supplied runner integrates **local stereo-inertial odometry**. Loop closure
and GPS/global fusion are not enabled. Historical maps, reports and OpenVINS
scripts remain historical baselines, including `build_report.py`,
`build_compare.py` and `stage7_viz.py`; they must not be used to label new VINS
outputs. No VINS accuracy/timing claim follows from the old OpenVINS results.

Playback defaults to 0.5x with a 15-second optimisation drain after the bag.
This is an asynchronous ROS run, not a deterministic serial benchmark. Inspect
`estimator.log` and trajectory coverage; use a slower `--rate` or longer
`--drain-seconds` if the machine cannot keep up. The trajectory excludes initial
frames before VINS initialises. Mapping rejects uncovered timestamps and gaps
over 151 ms. Mapping reads the existing disparity cache; running VINS alone does
not generate missing IGEV++ predictions.

The balanced runner defaults to VINS-Fusion and writes to
`outputs/SEQ/stage8_fusion_balanced_vins_fusion/`. Legacy input requires explicit
`--vio-backend openvins`. A custom `--poses` file must follow the same T_WI
contract and the selected backend's timestamp/calibration assumptions.

## Verified integration run (2026-11-12)

On `V1_01_easy`, the official estimator built and ran to completion: 1,446 IMU
poses over 144.5 seconds, initialization delay 1.050000128 seconds, zero lag from
the last image and maximum pose spacing 0.1 seconds. The balanced mapper consumed
578 of 583 cached depth keyframes and exported 337,323 surfels to
`outputs/V1_01_easy/stage8_fusion_balanced_vins_fusion/surfel_uniform/surface_support6_cc30_cloud.ply`.
Five early keyframes had no initialized pose. These are execution/coverage
checks, not an accuracy benchmark against OpenVINS or Leica. V1_02 and V1_03
have not been run through this integration. All 13 pipeline unit tests passed.
