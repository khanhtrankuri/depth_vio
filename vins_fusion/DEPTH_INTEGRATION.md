# IGEV++ depth inside VINS-Fusion

For the measured 9-setting weight/frequency sweep and a separate confirmation
replay, see [report 10](../reports/10_depth_weight_frequency_sweep.md) and
[DEPTH_SWEEP.md](DEPTH_SWEEP.md). The selected setting improves ATE on V1_01
but does not improve every RPE metric; it is not enabled by default.

This extension adds metric depth factors from inverse-depth observations to the native Ceres sliding
window. It changes pose estimation, not just the final map. Upstream base is
`be55a937a57436548ddfb1bd324bc1e9a9e828e0`; `apply_depth_patch.py` applies the
small integration to that source and fails if expected source locations change.
The baseline image and historical trajectories are preserved.

## Measurement path

1. `pipeline/export_igev_vins_depth.py` reads existing `disp` and `valid` caches,
   raw camera calibration, and stereo rectification. It does not read GT.
2. Each raw cam0 pixel ray is transformed into the rectified camera. Its
   rectified disparity is sampled, and rectified optical Z is converted to raw
   optical Z: `rho_raw = (R_rect_raw * ray_raw).z * disparity / (fx_rect * baseline)`.
   This rotation matters: copying rectified Z onto a raw image is incorrect.
3. Binary files contain width/height and float32 inverse depth + sigma per raw
   pixel. The C++ provider matches the exposure timestamp within 1 microsecond;
   it never propagates depth to another frame. It accepts only valid smooth
   3x3 neighbourhoods (inverse-depth range <8% of the mean).
4. Accepted measurements are attached to `FeaturePerFrame` observations. They
   move with the feature history when the sliding window changes.
5. For a landmark anchored in camera i, its inverse depth and the current poses
   predict optical depth in camera j. The residual is
   `sigma_z = sigma_inverse / observed_inverse_depth_j^2` and
   `r = (predicted_z_j - 1 / observed_inverse_depth_j) / sigma_z`.
   It depends on anchor pose, observing pose, camera extrinsic and landmark
   inverse depth. At the anchor itself it reduces to a unary inverse-depth prior.
   Only currently valid triangulated landmarks are constrained. A metric-Z
   residual keeps camera-plane trial points finite; positive landmark inverse
   depth is bounded in Ceres. This prevents invalid initial landmarks from
   terminating a solver window.
6. The corresponding right-camera reprojection factor is replaced at accepted
   observations. Left temporal reprojection, other stereo observations and IMU
   factors stay active. This avoids adding both constraints for that same stereo
   observation, but does not establish statistical independence of the learned
   predictions. Huber loss limits the influence of inconsistent measurements.
7. Depth terms for marginalized anchor tracks are included in the native prior,
   following VINS's existing oldest-frame marginalization policy.

Depth is restricted to 0.2–4 m. The default inverse-depth sigma is the maximum
of a 5% relative proxy and a 2-pixel disparity-noise proxy, transformed to the
raw camera frame. These values were fixed before evaluation, not selected using
GT. They are not calibrated uncertainty. Camera extrinsics and td=0 must stay
fixed for this implementation; incompatible settings fail explicitly.

## Build and reproduce

Start with the existing baseline Docker image and existing IGEV++ disparity
caches. Run from the repository root using a Python environment with NumPy,
SciPy, OpenCV, PyYAML and matplotlib:

```powershell
$env:EUROC_SEQ = 'V1_01_easy'
python pipeline/export_igev_vins_depth.py --output outputs/V1_01_easy/igev_vins_depth_v1
docker build -f vins_fusion/Dockerfile.depth -t depth-vio-vins-igev:depth-v2 .
python pipeline/benchmark_vins_depth.py --sequence V1_01_easy --depth-dir outputs/V1_01_easy/igev_vins_depth_v1 --bag outputs/V1_01_easy/stage7_vins_fusion/euroc.bag --work outputs/V1_01_easy/vins_depth_benchmark_v2
```

Use new output directories for another run. Exports/runners refuse to overwrite
completed input indexes or trajectories. The benchmark uses the same modified
executable with depth disabled/enabled, runs sequentially at 0.5x playback and
also scores the original saved baseline. It records image ID, source hashes,
configuration snapshots, depth provenance and active factor counts. This is
cached-depth offline execution, not live IGEV++ inference or a real-time claim.

To run only the depth-enabled estimator:

```powershell
python pipeline/run_vins_fusion.py --sequence V1_01_easy --depth-dir outputs/V1_01_easy/igev_vins_depth_v1 --bag outputs/V1_01_easy/stage7_vins_fusion/euroc.bag --work outputs/V1_01_easy/stage7_vins_igev
```

Pass its TUM output explicitly via `pipeline/run_balanced_fusion.py --poses ...`
and a fresh `--work` to map with the depth-assisted poses. The mapping preset's
default trajectory remains the original VINS-Fusion baseline until further
validation; the estimator extension is opt-in through `--depth-dir`.

## Validation and evaluation protocol

The Docker build runs C++ tests for all local pose/extrinsic/landmark Jacobians,
depth-driven translation optimisation, finite camera-plane/behind-camera trial
penalties, invalid inverse depth, same-exposure matching
and boundary rejection. Python tests cover rectified-to-raw depth geometry,
invalid/missing depths, SE3 invariance and scale-error detection in the metrics.

Evaluation uses only timestamps common to every compared method, nearest raw
EuRoC GT within 10 ms with unique associations, and one global SE3 alignment per
method fitted to all common positions. Scale stays fixed at 1. ATE is position
RMSE. RPE is relative SE3 translation/rotation RMSE at 1s and 5s, using all start
times whose end exists within 1ms. `evaluate_vins_depth.py --verify-evo` checks
every error against the installed evo implementation.

V1_01 has an existing reference-orientation caveat documented in the audit. The
same raw reference is used for all methods without method-derived orientation
correction. RPE inherits that reference limitation. Initialisation coverage and
dropouts must be checked alongside errors. A single run cannot establish a
general improvement; no benefit is assumed before measuring it.
