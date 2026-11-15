# Depth weight/frequency experiment

This experiment tunes only the weight and exposure frequency of the depth-v2
factors. The C++ executable, calibration, solver and tracking settings stay fixed.
The original baseline is preserved. This is GT-guided development on V1_01, not
an independent validation set.

The fixed grid is nominal information weights `0.25, 1, 4` and approximately
`0.5, 1, 2 Hz`. A weight w changes sigma by `1/sqrt(w)`, leaving all inverse-depth
values and masks identical. Because Huber loss acts on the normalised residual,
the complete robust objective is not simply multiplied by w for large residuals.

`prepare_depth_sweep.py` first selects cache exposures matching the backend's
existing odd image-index phase. Lower frequencies take every second/fourth one,
with fixed phase zero. This changes new depth observations, not the 10Hz backend
rate. Older observations and marginalisation priors can continue to constrain
the state between new depth exposures.

```powershell
python pipeline/prepare_depth_sweep.py --source outputs/V1_01_easy/igev_vins_depth_v1 --camera-csv data/V1_01_easy/mav0/cam0/data.csv --output outputs/V1_01_easy/depth_sweep_views_v1
python pipeline/sweep_vins_depth.py --configs outputs/V1_01_easy/depth_sweep_views_v1/configs.json --bag outputs/V1_01_easy/stage7_vins_fusion/euroc.bag --work outputs/V1_01_easy/vins_depth_sweep_v1
```

Screening runs each full sequence sequentially at 1x replay with a fresh baseline.
Order is shuffled once with seed 42. The lowest-ATE candidate is then run against
a fresh baseline at 0.5x, matching the earlier experiment's replay rate. Playback
speed is a scheduling control; sensor timestamps are unchanged. It can still
affect a time-limited asynchronous solver, so screening and confirmation scores
must be reported separately.

All evaluated methods share exposure timestamps and unique nearest GT matches
within 10ms. Each gets one SE3 alignment with scale fixed at 1; RPE uses all 1s
and 5s pairs. Every error is verified against evo. Runs with failed Ceres
evaluation, missing tail coverage, fewer than 1430 poses or pose gaps over 151ms
are rejected. RPE tradeoffs are retained even when selecting by ATE.

`progress.json` reports the current job. After an interruption the same command
reuses completed runs after validation; changing the plan/source requires a fresh
work directory. Incomplete trajectory outputs are never silently reused.

After completion:

```powershell
python reports/scripts/report_depth_sweep.py --work outputs/V1_01_easy/vins_depth_sweep_v1
```

The report contains every grid score, the confirmation pair, evidence hashes and
any failures. One replay on the same sequence cannot establish statistical
significance or generalisation. No estimator default is automatically promoted.
