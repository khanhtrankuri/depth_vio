Ngày đánh giá: **2026-11-07 (Asia/Bangkok)**. Phạm vi: snapshot workspace hiện tại.

**VERIFIED** = phép kiểm tra/tính toán được chạy trong đợt audit này, có script và artifact.
**REPORTED** = số được lưu/báo cáo trước đây, chưa chạy lại phép đo tương ứng.
**NOT EVALUATED** = chưa chạy hoặc thiếu điều kiện. Tái tính metric trên prediction lưu sẵn
không đồng nghĩa đã chạy lại inference hay xác thực tuyệt đối ground truth.

Nguồn số liệu audit: [audit_results.json](artifacts/audit_results.json),
[input_manifest.json](artifacts/input_manifest.json), [audit_run.log](artifacts/audit_run.log).
Mã production, estimator, weights và dữ liệu gốc được giữ nguyên; mọi file mới ở `reports/`.

# 07 — Experiment reproducibility

## Đã thực hiện trong đợt audit

| Kiểm tra | Sample/phạm vi | Trạng thái |
| --- | --- | --- |
| Dataset inventory | Tất cả CSV rows/ảnh listed của ba sequences | VERIFIED; integer timestamps, count/missing/gaps |
| Checkpoints | SceneFlow IGEV++ và RT-IGEV | VERIFIED tensor-read/finite/hash; chưa strict instantiate/infer |
| Geometry/reference | Frame 700 mỗi sequence; toàn 3.199.068-point scan | VERIFIED nội bộ: projection masks/depth khớp cached reference |
| Fair depth | 3 frame pairs; original và diagnostic reference | VERIFIED cached predictions, same-mask metrics + coverage/bins/CDF |
| Cached VIO | 2.758 / 1.581 / 1.972 associated poses | VERIFIED ATE/1s RPE/endpoint/timing recomputation |
| Fresh VIO | Ba full bags, existing v2.7 image | VERIFIED execution + trajectory metrics; child-node lifecycle abort được giữ trong log |
| Reconstruction | 6 PLY variants ×3 sequences | VERIFIED metric recomputation, legacy và fixed ROI |
| TSDF numerical controls | Plane Z=2m, 1.815 synthetic GPU voxels | VERIFIED integration arithmetic; không phải real-map accuracy |
| Fresh SGBM | Ba frame-700 pairs, 5 warmups +30 repeats | Xem sgbm_fresh.json/report 02; khác OpenCV version với cached run |
| RT-IGEV/RAFT/IGEV gốc | Không inference | NOT EVALUATED; không bịa accuracy/FPS |

## Environments thực sử dụng

Không cài packages, không sửa environment, không tải dataset/checkpoint/backbone.

1. Audit CPU: `C:/Users/Lenovo/Documents/BHYT_Vinmec/.venv/Scripts/python.exe`, Python 3.13,
   NumPy 2.3.5, SciPy 1.16.3, Matplotlib 3.10.6 và PyYAML; xem environment.json.
2. Checkpoint/TSDF controls: `C:/Users/Lenovo/Documents/meddies-health-ai/.venv-4060/Scripts/python.exe`,
   Python 3.11, PyTorch 2.13.0+cu130, NumPy 2.4.6, CUDA available;
   RTX 4060 Laptop GPU, 8.188MiB reported, driver 591.66. Không import modules dự án khác.
3. VIO/fresh SGBM: existing `openvins:noetic` image ID
   `sha256:8fd3b1cab2fbccaec6246bb6086cca16832a88fae4d5c6a35d141ba5bb1f275b`, v2.7 source;
   SGBM dùng OpenCV 4.2.0 / NumPy 1.17.4, default OpenCV threads được ghi trong artifact.
4. `python` mặc định trên PATH là WindowsApps alias không chạy được; dùng explicit executable.
   Docker daemon cần quyền host đọc/chạy container; audit dùng isolated read-only containers,
   network disabled, datasets mounts readonly và output mounts chỉ dưới reports/.

Model checkpoint hashes và sizes trong checkpoints.json; source/output inputs trong
input_manifest.json. Manifest không hash lại các bag/ZIP nhiều GB hoặc từng ảnh/cache NPZ;
đây là giới hạn provenance, không phải full cryptographic dataset manifest.
PLY và camera/IMU/GT YAML được hash qua inventory. Git/image IDs không thay lockfile dependencies.
Không có original environment lockfile hoặc run manifest gắn checkpoint hash với historical cache.

## Lệnh tái lập audit độc lập (PowerShell, workspace root)

```powershell
& 'C:\Users\Lenovo\Documents\BHYT_Vinmec\.venv\Scripts\python.exe' reports/scripts/audit_cached.py
& 'C:\Users\Lenovo\Documents\meddies-health-ai\.venv-4060\Scripts\python.exe' reports/scripts/inspect_checkpoints.py
& 'C:\Users\Lenovo\Documents\meddies-health-ai\.venv-4060\Scripts\python.exe' reports/scripts/check_tsdf.py
& 'C:\Users\Lenovo\Documents\BHYT_Vinmec\.venv\Scripts\python.exe' reports/scripts/evaluate_reruns.py
& 'C:\Users\Lenovo\Documents\BHYT_Vinmec\.venv\Scripts\python.exe' reports/scripts/write_reports.py
```

Các scripts ghi vào reports/ và sẽ thay audit artifacts/report cùng tên; không ghi outputs/.
`audit_cached.py` phải chạy xong mới render reports. `evaluate_reruns.py` cần fresh raw
trajectory/timing/process artifacts đã có ở reports/artifacts/rerun_*.

## Tái chạy OpenVINS trên dữ liệu có sẵn

Lệnh mẫu cho V1_01, output mới tách biệt. Thay sequence cho V1_02_medium/V1_03_difficult.
Không cần rebuild image hoặc network. Wrapper shell nằm trong reports/scripts, chỉ source ROS
và chạy original launch. `--tmpfs /tmp` cho ROS logs khi filesystem container readonly.

```powershell
$seq = 'V1_01_easy'
$rootPath = (Get-Location).Path
$outPath = Join-Path $rootPath "reports/artifacts/new_$seq"
New-Item -ItemType Directory -Path $outPath -Force | Out-Null
$bagPath = Join-Path $rootPath "vicon_room1/$seq"
$cfgPath = Join-Path $rootPath 'openvins'
$scriptsPath = Join-Path $rootPath 'reports/scripts'
docker run --rm --read-only --network none --tmpfs /tmp:rw,size=512m --env EUROC_SEQ=$seq --mount "type=bind,source=$bagPath,target=/bags,readonly" --mount "type=bind,source=$outPath,target=/out" --mount "type=bind,source=$cfgPath,target=/cfg,readonly" --mount "type=bind,source=$scriptsPath,target=/audit,readonly" --entrypoint /bin/bash openvins:noetic /audit/rerun_openvins.sh
```

Luôn capture stdout/stderr và `$LASTEXITCODE`, **đồng thời parse required-node exit trong log**.
roslaunch trả container exit 0 trong run audit dù child exit -6; không được dùng exit 0 làm
health check duy nhất. First audit-wrapper attempt failed vì `set -u` trước ROS setup;
giữ setup_attempt logs và sửa riêng wrapper, không đổi estimator.
ROS logs chi tiết trong container `/tmp` không được giữ sau --rm; run.log/stdout,
trajectory, timing và landmark bags được giữ. Muốn root-cause đầy đủ cần preserve ROS
per-node logs vào output của run tiếp theo. Không tuyên bố đã xác định thời điểm/cause cleanup.

## Tái lập fresh SGBM

```powershell
$rootPath = (Get-Location).Path
$outPath = Join-Path $rootPath 'reports/artifacts'
docker run --rm --read-only --network none --tmpfs /tmp:rw,size=128m --mount "type=bind,source=$rootPath,target=/workspace,readonly" --mount "type=bind,source=$outPath,target=/out" --entrypoint python3 openvins:noetic /workspace/reports/scripts/benchmark_sgbm.py
```

5 warmups, 30 repeats trên mỗi frame; measured preprocess, SGBM compute, rectify+compute+depth.
Calibration/map setup, disk decode, VIO/TSDF/I/O không nằm trong timed total.
Không lấy CPU container timing này gán cho historical Windows run. OpenCV versions làm
rectified image và disparity khác nhau; không overwrite cached baseline bằng fresh disparity.
Initial benign NumPy NaN-comparison warning được giữ trong initial_run log, đã loại trong
audit script bằng explicit finite replacement trước khi đo lại. Không có production fix.

## Existing production commands: điều kiện và cảnh báo phạm vi

`EUROC_SEQ`, `FRAME` (default 700), `IGEV_CKPT`, `STEP` (5), `VOXEL` (.03), `ZMAX_FUSE` (4)
là environment controls. `stage4_6.py`, `stage8_fusion.py`, `stage9_eval.py` viết vào
outputs/ theo tên cố định; **không chạy lại vào baseline cũ** nếu muốn giữ provenance.
Tạo workspace copy/output override trong một task triển khai rõ ràng trước run mới.
Các cached result trong audit đã giữ nguyên source logic và weights.

RT-IGEV EuRoC: có checkpoint/code nhưng chưa có EuRoC wrapper trong production;
official evaluate_stereo_rt.py chấm SceneFlow/KITTI/etc, không tự trở thành EuRoC evaluator.
Cần tạo independent adapter cho core_rt + same rectification/masks/metrics; pinned deps,
strict load, offline backbone initialization, CUDA synchronized timings như report 02.
Không đưa lệnh SceneFlow evaluator rồi gọi kết quả EuRoC. RAFT-Stereo/IGEV-Stereo cần
source/license/checkpoint hợp lệ có sẵn trước khi thêm adapter; không tải lớn trong audit.

E3/E4 cần SGBM sequence cache, E5 cần isolated config sweep. Independent corrected GT
chưa được lấy/xác minh frame trong local artifacts. Đây là điều kiện còn thiếu, không ghi
như experiments đã hoàn thành.

## Artifact map và limits

| Artifact | Mục đích |
| --- | --- |
| `audit_results.json` | Inventory, geometry residual, native/common depth, ATE/RPE, mapping legacy/fixed ROI |
| `input_manifest.json` | SHA256 production pipeline files, selected cache outputs, checkpoints |
| `checkpoints.json` | Tensor entries, finite checks, hashes; strict network instantiate chưa kiểm |
| `openvins_image.json` | Actual C++ source commit/tag và complete three calibration/config texts |
| `openvins_rerun_metrics.json` | Fresh binary runs + independent metrics + container/child health flags |
| `rerun_*/` | Raw covariance trajectory, timing, landmarks bag, process metadata, run logs |
| `tsdf_controls.json` | Synthetic integration control, không real-scene benchmark |
| `sgbm_fresh.json` | Fresh CPU latency/config/version và differences from cached prediction |
| `figures/` | Original/diagnostic same-image depth+C D F, cached và fresh ATE |

Input data hashes kiểm internal reproducibility, không thay publisher provenance.
Các number VERIFIED có thể vẫn conditional theo validity reference đã nêu; mỗi kết luận
phải giữ sample count, frame/pose reference, mask, range, hardware/version và alignment policy.
