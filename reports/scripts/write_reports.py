"""Render the seven requested Markdown reports from measured audit artifacts."""
import json
from pathlib import Path
import numpy as np
from audit_cached import SEQS, ROOT, DEST, J, nearest, pose, sensor

D = J(DEST/'artifacts/audit_results.json')
PRE = '''Ngày đánh giá: **2026-11-07 (Asia/Bangkok)**. Phạm vi: snapshot workspace hiện tại.

**VERIFIED** = phép kiểm tra/tính toán được chạy trong đợt audit này, có script và artifact.
**REPORTED** = số được lưu/báo cáo trước đây, chưa chạy lại phép đo tương ứng.
**NOT EVALUATED** = chưa chạy hoặc thiếu điều kiện. Tái tính metric trên prediction lưu sẵn
không đồng nghĩa đã chạy lại inference hay xác thực tuyệt đối ground truth.

Nguồn số liệu audit: [audit_results.json](artifacts/audit_results.json),
[input_manifest.json](artifacts/input_manifest.json), [audit_run.log](artifacts/audit_run.log).
Mã production, estimator, weights và dữ liệu gốc được giữ nguyên; mọi file mới ở `reports/`.

'''


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])+'\n\n'


def write(name, body):
    (DEST/name).write_text(PRE+body, encoding='utf-8')


def f(x, n=4): return f'{x:.{n}f}'
def pct(x): return f'{100*x:.2f}%'


architecture = '''# 01 — Source code và kiến trúc thực tế

## Kết luận

StereoFuse-VIO là **prototype tích hợp offline có đầu ra thực**, gồm pretrained stereo,
VIO độc lập và TSDF. Chưa có bằng chứng về model mới, fine-tuning, uncertainty-aware fusion,
hay tightly coupled depth–inertial estimator. Không thể xác nhận toàn hệ thống đã đúng/chính xác
chỉ từ hình render. Những phần hình học cốt lõi có tính nhất quán nội bộ, nhưng evaluation
còn hạn chế về reference orientation, fairness, registration và vùng quan sát.

## Cấu trúc đã kiểm tra

| Thư mục | Nội dung/thực trạng |
| --- | --- |
| `IGEV-plusplus/` | Source chính thức, `core/`, `core_rt/`, scripts train/eval; không thấy bằng chứng train cục bộ |
| `IGEV-plusplus/pretrained_models/` | IGEV++ SceneFlow/KITTI/Middlebury/ETH3D và RT SceneFlow/KITTI |
| `openvins/` | Dockerfile, launch, build log; source C++ nằm trong Docker image, không nằm ở host |
| `data/{sequence}/mav0/` | Ảnh ASL, IMU, Vicon, batch pose GT, calibration và Leica PLY |
| `vicon_room1/` | Ba rosbag và ba ZIP; cần phân biệt bag với ASL CSV khi tái lập |
| `pipeline/` | 11 Python files, hai HTML templates; inference, diagnostic, fusion, metric, visualization |
| `outputs/{sequence}/` | Disparity/depth/cache, point cloud/mesh, trajectory/covariance/timing, JSON và PNG |
| `report/` | Bốn HTML có sẵn; đây là báo cáo sinh từ artifact, không phải nguồn benchmark độc lập |
| `reports/` | Báo cáo, script và bằng chứng mới của đợt audit |

Không có AGENTS.md được tìm thấy trong workspace. Root không có Git repository;
IGEV++ commit `099dec97989780d151700459c682df1cc2c18e88`, không thấy tracked modification;
`pretrained_models/` là untracked. Docker image local được kiểm tra có OpenVINS
commit `93adc241390d13e99232652cf05cbe18a93c7bea`, tag `v2.7`.
Bằng chứng actual image/config: [openvins_image.json](artifacts/openvins_image.json).

## Sơ đồ từ code

```mermaid
flowchart TD
  ASL[EuRoC ASL cam0/cam1 CSV + PNG] --> SYNC[common.py: ghép timestamp bằng nhau]
  CAL[Sensor YAML cố định] --> RECT[OpenCV stereoRectify + remap]
  SYNC --> RECT
  RECT --> NET[IGEV++ SceneFlow, 16 iterations, FP16]
  RECT --> SGBM[SGBM: frame 700 baseline]
  NET --> LR[Hai forward passes + L-R binary mask]
  LR --> DEP[Z = fx B / disparity]
  LR --> CACHE[FP16 disparity cache, STEP=5]
  SGBM --> EVAL[Single-frame depth evaluation]
  DEP --> EVAL
  GT[ASL batch body pose GT] --> PROJ[Leica projection, 2x2 z-buffer]
  LEICA[Leica scan PLY] --> PROJ
  PROJ --> EVAL
  BAG[EuRoC rosbag: raw stereo + IMU] --> OV[OpenVINS v2.7 in Docker]
  OV --> RAW[Pose IMU + covariance blocks + timing]
  RAW --> TUM[TUM pose file]
  TUM --> ALIGN[SE3 alignment of full trajectory to GT]
  GT --> ALIGN
  ALIGN --> FUS[Custom GPU dense TorchTSDF]
  GT --> FUS
  CACHE --> FUS
  CAL --> FUS
  FUS --> MAP[Mesh + surface point cloud + sampled naive accumulation]
  MAP --> MET[Point-to-point distances against Leica]
  LEICA --> MET
  ASL --> FRONT[Separate OpenCV FAST/KLT visualization]
```

`stage1_3.py:1–5` nói rõ FAST/KLT được reimplement để minh họa, không phải output
front-end OpenVINS. Ngược lại `stage7_viz.py:20–43` đọc landmarks thật từ rosbag và
áp dụng cùng SE3 alignment với trajectory. Các hình features không được xem là
bằng chứng tracking của binary; mesh/trajectory/landmarks có artifact thật, không chỉ mô phỏng.

## IGEV++: từng bước kiểm tra

| Hạng mục | Bằng chứng/kết luận |
| --- | --- |
| Checkpoint | `pipeline/igev.py:12,24–29`: mặc định SceneFlow, có `IGEV_CKPT` override; strict state_dict load, CUDA, eval mode |
| Xác thực file | Đã đọc bằng `torch.load(weights_only=True)` trên CPU, tensor hữu hạn; xem checkpoints.json. Tên/hash không chứng minh nguồn tải chính thức hay chưa từng fine-tune |
| Gray → RGB | `igev.py:35–37`: repeat ba kênh giống nhau; không có color conversion giả |
| Rectification | `common.py:36–55`: `inv(T_B1) @ T_B0`, zero-disparity, alpha=0, remap hai camera |
| Resolution | 752×480, pad thành 768×480 chia hết cho 32, rồi unpad. Không resize nên không cần scale lại disparity |
| Normalization | `core/igev_stereo.py:164–165`: [0,255] → [-1,1], không normalize hai lần |
| Pixel units | `core/igev_stereo.py:158` nhân disparity low-resolution với 4 khi upsample; output đã là pixel full resolution |
| Intrinsics/baseline | `common.py:50–55`: dùng P0/P1 rectified, không dùng fx raw cho ảnh rectified |
| Depth | `stage4_6.py:68–75`: fx·B/d, invalid = NaN, clip >10 m; fusion invalid=0, clip >4 m |
| Confidence | `igev.py:49–57`: nearest-pixel L-R residual <1px, d>0.5; chỉ binary mask, không phải variance/probability calibrated |
| Raw/postprocess | NPY giữ raw dL/dR; depth được masking, không smoothing disparity. Cache mapping quantize disparity sang FP16 |

**VERIFIED**: fx=436.244295647102 px, B=0.1100778421917611 m, fxB=48.0208307333 px·m.
Depth lưu khớp công thức đến sai số float32 dưới 5e-7 m; cả ba reference projections
được tính lại khớp mask và depth đã lưu. Đây là kiểm tra thực thi phép biến đổi theo
calibration đã cung cấp, không phải chứng nhận calibration vật lý tuyệt đối.

Rủi ro: disparity threshold và L-R check không kiểm tra `isfinite` tường minh; SGBM có
range 96 px còn IGEV++ cấu hình max_disp 768; thiếu border mask trong cache mapping
(`stage8_fusion.py:37`) dù single-frame có (`stage4_6.py:31–38`). Với alpha=0 ở ba frame
đang đánh giá, ratio SGBM d>0.5 khớp artifact, nhưng không nên suy rộng cho cấu hình khác.
FP16 có thể gây quantization và max disparity lãng phí compute; chưa đo sensitivity.

## Frame convention và calibration

Ký hiệu T_AB biến điểm B sang A. Body B là IMU I vì imu0/T_BS=I.
Đúng về đại số: `T_WCraw = T_WI T_ICraw`,
`T_WCrect = T_WI T_ICraw diag(R0ᵀ,1)` (`common.py:116–120`).
`R0` biến camera raw sang rectified; bỏ R0ᵀ sẽ dùng sai hướng tia depth.
Integration nhận `T_CW = inv(T_WCrect)` (`stage8_fusion.py:131–135`).
Không phát hiện đảo extrinsic hay bỏ rectification rotation trong đường này.

Leica YAML khai báo T_WR=I; reference code coi scan đã ở world frame, phù hợp metadata
local hiện tại. Với Vicon raw S, cần T_WI=T_WS·inv(T_IS); code hiện đọc batch
`state_groundtruth_estimate0` nên không được nhân thêm extrinsic Vicon một lần nữa.
Thứ tự quaternion GT wxyz được đổi đúng; ROS/TUM xyzw dùng Hamilton world pose,
không được đảo quaternion chỉ vì OpenVINS nội bộ dùng JPL.

**Cấu hình đang có trong image**: stereo hai camera, FEJ=true, RK4, max_clones=11,
max_slam=50; online camera extrinsics/intrinsics/timeoffset=true; IMU intrinsic=false;
static init window 2 s, dynamic init=false; KLT 200 pts/cam, 5×5 grid, histogram=HISTOGRAM,
OpenCV threads=4. Hình minh họa stage3 dùng CLAHE, nên không tương đương front-end thật.
Camera/IMU calibration initial trong image khớp sensor YAML theo các giá trị đã kiểm tra.
Không lưu time-varying online calibration: mapping dùng static YAML, nên chưa biết sai lệch
do online calibration/time offset của estimator. Đây là vấn đề P1 cần đo, chưa có bằng chứng
đủ để kết luận chính nó gây ra toàn bộ lỗi mapping.

## Hoàn thiện, một phần, chưa có

| Mức | Thành phần |
| --- | --- |
| Có implementation + artifacts | Rectification, IGEV++ inference cache, SGBM single-frame, VIO binary runs, SE3 ATE, custom TSDF, exports, HTML |
| Một phần | Reference depth, visibility bằng point splat, pose association, covariance logging, mapping evaluation |
| Chưa có | Temporal gating khi integrate, uncertainty calibration/propagation, weighted TSDF, dynamic rejection, sparse/adaptive volume, online scheduling |
| Chưa benchmark | RT-IGEV/RAFT-Stereo/IGEV-Stereo, E3/E4 mapping SGBM, E5 sweep, single-frame vs map trên common observations |

## Lỗi/hạn chế theo ưu tiên

| Ưu tiên | Vấn đề và vị trí | Ảnh hưởng/hướng sửa đề xuất |
| --- | --- | --- |
| P0 | `stage4_6.py:105–114` dùng native mask riêng | Không suy ranking từ MAE native; dùng intersection + coverage như report 02 |
| P0 | `diag_gt_offset.py:15–24` fit orientation bằng VIO | Không dùng gtrotcorr làm GT độc lập; giữ như diagnostic, lấy reference độc lập |
| P0 | `stage9_eval.py:85–92` bbox riêng của mỗi map | Completeness khó so trực tiếp; common ROI + visibility + frame set |
| P0 | `build_report.py:187–189,246–250` suy nhân quả quá mạnh | Thay kết luận bằng conditional evidence; audit này chưa sửa báo cáo cũ |
| P1 | `stage8_fusion.py:128`, `common.py:104–113` nearest pose | Tên interp gây hiểu nhầm; dùng translation interpolation + SLERP và ghi dt |
| P1 | `stage8_fusion.py:124` fixed world volume + aligned VIO input | GT tham gia gauge trước integration; tích hợp native/local frame rồi register output |
| P1 | `stage8_fusion.py:33,38` cache không fingerprint | Có thể trộn model/iters/calibration; manifest/version cache và fail fast |
| P1 | `stage8_fusion.py:94–100,145–161` unit weights, temporal chỉ đo | Không hạn chế depth bất nhất; thêm robust gating sau baseline đúng |
| P1 | `stage8_fusion.py:109–110` extraction chỉ kiểm hai corner | Không đảm bảo mọi corner liên quan đều observed; test mesh holes/boundary |
| P2 | `stage8_fusion.py:89,153` hardcode 752×480 | Vỡ khi đổi resolution; lấy shape từ depth |
| P2 | `stage4_6.py:28` latency một lần | Không phải benchmark ổn định hay total/L-R time; warmup/repeats/tách stages |

Production chưa được sửa để giữ nguyên baseline. Roadmap và acceptance gates ở report 06.
'''
write('01_architecture_audit.md', architecture)

depth = '''# 02 — Depth, dataset và Leica reference

## Dataset thực có trên đĩa — VERIFIED

'''
depth += table(['Sequence', 'cam0 / cam1', 'stereo pairs', 'IMU', 'GT body', 'Vicon', 'Cache depth'],
 [[s, f"{x['inventory']['cam0']['n']} / {x['inventory']['cam1']['n']}", x['inventory']['stereo_intersection'],
   x['inventory']['imu0']['n'], x['inventory']['state_groundtruth_estimate0']['n'], x['inventory']['vicon0']['n'], x['cache']['n']] for s,x in D.items()])
depth += '''Ảnh 752×480, median ~20 Hz; IMU ~200 Hz, Vicon ~100 Hz, batch body GT ~200 Hz.
Inventory kiểm tra timestamps integer, duplicates/reversed, image files missing và số ảnh trên đĩa;
chi tiết trong JSON. V1_02 có thừa một timestamp cam1, ghép intersection là đúng và cần ghi nhận.
Ba PLY có cùng SHA256 `8836e1dffcdac96d149554865cd9e8df72137c55deeef46bea7f585d79e4291e`,
mỗi file 3.199.068 points: cùng scan phòng Vicon Room 1, không phải ba scan độc lập.
Đường tải local không được ghi trong download manifest; metadata nói Leica Nova MS50 từ sáu vị trí.
Không thể chứng thực nguyên trạng file với checksum publisher chỉ từ repository.

## Leica–Vicon–Camera: kết luận kiểm tra

`pointcloud0/sensor.yaml:8–15` khai báo T_WR=I, nên pointcloud ở frame world của dataset
theo metadata. `stage4_6.py:79–101` thực sự đọc Leica PLY, dùng pose IMU batch và
T_WCrect từ calibration để project. Có z-buffer (`np.minimum.at`) và 2×2 splat,
không phải gán pose GT thành depth GT. Pixel không có reference là NaN và không được chấm điểm.
Scan chỉ là geometric reference: point sampling/splat có thể đưa surface phía sau vào lỗ
của foreground; thiếu meshed visibility, edge erosion, static-region masking và đánh giá
độ nhạy với splat size. `astype(int)` truncate về zero, ảnh hưởng sát border âm.

Đợt audit tái dựng rectification rotation từ relative extrinsic và chiếu lại scan bằng
NumPy/SciPy, dùng intrinsics rectified lưu trong artifact. Kết quả cả ba sequence:
**0 pixel disagreement về mask, max depth difference = 0** với reference gốc đã lưu.
Không có ICP trong evaluation scripts hiện tại. Điều chưa được kiểm chứng độc lập là độ đúng
của orientation/reference registration vật lý và visibility, không phải việc code có chạy projection.

Pose lookup là nearest neighbor, GT tolerance=20 ms, fusion tolerance=30 ms,
không translation interpolation/SLERP. Cần ghi thời gian associate cho mỗi sample;
mật độ nominal 200 Hz không bảo đảm sai lệch <=2.5ms ở mọi gap/end boundary.
Timestamps GT được đọc float64 trong production; độ lượng tử cỡ dưới microsecond là nhỏ
so với nearest-neighbor milliseconds, nhưng nên giữ integer ns.

Tài liệu chính thức cảnh báo orientation **V1_01_easy** và cung cấp reference tính độc lập
từ Vicon/IMU; không xác nhận cùng lỗi cho V1_02/V1_03.
[OpenVINS supported datasets](https://docs.openvins.com/gs-datasets.html).
Code `diag_gt_offset.py` lấy mean(R_VIO⁻¹ R_GT), rồi sửa orientation GT; đây là reference
phụ thuộc phương pháp. Không được dùng nó để tuyên bố sai số tuyệt đối hay chứng minh nguyên nhân.

## Kiểm chứng số liệu đã báo cáo

Coverage disparity frame 700 được tính lại/đối chiếu với artifact; depth range được chuẩn hóa
về [0.2,10] m. SGBM disparity coverage trước clipping khác depth coverage sau clipping.

'''
rows = []
for s,x in D.items():
    summary=J(ROOT/'outputs'/s/'single_frame_summary.json')
    models=x['depth']['references']['original']['models']
    rows.append([s,pct(summary['stage4']['igev_valid_frac']),pct(summary['stage4']['sgbm_valid_frac']),
                 pct(models['IGEV++']['native_image_valid_ratio']),pct(models['SGBM']['native_image_valid_ratio'])])
depth += table(['Sequence', 'IGEV valid disparity', 'SGBM valid disparity', 'IGEV valid depth', 'SGBM valid depth'], rows)
depth += '''V1_01 native-mask MAE **0.223138 m / 0.348345 m** tái lập được, tương ứng 22,3/34,8 cm.
**VERIFIED về phép tính trên cached prediction/reference**, chưa là chứng nhận absolute depth
accuracy vì reference orientation V1_01 có cảnh báo và masks khác nhau. Mỗi sequence chỉ có
**một frame được so depth với Leica trong audit này**, index 700; không phải sequence-wide depth benchmark.

## Protocol công bằng đã chạy

G = finite reference trong [0.2,10] m. V_m = prediction hữu hạn trong cùng range.
Native = G∩V_m. Intersection = G∩V_IGEV∩V_SGBM, giống hệt cho cả hai model.
Coverage = |G∩V_m|/|G|, valid image ratio=|V_m|/(752·480).
MAE=mean|Z−Zref|; RMSE=sqrt(mean((Z−Zref)²)); AbsRel=mean(|Z−Zref|/Zref).
Trên full G, không định nghĩa MAE cho missing prediction: báo số missing và coverage,
không lặng lẽ loại rồi gọi là full-mask score; không impute lỗi zero.

### Reference gốc: intersection mask — VERIFIED có điều kiện

'''
rows=[]
for s,x in D.items():
    for m,a in x['depth']['references']['original']['models'].items():
        z=a['intersection']; rows.append([s,m,z['n'],f(z['MAE_m']),f(z['RMSE_m']),f(z['AbsRel']),pct(z['coverage_gt'])])
depth += table(['Sequence','Model','n common pixels','MAE m','RMSE m','AbsRel','Common/G'],rows)
depth += '### Native mask và full-G missing counts\n\n'
rows=[]
for s,x in D.items():
    for m,a in x['depth']['references']['original']['models'].items():
        z=a['native']; rows.append([s,m,z['n'],f(z['MAE_m']),f(z['RMSE_m']),f(z['AbsRel']),pct(z['coverage_gt']),a['gt_mask_missing_n']])
depth += table(['Sequence','Model','n native∩G','MAE m','RMSE m','AbsRel','Coverage/G','Missing trên G'], rows)
depth += '''IGEV++ có MAE/RMSE/AbsRel thấp hơn SGBM trên common mask của cả ba frame đang xét.
Đây là bằng chứng tốt hơn so native-mask MAE, nhưng không chứng minh generalization trên toàn
sequence/dataset, chưa có repeatability inference, confidence interval theo frame/sequence,
reference độc lập và kiểm tra occlusion. Không bootstrap hàng trăm nghìn pixel tương quan
như các sample độc lập để tạo ý nghĩa thống kê giả.

### Theo khoảng cách, phân phối lỗi

'''
rows=[]
for s,x in D.items():
    for m,a in x['depth']['references']['original']['models'].items():
        for z in a['bins_intersection']:
            rows.append([s,m,f"[{z['lo']},{z['hi']})",z['n'],f(z['MAE_m']) if z['n'] else 'N/A',f(z['RMSE_m']) if z['n'] else 'N/A'])
depth += table(['Sequence','Model','GT depth bin m','n','MAE m','RMSE m'],rows)
depth += '''JSON có q50/q90/q95/q99 và signed mean cho mọi mask/reference; CDF nằm cùng hình comparison.
Không suy kết quả ở range không có sample. Binned coverage dùng cùng denominator full G,
không phải coverage nội bộ từng bin.

### Reference sửa theo VIO — chỉ diagnostic

'''
rows=[]
for s,x in D.items():
    a=x['depth']['references']['vio_corrected_DIAGNOSTIC']['models']
    rows.append([s,a['IGEV++']['intersection']['n'],f(a['IGEV++']['intersection']['MAE_m']),f(a['SGBM']['intersection']['MAE_m'])])
depth+=table(['Sequence','n common','IGEV MAE m','SGBM MAE m'],rows)
for s in SEQS:
    depth+=f'![{s} raw reference](figures/{s}_depth_original.png)\n\n'
depth+='''Các hình tương tự cho reference diagnostic ở `figures/*_depth_vio_corrected_DIAGNOSTIC.png`.
Màu depth chung 0.2–6m dùng để xem; scoring range vẫn 0.2–10m.

## Model benchmarks và latency

| Model | Local code/weights | Trạng thái đợt audit |
| --- | --- | --- |
| IGEV++ | Có; SceneFlow 58.525.223 bytes, 737 tensor entries hữu hạn | Cached depth metrics VERIFIED; fresh inference/latency NOT EVALUATED |
| StereoSGBM | Code OpenCV có, cached disparity có | Cached metrics VERIFIED; fresh CPU latency NOT EVALUATED |
| RT-IGEV | `IGEV-plusplus/pretrained_models/igev_rt/sceneflow.pth`, 16.954.349 bytes, 490 tensor entries hữu hạn | Checkpoint readability VERIFIED; strict model load/inference/accuracy/speed NOT EVALUATED |
| RAFT-Stereo | Không tìm thấy source/checkpoint trong workspace | NOT EVALUATED |
| IGEV-Stereo gốc | Không có source/checkpoint riêng; IGEV++ không được đổi nhãn thành IGEV-Stereo | NOT EVALUATED |

Hashes và tensor checks ở [checkpoints.json](artifacts/checkpoints.json). Chưa có publisher
checksum để xác nhận provenance. Không tải model/dataset, không train hoặc sửa weights.
Windows Python có CUDA PyTorch nhưng thiếu `cv2`, `timm`, `open3d`, `evo` trong môi trường đó;
môi trường NumPy/SciPy/Matplotlib dùng audit riêng. Không ghép binary packages giữa Python 3.11/3.13.
MobileNet extractor gọi timm pretrained=True: strict state_dict hợp lệ vẫn chưa đủ nếu
backbone initialization muốn truy cập mạng/cache. Khi tái lập phải xác nhận cache backbone,
dependency versions và strict loading, giữ network offline để tránh tải ngoài ý muốn.

`stage4_6.py:27–29` có một warmup và synchronization; một timed call bao gồm tensor creation,
H2D, forward và D2H, không chỉ GPU kernel inference. Nó không tính right pass L-R mask,
rectification, I/O và TSDF. `build_report.py:169` nhắc 364ms như lời ghi chú hardcoded,
không có timing log tương ứng. Native JSON hiện tại có timing khác; 364ms và 15ms trong
prompt vẫn **REPORTED** về run đó, không phải latency được audit tái đo.
Cache timing L+R ~0.90–0.93s/pair là REPORTED từ infer_timing JSON, không có warmup tách
riêng/latency distribution đầy đủ, có thể chịu tải hệ thống. Không so 364ms một lượt với
CPU SGBM rồi coi đó là total pipeline speed.

Điều kiện chạy RT-IGEV: môi trường pinned có torch/torchvision/timm/OpenCV phù hợp,
backbone cache offline, loader `core_rt.rt_igev_stereo.IGEVStereo`, hidden_dim=96,
max_disp=192, official default 8 iterations. IGEV++ default hiện tại 16 iterations;
báo rõ từng cấu hình, không gọi là cùng iteration budget. Dùng cùng ảnh rectify,
range, reference/masks; warmup >=5, repeats >=30, CUDA events cho forward và synchronize
cho wall timing; đo riêng preprocess/H2D/forward/D2H/L-R/postprocess/total và peak GPU RAM.
Không có kết quả RT-IGEV giả hay yêu cầu tải checkpoint mới.
'''
write('02_depth_evaluation.md',depth)

vio_report='''# 03 — OpenVINS evaluation

## Version và actual configuration

Dockerfile yêu cầu v2.7 (`openvins/Dockerfile:9`); đã xác nhận trực tiếp source trong local
image tag v2.7, commit `93adc241390d13e99232652cf05cbe18a93c7bea`.
Image ID `sha256:8fd3b1cab2fbccaec6246bb6086cca16832a88fae4d5c6a35d141ba5bb1f275b`.
Build log cũ nhắc image digest khác, nên không dùng build log để khẳng định image hiện tại
bit-identical với historical run. [Actual configuration snapshot](artifacts/openvins_image.json).

Stereo-inertial, hai camera raw images, KLT, FEJ và RK4 propagation/update của OpenVINS,
không nhận depth IGEV vào estimator. Camera–IMU spatial/time calibration online bật;
mapping không đọc calibration online này. Các giá trị noise trong image khớp YAML dataset:
gyro density 1.6968e-4, walk 1.9393e-5; accel density 2e-3, walk 3e-3 (SI như YAML).
Static initialization window=2s; dynamic init tắt; không có ground-truth initialization
trong launch cục bộ. Không sửa estimator hay calibration trong audit.

## Protocol và kết quả cached trajectories — VERIFIED

Đọc raw GT batch CSV và exported IMU pose; nearest association <=10ms, kiểm tra
one-to-one trên tập này. Fit một SE3 bằng Umeyama/SVD trên position của toàn matched
trajectory, **scale=1**. Quaternion chuyển Hamilton đúng thứ tự, áp dụng cùng world rotation
cho orientation. Không Sim3. Phép tính NumPy/SciPy độc lập không gọi evo hoặc production.

'''
vio_report+=table(['Sequence','n','Duration s','Path m','ATE RMSE cm','ATE rot °','max |dt| ms'],
 [[s,x['vio']['n_matched'],f(x['vio']['duration_s'],2),f(x['vio']['path_length_m'],2),
   f(x['vio']['ATE_rmse_m']*100,3),f(x['vio']['ATE_rotation_rmse_deg'],3),f(x['vio']['association_max_ms'],3)] for s,x in D.items()])
vio_report+='''Chênh ATE translation với JSON cũ <2e-15 m. Vì vậy **5,7–6,3 cm được tái lập về
measurement trên trajectory đã lưu**. Không có scale fitting để che sai lệch scale.
ATE rotation V1_01 chịu ảnh hưởng cảnh báo GT orientation; translation ATE không cần
orientation GT cho alignment/scoring. Cần corrected reference độc lập để đánh giá orientation.

RPE mới dùng temporal delta=1s, matching endpoint <=30ms, tất cả starting poses hợp lệ;
E = ΔT_GT⁻¹ ΔT_est. Khác definition với RPE spatial 1m/5m của script cũ.

'''
vio_report+=table(['Sequence','RPE pairs @1s','RPE translation m','RPE rotation °','Relative endpoint m','first pose delay vs GT s','max pose gap s'],
 [[s,x['vio']['rpe']['1s']['n'],f(x['vio']['rpe']['1s']['trans_rmse_m']),f(x['vio']['rpe']['1s']['rot_rmse_deg']),
   f(x['vio']['endpoint_relative_translation_m']),f(x['vio']['first_estimate_delay_s'],3),f(x['vio']['estimate_max_gap_s'],3)] for s,x in D.items()])
vio_report+='''First-pose delay được tính từ timestamp đầu **GT**, không phải đo chính xác thời gian
initialization nội bộ; GT và camera bắt đầu khác nhau. Endpoint metric là relative pose error
giữa hai đầu matched segment, không phải drift chuẩn hóa toàn sequence. Pose gaps 0.10–0.15s
là mất/giảm mẫu output, chưa đủ bằng chứng tracking reset/failure. Số matched không bao trùm
mọi frame camera. Các logs historical V1_01/V1_02 có LibraryUnloadException/exit -6;
không được bỏ qua hoặc quy chắc thành tracking failure khi thiếu log chi tiết thời điểm crash.

### Spatial RPE cũ — REPORTED, chưa chạy lại evo filter-pair protocol

'''
rows=[]
for s in SEQS:
    a=J(ROOT/'outputs'/s/'stage9_eval/vio_metrics.json')
    rows.append([s,f(a['RPE_trans_rmse_m@1m']),f(a['RPE_rot_rmse_deg@1m']),f(a['RPE_trans_rmse_m@5m']),f(a['RPE_rot_rmse_deg@5m'])])
vio_report+=table(['Sequence','RPE trans @1m m','RPE rot @1m °','RPE trans @5m m','RPE rot @5m °'],rows)
vio_report+='## Runtime\n\n'
vio_report+=table(['Sequence','cached estimator mean ms','cached p90 ms'],[[s,f(x['vio']['timing_total_ms_mean'],3),f(x['vio']['timing_total_ms_p90'],3)] for s,x in D.items()])
vio_report+='''Các giá trị này được **VERIFIED bằng tính lại trên timing log**, nhưng clock measurements
vẫn là historical run; không tính rosbag I/O, recording, ROS publication hay toàn pipeline.
CPU utilization và peak RSS historical không có telemetry: NOT EVALUATED.
Không có VINS-Fusion/ORB-SLAM3 source, binaries hoặc trajectories trong workspace để so sánh.

## Covariance thực có và frame của nó

`openvins_traj.txt` có 20 columns: timestamp, position, xyzw, 6 upper-triangle entries
của rotational covariance, 6 của positional covariance. Đã kiểm tra mọi 3×3 block đối xứng
khôi phục được có eigenvalue dương trong các run đã lưu; không đồng nghĩa covariance calibrated.
`openvins_traj_tum.txt` chỉ 8 columns, mất covariance. Cross-cov position–orientation và
cross-time covariance không nằm trong text export.

Publisher v2.7 lấy marginal state `[p_IinG, q_GtoI]`, reorder vào ROS message position rồi
orientation nhưng không tự chuyển error-state thành world Hamilton tangent.
Position perturbation ở global frame; rotational error của JPL q_GtoI gắn với IMU tangent
theo convention OpenVINS. Muốn propagate sang camera/world phải derive Jacobian đúng sign,
left/right perturbation và lever arm, không coi ROS frame_id=global là đủ.
[Publisher source](https://raw.githubusercontent.com/rpng/open_vins/v2.7/ov_msckf/src/ros/ROS1Visualizer.cpp),
[Recorder source](https://raw.githubusercontent.com/rpng/open_vins/v2.7/ov_eval/src/utils/Recorder.h).

Để UATDF đầy đủ: record nguyên PoseWithCovarianceStamped 6×6, state timestamps,
online spatial/time calibration và cross-correlation cần thiết. Covariance thuộc gauge
OpenVINS trước alignment; alignment GT có uncertainty riêng, không được lặng lẽ xem như exact.

## Fresh OpenVINS runs

'''
fresh=DEST/'artifacts/openvins_rerun_metrics.json'
if fresh.exists():
    F=J(fresh)
    vio_report+=table(['Sequence','fresh n','ATE cm','RPE@1s m / °','estimator mean ms','container wall s','container exit / node abort'],
     [[s,a['metrics']['n_matched'],f(a['metrics']['ATE_rmse_m']*100,3),
       f"{a['metrics']['rpe']['1s']['trans_rmse_m']:.4f} / {a['metrics']['rpe']['1s']['rot_rmse_deg']:.4f}",
       f(a['metrics']['timing_total_ms_mean'],3),f(a['process']['wall_s'],2),
       f"{a['process']['exit_code']} / {a['process']['required_node_abort']}"] for s,a in F.items()])
    vio_report+='''**VERIFIED**: binary đã chạy mới trên ba bags có sẵn, default configuration trong image,
original launch, không dùng GT vào estimator, isolated containers --read-only --network none.
Recording artifacts nằm trong `artifacts/rerun_{sequence}/`; fresh metrics tính từ raw export
bằng cùng SE3/1s RPE evaluator độc lập. Một lần chạy/sequence, chưa phải multi-seed variance study.
Wall time bao gồm ROS startup/recording, không thay thế per-frame latency; chưa đo peak RSS/CPU utilization.
**Container exit 0 không bảo đảm node sạch**: roslaunch có thể trả 0 dù required node abort -6.
Đã đọc cả child-process message và LibraryUnloadException trong log; bảng ghi hai trạng thái riêng.
Trajectories mới trùng ATE/n với artifact cũ không loại bỏ lỗi lifecycle này; chưa có log chi tiết
để chứng minh abort chỉ xảy ra ở shutdown sau frame cuối. Cần sửa/kiểm chứng cleanup ở một task riêng.
Lần wrapper đầu lỗi shell `set -u` trước ROS setup; đã sửa wrapper riêng của audit và giữ
`setup_attempt*.log/json`, không chạm estimator. Historical crashes và fresh exit phải báo riêng.
'''
else:
    vio_report+='Fresh runs chưa hoàn tất; xem artifact rerun_* và không dùng số chưa có.\n\n'
for s in SEQS: vio_report+=f'![ATE {s}](figures/{s}_ate.png)\n\n'
write('03_vio_evaluation.md',vio_report)

mapping_report='''# 04 — Depth–VIO–TSDF mapping

## Implementation và numerical controls

Fusion thực dùng **custom dense TorchTSDF GPU**, không phải Open3D ScalableTSDFVolume.
Open3D dùng cho I/O/geometry/visualization. Docstring `stage8_fusion.py:63–68` nói Open3D
trả volume rỗng ở môi trường trước; không có standalone artifact của thử nghiệm đó trong
repository, nên lý do thay backend vẫn REPORTED, không được coi là đã chứng minh ở máy hiện tại.

| Tham số | Implementation hiện tại |
| --- | --- |
| Sampling | STEP=5; cache timestamp median spacing 0.25s, tức ~4 depth keyframes/s |
| Disparity | Hai IGEV forward passes, binary L-R <1px, d>0.5, FP16 cache |
| Depth | fxB/d; >4m thành 0; không zmin explicit khi tạo depth, integrate chỉ xét voxel camera-z>0.2m |
| Pose sync | Nearest timestamp, max 30ms; không interpolation |
| Extrinsic | Static dataset cam0 raw extrinsic + R0ᵀ; không online-calibrated extrinsic |
| Volume bounds | World x/y∈[-4.8,4.8], z∈[-0.6,3.6] m |
| Voxel/truncation | 0.03m / 0.12m; trunc=4·voxel, không sweep độc lập |
| Signed distance | d_obs−z_voxel; skip sdf<=−trunc, clamp positive to +1 |
| Weight | Mỗi observation hợp lệ +1; không cap, uncertainty/pose covariance không dùng |
| Extraction | Marching cubes, min_w=2, kiểm floor/ceil diagonal corners |
| Outliers/occlusion | Binary stereo mask + truncation occlusion phía sau surface; không robust temporal gate |
| Temporal consistency | Có warp depth trước và đo residual, nhưng không dùng residual để accept/reject integration |
| Dynamic objects | Không segmentation/motion masking hoặc dynamic state |

**VERIFIED synthetic control**: đã trích nguyên AST class TorchTSDF từ production và chạy
CUDA với plane Z=2m, không import side effects của pipeline. Kiểm tra signed distance,
behind-band rejection, hai lần integrate cộng weight đúng, depth zero không đổi state;
max TSDF numerical error ~6e-8. [tsdf_controls.json](artifacts/tsdf_controls.json).
Điều này kiểm chứng arithmetic integration trên case kiểm soát, chưa kiểm mesh extraction,
occlusion phức tạp, drifting poses hoặc real-world reconstruction quality.

Volume nominal 320×320×140 nhưng float32 lower bound + `ceil` tạo actual
**321×321×141=14.528.781 voxels**. Ba scalar volumes và XYZ centers FP32 chiếm khoảng
332.5 MiB trước temporaries; integrate còn Pc/projection/index/masks/tensor copies, nên
peak cao hơn nhiều. Đây là ước tính từ allocation, chưa là measured peak GPU memory.
Độ phức tạp mỗi frame O(V), memory O(V); giảm voxel một nửa làm chi phí volume xấp xỉ 8×.
Không thể gọi bộ tích hợp này scalable toàn cục khi volume cố định.

## E1–E5: điều gì đã và chưa chạy

'''
mapping_report+=table(['Experiment','Status','Evidence và giới hạn'],[
 ['E1: IGEV++ + GT pose','VERIFIED cached map metrics','Có tsdf_cloud_gt và fusion_gt trên ba sequences. Original GT orientation chưa đủ chắc để gọi upper bound sạch. Không rerun whole-map integration.'],
 ['E2: IGEV++ + OpenVINS','VERIFIED cached map metrics','Có tsdf_cloud_openvins. Pose được SE3-align toàn trajectory với GT trước integration; không phải run online hoàn toàn không GT.'],
 ['E3: SGBM + GT','NOT EVALUATED','Chỉ có SGBM một frame; chưa có sequence cache/maps.'],
 ['E4: SGBM + OpenVINS','NOT EVALUATED','Chưa có maps/cache; không suy kết quả từ E2.'],
 ['E5: parameter sensitivity','NOT EVALUATED','Không có sweep voxel/trunc/depth/sampling/weight/outliers.']])
mapping_report+='''Nhánh `gtrotcorr` có map thật nhưng chỉ diagnostic phụ thuộc OpenVINS. Không dùng nó
để xác nhận independent upper bound. E2 dùng rigid GT alignment để chọn global gauge,
không scale/ICP hoặc nonlinear trajectory correction; điều này có thể hợp lệ ở registration
benchmark nếu công bố rõ. Nhưng vì transform được dùng trước fixed-volume integration,
GT cũng ảnh hưởng clipping volume: không thể quảng bá là toàn bộ method runtime không dùng GT.
Giải pháp: integrate ở frame native/submap rồi apply cùng rigid registration cho output khi scoring.

Số frames thực integrate có thể khác giữa pose sources do start/end và tolerance;
so E1/E2 phải lấy **cùng timestamp subset** trước khi quy chênh lệch thành pose error.

'''
rows=[]
for s in SEQS:
    fs={t:J(ROOT/'outputs'/s/'stage8_fusion'/f'fusion_{t}.json') for t in ['gt','openvins','gtrotcorr']}
    rows.append([s,D[s]['cache']['n'],fs['gt']['n_frames'],fs['openvins']['n_frames'],fs['gtrotcorr']['n_frames'],f(fs['openvins']['integrate_ms_mean'],2)])
mapping_report+=table(['Sequence','cached','GT frames','VIO frames','diagnostic frames','reported VIO integrate ms'],rows)
mapping_report+='''## Mapping metrics đã chạy lại

Đọc PLY thực, voxel downsample 2cm bằng quy tắc origin/centroid tương đương Open3D,
dùng SciPy KDTree nearest-neighbor. Cả 18 map/accumulation variants được tính lại;
mean accuracy khớp JSON cũ đến floating-point precision. Không ICP.
**Đây là point-to-point surface sampling distance**, không phải exact point-to-surface
distance đến mesh scan. Không đổi tên metric thành surface accuracy tuyệt đối.

Legacy accuracy dùng toàn scan làm target. Legacy completeness chỉ dùng scan trong bbox
riêng của output +0.1m: denominator khác giữa methods, có nguy cơ reward coverage thấp.
Audit bổ sung ROI cố định như volume nominal, crop **cả prediction và reference** vào ROI,
và report completeness cùng denominator. ROI này vẫn chưa là common observable surface:
không phạt/loại đúng mọi vùng camera không nhìn thấy, nên chỉ là kiểm tra nhạy với denominator.

### VIO pose: TSDF vs naive accumulation — VERIFIED có điều kiện registration

'''
rows=[]
for s,x in D.items():
    for k in ['tsdf_cloud_openvins','naive_accum_openvins']:
        a=x['mapping'][k]
        rows.append([s,k.split('_openvins')[0],a['n'],f(a['accuracy_mean_m']),f(a['accuracy_rmse_m']),f(a['accuracy_median_m']),pct(a['accuracy_5cm']),pct(a['legacy_completeness_5cm']),pct(a['fixed_volume_completeness_5cm'])])
mapping_report+=table(['Sequence','Output','n 2cm points','mean m','RMSE m','median m','precision@5cm','legacy recall@5cm','fixed ROI recall@5cm'],rows)
mapping_report+='''Mean/RMSE cao hơn cho TSDF so naive trên cả ba sequence; median có thể tốt hơn một ít.
Không cherry-pick median để kết luận fusion tốt. Naive là random tối đa 3.000 points/frame,
còn TSDF là extracted mesh vertices: sampling/density và observed support khác nhau.
Đây chưa là controlled E2 ablation, nhưng đủ bác bỏ khẳng định “TSDF chắc chắn cải thiện mọi metric”.

### Cùng fixed ROI: unsigned Chamfer mean và accuracy

Chamfer ở đây = 0.5·(mean nearest distance prediction→reference + mean reverse), đơn vị m,
không squared distance. Đây là unweighted point-sampling Chamfer, chưa area-normalized.

'''
rows=[]
for s,x in D.items():
    for k,a in x['mapping'].items():
        rows.append([s,k,a['fixed_volume_reference_n'],a['fixed_volume_prediction_n'],f(a['fixed_volume_accuracy_mean_m']),pct(a['fixed_volume_completeness_5cm']),f(a['chamfer_unsquared_fixed_m'])])
mapping_report+=table(['Sequence','Variant','n reference ROI','n prediction ROI','mean accuracy m','recall@5cm','Chamfer m'],rows)
mapping_report+='''### Có hơn single-frame depth không?

**NOT EVALUATED**: chưa có controlled reconstruction comparison single-frame vs multi-frame
với cùng registered visible reference, depth range và spatial support. Local single-frame PLY
có trong stage6, nhưng pixel MAE không so trực tiếp với global nearest-neighbor mean/median.
Các hình mượt hơn không chứng minh accuracy, completeness hoặc temporal consistency tốt hơn.

## Thiết kế thí nghiệm độc lập tiếp theo

1. Chọn timestamps chung cho E1–E4; reference trajectory/scan registration được frozen trước
   benchmark, V1_01 dùng corrected reference độc lập và vẫn report original reference sensitivity.
2. Mỗi pair chạy IGEV++ và SGBM cùng calibration/range; giữ native mask và common mask variants.
   Integrate local/native frames, logging association dt và từng frame accepted/skipped.
3. Reference meshing hoặc visibility-aware raycast; freeze observed ROI từ union/intersection
   camera frusta không phụ thuộc output quality. Uniformly sample mesh surfaces cùng density.
4. E5: voxel {0.02,0.03,0.05}m; trunc độc lập {0.06,0.12,0.20}m; zmax {3,4,6}m;
   stride {1,2,5,10}; uniform vs robust vs inverse-variance; gating off/on. Bắt đầu one-factor
   sweep trên held-out đoạn ngắn, sau đó xác nhận interactions bằng limited factorial.
5. Report accuracy mean/median/RMSE, recall/precision/F-score @2/5/10cm, observable coverage,
   runtime/peak RAM, temporal error distribution; before/after registration nếu khảo sát ICP.
   Không fit ICP riêng rồi giấu native pose drift. Không dùng test sequence để tune thresholds.

## Sai số đến từ đâu?

| Nguồn | Bằng chứng hiện có | Có quy tỷ lệ đóng góp được không? |
| --- | --- | --- |
| Stereo | Common-mask IGEV tốt hơn SGBM trên ba frame; residual tails có thật | Chưa: một frame/seq, reference uncertainty |
| Pose | ATE ~6cm, orientation residual; maps GT/VIO khác | Chưa: orientation GT và frame subsets khác |
| Calibration/timestamp | Static mapping vs online calibration; nearest poses | Chưa: chưa calibration/time sweep |
| TSDF | Mean/RMSE kém hơn naive; plane-control arithmetic đúng | Chưa: sampling/coverage/extraction khác |

Không có cơ sở để chia final error thành phần trăm stereo/VIO/calibration/TSDF.
Phần cần giải quyết trước về accuracy là **tính hợp lệ protocol/reference**, còn runtime
bottleneck dự kiến là hai-pass IGEV++; xem roadmap cho budget 20FPS.
'''
write('04_mapping_evaluation.md',mapping_report)

research='''# 05 — Research novelty assessment và UATDF

## Đánh giá khoa học

Hiện chưa có scientific contribution mới được chứng minh bằng source hoặc ablation.
Stereo pretrained + OpenVINS + conventional TSDF là system integration đáng nghiên cứu,
nhưng không tự tạo novelty. Tên “Uncertainty-Aware Temporal Depth Fusion” cũng không đủ
để chứng minh ý tưởng mới; literature check đã thấy overlap trực tiếp.

| Công trình gốc | Liên quan trực tiếp | Hệ quả với UATDF |
| --- | --- | --- |
| [Curless & Levoy, SIGGRAPH 1996](https://graphics.stanford.edu/papers/volrange/) | Weighted signed-distance accumulation từ range images đã có từ lâu | Weighted averaging riêng lẻ không mới |
| [RoutedFusion, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Weder_RoutedFusion_Learning_Real-Time_Depth_Map_Fusion_CVPR_2020_paper.html) | Learned routing/fusion xử lý noise/outliers | Neural rejection/fusion cũng có tiền lệ |
| [Rosinol et al., WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Rosinol_Probabilistic_Volumetric_Fusion_for_Dense_Monocular_SLAM_WACV_2023_paper.html) | Propagate uncertainty từ dense SLAM sang volumetric fusion; có EuRoC | Inverse-depth-variance TSDF trên EuRoC không phải claim novelty đủ |
| [Jung et al., ICRA 2025](https://arxiv.org/abs/2409.12051) | Probabilistic stereo/motion depth và VI volumetric occupancy submaps | Stereo+VIO+temporal uncertainty có prior art gần; occupancy khác TSDF không xóa overlap |

Đây là targeted literature review, không phải exhaustive novelty/patent clearance.
Không có bằng chứng đủ để khẳng định UATDF novel. Cần formulate hypothesis cụ thể,
so với uncertainty-aware baselines hiện có và phân biệt bằng failure modes/derivation/metric.

## Covariance có thể dùng hợp lệ không?

Có: raw OpenVINS text giữ hai marginal 3×3 blocks, đã kiểm numerical PSD; TUM không giữ.
Không: text không giữ full 6×6 p–θ cross block, temporal cross-covariance, depth–pose
correlation, extrinsic/time-offset covariance. Tái dựng block diagonal là approximation
phải ghi rõ, không gọi là covariance đầy đủ. Publisher cung cấp 6×6 state marginal;
có thể record message hiện có thay vì sửa estimator. Cần verify orientation perturbation
và transform sang tangent convention của module fusion, xem report 03.

## Disparity/depth uncertainty từ IGEV++

Output hiện tại là disparity và binary L-R mask, không có calibrated variance.
Không coi valid ratio hoặc L-R residual trực tiếp là uncertainty xác suất.
Lộ trình không cần sửa weights: dùng L-R residual, photometric/gradient residual,
occlusion flags, iterative convergence và texture cues làm proxy, rồi calibrate variance
trên validation reference độc lập (range-dependent), evaluate reliability/NLL và
accuracy–coverage curves trên test. Iterative disagreement không phải independent ensemble.
Cost-volume entropy là proxy khác, có thể cần hook source; multimodal/correlated cost
không tự cho variance calibrated. Learned uncertainty head/fine-tuning là task nghiên cứu
riêng; không được giả định đã tồn tại hoặc huấn luyện trong audit này.

## Propagation: đề xuất toán học, chưa implementation

Với f=fx, z=fB/d, vector tia a=[(u−cx)/fx,(v−cy)/fy,1]ᵀ:

```text
∂z/∂d = −fB/d²
σz² ≈ (fB/d²)² σd²                         [f,B xem như cố định]
pC = z a
Jd = −(fB/d²) a
Juv = [[z/fx, 0], [0, z/fy], [0, 0]]
ΣC ≈ Jd σd² Jdᵀ + Juv Σuv Juvᵀ
pW = RWC pC + tWC
```

Chọn pose perturbation **riêng của fusion**: pW' = RWC exp([δθC]×)pC + tWC + δtW,
δξ=[δtW,δθC]. Khi đó Jpose=[I, −RWC[pC]×] và Jdepth=RWC.

```text
ΣW ≈ RWC ΣC RWCᵀ + Jpose Σpose Jposeᵀ
      + cross terms nếu measurement/pose phụ thuộc nhau
```

Trước khi dùng, phải map covariance OpenVINS từ error-state JPL IMU tangent sang δξ này,
bao gồm rotation sign/convention, T_IC lever arm, alignment gauge và covariance cross blocks.
Không áp dụng công thức trên trực tiếp vào bất kỳ ROS 6×6 nào chỉ dựa tên frame.
Depth và VIO dùng cùng stereo images nên có correlation; first-order independence là
approximation, đặc biệt temporal reuse. Calibration/time offset uncertainty có thể thêm
Jacobian tương ứng; σt² đóng góp theo scene motion/Jacobian timestamp.
Validate bằng finite differences trên SE3 và Monte Carlo trước real-scene score.

## Weighting và temporal consistency

TSDF hiện dùng projective signed distance s=d_obs(u,v)−z_voxel.
Nên propagate **variance của chính projective residual**, có image gradient/reprojection
và uncertainty pose, thay vì thay bằng point-normal variance không cùng residual.
Point-normal `nᵀΣWn` chỉ phù hợp nếu chuyển sang signed point-to-plane residual với normal đáng tin.

```text
w_obs = clip(ρ(residual_normalized) / (σs² + ε), 0, w_max)
F_new = (W F + w_obs clip(s/μ, −1, 1)) / (W + w_obs)
W_new = min(W + w_obs, W_cap)
```

ε, μ và σs dùng đơn vị nhất quán; nếu variance tính trên s/μ thì chia σs² cho μ².
Occluded/out-of-image/missing reference không phải inconsistent outlier. Warp previous
depth/map với z-buffer; chỉ gate ở mutual visibility, kiểm normalized residual với combined
variance. Không cộng weight như các observations independent khi cùng depth được warp/reuse
nhiều lần; cap information, keyframe novelty hoặc conservative correlation handling.
Pose covariance chỉ reweights không sửa systematic bias/calibration drift.

Không bắt buộc sửa IGEV++ ở phase đầu: wrapper proxies + calibration đủ làm baseline.
Không bắt buộc sửa OpenVINS estimator: record full pose message/calibration; có thể cần
thêm exporter nếu cần clone cross-covariances. Cần sửa/instrument **fusion module** trong
task phát triển tiếp theo; audit này chưa thay production. Làm gating-only baseline trước.

## Chi phí và real-time

Scalar variance + confidence FP32 tại 752×480 khoảng 2.75MiB/frame; full symmetric 3×3
point covariance (6 floats) khoảng 8.26MiB/frame, không nên lưu cho toàn sequence.
Giữ diagonal/scalar residual variance, compute on-the-fly cho visible voxels/keyframes.
Thêm scalar volume FP32 vào grid hiện tại tốn khoảng 55.4MiB; full covariance/voxel rất đắt.
Pixel warping/gating O(HW), sparse visible integration O(V_visible); dense hiện tại O(V).
Hai-pass deep stereo hiện là bottleneck latency; UATDF không tự làm hệ 20FPS.
Không có đo runtime UATDF, nên real-time feasibility vẫn NOT EVALUATED.

## Ba hướng nghiên cứu được xếp hạng

Xếp hạng dưới đây là judgment định tính, không phải kết quả benchmark.

| Rank / hướng | Novelty tiềm năng | Feasibility | Cost | Khó triển khai | Benefit có thể đo | Rủi ro |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Correlation-aware calibrated stereo–VIO temporal fusion | Trung bình, cần đối chiếu prior art sâu hơn | Trung bình | Thấp–TB nếu scalar residual | Cao: tangents/cross-time/correlation | Accuracy/recall, reliability, lỗi temporal khi reuse | Covariance không calibrated; correlation khó estimate |
| 2. Visibility-aware robust temporal gating, có uncertainty ablation | Thấp–TB; phải có insight cụ thể | Cao nhất | Thấp, O(HW) | Trung bình | Tail error, edge artifacts, precision–recall tradeoff | Gating loại nhầm làm mất coverage; dễ bị xem heuristic integration |
| 3. Budget-aware keyframe/submap fusion với RT stereo | Thấp–TB nếu chỉ engineering; cao hơn nếu policy có nguyên lý | Cao | Tiết kiệm GPU, thêm scheduler | TB–Cao | Latency/energy/accuracy Pareto @20Hz VIO | Chỉ thành optimization report, novelty yếu |

Khuyến nghị thực dụng: triển khai hướng 2 làm baseline sau khi reference/protocol đúng;
chọn hướng 1 làm hypothesis nghiên cứu nếu calibration/correlation cho gain vượt robust
gating và distance-only weighting trên held-out sequences. Hướng 3 cần cho deployment,
nhưng không nên gắn nhãn novel nếu chỉ thay network/backend.

Ablations tối thiểu: uniform; distance-only; L-R gate; temporal gate-only; depth uncertainty-only;
pose uncertainty-only; both independent; conservative correlated variant. Báo quality tại
cùng coverage và cùng runtime budget. Không chọn threshold trên test. EuRoC Room1 dùng chung
scene nên ba sequences chưa chứng minh cross-scene generalization; cần held-out rooms/datasets
trong công việc sau, không tải thêm trong đợt audit.
'''
write('05_research_novelty_assessment.md',research)

roadmap='''# 06 — Roadmap và Executive Research Assessment

## Roadmap theo bằng chứng

| Ưu tiên | Việc tiếp theo | Điều kiện hoàn thành | Chi phí/rủi ro |
| --- | --- | --- | --- |
| P0.1 | Freeze calibration, source/image digest, weights hashes, reference và timestamp frame sets | Immutable experiment manifest, output isolated, không silently reuse cache cũ | Rẻ về compute; cần provenance chính xác |
| P0.2 | Reference orientation V1_01 độc lập; visibility/registration Leica | Có transform chain và residual checks; original/corrected sensitivity, không dùng VIO-fit GT | Reference acquisition/validation; corrected GT có gauge cần kiểm riêng |
| P0.3 | Fair sequence depth benchmark | Native/common/full-G missing report, multiple spaced frames/seq, identical range/preprocess, held-out tuning | Compute stereo và scan projections, phụ thuộc environment |
| P0.4 | Diagnose OpenVINS LibraryUnloadException | Node lifecycle sạch hoặc xác nhận chính xác abort sau completion; exit code và last timestamps được kiểm | Không sửa accuracy logic để giấu crash; ROS cleanup task riêng |
| P1.1 | Pose interpolation và calibration logging | Integer time; linear+SLERP; full pose covariance; online extrinsic/timeoffset exports | Coordinate/tangent bugs nếu làm vội |
| P1.2 | Complete E1–E4 common-frame mapping | Four maps với same support; E2/E4 native-frame integration; scan registration chỉ evaluation | SGBM cache rẻ; deep cache có sẵn nhưng cần fingerprint |
| P1.3 | E5 + extraction validation | Voxel/trunc independent sweep, synthetic thin surfaces/occlusion; meshing support checks | Dense volume memory O(voxel^-3), nhanh OOM |
| P1.4 | RT-IGEV benchmark thực | Strict SceneFlow load, offline backbone cache, warmup/repeats, forward/L-R/total và accuracy | Dependency compatibility; không được suy FPS từ tên RT |
| P2.1 | Robust visibility-aware temporal gating baseline | Gain ở cùng coverage/runtime trên held-out samples | False rejection giảm completeness |
| P2.2 | Calibrated disparity + pose uncertainty fusion | Reliability/NLL, Jacobian finite-difference/MC, cross-cov assumptions công bố; vượt gating-only | Novelty overlap và covariance correlation |
| P2.3 | Sparse/submap backend, real-time scheduler | Measured p95 latency/backlog/memory, map integrity dưới budget | Engineering nhiều, novelty chưa tự có |

Không ước lượng lịch ngày công như một kết quả thực nghiệm; thứ tự trên phụ thuộc acceptance
gates. Chưa có cơ sở chọn novelty bằng cách bỏ qua reference hoặc chạy nhiều model hơn.

## 20FPS: hai mục tiêu phải tách rõ

20 stereo pairs/s cho toàn pipeline nghĩa là budget trung bình 50ms/pair, đồng thời p95,
queue/backlog và pose-to-map delay phải được báo. 20Hz VIO với mapping keyframe chậm hơn
là mục tiêu khác, có thể phù hợp hơn. Current depth stride=5 đã là ~4Hz sampling dữ liệu,
không chứng minh xử lý online 4Hz vì inference wall ~0.9s/pair được báo ở run cũ.

Reported single pass 364ms ~2.75FPS; cần >=7.28× acceleration chỉ riêng lượt đó để đạt 50ms.
Reported L+R ~0.90–0.93s ~1.1FPS, cần ~18× để vừa 50ms trước TSDF/I/O. Các phép chia
này là **budget arithmetic từ REPORTED timing**, không phải benchmark mới.
Reported GPU integrate ~40ms/frame ở V1_01 đã chiếm phần lớn budget GPU. CPU VIO có
thể overlap nhưng GPU stereo/fusion tranh tài nguyên; không cộng hoặc lấy max tùy ý mà
chưa đo concurrency. RT-IGEV chưa đo trên máy này nên không cam kết đạt 20FPS.

Kiến trúc deployment đề xuất: VIO 20Hz + bounded queues, timestamped asynchronous depth
worker (benchmark RT-IGEV trước), keyframe selection theo motion/coverage/budget, visibility-aware
fusion trên sparse voxel blocks/submaps. Nếu bắt buộc dense depth từng frame: cân nhắc resolution/
iterations/disparity range/backend tối ưu chỉ sau khi đo quality trade-off; khi resize phải scale
intrinsics và disparity đúng. Temporal reuse phải visibility-aware, không tạo thông tin độc lập giả.
Inference optimization, lower precision hoặc runtime export là task sau; không có source/weights
production nào bị sửa để làm đẹp số trong audit.

## Q1–Q10: trả lời trực tiếp

| Câu hỏi | Đánh giá dựa trên bằng chứng |
| --- | --- |
| Q1. Đúng toán học/kỹ thuật chưa? | Một phần. Rectification/extrinsic/depth math nhất quán, reprojection tái lập chính xác và TSDF plane arithmetic pass. Chưa đủ xác nhận end-to-end accuracy/online correctness; reference, protocol, lifecycle còn vấn đề. |
| Q2. Calibration/transform/time có lỗi không? | Không phát hiện đảo T_WI/T_WC hoặc bỏ R0ᵀ trong path chính. Có nearest-neighbor poses, static mapping vs online camera/time calibration, reference orientation V1_01 có cảnh báo và gtrotcorr phụ thuộc VIO. Chưa định lượng tác động từng nguồn. |
| Q3. IGEV++ hơn SGBM đáng tin không? | Có bằng chứng có điều kiện trên common masks của ba frame: MAE/RMSE/AbsRel tốt hơn. 22.3/34.8cm là native-mask số tái lập, chưa là independent sequence-wide benchmark. |
| Q4. ATE OpenVINS đo đúng/tái lập không? | Có cho metric-scale translation SE3 của trajectories đã lưu; chạy binary mới ba sequences tái lập ATE/n. Không Sim3. Vẫn có LibraryUnloadException/child abort, GT orientation caveat và chưa repeatability study. |
| Q5. TSDF hơn single-frame không? | Chưa được đánh giá có kiểm soát. So naive multi-frame, TSDF mean/RMSE xấu hơn trong cả ba sequence; một vài median/precision tốt hơn không đủ kết luận chung. |
| Q6. Bottleneck accuracy/runtime? | Accuracy: reference/protocol validity và thiếu controlled ablations là nút thắt kết luận; không thể quy lỗi cho network/VIO bằng tỷ lệ. Runtime dự kiến: two-pass IGEV++, tiếp đến dense GPU TSDF; fresh deep timing chưa chạy. |
| Q7. 20FPS cần gì? | Benchmark RT-IGEV, sparse/visible integration, asynchronous bounded scheduling và measured budget. Phân biệt 20Hz VIO với 20FPS dense mapping. Không hứa FPS khi chưa đo. |
| Q8. Hướng đóng góp mới rõ nhất? | Hypothesis calibrated correlation-aware temporal stereo–VIO fusion; phải vượt strong gating/uncertainty baselines và đối chiếu prior art. Chưa chứng minh novelty. |
| Q9. Nên uncertainty-aware fusion không? | Nên làm theo giai đoạn: reference/protocol → robust gating baseline → calibrated depth uncertainty → full pose propagation/correlation nếu gain đo được. Chưa nên thay toàn estimator/network ngay. |
| Q10. Roadmap ưu tiên? | P0 reference/fairness/lifecycle/provenance; P1 interpolation/logging/E1–E5/RT runtime; P2 robust fusion và novelty-driven ablations. |

## Executive Research Assessment

**Current maturity:** prototype offline có prediction, VIO và mapping artifacts thật;
chưa là validated real-time research system hoặc model mới.

**Critical issues:** GT orientation/registration chưa độc lập đầy đủ, masks và completeness
denominators khác nhau, native-frame mapping chưa tách khỏi GT alignment, temporal consistency
chỉ được đo chứ chưa gate, covariance không được dùng, ROS required node abort dù container exit 0.

**Validated results:** independent cached depth formula/projection/metrics, common-mask
IGEV superiority ở ba frame; metric-scale ATE tái lập và fresh VIO runs; 18 map metric
recomputations; finite SceneFlow state_dicts cho IGEV++/RT; synthetic GPU TSDF controls.
Fresh CPU SGBM benchmark được ghi riêng trong report 02 khi artifact có sẵn.

**Missing experiments:** independent orientation/visibility reference validation, sequence-wide
depth, fresh deep-model timings/RT predictions, E3/E4/E5, controlled single-frame vs map,
full-pose covariance calibration/propagation, process resource telemetry, cross-scene validation.

**Novelty gap:** generic uncertainty weighting/temporal fusion có prior art gần; chưa có
novel method, calibrated uncertainty module hoặc ablation gain trong source.

**Recommended architecture:** native-frame VIO/mapping với explicit frame/time contracts,
full covariance/calibration logging, fingerprinted stereo cache, visibility-aware robust
fusion và sparse bounded submaps; evaluate RT stereo bằng số đo thực trước đổi backend.

**Next implementation steps:** khóa reference/protocol và sửa lifecycle theo task riêng;
sau đó hoàn tất common-frame E1–E5, latency profiling; chỉ phát triển UATDF khi baseline
cho phép chứng minh gain độc lập và không phải artifact của evaluation.
'''
write('06_recommended_roadmap.md',roadmap)

repro='''# 07 — Experiment reproducibility

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
& 'C:\\Users\\Lenovo\\Documents\\BHYT_Vinmec\\.venv\\Scripts\\python.exe' reports/scripts/audit_cached.py
& 'C:\\Users\\Lenovo\\Documents\\meddies-health-ai\\.venv-4060\\Scripts\\python.exe' reports/scripts/inspect_checkpoints.py
& 'C:\\Users\\Lenovo\\Documents\\meddies-health-ai\\.venv-4060\\Scripts\\python.exe' reports/scripts/check_tsdf.py
& 'C:\\Users\\Lenovo\\Documents\\BHYT_Vinmec\\.venv\\Scripts\\python.exe' reports/scripts/evaluate_reruns.py
& 'C:\\Users\\Lenovo\\Documents\\BHYT_Vinmec\\.venv\\Scripts\\python.exe' reports/scripts/write_reports.py
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
'''
write('07_experiment_reproducibility.md',repro)

# Append fresh CPU results without conflating them with historical cached model accuracy.
sgbm_path=DEST/'artifacts/sgbm_fresh.json'
if sgbm_path.exists():
    S=J(sgbm_path)
    section='''\n## Fresh StereoSGBM CPU benchmark — VERIFIED

Đã chạy trong existing offline Docker image, OpenCV 4.2.0, NumPy 1.17.4,
5 warmups +30 repeats/frame, default OpenCV threads ghi trong JSON.
Không cài dependency mới. Compute time gồm SGBM và chuyển fixed-point disparity sang float;
total gồm remap, compute và depth conversion, chưa gồm disk decode/map initialization/VIO/TSDF.

'''
    section+=table(['Sequence','preprocess mean ms','compute mean ms','compute p90 ms','timed total mean ms','diff cached disparity','max rect image diff'],
      [[s,f(a['preprocess_ms_mean'],3),f(a['compute_ms_mean'],3),f(a['compute_ms_p90'],3),f(a['total_rectify_compute_depth_ms_mean'],3),
        pct(a['fraction_different_from_cached']),a['left_rect_max_difference']] for s,a in S['sequences'].items()])
    section+='''Đây là benchmark CPU mới riêng. Rectified intensity/disparity khác cached run do
backend/version; differences gồm invalid/valid transitions, không coi max disparity difference
là stereo accuracy. Accuracy comparison chính ở trên dùng cả hai cached predictions với
cùng cached preprocessing. Fresh SGBM không được lặng lẽ thay vào bảng đó hoặc dùng để
khẳng định bitwise reproducibility. [Fresh artifact](artifacts/sgbm_fresh.json).
'''
    path=DEST/'02_depth_evaluation.md'
    text=path.read_text(encoding='utf-8').replace('Cached metrics VERIFIED; fresh CPU latency NOT EVALUATED',
                                               'Cached metrics VERIFIED; fresh CPU benchmark riêng bên dưới')
    path.write_text(text+section,encoding='utf-8')

index='''# StereoFuse-VIO — audit index

Đọc [Executive Research Assessment và Q1–Q10](06_recommended_roadmap.md) trước.

1. [Architecture/source audit](01_architecture_audit.md)
2. [Depth/dataset/Leica/fair benchmarks](02_depth_evaluation.md)
3. [VIO, covariance, fresh runs](03_vio_evaluation.md)
4. [Mapping và E1–E5](04_mapping_evaluation.md)
5. [Research novelty và UATDF](05_research_novelty_assessment.md)
6. [Roadmap/executive assessment](06_recommended_roadmap.md)
7. [Reproducibility, commands, artifact map](07_experiment_reproducibility.md)

Không có production inference/estimator/weights nào được sửa. VERIFIED là phép đo được
chạy và có artifact, không có nghĩa mọi giả định ground truth đã được xác nhận độc lập.
'''
(DEST/'README.md').write_text(index,encoding='utf-8')
print('Wrote seven reports and index.')
