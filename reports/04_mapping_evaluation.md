Ngày đánh giá: **2026-11-07 (Asia/Bangkok)**. Phạm vi: snapshot workspace hiện tại.

**VERIFIED** = phép kiểm tra/tính toán được chạy trong đợt audit này, có script và artifact.
**REPORTED** = số được lưu/báo cáo trước đây, chưa chạy lại phép đo tương ứng.
**NOT EVALUATED** = chưa chạy hoặc thiếu điều kiện. Tái tính metric trên prediction lưu sẵn
không đồng nghĩa đã chạy lại inference hay xác thực tuyệt đối ground truth.

Nguồn số liệu audit: [audit_results.json](artifacts/audit_results.json),
[input_manifest.json](artifacts/input_manifest.json), [audit_run.log](artifacts/audit_run.log).
Mã production, estimator, weights và dữ liệu gốc được giữ nguyên; mọi file mới ở `reports/`.

# 04 — Depth–VIO–TSDF mapping

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

| Experiment | Status | Evidence và giới hạn |
| --- | --- | --- |
| E1: IGEV++ + GT pose | VERIFIED cached map metrics | Có tsdf_cloud_gt và fusion_gt trên ba sequences. Original GT orientation chưa đủ chắc để gọi upper bound sạch. Không rerun whole-map integration. |
| E2: IGEV++ + OpenVINS | VERIFIED cached map metrics | Có tsdf_cloud_openvins. Pose được SE3-align toàn trajectory với GT trước integration; không phải run online hoàn toàn không GT. |
| E3: SGBM + GT | NOT EVALUATED | Chỉ có SGBM một frame; chưa có sequence cache/maps. |
| E4: SGBM + OpenVINS | NOT EVALUATED | Chưa có maps/cache; không suy kết quả từ E2. |
| E5: parameter sensitivity | NOT EVALUATED | Không có sweep voxel/trunc/depth/sampling/weight/outliers. |

Nhánh `gtrotcorr` có map thật nhưng chỉ diagnostic phụ thuộc OpenVINS. Không dùng nó
để xác nhận independent upper bound. E2 dùng rigid GT alignment để chọn global gauge,
không scale/ICP hoặc nonlinear trajectory correction; điều này có thể hợp lệ ở registration
benchmark nếu công bố rõ. Nhưng vì transform được dùng trước fixed-volume integration,
GT cũng ảnh hưởng clipping volume: không thể quảng bá là toàn bộ method runtime không dùng GT.
Giải pháp: integrate ở frame native/submap rồi apply cùng rigid registration cho output khi scoring.

Số frames thực integrate có thể khác giữa pose sources do start/end và tolerance;
so E1/E2 phải lấy **cùng timestamp subset** trước khi quy chênh lệch thành pose error.

| Sequence | cached | GT frames | VIO frames | diagnostic frames | reported VIO integrate ms |
| --- | --- | --- | --- | --- | --- |
| V1_01_easy | 583 | 574 | 552 | 574 | 41.56 |
| V1_02_medium | 342 | 335 | 314 | 335 | 34.13 |
| V1_03_difficult | 430 | 418 | 393 | 418 | 32.95 |

## Mapping metrics đã chạy lại

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

| Sequence | Output | n 2cm points | mean m | RMSE m | median m | precision@5cm | legacy recall@5cm | fixed ROI recall@5cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1_01_easy | tsdf_cloud | 180419 | 0.0695 | 0.1297 | 0.0291 | 71.51% | 62.40% | 55.42% |
| V1_01_easy | naive_accum | 666497 | 0.0410 | 0.0597 | 0.0302 | 72.24% | 68.58% | 62.82% |
| V1_02_medium | tsdf_cloud | 224810 | 0.0547 | 0.0809 | 0.0344 | 67.31% | 58.18% | 61.34% |
| V1_02_medium | naive_accum | 642382 | 0.0480 | 0.0643 | 0.0361 | 64.76% | 64.99% | 76.44% |
| V1_03_difficult | tsdf_cloud | 266880 | 0.0672 | 0.1000 | 0.0425 | 57.36% | 56.83% | 60.00% |
| V1_03_difficult | naive_accum | 810857 | 0.0557 | 0.0750 | 0.0444 | 55.59% | 56.12% | 79.96% |

Mean/RMSE cao hơn cho TSDF so naive trên cả ba sequence; median có thể tốt hơn một ít.
Không cherry-pick median để kết luận fusion tốt. Naive là random tối đa 3.000 points/frame,
còn TSDF là extracted mesh vertices: sampling/density và observed support khác nhau.
Đây chưa là controlled E2 ablation, nhưng đủ bác bỏ khẳng định “TSDF chắc chắn cải thiện mọi metric”.

### Cùng fixed ROI: unsigned Chamfer mean và accuracy

Chamfer ở đây = 0.5·(mean nearest distance prediction→reference + mean reverse), đơn vị m,
không squared distance. Đây là unweighted point-sampling Chamfer, chưa area-normalized.

| Sequence | Variant | n reference ROI | n prediction ROI | mean accuracy m | recall@5cm | Chamfer m |
| --- | --- | --- | --- | --- | --- | --- |
| V1_01_easy | tsdf_cloud_openvins | 511871 | 179202 | 0.0703 | 55.42% | 0.1921 |
| V1_01_easy | naive_accum_openvins | 511871 | 656187 | 0.0414 | 62.82% | 0.1636 |
| V1_01_easy | tsdf_cloud_gt | 511871 | 203620 | 0.1184 | 27.70% | 0.2444 |
| V1_01_easy | naive_accum_gt | 511871 | 995220 | 0.0893 | 48.66% | 0.2030 |
| V1_01_easy | tsdf_cloud_gtrotcorr | 511871 | 181066 | 0.0679 | 55.24% | 0.1921 |
| V1_01_easy | naive_accum_gtrotcorr | 511871 | 681899 | 0.0373 | 63.00% | 0.1611 |
| V1_02_medium | tsdf_cloud_openvins | 511871 | 221954 | 0.0556 | 61.34% | 0.0909 |
| V1_02_medium | naive_accum_openvins | 511871 | 597973 | 0.0483 | 76.44% | 0.0735 |
| V1_02_medium | tsdf_cloud_gt | 511871 | 218783 | 0.0604 | 53.55% | 0.1005 |
| V1_02_medium | naive_accum_gt | 511871 | 572975 | 0.0459 | 73.50% | 0.0740 |
| V1_02_medium | tsdf_cloud_gtrotcorr | 511871 | 218031 | 0.0495 | 70.82% | 0.0842 |
| V1_02_medium | naive_accum_gtrotcorr | 511871 | 535912 | 0.0352 | 79.36% | 0.0639 |
| V1_03_difficult | tsdf_cloud_openvins | 511871 | 263743 | 0.0681 | 60.00% | 0.0817 |
| V1_03_difficult | naive_accum_openvins | 511871 | 752730 | 0.0561 | 79.96% | 0.0646 |
| V1_03_difficult | tsdf_cloud_gt | 511871 | 266229 | 0.0844 | 51.53% | 0.1024 |
| V1_03_difficult | naive_accum_gt | 511871 | 735286 | 0.0611 | 73.04% | 0.0736 |
| V1_03_difficult | tsdf_cloud_gtrotcorr | 511871 | 258384 | 0.0647 | 70.30% | 0.0783 |
| V1_03_difficult | naive_accum_gtrotcorr | 511871 | 687261 | 0.0461 | 81.66% | 0.0593 |

### Có hơn single-frame depth không?

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
