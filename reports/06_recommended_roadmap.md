Ngày đánh giá: **2026-11-07 (Asia/Bangkok)**. Phạm vi: snapshot workspace hiện tại.

**VERIFIED** = phép kiểm tra/tính toán được chạy trong đợt audit này, có script và artifact.
**REPORTED** = số được lưu/báo cáo trước đây, chưa chạy lại phép đo tương ứng.
**NOT EVALUATED** = chưa chạy hoặc thiếu điều kiện. Tái tính metric trên prediction lưu sẵn
không đồng nghĩa đã chạy lại inference hay xác thực tuyệt đối ground truth.

Nguồn số liệu audit: [audit_results.json](artifacts/audit_results.json),
[input_manifest.json](artifacts/input_manifest.json), [audit_run.log](artifacts/audit_run.log).
Mã production, estimator, weights và dữ liệu gốc được giữ nguyên; mọi file mới ở `reports/`.

# 06 — Roadmap và Executive Research Assessment

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
