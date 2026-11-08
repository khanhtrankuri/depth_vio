Ngày đánh giá: **2026-11-07 (Asia/Bangkok)**. Phạm vi: snapshot workspace hiện tại.

**VERIFIED** = phép kiểm tra/tính toán được chạy trong đợt audit này, có script và artifact.
**REPORTED** = số được lưu/báo cáo trước đây, chưa chạy lại phép đo tương ứng.
**NOT EVALUATED** = chưa chạy hoặc thiếu điều kiện. Tái tính metric trên prediction lưu sẵn
không đồng nghĩa đã chạy lại inference hay xác thực tuyệt đối ground truth.

Nguồn số liệu audit: [audit_results.json](artifacts/audit_results.json),
[input_manifest.json](artifacts/input_manifest.json), [audit_run.log](artifacts/audit_run.log).
Mã production, estimator, weights và dữ liệu gốc được giữ nguyên; mọi file mới ở `reports/`.

# 05 — Research novelty assessment và UATDF

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
