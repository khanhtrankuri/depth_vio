# StereoFuse-VIO — audit index

[Latest validated depth-assisted run, metrics and point cloud](latest_run.md).

Các báo cáo cũ được giữ làm tài liệu lịch sử; thư mục output/HTML của các lượt cũ đã được dọn.

Các báo cáo 01–08 lưu kết quả lịch sử với OpenVINS. Pipeline mặc định đã chuyển
sang [VINS-Fusion](../vins_fusion/README.md); các metric cũ không phải kết quả
VINS-Fusion.

Kết quả tích hợp depth vào bộ tối ưu và so sánh ATE/RPE:
[09 — IGEV++ depth inside VINS-Fusion](09_vins_depth_integration.md).

Khảo sát 9 tổ hợp trọng số/tần số depth và lượt xác nhận riêng:
[10 — Depth weight and frequency sweep](10_depth_weight_frequency_sweep.md).

Đọc [Executive Research Assessment và Q1–Q10](06_recommended_roadmap.md) trước.

1. [Architecture/source audit](01_architecture_audit.md)
2. [Depth/dataset/Leica/fair benchmarks](02_depth_evaluation.md)
3. [VIO, covariance, fresh runs](03_vio_evaluation.md)
4. [Mapping và E1–E5](04_mapping_evaluation.md)
5. [Research novelty và UATDF](05_research_novelty_assessment.md)
6. [Roadmap/executive assessment](06_recommended_roadmap.md)
7. [Reproducibility, commands, artifact map](07_experiment_reproducibility.md)

Các báo cáo 01–08 mô tả mã trước khi tích hợp depth vào VINS-Fusion; bản tích hợp mới có sửa estimator, không sửa pretrained weights. VERIFIED là phép đo được
chạy và có artifact, không có nghĩa mọi giả định ground truth đã được xác nhận độc lập.


## Architecture improvement — 2026-11-08

[Measured balanced fusion v2](08_fusion_architecture_improvement.md)
