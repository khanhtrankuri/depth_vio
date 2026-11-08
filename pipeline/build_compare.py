"""Cross-sequence comparison page (V1_01/02/03). Reads saved outputs only.
Usage: python build_compare.py <url_v101> <url_v102> <url_v103>"""
import base64, json, os, sys
import numpy as np
import cv2
from common import ROOT

SEQS = ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]
URLS = dict(zip(SEQS, sys.argv[1:4])) if len(sys.argv) >= 4 else {s: "#" for s in SEQS}


def J(s, p):
    return json.load(open(os.path.join(ROOT, "outputs", s, p)))


def img(s, rel, maxw=900, q=80):
    im = cv2.imread(os.path.join(ROOT, "outputs", s, rel), cv2.IMREAD_COLOR)
    if im.shape[1] > maxw:
        im = cv2.resize(im, (maxw, int(im.shape[0] * maxw / im.shape[1])), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


cm = lambda x: f"{100*x:.1f} cm"; pct = lambda x: f"{100*x:.1f}%"; deg = lambda x: f"{x:.2f}°"
rows = {}
for s in SEQS:
    sf = J(s, "single_frame_summary.json"); d = J(s, "stage5_depth_diag_gtrotcorr/single_frame_summary.json")
    s1 = J(s, "stage1_input/info.json"); s3 = J(s, "stage3_features/features.json")
    pf = s3["per_frame"][1:]; rj = sum(x["rej_lk"] + x["rej_fb"] + x["rej_ransac"] for x in pf)
    vm = J(s, "stage9_eval/vio_metrics.json"); off = J(s, "stage9_eval/gt_rot_offset_diag.json")
    rc = J(s, "stage9_eval/recon_metrics.json"); fu = {t: J(s, f"stage8_fusion/fusion_{t}.json") for t in ("openvins", "gt", "gtrotcorr")}
    cons = lambda t: np.median([c["median_rel"] for c in fu[t]["consistency"]])
    mi, mr = sf["stage5"]["metrics_igev_vs_leica"], d["stage5"]["metrics_igev_vs_leica"]
    rows[s] = [
        ("Dữ liệu", "Cặp stereo đồng bộ", f"{s1['n_synced_pairs']} ({s1['n_cam0']}/{s1['n_cam1']})"),
        ("Dữ liệu", "Quãng đường / thời lượng", f"{vm['path_length_m']:.1f} m / {vm['duration_s']:.0f} s"),
        ("Features (tái hiện)", "Tỉ lệ loại mỗi frame", pct(rj / max(1, rj + sum(x['tracked'] for x in pf)))),
        ("IGEV++ frame #700", "Pixel qua L-R check", pct(sf["stage4"]["igev_valid_frac"])),
        ("IGEV++ frame #700", "SGBM hợp lệ (baseline)", pct(sf["stage4"]["sgbm_valid_frac"])),
        ("Depth vs Leica", "MAE · GT gốc", cm(mi["MAE_m"])),
        ("Depth vs Leica", "AbsRel · GT gốc", pct(mi["AbsRel"])),
        ("Depth vs Leica", "MAE · GT bù offset (chẩn đoán)", cm(mr["MAE_m"])),
        ("Depth vs Leica", "δ<1.25 · GT bù offset", pct(mr["delta_1_25"])),
        ("OpenVINS", "ATE vị trí RMSE", cm(vm["ATE_trans_rmse_m"])),
        ("OpenVINS", "ATE xoay RMSE", deg(vm["ATE_rot_rmse_deg"])),
        ("OpenVINS", "RPE vị trí @ 1 m", cm(vm["RPE_trans_rmse_m@1m"])),
        ("OpenVINS", "RPE xoay @ 1 m", deg(vm["RPE_rot_rmse_deg@1m"])),
        ("Chẩn đoán GT", "Offset xoay GT↔IMU", deg(off["offset_deg"])),
        ("Chẩn đoán GT", "Lệch hướng sau bù (trung vị)", deg(off["rot_diff_median_after_deg"])),
        ("Fusion", "Nhất quán depth · OpenVINS", f"{100*cons('openvins'):.2f}%"),
        ("Fusion", "Nhất quán depth · GT gốc", f"{100*cons('gt'):.2f}%"),
        ("Bản đồ vs Leica", "TSDF · OpenVINS", cm(rc["tsdf_cloud_openvins"]["acc_median_m"])),
        ("Bản đồ vs Leica", "TSDF · GT gốc", cm(rc["tsdf_cloud_gt"]["acc_median_m"])),
        ("Bản đồ vs Leica", "TSDF · GT bù offset", cm(rc["tsdf_cloud_gtrotcorr"]["acc_median_m"])),
        ("Bản đồ vs Leica", "TSDF trong 5 cm · OpenVINS", pct(rc["tsdf_cloud_openvins"]["frac_within_5cm"])),
        ("Thời gian", "IGEV++ trái+phải / cặp", f"{J(s, 'stage8_fusion/infer_timing.json')['igev_LR_ms_mean']:.0f} ms"),
        ("Thời gian", "OpenVINS / frame", f"{J(s, 'stage7_vio/stage7_info.json')['timing_ms_mean']['total']:.1f} ms"),
    ]

body = ""
grp = None
for i, (g, name, _) in enumerate(rows[SEQS[0]]):
    if g != grp:
        body += f'<tr class="grp"><th colspan="4">{g}</th></tr>'; grp = g
    body += f"<tr><td>{name}</td>" + "".join(f"<td>{rows[s][i][2]}</td>" for s in SEQS) + "</tr>"
head = "".join(f'<th><a href="{URLS[s]}" target="_blank" rel="noopener">{s}</a></th>' for s in SEQS)

gallery = ""
for title, rel, kinds in (("Quỹ đạo OpenVINS so với GT", "stage9_eval/trajectory_vs_gt.png", "VIO output · OpenVINS / Tham chiếu"),
                          ("IGEV++ disparity, frame #700", "stage4_disparity/disparity.png", "Model output · IGEV++ / Baseline SGBM"),
                          ("Bản đồ fusion (cộng dồn thô và TSDF)", "stage8_fusion/fused_views.png", "Model + VIO output"),
                          ("Sai số bản đồ so với Leica", "stage9_eval/recon_error.png", "Model + VIO / Tham chiếu / Chẩn đoán")):
    gallery += f'<section class="row"><h2>{title}</h2><p class="src">{kinds}</p><div class="trio">' + "".join(
        f'<figure><figcaption><a href="{URLS[s]}" target="_blank" rel="noopener">{s}</a></figcaption><img src="{img(s, rel)}" alt="{title} {s}" loading="lazy"></figure>' for s in SEQS) + "</div></section>"

offs = [J(s, "stage9_eval/gt_rot_offset_diag.json")["offset_deg"] for s in SEQS]
html = open(os.path.join(os.path.dirname(__file__), "compare_template.html"), encoding="utf-8").read()
html = html.replace("{{LINKS}}", "".join(f'<a href="{URLS[s]}" target="_blank" rel="noopener">{s} →</a>' for s in SEQS)).replace("{{HEAD}}", head).replace("{{BODY}}", body).replace("{{GALLERY}}", gallery).replace(
    "{{OFFS}}", " / ".join(f"{o:.1f}°" for o in offs))
out = os.path.join(ROOT, "report", "compare_V1.html")
open(out, "w", encoding="utf-8").write(html)
print(out, f"{os.path.getsize(out)/1e6:.1f} MB")
