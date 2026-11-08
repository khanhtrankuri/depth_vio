"""Build a single self-contained HTML report from the saved outputs (no recomputation)."""
import base64, json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, ROOT, SEQ

J = lambda p: json.load(open(os.path.join(OUT, p)))
s1 = J("stage1_input/info.json"); s2 = J("stage2_rectify/rectify.json")
s3 = J("stage3_features/features.json"); t3 = J("stage3_features/timing.json")
sf = J("single_frame_summary.json"); s4, s5, s6 = sf["stage4"], sf["stage5"], sf["stage6"]
s5d = J("stage5_depth_diag_gtrotcorr/single_frame_summary.json")["stage5"]
s7 = J("stage7_vio/stage7_info.json"); vm = J("stage9_eval/vio_metrics.json"); off = J("stage9_eval/gt_rot_offset_diag.json")
fu = {t: J(f"stage8_fusion/fusion_{t}.json") for t in ("openvins", "gt", "gtrotcorr")}
rc = J("stage9_eval/recon_metrics.json"); it = J("stage8_fusion/infer_timing.json")


def img(rel, maxw=1500, q=84):
    im = cv2.imread(os.path.join(OUT, rel), cv2.IMREAD_COLOR)
    if im.shape[1] > maxw:
        im = cv2.resize(im, (maxw, int(im.shape[0] * maxw / im.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


# ---- interactive payloads ----
dL = np.load(os.path.join(OUT, "stage4_disparity/disp_igev.npy"))
Z = np.load(os.path.join(OUT, "stage5_depth/depth_igev_m.npy"))
Zr = np.load(os.path.join(OUT, "stage5_depth/depth_leica_ref_m.npy"))
Zc = np.load(os.path.join(OUT, "stage5_depth_diag_gtrotcorr/depth_leica_ref_m.npy"))
mm = lambda a: np.where(np.isfinite(a), np.clip(a * 1000, 0, 65535), 0).astype(np.uint16)
rgb = plt.get_cmap("turbo_r")(np.clip((np.nan_to_num(Z, nan=0) - 0.5) / 5.5, 0, 1))[..., :3]
rgb[~np.isfinite(Z)] = 0.08
ok, buf = cv2.imencode(".png", (rgb[..., ::-1] * 255).astype(np.uint8))
depth_png = "data:image/png;base64," + base64.b64encode(buf).decode()
left = cv2.imread(os.path.join(OUT, "stage2_rectify/left_rect.png"), 0)
ok, buf = cv2.imencode(".jpg", left, [cv2.IMWRITE_JPEG_QUALITY, 85]); left_jpg = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
probe = dict(disp=b64((np.clip(dL, 0, 655) * 100).astype(np.uint16)), z=b64(mm(Z)), zr=b64(mm(Zr)), zc=b64(mm(Zc)))

loc = np.load(os.path.join(OUT, "stage6_pointcloud/viewer_xyzg.npy"))[:45000]
glob_ = np.load(os.path.join(OUT, "stage8_fusion/viewer_global.npy"))
pack = lambda a: dict(xyz=b64(np.round(a[:, :3] * 1000).astype(np.int16)), g=b64((a[:, 3] * 255).astype(np.uint8)), n=len(a))
traj = np.loadtxt(os.path.join(OUT, "stage9_eval/vio_aligned_tum.txt"))[::4, 1:4]
gtt = np.loadtxt(os.path.join(OUT, "stage9_eval/gt_body_tum.txt"))[::40, 1:4]
cams = json.load(open(os.path.join(OUT, "stage8_fusion/cams_openvins.json")))[::12]
cam_T = [np.round(np.array(c["T_WC"])[:3, :4], 4).ravel().tolist() for c in cams]
payload = dict(probe=probe, fx=s5["fx_px"], B=s5["baseline_m"], loc=pack(loc), glob=pack(glob_),
               traj=np.round(traj, 3).ravel().tolist(), gt=np.round(gtt, 3).ravel().tolist(), cams=cam_T)

f = lambda x, n=2: f"{x:.{n}f}"
pct = lambda x, n=1: f"{100 * x:.{n}f}%"
cm = lambda x, n=1: f"{100 * x:.{n}f} cm"
mi, ms = s5["metrics_igev_vs_leica"], s5["metrics_sgbm_vs_leica"]
mid, msd = s5d["metrics_igev_vs_leica"], s5d["metrics_sgbm_vs_leica"]
last = s3["per_frame"][-1]
_pf = s3["per_frame"][1:]
_rj = sum(x["rej_lk"] + x["rej_fb"] + x["rej_ransac"] for x in _pf)
rej_rate = _rj / max(1, _rj + sum(x["tracked"] for x in _pf))
_bins = [b for b in s5["err_by_depth"] if b["median_abs"] is not None and b["n"] > 2000]
bins_txt = ", ".join(f"{cm(b['median_abs'],1)} ở {b['z0']:.1f}–{b['z1']:.1f} m" for b in _bins[::2])
cons = lambda t: np.median([c["median_rel"] for c in fu[t]["consistency"]])
consabs = lambda t: np.median([c["median_abs_m"] for c in fu[t]["consistency"]])
R = lambda k: rc[k]

TAG = {"model": ("Model output", "IGEV++"), "vio": ("VIO output", "OpenVINS"), "geo": ("Hình học", "geometry"),
       "base": ("Baseline", "không phải IGEV++"), "ref": ("Tham chiếu", "dataset GT"), "diag": ("Chẩn đoán", "không phải GT"),
       "reimpl": ("Tái hiện", "OpenCV, logic OpenVINS"), "input": ("Dữ liệu gốc", "EuRoC")}


def chip(k):
    a, b = TAG[k]
    return f'<span class="chip chip-{k}"><b>{a}</b> · {b}</span>'


def fig(src, cap, kinds):
    return f'<figure><div class="chips">{"".join(chip(k) for k in kinds)}</div><img src="{src}" alt="{cap}" loading="lazy"><figcaption>{cap}</figcaption></figure>'


def stage(n, title, role, status, status_kind, figs, io, algo, notes, concl, extra=""):
    return f"""
<section class="stage" id="s{n}">
  <header class="stage-head">
    <div class="stage-no">Stage {n}</div>
    <h2>{title}</h2>
    <span class="status status-{status_kind}">{status}</span>
  </header>
  <p class="role">{role}</p>
  {''.join(figs)}
  {extra}
  <div class="cols">
    <div class="col"><h3>Đầu vào → đầu ra</h3>{io}</div>
    <div class="col"><h3>Phép biến đổi / thuật toán</h3>{algo}</div>
    <div class="col"><h3>Nhận xét</h3>{notes}</div>
  </div>
  <p class="concl"><b>Kết luận:</b> {concl}</p>
</section>"""


def table(rows, head):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


S = []
S.append(stage(1, "Input Visualization", "Đọc cặp ảnh stereo gốc, kiểm tra kích thước, kiểu dữ liệu và đồng bộ thời gian giữa cam0 và cam1.",
    "Đã chạy", "ok",
    [fig(img("stage1_input/original_pair.png"), f"Cặp stereo gốc, frame #{s1['frame_index']} (t = {s1['timestamp_ns']} ns). Ảnh bị méo xuyên tâm rõ ở mép (ống kính góc rộng).", ["input"]),
     fig(img("stage1_input/frame_intervals.png", 900), "Khoảng cách thời gian giữa các frame cam0 trên toàn sequence.", ["input"])],
    f"<ul><li>Sequence <code>{SEQ}</code>, định dạng ASL đã giải nén từ <code>{SEQ}.zip</code>.</li><li>{s1['n_cam0']} ảnh cam0, {s1['n_cam1']} ảnh cam1; <b>{s1['n_synced_pairs']}</b> cặp trùng timestamp tuyệt đối.</li><li>{s1['shape'][1]}×{s1['shape'][0]}, {s1['dtype']}, 1 kênh.</li><li>Độ sáng trung bình: trái {f(s1['left_mean'],1)}, phải {f(s1['right_mean'],1)}.</li></ul>",
    "<p>Ghép cặp theo timestamp nanosecond giống hệt nhau trong <code>cam0/data.csv</code> và <code>cam1/data.csv</code>. Hai camera MT9M034 được trigger phần cứng nên không cần nội suy.</p>",
    f"<p>Δt trung vị {f(s1['cam0_dt_ms_median'],3)} ms (min {f(s1['cam0_dt_ms_min'],3)}, max {f(s1['cam0_dt_ms_max'],3)})" +
    (": đúng 20 Hz, không rơi frame." if s1['cam0_dt_ms_max'] < 60 else ": có khoảng trống lớn hơn một chu kỳ, tức là có frame bị rơi.") +
    (f" Số ảnh hai camera lệch nhau {abs(s1['n_cam0']-s1['n_cam1'])} ảnh; ảnh không có cặp cùng timestamp bị bỏ qua." if s1['n_cam0'] != s1['n_cam1'] else "") +
    (f" Ảnh {'phải tối' if s1['left_mean'] > s1['right_mean'] else 'trái tối'} hơn ảnh kia khoảng {abs(s1['left_mean']-s1['right_mean']):.0f} mức xám do auto-exposure riêng từng camera; matching cần bền với chênh lệch này." if abs(s1['left_mean']-s1['right_mean']) > 8 else " Độ sáng hai ảnh gần nhau.") + "</p>",
    "Thành công. Dữ liệu stereo, IMU (200 Hz), calibration và ground truth đều có sẵn nên chạy được toàn bộ pipeline."))

S.append(stage(2, "Stereo Preprocessing", "Undistort và rectify để đường epipolar thành các hàng ảnh nằm ngang, đưa bài toán matching về tìm kiếm 1D theo u.",
    "Đã chạy", "ok",
    [fig(img("stage2_rectify/rectified_epipolar.png"), "Trên: ảnh gốc. Dưới: sau rectification, cùng một vật nằm trên cùng một đường ngang ở hai ảnh.", ["geo"]),
     fig(img("stage2_rectify/rectified_matches.png"), "60 cặp ORB ngẫu nhiên giữa ảnh trái và phải đã rectify: các đường nối gần như nằm ngang.", ["geo"]),
     fig(img("stage2_rectify/dy_hist.png", 900), "Kiểm chứng định lượng: độ lệch dọc |v_trái − v_phải| của các match ORB trước và sau rectification.", ["geo"])],
    f"<ul><li>Intrinsics và distortion radial-tangential từ <code>cam0/1/sensor.yaml</code>.</li><li>T<sub>c1←c0</sub> = T<sub>B←c1</sub><sup>−1</sup>·T<sub>B←c0</sub>.</li><li>Đầu ra: f<sub>x</sub> = {f(s2['fx_rect'])} px, c<sub>x</sub> = {f(s2['cx_rect'])}, c<sub>y</sub> = {f(s2['cy_rect'])}, <b>baseline = {f(s2['baseline_m']*100,2)} cm</b>.</li></ul>",
    "<p><code>cv2.stereoRectify</code> (CALIB_ZERO_DISPARITY, alpha = 0) tính hai phép quay R0, R1 đưa trục quang song song và vuông góc với baseline, rồi <code>initUndistortRectifyMap</code> + <code>remap</code> bilinear. Ảnh giữ dạng uint8; IGEV++ tự chuẩn hóa về [−1, 1] và nhận 3 kênh lặp từ ảnh xám.</p>",
    f"<p>Trước rectification, lệch dọc trung vị là <b>{f(s2['dy_raw_median'],1)} px</b> (do c<sub>y</sub> khác nhau và quay tương đối giữa hai camera). Sau đó còn <b>{f(s2['dy_rect_median'],2)} px</b> trung vị, p90 = {f(s2['dy_rect_p90'],2)} px, đủ cho matching 1D. alpha = 0 cắt bỏ viền đen nên mất một ít trường nhìn ở góc.</p>",
    "Thành công. Calibration EuRoC chính xác dưới pixel."))

S.append(stage(3, "Feature Extraction & Tracking", "Front-end thị giác của VIO: phát hiện góc, theo dõi qua thời gian và ghép trái-phải.",
    "Đã chạy · tái hiện", "warn",
    [fig(img("stage3_features/keypoints.png", 900), "Frame cuối: xanh lá = có match stereo hợp lệ, vàng = chỉ có ở ảnh trái; ✕ đỏ = KLT mất dấu, ✕ tím = sai forward-backward, ✕ xanh dương = outlier RANSAC.", ["reimpl"]),
     fig(img("stage3_features/tracks.png", 900), f"Vết track của các feature còn sống qua {len(s3['per_frame'])} frame (2 s); màu từ xanh (cũ) tới đỏ (mới).", ["reimpl"]),
     fig(img("stage3_features/stereo_klt.png"), "KLT trái → phải trên ảnh gốc (chưa rectify), sau đó lọc theo ràng buộc epipolar.", ["reimpl"]),
     fig(img("stage3_features/feature_stats.png"), "Số feature theo frame và phân bố độ dài track.", ["reimpl"])],
    f"<ul><li>40 frame liên tiếp kết thúc ở frame #{s1['frame_index']}.</li><li>Frame cuối: {last['tracked']} track tiếp tục, {last['new']} mới, {last['stereo_ok']} có match stereo, loại {last['rej_lk']+last['rej_fb']+last['rej_ransac']}.</li><li>Tổng {s3['n_tracks']} track, dài trung bình {f(s3['mean_track_len'],1)} frame.</li><li>{f(t3['feature_track_ms_per_frame'],1)} ms/frame trên CPU.</li></ul>",
    "<p>Không có mã nguồn ARTI-VIO; theo lựa chọn của bạn, OpenVINS thay thế. Ở đây tôi tái hiện front-end <code>TrackKLT</code> của OpenVINS bằng OpenCV với tham số <code>euroc_mav</code>: CLAHE, FAST ngưỡng 20 trên lưới 5×5, tối đa 200 điểm, cách nhau ≥ 10 px, KLT pyramid cửa sổ 15. Loại bỏ theo 3 tầng: KLT thất bại, forward-backward &gt; 1 px, RANSAC ma trận fundamental trên tọa độ đã undistort.</p>",
    f"<p>Feature phân bố đều nhờ lưới; vùng tường trơn ít điểm. Track dài trên thảm có hoa văn và băng dán đen-trắng. Trong 40 frame, {pct(rej_rate)} feature đang track bị loại mỗi frame" +
    (", thấp vì chuyển động chậm." if rej_rate < 0.03 else ", cao hơn rõ rệt do chuyển động nhanh và motion blur làm KLT trôi; phần lớn bị loại bởi kiểm tra forward-backward và RANSAC.") + " Hình này cho thấy thuật toán làm gì nhưng không phải output của binary OpenVINS; số feature thật OpenVINS dùng có ở Stage 7.</p>",
    "Thành công dưới dạng tái hiện có ghi rõ nguồn gốc. Front-end thật của OpenVINS chạy bên trong Stage 7."))

probe_html = """
<div class="probe" id="probe">
  <div class="chips"><span class="chip chip-model"><b>Model output</b> · IGEV++</span><span class="chip chip-ref"><b>Tham chiếu</b> · Leica + pose GT</span></div>
  <div class="probe-grid">
    <div class="probe-canvas"><canvas id="pc" width="752" height="480" aria-label="Depth map, di chuột để đọc giá trị"></canvas>
      <div class="probe-toggle"><label><input type="radio" name="pv" id="pv-depth" value="depth" checked> Depth</label><label><input type="radio" name="pv" id="pv-gray" value="gray"> Ảnh xám</label></div></div>
    <div class="probe-read">
      <div class="cbar"><span>0.5 m</span><i></i><span>6 m</span></div>
      <dl>
        <dt>Pixel (u, v)</dt><dd id="r-uv">di chuột hoặc chạm vào ảnh</dd>
        <dt>Disparity IGEV++</dt><dd id="r-d">–</dd>
        <dt>Depth IGEV++ (L-R hợp lệ)</dt><dd id="r-z">–</dd>
        <dt>Leica, pose GT gốc</dt><dd id="r-zr">–</dd>
        <dt>Leica, GT đã bù offset</dt><dd id="r-zc">–</dd>
        <dt>Điểm 3D (camera)</dt><dd id="r-xyz">–</dd>
      </dl>
    </div>
  </div>
</div>"""

S.append(stage(4, "Stereo Matching với IGEV++", "Ước lượng disparity dày đặc cho mọi pixel ảnh trái bằng mạng IGEV++.",
    "Đã chạy · inference thật", "ok",
    [fig(img("stage4_disparity/disparity.png"), "Trên phải: disparity thô của IGEV++. Dưới trái: sau kiểm tra nhất quán trái-phải (&lt; 1 px). Dưới phải: OpenCV StereoSGBM làm baseline.", ["model", "base"])],
    f"<ul><li>Cặp ảnh đã rectify 752×480.</li><li>Checkpoint chính thức <code>igev_plusplus/sceneflow.pth</code> (Google Drive của tác giả), 16 vòng GRU, fp16 autocast.</li><li>Disparity hợp lệ: {f(s4['igev_disp_min'],1)} – {f(s4['igev_disp_max'],1)} px, trung vị {f(s4['igev_disp_median'],1)} px.</li><li><b>{pct(s4['igev_valid_frac'])}</b> pixel qua L-R check, so với {pct(s4['sgbm_valid_frac'])} của SGBM.</li><li>{f(s4['timing']['igev_ms'],0)} ms/lượt trên RTX 4060 Laptop (đo một lần; CPU có thể đang chạy tác vụ khác song song, lần đo không tranh chấp trên V1_01 là 364 ms). SGBM: {f(s4['timing']['sgbm_ms'],0)} ms.</li></ul>",
    "<p>IGEV++ xây Multi-range Geometry Encoding Volume (3 dải 48/96/192 px) từ feature MobileNetV2, hồi quy disparity khởi tạo bằng soft-argmin rồi tinh chỉnh bằng ConvGRU nhiều tầng. Để có mặt nạ hợp lệ, tôi chạy thêm một lượt với ảnh lật ngang và đổi vai trái/phải để có disparity của ảnh phải, rồi giữ pixel có |d<sub>L</sub>(u) − d<sub>R</sub>(u − d<sub>L</sub>)| &lt; 1 px.</p>",
    f"<p>Disparity mượt, cạnh vật thể sắc, kể cả tường xám không có texture, nơi SGBM bị thủng. Pixel bị loại tập trung ở dải mép trái ({pct(s4['invalid_left_band_frac'],0)} của 60 cột đầu không hợp lệ, vì phần này không có trong ảnh phải), vùng bị che khuất sau mép tấm panel, và vùng cửa sổ cháy sáng. Checkpoint huấn luyện trên SceneFlow tổng hợp nhưng tổng quát hóa tốt sang ảnh xám EuRoC.</p>",
    "Thành công. Đây là output thực của IGEV++ trên dữ liệu của bạn."))

S.append(stage(5, "Depth Estimation", "Chuyển disparity thành độ sâu metric và đo sai số so với scan laser Leica của phòng.",
    "Đã chạy", "ok",
    [fig(img("stage5_depth/depth.png"), "Depth IGEV++, depth SGBM, depth tham chiếu (scan Leica chiếu vào camera bằng pose GT gốc) và bản đồ sai số tuyệt đối.", ["model", "base", "ref"]),
     fig(img("stage5_depth/depth_stats.png"), "Phân bố depth hợp lệ và so sánh từng pixel với tham chiếu (pose GT gốc).", ["model", "ref"]),
     fig(img("stage5_depth_diag_gtrotcorr/depth.png"), f"Cùng phép so sánh nhưng tham chiếu được chiếu bằng GT đã bù offset xoay hằng {f(off['offset_deg'],1)}° (xem Stage 9; offset ước lượng từ OpenVINS, chỉ dùng để chẩn đoán).", ["model", "diag"])],
    f"<ul><li>Z = f<sub>x</sub>·B / d với f<sub>x</sub>·B = {f(s5['fxB'],2)} px·m.</li><li>Depth hợp lệ: {f(s5['depth_min'],2)} – {f(s5['depth_max'],2)} m, trung vị {f(s5['depth_median'],2)} m.</li><li>Pixel có tham chiếu Leica: {pct(s5['ref_coverage'],0)}.</li></ul>" +
    table([["IGEV++", cm(mi['MAE_m']), cm(mi['RMSE_m']), pct(mi['AbsRel']), pct(mi['delta_1_25'])],
           ["SGBM", cm(ms['MAE_m']), cm(ms['RMSE_m']), pct(ms['AbsRel']), pct(ms['delta_1_25'])],
           ["IGEV++ · GT bù offset", cm(mid['MAE_m']), cm(mid['RMSE_m']), pct(mid['AbsRel']), pct(mid['delta_1_25'])],
           ["SGBM · GT bù offset", cm(msd['MAE_m']), cm(msd['RMSE_m']), pct(msd['AbsRel']), pct(msd['delta_1_25'])]],
          ["Frame #700", "MAE", "RMSE", "AbsRel", "δ&lt;1.25"]),
    f"<p>Bỏ pixel không qua L-R check và pixel xa hơn 10 m. Tham chiếu: đám mây Leica MS50 (<code>pointcloud0</code>) biến đổi bằng T<sub>W←B</sub>(t) của GT nhân T<sub>B←c0</sub>, T<sub>c0←rect</sub>, chiếu bằng intrinsics rectified với z-buffer 2×2 px. Sai số lý thuyết tăng theo bình phương khoảng cách: σ<sub>z</sub> ≈ z²/(f<sub>x</sub>B)·σ<sub>d</sub>; với σ<sub>d</sub> = 0.25 px thì 2 m → 2 cm, 4 m → 8 cm.</p>",
    f"<p>Trung vị |Δz| theo khoảng cách (pose GT gốc): {bins_txt}. Sai số lớn nhất nằm ở mép vật (tấm panel, hộp), nơi chênh lệch nhỏ của pose tham chiếu tạo dải sai số rộng. " +
    (f"Khi bù offset xoay của GT, MAE IGEV++ giảm từ {cm(mi['MAE_m'])} xuống {cm(mid['MAE_m'])}, nghĩa là phần lớn \"sai số\" ban đầu đến từ pose tham chiếu chứ không phải từ IGEV++. "
     if mid['MAE_m'] < 0.8 * mi['MAE_m'] else f"Bù offset xoay của GT chỉ đổi MAE từ {cm(mi['MAE_m'])} sang {cm(mid['MAE_m'])}, nên ở frame này sai số chủ yếu đến từ chính depth. ") +
    ("IGEV++ tốt hơn SGBM ở mọi chỉ số và phủ nhiều pixel hơn." if (mi['MAE_m'] < ms['MAE_m'] and mi['AbsRel'] < ms['AbsRel'] and mi['delta_1_25'] > ms['delta_1_25'])
     else f"IGEV++ phủ {pct(s4['igev_valid_frac'])} pixel so với {pct(s4['sgbm_valid_frac'])} của SGBM; trên vùng SGBM dám trả lời, chỉ số hai bên gần nhau hơn (xem bảng).") + "</p>",
    "Thành công. Depth metric chính xác vài cm trong 2.5 m; dùng công cụ bên dưới để đọc giá trị tại từng pixel.",
    extra=probe_html))

viewer = lambda id_, h: f'<div class="viewer" id="{id_}" style="height:{h}px"><div class="vhint">Kéo để xoay · cuộn/pinch để zoom · chuột phải để dời</div></div>'
S.append(stage(6, "3D Point Cloud Generation", "Back-project depth hợp lệ thành đám mây điểm 3D trong hệ camera trái đã rectify.",
    "Đã chạy", "ok",
    [f'<figure><div class="chips">{chip("model")}{chip("geo")}</div>{viewer("v-local", 440)}<figcaption>Đám mây cục bộ của frame #700 ({loc.shape[0]:,} điểm lấy mẫu từ {s6["n_points"]:,}), màu theo độ xám. Trục: X phải, Y xuống, Z theo hướng nhìn.</figcaption></figure>',
     fig(img("stage6_pointcloud/pointcloud_views.png"), "Ba góc nhìn tĩnh của cùng đám mây.", ["model", "geo"])],
    f"<ul><li>{s6['n_points']:,} điểm; {s6['n_points_ds2cm']:,} sau voxel 2 cm.</li><li>Hộp bao X [{f(s6['bbox_min'][0],1)}, {f(s6['bbox_max'][0],1)}] m, Z [{f(s6['bbox_min'][2],1)}, {f(s6['bbox_max'][2],1)}] m.</li><li>File: <code>local_cloud_cam0rect.ply</code>, <code>local_cloud_ds2cm.ply</code> (Open3D).</li></ul>",
    "<p>X = (u − c<sub>x</sub>)·Z/f<sub>x</sub>, Y = (v − c<sub>y</sub>)·Z/f<sub>y</sub>, Z = Z, dùng intrinsics sau rectification. Tạo <code>open3d.geometry.PointCloud</code>, gán màu xám từ ảnh trái rectified, ghi .ply.</p>",
    "<p>Sàn, các tấm panel và hộp hiện rõ dạng phẳng. Tường xa có lớp nhiễu dày hơn vì sai số depth tăng theo z². Mép vật có vài điểm \"bay\" (flying pixels) do disparity được nội suy qua biên che khuất.</p>",
    "Thành công. Đây là bản đồ cục bộ một frame, chưa có pose."))

S.append(stage(7, "VIO Pose Estimation (OpenVINS thay cho ARTI-VIO)", "Ước lượng pose 6-DoF của camera từ stereo + IMU.",
    "Đã chạy · OpenVINS", "warn",
    [fig(img("stage9_eval/trajectory_vs_gt.png"), "Quỹ đạo OpenVINS (căn chỉnh SE(3) với GT bằng Umeyama) so với ground truth Vicon; màu = sai số vị trí.", ["vio", "ref"]),
     fig(img("stage7_vio/sparse_map.png"), "Bản đồ thưa: landmark SLAM (đỏ) và feature MSCKF (xám) do OpenVINS publish, cùng hệ trục camera mỗi 6 s.", ["vio"])],
    f"<ul><li>Input: <code>{SEQ}.bag</code> (cam0, cam1 20 Hz, IMU 200 Hz).</li><li>OpenVINS v2.7, ROS Noetic trong Docker, cấu hình chính thức <code>euroc_mav</code>, chế độ serial (xử lý tuần tự, tất định).</li><li>{s7['n_poses']} pose IMU trong {f(vm['duration_s'],0)} s; quãng đường {f(vm['path_length_m'],1)} m.</li><li>{s7['n_slam']:,} vị trí landmark SLAM được publish (một landmark có nhiều vị trí qua thời gian, tạo thành các vệt), {s7['n_msckf']:,} feature MSCKF.</li><li>Thời gian: {f(s7['timing_ms_mean']['total'],1)} ms/frame (tracking {f(s7['timing_ms_mean']['tracking'],1)}, SLAM update {f(s7['timing_ms_mean']['slam_update'],1)}).</li></ul>",
    "<p><b>Không tìm thấy mã nguồn ARTI-VIO</b>, nên theo lựa chọn của bạn tôi chạy OpenVINS: Multi-State Constraint Kalman Filter. IMU được tích phân để dự đoán trạng thái; feature KLT stereo dùng làm ràng buộc MSCKF (không giữ trong state) hoặc landmark SLAM (giữ trong state). Online calibration extrinsic và time offset camera-IMU được bật như trong cấu hình EuRoC gốc. Pose xuất ra là T<sub>W←IMU</sub>.</p>",
    f"<p>ATE vị trí RMSE <b>{cm(vm['ATE_trans_rmse_m'])}</b> (max {cm(vm['ATE_trans_max_m'])}); RPE vị trí trên 1 m quãng đường là {cm(vm['RPE_trans_rmse_m@1m'])}. ATE xoay RMSE {f(vm['ATE_rot_rmse_deg'],1)}°" +
    (f"; Stage 9 cho thấy phần lớn đến từ offset hằng {f(off['offset_deg'],1)}° giữa frame body của GT và IMU (sau khi bù còn {f(off['rot_diff_median_after_deg'],2)}° trung vị), không phải drift. " if off['offset_deg'] > 2 * off['rot_diff_median_after_deg'] else ". ") + "Node thoát với lỗi <code>class_loader</code> lúc tắt (sau frame cuối), không ảnh hưởng kết quả.</p>",
    "Thành công với OpenVINS. Stage 7 cho ARTI-VIO chưa chạy được vì thiếu mã nguồn; pipeline đã sẵn sàng để thay vào: chỉ cần file quỹ đạo dạng TUM."))

S.append(stage(8, "VIO + Depth Fusion", "Đưa depth IGEV++ của nhiều frame vào chung hệ thế giới bằng pose OpenVINS rồi hợp nhất thành một bản đồ.",
    "Đã chạy", "ok",
    [f'<figure><div class="chips">{chip("model")}{chip("vio")}{chip("geo")}</div>{viewer("v-global", 520)}<figcaption>Bản đồ TSDF toàn cục từ {fu["openvins"]["n_frames"]} keyframe ({glob_.shape[0]:,} điểm bề mặt hiển thị). Cam = quỹ đạo OpenVINS, xám mảnh = GT; khung xanh = camera mỗi 3 s.</figcaption></figure>',
     fig(img("stage8_fusion/fused_views.png"), "Trái: cộng dồn thô (3000 điểm/frame). Giữa: TSDF fusion 3 cm. Phải: lát cắt đứng so với scan Leica.", ["model", "vio", "ref"]),
     fig(img("stage8_fusion/consistency.png"), "Nhất quán depth giữa hai keyframe liên tiếp: depth frame k−1 được biến đổi SE(3) sang frame k và so với depth của frame k.", ["model", "vio"])],
    f"<ul><li>IGEV++ trên mỗi cặp thứ 5 (4 Hz): {it['n']} cặp, {f(it['igev_LR_ms_mean'],0)} ms/cặp cho 2 lượt (trái + phải).</li><li>Pose: quỹ đạo OpenVINS đã căn chỉnh vào hệ GT (để so được với scan Leica).</li><li>Depth cắt ở 4 m (σ<sub>z</sub> ≈ 8 cm).</li><li>Mesh TSDF: {fu['openvins']['mesh_vertices']:,} đỉnh. File: <code>tsdf_mesh_openvins.ply</code>, <code>tsdf_cloud_openvins.ply</code>, <code>naive_accum_openvins.ply</code>.</li></ul>",
    f"<p>Mỗi pixel → điểm trong hệ camera → P<sub>W</sub> = T<sub>W←B</sub>·T<sub>B←c0</sub>·T<sub>c0←rect</sub>·P<sub>c</sub>. Đồng bộ depth với pose bằng pose gần nhất (OpenVINS 20 Hz cùng timestamp ảnh). TSDF: lưới dày 3 cm trên GPU, truncation 12 cm, trung bình có trọng số theo Curless &amp; Levoy, trích bề mặt bằng marching cubes ({f(fu['openvins']['integrate_ms_mean'],0)} ms/frame). Open3D 0.20 trong môi trường này trả về volume rỗng (kể cả với mặt phẳng tổng hợp) nên TSDF được tự viết bằng PyTorch; Open3D vẫn dùng cho đám mây, I/O và đo khoảng cách.</p>",
    f"<p>Depth giữa các frame rất nhất quán: sai lệch tương đối trung vị <b>{pct(cons('openvins'),2)}</b> ({cm(consabs('openvins'),2)}) với pose OpenVINS. Cộng dồn thô bị nhòe vì nhiễu depth xa và mép vật; TSDF làm phẳng sàn và tường, bỏ phần lớn điểm bay. Bề mặt bên ngoài tường phòng (vùng nhìn qua cửa sổ, cửa ra vào) không có trong scan Leica nên hiện sai số lớn ở Stage 9.</p>",
    "Thành công: có bản đồ 3D toàn cục từ IGEV++ + VIO với quỹ đạo camera bên trong."))

S.append(stage(9, "Evaluation", "So sánh định lượng với ground truth: quỹ đạo Vicon và scan laser Leica.",
    "Đã chạy", "ok",
    [fig(img("stage9_eval/recon_error.png"), "Trái: khoảng cách từ mỗi điểm TSDF (pose OpenVINS) tới điểm Leica gần nhất. Phải: phân bố tích lũy sai số của các biến thể.", ["model", "vio", "ref", "diag"])],
    "<h4>VIO (evo, SE(3) Umeyama)</h4>" + table([
        ["ATE vị trí RMSE", cm(vm['ATE_trans_rmse_m'])], ["ATE vị trí max", cm(vm['ATE_trans_max_m'])],
        ["ATE xoay RMSE", f"{f(vm['ATE_rot_rmse_deg'],2)}°"],
        ["RPE vị trí @ 1 m", cm(vm['RPE_trans_rmse_m@1m'])], ["RPE xoay @ 1 m", f"{f(vm['RPE_rot_rmse_deg@1m'],2)}°"],
        ["RPE vị trí @ 5 m", cm(vm['RPE_trans_rmse_m@5m'])], ["RPE xoay @ 5 m", f"{f(vm['RPE_rot_rmse_deg@5m'],2)}°"]], ["Chỉ số", "OpenVINS"]) +
    "<h4>Bản đồ 3D so với scan Leica</h4>" + table([
        ["Cộng dồn thô · OpenVINS", cm(R('naive_accum_openvins')['acc_median_m']), pct(R('naive_accum_openvins')['frac_within_5cm'],0), pct(R('naive_accum_openvins')['completeness_5cm'],0)],
        ["TSDF · OpenVINS", cm(R('tsdf_cloud_openvins')['acc_median_m']), pct(R('tsdf_cloud_openvins')['frac_within_5cm'],0), pct(R('tsdf_cloud_openvins')['completeness_5cm'],0)],
        ["Cộng dồn thô · GT gốc", cm(R('naive_accum_gt')['acc_median_m']), pct(R('naive_accum_gt')['frac_within_5cm'],0), pct(R('naive_accum_gt')['completeness_5cm'],0)],
        ["TSDF · GT gốc", cm(R('tsdf_cloud_gt')['acc_median_m']), pct(R('tsdf_cloud_gt')['frac_within_5cm'],0), pct(R('tsdf_cloud_gt')['completeness_5cm'],0)],
        ["TSDF · GT bù offset (chẩn đoán)", cm(R('tsdf_cloud_gtrotcorr')['acc_median_m']), pct(R('tsdf_cloud_gtrotcorr')['frac_within_5cm'],0), pct(R('tsdf_cloud_gtrotcorr')['completeness_5cm'],0)]],
        ["Biến thể", "Accuracy trung vị", "Điểm trong 5 cm", "Completeness 5 cm"]) +
    "<h4>Thời gian xử lý</h4>" + table([
        ["Rectification", f"{f(t3['rectify_ms'],0)} ms", "CPU"], ["Feature/KLT (tái hiện)", f"{f(t3['feature_track_ms_per_frame'],1)} ms", "CPU"],
        ["IGEV++ 1 lượt", f"{f(s4['timing']['igev_ms'],0)} ms", "RTX 4060 Laptop"], ["IGEV++ trái + phải", f"{f(it['igev_LR_ms_mean'],0)} ms", "RTX 4060 Laptop"],
        ["StereoSGBM", f"{f(s4['timing']['sgbm_ms'],0)} ms", "CPU"], ["OpenVINS", f"{f(s7['timing_ms_mean']['total'],1)} ms", "CPU, Docker"],
        ["TSDF integrate", f"{f(fu['openvins']['integrate_ms_mean'],0)} ms", "GPU"]], ["Bước", "Mỗi frame", "Thiết bị"]),
    f"<p>Depth: MAE, RMSE, AbsRel, δ&lt;1.25 theo pixel (Stage 5). VIO: ATE/RPE bằng <code>evo</code>. Bản đồ: accuracy = khoảng cách điểm tái tạo → Leica; completeness = tỉ lệ điểm Leica trong hộp bao của bản đồ có điểm tái tạo trong 5 cm; cả hai sau voxel 2 cm.</p><p><b>Chẩn đoán offset GT:</b> ước lượng phép quay hằng R<sub>off</sub> sao cho R<sub>GT</sub>(t) ≈ R<sub>VIO</sub>(t)·R<sub>off</sub>. Kết quả {f(off['offset_deg'],2)}°; sau khi bù, lệch hướng trung vị giữa GT và OpenVINS giảm từ {f(off['rot_diff_median_before_deg'],2)}° xuống {f(off['rot_diff_median_after_deg'],2)}°. Vì R<sub>off</sub> được ước lượng từ OpenVINS, biến thể này <b>không phải ground truth độc lập</b>.</p>",
    f"<p>Fusion bằng pose OpenVINS đạt accuracy trung vị {cm(R('tsdf_cloud_openvins')['acc_median_m'])}. " +
    (f"Dùng orientation GT gốc lại cho bản đồ tệ hơn ({cm(R('tsdf_cloud_gt')['acc_median_m'])}) và nhất quán depth kém hơn ({pct(cons('gt'),2)} so với {pct(cons('openvins'),2)}). Bù offset hằng {f(off['offset_deg'],1)}° đưa GT về {cm(R('tsdf_cloud_gtrotcorr')['acc_median_m'])} và MAE depth Stage 5 từ {cm(mi['MAE_m'])} về {cm(mid['MAE_m'])}. Hai bằng chứng (scan Leica và nhất quán đa frame) cùng chỉ ra orientation trong <code>state_groundtruth_estimate0</code> của sequence này lệch so với hệ IMU/camera của calibration. "
     if R('tsdf_cloud_gt')['acc_median_m'] > 1.3 * R('tsdf_cloud_gtrotcorr')['acc_median_m'] else
     f"Pose GT gốc cho {cm(R('tsdf_cloud_gt')['acc_median_m'])}, GT bù offset {f(off['offset_deg'],1)}° cho {cm(R('tsdf_cloud_gtrotcorr')['acc_median_m'])}; khác biệt nhỏ nên offset orientation không phải yếu tố chính ở sequence này. ") +
    "TSDF có đuôi phân bố dài hơn cộng dồn thô: phần đuôi chủ yếu là bề mặt nhìn qua cửa sổ và cửa ra vào, nằm ngoài vùng scan Leica.</p>",
    "Đã đo đủ depth, VIO, bản đồ và thời gian." + (" Khi so pose hoặc chiếu depth bằng GT của sequence này, cần tính đến offset orientation." if R('tsdf_cloud_gt')['acc_median_m'] > 1.3 * R('tsdf_cloud_gtrotcorr')['acc_median_m'] else "")))

intro = f"""
<div class="summary">
  <div class="kpi"><span class="k">{pct(s4['igev_valid_frac'],0)}</span><span class="l">pixel IGEV++ qua L-R check</span></div>
  <div class="kpi"><span class="k">{cm(mid['median_abs_err_m'],1)}</span><span class="l">|Δz| trung vị IGEV++ vs Leica<br>(GT bù offset)</span></div>
  <div class="kpi"><span class="k">{cm(vm['ATE_trans_rmse_m'])}</span><span class="l">ATE OpenVINS trên {f(vm['path_length_m'],0)} m</span></div>
  <div class="kpi"><span class="k">{cm(R('tsdf_cloud_openvins')['acc_median_m'])}</span><span class="l">sai số trung vị bản đồ TSDF vs Leica</span></div>
</div>
<div class="legend">
  <p>Mỗi hình được gắn nhãn nguồn gốc:</p>
  <div class="chips">{''.join(chip(k) for k in ('input','model','vio','geo','base','reimpl','ref','diag'))}</div>
</div>
<div class="callout"><b>Khác với yêu cầu ban đầu:</b> không có mã nguồn ARTI-VIO nên Stage 7 dùng OpenVINS v2.7 (lựa chọn của bạn). Stage 3 là bản tái hiện front-end OpenVINS bằng OpenCV để trực quan hóa; front-end thật chạy bên trong OpenVINS ở Stage 7.</div>"""

nav = "".join(f'<a href="#s{i}"><span>{i}</span>{t}</a>' for i, t in enumerate(
    ["Input", "Rectification", "Features", "IGEV++", "Depth", "Point cloud", "VIO", "Fusion", "Evaluation"], 1))

html = open(os.path.join(os.path.dirname(__file__), "report_template.html"), encoding="utf-8").read()
html = (html.replace("{{TITLE}}", "EuRoC Depth–VIO Pipeline" if SEQ == "V1_01_easy" else f"Depth–VIO {SEQ}").replace("{{NAV}}", nav).replace("{{INTRO}}", intro).replace("{{STAGES}}", "".join(S))
        .replace("{{PAYLOAD}}", json.dumps(payload)).replace("{{DEPTH_PNG}}", depth_png).replace("{{LEFT_JPG}}", left_jpg)
        .replace("{{SEQ}}", SEQ).replace("{{FRAME}}", str(s1["frame_index"])))
out = os.path.join(ROOT, "report", "pipeline_report.html" if SEQ == "V1_01_easy" else f"pipeline_report_{SEQ}.html")
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w", encoding="utf-8").write(html)
print(out, f"{os.path.getsize(out)/1e6:.1f} MB")
