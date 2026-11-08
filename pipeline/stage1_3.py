"""Stage 1 (input), Stage 2 (rectification), Stage 3 (feature extraction/tracking).

Stage 3 front-end mirrors OpenVINS TrackKLT logic (grid FAST + pyramidal KLT +
fundamental-matrix RANSAC + stereo KLT), re-implemented with OpenCV for
visualisation. It is NOT the output of the OpenVINS binary.
"""
import json, os, sys, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

FRAME = int(os.environ.get("FRAME", 700))
NTRACK = int(os.environ.get("NTRACK", 40))
timing = {}

pairs, n0, n1 = stereo_pairs()
t_ns, pL, pR = pairs[FRAME]
L, R = read_gray(pL), read_gray(pR)

# ---------------- Stage 1 ----------------
d1 = outdir("stage1_input")
c0 = [t for t, _ in list_frames("cam0")]
dts = np.diff(np.array(c0)) / 1e6
info = dict(frame_index=FRAME, timestamp_ns=t_ns, left=os.path.basename(pL),
            right=os.path.basename(pR), shape=list(L.shape), dtype=str(L.dtype),
            left_min=int(L.min()), left_max=int(L.max()), left_mean=float(L.mean()),
            right_mean=float(R.mean()), n_cam0=n0, n_cam1=n1, n_synced_pairs=len(pairs),
            cam0_dt_ms_median=float(np.median(dts)), cam0_dt_ms_min=float(dts.min()),
            cam0_dt_ms_max=float(dts.max()), sequence=SEQ)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for a, im, nm in zip(ax, (L, R), ("cam0 (left)", "cam1 (right)")):
    a.imshow(im, cmap="gray", vmin=0, vmax=255)
    a.set_title(f"{nm}  t={t_ns} ns  {im.shape[1]}x{im.shape[0]} {im.dtype}")
    a.set_xlabel("u [px]"); a.set_ylabel("v [px]")
plt.tight_layout(); plt.savefig(f"{d1}/original_pair.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(dts, lw=0.6); ax.set_xlabel("cam0 frame index"); ax.set_ylabel("Δt [ms]")
ax.set_title("cam0 inter-frame interval (20 Hz → 50 ms)")
plt.tight_layout(); plt.savefig(f"{d1}/frame_intervals.png", dpi=110); plt.close()
json.dump(info, open(f"{d1}/info.json", "w"), indent=2)
print("Stage1", info)

# ---------------- Stage 2 ----------------
d2 = outdir("stage2_rectify")
t0 = time.perf_counter()
rect = stereo_rectify(alpha=0.0)
Lr, Rr = rectify_pair(rect, L, R)
timing["rectify_ms"] = (time.perf_counter() - t0) * 1e3


def epi_check(a, b):
    """ORB matches L↔R (cross-checked); returns vertical offset |dy|."""
    orb = cv2.ORB_create(2000)
    ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    m = [x for x in m if x.distance < 40]
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    dy = np.abs(pa[:, 1] - pb[:, 1]); dx = pa[:, 0] - pb[:, 0]
    keep = (dy < 20) & (dx > -2)  # discard gross mismatches
    return dy[keep], pa[keep], pb[keep]


dy_raw, _, _ = epi_check(L, R)
dy_rect, pa, pb = epi_check(Lr, Rr)
rinfo = dict(fx_rect=rect["fx"], fy_rect=rect["fy"], cx_rect=rect["cx"], cy_rect=rect["cy"],
             baseline_m=rect["baseline"], T_c1_c0=rect["T_10"].tolist(),
             P0=rect["P0"].tolist(), P1=rect["P1"].tolist(),
             dy_raw_median=float(np.median(dy_raw)), dy_raw_p90=float(np.percentile(dy_raw, 90)),
             dy_rect_median=float(np.median(dy_rect)), dy_rect_p90=float(np.percentile(dy_rect, 90)),
             n_matches_raw=int(len(dy_raw)), n_matches_rect=int(len(dy_rect)))


def epilines(img, step=40):
    c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for y in range(20, img.shape[0], step):
        cv2.line(c, (0, y), (img.shape[1], y), (0, 255, 0), 1)
    return c[..., ::-1]


fig, ax = plt.subplots(2, 2, figsize=(13, 8.6))
for a, im, nm in zip(ax.ravel(), (L, R, Lr, Rr),
                     ("raw cam0", "raw cam1", "rectified cam0", "rectified cam1")):
    a.imshow(epilines(im)); a.set_title(nm + " + horizontal lines"); a.set_xlabel("u [px]"); a.set_ylabel("v [px]")
plt.tight_layout(); plt.savefig(f"{d2}/rectified_epipolar.png", dpi=110); plt.close()
cat = np.hstack([cv2.cvtColor(Lr, cv2.COLOR_GRAY2BGR), cv2.cvtColor(Rr, cv2.COLOR_GRAY2BGR)])
for y in range(20, Lr.shape[0], 40):
    cv2.line(cat, (0, y), (cat.shape[1], y), (0, 200, 0), 1)
rng = np.random.default_rng(0)
for i in rng.choice(len(pa), min(60, len(pa)), replace=False):
    col = tuple(int(c) for c in rng.integers(60, 255, 3))
    cv2.line(cat, tuple(int(v) for v in pa[i]), (int(pb[i][0] + Lr.shape[1]), int(pb[i][1])), col, 1, cv2.LINE_AA)
cv2.imwrite(f"{d2}/rectified_matches.png", cat)
fig, ax = plt.subplots(figsize=(7, 3.2))
bins = np.linspace(0, 16, 65)
ax.hist(dy_raw, bins, alpha=.6, label=f"raw (median {np.median(dy_raw):.2f}px)")
ax.hist(dy_rect, bins, alpha=.6, label=f"rectified (median {np.median(dy_rect):.2f}px)")
ax.set_xlabel("|v_left − v_right| of ORB matches [px]"); ax.set_ylabel("count"); ax.legend()
ax.set_title("Epipolar alignment check")
plt.tight_layout(); plt.savefig(f"{d2}/dy_hist.png", dpi=110); plt.close()
cv2.imwrite(f"{d2}/left_rect.png", Lr); cv2.imwrite(f"{d2}/right_rect.png", Rr)
json.dump(rinfo, open(f"{d2}/rectify.json", "w"), indent=2)
print("Stage2", {k: v for k, v in rinfo.items() if not isinstance(v, list)})

# ---------------- Stage 3 ----------------
# OpenVINS euroc_mav defaults: num_pts=200, fast_threshold=20, grid 5x5, min_px_dist=10,
# histogram equalisation, KLT win 15, pyr 5 levels.
d3 = outdir("stage3_features")
NUM_PTS, FAST_T, GX, GY, MIN_DIST = 200, 20, 5, 5, 10
lk = dict(winSize=(15, 15), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
fast = cv2.FastFeatureDetector_create(FAST_T, True)


def detect(img, existing):
    h, w = img.shape
    mask = np.full((h, w), 255, np.uint8)
    for p in existing:
        cv2.circle(mask, (int(p[0]), int(p[1])), MIN_DIST, 0, -1)
    per = int(np.ceil((NUM_PTS - len(existing)) / (GX * GY)))
    new = []
    if per <= 0:
        return np.zeros((0, 2), np.float32)
    for gy in range(GY):
        for gx in range(GX):
            x0, x1 = gx * w // GX, (gx + 1) * w // GX; y0, y1 = gy * h // GY, (gy + 1) * h // GY
            kps = fast.detect(img[y0:y1, x0:x1], mask[y0:y1, x0:x1])
            kps = sorted(kps, key=lambda k: -k.response)[:per]
            new += [(k.pt[0] + x0, k.pt[1] + y0) for k in kps]
    return np.float32(new).reshape(-1, 2)


def klt(a, b, p):
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **lk)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **lk)
    fb = np.linalg.norm(p - pb, axis=1)
    h, w = a.shape
    inb = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    ok_lk = (st.ravel() == 1) & (st2.ravel() == 1) & inb
    ok_fb = fb < 1.0
    return q, ok_lk, ok_fb


def undist(p, K, D):
    return cv2.undistortPoints(p.reshape(-1, 1, 2), K, D).reshape(-1, 2)


K0, D0, K1, D1 = rect["K0"], rect["D0"], rect["K1"], rect["D1"]
clahe = cv2.createCLAHE(2.0, (8, 8))
ids = np.zeros(0, int); pts = np.zeros((0, 2), np.float32); next_id = 0
tracks = {}
stats = []
prev = None
t_feat = []
seq = [read_gray(pairs[FRAME - NTRACK + 1 + k][1]) for k in range(NTRACK)]
seqR = [read_gray(pairs[FRAME - NTRACK + 1 + k][2]) for k in range(NTRACK)]
for k, img in enumerate(seq):
    t0 = time.perf_counter()
    cur = clahe.apply(img)
    rej = dict(lk=np.zeros((0, 2)), fb=np.zeros((0, 2)), ransac=np.zeros((0, 2)))
    if prev is not None and len(pts):
        q, ok_lk, ok_fb = klt(prev, cur, pts)
        ok = ok_lk & ok_fb
        rej["lk"] = q[~ok_lk]; rej["fb"] = q[ok_lk & ~ok_fb]
        inl = np.zeros(len(pts), bool)
        if ok.sum() >= 8:
            a = undist(pts[ok], K0, D0); b = undist(q[ok], K0, D0)
            _, m = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.0 / K0[0, 0], 0.999)
            inl[np.where(ok)[0]] = m.ravel().astype(bool)
        rej["ransac"] = q[ok & ~inl]
        n_tracked = int(inl.sum())
        ids, pts = ids[inl], q[inl]
    else:
        n_tracked = 0
    newp = detect(cur, pts)
    ids = np.concatenate([ids, np.arange(next_id, next_id + len(newp))]); next_id += len(newp)
    pts = np.vstack([pts, newp]).astype(np.float32)
    curR = clahe.apply(seqR[k])
    qr, ok_lk_r, ok_fb_r = klt(cur, curR, pts)
    okr = ok_lk_r & ok_fb_r
    if okr.sum():
        ul = cv2.undistortPoints(pts.reshape(-1, 1, 2), K0, D0, R=rect["R0"], P=rect["P0"]).reshape(-1, 2)
        ur = cv2.undistortPoints(qr.reshape(-1, 1, 2), K1, D1, R=rect["R1"], P=rect["P1"]).reshape(-1, 2)
        okr &= (np.abs(ul[:, 1] - ur[:, 1]) < 2.0) & (ul[:, 0] - ur[:, 0] > 0)
    t_feat.append((time.perf_counter() - t0) * 1e3)
    for i, p in zip(ids, pts):
        tracks.setdefault(int(i), []).append((k, float(p[0]), float(p[1])))
    stats.append(dict(k=k, tracked=n_tracked, new=int(len(newp)), total=int(len(pts)),
                      stereo_ok=int(okr.sum()), rej_lk=int(len(rej["lk"])), rej_fb=int(len(rej["fb"])),
                      rej_ransac=int(len(rej["ransac"]))))
    prev = cur; rej_last = rej; okr_last, qr_last = okr, qr
timing["feature_track_ms_per_frame"] = float(np.mean(t_feat[1:]))

vis = cv2.cvtColor(seq[-1], cv2.COLOR_GRAY2BGR)
for p, s in zip(pts, okr_last):
    cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 220, 0) if s else (0, 200, 255), -1)
for key, col in (("lk", (0, 0, 255)), ("fb", (255, 0, 255)), ("ransac", (255, 80, 0))):
    for p in rej_last[key]:
        cv2.drawMarker(vis, (int(p[0]), int(p[1])), col, cv2.MARKER_TILTED_CROSS, 9, 2)
cv2.imwrite(f"{d3}/keypoints.png", vis)
tv = cv2.cvtColor(seq[-1], cv2.COLOR_GRAY2BGR)
cmap = plt.get_cmap("turbo")
alive = set(int(i) for i in ids)
lengths = []
for tid, tr in tracks.items():
    lengths.append(len(tr))
    if tid not in alive or len(tr) < 2:
        continue
    for (k1, x1, y1), (k2, x2, y2) in zip(tr[:-1], tr[1:]):
        c = np.array(cmap(k2 / NTRACK)[:3])[::-1] * 255
        cv2.line(tv, (int(x1), int(y1)), (int(x2), int(y2)), tuple(int(v) for v in c), 1, cv2.LINE_AA)
    cv2.circle(tv, (int(tr[-1][1]), int(tr[-1][2])), 2, (255, 255, 255), -1)
cv2.imwrite(f"{d3}/tracks.png", tv)
sv = np.hstack([cv2.cvtColor(seq[-1], cv2.COLOR_GRAY2BGR), cv2.cvtColor(seqR[-1], cv2.COLOR_GRAY2BGR)])
for p, q, s in zip(pts, qr_last, okr_last):
    if s:
        cv2.line(sv, (int(p[0]), int(p[1])), (int(q[0]) + 752, int(q[1])), (0, 220, 0), 1, cv2.LINE_AA)
cv2.imwrite(f"{d3}/stereo_klt.png", sv)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ks = [s["k"] for s in stats]
ax[0].stackplot(ks, [s["tracked"] for s in stats], [s["new"] for s in stats], labels=["tracked", "newly detected"], alpha=.8)
ax[0].plot(ks, [s["stereo_ok"] for s in stats], "k-", lw=1.2, label="stereo-matched")
ax[0].plot(ks, [s["rej_lk"] + s["rej_fb"] + s["rej_ransac"] for s in stats], "r--", lw=1, label="rejected")
ax[0].set_xlabel(f"frame (k=0 ↔ index {FRAME-NTRACK+1})"); ax[0].set_ylabel("# features"); ax[0].legend(fontsize=8)
ax[1].hist(lengths, bins=range(1, NTRACK + 2)); ax[1].set_xlabel("track length [frames]"); ax[1].set_ylabel("# tracks"); ax[1].set_yscale("log")
plt.tight_layout(); plt.savefig(f"{d3}/feature_stats.png", dpi=110); plt.close()
fstats = dict(per_frame=stats, n_tracks=len(tracks), mean_track_len=float(np.mean(lengths)),
              params=dict(num_pts=NUM_PTS, fast_threshold=FAST_T, grid=[GX, GY], min_px_dist=MIN_DIST,
                          klt_win=15, pyr_levels=5, fb_thresh_px=1.0, ransac_thresh_px=1.0))
json.dump(fstats, open(f"{d3}/features.json", "w"), indent=2)
json.dump(timing, open(f"{d3}/timing.json", "w"), indent=2)
print("Stage3 last frame", stats[-1], "tracks", len(tracks), "timing", timing)
