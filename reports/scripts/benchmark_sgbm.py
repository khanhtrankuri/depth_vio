"""Fresh CPU SGBM benchmark inside existing offline OpenVINS image; no installs."""
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
import yaml

root, dest = Path('/workspace'), Path('/out')
results = dict(opencv=cv2.__version__, numpy=np.__version__,
               backend='CPU, existing OpenVINS Docker image, default OpenCV threads',
               opencv_threads=cv2.getNumThreads(), warmup=5, repeats=30, sequences={})
for seq in ['V1_01_easy', 'V1_02_medium', 'V1_03_difficult']:
    mav = root/'data'/seq/'mav0'
    sensors = [yaml.safe_load((mav/c/'sensor.yaml').read_text()) for c in ['cam0', 'cam1']]
    K, dist, Tb = [], [], []
    for s in sensors:
        fx, fy, cx, cy = s['intrinsics']
        K.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=float))
        dist.append(np.array(s['distortion_coefficients']))
        Tb.append(np.array(s['T_BS']['data']).reshape(4, 4))
    rel = np.linalg.inv(Tb[1]) @ Tb[0]
    size = tuple(sensors[0]['resolution'])
    r0,r1,p0,p1,_,_,_ = cv2.stereoRectify(K[0],dist[0],K[1],dist[1],size,rel[:3,:3],rel[:3,3],flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    maps = [cv2.initUndistortRectifyMap(k,d,r,p,size,cv2.CV_32FC1) for k,d,r,p in zip(K,dist,[r0,r1],[p0,p1])]
    frames=[]
    for cam in ['cam0', 'cam1']:
        with (mav/cam/'data.csv').open() as f:
            frames.append({int(r[0]):r[1] for r in csv.reader(f) if r and not r[0].startswith('#')})
    ts=sorted(set(frames[0]) & set(frames[1]))[700]
    raw=[cv2.imread(str(mav/c/'data'/fr[ts]),cv2.IMREAD_UNCHANGED) for c,fr in zip(['cam0','cam1'],frames)]
    def prep(): return [cv2.remap(im,mp[0],mp[1],cv2.INTER_LINEAR) for im,mp in zip(raw,maps)]
    left,right=prep()
    matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=96,blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    for _ in range(5): matcher.compute(left,right)
    tp, ti, tt = [], [], []
    for _ in range(30):
        t=time.perf_counter(); l,r=prep(); t1=time.perf_counter()
        disp=matcher.compute(l,r).astype(np.float32)/16.; t2=time.perf_counter()
        valid=disp>.5; depth=np.full_like(disp,np.nan)
        depth[valid]=(-p1[0,3])/disp[valid]; depth[np.nan_to_num(depth)>10]=np.nan
        t3=time.perf_counter()
        tp.append((t1-t)*1000); ti.append((t2-t1)*1000); tt.append((t3-t)*1000)
    np.save(str(dest/(seq+'_sgbm_fresh_disp.npy')),disp)
    old=np.load(str(root/'outputs'/seq/'stage4_disparity/disp_sgbm.npy'))
    imgold=cv2.imread(str(root/'outputs'/seq/'stage2_rectify/left_rect.png'),cv2.IMREAD_UNCHANGED)
    results['sequences'][seq]=dict(timestamp_ns=ts, preprocess_ms_mean=float(np.mean(tp)),
        compute_ms_mean=float(np.mean(ti)),compute_ms_p50=float(np.percentile(ti,50)),compute_ms_p90=float(np.percentile(ti,90)),
        total_rectify_compute_depth_ms_mean=float(np.mean(tt)),
        timing_scope='in-memory images; excludes disk decode, calibration-map setup, VIO, mapping',
        disparity_valid_frac=float(valid.mean()), fraction_different_from_cached=float(np.mean(old!=disp)),
        max_disparity_difference_px=float(np.max(abs(old-disp))),
        left_rect_max_difference=int(np.max(abs(left.astype(int)-imgold.astype(int)))))
(dest/'sgbm_fresh.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
