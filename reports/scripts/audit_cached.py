"""Independent, read-only evaluation of existing artifacts; writes only reports/.

Requires numpy, scipy, matplotlib, pyyaml. No production imports, inference,
alignment fitting to geometry, downloads, training, or environment mutations.
"""
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'reports'
SEQS = ['V1_01_easy', 'V1_02_medium', 'V1_03_difficult']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def dump(name, obj):
    (DEST / 'artifacts' / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def J(p):
    return json.loads(p.read_text(encoding='utf-8'))


def pose(data, gt=False):
    if gt:
        q = data[:, [5, 6, 7, 4]]
    else:
        q = data[:, 4:8]
    T = np.tile(np.eye(4), (len(data), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    T[:, :3, 3] = data[:, 1:4]
    return T


def nearest(t, query):
    i = np.searchsorted(t, query)
    lo, hi = np.clip(i-1, 0, len(t)-1), np.clip(i, 0, len(t)-1)
    j = np.where(abs(t[lo]-query) <= abs(t[hi]-query), lo, hi)
    return j, abs(t[j]-query)


def se3_align(x, y):
    xc, yc = x-x.mean(0), y-y.mean(0)
    u, _, vt = np.linalg.svd(yc.T @ xc / len(x))
    d = np.eye(3); d[2, 2] = np.linalg.det(u @ vt)
    r = u @ d @ vt
    return r, y.mean(0)-r @ x.mean(0)


def sensor(mav, name):
    return yaml.safe_load((mav / name / 'sensor.yaml').read_text())


def ply(path):
    types = {'float': '<f4', 'double': '<f8', 'uchar': 'u1'}
    with path.open('rb') as f:
        props = []; fmt = None; n = 0
        while True:
            s = f.readline().decode('ascii').strip()
            if s.startswith('format '): fmt = s.split()[1]
            if s.startswith('element vertex '): n = int(s.split()[2])
            if s.startswith('property ') and not s.startswith('property list'):
                _, typ, name = s.split(); props.append((name, types[typ]))
            if s == 'end_header': break
        if fmt == 'ascii':
            return np.loadtxt(f, usecols=(0, 1, 2), max_rows=n)
        a = np.fromfile(f, dtype=np.dtype(props), count=n)
        return np.column_stack([a[k] for k in ('x', 'y', 'z')])


def voxel(p, size=.02):
    # Open3D legacy origin convention; average all points within each voxel.
    key = np.floor((p-(p.min(0)-size*.5))/size).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return np.column_stack([np.bincount(inv, weights=p[:, k])/cnt for k in range(3)])


def depth_metrics(z, ref, mask, denom):
    n = int(mask.sum())
    if not n: return {'n': 0, 'coverage_gt': 0}
    e = z[mask].astype(float)-ref[mask]
    ae = abs(e)
    return dict(n=n, coverage_gt=n/int(denom), MAE_m=float(ae.mean()),
                RMSE_m=float(np.sqrt(np.mean(e*e))), AbsRel=float(np.mean(ae/ref[mask])),
                signed_mean_m=float(e.mean()), q50_m=float(np.median(ae)),
                q90_m=float(np.quantile(ae, .90)), q95_m=float(np.quantile(ae, .95)),
                q99_m=float(np.quantile(ae, .99)))


def project(scan, T_WC, intr, shape):
    T = np.linalg.inv(T_WC)
    p = scan @ T[:3, :3].T + T[:3, 3]
    p = p[(p[:, 2] > .1) & (p[:, 2] < 20)]
    fx, fy, cx, cy = intr
    u, v = fx*p[:, 0]/p[:, 2]+cx, fy*p[:, 1]/p[:, 2]+cy
    h, w = shape
    z = np.full(h*w, np.inf, np.float32)
    for du in (0, 1):
        for dv in (0, 1):
            ui, vi = (u+du-.5).astype(int), (v+dv-.5).astype(int)
            k = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            np.minimum.at(z, vi[k]*w+ui[k], p[k, 2].astype(np.float32))
    z[~np.isfinite(z)] = np.nan
    return z.reshape(shape)


def inventory(seq):
    mav = ROOT / 'data' / seq / 'mav0'
    out = {}
    ts = {}
    for name in ['cam0', 'cam1', 'imu0', 'vicon0', 'state_groundtruth_estimate0']:
        with (mav/name/'data.csv').open() as f:
            rows = [r for r in csv.reader(f) if r and not r[0].startswith('#')]
        t = np.array([int(r[0]) for r in rows], dtype=np.int64); ts[name] = t
        dt = np.diff(t)/1e9
        out[name] = dict(n=len(t), duration_s=float((t[-1]-t[0])/1e9),
                         first_ns=int(t[0]), last_ns=int(t[-1]),
                         median_hz=float(1/np.median(dt)), max_gap_s=float(dt.max()),
                         duplicate_or_reversed=int((dt <= 0).sum()),
                         sensor_yaml_sha256=sha(mav/name/'sensor.yaml'))
        if name.startswith('cam'):
            out[name]['missing_images'] = sum(not (mav/name/'data'/r[1]).is_file() for r in rows)
            out[name]['files_on_disk'] = len(list((mav/name/'data').glob('*.png')))
    out['stereo_intersection'] = len(np.intersect1d(ts['cam0'], ts['cam1']))
    out['same_timestamp_order'] = bool(np.array_equal(ts['cam0'], ts['cam1']))
    out['leica'] = dict(sha256=sha(mav/'pointcloud0'/'data.ply'),
                        bytes=(mav/'pointcloud0'/'data.ply').stat().st_size,
                        calibration=sensor(mav, 'pointcloud0'))
    return out, ts


def vio(seq, source=None):
    op = ROOT/'outputs'/seq
    vp = Path(source) if source else op/'stage7_vio'
    gt = np.loadtxt(ROOT/'data'/seq/'mav0/state_groundtruth_estimate0/data.csv', delimiter=',')
    trajectory = vp/'openvins_traj_tum.txt'
    if not trajectory.exists(): trajectory = vp/'openvins_traj.txt'
    e = np.loadtxt(trajectory)[:, :8]
    j, dt = nearest(gt[:, 0]/1e9, e[:, 0])
    # EuRoC reference is dense enough that nearest matching is one-to-one here.
    k = dt <= .01; e = e[k]; j = j[k]; dt = dt[k]
    assert len(np.unique(j)) == len(j), 'Association requires one-to-one conflict resolution'
    T, G = pose(e), pose(gt[j], gt=True)
    r, t = se3_align(T[:, :3, 3], G[:, :3, 3])
    T[:, :3, :3] = r @ T[:, :3, :3]; T[:, :3, 3] = T[:, :3, 3] @ r.T+t
    er = np.linalg.norm(T[:, :3, 3]-G[:, :3, 3], axis=1)
    ori = Rotation.from_matrix(np.transpose(G[:, :3, :3], (0, 2, 1)) @ T[:, :3, :3]).magnitude()*180/np.pi
    raw = np.loadtxt(vp/'openvins_traj.txt')
    cov = {}
    for label, start in [('orientation', 8), ('position', 14)]:
        a = raw[:, start:start+6]; C = np.zeros((len(a), 3, 3))
        C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 1, 1], C[:, 1, 2], C[:, 2, 2] = a.T
        C[:, 1, 0], C[:, 2, 0], C[:, 2, 1] = C[:, 0, 1], C[:, 0, 2], C[:, 1, 2]
        cov[label] = dict(min_eigenvalue=float(np.linalg.eigvalsh(C).min()),
                          median_diagonal=np.median(np.diagonal(C, axis1=1, axis2=2), axis=0).tolist())
    timing = np.loadtxt(vp/'openvins_timing.txt', delimiter=',')
    old = J(op/'stage9_eval/vio_metrics.json')
    result = dict(n_matched=len(e), unique_gt=len(np.unique(j)),
                  ATE_rmse_m=float(np.sqrt(np.mean(er**2))),
                  ATE_max_m=float(er.max()), ATE_rotation_rmse_deg=float(np.sqrt(np.mean(ori**2))),
                  alignment='SE3, all associated positions, scale fixed to 1',
                  delta_from_saved_ATE_m=float(np.sqrt(np.mean(er**2))-old['ATE_trans_rmse_m']),
                  association_max_ms=float(dt.max()*1000),
                  first_estimate_delay_s=float(e[0, 0]-gt[0, 0]/1e9),
                  estimate_max_gap_s=float(np.diff(e[:, 0]).max()),
                  duration_s=float(e[-1, 0]-e[0, 0]),
                  path_length_m=float(np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1).sum()),
                  timing_total_ms_mean=float(timing[:, -1].mean()*1000),
                  timing_total_ms_p90=float(np.quantile(timing[:, -1], .90)*1000),
                  covariance_blocks=cov, raw_export_columns=raw.shape[1],
                  rpe={})
    # Fixed one-second temporal RPE, distinct from legacy evo spatial 1m/5m RPE.
    ii, gap = nearest(e[:, 0], e[:, 0]+1)
    a = np.where((gap <= .03) & (ii > np.arange(len(e))))[0]; b = ii[a]
    de = np.linalg.inv(T[a]) @ T[b]; dg = np.linalg.inv(G[a]) @ G[b]
    rel = np.linalg.inv(dg) @ de
    result['rpe']['1s'] = dict(n=len(a), trans_rmse_m=float(np.sqrt(np.mean(np.sum(rel[:, :3, 3]**2, axis=1)))),
                              rot_rmse_deg=float(np.sqrt(np.mean(Rotation.from_matrix(rel[:, :3, :3]).magnitude()**2))*180/np.pi))
    # Relative endpoint error (whole matched segment); not percent drift over full dataset.
    end = np.linalg.inv(np.linalg.inv(G[0])@G[-1]) @ (np.linalg.inv(T[0])@T[-1])
    result['endpoint_relative_translation_m'] = float(np.linalg.norm(end[:3, 3]))
    plt.figure(figsize=(9, 3)); plt.plot(e[:, 0]-e[0, 0], er)
    plt.xlabel('Time after first matched pose [s]'); plt.ylabel('SE3-aligned ATE [m]')
    suffix = '_rerun' if source else ''
    plt.title(seq+suffix); plt.tight_layout(); plt.savefig(DEST/'figures'/f'{seq}_ate{suffix}.png'); plt.close()
    return result


def depth(seq, scan):
    op = ROOT/'outputs'/seq; mav = ROOT/'data'/seq/'mav0'
    summary = J(op/'single_frame_summary.json'); rect = J(op/'stage2_rectify/rectify.json')
    z = np.load(op/'stage5_depth/depth_igev_m.npy')
    d = np.load(op/'stage4_disparity/disp_sgbm.npy')
    # alpha=0 remap support is full image; verify this against saved native mask ratio.
    vs = d > .5
    assert abs(vs.mean()-summary['stage4']['sgbm_valid_frac']) < 1e-12
    zs = np.full_like(d, np.nan); zs[vs] = rect['fx_rect']*rect['baseline_m']/d[vs]; zs[zs > 10] = np.nan
    ds = np.load(op/'stage4_disparity/disp_igev.npy')
    depth_formula_error = float(np.max(abs(z[np.isfinite(z)]-rect['fx_rect']*rect['baseline_m']/ds[np.isfinite(z)])))
    T_B0 = np.array(sensor(mav, 'cam0')['T_BS']['data']).reshape(4, 4)
    T_B1 = np.array(sensor(mav, 'cam1')['T_BS']['data']).reshape(4, 4)
    T10 = np.linalg.inv(T_B1)@T_B0
    half = Rotation.from_rotvec(-.5*Rotation.from_matrix(T10[:3, :3]).as_rotvec()).as_matrix()
    tv = half @ T10[:3, 3]; unit = np.array([np.sign(tv[0]), 0, 0]); axis = np.cross(tv, unit)
    angle = np.arccos(abs(tv[0])/np.linalg.norm(tv))
    wr = Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle).as_matrix()
    R0 = wr @ half.T
    ext = T_B0.copy(); ext[:3, :3] = T_B0[:3, :3]@R0.T
    original_pose = np.array(summary['gt_pose_T_W_B'])
    independent = project(scan, original_pose@ext,
                          [rect[k] for k in ['fx_rect', 'fy_rect', 'cx_rect', 'cy_rect']], z.shape)
    ref = np.load(op/'stage5_depth/depth_leica_ref_m.npy')
    both = np.isfinite(independent)&np.isfinite(ref)
    verification = dict(depth_formula_max_difference_m=depth_formula_error,
                        reprojection_mask_disagreement=int(np.count_nonzero(np.isfinite(independent)!=np.isfinite(ref))),
                        reprojection_max_difference_m=float(np.max(abs(independent[both]-ref[both]))),
                        fx=rect['fx_rect'], baseline_m=rect['baseline_m'])
    results = {'geometry_recomputation': verification, 'references': {}}
    for refname, refpath in [('original', op/'stage5_depth/depth_leica_ref_m.npy'),
                              ('vio_corrected_DIAGNOSTIC', op/'stage5_depth_diag_gtrotcorr/depth_leica_ref_m.npy')]:
        ref = np.load(refpath); gm = np.isfinite(ref)&(ref >= .2)&(ref <= 10)
        masks = {'IGEV++': np.isfinite(z)&(z >= .2)&(z <= 10), 'SGBM': np.isfinite(zs)&(zs >= .2)&(zs <= 10)}
        inter = gm&masks['IGEV++']&masks['SGBM']
        rr = dict(gt_valid_n=int(gm.sum()), gt_image_coverage=float(gm.mean()), models={})
        fig, ax = plt.subplots(2, 3, figsize=(14, 7))
        image = plt.imread(op/'stage2_rectify/left_rect.png')
        ax[0, 0].imshow(image, cmap='gray'); ax[0, 0].set_title('Rectified left, frame 700')
        im = ax[0, 1].imshow(ref, vmin=.2, vmax=6, cmap='turbo_r'); ax[0, 1].set_title(refname+' reference [m]')
        ax[0, 2].imshow(z, vmin=.2, vmax=6, cmap='turbo_r'); ax[0, 2].set_title('Cached IGEV++ [m]')
        for name, pred in [('IGEV++', z), ('SGBM', zs)]:
            native = gm&masks[name]
            bins = []
            for lo, hi in zip([.2, 1, 2, 3, 4, 6], [1, 2, 3, 4, 6, 10]):
                bins.append(dict(lo=lo, hi=hi, **depth_metrics(pred, ref, inter&(ref >= lo)&(ref < hi), gm.sum())))
            rr['models'][name] = dict(native_image_valid_ratio=float(masks[name].mean()),
                gt_mask_requested_n=int(gm.sum()), gt_mask_missing_n=int((gm&~masks[name]).sum()),
                gt_mask_score='undefined for missing predictions; no zero-error imputation',
                native=depth_metrics(pred, ref, native, gm.sum()),
                intersection=depth_metrics(pred, ref, inter, gm.sum()), bins_intersection=bins)
            err = abs(pred[inter]-ref[inter]); sortederr = np.sort(err)
            ax[1, 2].plot(sortederr, np.arange(1, len(err)+1)/len(err), label=name)
        ax[1, 0].imshow(zs, vmin=.2, vmax=6, cmap='turbo_r'); ax[1, 0].set_title('Cached SGBM [m]')
        ax[1, 1].imshow(np.where(inter, abs(z-ref), np.nan), vmin=0, vmax=.5, cmap='magma')
        ax[1, 1].set_title('IGEV++ error on common mask [m]')
        ax[1, 2].set_xlim(0, 1); ax[1, 2].legend(); ax[1, 2].set_xlabel('Common-mask absolute error [m]'); ax[1, 2].set_ylabel('CDF')
        fig.colorbar(im, ax=ax[0, :2], shrink=.7)
        fig.suptitle(seq+' | '+refname+' | reference registration not independently certified')
        fig.subplots_adjust(hspace=.35, wspace=.25, top=.88)
        fig.savefig(DEST/'figures'/f'{seq}_depth_{refname}.png', dpi=115); plt.close(fig)
        results['references'][refname] = rr
    return results


def mapping(seq, scan):
    op = ROOT/'outputs'/seq
    ref = voxel(scan); rt = cKDTree(ref)
    fixed = ref[((ref >= [-4.8, -4.8, -.6])&(ref <= [4.8, 4.8, 3.6])).all(1)]
    results = {}
    for tag in ['openvins', 'gt', 'gtrotcorr']:
        for kind in ['tsdf_cloud', 'naive_accum']:
            key = kind+'_'+tag
            p = voxel(ply(op/'stage8_fusion'/f'{key}.ply'))
            acc = rt.query(p, workers=2)[0]; tree = cKDTree(p)
            crop = ref[((ref >= p.min(0)-.1)&(ref <= p.max(0)+.1)).all(1)]
            comp = tree.query(crop, workers=2)[0]
            pf = p[((p >= [-4.8, -4.8, -.6])&(p <= [4.8, 4.8, 3.6])).all(1)]
            fixed_comp = cKDTree(pf).query(fixed, workers=2)[0]
            fixed_acc = cKDTree(fixed).query(pf, workers=2)[0]
            old = J(op/'stage9_eval/recon_metrics.json')[key]
            results[key] = dict(n=len(p), accuracy_mean_m=float(acc.mean()), accuracy_rmse_m=float(np.sqrt(np.mean(acc**2))),
                accuracy_median_m=float(np.median(acc)), accuracy_5cm=float(np.mean(acc < .05)),
                legacy_crop_reference_n=len(crop), legacy_completeness_5cm=float(np.mean(comp < .05)),
                legacy_completeness_median_m=float(np.median(comp)),
                fixed_volume_reference_n=len(fixed), fixed_volume_completeness_5cm=float(np.mean(fixed_comp < .05)),
                fixed_volume_prediction_n=len(pf), fixed_volume_accuracy_mean_m=float(fixed_acc.mean()),
                fixed_volume_completeness_mean_m=float(fixed_comp.mean()),
                chamfer_unsquared_fixed_m=float((fixed_acc.mean()+fixed_comp.mean())/2),
                delta_saved_accuracy_mean_m=float(acc.mean()-old['acc_mean_m']))
    return results


def main():
    # Meaningful numerical controls for independent alignment and metric calculations.
    rng = np.random.default_rng(19); p = rng.normal(size=(30, 3))
    r0 = Rotation.from_rotvec([.2, -.3, .4]).as_matrix(); t0 = np.array([1, 2, -1])
    r, t = se3_align(p, p@r0.T+t0)
    assert np.allclose(r, r0) and np.allclose(t, t0) and np.linalg.det(r) > .99999
    controls = depth_metrics(np.array([2., 4.]), np.array([1., 2.]), np.array([True, True]), 2)
    assert controls['MAE_m'] == 1.5 and controls['AbsRel'] == 1.
    versions = dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                    matplotlib=matplotlib.__version__, platform=platform.platform(), numerical_controls='passed')
    dump('environment.json', versions)
    allres = {}; scan_cache = {}
    manifest = {}
    for path in sorted((ROOT/'pipeline').glob('*')):
        if path.is_file(): manifest[str(path.relative_to(ROOT))] = sha(path)
    for path in sorted((ROOT/'IGEV-plusplus/pretrained_models').rglob('*.pth')):
        manifest[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    for seq in SEQS:
        print('AUDIT', seq, flush=True)
        inv, ts = inventory(seq)
        digest = inv['leica']['sha256']
        if digest not in scan_cache: scan_cache[digest] = ply(ROOT/'data'/seq/'mav0/pointcloud0/data.ply')
        scan = scan_cache[digest]; inv['leica']['n_vertices'] = len(scan)
        result = dict(inventory=inv, vio=vio(seq), depth=depth(seq, scan))
        print('MAPPING', seq, flush=True)
        result['mapping'] = mapping(seq, scan)
        cache = sorted((ROOT/'outputs'/seq/'stage8_fusion/disp').glob('*.npz'))
        result['cache'] = dict(n=len(cache), timestamp_step_median_s=float(np.median(np.diff([int(p.stem) for p in cache]))/1e9))
        for path in sorted((ROOT/'outputs'/seq).rglob('*')):
            if path.is_file() and path.suffix in ['.json', '.txt', '.npy'] and 'stage3' not in str(path):
                manifest[str(path.relative_to(ROOT))] = sha(path)
        allres[seq] = result
        dump('audit_results.json', allres)
    dump('input_manifest.json', manifest)
    print('COMPLETE: reports/artifacts/audit_results.json', flush=True)


if __name__ == '__main__':
    main()
