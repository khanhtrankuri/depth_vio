"""Stage 7 visualisation + Stage 9 evaluation.

python stage9_eval.py vio  <est_tum>          → ATE/RPE (evo, SE3 Umeyama alignment), trajectory plots
python stage9_eval.py recon <tag> [<tag2>...]  → TSDF / naive clouds vs Leica scan (accuracy / completeness)
"""
import json, os, sys
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

d9 = outdir("stage9_eval")


def write_gt_tum():
    t, T = load_gt()
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(T[:, :3, :3]).as_quat()  # x y z w
    p = os.path.join(d9, "gt_body_tum.txt")
    np.savetxt(p, np.column_stack([t / 1e9, T[:, :3, 3], q]), fmt="%.9f")
    return p


def vio(est):
    from evo.tools import file_interface
    from evo.core import sync, metrics
    from evo.core.trajectory import PoseTrajectory3D
    gt_p = write_gt_tum()
    ref = file_interface.read_tum_trajectory_file(gt_p)
    e = file_interface.read_tum_trajectory_file(est)
    ref, e = sync.associate_trajectories(ref, e, max_diff=0.01)
    e_al = PoseTrajectory3D(poses_se3=e.poses_se3, timestamps=e.timestamps)
    r, tvec, s = e_al.align(ref, correct_scale=False)
    ate = metrics.APE(metrics.PoseRelation.translation_part); ate.process_data((ref, e_al))
    ater = metrics.APE(metrics.PoseRelation.rotation_angle_deg); ater.process_data((ref, e_al))
    out = dict(est=est, n_matched=len(e.timestamps), duration_s=float(e.timestamps[-1] - e.timestamps[0]),
               path_length_m=float(ref.path_length),
               ATE_trans_rmse_m=ate.get_statistic(metrics.StatisticsType.rmse),
               ATE_trans_mean_m=ate.get_statistic(metrics.StatisticsType.mean),
               ATE_trans_max_m=ate.get_statistic(metrics.StatisticsType.max),
               ATE_rot_rmse_deg=ater.get_statistic(metrics.StatisticsType.rmse),
               align_R=np.asarray(r).tolist(), align_t=np.asarray(tvec).tolist())
    for delta in (1.0, 5.0):
        rp = metrics.RPE(metrics.PoseRelation.translation_part, delta=delta, delta_unit=metrics.Unit.meters, all_pairs=True)
        rp.process_data((ref, e_al))
        rr = metrics.RPE(metrics.PoseRelation.rotation_angle_deg, delta=delta, delta_unit=metrics.Unit.meters, all_pairs=True)
        rr.process_data((ref, e_al))
        out[f"RPE_trans_rmse_m@{delta:g}m"] = rp.get_statistic(metrics.StatisticsType.rmse)
        out[f"RPE_rot_rmse_deg@{delta:g}m"] = rr.get_statistic(metrics.StatisticsType.rmse)
    from scipy.spatial.transform import Rotation as Rot
    P = np.array(e_al.poses_se3)
    np.savetxt(os.path.join(d9, "vio_aligned_tum.txt"),
               np.column_stack([e_al.timestamps, P[:, :3, 3], Rot.from_matrix(P[:, :3, :3]).as_quat()]), fmt="%.9f")
    pr = ref.positions_xyz; pe = e_al.positions_xyz; err = ate.error
    np.save(os.path.join(d9, "traj_plot.npy"), np.column_stack([np.array(e.timestamps), pr, pe, err]))
    fig = plt.figure(figsize=(14, 4.6))
    a = fig.add_subplot(1, 3, 1, projection="3d")
    a.plot(*pr.T, "k-", lw=1, label="ground truth (Vicon)"); a.plot(*pe.T, "C1-", lw=1, label="OpenVINS (aligned)")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_zlabel("z [m]"); a.legend(fontsize=8); a.set_title("3D trajectory")
    a = fig.add_subplot(1, 3, 2)
    sc = a.scatter(pe[:, 0], pe[:, 1], c=err, s=2, cmap="viridis"); a.plot(pr[:, 0], pr[:, 1], "k-", lw=.5)
    a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_title("top view, colour = ATE")
    plt.colorbar(sc, ax=a, label="translation error [m]")
    a = fig.add_subplot(1, 3, 3)
    a.plot(np.array(e.timestamps) - e.timestamps[0], err, lw=.8); a.set_xlabel("time [s]"); a.set_ylabel("ATE [m]")
    a.set_title(f"ATE RMSE {out['ATE_trans_rmse_m']:.3f} m")
    plt.tight_layout(); plt.savefig(os.path.join(d9, "trajectory_vs_gt.png"), dpi=110); plt.close()
    json.dump(out, open(os.path.join(d9, "vio_metrics.json"), "w"), indent=2)
    print({k: v for k, v in out.items() if not k.startswith("align")})


def recon(tags):
    ref = o3d.io.read_point_cloud(os.path.join(MAV, "pointcloud0", "data.ply")).voxel_down_sample(0.02)
    d8 = os.path.join(OUT, "stage8_fusion")
    res = {}
    for tag in tags:
        for kind in ("tsdf_cloud", "naive_accum"):
            f = os.path.join(d8, f"{kind}_{tag}.ply")
            if not os.path.exists(f):
                continue
            pc = o3d.io.read_point_cloud(f).voxel_down_sample(0.02)
            acc = np.asarray(pc.compute_point_cloud_distance(ref))
            bb = pc.get_axis_aligned_bounding_box(); bb = o3d.geometry.AxisAlignedBoundingBox(bb.min_bound - .1, bb.max_bound + .1)
            refc = ref.crop(bb)
            comp = np.asarray(refc.compute_point_cloud_distance(pc))
            res[f"{kind}_{tag}"] = dict(n=len(acc), acc_mean_m=float(acc.mean()), acc_median_m=float(np.median(acc)),
                                       acc_rmse_m=float(np.sqrt((acc ** 2).mean())), frac_within_5cm=float((acc < .05).mean()),
                                       comp_median_m=float(np.median(comp)), completeness_5cm=float((comp < .05).mean()))
            np.save(os.path.join(d9, f"acc_{kind}_{tag}.npy"), acc.astype(np.float32))
    json.dump(res, open(os.path.join(d9, "recon_metrics.json"), "w"), indent=2)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    {"vio": lambda: vio(sys.argv[2]), "recon": lambda: recon(sys.argv[2:]), "gt": write_gt_tum}[sys.argv[1]]()
