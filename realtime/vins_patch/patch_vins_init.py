#!/usr/bin/env python3
"""Fix the stereo+IMU initialisation of VINS-Fusion-ROS2-Humble (also present in upstream VINS-Fusion).

Upstream copies Rs[i]/Ps[i] onto *every* entry of all_image_frame and then solves the gyroscope bias from
consecutive all_image_frame pairs. While the platform is static, non-keyframes are marginalised with
MARGIN_SECOND_NEW and stay in all_image_frame, so that map holds more frames than the window: poses are
assigned to the wrong frames and Rs[] is read past its end (WINDOW_SIZE + 1 entries). The resulting gyro bias
is random; at real-time rate it occasionally explodes ("numerical unstable in preintegration") and the
estimate diverges by kilometres (seen at the static start of EuRoC V1_01 in 2 of ~45 runs).

Fix: solve the gyro bias from the window itself (Rs[i], Rs[i+1] and pre_integrations[i+1], which VINS keeps
merged across dropped non-keyframes), and refuse to initialise (slide and retry) if the correction is
implausibly large. Idempotent; fails loudly if the upstream source changed.
"""
import pathlib
import sys

est = pathlib.Path(sys.argv[1]) / "vins" / "src" / "estimator" / "estimator.cpp"
OLD = """                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();"""
NEW = """                // [init-fix] gyro bias from the window (see realtime/vins_patch/patch_vins_init.py)
                Matrix3d bg_A = Matrix3d::Zero();
                Vector3d bg_b = Vector3d::Zero();
                for (int i = 0; i < WINDOW_SIZE; i++)
                {
                    Quaterniond q_ij(Rs[i].transpose() * Rs[i + 1]);
                    Matrix3d J = pre_integrations[i + 1]->jacobian.block<3, 3>(O_R, O_BG);
                    Vector3d r = 2 * (pre_integrations[i + 1]->delta_q.inverse() * q_ij).vec();
                    bg_A += J.transpose() * J;
                    bg_b += J.transpose() * r;
                }
                Vector3d delta_bg = bg_A.ldlt().solve(bg_b);
                if (!delta_bg.allFinite() || delta_bg.norm() > 0.1)
                {
                    ROS_WARN("init rejected: implausible gyro bias correction %f rad/s, retrying", delta_bg.norm());
                    slideWindow();
                    return;
                }
                ROS_INFO("init gyro bias correction %f %f %f", delta_bg.x(), delta_bg.y(), delta_bg.z());
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    Bgs[i] += delta_bg;
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();"""
text = est.read_text()
if NEW not in text:
    if OLD not in text:
        sys.exit("patch_vins_init: anchor not found (upstream changed?)")
    text = text.replace(OLD, NEW, 1)
    est.write_text(text)
print("patched vins init")
