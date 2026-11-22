"""Patch VINS-Fusion-ROS2-Humble (fanhong-li, commit 8993f55) with live learned-depth factors.

    python3 apply_ros2_depth_patch.py <path/to/VINS-Fusion-ROS2-Humble>

Adds (all opt-in through the YAML key `depth_topic`; without it the estimator is unchanged):
  * DepthTopicProvider: buffers 32FC2 [inverse depth, sigma] images from the online depth node,
    matched to the exact image exposure.
  * Metric-depth factors (IgevDepthFactor / IgevAnchorDepthFactor) in the sliding-window Ceres problem
    and in marginalization. At observations with accepted depth the correlated right-camera reprojection
    factor is replaced by the depth factor.
  * Depth-based landmark initialisation for features that have no stereo match yet.
  * Take-off guard (depth only after the estimator has moved depth_start_dist metres, then depth_start_delay s),
    optional depth_keep_stereo, and a consistency gate (factor skipped if it disagrees with the estimate).
Fails loudly if the upstream source does not match the expected anchors. Idempotent: refuses double patching.
"""
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
FACTOR = HERE.parent.parent / 'vins_fusion' / 'depth' / 'igev_depth_factor.h'
MARK = '// LIVE_DEPTH_PATCH'


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f'Unexpected upstream source ({text.count(old)} matches) at: {old[:120]!r}')
    return text.replace(old, new)


def apply(repo):
    src = Path(repo) / 'vins' / 'src'
    est = src / 'estimator' / 'estimator.cpp'
    if MARK in est.read_text():
        raise SystemExit('already patched')
    shutil.copyfile(FACTOR, src / 'factor' / 'igev_depth_factor.h')
    shutil.copyfile(HERE / 'depth_topic_provider.h', src / 'factor' / 'depth_topic_provider.h')

    # ---- parameters -------------------------------------------------------------------------
    p = src / 'estimator' / 'parameters.h'; s = p.read_text()
    s = replace(s, 'extern double TD;', 'extern double TD;\n'
                'extern std::string DEPTH_TOPIC;\nextern double DEPTH_WEIGHT, DEPTH_PERIOD, DEPTH_WAIT_MS, DEPTH_MAX_REL_RANGE, DEPTH_START_DELAY, DEPTH_GATE, DEPTH_START_DIST;\n'
                'extern int DEPTH_INIT_LANDMARKS, DEPTH_KEEP_STEREO;')
    p.write_text(s)
    p = src / 'estimator' / 'parameters.cpp'; s = p.read_text()
    s = replace(s, 'double TD;', 'double TD;\n' + MARK + '\nstd::string DEPTH_TOPIC;\n'
                'double DEPTH_WEIGHT = 16.0, DEPTH_PERIOD = 0.2, DEPTH_WAIT_MS = 80.0, DEPTH_MAX_REL_RANGE = 0.08, DEPTH_START_DELAY = 1.0, DEPTH_GATE = 0.1, DEPTH_START_DIST = 0.5;\n'
                'int DEPTH_INIT_LANDMARKS = 1, DEPTH_KEEP_STEREO = 0;')
    s = replace(s, '    fsSettings.release();\n}', '''    DEPTH_TOPIC.clear();
    if (!fsSettings["depth_topic"].empty()) fsSettings["depth_topic"] >> DEPTH_TOPIC;
    if (!fsSettings["depth_weight"].empty()) DEPTH_WEIGHT = fsSettings["depth_weight"];
    if (!fsSettings["depth_period"].empty()) DEPTH_PERIOD = fsSettings["depth_period"];
    if (!fsSettings["depth_wait_ms"].empty()) DEPTH_WAIT_MS = fsSettings["depth_wait_ms"];
    if (!fsSettings["depth_max_rel_range"].empty()) DEPTH_MAX_REL_RANGE = fsSettings["depth_max_rel_range"];
    if (!fsSettings["depth_init_landmarks"].empty()) DEPTH_INIT_LANDMARKS = fsSettings["depth_init_landmarks"];
    if (!fsSettings["depth_start_delay"].empty()) DEPTH_START_DELAY = fsSettings["depth_start_delay"];
    if (!fsSettings["depth_gate"].empty()) DEPTH_GATE = fsSettings["depth_gate"];
    if (!fsSettings["depth_start_dist"].empty()) DEPTH_START_DIST = fsSettings["depth_start_dist"];
    if (!fsSettings["depth_keep_stereo"].empty()) DEPTH_KEEP_STEREO = fsSettings["depth_keep_stereo"];
    if (!DEPTH_TOPIC.empty())
        ROS_INFO("LIVE_DEPTH topic=%s weight=%.3f period=%.3fs wait=%.1fms rel_range=%.3f init=%d", DEPTH_TOPIC.c_str(),
                 DEPTH_WEIGHT, DEPTH_PERIOD, DEPTH_WAIT_MS, DEPTH_MAX_REL_RANGE, DEPTH_INIT_LANDMARKS);

    fsSettings.release();
}''')
    p.write_text(s)

    # ---- feature manager: per-observation depth --------------------------------------------
    p = src / 'estimator' / 'feature_manager.h'; s = p.read_text()
    s = replace(s, '    bool is_stereo;', '    bool is_stereo;\n    double igev_inverse_depth = 0.0;\n'
                '    double igev_sigma = 0.0;\n    bool igev_active = false;')
    p.write_text(s)
    # Landmarks without stereo triangulation: initialise from the learned depth of their first observation.
    p = src / 'estimator' / 'feature_manager.cpp'; s = p.read_text()
    old = '''        if (it_per_id.estimated_depth > 0)
            continue;'''
    if s.count(old) != 1:
        raise RuntimeError('feature_manager.cpp triangulate anchor changed')
    s = s.replace(old, old + '''
        if (DEPTH_INIT_LANDMARKS && !(STEREO && it_per_id.feature_per_frame[0].is_stereo) &&
            it_per_id.feature_per_frame[0].igev_inverse_depth > 0)
        {
            it_per_id.estimated_depth = 1.0 / it_per_id.feature_per_frame[0].igev_inverse_depth;
            continue;
        }''')
    p.write_text(s)

    # ---- estimator ---------------------------------------------------------------------------
    p = src / 'estimator' / 'estimator.h'; s = p.read_text()
    s = replace(s, '#include "parameters.h"', '#include "parameters.h"\n#include "../factor/igev_depth_factor.h"\n'
                '#include "../factor/depth_topic_provider.h"')
    s = replace(s, '    FeatureTracker featureTracker;', '    FeatureTracker featureTracker;\n    DepthTopicProvider depth;\n    Eigen::Vector3d depth_origin = Eigen::Vector3d::Zero();\n    bool depth_origin_set = false, depth_motion_ready = false;')
    p.write_text(s)

    s = est.read_text()
    s = replace(s, '    td = TD;', MARK + '''
    depth.configure(!DEPTH_TOPIC.empty(), DEPTH_WEIGHT, DEPTH_PERIOD, DEPTH_WAIT_MS, DEPTH_MAX_REL_RANGE);
    depth.start_delay_s = DEPTH_START_DELAY;
    if (depth.enabled && (ESTIMATE_TD || ESTIMATE_EXTRINSIC || std::abs(TD) > 1e-10))
        throw std::runtime_error("live depth factors require fixed extrinsics and td=0");
    td = TD;''')
    s = replace(s, '    Headers[frame_count] = header;', '''    Headers[frame_count] = header;
    // Depth starts only after the initialised estimator has moved DEPTH_START_DIST metres (take-off guard).
    if (solver_flag != NON_LINEAR) depth_motion_ready = false, depth_origin_set = false;
    else if (!depth_origin_set) depth_origin = Ps[frame_count], depth_origin_set = true;
    if (depth_origin_set && !depth_motion_ready && (Ps[frame_count] - depth_origin).norm() >= DEPTH_START_DIST)
        depth_motion_ready = true;
    if (depth.enabled && depth.load(header, solver_flag == NON_LINEAR && depth_motion_ready)) {
        int accepted = 0;
        for (auto &feature : f_manager.feature) {
            int k = frame_count - feature.start_frame;
            if (k < 0 || k >= int(feature.feature_per_frame.size())) continue;
            auto &observation = feature.feature_per_frame[k];
            if (depth.sample(observation.uv.x(), observation.uv.y(), observation.igev_inverse_depth, observation.igev_sigma))
                ++accepted;
            else
                observation.igev_inverse_depth = 0.;
        }
        ROS_INFO("LIVE_DEPTH_OBS t=%.9f accepted=%d received=%lld used=%lld missed=%lld skipped=%lld", header, accepted,
                 depth.received.load(), depth.used, depth.missed, depth.skipped);
    }''')
    s = replace(s, 'void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)\n{\n    inputImageCnt++;\n',
                'void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)\n{\n    inputImageCnt++;\n'
                '    // announce frames the backend will process, so the depth node computes exactly those\n'
                '    if (depth.enabled && depth.on_backend_frame && (!MULTIPLE_THREAD || inputImageCnt % 2 == 0))\n'
                '        depth.on_backend_frame(t);\n')
    prefix, rest = s.split('void Estimator::optimization()', 1)
    body, suffix = rest.split('void Estimator::slideWindow()', 1)
    body = replace(body, '    int f_m_cnt = 0;', '    int depth_factor_count = 0;\n    int f_m_cnt = 0;')
    marker = '\n            if(STEREO && it_per_frame.is_stereo)\n            {'
    body = replace(body, marker, '''
            // Only constrain landmarks with a valid current inverse depth.
            it_per_frame.igev_active = it_per_frame.igev_inverse_depth > 0 &&
                std::isfinite(para_Feature[feature_index][0]) && para_Feature[feature_index][0] > 1e-6;
            if (it_per_frame.igev_active) {
                ceres::CostFunction *depth_factor;
                std::vector<double *> blocks;
                if (imu_i == imu_j) {
                    depth_factor = new IgevAnchorDepthFactor(it_per_frame.igev_inverse_depth, it_per_frame.igev_sigma);
                    blocks = {para_Feature[feature_index]};
                } else {
                    depth_factor = new IgevDepthFactor(pts_i, it_per_frame.igev_inverse_depth, it_per_frame.igev_sigma);
                    blocks = {para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]};
                }
                // Consistency gate: relative depth disagreement with the current estimate.
                double r = 0;
                const bool ok = depth_factor->Evaluate(blocks.data(), &r, nullptr) && std::isfinite(r);
                const double rho = it_per_frame.igev_inverse_depth, s = it_per_frame.igev_sigma;
                const double rel = std::abs(r) * s / rho;  // = |dz|/z (non-anchor) = |d rho|/rho (anchor)
                if (ok && rel < DEPTH_GATE) {
                    problem.AddResidualBlock(depth_factor, loss_function, blocks);
                    problem.SetParameterLowerBound(para_Feature[feature_index], 0, 1e-6);
                    ++depth_factor_count;
                } else {
                    delete depth_factor;
                    it_per_frame.igev_active = false;
                }
            }
            // Replace the correlated right-camera factor only where depth is accepted.
            if(STEREO && it_per_frame.is_stereo && (DEPTH_KEEP_STEREO || !it_per_frame.igev_active))
            {''')
    body = replace(body, '    ROS_DEBUG("visual measurement count: %d", f_m_cnt);',
                   '    ROS_DEBUG("visual measurement count: %d", f_m_cnt);\n'
                   '    if (depth.enabled && depth_factor_count) ROS_DEBUG("LIVE_DEPTH_FACTORS %d", depth_factor_count);')
    marker = '\n                    if(STEREO && it_per_frame.is_stereo)\n                    {'
    body = replace(body, marker, '''
                    if (it_per_frame.igev_active) {
                        if (imu_i == imu_j)
                            marginalization_info->addResidualBlockInfo(new ResidualBlockInfo(
                                new IgevAnchorDepthFactor(it_per_frame.igev_inverse_depth, it_per_frame.igev_sigma),
                                loss_function, vector<double *>{para_Feature[feature_index]}, vector<int>{0}));
                        else
                            marginalization_info->addResidualBlockInfo(new ResidualBlockInfo(
                                new IgevDepthFactor(pts_i, it_per_frame.igev_inverse_depth, it_per_frame.igev_sigma),
                                loss_function, vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                vector<int>{0, 3}));
                    }
                    if(STEREO && it_per_frame.is_stereo && (DEPTH_KEEP_STEREO || !it_per_frame.igev_active))
                    {''')
    est.write_text(prefix + 'void Estimator::optimization()' + body + 'void Estimator::slideWindow()' + suffix)

    # ---- ROS node: subscribe to the depth topic ------------------------------------------
    p = src / 'rosNodeTest.cpp'; s = p.read_text()
    s = replace(s, '#include "utility/visualization.h"', '#include "utility/visualization.h"\n#include <std_msgs/msg/header.hpp>\n#include <cmath>')
    s = replace(s, 'void imu_callback(', '''void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (msg->encoding != "32FC2" || msg->is_bigendian) {
        ROS_WARN("depth topic must be little-endian 32FC2");
        return;
    }
    const long long stamp = (long long)msg->header.stamp.sec * 1000000000LL + msg->header.stamp.nanosec;
    if (!estimator.depth.push(stamp, msg->width, msg->height, msg->data.data(), msg->step, msg->data.size()))
        ROS_WARN("invalid depth image dimensions, stride or payload size");
}

rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr pub_depth_request;

void depth_pending_callback(const std_msgs::msg::Header::SharedPtr msg)
{
    estimator.depth.mark_pending((long long)msg->stamp.sec * 1000000000LL + msg->stamp.nanosec);
}

void imu_callback(''')
    s = replace(s, '    std::thread sync_thread{sync_process};', '''    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth = NULL;
    if (!DEPTH_TOPIC.empty())
        sub_depth = n->create_subscription<sensor_msgs::msg::Image>(DEPTH_TOPIC, rclcpp::QoS(rclcpp::KeepLast(20)), depth_callback);
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr sub_depth_pending = NULL;
    if (!DEPTH_TOPIC.empty()) {
        sub_depth_pending = n->create_subscription<std_msgs::msg::Header>(DEPTH_TOPIC + "/pending",
            rclcpp::QoS(rclcpp::KeepLast(50)), depth_pending_callback);
        pub_depth_request = n->create_publisher<std_msgs::msg::Header>(DEPTH_TOPIC + "/request", rclcpp::QoS(rclcpp::KeepLast(50)));
        estimator.depth.on_backend_frame = [](double t) {
            long long ns = std::llround(t * 1e9);
            std_msgs::msg::Header h;
            h.stamp.sec = (int32_t)(ns / 1000000000LL);
            h.stamp.nanosec = (uint32_t)(ns % 1000000000LL);
            pub_depth_request->publish(h);
        };
    }

    std::thread sync_thread{sync_process};''')
    p.write_text(s)
    print('patched', repo)


if __name__ == '__main__':
    apply(sys.argv[1])
