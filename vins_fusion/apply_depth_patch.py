"""Apply the depth extension to the pinned upstream source; fail on source drift."""
from pathlib import Path
import shutil
import sys


def replace(text, old, new, count=1):
    if text.count(old)!=count:
        raise RuntimeError('Unexpected upstream source at: '+old[:100])
    return text.replace(old,new)


def apply(root):
    root=Path(root)/'vins_estimator/src'
    here=Path(__file__).resolve().parent
    for header in (here/'depth').glob('*.h'):
        shutil.copyfile(header,root/'factor'/header.name)
    path=root/'estimator/feature_manager.h'
    s=path.read_text()
    s=replace(s,'    bool is_stereo;','    bool is_stereo;\n    double igev_inverse_depth = 0.0;\n    double igev_sigma = 0.0;\n    bool igev_active = false;')
    path.write_text(s)
    path=root/'estimator/estimator.h';s=path.read_text()
    s=replace(s,'#include "parameters.h"','#include "parameters.h"\n#include "../factor/igev_depth_factor.h"\n#include "../factor/igev_depth_provider.h"')
    s=replace(s,'    FeatureTracker featureTracker;','    FeatureTracker featureTracker;\n    IgevDepthProvider igev_depth;')
    path.write_text(s)
    path=root/'estimator/estimator.cpp';s=path.read_text()
    s=replace(s,'    td = TD;','    if (igev_depth.enabled && (ESTIMATE_TD || ESTIMATE_EXTRINSIC))\n        throw std::runtime_error("IGEV integration requires fixed extrinsic and td=0");\n    if (igev_depth.enabled && std::abs(TD)>1e-10)\n        throw std::runtime_error("IGEV integration requires td=0");\n    td = TD;')
    s=replace(s,'    Headers[frame_count] = header;', '''    Headers[frame_count] = header;
    if (igev_depth.enabled && igev_depth.load(header)) {
        int accepted=0;
        for (auto &feature : f_manager.feature) {
            int k=frame_count-feature.start_frame;
            if(k<0 || k>=int(feature.feature_per_frame.size())) continue;
            auto &observation=feature.feature_per_frame[k];
            if(igev_depth.sample(observation.uv.x(),observation.uv.y(),
                                  observation.igev_inverse_depth,observation.igev_sigma)) ++accepted;
            else observation.igev_inverse_depth=0.;
        }
        ROS_INFO("IGEV_OBSERVATIONS timestamp=%.9f accepted=%d",header,accepted);
    }''')
    prefix,s=s.split('void Estimator::optimization()',1)
    s,suffix=s.split('void Estimator::slideWindow()',1)
    s=replace(s,'    int f_m_cnt = 0;','    int igev_factor_count = 0;\n    int f_m_cnt = 0;')
    marker='''            if(STEREO && it_per_frame.is_stereo)
            {'''
    s=replace(s,'\n'+marker,'\n'+'''            // Only constrain landmarks with a valid current triangulated inverse depth.
            it_per_frame.igev_active=it_per_frame.igev_inverse_depth>0 &&
                std::isfinite(para_Feature[feature_index][0]) && para_Feature[feature_index][0]>1e-6;
            if (it_per_frame.igev_active) {
                if (imu_i==imu_j)
                    problem.AddResidualBlock(new IgevAnchorDepthFactor(it_per_frame.igev_inverse_depth,it_per_frame.igev_sigma),
                                             loss_function,para_Feature[feature_index]);
                else
                    problem.AddResidualBlock(new IgevDepthFactor(pts_i,it_per_frame.igev_inverse_depth,it_per_frame.igev_sigma),
                                             loss_function,para_Pose[imu_i],para_Pose[imu_j],para_Ex_Pose[0],para_Feature[feature_index]);
                problem.SetParameterLowerBound(para_Feature[feature_index],0,1e-6);
                ++igev_factor_count;
            }
            // Replace the correlated right-stereo factor only at accepted depth observations.
            if(STEREO && it_per_frame.is_stereo && !it_per_frame.igev_active)
            {''')
    s=replace(s,'    ROS_DEBUG("visual measurement count: %d", f_m_cnt);',
              '    ROS_DEBUG("visual measurement count: %d", f_m_cnt);\n    if(igev_depth.enabled) ROS_INFO("IGEV_FACTORS timestamp=%.9f count=%d",Headers[frame_count],igev_factor_count);')
    marker='''                    if(STEREO && it_per_frame.is_stereo)
                    {'''
    s=replace(s,'\n'+marker,'\n'+'''                    if (it_per_frame.igev_active) {
                        if (imu_i==imu_j)
                            marginalization_info->addResidualBlockInfo(new ResidualBlockInfo(
                                new IgevAnchorDepthFactor(it_per_frame.igev_inverse_depth,it_per_frame.igev_sigma),
                                loss_function,vector<double *>{para_Feature[feature_index]},vector<int>{0}));
                        else
                            marginalization_info->addResidualBlockInfo(new ResidualBlockInfo(
                                new IgevDepthFactor(pts_i,it_per_frame.igev_inverse_depth,it_per_frame.igev_sigma),
                                loss_function,vector<double *>{para_Pose[imu_i],para_Pose[imu_j],para_Ex_Pose[0],para_Feature[feature_index]},vector<int>{0,3}));
                    }
                    if(STEREO && it_per_frame.is_stereo && !it_per_frame.igev_active)
                    {''')
    path.write_text(prefix+'void Estimator::optimization()'+s+'void Estimator::slideWindow()'+suffix)


if __name__=='__main__':
    apply(sys.argv[1])
