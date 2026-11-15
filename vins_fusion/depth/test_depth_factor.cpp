// SPDX-License-Identifier: GPL-3.0-or-later
#include "igev_depth_factor.h"
#include "igev_depth_provider.h"
#include <iostream>
#include <array>
#include <cassert>
#include <sys/stat.h>

void plus(double* p,int k,double delta) {
    if(k<3) p[k]+=delta;
    else {
        Eigen::Vector3d v=Eigen::Vector3d::Zero();v[k-3]=delta;
        Eigen::Quaterniond q(p[6],p[3],p[4],p[5]),dq(1,v.x()/2,v.y()/2,v.z()/2);
        q=(q*dq).normalized();
        p[3]=q.x();p[4]=q.y();p[5]=q.z();p[6]=q.w();
    }
}

int main() {
    std::array<std::array<double,7>,3> p;
    for(int b=0;b<3;b++) {
        p[b]={{.1*b,-.03*b,.04*b,0,0,0,1}};
        plus(p[b].data(),3,.04*(b+1));plus(p[b].data(),4,-.07*b);plus(p[b].data(),5,.03*b);
    }
    double rho=.35;const double* blocks[]={p[0].data(),p[1].data(),p[2].data(),&rho};
    IgevDepthFactor factor(Eigen::Vector3d(.2,-.1,1),.37,.03);
    double r,J0[7],J1[7],J2[7],J3[1];double* jac[]={J0,J1,J2,J3};
    assert(factor.Evaluate(blocks,&r,jac));double max_error=0;
    for(int b=0;b<4;b++) for(int k=0;k<(b==3?1:6);k++) {
        auto original=p;double original_rho=rho;
        if(b==3)rho+=1e-7;else plus(p[b].data(),k,1e-7);
        double rp;assert(factor.Evaluate(blocks,&rp,nullptr));p=original;rho=original_rho;
        if(b==3)rho-=1e-7;else plus(p[b].data(),k,-1e-7);
        double rm;assert(factor.Evaluate(blocks,&rm,nullptr));p=original;rho=original_rho;
        max_error=std::max(max_error,std::abs((rp-rm)/2e-7-jac[b][k]));
    }
    assert(max_error<1e-5);
    // A metric depth observation must change translation, not just initialize landmarks.
    double a[]={0,0,0,0,0,0,1},b[]={0,0,.2,0,0,0,1},e[]={0,0,0,0,0,0,1},inverse=.25;
    ceres::Problem problem;
    problem.AddResidualBlock(new IgevDepthFactor(Eigen::Vector3d(0,0,1),1./3.,.03),nullptr,a,b,e,&inverse);
    problem.SetParameterBlockConstant(a);problem.SetParameterBlockConstant(e);problem.SetParameterBlockConstant(&inverse);
    problem.SetParameterization(b,new ceres::SubsetParameterization(7,{0,1,3,4,5,6}));
    ceres::Solver::Options options;options.max_num_iterations=30;options.linear_solver_type=ceres::DENSE_QR;
    ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
    assert(summary.IsSolutionUsable());assert(std::abs(b[2]-1.)<1e-5);
    // A trial behind the observing camera must incur a finite penalty, not kill Ceres.
    b[2]=5;const double* invalid[]={a,b,e,&inverse};assert(factor.Evaluate(invalid,&r,nullptr));assert(std::abs(r)>1);
    double negative=-.1;const double* bad_rho[]={a,b,e,&negative};assert(!factor.Evaluate(bad_rho,&r,nullptr));
    IgevAnchorDepthFactor anchor(.25,.02);const double* only[]={&inverse};double derivative;double* aj[]={&derivative};
    assert(anchor.Evaluate(only,&r,aj));assert(std::abs(r)<1e-12 && std::abs(derivative-50)<1e-12);
    // Binary sampler must reject stale timestamps and discontinuities.
    mkdir("/tmp/igev-depth-test",0700);
    {std::ofstream index("/tmp/igev-depth-test/index.txt");index<<"1000000000 sample.bin\n";}
    {std::ofstream out("/tmp/igev-depth-test/sample.bin",std::ios::binary);uint32_t size=5;
     out.write(reinterpret_cast<char*>(&size),4);out.write(reinterpret_cast<char*>(&size),4);
     for(int i=0;i<25;i++){float values[]={.5f,.03f};out.write(reinterpret_cast<char*>(values),8);}}
    setenv("IGEV_DEPTH_DIR","/tmp/igev-depth-test",1);IgevDepthProvider provider;
    double measured,sigma;assert(provider.load(1.0000001));assert(provider.sample(2,2,measured,sigma));
    assert(std::abs(measured-.5)<1e-6);assert(!provider.sample(0,0,measured,sigma));
    assert(!provider.load(1.01));assert(!provider.sample(2,2,measured,sigma));
    std::cout<<"PASS: local Jacobians max error="<<max_error<<", depth-driven pose solve, invalid geometry, sampler timestamp/border checks\n";
}
