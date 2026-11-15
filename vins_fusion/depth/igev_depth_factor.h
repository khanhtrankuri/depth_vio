// SPDX-License-Identifier: GPL-3.0-or-later
// Local-pose Jacobians follow VINS-Fusion's [translation, right rotation, 0] contract.
#pragma once
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

inline Eigen::Matrix3d depthSkew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d m; m << 0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0; return m;
}

class IgevAnchorDepthFactor : public ceres::SizedCostFunction<1,1> {
    double inverse_, sigma_;
public:
    IgevAnchorDepthFactor(double inverse, double sigma):inverse_(inverse),sigma_(sigma) {}
    bool Evaluate(double const* const* p,double* r,double** j) const override {
        r[0]=(p[0][0]-inverse_)/sigma_;
        if(j && j[0]) j[0][0]=1./sigma_;
        return std::isfinite(r[0]);
    }
};

class IgevDepthFactor : public ceres::SizedCostFunction<1,7,7,7,1> {
    Eigen::Vector3d ray_;
    double inverse_,sigma_;
public:
    IgevDepthFactor(const Eigen::Vector3d &ray,double inverse,double sigma)
      :ray_(ray),inverse_(inverse),sigma_(sigma) {}
    bool Evaluate(double const* const* p,double* r,double** jac) const override {
        const Eigen::Vector3d Pi=Eigen::Map<const Eigen::Vector3d>(p[0]);
        const Eigen::Vector3d Pj=Eigen::Map<const Eigen::Vector3d>(p[1]);
        const Eigen::Vector3d t=Eigen::Map<const Eigen::Vector3d>(p[2]);
        const Eigen::Matrix3d Ri=Eigen::Quaterniond(p[0][6],p[0][3],p[0][4],p[0][5]).toRotationMatrix();
        const Eigen::Matrix3d Rj=Eigen::Quaterniond(p[1][6],p[1][3],p[1][4],p[1][5]).toRotationMatrix();
        const Eigen::Matrix3d Rc=Eigen::Quaterniond(p[2][6],p[2][3],p[2][4],p[2][5]).toRotationMatrix();
        const double rho=p[3][0];
        if(rho<=1e-8) return false;
        const Eigen::Vector3d ci=ray_/rho, bi=Rc*ci+t;
        const Eigen::Vector3d bj=Rj.transpose()*(Ri*bi+Pi-Pj);
        const Eigen::Vector3d cj=Rc.transpose()*(bj-t);
        if(!std::isfinite(cj.z())) return false;
        // Optical-Z residual avoids a singularity at a trial point's camera plane.
        // Propagate the inverse-depth noise proxy into metric Z at the observation.
        const double sqrt_information=inverse_*inverse_/sigma_;
        r[0]=(cj.z()-1./inverse_)*sqrt_information;
        if(jac) {
            Eigen::RowVector3d reduce(0,0,sqrt_information);
            const Eigen::Matrix3d A=Rc.transpose()*Rj.transpose()*Ri;
            if(jac[0]) {
                Eigen::Map<Eigen::Matrix<double,1,7,Eigen::RowMajor>> J(jac[0]);J.setZero();
                J.block<1,3>(0,0)=reduce*Rc.transpose()*Rj.transpose();
                J.block<1,3>(0,3)=-reduce*A*depthSkew(bi);
            }
            if(jac[1]) {
                Eigen::Map<Eigen::Matrix<double,1,7,Eigen::RowMajor>> J(jac[1]);J.setZero();
                J.block<1,3>(0,0)=-reduce*Rc.transpose()*Rj.transpose();
                J.block<1,3>(0,3)=reduce*Rc.transpose()*depthSkew(bj);
            }
            if(jac[2]) {
                Eigen::Map<Eigen::Matrix<double,1,7,Eigen::RowMajor>> J(jac[2]);J.setZero();
                J.block<1,3>(0,0)=reduce*Rc.transpose()*(Rj.transpose()*Ri-Eigen::Matrix3d::Identity());
                J.block<1,3>(0,3)=reduce*(depthSkew(cj)-A*Rc*depthSkew(ci));
            }
            if(jac[3]) jac[3][0]=(reduce*(-A*Rc*ray_/(rho*rho)))(0);
        }
        return std::isfinite(r[0]);
    }
};
