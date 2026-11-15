// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <fstream>
#include <map>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <string>

class IgevDepthProvider {
    std::string directory_;
    std::map<long long,std::string> files_;
    std::vector<float> pixels_;
    uint32_t width_=0,height_=0;
public:
    bool enabled=false;
    IgevDepthProvider() {
        const char* directory=std::getenv("IGEV_DEPTH_DIR");
        if(!directory || !directory[0]) return;
        directory_=directory;
        std::ifstream index(directory_+"/index.txt");
        long long timestamp;std::string file;
        while(index>>timestamp>>file) files_[timestamp]=file;
        if(files_.empty()) throw std::runtime_error("IGEV depth index is empty/missing");
        enabled=true;
    }
    bool load(double seconds) {
        pixels_.clear();
        if(!enabled) return false;
        const long long query=std::llround(seconds*1e9);
        auto right=files_.lower_bound(query),selected=right;
        if(right!=files_.begin()) {
            auto left=right;--left;
            if(right==files_.end() || std::llabs(left->first-query)<std::llabs(right->first-query)) selected=left;
        }
        // Only the same exposure: no stale/nearest-neighbour depth across images.
        if(selected==files_.end() || std::llabs(selected->first-query)>1000) return false;
        std::ifstream input(directory_+"/"+selected->second,std::ios::binary);
        input.read(reinterpret_cast<char*>(&width_),4);input.read(reinterpret_cast<char*>(&height_),4);
        if(!input || width_==0 || height_==0 || width_>4096 || height_>4096)
            throw std::runtime_error("Invalid IGEV depth binary header");
        pixels_.resize(static_cast<size_t>(width_)*height_*2);
        input.read(reinterpret_cast<char*>(pixels_.data()),pixels_.size()*sizeof(float));
        if(!input) throw std::runtime_error("Truncated IGEV depth binary");
        return true;
    }
    bool sample(double u,double v,double &inverse,double &sigma) const {
        if(pixels_.empty() || !std::isfinite(u) || !std::isfinite(v)) return false;
        int x=std::lround(u),y=std::lround(v);
        if(x<1 || y<1 || x>=int(width_)-1 || y>=int(height_)-1) return false;
        double lo=1e10,hi=0,sum=0;sigma=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
            size_t id=(size_t(y+dy)*width_+x+dx)*2;
            double rho=pixels_[id],s=pixels_[id+1];
            if(!std::isfinite(rho) || !std::isfinite(s) || rho<=0 || s<=0) return false;
            lo=std::min(lo,rho);hi=std::max(hi,rho);sum+=rho;sigma=std::max(sigma,s);
        }
        inverse=sum/9.;
        // Reject depth discontinuities; do not interpolate across foreground/background.
        return (hi-lo)/inverse<.08;
    }
};
