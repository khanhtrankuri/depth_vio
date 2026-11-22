// SPDX-License-Identifier: GPL-3.0-or-later
// Live learned-depth provider for VINS-Fusion (ROS2).
// Receives 32FC2 images [inverse optical Z (1/m), sigma of inverse depth] on the raw cam0 pixel grid,
// produced online by realtime/depth_node. Depth is only ever used for the exact exposure it was computed
// from (|dt| <= 1 us); it is never propagated to another frame.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

class DepthTopicProvider {
public:
    struct Frame {
        uint32_t width = 0, height = 0;
        std::vector<float> pixels;  // interleaved inverse depth, sigma
    };

    bool enabled = false;
    double period_s = 1.0;      // minimum spacing between frames that receive depth factors
    double wait_s = 0.08;       // how long the backend may wait for the depth of a selected frame
    double sigma_scale = 1.0;   // 1/sqrt(information weight)
    double max_rel_range = 0.08;// local smoothness gate (3x3 inverse-depth range / mean)
    std::atomic<long long> received{0};  // written by the ROS thread, read by the estimator
    long long used = 0, missed = 0;

    void configure(bool on, double weight, double period, double wait_ms, double rel_range) {
        enabled = on;
        sigma_scale = 1.0 / std::sqrt(std::max(weight, 1e-6));
        period_s = period;
        wait_s = wait_ms * 1e-3;
        max_rel_range = rel_range;
    }

    // Called from the ROS subscription thread.
    bool push(long long stamp_ns, uint32_t w, uint32_t h, const uint8_t *data, size_t step, size_t size) {
        constexpr size_t pixel_bytes = 2 * sizeof(float);
        constexpr auto max_dimension = static_cast<uint32_t>(std::numeric_limits<int>::max());
        if (!data || w == 0 || h == 0 || w > std::numeric_limits<size_t>::max() / pixel_bytes ||
            w > max_dimension || h > max_dimension)
            return false;
        const size_t row_bytes = size_t(w) * pixel_bytes;
        if (step < row_bytes || h > size / step)
            return false;

        auto f = std::make_shared<Frame>();
        f->width = w; f->height = h; f->pixels.resize(size_t(w) * h * 2);
        for (uint32_t y = 0; y < h; ++y)
            std::memcpy(&f->pixels[size_t(y) * w * 2], data + size_t(y) * step, row_bytes);
        {
            std::lock_guard<std::mutex> lock(m_);
            frames_[stamp_ns] = f;
            ++received;
            while (frames_.size() > 64) frames_.erase(frames_.begin());
        }
        cv_.notify_all();
        return true;
    }

    double start_delay_s = 0.0; // depth factors only after the estimator has been non-linear this long

    // Called by the estimator for each backend frame. Returns true if depth is loaded for this exposure.
    // `estimator_ready`: the sliding window is initialised (solver_flag == NON_LINEAR).
    bool load(double seconds, bool estimator_ready = true) {
        current_.reset();
        if (!enabled) return false;
        if (!estimator_ready) { first_ready_ = -1; return false; }
        if (first_ready_ < 0) first_ready_ = seconds;
        if (seconds - first_ready_ < start_delay_s) return false;
        if (last_used_ >= 0 && seconds - last_used_ < period_s - 1e-3) return false;
        const long long query = std::llround(seconds * 1e9);
        std::unique_lock<std::mutex> lock(m_);
        auto find = [&]() -> std::shared_ptr<Frame> {
            auto it = frames_.lower_bound(query - 1000);
            if (it != frames_.end() && std::llabs(it->first - query) <= 1000) return it->second;
            return nullptr;
        };
        auto newer_exists = [&]() { return !frames_.empty() && frames_.rbegin()->first > query + 1000; };
        std::shared_ptr<Frame> f = find();
        if (!f && !newer_exists()) {
            cv_.wait_for(lock, std::chrono::duration<double>(wait_s), [&]() { f = find(); return f || newer_exists(); });
            if (!f) f = find();
        }
        // Drop everything older than this exposure; it can never be used again.
        frames_.erase(frames_.begin(), frames_.lower_bound(query - 1000));
        if (!f) { ++missed; return false; }
        current_ = f;
        last_used_ = seconds;
        ++used;
        return true;
    }

    bool sample(double u, double v, double &inverse, double &sigma) const {
        if (!current_ || !std::isfinite(u) || !std::isfinite(v)) return false;
        const int W = current_->width, H = current_->height;
        const int x = std::lround(u), y = std::lround(v);
        if (x < 1 || y < 1 || x >= W - 1 || y >= H - 1) return false;
        double lo = 1e10, hi = 0, sum = 0; sigma = 0;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                const size_t id = (size_t(y + dy) * W + x + dx) * 2;
                const double rho = current_->pixels[id], s = current_->pixels[id + 1];
                if (!std::isfinite(rho) || !std::isfinite(s) || rho <= 0 || s <= 0) return false;
                lo = std::min(lo, rho); hi = std::max(hi, rho); sum += rho; sigma = std::max(sigma, s);
            }
        inverse = sum / 9.;
        sigma *= sigma_scale;
        // Reject depth discontinuities; never interpolate across foreground/background.
        return (hi - lo) / inverse < max_rel_range;
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::map<long long, std::shared_ptr<Frame>> frames_;
    std::shared_ptr<Frame> current_;
    double last_used_ = -1;
    double first_ready_ = -1;
};
