// g++ -std=c++14 -O2 -pthread test_depth_topic_provider.cpp -o /tmp/t && /tmp/t
#include "depth_topic_provider.h"
#include <cassert>
#include <cstdio>
#include <thread>

static std::vector<float> img(uint32_t w, uint32_t h, float inv, float sig) {
    std::vector<float> v(size_t(w) * h * 2);
    for (size_t i = 0; i < v.size(); i += 2) { v[i] = inv; v[i + 1] = sig; }
    return v;
}

int main() {
    const uint32_t W = 32, H = 24;
    DepthTopicProvider p;
    p.configure(true, 4.0, 0.5, 50.0, 0.08);
    auto a = img(W, H, 0.5f, 0.01f);
    const long long t0 = 1403715536162142992LL;
    p.push(t0, W, H, reinterpret_cast<uint8_t *>(a.data()), W * 8);
    double inv, sig;
    // exact exposure match (double seconds loses ~100 ns: tolerance is 1 us)
    assert(p.load(t0 * 1e-9));
    assert(p.sample(10, 10, inv, sig));
    assert(std::abs(inv - 0.5) < 1e-6 && std::abs(sig - 0.005) < 1e-7);  // sigma scaled by 1/sqrt(4)
    assert(!p.sample(0, 10, inv, sig));                                     // border rejected
    // period gate: 0.1 s later is not selected even if depth exists
    p.push(t0 + 100000000LL, W, H, reinterpret_cast<uint8_t *>(a.data()), W * 8);
    assert(!p.load((t0 + 100000000LL) * 1e-9));
    // missing frame with a newer one already present: no waiting, no stale reuse
    p.push(t0 + 700000000LL, W, H, reinterpret_cast<uint8_t *>(a.data()), W * 8);
    assert(!p.load((t0 + 600000000LL) * 1e-9));
    assert(!p.sample(10, 10, inv, sig));
    // depth arriving while the estimator waits
    std::thread late([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        p.push(t0 + 800000000LL, W, H, reinterpret_cast<uint8_t *>(a.data()), W * 8);
    });
    assert(p.load((t0 + 800000000LL) * 1e-9));
    late.join();
    // discontinuity rejected
    auto b = img(W, H, 0.5f, 0.01f);
    b[(size_t(10) * W + 11) * 2] = 1.0f;
    p.push(t0 + 2000000000LL, W, H, reinterpret_cast<uint8_t *>(b.data()), W * 8);
    assert(p.load((t0 + 2000000000LL) * 1e-9));
    assert(!p.sample(10, 10, inv, sig));
    assert(p.sample(20, 20, inv, sig));
    // timeout when nothing arrives
    auto s = std::chrono::steady_clock::now();
    assert(!p.load((t0 + 3000000000LL) * 1e-9));
    double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
    assert(waited > 0.04 && waited < 0.2);
    std::printf("depth_topic_provider tests passed (used=%lld missed=%lld)\n", p.used, p.missed);
    return 0;
}
