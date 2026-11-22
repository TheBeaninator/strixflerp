// runtime.flerp.chain test on bench box K: (1) 2,000 dependent kernels, plain vs chained, bytewise identical final
// buffers at grids 10, 40, 160, 1280; (2) time a launch each way; (3) a skipped predecessor ends in the error word,
// within the bounded wait, not a hang.
#include "flerp/chain/chain.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int kT = 256;
__device__ __forceinline__ float step(float v, int s) { return fmaf(v, 1.0001f, (float)(s & 7) * 1e-3f); }
__global__ void __launch_bounds__(kT) pp_plain(const float *src, float *dst, int s) {
    const int i = blockIdx.x * kT + threadIdx.x;
    dst[i] = step(src[(i + 37 * s) % (gridDim.x * kT)], s);  // reads another block's element: a real dependency
}
__global__ void __launch_bounds__(kT) pp_chained(const float *src, float *dst, int s, flerp::chain::Arg ca) {
    flerp::chain::wait(ca);
    const int i = blockIdx.x * kT + threadIdx.x;
    dst[i] = step(src[(i + 37 * s) % (gridDim.x * kT)], s);
    flerp::chain::done(ca);
}
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("{\"error\":\"%s\",\"at\":%d}\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

int main() {
    hipStream_t st;
    CK(hipStreamCreate(&st));
    const int N = 2000;
    bool all_same = true;
    for (int G : {10, 40, 160, 1280}) {
        const size_t n = (size_t)G * kT;
        std::vector<float> init(n);
        for (size_t i = 0; i < n; ++i) init[i] = 1.0f + 0.001f * (float)(i % 977);
        float *a, *b;
        CK(hipMalloc(&a, n * 4));
        CK(hipMalloc(&b, n * 4));
        auto run = [&](bool chained, flerp::chain::Chain *ch) -> double {
            CK(hipMemcpy(a, init.data(), n * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(b, init.data(), n * 4, hipMemcpyHostToDevice));
            if (ch) ch->reset();
            CK(hipStreamSynchronize(st));
            const auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s < N; ++s) {
                const float *src = (s & 1) ? b : a;
                float *dst = (s & 1) ? a : b;
                if (chained) ch->launch(pp_chained, dim3(G), dim3(kT), 0, src, dst, s);
                else hipLaunchKernelGGL(pp_plain, dim3(G), dim3(kT), 0, st, src, dst, s);
            }
            CK(hipStreamSynchronize(st));
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
        };
        flerp::chain::Chain ch(st);
        double plain_us = 0, chain_us = 0;
        std::vector<float> rp(n), rc(n);
        for (int rep = 0; rep < 3; ++rep) {  // ABAB, keep the last pair (warm)
            plain_us = run(false, nullptr);
            CK(hipMemcpy(rp.data(), (N & 1) ? b : a, n * 4, hipMemcpyDeviceToHost));
            chain_us = run(true, &ch);
            CK(hipMemcpy(rc.data(), (N & 1) ? b : a, n * 4, hipMemcpyDeviceToHost));
        }
        try { ch.check(); } catch (const std::exception &e) { std::printf("{\"error\":\"%s\"}\n", e.what()); return 1; }
        const bool same = std::memcmp(rp.data(), rc.data(), n * 4) == 0;
        all_same = all_same && same;
        std::printf("{\"grid\":%d,\"n\":%d,\"plain_us\":%.3f,\"chained_us\":%.3f,\"saved_us\":%.3f,\"bit_identical\":%s}\n", G, N,
                    plain_us, chain_us, plain_us - chain_us, same ? "true" : "false");
        CK(hipFree(a));
        CK(hipFree(b));
    }
    // watchdog: a predecessor counted but never launched
    {
        const int G = 40;
        float *a, *b;
        CK(hipMalloc(&a, G * kT * 4));
        CK(hipMalloc(&b, G * kT * 4));
        CK(hipMemset(a, 0, G * kT * 4));
        flerp::chain::Chain ch(st, 0.05);  // 50 ms bounded wait for the test
        CK(hipStreamSynchronize(st));
        const auto t0 = std::chrono::steady_clock::now();
        ch.launch(pp_chained, dim3(G), dim3(kT), 0, (const float *)a, b, 0);
        ch.skip(G);  // a launch that never ran
        for (int s = 1; s <= 20; ++s) ch.launch(pp_chained, dim3(G), dim3(kT), 0, (const float *)((s & 1) ? b : a), (s & 1) ? a : b, s);
        CK(hipStreamSynchronize(st));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        bool fired = false;
        try { ch.check(); } catch (const std::exception &) { fired = true; }
        std::printf("{\"watchdog\":{\"error_word_set\":%s,\"elapsed_ms\":%.1f,\"timeout_ms\":50,\"hung\":false}}\n", fired ? "true" : "false", ms);
    }
    std::printf("{\"summary\":{\"bit_identical_all_grids\":%s}}\n", all_same ? "true" : "false");
    return 0;
}
