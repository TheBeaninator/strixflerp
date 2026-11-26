// Targeted Threshold Benchmark: GEMV vs WMMA on gfx1151 (Strix Halo)
// Measures crossover points for Linear Q4/Q6/Q8, DeltaRule, HC Mix, and MoE Grouped
// across T = 8, 10, 11, 12, 14, 16, 20, 24, 32, 48, 64.
// Flushes 128 MiB MALL before each timed trial.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q4.hpp"
#include "formats/q6.hpp"
#include "formats/q8.hpp"
#include "kernels/gdn.hpp"
#include "kernels/hc.hpp"
#include "kernels/hc_wmma.hpp"
#include "kernels/linear.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/linear_q6.hpp"
#include "kernels/linear_q8.hpp"
#include "kernels/linear_wmma.hpp"
#include "kernels/moe_grouped.hpp"
#include "kernels/moe_router.hpp"
#include "kernels/norm.hpp"
#include "kernels/swiglu.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q6_device.hpp"
#include "runtime/q8_device.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

inline float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

Q4Device make_synthetic_q4(int64_t N, int64_t K, int64_t G, const std::string &name) {
    Q4Weight w;
    w.N = N; w.K = K; w.G = G;
    w.q.resize((size_t)(N * K / 2), 0x55);
    int64_t groups = N * (K / G);
    w.scale.resize((size_t)groups, 0x3c00);
    w.minv.resize((size_t)groups, 0x0000);
    return Q4Device::upload(w, name);
}

Q6Device make_synthetic_q6(int64_t N, int64_t K, int64_t G, const std::string &name) {
    Q6Weight w;
    w.N = N; w.K = K; w.G = G;
    int64_t rb = q6_row_bytes(K);
    w.q.resize((size_t)(N * rb), 0x33);
    int64_t groups = N * (K / G);
    w.scale.resize((size_t)groups, 0x3c00);
    w.minv.resize((size_t)groups, 0x0000);
    return Q6Device::upload(w, name);
}

Q8Device make_synthetic_q8(int64_t N, int64_t K, int64_t G, const std::string &name) {
    Q8Weight w;
    w.N = N; w.K = K; w.G = G;
    w.q.resize((size_t)(N * K), 0x7f);
    int64_t groups = N * (K / G);
    w.scale.resize((size_t)groups, 0x3c00);
    w.minv.resize((size_t)groups, 0x0000);
    return Q8Device::upload(w, name);
}

Q8ChunkMajorDevice make_synthetic_q8_cm(int64_t N, int64_t K, int64_t G, const std::string &name) {
    Q8Weight w;
    w.N = N; w.K = K; w.G = G;
    w.q.resize((size_t)(N * K), 0x7f);
    int64_t groups = N * (K / G);
    w.scale.resize((size_t)groups, 0x3c00);
    w.minv.resize((size_t)groups, 0x0000);
    Q8ChunkMajor qcm = to_chunk_major(w, name.c_str());
    return Q8ChunkMajorDevice::upload(qcm, name);
}

} // namespace

int main() {
    std::printf("========================================================================================\n");
    std::printf("   STEP 0: GEMV vs WMMA DETAILED CROSSOVER BENCHMARK (GFX1151 / COLD LPDDR5X)          \n");
    std::printf("========================================================================================\n\n");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    const int64_t d = 2560, H = 4, n4 = 10240, r = 320, R_hc = 324;
    const int64_t gk = 16, gv = 48, kHD = 128, conv_c = 10240, gz = 6144, gstride = 16480;
    const int64_t astride = 13952, hq = 24, hd = 256;
    const int64_t E = 512, top_k = 10, inter = 640;

    std::printf("Allocating synthetic weights...\n");
    DeviceBuffer<uint8_t> mall_flush(128 * 1024 * 1024, "mall_flush");

    // Weights
    Q4Device w_gdn_in = make_synthetic_q4(gstride, d, 128, "w_gdn_in");
    Q6Device w_gdn_out = make_synthetic_q6(d, gz, 64, "w_gdn_out");
    Q6Device w_attn_qkv = make_synthetic_q6(astride, d, 64, "w_attn_qkv");
    Q6Device w_attn_o = make_synthetic_q6(d, hq * hd, 64, "w_attn_o");
    Q8Device w_hc_down = make_synthetic_q8(R_hc, n4, 64, "w_hc_down");
    Q8ChunkMajorDevice w_hc_up = make_synthetic_q8_cm(n4, r, 64, "w_hc_up");
    DeviceBuffer<float> gdn_norm = DeviceBuffer<float>::from_host(std::vector<float>(128, 1.0f), "gdn_norm");
    Q8Device w_lm_head = make_synthetic_q8(248320, d, 64, "w_lm_head");

    // Router with realistic random weights so tokens route across ~50-150 diverse experts
    std::mt19937 rng(1337);
    std::normal_distribution<float> norm_dist(0.0f, 0.05f);
    std::vector<uint16_t> host_router((size_t)((E + 1) * d));
    for (size_t i = 0; i < host_router.size(); ++i) host_router[i] = f32_to_bf16_bits(norm_dist(rng));
    DeviceBuffer<uint16_t> w_router = DeviceBuffer<uint16_t>::from_host(host_router, "w_router");

    Q4Device w_gate_up = make_synthetic_q4(E * 2 * inter, d, 128, "w_gate_up");
    Q4Device w_down = make_synthetic_q4(E * d, inter, 128, "w_down");

    // GDN front & states
    DeviceBuffer<float> conv_w = DeviceBuffer<float>::from_host(std::vector<float>(conv_c * 4, 0.01f), "conv_w");
    DeviceBuffer<float> A_log = DeviceBuffer<float>::from_host(std::vector<float>(gv, -0.5f), "A_log");
    DeviceBuffer<float> dt_bias = DeviceBuffer<float>::from_host(std::vector<float>(gv, 0.1f), "dt_bias");

    std::printf("Weights ready.\n\n");

    auto time_fn = [&](auto fn, int iters = 15) -> float {
        fn();
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "warmup");
        float total_ms = 0.0f;
        for (int i = 0; i < iters; ++i) {
            STRIX_HIP_CHECK(hipMemsetAsync(mall_flush.get(), 0x5a + (uint8_t)i, mall_flush.size(), stream), "flush");
            STRIX_HIP_CHECK(hipStreamSynchronize(stream), "flush sync");

            STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
            fn();
            STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
            STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
            float ms = 0.0f;
            STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
            total_ms += ms;
        }
        return total_ms / iters;
    };

    std::vector<int64_t> T_list = {8, 10, 11, 12, 16, 20, 64, 512};

    // -----------------------------------------------------------------------------------------
    // 1. LINEAR Q6: GEMV vs WMMA
    // -----------------------------------------------------------------------------------------
    std::printf("========================================================================================\n");
    std::printf(" 1. LINEAR Q6: attn_qkv [13952 x 2560] (GEMV vs WMMA)\n");
    std::printf("========================================================================================\n");
    std::printf("%4s | %14s | %14s | %14s | %s\n", "T", "GEMV (linear_q6)", "WMMA (q6_wmma)", "Delta (ms)", "Winner");
    std::printf("----------------------------------------------------------------------------------------\n");

    for (int64_t T : T_list) {
        DeviceBuffer<uint16_t> x_u((size_t)(T * d), "u");
        DeviceBuffer<uint16_t> y_gemv((size_t)(T * astride), "y_gemv");
        DeviceBuffer<uint16_t> y_wmma((size_t)(T * astride), "y_wmma");

        float ms_gemv = time_fn([&]() { linear_q6(x_u.get(), w_attn_qkv.view(), y_gemv.get(), T, Act::BF16, Act::BF16, stream); });
        float ms_wmma = time_fn([&]() { linear_q6_wmma(x_u.get(), w_attn_qkv.view(), y_wmma.get(), T, Act::BF16, Act::BF16, stream); });

        const char *winner = (ms_gemv < ms_wmma) ? "GEMV is FASTER" : "WMMA is FASTER";
        std::printf("%4ld | %11.3f ms | %11.3f ms | %+11.3f ms | %s\n", (long)T, ms_gemv, ms_wmma, ms_wmma - ms_gemv, winner);
    }
    std::printf("\n");

    // -----------------------------------------------------------------------------------------
    // 2. LINEAR Q4: GEMV vs WMMA
    // -----------------------------------------------------------------------------------------
    std::printf("========================================================================================\n");
    std::printf(" 2. LINEAR Q4: gdn_in [16480 x 2560] (GEMV vs WMMA)\n");
    std::printf("========================================================================================\n");
    std::printf("%4s | %14s | %14s | %14s | %s\n", "T", "GEMV (linear_q4)", "WMMA (q4_wmma)", "Delta (ms)", "Winner");
    std::printf("----------------------------------------------------------------------------------------\n");

    for (int64_t T : T_list) {
        DeviceBuffer<uint16_t> x_u((size_t)(T * d), "u");
        DeviceBuffer<uint16_t> y_gemv((size_t)(T * gstride), "y_gemv");
        DeviceBuffer<uint16_t> y_wmma((size_t)(T * gstride), "y_wmma");

        float ms_gemv = time_fn([&]() { linear_q4(x_u.get(), w_gdn_in.view(), y_gemv.get(), T, Act::BF16, Act::BF16, stream); });
        float ms_wmma = time_fn([&]() { linear_q4_wmma(x_u.get(), w_gdn_in.view(), y_wmma.get(), T, Act::BF16, Act::BF16, stream); });

        const char *winner = (ms_gemv < ms_wmma) ? "GEMV is FASTER" : "WMMA is FASTER";
        std::printf("%4ld | %11.3f ms | %11.3f ms | %+11.3f ms | %s\n", (long)T, ms_gemv, ms_wmma, ms_wmma - ms_gemv, winner);
    }
    std::printf("\n");

    // -----------------------------------------------------------------------------------------
    // 3. LINEAR Q8: LM HEAD [248320 x 2560] GEMV vs WMMA
    // -----------------------------------------------------------------------------------------
    std::printf("========================================================================================\n");
    std::printf(" 3. LINEAR Q8: lm_head [248320 x 2560] (GEMV vs F16-WMMA)\n");
    std::printf("========================================================================================\n");
    std::printf("%4s | %14s | %14s | %14s | %s\n", "T", "GEMV (linear_q8)", "WMMA (q8_wmma)", "Delta (ms)", "Winner");
    std::printf("----------------------------------------------------------------------------------------\n");

    for (int64_t T : T_list) {
        DeviceBuffer<uint16_t> x_u((size_t)(T * d), "u");
        DeviceBuffer<float> y_gemv((size_t)(T * 248320), "y_gemv");
        DeviceBuffer<float> y_wmma((size_t)(T * 248320), "y_wmma");

        float ms_gemv = time_fn([&]() { linear_q8(x_u.get(), w_lm_head.view(), y_gemv.get(), T, Act::BF16, Act::F32, stream); });
        float ms_wmma = time_fn([&]() { linear_q8_wmma(x_u.get(), w_lm_head.view(), y_wmma.get(), T, Act::BF16, Act::F32, stream); });

        const char *winner = (ms_gemv < ms_wmma) ? "GEMV is FASTER" : "WMMA is FASTER";
        std::printf("%4ld | %11.3f ms | %11.3f ms | %+11.3f ms | %s\n", (long)T, ms_gemv, ms_wmma, ms_wmma - ms_gemv, winner);
    }
    std::printf("\n");

    // -----------------------------------------------------------------------------------------
    // 4. GDN DELTA RULE: PerToken vs ChunkedWmma
    // -----------------------------------------------------------------------------------------
    std::printf("========================================================================================\n");
    std::printf(" 4. GDN DELTA RULE: PerToken vs ChunkedWmma (HalfU)\n");
    std::printf("========================================================================================\n");
    std::printf("%4s | %14s | %14s | %14s | %s\n", "T", "PerToken", "ChunkedWmma", "Delta (ms)", "Winner");
    std::printf("----------------------------------------------------------------------------------------\n");

    for (int64_t T : T_list) {
        DeviceBuffer<uint16_t> qkv_buf((size_t)(T * conv_c), "qkv_buf");
        DeviceBuffer<float> beta_buf((size_t)(T * gv), "beta");
        DeviceBuffer<float> g_buf((size_t)(T * gv), "g");
        DeviceBuffer<float> rec_state((size_t)(gv * kHD * kHD), "rec_state");
        DeviceBuffer<uint16_t> core_buf((size_t)(T * gz), "core");
        DeltaRuleShape drs{T, gk, gv, conv_c};

        float ms_pertoken = time_fn([&]() {
            delta_rule(qkv_buf.get(), beta_buf.get(), g_buf.get(), drs, rec_state.get(), rec_state.get(),
                       false, core_buf.get(), Act::BF16, stream, DeltaRulePath::PerToken);
        });

        float ms_chunked = time_fn([&]() {
            delta_rule(qkv_buf.get(), beta_buf.get(), g_buf.get(), drs, rec_state.get(), rec_state.get(),
                       false, core_buf.get(), Act::BF16, stream, DeltaRulePath::ChunkedWmma, DrPrecision::HalfU);
        });

        const char *winner = (ms_pertoken < ms_chunked) ? "PerToken is FASTER" : "ChunkedWmma is FASTER";
        std::printf("%4ld | %11.3f ms | %11.3f ms | %+11.3f ms | %s\n", (long)T, ms_pertoken, ms_chunked, ms_chunked - ms_pertoken, winner);
    }
    std::printf("\n");

    // -----------------------------------------------------------------------------------------
    // 5. HYPER-CONNECTION: GEMV vs WMMA
    // -----------------------------------------------------------------------------------------
    std::printf("========================================================================================\n");
    std::printf(" 5. HYPER-CONNECTION: hc_mix_down + hc_mix_up (GEMV vs WMMA)\n");
    std::printf("========================================================================================\n");
    std::printf("%4s | %14s | %14s | %14s | %s\n", "T", "GEMV (hc)", "WMMA (hc_wmma)", "Delta (ms)", "Winner");
    std::printf("----------------------------------------------------------------------------------------\n");

    for (int64_t T : T_list) {
        DeviceBuffer<uint16_t> X_buf((size_t)(T * n4), "X");
        DeviceBuffer<float> h_hc((size_t)(T * r), "h_hc");
        DeviceBuffer<float> w_in_buf((size_t)(T * H), "w_in");
        DeviceBuffer<float> inv_buf((size_t)(T * H), "inv");
        DeviceBuffer<uint16_t> x_u((size_t)(T * d), "u");
        size_t hc_ws_bytes = hc_wmma_workspace_bytes(T, H, R_hc);
        DeviceBuffer<float> hc_ws(hc_ws_bytes / 4 + 1, "hc_ws");

        float ms_gemv = time_fn([&]() {
            hc_mix_down(X_buf.get(), w_hc_down.view(), T, H, d, r, true, 1e-6f, h_hc.get(),
                        w_in_buf.get(), inv_buf.get(), Act::BF16, stream);
            hc_mix_up(h_hc.get(), w_hc_up.view(), X_buf.get(), gdn_norm.get(), inv_buf.get(),
                      T, H, d, x_u.get(), Act::BF16, stream);
        });

        float ms_wmma = time_fn([&]() {
            hc_mix_down_wmma(X_buf.get(), w_hc_down.view(), T, H, d, r, true, 1e-6f, h_hc.get(),
                             w_in_buf.get(), inv_buf.get(), hc_ws.get(), hc_ws_bytes, Act::BF16, stream, false);
            hc_mix_up_wmma(h_hc.get(), w_hc_up.view(), X_buf.get(), gdn_norm.get(), inv_buf.get(),
                           T, H, d, x_u.get(), Act::BF16, stream);
        });

        const char *winner = (ms_gemv < ms_wmma) ? "GEMV is FASTER" : "WMMA is FASTER";
        std::printf("%4ld | %11.3f ms | %11.3f ms | %+11.3f ms | %s\n", (long)T, ms_gemv, ms_wmma, ms_wmma - ms_gemv, winner);
    }
    std::printf("\n");

    // -----------------------------------------------------------------------------------------
    // 6. MoE: Ungrouped GEMV vs Grouped Narrow vs Grouped Wide256 (RANDOM ROUTING)
    // -----------------------------------------------------------------------------------------
    std::printf("========================================================================================\n");
    std::printf(" 6. MoE EXPERTS (512 experts, top_k=10): Ungrouped vs Narrow vs Wide256 (RANDOM ROUTING)\n");
    std::printf("========================================================================================\n");
    std::printf("%4s | %10s | %10s | %10s | %s\n", "T", "Ungrouped", "GroupedNarrow", "GroupedWide256", "Best Method");
    std::printf("----------------------------------------------------------------------------------------\n");

    for (int64_t T : T_list) {
        DeviceBuffer<uint16_t> x_u((size_t)(T * d), "u");
        DeviceBuffer<float> router_logits((size_t)(T * (E + 1)), "router_logits");
        linear_bf16w(x_u.get(), w_router.get(), router_logits.get(), T, E + 1, d, Act::BF16, Act::F32, stream);

        DeviceBuffer<int32_t> route_ids((size_t)(T * top_k), "route_ids");
        DeviceBuffer<float> route_coef((size_t)(T * top_k), "route_coef");
        DeviceBuffer<uint32_t> router_err(3, "router_err");
        moe_router(router_logits.get(), E + 1, T, E, top_k, router_logits.get() + E, E + 1, (int32_t)E,
                   route_ids.get(), route_coef.get(), Act::F32, router_err.get(), stream);
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "router sync");

        // Inspect how many unique experts are touched
        std::vector<int32_t> host_ids((size_t)(T * top_k));
        STRIX_HIP_CHECK(hipMemcpy(host_ids.data(), route_ids.get(), host_ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost), "host_ids copy");
        std::vector<bool> touched(E, false);
        int unique_cnt = 0;
        for (int32_t id : host_ids) {
            if (id >= 0 && id < E && !touched[id]) {
                touched[id] = true;
                unique_cnt++;
            }
        }

        DeviceBuffer<uint16_t> h_swi((size_t)(T * top_k * inter), "h_swi");
        DeviceBuffer<float> partial((size_t)(T * top_k * d), "partial");
        DeviceBuffer<uint16_t> y_buf((size_t)(T * d), "y");
        size_t group_ws_bytes = moe_route_workspace_bytes(T, top_k, E);
        DeviceBuffer<uint8_t> group_ws(group_ws_bytes, "group_ws");

        float ms_ungrouped = time_fn([&]() {
            linear_q4_experts_gather_swiglu(x_u.get(), w_gate_up.view(), E, route_ids.get(),
                                            T, top_k, h_swi.get(), Act::BF16, router_err.get(), stream);
            linear_q4_experts_combine(h_swi.get(), w_down.view(), E, route_ids.get(), route_coef.get(),
                                      T, top_k, y_buf.get(), Act::BF16, router_err.get(), stream);
        });

        float ms_narrow = time_fn([&]() {
            moe_group_routes(route_ids.get(), T, top_k, E, group_ws.get(), group_ws_bytes, router_err.get(), stream);
            linear_q4_experts_gather_grouped_swiglu(x_u.get(), w_gate_up.view(), E, group_ws.get(), group_ws_bytes,
                                                    T, top_k, h_swi.get(), Act::BF16, MoeMath::WmmaF16Narrow, stream);
            linear_q4_experts_combine_grouped(h_swi.get(), w_down.view(), E, route_ids.get(), route_coef.get(),
                                              group_ws.get(), group_ws_bytes, partial.get(), T, top_k, y_buf.get(),
                                              Act::BF16, MoeMath::WmmaF16Narrow, stream);
        });

        float ms_wide256 = time_fn([&]() {
            moe_group_routes(route_ids.get(), T, top_k, E, group_ws.get(), group_ws_bytes, router_err.get(), stream);
            linear_q4_experts_gather_grouped_swiglu(x_u.get(), w_gate_up.view(), E, group_ws.get(), group_ws_bytes,
                                                    T, top_k, h_swi.get(), Act::BF16, MoeMath::WmmaF16Wide256, stream);
            linear_q4_experts_combine_grouped(h_swi.get(), w_down.view(), E, route_ids.get(), route_coef.get(),
                                              group_ws.get(), group_ws_bytes, partial.get(), T, top_k, y_buf.get(),
                                              Act::BF16, MoeMath::WmmaF16Wide256, stream);
        });

        const char *best = (ms_ungrouped <= ms_narrow && ms_ungrouped <= ms_wide256) ? "Ungrouped"
                         : (ms_narrow <= ms_wide256 ? "GroupedNarrow" : "GroupedWide256");

        char best_buf[64];
        std::snprintf(best_buf, sizeof(best_buf), "%s (%d unique experts)", best, unique_cnt);
        std::printf("%4ld | %7.3f ms | %10.3f ms | %11.3f ms | %s\n",
                    (long)T, ms_ungrouped, ms_narrow, ms_wide256, best_buf);
    }
    std::printf("========================================================================================\n");

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return 0;
}
