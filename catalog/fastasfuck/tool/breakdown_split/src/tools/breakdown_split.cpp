// Comprehensive Layer Breakdown Benchmark on gfx1151 (Strix Halo)
// Measures every kernel in GDN, Attention (dense pos=4k & QSA pos=32k), MoE (E=512, top_k=10),
// Hyper-Connections, LM Head (248k Q8), and MTP Draft Head (160k Q4).
// Incorporates 128 MiB MALL (Infinity Cache) flushes before each timed trial.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q4.hpp"
#include "formats/q6.hpp"
#include "formats/q8.hpp"
#include "kernels/attention_gqa.hpp"
#include "kernels/gdn.hpp"
#include "kernels/hc.hpp"
#include "kernels/hc_wmma.hpp"
#include "kernels/kv8.hpp"
#include "kernels/linear.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/linear_q6.hpp"
#include "kernels/linear_q8.hpp"
#include "kernels/linear_wmma.hpp"
#include "kernels/moe_grouped.hpp"
#include "kernels/moe_router.hpp"
#include "kernels/norm.hpp"
#include "kernels/qsa_indexer.hpp"
#include "kernels/swiglu.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q6_device.hpp"
#include "runtime/q8_device.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

Q4Device make_synthetic_q4(int64_t N, int64_t K, int64_t G, const std::string &name) {
    Q4Weight w;
    w.N = N; w.K = K; w.G = G;
    w.q.resize((size_t)(N * K / 2), 0x55);
    int64_t groups = N * (K / G);
    w.scale.resize((size_t)groups, 0x3c00); // 1.0 in bf16
    w.minv.resize((size_t)groups, 0x0000);  // 0.0 in bf16
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

struct BreakdownResult {
    int64_t T = 0;
    float ms_gdn_total = 0.0f;
    float ms_attn_dense = 0.0f;
    float ms_attn_qsa = 0.0f;
    float ms_moe_total = 0.0f;
    float ms_hc_total = 0.0f;
    float ms_lm_head = 0.0f;
    float ms_step_dense = 0.0f;
    float ms_step_qsa = 0.0f;
};

// Global synthetic weight set allocated ONCE
struct ModelWeights {
    // GDN
    Q4Device w_gdn_in;
    DeviceBuffer<float> conv_w;
    DeviceBuffer<float> A_log;
    DeviceBuffer<float> dt_bias;
    DeviceBuffer<float> gdn_norm;
    Q6Device w_gdn_out;

    // Attention
    Q6Device w_attn_qkv;
    DeviceBuffer<float> q_norm, k_norm, inv_freq;
    Q6Device w_attn_o;

    // MoE
    DeviceBuffer<uint16_t> w_router;
    Q4Device w_gate_up;
    Q4Device w_down;
    Q6Device w_sh_gu;
    Q6Device w_sh_dn;

    // HC
    Q8Device w_hc_down;
    Q8Device w_hc_final;
    Q8ChunkMajorDevice w_hc_up;

    // LM Head
    Q8Device w_lm_head;

    // MTP Draft Head
    Q4Device w_draft_head;

    // MALL flush buffer (128 MiB)
    DeviceBuffer<uint8_t> mall_flush;

    ModelWeights()
        : w_gdn_in(make_synthetic_q4(16480, 2560, 128, "w_gdn_in")),
          conv_w(DeviceBuffer<float>::from_host(std::vector<float>(10240 * 4, 0.01f), "conv_w")),
          A_log(DeviceBuffer<float>::from_host(std::vector<float>(48, -0.5f), "A_log")),
          dt_bias(DeviceBuffer<float>::from_host(std::vector<float>(48, 0.1f), "dt_bias")),
          gdn_norm(DeviceBuffer<float>::from_host(std::vector<float>(128, 1.0f), "gdn_norm")),
          w_gdn_out(make_synthetic_q6(2560, 6144, 64, "w_gdn_out")),
          w_attn_qkv(make_synthetic_q6(13952, 2560, 64, "w_attn_qkv")),
          q_norm(DeviceBuffer<float>::from_host(std::vector<float>(256, 1.0f), "q_norm")),
          k_norm(DeviceBuffer<float>::from_host(std::vector<float>(256, 1.0f), "k_norm")),
          inv_freq(DeviceBuffer<float>::from_host(std::vector<float>(128, 0.01f), "inv_freq")),
          w_attn_o(make_synthetic_q6(2560, 6144, 64, "w_attn_o")),
          w_router(DeviceBuffer<uint16_t>::from_host(std::vector<uint16_t>(513 * 2560, 0x3c00), "w_router")),
          w_gate_up(make_synthetic_q4(512 * 1280, 2560, 128, "w_gate_up")),
          w_down(make_synthetic_q4(512 * 2560, 640, 128, "w_down")),
          w_sh_gu(make_synthetic_q6(1280, 2560, 64, "w_sh_gu")),
          w_sh_dn(make_synthetic_q6(2560, 640, 64, "w_sh_dn")),
          w_hc_down(make_synthetic_q8(324, 10240, 64, "w_hc_down")),
          w_hc_final(make_synthetic_q8(320, 10240, 64, "w_hc_final")),
          w_hc_up(make_synthetic_q8_cm(10240, 320, 64, "w_hc_up")),
          w_lm_head(make_synthetic_q8(248320, 2560, 64, "w_lm_head")),
          w_draft_head(make_synthetic_q4(160904, 2560, 64, "w_draft_head")),
          mall_flush(128 * 1024 * 1024, "mall_flush")
    {}
};

BreakdownResult run_breakdown(int64_t T, const ModelWeights &W, bool verbose = true) {
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    auto time_kernel = [&](auto fn, const char *label, int iters = 10) -> float {
        // Untimed warmup
        fn();
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "warmup sync");

        float total_ms = 0.0f;
        for (int i = 0; i < iters; ++i) {
            // Evict L2 (2 MiB) and MALL Infinity Cache (32 MiB) via 128 MiB flush buffer
            STRIX_HIP_CHECK(hipMemsetAsync(W.mall_flush.get(), 0x5a + (uint8_t)i, W.mall_flush.size(), stream), "flush");
            STRIX_HIP_CHECK(hipStreamSynchronize(stream), "flush sync");

            STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
            fn();
            STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
            STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
            float ms = 0.0f;
            STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
            total_ms += ms;
        }
        float avg_ms = total_ms / iters;
        if (verbose) {
            std::printf("  %-40s : %7.3f ms\n", label, avg_ms);
        }
        return avg_ms;
    };

    if (verbose) {
        std::printf("================================================================================\n");
        std::printf("   LAYER COMPONENT BREAKDOWN BENCHMARK AT T = %ld TOKENS (GFX1151)              \n", (long)T);
        std::printf("================================================================================\n\n");
    }

    const int64_t d = 2560;
    const int64_t H = 4;
    const int64_t n4 = H * d; // 10240

    // --------------------------------------------------------------------------
    // 1. GDN Layer (36 layers)
    // --------------------------------------------------------------------------
    if (verbose) std::printf("--- 1. GDN Layer Breakdown (36 layers) ---\n");
    const int64_t gk = 16, gv = 48, kHD = 128;
    const int64_t conv_c = 10240, gz = 6144, gstride = 16480;

    DeviceBuffer<uint16_t> x_u((size_t)(T * d), "u");
    DeviceBuffer<uint16_t> gdn_gp((size_t)(T * gstride), "gp");

    float ms_gdn_in = time_kernel([&]() {
        if (T < 12) linear_q4(x_u.get(), W.w_gdn_in.view(), gdn_gp.get(), T, Act::BF16, Act::BF16, stream);
        else linear_q4_wmma(x_u.get(), W.w_gdn_in.view(), gdn_gp.get(), T, Act::BF16, Act::BF16, stream);
    }, T < 12 ? "gdn_in (linear_q4 [16480x2560])" : "gdn_in (linear_q4_wmma [16480x2560])");

    DeviceBuffer<uint16_t> conv_state((size_t)(3 * conv_c), "conv_state");
    DeviceBuffer<uint16_t> qkv_buf((size_t)(T * conv_c), "qkv_buf");
    DeviceBuffer<float> beta_buf((size_t)(T * gv), "beta");
    DeviceBuffer<float> g_buf((size_t)(T * gv), "g");

    GdnFrontShape gfs{T, gk, kHD, gv, kHD, gstride, conv_c + gz, conv_c + gz + gv};
    float ms_gdn_front = time_kernel([&]() {
        gdn_front(gdn_gp.get(), gfs, W.conv_w.get(), conv_state.get(), conv_state.get(), false,
                  W.A_log.get(), W.dt_bias.get(), qkv_buf.get(), beta_buf.get(), g_buf.get(), Act::BF16, stream);
    }, "gdn_front (conv causal + gates)");

    DeviceBuffer<float> rec_state((size_t)(gv * kHD * kHD), "rec_state");
    DeviceBuffer<uint16_t> core_buf((size_t)(T * gz), "core");
    DeltaRuleShape drs{T, gk, gv, conv_c};

    float ms_delta_rule = time_kernel([&]() {
        if (T < 12) {
            delta_rule(qkv_buf.get(), beta_buf.get(), g_buf.get(), drs, rec_state.get(), rec_state.get(),
                       false, core_buf.get(), Act::BF16, stream, DeltaRulePath::PerToken);
        } else {
            delta_rule(qkv_buf.get(), beta_buf.get(), g_buf.get(), drs, rec_state.get(), rec_state.get(),
                       false, core_buf.get(), Act::BF16, stream, DeltaRulePath::ChunkedWmma, DrPrecision::HalfU);
        }
    }, T < 12 ? "delta_rule (PerToken)" : "delta_rule (ChunkedWmma / HalfU)");

    DeviceBuffer<uint16_t> gnorm_buf((size_t)(T * gz), "gnorm");
    float ms_rmsnorm_gated = time_kernel([&]() {
        rmsnorm_gated(core_buf.get(), gdn_gp.get() + conv_c, W.gdn_norm.get(), gnorm_buf.get(),
                      T * gv, kHD, 1e-6f, GateAct::Sigmoid, Act::BF16, stream, gv, gstride);
    }, "rmsnorm_gated (RMSNorm + Sigmoid)");

    DeviceBuffer<uint16_t> y_buf((size_t)(T * d), "y");
    float ms_gdn_out = time_kernel([&]() {
        if (T < 12) linear_q6(gnorm_buf.get(), W.w_gdn_out.view(), y_buf.get(), T, Act::BF16, Act::BF16, stream);
        else linear_q6_wmma(gnorm_buf.get(), W.w_gdn_out.view(), y_buf.get(), T, Act::BF16, Act::BF16, stream);
    }, T < 12 ? "gdn_out (linear_q6 [2560x6144])" : "gdn_out (linear_q6_wmma [2560x6144])");

    float ms_gdn_layer = ms_gdn_in + ms_gdn_front + ms_delta_rule + ms_rmsnorm_gated + ms_gdn_out;
    float ms_gdn_36 = ms_gdn_layer * 36;
    if (verbose) {
        std::printf("  TOTAL GDN Layer (single layer)         : %7.3f ms\n", ms_gdn_layer);
        std::printf("  TOTAL GDN Layers (36 layers)           : %7.3f ms\n\n", ms_gdn_36);
    }

    // --------------------------------------------------------------------------
    // 2. Attention Layer (12 layers)
    // --------------------------------------------------------------------------
    if (verbose) std::printf("--- 2. Attention Layer Breakdown (12 layers) ---\n");
    const int64_t hq = 24, hkv = 2, hd = 256;
    const int64_t astride = 13952;
    const int64_t idx_h = 4, idx_d = 128, idx_col = 13312;
    DeviceBuffer<uint16_t> attn_proj((size_t)(T * astride), "attn_proj");

    float ms_attn_qkv = time_kernel([&]() {
        if (T < 12) linear_q6(x_u.get(), W.w_attn_qkv.view(), attn_proj.get(), T, Act::BF16, Act::BF16, stream);
        else linear_q6_wmma(x_u.get(), W.w_attn_qkv.view(), attn_proj.get(), T, Act::BF16, Act::BF16, stream);
    }, T < 12 ? "attn_qkv (linear_q6 [13952x2560])" : "attn_qkv (linear_q6_wmma [13952x2560])");

    DeviceBuffer<uint16_t> k_cache((size_t)(T * hkv * hd), "k_cache");
    DeviceBuffer<uint16_t> v_cache((size_t)(T * hkv * hd), "v_cache");

    NormRopeJob nr[3] = {
        {attn_proj.get(), astride, 2 * hd, attn_proj.get(), astride, 2 * hd, hq, hd, W.q_norm.get()},
        {attn_proj.get() + hq * 2 * hd, astride, hd, k_cache.get(), hkv * hd, hd, hkv, hd, W.k_norm.get()},
        {attn_proj.get() + hq * 2 * hd + hkv * hd, astride, hd, v_cache.get(), hkv * hd, hd, hkv, hd, nullptr}
    };

    float ms_qk_norm_rope = time_kernel([&]() {
        qk_norm_rope_jobs(nr, 3, T, 1e-6f, 64, 0, W.inv_freq.get(), Act::BF16, stream);
    }, "qk_norm_rope_jobs (q, k, v RoPE)");

    // Attention Core: Context setup for pos = 4096 (Dense KvFormat::Q8)
    const int64_t cap = 65536, W_kv = hkv * hd; // 512
    DeviceBuffer<uint8_t> kv8_store_buf(kernels::kv8_bytes(cap, W_kv), "kv8_store");
    DeviceBuffer<uint8_t> kv8_tail_buf(kernels::kv8_tail_bytes(W_kv, Act::BF16), "kv8_tail");
    DeviceBuffer<uint16_t> k_new((size_t)(T * W_kv), "k_new");
    DeviceBuffer<uint16_t> v_new((size_t)(T * W_kv), "v_new");
    Kv8View kv8_view{kv8_store_buf.get(), cap, W_kv, kv8_tail_buf.get(), k_new.get(), v_new.get(), W_kv, 4096, nullptr, nullptr, 0};

    AttentionShape as_dense{T, 4096, hq, hkv, hd, astride, 2 * hd};
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    size_t attn_ws_bytes = attention_wmma_workspace_bytes(as_dense);
    DeviceBuffer<float> attn_ws(attn_ws_bytes / 4 + 1, "attn_ws");
    DeviceBuffer<uint16_t> attn_core((size_t)(T * hq * hd), "attn_core");

    float ms_attn_dense = time_kernel([&]() {
        attention_wmma(attn_proj.get(), attn_proj.get() + hd, kv8_view, as_dense, attn_scale,
                       attn_core.get(), attn_ws.get(), attn_ws_bytes, Act::BF16, stream, AttnPrecision::Fast);
    }, "attention_dense pos=4096 (Fast Kv8)");

    // Attention Core: Context setup for pos = 32768 (QSA)
    const int64_t cap_blocks = cap / 4; // 16384
    const int64_t qsa_k = 2048 / 4;        // 512
    DeviceBuffer<uint16_t> block_keys((size_t)(cap_blocks * idx_d), "block_keys");
    DeviceBuffer<float> qsa_scores_buf((size_t)(T * cap_blocks), "qsa_scores");
    DeviceBuffer<int32_t> qsa_sel((size_t)(T * qsa_k), "qsa_sel");
    DeviceBuffer<int32_t> qsa_nsel((size_t)T, "qsa_nsel");
    size_t qsa_topk_ws_bytes = kernels::qsa_topk_workspace_bytes(T);
    DeviceBuffer<uint8_t> qsa_topk_ws(qsa_topk_ws_bytes, "qsa_topk_ws");
    DeviceBuffer<uint32_t> attn_err(3, "attn_err");

    float ms_qsa_scores = time_kernel([&]() {
        qsa_scores(attn_proj.get() + idx_col, astride, idx_d, T, 32768, idx_h, idx_d,
                   block_keys.get(), cap_blocks, qsa_scores_buf.get(), cap_blocks, Act::BF16, stream);
    }, "qsa_scores pos=32768 (8192 blocks)");

    float ms_qsa_topk = time_kernel([&]() {
        qsa_topk(qsa_scores_buf.get(), cap_blocks, T, 32768, qsa_k, qsa_sel.get(), qsa_nsel.get(),
                 qsa_topk_ws.get(), qsa_topk_ws_bytes, stream);
    }, "qsa_topk pos=32768 (top 512 blocks)");

    AttentionShape as_qsa{T, 32768, hq, hkv, hd, astride, 2 * hd};
    size_t qsa_attn_ws_bytes = attention_gathered_wmma_workspace_bytes(as_qsa, qsa_k);
    if (qsa_attn_ws_bytes > attn_ws_bytes) {
        attn_ws = DeviceBuffer<float>(qsa_attn_ws_bytes / 4 + 1, "attn_ws");
        attn_ws_bytes = qsa_attn_ws_bytes;
    }

    float ms_attn_gathered = time_kernel([&]() {
        Kv8View kv8_view_qsa{kv8_store_buf.get(), cap, W_kv, kv8_tail_buf.get(), k_new.get(), v_new.get(), W_kv, 32768, nullptr, nullptr, 0};
        attention_gathered_wmma(attn_proj.get(), attn_proj.get() + hd, kv8_view_qsa, as_qsa,
                                qsa_sel.get(), qsa_nsel.get(), qsa_k, attn_scale, attn_core.get(),
                                attn_ws.get(), attn_ws_bytes, attn_err.get(), Act::BF16, stream, AttnPrecision::Fast);
    }, "attention_gathered pos=32768 (Fast Kv8)");

    float ms_attn_o = time_kernel([&]() {
        if (T < 12) linear_q6(attn_core.get(), W.w_attn_o.view(), y_buf.get(), T, Act::BF16, Act::BF16, stream);
        else linear_q6_wmma(attn_core.get(), W.w_attn_o.view(), y_buf.get(), T, Act::BF16, Act::BF16, stream);
    }, T < 12 ? "attn_o (linear_q6 [2560x6144])" : "attn_o (linear_q6_wmma [2560x6144])");

    float ms_attn_dense_layer = ms_attn_qkv + ms_qk_norm_rope + ms_attn_dense + ms_attn_o;
    float ms_attn_qsa_layer = ms_attn_qkv + ms_qk_norm_rope + ms_qsa_scores + ms_qsa_topk + ms_attn_gathered + ms_attn_o;
    if (verbose) {
        std::printf("  TOTAL Attention Layer (dense pos=4096) : %7.3f ms\n", ms_attn_dense_layer);
        std::printf("  TOTAL Attention Layers 12x (dense)     : %7.3f ms\n", ms_attn_dense_layer * 12);
        std::printf("  TOTAL Attention Layer (QSA pos=32768)  : %7.3f ms\n", ms_attn_qsa_layer);
        std::printf("  TOTAL Attention Layers 12x (QSA)       : %7.3f ms\n\n", ms_attn_qsa_layer * 12);
    }

    // --------------------------------------------------------------------------
    // 3. MoE Layer (48 layers)
    // --------------------------------------------------------------------------
    if (verbose) std::printf("--- 3. MoE Layer Breakdown (48 layers) ---\n");
    const int64_t E = 512;
    const int64_t top_k = 10;
    const int64_t inter = 640;

    DeviceBuffer<float> router_logits((size_t)(T * (E + 1)), "router_logits");
    float ms_router_gemm = time_kernel([&]() {
        if (T < 12) linear_bf16w(x_u.get(), W.w_router.get(), router_logits.get(), T, E + 1, d, Act::BF16, Act::F32, stream);
        else linear_bf16w_wmma(x_u.get(), W.w_router.get(), router_logits.get(), T, E + 1, d, Act::BF16, Act::F32, stream);
    }, T < 12 ? "router_gemm (linear_bf16w [513x2560])" : "router_gemm (linear_bf16w_wmma [513x2560])");

    DeviceBuffer<int32_t> route_ids((size_t)(T * top_k), "route_ids");
    DeviceBuffer<float> route_coef((size_t)(T * top_k), "route_coef");
    DeviceBuffer<uint32_t> router_err(3, "router_err");

    float ms_moe_router = time_kernel([&]() {
        moe_router(router_logits.get(), E + 1, T, E, top_k, router_logits.get() + E, E + 1, (int32_t)E,
                   route_ids.get(), route_coef.get(), Act::F32, router_err.get(), stream);
    }, "moe_router (top-k=10 router)");

    DeviceBuffer<uint16_t> h_swi((size_t)(T * top_k * inter), "h_swi");
    DeviceBuffer<float> partial((size_t)(T * top_k * d), "partial");
    size_t group_ws_bytes = moe_route_workspace_bytes(T, top_k, E);
    DeviceBuffer<uint8_t> group_ws(group_ws_bytes, "group_ws");

    float ms_moe_experts = 0.0f;
    if (T < 6) {
        float ms_gather = time_kernel([&]() {
            linear_q4_experts_gather_swiglu(x_u.get(), W.w_gate_up.view(), E, route_ids.get(),
                                            T, top_k, h_swi.get(), Act::BF16, router_err.get(), stream);
        }, "experts_gather_swiglu (ungrouped)");

        float ms_combine = time_kernel([&]() {
            linear_q4_experts_combine(h_swi.get(), W.w_down.view(), E, route_ids.get(), route_coef.get(),
                                      T, top_k, y_buf.get(), Act::BF16, router_err.get(), stream);
        }, "experts_combine (ungrouped)");
        ms_moe_experts = ms_gather + ms_combine;
    } else {
        float ms_group_routes = time_kernel([&]() {
            moe_group_routes(route_ids.get(), T, top_k, E, group_ws.get(), group_ws_bytes, router_err.get(), stream);
        }, "moe_group_routes");

        float ms_gather = time_kernel([&]() {
            linear_q4_experts_gather_grouped_swiglu(x_u.get(), W.w_gate_up.view(), E, group_ws.get(), group_ws_bytes,
                                                    T, top_k, h_swi.get(), Act::BF16, MoeMath::WmmaF16Wide256, stream);
        }, "gather_grouped_swiglu (Wide256)");

        float ms_combine = time_kernel([&]() {
            linear_q4_experts_combine_grouped(h_swi.get(), W.w_down.view(), E, route_ids.get(), route_coef.get(),
                                              group_ws.get(), group_ws_bytes, partial.get(), T, top_k, y_buf.get(),
                                              Act::BF16, MoeMath::WmmaF16Wide, stream);
        }, "combine_grouped (Wide)");
        ms_moe_experts = ms_group_routes + ms_gather + ms_combine;
    }

    // Shared Expert (Q6)
    DeviceBuffer<uint16_t> sh_gu_buf((size_t)(T * 2 * inter), "sh_gu");
    DeviceBuffer<uint16_t> sh_hh((size_t)(T * inter), "sh_hh");
    DeviceBuffer<uint16_t> sh_out((size_t)(T * d), "sh_out");

    float ms_shared_gu = time_kernel([&]() {
        if (T < 12) linear_q6(x_u.get(), W.w_sh_gu.view(), sh_gu_buf.get(), T, Act::BF16, Act::BF16, stream);
        else linear_q6_wmma(x_u.get(), W.w_sh_gu.view(), sh_gu_buf.get(), T, Act::BF16, Act::BF16, stream);
    }, T < 12 ? "shared_expert gate_up (linear_q6)" : "shared_expert gate_up (linear_q6_wmma)");

    float ms_shared_swi = time_kernel([&]() {
        swiglu(sh_gu_buf.get(), sh_hh.get(), T, inter, Act::BF16, stream);
    }, "shared_expert swiglu");

    float ms_shared_dn = time_kernel([&]() {
        if (T < 12) linear_q6(sh_hh.get(), W.w_sh_dn.view(), sh_out.get(), T, Act::BF16, Act::BF16, stream);
        else linear_q6_wmma(sh_hh.get(), W.w_sh_dn.view(), sh_out.get(), T, Act::BF16, Act::BF16, stream);
    }, T < 12 ? "shared_expert down (linear_q6)" : "shared_expert down (linear_q6_wmma)");

    float ms_shared_total = ms_shared_gu + ms_shared_swi + ms_shared_dn;
    float ms_moe_layer = ms_router_gemm + ms_moe_router + ms_moe_experts + ms_shared_total;
    float ms_moe_48 = ms_moe_layer * 48;
    if (verbose) {
        std::printf("  TOTAL Shared Expert (Q6)               : %7.3f ms\n", ms_shared_total);
        std::printf("  TOTAL MoE Layer (single layer)         : %7.3f ms\n", ms_moe_layer);
        std::printf("  TOTAL MoE Layers (48 layers)           : %7.3f ms\n\n", ms_moe_48);
    }

    // --------------------------------------------------------------------------
    // 4. Hyper-Connection & Injection (96 calls)
    // --------------------------------------------------------------------------
    if (verbose) std::printf("--- 4. Hyper-Connection & Injection Breakdown (96 calls) ---\n");
    const int64_t r = 320, R_hc = 324;
    DeviceBuffer<uint16_t> X_buf((size_t)(T * n4), "X");
    DeviceBuffer<float> h_hc((size_t)(T * r), "h_hc");
    DeviceBuffer<float> w_in_buf((size_t)(T * H), "w_in");
    DeviceBuffer<float> inv_buf((size_t)(T * H), "inv");
    size_t hc_ws_bytes = hc_wmma_workspace_bytes(T, H, R_hc);
    DeviceBuffer<float> hc_ws(hc_ws_bytes / 4 + 1, "hc_ws");

    float ms_hc_down = time_kernel([&]() {
        if (T < 12) {
            hc_mix_down(X_buf.get(), W.w_hc_down.view(), T, H, d, r, true, 1e-6f, h_hc.get(),
                        w_in_buf.get(), inv_buf.get(), Act::BF16, stream);
        } else {
            hc_mix_down_wmma(X_buf.get(), W.w_hc_down.view(), T, H, d, r, true, 1e-6f, h_hc.get(),
                             w_in_buf.get(), inv_buf.get(), hc_ws.get(), hc_ws_bytes, Act::BF16, stream, true);
        }
    }, T < 12 ? "hc_mix_down (GEMV)" : "hc_mix_down_wmma (inv_ready)");

    float ms_hc_up = time_kernel([&]() {
        if (T < 12) {
            hc_mix_up(h_hc.get(), W.w_hc_up.view(), X_buf.get(), W.gdn_norm.get(), inv_buf.get(),
                      T, H, d, x_u.get(), Act::BF16, stream);
        } else {
            hc_mix_up_wmma(h_hc.get(), W.w_hc_up.view(), X_buf.get(), W.gdn_norm.get(), inv_buf.get(),
                           T, H, d, x_u.get(), Act::BF16, stream);
        }
    }, T < 12 ? "hc_mix_up (GEMV)" : "hc_mix_up_wmma");

    float ms_hc_inject = time_kernel([&]() {
        if (T < 12) {
            hc_inject(X_buf.get(), w_in_buf.get(), y_buf.get(), T, H, d, Act::BF16, stream);
        } else {
            hc_inject_inv(X_buf.get(), w_in_buf.get(), y_buf.get(), T, H, d, 1e-6f, inv_buf.get(), Act::BF16, stream);
        }
    }, T < 12 ? "hc_inject" : "hc_inject_inv (fused)");

    float ms_hc_call = ms_hc_down + ms_hc_up + ms_hc_inject;
    float ms_hc_96 = ms_hc_call * 96;
    if (verbose) {
        std::printf("  TOTAL Hyper-Connection (per call)      : %7.3f ms\n", ms_hc_call);
        std::printf("  TOTAL Hyper-Connection (96 calls)      : %7.3f ms\n\n", ms_hc_96);
    }

    // --------------------------------------------------------------------------
    // 5. LM Head (1 call per forward)
    // --------------------------------------------------------------------------
    if (verbose) std::printf("--- 5. LM Head Breakdown (1 call) ---\n");
    DeviceBuffer<float> logits((size_t)(T * 248320), "logits");
    float ms_lm_head = time_kernel([&]() {
        if (T < 12) linear_q8(x_u.get(), W.w_lm_head.view(), logits.get(), T, Act::BF16, Act::F32, stream);
        else linear_q8_wmma(x_u.get(), W.w_lm_head.view(), logits.get(), T, Act::BF16, Act::F32, stream);
    }, T < 12 ? "lm_head (linear_q8 [248320x2560])" : "lm_head (linear_q8_wmma [248320x2560])");

    if (verbose) std::printf("\n");

    // Summary calculation
    float ms_step_dense = ms_gdn_36 + (ms_attn_dense_layer * 12) + ms_moe_48 + ms_hc_96 + ms_lm_head;
    float ms_step_qsa = ms_gdn_36 + (ms_attn_qsa_layer * 12) + ms_moe_48 + ms_hc_96 + ms_lm_head;

    if (verbose) {
        std::printf("================================================================================\n");
        std::printf("   TOTAL STEP FORWARD TIME ACCOUNTING (T = %ld)\n", (long)T);
        std::printf("================================================================================\n");
        std::printf("  36 GDN Layers Total                    : %7.3f ms  (%5.1f%%)\n", ms_gdn_36, ms_gdn_36 / ms_step_dense * 100.0f);
        std::printf("  12 Attention Layers (dense)            : %7.3f ms  (%5.1f%%)\n", ms_attn_dense_layer * 12, (ms_attn_dense_layer * 12) / ms_step_dense * 100.0f);
        std::printf("  12 Attention Layers (QSA)              : %7.3f ms\n", ms_attn_qsa_layer * 12);
        std::printf("  48 MoE Layers Total                    : %7.3f ms  (%5.1f%%)\n", ms_moe_48, ms_moe_48 / ms_step_dense * 100.0f);
        std::printf("  96 Hyper-Connection Calls              : %7.3f ms  (%5.1f%%)\n", ms_hc_96, ms_hc_96 / ms_step_dense * 100.0f);
        std::printf("   1 LM Head Call (248k rows)            : %7.3f ms  (%5.1f%%)\n", ms_lm_head, ms_lm_head / ms_step_dense * 100.0f);
        std::printf("--------------------------------------------------------------------------------\n");
        std::printf("  SUM WITH DENSE ATTENTION (pos=4096)    : %7.3f ms -> %7.1f tok/s\n", ms_step_dense, (float)T / (ms_step_dense / 1000.0f));
        std::printf("  SUM WITH QSA ATTENTION (pos=32768)     : %7.3f ms -> %7.1f tok/s\n", ms_step_qsa, (float)T / (ms_step_qsa / 1000.0f));
        std::printf("================================================================================\n\n");
    }

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);

    return {T, ms_gdn_36, ms_attn_dense_layer * 12, ms_attn_qsa_layer * 12, ms_moe_48, ms_hc_96, ms_lm_head, ms_step_dense, ms_step_qsa};
}

void benchmark_mtp_draft_block(const ModelWeights &W) {
    std::printf("================================================================================\n");
    std::printf("   MTP DRAFT BLOCK BENCHMARK: 4x(T=1) SERIAL vs 1x(T=4) BATCHED DRAFT           \n");
    std::printf("================================================================================\n");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    auto time_draft_step = [&](int64_t T, int iters = 15) -> float {
        const int64_t d = 2560, hq = 24, hkv = 2, hd = 256, astride = 13952, E = 512, top_k = 10, inter = 640;
        const int64_t H = 4, n4 = H * d, r = 320, R_hc = 324;
        DeviceBuffer<uint16_t> x_u((size_t)(T * d), "u");
        DeviceBuffer<uint16_t> attn_proj((size_t)(T * astride), "attn_proj");
        DeviceBuffer<uint16_t> attn_core((size_t)(T * hq * hd), "attn_core");
        DeviceBuffer<uint16_t> y_buf((size_t)(T * d), "y");

        const int64_t cap = 65536, W_kv = hkv * hd;
        DeviceBuffer<uint8_t> kv8_store_buf(kernels::kv8_bytes(cap, W_kv), "kv8_store");
        DeviceBuffer<uint8_t> kv8_tail_buf(kernels::kv8_tail_bytes(W_kv, Act::BF16), "kv8_tail");
        DeviceBuffer<uint16_t> k_new((size_t)(T * W_kv), "k_new");
        DeviceBuffer<uint16_t> v_new((size_t)(T * W_kv), "v_new");
        Kv8View kv8_view{kv8_store_buf.get(), cap, W_kv, kv8_tail_buf.get(), k_new.get(), v_new.get(), W_kv, 4096, nullptr, nullptr, 0};
        AttentionShape as{T, 4096, hq, hkv, hd, astride, 2 * hd};
        const float attn_scale = 1.0f / std::sqrt((float)hd);
        size_t attn_ws_bytes = attention_wmma_workspace_bytes(as);
        DeviceBuffer<float> attn_ws(attn_ws_bytes / 4 + 1, "attn_ws");

        DeviceBuffer<float> router_logits((size_t)(T * (E + 1)), "router_logits");
        DeviceBuffer<int32_t> route_ids((size_t)(T * top_k), "route_ids");
        DeviceBuffer<float> route_coef((size_t)(T * top_k), "route_coef");
        DeviceBuffer<uint32_t> router_err(3, "router_err");
        DeviceBuffer<uint16_t> h_swi((size_t)(T * top_k * inter), "h_swi");
        DeviceBuffer<uint16_t> sh_gu_buf((size_t)(T * 2 * inter), "sh_gu");
        DeviceBuffer<uint16_t> sh_hh((size_t)(T * inter), "sh_hh");
        DeviceBuffer<uint16_t> sh_out((size_t)(T * d), "sh_out");

        DeviceBuffer<uint16_t> X_buf((size_t)(T * n4), "X");
        DeviceBuffer<float> h_hc((size_t)(T * r), "h_hc");
        DeviceBuffer<float> w_in_buf((size_t)(T * H), "w_in");
        DeviceBuffer<float> inv_buf((size_t)(T * H), "inv");

        DeviceBuffer<float> draft_logits((size_t)(T * 160904), "draft_logits");

        auto run_draft = [&]() {
            // 1. Attention layer
            linear_q6(x_u.get(), W.w_attn_qkv.view(), attn_proj.get(), T, Act::BF16, Act::BF16, stream);
            attention_wmma(attn_proj.get(), attn_proj.get() + hd, kv8_view, as, attn_scale,
                           attn_core.get(), attn_ws.get(), attn_ws_bytes, Act::BF16, stream, AttnPrecision::Fast);
            linear_q6(attn_core.get(), W.w_attn_o.view(), y_buf.get(), T, Act::BF16, Act::BF16, stream);

            // 2. MoE layer (ungrouped for T < 6)
            linear_bf16w(x_u.get(), W.w_router.get(), router_logits.get(), T, E + 1, d, Act::BF16, Act::F32, stream);
            moe_router(router_logits.get(), E + 1, T, E, top_k, router_logits.get() + E, E + 1, (int32_t)E,
                       route_ids.get(), route_coef.get(), Act::F32, router_err.get(), stream);
            linear_q4_experts_gather_swiglu(x_u.get(), W.w_gate_up.view(), E, route_ids.get(),
                                            T, top_k, h_swi.get(), Act::BF16, router_err.get(), stream);
            linear_q4_experts_combine(h_swi.get(), W.w_down.view(), E, route_ids.get(), route_coef.get(),
                                      T, top_k, y_buf.get(), Act::BF16, router_err.get(), stream);
            linear_q6(x_u.get(), W.w_sh_gu.view(), sh_gu_buf.get(), T, Act::BF16, Act::BF16, stream);
            swiglu(sh_gu_buf.get(), sh_hh.get(), T, inter, Act::BF16, stream);
            linear_q6(sh_hh.get(), W.w_sh_dn.view(), sh_out.get(), T, Act::BF16, Act::BF16, stream);

            // 3. Final Hyper-Connection (inject = false -> w_in = nullptr, 320 rows)
            hc_mix_down(X_buf.get(), W.w_hc_final.view(), T, H, d, r, false, 1e-6f, h_hc.get(),
                        nullptr, inv_buf.get(), Act::BF16, stream);
            hc_mix_up(h_hc.get(), W.w_hc_up.view(), X_buf.get(), W.gdn_norm.get(), inv_buf.get(),
                      T, H, d, x_u.get(), Act::BF16, stream);

            // 4. Draft Head scoring (160904 rows, Q4G64)
            linear_q4(x_u.get(), W.w_draft_head.view(), draft_logits.get(), T, Act::BF16, Act::F32, stream);
        };

        // Warmup
        run_draft();
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "warmup sync");

        float total_ms = 0.0f;
        for (int i = 0; i < iters; ++i) {
            STRIX_HIP_CHECK(hipMemsetAsync(W.mall_flush.get(), 0x3c + (uint8_t)i, W.mall_flush.size(), stream), "flush");
            STRIX_HIP_CHECK(hipStreamSynchronize(stream), "flush sync");

            STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
            run_draft();
            STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
            STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
            float ms = 0.0f;
            STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
            total_ms += ms;
        }
        return total_ms / iters;
    };

    float ms_t1 = time_draft_step(1);
    float ms_t4 = time_draft_step(4);
    float ms_4x_t1 = 4.0f * ms_t1;
    float speedup = ms_4x_t1 / ms_t4;

    std::printf("  1x Single-slot Draft Step (T = 1)      : %7.3f ms\n", ms_t1);
    std::printf("  4x Serial Single-slot Drafts (4 * T=1) : %7.3f ms\n", ms_4x_t1);
    std::printf("  1x Batched 4-slot Draft Step (T = 4)   : %7.3f ms\n", ms_t4);
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("  MTP BATCHING ADVANTAGE                 : %7.3f ms saved per draft turn\n", ms_4x_t1 - ms_t4);
    std::printf("  THEORETICAL SPEEDUP RATIO              : %7.2fx faster drafting\n", speedup);
    std::printf("================================================================================\n\n");

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
}

}  // namespace

int main(int argc, char **argv) {
    std::printf("Initializing ModelWeights for gfx1151 (Strix Halo Hybrid Q6)...\n");
    ModelWeights W;
    std::printf("Weights allocated successfully.\n\n");

    if (argc > 1 && std::string(argv[1]) != "grid") {
        int64_t T = std::atoll(argv[1]);
        run_breakdown(T, W, true);
        if (T == 1 || T == 4) benchmark_mtp_draft_block(W);
        return 0;
    }

    // Run MTP draft comparison first
    benchmark_mtp_draft_block(W);

    // Run grid across T = 1, 4, 6, 12, 16, 20, 64, 512
    std::vector<int64_t> grid = {1, 4, 6, 12, 16, 20, 64, 512};
    std::vector<BreakdownResult> results;

    for (int64_t T : grid) {
        results.push_back(run_breakdown(T, W, false));
    }

    std::printf("========================================================================================================================\n");
    std::printf("                               SUMMARY GRID BREAKDOWN (COLD LPDDR5X / MALL FLUSHED)                                     \n");
    std::printf("========================================================================================================================\n");
    std::printf("%4s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s\n",
                "T", "GDN (36x)", "Attn-4k", "Attn-32k", "MoE (48x)", "HC (96x)", "LM Head", "Step (4k)", "Step (32k)", "tok/s (4k)");
    std::printf("------------------------------------------------------------------------------------------------------------------------\n");

    for (const auto &r : results) {
        float toks = (float)r.T / (r.ms_step_dense / 1000.0f);
        std::printf("%4ld | %7.2f ms | %7.2f ms | %7.2f ms | %7.2f ms | %7.2f ms | %7.2f ms | %7.2f ms | %7.2f ms | %7.1f t/s\n",
                    (long)r.T, r.ms_gdn_total, r.ms_attn_dense, r.ms_attn_qsa, r.ms_moe_total,
                    r.ms_hc_total, r.ms_lm_head, r.ms_step_dense, r.ms_step_qsa, toks);
    }
    std::printf("========================================================================================================================\n");

    return 0;
}
