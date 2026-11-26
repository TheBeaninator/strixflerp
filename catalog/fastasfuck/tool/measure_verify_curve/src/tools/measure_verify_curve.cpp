// Benchmark tool to measure live C_verify(M) curve on AMD Strix Halo APU
// Measures verify forward latency for single-slot M in 1..16 (and 20, 24)
// and 4-slot batched verify for M in 4, 8, 12, 16 on real weights and diverse tokens.

#include "runtime/qwen4exp.hpp"
#include "serve/tokenizer.hpp"
#include "common/check.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <numeric>

using namespace strix;

namespace {

float compute_median(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

float compute_stdev(const std::vector<float> &v, float mean) {
    if (v.size() <= 1) return 0.0f;
    float sum_sq = 0.0f;
    for (float x : v) sum_sq += (x - mean) * (x - mean);
    return std::sqrt(sum_sq / (float)(v.size() - 1));
}

} // namespace

int main(int argc, char **argv) {
    std::string weights_path = "/models/Thinking-Cap-Qwen-3.8-FN-faf/weights-q6-hybrid.faf";
    std::string ngram_path = "/models/Thinking-Cap-Qwen-3.8-FN-faf/ngram.table";
    std::string tokenizer_path = "/models/Thinking-Cap-Qwen-3.8-FN-faf/tokenizer.json";
    std::string text_path = "/models/wikitext-2-test.txt";
    std::string output_json = "verify_curve_results.json";
    int n_trials = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--ngram" && i + 1 < argc) ngram_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--text" && i + 1 < argc) text_path = argv[++i];
        else if (arg == "--trials" && i + 1 < argc) n_trials = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_json = argv[++i];
    }

    std::cout << "=================================================================\n";
    std::cout << "   MEASURING LIVE C_VERIFY(M) LATENCY CURVE ON AMD STRIX HALO    \n";
    std::cout << "=================================================================\n";
    std::cout << "Weights   : " << weights_path << "\n";
    std::cout << "Corpus    : " << text_path << "\n";
    std::cout << "Trials    : " << n_trials << " per data point (+1 untimed warmup)\n";
    std::cout << "=================================================================\n\n";

    // 1. Load Tokenizer & Corpus
    std::cout << "[1/4] Loading Tokenizer and extracting diverse realistic tokens...\n";
    Tokenizer tokenizer(tokenizer_path);
    std::ifstream text_file(text_path);
    STRIX_CHECK(text_file.is_open(), "Cannot open corpus: ", text_path);
    std::stringstream ss;
    ss << text_file.rdbuf();
    std::vector<int32_t> all_tokens = tokenizer.encode(ss.str());
    std::cout << "      Total tokens available: " << all_tokens.size() << "\n\n";
    STRIX_CHECK(all_tokens.size() >= 2048, "Corpus too short");

    // 2. Load Model & Session (4 slots)
    std::cout << "[2/4] Loading Qwen4ExpModel into Unified LPDDR5X Memory...\n";
    auto t_load_0 = std::chrono::steady_clock::now();
    Qwen4ExpModel model(weights_path, ngram_path, kernels::Act::BF16, false,
                        8388608, 1.0f, WeightPlacement::Device);
    auto t_load_1 = std::chrono::steady_clock::now();
    double load_s = std::chrono::duration<double>(t_load_1 - t_load_0).count();
    std::cout << "      Model loaded in " << std::fixed << std::setprecision(2) << load_s << " s\n\n";

    std::cout << "[3/4] Initializing 4-slot session and prefilling context (pos = 128)...\n";
    const int64_t slot_capacity = 2048;
    const int64_t chunk_size = 2048;
    const int n_slots = 4;
    Qwen4ExpSession session(model, slot_capacity, chunk_size, PrefillMath::WmmaBf16,
                            false, KvFormat::Q8, n_slots);
    PrecisionOptions prec;
    prec.gdn = kernels::DrPrecision::HalfU;
    prec.attention = kernels::AttnPrecision::Fast;
    session.set_precision(prec);

    // Warmup ping
    std::vector<int32_t> ping_tok = {all_tokens[0]};
    session.forward(ping_tok, 1);
    session.reset();

    // Prefill 128 tokens into each of the 4 slots to establish realistic KV cache history
    for (int s = 0; s < n_slots; ++s) {
        std::vector<int32_t> prefill_tok(all_tokens.begin() + s * 128, all_tokens.begin() + (s + 1) * 128);
        std::vector<Qwen4ExpSession::BatchRun> br = {{s, prefill_tok, 1}};
        session.forward_batch(br);
    }
    std::cout << "      4 slots established at pos = 128 (KV cache active, GDN recurrent states warm).\n\n";

    std::cout << "[4/4] Executing Live Benchmark Sweeps...\n";

    // -------------------------------------------------------------------------
    // SWEEP A: 4-SLOT BATCHED VERIFY (forward_batch_verify)
    // M = 4, 8, 12, 16 (tokens per slot: k = 1, 2, 3, 4)
    // -------------------------------------------------------------------------
    std::cout << "\n>>> SECTION A: 4-SLOT BATCHED VERIFY (M = 4, 8, 12, 16)\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "| M (Total) | k/Slot | Median (ms) | Mean ± Stdev (ms) | Marg. Slope / tok |\n";
    std::cout << "|-----------|--------|-------------|-------------------|-------------------|\n";

    struct ResultPoint {
        int64_t M = 0;
        int64_t k_per_slot = 0;
        float median_ms = 0.0f;
        float mean_ms = 0.0f;
        float stdev_ms = 0.0f;
        float delta_per_tok = 0.0f;
    };
    std::vector<ResultPoint> batch_results;

    float prev_batch_med = 0.0f;
    int64_t prev_batch_M = 0;

    for (int k = 1; k <= 4; ++k) {
        int64_t M = n_slots * k;
        // Diverse tokens per slot
        std::vector<Qwen4ExpSession::BatchRun> runs;
        for (int s = 0; s < n_slots; ++s) {
            size_t start = 600 + s * 50 + k * 10;
            std::vector<int32_t> slot_ids(all_tokens.begin() + start, all_tokens.begin() + start + k);
            runs.push_back({s, slot_ids, 1});
        }

        // Untimed warmup
        session.forward_batch_verify(runs);
        for (int s = 0; s < n_slots; ++s) session.drop_verify(s);

        // n_trials timed
        std::vector<float> trials;
        for (int t = 0; t < n_trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            session.forward_batch_verify(runs);
            const auto t1 = std::chrono::steady_clock::now();
            for (int s = 0; s < n_slots; ++s) session.drop_verify(s);
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            trials.push_back(ms);
        }

        float med = compute_median(trials);
        float mean = std::accumulate(trials.begin(), trials.end(), 0.0f) / (float)trials.size();
        float stdev = compute_stdev(trials, mean);
        float slope = (prev_batch_M > 0) ? (med - prev_batch_med) / (float)(M - prev_batch_M) : 0.0f;

        batch_results.push_back({M, k, med, mean, stdev, slope});
        prev_batch_med = med;
        prev_batch_M = M;

        std::cout << "| " << std::setw(9) << M << " | "
                  << std::setw(6) << k << " | "
                  << std::setw(11) << std::fixed << std::setprecision(2) << med << " | "
                  << std::setw(6) << mean << " ± " << std::setw(4) << stdev << "   | "
                  << (slope > 0 ? ("+" + std::to_string(slope).substr(0, 4) + " ms/tok") : "  baseline  ") << " |\n";
    }

    // -------------------------------------------------------------------------
    // SWEEP B: SINGLE-SLOT VERIFY (forward_verify)
    // M = 1, 2, 3, 4, 6, 8, 10, 12, 14, 16
    // -------------------------------------------------------------------------
    std::cout << "\n>>> SECTION B: SINGLE-SLOT VERIFY (M = 1..16)\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "| M (Tokens) | Median (ms) | Mean ± Stdev (ms) | Marg. Slope / tok |\n";
    std::cout << "|------------|-------------|-------------------|-------------------|\n";

    std::vector<int64_t> m_targets = {1, 2, 3, 4, 6, 8, 10, 12, 14, 16};
    std::vector<ResultPoint> single_results;
    float prev_single_med = 0.0f;
    int64_t prev_single_M = 0;

    for (int64_t M : m_targets) {
        size_t start = 1000 + M * 15;
        std::vector<int32_t> test_ids(all_tokens.begin() + start, all_tokens.begin() + start + M);

        // Warmup
        session.forward_verify(test_ids, 1);
        session.drop_verify();

        // Timed trials
        std::vector<float> trials;
        for (int t = 0; t < n_trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            session.forward_verify(test_ids, 1);
            const auto t1 = std::chrono::steady_clock::now();
            session.drop_verify();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            trials.push_back(ms);
        }

        float med = compute_median(trials);
        float mean = std::accumulate(trials.begin(), trials.end(), 0.0f) / (float)trials.size();
        float stdev = compute_stdev(trials, mean);
        float slope = (prev_single_M > 0) ? (med - prev_single_med) / (float)(M - prev_single_M) : 0.0f;

        single_results.push_back({M, 0, med, mean, stdev, slope});
        prev_single_med = med;
        prev_single_M = M;

        std::cout << "| " << std::setw(10) << M << " | "
                  << std::setw(11) << std::fixed << std::setprecision(2) << med << " | "
                  << std::setw(6) << mean << " ± " << std::setw(4) << stdev << "   | "
                  << (slope > 0 ? ("+" + std::to_string(slope).substr(0, 4) + " ms/tok") : "  baseline  ") << " |\n";
    }

    // -------------------------------------------------------------------------
    // SWEEP C: TRUNK FORWARD EXTENSION FOR M = 16, 20, 24
    // -------------------------------------------------------------------------
    std::cout << "\n>>> SECTION C: TRUNK FORWARD EXTENSION (M = 16, 20, 24)\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "| M (Tokens) | Median (ms) | Mean ± Stdev (ms) | Marg. Slope / tok |\n";
    std::cout << "|------------|-------------|-------------------|-------------------|\n";

    std::vector<int64_t> ext_targets = {16, 20, 24};
    std::vector<ResultPoint> ext_results;
    float prev_ext_med = 0.0f;
    int64_t prev_ext_M = 0;

    Qwen4ExpSnapshot snap0 = session.make_snapshot(0);
    session.save(0, snap0);

    for (int64_t M : ext_targets) {
        size_t start = 1400 + M * 10;
        std::vector<int32_t> test_ids(all_tokens.begin() + start, all_tokens.begin() + start + M);

        // Warmup
        session.restore(0, snap0);
        std::vector<Qwen4ExpSession::BatchRun> runs = {{0, test_ids, 1}};
        session.forward_batch(runs);

        // Timed trials
        std::vector<float> trials;
        for (int t = 0; t < n_trials; ++t) {
            session.restore(0, snap0);
            runs = {{0, test_ids, 1}};
            const auto t0 = std::chrono::steady_clock::now();
            session.forward_batch(runs);
            const auto t1 = std::chrono::steady_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            trials.push_back(ms);
        }

        float med = compute_median(trials);
        float mean = std::accumulate(trials.begin(), trials.end(), 0.0f) / (float)trials.size();
        float stdev = compute_stdev(trials, mean);
        float slope = (prev_ext_M > 0) ? (med - prev_ext_med) / (float)(M - prev_ext_M) : 0.0f;

        ext_results.push_back({M, 0, med, mean, stdev, slope});
        prev_ext_med = med;
        prev_ext_M = M;

        std::cout << "| " << std::setw(10) << M << " | "
                  << std::setw(11) << std::fixed << std::setprecision(2) << med << " | "
                  << std::setw(6) << mean << " ± " << std::setw(4) << stdev << "   | "
                  << (slope > 0 ? ("+" + std::to_string(slope).substr(0, 4) + " ms/tok") : "  baseline  ") << " |\n";
    }

    // Save JSON
    std::ofstream out(output_json);
    out << "{\n";
    out << "  \"batch_4slots\": [\n";
    for (size_t i = 0; i < batch_results.size(); ++i) {
        const auto &r = batch_results[i];
        out << "    {\"M\": " << r.M << ", \"k\": " << r.k_per_slot << ", \"median_ms\": " << r.median_ms
            << ", \"mean_ms\": " << r.mean_ms << ", \"stdev_ms\": " << r.stdev_ms << ", \"slope_per_tok\": " << r.delta_per_tok << "}"
            << (i + 1 < batch_results.size() ? ",\n" : "\n");
    }
    out << "  ],\n";
    out << "  \"single_verify\": [\n";
    for (size_t i = 0; i < single_results.size(); ++i) {
        const auto &r = single_results[i];
        out << "    {\"M\": " << r.M << ", \"median_ms\": " << r.median_ms
            << ", \"mean_ms\": " << r.mean_ms << ", \"stdev_ms\": " << r.stdev_ms << ", \"slope_per_tok\": " << r.delta_per_tok << "}"
            << (i + 1 < single_results.size() ? ",\n" : "\n");
    }
    out << "  ],\n";
    out << "  \"extension_forward\": [\n";
    for (size_t i = 0; i < ext_results.size(); ++i) {
        const auto &r = ext_results[i];
        out << "    {\"M\": " << r.M << ", \"median_ms\": " << r.median_ms
            << ", \"mean_ms\": " << r.mean_ms << ", \"stdev_ms\": " << r.stdev_ms << ", \"slope_per_tok\": " << r.delta_per_tok << "}"
            << (i + 1 < ext_results.size() ? ",\n" : "\n");
    }
    out << "  ]\n";
    out << "}\n";
    out.close();

    std::cout << "\nResults saved to " << output_json << "\n";
    return 0;
}
