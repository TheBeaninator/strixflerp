#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include "serve/json.hpp"

namespace strix {

struct MtpCalibrationConfig {
    bool enabled = true;
    int draft_samples = 300;
    int verify_samples_min = 25;
    int verify_samples_max = 80;
    int warmup_samples = 15;
    float tolerance_ms = 0.08f;
    int triplets_target = 1000;
    bool cache_enabled = true;
    std::string cache_path;

    float fallback_draft_ms = 1.64f;
    float fallback_verify_base_ms = 26.84f;
    float fallback_verify_step_ms = 4.30f;
    std::vector<float> fallback_verify_grid;
    float fallback_alpha = 0.85f;
    float fallback_beta = -0.04f;
    float fallback_bias = 0.0f;
};

struct MtpHardwareProfile {
    float draft_step_ms = 1.64f;            // solo draft latency (m = 1)
    float draft_batch_step_ms = 1.64f;      // batched draft latency (m = n_slots)
    float verify_base_ms = 25.10f;
    float verify_step_ms = 2.50f;
    std::vector<float> verify_latencies = {
        0.0f, 25.10f, 30.30f, 34.96f, 39.63f, 42.66f, 45.69f, 50.65f, 55.61f,
        58.05f, 60.49f, 63.20f, 65.92f, 68.72f, 71.53f, 72.39f, 73.25f, 75.87f,
        78.85f, 81.84f, 85.46f, 89.09f, 92.70f, 96.30f, 100.0f
    };
    std::vector<float> solo_verify_latencies = {
        0.0f, 25.10f, 30.30f, 34.96f, 39.63f, 42.66f, 45.69f, 50.65f, 55.61f
    };

    float get_verify_cost(int64_t T, bool is_solo = false) const {
        if (T <= 0) return 0.0f;
        if (is_solo && !solo_verify_latencies.empty()) {
            if ((size_t)T < solo_verify_latencies.size()) {
                return solo_verify_latencies[(size_t)T];
            }
            float solo_step = (solo_verify_latencies.size() >= 2)
                ? (solo_verify_latencies.back() - solo_verify_latencies[1]) / (float)(solo_verify_latencies.size() - 2)
                : 4.3f;
            return solo_verify_latencies.back() + (float)(T - (int64_t)(solo_verify_latencies.size() - 1)) * solo_step;
        }
        if (verify_latencies.empty()) return (float)T * 4.0f;
        if ((size_t)T < verify_latencies.size()) {
            return verify_latencies[(size_t)T];
        }
        return verify_latencies.back() + (float)(T - (int64_t)(verify_latencies.size() - 1)) * verify_step_ms;
    }

    float get_draft_cost(int m, int total_slots) const {
        if (m <= 1 || total_slots <= 1 || draft_batch_step_ms <= 0.0f) {
            return draft_step_ms;
        }
        float frac = std::clamp((float)(m - 1) / (float)(total_slots - 1), 0.0f, 1.0f);
        return draft_step_ms + frac * (draft_batch_step_ms - draft_step_ms);
    }
};

struct MtpAcceptanceProfile {
    float alpha = 0.85f;
    float beta = -0.04f;
    float bias = 0.0f;
    int sample_count = 0;
};

struct MtpCalibratedProfile {
    std::string fingerprint;
    std::string model_id;
    std::string weights_path;
    uint64_t weights_strix_hash = 0;
    uint64_t weights_size = 0;
    int64_t weights_mtime = 0;
    int draft_head_bits = 4;
    bool draft_head_q4 = true;
    std::string mtp_vocab_filter = "all";
    std::string gdn_precision = "half-u";

    MtpHardwareProfile hw;
    MtpAcceptanceProfile acceptance;
};

float compute_median(std::vector<float> values);
float compute_trimmed_mean(std::vector<float> values, float trim_pct = 0.10f);
std::vector<float> filter_iqr_outliers(const std::vector<float>& values, float k = 1.5f);

std::string compute_weights_fingerprint(
    const std::string& weights_path,
    const std::string& model_id,
    int draft_head_bits,
    const std::string& mtp_vocab_filter,
    const std::string& gdn_precision,
    uint64_t* out_strix_hash = nullptr,
    uint64_t* out_file_size = nullptr,
    int64_t* out_mtime = nullptr
);

struct TripletSample {
    int64_t step;
    float delta_logits;
    int accepted;
};

MtpAcceptanceProfile fit_acceptance_model(
    const std::vector<TripletSample>& samples,
    float l2_reg = 1e-4f
);

std::optional<MtpCalibratedProfile> load_cached_profile(
    const std::string& cache_file,
    const std::string& fingerprint
);

bool save_cached_profile(
    const std::string& cache_file,
    const MtpCalibratedProfile& profile
);

} // namespace strix
