#include "serve/mtp_calib.hpp"
#include "formats/faf.hpp"
#include "serve/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace strix {

float compute_median(std::vector<float> values) {
    if (values.empty()) return 0.0f;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 1) return values[n / 2];
    return 0.5f * (values[n / 2 - 1] + values[n / 2]);
}

float compute_trimmed_mean(std::vector<float> values, float trim_pct) {
    if (values.empty()) return 0.0f;
    std::sort(values.begin(), values.end());
    size_t trim_k = static_cast<size_t>(values.size() * trim_pct);
    if (2 * trim_k >= values.size()) return compute_median(values);

    double sum = 0.0;
    size_t count = 0;
    for (size_t i = trim_k; i < values.size() - trim_k; ++i) {
        sum += values[i];
        count++;
    }
    return static_cast<float>(sum / count);
}

std::vector<float> filter_iqr_outliers(const std::vector<float>& values, float k) {
    if (values.size() < 4) return values;
    std::vector<float> s = values;
    std::sort(s.begin(), s.end());
    size_t n = s.size();
    float q1 = s[n / 4];
    float q3 = s[(3 * n) / 4];
    float iqr = q3 - q1;
    float upper = q3 + k * iqr;
    float lower = q1 - k * iqr;

    std::vector<float> filtered;
    filtered.reserve(n);
    for (float v : values) {
        if (v >= lower && v <= upper) {
            filtered.push_back(v);
        }
    }
    return filtered;
}

std::string compute_weights_fingerprint(
    const std::string& weights_path,
    const std::string& model_id,
    int draft_head_bits,
    const std::string& mtp_vocab_filter,
    const std::string& gdn_precision,
    uint64_t* out_strix_hash,
    uint64_t* out_file_size,
    int64_t* out_mtime
) {
    struct stat st;
    uint64_t file_size = 0;
    int64_t mtime_ns = 0;
    uint64_t header_index_hash = 0;

    if (::stat(weights_path.c_str(), &st) == 0) {
        file_size = static_cast<uint64_t>(st.st_size);
        mtime_ns = static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
    }

    int fd = ::open(weights_path.c_str(), O_RDONLY);
    if (fd >= 0) {
        uint8_t hdr[128];
        ssize_t rd = ::read(fd, hdr, sizeof(hdr));
        if (rd == 128 && (std::memcmp(hdr, "FASTASFK", 8) == 0 || std::memcmp(hdr, "STRIXWT\0", 8) == 0)) {
            std::memcpy(&header_index_hash, hdr + 48, sizeof(uint64_t));
        }
        ::close(fd);
    }

    if (out_strix_hash) *out_strix_hash = header_index_hash;
    if (out_file_size) *out_file_size = file_size;
    if (out_mtime) *out_mtime = mtime_ns;

    std::ostringstream ss;
    ss << model_id << "|"
       << weights_path << "|"
       << "0x" << std::hex << header_index_hash << "|"
       << file_size << "|"
       << mtime_ns << "|"
       << "q" << draft_head_bits << "|"
       << mtp_vocab_filter << "|"
       << gdn_precision;

    std::string desc = ss.str();
    uint64_t h1 = strix_hash64(desc.data(), desc.size());
    std::string desc_rev = desc;
    std::reverse(desc_rev.begin(), desc_rev.end());
    uint64_t h2 = strix_hash64(desc_rev.data(), desc_rev.size());

    std::ostringstream out;
    out << std::hex << std::setfill('0')
        << std::setw(16) << h1
        << std::setw(16) << h2;
    return out.str();
}

MtpAcceptanceProfile fit_acceptance_model(
    const std::vector<TripletSample>& samples,
    float l2_reg
) {
    MtpAcceptanceProfile res;
    res.sample_count = static_cast<int>(samples.size());
    if (samples.empty()) return res;

    double w[3] = {0.85, -0.04, 0.0};

    for (int iter = 0; iter < 20; ++iter) {
        double grad[3] = {0.0, 0.0, 0.0};
        double H[3][3] = {{0.0}};

        grad[0] += 2.0 * l2_reg * w[0];
        grad[1] += 2.0 * l2_reg * w[1];
        H[0][0] += 2.0 * l2_reg;
        H[1][1] += 2.0 * l2_reg;

        for (const auto& s : samples) {
            double eta = w[0] * s.delta_logits + w[1] * static_cast<double>(s.step) + w[2];
            double p = 1.0 / (1.0 + std::exp(-std::clamp(eta, -20.0, 20.0)));
            double diff = p - static_cast<double>(s.accepted);
            double weight = std::max(1e-6, p * (1.0 - p));

            double x[3] = {static_cast<double>(s.delta_logits), static_cast<double>(s.step), 1.0};

            for (int i = 0; i < 3; ++i) {
                grad[i] += x[i] * diff;
                for (int j = 0; j < 3; ++j) {
                    H[i][j] += x[i] * x[j] * weight;
                }
            }
        }

        double det = H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1])
                   - H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0])
                   + H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]);

        if (std::abs(det) < 1e-12) break;

        auto solve_cramer = [&](int col) {
            double Mc[3][3];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    Mc[r][c] = (c == col) ? grad[r] : H[r][c];
                }
            }
            double det_c = Mc[0][0] * (Mc[1][1] * Mc[2][2] - Mc[1][2] * Mc[2][1])
                         - Mc[0][1] * (Mc[1][0] * Mc[2][2] - Mc[1][2] * Mc[2][0])
                         + Mc[0][2] * (Mc[1][0] * Mc[2][1] - Mc[1][1] * Mc[2][0]);
            return det_c / det;
        };

        double dw0 = solve_cramer(0);
        double dw1 = solve_cramer(1);
        double dw2 = solve_cramer(2);

        w[0] -= dw0;
        w[1] -= dw1;
        w[2] -= dw2;

        if (std::abs(dw0) < 1e-4 && std::abs(dw1) < 1e-4 && std::abs(dw2) < 1e-4) {
            break;
        }
    }

    res.alpha = static_cast<float>(std::clamp(w[0], 0.05, 5.0));
    res.beta = static_cast<float>(std::clamp(w[1], -1.0, 0.5));
    res.bias = static_cast<float>(std::clamp(w[2], -5.0, 5.0));
    return res;
}

std::optional<MtpCalibratedProfile> load_cached_profile(
    const std::string& cache_file,
    const std::string& fingerprint
) {
    std::ifstream f(cache_file);
    if (!f.is_open()) return std::nullopt;

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    try {
        json::Value root = json::Value::parse(content);
        if (!root.is_object()) return std::nullopt;

        const json::Value *profiles = root.find("profiles");
        if (!profiles || !profiles->is_object()) return std::nullopt;

        const json::Value *prof = profiles->find(fingerprint);
        if (!prof || !prof->is_object()) return std::nullopt;

        MtpCalibratedProfile res;
        res.fingerprint = fingerprint;

        const json::Value *hw = prof->find("hardware_latencies");
        if (!hw || !hw->is_object()) return std::nullopt;

        const json::Value *v_draft = hw->find("draft_step_ms");
        if (v_draft) res.hw.draft_step_ms = static_cast<float>(v_draft->as_double("draft_step_ms"));

        const json::Value *v_draft_b = hw->find("draft_batch_step_ms");
        if (v_draft_b) res.hw.draft_batch_step_ms = static_cast<float>(v_draft_b->as_double("draft_batch_step_ms"));
        else res.hw.draft_batch_step_ms = res.hw.draft_step_ms;

        const json::Value *v_base = hw->find("verify_base_ms");
        if (v_base) res.hw.verify_base_ms = static_cast<float>(v_base->as_double("verify_base_ms"));

        const json::Value *v_step = hw->find("verify_step_ms");
        if (v_step) res.hw.verify_step_ms = static_cast<float>(v_step->as_double("verify_step_ms"));

        const json::Value *grid = hw->find("verify_ms_grid");
        if (grid && grid->is_array()) {
            res.hw.verify_latencies.clear();
            for (const auto &item : grid->as_array("verify_ms_grid")) {
                res.hw.verify_latencies.push_back(static_cast<float>(item.as_double("grid_item")));
            }
        }
        if (res.hw.verify_latencies.size() < 25) return std::nullopt;

        const json::Value *solo_grid = hw->find("solo_verify_ms_grid");
        if (solo_grid && solo_grid->is_array()) {
            res.hw.solo_verify_latencies.clear();
            for (const auto &item : solo_grid->as_array("solo_verify_ms_grid")) {
                res.hw.solo_verify_latencies.push_back(static_cast<float>(item.as_double("grid_item")));
            }
        }
        if (res.hw.solo_verify_latencies.size() < 9) return std::nullopt;

        const json::Value *acc = prof->find("acceptance_model");
        if (acc && acc->is_object()) {
            const json::Value *v_alpha = acc->find("alpha");
            if (v_alpha) res.acceptance.alpha = static_cast<float>(v_alpha->as_double("alpha"));

            const json::Value *v_beta = acc->find("beta");
            if (v_beta) res.acceptance.beta = static_cast<float>(v_beta->as_double("beta"));

            const json::Value *v_bias = acc->find("bias");
            if (v_bias) res.acceptance.bias = static_cast<float>(v_bias->as_double("bias"));

            const json::Value *v_cnt = acc->find("triplets_count");
            if (v_cnt) res.acceptance.sample_count = static_cast<int>(v_cnt->as_int("triplets_count"));
        }

        const json::Value *meta = prof->find("metadata");
        if (meta && meta->is_object()) {
            const json::Value *v_bits = meta->find("draft_head_bits");
            if (v_bits) {
                res.draft_head_bits = static_cast<int>(v_bits->as_int("draft_head_bits"));
                res.draft_head_q4 = (res.draft_head_bits == 4);
            } else {
                const json::Value *v_q4 = meta->find("draft_head_q4");
                if (v_q4) {
                    res.draft_head_q4 = v_q4->as_bool("draft_head_q4");
                    res.draft_head_bits = res.draft_head_q4 ? 4 : 8;
                }
            }
        }

        return res;
    } catch (...) {
        return std::nullopt;
    }
}

bool save_cached_profile(
    const std::string& cache_file,
    const MtpCalibratedProfile& profile
) {
    json::Value root = json::Value::object();
    std::ifstream in(cache_file);
    if (in.is_open()) {
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        try {
            root = json::Value::parse(text);
        } catch (...) {
            root = json::Value::object();
        }
        in.close();
    }

    if (!root.is_object()) root = json::Value::object();
    const json::Value *existing_profiles = root.find("profiles");
    json::Value profiles = (existing_profiles && existing_profiles->is_object()) ? *existing_profiles : json::Value::object();

    json::Value p = json::Value::object();
    json::Value meta = json::Value::object();
    meta.set("model_id", json::Value::string(profile.model_id));
    meta.set("weights_file", json::Value::string(profile.weights_path));
    meta.set("weights_strix_hash", json::Value::integer(profile.weights_strix_hash));
    meta.set("weights_size", json::Value::integer(profile.weights_size));
    meta.set("weights_mtime", json::Value::integer(profile.weights_mtime));
    meta.set("draft_head_bits", json::Value::integer(profile.draft_head_bits));
    meta.set("draft_head_q4", json::Value::boolean(profile.draft_head_q4));
    meta.set("mtp_vocab_filter", json::Value::string(profile.mtp_vocab_filter));
    meta.set("gdn_precision", json::Value::string(profile.gdn_precision));
    p.set("metadata", meta);

    json::Value hw = json::Value::object();
    hw.set("draft_step_ms", json::Value::number(profile.hw.draft_step_ms));
    hw.set("draft_batch_step_ms", json::Value::number(profile.hw.draft_batch_step_ms));
    hw.set("verify_base_ms", json::Value::number(profile.hw.verify_base_ms));
    hw.set("verify_step_ms", json::Value::number(profile.hw.verify_step_ms));
    json::Value grid = json::Value::array();
    for (float v : profile.hw.verify_latencies) {
        grid.push(json::Value::number(v));
    }
    hw.set("verify_ms_grid", grid);

    json::Value solo_grid = json::Value::array();
    for (float v : profile.hw.solo_verify_latencies) {
        solo_grid.push(json::Value::number(v));
    }
    hw.set("solo_verify_ms_grid", solo_grid);
    p.set("hardware_latencies", hw);

    json::Value acc = json::Value::object();
    acc.set("alpha", json::Value::number(profile.acceptance.alpha));
    acc.set("beta", json::Value::number(profile.acceptance.beta));
    acc.set("bias", json::Value::number(profile.acceptance.bias));
    acc.set("triplets_count", json::Value::integer(profile.acceptance.sample_count));
    p.set("acceptance_model", acc);

    profiles.set(profile.fingerprint, p);
    root.set("profiles", profiles);
    root.set("version", json::Value::integer(1));

    std::ofstream out(cache_file);
    if (!out.is_open()) return false;
    out << root.dump();
    return true;
}

} // namespace strix
