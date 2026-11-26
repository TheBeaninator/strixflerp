// Evaluator and ground-truth validation tool for PPL (Perplexity) and KLD (Kullback-Leibler Divergence)
// Compares Baseline Q8 vs Hybrid Q6 token-by-token and dumps raw logits / statistics.

#include "runtime/qwen4exp.hpp"
#include "serve/tokenizer.hpp"
#include "common/check.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <queue>

using namespace strix;

namespace {

std::string escape_json(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20 || c == 0x7f) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += (char)c;
        }
    }
    return out;
}

std::string escape_csv(const std::string &s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out += c;
    }
    out += "\"";
    return out;
}

struct TokenCandidate {
    int32_t id = 0;
    float logit = 0.0f;
    double prob = 0.0;
};

std::vector<TokenCandidate> get_top_k(const float *z, int64_t V, double log_sum_exp, int K) {
    auto comp = [](const TokenCandidate &a, const TokenCandidate &b) {
        return a.logit > b.logit;
    };
    std::priority_queue<TokenCandidate, std::vector<TokenCandidate>, decltype(comp)> pq(comp);

    for (int32_t i = 0; i < (int32_t)V; ++i) {
        float val = z[i];
        if ((int)pq.size() < K) {
            pq.push({i, val, 0.0});
        } else if (val > pq.top().logit) {
            pq.pop();
            pq.push({i, val, 0.0});
        }
    }

    std::vector<TokenCandidate> res;
    while (!pq.empty()) {
        TokenCandidate tc = pq.top();
        pq.pop();
        tc.prob = std::exp((double)tc.logit - log_sum_exp);
        res.push_back(tc);
    }
    std::reverse(res.begin(), res.end());
    return res;
}

} // namespace

int main(int argc, char **argv) {
    std::string weights_q8 = "/models/Thinking-Cap-Qwen-3.8-FN-faf/weights.faf";
    std::string weights_q6 = "/models/Thinking-Cap-Qwen-3.8-FN-faf/weights-q6-hybrid.faf";
    std::string ngram_path = "/models/Thinking-Cap-Qwen-3.8-FN-faf/ngram.table";
    std::string tokenizer_path = "/models/Thinking-Cap-Qwen-3.8-FN-faf/tokenizer.json";
    std::string text_path = "/models/wikitext-2-test.txt";
    std::string output_dir = "/models/ppl_kld_eval";
    int64_t num_eval_tokens = 2048;
    int64_t chunk_size = 16;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights-q8" && i + 1 < argc) weights_q8 = argv[++i];
        else if (arg == "--weights-q6" && i + 1 < argc) weights_q6 = argv[++i];
        else if (arg == "--ngram" && i + 1 < argc) ngram_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--text" && i + 1 < argc) text_path = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) num_eval_tokens = std::stoll(argv[++i]);
        else if (arg == "--chunk-size" && i + 1 < argc) { chunk_size = std::stoll(argv[++i]); if (chunk_size > 16) chunk_size = 16; }
        else if (arg == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
    }

    std::filesystem::create_directories(output_dir);
    std::string q8_tmp_bin = output_dir + "/q8_logits_temp.bin";

    std::cout << "=================================================================" << std::endl;
    std::cout << "  FASTASFUCK GROUND-TRUTH ACCURACY & DIVERGENCE BENCHMARK (PPL/KLD)" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "Q8 Baseline Weights : " << weights_q8 << std::endl;
    std::cout << "Hybrid Q6 Weights   : " << weights_q6 << std::endl;
    std::cout << "Evaluation Corpus   : " << text_path << std::endl;
    std::cout << "Tokens to evaluate  : " << num_eval_tokens << " (chunk size: " << chunk_size << ")" << std::endl;
    std::cout << "Output Directory    : " << output_dir << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Load Tokenizer & Corpus
    std::cout << "[1/4] Loading Tokenizer and encoding corpus..." << std::endl;
    Tokenizer tok(tokenizer_path);
    std::ifstream text_file(text_path);
    STRIX_CHECK(text_file.is_open(), "Cannot open text corpus: ", text_path);
    std::stringstream text_ss;
    text_ss << text_file.rdbuf();
    std::string raw_text = text_ss.str();

    std::vector<int32_t> all_tokens = tok.encode(raw_text);
    std::cout << "      Total corpus tokens: " << all_tokens.size() << std::endl;
    STRIX_CHECK((int64_t)all_tokens.size() >= num_eval_tokens + 1,
                "Corpus has fewer tokens (", all_tokens.size(), ") than requested (", num_eval_tokens + 1, ")");

    int64_t V = 0;

    // Phase 1: Forward Baseline Q8
    std::cout << "\n[2/4] Executing Phase 1: Baseline Q8 Forward Pass..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    {
        Qwen4ExpModel model_q8(weights_q8, ngram_path, kernels::Act::BF16, false,
                               8388608, 1.0f, WeightPlacement::Device);
        V = model_q8.dims().vocab;
        std::cout << "      Vocabulary size: " << V << " classes" << std::endl;

        Qwen4ExpSession session_q8(model_q8, num_eval_tokens + 128, num_eval_tokens + 128,
                                   PrefillMath::WmmaBf16, false, KvFormat::Q8, 1);
        PrecisionOptions prec;
        prec.gdn = kernels::DrPrecision::HalfU;
        prec.attention = kernels::AttnPrecision::Fast;
        session_q8.set_precision(prec);

        FILE *fp_out = fopen(q8_tmp_bin.c_str(), "wb");
        STRIX_CHECK(fp_out != nullptr, "Cannot open temp file for writing: ", q8_tmp_bin);

        int64_t processed = 0;
        while (processed < num_eval_tokens) {
            int64_t cur_chunk = std::min(chunk_size, num_eval_tokens - processed);
            std::vector<int32_t> chunk_ids(all_tokens.begin() + processed,
                                           all_tokens.begin() + processed + cur_chunk);
            std::vector<float> logits = session_q8.forward(chunk_ids, cur_chunk);
            STRIX_CHECK(logits.size() == (size_t)(cur_chunk * V), "Logits size mismatch in Q8 forward");
            fwrite(logits.data(), sizeof(float), logits.size(), fp_out);
            processed += cur_chunk;
            std::cout << "\r      Q8 Processed: " << processed << " / " << num_eval_tokens << " tokens..." << std::flush;
        }
        fclose(fp_out);
        std::cout << std::endl;
    }
    auto t1 = std::chrono::steady_clock::now();
    double q8_time_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "      Phase 1 Complete in " << std::fixed << std::setprecision(2) << q8_time_s << " s ("
              << (double)num_eval_tokens / q8_time_s << " tok/s)" << std::endl;

    // Phase 2: Forward Hybrid Q6 and stream-compare with Q8
    std::cout << "\n[3/4] Executing Phase 2: Hybrid Q6 Forward Pass & Ground-Truth Divergence Analysis..." << std::endl;
    auto t2 = std::chrono::steady_clock::now();

    std::ofstream jsonl_out(output_dir + "/token_by_token.jsonl");
    std::ofstream csv_out(output_dir + "/token_by_token.csv");
    csv_out << "pos,token_id,token_text,target_id,target_text,q8_nll,q6_nll,q8_top1_id,q8_top1_text,q6_top1_id,q6_top1_text,top1_match,kld,cosine_sim,max_abs_logit_diff,l2_diff\n";

    FILE *fp_in = fopen(q8_tmp_bin.c_str(), "rb");
    STRIX_CHECK(fp_in != nullptr, "Cannot open temp file for reading: ", q8_tmp_bin);

    double sum_q8_nll = 0.0;
    double sum_q6_nll = 0.0;
    double sum_kld = 0.0;
    double max_kld = 0.0;
    std::vector<double> all_kld;
    all_kld.reserve(num_eval_tokens);

    int64_t top1_matches = 0;
    int64_t top5_matches = 0;
    double sum_cosine_sim = 0.0;
    double sum_max_abs_diff = 0.0;
    double global_max_abs_diff = 0.0;

    {
        Qwen4ExpModel model_q6(weights_q6, ngram_path, kernels::Act::BF16, false,
                               8388608, 1.0f, WeightPlacement::Device);
        Qwen4ExpSession session_q6(model_q6, num_eval_tokens + 128, num_eval_tokens + 128,
                                   PrefillMath::WmmaBf16, false, KvFormat::Q8, 1);
        PrecisionOptions prec;
        prec.gdn = kernels::DrPrecision::HalfU;
        prec.attention = kernels::AttnPrecision::Fast;
        session_q6.set_precision(prec);

        std::vector<float> q8_chunk_buf(chunk_size * V);
        int64_t processed = 0;

        while (processed < num_eval_tokens) {
            int64_t cur_chunk = std::min(chunk_size, num_eval_tokens - processed);
            std::vector<int32_t> chunk_ids(all_tokens.begin() + processed,
                                           all_tokens.begin() + processed + cur_chunk);

            std::vector<float> q6_logits = session_q6.forward(chunk_ids, cur_chunk);
            STRIX_CHECK(q6_logits.size() == (size_t)(cur_chunk * V), "Logits size mismatch in Q6 forward");

            size_t read_cnt = fread(q8_chunk_buf.data(), sizeof(float), cur_chunk * V, fp_in);
            STRIX_CHECK(read_cnt == (size_t)(cur_chunk * V), "Failed to read Q8 logits from temp file");

            for (int64_t local_idx = 0; local_idx < cur_chunk; ++local_idx) {
                int64_t global_idx = processed + local_idx;
                int32_t cur_token_id = all_tokens[global_idx];
                int32_t target_token_id = all_tokens[global_idx + 1];

                const float *z_p = q8_chunk_buf.data() + local_idx * V;
                const float *z_q = q6_logits.data() + local_idx * V;

                // Max and log-sum-exp for P (Q8) and Q (Q6)
                float max_p = -1e30f, max_q = -1e30f;
                int32_t top1_p_id = 0, top1_q_id = 0;
                float top1_p_val = -1e30f, top1_q_val = -1e30f;

                for (int32_t i = 0; i < V; ++i) {
                    if (z_p[i] > max_p) max_p = z_p[i];
                    if (z_q[i] > max_q) max_q = z_q[i];
                    if (z_p[i] > top1_p_val) { top1_p_val = z_p[i]; top1_p_id = i; }
                    if (z_q[i] > top1_q_val) { top1_q_val = z_q[i]; top1_q_id = i; }
                }

                double sum_exp_p = 0.0, sum_exp_q = 0.0;
                for (int32_t i = 0; i < V; ++i) {
                    sum_exp_p += std::exp((double)z_p[i] - max_p);
                    sum_exp_q += std::exp((double)z_q[i] - max_q);
                }
                double lse_p = max_p + std::log(sum_exp_p);
                double lse_q = max_q + std::log(sum_exp_q);

                double nll_p = -((double)z_p[target_token_id] - lse_p);
                double nll_q = -((double)z_q[target_token_id] - lse_q);
                sum_q8_nll += nll_p;
                sum_q6_nll += nll_q;

                // Exact KLD over full 248,320 vocabulary classes
                double kld_step = 0.0;
                double dot = 0.0, norm_p = 0.0, norm_q = 0.0;
                double max_abs_diff_step = 0.0;
                double l2_diff_sq_step = 0.0;

                for (int32_t i = 0; i < V; ++i) {
                    double val_p = z_p[i];
                    double val_q = z_q[i];
                    double log_p = val_p - lse_p;
                    double log_q = val_q - lse_q;
                    double p = std::exp(log_p);

                    kld_step += p * (log_p - log_q);

                    dot += val_p * val_q;
                    norm_p += val_p * val_p;
                    norm_q += val_q * val_q;

                    double diff = std::abs(val_p - val_q);
                    if (diff > max_abs_diff_step) max_abs_diff_step = diff;
                    l2_diff_sq_step += diff * diff;
                }

                if (kld_step < 0.0 && kld_step > -1e-7) kld_step = 0.0; // clamp machine epsilon
                sum_kld += kld_step;
                all_kld.push_back(kld_step);
                if (kld_step > max_kld) max_kld = kld_step;

                double cos_sim = (norm_p > 0.0 && norm_q > 0.0) ? (dot / (std::sqrt(norm_p) * std::sqrt(norm_q))) : 1.0;
                sum_cosine_sim += cos_sim;
                sum_max_abs_diff += max_abs_diff_step;
                if (max_abs_diff_step > global_max_abs_diff) global_max_abs_diff = max_abs_diff_step;

                bool top1_match = (top1_p_id == top1_q_id);
                if (top1_match) top1_matches++;

                auto top5_p = get_top_k(z_p, V, lse_p, 5);
                auto top5_q = get_top_k(z_q, V, lse_q, 5);

                bool top5_overlap = false;
                for (const auto &cand_p : top5_p) {
                    for (const auto &cand_q : top5_q) {
                        if (cand_p.id == cand_q.id) { top5_overlap = true; break; }
                    }
                    if (top5_overlap) break;
                }
                if (top5_overlap) top5_matches++;

                std::string cur_tok_str = tok.token_bytes(cur_token_id);
                std::string tgt_tok_str = tok.token_bytes(target_token_id);
                std::string top1_p_str = tok.token_bytes(top1_p_id);
                std::string top1_q_str = tok.token_bytes(top1_q_id);

                // Write JSONL
                jsonl_out << "{\"pos\":" << global_idx
                          << ",\"token_id\":" << cur_token_id
                          << ",\"token_text\":\"" << escape_json(cur_tok_str) << "\""
                          << ",\"target_id\":" << target_token_id
                          << ",\"target_text\":\"" << escape_json(tgt_tok_str) << "\""
                          << ",\"q8_nll\":" << nll_p
                          << ",\"q6_nll\":" << nll_q
                          << ",\"q8_target_prob\":" << std::exp(-nll_p)
                          << ",\"q6_target_prob\":" << std::exp(-nll_q)
                          << ",\"q8_top1_id\":" << top1_p_id
                          << ",\"q8_top1_text\":\"" << escape_json(top1_p_str) << "\""
                          << ",\"q6_top1_id\":" << top1_q_id
                          << ",\"q6_top1_text\":\"" << escape_json(top1_q_str) << "\""
                          << ",\"top1_match\":" << (top1_match ? "true" : "false")
                          << ",\"kld\":" << kld_step
                          << ",\"cosine_sim\":" << cos_sim
                          << ",\"max_abs_logit_diff\":" << max_abs_diff_step
                          << ",\"l2_diff\":" << std::sqrt(l2_diff_sq_step)
                          << ",\"q8_top5\":[";
                for (size_t k = 0; k < top5_p.size(); ++k) {
                    if (k > 0) jsonl_out << ",";
                    jsonl_out << "{\"id\":" << top5_p[k].id
                              << ",\"text\":\"" << escape_json(tok.token_bytes(top5_p[k].id)) << "\""
                              << ",\"prob\":" << top5_p[k].prob
                              << ",\"logit\":" << top5_p[k].logit << "}";
                }
                jsonl_out << "],\"q6_top5\":[";
                for (size_t k = 0; k < top5_q.size(); ++k) {
                    if (k > 0) jsonl_out << ",";
                    jsonl_out << "{\"id\":" << top5_q[k].id
                              << ",\"text\":\"" << escape_json(tok.token_bytes(top5_q[k].id)) << "\""
                              << ",\"prob\":" << top5_q[k].prob
                              << ",\"logit\":" << top5_q[k].logit << "}";
                }
                jsonl_out << "]}\n";

                // Write CSV
                csv_out << global_idx << ","
                        << cur_token_id << ","
                        << escape_csv(cur_tok_str) << ","
                        << target_token_id << ","
                        << escape_csv(tgt_tok_str) << ","
                        << nll_p << ","
                        << nll_q << ","
                        << top1_p_id << ","
                        << escape_csv(top1_p_str) << ","
                        << top1_q_id << ","
                        << escape_csv(top1_q_str) << ","
                        << (top1_match ? "1" : "0") << ","
                        << kld_step << ","
                        << cos_sim << ","
                        << max_abs_diff_step << ","
                        << std::sqrt(l2_diff_sq_step) << "\n";
            }

            processed += cur_chunk;
            std::cout << "\r      Q6 & Compare Processed: " << processed << " / " << num_eval_tokens << " tokens..." << std::flush;
        }
        std::cout << std::endl;
    }
    fclose(fp_in);
    jsonl_out.close();
    csv_out.close();
    std::filesystem::remove(q8_tmp_bin);

    auto t3 = std::chrono::steady_clock::now();
    double q6_time_s = std::chrono::duration<double>(t3 - t2).count();
    std::cout << "      Phase 2 Complete in " << std::fixed << std::setprecision(2) << q6_time_s << " s ("
              << (double)num_eval_tokens / q6_time_s << " tok/s)" << std::endl;

    // Phase 4: Compute Aggregates & Print
    std::cout << "\n[4/4] Computing Final Mathematical Invariants & Summary Statistics..." << std::endl;

    double ppl_q8 = std::exp(sum_q8_nll / (double)num_eval_tokens);
    double ppl_q6 = std::exp(sum_q6_nll / (double)num_eval_tokens);
    double ppl_delta = ppl_q6 - ppl_q8;
    double ppl_rel_pct = (ppl_delta / ppl_q8) * 100.0;

    double mean_kld = sum_kld / (double)num_eval_tokens;
    std::sort(all_kld.begin(), all_kld.end());
    double median_kld = all_kld[all_kld.size() / 2];
    double p95_kld = all_kld[(size_t)(all_kld.size() * 0.95)];
    double p99_kld = all_kld[(size_t)(all_kld.size() * 0.99)];

    double top1_agreement = (double)top1_matches / (double)num_eval_tokens * 100.0;
    double top5_agreement = (double)top5_matches / (double)num_eval_tokens * 100.0;
    double mean_cosine = sum_cosine_sim / (double)num_eval_tokens;
    double mean_max_abs_diff = sum_max_abs_diff / (double)num_eval_tokens;

    // Write summary.json
    std::ofstream summary_out(output_dir + "/summary.json");
    summary_out << "{\n"
                << "  \"corpus\": \"" << text_path << "\",\n"
                << "  \"tokens_evaluated\": " << num_eval_tokens << ",\n"
                << "  \"q8_ppl\": " << std::setprecision(6) << ppl_q8 << ",\n"
                << "  \"q6_ppl\": " << std::setprecision(6) << ppl_q6 << ",\n"
                << "  \"ppl_delta\": " << std::setprecision(6) << ppl_delta << ",\n"
                << "  \"ppl_rel_pct\": " << std::setprecision(4) << ppl_rel_pct << ",\n"
                << "  \"mean_kld\": " << std::setprecision(8) << mean_kld << ",\n"
                << "  \"median_kld\": " << std::setprecision(8) << median_kld << ",\n"
                << "  \"p95_kld\": " << std::setprecision(8) << p95_kld << ",\n"
                << "  \"p99_kld\": " << std::setprecision(8) << p99_kld << ",\n"
                << "  \"max_kld\": " << std::setprecision(8) << max_kld << ",\n"
                << "  \"top1_agreement_pct\": " << std::setprecision(4) << top1_agreement << ",\n"
                << "  \"top5_agreement_pct\": " << std::setprecision(4) << top5_agreement << ",\n"
                << "  \"mean_cosine_similarity\": " << std::setprecision(6) << mean_cosine << ",\n"
                << "  \"mean_abs_logit_diff\": " << std::setprecision(6) << mean_max_abs_diff << ",\n"
                << "  \"global_max_abs_logit_diff\": " << std::setprecision(6) << global_max_abs_diff << ",\n"
                << "  \"q8_time_seconds\": " << std::setprecision(2) << q8_time_s << ",\n"
                << "  \"q6_time_seconds\": " << std::setprecision(2) << q6_time_s << "\n"
                << "}\n";
    summary_out.close();

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "                      FINAL EVALUATION RESULTS                   " << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << " Evaluated Tokens         : " << num_eval_tokens << std::endl;
    std::cout << " Baseline Q8 PPL           : " << std::fixed << std::setprecision(4) << ppl_q8 << std::endl;
    std::cout << " Hybrid Q6 PPL             : " << std::fixed << std::setprecision(4) << ppl_q6 << std::endl;
    std::cout << " Delta PPL (Q6 - Q8)       : " << std::showpos << ppl_delta << " (" << ppl_rel_pct << "%)" << std::noshowpos << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << " Mean KL-Divergence (KLD)  : " << std::scientific << std::setprecision(6) << mean_kld << std::defaultfloat << std::endl;
    std::cout << " Median KLD                : " << std::scientific << std::setprecision(6) << median_kld << std::defaultfloat << std::endl;
    std::cout << " 95th Percentile KLD       : " << std::scientific << std::setprecision(6) << p95_kld << std::defaultfloat << std::endl;
    std::cout << " Max KLD                   : " << std::scientific << std::setprecision(6) << max_kld << std::defaultfloat << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << " Top-1 Token Agreement     : " << std::fixed << std::setprecision(2) << top1_agreement << " %" << std::endl;
    std::cout << " Top-5 Token Overlap       : " << std::fixed << std::setprecision(2) << top5_agreement << " %" << std::endl;
    std::cout << " Mean Logit Cosine Sim     : " << std::fixed << std::setprecision(6) << mean_cosine << std::endl;
    std::cout << " Mean Abs Logit Diff       : " << std::fixed << std::setprecision(4) << mean_max_abs_diff << std::endl;
    std::cout << " Global Max Logit Diff     : " << std::fixed << std::setprecision(4) << global_max_abs_diff << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << " Raw token-by-token dump   : " << output_dir << "/token_by_token.jsonl" << std::endl;
    std::cout << " Raw CSV dump              : " << output_dir << "/token_by_token.csv" << std::endl;
    std::cout << " Summary JSON              : " << output_dir << "/summary.json" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
