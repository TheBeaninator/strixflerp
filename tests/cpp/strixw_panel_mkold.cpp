// Writes small strixw v4 files with strixite's own (harvested) StrixwWriter, laid out like the fleet's
// Qwen3.8-Flash-Next file (tensor names and classes of convert_qwen4exp make_plan, every encoding the fleet uses,
// small shapes), for tests/test_convert_strixw_panel.py. Built by
// catalog/flerp/tool/strixw_patch/src/flerp/strixw_patch/build.sh OUT this.cpp (harvested sources only).
//   mkold OLD.strixw BAD.strixw
// BAD is the same with mtp.fc_hidden.weight at 130 rows (not a whole number of 16-row panels).

#include "common/check.hpp"
#include "formats/q4.hpp"
#include "formats/q8.hpp"
#include "formats/strixw.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strix;

namespace {

uint64_t rng = 12345;
float rnd() {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return (float)((int64_t)(rng % 20001) - 10000) / 70001.0f;
}
std::vector<float> rf(int64_t n) {
    std::vector<float> v((size_t)n);
    for (float &x : v) x = rnd();
    return v;
}

struct T {
    std::string name;
    StrixwEncoding enc;
    int64_t N, K, G, experts;
};

void make(const std::string &path, int64_t fc_rows) {
    const std::string L = "model.language_model.", L0 = L + "layers.0.", L1 = L + "layers.1.", M = "mtp.layers.0.";
    const auto Q4 = StrixwEncoding::Q4RowMajor, Q8 = StrixwEncoding::Q8RowMajor;
    std::vector<T> ts = {
        {L + "embed_tokens.weight", StrixwEncoding::BF16, 64, 256, 0, 0},
        {L + "hyper_connection_mixer.mix_down", Q8, 32, 256, 64, 0},
        {L + "hyper_connection_mixer.mix_up", StrixwEncoding::Q8ChunkMajor, 64, 64, 64, 0},
        {L + "hyper_connection_mixer.hc_norm.weight", StrixwEncoding::F32, 64, 0, 0, 0},
        {"lm_head.weight", Q8, 96, 256, 64, 0},
        {L0 + "mlp.router", StrixwEncoding::BF16, 5, 256, 0, 0},
        {L0 + "mlp.experts_gate_up", Q4, 4 * 32, 256, 128, 4},
        {L0 + "mlp.shared_expert.gate_up", Q8, 32, 256, 64, 0},
        {L0 + "mlp.experts_down", Q4, 4 * 256, 64, 32, 4},
        {L0 + "mlp.shared_expert.down_proj.weight", Q8, 256, 64, 32, 0},
        {L0 + "linear_attn.in_proj", Q4, 48, 256, 128, 0},
        {L0 + "linear_attn.out_proj.weight", Q8, 256, 128, 64, 0},
        {L0 + "linear_attn.A_log", StrixwEncoding::F32, 4, 0, 0, 0},
        {L0 + "attn_hyper_connection.mix_down", Q8, 32, 256, 64, 0},
        {L0 + "attn_hyper_connection.mix_up", StrixwEncoding::Q8ChunkMajor, 64, 64, 64, 0},
        {L0 + "ple.kv_proj", Q8, 32, 256, 64, 0},
        {L0 + "ple.ple_embedding.layer_multipliers", StrixwEncoding::I64, 3, 0, 0, 0},
        {L1 + "self_attn.qkv", Q8, 80, 256, 64, 0},
        {L1 + "self_attn.o_proj.weight", Q8, 256, 128, 64, 0},
        {L1 + "self_attn.q_norm.weight", StrixwEncoding::F32, 32, 0, 0, 0},
        {"mtp.fc_embedding.weight", Q8, 256, 512, 64, 0},
        {"mtp.fc_hidden.weight", Q8, fc_rows, 512, 64, 0},
        {M + "mlp.experts_gate_up", Q4, 4 * 32, 256, 128, 4},
        {M + "mlp.experts_down", Q4, 4 * 256, 64, 32, 4},
    };
    std::vector<StrixwTensor> recs;
    for (const T &t : ts) {
        StrixwTensor r;
        r.name = t.name, r.encoding = t.enc, r.group_size = t.G, r.experts = t.experts;
        r.shape = t.K ? std::vector<int64_t>{t.N, t.K} : std::vector<int64_t>{t.N};
        recs.push_back(r);
    }
    const std::map<std::string, std::string> meta = {
        {"layout", "exp_gu=q4g128,exp_down=q4g32,sh_gu=q8g64,sh_down=q8g32,gdn_in=q4g128,gdn_out=q8g64,attn_qkv=q8g64,"
                   "attn_o=q8g64,hc_down=q8g64,hc_up=q8g64,ple_kv=q8g64,lm_head=q8g64,mtp_fc=q8g64,embed=bf16"},
        {"source_fingerprint", "0123456789abcdef"},
        {"converter_git", "selftest"},
        {"shared_expert", "separate"}};
    StrixwWriter w(path, meta, recs);
    for (size_t i = 0; i < ts.size(); ++i) {
        const T &t = ts[i];
        auto put3 = [&](const std::vector<uint8_t> &q, const std::vector<uint16_t> &s, const std::vector<uint16_t> &m) {
            w.write(i, StrixwRole::Q, q.data(), q.size());
            w.write(i, StrixwRole::Scale, s.data(), s.size() * 2);
            w.write(i, StrixwRole::Min, m.data(), m.size() * 2);
        };
        if (t.enc == Q4) {
            const auto f = rf(t.N * t.K);
            const Q4Weight q = quantize_q4(f.data(), t.N, t.K, t.G);
            put3(q.q, q.scale, q.minv);
        } else if (t.enc == Q8) {
            const auto f = rf(t.N * t.K);
            const Q8Weight q = quantize_q8(f.data(), t.N, t.K, t.G);
            put3(q.q, q.scale, q.minv);
        } else if (t.enc == StrixwEncoding::Q8ChunkMajor) {
            const auto f = rf(t.N * t.K);
            const Q8ChunkMajor q = to_chunk_major(quantize_q8(f.data(), t.N, t.K, t.G), t.name.c_str());
            put3(q.q, q.scale, q.minv);
        } else {
            std::vector<uint8_t> d(w.tensors()[i].components[0].bytes);
            for (auto &b : d) b = (uint8_t)(rnd() * 1000);
            w.write(i, StrixwRole::Data, d.data(), d.size());
        }
    }
    w.finish();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: mkold OLD BAD\n");
        return 2;
    }
    try {
        make(argv[1], 256);
        make(argv[2], 130);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "mkold: %s\n", e.what());
        return 1;
    }
    return 0;
}
