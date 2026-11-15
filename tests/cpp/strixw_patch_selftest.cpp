// Builds the two reference files tests/test_strixw_patch.py compares tool.flerp.strixw_patch against, with strixite's
// own writer and quantizers (the converter's per-tensor ops): from a small BF16 checkpoint CKPT,
//   OLD.strixw       mtp_fc at q8gG_old (like the fleet file)
//   EXPECTED.strixw  the same file with mtp_fc at q4g<G_new> - what the tool must produce byte for byte
// Both carry identical metadata except the layout string (source_fingerprint from checkpoint_fingerprint).
//   strixw_patch_selftest CKPT OLD EXPECTED G_new
#include "formats/convert_qwen4exp.hpp"
#include "formats/q4.hpp"
#include "formats/q8.hpp"
#include "formats/safetensors.hpp"
#include "formats/strixw.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace strix;

namespace {

std::vector<float> widen(const SafetensorsFile &f, const TensorInfo &ti) {
    const auto *b = static_cast<const uint16_t *>(f.data(ti));
    std::vector<float> out((size_t)ti.numel());
    for (size_t i = 0; i < out.size(); ++i) {
        const uint32_t u = (uint32_t)b[i] << 16;
        std::memcpy(&out[i], &u, 4);
    }
    return out;
}

void make(const std::string &ckpt, const std::string &path, bool q4_mtp, int64_t g_new) {
    const SafetensorsFile f((std::filesystem::path(ckpt) / "model.safetensors").string());
    const std::string mtp_fmt = q4_mtp ? "q4g" + std::to_string(g_new) : "q8g64";
    std::map<std::string, std::string> meta = {
        {"arch", "qwen4exp"},
        {"layout", "exp_gu=q4g128,lm_head=q8g64,mtp_fc=" + mtp_fmt + ",embed=bf16"},
        {"source_fingerprint", checkpoint_fingerprint(ckpt)},
        {"converter_git", "selftest"},
        {"created", "2026-11-11T00:00:00Z"}};
    // order mirrors the converter's: globals, a layer, then MTP
    struct Plan { std::string name; StrixwEncoding enc; int64_t G; };
    const std::vector<Plan> plan = {
        {"model.embed.weight", StrixwEncoding::BF16, 0},
        {"lm_head.weight", StrixwEncoding::Q8RowMajor, 64},
        {"model.layers.0.experts.weight", StrixwEncoding::Q4RowMajor, 128},
        {"mtp.fc_embedding.weight", q4_mtp ? StrixwEncoding::Q4RowMajor : StrixwEncoding::Q8RowMajor,
         q4_mtp ? g_new : 64},
        {"mtp.fc_hidden.weight", q4_mtp ? StrixwEncoding::Q4RowMajor : StrixwEncoding::Q8RowMajor,
         q4_mtp ? g_new : 64},
        {"mtp.norm.weight", StrixwEncoding::F32, 0}};
    std::vector<StrixwTensor> ts;
    for (const Plan &p : plan) {
        StrixwTensor t;
        t.name = p.name, t.encoding = p.enc, t.group_size = p.G, t.shape = f.get(p.name).shape;
        if (p.name == "model.layers.0.experts.weight") t.experts = 4, t.parts.push_back({p.name, 0, t.shape[0]});
        ts.push_back(t);
    }
    StrixwWriter w(path, meta, ts);
    for (size_t i = 0; i < plan.size(); ++i) {
        const TensorInfo &ti = f.get(plan[i].name);
        const std::vector<float> x = widen(f, ti);
        const int64_t N = ti.shape[0], K = ti.numel() / N;
        switch (plan[i].enc) {
        case StrixwEncoding::BF16: w.write(i, StrixwRole::Data, f.data(ti), ti.byte_length); break;
        case StrixwEncoding::F32: w.write(i, StrixwRole::Data, x.data(), x.size() * 4); break;
        case StrixwEncoding::Q4RowMajor: {
            const Q4Weight q = quantize_q4(x.data(), N, K, plan[i].G);
            w.write(i, StrixwRole::Q, q.q.data(), q.q.size());
            w.write(i, StrixwRole::Scale, q.scale.data(), q.scale.size() * 2);
            w.write(i, StrixwRole::Min, q.minv.data(), q.minv.size() * 2);
            break;
        }
        case StrixwEncoding::Q8RowMajor: {
            const Q8Weight q = quantize_q8(x.data(), N, K, plan[i].G);
            w.write(i, StrixwRole::Q, q.q.data(), q.q.size());
            w.write(i, StrixwRole::Scale, q.scale.data(), q.scale.size() * 2);
            w.write(i, StrixwRole::Min, q.minv.data(), q.minv.size() * 2);
            break;
        }
        default: return;
        }
    }
    w.finish();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: strixw_patch_selftest CKPT OLD EXPECTED G_new\n");
        return 2;
    }
    try {
        make(argv[1], argv[2], false, 0);
        make(argv[1], argv[3], true, std::stoll(argv[4]));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
