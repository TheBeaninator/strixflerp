// steer_check (STEER-2): activation steering is a run's own, bit for bit.
//
// Two slots, two prompts (each prefilled alone), then decode steps forwarded together in one batch, twice: unsteered, then with a vector on slot 1 only. Slot
// 0's logits must be bit-identical between the two batches (same batch shape, so no rounding difference is
// allowed: a run's results depend on its own slot and ids only); slot 1's must differ. Exit 0 when every check holds.
//
//   steer_check --weights W --ngram T --tokenizer J [--layer 24] [--scale 8]
#include "common/check.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace strix;

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> a;
        for (int i = 1; i + 1 < argc; i += 2) a[std::string(argv[i]).substr(2)] = argv[i + 1];
        const int64_t layer = a.count("layer") ? std::stoll(a["layer"]) : 24;
        const float scale = a.count("scale") ? std::stof(a["scale"]) : 8.0f;
        const Tokenizer tok(a.at("tokenizer"));
        Qwen4ExpModel model(a.at("weights"), a.at("ngram"), kernels::Act::BF16, false, 262144, 1.0f);
        Qwen4ExpSession ses(model, 8192, 2048, PrefillMath::WmmaBf16, /*mtp=*/false, parse_kv_format("q8"), 2);
        const auto &D = model.dims();
        const int64_t n4 = D.H * D.d;
        std::vector<float> host((size_t)n4);
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 0.01f);
        for (float &f : host) f = nd(rng);
        auto buf = std::make_shared<DeviceBuffer<float>>(host.size(), "steer check");
        buf->upload(host);
        auto set = std::make_shared<SteerSet>();
        set->layers.push_back(SteerLayer{layer, buf, scale});
        set->signature = "check";

        const std::vector<int32_t> p0 = tok.encode("<|im_start|>user\nName three rivers in Europe.<|im_end|>\n<|im_start|>assistant\n");
        const std::vector<int32_t> p1 = tok.encode("<|im_start|>user\nWrite one line about the sea.<|im_end|>\n<|im_start|>assistant\n");
        const int64_t V = D.vocab;
        auto batch = [&](bool steer, std::vector<float> *l0, std::vector<float> *l1) {
            ses.reset(0), ses.reset(1);
            ses.set_steer(0, nullptr);
            ses.set_steer(1, steer ? set : nullptr);
            // prompts prefill one run at a time (a multi-run forward is decode-sized); then decode steps batched
            std::vector<Qwen4ExpSession::BatchRun> one(1);
            one[0].seq = 0, one[0].ids = p0;
            std::vector<float> a0 = ses.forward_batch(one);
            one[0].seq = 1, one[0].ids = p1;
            std::vector<float> a1 = ses.forward_batch(one);
            std::vector<Qwen4ExpSession::BatchRun> runs(2);
            runs[0].seq = 0, runs[1].seq = 1;
            std::vector<float> out;
            for (int step = 0; step < 6; ++step) {
                auto argmax = [&](const std::vector<float> &l) { return (int32_t)(std::max_element(l.begin(), l.end()) - l.begin()); };
                runs[0].ids = {argmax(std::vector<float>(a0.end() - V, a0.end()))};
                runs[1].ids = {argmax(std::vector<float>(a1.end() - V, a1.end()))};
                out = ses.forward_batch(runs);
                a0.insert(a0.end(), out.begin(), out.begin() + V);
                a1.insert(a1.end(), out.begin() + V, out.begin() + 2 * V);
            }
            *l0 = std::move(a0), *l1 = std::move(a1);
        };
        std::vector<float> u0, u1, s0, s1, u0b, u1b;
        batch(false, &u0, &u1);
        batch(true, &s0, &s1);
        batch(false, &u0b, &u1b);
        const bool repeat = u0 == u0b && u1 == u1b;
        const bool own = u0 == s0;
        double diff = 0;
        for (size_t i = 0; i < u1.size(); ++i) diff = std::max(diff, (double)std::fabs(u1[i] - s1[i]));
        std::printf("unsteered batch repeats bit for bit: %s\n", repeat ? "yes" : "NO");
        std::printf("slot 0 unchanged by slot 1's steering (bit for bit): %s\n", own ? "yes" : "NO");
        std::printf("slot 1 changed by its steering: max |dlogit| %.4f %s\n", diff, diff > 1e-3 ? "yes" : "NO");
        return repeat && own && diff > 1e-3 ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "steer_check: %s\n", e.what());
        return 2;
    }
}
