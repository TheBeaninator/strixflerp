// The strixite executor behind include/flerp/executor.h: a C++ shim over LmBackend (serve/engine.hpp), with
// Qwen4ExpBackend as the backend it opens. docs/EXECUTOR-ABI.md has the call-by-call mapping.
//
// flerp_open keys (all strings): weights, ngram (required); capacity, chunk, mtp (on|off), mtp_vocab, kv_cache
// (bf16|q8), gdn_precision, attention_precision, rope_yarn_factor, seqs, kv_generations, ngram_cache_rows, verify
// (single|batch). The
// defaults are the served ones (serve/server_config.cpp at 20937dc), so an open with only the two paths is the
// server's backend.
//
// verify: LmBackend has two verify paths and they are not the same arithmetic. A one-slot server verifies with
// forward_verify_rows + keep_verify / keep_verify_prefix / drop_verify (Engine::run); a server with several slots
// goes through its arbiter to verify_batch + keep_verify(seq, rows, carry) / drop_verify(seq), one forward for every
// slot's run. The adapter does what the server does: the single path when the executor has one seq, the batched
// path otherwise. verify=batch forces the batched path on a one-seq executor (the conformance gate uses it to show
// the difference); verify=single on a several-seq executor is refused.
#include "flerp/executor.h"

#include "common/check.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/engine.hpp"
#include "serve/host_buffer.hpp"
#include "serve/qwen4exp_backend.hpp"
#include "serve/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace strix;

namespace {

// LmBackend's snapshot slots a caller may hold: kTurnSlot, kSystemSlot, kUserSlot. kDraftSlot is the verify's.
constexpr int kSlots[3] = {LmBackend::kTurnSlot, LmBackend::kSystemSlot, LmBackend::kUserSlot};

struct Seq {
    bool open = false;
    bool pending = false;       // a verify awaits keep / keep_prefix / drop
    int64_t verify_rows = 0;
    std::vector<int32_t> verify_ids;  // the single path's keep_verify_prefix takes the ids back
    bool snap[3] = {false, false, false};
};

std::string g_open_error;

}  // namespace

struct flerp_executor {
    std::unique_ptr<Qwen4ExpModel> model;
    std::unique_ptr<Qwen4ExpBackend> backend;
    LmBackend *b = nullptr;
    std::vector<Seq> seqs;
    mutable std::string error;
    std::string describe, fingerprint;
    flerp_stats stats{};
    bool mtp = false;
    bool batch_verify = false;  // verify_batch (the several-slot path) instead of forward_verify_rows
};

namespace {

flerp_status fail(const flerp_executor *ex, flerp_status st, const std::string &what) {
    ex->error = what;
    return st;
}

// Runs f; a strixite Error (STRIX_CHECK) or any exception becomes FLERP_E_FAILED with its text.
template <class F> flerp_status guarded(flerp_executor *ex, F &&f) {
    try {
        return f();
    } catch (const std::exception &e) {
        return fail(ex, FLERP_E_FAILED, e.what());
    }
}

flerp_status seq_ok(const flerp_executor *ex, int32_t seq, bool want_pending) {
    if (seq < 0 || seq >= (int32_t)ex->seqs.size() || !ex->seqs[(size_t)seq].open)
        return fail(ex, FLERP_E_ARGUMENT, "seq " + std::to_string(seq) + " is not open");
    if (ex->seqs[(size_t)seq].pending != want_pending)
        return fail(ex, FLERP_E_STATE, want_pending ? "no verify is pending on seq " + std::to_string(seq)
                                                    : "a verify is pending on seq " + std::to_string(seq) +
                                                          ": keep, keep_prefix or drop first");
    return FLERP_OK;
}

// flerp_masks -> strixite LogitMasks (null when there is nothing to apply).
std::unique_ptr<LogitMasks> masks_of(const flerp_masks *m) {
    if (m == nullptr || (m->rows == 0 && m->n_bias == 0)) return nullptr;
    auto out = std::make_unique<LogitMasks>();
    if (m->rows > 0) {
        out->rows = m->rows, out->words = m->words;
        out->bits.assign(m->bits, m->bits + m->rows * m->words);
    }
    if (m->n_bias > 0) {
        auto bias = std::make_shared<LogitBias>();
        bias->id.assign(m->bias_id, m->bias_id + m->n_bias);
        bias->v.assign(m->bias, m->bias + m->n_bias);
        out->bias = std::move(bias);
    }
    return out;
}

// One strixite LogitRows into the caller's buffers with the asked delivery.
flerp_status deliver(const flerp_executor *ex, const LogitRows &l, uint32_t delivery, int64_t n_valid, flerp_rows *out) {
    if (out == nullptr) return fail(ex, FLERP_E_ARGUMENT, "results were asked for without a flerp_rows");
    const int64_t rows = l.rows, K = Sampler::kCandidates;
    out->rows = 0;
    if (out->nan != nullptr) {
        if ((int64_t)out->nan_cap < rows) return fail(ex, FLERP_E_ARGUMENT, "flerp_rows.nan is too small");
        for (int64_t r = 0; r < rows; ++r) out->nan[r] = r < (int64_t)l.nan.size() ? l.nan[(size_t)r] : 0;
    }
    if (delivery == FLERP_OUT_FULL) {
        if (l.cands != 0) return fail(ex, FLERP_E_FAILED, "the backend returned candidates where full rows were asked");
        if (out->full == nullptr || out->full_cap < l.full.size())
            return fail(ex, FLERP_E_ARGUMENT, "flerp_rows.full is too small");
        std::memcpy(out->full, l.full.data(), l.full.size() * sizeof(float));
    } else if (delivery == FLERP_OUT_CANDIDATES || delivery == FLERP_OUT_TOKENS) {
        const bool tokens = delivery == FLERP_OUT_TOKENS;
        if (tokens ? (out->tokens == nullptr || (int64_t)out->tokens_cap < rows)
                   : (out->cand_id == nullptr || out->cand_logit == nullptr || (int64_t)out->cand_cap < rows * K))
            return fail(ex, FLERP_E_ARGUMENT, tokens ? "flerp_rows.tokens is too small" : "flerp_rows.cand_* are too small");
        for (int64_t r = 0; r < rows; ++r) {
            if (l.cands > 0) {  // value descending, the lower id first on a tie (kernels/logits_topk)
                if (tokens) out->tokens[r] = l.cand_id[(size_t)(r * l.cands)];
                else
                    for (int64_t c = 0; c < K; ++c) {
                        const bool have = c < l.cands;
                        out->cand_id[r * K + c] = have ? l.cand_id[(size_t)(r * l.cands + c)] : -1;
                        out->cand_logit[r * K + c] = have ? l.cand_v[(size_t)(r * l.cands + c)] : -INFINITY;
                    }
                continue;
            }
            // full rows from a backend that does not reduce them: the same order on the host
            const float *x = l.full.data() + r * l.row;
            const int64_t n = n_valid > 0 ? std::min(n_valid, l.row) : l.row;
            if (tokens) {
                out->tokens[r] = top2(x, n).best;
                continue;
            }
            std::vector<int32_t> idx((size_t)n);
            for (int64_t i = 0; i < n; ++i) idx[(size_t)i] = (int32_t)i;
            const int64_t k = std::min(K, n);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int32_t a, int32_t c) { return x[a] > x[c] || (x[a] == x[c] && a < c); });
            for (int64_t c = 0; c < K; ++c) {
                out->cand_id[r * K + c] = c < k ? idx[(size_t)c] : -1;
                out->cand_logit[r * K + c] = c < k ? x[idx[(size_t)c]] : -INFINITY;
            }
        }
    } else {
        return fail(ex, FLERP_E_ARGUMENT, "delivery " + std::to_string(delivery) + " is not one FLERP_OUT_* value");
    }
    out->rows = rows;
    return FLERP_OK;
}

}  // namespace

extern "C" {

flerp_status flerp_open(const flerp_open_args *args, flerp_executor **out) {
    g_open_error.clear();
    if (args == nullptr || out == nullptr || args->size < sizeof(flerp_open_args)) {
        g_open_error = "flerp_open: null or short arguments";
        return FLERP_E_ARGUMENT;
    }
    std::map<std::string, std::string> cfg = {
        {"weights", ""},     {"ngram", ""},           {"capacity", "524288"},    {"chunk", "16384"},
        {"mtp", "on"},       {"mtp_vocab", "65536"},  {"kv_cache", "q8"},        {"gdn_precision", "half-u"},
        {"attention_precision", "exact"}, {"seqs", "1"}, {"kv_generations", "0"}, {"ngram_cache_rows", "8388608"},
        {"verify", ""}, {"rope_yarn_factor", "2"}};
    for (size_t i = 0; i < args->n_config; ++i) {
        const std::string key = args->config[i].key ? args->config[i].key : "";
        if (cfg.find(key) == cfg.end()) {
            g_open_error = "flerp_open (strixite): unknown key '" + key + "'";
            return FLERP_E_CONFIG;
        }
        cfg[key] = args->config[i].value ? args->config[i].value : "";
    }
    if (cfg["weights"].empty() || cfg["ngram"].empty()) {
        g_open_error = "flerp_open (strixite): weights and ngram are required";
        return FLERP_E_CONFIG;
    }
    if (cfg["mtp"] != "on" && cfg["mtp"] != "off") {
        g_open_error = "flerp_open (strixite): mtp must be on or off";
        return FLERP_E_CONFIG;
    }
    auto ex = std::make_unique<flerp_executor>();
    try {
        PrecisionOptions precision;
        precision.gdn = parse_gdn_precision(cfg["gdn_precision"]);
        precision.attention = parse_attention_precision(cfg["attention_precision"]);
        const KvFormat kv = parse_kv_format(cfg["kv_cache"]);
        ex->mtp = cfg["mtp"] == "on";
        ex->model = std::make_unique<Qwen4ExpModel>(cfg["weights"], cfg["ngram"], kernels::Act::BF16, false,
                                                    std::stoll(cfg["ngram_cache_rows"]), std::stof(cfg["rope_yarn_factor"]));
        ex->backend = std::make_unique<Qwen4ExpBackend>(*ex->model, std::stoll(cfg["capacity"]), std::stoll(cfg["chunk"]),
                                                        ex->mtp, std::stoll(cfg["mtp_vocab"]), precision, kv,
                                                        std::stoll(cfg["seqs"]), std::stoll(cfg["kv_generations"]));
        ex->b = ex->backend.get();
        ex->seqs.resize((size_t)ex->b->n_seqs());
        if (!cfg["verify"].empty() && cfg["verify"] != "single" && cfg["verify"] != "batch")
            throw Error("verify must be single or batch");
        if (cfg["verify"] == "single" && ex->b->n_seqs() > 1)
            throw Error("verify=single on an executor with several seqs: its server verifies through verify_batch");
        ex->batch_verify = cfg["verify"] == "batch" || ex->b->n_seqs() > 1;
        ex->describe = ex->b->describe();
        ex->fingerprint = ex->b->state_fingerprint();
    } catch (const std::exception &e) {
        g_open_error = std::string("flerp_open (strixite): ") + e.what();
        return FLERP_E_FAILED;
    }
    *out = ex.release();
    return FLERP_OK;
}

void flerp_close(flerp_executor *ex) { delete ex; }

flerp_status flerp_get_caps(const flerp_executor *ex, flerp_caps *out) {
    // a version 1 caller's struct ends at `flags`: it is filled as far as it goes and never written past
    if (out == nullptr || out->size < FLERP_HAS(flerp_caps, flags)) return fail(ex, FLERP_E_ARGUMENT, "flerp_caps is null or short");
    flerp_caps c{};
    c.size = (uint32_t)std::min<size_t>(out->size, FLERP_CAPS_V2_SIZE), c.abi = 2;  // version 3 calls: FLERP-CACHE, FLERP-STEER, FLERP-VISION
    c.positions = ex->b->capacity();
    c.max_forward = ex->b->max_chunk();
    c.max_verify_rows = ex->b->max_verify_rows();
    c.logits_row = ex->b->logits_row();
    c.candidates = (int32_t)Sampler::kCandidates;
    c.seqs = ex->b->n_seqs();
    c.snapshots = 3;
    c.max_draft = ex->b->has_mtp() ? 15 : 0;  // the engine's mtp-draft ceiling (kMtpMaxDraft)
    c.deliveries = FLERP_OUT_FULL | FLERP_OUT_CANDIDATES | FLERP_OUT_TOKENS;
    c.flags = FLERP_CAP_MASKS | FLERP_CAP_KEEP_PREFIX | FLERP_CAP_SNAPSHOT | FLERP_CAP_SNAPSHOT_BYTES |
              FLERP_CAP_SNAPSHOT_ANYWHERE | FLERP_CAP_FORWARD_ROWS |
              (ex->b->has_mtp() ? FLERP_CAP_DRAFT_STEP : 0u) | (ex->b->has_vision() ? FLERP_CAP_VISION : 0u) |
              (ex->b->n_seqs() > 1 ? FLERP_CAP_VERIFY_BATCH : 0u);
    c.sampling = 0, c.top_k_max = 0;  // version 2: no draws on the device (no FLERP_CAP_SAMPLED_TOKENS); the host samples
    std::memcpy(out, &c, c.size);
    return FLERP_OK;
}

const char *flerp_describe(const flerp_executor *ex) { return ex->describe.c_str(); }
const char *flerp_state_fingerprint(const flerp_executor *ex) { return ex->fingerprint.c_str(); }
const char *flerp_last_error(const flerp_executor *ex) { return ex ? ex->error.c_str() : g_open_error.c_str(); }
int flerp_failed(const flerp_executor *ex) { return ex->b->failed() ? 1 : 0; }

flerp_status flerp_seq_new(flerp_executor *ex, const flerp_seq_args *args, int32_t *seq) {
    if (seq == nullptr) return fail(ex, FLERP_E_ARGUMENT, "flerp_seq_new: null seq");
    if (args != nullptr && args->sampling != nullptr && args->sampling->temperature > 0)
        return fail(ex, FLERP_E_UNSUPPORTED, "strixite draws on the host: a sampled seq takes FLERP_OUT_FULL or "
                                             "FLERP_OUT_CANDIDATES rows and its own sampler, not FLERP_OUT_TOKENS");
    for (size_t i = 0; i < ex->seqs.size(); ++i)
        if (!ex->seqs[i].open)
            return guarded(ex, [&] {
                ex->b->reset((int)i);
                ex->seqs[i] = Seq{};
                ex->seqs[i].open = true;
                *seq = (int32_t)i;
                return FLERP_OK;
            });
    return fail(ex, FLERP_E_FULL, "every seq is open (" + std::to_string(ex->seqs.size()) + ")");
}

flerp_status flerp_seq_free(flerp_executor *ex, int32_t seq) {
    if (seq < 0 || seq >= (int32_t)ex->seqs.size() || !ex->seqs[(size_t)seq].open)
        return fail(ex, FLERP_E_ARGUMENT, "seq " + std::to_string(seq) + " is not open");
    return guarded(ex, [&] {
        if (ex->seqs[(size_t)seq].pending) ex->b->drop_verify(seq);
        ex->b->reset(seq);
        ex->seqs[(size_t)seq] = Seq{};
        return FLERP_OK;
    });
}

flerp_status flerp_seq_reset(flerp_executor *ex, int32_t seq) {
    if (flerp_status st = seq_ok(ex, seq, false)) return st;
    return guarded(ex, [&] {
        ex->b->reset(seq);
        std::fill(std::begin(ex->seqs[(size_t)seq].snap), std::end(ex->seqs[(size_t)seq].snap), false);
        return FLERP_OK;
    });
}

int64_t flerp_seq_pos(const flerp_executor *ex, int32_t seq) {
    if (seq < 0 || seq >= (int32_t)ex->seqs.size() || !ex->seqs[(size_t)seq].open) return FLERP_E_ARGUMENT;
    return ex->b->pos(seq);
}

flerp_status flerp_forward(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n, const flerp_want *want,
                           flerp_rows *out, const flerp_token *lookahead, size_t n_lookahead) {
    if (flerp_status st = seq_ok(ex, seq, false)) return st;
    if (ids == nullptr || n == 0) return fail(ex, FLERP_E_ARGUMENT, "flerp_forward: no ids");
    const int64_t rows = want ? want->rows : 0;
    if (rows < 0 || rows > (int64_t)n) return fail(ex, FLERP_E_ARGUMENT, "flerp_forward: rows outside 0..n");
    if ((int64_t)n > ex->b->max_chunk()) return fail(ex, FLERP_E_FULL, "flerp_forward: more ids than caps.max_forward");
    return guarded(ex, [&] {
        if (lookahead != nullptr && n_lookahead > 0)
            ex->b->set_lookahead(std::vector<int32_t>(lookahead, lookahead + n_lookahead));
        const std::vector<int32_t> v(ids, ids + n);
        const std::unique_ptr<LogitMasks> masks = masks_of(want ? want->masks : nullptr);
        const bool cands = rows > 0 && want->delivery != FLERP_OUT_FULL;
        const int64_t n_valid = want && want->n_valid > 0 ? want->n_valid : ex->b->logits_row();
        const LogitRows l = ex->b->forward_rows(seq, v, rows, cands, n_valid, masks.get());
        ++ex->stats.forwards, ex->stats.forward_tokens += (int64_t)n;
        if (rows == 0) return FLERP_OK;
        return deliver(ex, l, want->delivery, n_valid, out);
    });
}

flerp_status flerp_draft_step(flerp_executor *ex, int32_t seq, flerp_token token, int32_t step, flerp_draft *out) {
    if (flerp_status st = seq_ok(ex, seq, false)) return st;
    if (!ex->b->has_mtp()) return fail(ex, FLERP_E_UNSUPPORTED, "this executor was opened without its MTP head");
    if (out == nullptr) return fail(ex, FLERP_E_ARGUMENT, "flerp_draft_step: null out");
    return guarded(ex, [&] {
        const Top2 t = ex->b->forward_mtp_top2(seq, token, step);
        if (t.nan) return fail(ex, FLERP_E_FAILED, "NaN in the MTP draft logits");
        out->token = t.best, out->best = t.best_v, out->second = t.second_v, out->probability = NAN;
        ++ex->stats.draft_tokens;
        return FLERP_OK;
    });
}

flerp_status flerp_draft_hold(flerp_executor *ex, int32_t, flerp_token, int32_t) {
    return fail(ex, FLERP_E_UNSUPPORTED, "strixite's draft head hands its drafts to the caller: flerp_draft_step");
}

flerp_status flerp_verify(flerp_executor *ex, flerp_verify_run *runs, size_t n_runs, const flerp_want *want) {
    if (runs == nullptr || n_runs == 0 || want == nullptr) return fail(ex, FLERP_E_ARGUMENT, "flerp_verify: no runs or no want");
    int64_t total = 0;
    for (size_t i = 0; i < n_runs; ++i) {
        if (flerp_status st = seq_ok(ex, runs[i].seq, false)) return st;
        if (runs[i].held != 0) return fail(ex, FLERP_E_UNSUPPORTED, "strixite holds no drafts on the device (run.held)");
        if (runs[i].ids == nullptr || runs[i].n_ids == 0) return fail(ex, FLERP_E_ARGUMENT, "flerp_verify: a run without ids");
        for (size_t j = 0; j < i; ++j)
            if (runs[j].seq == runs[i].seq) return fail(ex, FLERP_E_ARGUMENT, "flerp_verify: two runs of one seq");
        total += (int64_t)runs[i].n_ids;
    }
    if (total > ex->b->max_verify_rows()) return fail(ex, FLERP_E_FULL, "flerp_verify: more rows than caps.max_verify_rows");
    return guarded(ex, [&] {
        std::vector<LmBackend::VerifyRun> vr(n_runs);
        std::vector<std::unique_ptr<LogitMasks>> masks(n_runs);
        for (size_t i = 0; i < n_runs; ++i) {
            vr[i].seq = runs[i].seq;
            vr[i].ids.assign(runs[i].ids, runs[i].ids + runs[i].n_ids);
            masks[i] = masks_of(runs[i].masks);
            vr[i].masks = masks[i].get();
        }
        const int64_t n_valid = want->n_valid > 0 ? want->n_valid : ex->b->logits_row();
        const bool cands = want->delivery != FLERP_OUT_FULL;
        std::vector<LogitRows> ls;
        if (ex->batch_verify) {
            ls = ex->b->verify_batch(vr, cands, n_valid);
        } else {  // one seq: the calls Engine::run makes for a one-slot server
            ls.push_back(ex->b->forward_verify_rows(vr[0].ids, (int64_t)vr[0].ids.size(), cands, n_valid, vr[0].masks));
            ex->seqs[(size_t)runs[0].seq].verify_ids = vr[0].ids;
        }
        ++ex->stats.verifies, ex->stats.verify_rows += total;
        for (size_t i = 0; i < n_runs; ++i) {  // pending from here, whatever the delivery below says
            ex->seqs[(size_t)runs[i].seq].pending = true;
            ex->seqs[(size_t)runs[i].seq].verify_rows = (int64_t)runs[i].n_ids;
        }
        for (size_t i = 0; i < n_runs; ++i)
            if (flerp_status st = deliver(ex, ls[i], want->delivery, n_valid, runs[i].out)) return st;
        return FLERP_OK;
    });
}

static flerp_status resolve(flerp_executor *ex, int32_t seq, int64_t rows, flerp_token carry) {
    if (flerp_status st = seq_ok(ex, seq, true)) return st;
    Seq &s = ex->seqs[(size_t)seq];
    if (rows < 0 || rows > s.verify_rows) return fail(ex, FLERP_E_ARGUMENT, "keep: rows outside the pending verify");
    return guarded(ex, [&] {
        if (ex->batch_verify) {
            if (rows == 0) ex->b->drop_verify(seq);
            else ex->b->keep_verify(seq, rows, carry);
        } else if (rows == 0) {
            ex->b->drop_verify();
        } else if (rows == s.verify_rows) {
            ex->b->keep_verify();
        } else {
            ex->b->keep_verify_prefix(s.verify_ids, rows);
        }
        ex->stats.kept_rows += rows, ex->stats.dropped_rows += s.verify_rows - rows;
        s.pending = false, s.verify_rows = 0;
        return FLERP_OK;
    });
}

flerp_status flerp_keep(flerp_executor *ex, int32_t seq) {
    if (flerp_status st = seq_ok(ex, seq, true)) return st;
    return resolve(ex, seq, ex->seqs[(size_t)seq].verify_rows, -1);
}

flerp_status flerp_keep_prefix(flerp_executor *ex, int32_t seq, int64_t rows, flerp_token carry) {
    if (flerp_status st = seq_ok(ex, seq, true)) return st;
    if (rows < 1 || rows >= ex->seqs[(size_t)seq].verify_rows)
        return fail(ex, FLERP_E_ARGUMENT, "flerp_keep_prefix: rows must be 1 .. the run's rows - 1");
    return resolve(ex, seq, rows, carry);
}

flerp_status flerp_drop(flerp_executor *ex, int32_t seq) { return resolve(ex, seq, 0, -1); }

static int slot_of(const flerp_executor *ex, int32_t seq, flerp_snapshot snap) {
    if (snap < 1 || snap > 3 || !ex->seqs[(size_t)seq].snap[snap - 1]) return -1;
    return kSlots[snap - 1];
}

flerp_status flerp_snapshot_save(flerp_executor *ex, int32_t seq, flerp_snapshot *out) {
    if (flerp_status st = seq_ok(ex, seq, false)) return st;
    if (out == nullptr) return fail(ex, FLERP_E_ARGUMENT, "flerp_snapshot_save: null out");
    Seq &s = ex->seqs[(size_t)seq];
    for (int i = 0; i < 3; ++i)
        if (!s.snap[i])
            return guarded(ex, [&] {
                ex->b->save_snapshot(seq, kSlots[i]);
                s.snap[i] = true;
                *out = (flerp_snapshot)(i + 1);
                ++ex->stats.snapshots_saved;
                return FLERP_OK;
            });
    return fail(ex, FLERP_E_FULL, "the seq holds its 3 snapshots: drop one");
}

flerp_status flerp_snapshot_restore(flerp_executor *ex, int32_t seq, flerp_snapshot snap) {
    if (flerp_status st = seq_ok(ex, seq, false)) return st;
    const int slot = slot_of(ex, seq, snap);
    if (slot < 0) return fail(ex, FLERP_E_ARGUMENT, "not a snapshot of this seq");
    return guarded(ex, [&] {
        if (!ex->b->can_restore_snapshot(seq, slot)) return fail(ex, FLERP_E_STATE, "the snapshot is no longer valid for the seq's state");
        ex->b->restore_snapshot(seq, slot);
        ++ex->stats.snapshots_restored;
        return FLERP_OK;
    });
}

flerp_status flerp_snapshot_drop(flerp_executor *ex, int32_t seq, flerp_snapshot snap) {
    if (flerp_status st = seq_ok(ex, seq, false)) return st;
    if (slot_of(ex, seq, snap) < 0) return fail(ex, FLERP_E_ARGUMENT, "not a snapshot of this seq");
    ex->seqs[(size_t)seq].snap[snap - 1] = false;  // the slot is free for the next save, which overwrites it
    return FLERP_OK;
}

int64_t flerp_snapshot_pos(const flerp_executor *ex, int32_t seq, flerp_snapshot snap) {
    if (seq < 0 || seq >= (int32_t)ex->seqs.size() || !ex->seqs[(size_t)seq].open) return FLERP_E_ARGUMENT;
    const int slot = slot_of(ex, seq, snap);
    return slot < 0 ? (int64_t)FLERP_E_ARGUMENT : ex->b->snapshot_pos(seq, slot);
}

int64_t flerp_snapshot_bytes(const flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from) {
    if (seq < 0 || seq >= (int32_t)ex->seqs.size() || !ex->seqs[(size_t)seq].open) return FLERP_E_ARGUMENT;
    const int slot = slot_of(ex, seq, snap);
    if (slot < 0) return FLERP_E_ARGUMENT;
    try {
        return (int64_t)ex->b->snapshot_bytes(seq, slot, from);
    } catch (const std::exception &e) {
        return fail(ex, FLERP_E_FAILED, e.what());
    }
}

flerp_status flerp_snapshot_export(flerp_executor *ex, int32_t seq, flerp_snapshot snap, int64_t from, void *bytes,
                                   size_t cap, size_t *written) {
    if (flerp_status st = seq_ok(ex, seq, false)) return st;
    const int slot = slot_of(ex, seq, snap);
    if (slot < 0 || bytes == nullptr) return fail(ex, FLERP_E_ARGUMENT, "not a snapshot of this seq, or no buffer");
    return guarded(ex, [&] {
        if (ex->b->snapshot_bytes(seq, slot, from) > cap) return fail(ex, FLERP_E_ARGUMENT, "the buffer is smaller than flerp_snapshot_bytes");
        HostBuffer hb;  // LmBackend exports into its own page-aligned buffer; one copy out to the caller's
        ex->b->export_snapshot(seq, slot, hb, from);
        std::memcpy(bytes, hb.data(), hb.size());
        if (written != nullptr) *written = hb.size();
        return FLERP_OK;
    });
}

flerp_status flerp_state_import(flerp_executor *ex, int32_t seq, const void *bytes, size_t n_bytes, int64_t n, int64_t from) {
    if (flerp_status st = seq_ok(ex, seq, false)) return st;
    if (bytes == nullptr) return fail(ex, FLERP_E_ARGUMENT, "flerp_state_import: no bytes");
    return guarded(ex, [&] {
        HostBuffer hb(n_bytes);
        std::memcpy(hb.data(), bytes, n_bytes);
        ex->b->import_state(seq, hb, n, from);
        std::fill(std::begin(ex->seqs[(size_t)seq].snap), std::end(ex->seqs[(size_t)seq].snap), false);
        return FLERP_OK;
    });
}

flerp_status flerp_snapshot_write(flerp_executor *ex, int32_t, flerp_snapshot, const char *, uint64_t) {
    return fail(ex, FLERP_E_UNSUPPORTED, "strixite states leave as host bytes (flerp_snapshot_export); files are its prompt cache's");
}

flerp_status flerp_snapshot_read(flerp_executor *ex, int32_t, const char *, uint64_t, int64_t, flerp_snapshot *) {
    return fail(ex, FLERP_E_UNSUPPORTED, "strixite states arrive as host bytes (flerp_state_import)");
}

flerp_status flerp_get_stats(const flerp_executor *ex, flerp_stats *out) {
    if (out == nullptr || out->size < sizeof(flerp_stats)) return fail(ex, FLERP_E_ARGUMENT, "flerp_stats is null or short");
    const uint32_t size = out->size;
    *out = ex->stats;
    out->size = size, out->device_bytes = -1;
    return FLERP_OK;
}

size_t flerp_stats_json(const flerp_executor *ex, char *out, size_t cap) {
    const BackendStats s = ex->b->backend_stats();
    char buf[512];
    const int n = std::snprintf(buf, sizeof(buf),
                                "{\"ngram_rows_requested\": %lld, \"ngram_rows_read\": %lld, \"ple_gathers\": %lld, "
                                "\"ple_waits\": %lld, \"vision_images\": %lld, \"kv_renewals\": %lld}",
                                (long long)s.ngram_rows_requested, (long long)s.ngram_rows_read, (long long)s.ple_gathers,
                                (long long)s.ple_waits, (long long)s.vision_images, (long long)s.kv_renewals);
    if (out != nullptr && cap > 0) {
        const size_t m = std::min((size_t)n, cap - 1);
        std::memcpy(out, buf, m);
        out[m] = 0;
    }
    return (size_t)n;
}

}  // extern "C"
