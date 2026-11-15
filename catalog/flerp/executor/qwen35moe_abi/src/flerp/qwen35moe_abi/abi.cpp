// SPDX-License-Identifier: MIT
// include/flerp/executor.h over executor.flerp.qwen35moe run as N stages (docs/PHASE5-FOURNODE-DESIGN.md section
// 3.9): rank 0's Pipeline (stages.hpp) behind the ABI. World 1 is the same code with no links. First version:
//   one seq;
//   FLERP_OUT_TOKENS (greedy: the last stage's argmax of the row, the lower id first on a tie) and FLERP_OUT_FULL
//   (the logits row of the forward's last position); one row a forward (no FLERP_CAP_FORWARD_ROWS);
//   max_forward = the arena's prompt batch (QwenGpuExecutor::GetMaxPromptBatch), so a prompt fed in forwards of
//   max_forward is cut where the origin's ForwardPromptBatch / stage P2 cut it;
//   no snapshots, no masks, no sampled draws: each is FLERP_E_UNSUPPORTED with its reason, and the caps say so (never
//   a silent fallback).
// Drafts (ORNITH-O3 step two, gate/ornith-o3-spec.toml), when flerp_open has drafts=N (TCP, the last stage process
//   started with draft_head=on): caps.max_draft N, FLERP_CAP_DRAFT_HELD | FLERP_CAP_KEEP_PREFIX. A seq opened with
//   args->drafts > 0 is a drafting session: the last stage's MTP head drafts after every reply and holds the drafts
//   (flerp_draft_hold names how many the next verify runs); flerp_verify runs the pending id and those drafts as
//   back-to-back single-row decode steps (the decode step's own graph, so each row is exactly a decode step's) and
//   returns every row's greedy id. The keep is the last stage's own greedy chain: flerp_keep / flerp_keep_prefix must
//   keep exactly the rows whose drafts equal the id drawn from the row before, plus that row (what a greedy caller
//   keeps); anything else is FLERP_E_STATE. A drafting seq is greedy: after its prompt, a forward is one id, the id
//   the last reply drew. No flerp_drop (the round's recurrent ring has no slot for it).
//   With draft_value=on (rank 0) the last stage's value-aware rule holds at most N, fewer when a draft is not worth
//   its row: flerp_draft_hold of more than it holds holds what it has, and flerp_verify of run.held >= that runs
//   those (out->rows = 1 + held; the decode loop takes the row count, catalog/flerp/runtime/decode_loop). Without
//   draft_value a hold of more than the stage holds is FLERP_E_STATE, as before. (card 3f6d20b703)
// Forward shape. The executor has two paths and they are not the same kernels: a prompt chunk
// (ForwardPromptBatch) and a decode step (ForwardToken, captured as a graph). A forward is a decode step when it
// carries one id and the seq has already returned a row (its prompt was forwarded); every other forward is a prompt
// chunk. So the forwards a caller makes with flerp_feed_ids and then one id a forward are the generator's: prompt
// chunks of the arena's batch, then a decode step a token (tool.flerp.qwen35moe_generate, stage P1 and P2).
// flerp_open keys: stages.hpp (weights, context, world, rank = 0, peers, placement, epoch, deadline_ms,
// setup_deadline_ms, graph, single_reader, pause_ms).
#include "flerp/executor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "flerp/qwen35moe_abi/stages.hpp"

namespace qabi = flerp::qwen35moe_abi;

namespace {
std::string g_open_error;
}  // namespace

struct flerp_executor {
    qabi::Options opt;
    std::unique_ptr<qabi::Pipeline> p;
    bool open = false;      // the one seq
    bool prompted = false;  // the seq has returned a row: a one-id forward from here on is a decode step
    int64_t pos = 0;
    // a drafting seq (args->drafts > 0)
    uint32_t drafts = 0;
    int32_t pending = -1;   // the id the last reply drew, not yet fed
    int32_t hold = 0;       // drafts named by flerp_draft_hold for the next verify
    bool verifying = false; // a verify is pending keep / keep_prefix
    bool left = false;      // a keep left the last stage's greedy chain (an end id kept short, a cut): reset next
    qabi::Round round;
    // Slot-resident prefix reuse (ORNITH-PCACHE, key reuse_tail > 0; gate/ornith-pcache.toml). The stages keep one snapshot of
    // their KV and recurrent state, taken at the start of the last prompt forward when at least reuse_tail ids follow (so the
    // chat template's generation prompt, which the next request's rendering of the same thread does not repeat, stays out of
    // it; prompt forwards are cut at the reuse grid when reuse_grid > 0). A seq whose
    // first ids equal the snapshot's ids leaves the stages alone (those forwards are skipped), restores the snapshot at its
    // end and forwards only what follows; any other prompt resets the stages and forwards everything, as before.
    std::vector<int32_t> hist;          // the prompt's ids at positions 0.. of this seq (prompt forwards only)
    std::vector<std::pair<int64_t, size_t>> skipped;   // (start, n) of the forwards skipped while the match holds
    bool deferred = false;              // seq_new did not reset the stages: the ids so far equal the snapshot's and were skipped
    int64_t snap_K = 0;                 // positions the snapshot covers (0: none)
    std::vector<int32_t> snap_ids;      // its ids
    uint32_t snap_drafts = 0;           // the draft depth of the session that saved it (the draft head's edge exists only then)
    int64_t reuse_hits = 0, reuse_misses = 0, reuse_tokens = 0, reuse_saves = 0;
    mutable std::string error;
    flerp_stats stats{};
};

namespace {

flerp_status fail(const flerp_executor *ex, flerp_status st, const std::string &what) {
    ex->error = what;
    return st;
}

flerp_status unsupported(const flerp_executor *ex, const char *what) {
    return fail(ex, FLERP_E_UNSUPPORTED, std::string("executor.flerp.qwen35moe_abi: ") + what);
}

flerp_status seq_ok(const flerp_executor *ex, int32_t seq) {
    if (seq != 0 || !ex->open) return fail(ex, FLERP_E_ARGUMENT, "seq " + std::to_string(seq) + " is not open (this executor holds one seq, 0)");
    if (ex->p->failed()) return fail(ex, FLERP_E_FAILED, "the stages failed: " + ex->p->reason());
    return FLERP_OK;
}

flerp_status reset_all(flerp_executor *ex) {
    try {
        ex->p->Reset(ex->drafts);
    } catch (const std::exception &e) {
        return fail(ex, FLERP_E_FAILED, std::string("reset: ") + e.what());
    }
    ex->pos = 0, ex->prompted = false;
    ex->pending = -1, ex->hold = 0, ex->verifying = false, ex->left = false;
    ex->deferred = false, ex->hist.clear(), ex->skipped.clear();
    ex->snap_K = 0;   // a prompt forwarded from position 0 overwrites the KV the snapshot stands on
    return FLERP_OK;
}

// A new seq. With a snapshot of the same session kind the stages are left as they are (no reset): the prompt's forwards
// are matched against the snapshot's ids first (prompt_with_reuse). Otherwise the stages reset now.
flerp_status begin_session(flerp_executor *ex) {
    if (ex->opt.reuse_tail > 0 && ex->snap_K > 0 && ex->snap_drafts == ex->drafts) {
        ex->pos = 0, ex->prompted = false;
        ex->pending = -1, ex->hold = 0, ex->verifying = false, ex->left = false;
        ex->deferred = true, ex->hist.clear(), ex->skipped.clear();
        return FLERP_OK;
    }
    return reset_all(ex);
}

// The match failed (or the final forward must run): the stages reset and the forwards skipped so far run, in the chunks
// they came in, so the prompt reaches the stages as a cold prompt does.
void replay_skipped(flerp_executor *ex, const char *why) {
    ex->deferred = false;
    ex->snap_K = 0;
    ex->p->Reset(ex->drafts);
    for (const auto &[start, n] : ex->skipped)
        (void)ex->p->Forward(false, std::span<const int32_t>(ex->hist.data() + start, n), (uint32_t)start, false, false);
    ex->skipped.clear();
    ++ex->reuse_misses;
    qabi::Emit(std::string("{\"event\":\"prefix_miss\",\"why\":\"") + why + "\",\"replayed_to\":" + std::to_string(ex->hist.size()) + "}");
}

// One prompt piece ids[0, n) at position q (a forward, or part of one cut at the reuse grid), reuse on. true: the stages ran
// it and *r is their reply; false: skipped. ex->hist already holds the ids at their positions.
bool prompt_piece(flerp_executor *ex, const int32_t *ids, size_t n, int64_t q, bool want, bool full, qabi::Reply *r) {
    const int64_t P = q + (int64_t)n;
    size_t off = 0;   // ids[0, off) matched the snapshot and are not forwarded
    if (ex->deferred) {
        const int64_t K = ex->snap_K;
        const int64_t cmp = std::min<int64_t>((int64_t)n, K - q);
        const bool equal = cmp <= 0 || std::equal(ids, ids + cmp, ex->snap_ids.begin() + q);
        if (!equal) {
            replay_skipped(ex, "ids differ from the kept prefix");
        } else if (P <= K) {
            if (want) {
                replay_skipped(ex, "the prompt ends inside the kept prefix");   // the final forward's row needs the stages at the prompt's end
            } else {
                ex->skipped.emplace_back(q, n);
                return false;
            }
        } else {   // the prompt runs past the kept prefix: put the snapshot back, forward from its end
            ex->p->Reset(ex->drafts, (uint32_t)K);
            ex->deferred = false;
            ex->skipped.clear();
            ++ex->reuse_hits, ex->reuse_tokens += K;
            off = (size_t)(K - q);
            qabi::Emit("{\"event\":\"prefix_reuse\",\"kept\":" + std::to_string(K) + "}");
        }
    }
    const int64_t start = q + (int64_t)off;
    if (want) {
        // The final piece: the stages stand at its start, a piece boundary every cold prefill of this prompt has too (same
        // chunks, so a warm run that restores here forwards the very chunks a cold run does and its ids are the cold run's).
        // Keep the state here when at least reuse_tail ids follow (the chat template's generation prompt must stay out of the
        // kept prefix: the next request's rendering of the thread does not repeat it); otherwise the older snapshot, if any, stays.
        if (start > 0 && P - start >= (int64_t)ex->opt.reuse_tail && start != ex->snap_K) {
            ex->p->SnapSave((uint32_t)start);
            ex->snap_K = start, ex->snap_drafts = ex->drafts;
            ex->snap_ids.assign(ex->hist.begin(), ex->hist.begin() + start);
            ++ex->reuse_saves;
        }
    }
    *r = ex->p->Forward(false, std::span<const int32_t>(ids + off, n - off), (uint32_t)start, want, full);
    return true;
}

// A prompt forward of ids[0, n) at ex->pos, cut at multiples of the reuse grid (key reuse_grid; 0: as the caller made it).
bool prompt_with_reuse(flerp_executor *ex, const int32_t *ids, size_t n, bool want, bool full, qabi::Reply *r) {
    const int64_t q0 = ex->pos;
    const int64_t grid = ex->opt.reuse_grid;
    bool ran = false;
    size_t i = 0;
    while (i < n) {
        const int64_t q = q0 + (int64_t)i;
        size_t m = n - i;
        if (grid > 0) m = std::min<size_t>(m, (size_t)(grid - q % grid));
        ran |= prompt_piece(ex, ids + i, m, q, want && i + m == n, full, r);
        i += m;
    }
    return ran;
}

}  // namespace

extern "C" {

flerp_status flerp_open(const flerp_open_args *args, flerp_executor **out) {
    g_open_error.clear();
    if (args == nullptr || out == nullptr || args->size < sizeof(flerp_open_args)) {
        g_open_error = "flerp_open: null or short arguments";
        return FLERP_E_ARGUMENT;
    }
    auto ex = std::make_unique<flerp_executor>();
    for (size_t i = 0; i < args->n_config; ++i) {
        const std::string key = args->config[i].key ? args->config[i].key : "";
        const std::string value = args->config[i].value ? args->config[i].value : "";
        std::string error;
        if (!qabi::SetKey(ex->opt, key, value, &error)) {
            g_open_error = "flerp_open (qwen35moe_abi): " + error;
            return FLERP_E_CONFIG;
        }
    }
    if (const std::string e = qabi::CheckOptions(ex->opt); !e.empty()) {
        g_open_error = "flerp_open (qwen35moe_abi): " + e;
        return FLERP_E_CONFIG;
    }
    if (ex->opt.rank != 0) {
        g_open_error = "flerp_open (qwen35moe_abi): an executor is rank 0; ranks 1.." + std::to_string(ex->opt.world - 1) +
                       " run flerp-qwen35moe-stage";
        return FLERP_E_CONFIG;
    }
    try {
        ex->p = std::make_unique<qabi::Pipeline>(ex->opt);
    } catch (const std::invalid_argument &e) {
        g_open_error = std::string("flerp_open (qwen35moe_abi): ") + e.what();
        return FLERP_E_CONFIG;
    } catch (const std::exception &e) {
        g_open_error = std::string("flerp_open (qwen35moe_abi): ") + e.what();
        return FLERP_E_FAILED;
    }
    ex->stats.size = sizeof(flerp_stats);
    *out = ex.release();
    return FLERP_OK;
}

void flerp_close(flerp_executor *ex) {
    if (ex == nullptr) return;
    if (ex->p) ex->p->Close();  // CLOSE to every stage, which then exits; then the stages here, then the load
    delete ex;
}

flerp_status flerp_get_caps(const flerp_executor *ex, flerp_caps *out) {
    // a version 1 caller's struct ends at `flags`: it is filled as far as it goes and never written past
    if (out == nullptr || out->size < FLERP_HAS(flerp_caps, flags)) return fail(ex, FLERP_E_ARGUMENT, "flerp_caps is null or short");
    flerp_caps c{};
    c.size = (uint32_t)std::min<size_t>(out->size, FLERP_CAPS_V2_SIZE), c.abi = 2;  // a version 2 executor
    c.positions = ex->p->context();
    c.max_forward = (int64_t)ex->p->max_rows();
    c.max_verify_rows = 1 + (int64_t)ex->opt.drafts;  // a verify: the pending id and the held drafts
    c.logits_row = ex->p->vocab();
    c.candidates = 0;
    c.seqs = 1;
    c.snapshots = 0;
    c.max_draft = (int32_t)ex->opt.drafts;
    c.deliveries = FLERP_OUT_FULL | FLERP_OUT_TOKENS;
    // no vision, masks, verify batches, snapshots, sampled draws, rows beyond the last; drafts held by the last stage
    c.flags = ex->opt.drafts > 0 ? (FLERP_CAP_DRAFT_HELD | FLERP_CAP_KEEP_PREFIX) : 0;
    c.sampling = 0, c.top_k_max = 0;
    std::memcpy(out, &c, c.size);
    return FLERP_OK;
}

const char *flerp_describe(const flerp_executor *ex) { return ex->p->describe().c_str(); }
const char *flerp_state_fingerprint(const flerp_executor *ex) { return ex->p->fingerprint().c_str(); }
const char *flerp_last_error(const flerp_executor *ex) { return ex ? ex->error.c_str() : g_open_error.c_str(); }
int flerp_failed(const flerp_executor *ex) { return ex->p->failed() ? 1 : 0; }

flerp_status flerp_seq_new(flerp_executor *ex, const flerp_seq_args *args, int32_t *seq) {
    if (seq == nullptr) return fail(ex, FLERP_E_ARGUMENT, "flerp_seq_new: null seq");
    if (args != nullptr && args->sampling != nullptr && args->sampling->temperature > 0)
        return unsupported(ex, "tokens are greedy only (no FLERP_CAP_SAMPLED_TOKENS): sample on the host from FLERP_OUT_FULL rows");
    if (ex->open) return fail(ex, FLERP_E_FULL, "the executor's one seq is open");
    if (ex->p->failed()) return fail(ex, FLERP_E_FAILED, "the stages failed: " + ex->p->reason());
    const int32_t want = args != nullptr && args->size >= FLERP_SEQ_ARGS_V2_SIZE ? args->drafts : 0;
    if (want < 0) return fail(ex, FLERP_E_ARGUMENT, "flerp_seq_new: drafts < 0");
    if (want > 0 && ex->opt.drafts == 0) return unsupported(ex, "no draft head (flerp_open drafts=N with a last stage started draft_head=on; caps.max_draft 0)");
    ex->drafts = (uint32_t)std::min<int64_t>(want, ex->opt.drafts);
    if (const flerp_status st = begin_session(ex)) return st;
    ex->open = true;
    *seq = 0;
    return FLERP_OK;
}

flerp_status flerp_seq_free(flerp_executor *ex, int32_t seq) {
    if (seq != 0 || !ex->open) return fail(ex, FLERP_E_ARGUMENT, "seq " + std::to_string(seq) + " is not open");
    ex->open = false;  // the stages are reset by the next flerp_seq_new
    return FLERP_OK;
}

flerp_status flerp_seq_reset(flerp_executor *ex, int32_t seq) {
    if (const flerp_status st = seq_ok(ex, seq)) return st;
    return reset_all(ex);
}

int64_t flerp_seq_pos(const flerp_executor *ex, int32_t seq) {
    if (seq != 0 || !ex->open) return FLERP_E_ARGUMENT;
    return ex->pos;
}

flerp_status flerp_forward(flerp_executor *ex, int32_t seq, const flerp_token *ids, size_t n, const flerp_want *want,
                           flerp_rows *out, const flerp_token *, size_t) {
    if (const flerp_status st = seq_ok(ex, seq)) return st;
    if (ids == nullptr || n == 0) return fail(ex, FLERP_E_ARGUMENT, "flerp_forward: no ids");
    const int64_t rows = want ? want->rows : 0;
    if (rows < 0 || rows > (int64_t)n) return fail(ex, FLERP_E_ARGUMENT, "flerp_forward: rows outside 0..n");
    if (rows > 1) return unsupported(ex, "a forward returns its last row only (no FLERP_CAP_FORWARD_ROWS)");
    if (want != nullptr && want->masks != nullptr) return unsupported(ex, "no masks or logit bias (no FLERP_CAP_MASKS)");
    if ((int64_t)n > (int64_t)ex->p->max_rows())
        return fail(ex, FLERP_E_FULL, "flerp_forward: " + std::to_string(n) + " ids, caps.max_forward is " + std::to_string(ex->p->max_rows()));
    if (ex->pos + (int64_t)n > (int64_t)ex->p->context())
        return fail(ex, FLERP_E_FULL, "flerp_forward: the seq holds " + std::to_string(ex->pos) + " of " + std::to_string(ex->p->context()) + " positions");
    for (size_t i = 0; i < n; ++i)
        if (ids[i] < 0 || (uint32_t)ids[i] >= ex->p->vocab()) return fail(ex, FLERP_E_ARGUMENT, "flerp_forward: id " + std::to_string(ids[i]) + " is outside the vocabulary");
    const uint32_t vocab = ex->p->vocab();
    int64_t n_valid = 0;
    uint32_t delivery = FLERP_OUT_NONE;
    if (rows == 1) {
        delivery = want->delivery;
        if (delivery != FLERP_OUT_FULL && delivery != FLERP_OUT_TOKENS) return unsupported(ex, "the deliveries are FLERP_OUT_FULL and FLERP_OUT_TOKENS");
        if (out == nullptr) return fail(ex, FLERP_E_ARGUMENT, "results were asked for without a flerp_rows");
        if (delivery == FLERP_OUT_FULL && (out->full == nullptr || out->full_cap < vocab)) return fail(ex, FLERP_E_ARGUMENT, "flerp_rows.full is too small");
        if (delivery == FLERP_OUT_TOKENS && (out->tokens == nullptr || out->tokens_cap < 1)) return fail(ex, FLERP_E_ARGUMENT, "flerp_rows.tokens is too small");
        if (out->nan != nullptr && out->nan_cap < 1) return fail(ex, FLERP_E_ARGUMENT, "flerp_rows.nan is too small");
        n_valid = want->n_valid > 0 && want->n_valid < (int64_t)vocab ? want->n_valid : 0;
    }
    // tokens from a part of the vocabulary: the argmax over it on the host, from the full row
    const bool full = rows == 1 && (delivery == FLERP_OUT_FULL || n_valid > 0);
    const bool decode = n == 1 && ex->prompted;
    if (ex->drafts > 0) {
        if (ex->verifying) return fail(ex, FLERP_E_STATE, "a verify is pending keep / keep_prefix");
        if (ex->left) return fail(ex, FLERP_E_STATE, "the seq left the last stage's greedy chain at its last keep: reset it");
        if (full) return unsupported(ex, "a drafting seq returns tokens (FLERP_OUT_TOKENS over the whole vocabulary)");
        if (ex->prompted && (n != 1 || ids[0] != ex->pending))
            return unsupported(ex, "a drafting seq takes, after its prompt, one id a forward: the id its last reply drew");
    }
    qabi::Reply r;
    bool ran = true;
    try {
        if (!decode && !ex->prompted && ex->opt.reuse_tail > 0) {
            ex->hist.insert(ex->hist.end(), ids, ids + n);
            ran = prompt_with_reuse(ex, ids, n, rows == 1, full, &r);
        } else {
            r = ex->p->Forward(decode, std::span<const int32_t>(ids, n), (uint32_t)ex->pos, rows == 1, full);
        }
    } catch (const std::exception &e) {
        ex->snap_K = 0, ex->deferred = false;
        return fail(ex, FLERP_E_FAILED, std::string("forward: ") + e.what());
    }
    ex->pos += (int64_t)n;
    if (!ran) return FLERP_OK;   // a prompt forward the kept prefix covers: no stage ran it, and it asked for no row
    ++ex->stats.forwards, ex->stats.forward_tokens += (int64_t)n;
    if (ex->drafts > 0 && rows == 1) ex->pending = r.token, ex->hold = 0, ex->stats.draft_tokens += (int64_t)ex->p->held().size();
    if (rows == 0) return FLERP_OK;
    ex->prompted = true;
    ex->hist.clear();
    out->rows = 0;
    bool nan = false;
    if (full) {
        if (r.logits.size() != vocab) return fail(ex, FLERP_E_FAILED, "the last stage returned no logits row");
        for (const float x : r.logits) nan = nan || std::isnan(x);
    }
    if (delivery == FLERP_OUT_FULL) {
        std::memcpy(out->full, r.logits.data(), (size_t)vocab * sizeof(float));
    } else if (n_valid > 0) {
        int64_t best = 0;  // the lower id first on a tie, as the device argmax
        for (int64_t i = 1; i < n_valid; ++i)
            if (r.logits[(size_t)i] > r.logits[(size_t)best]) best = i;
        out->tokens[0] = (flerp_token)best;
    } else {
        out->tokens[0] = r.token;
    }
    if (out->nan != nullptr) out->nan[0] = nan ? 1 : 0;
    out->rows = 1;
    return FLERP_OK;
}

flerp_status flerp_draft_step(flerp_executor *ex, int32_t, flerp_token, int32_t, flerp_draft *) {
    return unsupported(ex, "drafts are held by the last stage (FLERP_CAP_DRAFT_HELD), not stepped");
}
flerp_status flerp_draft_hold(flerp_executor *ex, int32_t seq, flerp_token pending, int32_t depth) {
    if (const flerp_status st = seq_ok(ex, seq)) return st;
    if (ex->drafts == 0) return unsupported(ex, "the seq was not opened with drafts (or caps.max_draft is 0)");
    if (ex->verifying) return fail(ex, FLERP_E_STATE, "a verify is pending keep / keep_prefix");
    if (ex->left) return fail(ex, FLERP_E_STATE, "the seq left the last stage's greedy chain at its last keep: reset it");
    if (!ex->prompted || pending != ex->pending) return fail(ex, FLERP_E_STATE, "flerp_draft_hold: pending is not the id the seq's last reply drew");
    if (depth < 0) return fail(ex, FLERP_E_ARGUMENT, "flerp_draft_hold: depth < 0");
    if ((size_t)depth > ex->p->held().size() && ex->opt.draft_value) depth = (int32_t)ex->p->held().size();
    if ((size_t)depth > ex->p->held().size())
        return fail(ex, FLERP_E_STATE, "flerp_draft_hold: " + std::to_string(depth) + " drafts asked, the last stage holds " +
                                           std::to_string(ex->p->held().size()) + " (fewer near the end of the context)");
    if (ex->pos + depth + 1 > (int64_t)ex->p->context()) return fail(ex, FLERP_E_FULL, "flerp_draft_hold: no room in the context");
    ex->hold = depth;
    return FLERP_OK;
}
flerp_status flerp_verify(flerp_executor *ex, flerp_verify_run *runs, size_t n_runs, const flerp_want *want) {
    if (runs == nullptr || n_runs != 1) return unsupported(ex, "one run a verify (no FLERP_CAP_VERIFY_BATCH)");
    flerp_verify_run &run = runs[0];
    if (const flerp_status st = seq_ok(ex, run.seq)) return st;
    if (ex->drafts == 0) return unsupported(ex, "the seq was not opened with drafts (or caps.max_draft is 0)");
    if (ex->verifying) return fail(ex, FLERP_E_STATE, "a verify is already pending");
    if (run.ids == nullptr || run.n_ids != 1 || run.ids[0] != ex->pending)
        return unsupported(ex, "a verify runs the pending id and the held drafts only (n_ids 1, ids[0] the last drawn id)");
    if (run.held != ex->hold && !(ex->opt.draft_value && run.held > ex->hold))
        return fail(ex, FLERP_E_STATE, "run.held differs from the last flerp_draft_hold");
    const int32_t held = ex->hold;   // draft_value: what the stage held (run.held may name more)
    if (run.masks != nullptr || (want != nullptr && want->masks != nullptr)) return unsupported(ex, "no masks (no FLERP_CAP_MASKS)");
    if (want != nullptr && want->delivery != FLERP_OUT_TOKENS) return unsupported(ex, "a verify returns tokens (FLERP_OUT_TOKENS)");
    const int64_t rows = 1 + (int64_t)held;
    if (run.out == nullptr || run.out->tokens == nullptr || run.out->tokens_cap < (size_t)rows) return fail(ex, FLERP_E_ARGUMENT, "flerp_verify: out->tokens is too small");
    if (held > 0 && (run.held_out == nullptr || run.held_cap < (size_t)held)) return fail(ex, FLERP_E_ARGUMENT, "flerp_verify: held_out is too small");
    std::vector<int32_t> drafts(ex->p->held().begin(), ex->p->held().begin() + held);
    try {
        ex->round = ex->p->Verify(ex->pending, (uint32_t)held, (uint32_t)ex->pos);
    } catch (const std::exception &e) {
        return fail(ex, FLERP_E_FAILED, std::string("verify: ") + e.what());
    }
    for (int64_t i = 0; i < rows; ++i) run.out->tokens[i] = ex->round.drawn[(size_t)i];
    if (run.out->nan != nullptr && run.out->nan_cap >= (size_t)rows) std::memset(run.out->nan, 0, (size_t)rows);
    run.out->rows = rows;
    for (int32_t i = 0; i < held; ++i) run.held_out[i] = drafts[(size_t)i];
    ex->verifying = true;
    ++ex->stats.verifies, ex->stats.verify_rows += rows;
    ex->stats.draft_tokens += (int64_t)ex->p->held().size();
    return FLERP_OK;
}

namespace {
// The keep of a verify: `rows` rows (the pending id and rows - 1 drafts) and the carried id must be the last stage's
// own greedy chain, which it has already drafted after.
flerp_status keep_rows(flerp_executor *ex, int32_t seq, int64_t rows, flerp_token carry) {
    if (const flerp_status st = seq_ok(ex, seq)) return st;
    if (!ex->verifying) return fail(ex, FLERP_E_STATE, "nothing to keep: no verify is pending");
    const int64_t kept = (int64_t)ex->round.kept.size();   // the stage's kept drafts + 1
    const int64_t all = 1 + (int64_t)ex->hold;
    if (rows < 1 || rows > all) return fail(ex, FLERP_E_ARGUMENT, "keep of " + std::to_string(rows) + " rows of " + std::to_string(all));
    if (rows > kept || (rows == kept && carry >= 0 && carry != ex->round.kept.back()))
        return fail(ex, FLERP_E_STATE, "keep of " + std::to_string(rows) + " rows (carry " + std::to_string(carry) + "): the last stage kept " +
                                           std::to_string(kept) + " (carry " + std::to_string(ex->round.kept.back()) +
                                           "), its greedy chain; a drafting seq keeps that, or fewer to end the reply");
    // fewer rows than the stage kept (the caller ends the reply at an end id, or cuts for fed ids): the stages are
    // past this point, so the seq takes no more forwards until it is reset
    ex->left = rows < kept;
    ex->pos += rows;
    ex->pending = ex->round.kept.back();
    ex->stats.kept_rows += rows, ex->stats.dropped_rows += all - rows;
    ex->verifying = false, ex->hold = 0;
    return FLERP_OK;
}
}  // namespace

flerp_status flerp_keep(flerp_executor *ex, int32_t seq) { return keep_rows(ex, seq, 1 + (int64_t)ex->hold, -1); }
flerp_status flerp_keep_prefix(flerp_executor *ex, int32_t seq, int64_t rows, flerp_token carry) {
    return keep_rows(ex, seq, rows, carry);
}
flerp_status flerp_drop(flerp_executor *ex, int32_t) {
    return unsupported(ex, "no drop: a round's rows are kept by the last stage's greedy chain (its recurrent ring keeps no slot for an undo)");
}

flerp_status flerp_snapshot_save(flerp_executor *ex, int32_t, flerp_snapshot *) {
    return unsupported(ex, "no snapshots in this version (no FLERP_CAP_SNAPSHOT)");
}
flerp_status flerp_snapshot_restore(flerp_executor *ex, int32_t, flerp_snapshot) {
    return unsupported(ex, "no snapshots in this version (no FLERP_CAP_SNAPSHOT)");
}
flerp_status flerp_snapshot_drop(flerp_executor *ex, int32_t, flerp_snapshot) {
    return unsupported(ex, "no snapshots in this version (no FLERP_CAP_SNAPSHOT)");
}
int64_t flerp_snapshot_pos(const flerp_executor *ex, int32_t, flerp_snapshot) {
    return unsupported(ex, "no snapshots in this version (no FLERP_CAP_SNAPSHOT)");
}
int64_t flerp_snapshot_bytes(const flerp_executor *ex, int32_t, flerp_snapshot, int64_t) {
    return unsupported(ex, "no snapshots in this version (no FLERP_CAP_SNAPSHOT_BYTES)");
}
flerp_status flerp_snapshot_export(flerp_executor *ex, int32_t, flerp_snapshot, int64_t, void *, size_t, size_t *) {
    return unsupported(ex, "no snapshots in this version (no FLERP_CAP_SNAPSHOT_BYTES)");
}
flerp_status flerp_state_import(flerp_executor *ex, int32_t, const void *, size_t, int64_t, int64_t) {
    return unsupported(ex, "no state import in this version (no FLERP_CAP_SNAPSHOT_BYTES)");
}
flerp_status flerp_snapshot_write(flerp_executor *ex, int32_t, flerp_snapshot, const char *, uint64_t) {
    return unsupported(ex, "no snapshot files (no FLERP_CAP_SNAPSHOT_FILES)");
}
flerp_status flerp_snapshot_read(flerp_executor *ex, int32_t, const char *, uint64_t, int64_t, flerp_snapshot *) {
    return unsupported(ex, "no snapshot files (no FLERP_CAP_SNAPSHOT_FILES)");
}

flerp_status flerp_get_stats(const flerp_executor *ex, flerp_stats *out) {
    if (out == nullptr || out->size < sizeof(flerp_stats)) return fail(ex, FLERP_E_ARGUMENT, "flerp_stats is null or short");
    const uint32_t size = out->size;
    *out = ex->stats;
    out->size = size, out->device_bytes = ex->p->device_bytes();
    return FLERP_OK;
}

size_t flerp_stats_json(const flerp_executor *ex, char *out, size_t cap) {
    char buf[512];
    const int n = std::snprintf(buf, sizeof(buf),
                                "{\"world\": %u, \"positions_held\": %lld, \"failed\": %s, \"prefix_hits\": %lld, \"prefix_misses\": %lld, "
                                "\"prefix_tokens_reused\": %lld, \"prefix_saves\": %lld, \"prefix_kept\": %lld}",
                                ex->opt.world, (long long)ex->pos, ex->p->failed() ? "true" : "false", (long long)ex->reuse_hits,
                                (long long)ex->reuse_misses, (long long)ex->reuse_tokens, (long long)ex->reuse_saves, (long long)ex->snap_K);
    if (out != nullptr && cap > 0) {
        const size_t m = std::min((size_t)n, cap - 1);
        std::memcpy(out, buf, m);
        out[m] = 0;
    }
    return (size_t)n;
}

}  // extern "C"
