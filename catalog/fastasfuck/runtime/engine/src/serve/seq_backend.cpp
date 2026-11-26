#include "serve/seq_backend.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <chrono>

namespace strix {

namespace {
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

GpuArbiter::GpuArbiter(LmBackend &be)
    : be_(be),
      epoch_((size_t)be.n_seqs(), 0),
      last_prefill_batch_(-kPrefillYieldBatches),
      concurrent_((size_t)be.n_seqs(), 0) {
    STRIX_CHECK(be.n_seqs() >= 1, "GpuArbiter: the backend holds ", be.n_seqs(), " conversations, expected >= 1");
}

void GpuArbiter::check_ok_locked(int seq) const {
    STRIX_CHECK(seq >= 0 && seq < (int)epoch_.size(), "GpuArbiter: seq ", seq, ", expected 0..", epoch_.size() - 1);
    STRIX_CHECK(!failed_ && epoch_[(size_t)seq] == failures_, "GpuArbiter: seq ", seq,
                ": a call on the shared GPU state failed (on this or another slot) since this slot's last reset - "
                "reset it first");
}

void GpuArbiter::fail_locked() {
    failed_ = true, ++failures_;
    resolving_.clear();
    for (auto &[s, p] : pending_)
        if (!p.done) p.done = true, p.error = std::make_exception_ptr(std::runtime_error(
                                         "GpuArbiter: the batched verify was abandoned - another slot's call failed"));
    for (auto &[s, d] : pending_drafts_)
        if (!d.done) d.done = true, d.error = std::make_exception_ptr(std::runtime_error(
                                         "GpuArbiter: the batched draft was abandoned - another slot's call failed"));
}

bool GpuArbiter::has_pending_verifies_locked() const {
    for (const auto &[s, p] : pending_)
        if (!p.done) return true;
    return false;
}

bool GpuArbiter::others_decoding_locked(int seq) const {
    for (int s : decoding_)
        if (s != seq) return true;
    return false;
}

bool GpuArbiter::can_run_locked(int seq, Kind kind) const {
    if (busy_) return false;
    if (kind == Kind::Hint) return true;
    if (kind == Kind::Resolve) return resolving_.count(seq) > 0;
    if (!resolving_.empty()) return false;  // the batch's resolutions first (its shared workspaces)

    // Batch runs before heavy Prefill, but lightweight Call (restore_snapshot, short prefill <= 128)
    // must slip in so waiting slots can join the active decode batch immediately.
    if (kind == Kind::Prefill && lead_ready_locked(now_s())) return false;
    if (kind == Kind::Prefill && has_pending_verifies_locked()) return false;

    // Deficit Round Robin Event-Driven Quantum Interleaving:
    // If other slots are actively decoding and this slot is running an exclusive prefill chunk (not decoding itself),
    // it must yield until active decode slots complete their quantum of verify batches.
    if (kind == Kind::Prefill && others_decoding_locked(seq) && !decoding_.count(seq)) {
        if (stats_.verify_batches - last_prefill_batch_ < kPrefillYieldBatches) {
            return false;  // yield GPU until active decode slots run their verify batch quantum
        }
        // Fair round-robin across concurrent prefill slots:
        // If this slot was the last one to run a prefill chunk and another prefill slot is waiting, let the other run first.
        if (last_prefill_seq_ == seq && prefill_waiting_.size() > 1) {
            for (int s : prefill_waiting_) {
                if (s != seq && !decoding_.count(s)) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool GpuArbiter::poisoned_locked(int seq) const {
    return failed_ || epoch_[(size_t)seq] != failures_;
}

bool GpuArbiter::run(int seq, Kind kind, const std::function<void()> &f) {
    std::unique_lock<std::mutex> lock(mu_);
    // A Hint is advisory (reads, prefetches): after a failed call there is nothing trustworthy left to read,
    // so it skips f and reports false instead of throwing - an advisory call must never be what takes a worker
    // down (the production abort: a post-request snapshot_pos Hint threw after another slot's clean failure).
    if (kind == Kind::Hint && poisoned_locked(seq)) return false;
    check_ok_locked(seq);
    const bool call = kind == Kind::Call;
    const bool prefill = kind == Kind::Prefill;
    auto waiting_call = call ? calls_waiting_.insert(seq) : calls_waiting_.end();
    auto waiting_prefill = prefill ? prefill_waiting_.insert(seq) : prefill_waiting_.end();
    while (!failed_ && !can_run_locked(seq, kind)) {
        cv_.wait(lock);
    }
    if (call) {
        calls_waiting_.erase(waiting_call);
    }
    if (prefill) {
        prefill_waiting_.erase(waiting_prefill);
        if (!decoding_.count(seq)) {
            last_prefill_batch_ = stats_.verify_batches;
            last_prefill_seq_ = seq;
        }
    }
    if (kind == Kind::Hint && poisoned_locked(seq)) return false;  // poisoned while waiting: same skip
    check_ok_locked(seq);
    STRIX_CHECK(kind != Kind::Resolve || resolving_.count(seq), "GpuArbiter: seq ", seq,
                " resolves a verify it has no pending run of");
    busy_ = true;
    lock.unlock();
    try {
        f();
    } catch (...) {
        lock.lock();
        busy_ = false;
        if (be_.failed()) fail_locked();
        lock.unlock();
        cv_.notify_all();
        throw;
    }
    lock.lock();
    busy_ = false;
    if (kind == Kind::Resolve) resolving_.erase(seq);
    lock.unlock();
    cv_.notify_all();
    return true;  // f ran
}

bool GpuArbiter::run_concurrent(int seq, bool advisory, const std::function<void()> &f) {
    std::unique_lock<std::mutex> lock(mu_);
    STRIX_CHECK(seq >= 0 && seq < (int)epoch_.size(), "GpuArbiter::run_concurrent: seq ", seq, ", expected 0..",
                epoch_.size() - 1);
    if (advisory && poisoned_locked(seq)) return false;
    check_ok_locked(seq);
    ++concurrent_[(size_t)seq];
    ++concurrent_total_;
    lock.unlock();
    auto finish = [&](bool thrown) {
        lock.lock();
        const bool zero_seq = --concurrent_[(size_t)seq] == 0;
        const bool zero_all = --concurrent_total_ == 0;
        // Skip be_.failed() while busy_ holds an exclusive GPU call: Qwen4ExpSession::broken_ is transiently true
        // during an active forward/draft on another slot.
        const bool failed = thrown && !failed_ && !busy_ && be_.failed();
        if (failed) fail_locked();
        lock.unlock();
        if (zero_seq || zero_all || failed) cv_.notify_all();
    };
    try {
        f();
    } catch (...) {
        finish(true);
        throw;
    }
    finish(false);
    return true;
}

bool GpuArbiter::lead_ready_locked(double now) const {
    if (busy_ || !resolving_.empty() || !calls_waiting_.empty()) return false;
    double first = 0;
    int waiting = 0;
    bool first_alone = false;  // the oldest waiting run can't share a forward: no partner to wait for
    for (const auto &[s, p] : pending_)
        if (!p.done && (!waiting++ || p.since < first)) first = p.since, first_alone = !p.batchable;
    if (waiting == 0) return false;
    if (first_alone) return true;

    const bool timed_out = (now - first >= batch_wait_locked());

    // If not timed out, let active drafters run on GPU first
    if (!timed_out && !pending_drafts_.empty()) {
        for (const auto &[s, d] : pending_drafts_) {
            if (!d.done) return false;
        }
    }

    bool host_side = false;  // a decoding seq still to come that is busy on the host (sampling, parsing, streaming) or drafting
    for (int s : decoding_) {
        if (pending_.count(s) && !pending_.at(s).done) continue;
        if (!timed_out && pending_drafts_.count(s) && !pending_drafts_.at(s).done) continue;
        host_side = true;
    }
    return !host_side || timed_out;  // every decoding seq is in, or the host-side wait ran out
}

double GpuArbiter::batch_wait_locked() const {
    if (!calls_waiting_.empty()) {
        return kBatchWaitMinSeconds;
    }
    return std::clamp(0.35 * last_batch_seconds_, kBatchWaitMinSeconds, kBatchWaitMaxSeconds);
}

void GpuArbiter::run_batch(std::unique_lock<std::mutex> &lock) {
    // Pack the waiting runs, oldest first, into the row budget; the rest wait for the next batch.
    std::vector<int> order;
    for (const auto &[s, p] : pending_)
        if (!p.done) order.push_back(s);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return pending_[a].since < pending_[b].since; });
    std::vector<int> members;
    std::vector<LmBackend::VerifyRun> runs;
    int64_t rows = 0;
    const int64_t budget = be_.max_verify_rows();
    bool cands = false;
    int64_t n_valid = 0;
    cv_.wait(lock, [&] { return !busy_; });
    for (int s : order) {
        const Pending &p = pending_[s];
        const int64_t r = (int64_t)p.run.ids.size();
        // A run that can't share a forward (LmBackend::batchable: image rows, mRoPE positions) goes alone: it
        // leads a batch of one when it is the oldest, and otherwise waits for the next batch (upstream #29: both
        // requests of a mixed batch failed the multi-run forward's scalar-positions check).
        if (!runs.empty() && (!p.batchable || rows + r > budget || p.cands != cands || p.n_valid != n_valid)) continue;
        if (runs.empty()) cands = p.cands, n_valid = p.n_valid;
        runs.push_back(p.run), members.push_back(s), rows += r;
        if (!p.batchable) break;
    }
    busy_ = true;
    lock.unlock();
    std::vector<LogitRows> out;
    std::exception_ptr err;
    const double t0 = now_s();
    try {
        out = be_.verify_batch(runs, cands, n_valid);
        STRIX_CHECK(out.size() == runs.size(), "GpuArbiter: verify_batch returned ", out.size(), " results for ",
                    runs.size(), " runs");
    } catch (...) {
        err = std::current_exception();
    }
    lock.lock();
    busy_ = false;
    const double actual_verify_ms = (now_s() - t0) * 1e3;
    last_batch_seconds_ = actual_verify_ms / 1e3;
    if (!err && hw_prof_) {
        float expected_v_ms = hw_prof_->get_verify_cost(rows);
        if (expected_v_ms > 1.0f) {
            float ratio = (float)actual_verify_ms / expected_v_ms;
            ratio = std::clamp(ratio, 0.5f, 3.0f);
            s_verify_ = (1.0f - kEmaDecay) * s_verify_ + kEmaDecay * ratio;
        }
        float round_gpu_ms = (float)actual_verify_ms + (float)draft_ms_accum_;
        draft_ms_accum_ = 0.0;
        ema_round_gpu_ms_ = (1.0f - kEmaDecay) * ema_round_gpu_ms_ + kEmaDecay * round_gpu_ms;
        ema_batch_runs_ = (1.0f - kEmaDecay) * ema_batch_runs_ + kEmaDecay * (float)members.size();
    }
    if (err && be_.failed()) {
        for (int m : members) pending_[m].done = true, pending_[m].error = err;  // the batch's own error
        fail_locked();  // the rest: abandoned
    } else {
        for (size_t i = 0; i < members.size(); ++i) {
            Pending &p = pending_[members[i]];
            p.done = true;
            if (err) p.error = err;
            else p.rows = std::move(out[i]), resolving_.insert(members[i]);
        }
        if (!err) {
            ++stats_.verify_batches, stats_.verify_runs += (int64_t)members.size();
            if (members.size() == 1 && !pending_[members[0]].batchable) ++stats_.verify_alone;
            stats_.max_runs = std::max(stats_.max_runs, (int64_t)members.size());
        }
    }
    cv_.notify_all();
}

bool GpuArbiter::draft_ready_locked(double now) const {
    if (busy_ || !resolving_.empty() || !calls_waiting_.empty()) return false;
    double first = 0;
    int waiting = 0;
    for (const auto &[s, d] : pending_drafts_) {
        if (!d.done) {
            if (!waiting++ || d.since < first) first = d.since;
        }
    }
    if (waiting == 0) return false;
    if (decoding_.size() <= 1) return true;

    bool all_in = true;
    for (int s : decoding_) {
        if (resolving_.count(s)) continue;
        if (pending_.count(s) && !pending_.at(s).done) continue;
        if (pending_drafts_.count(s) && !pending_drafts_.at(s).done) continue;
        all_in = false;
        break;
    }
    return all_in || (now - first >= kDraftWaitMaxSeconds);
}

void GpuArbiter::run_draft_batch(std::unique_lock<std::mutex> &lock) {
    cv_.wait(lock, [&] { return !busy_; });
    std::vector<int> members;
    std::vector<LmBackend::MtpDraftRun> runs;
    for (const auto &[s, d] : pending_drafts_) {
        if (!d.done) {
            members.push_back(s);
            runs.push_back(d.run);
        }
    }
    if (runs.empty()) return;
    busy_ = true;
    lock.unlock();

    std::vector<Top2> results;
    std::exception_ptr err;
    const double t0 = now_s();
    try {
        results = be_.forward_mtp_top2_batch(runs);
        STRIX_CHECK(results.size() == runs.size(), "GpuArbiter: forward_mtp_top2_batch returned ",
                    results.size(), " results for ", runs.size(), " runs");
    } catch (...) {
        err = std::current_exception();
    }

    lock.lock();
    busy_ = false;
    const double actual_draft_ms = (now_s() - t0) * 1e3;
    draft_ms_accum_ += actual_draft_ms;
    if (!err && hw_prof_) {
        float expected_d_ms = hw_prof_->get_draft_cost((int)members.size(), (int)epoch_.size());
        if (expected_d_ms > 0.1f) {
            float ratio = (float)actual_draft_ms / expected_d_ms;
            ratio = std::clamp(ratio, 0.5f, 3.0f);
            s_draft_ = (1.0f - kEmaDecay) * s_draft_ + kEmaDecay * ratio;
        }
    }
    if (err && be_.failed()) {
        for (int m : members) {
            pending_drafts_[m].done = true;
            pending_drafts_[m].error = err;
        }
        fail_locked();
    } else {
        ++stats_.draft_batches;
        stats_.draft_runs += (int64_t)members.size();
        for (size_t i = 0; i < members.size(); ++i) {
            PendingDraft &d = pending_drafts_[members[i]];
            d.done = true;
            if (err) d.error = err;
            else d.result = results[i];
        }
    }
    cv_.notify_all();
}

Top2 GpuArbiter::draft_top2(int seq, int32_t token_id, int64_t step) {
    std::unique_lock<std::mutex> lock(mu_);
    check_ok_locked(seq);
    STRIX_CHECK(!pending_drafts_.count(seq), "GpuArbiter: seq ", seq, " drafts again before previous draft finished");
    PendingDraft &me = pending_drafts_[seq];
    me.run = {seq, token_id, step};
    me.since = now_s();
    me.done = false;
    cv_.notify_all();

    for (;;) {
        if (me.done) break;
        if (!busy_ && draft_ready_locked(now_s())) {
            run_draft_batch(lock);
            continue;
        }
        if (decoding_.size() <= 1) {
            cv_.wait(lock, [&] { return me.done || (!busy_ && draft_ready_locked(now_s())); });
        } else {
            cv_.wait_for(lock, std::chrono::duration<double>(kDraftWaitMaxSeconds / 2));
        }
    }
    PendingDraft done = std::move(me);
    pending_drafts_.erase(seq);
    if (done.error) std::rethrow_exception(done.error);
    return done.result;
}

LogitRows GpuArbiter::verify(int seq, const LmBackend::VerifyRun &run, bool cands, int64_t n_valid) {
    std::unique_lock<std::mutex> lock(mu_);
    check_ok_locked(seq);
    STRIX_CHECK(!pending_.count(seq) && !resolving_.count(seq), "GpuArbiter: seq ", seq,
                " verifies again before its last verify was resolved");
    STRIX_CHECK(run.seq == seq, "GpuArbiter: seq ", seq, " sends a run of seq ", run.seq);
    Pending &me = pending_[seq];
    me.run = run, me.cands = cands, me.n_valid = n_valid, me.since = now_s();
    me.batchable = be_.batchable(seq);  // read here: this seq's own state, between its own calls
    cv_.notify_all();
    for (;;) {
        if (me.done) break;
        if (lead_ready_locked(now_s())) {
            run_batch(lock);
            continue;
        }
        cv_.wait_for(lock, std::chrono::duration<double>(kBatchWaitMinSeconds / 2));
    }
    Pending done = std::move(me);
    pending_.erase(seq);
    if (done.error) std::rethrow_exception(done.error);
    return std::move(done.rows);
}

void GpuArbiter::reset(int seq) {
    std::unique_lock<std::mutex> lock(mu_);
    STRIX_CHECK(seq >= 0 && seq < (int)epoch_.size(), "GpuArbiter::reset: seq ", seq, ", expected 0..",
                epoch_.size() - 1);
    cv_.wait(lock, [&] { return !busy_ && (failed_ ? concurrent_total_ == 0 : concurrent_[(size_t)seq] == 0); });
    busy_ = true;
    const bool whole = failed_;
    const bool drop = !whole && resolving_.count(seq) > 0;  // a request failed between its verify and its resolution
    lock.unlock();
    std::exception_ptr err;
    try {
        if (whole) be_.reset();
        else {
            if (drop) be_.drop_verify(seq);
            be_.reset(seq);
        }
    } catch (...) {
        err = std::current_exception();
    }
    lock.lock();
    busy_ = false;
    if (!err) {
        if (whole) {
            failed_ = false;  // the other seqs' epochs stay behind: each resets before its next call
            last_prefill_batch_ = -kPrefillYieldBatches;
        }
        epoch_[(size_t)seq] = failures_;
        if (last_prefill_seq_ == seq) last_prefill_seq_ = -1;
        resolving_.erase(seq);
        pending_drafts_.erase(seq);
    } else if (be_.failed()) {
        fail_locked();
    }
    lock.unlock();
    cv_.notify_all();
    if (err) std::rethrow_exception(err);
}

void GpuArbiter::set_decoding(int seq, bool on) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (on) decoding_.insert(seq);
        else decoding_.erase(seq);
    }
    cv_.notify_all();
}

bool GpuArbiter::stale(int seq) const {
    std::lock_guard<std::mutex> lock(mu_);
    STRIX_CHECK(seq >= 0 && seq < (int)epoch_.size(), "GpuArbiter::stale: seq ", seq, ", expected 0..", epoch_.size() - 1);
    return failed_ || epoch_[(size_t)seq] != failures_;
}

int GpuArbiter::decoding_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)decoding_.size();
}

int GpuArbiter::pending_verifies_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)pending_.size();
}

int64_t GpuArbiter::pending_verify_rows() const {
    std::lock_guard<std::mutex> lock(mu_);
    int64_t rows = 0;
    for (const auto &[s, p] : pending_) {
        rows += (int64_t)p.run.ids.size();
    }
    return rows;
}

int GpuArbiter::active_drafting_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    int n = (int)decoding_.size();
    int pending_v = (int)pending_.size();
    return std::max(1, n - pending_v);
}

float GpuArbiter::cluster_tau() const {
    std::lock_guard<std::mutex> lock(mu_);
    float denom = std::max(1.0f, ema_batch_runs_ * ema_slot_E_);
    return std::max(1.0f, ema_round_gpu_ms_ / denom);
}

float GpuArbiter::scale_verify() const {
    std::lock_guard<std::mutex> lock(mu_);
    return s_verify_;
}

float GpuArbiter::scale_draft() const {
    std::lock_guard<std::mutex> lock(mu_);
    return s_draft_;
}

void GpuArbiter::report_slot_acceptance(int accepted) {
    std::lock_guard<std::mutex> lock(mu_);
    ema_slot_E_ = (1.0f - kEmaDecay) * ema_slot_E_ + kEmaDecay * (float)accepted;
}

bool GpuArbiter::others_decoding(int seq) const {
    std::lock_guard<std::mutex> lock(mu_);
    STRIX_CHECK(seq >= 0 && seq < (int)epoch_.size(), "GpuArbiter::others_decoding: seq ", seq, ", expected 0..",
                epoch_.size() - 1);
    return others_decoding_locked(seq);
}

GpuArbiter::Stats GpuArbiter::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

// ---------------------------------------------------------------- SeqBackend

SeqBackend::SeqBackend(GpuArbiter &arb, int seq) : arb_(arb), seq_(seq) {
    STRIX_CHECK(seq >= 0 && seq < arb.backend().n_seqs(), "SeqBackend: seq ", seq, ", expected 0..",
                arb.backend().n_seqs() - 1);
}

int64_t SeqBackend::pos() const {
    int64_t p = 0;  // poisoned: 0 (its next Call fails the request anyway; nothing trusts positions mid-failure)
    arb_.run_concurrent(seq_, true, [&] { p = arb_.backend().pos(seq_); });
    return p;
}

void SeqBackend::reset() {
    arb_.reset(seq_);
    verify_rows_ = 0;
}

std::vector<float> SeqBackend::forward(const std::vector<int32_t> &ids, bool want_logits) {
    return forward(ids, (int64_t)(want_logits ? 1 : 0));
}

std::vector<float> SeqBackend::forward(const std::vector<int32_t> &ids, int64_t n_logits) {
    return forward_rows(ids, n_logits, false, 0).full;
}

void SeqBackend::set_lookahead(const std::vector<int32_t> &next_ids) {
    arb_.run_concurrent(seq_, true, [&] { arb_.backend().set_lookahead(seq_, next_ids); });
}

Top2 SeqBackend::forward_mtp_top2(int32_t token_id, int64_t step) {
    return arb_.draft_top2(seq_, token_id, step);
}

std::vector<float> SeqBackend::forward_verify(const std::vector<int32_t> &ids, int64_t n_logits) {
    return forward_verify_rows(ids, n_logits, false, 0).full;
}

LogitRows SeqBackend::forward_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                                   const LogitMasks *masks) {
    LogitRows l;
    const GpuArbiter::Kind k = (ids.size() > 128) ? GpuArbiter::Kind::Prefill : GpuArbiter::Kind::Call;
    arb_.run(seq_, k,
             [&] { l = arb_.backend().forward_rows(seq_, ids, n_logits, cands, n_valid, masks); });
    return l;
}

LogitRows SeqBackend::forward_verify_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands,
                                          int64_t n_valid, const LogitMasks *masks) {
    STRIX_CHECK(n_logits == (int64_t)ids.size(), "SeqBackend::forward_verify_rows: ", n_logits, " logits rows for ",
                ids.size(), " ids - a batched verify returns every row");
    LogitRows l = arb_.verify(seq_, {seq_, ids, masks}, cands, n_valid);
    verify_rows_ = (int64_t)ids.size();
    return l;
}

LogitRows SeqBackend::forward_image_prefill(const std::vector<int32_t> &ids, const PrefillImage &image,
                                            const std::vector<int64_t> &pos3, int64_t n_logits, bool cands,
                                            int64_t n_valid, const LogitMasks *masks) {
    LogitRows l;
    arb_.run(seq_, GpuArbiter::Kind::Prefill, [&] {
        l = arb_.backend().forward_image_prefill(seq_, ids, image, pos3, n_logits, cands, n_valid, masks);
    });
    return l;
}

void SeqBackend::prefetch_ple(const std::vector<int32_t> &ids, int64_t first) {
    arb_.run_concurrent(seq_, true, [&] { arb_.backend().prefetch_ple(seq_, ids, first); });
}

void SeqBackend::keep_verify() {
    STRIX_CHECK(verify_rows_ >= 1, "SeqBackend::keep_verify: slot ", seq_, " has no verify to keep");
    const int64_t rows = verify_rows_;
    arb_.run(seq_, GpuArbiter::Kind::Resolve, [&] { arb_.backend().keep_verify(seq_, rows, -1); });
    verify_rows_ = 0;
}

void SeqBackend::drop_verify() {
    arb_.run(seq_, GpuArbiter::Kind::Resolve, [&] { arb_.backend().drop_verify(seq_); });
    verify_rows_ = 0;
}

void SeqBackend::keep_verify_prefix(const std::vector<int32_t> &ids, int64_t rows) {
    STRIX_CHECK(verify_rows_ >= 1 && (int64_t)ids.size() == verify_rows_ && rows >= 1 && rows < verify_rows_,
                "SeqBackend::keep_verify_prefix: keep ", rows, " of ", ids.size(), " ids, the pending verify has ",
                verify_rows_, " rows");
    arb_.run(seq_, GpuArbiter::Kind::Resolve, [&] { arb_.backend().keep_verify(seq_, rows, -1); });
    verify_rows_ = 0;
}

void SeqBackend::save_snapshot(int slot) {
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().save_snapshot(seq_, slot); });
}

int64_t SeqBackend::snapshot_pos(int slot) const {
    int64_t p = -1;  // poisoned: -1 - no snapshot (a turn_pos of -1 = none; the engine never saves against it)
    arb_.run_concurrent(seq_, true, [&] { p = arb_.backend().snapshot_pos(seq_, slot); });
    return p;
}

bool SeqBackend::can_restore_snapshot(int slot) const {
    bool ok = false;  // poisoned: false - refuse the restore (the caller's next Call fails cleanly)
    arb_.run_concurrent(seq_, true, [&] { ok = arb_.backend().can_restore_snapshot(seq_, slot); });
    return ok;
}

void SeqBackend::restore_snapshot(int slot) {
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().restore_snapshot(seq_, slot); });
}

void SeqBackend::export_snapshot(int slot, HostBuffer &out, int64_t from) {
    arb_.run_concurrent(seq_, false, [&] { arb_.backend().export_snapshot(seq_, slot, out, from); });
}

uint64_t SeqBackend::snapshot_bytes(int slot, int64_t from) const {
    uint64_t b = 0;  // poisoned: 0 (never reached: the save path's snapshot_pos / can_restore hints skip first)
    arb_.run_concurrent(seq_, true, [&] { b = arb_.backend().snapshot_bytes(seq_, slot, from); });
    return b;
}

void SeqBackend::import_state(const HostBuffer &state, int64_t n, int64_t from) {
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().import_state(seq_, state, n, from); });
}

PagedSnapshot SeqBackend::park_snapshot(int slot) {
    PagedSnapshot out;
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { out = arb_.backend().park_snapshot(seq_, slot); });
    return out;
}

void SeqBackend::resume_paged(const HostBuffer &small, const PagedState &paged, int64_t n) {
    // No release on a failed resume: the session spends the parked id as part of mapping it, and a refusal that
    // happened before any mapping leaves the entry - still pinned by the cache - to release it when it leaves RAM.
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().resume_paged(seq_, small, paged, n); });
}

uint64_t SeqBackend::small_snapshot_bytes(int slot) const {
    uint64_t b = 0;
    arb_.run_concurrent(seq_, true, [&] { b = arb_.backend().small_snapshot_bytes(seq_, slot); });
    return b;
}

void SeqBackend::export_small_snapshot(int slot, HostBuffer &out) {
    arb_.run_concurrent(seq_, false, [&] { arb_.backend().export_small_snapshot(seq_, slot, out); });
}

void SeqBackend::resume_small(const HostBuffer &small, int64_t n) {
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().resume_small(seq_, small, n); });
}

void SeqBackend::release_parked(int64_t parked_id) {
    arb_.run(seq_, GpuArbiter::Kind::Call, [&] { arb_.backend().release_parked(seq_, parked_id); });
}

bool SeqBackend::park_budget_left() const {
    bool left = false;  // poisoned: false - the save path takes the host copy (a park needs the GPU anyway)
    arb_.run_concurrent(seq_, true, [&] { left = arb_.backend().park_budget_left(seq_); });
    return left;
}

}  // namespace strix
