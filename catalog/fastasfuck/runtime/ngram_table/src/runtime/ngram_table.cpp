#include "runtime/ngram_table.hpp"

#include "common/check.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace strix {

// Concurrent row reads (the SSD queue depth). 64 reads a cold gather ~1.25x faster than 32 (16,384 random rows 168 ->
// 134 ms, 65,536 rows 630 -> 494 ms); 128 and 256 gain nothing more.
constexpr size_t kReadThreads = 64;
size_t ngram_read_threads() { return kReadThreads; }

namespace {

// Persistent row readers for parallel_rows: starting a std::thread per read cost more than the read itself (a 16-row
// gather ~1 ms of thread start-up vs ~0.17 ms of SSD reads in parallel). A batch is n indices; the caller works on its
// own batch too and returns when every index is done. Concurrent batches (a gather and a prefetch) share the threads.
class ReadPool {
public:
    static ReadPool &get() {
        static ReadPool pool(kReadThreads);
        return pool;
    }
    ~ReadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread &t : threads_) t.join();
    }
    void run(size_t n, size_t max_helpers, const std::function<void(size_t)> &fn) {
        Batch b;
        b.fn = &fn, b.n = n, b.helpers_left = std::min(max_helpers, threads_.size());
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push_back(&b);
        }
        cv_.notify_all();
        work(b);  // the caller too
        std::unique_lock<std::mutex> lock(mu_);
        // Off the queue before it goes out of scope; then wait for helpers still inside it.
        queue_.erase(std::find(queue_.begin(), queue_.end(), &b));
        done_cv_.wait(lock, [&] { return b.active == 0; });
        if (b.err) std::rethrow_exception(b.err);
    }

private:
    struct Batch {
        const std::function<void(size_t)> *fn = nullptr;
        size_t n = 0, helpers_left = 0, active = 0;  // helpers_left / active under mu_
        std::atomic<size_t> next{0};
        std::exception_ptr err;  // under mu_
    };
    explicit ReadPool(size_t threads) {
        for (size_t k = 0; k < threads; ++k) threads_.emplace_back([this] { loop(); });
    }
    void work(Batch &b) {  // indices until none are left; the first failure is kept, the rest still run
        for (size_t u; (u = b.next.fetch_add(1)) < b.n;) {
            try {
                (*b.fn)(u);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mu_);
                if (!b.err) b.err = std::current_exception();
            }
        }
    }
    void loop() {
        for (;;) {
            Batch *b = nullptr;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] {
                    if (stop_) return true;
                    for (Batch *q : queue_)
                        if (q->helpers_left > 0 && q->next.load() < q->n) return true;
                    return false;
                });
                if (stop_) return;
                // `next` moves without the lock (the caller works on its own batch): the batch that woke this thread
                // may be used up by now - then wait again.
                for (Batch *q : queue_)
                    if (q->helpers_left > 0 && q->next.load() < q->n) {
                        b = q;
                        break;
                    }
                if (b == nullptr) continue;
                --b->helpers_left, ++b->active;
            }
            work(*b);
            {
                std::lock_guard<std::mutex> lock(mu_);
                --b->active;
            }
            done_cv_.notify_all();
        }
    }
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::vector<Batch *> queue_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
};

}  // namespace

void parallel_rows(size_t n, size_t max_threads, const std::function<void(size_t)> &fetch) {
    STRIX_CHECK(max_threads >= 1, "parallel_rows: max_threads = 0");
    STRIX_CHECK(fetch != nullptr, "parallel_rows: empty fetch function");
    if (n == 0) return;
    if (n == 1 || max_threads == 1) {
        for (size_t u = 0; u < n; ++u) fetch(u);
        return;
    }
    ReadPool::get().run(n, std::min(max_threads, n) - 1, fetch);
}

NgramTableRows::NgramTableRows(const std::string &path, int64_t cache_rows) : file_(path), cap_(0) {
    STRIX_CHECK(cache_rows >= 0 && cache_rows <= (1ll << 26), "NgramTableRows: cache_rows = ", cache_rows,
                ", expected 0..2^26 (up to ", 1ll << 26, " rows)");
    STRIX_CHECK(file_.info().rows < (int64_t)UINT32_MAX, "NgramTableRows: rows = ", file_.info().rows,
                " exceeds uint32_t limit (", (uint64_t)UINT32_MAX, ")");
    resize_cache(cache_rows);
}

int32_t NgramTableRows::find_slot(uint32_t row) const {
    if (table_.empty()) return -1;
    uint32_t idx = fmix32(row) & table_mask_;
    for (;;) {
        const TableEntry &e = table_[idx];
        if (e.row == kEmptyRow) return -1;
        if (e.row == row) return e.slot;
        idx = (idx + 1) & table_mask_;
    }
}

void NgramTableRows::table_insert(uint32_t row, int32_t slot) const {
    uint32_t idx = fmix32(row) & table_mask_;
    for (;;) {
        TableEntry &e = table_[idx];
        if (e.row == kEmptyRow) {
            e.row = row;
            e.slot = slot;
            return;
        }
        if (e.row == row) {
            e.slot = slot;
            return;
        }
        idx = (idx + 1) & table_mask_;
    }
}

void NgramTableRows::table_erase(uint32_t row) const {
    if (table_.empty()) return;
    uint32_t idx = fmix32(row) & table_mask_;
    for (;;) {
        if (table_[idx].row == kEmptyRow) return;  // not found
        if (table_[idx].row == row) break;
        idx = (idx + 1) & table_mask_;
    }
    uint32_t curr = idx;
    uint32_t next = (curr + 1) & table_mask_;
    for (;;) {
        if (table_[next].row == kEmptyRow) {
            table_[curr].row = kEmptyRow;
            table_[curr].slot = -1;
            return;
        }
        uint32_t natural = fmix32(table_[next].row) & table_mask_;
        if (((curr - natural) & table_mask_) < ((next - natural) & table_mask_)) {
            table_[curr] = table_[next];
            curr = next;
        }
        next = (next + 1) & table_mask_;
    }
}

int32_t NgramTableRows::take_slot(uint32_t row) const {
    if (used_ < cap_) {
        int32_t s = used_++;
        slot_row_[(size_t)s] = row;
        clock_ref_[(size_t)s] = 1;
        table_insert(row, s);
        return s;
    }
    for (;;) {
        int32_t s = clock_hand_;
        clock_hand_ = (clock_hand_ + 1 == cap_) ? 0 : clock_hand_ + 1;
        if (clock_ref_[(size_t)s] == 0) {
            uint32_t old_row = slot_row_[(size_t)s];
            table_erase(old_row);
            slot_row_[(size_t)s] = row;
            clock_ref_[(size_t)s] = 1;
            table_insert(row, s);
            return s;
        }
        clock_ref_[(size_t)s] = 0;
    }
}

std::vector<int64_t> NgramTableRows::unique_rows(const int64_t *rows, int64_t n, const char *fn) const {
    STRIX_CHECK(rows || n == 0, "NgramTableRows::", fn, ": null rows with n = ", n);
    STRIX_CHECK(n >= 0, "NgramTableRows::", fn, ": n = ", n);
    const NgramTableInfo &info = file_.info();
    std::vector<int64_t> uniq(rows, rows + n);
    for (int64_t r : uniq)
        STRIX_CHECK(r >= 0 && r < info.rows, "NgramTableRows::", fn, ": row id ", r, " outside the table's 0..",
                    info.rows - 1);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    return uniq;
}

void NgramTableRows::gather(const int64_t *rows, int64_t n, uint16_t *out) const {
    STRIX_CHECK(out || n == 0, "NgramTableRows::gather: null out with n = ", n);
    const std::vector<int64_t> uniq = unique_rows(rows, n, "gather");
    const size_t D = (size_t)file_.info().row_dim;
    const size_t rb = (size_t)file_.info().row_bytes();
    if (uniq.empty()) return;

    std::vector<uint8_t> raw_uniq(uniq.size() * rb);
    std::vector<size_t> miss;
    {
        std::lock_guard<std::mutex> lock(mu_);
        stats_.requested += (uint64_t)n;
        stats_.unique += uniq.size();
        for (size_t u = 0; u < uniq.size(); ++u) {
            int32_t s = cap_ > 0 ? find_slot((uint32_t)uniq[u]) : -1;
            if (s < 0) {
                miss.push_back(u);
            } else {
                std::memcpy(raw_uniq.data() + u * rb, data_.data() + (size_t)s * rb, rb);
                clock_ref_[(size_t)s] = 1;
            }
        }
        stats_.hits += uniq.size() - miss.size();
        stats_.reads += miss.size();
    }

    if (!miss.empty()) {
        const size_t kReadThreads = ngram_read_threads();
        parallel_rows(miss.size(), kReadThreads, [&](size_t m) {
            const size_t u = miss[m];
            uint8_t *p = raw_uniq.data() + u * rb;
            uint64_t at = kNgramTableDataOffset + (uint64_t)uniq[u] * rb;
            size_t left = rb;
            while (left > 0) {
                const ssize_t got = ::pread(file_.fd(), p, left, (off_t)at);
                if (got < 0 && errno == EINTR) continue;
                STRIX_CHECK(got > 0, "NgramTableRows: read of row ", uniq[u], " from '", file_.path(),
                            "' failed: ", got < 0 ? std::strerror(errno) : "end of file");
                p += got, left -= (size_t)got, at += (uint64_t)got;
            }
        });
        if (cap_ > 0) {
            std::lock_guard<std::mutex> lock(mu_);
            if (cap_ > 0) {
                for (size_t m = 0; m < miss.size(); ++m) {
                    const size_t u = miss[m];
                    uint32_t r = (uint32_t)uniq[u];
                    int32_t s = find_slot(r);
                    if (s >= 0) {
                        clock_ref_[(size_t)s] = 1;
                    } else {
                        int32_t ns = take_slot(r);
                        std::memcpy(data_.data() + (size_t)ns * rb, raw_uniq.data() + u * rb, rb);
                    }
                }
            }
        }
    }

    // Decode directly outside of lock
    std::vector<uint16_t> decoded_uniq(uniq.size() * D);
    for (size_t u = 0; u < uniq.size(); ++u) {
        ngram_decode_row(file_.info(), raw_uniq.data() + u * rb, decoded_uniq.data() + u * D);
    }
    const size_t vb = D * 2;
    for (int64_t i = 0; i < n; ++i) {
        const size_t u = (size_t)(std::lower_bound(uniq.begin(), uniq.end(), rows[i]) - uniq.begin());
        std::memcpy(out + i * (int64_t)D, decoded_uniq.data() + u * D, vb);
    }
}

void NgramTableRows::prefetch(const int64_t *rows, int64_t n) const {
    const std::vector<int64_t> uniq = unique_rows(rows, n, "prefetch");
    if (cap_ == 0 || uniq.empty()) return;
    const size_t rb = (size_t)file_.info().row_bytes();

    std::vector<size_t> miss;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (cap_ == 0) return;
        for (size_t u = 0; u < uniq.size(); ++u) {
            if (find_slot((uint32_t)uniq[u]) < 0) {
                miss.push_back(u);
            }
        }
    }
    if (miss.empty()) return;

    std::vector<uint8_t> raw_miss(miss.size() * rb);
    const size_t kReadThreads = ngram_read_threads();
    parallel_rows(miss.size(), kReadThreads, [&](size_t m) {
        const size_t u = miss[m];
        uint8_t *p = raw_miss.data() + m * rb;
        uint64_t at = kNgramTableDataOffset + (uint64_t)uniq[u] * rb;
        size_t left = rb;
        while (left > 0) {
            const ssize_t got = ::pread(file_.fd(), p, left, (off_t)at);
            if (got < 0 && errno == EINTR) continue;
            STRIX_CHECK(got > 0, "NgramTableRows: prefetch read of row ", uniq[u], " failed: ",
                        got < 0 ? std::strerror(errno) : "end of file");
            p += got, left -= (size_t)got, at += (uint64_t)got;
        }
    });

    size_t inserted = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (cap_ > 0) {
            for (size_t m = 0; m < miss.size(); ++m) {
                const size_t u = miss[m];
                uint32_t r = (uint32_t)uniq[u];
                if (find_slot(r) < 0) {
                    int32_t s = take_slot(r);
                    std::memcpy(data_.data() + (size_t)s * rb, raw_miss.data() + m * rb, rb);
                    ++inserted;
                }
            }
            stats_.prefetched += inserted;
        }
    }
}

void NgramTableRows::clear_cache() const {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto &e : table_) {
        e.row = kEmptyRow;
        e.slot = -1;
    }
    std::fill(clock_ref_.begin(), clock_ref_.end(), 0);
    std::fill(slot_row_.begin(), slot_row_.end(), kEmptyRow);
    clock_hand_ = 0;
    used_ = 0;
}

int64_t NgramTableRows::cache_rows() const {
    std::lock_guard<std::mutex> lock(mu_);
    return cap_;
}

void NgramTableRows::resize_cache(int64_t rows) const {
    STRIX_CHECK(rows >= 0 && rows <= (1ll << 26), "NgramTableRows::resize_cache: rows = ", rows, ", expected 0..2^26");
    std::lock_guard<std::mutex> lock(mu_);
    const size_t rb = (size_t)file_.info().row_bytes();
    cap_ = rows;
    used_ = 0;
    clock_hand_ = 0;
    std::vector<uint8_t>((size_t)(rows * rb)).swap(data_);
    std::vector<uint32_t>((size_t)rows, kEmptyRow).swap(slot_row_);
    std::vector<uint8_t>((size_t)rows, 0).swap(clock_ref_);
    if (rows > 0) {
        size_t ts = 16;
        while (ts < (size_t)rows * 2) ts <<= 1;
        table_mask_ = (uint32_t)(ts - 1);
        std::vector<TableEntry>(ts, TableEntry{kEmptyRow, -1}).swap(table_);
    } else {
        table_mask_ = 0;
        std::vector<TableEntry>().swap(table_);
    }
}

void NgramTableRows::drop_page_cache() const {
    const int rc = ::posix_fadvise(file_.fd(), 0, 0, POSIX_FADV_DONTNEED);  // returns the error number, not errno
    STRIX_CHECK(rc == 0, "NgramTableRows::drop_page_cache: posix_fadvise(DONTNEED) on '", file_.path(), "' failed: ",
                std::strerror(rc));
}

NgramTableRows::Stats NgramTableRows::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

}  // namespace strix
