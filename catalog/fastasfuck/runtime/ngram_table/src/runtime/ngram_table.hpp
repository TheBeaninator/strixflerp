#pragma once

// PLE n-gram rows for the forward. NgramRowSource is what Qwen4ExpModel reads rows through; two sources:
//   NgramTableRows      - the converted table file (formats/ngram_table.hpp): rows read on demand from SSD
//                         (the table stays on SSD), deduplicated and fetched with up to 64 concurrent preads, and an
//                         LRU cache of recently used rows (n-gram ids are heavily skewed - frequent n-grams repeat
//                         across a conversation and its prefills).
//   NgramRowsFromShards - interim (runtime/ngram_rows.hpp): straight from the HF checkpoint's shards.
// The caller picks one explicitly (Qwen4ExpModel: a table file or a checkpoint directory) - not a fallback.

#include "formats/ngram_table.hpp"
#include "runtime/ple_hash.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strix {

class NgramRowSource {
public:
    virtual ~NgramRowSource() = default;
    virtual const PleHashParams &hash_params() const = 0;
    virtual int64_t row_dim() const = 0;
    // rows [n] table row ids (ple_ngram_ids output) -> out [n, row_dim] BF16 bits. Throws on an id outside the table.
    virtual void gather(const int64_t *rows, int64_t n, uint16_t *out) const = 0;
    // A hint: rows [n] a gather will soon ask for - a source with a row cache reads the missing ones into it now,
    // so that gather finds them there. A source without a cache does nothing (the gather reads them as always).
    virtual void prefetch(const int64_t *rows, int64_t n) const { (void)rows, (void)n; }
};

// Concurrent row fetch shared by both sources: fetch(u) for u in [0, n) on the caller's thread and up to
// max_threads - 1 of a persistent pool of reader threads (shared by concurrent calls); returns when all n are done
// and rethrows the first failure.
void parallel_rows(size_t n, size_t max_threads, const std::function<void(size_t)> &fetch);
// The reader pool's thread count (the value parallel_rows caps at; bench provenance fields record it).
size_t ngram_read_threads();

class NgramTableRows : public NgramRowSource {
public:
    static constexpr int64_t kDefaultCacheRows = 1 << 18;
    explicit NgramTableRows(const std::string &path, int64_t cache_rows = kDefaultCacheRows);
    const PleHashParams &hash_params() const override { return file_.info().params; }
    int64_t row_dim() const override { return file_.info().row_dim; }
    void gather(const int64_t *rows, int64_t n, uint16_t *out) const override;
    void prefetch(const int64_t *rows, int64_t n) const override;
    int64_t cache_rows() const;
    // RAM per cache row: row_bytes (162 B for Q8Row) + slot_row_ (4 B) + clock_ref_ (1 B) + table overhead (24 B).
    uint64_t cache_bytes_per_row() const { return (uint64_t)file_.info().row_bytes() + 4 + 1 + 24; }
    // Empties the cache and reallocates it at `rows` (0..2^26), freeing the old arrays - the startup memory plan's
    // shrink (serve/memory_plan.hpp). Thread-safe; gathers meanwhile wait.
    void resize_cache(int64_t rows) const;
    const NgramTableFile &file() const { return file_; }
    // Gathers: rows requested, distinct rows per gather (summed), of those served from the cache / read from the
    // file. Prefetches: rows read from the file into the cache (a later gather counts them as hits).
    struct Stats {
        uint64_t requested = 0, unique = 0, hits = 0, reads = 0, prefetched = 0;
    };
    Stats stats() const;
    // For measurements: empties the row cache (its counters stay), as after a
    // server restart; and asks the kernel to drop the table file's clean pages from the page cache, as under memory
    // pressure (posix_fadvise DONTNEED - advisory: pages another process holds mapped may stay).
    void clear_cache() const;
    void drop_page_cache() const;

private:
    struct TableEntry {
        uint32_t row;
        int32_t slot;
    };
    static constexpr uint32_t kEmptyRow = UINT32_MAX;

    static inline uint32_t fmix32(uint32_t h) {
        h ^= h >> 16;
        h *= 0x7feb352du;
        h ^= h >> 15;
        h *= 0x846ca68bu;
        h ^= h >> 16;
        return h;
    }

    NgramTableFile file_;
    mutable int64_t cap_ = 0;  // under mu_ (resize_cache)
    mutable std::mutex mu_;
    mutable std::vector<TableEntry> table_;
    mutable uint32_t table_mask_ = 0;
    mutable std::vector<uint32_t> slot_row_;
    mutable std::vector<uint8_t> clock_ref_;
    mutable std::vector<uint8_t> data_;  // raw file row bytes (e.g. 162 B for Q8Row)
    mutable int32_t clock_hand_ = 0, used_ = 0;
    mutable Stats stats_;

    int32_t find_slot(uint32_t row) const;
    void table_insert(uint32_t row, int32_t slot) const;
    void table_erase(uint32_t row) const;
    int32_t take_slot(uint32_t row) const;
    std::vector<int64_t> unique_rows(const int64_t *rows, int64_t n, const char *fn) const;
};

}  // namespace strix
