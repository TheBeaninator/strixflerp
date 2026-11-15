// CTXMOVE-XFER (card a4a320429d): the cache transfer API that the server's POST /cache/export and /cache/import
// routes run. Host only - no GPU, no weights: cache entries are token prefixes over opaque state bytes, so the
// states here are synthetic. What this proves is the transport's contract:
//   1  an entry exports as its exact disk file bytes (header checked independently here, not only by the import)
//      and imports into a fresh cache: best/in_ram agree, and the state bytes come back identical;
//   2  a delta exports base first, then itself (two files, the second's header naming the first's length), and a
//      fresh cache that imports the body can resume the full prompt;
//   3  a foreign fingerprint is refused with Foreign and puts nothing;
//   4  a flipped state byte, a flipped token byte and a cut body are Damaged, and nothing goes in;
//   5  a delta body whose base is missing answers BaseMissing (the same body into a cache that has the base is Ok);
//   6  re-importing what a cache already has is Ok (a refresh), and exporting a prompt nothing covers answers 0.
//
// usage: cache_xfer_check <scratch_dir>
// one ok/FAIL line per check; the exit status is the number of failures (2: usage).

#include "formats/strixw.hpp"  // strix_hash64, for checking the exported headers independently
#include "serve/host_buffer.hpp"
#include "serve/prompt_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace strix;
namespace fs = std::filesystem;

namespace {

int failures = 0;
void check(bool ok, const char *what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    std::fflush(stdout);
    if (!ok) ++failures;
}

constexpr uint64_t kHeader = 4096;
uint64_t align4k(uint64_t x) { return (x + 4095) & ~uint64_t{4095}; }
uint64_t file_total(uint64_t n, uint64_t state_bytes) { return align4k(kHeader + n * 4) + state_bytes; }

// One exported file's header, read here independently of PromptCache::import_entries: a bug shared between the
// writer and the importer would hide, so these checks recompute the layout and both hashes from first principles.
struct FileHead {
    bool ok = false;
    uint64_t tokens = 0, state_bytes = 0, state_at = 0, base_n = 0;
    uint32_t kind = 0;
};
FileHead check_file(const std::string &b, size_t off) {
    FileHead r;
    if (b.size() - off < kHeader || std::memcmp(b.data() + off, "STRXPCCH", 8) != 0) return r;
    const uint8_t *h = reinterpret_cast<const uint8_t *>(b.data()) + off;
    auto rd = [&](size_t o, auto v) { std::memcpy(&v, h + o, sizeof(v)); return v; };
    if (rd((size_t)8, uint32_t{}) != 2) return r;  // the writer emits version 2
    if (rd(256, uint64_t{}) != strix_hash64(h, 256)) return r;
    r.kind = rd(12, uint32_t{}), r.tokens = rd(16, uint64_t{}), r.state_bytes = rd(24, uint64_t{});
    r.state_at = rd(32, uint64_t{}), r.base_n = rd(248, uint64_t{});
    if (rd(40, uint64_t{}) != strix_hash64(h + kHeader, (size_t)r.tokens * 4)) return r;
    if (r.state_at != align4k(kHeader + r.tokens * 4) || b.size() - off < r.state_at + r.state_bytes) return r;
    if (rd(48, uint64_t{}) != strix_hash64(h + r.state_at, (size_t)r.state_bytes)) return r;
    r.ok = true;
    return r;
}

std::vector<int32_t> make_tokens(int64_t n, int32_t seed) {
    std::vector<int32_t> t((size_t)n);
    for (int64_t i = 0; i < n; ++i) t[(size_t)i] = seed + (int32_t)(i * 7 + 11) % 100000;
    return t;
}

HostBuffer make_state(uint64_t seed, size_t bytes) {
    HostBuffer b(bytes);
    for (size_t i = 0; i < bytes; ++i) b.data()[i] = (uint8_t)((seed * 1103515245ull + i * 2654435761ull) >> 13);
    return b;
}

bool state_equals(const HostBuffer &b, uint64_t seed, size_t bytes) {
    if (b.size() != bytes) return false;
    for (size_t i = 0; i < bytes; ++i)
        if (b.data()[i] != (uint8_t)((seed * 1103515245ull + i * 2654435761ull) >> 13)) return false;
    return true;
}

PromptCache::Options quiet_options() {  // fake the memory probes: the checks are about bytes, not about this box
    PromptCache::Options o;
    o.mem_available = [] { return 100ull << 30; };
    o.disk_free = [] { return 100ull << 30; };
    o.prefault = false;
    return o;
}

std::string g_dir;  // the scratch dir
PromptCache *new_cache(const char *name, const std::string &fingerprint) {
    return new PromptCache(g_dir + "/" + name, 256ull << 20, fingerprint, quiet_options());
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: cache_xfer_check <scratch_dir>\n");
        return 2;
    }
    g_dir = argv[1];
    fs::create_directories(g_dir);
    const std::string fp = "xfer-check-fingerprint";

    PromptCache *a = new_cache("a", fp);

    // --- 1. whole entry round trip ---
    const std::vector<int32_t> t48 = make_tokens(48, 1000);
    a->put(t48, PromptCache::Kind::Checkpoint, make_state(7, 8192));
    std::string blob;
    const int64_t n1 = a->export_for(t48, &blob);
    check(n1 == 48 && blob.size() == file_total(48, 8192), "export answers the whole entry's bytes");
    const FileHead f1 = check_file(blob, 0);
    check(f1.ok && f1.tokens == 48 && f1.kind == 3 && f1.base_n == 0 && f1.state_bytes == 8192,
          "the file header names 48 checkpoint tokens, whole, with intact hashes");
    PromptCache *b = new_cache("b", fp);
    check(b->import_entries(blob) == PromptCache::ImportStatus::Ok, "a fresh server imports the body");
    check(b->best(t48, 48) == 48 && b->contains(t48.data(), 48), "the imported entry answers the prompt");
    check(b->in_ram(t48.data(), 48), "it sits in the RAM tier (not only on disk)");
    bool same = false;
    b->with_state(t48, 48, [&](const HostBuffer &st) { same = state_equals(st, 7, 8192); });
    check(same, "the state bytes come back identical");
    check(b->import_entries(blob) == PromptCache::ImportStatus::Ok, "importing what is already there is Ok (a refresh)");

    // --- 2. a delta: base first, then itself; imports as a pair ---
    // (delta_base's rule: a delta must stay a quarter of its base or smaller - so 8 tokens on a 64-token base.)
    std::vector<int32_t> t72 = make_tokens(72, 1000);  // the same conversation, 8 tokens past the new checkpoint
    a->put(std::vector<int32_t>(t72.begin(), t72.begin() + 64), PromptCache::Kind::Checkpoint, make_state(8, 6144));
    for (size_t i = 64; i < 72; ++i) t72[i] = 500000 + (int32_t)i;
    const int64_t base_n = a->delta_base(t72.data(), 72, PromptCache::Kind::Turn);
    check(base_n == 64, "the 8-token turn deltas on the 64-token base");
    check(a->put(t72, PromptCache::Kind::Turn, make_state(9, 4096), false, base_n), "the delta saves on its base");
    std::string pair;
    const int64_t n2 = a->export_for(t72, &pair);
    const FileHead g1 = check_file(pair, 0);
    const FileHead g2 = g1.ok ? check_file(pair, (size_t)(g1.state_at + g1.state_bytes)) : FileHead{};
    check(n2 == 72 && g1.ok && g1.tokens == 64 && g2.ok && g2.tokens == 72 && g2.base_n == 64 &&
              pair.size() == (g1.state_at + g1.state_bytes) + (g2.state_at + g2.state_bytes),
          "the delta body carries the base's file first, then its own");
    PromptCache *c = new_cache("c", fp);
    check(c->import_entries(pair) == PromptCache::ImportStatus::Ok, "a fresh server imports the pair");
    check(c->best(t72, 72) == 72 && c->in_ram(t72.data(), 72), "and resumes the full 72 tokens from it");

    // --- 3. foreign fingerprint: refused, nothing in ---
    PromptCache *d = new_cache("d", "some-other-server");
    check(d->import_entries(blob) == PromptCache::ImportStatus::Foreign, "a foreign fingerprint is refused (409)");
    check(d->best(t48, 48) == 0, "and puts nothing");

    // --- 4. damaged: a flipped byte, wherever it lands; and a cut body ---
    PromptCache *e = new_cache("e", fp);
    std::string bad = blob;
    bad[(size_t)(align4k(kHeader + 48 * 4) + 100)] ^= 0x40;  // a state byte
    check(e->import_entries(bad) == PromptCache::ImportStatus::Damaged, "a flipped state byte is Damaged (400)");
    bad = blob;
    bad[kHeader + 7] ^= 0x01;  // a token id
    check(e->import_entries(bad) == PromptCache::ImportStatus::Damaged, "a flipped token byte is Damaged (400)");
    check(e->import_entries(blob.substr(0, blob.size() - 4000)) == PromptCache::ImportStatus::Damaged,
          "a body cut mid-state is Damaged (400)");
    check(e->best(t48, 48) == 0, "none of them put anything");

    // --- 5. a delta whose base is missing ---
    const std::string delta_only = pair.substr((size_t)(g1.state_at + g1.state_bytes));
    PromptCache *f = new_cache("f", fp);
    check(f->import_entries(delta_only) == PromptCache::ImportStatus::BaseMissing,
          "a delta without its base answers BaseMissing");
    check(f->best(t72, 72) == 0 && f->best(std::vector<int32_t>(t72.begin(), t72.begin() + 64), 64) == 0,
          "and put neither the delta nor a base");
    check(b->import_entries(pair) == PromptCache::ImportStatus::Ok &&
              b->import_entries(delta_only) == PromptCache::ImportStatus::Ok,
          "the same delta goes into a server that has its base (an earlier body)");

    // --- 6. exporting what nothing covers ---
    std::string out2 = "keep";
    check(a->export_for(make_tokens(32, 777777), &out2) == 0 && out2 == "keep",
          "a prompt no entry covers exports 0 and leaves the buffer");

    // --- 7. the shapes the serving path produces (a4a320429d's e2e): a RAM-only one-shot entry, and an
    // export body longer than any saved entry (the engine serves to the prompt's end; checkpoints stop short).
    const std::vector<int32_t> t30 = make_tokens(30, 900000);
    check(a->put(t30, PromptCache::Kind::Turn, make_state(11, 4096), /*ram_only=*/true), "a RAM-only entry saves");
    std::string oneshot;
    const int64_t n7 = a->export_for(t30, &oneshot);
    const FileHead h7 = check_file(oneshot, 0);
    check(n7 == 30 && h7.ok && h7.tokens == 30 && h7.base_n == 0, "a RAM-only one-shot entry exports too");
    PromptCache *g = new_cache("g", fp);
    check(g->import_entries(oneshot) == PromptCache::ImportStatus::Ok && g->in_ram(t30.data(), 30),
          "a fresh server imports it and can resume from it");
    std::vector<int32_t> t96 = t72;
    for (int32_t i = 0; i < 24; ++i) t96.push_back(600000 + i);
    std::string lower;
    const int64_t n8 = a->export_for(t96, &lower);
    const FileHead k1 = check_file(lower, 0);
    const FileHead k2 = k1.ok ? check_file(lower, (size_t)(k1.state_at + k1.state_bytes)) : FileHead{};
    check(n8 == 72 && k1.ok && k1.tokens == 64 && k2.ok && k2.tokens == 72 && k2.base_n == 64,
          "a prompt longer than any entry exports the base+delta pair below its end");
    // (A delta whose base vanished on its own cannot exist: remove_locked takes a base's deltas with it. The
    // export_for fallback loop covers the race between best() and the base's removal - only a live server hits it,
    // so the loop keeps the longest entry's result identical to before, which cases 1-2 re-prove every run.)

    delete a; delete b; delete c; delete d; delete e; delete f; delete g;
    return failures;
}
