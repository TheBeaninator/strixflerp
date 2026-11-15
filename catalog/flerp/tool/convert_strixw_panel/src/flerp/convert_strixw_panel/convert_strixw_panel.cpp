// convert_strixw_panel: copy a strixw (v1..v4) to a strixw v5 with chosen Q4/Q8 classes re-laid out as
// Q4Panel16 / Q8Panel16 (format.flerp.q4panel / q8panel; docs/STRIXW-LAYOUT-NEXT.md section 2 candidate A).
// Order-only: no checkpoint is read and nothing is re-quantized.
//
//   convert_strixw_panel --old OLD --out NEW [--classes a,b,..]
//   convert_strixw_panel --old OLD --plan [--classes a,b,..] [--scratch DIR]
//   convert_strixw_panel --old OLD --check-tensors NAME[,NAME..]
//
// Streaming: one tensor at a time, OLD read with large O_DIRECT preads when the filesystem allows (page cache
// dropped behind otherwise). Every component read from OLD is hashed and compared with OLD's index before it is
// used. Copied tensors keep their encoding and bytes; converted tensors become one Q component. Offsets, hashes,
// index and header come from the v5 StrixwWriter (format.flerp.strixw_v5); metadata is OLD's with only `layout`
// changed (`<class>=<fmt>p16` for each converted class), so source_fingerprint (vision / n-gram pairing) is kept
// and the layout hash (prompt-cache key) changes.
//
// After finish(): the output is reopened with the v5 StrixwFile (header, index, every record), then one pass reads
// every output component, checks its hash against the index, and for converted tensors un-relayouts the panels and
// compares them byte for byte with OLD's q / scale / min (re-read from OLD); copied components are compared with
// OLD's bytes too. A failed run deletes its output.
//
// --plan: OLD's header and index only; plans the output with the writer on a sparse scratch file (deleted again)
// and prints per-class sizes, panel geometry and every tensor's new offsets.
// --check-tensors: reads only the named tensors' byte ranges of OLD, relayouts and inverts them in memory and
// compares (gate on a real file without writing one).

#include "common/check.hpp"
#include "flerp/q4panel/q4panel.hpp"
#include "formats/strixw.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strix;

namespace {

bool created_out = false;

// The design's recommendation (STRIXW-LAYOUT-NEXT.md section 4): every Q4/Q8 class but the HC mixes and BF16.
const char *const kDefaultClasses = "exp_gu,exp_down,sh_gu,sh_down,gdn_in,gdn_out,attn_qkv,attn_o,ple_kv,lm_head,mtp_fc";
// May be converted on request (A optional for mix_down, section 4); hc_up stays chunk-major, embed BF16.
const std::set<std::string> kConvertible = {"exp_gu", "exp_down", "sh_gu",  "sh_down", "gdn_in",  "gdn_out",
                                            "attn_qkv", "attn_o", "ple_kv", "lm_head", "mtp_fc", "hc_down"};

bool ends_with(const std::string &s, const std::string &x) {
    return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0;
}

// The layout class a tensor was planned under (convert_qwen4exp.cpp Planner::layer / make_plan), "" for the
// BF16 / F32 / I64 tensors no class names (router, norms, conv, ...).
std::string tensor_class(const std::string &n) {
    if (ends_with(n, ".mlp.experts_gate_up")) return "exp_gu";
    if (ends_with(n, ".mlp.experts_down")) return "exp_down";
    if (ends_with(n, ".mlp.shared_expert.gate_up")) return "sh_gu";
    if (ends_with(n, ".mlp.shared_expert.down_proj.weight")) return "sh_down";
    if (ends_with(n, ".linear_attn.in_proj")) return "gdn_in";
    if (ends_with(n, ".linear_attn.out_proj.weight")) return "gdn_out";
    if (ends_with(n, ".self_attn.qkv")) return "attn_qkv";
    if (ends_with(n, ".self_attn.o_proj.weight")) return "attn_o";
    if (ends_with(n, "hyper_connection.mix_down") || ends_with(n, "hyper_connection_mixer.mix_down")) return "hc_down";
    if (ends_with(n, "hyper_connection.mix_up") || ends_with(n, "hyper_connection_mixer.mix_up")) return "hc_up";
    if (ends_with(n, ".ple.kv_proj")) return "ple_kv";
    if (n == "lm_head.weight") return "lm_head";
    if (n == "mtp.fc_embedding.weight" || n == "mtp.fc_hidden.weight") return "mtp_fc";
    if (ends_with(n, "embed_tokens.weight")) return "embed";
    return "";
}

std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> r;
    size_t at = 0;
    while (at <= s.size()) {
        const size_t c = s.find(',', at);
        const std::string p = s.substr(at, c == std::string::npos ? std::string::npos : c - at);
        if (!p.empty()) r.push_back(p);
        if (c == std::string::npos) break;
        at = c + 1;
    }
    return r;
}

std::string layout_get(const std::string &layout, const std::string &cls) {
    for (const std::string &part : split(layout))
        if (part.rfind(cls + "=", 0) == 0) return part.substr(cls.size() + 1);
    return "";
}

std::string layout_set(const std::string &layout, const std::string &cls, const std::string &fmt) {
    std::string r;
    bool done = false;
    for (const std::string &part : split(layout)) {
        r += r.empty() ? "" : ",";
        if (part.rfind(cls + "=", 0) == 0) r += cls + "=" + fmt, done = true;
        else r += part;
    }
    STRIX_CHECK(done, "convert_strixw_panel: layout has no ", cls);
    return r;
}

std::string enc_fmt(const StrixwTensor &t) {
    switch (t.encoding) {
    case StrixwEncoding::Q4RowMajor: return "q4g" + std::to_string(t.group_size);
    case StrixwEncoding::Q5RowMajor: return "q5g" + std::to_string(t.group_size);
    case StrixwEncoding::Q6RowMajor: return "q6g" + std::to_string(t.group_size);
    case StrixwEncoding::Q8RowMajor: return "q8g" + std::to_string(t.group_size);
    case StrixwEncoding::Q4Panel16: return "q4g" + std::to_string(t.group_size) + "p16";
    case StrixwEncoding::Q8Panel16: return "q8g" + std::to_string(t.group_size) + "p16";
    default: return strixw_encoding_name(t.encoding);
    }
}

uint32_t header_version(const std::string &path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    STRIX_CHECK(fd >= 0, "convert_strixw_panel: open '", path, "': ", std::strerror(errno));
    uint32_t v = 0;
    const ssize_t r = ::pread(fd, &v, 4, 8);
    ::close(fd);
    STRIX_CHECK(r == 4, "convert_strixw_panel: short header read of '", path, "'");
    return v;
}

// ---------------------------------------------------------------- byte ranges (tool.flerp.strixw_patch's reader)

class RangeReader {
public:
    explicit RangeReader(const std::string &path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        direct_ = fd_ >= 0;
        if (!direct_) fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        STRIX_CHECK(fd_ >= 0, "convert_strixw_panel: open '", path, "': ", std::strerror(errno));
    }
    ~RangeReader() {
        if (fd_ >= 0) ::close(fd_);
        std::free(buf_);
    }
    bool direct() const { return direct_; }
    // Bytes [off, off + n); the pointer is valid until the next read.
    const uint8_t *read(uint64_t off, uint64_t n) {
        const uint64_t a0 = off / kAlign * kAlign, a1 = (off + n + kAlign - 1) / kAlign * kAlign;
        reserve(a1 - a0);
        uint64_t at = a0, got = 0;
        while (at < a1) {
            const size_t chunk = (size_t)std::min<uint64_t>(a1 - at, 64ull << 20);
            ssize_t r = ::pread(fd_, buf_ + (at - a0), chunk, (off_t)at);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0 && direct_ && errno == EINVAL) {
                ::close(fd_);
                fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
                STRIX_CHECK(fd_ >= 0, "convert_strixw_panel: reopen '", path_, "': ", std::strerror(errno));
                direct_ = false;
                continue;
            }
            STRIX_CHECK(r >= 0, "convert_strixw_panel: pread '", path_, "' at ", at, ": ", std::strerror(errno));
            if (r == 0) break;
            at += (uint64_t)r, got = at - a0;
        }
        STRIX_CHECK(got >= off + n - a0, "convert_strixw_panel: '", path_, "' ends inside bytes [", off, ", ", off + n, ")");
        if (!direct_) (void)::posix_fadvise(fd_, (off_t)a0, (off_t)(a1 - a0), POSIX_FADV_DONTNEED);
        return buf_ + (off - a0);
    }
    // The component's bytes, checked against the index hash.
    const uint8_t *read_checked(const StrixwTensor &t, const StrixwComponent &c) {
        const uint8_t *d = read(c.offset, c.bytes);
        const uint64_t h = strix_hash64(d, c.bytes);
        STRIX_CHECK(h == c.hash, "convert_strixw_panel: '", path_, "' tensor '", t.name, "' ", strixw_role_name(c.role),
                    " hashes to ", h, ", its index says ", c.hash, " - the file is corrupt");
        return d;
    }

private:
    static constexpr uint64_t kAlign = 4096;
    void reserve(uint64_t n) {
        if (n <= cap_) return;
        std::free(buf_);
        buf_ = static_cast<uint8_t *>(std::aligned_alloc(kAlign, n));
        STRIX_CHECK(buf_ != nullptr, "convert_strixw_panel: allocating ", n, " bytes failed");
        cap_ = n;
    }
    std::string path_;
    int fd_ = -1;
    bool direct_ = false;
    uint8_t *buf_ = nullptr;
    uint64_t cap_ = 0;
};

const StrixwComponent &comp(const StrixwTensor &t, StrixwRole r) {
    for (const StrixwComponent &c : t.components)
        if (c.role == r) return c;
    STRIX_CHECK(false, "convert_strixw_panel: '", t.name, "' has no ", strixw_role_name(r));
    return t.components[0];
}

flerp::Panel16Geom geom_of(const StrixwTensor &t) {
    const int bits = (t.encoding == StrixwEncoding::Q4RowMajor || t.encoding == StrixwEncoding::Q4Panel16) ? 4 : 8;
    return flerp::panel16_geom(t.shape.at(0), t.shape.at(1), t.group_size, bits, t.name.c_str());
}

// Row-major planes of OLD's tensor t, read and hash-checked. q, scale, min copied into the vectors.
struct Planes {
    std::vector<uint8_t> q, s, m;
};
Planes read_planes(RangeReader &rr, const StrixwTensor &t) {
    Planes p;
    const StrixwComponent &cs = comp(t, StrixwRole::Scale), &cm = comp(t, StrixwRole::Min), &cq = comp(t, StrixwRole::Q);
    const uint8_t *d = rr.read_checked(t, cs);
    p.s.assign(d, d + cs.bytes);
    d = rr.read_checked(t, cm);
    p.m.assign(d, d + cm.bytes);
    d = rr.read_checked(t, cq);
    p.q.assign(d, d + cq.bytes);
    return p;
}

// ---------------------------------------------------------------- the plan

struct Plan {
    std::map<std::string, std::string> meta;
    std::vector<StrixwTensor> tensors;  // records for the writer
    std::vector<bool> convert;          // per tensor
    std::vector<std::string> cls;       // per tensor
    std::vector<std::string> classes;   // converted classes, in --classes order
};

Plan make_plan(const StrixwFile &old, const std::string &old_path, const std::string &classes_arg) {
    const uint32_t v = header_version(old_path);
    STRIX_CHECK(v >= 1 && v <= 4, "convert_strixw_panel: '", old_path, "' is strixw v", v,
                " - expected a v1..v4 file (no panel encodings) - refused");
    STRIX_CHECK(old.meta().count("layout") == 1, "convert_strixw_panel: '", old_path, "' has no layout metadata");
    Plan p;
    p.meta = old.meta();
    std::string layout = p.meta.at("layout");
    std::set<std::string> want;
    for (const std::string &c : split(classes_arg)) {
        STRIX_CHECK(kConvertible.count(c) == 1, "convert_strixw_panel: class '", c, "' refused (convertible: exp_gu ",
                    "exp_down sh_gu sh_down gdn_in gdn_out attn_qkv attn_o ple_kv lm_head mtp_fc hc_down; hc_up stays ",
                    "chunk-major, embed and the router stay BF16)");
        STRIX_CHECK(want.insert(c).second, "convert_strixw_panel: class '", c, "' given twice - refused");
        const std::string f = layout_get(layout, c);
        STRIX_CHECK(f.size() >= 4 && (f.rfind("q4g", 0) == 0 || f.rfind("q8g", 0) == 0) &&
                        f.find('p') == std::string::npos,
                    "convert_strixw_panel: layout says ", c, "=", f.empty() ? "(absent)" : f,
                    ", expected row-major q4g<G> or q8g<G> - refused");
        layout = layout_set(layout, c, f + "p16");
        p.classes.push_back(c);
    }
    STRIX_CHECK(!p.classes.empty(), "convert_strixw_panel: no classes to convert");
    p.meta["layout"] = layout;
    std::map<std::string, int> found;
    for (const StrixwTensor &t : old.tensors()) {
        StrixwTensor n;
        n.name = t.name, n.encoding = t.encoding, n.shape = t.shape, n.group_size = t.group_size;
        n.experts = t.experts, n.parts = t.parts;
        const std::string c = tensor_class(t.name);
        const bool conv = want.count(c) == 1;
        if (conv) {
            const std::string f = layout_get(old.meta("layout"), c);
            STRIX_CHECK(enc_fmt(t) == f, "convert_strixw_panel: '", t.name, "' is ", enc_fmt(t), ", the layout says ", c,
                        "=", f, " - refused");
            STRIX_CHECK(t.encoding == StrixwEncoding::Q4RowMajor || t.encoding == StrixwEncoding::Q8RowMajor,
                        "convert_strixw_panel: '", t.name, "' is ", strixw_encoding_name(t.encoding));
            n.encoding = t.encoding == StrixwEncoding::Q4RowMajor ? StrixwEncoding::Q4Panel16 : StrixwEncoding::Q8Panel16;
            geom_of(t);  // refuses N % 16, K % 32, K % G here, before anything is written
            ++found[c];
        }
        p.tensors.push_back(std::move(n));
        p.convert.push_back(conv);
        p.cls.push_back(c);
    }
    for (const std::string &c : p.classes)
        STRIX_CHECK(found[c] >= 1, "convert_strixw_panel: no tensor of class ", c, " in '", old_path, "' - refused");
    return p;
}

std::string scratch_path(const std::string &dir) {
    return (fs::path(dir) / ("convert_strixw_panel_plan." + std::to_string(::getpid()) + ".strixw")).string();
}

// ---------------------------------------------------------------- modes

int run_plan(const std::string &old_path, const std::string &classes, const std::string &scratch) {
    StrixwFile old(old_path);
    const Plan p = make_plan(old, old_path, classes);
    const std::string tmp = scratch_path(scratch);
    std::vector<StrixwTensor> planned;
    uint64_t bytes = 0;
    try {
        StrixwWriter w(tmp, p.meta, p.tensors);  // plans in its constructor; writes header + index only (sparse)
        planned = w.tensors(), bytes = w.file_bytes();
    } catch (...) {
        fs::remove(tmp);
        throw;
    }
    fs::remove(tmp);
    std::printf("old %s v%u\n", old_path.c_str(), header_version(old_path));
    std::printf("layout_old %s\nlayout_new %s\n", old.meta("layout").c_str(), p.meta.at("layout").c_str());
    std::printf("source_fingerprint %s (kept)\n", p.meta.count("source_fingerprint") ? p.meta.at("source_fingerprint").c_str() : "(none)");
    std::printf("old_bytes %llu new_bytes %llu delta %lld data_offset %llu tensors %zu\n",
                (unsigned long long)old.file_bytes(), (unsigned long long)bytes,
                (long long)bytes - (long long)old.file_bytes(),
                (unsigned long long)planned.front().components.front().offset, planned.size());
    // per class
    struct Agg {
        int n = 0;
        uint64_t old_b = 0, new_b = 0;
        std::set<std::string> geoms;
    };
    std::map<std::string, Agg> agg;
    uint64_t conv_b = 0;
    for (size_t i = 0; i < planned.size(); ++i) {
        const std::string c = p.convert[i] ? p.cls[i] : "(copied)";
        Agg &a = agg[c];
        ++a.n, a.old_b += old.tensors()[i].bytes(), a.new_b += planned[i].bytes();
        if (p.convert[i]) {
            conv_b += planned[i].bytes();
            const flerp::Panel16Geom g = geom_of(planned[i]);
            char s[256];
            std::snprintf(s, sizeof s, "N=%lld K=%lld G=%lld panels=%lld panel_bytes=%llu (codes %llu + s/m %llu) %s",
                          (long long)g.N, (long long)g.K, (long long)g.G, (long long)g.panels,
                          (unsigned long long)g.panel_bytes, (unsigned long long)g.codes_bytes,
                          (unsigned long long)g.sm_bytes,
                          g.panel_bytes % 128 == 0 && g.codes_bytes % 128 == 0 ? "128-aligned" : "NOT-128-aligned");
            a.geoms.insert(s);
        }
    }
    for (const auto &[c, a] : agg) {
        std::printf("class %-9s tensors %3d old %llu new %llu\n", c.c_str(), a.n, (unsigned long long)a.old_b,
                    (unsigned long long)a.new_b);
        for (const std::string &g : a.geoms) std::printf("    %s\n", g.c_str());
    }
    std::printf("converted_bytes %llu\n", (unsigned long long)conv_b);
    for (size_t i = 0; i < planned.size(); ++i) {
        const StrixwTensor &t = planned[i], &ot = old.tensors()[i];
        std::printf("%s %s %s", p.convert[i] ? "PANEL" : "copy ", t.name.c_str(), enc_fmt(t).c_str());
        for (const StrixwComponent &c : t.components)
            std::printf(" %s@%llu+%llu", strixw_role_name(c.role), (unsigned long long)c.offset,
                        (unsigned long long)c.bytes);
        std::printf(" (old@%llu)\n", (unsigned long long)ot.components[0].offset);
    }
    return 0;
}

int run_check(const std::string &old_path, const std::string &names) {
    StrixwFile old(old_path);
    RangeReader rr(old_path);
    int bad = 0;
    for (const std::string &name : split(names)) {
        const StrixwTensor &t = old.get(name);
        STRIX_CHECK(t.encoding == StrixwEncoding::Q4RowMajor || t.encoding == StrixwEncoding::Q8RowMajor,
                    "convert_strixw_panel: '", name, "' is ", strixw_encoding_name(t.encoding), ", not row-major Q4/Q8");
        const auto t0 = std::chrono::steady_clock::now();
        const Planes pl = read_planes(rr, t);  // hash-checked against OLD's index
        const flerp::Panel16Geom g = geom_of(t);
        std::vector<uint8_t> panels(g.total_bytes);
        flerp::panel16_pack(g, pl.q.data(), pl.s.data(), pl.m.data(), panels.data());
        // The offsets the design states, checked on every row's first and last chunk and group.
        uint64_t off_bad = 0;
        for (int64_t n = 0; n < g.N; ++n)
            for (int64_t c : {(int64_t)0, g.chunks - 1}) {
                if (std::memcmp(panels.data() + flerp::panel16_code_offset(g, n, c),
                                pl.q.data() + (uint64_t)n * g.row_code_bytes + c * 16, 16) != 0)
                    ++off_bad;
                const int64_t gr = c == 0 ? 0 : g.groups - 1;
                const uint8_t *sm = panels.data() + flerp::panel16_sm_offset(g, n, gr);
                if (std::memcmp(sm, pl.s.data() + (n * g.groups + gr) * 2, 2) ||
                    std::memcmp(sm + 2, pl.m.data() + (n * g.groups + gr) * 2, 2))
                    ++off_bad;
            }
        std::vector<uint8_t> q(pl.q.size()), s(pl.s.size()), m(pl.m.size());
        flerp::panel16_unpack(g, panels.data(), q.data(), s.data(), m.data());
        const bool same = q == pl.q && s == pl.s && m == pl.m;
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%s %s [%lld,%lld] experts %lld: %llu panels x %llu B (%s), total %llu B = q+s+m %llu B; "
                    "offsets %s; round trip %s; %.1f s\n",
                    name.c_str(), enc_fmt(t).c_str(), (long long)g.N, (long long)g.K, (long long)t.experts,
                    (unsigned long long)g.panels, (unsigned long long)g.panel_bytes,
                    g.panel_bytes % 128 == 0 && g.codes_bytes % 128 == 0 ? "128-aligned" : "NOT 128-aligned",
                    (unsigned long long)g.total_bytes, (unsigned long long)(pl.q.size() + pl.s.size() + pl.m.size()),
                    off_bad ? "WRONG" : "as designed", same ? "IDENTICAL" : "DIFFERENT", sec);
        std::fflush(stdout);
        if (!same || off_bad || g.total_bytes != pl.q.size() + pl.s.size() + pl.m.size()) ++bad;
    }
    std::printf("%s\n", bad ? "RESULT: FAILED" : "RESULT: every tensor round-trips byte-identical");
    return bad ? 1 : 0;
}

int run_convert(const std::string &old_path, const std::string &classes, const std::string &out) {
    STRIX_CHECK(!fs::exists(out), "convert_strixw_panel: '", out, "' exists - refusing to overwrite");
    STRIX_CHECK(fs::absolute(out) != fs::absolute(old_path), "convert_strixw_panel: --out is --old");
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    StrixwFile old(old_path);
    const Plan p = make_plan(old, old_path, classes);
    RangeReader rr(old_path);
    uint64_t moved = 0;
    {
        created_out = true;
        StrixwWriter w(out, p.meta, p.tensors);
        std::printf("writing %s: %llu bytes (OLD %llu), %zu tensors, layout %s, reads %s\n", out.c_str(),
                    (unsigned long long)w.file_bytes(), (unsigned long long)old.file_bytes(), p.tensors.size(),
                    p.meta.at("layout").c_str(), rr.direct() ? "O_DIRECT" : "buffered");
        std::vector<uint8_t> panels;
        for (size_t ti = 0; ti < p.tensors.size(); ++ti) {
            const StrixwTensor &ot = old.tensors()[ti];
            STRIX_CHECK(w.tensors()[ti].name == ot.name, "convert_strixw_panel: tensor order changed at ", ti);
            if (p.convert[ti]) {
                const flerp::Panel16Geom g = geom_of(ot);
                const StrixwComponent &cs = comp(ot, StrixwRole::Scale), &cm = comp(ot, StrixwRole::Min),
                                      &cq = comp(ot, StrixwRole::Q);
                const uint8_t *d = rr.read_checked(ot, cs);
                const std::vector<uint8_t> s(d, d + cs.bytes);
                d = rr.read_checked(ot, cm);
                const std::vector<uint8_t> m(d, d + cm.bytes);
                const uint8_t *q = rr.read_checked(ot, cq);  // valid until the next read
                panels.assign(g.total_bytes, 0);
                flerp::panel16_pack(g, q, s.data(), m.data(), panels.data());
                w.write(ti, StrixwRole::Q, panels.data(), panels.size());
                moved += panels.size();
                std::vector<uint8_t>().swap(panels);
            } else {
                STRIX_CHECK(w.tensors()[ti].components.size() == ot.components.size(), "convert_strixw_panel: '",
                            ot.name, "' component count changed");
                for (size_t ci = 0; ci < ot.components.size(); ++ci) {
                    const StrixwComponent &c = ot.components[ci];
                    const uint8_t *d = rr.read_checked(ot, c);
                    w.write(ti, c.role, d, c.bytes);
                    STRIX_CHECK(w.tensors()[ti].components[ci].hash == c.hash, "convert_strixw_panel: '", ot.name,
                                "' copy hash");
                    moved += c.bytes;
                }
            }
            if (ti % 32 == 0) {
                std::printf("  %zu/%zu tensors, %.1f GB written, %.0f s\n", ti, p.tensors.size(), moved / 1e9, secs());
                std::fflush(stdout);
            }
        }
        w.finish();
    }
    std::printf("written in %.0f s; verifying\n", secs());
    std::fflush(stdout);

    // 1. The v5 reader: header, index, every record and component size.
    StrixwFile neu(out);
    STRIX_CHECK(neu.meta() == p.meta, "convert_strixw_panel: output metadata differs from the plan");
    STRIX_CHECK(neu.tensors().size() == old.tensors().size(), "convert_strixw_panel: output tensor count");
    // 2. One pass over the output: every component's hash; converted tensors inverted and compared with OLD.
    RangeReader rn(out);
    uint64_t checked = 0;
    for (size_t ti = 0; ti < old.tensors().size(); ++ti) {
        const StrixwTensor &a = old.tensors()[ti], &b = neu.tensors()[ti];
        STRIX_CHECK(a.name == b.name && a.shape == b.shape && a.experts == b.experts && a.group_size == b.group_size &&
                        a.parts.size() == b.parts.size(),
                    "convert_strixw_panel: output record ", ti, " '", b.name, "' differs from OLD's");
        if (p.convert[ti]) {
            STRIX_CHECK(b.encoding == (a.encoding == StrixwEncoding::Q4RowMajor ? StrixwEncoding::Q4Panel16
                                                                                 : StrixwEncoding::Q8Panel16),
                        "convert_strixw_panel: '", b.name, "' encoding");
            const flerp::Panel16Geom g = geom_of(b);
            const uint8_t *panels = rn.read_checked(b, comp(b, StrixwRole::Q));
            std::vector<uint8_t> q((size_t)(g.N * g.row_code_bytes)), s((size_t)(g.N * g.groups * 2)), m(s.size());
            flerp::panel16_unpack(g, panels, q.data(), s.data(), m.data());
            const Planes pl = read_planes(rr, a);
            STRIX_CHECK(q == pl.q && s == pl.s && m == pl.m, "convert_strixw_panel: '", b.name,
                        "' panels do not invert to OLD's planes");
            checked += g.total_bytes;
        } else {
            STRIX_CHECK(a.encoding == b.encoding, "convert_strixw_panel: '", b.name, "' encoding changed");
            for (size_t ci = 0; ci < a.components.size(); ++ci) {
                const StrixwComponent &ca = a.components[ci], &cb = b.components[ci];
                STRIX_CHECK(ca.role == cb.role && ca.bytes == cb.bytes && ca.hash == cb.hash, "convert_strixw_panel: '",
                            b.name, "' component ", ci, " differs from OLD's in the index");
                const uint8_t *d = rn.read_checked(b, cb);
                const std::vector<uint8_t> mine(d, d + cb.bytes);
                STRIX_CHECK(std::memcmp(mine.data(), rr.read(ca.offset, ca.bytes), ca.bytes) == 0,
                            "convert_strixw_panel: '", b.name, "' component ", ci, " bytes differ from OLD's");
                checked += cb.bytes;
            }
        }
        if (ti % 32 == 0) {
            std::printf("  verify %zu/%zu, %.1f GB, %.0f s\n", ti, old.tensors().size(), checked / 1e9, secs());
            std::fflush(stdout);
        }
    }
    std::printf("OK %s: v5, %llu bytes, layout %s, every component hash-checked, converted tensors invert to OLD "
                "byte for byte, %.0f s\n",
                out.c_str(), (unsigned long long)neu.file_bytes(), neu.meta("layout").c_str(), secs());
    return 0;
}

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: convert_strixw_panel --old OLD --out NEW [--classes a,b,..]\n"
                 "       convert_strixw_panel --old OLD --plan [--classes a,b,..] [--scratch DIR]\n"
                 "       convert_strixw_panel --old OLD --check-tensors NAME[,NAME..]\n"
                 "default --classes %s\n",
                 kDefaultClasses);
    std::exit(2);
}

}  // namespace

int main(int argc, char **argv) {
    std::string old_path, out, check, classes = kDefaultClasses, scratch = fs::temp_directory_path().string();
    bool plan = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (a == "--old") old_path = val();
        else if (a == "--out") out = val();
        else if (a == "--classes") classes = val();
        else if (a == "--check-tensors") check = val();
        else if (a == "--scratch") scratch = val();
        else if (a == "--plan") plan = true;
        else usage();
    }
    try {
        if (old_path.empty() || (int)plan + (int)!out.empty() + (int)!check.empty() != 1) usage();
        if (!check.empty()) return run_check(old_path, check);
        if (plan) return run_plan(old_path, classes, scratch);
        try {
            return run_convert(old_path, classes, out);
        } catch (...) {
            if (created_out) fs::remove(out);  // run_convert refuses an existing --out before creating anything
            throw;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "convert_strixw_panel: FAILED: %s\n", e.what());
        return 1;
    }
}
