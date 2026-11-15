// strixw_patch: copy a weights.strixw with ONE tensor class re-encoded from the source checkpoint.
//
// Only the declared override is accepted: mtp_fc=q4g<32|64|128> (the MTP head's two input projections,
// convert_qwen4exp.cpp make_plan: quant_single("mtp.fc_embedding.weight" / "mtp.fc_hidden.weight", "mtp_fc")).
// Every other tensor's components are copied byte for byte from OLD, one component at a time (large sequential
// O_DIRECT reads when the filesystem allows, page cache dropped behind). The two mtp_fc tensors are quantized from
// the checkpoint's BF16 tensors with formats/q4's quantize_q4, the function the converter's Q4 path calls
// (quantize_rows splits rows over threads; Q4 groups run along K inside a row, so one call over all rows gives the
// same bytes). Offsets, component hashes, index and header come from strixite's StrixwWriter; the result is
// re-opened with StrixwFile and every component hash checked before the tool reports success.
//
//   strixw_patch --old OLD --source CKPT_DIR --override mtp_fc=q4g64 --out NEW.strixw [--threads N]
//   strixw_patch --old OLD [--source CKPT_DIR] --override mtp_fc=q4g64 --plan [--scratch DIR]
//   strixw_patch --source CKPT_DIR --override mtp_fc=q4g64 --check-against REF.strixw
//
// --plan: header and index of OLD only; plans the output with StrixwWriter on a sparse scratch file (header +
// index bytes only, deleted again) and prints layout, size and every tensor's new offsets.
// --check-against: quantizes the class from the checkpoint and compares it, byte range by byte range, with
// REF's same-named components (reads only those ranges of REF).

#include "common/check.hpp"
#include "formats/convert_qwen4exp.hpp"
#include "formats/q4.hpp"
#include "formats/safetensors.hpp"
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
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strix;

namespace {

bool created_out = false;  // set once run_patch has created --out

const char *const kMtpFc[] = {"mtp.fc_embedding.weight", "mtp.fc_hidden.weight"};

struct Override {
    std::string cls;
    int64_t G = 0;
    std::string fmt;  // "q4g64"
};

Override parse_override(const std::string &s) {
    const size_t eq = s.find('=');
    STRIX_CHECK(eq != std::string::npos && s.find(',') == std::string::npos, "strixw_patch: --override '", s,
                "' refused: exactly one class=format");
    Override o;
    o.cls = s.substr(0, eq), o.fmt = s.substr(eq + 1);
    STRIX_CHECK(o.cls == "mtp_fc", "strixw_patch: --override class '", o.cls,
                "' refused: this tool re-encodes only mtp_fc (the MTP draft input projections)");
    STRIX_CHECK(o.fmt == "q4g32" || o.fmt == "q4g64" || o.fmt == "q4g128", "strixw_patch: --override format '",
                o.fmt, "' refused: only q4g32 / q4g64 / q4g128 (the converter's Q4 path)");
    o.G = std::stoll(o.fmt.substr(3));
    return o;
}

uint32_t header_version(const std::string &path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    STRIX_CHECK(fd >= 0, "strixw_patch: open '", path, "': ", std::strerror(errno));
    uint32_t v = 0;
    const ssize_t r = ::pread(fd, &v, 4, 8);
    ::close(fd);
    STRIX_CHECK(r == 4, "strixw_patch: short header read of '", path, "'");
    return v;
}

// "a=x,mtp_fc=q8g64,b=y" -> value of cls, or "".
std::string layout_get(const std::string &layout, const std::string &cls) {
    size_t at = 0;
    while (at <= layout.size()) {
        const size_t comma = layout.find(',', at);
        const std::string part = layout.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        if (part.rfind(cls + "=", 0) == 0) return part.substr(cls.size() + 1);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return "";
}

std::string layout_set(const std::string &layout, const std::string &cls, const std::string &fmt) {
    const std::string from = cls + "=" + layout_get(layout, cls);
    const size_t at = (layout.rfind(from + ",", 0) == 0 || layout == from) ? 0 : layout.find("," + from) + 1;
    return layout.substr(0, at) + cls + "=" + fmt + layout.substr(at + from.size());
}

std::string enc_fmt(const StrixwTensor &t) {
    switch (t.encoding) {
    case StrixwEncoding::Q4RowMajor: return "q4g" + std::to_string(t.group_size);
    case StrixwEncoding::Q5RowMajor: return "q5g" + std::to_string(t.group_size);
    case StrixwEncoding::Q6RowMajor: return "q6g" + std::to_string(t.group_size);
    case StrixwEncoding::Q8RowMajor: return "q8g" + std::to_string(t.group_size);
    default: return strixw_encoding_name(t.encoding);
    }
}

bool is_mtp_fc(const std::string &name) {
    return name == kMtpFc[0] || name == kMtpFc[1];
}

std::string shape_s(const std::vector<int64_t> &s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
    return r + "]";
}

// ---------------------------------------------------------------- the checkpoint side

class Checkpoint {
public:
    explicit Checkpoint(const std::string &dir) : dir_(dir) {
        STRIX_CHECK(fs::is_directory(dir), "strixw_patch: --source '", dir, "' is not a directory");
        std::vector<std::string> paths;
        for (const auto &e : fs::directory_iterator(dir))
            if (e.path().extension() == ".safetensors") paths.push_back(e.path().string());
        std::sort(paths.begin(), paths.end());
        for (const std::string &p : paths) files_.push_back(std::make_unique<SafetensorsFile>(p));
        STRIX_CHECK(!files_.empty(), "strixw_patch: no *.safetensors in '", dir, "'");
    }
    // The tensor as FP32 (BF16 widened exactly, like the converter's bf16_to_f32).
    std::vector<float> read_bf16(const std::string &name, const std::vector<int64_t> &shape) const {
        for (const auto &f : files_)
            if (const TensorInfo *ti = f->find(name)) {
                STRIX_CHECK(ti->dtype == Dtype::BF16 && ti->shape == shape, "strixw_patch: checkpoint '", name,
                            "' is ", dtype_name(ti->dtype), " ", shape_s(ti->shape), ", expected BF16 ",
                            shape_s(shape));
                const auto *b = static_cast<const uint16_t *>(f->data(*ti));
                std::vector<float> out((size_t)ti->numel());
                for (size_t i = 0; i < out.size(); ++i) {
                    const uint32_t u = (uint32_t)b[i] << 16;
                    std::memcpy(&out[i], &u, 4);
                }
                return out;
            }
        STRIX_CHECK(false, "strixw_patch: '", name, "' not in the checkpoint '", dir_, "'");
        return {};
    }

private:
    std::string dir_;
    std::vector<std::unique_ptr<SafetensorsFile>> files_;
};

Q4Weight quantize_class_tensor(const Checkpoint &ck, const StrixwTensor &t, int64_t G) {
    STRIX_CHECK(t.shape.size() == 2 && t.parts.empty() && t.experts == 0, "strixw_patch: '", t.name,
                "' is not a single [N, K] tensor (", t.parts.size(), " parts, ", t.experts, " experts)");
    const std::vector<float> f = ck.read_bf16(t.name, t.shape);
    return quantize_q4(f.data(), t.shape[0], t.shape[1], G);
}

const void *plane(const Q4Weight &q, StrixwRole r, size_t *bytes) {
    switch (r) {
    case StrixwRole::Q: *bytes = q.q.size(); return q.q.data();
    case StrixwRole::Scale: *bytes = q.scale.size() * 2; return q.scale.data();
    case StrixwRole::Min: *bytes = q.minv.size() * 2; return q.minv.data();
    default: STRIX_CHECK(false, "strixw_patch: Q4 has no ", strixw_role_name(r), " plane"); return nullptr;
    }
}

// ---------------------------------------------------------------- byte ranges of a file

class RangeReader {
public:
    explicit RangeReader(const std::string &path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        direct_ = fd_ >= 0;
        if (!direct_) fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        STRIX_CHECK(fd_ >= 0, "strixw_patch: open '", path, "': ", std::strerror(errno));
    }
    ~RangeReader() {
        if (fd_ >= 0) ::close(fd_);
        std::free(buf_);
    }
    bool direct() const { return direct_; }
    // Bytes [off, off + n) of the file; the pointer is valid until the next read.
    const uint8_t *read(uint64_t off, uint64_t n) {
        const uint64_t a0 = off / kAlign * kAlign, a1 = (off + n + kAlign - 1) / kAlign * kAlign;
        reserve(a1 - a0);
        uint64_t at = a0, got = 0;
        while (at < a1) {
            const size_t chunk = (size_t)std::min<uint64_t>(a1 - at, 64ull << 20);
            ssize_t r = ::pread(fd_, buf_ + (at - a0), chunk, (off_t)at);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0 && direct_ && errno == EINVAL) {  // filesystem refuses O_DIRECT reads after all
                ::close(fd_);
                fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
                STRIX_CHECK(fd_ >= 0, "strixw_patch: reopen '", path_, "': ", std::strerror(errno));
                direct_ = false;
                continue;
            }
            STRIX_CHECK(r >= 0, "strixw_patch: pread '", path_, "' at ", at, ": ", std::strerror(errno));
            if (r == 0) break;  // end of file (the last component need not end on a 4096 boundary)
            at += (uint64_t)r, got = at - a0;
        }
        STRIX_CHECK(got >= off + n - a0, "strixw_patch: '", path_, "' ends inside bytes [", off, ", ", off + n, ")");
        if (!direct_) (void)::posix_fadvise(fd_, (off_t)a0, (off_t)(a1 - a0), POSIX_FADV_DONTNEED);
        return buf_ + (off - a0);
    }

private:
    static constexpr uint64_t kAlign = 4096;
    void reserve(uint64_t n) {
        if (n <= cap_) return;
        std::free(buf_);
        buf_ = static_cast<uint8_t *>(std::aligned_alloc(kAlign, n));
        STRIX_CHECK(buf_ != nullptr, "strixw_patch: allocating ", n, " bytes failed");
        cap_ = n;
    }
    std::string path_;
    int fd_ = -1;
    bool direct_ = false;
    uint8_t *buf_ = nullptr;
    uint64_t cap_ = 0;
};

// ---------------------------------------------------------------- the patch plan

struct Patch {
    std::map<std::string, std::string> meta;
    std::vector<StrixwTensor> tensors;  // records for the writer (components recomputed by it)
    std::string old_fmt;
};

Patch make_patch(const StrixwFile &old, const std::string &old_path, const Override &o) {
    const uint32_t v = header_version(old_path);
    STRIX_CHECK(v == kStrixwVersion, "strixw_patch: '", old_path, "' is strixw v", v, "; StrixwWriter writes v",
                kStrixwVersion, ", so the header would not be the same - refused");
    STRIX_CHECK(old.meta().count("layout") == 1, "strixw_patch: '", old_path, "' has no layout metadata");
    STRIX_CHECK(old.meta().count("mtp_source") == 0, "strixw_patch: '", old_path,
                "' has a replaced draft head (mtp_source) - its mtp_fc does not come from the checkpoint - refused");
    Patch p;
    p.meta = old.meta();
    const std::string &layout = p.meta.at("layout");
    p.old_fmt = layout_get(layout, o.cls);
    STRIX_CHECK(!p.old_fmt.empty(), "strixw_patch: layout '", layout, "' has no ", o.cls);
    STRIX_CHECK(p.old_fmt != o.fmt, "strixw_patch: ", o.cls, " is already ", o.fmt, " - nothing to patch");
    p.meta["layout"] = layout_set(layout, o.cls, o.fmt);
    STRIX_CHECK(layout_get(p.meta["layout"], o.cls) == o.fmt, "strixw_patch: layout rewrite failed");
    int found = 0;
    for (const StrixwTensor &t : old.tensors()) {
        StrixwTensor n;
        n.name = t.name, n.encoding = t.encoding, n.shape = t.shape, n.group_size = t.group_size;
        n.experts = t.experts, n.parts = t.parts;
        if (is_mtp_fc(t.name)) {
            STRIX_CHECK(enc_fmt(t) == p.old_fmt, "strixw_patch: '", t.name, "' is ", enc_fmt(t), ", the layout says ",
                        o.cls, "=", p.old_fmt);
            STRIX_CHECK(t.parts.empty() && t.experts == 0 && t.shape.size() == 2, "strixw_patch: '", t.name,
                        "' is not a single-source [N, K] tensor");
            n.encoding = StrixwEncoding::Q4RowMajor, n.group_size = o.G;
            ++found;
        }
        p.tensors.push_back(std::move(n));
    }
    STRIX_CHECK(found == 2, "strixw_patch: found ", found, " of the 2 mtp_fc tensors in '", old_path, "'");
    return p;
}

void check_source(const StrixwFile &old, const std::string &src) {
    const std::string fp = checkpoint_fingerprint(src);
    const std::string &want = old.meta("source_fingerprint");
    STRIX_CHECK(fp == want, "strixw_patch: checkpoint '", src, "' has fingerprint ", fp, ", the file was converted from ",
                want, " - refused");
    std::printf("source %s fingerprint %s (matches OLD's source_fingerprint)\n", src.c_str(), fp.c_str());
}

void print_plan(const StrixwFile &old, const Patch &p, const std::vector<StrixwTensor> &planned, uint64_t bytes,
                uint64_t data_offset) {
    std::printf("layout %s\n", p.meta.at("layout").c_str());
    std::printf("old_bytes %llu new_bytes %llu delta %lld data_offset %llu tensors %zu\n",
                (unsigned long long)old.file_bytes(), (unsigned long long)bytes,
                (long long)bytes - (long long)old.file_bytes(), (unsigned long long)data_offset, planned.size());
    for (size_t i = 0; i < planned.size(); ++i) {
        const StrixwTensor &t = planned[i], &ot = old.tensors()[i];
        std::printf("%s %s %s g%lld", is_mtp_fc(t.name) ? "PATCH" : "copy ", t.name.c_str(), enc_fmt(t).c_str(),
                    (long long)t.group_size);
        for (size_t c = 0; c < t.components.size(); ++c)
            std::printf(" %s@%llu+%llu", strixw_role_name(t.components[c].role),
                        (unsigned long long)t.components[c].offset, (unsigned long long)t.components[c].bytes);
        std::printf(" (old@%llu)\n", (unsigned long long)ot.components[0].offset);
    }
}

std::string scratch_path(const std::string &dir) {
    return (fs::path(dir) / ("strixw_patch_plan." + std::to_string(::getpid()) + ".strixw")).string();
}

// ---------------------------------------------------------------- modes

int run_plan(const std::string &old_path, const std::string &src, const Override &o, const std::string &scratch) {
    StrixwFile old(old_path);
    if (!src.empty()) check_source(old, src);
    const Patch p = make_patch(old, old_path, o);
    const std::string tmp = scratch_path(scratch);
    std::vector<StrixwTensor> planned;
    uint64_t bytes = 0, data_offset = 0;
    try {
        // StrixwWriter plans in its constructor and writes only header + index; the file is sparse.
        StrixwWriter w(tmp, p.meta, p.tensors);
        planned = w.tensors(), bytes = w.file_bytes();
        data_offset = planned.front().components.front().offset;
    } catch (...) {
        fs::remove(tmp);
        throw;
    }
    fs::remove(tmp);
    print_plan(old, p, planned, bytes, data_offset);
    return 0;
}

int run_check(const std::string &ref_path, const std::string &src, const Override &o) {
    StrixwFile ref(ref_path);
    check_source(ref, src);
    const Checkpoint ck(src);
    RangeReader rr(ref_path);
    int bad = 0;
    for (const char *name : kMtpFc) {
        const StrixwTensor &rt = ref.get(name);
        std::printf("%s ref %s %s\n", name, enc_fmt(rt).c_str(), shape_s(rt.shape).c_str());
        if (rt.encoding != StrixwEncoding::Q4RowMajor || rt.group_size != o.G) {
            std::printf("  REF IS NOT %s - no byte comparison\n", o.fmt.c_str());
            ++bad;
            continue;
        }
        const Q4Weight q = quantize_class_tensor(ck, rt, o.G);
        for (const StrixwComponent &c : rt.components) {
            size_t n = 0;
            const auto *mine = static_cast<const uint8_t *>(plane(q, c.role, &n));
            STRIX_CHECK(n == c.bytes, "strixw_patch: ", name, " ", strixw_role_name(c.role), " is ", n,
                        " bytes, ref ", c.bytes);
            const uint8_t *theirs = rr.read(c.offset, c.bytes);
            uint64_t diff = 0, first = UINT64_MAX;
            for (size_t i = 0; i < n; ++i)
                if (mine[i] != theirs[i]) {
                    ++diff;
                    if (first == UINT64_MAX) first = i;
                }
            const uint64_t h = strix_hash64(mine, n);
            std::printf("  %-5s %9zu bytes at %llu: %s, %llu bytes differ%s, hash mine %016llx ref-index %016llx%s\n",
                        strixw_role_name(c.role), n, (unsigned long long)c.offset, diff ? "DIFFERENT" : "IDENTICAL",
                        (unsigned long long)diff,
                        diff ? (" (first at " + std::to_string(first) + ")").c_str() : "", (unsigned long long)h,
                        (unsigned long long)c.hash, h == c.hash ? "" : " MISMATCH");
            if (diff || h != c.hash) ++bad;
        }
    }
    std::printf("%s\n", bad ? "RESULT: differs" : "RESULT: byte-identical");
    return bad ? 1 : 0;
}

int run_patch(const std::string &old_path, const std::string &src, const Override &o, const std::string &out,
              int threads) {
    STRIX_CHECK(!fs::exists(out), "strixw_patch: '", out, "' exists - refusing to overwrite");
    STRIX_CHECK(fs::absolute(out) != fs::absolute(old_path), "strixw_patch: --out is --old");
    const auto t0 = std::chrono::steady_clock::now();
    StrixwFile old(old_path);
    check_source(old, src);
    const Patch p = make_patch(old, old_path, o);
    const Checkpoint ck(src);
    std::map<std::string, Q4Weight> quant;
    for (const char *name : kMtpFc) quant.emplace(name, quantize_class_tensor(ck, old.get(name), o.G));

    RangeReader rr(old_path);
    uint64_t copied = 0;
    {
        created_out = true;
        StrixwWriter w(out, p.meta, p.tensors);
        std::printf("writing %s: %llu bytes, %zu tensors, reads %s\n", out.c_str(),
                    (unsigned long long)w.file_bytes(), p.tensors.size(), rr.direct() ? "O_DIRECT" : "buffered");
        for (size_t ti = 0; ti < p.tensors.size(); ++ti) {
            const StrixwTensor &ot = old.tensors()[ti];
            const StrixwTensor &nt = w.tensors()[ti];
            STRIX_CHECK(nt.name == ot.name, "strixw_patch: tensor order changed at ", ti);
            if (is_mtp_fc(ot.name)) {
                const Q4Weight &q = quant.at(ot.name);
                for (const StrixwComponent &c : nt.components) {
                    size_t n = 0;
                    const void *d = plane(q, c.role, &n);
                    w.write(ti, c.role, d, n);
                }
                continue;
            }
            STRIX_CHECK(nt.components.size() == ot.components.size(), "strixw_patch: '", ot.name,
                        "' component count changed");
            for (size_t ci = 0; ci < ot.components.size(); ++ci) {
                const StrixwComponent &c = ot.components[ci];
                const uint8_t *d = rr.read(c.offset, c.bytes);
                w.write(ti, c.role, d, c.bytes);
                const uint64_t h = w.tensors()[ti].components[ci].hash;
                STRIX_CHECK(h == c.hash, "strixw_patch: '", ot.name, "' ", strixw_role_name(c.role),
                            " read from OLD hashes to ", h, ", OLD's index says ", c.hash, " - OLD is corrupt");
                copied += c.bytes;
            }
            if (ti % 64 == 0) {
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("  %zu/%zu tensors, %.1f GB copied, %.0f s\n", ti, p.tensors.size(), copied / 1e9, s);
                std::fflush(stdout);
            }
        }
        w.finish();
    }

    // Verify with the harvested reader: header, index, every record, every component hash.
    StrixwFile neu(out);
    STRIX_CHECK(neu.meta() == p.meta, "strixw_patch: output metadata differs from the plan");
    STRIX_CHECK(neu.tensors().size() == old.tensors().size(), "strixw_patch: output tensor count");
    for (size_t ti = 0; ti < old.tensors().size(); ++ti) {
        const StrixwTensor &a = old.tensors()[ti], &b = neu.tensors()[ti];
        STRIX_CHECK(a.name == b.name && a.shape == b.shape && a.experts == b.experts &&
                        a.parts.size() == b.parts.size(),
                    "strixw_patch: output record ", ti, " '", b.name, "' differs from OLD's");
        if (is_mtp_fc(a.name)) {
            STRIX_CHECK(b.encoding == StrixwEncoding::Q4RowMajor && b.group_size == o.G, "strixw_patch: '", b.name,
                        "' not ", o.fmt);
            for (const StrixwComponent &c : b.components) {
                size_t n = 0;
                const void *d = plane(quant.at(a.name), c.role, &n);
                STRIX_CHECK(c.hash == strix_hash64(d, n), "strixw_patch: '", b.name, "' hash");
            }
            continue;
        }
        STRIX_CHECK(a.encoding == b.encoding && a.group_size == b.group_size, "strixw_patch: '", b.name,
                    "' encoding changed");
        for (size_t ci = 0; ci < a.components.size(); ++ci)
            STRIX_CHECK(a.components[ci].hash == b.components[ci].hash && a.components[ci].bytes == b.components[ci].bytes,
                        "strixw_patch: '", b.name, "' component ", ci, " differs from OLD's");
    }
    std::printf("verifying every component hash of %s (%d threads)\n", out.c_str(), threads);
    std::fflush(stdout);
    neu.verify_hashes(threads);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("OK %s: %llu bytes, layout %s, %.0f s\n", out.c_str(), (unsigned long long)neu.file_bytes(),
                neu.meta("layout").c_str(), s);
    return 0;
}

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: strixw_patch --old OLD --source CKPT --override mtp_fc=q4g64 --out NEW [--threads N]\n"
                 "       strixw_patch --old OLD [--source CKPT] --override mtp_fc=q4g64 --plan [--scratch DIR]\n"
                 "       strixw_patch --source CKPT --override mtp_fc=q4g64 --check-against REF\n");
    std::exit(2);
}

}  // namespace

int main(int argc, char **argv) {
    std::string old_path, src, ovr, out, ref, scratch = fs::temp_directory_path().string();
    bool plan = false;
    int threads = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (a == "--old") old_path = val();
        else if (a == "--source") src = val();
        else if (a == "--override") ovr = val();
        else if (a == "--out") out = val();
        else if (a == "--check-against") ref = val();
        else if (a == "--scratch") scratch = val();
        else if (a == "--plan") plan = true;
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else usage();
    }
    try {
        if (ovr.empty()) usage();
        const Override o = parse_override(ovr);
        if (!ref.empty()) {
            if (src.empty() || !old_path.empty() || plan || !out.empty()) usage();
            return run_check(ref, src, o);
        }
        if (old_path.empty()) usage();
        if (plan) {
            if (!out.empty()) usage();
            return run_plan(old_path, src, o, scratch);
        }
        if (src.empty() || out.empty()) usage();
        STRIX_CHECK(threads >= 1 && threads <= 64, "strixw_patch: --threads ", threads);
        try {
            return run_patch(old_path, src, o, out, threads);
        } catch (...) {
            // A partial or unverified output must not be mistaken for a good one; the error names what failed.
            // (run_patch refuses an existing --out before creating anything, so this never deletes a prior file.)
            if (created_out) fs::remove(out);
            throw;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "strixw_patch: FAILED: %s\n", e.what());
        return 1;
    }
}
