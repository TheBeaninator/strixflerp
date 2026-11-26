#include "formats/faf.hpp"

#include "common/check.hpp"
#include "formats/q4.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>

namespace strix {

namespace {

constexpr char kMagic[8] = {'F', 'A', 'S', 'T', 'A', 'S', 'F', 'K'};
constexpr char kLegacyMagic[8] = {'S', 'T', 'R', 'I', 'X', 'W', 'T', '\0'};
constexpr uint64_t kK1 = 0x9e3779b97f4a7c15ull, kK2 = 0xd6e8feb86659fd93ull;

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

std::string shape_str(const std::vector<int64_t> &s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}

inline uint64_t mix(uint64_t h, uint64_t w) {
    h ^= w * kK1;
    return (h ^ (h >> 31)) * kK2;
}

// Little-endian append/read helpers for the index (the host is x86-64: memcpy is the encoding).
template <typename T> void put(std::vector<uint8_t> &b, T v) {
    const size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
}
void put_str16(std::vector<uint8_t> &b, const std::string &s, const char *what) {
    STRIX_CHECK(s.size() <= 0xffff, "faf: ", what, " '", s.substr(0, 64), "...' is ", s.size(),
                " bytes, limit 65535");
    put<uint16_t>(b, (uint16_t)s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Cursor {
    const uint8_t *p;
    size_t n, at = 0;
    const std::string &path;
    template <typename T> T get(const char *what) {
        STRIX_CHECK(n - at >= sizeof(T), "faf '", path, "': index ends inside ", what, " (at index byte ", at,
                    " of ", n, ")");
        T v;
        std::memcpy(&v, p + at, sizeof(T));
        at += sizeof(T);
        return v;
    }
    std::string str(size_t len, const char *what) {
        STRIX_CHECK(n - at >= len, "faf '", path, "': index ends inside ", what, " (", len, " bytes at index byte ",
                    at, " of ", n, ")");
        std::string s(reinterpret_cast<const char *>(p + at), len);
        at += len;
        return s;
    }
};

// Checks what the writer and the reader both require of a record (components excluded).
void validate_record(const FafTensor &t, const std::string &where) {
    STRIX_CHECK(!t.name.empty(), where, ": tensor with an empty name");
    STRIX_CHECK(!t.shape.empty() && t.shape.size() <= 8, where, ": tensor '", t.name, "' has ", t.shape.size(),
                " dims, expected 1..8");
    for (int64_t d : t.shape)
        STRIX_CHECK(d >= 1, where, ": tensor '", t.name, "' has shape ", shape_str(t.shape));
    STRIX_CHECK(t.experts >= 0 && (t.experts == 0 || t.shape[0] % t.experts == 0), where, ": tensor '", t.name,
                "': ", t.shape[0], " rows don't split into ", t.experts, " experts");
    int64_t next = 0;
    for (const FafPart &p : t.parts) {
        STRIX_CHECK(!p.source.empty() && p.rows >= 1 && p.row_offset == next, where, ": tensor '", t.name,
                    "' part '", p.source, "' at row ", p.row_offset, " (", p.rows, " rows), expected to start at row ",
                    next);
        next += p.rows;
    }
    STRIX_CHECK(t.parts.empty() || next == t.shape[0], where, ": tensor '", t.name, "' parts cover ", next,
                " rows of ", t.shape[0]);
}

}  // namespace

const char *faf_encoding_name(FafEncoding e) {
    switch (e) {
    case FafEncoding::Q4RowMajor: return "q4";
    case FafEncoding::Q4ChunkMajor: return "q4-chunk-major";
    case FafEncoding::BF16: return "bf16";
    case FafEncoding::F32: return "f32";
    case FafEncoding::I64: return "i64";
    case FafEncoding::Q8RowMajor: return "q8";
    case FafEncoding::Q8ChunkMajor: return "q8-chunk-major";
    case FafEncoding::Q6RowMajor: return "q6";
    case FafEncoding::Q5RowMajor: return "q5";
    }
    return "unknown";
}

const char *faf_role_name(FafRole r) {
    switch (r) {
    case FafRole::Data: return "data";
    case FafRole::Q: return "q";
    case FafRole::Scale: return "scale";
    case FafRole::Min: return "min";
    }
    return "unknown";
}

int64_t FafTensor::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) {
        STRIX_CHECK(d >= 1 && n <= INT64_MAX / d, "faf tensor '", name, "': bad shape ", shape_str(shape));
        n *= d;
    }
    return n;
}

uint64_t FafTensor::bytes() const {
    uint64_t b = 0;
    for (const FafComponent &c : components) b += c.bytes;
    return b;
}

std::vector<FafComponent> faf_components(const FafTensor &t) {
    const int64_t n = t.numel();
    auto comp = [](FafRole r, int64_t bytes) { return FafComponent{r, 0, (uint64_t)bytes, 0}; };
    switch (t.encoding) {
    case FafEncoding::Q4RowMajor:
    case FafEncoding::Q4ChunkMajor: {
        STRIX_CHECK(t.shape.size() == 2, "faf tensor '", t.name, "': ", faf_encoding_name(t.encoding),
                    " needs shape [N, K], got ", shape_str(t.shape));
        const int64_t N = t.shape[0], K = t.shape[1], G = t.group_size;
        STRIX_CHECK(q4_group_size_supported(G) && K % G == 0, "faf tensor '", t.name, "': ",
                    faf_encoding_name(t.encoding), " with K = ", K, ", group size ", G,
                    " (must be 32/64/128 and divide K)");
        return {comp(FafRole::Q, N * K / 2), comp(FafRole::Scale, N * (K / G) * 2),
                comp(FafRole::Min, N * (K / G) * 2)};
    }
    case FafEncoding::Q8RowMajor:
    case FafEncoding::Q8ChunkMajor: {
        STRIX_CHECK(t.shape.size() == 2, "faf tensor '", t.name, "': ", faf_encoding_name(t.encoding),
                    " needs shape [N, K], got ", shape_str(t.shape));
        const int64_t N = t.shape[0], K = t.shape[1], G = t.group_size;
        STRIX_CHECK(q4_group_size_supported(G) && K % G == 0, "faf tensor '", t.name, "': ",
                    faf_encoding_name(t.encoding), " with K = ", K, ", group size ", G,
                    " (must be 32/64/128 and divide K)");
        return {comp(FafRole::Q, N * K), comp(FafRole::Scale, N * (K / G) * 2),
                comp(FafRole::Min, N * (K / G) * 2)};
    }
    case FafEncoding::Q6RowMajor: {  // formats/q6.hpp: per row a low plane [K/2] then a high plane [K/4]
        STRIX_CHECK(t.shape.size() == 2, "faf tensor '", t.name, "': ", faf_encoding_name(t.encoding),
                    " needs shape [N, K], got ", shape_str(t.shape));
        const int64_t N = t.shape[0], K = t.shape[1], G = t.group_size;
        STRIX_CHECK(q4_group_size_supported(G) && K % G == 0 && K % 64 == 0, "faf tensor '", t.name, "': ",
                    faf_encoding_name(t.encoding), " with K = ", K, ", group size ", G,
                    " (must be 32/64/128 and divide K; K a multiple of 64)");
        return {comp(FafRole::Q, N * (K / 2 + K / 4)), comp(FafRole::Scale, N * (K / G) * 2),
                comp(FafRole::Min, N * (K / G) * 2)};
    }
    case FafEncoding::Q5RowMajor: {  // formats/q5.hpp: per row a low plane [K/2] then a high plane [K/8]
        STRIX_CHECK(t.shape.size() == 2, "faf tensor '", t.name, "': ", faf_encoding_name(t.encoding),
                    " needs shape [N, K], got ", shape_str(t.shape));
        const int64_t N = t.shape[0], K = t.shape[1], G = t.group_size;
        STRIX_CHECK(q4_group_size_supported(G) && K % G == 0 && K % 128 == 0, "faf tensor '", t.name, "': ",
                    faf_encoding_name(t.encoding), " with K = ", K, ", group size ", G,
                    " (must be 32/64/128 and divide K; K a multiple of 128)");
        return {comp(FafRole::Q, N * (K / 2 + K / 8)), comp(FafRole::Scale, N * (K / G) * 2),
                comp(FafRole::Min, N * (K / G) * 2)};
    }
    case FafEncoding::BF16:
    case FafEncoding::F32:
    case FafEncoding::I64: {
        STRIX_CHECK(t.group_size == 0, "faf tensor '", t.name, "': ", faf_encoding_name(t.encoding),
                    " has group size ", t.group_size, ", expected 0");
        const int64_t es = t.encoding == FafEncoding::BF16 ? 2 : t.encoding == FafEncoding::F32 ? 4 : 8;
        STRIX_CHECK(n <= INT64_MAX / es, "faf tensor '", t.name, "': ", n, " elements overflow");
        return {comp(FafRole::Data, n * es)};
    }
    }
    STRIX_CHECK(false, "faf tensor '", t.name, "': unknown encoding ", (int)t.encoding);
    return {};
}

StrixHasher::StrixHasher(size_t n) : total_(n) {
    h_[0] = n ^ kK1, h_[1] = n ^ kK2, h_[2] = ~n ^ kK1, h_[3] = ~n ^ kK2;
}

void StrixHasher::consume_block(const uint8_t *p) {
    uint64_t w[4];
    std::memcpy(w, p, 32);
    h_[0] = mix(h_[0], w[0]), h_[1] = mix(h_[1], w[1]), h_[2] = mix(h_[2], w[2]), h_[3] = mix(h_[3], w[3]);
}

void StrixHasher::update(const void *data, size_t len) {
    STRIX_CHECK(fed_ + len <= total_, "StrixHasher: update of ", len, " bytes with ", fed_, " of ", total_,
                " already fed - more than the length the hasher was made for");
    const uint8_t *p = static_cast<const uint8_t *>(data);
    if (len == 0) return;
    if (!carry_.empty()) {  // top the partial block up, then consume the caller's whole blocks directly
        const size_t take = std::min(len, (size_t)32 - carry_.size());
        carry_.insert(carry_.end(), p, p + take);
        p += take, len -= take, fed_ += take;
        if (carry_.size() == 32) consume_block(carry_.data()), carry_.clear();
    }
    while (len >= 32) consume_block(p), p += 32, len -= 32, fed_ += 32;
    if (len > 0) carry_.insert(carry_.end(), p, p + len), fed_ += len;
}

uint64_t StrixHasher::final() {
    STRIX_CHECK(fed_ == total_, "StrixHasher::final: ", fed_, " of ", total_, " bytes fed - the state is short");
    STRIX_CHECK(carry_.size() < 32, "StrixHasher::final: ", carry_.size(), " bytes carried (a full block was "
                "never consumed - an update bug)");
    const uint8_t *p = carry_.data();
    size_t i = 0;
    for (int lane = 0; i + 8 <= carry_.size(); i += 8, ++lane) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h_[lane] = mix(h_[lane], w);
    }
    if (i < carry_.size()) {
        uint64_t w = 0;
        std::memcpy(&w, p + i, carry_.size() - i);
        h_[0] = mix(h_[0], w ^ ((uint64_t)(carry_.size() - i) << 56));
    }
    uint64_t r = mix(mix(mix(h_[0], h_[1]), h_[2]), h_[3]);
    r ^= r >> 33;
    r *= kK1;
    return r ^ (r >> 29);
}

// One definition of the arithmetic: the one-shot hash is the streaming one over a single update, so a file
// written in pieces and a file written at once can never disagree.
uint64_t strix_hash64(const void *data, size_t n) {
    StrixHasher h(n);
    h.update(data, n);
    return h.final();
}

// ---------------------------------------------------------------- writer

FafWriter::FafWriter(const std::string &path, std::map<std::string, std::string> meta,
                           std::vector<FafTensor> tensors)
    : path_(path), meta_(std::move(meta)), tensors_(std::move(tensors)) {
    const std::string where = "FafWriter('" + path + "')";
    STRIX_CHECK(!path.empty(), "FafWriter: empty path");
    STRIX_CHECK(!tensors_.empty(), where, ": no tensors");
    std::set<std::string> names;
    for (FafTensor &t : tensors_) {
        validate_record(t, where);
        STRIX_CHECK(names.insert(t.name).second, where, ": tensor '", t.name, "' planned twice");
        t.components = faf_components(t);
    }
    index_bytes_ = encode_index().size();  // hashes are fixed-width: the size is final
    data_offset_ = align_up(kFafHeaderBytes + index_bytes_, kFafTensorAlign);
    uint64_t cur = data_offset_;
    for (FafTensor &t : tensors_) {
        cur = align_up(cur, kFafTensorAlign);
        for (FafComponent &c : t.components) {
            cur = align_up(cur, kFafComponentAlign);
            c.offset = cur;
            cur += c.bytes;
        }
        written_.emplace_back(t.components.size(), false);
    }
    file_bytes_ = cur;
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    STRIX_CHECK(fd_ >= 0, where, ": open for writing failed: ", std::strerror(errno));
    STRIX_CHECK(::ftruncate(fd_, (off_t)file_bytes_) == 0, where, ": ftruncate to ", file_bytes_, " bytes failed: ",
                std::strerror(errno));
    write_header_and_index();
}

FafWriter::~FafWriter() {
    if (fd_ >= 0) ::close(fd_);
}

std::vector<uint8_t> FafWriter::encode_index() const {
    std::vector<uint8_t> b;
    put<uint32_t>(b, (uint32_t)meta_.size());
    for (const auto &[k, v] : meta_) {
        put_str16(b, k, "metadata key");
        STRIX_CHECK(v.size() <= 0xffffffffu, "faf: metadata '", k, "' too long");
        put<uint32_t>(b, (uint32_t)v.size());
        b.insert(b.end(), v.begin(), v.end());
    }
    put<uint32_t>(b, (uint32_t)tensors_.size());
    for (const FafTensor &t : tensors_) {
        put_str16(b, t.name, "tensor name");
        put<uint8_t>(b, (uint8_t)t.encoding);
        put<uint8_t>(b, (uint8_t)t.shape.size());
        for (int64_t d : t.shape) put<int64_t>(b, d);
        put<int64_t>(b, t.group_size);
        put<int64_t>(b, t.experts);
        STRIX_CHECK(t.parts.size() <= 0xffff, "faf: tensor '", t.name, "' has ", t.parts.size(), " parts");
        put<uint16_t>(b, (uint16_t)t.parts.size());
        for (const FafPart &p : t.parts) {
            put_str16(b, p.source, "part source name");
            put<int64_t>(b, p.row_offset);
            put<int64_t>(b, p.rows);
        }
        put<uint8_t>(b, (uint8_t)t.components.size());
        for (const FafComponent &c : t.components) {
            put<uint8_t>(b, (uint8_t)c.role);
            put<uint64_t>(b, c.offset);
            put<uint64_t>(b, c.bytes);
            put<uint64_t>(b, c.hash);
        }
    }
    return b;
}

namespace {

void pwrite_all(int fd, const void *data, size_t bytes, uint64_t offset, const std::string &where) {
    const auto *p = static_cast<const uint8_t *>(data);
    while (bytes > 0) {
        const size_t chunk = std::min<size_t>(bytes, 1u << 30);
        const ssize_t w = ::pwrite(fd, p, chunk, (off_t)offset);
        if (w < 0 && errno == EINTR) continue;
        STRIX_CHECK(w > 0, where, ": pwrite of ", chunk, " bytes at offset ", offset, " failed: ",
                    w < 0 ? std::strerror(errno) : "wrote 0 bytes");
        p += w, bytes -= (size_t)w, offset += (uint64_t)w;
    }
}

}  // namespace

void FafWriter::write_header_and_index() {
    const std::vector<uint8_t> index = encode_index();
    STRIX_CHECK(index.size() == index_bytes_, "FafWriter('", path_, "'): index changed size (", index.size(),
                " vs planned ", index_bytes_, ")");
    std::vector<uint8_t> h(kFafHeaderBytes, 0);
    std::memcpy(h.data(), kMagic, 8);
    auto at = [&](size_t off, auto v) { std::memcpy(h.data() + off, &v, sizeof(v)); };
    at(8, kFafVersion);
    at(12, (uint32_t)kFafHeaderBytes);
    at(16, (uint64_t)kFafHeaderBytes);
    at(24, index_bytes_);
    at(32, data_offset_);
    at(40, file_bytes_);
    at(48, strix_hash64(index.data(), index.size()));
    at(120, strix_hash64(h.data(), 120));
    pwrite_all(fd_, h.data(), h.size(), 0, "FafWriter('" + path_ + "') header");
    pwrite_all(fd_, index.data(), index.size(), kFafHeaderBytes, "FafWriter('" + path_ + "') index");
}

void FafWriter::write(size_t ti, FafRole role, const void *data, size_t bytes) {
    STRIX_CHECK(!finished_ && fd_ >= 0, "FafWriter('", path_, "'): write after finish");
    STRIX_CHECK(ti < tensors_.size(), "FafWriter('", path_, "'): tensor index ", ti, " of ", tensors_.size());
    FafTensor &t = tensors_[ti];
    for (size_t ci = 0; ci < t.components.size(); ++ci) {
        FafComponent &c = t.components[ci];
        if (c.role != role) continue;
        STRIX_CHECK(data != nullptr, "FafWriter: null data for '", t.name, "' ", faf_role_name(role));
        STRIX_CHECK(bytes == c.bytes, "FafWriter: '", t.name, "' ", faf_role_name(role), " is ", bytes,
                    " bytes, planned ", c.bytes, " (", faf_encoding_name(t.encoding), " ", shape_str(t.shape),
                    ")");
        STRIX_CHECK(!written_[ti][ci], "FafWriter: '", t.name, "' ", faf_role_name(role), " written twice");
        pwrite_all(fd_, data, bytes, c.offset, "FafWriter('" + path_ + "') tensor '" + t.name + "'");
        c.hash = strix_hash64(data, bytes);
        // Push it to disk now and drop it from the page cache: ~67 GiB of dirty pages would otherwise sit in
        // RAM until the final sync (and the verify pass should read what's on disk).
        STRIX_CHECK(::sync_file_range(fd_, (off_t)c.offset, (off_t)bytes,
                                      SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER) == 0,
                    "FafWriter('", path_, "'): sync_file_range for '", t.name, "' failed: ", std::strerror(errno));
        (void)::posix_fadvise(fd_, (off_t)c.offset, (off_t)bytes, POSIX_FADV_DONTNEED);
        written_[ti][ci] = true;
        return;
    }
    STRIX_CHECK(false, "FafWriter: '", t.name, "' (", faf_encoding_name(t.encoding), ") has no ",
                faf_role_name(role), " component");
}

void FafWriter::finish() {
    STRIX_CHECK(!finished_, "FafWriter('", path_, "'): finish called twice");
    for (size_t ti = 0; ti < tensors_.size(); ++ti)
        for (size_t ci = 0; ci < written_[ti].size(); ++ci)
            STRIX_CHECK(written_[ti][ci], "FafWriter('", path_, "'): tensor '", tensors_[ti].name, "' ",
                        faf_role_name(tensors_[ti].components[ci].role), " was never written");
    write_header_and_index();
    STRIX_CHECK(::fdatasync(fd_) == 0, "FafWriter('", path_, "'): fdatasync failed: ", std::strerror(errno));
    STRIX_CHECK(::close(fd_) == 0, "FafWriter('", path_, "'): close failed: ", std::strerror(errno));
    fd_ = -1;
    finished_ = true;
}

// ---------------------------------------------------------------- reader

FafFile::FafFile(const std::string &path) : path_(path) {
    const std::string where = "faf '" + path + "'";
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    STRIX_CHECK(fd >= 0, where, ": open failed: ", std::strerror(errno));
    struct stat st{};
    const int sr = ::fstat(fd, &st);
    const int err = errno;
    if (sr != 0) ::close(fd);
    STRIX_CHECK(sr == 0, where, ": fstat failed: ", std::strerror(err));
    const uint64_t size = (uint64_t)st.st_size;
    if (size < kFafHeaderBytes) ::close(fd);
    STRIX_CHECK(size >= kFafHeaderBytes, where, ": ", size, " bytes, smaller than the ", kFafHeaderBytes,
                "-byte header");
    map_ = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    STRIX_CHECK(map_ != MAP_FAILED, where, ": mmap of ", size, " bytes failed: ", std::strerror(errno));
    file_bytes_ = size;
    const auto *b = static_cast<const uint8_t *>(map_);
    try {
        auto rd = [&](size_t off, auto v) {
            std::memcpy(&v, b + off, sizeof(v));
            return v;
        };
        STRIX_CHECK(std::memcmp(b, kMagic, 8) == 0 || std::memcmp(b, kLegacyMagic, 8) == 0, where, ": not a .faf or legacy .strixw file (bad magic)");
        const uint32_t version = rd(8, uint32_t{}), hsize = rd(12, uint32_t{});
        STRIX_CHECK(version >= kFafOldestVersion && version <= kFafVersion, where, ": format version ", version,
                    ", this build reads ", kFafOldestVersion, "..", kFafVersion, " (re-run the converter)");
        STRIX_CHECK(hsize == kFafHeaderBytes, where, ": header size ", hsize, ", expected ", kFafHeaderBytes);
        STRIX_CHECK(rd(120, uint64_t{}) == strix_hash64(b, 120), where, ": header hash mismatch (corrupt header)");
        const uint64_t index_off = rd(16, uint64_t{}), index_bytes = rd(24, uint64_t{}),
                       file_bytes = rd(40, uint64_t{});
        data_offset_ = rd(32, uint64_t{});
        STRIX_CHECK(file_bytes == size, where, ": header says ", file_bytes, " bytes, the file has ", size,
                    " (truncated or appended to)");
        STRIX_CHECK(index_off == kFafHeaderBytes && index_bytes <= size - index_off, where, ": index at ",
                    index_off, " (", index_bytes, " bytes) doesn't fit the ", size, "-byte file");
        STRIX_CHECK(data_offset_ % kFafTensorAlign == 0 && data_offset_ >= index_off + index_bytes &&
                        data_offset_ <= size,
                    where, ": data offset ", data_offset_, " (index ends at ", index_off + index_bytes,
                    ", alignment ", kFafTensorAlign, ")");
        STRIX_CHECK(rd(48, uint64_t{}) == strix_hash64(b + index_off, index_bytes), where,
                    ": index hash mismatch (corrupt index)");

        Cursor c{b + index_off, index_bytes, 0, path_};
        const uint32_t nmeta = c.get<uint32_t>("metadata count");
        for (uint32_t i = 0; i < nmeta; ++i) {
            std::string k = c.str(c.get<uint16_t>("metadata key length"), "metadata key");
            std::string v = c.str(c.get<uint32_t>("metadata value length"), "metadata value");
            STRIX_CHECK(meta_.emplace(k, std::move(v)).second, where, ": metadata key '", k, "' twice");
        }
        const uint32_t ntensors = c.get<uint32_t>("tensor count");
        STRIX_CHECK(ntensors >= 1, where, ": no tensors");
        std::vector<std::pair<uint64_t, uint64_t>> spans;
        for (uint32_t i = 0; i < ntensors; ++i) {
            FafTensor t;
            t.name = c.str(c.get<uint16_t>("tensor name length"), "tensor name");
            t.encoding = (FafEncoding)c.get<uint8_t>("encoding");
            const uint8_t ndim = c.get<uint8_t>("ndim");
            for (uint8_t d = 0; d < ndim; ++d) t.shape.push_back(c.get<int64_t>("shape"));
            t.group_size = c.get<int64_t>("group size");
            t.experts = c.get<int64_t>("experts");
            const uint16_t nparts = c.get<uint16_t>("part count");
            for (uint16_t p = 0; p < nparts; ++p) {
                FafPart part;
                part.source = c.str(c.get<uint16_t>("part name length"), "part name");
                part.row_offset = c.get<int64_t>("part row offset");
                part.rows = c.get<int64_t>("part rows");
                t.parts.push_back(std::move(part));
            }
            const uint8_t ncomp = c.get<uint8_t>("component count");
            for (uint8_t k = 0; k < ncomp; ++k) {
                FafComponent comp;
                comp.role = (FafRole)c.get<uint8_t>("component role");
                comp.offset = c.get<uint64_t>("component offset");
                comp.bytes = c.get<uint64_t>("component bytes");
                comp.hash = c.get<uint64_t>("component hash");
                t.components.push_back(comp);
            }
            validate_record(t, where);
            const std::vector<FafComponent> want = faf_components(t);
            STRIX_CHECK(want.size() == t.components.size(), where, ": tensor '", t.name, "' (",
                        faf_encoding_name(t.encoding), ") has ", t.components.size(), " components, expected ",
                        want.size());
            for (size_t k = 0; k < want.size(); ++k) {
                const FafComponent &g = t.components[k];
                STRIX_CHECK(g.role == want[k].role && g.bytes == want[k].bytes, where, ": tensor '", t.name,
                            "' component ", k, " is ", faf_role_name(g.role), " of ", g.bytes, " bytes, expected ",
                            faf_role_name(want[k].role), " of ", want[k].bytes, " for ",
                            faf_encoding_name(t.encoding), " ", shape_str(t.shape));
                // g.offset <= size first: past it, size - g.offset wraps and any g.bytes would pass.
                STRIX_CHECK(g.offset >= data_offset_ && g.offset <= size && g.offset % kFafComponentAlign == 0 &&
                                g.bytes <= size - g.offset,
                            where, ": tensor '", t.name, "' ", faf_role_name(g.role), " at ", g.offset, " (",
                            g.bytes, " bytes) is outside the data region [", data_offset_, ", ", size,
                            ") or not ", kFafComponentAlign, "-aligned");
                spans.emplace_back(g.offset, g.offset + g.bytes);
            }
            STRIX_CHECK(by_name_.emplace(t.name, tensors_.size()).second, where, ": tensor '", t.name, "' twice");
            tensors_.push_back(std::move(t));
        }
        STRIX_CHECK(c.at == c.n, where, ": ", c.n - c.at, " unread bytes after the last index record");
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size(); ++i)
            STRIX_CHECK(spans[i].first >= spans[i - 1].second, where, ": components overlap at byte ", spans[i].first);
    } catch (...) {
        ::munmap(map_, file_bytes_);
        map_ = nullptr;
        throw;
    }
}

FafFile::~FafFile() {
    if (map_) ::munmap(map_, file_bytes_);
}

const std::string &FafFile::meta(const std::string &key) const {
    auto it = meta_.find(key);
    STRIX_CHECK(it != meta_.end(), "faf '", path_, "': no metadata key '", key, "'");
    return it->second;
}

const FafTensor *FafFile::find(const std::string &name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const FafTensor &FafFile::get(const std::string &name) const {
    const FafTensor *t = find(name);
    STRIX_CHECK(t != nullptr, "faf '", path_, "': no tensor '", name, "' (", tensors_.size(), " tensors)");
    return *t;
}

const void *FafFile::data(const FafTensor &t, FafRole role) const {
    STRIX_CHECK(find(t.name) == &t, "faf '", path_, "': tensor '", t.name, "' is not from this file");
    for (const FafComponent &c : t.components)
        if (c.role == role) return static_cast<const uint8_t *>(map_) + c.offset;
    STRIX_CHECK(false, "faf '", path_, "': tensor '", t.name, "' (", faf_encoding_name(t.encoding),
                ") has no ", faf_role_name(role), " component");
    return nullptr;
}

void FafFile::verify_hashes(int threads) const {
    STRIX_CHECK(threads >= 1 && threads <= 1024, "verify_hashes: threads = ", threads);
    std::vector<std::pair<const FafTensor *, const FafComponent *>> work;
    for (const FafTensor &t : tensors_)
        for (const FafComponent &c : t.components) work.emplace_back(&t, &c);
    std::atomic<size_t> next{0};
    std::mutex mu;
    std::string first_error;
    auto worker = [&] {
        for (size_t i; (i = next.fetch_add(1)) < work.size();) {
            const auto [t, c] = work[i];
            const uint64_t h = strix_hash64(static_cast<const uint8_t *>(map_) + c->offset, c->bytes);
            if (h != c->hash) {
                std::lock_guard<std::mutex> lock(mu);
                if (first_error.empty())
                    first_error = "tensor '" + t->name + "' " + faf_role_name(c->role) + " (" +
                                  std::to_string(c->bytes) + " bytes at " + std::to_string(c->offset) + ")";
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
    for (std::thread &th : pool) th.join();
    STRIX_CHECK(first_error.empty(), "faf '", path_, "': hash mismatch in ", first_error, " - the file is corrupt");
}

}  // namespace strix
