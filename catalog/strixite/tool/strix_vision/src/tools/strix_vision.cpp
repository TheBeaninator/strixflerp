// strix_vision: one image file through the engine's own vision pipeline -
// serve/vision_preprocess (decode, smart_resize, normalize, patchify) -> VisionTower::forward (the v0.3.0
// tower from vision.strixw, on the GPU) -> the merger's rows, the embeddings an image prefill splices at
// its image_pad rows, written as [N, 2560] BF16. The G1 gate (plan section 3) is --golden / --golden-root:
// EVERY stage available is compared against the HF dumps - pixel_values first, so preprocessing is checked
// separately from the tower, then patch_embed, pos_added, block0, block13, block26, merger - printing
// per-stage max-abs and cosine, and applying the gate:
//
//   floor(stage, case) = max|hf16 - hf32| for that stage, measured from the golden dumps where both
//   precisions exist (recomputed from the .npy files here and cross-checked against floor.json).
//   PASS per case = every dumped stage has max|strix - hf32| <= 1.5 x floor(stage, case); the merger
//   cosine vs hf32 >= 0.9995 is reported as the sanity floor (it never replaces the ratio rule).
//   A stage with no floor (no BF16 twin in the golden) cannot pass the ratio rule - it is reported as
//   such, not waved through. pixel_values has no BF16 twin by design (it is the input, engine rounding
//   starts after it): it is compared to the FP32 golden, exactness reported, and excluded from the ratio
//   verdict (nothing of ours is rounded before it; a difference here is preprocessing, not precision).
//
// The reference is the HF model, never the engine. A stage beyond 1.5 x floor FAILS the case and the
// table names the stage (the named suspects for a mismatch are the position-embedding interpolation -
// upstream bilinear align_corners vs the plan's bicubic - and the resize filter; the per-stage table
// localizes which one it is).
//
//   encode one image: strix_vision --weights W.strixw --vision V.strixw --config C.json --preproc P.json \
//                          --image photo.jpg --out emb          -> emb.bin (bf16) + emb.f32.npy + emb.json
//   the golden gate:    ... ... --golden-root /path/to/goldens \
//                          --floor /path/to/goldens/floor.json --results-dir ./results
//   the bench:          ... --bench --bench-src some.jpg [--bench-sizes 0.25,1,2.5] [--repeat 5]
//
// --act bf16 is the serving precision (WMMA attention, n <= kMaxVisionSeqWmma); --act f32 is the dev-stage
// reference (naive kernel, n <= kMaxVisionSeq). The resize filter (--resize bicubic_aa default | bilinear)
// and the size rule (--min-pixels, --max-pixels, --max-patches) default to the golden's rule (README: min
// 65536 px, the serve cap 2621440 px, no patch clamp); the serve parity is --max-patches 8192.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "formats/strixw.hpp"    // StrixwFile: the text side-file the tower's fingerprint must match
#include "kernels/norm.hpp"      // kernels::Act
#include "kernels/vision.hpp"    // kMaxVisionSeq / kMaxVisionSeqWmma (attention caps)
#include "runtime/device_buffer.hpp"
#include "runtime/vision_tower.hpp"
#include "serve/json.hpp"
#include "serve/vision_preprocess.hpp"
#include "serve/vision_request.hpp"  // load_vision_config

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace strix;

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t es(kernels::Act a) { return a == kernels::Act::F32 ? 4 : 2; }

float bf16_bits_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// The session's host-side rounding (runtime/qwen4exp.cpp, anonymous namespace): round-to-nearest-even,
// non-finite preserved. The tool is a separate binary, so it carries its own copy of the same rule.
uint16_t f32_to_bf16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// ---------------------------------------------------------------- arguments

struct Args {
    std::map<std::string, std::string> kv;
    bool has(const std::string &k) const { return kv.count(k) != 0; }
    std::string get(const std::string &k, const std::string &d = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
    int64_t num(const std::string &k, int64_t d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : std::stoll(it->second);
    }
};

std::vector<std::string> split(const std::string &s, char c) {
    std::vector<std::string> r;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, c);)
        if (!x.empty()) r.push_back(x);
    return r;
}

// ---------------------------------------------------------------- files

std::vector<uint8_t> read_bytes(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    STRIX_CHECK(f.good(), "read: can't open '", path, "'");
    std::vector<uint8_t> out((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    STRIX_CHECK(!out.empty(), "read: '", path, "' is empty");
    return out;
}

std::string read_text(const std::string &path) {
    auto b = read_bytes(path);
    return std::string(b.begin(), b.end());
}

bool file_exists(const std::string &p) {
    std::ifstream f(p);
    return f.good();
}

void write_file(const std::string &path, const void *data, size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    STRIX_CHECK(f.good(), "write: can't create '", path, "'");
    f.write((const char *)data, (std::streamsize)n);
    STRIX_CHECK(f.good(), "write: failed writing '", path, "'");
}

// ---------------------------------------------------------------- .npy (the golden dumps)

// The golden dumps are .npy whose payload is FP32 (descr <f4) in EVERY file, the .bf16.npy ones included
// (the golden README's dtype quirk: read the dtype from the file, not the name).
struct NpyArray {
    std::vector<float> data;
    std::vector<int64_t> shape;
    std::string shape_str() const {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
        return s + "]";
    }
};

NpyArray load_npy_f4(const std::string &path) {
    const char *fn = "load_npy";
    auto raw = read_bytes(path);
    STRIX_CHECK(raw.size() > 12 && std::memcmp(raw.data(), "\x93" "NUMPY", 6) == 0, fn, ": '", path,
                "' is not an npy file");
    const uint8_t ver = raw[6];
    size_t hlen = 0, body = 0;
    if (ver == 1) {
        hlen = (size_t)raw[8] | ((size_t)raw[9] << 8);
        body = 10 + hlen;
    } else if (ver == 2 || ver == 3) {
        hlen = (size_t)raw[8] | ((size_t)raw[9] << 8) | ((size_t)raw[10] << 16) | ((size_t)raw[11] << 24);
        body = 12 + hlen;
    } else
        STRIX_CHECK(false, fn, ": '", path, "' npy version ", (int)ver, ", expected 1, 2 or 3");
    STRIX_CHECK(raw.size() >= body, fn, ": '", path, "' truncated header");
    const std::string hdr(raw.begin() + (ver == 1 ? 10 : 12), raw.begin() + (long)body);
    const auto field = [&](const std::string &key) -> std::string {
        const std::string k = "'" + key + "':";
        size_t p = hdr.find(k);
        STRIX_CHECK(p != std::string::npos, fn, ": '", path, "' header has no '", key, "' (", hdr, ")");
        size_t e = hdr.size();
        for (int depth = 0, i = (int)(p + k.size()); i < (int)hdr.size(); ++i) {
            const char c = hdr[(size_t)i];
            if (c == '(') ++depth;
            else if (c == ')') --depth;
            else if (depth == 0 && (c == ',' || c == '}')) { e = (size_t)i; break; }
        }
        STRIX_CHECK(e < hdr.size(), fn, ": '", path, "' header malformed after '", key, "'");
        std::string v = hdr.substr(p + k.size(), e - p - k.size());
        while (!v.empty() && (v.front() == ' ' || v.front() == '\'')) v.erase(v.begin());
        while (!v.empty() && (v.back() == ' ' || v.back() == '\'')) v.pop_back();
        return v;
    };
    const std::string descr = field("descr");
    STRIX_CHECK(descr == "<f4", fn, ": '", path, "' descr is '", descr, "', expected '<f4' - the golden dumps "
                "store FP32 payloads in every file, the .bf16.npy included (README dtype quirk)");
    STRIX_CHECK(field("fortran_order") == "False", fn, ": '", path, "' is Fortran-order, unsupported");
    const std::string shp = field("shape");
    NpyArray a;
    {
        std::stringstream ss(shp.find('(') != std::string::npos
                                 ? shp.substr(shp.find('(') + 1, shp.rfind(')') - shp.find('(') - 1)
                                 : shp);
        for (std::string x; std::getline(ss, x, ',');) {
            while (!x.empty() && (x.front() == ' ')) x.erase(x.begin());
            while (!x.empty() && (x.back() == ' ')) x.pop_back();
            if (x.empty()) continue;
            a.shape.push_back(std::stoll(x));
        }
    }
    int64_t n = 1;
    for (int64_t d : a.shape) n *= d;
    STRIX_CHECK(n >= 0 && raw.size() == body + (size_t)n * 4, fn, ": '", path, "' holds ",
                (raw.size() - body) / 4, " floats for shape ", shp, ", expected ", n);
    a.data.resize((size_t)n);
    std::memcpy(a.data.data(), raw.data() + body, (size_t)n * 4);
    return a;
}

void write_npy_f4(const std::string &path, const std::vector<float> &data, const std::vector<int64_t> &shape) {
    std::string shp = "(";
    for (size_t i = 0; i < shape.size(); ++i) shp += std::to_string(shape[i]) + (i + 1 < shape.size() ? ", " : "");
    if (shape.size() == 1) shp += ",";
    shp += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shp + ", }";
    hdr.append((64 - (10 + hdr.size() + 1) % 64) % 64, ' ');
    hdr += "\n";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    STRIX_CHECK(f.good(), "write_npy: can't create '", path, "'");
    f.write("\x93" "NUMPY", 6);
    f.put(1);
    f.put(0);
    const uint16_t hl = (uint16_t)hdr.size();
    f.put((char)(hl & 0xff)).put((char)(hl >> 8));
    f.write(hdr.data(), (std::streamsize)hdr.size());
    f.write((const char *)data.data(), (std::streamsize)data.size() * 4);
    STRIX_CHECK(f.good(), "write_npy: failed writing '", path, "'");
}

// ---------------------------------------------------------------- comparisons

struct StageStats {
    int64_t n = 0, n_exact = 0;
    double max_abs = 0, mean_abs = 0, cos = 1.0;
    bool nan_found = false;
};

StageStats compare_stages(const std::vector<float> &a, const std::vector<float> &b) {
    STRIX_CHECK(a.size() == b.size(), "compare_stages: ", a.size(), " vs ", b.size(), " elements");
    StageStats s;
    long double dot = 0, na = 0, nb = 0;
    double sum = 0, max_abs = 0;
    int64_t exact = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float x = a[i], y = b[i];
        if (!std::isfinite(x) || !std::isfinite(y)) s.nan_found = true;
        const double d = std::fabs((double)x - (double)y);
        if (d > max_abs) max_abs = d;
        sum += d;
        exact += (x == y) ? 1 : 0;
        dot += (long double)x * (long double)y;
        na += (long double)x * (long double)x;
        nb += (long double)y * (long double)y;
    }
    s.n = (int64_t)a.size();
    s.n_exact = exact;
    s.max_abs = max_abs;
    s.mean_abs = a.empty() ? 0 : sum / (double)a.size();
    const long double den = std::sqrt(na) * std::sqrt(nb);
    s.cos = den > 0 ? (double)(dot / den) : 0.0;
    return s;
}

// ---------------------------------------------------------------- the run

struct Options {
    std::string weights, vision, config, preproc, image, out, golden, golden_root, floor, results_dir;
    std::string bench_src;
    std::string image_root;  // mirror root for the meta.json image paths (workers see the golden tree
                             // under the NFS export, a local mirror of the golden tree: --image-root re-roots the meta paths here)
    std::string feed = "golden";  // --feed golden|preprocessed: in golden mode, the tower runs on the
                                  // golden's pixel_values (the stage dumps are ITS forward, so the tower
                                  // half of the check is apples-to-apples; preprocessing is its own line)
                                  // or on our own re-preprocessed pixels (the end-to-end serve chain)
    bool preproc_only = false;  // --preproc-only: compare pixel_values against the HF dump only, no tower
                                // (the preprocessing half of the check, runnable where no GPU exists)
    std::string dump_pv, dump_stages;  // --dump-pv DIR / --dump-stages DIR: also write OUR pixel_values and
                                       // our probed stage tensors as FP32 .npy into an existing DIR. The
                                       // stats in the results say HOW FAR a stage is off; the arrays say
                                       // WHERE, which is what separates rounding noise (spread over the
                                       // whole tensor, ~1 ulp) from a structural difference (a row, a
                                       // channel, a half-plane). The dirs must exist.
    kernels::Act act = kernels::Act::BF16;
    int64_t min_pixels = 65536, max_pixels = 2621440, max_patches = 0;  // the golden's rule (README)
    vision::ResizeFilter resize = vision::ResizeFilter::BicubicAntialias;
    std::vector<double> bench_sizes = {0.25, 1.0, 2.5};  // MP
    int64_t repeat = 5;
    bool bench = false;
};

// One tower + the grow-on-larger activation set (the session's ensure_vision_scratch pattern), on the
// default stream (the tool is single-threaded).
struct TowerRun {
    StrixwFile text_index;
    VisionTower tower;
    hipStream_t stream = nullptr;
    int64_t patches = 0;
    size_t act_bytes = 2;
    DeviceBuffer<uint8_t> pv, x, conv, ln, proj, qkv, mlp, merg, merged, idx, w;

    TowerRun(const std::string &weights, const std::string &vision_path, kernels::Act act)
        // v0.5.0: the tower constructor takes no placement - the tower always loads to the device.
        : text_index(weights), tower(vision_path, text_index, act) {
        act_bytes = es(act);
    }

    void ensure(int64_t grid_h, int64_t grid_w) {
        const int64_t p = grid_h * grid_w;
        if (patches >= p && pv.get() != nullptr) return;
        const TowerScratchBytes b = tower.scratch_bytes(grid_h, grid_w);
        pv = DeviceBuffer<uint8_t>((size_t)b.patch_in * act_bytes, "pv");
        x = DeviceBuffer<uint8_t>((size_t)b.hidden * act_bytes, "x");
        conv = DeviceBuffer<uint8_t>((size_t)b.hidden * act_bytes, "conv");
        ln = DeviceBuffer<uint8_t>((size_t)b.hidden * act_bytes, "ln");
        proj = DeviceBuffer<uint8_t>((size_t)b.hidden * act_bytes, "proj");
        qkv = DeviceBuffer<uint8_t>((size_t)b.qkv * act_bytes, "qkv");
        mlp = DeviceBuffer<uint8_t>((size_t)b.mlp * act_bytes, "mlp");
        merg = DeviceBuffer<uint8_t>((size_t)b.merg * act_bytes, "merg");
        merged = DeviceBuffer<uint8_t>((size_t)b.merged * act_bytes, "merged");
        idx = DeviceBuffer<uint8_t>((size_t)p * 4 * sizeof(int32_t), "pos-idx");
        w = DeviceBuffer<uint8_t>((size_t)p * 4 * sizeof(float), "pos-w");
        patches = p;
    }

    TowerScratch scratch() {
        return TowerScratch{pv.get(), x.get(), conv.get(), ln.get(), proj.get(), qkv.get(), mlp.get(),
                            merg.get(), (int32_t *)idx.get(), (float *)w.get(), merged.get()};
    }

    // Upload pixel_values (the session's rounding point: FP32 in, activation bits out) + the grid taps,
    // run the forward. want_stages names the probe stages to read back into stages_out (empty = none, the
    // probe is not even installed then); merged_f32 (if given) gets the merger rows as FP32. Returns the
    // upload + forward + final-sync wall ms; stage readbacks (when requested) are inside that window.
    double encode(const vision::Preprocessed &img, const std::vector<std::string> &want_stages,
                  std::map<std::string, std::vector<float>> &stages_out, std::vector<float> *merged_f32) {
        const char *fn = "encode";
        const int64_t n = img.grid_h * img.grid_w;
        const int64_t m2 = tower.dims().merge * tower.dims().merge;
        ensure(img.grid_h, img.grid_w);
        std::vector<int32_t> taps_idx;
        std::vector<float> taps_w;
        tower.grid_taps(img.grid_h, img.grid_w, taps_idx, taps_w);

        const size_t pv_elems = (size_t)(n * tower.dims().patch_in);
        STRIX_CHECK(img.pixel_values.size() == pv_elems, fn, ": pixel_values has ", img.pixel_values.size(),
                    " elements, the grid wants ", pv_elems);
        if (tower.act() == kernels::Act::F32) {
            STRIX_HIP_CHECK(hipMemcpyAsync(pv.get(), img.pixel_values.data(), pv_elems * 4,
                                            hipMemcpyHostToDevice, stream), fn, ": upload pixel_values");
        } else {
            std::vector<uint16_t> bits(pv_elems);
            for (size_t i = 0; i < pv_elems; ++i) bits[i] = f32_to_bf16_bits(img.pixel_values[i]);
            STRIX_HIP_CHECK(hipMemcpyAsync(pv.get(), bits.data(), pv_elems * 2, hipMemcpyHostToDevice, stream),
                            fn, ": upload pixel_values");
        }
        STRIX_HIP_CHECK(hipMemcpyAsync(idx.get(), taps_idx.data(), (size_t)n * 4 * sizeof(int32_t),
                                        hipMemcpyHostToDevice, stream), fn, ": upload tap indices");
        STRIX_HIP_CHECK(hipMemcpyAsync(w.get(), taps_w.data(), (size_t)n * 4 * sizeof(float),
                                        hipMemcpyHostToDevice, stream), fn, ": upload tap weights");

        const int64_t hid = tower.dims().hidden;
        VisionTower::StageProbe probe = nullptr;
        if (!want_stages.empty())
            probe = [&](const char *stage, const void *dev) {
                if (std::find(want_stages.begin(), want_stages.end(), std::string(stage)) == want_stages.end())
                    return;
                std::vector<float> host;
                STRIX_HIP_CHECK(hipStreamSynchronize(stream), fn, ": stage sync at '", stage, "'");
                if (tower.act() == kernels::Act::F32) {
                    host.resize((size_t)n * hid);
                    STRIX_HIP_CHECK(hipMemcpy(host.data(), dev, (size_t)n * hid * 4, hipMemcpyDeviceToHost),
                                    fn, ": read stage '", stage, "'");
                } else {
                    // its own bits buffer, then widened: decoding bf16 in place over the copy destination
                    // would alias (each FP32 write eats the next two source words)
                    std::vector<uint16_t> bits((size_t)n * hid);
                    STRIX_HIP_CHECK(hipMemcpy(bits.data(), dev, (size_t)n * hid * 2, hipMemcpyDeviceToHost),
                                    fn, ": read stage '", stage, "'");
                    host.resize(bits.size());
                    for (size_t i = 0; i < bits.size(); ++i) host[i] = bf16_bits_to_f32(bits[i]);
                }
                stages_out[stage] = std::move(host);
            };

        const double t0 = now_ms();
        tower.forward(img.grid_h, img.grid_w, (const int32_t *)idx.get(), (const float *)w.get(), scratch(),
                      stream, probe);
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), fn, ": forward sync");
        const double ms = now_ms() - t0;

        if (merged_f32) {
            const size_t rows = (size_t)(n / m2), cols = (size_t)tower.dims().out_hidden;
            std::vector<float> host(rows * cols);
            if (tower.act() == kernels::Act::F32) {
                STRIX_HIP_CHECK(hipMemcpy(host.data(), merged.get(), rows * cols * 4, hipMemcpyDeviceToHost),
                                fn, ": read merged");
            } else {
                std::vector<uint16_t> bits(rows * cols);
                STRIX_HIP_CHECK(hipMemcpy(bits.data(), merged.get(), rows * cols * 2, hipMemcpyDeviceToHost),
                                fn, ": read merged");
                for (size_t i = 0; i < bits.size(); ++i) host[i] = bf16_bits_to_f32(bits[i]);
            }
            *merged_f32 = std::move(host);
        }
        return ms;
    }
};

// ---------------------------------------------------------------- the golden gate

const char *kStageFiles[] = {"patch_embed", "pos_added", "block0", "block13", "block26", "merger"};
constexpr int kNumStages = 6;

struct FloorDoc {  // floor.json: {"per_image": {"<case>": {"<stage>": {"max_abs": x, ...}}}}
    std::map<std::string, std::map<std::string, double>> per_image;
};

FloorDoc load_floor(const std::string &path) {
    FloorDoc f;
    if (path.empty() || !file_exists(path)) return f;
    json::Value root = json::Value::parse(read_text(path));
    const json::Value *pi = root.find("per_image");
    if (pi == nullptr || !pi->is_object()) return f;
    for (const auto &c : pi->as_object("per_image")) {
        if (!c.second.is_object()) continue;
        for (const auto &st : c.second.as_object(c.first)) {
            const json::Value *ma = st.second.find("max_abs");
            if (ma != nullptr && ma->is_number())
                f.per_image[c.first][st.first] = ma->as_double(c.first + "." + st.first + ".max_abs");
        }
    }
    return f;
}

struct CaseStage {
    std::string name;
    bool present = false;                 // the golden has this stage's dumps
    bool shape_ok = false;
    std::string shape;
    StageStats vs_f32, vs_bf16;           // strix vs the FP32 / BF16 dumps
    bool has_floor = false;
    double floor_abs = 0, floor_cos = 1;  // measured max|hf16 - hf32| and cos(hf16, hf32)
    bool ratio_ok = false;
    double ratio = 0;                     // max|strix - hf32| / floor
};

struct CaseResult {
    std::string name, image;
    bool ran = false;
    bool ok = true;
    std::string fail_reason;
    int64_t grid_h = 0, grid_w = 0;
    double preproc_ms = 0, tower_ms = 0;
    double image_bytes = 0;
    int64_t patches = 0, merged_tokens = 0;
    bool pv_present = false, pv_exact = false, pv_shape_ok = false, pv_cos_ok = false;
    StageStats pv_stats;
    std::vector<CaseStage> stages;
};

// One image through preprocessing + tower, compared against the case's dumps. case_dir "" = plain encode
// (no golden), which still runs and returns timings. image_root re-roots the meta's absolute image
// path on its /images/ tail onto a local mirror of the golden tree, so a tree generated anywhere
// runs unchanged. run == nullptr (preproc_only): stop after the pixel_values compare -
// the tower half is never reached, and a case passes iff its pixel_values are exact (no engine rounding
// happens before the tower input, so exactness is the bar here; a difference says the resize rule or
// filter differs, which the downstream stages could never separate from a tower bug).
CaseResult run_case(TowerRun *run, const vision::PreprocConfig &cfg, const FloorDoc &floors,
                    const std::string &case_name, const std::string &image_path, const std::string &case_dir,
                    int64_t hf_grid_h, int64_t hf_grid_w, const std::string &image_root,
                    const std::string &feed, const std::string &dump_pv = {}, const std::string &dump_st = {}) {
    const char *fn = "golden";
    CaseResult r;
    r.name = case_name;
    r.image = image_path;

    std::string path = image_path;  // re-root the meta's absolute path on its /images/ tail
    if (!image_root.empty() && !path.empty() && path[0] == '/') {
        const std::string::size_type pos = path.find("/images/");
        if (pos != std::string::npos) path = image_root + path.substr(pos);
    }
    const std::vector<uint8_t> bytes = read_bytes(path);
    r.image_bytes = (double)bytes.size();
    const double t_pre0 = now_ms();
    vision::Preprocessed img;
    try {
        img = vision::preprocess_image(bytes, cfg);
    } catch (const Error &e) {
        r.ok = false;
        r.ran = false;
        r.fail_reason = std::string("preprocess refused: ") + e.what();
        return r;
    }
    r.preproc_ms = now_ms() - t_pre0;
    r.grid_h = img.grid_h;
    r.grid_w = img.grid_w;
    r.patches = img.grid_h * img.grid_w;
    r.merged_tokens = img.merged_tokens;
    bool grid_mismatch =
        hf_grid_h > 0 && (hf_grid_h != img.grid_h || hf_grid_w != img.grid_w);
    if (grid_mismatch)
        r.fail_reason = "grid " + std::to_string(img.grid_h) + "x" + std::to_string(img.grid_w) +
                        " != HF golden grid " + std::to_string(hf_grid_h) + "x" + std::to_string(hf_grid_w);

    // 1. The input stage: preprocessed pixel_values vs the HF FP32 dump (preprocessing on its own, before
    //    anything of ours is rounded).
    const std::string pv_path = case_dir + "/pixel_values.f32.npy";
    std::vector<float> golden_pv;  // the dump's own tower input (the golden feed)
    if (!case_dir.empty() && file_exists(pv_path)) {
        r.pv_present = true;
        NpyArray pv = load_npy_f4(pv_path);
        if (pv.data.size() != img.pixel_values.size()) {
            r.pv_stats.n = (int64_t)pv.data.size();
            r.pv_shape_ok = false;
            r.ok = false;
            r.fail_reason = (r.fail_reason.empty() ? "" : r.fail_reason + "; ") + "pixel_values shape " +
                            pv.shape_str() + " vs strix [n=" + std::to_string(r.patches) +
                            (run ? ", " + std::to_string(run->tower.dims().patch_in) : "") +
                            "] (the resize rule or filter differs)";
        } else {
            r.pv_shape_ok = true;
            r.pv_stats = compare_stages(img.pixel_values, pv.data);
            r.pv_exact = r.pv_stats.n_exact == r.pv_stats.n;
            // pixel_values has no BF16 twin (nothing of ours is rounded before it): the floor is missing,
            // so the card's provisional rule applies - cosine >= 0.9995 - with exactness reported.
            r.pv_cos_ok = r.pv_stats.cos >= 0.9995;
            if (!dump_pv.empty())  // the arrays, not just the stats: max_abs and n_exact cannot say whether
                                   // the delta is 1 FP32 ulp spread over the tensor (the normalise's op
                                   // order) or whole pixels off by a uint8 level (the resize's rounding)
                write_npy_f4(dump_pv + "/" + case_name + ".pixel_values.f32.npy", img.pixel_values, pv.shape);
            golden_pv = std::move(pv.data);
        }
    }
    if (r.pv_present && r.pv_shape_ok && !r.pv_cos_ok) {
        r.ok = false;
        r.fail_reason = (r.fail_reason.empty() ? "" : r.fail_reason + "; ") + "pixel_values cos " +
                        std::to_string(r.pv_stats.cos) + " < 0.9995 (floor missing for the input; the "
                        "provisional cosine rule says the preprocessing itself diverges)";
    }

    // preproc_only: the verdict is the pixel_values line alone (cosine rule above, exactness reported).
    if (run == nullptr) {
        if (!r.pv_present) {
            r.ok = false;
            r.fail_reason = (r.fail_reason.empty() ? "" : r.fail_reason + "; ") + "no pixel_values dump";
        }
        return r;
    }

    // 2. The tower. With a grid mismatch, HF's stage dumps belong to a different tensor shape - compare
    //    nothing, the case already failed on the grid/pv line.
    if (grid_mismatch) {
        r.ok = false;
        return r;
    }
    // Feed: the golden's own pixel_values (default - the stage dumps are the forward of THAT input, so
    // the tower half of the gate is apples-to-apples and preprocessing stays its own line) or our
    // re-preprocessed pixels (--feed preprocessed, the end-to-end serve chain).
    const vision::Preprocessed *tower_in = &img;
    vision::Preprocessed golden_feed;
    if (feed == "golden" && r.pv_present && r.pv_shape_ok) {
        golden_feed.pixel_values = std::move(golden_pv);
        golden_feed.grid_t = 1;
        golden_feed.grid_h = r.grid_h;
        golden_feed.grid_w = r.grid_w;
        golden_feed.resized_height = img.resized_height;
        golden_feed.resized_width = img.resized_width;
        golden_feed.merged_tokens = r.merged_tokens;
        tower_in = &golden_feed;
    }
    std::map<std::string, std::vector<float>> stages;
    std::vector<float> merged;
    std::vector<std::string> want;
    if (!case_dir.empty()) want.assign(kStageFiles, kStageFiles + kNumStages);
    r.tower_ms = run->encode(*tower_in, want, stages, &merged);
    r.ran = true;

    // 3. Every dumped stage against both precisions; floor recomputed from the dumps, cross-checked with
    //    floor.json (they must agree - same files).
    for (int si = 0; si < kNumStages; ++si) {
        const std::string stage = kStageFiles[si];
        CaseStage cs;
        cs.name = stage;
        const std::string f32_path = case_dir + "/" + stage + ".f32.npy";
        if (case_dir.empty() || !file_exists(f32_path)) {
            r.stages.push_back(std::move(cs));  // present = false
            continue;
        }
        cs.present = true;
        NpyArray g32 = load_npy_f4(f32_path);
        NpyArray g16;
        const std::string bf16_path = case_dir + "/" + stage + ".bf16.npy";
        if (file_exists(bf16_path)) g16 = load_npy_f4(bf16_path);
        cs.shape = g32.shape_str();

        const std::vector<float> *strix = stage == "merger" ? &merged : nullptr;
        if (strix == nullptr) {
            auto it = stages.find(stage);
            if (it != stages.end()) strix = &it->second;
        }
        const int64_t m2 = run->tower.dims().merge * run->tower.dims().merge;
        if (strix == nullptr || strix->size() != g32.data.size() || g32.shape.size() != 2 ||
            g32.shape[0] != (stage == "merger" ? r.patches / m2 : r.patches)) {
            cs.shape_ok = false;
            r.ok = false;
            r.fail_reason = (r.fail_reason.empty() ? "" : r.fail_reason + "; ") + "stage '" + stage +
                            "' golden shape " + cs.shape + ", strix produced " +
                            std::to_string(strix ? strix->size() : 0) + " elements";
            r.stages.push_back(std::move(cs));
            continue;
        }
        cs.shape_ok = true;
        cs.vs_f32 = compare_stages(*strix, g32.data);
        if (!dump_st.empty())  // our stage tensor, widened to FP32 (what the probe read back), in the
                               // golden's own shape - the input of the WHERE analysis
            write_npy_f4(dump_st + "/" + case_name + "." + stage + ".f32.npy", *strix, g32.shape);
        if (!g16.data.empty()) {
            STRIX_CHECK(g16.data.size() == g32.data.size(), fn, ": ", case_name, "/", stage, " bf16 dump has ",
                        g16.data.size(), " elements, f32 dump has ", g32.data.size());
            cs.vs_bf16 = compare_stages(*strix, g16.data);
            const StageStats fl = compare_stages(g16.data, g32.data);
            cs.floor_abs = fl.max_abs;
            cs.floor_cos = fl.cos;
            cs.has_floor = cs.floor_abs > 0;
            auto it = floors.per_image.find(case_name);
            if (it != floors.per_image.end()) {
                auto st = it->second.find(stage);
                if (st != it->second.end()) {
                    const double rel =
                        cs.floor_abs > 0 ? std::fabs(cs.floor_abs - st->second) / cs.floor_abs : (st->second > 0 ? 1.0 : 0.0);
                    STRIX_CHECK(rel < 1e-6, fn, ": floor.json ", case_name, ".", stage, " max_abs ",
                                st->second, " vs recomputed from the dumps ", cs.floor_abs,
                                " - the doc and the files disagree, which is the gate source?");
                }
            }
        }
        if (cs.has_floor) {
            cs.ratio = cs.vs_f32.max_abs / cs.floor_abs;
            cs.ratio_ok = cs.ratio <= 1.5;
        }
        if (!cs.ratio_ok) {
            r.ok = false;
            r.fail_reason = (r.fail_reason.empty() ? "" : r.fail_reason + "; ") + "stage '" + stage + "': " +
                            (cs.has_floor
                                 ? "ratio " + std::to_string(cs.ratio) + " > 1.5 (max|strix-hf32| " +
                                       std::to_string(cs.vs_f32.max_abs) + " vs floor " + std::to_string(cs.floor_abs) + ")"
                                 : "no BF16 twin in the golden - floor missing, ratio rule not evaluable");
        }
        r.stages.push_back(std::move(cs));
    }
    return r;
}

// ---------------------------------------------------------------- per-case meta + JSON output

struct Meta {
    std::vector<std::string> images;  // meta.json "image" (photos) or "images" [..] (the single-image dirs)
    int64_t grid_h = 0, grid_w = 0;
};

Meta read_meta(const std::string &dir) {
    Meta m;
    const json::Value v = json::Value::parse(read_text(dir + "/meta.json"));
    if (const json::Value *im = v.find("image"); im != nullptr && im->is_string())
        m.images.push_back(im->as_string(dir + ".image"));
    if (const json::Value *im = v.find("images"); im != nullptr && im->is_array())
        for (const auto &s : im->as_array(dir + ".images"))
            if (s.is_string()) m.images.push_back(s.as_string(dir + ".images[]"));
    if (const json::Value *g = v.find("grid_thw");
        g != nullptr && g->is_array() && !g->as_array("grid_thw").empty()) {
        const auto &one = g->as_array("grid_thw")[0];
        STRIX_CHECK(one.is_array() && one.as_array("grid_thw[0]").size() == 3, dir,
                    ": meta.json grid_thw must be [[t, h, w]]");
        m.grid_h = one.as_array("grid_thw[0]")[1].as_int(dir + ".grid_thw[1]");
        m.grid_w = one.as_array("grid_thw[0]")[2].as_int(dir + ".grid_thw[2]");
    }
    return m;
}

json::Value case_to_json(const CaseResult &r) {
    json::Value c = json::Value::object();
    c.set("case", json::Value::string(r.name));
    c.set("image", json::Value::string(r.image));
    c.set("image_bytes", json::Value::number(r.image_bytes));
    {
        json::Value g = json::Value::array();
        g.push(json::Value::integer(r.grid_h));
        g.push(json::Value::integer(r.grid_w));
        c.set("grid", std::move(g));
    }
    c.set("patches", json::Value::integer(r.patches));
    c.set("merged_tokens", json::Value::integer(r.merged_tokens));
    c.set("preproc_ms", json::Value::number(r.preproc_ms));
    c.set("tower_ms", json::Value::number(r.tower_ms));
    c.set("us_per_token", json::Value::number(r.merged_tokens > 0 ? r.tower_ms * 1000.0 / (double)r.merged_tokens
                                                                  : 0.0));
    c.set("ok", json::Value::boolean(r.ok));
    c.set("fail_reason", json::Value::string(r.fail_reason));
    {
        json::Value pv = json::Value::object();
        pv.set("present", json::Value::boolean(r.pv_present));
        pv.set("shape_ok", json::Value::boolean(r.pv_shape_ok));
        pv.set("exact", json::Value::boolean(r.pv_exact));
        pv.set("n", json::Value::integer(r.pv_stats.n));
        pv.set("n_exact", json::Value::integer(r.pv_stats.n_exact));
        pv.set("max_abs", json::Value::number(r.pv_stats.max_abs));
        pv.set("mean_abs", json::Value::number(r.pv_stats.mean_abs));
        pv.set("cos", json::Value::number(r.pv_stats.cos));
        c.set("pixel_values", std::move(pv));
    }
    json::Value st = json::Value::array();
    for (const CaseStage &cs : r.stages) {
        json::Value e = json::Value::object();
        e.set("stage", json::Value::string(cs.name));
        e.set("present", json::Value::boolean(cs.present));
        e.set("shape_ok", json::Value::boolean(cs.shape_ok));
        e.set("shape", json::Value::string(cs.shape));
        e.set("strix_vs_hf32_max_abs", json::Value::number(cs.vs_f32.max_abs));
        e.set("strix_vs_hf32_mean_abs", json::Value::number(cs.vs_f32.mean_abs));
        e.set("strix_vs_hf32_cos", json::Value::number(cs.vs_f32.cos));
        e.set("strix_vs_hf16_max_abs", json::Value::number(cs.vs_bf16.max_abs));
        e.set("strix_vs_hf16_cos", json::Value::number(cs.vs_bf16.cos));
        e.set("floor_hf16_vs_hf32_max_abs", json::Value::number(cs.floor_abs));
        e.set("floor_cos", json::Value::number(cs.floor_cos));
        e.set("has_floor", json::Value::boolean(cs.has_floor));
        e.set("ratio", json::Value::number(cs.ratio));
        e.set("ratio_ok", json::Value::boolean(cs.ratio_ok));
        st.push(std::move(e));
    }
    c.set("stages", std::move(st));
    return c;
}

// ---------------------------------------------------------------- bench

void run_bench(TowerRun &run, const Options &opt, const vision::PreprocConfig &cfg) {
    STRIX_CHECK(!opt.bench_src.empty(), "--bench: --bench-src <image file> is required");
    const std::vector<uint8_t> bytes = read_bytes(opt.bench_src);
    const vision::ImageU8 src = vision::decode_image(bytes, cfg);
    std::printf("bench: source %s (%lldx%lld %s)\n", opt.bench_src.c_str(), (long long)src.width,
                (long long)src.height, src.format.c_str());
    json::Value out = json::Value::object();
    for (const double mp : opt.bench_sizes) {
        // nearest square target on the 32-px factor grid (tall or wide frames are the image's problem, the
        // bench sizes the plan asks for are square pixel budgets)
        const int64_t side = (int64_t)std::llround(std::sqrt(mp * 1e6) / 32.0) * 32;
        STRIX_CHECK(side > 0, "bench size ", mp, " MP rounds to side ", side);
        const vision::ImageU8 im = vision::resize_bilinear_u8(src, side, side);
        const double tp0 = now_ms();
        const vision::Preprocessed img = vision::preprocess_image(im, cfg);
        const double preproc_ms = now_ms() - tp0;
        std::vector<std::string> no_stages;
        std::map<std::string, std::vector<float>> no_readback;
        std::vector<double> ms;
        for (int64_t i = 0; i < opt.repeat; ++i) ms.push_back(run.encode(img, no_stages, no_readback, nullptr));
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        const int64_t tok = img.merged_tokens;
        std::printf("  %6.2f MP -> %lldx%lld, grid %lldx%lld, %lld patches, %lld tokens: preprocess %.1f ms, "
                    "tower %.2f ms med (%.2f min / %.2f max of %lld) = %.2f us/token\n",
                    mp, (long long)img.resized_height, (long long)img.resized_width, (long long)img.grid_h,
                    (long long)img.grid_w, (long long)(img.grid_h * img.grid_w), (long long)tok, preproc_ms,
                    med, ms.front(), ms.back(), (long long)opt.repeat,
                    tok > 0 ? med * 1000.0 / (double)tok : 0.0);
        json::Value e = json::Value::object();
        e.set("mp_target", json::Value::number(mp));
        e.set("resized_h", json::Value::integer(img.resized_height));
        e.set("resized_w", json::Value::integer(img.resized_width));
        e.set("grid_h", json::Value::integer(img.grid_h));
        e.set("grid_w", json::Value::integer(img.grid_w));
        e.set("patches", json::Value::integer(img.grid_h * img.grid_w));
        e.set("tokens", json::Value::integer(tok));
        e.set("preproc_ms", json::Value::number(preproc_ms));
        e.set("tower_ms_med", json::Value::number(med));
        e.set("tower_ms_min", json::Value::number(ms.front()));
        e.set("tower_ms_max", json::Value::number(ms.back()));
        e.set("us_per_token", json::Value::number(tok > 0 ? med * 1000.0 / (double)tok : 0.0));
        e.set("reps", json::Value::integer(opt.repeat));
        std::string key = std::to_string((long long)std::llround(mp * 100));
        out.set(key, std::move(e));
    }
    if (!opt.results_dir.empty()) {
        std::ofstream f(opt.results_dir + "/bench.json", std::ios::trunc);
        STRIX_CHECK(f.good(), "--bench: can't write '", opt.results_dir, "/bench.json'");
        f << out.dump() << "\n";
    }
}

// ---------------------------------------------------------------- main

void usage() {
    std::printf(
        "strix_vision: image -> preprocess -> VisionTower::forward -> merger; G1 golden gate.\n"
        "  --weights W.strixw --vision V.strixw --config config.json --preproc preprocessor_config.json\n"
        "  --image FILE [--out PREFIX]     encode one image -> PREFIX.bin ([N,2560] bf16), .f32.npy, .json\n"
        "  --golden-root DIR [--floor F.json] [--results-dir DIR] [--image-root DIR]   the G1 gate per case\n"
        "  [--feed golden|preprocessed]   tower input: the golden's pixel_values (default; the stage\n"
        "                                 dumps are that input's forward) or our re-preprocessed pixels\n"
        "  [--preproc-only]               compare pixel_values only (no tower, no GPU)\n"
        "  [--dump-pv DIR] [--dump-stages DIR]  also write our pixel_values / stage tensors as <case>.*\n"
        "                                 .f32.npy into these existing directories (the where-analysis)\n"
        "  --golden CASEDIR --image FILE   one golden case by hand\n"
        "  --bench --bench-src FILE [--bench-sizes 0.25,1,2.5] [--repeat 5]\n"
        "  [--act bf16|f32] [--min-pixels N] [--max-pixels N] [--max-patches N] [--resize bicubic_aa|bilinear]\n"
        "  attention caps: f32 n <= %lld patches (kMaxVisionSeq), bf16 WMMA n <= %lld (kMaxVisionSeqWmma)\n",
        (long long)kernels::kMaxVisionSeq, (long long)kernels::kMaxVisionSeqWmma);
}

}  // namespace

namespace strix {
namespace {

int real_main(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "-h" || s == "--help") {
            usage();
            return 0;
        }
        STRIX_CHECK(s.rfind("--", 0) == 0, "argument '", s, "': expected --key [value]");
        std::string k = s.substr(2);
        // a value starting with -- is the next flag, not this one's argument (a flag's value is "1")
        if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
            a.kv[k] = "1";
        else
            a.kv[k] = argv[++i];
    }
    Options opt;
    opt.weights = a.get("weights");
    opt.vision = a.get("vision");
    opt.config = a.get("config");
    opt.preproc = a.get("preproc");
    opt.image = a.get("image");
    opt.out = a.get("out");
    opt.golden = a.get("golden");
    opt.golden_root = a.get("golden-root");
    opt.floor = a.get("floor");
    opt.results_dir = a.get("results-dir");
    opt.bench_src = a.get("bench-src");
    opt.image_root = a.get("image-root");
    if (a.has("feed")) {
        STRIX_CHECK(a.get("feed") == "golden" || a.get("feed") == "preprocessed",
                    "--feed: expected golden or preprocessed, got '", a.get("feed"), "'");
        opt.feed = a.get("feed");
    }
    STRIX_CHECK(!opt.config.empty() && !opt.preproc.empty(),
                "strix_vision: --config and --preproc are required");
    if (!a.has("preproc-only"))
        STRIX_CHECK(!opt.weights.empty() && !opt.vision.empty(),
                    "strix_vision: --weights and --vision are required (unless --preproc-only)");
    if (a.has("act")) {
        if (a.get("act") == "f32")
            opt.act = kernels::Act::F32;
        else
            STRIX_CHECK(a.get("act") == "bf16", "--act: expected bf16 or f32, got '", a.get("act"), "'");
    }
    opt.min_pixels = a.num("min-pixels", opt.min_pixels);
    opt.max_pixels = a.num("max-pixels", opt.max_pixels);
    opt.max_patches = a.num("max-patches", opt.max_patches);
    if (a.has("resize")) {
        if (a.get("resize") == "bilinear")
            opt.resize = vision::ResizeFilter::Bilinear;
        else
            STRIX_CHECK(a.get("resize") == "bicubic_aa", "--resize: expected bicubic_aa or bilinear, got '",
                        a.get("resize"), "'");
    }
    if (a.has("bench-sizes")) {
        opt.bench_sizes.clear();
        for (const std::string &x : split(a.get("bench-sizes"), ',')) opt.bench_sizes.push_back(std::stod(x));
        STRIX_CHECK(!opt.bench_sizes.empty(), "--bench-sizes: empty");
    }
    opt.repeat = a.num("repeat", 5);
    STRIX_CHECK(opt.repeat >= 1, "--repeat: expected >= 1, got ", opt.repeat);
    opt.bench = a.has("bench");
    opt.dump_pv = a.get("dump-pv");
    opt.dump_stages = a.get("dump-stages");
    for (const std::pair<const char *, std::string *> d :
         {std::make_pair("dump-pv", &opt.dump_pv), std::make_pair("dump-stages", &opt.dump_stages)})
        if (!d.second->empty()) {
            DIR *h = opendir(d.second->c_str());
            STRIX_CHECK(h != nullptr, "--", d.first, ": can't open '", *d.second,
                        "' (the caller creates the directory)");
            closedir(h);
        }
    opt.preproc_only = a.has("preproc-only");
    STRIX_CHECK(!opt.preproc_only || !opt.golden_root.empty() || !opt.golden.empty(),
                "--preproc-only: the pixel_values comparison is the golden check - give --golden-root or "
                "--golden (encoding and --bench need the tower, so the GPU)");
    STRIX_CHECK(!opt.bench || !opt.preproc_only, "--bench needs the tower; --preproc-only excludes it");
    STRIX_CHECK(!opt.image.empty() || !opt.golden_root.empty() || opt.bench,
                "strix_vision: nothing to do - give --image, --golden-root or --bench");

    // The engine's config path (the checkpoint's own files), with the golden's size rule applied on top:
    // min 65536 / the serve cap 2621440 px / no patch clamp (the goldens are at serve resolution; serve
    // parity is --max-patches 8192, the F32 attention clamp).
    const vision::ServedVision sv = vision::load_vision_config(opt.preproc, opt.config, opt.max_patches);
    vision::PreprocConfig cfg = sv.cfg;
    cfg.min_pixels = opt.min_pixels;
    cfg.max_pixels = opt.max_pixels;
    cfg.max_patches = opt.max_patches;
    cfg.resize = opt.resize;
    cfg.validate();

    // The tower (and with it the GPU) is built only when the run needs it: --preproc-only compares
    // pixel_values, which lives entirely host-side.
    std::unique_ptr<TowerRun> run;
    if (!opt.preproc_only) run = std::make_unique<TowerRun>(opt.weights, opt.vision, opt.act);
    std::printf("strix_vision: %s, act %s, resize %s, min %lld max %lld px (effective %lld), max_patches %lld\n",
                opt.preproc_only ? "preproc-only (no tower, no GPU)" : "tower",
                opt.act == kernels::Act::F32 ? "f32" : "bf16",
                cfg.resize == vision::ResizeFilter::BicubicAntialias ? "bicubic_aa" : "bilinear",
                (long long)cfg.min_pixels, (long long)cfg.max_pixels, (long long)cfg.effective_max_pixels(),
                (long long)cfg.max_patches);
    if (run) {
        const VisionTowerDims &D = run->tower.dims();
        std::printf("tower %s: depth %lld hidden %lld heads %lld inter %lld patch_in %lld merge %lld out %lld "
                    "pos %lldx%lld\n",
                    opt.vision.c_str(), (long long)D.depth, (long long)D.hidden, (long long)D.heads,
                    (long long)D.inter, (long long)D.patch_in, (long long)D.merge, (long long)D.out_hidden,
                    (long long)D.pos_side, (long long)D.pos_side);
    }

    const FloorDoc floors = load_floor(opt.floor);
    std::vector<CaseResult> results;

    if (!opt.golden_root.empty()) {
        // Every subdirectory with a meta.json and tower dumps is a single-image golden case (the multi-
        // image dirs have no per-image pixel_values file and are out of this gate's scope).
        std::vector<std::string> names;
        DIR *d = opendir(opt.golden_root.c_str());
        STRIX_CHECK(d != nullptr, "--golden-root: can't open '", opt.golden_root, "'");
        while (struct dirent *e = readdir(d))
            if (e->d_name[0] != '.') names.push_back(e->d_name);
        closedir(d);
        std::sort(names.begin(), names.end());
        for (const std::string &name : names) {
            const std::string dir = opt.golden_root + "/" + name;
            if (!file_exists(dir + "/meta.json") || !file_exists(dir + "/pixel_values.f32.npy")) {
                std::printf("skipping %s (no meta.json / pixel_values dump: not a single-image case)\n",
                            name.c_str());
                continue;
            }
            CaseResult cr;
            cr.name = name;
            try {
                const Meta m = read_meta(dir);
                STRIX_CHECK(m.images.size() == 1, name, "/meta.json: expected exactly one image in this "
                            "single-image case, got ", m.images.size());
                cr = run_case(run.get(), cfg, floors, name, m.images[0], dir, m.grid_h, m.grid_w,
                              opt.image_root, opt.feed, opt.dump_pv, opt.dump_stages);
            } catch (const std::exception &ex) {
                // one case's problem (e.g. over the naive-F32 attention cap) must not end the gate - it
                // fails as a case with the reason printed, the table stays complete
                cr.name = name;
                cr.ok = false;
                cr.fail_reason = ex.what();
            }
            results.push_back(std::move(cr));
        }
    } else if (!opt.golden.empty()) {
        const Meta m = read_meta(opt.golden);
        STRIX_CHECK(!opt.image.empty(), "--golden: --image is required");
        // The case name is the golden directory's own name, not a literal "case": the name is the key
        // floor.json is looked up by, and a per-case --dump-stages loop would otherwise land every
        // case's arrays on one filename, each run silently overwriting the last (measured 2026-10-16).
        std::string name = opt.golden;
        while (name.size() > 1 && name.back() == '/') name.pop_back();
        const std::size_t slash = name.find_last_of('/');
        if (slash != std::string::npos && slash + 1 < name.size()) name = name.substr(slash + 1);
        results.push_back(run_case(run.get(), cfg, floors, name, opt.image, opt.golden, m.grid_h,
                                   m.grid_w, opt.image_root, opt.feed, opt.dump_pv, opt.dump_stages));
    } else {
        // The plain verb: encode one image, embeddings out.
        STRIX_CHECK(!opt.image.empty(), "--image: not found");
        const VisionTowerDims &D = run->tower.dims();
        const double t_pre0 = now_ms();
        vision::Preprocessed img = vision::preprocess_image(read_bytes(opt.image), cfg);
        const double preproc_ms = now_ms() - t_pre0;
        CaseResult r;
        r.name = "image";
        r.image = opt.image;
        r.grid_h = img.grid_h;
        r.grid_w = img.grid_w;
        r.patches = img.grid_h * img.grid_w;
        r.merged_tokens = img.merged_tokens;
        r.preproc_ms = preproc_ms;
        std::map<std::string, std::vector<float>> no_readback;
        std::vector<float> merged;
        r.tower_ms = run->encode(img, {}, no_readback, &merged);
        r.ran = true;
        results.push_back(r);
        if (!opt.out.empty()) {
            std::vector<uint16_t> bits(merged.size());
            for (size_t i = 0; i < merged.size(); ++i) bits[i] = f32_to_bf16_bits(merged[i]);
            write_file(opt.out + ".bin", bits.data(), bits.size() * 2);  // [N, 2560] BF16 (raw bits)
            write_npy_f4(opt.out + ".f32.npy", merged, {r.merged_tokens, D.out_hidden});
            json::Value info = case_to_json(r);
            info.set("embedding_rows", json::Value::integer(r.merged_tokens));
            info.set("embedding_cols", json::Value::integer(D.out_hidden));
            std::ofstream f(opt.out + ".json", std::ios::trunc);
            STRIX_CHECK(f.good(), "--out: can't write '", opt.out, ".json'");
            f << info.dump() << "\n";
            std::printf("wrote %s.bin + %s.f32.npy ([%lld, %lld]) + %s.json\n", opt.out.c_str(),
                        opt.out.c_str(), (long long)r.merged_tokens, (long long)D.out_hidden, opt.out.c_str());
        }
    }

    if (opt.bench) run_bench(*run, opt, cfg);

    if (results.empty()) return 0;

    // ---- the per-stage table (one column per stage, ratio with '!' when the stage failed; pixel_values
    // exactness in its own column - it has no floor, it is reported, not ratio-gated)
    std::printf("\n%-14s %-11s %-7s %7s %10s %10s %10s %10s %10s %10s  %s\n", "case", "grid", "tokens",
                "pixval", "patch_emb", "pos_added", "block0", "block13", "block26", "merger", "verdict");
    int64_t npass = 0, nfail = 0;
    double worst_ratio[kNumStages] = {0, 0, 0, 0, 0, 0};
    double worst_cos[kNumStages] = {2, 2, 2, 2, 2, 2};
    double pv_worst_max_abs = 0;
    int64_t pv_cases = 0, pv_exact_cases = 0;
    json::Value all = json::Value::array();
    for (const CaseResult &r : results) {
        char cell[16];
        auto fmt = [&](const char *nm) -> std::string {
            for (const CaseStage &cs : r.stages) {
                if (cs.name != nm) continue;
                if (!cs.present) return "     ---  ";
                if (!cs.shape_ok) return "  SHAPE!  ";
                if (!cs.has_floor) return " no-floor ";
                std::snprintf(cell, sizeof cell, "%9.3f%c", cs.ratio, cs.ratio_ok ? ' ' : '!');
                return std::string(cell);
            }
            return "     ---  ";
        };
        std::string pvs = !r.pv_present ? "  ---  "
                            : !r.pv_shape_ok ? " SHAPE "
                            : r.pv_exact     ? " exact"
                            : r.pv_cos_ok    ? " cos-ok"
                                             : "  BAD ";
        std::printf("%-14s %4lldx%-5lld %7lld %7s %s%s%s%s%s%s  %s\n", r.name.c_str(), (long long)r.grid_h,
                    (long long)r.grid_w, (long long)r.merged_tokens, pvs.c_str(),
                    fmt("patch_embed").c_str(), fmt("pos_added").c_str(), fmt("block0").c_str(),
                    fmt("block13").c_str(), fmt("block26").c_str(), fmt("merger").c_str(),
                    r.ran ? (r.ok ? "PASS" : "FAIL") : "NO-RUN");
        if (!r.fail_reason.empty()) std::printf("    fail: %s\n", r.fail_reason.c_str());
        if (r.pv_present && !r.pv_exact)
            std::printf("    pixel_values: max_abs %.3e, %lld/%lld exact, cos %.10f\n", r.pv_stats.max_abs,
                        (long long)r.pv_stats.n_exact, (long long)r.pv_stats.n, r.pv_stats.cos);
        if (r.ran) std::printf("               %.1f ms preprocess + %.1f ms tower = %.2f us/token\n",
                               r.preproc_ms, r.tower_ms,
                               r.merged_tokens > 0 ? r.tower_ms * 1000.0 / (double)r.merged_tokens : 0.0);
        if (r.ok)
            ++npass;
        else
            ++nfail;
        for (int i = 0; i < kNumStages && i < (int)r.stages.size(); ++i) {
            const CaseStage &cs = r.stages[i];
            if (cs.present && cs.shape_ok && cs.has_floor) {
                worst_ratio[i] = std::max(worst_ratio[i], cs.ratio);
                worst_cos[i] = std::min(worst_cos[i], cs.vs_f32.cos);
            }
        }
        if (r.pv_present && r.pv_shape_ok) {
            ++pv_cases;
            pv_exact_cases += r.pv_exact ? 1 : 0;
            pv_worst_max_abs = std::max(pv_worst_max_abs, r.pv_stats.max_abs);
        }
        if (!opt.results_dir.empty()) {
            json::Value c = case_to_json(r);
            std::ofstream f(opt.results_dir + "/" + r.name + ".json", std::ios::trunc);
            STRIX_CHECK(f.good(), "--results-dir: can't write '", opt.results_dir, "/", r.name, ".json'");
            f << c.dump() << "\n";
            all.push(c);
        }
    }
    std::printf("\nper-stage worst over all cases (ratio = max|strix-hf32| / floor(hf16-vs-hf32), gate 1.500):\n");
    std::printf("  %-12s : pixel_values vs hf32: %lld/%lld cases exact, worst max_abs %.3e%s\n", "pixel_values",
                (long long)pv_exact_cases, (long long)pv_cases, pv_worst_max_abs,
                opt.preproc_only
                    ? " (preproc-only: no floor for the input - provisional cosine rule 0.9995, exactness "
                      "reported)"
                    : " (no floor for the input: reported with the provisional cosine rule; the tower "
                      "stages carry the gate)");
    if (!opt.preproc_only) {
        for (int i = 0; i < kNumStages; ++i)
            std::printf("  %-12s : worst ratio %.3f%s\n", kStageFiles[i], worst_ratio[i],
                        worst_cos[i] < 1.5
                            ? (std::string(", worst cos(vs hf32) ") + std::to_string(worst_cos[i])).c_str()
                            : "");
        const double merger_cos = worst_cos[kNumStages - 1] < 1.5 ? worst_cos[kNumStages - 1] : 0.0;
        std::printf("merger cosine vs hf32 worst %.10f (card sanity floor 0.9995: %s)\n", merger_cos,
                    merger_cos >= 0.9995 ? "met" : "NOT met");
    }
    std::printf("%s: %lld/%lld cases PASS%s\n", opt.preproc_only ? "preproc-only verdict" : "G1 verdict",
                (long long)npass, (long long)(npass + nfail),
                opt.preproc_only
                    ? " (pixel_values: no floor -> provisional cosine 0.9995; the tower half still needs "
                      "the GPU run)"
                    : " (every dumped stage within 1.5 x the hf16-vs-hf32 floor)");
    if (!opt.results_dir.empty()) {
        json::Value sum = json::Value::object();
        sum.set("pass", json::Value::integer(npass));
        sum.set("fail", json::Value::integer(nfail));
        sum.set("preproc_only", json::Value::boolean(opt.preproc_only));
        sum.set("merger_cos_worst", json::Value::number(opt.preproc_only ? 0.0
                                                                         : worst_cos[kNumStages - 1] < 1.5
                                                                                ? worst_cos[kNumStages - 1]
                                                                                : 0.0));
        json::Value wr = json::Value::object(), wc = json::Value::object();
        for (int i = 0; i < kNumStages; ++i) {
            wr.set(kStageFiles[i], json::Value::number(worst_ratio[i]));
            wc.set(kStageFiles[i], json::Value::number(worst_cos[i] < 1.5 ? worst_cos[i] : 0.0));
        }
        sum.set("worst_ratio", std::move(wr));
        sum.set("worst_cos", std::move(wc));
        sum.set("pixel_values_cases", json::Value::integer(pv_cases));
        sum.set("pixel_values_exact_cases", json::Value::integer(pv_exact_cases));
        sum.set("pixel_values_worst_max_abs", json::Value::number(pv_worst_max_abs));
        sum.set("cases", std::move(all));
        std::ofstream f(opt.results_dir + "/golden-gate.json", std::ios::trunc);
        STRIX_CHECK(f.good(), "--results-dir: can't write golden-gate.json");
        f << sum.dump() << "\n";
    }
    return nfail == 0 ? 0 : 2;
}

}  // namespace
}  // namespace strix

int main(int argc, char **argv) {
    try {
        return strix::real_main(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "strix_vision FAILED: %s\n", e.what());
        return 1;
    }
}
