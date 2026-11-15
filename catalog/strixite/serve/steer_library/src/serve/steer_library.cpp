#include "serve/steer_library.hpp"

#include "common/check.hpp"
#include "runtime/device_buffer.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <sstream>

namespace strix {
namespace {

uint64_t fnv(const void *p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const auto *b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

std::string slurp(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    STRIX_CHECK(f.good(), "steer library: cannot read ", p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

SteerLibrary::SteerLibrary(const std::string &dir, int64_t layers, int64_t H, int64_t d)
    : dir_(dir), layers_(layers), H_(H), d_(d) {
    vecs_ = load();
}

json::Value SteerLibrary::reload() const {
    std::map<std::string, Vec> fresh = load();  // outside the lock: a bad file throws and the old set stays
    {
        std::unique_lock<std::shared_mutex> lock(mu_);
        vecs_.swap(fresh);
    }
    return list();
}

std::map<std::string, SteerLibrary::Vec> SteerLibrary::load() const {
    const std::string &dir = dir_;
    const int64_t layers = layers_, H = H_, d = d_;
    std::map<std::string, Vec> vecs;
    namespace fs = std::filesystem;
    STRIX_CHECK(fs::is_directory(dir), "steer library: ", dir, " is not a directory");
    for (const auto &e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".f32") continue;
        const std::string name = e.path().stem().string();
        const std::string raw = slurp(e.path().string());
        const json::Value meta = json::Value::parse(slurp((e.path().parent_path() / (name + ".json")).string()));
        Vec v;
        v.layer = meta.find("layer") ? meta.find("layer")->as_int(name + ".json layer") : -1;
        STRIX_CHECK(v.layer >= 0 && v.layer < layers, "steer library: ", name, ": layer ", v.layer, " of ", layers);
        const size_t n = raw.size() / 4;
        STRIX_CHECK(raw.size() % 4 == 0 && (n == (size_t)d || n == (size_t)(H * d)), "steer library: ", name, ".f32 has ",
                    raw.size(), " bytes: expected ", d, " or ", H * d, " float32 values");
        std::vector<float> host((size_t)(H * d));
        for (size_t i = 0; i < host.size(); ++i) std::memcpy(&host[i], raw.data() + 4 * (n == (size_t)d ? i % (size_t)d : i), 4);
        for (float f : host) STRIX_CHECK(std::isfinite(f), "steer library: ", name, " has a non-finite value");
        auto buf = std::make_shared<DeviceBuffer<float>>(host.size(), "steer " + name);
        buf->upload(host);
        v.v = std::move(buf);
        v.hash = fnv(host.data(), host.size() * 4);
        v.values = (int64_t)n;
        vecs[name] = std::move(v);
    }
    return vecs;
}

json::Value SteerLibrary::list() const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    json::Value arr = json::Value::array();
    for (const auto &[name, v] : vecs_) {
        json::Value o = json::Value::object();
        char h[20];
        std::snprintf(h, sizeof h, "%016llx", (unsigned long long)v.hash);
        o.set("name", json::Value::string(name));
        o.set("layer", json::Value::integer(v.layer));
        o.set("values", json::Value::integer(v.values));
        o.set("hash", json::Value::string(h));
        arr.push(std::move(o));
    }
    json::Value b = json::Value::object();
    b.set("vectors", std::move(arr));
    return b;
}

std::vector<std::string> SteerLibrary::names() const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    std::vector<std::string> out;
    for (const auto &[k, v] : vecs_) out.push_back(k);
    return out;
}

std::shared_ptr<const SteerSet> SteerLibrary::resolve(const json::Value &steer) const {
    const std::vector<json::Value> &items = steer.as_array("steer");
    if (items.empty()) return nullptr;
    std::shared_lock<std::shared_mutex> lock(mu_);
    STRIX_CHECK(items.size() <= 16, "steer: ", items.size(), " vectors, at most 16");
    auto s = std::make_shared<SteerSet>();
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < items.size(); ++i) {
        const json::Value &it = items[i];
        const std::string at = "steer[" + std::to_string(i) + "]";
        STRIX_CHECK(it.is_object() && it.find("vector") && it.find("scale"), at, ": needs \"vector\" and \"scale\"");
        const std::string name = it.find("vector")->as_string(at + ".vector");
        auto v = vecs_.find(name);
        STRIX_CHECK(v != vecs_.end(), at, ": no vector named '", name, "' (", vecs_.size(), " loaded)");
        const double scale = it.find("scale")->as_double(at + ".scale");
        STRIX_CHECK(std::isfinite(scale) && std::fabs(scale) <= 1000, at, ": scale ", scale, " (|scale| <= 1000)");
        const int64_t layer = it.find("layer") ? it.find("layer")->as_int(at + ".layer") : v->second.layer;
        STRIX_CHECK(layer >= 0 && layer < layers_, at, ": layer ", layer, " of ", layers_);
        if (scale == 0) continue;
        s->layers.push_back(SteerLayer{layer, v->second.v, (float)scale});
        char row[160];
        std::snprintf(row, sizeof row, "%s%s@L%lld x%g (%016llx)", s->signature.empty() ? "" : ", ", name.c_str(),
                      (long long)layer, scale, (unsigned long long)v->second.hash);
        s->signature += row;
        h = fnv(&layer, sizeof layer, h), h = fnv(&scale, sizeof scale, h), h = fnv(&v->second.hash, 8, h);
    }
    if (s->layers.empty()) return nullptr;
    s->salt = -1 - (int32_t)(h % 0x3fffffffull);  // negative: never a real token id
    return s;
}

}  // namespace strix
