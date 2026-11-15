#pragma once

// The steering vector library (config steer-dir, STEER-2): <name>.f32 holds float32 values, <name>.json holds
// {"layer": L} (the layer it is added after, unless a request names another). A vector of d values is the same in
// every stream; one of H * d values is per stream, as strix_acts reads the "out" point. Loaded once onto the GPU.

#include "runtime/steer.hpp"
#include "serve/http_server.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace strix {

class SteerLibrary : public SteerResolver {
public:
    SteerLibrary(const std::string &dir, int64_t layers, int64_t H, int64_t d);
    std::shared_ptr<const SteerSet> resolve(const json::Value &steer) const override;
    std::vector<std::string> names() const;
    json::Value list() const override;
    json::Value reload() const override;

private:
    struct Vec {
        int64_t layer = 0;
        std::shared_ptr<const DeviceBuffer<float>> v;
        uint64_t hash = 0;
        int64_t values = 0;  // as the file holds them: d (every stream) or H * d
    };
    std::map<std::string, Vec> load() const;
    std::string dir_;
    int64_t layers_ = 0, H_ = 0, d_ = 0;
    mutable std::shared_mutex mu_;
    mutable std::map<std::string, Vec> vecs_;
};

}  // namespace strix
