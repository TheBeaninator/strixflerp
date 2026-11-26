#pragma once

#include "serve/json.hpp"

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace strix {

struct StoredResponse {
    std::string id;
    std::vector<int32_t> tokens;
    std::optional<std::string> instructions;
    std::optional<json::Value> tools;
    uint64_t config_hash = 0;
    int64_t created_at = 0;
};

class ResponseStore {
public:
    explicit ResponseStore(size_t max_entries = 256, size_t max_bytes = 256ull << 20);

    void put(StoredResponse resp);
    std::optional<StoredResponse> get(const std::string &id);
    void clear();
    size_t size() const;
    size_t bytes() const;

private:
    void prune_locked();

    size_t max_entries_;
    size_t max_bytes_;
    size_t current_bytes_ = 0;

    mutable std::mutex mu_;
    std::deque<std::string> lru_;  // oldest at front, newest at back
    std::unordered_map<std::string, StoredResponse> entries_;
};

}  // namespace strix
