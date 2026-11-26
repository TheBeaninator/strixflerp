#include "serve/response_store.hpp"

#include <algorithm>

namespace strix {

namespace {

size_t calc_bytes(const StoredResponse &r) {
    size_t b = sizeof(StoredResponse) + r.id.size();
    b += r.tokens.size() * sizeof(int32_t);
    if (r.instructions.has_value()) b += r.instructions->size();
    return b;
}

}  // namespace

ResponseStore::ResponseStore(size_t max_entries, size_t max_bytes)
    : max_entries_(max_entries), max_bytes_(max_bytes) {}

void ResponseStore::prune_locked() {
    while (!lru_.empty() && (entries_.size() > max_entries_ || current_bytes_ > max_bytes_)) {
        const std::string oldest_id = lru_.front();
        lru_.pop_front();
        auto it = entries_.find(oldest_id);
        if (it != entries_.end()) {
            const size_t b = calc_bytes(it->second);
            current_bytes_ = current_bytes_ > b ? current_bytes_ - b : 0;
            entries_.erase(it);
        }
    }
}

void ResponseStore::put(StoredResponse resp) {
    std::lock_guard<std::mutex> lock(mu_);
    const size_t b = calc_bytes(resp);
    auto it = entries_.find(resp.id);
    if (it != entries_.end()) {
        const size_t old_b = calc_bytes(it->second);
        current_bytes_ = current_bytes_ > old_b ? current_bytes_ - old_b : 0;
        entries_.erase(it);
        lru_.erase(std::remove(lru_.begin(), lru_.end(), resp.id), lru_.end());
    }
    current_bytes_ += b;
    lru_.push_back(resp.id);
    entries_[resp.id] = std::move(resp);
    prune_locked();
}

std::optional<StoredResponse> ResponseStore::get(const std::string &id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return std::nullopt;
    lru_.erase(std::remove(lru_.begin(), lru_.end(), id), lru_.end());
    lru_.push_back(id);
    return it->second;
}

void ResponseStore::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    entries_.clear();
    lru_.clear();
    current_bytes_ = 0;
}

size_t ResponseStore::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

size_t ResponseStore::bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return current_bytes_;
}

}  // namespace strix
