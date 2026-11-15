#pragma once

// Activation steering (STEER-2): a conversation's vectors, added to the residual streams after chosen layers.
//
// A vector is one row of the streams (H * d values, the shape of x_ per token) and is added, times its scale, to
// every token row of the conversation's forwards at the end of its layer - prompt and generated tokens alike - so
// the state a steered conversation caches is the steered state. Such state must never resume an unsteered prompt
// (or one steered differently): the request's cache key carries the salt as its first token (a negative id, never a
// real token), so every prefix match - the prompt cache, a slot's live state, /cache/export - diverges at token 0.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strix {

template <typename T>
class DeviceBuffer;  // runtime/device_buffer.hpp: only the model side touches the values

struct SteerLayer {
    int64_t layer = 0;
    std::shared_ptr<const DeviceBuffer<float>> v;  // H * d floats
    float scale = 0.0f;
};

struct SteerSet {
    std::vector<SteerLayer> layers;
    std::string signature;  // what was asked, with each vector's content hash: the log row and the salt's source
    int32_t salt = 0;       // the cache key's token 0 for a prompt steered by exactly this set
};


}  // namespace strix
