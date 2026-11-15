// SPDX-License-Identifier: MIT
// The one symbol of gufo's src/core/image.cpp that the Qwen vision library names (vision/prompt.cpp and
// vision/image_cache.cpp call it; the text executor links that library). image.cpp itself needs libpng, libjpeg,
// libwebp and libcurl, so a text-only build without them (FLERP_GUFO_IMAGE=OFF) links this instead.
// A text prompt never reaches it; a build that is given an image says so.
#include <stdexcept>

#include "src/core/image.hpp"

namespace gufo::core {

Image DecodeImage(std::span<const std::uint8_t>) {
  throw std::runtime_error("image input is not built into this recipe (FLERP_GUFO_IMAGE=OFF)");
}

}  // namespace gufo::core
