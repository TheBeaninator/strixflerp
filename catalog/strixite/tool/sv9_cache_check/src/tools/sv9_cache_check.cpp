// SV-9 (card 01b207e967): the prompt cache is keyed on the image CONTENT, not on shape.
//
// The cache-key stream (serve/vision_request.cpp's plan_vision_prompt) replaces every image_pad run with two
// pseudo-id runs derived from a 64-bit hash of the image bytes; the engine resumes and saves on that stream
// (serve/engine.cpp), and a disk entry's header hashes exactly those tokens. This program proves it, host
// only (no GPU, no model weights):
//   1  two equal-size, content-different images give different digests and different key streams;
//   2  the equal-size different image does not hit an entry saved for the first image (PromptCache::best only
//      answers with a whole entry, so this is the sharpest form: NO hit at all; and the real prompts, keyed
//      by pad ids as v0.3.0 did, are equal end to end - the very collision the card names);
//   3  identical bytes give an identical stream and do hit (so the miss is content, never shape);
//   4  the pure pre-image text, keyed as its own entry, hits - a same-shape new image may reuse the text;
//   5  swapping two different images swaps the key stream (order matters, no cross-hit);
//   6  the same holds after a disk round trip (close, reopen with the RAM tier off): the streams live in the
//      file headers, and a different image still answers 0;
//   7  the images' grids and merged-token counts are equal (so 1-6 are a fair equal-size test).
//
// usage: sv9_cache_check <tokenizer.json> <preprocessor_config.json> <config.json> <scratch_dir>
// one ok/FAIL line per check; the exit status is the number of failures (2: usage).

#include "common/check.hpp"
#include "serve/prompt_cache.hpp"
#include "serve/tokenizer.hpp"
#include "serve/vision_request.hpp"

#include "stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace strix;
namespace vision = strix::vision;

namespace {

int failures = 0;
void check(bool ok, const char *what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    std::fflush(stdout);  // a refusal aborts the process: the lines already printed must be on disk
    if (!ok) ++failures;
}

// A deterministic RGB image: a gradient plus a per-variant pattern (the contents differ, the size does not),
// PNG-encoded through stb (third_party/stb_impl's test-only encode side). The digest is over the file bytes,
// so any pixel change is a digest change - and a re-encode of the same pixels is not.
std::vector<uint8_t> make_png(int variant, int w, int h) {
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t *p = rgb.data() + ((size_t)y * w + x) * 3;
            p[0] = (uint8_t)((x * 4 + variant * 37) & 0xff);
            p[1] = (uint8_t)((y * 7 + variant * 11) & 0xff);
            p[2] = (uint8_t)((variant == 1 ? ~(x ^ y) : (x ^ y)) & 0xff);
        }
    if (variant == 2) rgb[4096] ^= 0x80;  // one flipped pixel: variant 2 differs from 0, same dimensions
    std::vector<uint8_t> png;
    STRIX_CHECK(stbi_write_png_to_func([](void *c, void *d, int n) {
                    auto *v = (std::vector<uint8_t> *)c;
                    v->insert(v->end(), (const uint8_t *)d, (const uint8_t *)d + n);
                },
                                        &png, w, h, 3, rgb.data(), w * 3) != 0,
                "stbi_write_png_to_func failed");
    STRIX_CHECK(!png.empty(), "stb produced an empty PNG");
    return png;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <tokenizer.json> <preprocessor_config.json> <config.json> <scratch_dir>\n",
                     argv[0]);
        return 2;
    }
    const std::string tok_path = argv[1], pp = argv[2], cfgj = argv[3], dir = argv[4];
    std::filesystem::remove_all(dir);
    vision::ServedVision v = vision::load_vision_config(pp, cfgj, 1u << 30);  // no engine here: no patch cap
    const Tokenizer tok(tok_path);
    const int32_t VS = tok.id_of("<|vision_start|>"), VE = tok.id_of("<|vision_end|>"), IP = v.image_token_id;

    // Four PNGs, all from the same 512x384 source: A and B differ in content, A2 re-encodes A's pixels
    // byte-identically (a fresh upload of the same image), C differs from both.
    std::vector<vision::ParsedImage> parsed;
    for (int variant : {0, 1, 0, 2}) {
        vision::ParsedImage pi;
        pi.bytes = make_png(variant, 512, 384);
        pi.mime = "image/png";
        parsed.push_back(std::move(pi));
    }
    vision::ParsedImage &A = parsed[0], &B = parsed[1], &A2 = parsed[2], &C = parsed[3];
    std::printf("info: PNG bytes %zu / %zu / %zu / %zu; A2 re-encode byte-identical to A: %d\n", A.bytes.size(),
                B.bytes.size(), A2.bytes.size(), C.bytes.size(), (int)(A.bytes == A2.bytes));
    check(A.bytes != B.bytes && A.bytes != C.bytes && B.bytes != C.bytes, "fixture: A, B and C are content-different");
    check(A.bytes == A2.bytes, "check 3: A2 has A's bytes exactly");

    std::vector<vision::ServedImage> imgs = vision::prepare_images(parsed, v);
    check(imgs[0].prefill.grid_h == imgs[1].prefill.grid_h && imgs[0].prefill.grid_w == imgs[1].prefill.grid_w &&
              imgs[0].prefill.grid_h == imgs[3].prefill.grid_h,
          "check 7: A, B and C have the same grid (equal size)");
    check(imgs[0].merged_tokens == imgs[1].merged_tokens && imgs[0].merged_tokens > 1,
          "check 7: equal merged-token counts (> 1)");
    check(imgs[0].digest != imgs[1].digest && imgs[0].digest != imgs[3].digest,
          "check 1: equal-size different images digest differently");
    check(imgs[0].digest == imgs[2].digest, "check 3: identical bytes digest identically");

    // The prompt: fixed text around two vision triplets, pads expanded to the real merged counts. Which
    // images the blocks carry comes from the pair handed to plan().
    const std::vector<int32_t> text_a = tok.encode(std::string("Describe the first picture.\n"));
    const std::vector<int32_t> text_b = tok.encode(std::string("\nAnd the second one.\n"));
    const int64_t merged = imgs[0].merged_tokens;
    std::vector<int32_t> prompt;
    {
        const auto triplet = [&] {
            prompt.push_back(VS);
            prompt.insert(prompt.end(), (size_t)merged, IP);
            prompt.push_back(VE);
        };
        prompt.insert(prompt.end(), text_a.begin(), text_a.end());
        triplet();
        prompt.insert(prompt.end(), text_b.begin(), text_b.end());
        triplet();
    }
    int64_t pads = 0;
    for (const int32_t id : prompt) pads += id == IP;
    check(pads == 2 * merged, "fixture: the expanded prompt carries both pad runs");

    const auto plan = [&](const vision::ParsedImage &first, const vision::ParsedImage &second) {
        std::vector<vision::ParsedImage> two;
        two.push_back(vision::ParsedImage{std::vector<uint8_t>(first.bytes), first.mime});
        two.push_back(vision::ParsedImage{std::vector<uint8_t>(second.bytes), second.mime});
        std::vector<vision::ServedImage> im = vision::prepare_images(two, v);
        STRIX_CHECK(im[0].merged_tokens == merged && im[1].merged_tokens == merged,
                    "sv9_cache_check: a planned image changed the pad count (internal)");
        return vision::plan_vision_prompt(prompt, im, v.cfg, IP, 1u << 30);
    };
    const vision::VisionPromptPlan k_ab = plan(A, B), k_ba = plan(B, A), k_ac = plan(A, C),
                                     k_ca = plan(C, A), k_ab3 = plan(A2, B);

    // The key stream's layout: [text_a][VS][block][VE][text_b][VS][block][VE].
    const int64_t first_block = (int64_t)text_a.size(), TP = first_block + 1,
                  L = (int64_t)k_ab.cache_prompt.size();
    check(k_ab.cache_prompt[(size_t)first_block] == VS && k_ab.cache_prompt[(size_t)(TP + merged)] == VE &&
              k_ab.cache_prompt[(size_t)(TP + merged + 1)] != VS,
          "fixture: the first block sits between the markers, right after the text run");
    check(k_ab.cache_prompt[(size_t)TP] < 0 && k_ab.cache_prompt[(size_t)(TP + merged - 1)] < 0,
          "check 1: the pad run became negative pseudo-id rows");
    int64_t block_len = 0;
    while (k_ab.cache_prompt[(size_t)(TP + block_len)] != VE) ++block_len;
    check(block_len == merged, "fixture: the block between the markers is the merged pads");

    // checks 1, 3, 5 at the stream level - and the collision itself:
    check(k_ab.cache_prompt != k_ac.cache_prompt,
          "check 1: the second block holds an equal-size different image - different key stream");
    check(k_ab.cache_prompt == k_ab3.cache_prompt,
          "check 3: A2 (same bytes) in place of A leaves the key stream identical");
    int64_t common_ac = 0, common_ca = 0;
    while (common_ac < L && k_ab.cache_prompt[(size_t)common_ac] == k_ac.cache_prompt[(size_t)common_ac]) ++common_ac;
    while (common_ca < L && k_ab.cache_prompt[(size_t)common_ca] == k_ca.cache_prompt[(size_t)common_ca]) ++common_ca;
    std::printf("info: L %lld merged %lld TP %lld common_ac %lld common_ca %lld\n", (long long)L,
                (long long)merged, (long long)TP, (long long)common_ac, (long long)common_ca);
    // second block pseudo rows: [L - 1 - merged, L - 2] (the last VE closes the stream) - the split is there.
    check(common_ac == L - 1 - merged, "check 1: the differing pair splits inside the second block");
    check(common_ca == TP, "check 2: a different FIRST image shares only the leading text and vision_start");
    check((int64_t)k_ac.cache_prompt.size() == L && k_ab.rope_positions_reached == k_ca.rope_positions_reached,
          "check 2: equal length, equal position walk - same shape, different key");

    const auto fresh_state = [](size_t bytes, uint8_t fill) {
        HostBuffer b(bytes);
        std::memset(b.data(), fill, b.size());
        return b;
    };
    PromptCache::Options opts;
    opts.mem_available = [] { return 64ull << 30; };  // fakes: this runs beside everything else on the box
    opts.disk_free = [] { return 512ull << 30; };

    {  // RAM tier; the destructor flushes the entries to disk
        PromptCache cache(dir, 1ull << 30, "sv9-cache-check", opts);
        check(cache.put(k_ab.cache_prompt, PromptCache::Kind::Turn, fresh_state(1 << 16, 0x5a)),
              "check 4: put of the full key stream");
        check(cache.best(k_ab.cache_prompt, L) == L, "check 4: the stored stream hits itself (whole entry)");
        check(cache.best(k_ab3.cache_prompt, L) == L,
              "check 3: the same-bytes stream hits the stored entry - a hit is allowed");
        check(cache.best(k_ac.cache_prompt, L) == 0,
              "check 2 (cache): the equal-size different image does NOT hit - no entry, no prefix hit "
              "(the v0.3.0 pad-id key would have matched end to end)");
        check(cache.best(k_ba.cache_prompt, L) == 0,
              "check 5 (cache): the swapped-order stream does not hit the stored (A,B) entry");
        check(!cache.contains(k_ac.cache_prompt.data(), L) && !cache.contains(k_ba.cache_prompt.data(), L),
              "check 5: the different-content streams are not entries");
        // The pure text, keyed as an entry of its own: the real ids up to the first vision_start - exactly
        // what a text-only prefix entry holds. The streams share only those rows (common_ca == TP).
        std::vector<int32_t> text_ids(text_a.begin(), text_a.end());
        text_ids.push_back(VS);
        check((int64_t)text_ids.size() == TP, "fixture: the pre-image stream is the text plus vision_start");
        check(cache.put(std::vector<int32_t>(text_ids), PromptCache::Kind::System, fresh_state(1 << 12, 0x33)),
              "check 4: put of the pre-image text entry");
        check(cache.best(text_ids, TP) == TP, "check 4: the pre-image text hits its own entry");
        check(cache.best(k_ca.cache_prompt, L) == TP,
              "check 4: a different first image resumes exactly at the text entry - never past it");
        std::printf("info: full entry in_ram %d on_disk %d\n", (int)cache.in_ram(k_ab.cache_prompt.data(), L),
                    (int)cache.on_disk(k_ab.cache_prompt.data(), L));
    }  // the destructor writes the RAM entries to disk

    {  // check 6: reopen on the same directory, cold (ram_tier off: nothing may answer from RAM)
        PromptCache::Options cold = opts;
        cold.ram_tier = false;
        PromptCache cache(dir, 1ull << 30, "sv9-cache-check", cold);
        check(!cache.in_ram(k_ab.cache_prompt.data(), L) && cache.on_disk(k_ab.cache_prompt.data(), L),
              "check 6: the reopened cache starts cold and the image entry is a file");
        check(cache.best(k_ab.cache_prompt, L) == L, "check 6: after a reopen the stored stream still hits from disk");
        check(cache.best(k_ac.cache_prompt, L) == TP,
              "check 6: after a reopen the equal-size different image stops at the text entry - the file header "
              "hashes the content pseudo-ids, no hit past them");
        std::vector<int32_t> text_ids(text_a.begin(), text_a.end());
        text_ids.push_back(VS);
        check(cache.best(text_ids, TP) == TP, "check 6: after a reopen the text prefix still hits its own entry");
        check(cache.best(k_ca.cache_prompt, L) == TP, "check 6: after a reopen the stop-at-text still holds");
    }

    std::filesystem::remove_all(dir);
    std::printf("%s\n", failures == 0 ? "SV9-CACHE-CHECK PASS" : "SV9-CACHE-CHECK: FAILURES");
    return failures;
}
