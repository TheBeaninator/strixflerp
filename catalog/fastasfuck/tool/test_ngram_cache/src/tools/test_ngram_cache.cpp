#include "common/check.hpp"
#include "runtime/ngram_table.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

using namespace strix;

int main(int argc, char **argv) {
    namespace fs = std::filesystem;
    std::string default_path = "/models/Thinking-Cap-Qwen-3.8-FN-faf/ngram.table";
    if (!fs::exists(default_path) && fs::exists("/models/Thinking-Cap-Qwen-3.8-FN-strixw/ngram.table")) {
        default_path = "/models/Thinking-Cap-Qwen-3.8-FN-strixw/ngram.table";
    }
    const std::string path = argc > 1 ? argv[1] : default_path;
    std::cout << "Testing NgramTableRows compact Q8 + CLOCK on: " << path << std::endl;

    // Ground truth reference: cache_rows = 0 (always reads directly from NVMe)
    NgramTableRows ref(path, 0);
    const int64_t D = ref.row_dim();
    const int64_t total_rows = ref.file().info().rows;
    STRIX_CHECK(total_rows > 100000, "Table has too few rows: ", total_rows);

    std::cout << "Table rows: " << total_rows << ", row_dim: " << D << ", dtype: " << (int)ref.file().info().dtype << std::endl;
    std::cout << "cache_bytes_per_row: " << ref.cache_bytes_per_row() << " bytes" << std::endl;

    // Test subject: small cache of 1024 rows to aggressively force CLOCK evictions and backward-shift deletions
    const int64_t kTestCap = 1024;
    NgramTableRows test(path, kTestCap);
    STRIX_CHECK(test.cache_rows() == kTestCap, "Cache cap mismatch");

    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<int64_t> row_dist(0, std::min<int64_t>(total_rows - 1, 50000));
    std::uniform_int_distribution<int> batch_dist(1, 16);
    std::uniform_int_distribution<int> op_dist(0, 9);

    std::vector<uint16_t> out_ref, out_test;
    const int kTotalOps = 50000;
    std::cout << "Running " << kTotalOps << " operations with cap = " << kTestCap << "..." << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < kTotalOps; ++it) {
        int n = batch_dist(rng);
        std::vector<int64_t> rows(n);
        for (int i = 0; i < n; ++i) {
            rows[i] = row_dist(rng);
        }

        int op = op_dist(rng);
        if (op < 2) {
            // 20% prefetch
            test.prefetch(rows.data(), n);
        } else {
            // 80% gather
            out_ref.resize(n * D);
            out_test.resize(n * D);
            ref.gather(rows.data(), n, out_ref.data());
            test.gather(rows.data(), n, out_test.data());

            if (std::memcmp(out_ref.data(), out_test.data(), (size_t)(n * D * 2)) != 0) {
                std::cerr << "BIT MISMATCH at iteration " << it << " for rows:";
                for (int i = 0; i < n; ++i) std::cerr << " " << rows[i];
                std::cerr << std::endl;
                STRIX_FAIL("test_ngram_cache: bit mismatch detected!");
            }
        }

        if ((it + 1) % 10000 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
            NgramTableRows::Stats s = test.stats();
            std::cout << "Completed " << (it + 1) << " ops in " << elapsed << " ms | requested=" << s.requested
                      << ", unique=" << s.unique << ", hits=" << s.hits << ", reads=" << s.reads
                      << ", prefetched=" << s.prefetched << std::endl;
        }
    }

    NgramTableRows::Stats s = test.stats();
    std::cout << "50,000 ops completed with 100% exact bit match!" << std::endl;
    STRIX_CHECK(s.hits > 0, "Expected hits > 0, got ", s.hits);
    STRIX_CHECK(s.reads > 0, "Expected reads > 0, got ", s.reads);
    STRIX_CHECK(s.reads > (uint64_t)kTestCap, "Expected evictions: reads ", s.reads, " > cap ", kTestCap);

    // Test edge cases:
    std::cout << "Testing edge case: clear_cache()..." << std::endl;
    test.clear_cache();
    std::vector<int64_t> test_rows = {10, 20, 30, 40};
    out_ref.resize(4 * D);
    out_test.resize(4 * D);
    ref.gather(test_rows.data(), 4, out_ref.data());
    test.gather(test_rows.data(), 4, out_test.data());
    STRIX_CHECK(std::memcmp(out_ref.data(), out_test.data(), (size_t)(4 * D * 2)) == 0, "clear_cache check failed");

    std::cout << "Testing edge case: resize_cache(0)..." << std::endl;
    test.resize_cache(0);
    STRIX_CHECK(test.cache_rows() == 0, "cache_rows should be 0");
    test.gather(test_rows.data(), 4, out_test.data());
    STRIX_CHECK(std::memcmp(out_ref.data(), out_test.data(), (size_t)(4 * D * 2)) == 0, "resize_cache(0) check failed");

    std::cout << "Testing edge case: resize_cache(2048)..." << std::endl;
    test.resize_cache(2048);
    STRIX_CHECK(test.cache_rows() == 2048, "cache_rows should be 2048");
    for (int it = 0; it < 5000; ++it) {
        int n = batch_dist(rng);
        std::vector<int64_t> rows(n);
        for (int i = 0; i < n; ++i) rows[i] = row_dist(rng);
        out_ref.resize(n * D);
        out_test.resize(n * D);
        ref.gather(rows.data(), n, out_ref.data());
        test.gather(rows.data(), n, out_test.data());
        STRIX_CHECK(std::memcmp(out_ref.data(), out_test.data(), (size_t)(n * D * 2)) == 0, "resize_cache(2048) check failed");
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "ALL TESTS PASSED: CLOCK EVICTION & BACKWARD-SHIFT 100% OK!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
