// SPDX-License-Identifier: MIT
// strixflerp adapter runtime: what every launch-level adapter needs and nothing else.
//
//  CodeObject   a device code object of another engine, embedded in the binary by `flerp assemble` between the
//               symbols <symbol> and <symbol>_end, loaded on first use and launched with the HIP module API -
//               the way StrixFold (kit of raw code objects) and hipfire (hipcc --genco bundles) launch their own.
//  Tally        counts launches that went to the foreign provider and launches handed back to the origin's
//               kernel, and says so on stderr at the first launch and at exit. A gate reads these lines: a swap
//               that never ran is not a swap.
#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>

namespace flerp {

class CodeObject {
public:
    CodeObject(const unsigned char *begin, const unsigned char *end, const char *kernel)
        : begin_(begin), end_(end), kernel_(kernel) {}
    CodeObject(const CodeObject &) = delete;
    CodeObject &operator=(const CodeObject &) = delete;

    // args: one pointer per kernel parameter, in order (the HIP module API's kernelParams form).
    void launch(dim3 grid, dim3 block, hipStream_t stream, void **args, unsigned shared_bytes = 0) {
        std::call_once(once_, [this] { load(); });
        const hipError_t e = hipModuleLaunchKernel(fn_, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                                                   shared_bytes, stream, args, nullptr);
        if (e != hipSuccess) fail("hipModuleLaunchKernel", e);
    }
    size_t bytes() const { return (size_t)(end_ - begin_); }

private:
    void load() {
        if (end_ <= begin_) throw std::runtime_error(std::string("flerp code object for ") + kernel_ + " is empty");
        hipError_t e = hipModuleLoadData(&module_, begin_);
        if (e != hipSuccess) fail("hipModuleLoadData", e);
        e = hipModuleGetFunction(&fn_, module_, kernel_);
        if (e != hipSuccess) fail("hipModuleGetFunction", e);
    }
    [[noreturn]] void fail(const char *what, hipError_t e) const {
        throw std::runtime_error(std::string("flerp code object ") + kernel_ + ": " + what + " failed: " +
                                 hipGetErrorString(e));
    }
    const unsigned char *begin_, *end_;
    const char *kernel_;
    std::once_flag once_;
    hipModule_t module_ = nullptr;
    hipFunction_t fn_ = nullptr;
};

class Tally {
public:
    Tally(const char *adapter, const char *provider) : adapter_(adapter), provider_(provider) {}
    ~Tally() { report("exit"); }
    void foreign() {
        if (foreign_.fetch_add(1, std::memory_order_relaxed) == 0) report("first launch through the provider");
    }
    void origin() { origin_.fetch_add(1, std::memory_order_relaxed); }

private:
    void report(const char *when) const {
        std::fprintf(stderr, "flerp-adapter %s provider=%s foreign_launches=%llu origin_launches=%llu (%s)\n", adapter_,
                     provider_, (unsigned long long)foreign_.load(), (unsigned long long)origin_.load(), when);
        std::fflush(stderr);
    }
    const char *adapter_, *provider_;
    std::atomic<unsigned long long> foreign_{0}, origin_{0};
};

}  // namespace flerp
