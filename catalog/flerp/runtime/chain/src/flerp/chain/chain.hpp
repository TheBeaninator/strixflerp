// SPDX-License-Identifier: MIT
// runtime.flerp.chain: chained kernel launches on one HIP stream.
//
// A kernel launched with hipExtAnyOrderLaunch has no AQL barrier bit, so the command processor dispatches it without
// waiting for the previous kernel to drain; ordering is kept on the device instead: every block of a chained kernel
// waits in its prologue until the stream's completion counter reaches the blocks issued before its kernel, and adds
// itself when done. Same kernels, same arithmetic, only the schedule changes, so outputs are bit-identical by
// construction (decode-launch-design.md, section 4; radiance libhalo halo_chain.h is the same idea in raw AQL).
//
//   device:  __global__ void k(..., flerp::chain::Arg ca) { flerp::chain::wait(ca); ...work...; flerp::chain::done(ca); }
//   host:    flerp::chain::Chain ch(stream);  ch.launch(k, grid, block, lds, args...);   ...   ch.plain();  // after any
//            non-chained op (memcpy, memset, an unchained kernel): the next launch keeps its barrier bit
//            ch.check();  // after a sync: throws if a wait ever timed out
//
// Bounded wait: one lane polls; past `timeout` wall-clock ticks (default 1 s) it sets the error word and releases the
// chain (adds its block anyway and skips nothing else). It does not s_trap: a trap kills the queue, and the error word
// could then no longer be read. Once the error word is set every later wait returns at once, so one failed launch ends
// the whole chain within milliseconds with a readable error instead of a hang or a cascade of 1 s timeouts; the
// outputs of that forward are garbage and Chain::check() says so.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace flerp::chain {

struct Arg {
    unsigned long long *ctr = nullptr;   // the stream's completion counter (blocks finished)
    unsigned *err = nullptr;             // error word: 1 = a wait timed out
    unsigned long long target = 0;       // blocks issued before this kernel
    unsigned long long timeout = 0;      // wall-clock ticks
};

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
__device__ __forceinline__ void wait(const Arg &a) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const unsigned long long t0 = wall_clock64();
        while (__hip_atomic_load(a.ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < a.target) {
            if (__hip_atomic_load(a.err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u) break;
            if (wall_clock64() - t0 > a.timeout) {
                __hip_atomic_store(a.err, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                break;
            }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}
__device__ __forceinline__ void done(const Arg &a) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        __hip_atomic_fetch_add(a.ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif

class Chain {
public:
    explicit Chain(hipStream_t s, double timeout_s = 1.0) : s_(s) {
        ok(hipMalloc(&ctr_, sizeof(*ctr_)), "counter");
        ok(hipMalloc(&err_, sizeof(*err_)), "error word");
        int dev = 0, rate_khz = 0;
        ok(hipGetDevice(&dev), "device");
        ok(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, dev), "wall clock rate");
        timeout_ = (unsigned long long)(timeout_s * rate_khz * 1000.0);
        reset();
    }
    ~Chain() { (void)hipFree(ctr_); (void)hipFree(err_); }
    Chain(const Chain &) = delete;
    Chain &operator=(const Chain &) = delete;

    // Zero the counter and the error word (stream-ordered); the next launch keeps its barrier bit.
    void reset() {
        ok(hipMemsetAsync(ctr_, 0, sizeof(*ctr_), s_), "counter reset");
        ok(hipMemsetAsync(err_, 0, sizeof(*err_), s_), "error reset");
        issued_ = 0, barrier_ = true;
    }
    // Call after any non-chained operation on the stream: the next chained kernel waits for it by barrier bit.
    void plain() { barrier_ = true; }
    // Count `blocks` as issued without launching (tests only: simulates a launch that never ran).
    void skip(unsigned long long blocks) { issued_ += blocks; }

    template <typename... KArgs, typename... Args>
    void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t lds, Args... args) {
        Arg a{ctr_, err_, issued_, timeout_};
        issued_ += (unsigned long long)grid.x * grid.y * grid.z;
        const unsigned flags = barrier_ ? 0u : hipExtAnyOrderLaunch;
        barrier_ = false;
        hipExtLaunchKernelGGL(kernel, grid, block, (uint32_t)lds, s_, nullptr, nullptr, flags, args..., a);
        ok(hipGetLastError(), "chained launch");
    }
    // After the stream is synchronized: throws if any wait of this chain timed out.
    void check() const {
        unsigned e = 0;
        ok(hipMemcpy(&e, err_, sizeof(e), hipMemcpyDeviceToHost), "error read");
        if (e) throw std::runtime_error("flerp::chain: a chained kernel's wait timed out (a predecessor never finished)");
    }
    unsigned long long issued() const { return issued_; }
    unsigned long long timeout_ticks() const { return timeout_; }

private:
    static void ok(hipError_t e, const char *what) {
        if (e != hipSuccess) throw std::runtime_error(std::string("flerp::chain: ") + what + ": " + hipGetErrorString(e));
    }
    hipStream_t s_;
    unsigned long long *ctr_ = nullptr;
    unsigned *err_ = nullptr;
    unsigned long long issued_ = 0, timeout_ = 0;
    bool barrier_ = true;
};

}  // namespace flerp::chain
