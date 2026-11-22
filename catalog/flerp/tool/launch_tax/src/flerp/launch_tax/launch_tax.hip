// launch_tax: what fusing dependent decode launches into one kernel can save on this GPU.
//
// A fused "inject + mix_down + mix_up" kernel (owner's 2026-11-16 package, fused residual + RMSNorm + HC + gate) still
// has two grid-wide dependencies inside it (the mix_down's RMS and dot need every stream column; the mix_up needs the
// whole h), so one launch replaces three only by adding two grid barriers. The saving per fused boundary is
// 2 x (dependent launch - grid barrier). This bench measures both at the decode shapes of executor.strixite.qwen4exp
// (40 blocks x 256 threads: hc_inject over H*d = 10,240 values, hc_mix_up over d / 64 = 40 blocks), plus the work alone.
//
//   launch_tax [--n 4000] [--blocks 40] [--reps 5]
// Prints one JSON line per rep: launch_us (a dependent launch on one stream, end to end, divided by n),
// barrier_us (one grid barrier inside a persistent kernel), work_us (the same work in one kernel, no barrier).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "launch_tax: %s: %s\n", #x, hipGetErrorString(e_));             \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

constexpr int kThreads = 256;
constexpr int kN = 10240;  // H * d at decode, T = 1

// The per-step work: every thread reads its element and its neighbour's (written by another block in the previous
// step, so the steps really depend on each other) and writes one value - about hc_inject's traffic.
__device__ __forceinline__ void step(float *buf, int s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = i; j < kN; j += gridDim.x * blockDim.x) {
        const float a = buf[j], b = buf[(j + 997) % kN];
        buf[j] = a * 0.999f + b * 0.001f + (float)s * 1e-9f;
    }
}

__global__ void __launch_bounds__(kThreads) one_step(float *buf, int s) { step(buf, s); }

// Sense-reversing grid barrier over all resident blocks: one atomic per block, block 0's thread 0 flips the sense.
__device__ __forceinline__ void grid_barrier(unsigned *count, volatile unsigned *sense, unsigned &local) {
    __syncthreads();
    if (threadIdx.x == 0) {
        local ^= 1u;
        __threadfence();
        if (atomicAdd(count, 1u) == gridDim.x - 1) {
            *count = 0;
            __threadfence();
            *sense = local;
        } else {
            while (*sense != local) __builtin_amdgcn_s_sleep(1);
        }
        __threadfence();
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kThreads) steps_with_barrier(float *buf, int n, unsigned *count, unsigned *sense) {
    unsigned local = *sense;
    for (int s = 0; s < n; ++s) {
        step(buf, s);
        grid_barrier(count, sense, local);
    }
}

__global__ void __launch_bounds__(kThreads) steps_no_barrier(float *buf, int n) {
    for (int s = 0; s < n; ++s) step(buf, s);
}

int main(int argc, char **argv) {
    int n = 4000, blocks = 40, reps = 5;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--n")) n = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--blocks")) blocks = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
    }
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));
    if (blocks > p.multiProcessorCount) {
        std::fprintf(stderr, "launch_tax: %d blocks > %d CUs; the barrier needs every block resident\n", blocks,
                     p.multiProcessorCount);
        return 1;
    }
    float *buf;
    unsigned *sync;
    CK(hipMalloc(&buf, kN * sizeof(float)));
    CK(hipMalloc(&sync, 2 * sizeof(unsigned)));
    CK(hipMemset(buf, 0, kN * sizeof(float)));
    CK(hipMemset(sync, 0, 2 * sizeof(unsigned)));
    hipStream_t st;
    CK(hipStreamCreate(&st));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto timed = [&](auto &&body) {
        CK(hipEventRecord(e0, st));
        body();
        CK(hipEventRecord(e1, st));
        CK(hipEventSynchronize(e1));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, e0, e1));
        return (double)ms * 1000.0 / n;
    };
    for (int r = -1; r < reps; ++r) {  // r = -1: warm-up, not printed
        const double launch = timed([&] {
            for (int s = 0; s < n; ++s) one_step<<<blocks, kThreads, 0, st>>>(buf, s);
        });
        const double barrier = timed([&] { steps_with_barrier<<<blocks, kThreads, 0, st>>>(buf, n, sync, sync + 1); });
        const double work = timed([&] { steps_no_barrier<<<blocks, kThreads, 0, st>>>(buf, n); });
        CK(hipGetLastError());
        if (r >= 0)
            std::printf("{\"rep\":%d,\"gpu\":\"%s\",\"cus\":%d,\"blocks\":%d,\"n\":%d,\"launch_us\":%.3f,\"barrier_us\":%.3f,"
                        "\"work_us\":%.3f}\n",
                        r, p.gcnArchName, p.multiProcessorCount, blocks, n, launch, barrier, work);
    }
    return 0;
}
