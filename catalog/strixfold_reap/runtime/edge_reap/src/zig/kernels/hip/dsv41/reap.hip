/* Edge-REAP routing statistics. Read-only: nothing here writes an expert output.
 * Four kernels, fixed-width parameters, no shared memory, no atomics. Index formulas are reap.zig's
 * packSrc / packDst / cellIndex. gfx1151 scan: this file only, the kit's flags, no extra -D.
 *
 * pack copies sampled slot vectors out of a strided [layers, rows_cap, slots, dn] buffer.
 * ssq is ||y0 + y1||^2 in f64 (y1 == 0: ||y0||^2). One thread per vector.
 * route / norm: one thread per (layer, expert) walks the round's rows and updates every bucket that
 * row names. That thread is the only writer of that (layer, expert) across buckets.
 */
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(256)
tf_ds_reap_pack(uint64_t src, uint64_t dst, uint64_t rows_idx,
                uint32_t layers, uint32_t rows_cap, uint32_t slots, uint32_t topk,
                uint32_t dn, uint32_t n_sample) {
    if (dn == 0 || topk == 0 || n_sample == 0 || layers == 0) return;
    const float* __restrict__ s = reinterpret_cast<const float*>(src);
    float* __restrict__ d = reinterpret_cast<float*>(dst);
    const int32_t* __restrict__ idx = reinterpret_cast<const int32_t*>(rows_idx);
    const uint32_t n = layers * n_sample * topk * dn;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        uint32_t col = i % dn;
        uint32_t t = i / dn;
        uint32_t slot = t % topk;
        t /= topk;
        uint32_t sample = t % n_sample;
        uint32_t layer = t / n_sample;
        uint32_t row = idx ? static_cast<uint32_t>(idx[sample]) : sample;
        size_t src_i = ((static_cast<size_t>(layer) * rows_cap + row) * slots + slot) * dn + col;
        d[i] = s[src_i];
    }
}

extern "C" __global__ void __launch_bounds__(256)
tf_ds_reap_ssq(uint64_t y0, uint64_t y1, uint64_t out, uint32_t nvec, uint32_t dn) {
    if (dn == 0) return;
    const float* __restrict__ a = reinterpret_cast<const float*>(y0);
    const float* __restrict__ b = reinterpret_cast<const float*>(y1);
    double* __restrict__ o = reinterpret_cast<double*>(out);
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < nvec; v += gridDim.x * blockDim.x) {
        const float* p0 = a + static_cast<size_t>(v) * dn;
        const float* p1 = b ? b + static_cast<size_t>(v) * dn : nullptr;
        double sum = 0;
#pragma unroll 1
        for (uint32_t i = 0; i < dn; i++) {
            double x = static_cast<double>(p0[i]);
            if (p1) x += static_cast<double>(p1[i]);
            sum += x * x;
        }
        o[v] = sum;
    }
}

extern "C" __global__ void __launch_bounds__(128)
tf_ds_reap_route(uint64_t pick, uint64_t wts, uint64_t pos, uint64_t commit, uint64_t bucket,
                 uint64_t token, uint64_t gate, uint64_t early,
                 uint32_t layers, uint32_t rows, uint32_t row_stride, uint32_t slots,
                 uint32_t topk, uint32_t nexp, uint32_t nbucket) {
    if (nexp == 0 || layers == 0) return;
    const uint32_t ncell = layers * nexp;
    const int32_t* __restrict__ pk = reinterpret_cast<const int32_t*>(pick);
    const float* __restrict__ wt = reinterpret_cast<const float*>(wts);
    const int32_t* __restrict__ ps = reinterpret_cast<const int32_t*>(pos);
    const uint8_t* __restrict__ cm = reinterpret_cast<const uint8_t*>(commit);
    const uint8_t* __restrict__ bk = reinterpret_cast<const uint8_t*>(bucket);
    int64_t* __restrict__ tok = reinterpret_cast<int64_t*>(token);
    double* __restrict__ gt = reinterpret_cast<double*>(gate);
    int64_t* __restrict__ er = reinterpret_cast<int64_t*>(early);
    for (uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x; cell < ncell; cell += gridDim.x * blockDim.x) {
        uint32_t layer = cell / nexp;
        uint32_t expert = cell - layer * nexp;
#pragma unroll 1
        for (uint32_t row = 0; row < rows; row++) {
            if (cm[row] == 0) continue;
            uint32_t b = bk[row];
            if (b >= nbucket) continue;
            int32_t p = ps[row];
            const int32_t* pr = pk + (static_cast<size_t>(layer) * row_stride + row) * slots;
            const float* wr = wt + (static_cast<size_t>(layer) * row_stride + row) * slots;
#pragma unroll 1
            for (uint32_t k = 0; k < topk; k++) {
                int32_t id = pr[k];
                if (id < 0 || static_cast<uint32_t>(id) >= nexp || static_cast<uint32_t>(id) != expert) continue;
                size_t at = (static_cast<size_t>(b) * layers + layer) * nexp + expert;
                tok[at] += 1;
                gt[at] += static_cast<double>(wr[k]);
                if (p < 64) er[at] += 1;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128)
tf_ds_reap_norm(uint64_t pick, uint64_t wts, uint64_t bucket, uint64_t rows_idx,
                uint64_t ss0, uint64_t ss1, uint64_t raw, uint64_t ncount, uint64_t maxn,
                uint32_t layers, uint32_t row_stride, uint32_t slots, uint32_t topk,
                uint32_t nexp, uint32_t nbucket, uint32_t n_sample) {
    if (nexp == 0 || layers == 0 || n_sample == 0) return;
    const uint32_t ncell = layers * nexp;
    const int32_t* __restrict__ pk = reinterpret_cast<const int32_t*>(pick);
    const float* __restrict__ wt = reinterpret_cast<const float*>(wts);
    const uint8_t* __restrict__ bk = reinterpret_cast<const uint8_t*>(bucket);
    const int32_t* __restrict__ idx = reinterpret_cast<const int32_t*>(rows_idx);
    const double* __restrict__ s0 = reinterpret_cast<const double*>(ss0);
    const double* __restrict__ s1 = reinterpret_cast<const double*>(ss1);
    double* __restrict__ rw = reinterpret_cast<double*>(raw);
    int64_t* __restrict__ nc = reinterpret_cast<int64_t*>(ncount);
    double* __restrict__ mx = reinterpret_cast<double*>(maxn);
    for (uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x; cell < ncell; cell += gridDim.x * blockDim.x) {
        uint32_t layer = cell / nexp;
        uint32_t expert = cell - layer * nexp;
#pragma unroll 1
        for (uint32_t sample = 0; sample < n_sample; sample++) {
            uint32_t row = idx ? static_cast<uint32_t>(idx[sample]) : sample;
            uint32_t b = bk[row];
            if (b >= nbucket) continue;
            const int32_t* pr = pk + (static_cast<size_t>(layer) * row_stride + row) * slots;
            const float* wr = wt + (static_cast<size_t>(layer) * row_stride + row) * slots;
#pragma unroll 1
            for (uint32_t k = 0; k < topk; k++) {
                int32_t id = pr[k];
                if (id < 0 || static_cast<uint32_t>(id) >= nexp || static_cast<uint32_t>(id) != expert) continue;
                uint32_t vi = (layer * n_sample + sample) * topk + k;
                double ss = s0[vi];
                if (s1) ss += s1[vi];
                double norm = sqrt(ss);
                size_t at = (static_cast<size_t>(b) * layers + layer) * nexp + expert;
                rw[at] += static_cast<double>(wr[k]) * norm;
                nc[at] += 1;
                if (norm > mx[at]) mx[at] = norm;
            }
        }
    }
}
