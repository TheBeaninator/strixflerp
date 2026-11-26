/* Per-request routing records. Read-only: adds gate mass into a
 * rank-0 grid. gfx1151 scan: this file only, the kit's flags, no extra -D.
 *
 * One thread per (layer, expert) walks the round's rows. That thread is the only writer
 * of its (layer, expert) across slots, so there are no atomics. Commit and slot rules
 * match tf_ds_reap_route. nlayer is the grid stride (40), not how many layers this launch ran.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(128)
tf_ds_route_rec(uint64_t pick, uint64_t wts, uint64_t commit, uint64_t slotmap,
                uint64_t mass, uint64_t count,
                uint32_t layers, uint32_t rows, uint32_t row_stride, uint32_t slots,
                uint32_t topk, uint32_t nexp, uint32_t nslot, uint32_t nlayer) {
    if (nexp == 0 || layers == 0 || nlayer == 0) return;
    const uint32_t ncell = layers * nexp;
    const int32_t* __restrict__ pk = reinterpret_cast<const int32_t*>(pick);
    const float* __restrict__ wt = reinterpret_cast<const float*>(wts);
    const uint8_t* __restrict__ cm = reinterpret_cast<const uint8_t*>(commit);
    const uint8_t* __restrict__ sm = reinterpret_cast<const uint8_t*>(slotmap);
    double* __restrict__ ms = reinterpret_cast<double*>(mass);
    uint32_t* __restrict__ ct = reinterpret_cast<uint32_t*>(count);
    for (uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x; cell < ncell; cell += gridDim.x * blockDim.x) {
        uint32_t layer = cell / nexp;
        uint32_t expert = cell - layer * nexp;
#pragma unroll 1
        for (uint32_t row = 0; row < rows; row++) {
            if (cm[row] == 0) continue;
            uint32_t slot = sm[row];
            if (slot == 0xFFu || slot >= nslot) continue;
            const int32_t* pr = pk + (static_cast<size_t>(layer) * row_stride + row) * slots;
            const float* wr = wt + (static_cast<size_t>(layer) * row_stride + row) * slots;
#pragma unroll 1
            for (uint32_t k = 0; k < topk; k++) {
                int32_t id = pr[k];
                if (id < 0 || static_cast<uint32_t>(id) >= nexp || static_cast<uint32_t>(id) != expert) continue;
                size_t at = (static_cast<size_t>(slot) * nlayer + layer) * nexp + expert;
                ms[at] += static_cast<double>(wr[k]);
                ct[at] += 1;
            }
        }
    }
}
