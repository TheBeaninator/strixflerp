// SPDX-License-Identifier: MIT
// Interface gate driver for residual.add/-/f32, through strixite's launch API (the consumer's side of the swap):
//   strix::kernels::residual_add(x, y, n, Act::F32, stream)
// Whichever module the gate bound to the interface provides that symbol.
#include "flerp_gate/gate_common.hpp"
#include "kernels/residual.hpp"
#include "residual_add.hpp"

using namespace gate;

// Values that make an add interesting: ordinary activations, cancellations, zeros of both signs, subnormals,
// huge and tiny magnitudes side by side, infinities and NaNs.
static void fill(std::vector<float> &x, std::vector<float> &y, uint64_t seed, const std::string &mix) {
    Rng r(seed);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = r.gauss(3.0f);
        y[i] = r.gauss(3.0f);
        if (mix != "special") continue;
        switch (r.next() % 16) {
            case 0: y[i] = -x[i]; break;                                    // exact cancellation
            case 1: x[i] = 0.0f, y[i] = -0.0f; break;
            case 2: x[i] = -0.0f, y[i] = -0.0f; break;
            case 3: x[i] = bits_f32((uint32_t)(r.next() & 0x7fffff) | 1u), y[i] = bits_f32((uint32_t)(r.next() & 0x7fffff)); break;   // subnormals
            case 4: x[i] = 3.0e38f, y[i] = 3.0e38f; break;                  // overflow to +inf
            case 5: x[i] = INFINITY, y[i] = -INFINITY; break;               // inf - inf = NaN
            case 6: x[i] = NAN; break;
            case 7: x[i] = r.sym(1e30f), y[i] = r.sym(1e-30f); break;       // absorbed addend
            case 8: x[i] = 16777216.0f, y[i] = 1.0f; break;                 // tie, round to even
            case 9: x[i] = 16777218.0f, y[i] = 1.0f; break;                 // tie, the other way
            default: break;
        }
    }
}

int main(int argc, char **argv) {
    return run_driver(argc, argv, "residual_add", [](const Case &c, Obj &o, hipStream_t stream, const std::string &dumps) {
        const size_t n = (size_t)c.num("n");
        o.i("n", (long long)n);
        if (c.str("kind") == "parity") {
            std::vector<float> x(n), y(n);
            fill(x, y, (uint64_t)c.num("seed"), c.str("mix", "plain"));
            const bool alias = c.num("alias", 0) != 0;      // y == x: strixite allows it, x doubles
            Dev<float> dx(n), dy(n);
            dx.put(x), dy.put(y);
            strix::kernels::residual_add(dx.p, alias ? dx.p : dy.p, (int64_t)n, strix::kernels::Act::F32, stream);
            hip_ok(hipStreamSynchronize(stream), "sync");
            std::vector<float> got = dx.get(), want = x;
            reference::residual_add_f32(want.data(), alias ? x.data() : y.data(), n);
            put_diff(o, "", diff_f32(got.data(), want.data(), n));
            dump(dumps, c.name, got.data(), n * 4);
        } else {      // bench: y = 0 keeps x fixed over thousands of in-place launches
            std::vector<float> x(n), y(n, 0.0f);
            Rng r(1);
            for (auto &v : x) v = r.gauss(1.0f);
            Dev<float> dx(n), dy(n);
            dx.put(x), dy.put(y);
            const Timing t = time_launches([&] { strix::kernels::residual_add(dx.p, dy.p, (int64_t)n, strix::kernels::Act::F32, stream); },
                                           stream, (int)c.num("iters"), (int)c.num("repeats"));
            put_timing(o, t, 12.0 * (double)n, 1);      // reads x and y, writes x
        }
    });
}
