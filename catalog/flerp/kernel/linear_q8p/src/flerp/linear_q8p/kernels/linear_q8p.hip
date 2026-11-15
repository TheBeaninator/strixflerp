// kernel.flerp.linear_q8p: the Q8 panel GEMV (kernels/panel_gemv.hpp at 16 codes a chunk) and the adapter over
// kernel.strixite.linear_q8.

#include "kernels/linear_q8p.hpp"

#include "flerp/adapter_rt.hpp"
#include "kernels/panel_gemv.hpp"

namespace strix::kernels {

void linear_q8_flerp_origin(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, Act out_act,
                            hipStream_t stream);

namespace {

flerp::Tally &tally() {
    static flerp::Tally t("kernel.flerp.linear_q8p", "panel kernels (Q8Panel16)");
    return t;
}

// linear_q8.hip split_k: K <= 2560 one wave a row, else ~1024 codes a wave.
int q8_split_k(int64_t K) {
    if (K <= 2560) return 1;
    int ks = 1;
    while (ks < 8 && K / (1024 * (ks * 2)) >= 1) ks *= 2;
    return ks;
}

}  // namespace

void linear_q8p(const void *x, const flerp::PanelView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    panel_gemv<8>("linear_q8p", x, w, y, M, act, out_act, q8_split_k(w.K), stream);
}

void linear_q8(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream) {
    linear_q8(x, w, y, M, act, act, stream);
}
void linear_q8(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    if (flerp::is_panel(w)) {
        tally().foreign();
        linear_q8p(x, flerp::panel_of(w), y, M, act, out_act, stream);
    } else {
        tally().origin();
        linear_q8_flerp_origin(x, w, y, M, act, out_act, stream);
    }
}

}  // namespace strix::kernels
