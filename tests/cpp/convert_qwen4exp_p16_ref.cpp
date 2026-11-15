// The ORIGIN converter (format.strixite.convert_qwen4exp, strixw v4) as a tiny CLI for tests/test_convert_qwen4exp_p16.py:
//   convert_qwen4exp_p16_ref SRC OUT_DIR LAYOUT
// Built with tool.flerp.strixw_patch's build.sh (harvested strixite sources only).
#include "formats/convert_qwen4exp.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: convert_qwen4exp_p16_ref SRC OUT_DIR LAYOUT\n");
        return 2;
    }
    try {
        strix::ConvertOptions o;
        o.src_dir = argv[1], o.out_dir = argv[2], o.layout = argv[3], o.git = "selftest", o.threads = 2;
        o.console = nullptr;
        strix::convert_qwen4exp(o);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
