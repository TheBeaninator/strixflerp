// tool.flerp.convert_qwen4exp_p16: adapted copy of strixite's tools/convert_qwen4exp.cpp (20937dc); changes marked
// `flerp`. Calls format.flerp.convert_qwen4exp_p16 (layout classes may carry p16; writes strixw v5).
//
// Offline converter: Qwen3.8-Flash-Next HF safetensors -> <out>/weights.strixw (formats/strixw.hpp).
// Verbose by default: every step is logged to stderr and to <out>/convert.log.
//
// Usage: convert_qwen4exp --src DIR --out DIR --git HASH [--threads N] [--layout default=q4g64,...]
//                         [--layers 0,3,7] [--no-globals] [--no-mtp] [--mtp-from FILE] [--vision]
//                         [--tower-only] [--plan]
//   --mtp-from    replace the checkpoint's mtp.* tensors with this safetensors file's (a fine-tuned draft head);
//                 the trunk and its fingerprint stay the checkpoint's
//   --vision      also write <out>/vision.strixw: the vision tower as exact BF16, validated against the
//                 checkpoint's config.json "vision_config"
//   --tower-only  (implies --vision) write <out>/vision.strixw and nothing else - adds the tower to an
//                 output directory converted without --vision; never touches an existing weights.strixw
//   --plan        read the headers, log the full plan (every output tensor), write nothing
//   --layers ...  convert only these decoder layers (a partial file, marked complete=no)
//   flerp: --plan-scratch DIR  with --plan, also plan the exact file size on a sparse scratch file in DIR
//   flerp: --check-against REF --tensors NAME,..|all  quantize only those tensors and compare with REF (no output)
//   flerp: default --layout = the fleet's layout with p16 on the classes docs/STRIXW-LAYOUT-NEXT.md recommends

#include "common/args.hpp"
#include "flerp/convert_qwen4exp_p16/convert_qwen4exp_p16.hpp"
#include "serve/json.hpp"

#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

using namespace strix;
using flerp::ConvertOptions;

// flerp: the fleet's layout (gate/results/strixw-layout/header-diff.md) with p16 on every Q4/Q8 class but the HC mixes.
const char *const kDefaultLayout =
    "default=q8g64,exp_gu=q4g128p16,exp_down=q4g32p16,sh_gu=q8g64p16,sh_down=q8g32p16,gdn_in=q4g128p16,"
    "gdn_out=q8g64p16,attn_qkv=q8g64p16,attn_o=q8g64p16,hc_down=q8g64,hc_up=q8g64,ple_kv=q8g64p16,lm_head=q8g64p16,"
    "mtp_fc=q8g64p16,embed=bf16";

namespace {

// The checkpoint's config.json "vision_config" -> the fields the converter validates against, plus the
// raw JSON text to record in the new file's index. Missing or wrong-typed fields are errors.
void read_vision_config(const std::string &src_dir, ConvertOptions &o) {
    const std::string path = src_dir + "/config.json";
    std::ifstream in(path);
    STRIX_CHECK(in.is_open(), "convert --vision: can't open '", path, "' (vision_config comes from it)");
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json::Value cfg = json::Value::parse(text, 64);
    const json::Value *vc = cfg.find("vision_config");
    STRIX_CHECK(vc != nullptr && vc->is_object(), "convert --vision: '", path, "' has no vision_config object");
    auto num = [&](const char *key) -> int64_t {
        const json::Value *v = vc->find(key);
        STRIX_CHECK(v != nullptr && v->is_int(), "convert --vision: vision_config.", key, " is missing or not an integer (in '",
                    path, "')");
        return v->as_int(std::string("vision_config.") + key);
    };
    o.vision_dims.depth = num("depth");
    o.vision_dims.hidden_size = num("hidden_size");
    o.vision_dims.num_heads = num("num_heads");
    o.vision_dims.intermediate_size = num("intermediate_size");
    o.vision_dims.in_channels = num("in_channels");
    o.vision_dims.patch_size = num("patch_size");
    o.vision_dims.temporal_patch_size = num("temporal_patch_size");
    o.vision_dims.spatial_merge_size = num("spatial_merge_size");
    o.vision_dims.out_hidden_size = num("out_hidden_size");
    o.vision_dims.num_position_embeddings = num("num_position_embeddings");
    const json::Value *act = vc->find("hidden_act");
    STRIX_CHECK(act != nullptr && act->is_string(), "convert --vision: vision_config.hidden_act is missing or not a string (in '",
                path, "')");
    o.vision_dims.hidden_act = act->as_string("vision_config.hidden_act");
    o.vision_config_json = vc->dump();
}

int run(int argc, char **argv) {
    ConvertOptions o;
    o.layout = kDefaultLayout;  // flerp
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char *flag) {
            STRIX_CHECK(i + 1 < argc, flag, " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--src") o.src_dir = value("--src");
        else if (a == "--out") o.out_dir = value("--out");
        else if (a == "--git") o.git = value("--git");
        else if (a == "--threads") o.threads = (int)parse_int(value("--threads"), "--threads", 1, 1024);
        else if (a == "--layout") o.layout = value("--layout");
        else if (a == "--layers") {
            std::stringstream ss(value("--layers"));
            for (std::string t; std::getline(ss, t, ',');) o.layers.push_back(parse_int(t, "--layers entry", 0, 1 << 20));
        } else if (a == "--no-globals") o.globals = false;
        else if (a == "--no-mtp") o.mtp = false;
        else if (a == "--mtp-from") o.mtp_from = value("--mtp-from");
        else if (a == "--vision") o.vision = true;
        else if (a == "--tower-only") o.vision = true, o.tower_only = true;
        else if (a == "--plan") o.plan_only = true;
        else if (a == "--plan-scratch") o.plan_scratch = value("--plan-scratch");  // flerp
        else if (a == "--check-against") o.check_ref = value("--check-against");  // flerp
        else if (a == "--tensors") {  // flerp
            std::stringstream ss(value("--tensors"));
            for (std::string t; std::getline(ss, t, ',');) if (!t.empty()) o.check_names.push_back(t);
        }
        else {
            std::fprintf(stderr,
                         "usage: %s --src DIR --out DIR --git HASH [--threads N] [--layout default=q4g64,...] "
                         "[--layers 0,3] [--no-globals] [--no-mtp] [--mtp-from FILE] [--vision] "
                         "[--tower-only] [--plan [--plan-scratch DIR]] [--check-against REF --tensors NAME,..|all]\n",
                         argv[0]);
            return 2;
        }
    }
    STRIX_CHECK(!o.src_dir.empty() && !o.git.empty() && (o.plan_only || !o.check_ref.empty() || !o.out_dir.empty()),
                "--src, --git and (unless --plan / --check-against) --out are required");
    STRIX_CHECK(o.check_ref.empty() == o.check_names.empty(), "--check-against and --tensors go together");  // flerp
    if (o.vision) read_vision_config(o.src_dir, o);
    flerp::convert_qwen4exp_p16(o);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "convert_qwen4exp_p16: %s\n", e.what());
        return 1;
    }
}
