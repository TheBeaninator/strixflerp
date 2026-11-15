#!/usr/bin/env python3
"""Writes the swapped variants of recipes/qwen38-flash-next.strixite.toml (SPEC.md section 9, phase 4): the
unmodified recipe with one or more interface bindings pointed at an adapter, plus the bindings the foreign
modules need. A variant differs from the unmodified recipe in exactly the lines this script prints; run it again
after the unmodified recipe changes.    python3 tools/make-variant-recipes.py"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "qwen38-flash-next.strixite"
GGUF = {"format.device-view/gguf-kquant/-": "format.gufo.gguf_kquant"}
SWAPS = {      # interface -> (strixite module, adapter, what it is)
    "logits.topk/-/-": ("kernel.strixite.logits_topk", "kernel.flerp.logits_topk_gufo",
                        "the top-20 logits candidates come from gufo's LaunchGPUSortLogits (kernel.gufo.qwen_sampling, "
                        "built by gufo's compile recipe)"),
    "linear.dense/bf16/bf16": ("kernel.strixite.linear", "kernel.flerp.dense_bf16_strixfold",
                               "the MoE router logits come from StrixFold's tf_ds_dense_gemv (kernel.strixfold.dense_bf16, "
                               "a code object built by StrixFold's hipcc recipe with TheRock 10.2)"),
    "moe.router/-/bf16": ("kernel.strixite.moe_router", "kernel.flerp.moe_router_gufo",
                          "the MoE router's top-k and weights come from gufo's RouterTopK (kernel.gufo.qfn_kernels, "
                          "built by gufo's compile recipe)"),
}
VARIANTS = {
    "topk-gufo": (["logits.topk/-/-"], GGUF, "token-identity",
                  "Exact selection: the gate asks for token identity."),
    "dense-strixfold": (["linear.dense/bf16/bf16"], {}, "token-agreement",
                        "A different summation order: the module gate's bound is the criterion, token agreement is measured."),
    "router-gufo": (["moe.router/-/bf16"], GGUF, "token-agreement",
                    "Weights equal to fp32 rounding, not bit for bit: the module gate's tolerance is the criterion, token "
                    "agreement is measured."),
    "three-engines": (["logits.topk/-/-", "linear.dense/bf16/bf16", "moe.router/-/bf16"], GGUF, "token-agreement",
                      "All three swaps at once: strixite's executor with kernels of gufo and StrixFold in one binary."),
}


def main():
    with open(os.path.join(ROOT, "recipes", BASE + ".toml")) as f:
        base = f.read()
    for name, (ifaces, extra, criterion, note) in VARIANTS.items():
        s = base
        first = s.index("\n") + 1
        what = "; ".join(SWAPS[i][2] for i in ifaces)
        head = (f"# Swapped variant of {BASE} (written by tools/make-variant-recipes.py): {what}, through "
                f"{', '.join(SWAPS[i][1] for i in ifaces)}. Everything else is strixite v0.7.0-pod as in the unmodified "
                f"recipe. {note}\n")
        s = head + s[first:]
        s = s.replace(f'name      = "{BASE}"', f'name      = "qwen38-flash-next.{name}"')
        s = s.replace(f'"strixflerp-{BASE}"', f'"strixflerp-qwen38-flash-next.{name}"')
        for i in ifaces:
            old, new, _ = SWAPS[i]
            key = f'"{i}"'
            line = next(ln for ln in s.splitlines() if ln.startswith(key) and old in ln)
            s = s.replace(line, f'{key.ljust(36)} = "{new}"      # swap: was {old}')
        if extra:
            marker = "\n# Modules the origin builds but no binary links"
            add = "".join(f'{("\"" + k + "\"").ljust(36)} = "{v}"\n' for k, v in extra.items())
            s = s.replace(marker, "\n# Bindings the swap adds: the foreign modules' own requirements.\n" + add + marker, 1)
        s = s.replace(f'config = "gate/{BASE}.toml"', f'config = "gate/qwen38-flash-next.{name}.toml"\ncriterion = "{criterion}"')
        with open(os.path.join(ROOT, "recipes", f"qwen38-flash-next.{name}.toml"), "w") as f:
            f.write(s)
        print(f"recipes/qwen38-flash-next.{name}.toml: {', '.join(ifaces)} -> {criterion}")


if __name__ == "__main__":
    main()
