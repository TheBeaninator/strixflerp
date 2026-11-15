#!/usr/bin/env bash
# Build tool.flerp.convert_qwen4exp_p16 with the host C++ compiler (no GPU, no HIP): its main, format.flerp.
# convert_qwen4exp_p16, format.flerp.q4panel / q8panel, format.flerp.strixw_v5's strixw.cpp (its include root first,
# so "formats/strixw.hpp" is the v5 copy) and the harvested strixite sources (quantizers, safetensors, merge plan,
# args, json). Usage: build.sh OUT_BINARY [another main.cpp instead of the tool's]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../../../../.." && pwd)"
f="$root/catalog/strixite/format"
fl="$root/catalog/flerp/format"
out="$1"
main="${2:-$here/convert_qwen4exp_p16_main.cpp}"
inc=(-I "$fl/strixw_v5/src/flerp/strixw_v5" -I "$fl/convert_qwen4exp_p16/src" -I "$fl/q4panel/src" -I "$fl/q8panel/src")
for m in q4 q5 q6 q8 safetensors merge_plan; do inc+=(-I "$f/$m/src"); done
inc+=(-I "$root/catalog/strixite/runtime/check/src" -I "$root/catalog/strixite/runtime/args/src"
      -I "$root/catalog/strixite/runtime/json/src")
srcs=("$fl/strixw_v5/src/flerp/strixw_v5/formats/strixw.cpp" "$fl/q4panel/src/flerp/q4panel/q4panel.cpp"
      "$fl/q8panel/src/flerp/q8panel/q8panel.cpp"
      "$fl/convert_qwen4exp_p16/src/flerp/convert_qwen4exp_p16/convert_qwen4exp_p16.cpp"
      "$root/catalog/strixite/runtime/json/src/serve/json.cpp")
for m in q4 q5 q6 q8 safetensors merge_plan; do srcs+=("$f/$m/src/formats/$m.cpp"); done
"${CXX:-g++}" -std=c++20 -O2 -ffp-contract=off -pthread "${inc[@]}" "$main" "${srcs[@]}" -o "$out"
