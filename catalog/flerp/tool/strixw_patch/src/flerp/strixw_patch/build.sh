#!/usr/bin/env bash
# Build tool.flerp.strixw_patch with the host C++ compiler (no GPU, no HIP): its source plus the harvested strixite
# format sources it requires. Usage: build.sh OUT_BINARY [extra main.cpp instead of the tool's]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../../../../.." && pwd)"
f="$root/catalog/strixite/format"
out="$1"
main="${2:-$here/strixw_patch.cpp}"
inc=()
for m in strixw q4 q5 q6 q8 safetensors merge_plan convert_qwen4exp; do inc+=(-I "$f/$m/src"); done
inc+=(-I "$root/catalog/strixite/runtime/check/src")
srcs=()
for m in strixw q4 q5 q6 q8 safetensors merge_plan convert_qwen4exp; do srcs+=("$f/$m/src/formats/$m.cpp"); done
"${CXX:-g++}" -std=c++20 -O2 -ffp-contract=off -pthread "${inc[@]}" "$main" "${srcs[@]}" -o "$out"
