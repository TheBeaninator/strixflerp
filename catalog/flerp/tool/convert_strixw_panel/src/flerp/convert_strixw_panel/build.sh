#!/usr/bin/env bash
# Build tool.flerp.convert_strixw_panel with the host C++ compiler (no GPU, no HIP): its source, format.flerp.q4panel /
# q8panel, format.flerp.strixw_v5's adapted strixw.cpp (its include root first, so "formats/strixw.hpp" is the v5
# copy) and the harvested Q4/Q8 sources. Usage: build.sh OUT_BINARY [another main.cpp instead of the tool's]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../../../../.." && pwd)"
f="$root/catalog/strixite/format"
fl="$root/catalog/flerp/format"
out="$1"
main="${2:-$here/convert_strixw_panel.cpp}"
inc=(-I "$fl/strixw_v5/src/flerp/strixw_v5" -I "$fl/q4panel/src" -I "$fl/q8panel/src" -I "$f/q4/src" -I "$f/q8/src"
     -I "$root/catalog/strixite/runtime/check/src")
srcs=("$fl/strixw_v5/src/flerp/strixw_v5/formats/strixw.cpp" "$fl/q4panel/src/flerp/q4panel/q4panel.cpp"
      "$fl/q8panel/src/flerp/q8panel/q8panel.cpp" "$f/q4/src/formats/q4.cpp" "$f/q8/src/formats/q8.cpp")
"${CXX:-g++}" -std=c++20 -O2 -ffp-contract=off -pthread "${inc[@]}" "$main" "${srcs[@]}" -o "$out"
