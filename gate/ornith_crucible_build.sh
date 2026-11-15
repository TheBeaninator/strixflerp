#!/usr/bin/env bash
# ORNITH-CRUCIBLE (gate/ornith-crucible.toml) kernel bench build, on control-node, CPU only, in this checkout (an export,
# never the shared /data/projects/strixflerp): build/ocr/ornith-crucible-bench from gate/ornith_crucible_bench.hip, which
# #includes the catalogue's moe_kquant.hip (the current decode path), moe_sg.hip and draft_coarse.hip. therock-10.1 clang
# (the recipes' toolchain), -O3 as the recipe builds, no fast-math (moe_kquant.hip's options in the tool CMake files).
set -euo pipefail
cd "$(dirname "$0")/.."
R=/data/tools/therock-10.1.0a20260823; O=build/ocr; C=catalog
mkdir -p $O
ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ -DENGINE_ENABLE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I$C/flerp/kernel/moe_kquant/src -I$C/flerp/kernel/moe_sg/src -I$C/flerp/kernel/draft_coarse/src \
  -I$C/gufo/kernel/qwen_ops/src -I$C/gufo/kernel/core_headers/src -I$C/gufo/runtime/ext_api/src/include \
  -O3 -DNDEBUG -std=c++20 --offload-arch=gfx1151 -x hip -c gate/ornith_crucible_bench.hip -o $O/bench.o
ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ --offload-arch=gfx1151 --hip-link $O/bench.o -o $O/ornith-crucible-bench \
  $R/lib/libamdhip64.so -lpthread -Wl,-rpath,$R/lib:$R/lib/llvm/lib
sha256sum $O/ornith-crucible-bench gate/ornith_crucible_bench.hip $C/flerp/kernel/moe_kquant/src/flerp/moe_kquant/moe_kquant.hip \
  $C/flerp/kernel/moe_sg/src/flerp/moe_sg/moe_sg.hip $C/flerp/kernel/draft_coarse/src/flerp/draft_coarse/draft_coarse.hip > $O/build-sha256.txt
echo "built $O/ornith-crucible-bench"
