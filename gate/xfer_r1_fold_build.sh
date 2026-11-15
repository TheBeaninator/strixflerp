#!/usr/bin/env bash
# Build build/xfer-r1-fold/xfer-r1-fold from gate/xfer_r1_fold.hip with moe_kquant.hip's compile flags (as
# gate/xfer_r1_build.sh): the new moe_kquant.hip from the catalogue, the reference from gate/xfer_r1_fold_ref.hip,
# every other header from the assembled tree of the recipe (build/xfer-r1-fold/before/tree).
set -euo pipefail
cd /data/projects/strixflerp
R=/data/tools/therock-10.1.0a20260823; T=$PWD/build/xfer-r1-fold/before/tree; O=build/xfer-r1-fold; L=build/xfer-r6/lib
mkdir -p $O
ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ -DENGINE_ENABLE_HIP=1 -DHIPBLASLT_USE_ROCROLLER -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I$PWD/catalog/flerp/kernel/moe_kquant/src -I$PWD/gate -I$T/flerp/qwen35moe -I$T -I$T/include -isystem $PWD/build/gate-scratch-moe/openssl-include \
  -O2 -g -DNDEBUG -std=c++20 --offload-arch=gfx1151 -x hip -c gate/xfer_r1_fold.hip -o $O/fold.o ${SAVE_TEMPS:+-save-temps=obj}
ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ --offload-arch=gfx1151 --hip-link $O/fold.o -o $O/xfer-r1-fold \
  $L/libgufo_core.a $L/gufo/src/ext/libgufo_logging.a $L/gufo/src/core/platform/libgufo_platform.a $L/libgufo_hip_host.a \
  /usr/lib/x86_64-linux-gnu/libicuuc.so /usr/lib/x86_64-linux-gnu/libicui18n.so /usr/lib/x86_64-linux-gnu/libcrypto.so.3 \
  $R/lib/llvm/lib/libomp.so $R/lib/libamdhip64.so.7.16.26332-0000000 -lpthread -Wl,-rpath,$R/lib:$R/lib/llvm/lib
echo built $O/xfer-r1-fold
