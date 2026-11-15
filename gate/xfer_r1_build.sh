#!/usr/bin/env bash
# Build build/xfer-r1/xfer-r1-splitk from gate/xfer_r1_splitk.hip with moe_kquant.hip's compile flags (build/flerp-p5).
set -euo pipefail
cd /data/projects/strixflerp
R=/data/tools/therock-10.1.0a20260823; T=$PWD/build/flerp-p5/tree; O=build/xfer-r1; L=build/xfer-r6/lib
mkdir -p $O
nice -n 19 $R/lib/llvm/bin/clang++ -DENGINE_ENABLE_HIP=1 -DHIPBLASLT_USE_ROCROLLER -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
  -I$T/flerp/qwen35moe -I$T -I$T/include -isystem $PWD/build/gate-scratch-moe/openssl-include -O2 -g -DNDEBUG -std=c++20 \
  --offload-arch=gfx1151 -x hip -c gate/xfer_r1_splitk.hip -o $O/splitk.o
nice -n 19 $R/lib/llvm/bin/clang++ --offload-arch=gfx1151 --hip-link $O/splitk.o -o $O/xfer-r1-splitk \
  $L/libgufo_core.a $L/gufo/src/ext/libgufo_logging.a $L/gufo/src/core/platform/libgufo_platform.a $L/libgufo_hip_host.a \
  /usr/lib/x86_64-linux-gnu/libicuuc.so /usr/lib/x86_64-linux-gnu/libicui18n.so /usr/lib/x86_64-linux-gnu/libcrypto.so.3 \
  $R/lib/llvm/lib/libomp.so $R/lib/libamdhip64.so.7.16.26332-0000000 -lpthread -Wl,-rpath,$R/lib:$R/lib/llvm/lib
echo built $O/xfer-r1-splitk
