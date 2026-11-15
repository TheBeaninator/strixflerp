#!/usr/bin/env bash
# Build build/xfer-r6/xfer-verify-bench (gate/xfer_verify_bench.cpp) against a snapshot of build/flerp-p5's compiled
# libraries (qwen35moe_generate's build of the qwen35moe executor). Compile flags = generate.cpp's; link = its link line.
set -euo pipefail
cd /data/projects/strixflerp
R=/data/tools/therock-10.1.0a20260823; S=build/flerp-p5; O=build/xfer-r6
mkdir -p $O/lib
for a in $(cd $S/out && find . -name '*.a'); do mkdir -p $O/lib/$(dirname $a); cp -p $S/out/$a $O/lib/$a; done
cp -p "$S/out/CMakeFiles/flerp-qwen35moe-generate.dir/data/projects/strixflerp/build/flerp-p5/tree/flerp/gufo_qwen/no_image.cpp.o" $O/no_image.o
T=$PWD/$S/tree
nice -n 19 $R/lib/llvm/bin/clang++ -DENGINE_ENABLE_HIP=1 -DHIPBLASLT_USE_ROCROLLER -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 \
  -I$T/flerp/qwen35moe -I$T -I$T/include -isystem $R/include -isystem $PWD/build/gate-scratch-moe/openssl-include \
  -O2 -g -DNDEBUG -std=c++20 -include $T/src/core/hip/libstdcxx_noinline.hpp -fopenmp=libomp \
  -o $O/bench.o -c gate/xfer_verify_bench.cpp
L=$O/lib
nice -n 19 $R/lib/llvm/bin/clang++ -O2 -g -DNDEBUG --offload-arch=gfx1151 --hip-link $O/bench.o $O/no_image.o -o $O/xfer-verify-bench \
  $L/gufo/src/models/qwen/libgufo_qwen27b.a $L/gufo/src/families/qwen/config/libgufo_qwen_config.a $L/gufo/src/families/qwen/text/libgufo_qwen_text.a \
  $L/gufo/src/families/qwen/vision/libgufo_qwen_vision.a $L/libgufo_core.a $L/gufo/src/ext/libgufo_logging.a $L/libflerp_stage_relay.a \
  $L/gufo/src/families/qwen/text/libgufo_qwen_text.a $L/gufo/src/families/qwen/config/libgufo_qwen_config.a $L/libgufo_cache_hip.a \
  $L/gufo/src/cache/libgufo_cache.a $L/gufo/src/families/qwen/kernels/libgufo_qwen_kernels.a $L/libflerp_moe_kernels.a $L/libflerp_qfn_kernels.a \
  $L/libflerp_gguf_placed.a $L/libgufo_core.a /usr/lib/x86_64-linux-gnu/libicuuc.so /usr/lib/x86_64-linux-gnu/libicui18n.so \
  /usr/lib/x86_64-linux-gnu/libcrypto.so.3 $R/lib/llvm/lib/libomp.so /lib/x86_64-linux-gnu/libpthread.a $L/libgufo_hip_host.a \
  $R/lib/libhipblas.so.3.7 $R/lib/librocblas.so.5.7 $R/lib/libhipblaslt.so.1.4 $R/lib/libamdhip64.so.7.16.26332-0000000 \
  $R/lib/llvm/lib/clang/23/lib/linux/libclang_rt.builtins-x86_64.a $L/gufo/src/core/platform/libgufo_platform.a \
  -Wl,-rpath,$R/lib:$R/lib/llvm/lib
echo built $O/xfer-verify-bench
