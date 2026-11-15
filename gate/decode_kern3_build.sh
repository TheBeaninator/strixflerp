#!/usr/bin/env bash
# DECODE-KERN3 (gate/decode-kern3.toml) builds, on control-node, CPU only:
#   bash gate/decode_kern3_build.sh assemble   before/after binaries of qwen36-35b-a3b.flerp-moe from the two source
#                                              exports build/kern3/src-before (HEAD at the criteria commit) and
#                                              build/kern3/src-after (the same plus the new kernels)
#   bash gate/decode_kern3_build.sh bench      build/kern3/decode-kern3 from gate/decode_kern3.hip: the NEW
#                                              moe_kquant.hip and gemv_quant.hip from src-after's catalogue, the
#                                              reference gate/decode_kern3_ref.hip, other headers from after/tree
set -euo pipefail
cd /data/projects/strixflerp
R=/data/tools/therock-10.1.0a20260823; O=build/kern3
case ${1:?assemble|bench} in
assemble)
  for w in ${2:-before after}; do
    (cd $O/src-$w && python3 ./flerp assemble qwen36-35b-a3b.flerp-moe -o /data/projects/strixflerp/$O/$w)
    rm -rf $O/$w/out
    FLERP_JOBS=4 bash $O/$w/build.sh -DFLERP_OPENSSL_INCLUDE_DIR=$PWD/build/gate-scratch-moe/openssl-include \
      -DFLERP_OPENSSL_CRYPTO_LIBRARY=/usr/lib/x86_64-linux-gnu/libcrypto.so.3 > $O/$w-build.log 2>&1 || { tail -30 $O/$w-build.log; exit 1; }
    echo "built $O/$w"
  done ;;
bench)
  S=$PWD/$O/src-after/catalog; T=$PWD/$O/after/tree; L=build/xfer-r6/lib
  ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ -DENGINE_ENABLE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
    -I$S/flerp/kernel/moe_kquant/src -I$S/gufo/kernel/qwen_linear_decode/src -I$PWD/gate -I$T/flerp/qwen35moe -I$T -I$T/include \
    -isystem $PWD/build/gate-scratch-moe/openssl-include \
    -O2 -g -DNDEBUG -std=c++20 --offload-arch=gfx1151 -x hip -c gate/decode_kern3.hip -o $O/bench.o
  ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ --offload-arch=gfx1151 --hip-link $O/bench.o -o $O/decode-kern3 \
    $L/libgufo_core.a $L/gufo/src/ext/libgufo_logging.a $L/gufo/src/core/platform/libgufo_platform.a $L/libgufo_hip_host.a \
    /usr/lib/x86_64-linux-gnu/libicuuc.so /usr/lib/x86_64-linux-gnu/libicui18n.so /usr/lib/x86_64-linux-gnu/libcrypto.so.3 \
    $R/lib/llvm/lib/libomp.so $R/lib/libamdhip64.so.7.16.26332-0000000 -lpthread -Wl,-rpath,$R/lib:$R/lib/llvm/lib
  echo "built $O/decode-kern3" ;;
esac
