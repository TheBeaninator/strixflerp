#!/usr/bin/env bash
# ATTN-PROJ (gate/attn-proj.toml) builds, on control-node, CPU only:
#   bash gate/attn_proj_build.sh src        src-before = build/attnproj/src-before.tar (git archive of the criteria commit);
#                                           src-after = the same plus this tree's ATTN-PROJ files ($ATTN_PROJ_FILES)
#   bash gate/attn_proj_build.sh assemble   before/after binaries of qwen36-35b-a3b.flerp-moe from the two exports
#   bash gate/attn_proj_build.sh bench      build/attnproj/attn-proj from gate/attn_proj.hip with src-after's catalogue
#   bash gate/attn_proj_build.sh tune       one bench a [tune] variant, kWideProjRows/kWideProjWaves patched into a copy
set -euo pipefail
cd /data/projects/strixflerp
R=/data/tools/therock-10.1.0a20260823; O=build/attnproj; L=build/xfer-r6/lib
ATTN_PROJ_FILES="catalog/gufo/kernel/qwen_ops/src/src/families/qwen/hip_common/quant_ops.hpp catalog/gufo/kernel/qwen_ops/module.toml
  catalog/gufo/kernel/qwen_linear_decode/src/src/families/qwen/kernels/qkv.hip catalog/gufo/kernel/qwen_linear_decode/module.toml
  catalog/gufo/kernel/qwen_gdn/src/src/families/qwen/kernels/ssm.hip catalog/gufo/kernel/qwen_gdn/module.toml"
bench() {  # $1 catalogue dir, $2 output dir
  local S=$PWD/$1 T=$PWD/$O/after/tree
  ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ -DENGINE_ENABLE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
    -I$S/gufo/kernel/qwen_ops/src -I$S/gufo/kernel/qwen_linear_decode/src -I$S/gufo/kernel/qwen_gdn/src -I$PWD/gate \
    -I$T/flerp/qwen35moe -I$T -I$T/include -isystem $PWD/build/gate-scratch-moe/openssl-include \
    -O2 -g -DNDEBUG -std=c++20 --offload-arch=gfx1151 -x hip -c gate/attn_proj.hip -o $2/bench.o
  ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ --offload-arch=gfx1151 --hip-link $2/bench.o -o $2/attn-proj \
    $L/libgufo_core.a $L/gufo/src/ext/libgufo_logging.a $L/gufo/src/core/platform/libgufo_platform.a $L/libgufo_hip_host.a \
    /usr/lib/x86_64-linux-gnu/libicuuc.so /usr/lib/x86_64-linux-gnu/libicui18n.so /usr/lib/x86_64-linux-gnu/libcrypto.so.3 \
    $R/lib/llvm/lib/libomp.so $R/lib/libamdhip64.so.7.16.26332-0000000 -lpthread -Wl,-rpath,$R/lib:$R/lib/llvm/lib
  echo "built $2/attn-proj"
}
case ${1:?src|assemble|bench|tune} in
src)
  rm -rf $O/src-before $O/src-after; mkdir -p $O/src-before $O/src-after
  tar -xf $O/src-before.tar -C $O/src-before
  # after = the criteria commit plus only the ATTN-PROJ files (another session's work in the tree stays out)
  tar -xf $O/src-before.tar -C $O/src-after
  for f in $ATTN_PROJ_FILES; do cp $f $O/src-after/$f; done
  echo "src-before $(sha256sum $O/src-before/catalog/gufo/kernel/qwen_ops/src/src/families/qwen/hip_common/quant_ops.hpp | cut -c1-12)" \
       "src-after $(sha256sum $O/src-after/catalog/gufo/kernel/qwen_ops/src/src/families/qwen/hip_common/quant_ops.hpp | cut -c1-12)" ;;
assemble)
  for w in ${2:-before after}; do
    (cd $O/src-$w && python3 ./flerp assemble qwen36-35b-a3b.flerp-moe -o /data/projects/strixflerp/$O/$w)
    rm -rf $O/$w/out
    FLERP_JOBS=4 bash $O/$w/build.sh -DFLERP_OPENSSL_INCLUDE_DIR=$PWD/build/gate-scratch-moe/openssl-include \
      -DFLERP_OPENSSL_CRYPTO_LIBRARY=/usr/lib/x86_64-linux-gnu/libcrypto.so.3 > $O/$w-build.log 2>&1 || { tail -30 $O/$w-build.log; exit 1; }
    echo "built $O/$w"
  done ;;
bench)
  bench $O/src-after/catalog $O ;;
tune)
  for v in "v1 2 8" "v2 2 4" "v3 4 4" "v4 4 8"; do
    read -r n r w <<< "$v"; d=$O/tune/$n; rm -rf $d; mkdir -p $d; cp -r $O/src-after/catalog $d/catalog
    q=$d/catalog/gufo/kernel/qwen_ops/src/src/families/qwen/hip_common/quant_ops.hpp
    sed -i "s/^constexpr std::uint32_t kWideProjRows = .*/constexpr std::uint32_t kWideProjRows = $r;/; s/^constexpr std::uint32_t kWideProjWaves = .*/constexpr std::uint32_t kWideProjWaves = $w;/" $q
    grep -q "kWideProjRows = $r;" $q && grep -q "kWideProjWaves = $w;" $q
    bench $d/catalog $d
  done ;;
esac
