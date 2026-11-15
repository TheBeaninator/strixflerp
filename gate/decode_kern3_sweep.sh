#!/usr/bin/env bash
# DECODE-KERN3 tuning sweep (not a gate: picks the launch shape that the gate then judges). On control-node:
#   bash gate/decode_kern3_sweep.sh build    CPU only: one bench binary a variant, build/kern3/sweep/<v>/decode-kern3
#   bash gate/decode_kern3_sweep.sh run      GPU (take the lease first: gate/decode_kern3_run.sh sweep does): every
#                                            variant on the router / out-proj / lm-head / qkv cases, cache-cold
# A variant is "name router_waves router_group gemv_group gemv_waves", patched into copies of src-after's two sources.
set -euo pipefail
cd /data/projects/strixflerp
O=build/kern3; W=$O/sweep; R=/data/tools/therock-10.1.0a20260823; L=build/xfer-r6/lib
VARIANTS=${K3_VARIANTS:-"b1 2 16 1 4;b2 1 16 2 4;b3 4 16 4 4;b4 2 8 1 8;b5 2 16 2 8"}
case $1 in
build)
  IFS=';'; for v in $VARIANTS; do IFS=' ' read -r n rw rg gg gw <<< "$v"
    d=$W/$n; rm -rf $d; mkdir -p $d
    cp -r $O/src-after/catalog $d/catalog
    m=$d/catalog/flerp/kernel/moe_kquant/src/flerp/moe_kquant/moe_kquant.hip
    g=$d/catalog/gufo/kernel/qwen_linear_decode/src/src/families/qwen/kernels/gemv_quant.hip
    sed -i "s/^constexpr std::uint32_t kRouterWaves = .*/constexpr std::uint32_t kRouterWaves = $rw;/; s/^constexpr std::size_t kRouterGroup = [0-9]*;/constexpr std::size_t kRouterGroup = $rg;/" $m
    sed -i "s/^constexpr std::size_t kWideGroup = .*/constexpr std::size_t kWideGroup = $gg;/; s/^constexpr std::uint32_t kWideWaves = .*/constexpr std::uint32_t kWideWaves = $gw;/" $g
    grep -q "kRouterWaves = $rw;" $m && grep -q "kWideGroup = $gg;" $g
    S=$PWD/$d/catalog; T=$PWD/$O/after/tree
    ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ -DENGINE_ENABLE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
      -I$S/flerp/kernel/moe_kquant/src -I$S/gufo/kernel/qwen_linear_decode/src -I$PWD/gate -I$T/flerp/qwen35moe -I$T -I$T/include \
      -isystem $PWD/build/gate-scratch-moe/openssl-include -O2 -g -DNDEBUG -std=c++20 --offload-arch=gfx1151 -x hip -c gate/decode_kern3.hip -o $d/bench.o
    ionice -c3 nice -n 19 $R/lib/llvm/bin/clang++ --offload-arch=gfx1151 --hip-link $d/bench.o -o $d/decode-kern3 \
      $L/libgufo_core.a $L/gufo/src/ext/libgufo_logging.a $L/gufo/src/core/platform/libgufo_platform.a $L/libgufo_hip_host.a \
      /usr/lib/x86_64-linux-gnu/libicuuc.so /usr/lib/x86_64-linux-gnu/libicui18n.so /usr/lib/x86_64-linux-gnu/libcrypto.so.3 \
      $R/lib/llvm/lib/libomp.so $R/lib/libamdhip64.so.7.16.26332-0000000 -lpthread -Wl,-rpath,$R/lib:$R/lib/llvm/lib
    echo "built $d ($v)"
  done ;;
run)
  python3 - <<'PY'
import json, os, subprocess, sys
sys.path.insert(0, "gate")
import decode_kern3 as K
want = {"router-blk10", "attn-out-blk3", "ssm-out-blk0", "lm-head", "attn-qkv-blk0", "attn-gate-blk0"}
cases = [(n, a) for n, a in K.kernel_cases() if n in want]
out = os.path.join(K.RES, "sweep.jsonl")
variants = os.environ.get("K3_VARIANTS", "b1 2 16 1 4;b2 1 16 2 4;b3 4 16 4 4;b4 2 8 1 8;b5 2 16 2 8").split(";")
with open(out, "a") as f:
    for v in variants:
        n = v.split()[0]
        for c, a in cases:
            a = [x for x in a]
            if "--tokens-equal" in a:
                i = a.index("--tokens-equal"); a[i + 1] = "1"
            l0 = K.R1.load1()
            p = subprocess.run(K.R1.POLITE + [f"build/kern3/sweep/{n}/decode-kern3", "--case", c, "--cold-mib", "512"] + a,
                               capture_output=True, text=True)
            for line in p.stdout.splitlines():
                r = json.loads(line)
                if r["timing"]:
                    r.update(variant=v, load_start=l0, load_end=K.R1.load1(), rc=p.returncode)
                    f.write(json.dumps(r) + "\n"); f.flush()
                    print(v, c, r["outputs_differing"], r["timing"]["ref"]["GBps"], r["timing"]["new"]["GBps"], flush=True)
PY
  ;;
esac
