#!/usr/bin/env bash
# decode2 (gate/strixw-panel-decode2.toml): rocprofv3 FETCH_SIZE of the decode GEMVs, origin and panel, on gdn_in
# (Q4 real, Q8 synthetic) and sh_gu at M = 1 and 8, with the wave map $1 = RW:TPW or "table". Reads ~26 MB.
# Usage: tools/decode2-fetch.sh <rw:tpw|table> <out-name>
set -u
cd "$(dirname "$0")/.."
map=$1 name=$2
res=gate/results/strixw-panel-decode2/fetch; mkdir -p "$res"
W=/data/models/strix-infer/converted/qwen38-fnext-tpn/weights.strixw
B=build/qwen38-flash-next.strixite-p16/ninja/strixflerp-panel-kernel-bench
L=model.language_model.layers
unset FLERP_PANEL_RW FLERP_PANEL_TPW
[ "$map" = table ] || export FLERP_PANEL_RW=${map%:*} FLERP_PANEL_TPW=${map#*:}
ionice -c3 nice -n 19 /data/tools/therock-10.1.0a20260823/bin/rocprofv3 --pmc FETCH_SIZE --kernel-trace \
  --output-format csv -d "$res/$name" -o prof -- "$B" --weights "$W" --out "$res/$name.jsonl" --decode-m 1,8 \
  --prefill-m "" --reps 3 --warm 0 --only dense --class gdn_in=$L.0.linear_attn.in_proj \
  --class sh_gu=$L.0.mlp.shared_expert.gate_up > "$res/$name.log" 2>&1
