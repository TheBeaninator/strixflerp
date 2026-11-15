#!/usr/bin/env bash
# decode2 sweep job (gate/strixw-panel-decode2.toml): waits for the 1-minute load <= 6 (up to 30 min), records the load
# before and after, runs the panel kernel bench with the wave map $1 = RW:TPW (FLERP_PANEL_RW / FLERP_PANEL_TPW) or
# "table" (the dispatch table) on the class set $2 (dense | experts | lm_head | sh_down | ple_kv).
# Usage: tools/decode2-bench.sh <rw:tpw|table> <set> <out-name> [reps]
set -u
cd "$(dirname "$0")/.."
map=$1 set=$2 name=$3 reps=${4:-20}
res=gate/results/strixw-panel-decode2/sweep; mkdir -p "$res"
W=/data/models/strix-infer/converted/qwen38-fnext-tpn/weights.strixw
B=build/qwen38-flash-next.strixite-p16/ninja/strixflerp-panel-kernel-bench
L=model.language_model.layers
case $set in
  dense) cls="--only dense --class sh_gu=$L.0.mlp.shared_expert.gate_up --class sh_down=$L.0.mlp.shared_expert.down_proj.weight --class gdn_in=$L.0.linear_attn.in_proj --class gdn_out=$L.0.linear_attn.out_proj.weight --class attn_qkv=$L.3.self_attn.qkv --class attn_o=$L.3.self_attn.o_proj.weight --class ple_kv=$L.1.ple.kv_proj --class mtp_fc=mtp.fc_hidden.weight";;
  experts) cls="--only experts --class exp_gu=$L.0.mlp.experts_gate_up --class exp_down=$L.0.mlp.experts_down";;
  lm_head) cls="--only dense --class lm_head=lm_head.weight";;
  sh_down) cls="--only dense --class sh_down=$L.0.mlp.shared_expert.down_proj.weight";;
  ple_kv) cls="--only dense --class ple_kv=$L.1.ple.kv_proj";;
esac
for i in $(seq 60); do awk '{exit !($1 <= 6)}' /proc/loadavg && break; sleep 30; done
echo "load_before $(cat /proc/loadavg)" > "$res/$name.load"
unset FLERP_PANEL_RW FLERP_PANEL_TPW
[ "$map" = table ] || export FLERP_PANEL_RW=${map%:*} FLERP_PANEL_TPW=${map#*:}
ionice -c3 nice -n 19 "$B" --weights "$W" --out "$res/$name.jsonl" --decode-m 1,2,3,4,5,6,7,8 --prefill-m "" \
  --active 10 --reps "$reps" --warm 3 $cls 2> "$res/$name.log"
rc=$?
echo "load_after $(cat /proc/loadavg)" >> "$res/$name.load"
exit $rc
