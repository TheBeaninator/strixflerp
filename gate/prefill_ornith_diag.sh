#!/usr/bin/env bash
# PREFILL-ORNITH diagnostic: whole MoE blocks of the 35B through the grouped path (B) and the current path (A),
# dumped by --stage-bench, checked against gate/moe_block_check.py (numpy float64, llama.cpp gguf-py).
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
O=$R/build/pfo-diag
mkdir -p $O
ROCM=/data/tools/therock-10.1.0a20260823
export LD_LIBRARY_PATH=$ROCM/lib:$ROCM/lib/llvm/lib
BIN=$R/build/qwen36-35b-a3b.flerp-moe/out/flerp-qwen35moe-generate
M=/data/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf
head -c 100000 $R/gate/results/ornith15-397b.p4/long.ids > $O/long.ids
for arm in B A; do
  g=$([ $arm = B ] && echo 1 || echo 0)
  FLERP_MOE_GROUPED=$g $BIN --model $M --placed --blocks 0:40 --context 4096 --stage-bench --chunk-rows 1024 --decode-steps 2 \
    --dump $O/dump-$arm.bin --dump-layers 0,1,19,39 --dump-rows 0,1,2,63,64,100,511,1023 --ids-file $O/long.ids > $O/bench-$arm.jsonl 2> $O/bench-$arm.err
done
