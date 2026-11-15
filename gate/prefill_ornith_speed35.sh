#!/usr/bin/env bash
# PREFILL-ORNITH T1 re-measure after the column-fast grid order and BM 32/128 (one load, arms A B x3).
R=$(cd "$(dirname "$0")/.." && pwd)
ROCM=/data/tools/therock-10.1.0a20260823
export LD_LIBRARY_PATH=$ROCM/lib:$ROCM/lib/llvm/lib
O=$R/build/${SPEED_OUT:-pfo-35b-speed3}; mkdir -p $O
$R/build/qwen36-35b-a3b.flerp-moe/out/flerp-qwen35moe-generate --model /data/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --context 4096 --ids-file $R/build/pfo-35b-r2/p6.ids --ids-file $R/build/pfo-35b-r2/p7.ids --max-tokens 2 --arms ${ARMS:-A,B,A,B,A,B} > $O/out.jsonl 2> $O/err.txt
