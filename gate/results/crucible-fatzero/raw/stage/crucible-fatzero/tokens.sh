#!/bin/bash
# tokens.sh "TREE:SPECS ..." : gen on the three gate prompts; outputs in TREE/build/kbout/<label>.out|.err
for ts in "$@"; do
  T=${ts%%:*}; SPECS=${ts#*:}
  mkdir -p $T/build/kbout
  for p in short-chat-thinking long-chat raw-completion; do
    for s in ${SPECS//,/ }; do
      L=$T-$p-spec$s; t0=$(date +%s)
      bash gen.sh $T $p $s > $T/build/kbout/$L.out 2> $T/build/kbout/$L.err; rc=$?
      echo "$L rc=$rc wall=$(( $(date +%s) - t0 ))s $(grep -h -E '^loaded' $T/build/kbout/$L.out $T/build/kbout/$L.err | head -1)"
      grep -h -E '^prefill' $T/build/kbout/$L.out; grep -h -E '^ids:' $T/build/kbout/$L.out | md5sum
      [ $rc = 0 ] || { tail -5 $T/build/kbout/$L.err; exit $rc; }
    done
  done
done
