#!/bin/bash
# gen.sh TREE PROMPT SPEC [extra crucible gen args]: one crucible gen of kbin/PROMPT.ids with TREE's build, N tokens (default 128).
T=$1; P=$2; S=$3; shift 3
cd $T && exec build/engine/crucible.mono gen /data/models/crucible-exl3/4.05bpw_h6_ng6 --ids-file ../kbin/$P.ids --n ${N:-128} --spec $S "$@"
