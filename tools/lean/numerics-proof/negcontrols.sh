#!/bin/bash
# negative controls for lake comparator; restores Proofs.lean at the end
cd "${1:?usage: negcontrols.sh <work dir with the built project>}"
export ELAN_HOME=/data/models/tools/lean-formal/elan-home PATH=/data/models/tools/lean-formal/elan-home/bin:/data/models/tools/lean-formal/bin:$PATH ELAN_TOOLCHAIN=leanprover/lean4:v4.36.0
cmp(){ systemd-run --user --wait --pipe --collect -p Nice=19 -p RestrictAddressFamilies=~AF_UNIX -E PATH="$PATH" -E ELAN_HOME="$ELAN_HOME" -E ELAN_TOOLCHAIN="$ELAN_TOOLCHAIN" --working-directory "$PWD" -- lake comparator --config comparator.json; }
cp Proofs.lean Proofs.good
python3 - <<'EOF'
s = open('Proofs.good').read()
i = s.index('theorem ue8m0Scale_tight'); j = s.index(':= by', i) + 5; k = s.index('\ntheorem ', j)
open('Proofs.lean', 'w').write(s[:j] + '\n  sorry\n' + s[k:])
EOF
nice -n 19 lake build Proofs 2>&1 | tail -1
cmp > comparator-sorry.out 2>&1; echo "rc_sorry=$?"
python3 - <<'EOF'
s = open('Proofs.good').read()
old = '    |x / ue8m0Scale qmax amax| ≤ qmax := by\n  have hs'
assert s.count(old) == 1
open('Proofs.lean', 'w').write(s.replace(old, '    |x / ue8m0Scale qmax amax| ≤ qmax + 1 := by\n  have hs'))
EOF
nice -n 19 lake build Proofs 2>&1 | tail -1
cmp > comparator-weak.out 2>&1; echo "rc_weak=$?"
cp Proofs.good Proofs.lean && rm Proofs.good && nice -n 19 lake build 2>&1 | tail -1
cmp > comparator-good.out 2>&1; echo "rc_good=$?"
echo NEGDONE
