# numerics-proof: Lean proofs for the dsv41 numerics contract rules

This is a Lean 4.36 + Mathlib v4.36.0 project with two modules:

- `Statements.lean` holds the definitions and the 19 theorem statements, each proved by `sorry`. **This is the file a reviewer must trust.** Every definition has a comment naming the contract text or Python function it models.
- `Proofs.lean` repeats each definition and statement word for word and proves them. It adds its own helper lemmas.
- `lake comparator` (config in `comparator.json`) checks two things: that every named theorem in Proofs has the same statement as in Statements, and that the proofs use only `propext`, `Classical.choice` and `Quot.sound`. Both the Lean kernel and nanoda check the proofs.
- `Axioms.lean` prints `#print axioms` for each theorem.

## What is stated, and the verdicts

| # | theorem | verdict |
|---|---|---|
| 1a | `exact_no_false_fail`: every pick set an in-band score vector produces passes band_rule.set_ok | proved |
| 1b | `exact_no_false_pass`: every pick set set_ok accepts is produced by some in-band vector (bands >= 0) | proved |
| 1c | `exact_accept_iff_inf_sup`: the pairwise form equals set_ok's min/max form | proved |
| 1d | `classify_may`: classify's may (must or free) is exactly "in some reachable set" | proved |
| 1e | `classify_must`: classify's must is exactly "in every reachable set" | proved |
| 1f | `argmax_exact`: argmax_ok (k = 1) is exact | proved |
| 2a, 2b | `old_rule_false_fail_ce1` / `_ce2`: the old summed-band rule (numerics.json 1.5.0-1.7.0, t_route.py) fails in-band results when bands differ | counterexamples, kernel-checked |
| 2c | `old_rule_unsound`: "the old rule accepts every reachable set" is false | proved (from CE1) |
| 2d | `old_rule_sound_equal_bands`: with one band for all positions the old rule never false-fails | proved |
| 2e | `old_rule_false_pass_fp1`: even with equal bands the old rule accepts an unreachable set | counterexample, kernel-checked |
| 3a | `e4m3_ue8m0_bf16_literal_false`: "e4m3 x UE8M0 is exact in bf16" for every UE8M0 scale | **false** |
| 3a | `e4m3_ue8m0_bf16_iff`: true exactly for scale codes 3..246 (2^-124 .. 2^119) | proved (iff) |
| 3a | `e4m3_ue8m0_bf16_edges`: 448 x 2^120 overflows bf16, and 2^-9 x 2^-125 = 2^-134 underflows | proved |
| 3b | `ue8m0Scale_bound`: s = 2^ceil(log2(amax/qmax)) keeps abs(x/s) <= qmax whenever abs(x) <= amax | proved |
| 3b | `ue8m0Scale_tight`: amax/s > qmax/2 (s is the smallest such power of two) | proved |
| 3b | `ue8m0Scale_e4m3_exponent`: for amax in [1e-4, 2^128] the exponent lies in -22..120 | proved |
| 3c | `e2m1_ue8m0_fp32_literal_false`: "E2M1 x UE8M0 is exact in fp32" for every UE8M0 scale | **false** |
| 3c | `e2m1_ue8m0_fp32_iff`: true exactly for scale codes <= 252 (6 x 2^126 overflows fp32) | proved (iff) |

How the formats are modelled: each format is decoded bit for bit from its code into an exact rational. bf16 and fp32 use one IEEE 754 decoder (`ieeeDecode 8 7`, `ieeeDecode 8 23`). E4M3 (fn variant, no infinities) and E2M1 have their own decoders. UE8M0 is the set of 2^(c-127) for c < 255. The scale lemma (3b) is stated over Mathlib reals, so it does not model the float rounding of amax/qmax; the contract's clamp to +-448 covers that.

## Build and check (control-node, offline)

The toolchain is the offline install under `/data/models/tools/lean-formal`. `.lake/packages` is not committed. Fill it with hard links from the Mathlib tree that install-lean-formal.sh builds:

```bash
export ELAN_HOME=/data/models/tools/lean-formal/elan-home ELAN_TOOLCHAIN=leanprover/lean4:v4.36.0
export PATH=$ELAN_HOME/bin:/data/models/tools/lean-formal/bin:$PATH
D=/srv/numerics-proof               # any dir outside /home: lake comparator's bwrap hides home dirs
cp -r tools/lean/numerics-proof/. $D/ && cd $D
mkdir -p .lake/packages && cp -al "$MATHLIB_TREE"/.lake/packages/. .lake/packages/   # MATHLIB_TREE: a built Mathlib v4.36.0 project
nice -n 19 lake build                                   # about 10 s with the Mathlib cache
nice -n 19 lake env lean Axioms.lean                    # each theorem: [propext, Classical.choice, Quot.sound]
systemd-run --user --wait --pipe --collect -p Nice=19 -p RestrictAddressFamilies=~AF_UNIX \
  -E PATH="$PATH" -E ELAN_HOME="$ELAN_HOME" -E ELAN_TOOLCHAIN="$ELAN_TOOLCHAIN" --working-directory "$D" \
  -- lake comparator --config comparator.json           # ends "Your solution is okay!"
```

`comparator.json` gives the path of the nanoda binary bundled with the 4.36.0 toolchain. `negcontrols.sh` runs two negative controls and then restores Proofs.lean. In the first, one proof is replaced by `sorry`. In the second, one statement is weakened. The comparator must reject both.
