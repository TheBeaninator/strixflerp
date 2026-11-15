# strixw panel encodings: results against gate/strixw-panel.toml (2026-11-12, commit 2bbe325)

No GPU, no worker, no service. control-node runs under `ionice -c3 nice -n 19`, load checked first (waited 19.4 -> 7.4).
Tool built on control-node from `catalog/flerp/tool/convert_strixw_panel/src/flerp/convert_strixw_panel/build.sh`.

## (a) unit, laptop + clean worktree of 2bbe325: PASS
`python3 -m unittest tests.test_convert_strixw_panel`: 7 tests OK. The full suite also passes (194 tests), and `./flerp check`
reports 0 errors. The C++ selftest covers 13 shapes x {Q4, Q8} (the 11 fleet classes plus upstream exp_down G128 and
sh_down G64). For each: offsets as designed, round trip exact, component size = q+s+m. Every fleet class is
128-aligned. **Finding:** the design's inference "64K/G % 128 == 0 for every class with K >= 256" fails for K=640 at
G128 (upstream exp_down: panel 5,440 B, 64-aligned only). The fleet's exp_down is G32 and is aligned.

## (b) real tensors, read-only byte ranges of the fleet file: PASS
`check-tensors-fleet.txt` covers 12 tensors, one or more per default class (exp_gu, exp_down x2, sh_gu, sh_down, gdn_in,
gdn_out, attn_qkv, attn_o, ple_kv, lm_head, mtp_fc). For every one:
- each component's hash equals OLD's index;
- offsets are as designed;
- panel total = q+s+m;
- the inverse is IDENTICAL.

The run read 2.75 GB in 4.1 s, with 3.4 GB max RSS.

## (c) --plan on the fleet file: PASS
`plan-fleet-default.txt` covers the default classes. Tensor counts are 49 49 49 49 36 36 13 13 1 1 2, as in
header-diff.md, and every class is 128-aligned. hc_up, hc_down, embed, router and F32/I64 are copied (555 tensors,
2,111,089,432 B). 298 tensors are converted (72,467,466,240 B).
- **Size: 74,579,640,320 B vs OLD 74,579,656,704 B, -16,384 B.** Every tensor's bytes are unchanged.
- The whole difference is the index: 298 tensors x 2 fewer component records (25 B each) moves data_offset from
  184,320 to 167,936.
- No 256-byte padding existed between OLD's planes.
- source_fingerprint d1121c2771a0beb1 is kept. The layout string gains `p16` on the 11 classes.

## Production run (not done; the owner schedules it)
Requirements:
- On a worker with >= 75 GB free on `/cache` (74,579,640,320 B output) and OLD at
  `/cache/models/qwen38-fnext-tpn/weights.strixw`.
- Not while that worker serves from the same disk, if avoidable.
- Build on control-node (`build.sh /tmp/convert_strixw_panel`) and copy the binary over.

```
mkdir -p /cache/models/qwen38-fnext-tpn-p16 && ionice -c3 nice -n 19 /tmp/convert_strixw_panel \
  --old /cache/models/qwen38-fnext-tpn/weights.strixw --out /cache/models/qwen38-fnext-tpn-p16/weights.strixw
```

Estimated cost, not measured:
- I/O is about 300 GB: OLD read once to write, plus once more for the 72.5 GB converted while verifying; the output
  written once and read once. At 1.5-2 GB/s local NVMe that is about 25-35 min.
- One CPU thread.
- Memory peaks at about 4 GB, during verification of an exp_gu tensor.

The output is v5, so no current strixite binary will load it; that needs stages 3-5. vision.strixw and the n-gram table
pair unchanged (same source_fingerprint).
