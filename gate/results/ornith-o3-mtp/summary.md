# ORNITH-O3 step one: MTP draft acceptance, 35B proxy

Written 2026-11-13T20:17:10Z on control-node by gate/ornith_o3_mtp.py judge. Criteria: gate/ornith-o3-mtp.toml (committed before any GPU run). **PASS**

## Criteria

- **1_ids_unchanged**: pass - {}
- **2_unit_check**: pass - {"steps": 71, "min_cos": 0.9999999999996994, "max_rel_l2": 7.751425615781352e-07, "logits_steps": 24, "argmax_agree": 24, "confident_steps": 20, "confident_agree": 20, "kinds": {"0": 1, "1": 54, "2": 16}}
- **3_llama_cpp**: pass - {"llama": {"1": {"draft_n": 1031, "draft_n_accepted": 890, "rate": 0.8632395732298739, "error": null}, "3": {"draft_n": 1945, "draft_n_accepted": 1276, "rate": 0.6560411311053984, "error": null}}, "probe_sched_n1": 0.8700823421774931, "probe_sched_n3": 0.6585721223895095}

## Acceptance (Qwen3.6-35B-A3B blk.40, greedy, 256 tokens, chat, thinking off)

| class | k | frontiers | marginal P(d_k right) | chain P(d_1..d_k right) | conditional a_k | E tokens/round at k |
|---|---|---|---|---|---|---|
| code | 1 | 1020 | 0.955 | 0.955 | 0.955 | 1.955 |
| code | 2 | 1016 | 0.880 | 0.869 | 0.910 | 2.824 |
| code | 3 | 1012 | 0.788 | 0.757 | 0.871 | 3.581 |
| prose | 1 | 1020 | 0.783 | 0.783 | 0.783 | 1.783 |
| prose | 2 | 1016 | 0.564 | 0.538 | 0.687 | 2.322 |
| prose | 3 | 1012 | 0.393 | 0.339 | 0.630 | 2.661 |
| all | 1 | 2040 | 0.869 | 0.869 | 0.869 | 1.869 |
| all | 2 | 2032 | 0.722 | 0.704 | 0.810 | 2.573 |
| all | 3 | 2024 | 0.590 | 0.548 | 0.779 | 3.121 |

llama.cpp-schedule simulation of the probe (rounds as draft-mtp runs them): code: n_max 1 0.964, n_max 3 0.852, prose: n_max 1 0.785, n_max 3 0.520, all: n_max 1 0.870, n_max 3 0.659

## Staged hook (--staged tcp world 2 on control-node, probe on rank 1)

```
{
 "code0": {
  "probe_ids_prefix_equals_rank0": true,
  "rank0_equals_on_run": true,
  "drafts_equal_on_run": "256/256",
  "prompt_rows_seen": 47,
  "prompt_tokens": 47
 },
 "code1": {
  "probe_ids_prefix_equals_rank0": true,
  "rank0_equals_on_run": true,
  "drafts_equal_on_run": "256/256",
  "prompt_rows_seen": 34,
  "prompt_tokens": 34
 },
 "code2": {
  "probe_ids_prefix_equals_rank0": true,
  "rank0_equals_on_run": true,
  "drafts_equal_on_run": "256/256",
  "prompt_rows_seen": 35,
  "prompt_tokens": 35
 },
 "code3": {
  "probe_ids_prefix_equals_rank0": true,
  "rank0_equals_on_run": true,
  "drafts_equal_on_run": "256/256",
  "prompt_rows_seen": 30,
  "prompt_tokens": 30
 },
 "prose0": {
  "probe_ids_prefix_equals_rank0": true,
  "rank0_equals_on_run": true,
  "drafts_equal_on_run": "256/256",
  "prompt_rows_seen": 31,
  "prompt_tokens": 31
 },
 "prose1": {
  "probe_ids_prefix_equals_rank0": true,
  "rank0_equals_on_run": true,
  "drafts_equal_on_run": "256/256",
  "prompt_rows_seen": 29,
  "prompt_tokens": 29
 },
 "prose2": {
  "probe_ids_prefix_equals_rank0": true,
  "rank0_equals_on_run": true,
  "drafts_equal_on_run": "256/256",
  "prompt_rows_seen": 27,
  "prompt_tokens": 27
 },
 "prose3": {
  "probe_ids_prefix_equals_rank0": true,
  "rank0_equals_on_run": true,
  "drafts_equal_on_run": "256/256",
  "prompt_rows_seen": 28,
  "prompt_tokens": 28
 }
}
```

MTP ms a draft step (35B, recorded): {"code0": 2.486, "code1": 2.5046, "code2": 2.5019, "code3": 2.52, "prose0": 2.5327, "prose1": 2.8236, "prose2": 2.5237, "prose3": 2.5201}

