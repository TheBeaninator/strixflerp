# Edge-REAP datasets from normal traffic

owner's expert-pruning script (`catalog/flerp/tool/edge_reap_keepset/src/flerp/edge_reap_keepset/strix_moe_surgery.py`,
credited to owner, unchanged) picks the routed experts each MoE layer keeps from calibration statistics: per layer and
expert, how often the router picked it, its gate mass, its REAP saliency (gate weight times the L2 norm of the expert's
full output, averaged over the tokens it saw), its peak output norm and how often it fired on the first 64 positions.
The serving engine collects those statistics passively while it answers ordinary requests, so the calibration set
is the pod's real traffic, split by what the traffic was for.

## Where the numbers come from

The engine is StrixFold's DeepSeek-V4.1-Flash (TP4 on four nodes, or TP2 on two) with `REAP_STATS=1` in the
serving conf (off by default; with it off the build behaves exactly as before). The collector is the module
`runtime.strixfold_reap.edge_reap` (`catalog/strixfold_reap`), whose design and cost are in its origin's
`docs/experiments/tp4-reap-stats.md`. In short:

- **Routing tier, every committed token.** `token_count`, `gate_mass_sum` (the gate weight after the top-6
  renormalisation and the 1.5 routed scaling, what the combine multiplies by) and `early_pos_count` come from the
  router's own picks and weights, on rank 0, with no extra traffic between nodes. Prompt tokens all count; in a
  speculative verify round only the accepted rows count. Drafter (DSpark/MTP) forwards and the three MTP layers
  40..42 never count.
- **Norm tier, a fixed sample.** `||f_j(t)||` must be the norm of expert j's whole output. Under tensor parallelism
  each node holds only a K-partial of it (the experts' intermediate is split between the TP2 ranks), and the sum of
  partial norms is not the norm of the sum. So on a sampled round the TP2 partner's per-slot partial is sent to the
  pair's rank 0, which adds the two vectors and takes the sum of squares over its columns; at TP4 the second pair's
  sums of squares follow to rank 0. That moves ~2.5 MB (TP4) or ~4.9 MB (TP2) per sampled token, so the engine
  samples: row 0 of every `REAP_EVERY`-th decode round (default 8) and `REAP_PREFILL_ROWS` evenly spaced rows of
  each prompt chunk (default 4). Expected cost at the defaults is about 0.3 % of a TP4 decode round and 0.5 % of a
  TP2 one (computed in the origin doc; the hardware check measures it).
- **Export semantics.** `reap_saliency` is the mean of `g * ||f||` over the sampled pairs, and
  `reap_weighted_norm_sum = reap_saliency * token_count`, so the script's identity
  `reap_saliency = reap_weighted_norm_sum / token_count` holds and `compute_edge_reap_keepset` reads the file as it
  reads its own collector's. `max_l2_norm` is over the sampled pairs. The raw sampled sums are kept under
  `sampled` (`norm_count`, `reap_weighted_norm_sum`). With `REAP_EVERY=1` and `REAP_PREFILL_ROWS` at least the
  chunk size every committed token is sampled and the file equals the script's `export_report`.

## Domain tags

A request names what it is for with `"reap_domain": "<tag>"` in the body (OpenAI chat and completions, Anthropic
messages) or the header `x-reap-domain: <tag>` (the body wins). A tag is 1-32 characters of `a-z 0-9 _ . -`;
untagged requests count under `default`. Each tag has two buckets, text and vision (a token's modality is which
router bias it was routed with: `gate.bias` for text, `gate.bias_vl` for image tokens). `REAP_BUCKETS` (default 16)
caps the number of tags: a request that names a new tag past the cap gets HTTP 400 naming the cap; known tags and
untagged requests always work. Suggested tags: `code`, `chat`, `vision`, `game`, one per client that matters.

## Getting the data out

- **Endpoint.** `GET /stats/reap?domain=<tag>&modality=text|vision|all` (default `all`: the two buckets added,
  peaks maxed) with the server's admin key, the same key file and header as the `/cache` routes
  (`--cache-transfer-key`, header `x-cache-transfer-key`). The body is the script's report (`model_id`,
  `num_layers`, `num_experts`, `reap_saliency`, `reap_weighted_norm_sum`, `gate_mass_sum`, `token_count`,
  `max_l2_norm`, `early_pos_count`, in that order) then `domain`, `modality`, `tokens`, `norm_rows`, `sampled`
  and `build`. `GET /stats/reap` with no domain lists the buckets and their token totals. 404 when
  `REAP_STATS` is off or no key is set, 401 without the key.
- **Snapshots.** With `REAP_DIR=<dir>` the server writes `<dir>/<tag>.<modality>.json` every
  `REAP_SNAPSHOT_MIN` minutes (default 10) and at a clean shutdown, atomically (temp file, fsync, rename). On start it
  reloads every file whose model and shape match and keeps accumulating, so a restart loses at most one interval. A
  file that does not match is renamed `.bad` and left alone. Put `REAP_DIR` on the serving node's local disk and
  copy it off with the rest of the node's state; the files are small (about 1 MB each).

## Feeding the pruning script

`edge_reap_report.py` (next to the script) reads, merges and solves:

```bash
T=catalog/flerp/tool/edge_reap_keepset/src/flerp/edge_reap_keepset
python3 $T/edge_reap_report.py fetch --url http://<rank0>:<port> --key-file <key> --domain code --out code.json
python3 $T/edge_reap_report.py merge snapshots/code.text.json snapshots/code.vision.json --out code.json
python3 $T/edge_reap_report.py merge snapshots/*.json --out core.json        # every domain: a general core report
python3 $T/edge_reap_report.py keepset --report code.json --core-report core.json --keep-experts 192 --out keep.json
python3 $T/strix_moe_surgery.py slice --src-dir <hf checkpoint> --out-dir <pruned> \
    --domain-report code.json --core-report core.json --keep-experts 192
```

`merge` adds reports of one model (counts and gate mass summed, peaks maxed, saliency recomputed from the raw
sampled sums) and its output is a valid `--domain-report` / `--core-report` for the script. A domain needs enough
traffic before its keep-set means anything: `compute_edge_reap_keepset` damps experts seen fewer than about
`tau_domain` (64) times, and with 6 of 384 experts per token a layer's experts see a token about every 64 routed
tokens, so plan on tens of thousands of tokens per domain, and more sampled rows (`norm_rows` in the export) for
the saliency to settle.

## Masks for the engine's MASK_EXPERTS (KCUT)

`mask_keepset.py` turns a domain's snapshots into the file the engine's `MASK_EXPERTS` loads (the removed experts, one
`layer:e1,e2,...` line a layer), for masked trials before any checkpoint is cut:

```bash
python3 $T/mask_keepset.py --snap-dir snapshots --domain code --keep 0.5 --out code-k50.mask --report code-k50.json
```

Per layer: budget = round(keep x 384), never below 6 (the engine refuses fewer); an expert holding at least
`--hub-share` (0.02) of the layer's gate mass in any snapshot of any domain in the directory is kept; the rest of the
budget goes to the highest `reap_saliency` (ties to the lower id; `--count-tau` damps rare experts as the script's
score does). MTP layers and the shared expert are never named. `--report` writes the per-layer coverage of saliency
mass, traffic and gate mass (json and a .md beside it). Tests: `tests/test_mask_keepset.py` (the mask is parsed with
`tools/ablate.py`'s `parse_mask`, which pins the engine's refusals and canonical id).

Tests: `tests/test_edge_reap_keepset.py` (a report the script's own collector exports, and the engine's export
format, round-tripped into keep-sets; merges; refusals).
