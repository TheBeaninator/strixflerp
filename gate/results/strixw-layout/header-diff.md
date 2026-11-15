# Qwen3.8-Flash-Next strixw: fleet (OLD) vs upstream (NEW), headers and indexes only

Read 2026-11-11 with `gate/strixw_header.py` (header hash and index hash checked against the file; no data region read;
`sha256.txt` not re-verified, that would read the data).

- OLD: worker B `/cache/models/qwen38-fnext-tpn/weights.strixw`. B's live process (pid 4355):
  `strix_server.v070tg --weights /cache/models/qwen38-fnext-tpn/weights.strixw --ngram
  /data/models/strix-infer/converted/qwen38-fnext-ngram-int8/ngram.table ... --chunk 8192 --mtp on --total-slots 2`
  (no vision flag in the args; vision.strixw sits beside the weights).
  strixw v4, 74,579,656,704 B, converter_git ed4c1f4, created 2026-10-05T07:56:13Z.
- NEW: ark `wemoh__Qwen3.8-Flash-Next-strixw/raw/converted/Qwen3.8-Flash-Next.U-gdn_in-g128/weights.strixw`.
  Header **version 3** (README says 4; the reader accepts 1..4, `strixw.hpp:24`), 70,714,568,704 B,
  converter_git eb57cdb, created 2026-09-28T00:47:55Z. sha256 per `sha256.txt`: `2d166ca5...d2e74b`.

## Layout strings (index metadata `layout`)

```
OLD exp_gu=q4g128,exp_down=q4g32, sh_gu=q8g64,sh_down=q8g32,gdn_in=q4g128,...,lm_head=q8g64,mtp_fc=q8g64,embed=bf16
NEW exp_gu=q4g128,exp_down=q4g128,sh_gu=q8g64,sh_down=q8g64,gdn_in=q4g128,...,lm_head=q8g64,mtp_fc=q4g64,embed=bf16
```

## Per class (bytes from the index; main model + MTP layer)

| class | OLD enc / G | OLD bytes | NEW enc / G | NEW bytes |
|---|---|---|---|---|
| routed experts gate/up [655360,2560] E512, 48+1 | q4 G128 | 42,781,900,800 + 891,289,600 | q4 G128 | same |
| **routed experts down** [1310720,640] E512, 48+1 | q4 **G32** | 25,165,824,000 + 524,288,000 | q4 **G128** | 21,390,950,400 + 445,644,800 |
| shared gate/up [1280,2560], 48+1 | q8 G64 | 167,116,800 + 3,481,600 | q8 G64 | same |
| **shared down** [2560,640], 48+1 | q8 **G32** | 88,473,600 + 1,843,200 | q8 **G64** | 83,558,400 + 1,740,800 |
| GDN in_proj [16480,2560] x36 | q4 G128 | 806,860,800 | q4 G128 | same |
| GDN out_proj [2560,6144] x36 | q8 G64 | 601,620,480 | q8 G64 | same |
| attn qkv [13952,2560] x12+1 | q8 G64 | 455,393,280 + 37,949,440 | same | same |
| attn o [2560,6144] x12+1 | q8 G64 | 200,540,160 + 16,711,680 | same | same |
| HC mix_down 96 + global + 2 MTP + MTP mixer | q8 G64 | 338,411,520 + 3,481,600 + 10,531,840 | same | same |
| HC mix_up 96 + global + 3 MTP | q8-chunk-major G64 | 334,233,600 + 3,481,600 + 10,444,800 | same | same |
| PLE kv_proj [12800,2560] | q8 G64 | 34,816,000 | same | same |
| LM head [248320,2560] | q8 G64 | 675,430,400 | same | same |
| **MTP fc_embedding + fc_hidden** [2560,2560] x2 | **q8 G64** | 13,926,400 | **q4 G64** | 7,372,800 |
| embedding [248320,2560] | bf16 | 1,271,398,400 | bf16 | same |
| router + shared gate [513,2560] x48+1 | bf16 | 126,074,880 + 2,626,560 | same | same |
| norms, conv, A_log, dt_bias, i64 | f32/i64 | same | f32/i64 | same |
| **total tensor bytes** | | 74,578,555,672 | | 70,713,467,672 (-3.865 GB, -5.2%) |

Exactly 100 of 853 records differ, all in the three expected classes (49 experts_down, 49 shared down_proj,
2 MTP fc). Tensor names, order, shapes, expert counts and part counts are otherwise identical. Other metadata
differences: `source_dir`, `created`, `converter_git`; `shared_expert = separate` in both.

## Fingerprints and pairing

| file | source_fingerprint |
|---|---|
| OLD weights.strixw | d1121c2771a0beb1 |
| NEW weights.strixw | d1121c2771a0beb1 |
| OLD vision.strixw (kind vision, layout bf16, converter bbb3f65) | d1121c2771a0beb1 |
| fleet ngram.table `qwen38-fnext-ngram-int8` (v2, dtype 3, converter ed4c1f4, 51,840,252,928 B) | d1121c2771a0beb1 |
| ark ngram.table `ngram-q8` (v2, dtype 3, converter 5d30435, 51,840,252,928 B) | d1121c2771a0beb1 |

All five come from the same checkpoint, so NEW pairs with the fleet's vision.strixw and with either n-gram table.
Rule: the n-gram table is refused on mismatch, `STRIX_CHECK(got == want ...)` "convert both from the same
checkpoint", `catalog/strixite/executor/qwen4exp/src/runtime/qwen4exp.cpp:169-171` (a throw at model load, not a
warning). The vision tower follows the same refusal, `catalog/strixite/executor/vision_tower/src/runtime/vision_tower.cpp:54-56`.
Swapping in NEW does change the prompt-cache key: `state_fingerprint` hashes `layout` with the fingerprint
(`catalog/strixite/executor/qwen4exp_backend/src/serve/qwen4exp_backend.cpp:225-241`), so saved states from OLD are
not reused (dropped, not misread).

## Bytes per decoded token (config.json: num_experts 512, num_experts_per_tok 10)

Per expert per layer: gate/up 1,740,800 B (both); down 1,024,000 B OLD vs 870,400 B NEW.
- Routed: 10 x 48 x (1,740,800 + down) = 1,327.1 MB OLD vs 1,253.4 MB NEW (-73.7 MB).
- Dense (everything else except the embedding stack and the expert stacks, LM head and router included):
  3,846.2 MB OLD vs 3,841.2 MB NEW (-4.9 MB, all shared down: 48 x 102,400 B). Embedding row 5 KB.
- **Total: 5.173 GB OLD vs 5.095 GB NEW per token, -78.6 MB (-1.52%).**
- MTP draft step (MTP layer: 10 experts + dense + fc, LM head's Q4 draft copy excluded):
  125.3 MB vs 117.1 MB (-8.2 MB; fc alone -6.55 MB).
- N-gram rows (from SSD) are the same in both.

## Bytes per 8,192-token prefill chunk

8,192 tokens x 10 / 512 = 160 routed rows per expert per layer on average; P(an expert untouched) ~ e^-160, so every
expert stack is read: all expert stacks + dense per chunk (MTP excluded) = **71.79 GB OLD vs 68.01 GB NEW
(-3.78 GB, -5.3%)** at one pass per chunk (more if a kernel re-reads weights per token tile).

## Expected effect (inference, not measured)

- Decode: if bandwidth-bound, time per token scales with bytes: ~1.5% faster (e.g. 33 -> 33.5 tok/s plain;
  the MTP verify step reads the same main-model bytes, so a similar ~1.5% on MTP decode).
- Prefill: one pass of 72 GB at ~210-250 GB/s is ~0.3 s against ~6.8 s per chunk at the fleet's ~1,200 tok/s, so
  prefill is not bandwidth-bound on weights; saving 3.78 GB is at most ~0.02 s/chunk at one pass. Any gain would come
  from G128's 4x fewer scale/min windows in the grouped expert kernel (compute), not bytes: expect <~3%, unmeasured.
- Accuracy: NEW is the layout upstream measured (KL 0.028, 226/242 top-1, README); OLD's G32 is finer, so not worse.

## Why the pod converted with G32 (record found)

`/data/pod/plans/strixite-tp2/prep.md` (ST-prep2, card fbdbff0d05, 2026-10-05):
> One g32 conversion serves both N = 2 and N = 4: at N = 4 the expert intermediate splits 640 -> 4 x 160, and 160 is
> not a multiple of 64 (exp_down q4g64 would fail K % G at the split loader), so `exp_down` and `sh_down` are
> converted at group size 32 (160 % 32 == 0; N = 2 gives 320 % 32 == 0 too).

Its layout input was `default=q8g64,exp_gu=q4g128,exp_down=q4g32,sh_down=q8g32,gdn_in=q4g128`, so `mtp_fc=q8g64`
came from `default=q8g64`, not a choice. Corroborated: `/data/pod/plans/strixite-perf-plan-2026-10-05.md:7,48-58`
("the g32 weights"), `2026-10-05-synergy-optimization.html` ("v3 says use the g32 weights from ST-prep2"), and
`convert.log` beside the file (line 3, the layout). Consequence: NEW's G128 down cannot be split for tensor
parallelism at all (640/2 = 320 and 640/4 = 160 are not multiples of 128); it suits single-node serving only, which
is how B runs today (no TP args). No engines.conf comment about the layout was found.
