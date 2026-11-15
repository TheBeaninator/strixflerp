# Raw-checkpoint path to strixw v5 panels (format/tool.flerp.convert_qwen4exp_p16), 2026-11-12

No GPU, no service. Everything ran on control-node under `ionice -c3 nice -n 19` with 4 threads, after checking the
load first (6-9). The checkpoint is `/mnt/ark/models/Qwen__Qwen3.8-Flash-Next/raw` (131 shards, 335.28 GiB,
`checkpoint_fingerprint` d1121c2771a0beb1). Only the named tensors' byte ranges were read.

## Unit tests (laptop): PASS
`python3 -m unittest tests.test_convert_qwen4exp_p16` ran 5 tests, all OK. The checkpoint is synthetic: GDN and
full-attention layers, PLE, MTP, two shards.
- Without p16, the raw path's output equals the ORIGIN converter's (strixite 20937dc) tensor for tensor.
- With p16, it equals `convert_strixw_panel` applied to the origin's output: every component's bytes and hash, with
  the shared expert both split and stacked, and hc_down in panels.
- `--check-against` passes against both, and fails on a real difference.
- The `--plan` file size equals the written file's size.
- Refusals: p16 on hc_up, embed, Q5 and default; a malformed suffix; Q8 experts.

## Real tensors vs the fleet file: PASS (`check-tensors-fleet.txt`)
The REF is `/data/models/strix-infer/converted/qwen38-fnext-tpn/weights.strixw` (v4). That file's bytes are what
the harvested converter produced at the fleet layout. 18 tensors were compared, with 0 differences.
- **p16 classes:**
  - exp_gu (layer 0)
  - exp_down (layer 0 and MTP)
  - sh_gu
  - sh_down
  - gdn_in
  - gdn_out
  - attn_qkv (layer 3)
  - attn_o (layer 3)
  - ple_kv
  - lm_head
  - mtp_fc
- **Two checks per p16 tensor:**
  - (a) our panels, inverted, are byte-identical to REF's q/scale/min planes;
  - (b) REF's planes, packed with `panel16_pack` (convert_strixw_panel's relayout), are byte-identical to our
    panels.
- **Other classes, byte-identical components:** hc_down (fold), hc_up (chunk-major), embed (bf16), router (bf16),
  f32 (A_log), i64 (PLE multipliers).
- **Cost:** 2 min 28 s, 13.6 GB read, 4.5 GB maximum RSS.

## --plan for the full model (`plan-default.txt`)
- **Layout:** `exp_gu=q4g128p16,exp_down=q4g32p16,sh_gu=q8g64p16,sh_down=q8g32p16,gdn_in=q4g128p16,gdn_out=q8g64p16,attn_qkv=q8g64p16,attn_o=q8g64p16,hc_down=q8g64,hc_up=q8g64,ple_kv=q8g64p16,lm_head=q8g64p16,mtp_fc=q8g64p16,embed=bf16`
- **source_fingerprint:** d1121c2771a0beb1.
- **Tensor bytes:** 74,578,555,672, equal to the fleet's (header-diff.md).
- **File size:** 74,579,640,320 B, v5, data offset 167,936. This is exactly the size of the order-only output in
  `../strixw-panel-run/`.

## Full conversion from the raw checkpoint (NOT run)
```
catalog/flerp/tool/convert_qwen4exp_p16/src/flerp/convert_qwen4exp_p16/build.sh /tmp/cq
ionice -c3 nice -n 19 /tmp/cq --src /mnt/ark/models/Qwen__Qwen3.8-Flash-Next/raw --git <strixflerp commit> \
  --threads 4 --out <dir on a disk with >= 75 GB free>
```
- **Output:** 74,579,640,320 B, plus convert.log. Companions (vision.strixw and the configs) pair by fingerprint, so
  they can be linked or copied from the fleet directory.
- **Reads:** the whole checkpoint, 335 GiB, from the ark disk.
- **Estimated time (not measured):** the check above quantized at about 92 MB/s of source with 4 idle-priority
  threads. At that rate the run takes about 1 h 05 min; more threads would go faster, up to the ark disk's read
  rate.
- **Memory:** about 2-5 GB (the origin streams 256 MB chunks; the largest relayout is 0.9 GB).
