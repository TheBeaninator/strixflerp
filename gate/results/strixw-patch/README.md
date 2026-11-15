# tool.flerp.strixw_patch: proof on the real files (2026-11-11, control-node, read-only)

Binary built on control-node from the module's `build.sh` (g++, host only).

## Source checkpoint
`/mnt/ark/models/Qwen__Qwen3.8-Flash-Next/raw` (= `/data/ark/models/Qwen__Qwen3.8-Flash-Next/raw`):
`checkpoint_fingerprint` (strixite's own) = `d1121c2771a0beb1`, equal to OLD's and NEW's `source_fingerprint`.

## mtp_fc against the upstream file
`strixw_patch --source <ckpt> --override mtp_fc=q4g64 --check-against
/data/ark/models/wemoh__Qwen3.8-Flash-Next-strixw/raw/converted/Qwen3.8-Flash-Next.U-gdn_in-g128/weights.strixw`
(reads only the six component byte ranges of NEW):

```
mtp.fc_embedding.weight  q 3276800 / scale 204800 / min 204800 bytes: IDENTICAL, hashes ae2d1e3ad3fbf2c6 2ef0e426b58116f3 20d62deebcfe0092 = NEW's index
mtp.fc_hidden.weight     q 3276800 / scale 204800 / min 204800 bytes: IDENTICAL, hashes b4a879392d395b78 85ce8b82f2378346 1bcf49f71a70a4b1 = NEW's index
RESULT: byte-identical
```

Converter versions differ (NEW eb57cdb, fleet ed4c1f4; harvested code 20937dc) and the bytes still agree.

## Plan for the fleet file
OLD = control-node `/data/models/strix-infer/converted/qwen38-fnext-tpn/weights.strixw` (its 128-byte header, which
carries the index hash, has the same sha256 as worker C's `/cache/models/qwen38-fnext-tpn/weights.strixw`).
`plan-fleet-mtp_fc-q4g64.txt`: 853 tensors, 851 copied, 2 patched; 74,579,656,704 -> 74,573,103,104 bytes
(-6,553,600); data offset 184,320; tensors before the MTP fc keep their offsets; 23 records (the second fc and everything after it) move.

## Production run (not done yet)
On a worker with >= 75 GB free on `/cache` (worker C had 8.4 GB free on 2026-11-11), after building on control-node
(`catalog/flerp/tool/strixw_patch/src/flerp/strixw_patch/build.sh /tmp/strixw_patch`) and copying the binary over:

```
mkdir -p /cache/models/qwen38-fnext-tpn-mtpq4 && ionice -c3 nice -n 19 /tmp/strixw_patch --old /cache/models/qwen38-fnext-tpn/weights.strixw \
  --source /data/ark/models/Qwen__Qwen3.8-Flash-Next/raw --override mtp_fc=q4g64 \
  --out /cache/models/qwen38-fnext-tpn-mtpq4/weights.strixw --threads 4
```

Disk: 74,573,103,104 bytes new. I/O: OLD read once, output written once and re-read once by `verify_hashes`
(~224 GB of I/O; ~10-15 min at 1.5-2 GB/s local NVMe, estimated, not measured). Memory: one component at a time,
<= ~840 MB. `vision.strixw` and the n-gram table pair unchanged (same source_fingerprint).
