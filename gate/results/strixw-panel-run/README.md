# Order-only panel conversion on worker A (convert_strixw_panel, production command of ../strixw-panel), 2026-11-12

## Pre-checks (00:23Z)
- `/cache` had 151.9 GB free.
- No hold and no lease on A (`pod-quad status`, `pod-lease status`).
- strix_server pid 1657436 serves `/cache/models/qwen38-fnext-tpn/weights.strixw`. That file and its directory were
  not touched.
- The output directory did not exist.

## Build and run
- **Build:** binary built on control-node from commit 0b0a3ff's sources (`build.sh`, unchanged since 87d549a),
  sha256 `d967eae97c104848f2d91a74797de84fec2c2c89afe9378e56f131cddedb2074`.
- **noexec:** `/cache` on A is mounted noexec. The binary was copied to `/cache/p16-tool` as briefed but ran from
  `/run/user/1000/p16-tool`. The log stayed in `/cache/p16-tool`.
- **Command:**
```
systemd-run --user --collect --unit=p16-convert /bin/sh -c "exec ionice -c3 nice -n 19 /usr/bin/time -v \
  /run/user/1000/p16-tool/convert_strixw_panel --old /cache/models/qwen38-fnext-tpn/weights.strixw \
  --out /cache/models/qwen38-fnext-tpn-p16/weights.strixw > /cache/p16-tool/convert.log 2>&1"
```
- **Timing:** started 00:26:32Z. Write took 322 s and verify 173 s, for 8 min 15 s wall. It used one thread (47%
  CPU), 4.2 GB maximum RSS, read 223.7 GB and wrote 74.6 GB.
- **Watch:** A's `/health` was polled every 2 minutes from control-node and stayed `"status":"ok"` with 2 busy slots
  throughout. Load on A was 1.8-5.2.

## Result: PASS (`convert.log`)
- **Tool verification:** v5, every component hash-checked, and every converted tensor inverts to OLD byte for byte.
- **Output:** `/cache/models/qwen38-fnext-tpn-p16/weights.strixw`, 74,579,640,320 B, equal to the --plan size.
- **sha256** (on A, ionice idle): `363bcc344bc7e6f4267a7780ea8db2bb3cad494bb37f3660ba7c5f85dc9e84c5`.
- **Layout:** `exp_gu=q4g128p16,exp_down=q4g32p16,sh_gu=q8g64p16,sh_down=q8g32p16,gdn_in=q4g128p16,gdn_out=q8g64p16,attn_qkv=q8g64p16,attn_o=q8g64p16,hc_down=q8g64,hc_up=q8g64,ple_kv=q8g64p16,lm_head=q8g64p16,mtp_fc=q8g64p16,embed=bf16`
- **Companions:** symlinks to OLD's directory: vision.strixw, preprocessor_config.json, config.json.
- **Cleanup:** `/cache/p16-tool` and `/run/user/1000/p16-tool` removed; the log is here.

The output is deterministic: OLD plus the same command gives the same bytes.
