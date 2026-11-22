<!-- docs/experiments/tp4-reap-stats.md at feb5614b230e, copied by docs/harvest/harvest_strixfold_reap.py -->
# TP4 Edge-REAP routing statistics (passive)

Read-only side path for the owner's `DistributedEdgeREAPCollector` (`ref/strix-moe-surgery.py`, `record_batch` / `export_report`). MoE layers 0..39 only. Never layers 40..42, never a drafter forward. Replies stay byte-identical: the expert combine still writes `out` from the same per-slot values. `REAP_STATS` default off leaves every plan, launch argument (including `discard`), exchange and reply as the base.

World 4: node g = 2p + r is TP2 rank r inside pair p (`plan.zig` `node`). Pair p owns columns `[p·2560, (p+1)·2560)` of the 5120-wide down (`plan.zig` `expertParts`, `down0` = 20). Each TP2 rank owns K = 1152 of 2304. World 2: each rank owns K = 1152 and all 5120 columns. World 1 owns both. Routed slots = 6; the table is 7 (shared expert last, id 384). Stats ignore id ≥ 384.

Every number is **computed** (arithmetic here) or **measured** (source named). Link rate 4.5 GB/s is the pod's measured rail. On-node DRAM floor 230 GB/s is the prefill note. Launch cost ~7 µs is the gather note's remainder (four `copyRows` accounting), not a new measurement.

## Where a slot's output lives

Decode, both widths. `exl3_experts.decodeLaunches` (world 2) and `exl3_experts2d.downLaunch` (world 4) pass `sc.y` as the down epilogue `y`. `exl3.hip` `grouped_cp_body` stores slot `p` at `y + p * N + n`. `member_p` is `(code >> 5) * slots + (code & 31)`: row-major `[R, slots, N]`, N this rank's columns (5120 or 2560). The completing program reads that `y` and writes the combined `out`. HIP does not read `discard` (`exl3.hip` header: no PDL / L2 discard on AMD), so `y` already reaches memory. The launch argument stays `discard = 1` while REAP is off (`4 * slots ≤ 32` is true at slots = 7).

Prompt, rows ≥ `exact_rows` (64), the served pf path (`exl3_experts.prompt` / `prompt2d.expertsDown` when `pfPlan` succeeds). `exl3_pf.hip` `place_body` sets `order[sorted_pos] = assignment i` (`i = row * slots + slot`). `down_body` writes `ypart[order[job] * H]`. `combine_body` reads `(r * topk + k) * H` with `topk = PfArgs.slots`. After down, `z` is tight `[rows, slots, H]` at `(row * slots + slot) * H`. `H` is this rank's columns (5120 TP2, 2560 TP4). The non-pf `down_combine` (`exl3.hip`, `store_y = 0`) keeps the slot vector in registers; `Z` is pre-Hadamard, not `f_j`. That layer is skipped for the norm (routing still counts). Rows under 64 take the decode window.

REAP redirects only the down epilogue's `y` (`DecodeEpi.y` in `decodeLaunches` / `downLaunch`) and that epilogue's `discard`. `decode_prep` still reads `sc.y`. Prep writes `out` for rows that routed zero slots; it does not write the routed rows' `y`. Gate and up epilogues are untouched. With REAP on, `discard` on the down epi is 0 so a captured graph does not depend on the sample bit. The down program still reads the same `y` it wrote, so `out` keeps its bits.

`gather_m` (`round.zig` stretch → `prompt2d.quarters` / `prompt.gatherF32`) moves the combined partial `[2, R, N]`. It has no per-slot payload. No rank holds a full `f_j(t)`.

## Where the norm is computed, and why it is the full output

`f_j(t)` is the sum of the K-partials over the 5120 columns. `||f||^2 = ||f_pair0||^2 + ||f_pair1||^2`, and `f_pair_p = y(p,0) + y(p,1)` over that pair's 2560 columns. Sum of per-rank norms is a different number (`||a|| + ||b|| ≠ ||a + b||`); the host test builds a case that fails if a rank-norm sum is used.

On a sampled round, after the forward, on the engine stream:

1. Pack row 0's six routed slots, 40 layers, into one buffer (the per-layer `y` slices are strided by `round.max_rows`, 48).
2. Hop 1. `Coll.exchange` with peer `g ^ 1`, site `reap_pair`. Both sides send the packed partial (the API is two-way; full duplex, so the time is one payload). The pair's r = 0 adds the two vectors in f64 and writes one sum of squares per `(layer, slot)`.
3. Hop 2, world 4 only. `Coll.exchange` with peer `g ^ 2`, site `reap_scalar`, 40 · 6 f64 sums of squares. Ranks 1 and 3 exchange a same-sized zero buffer so every rank issues the same site, peer and byte count. Rank 0 sets `||f|| = sqrt(ss_pair0 + ss_pair1)` in f64.
4. Rank 0's norm kernel updates the device accumulators. World 2 skips hop 2 (the hop-1 sum is already over 5120 columns). World 1 skips both hops (the local `y` is the full vector).

One 4-rank all-gather of the partials would move 3× the payload. The two hops move one partial per pair plus 1,920 B.

Prompt chunks do not redirect `y`. A decode-window layer (rows under `exact_rows`) packs that layer's sampled rows from scratch `y` after the layer, one launch with `layers = 1` (the pack kernel then takes the row from the sample index, and `rows_cap` is not in the address). A pf layer packs from `z` the same way. `fillSplit` runs two chunks at once, so each has its own pack region. Then one exchange after the chunk. Pack copies the first `topk` slots; the stride is the slots the down kernel was launched with (6 after TP2 `compactSlots`, 7 on the served path where the shared expert stays in the table, both widths). A mix of those strides across MoE layers turns the norm tier off on every rank. The route tier stays on.

## Routing tier (no wire)

`token_count`, `gate_mass_sum`, `early_pos_count` come from the router's `pick` / `wts`. `tri_norm.gateRoute` already writes the weight after the top-k renorm and after `routed_scaling_factor` (1.5). With REAP on, those pointers are per-layer slices of a `[40, R, slots]` staging buffer for the whole session, so a captured graph stays valid. One rank-0 kernel after acceptance scans them. No per-layer stats launch.

Committed rows (`lanes.zig` `Telemetry.verified`): a verify window of `rows` ids keeps `kept` drafts while `drawn[kept] == ids[kept + 1]`; committed = `kept + 1`. Row 0 is always committed. Drafted rows count only when accepted. Prompt rows are all committed. The norm sample is row 0 of a verify window, so the exchange does not wait on acceptance. Positions `< 64` increment `early_pos_count`. Modality is which bias the router used: a row whose position sits in an image span was re-routed with `gate.bias_vl` (`prompt.routeSpans`); every other row is text.

An encoder-only prompt chunk stops at layer 20 (`fillWhole`). Layers that did not run do not see those rows. That is the forward the engine actually computed.

## Bytes, wire, launches

Routed partial, one row, 40 layers. **Computed.**

| | floats | bytes | wire at 4.5 GB/s |
| --- | ---: | ---: | ---: |
| TP4 hop 1 (dn = 2560) | 40·6·2560 = 614,400 | 2,457,600 | 0.546 ms |
| TP2 hop 1 (dn = 5120) | 1,228,800 | 4,915,200 | 1.092 ms |
| hop 2 scalars | 240 f64 | 1,920 | 0.43 µs |

World-4 rdma slot is 8,912,896 B (`TF_COLL_SLOT_KIB` default 8704, 4096-aligned). `exchangeHalf` = alignDown((slot − 4·64) / 2, 64) = 4,456,320 B (**computed**). Hop 1's 2,457,600 B is one slot exchange: one `coll.stage`, one `coll.collect`. TP2's 4,915,200 B is two of those halves (2 · 4,456,320 = 8,912,640), so two slot exchanges on the world-4 layout, or one bulk send on world-2 `tcp_host` when the gather exceeds the 4,456,448 B slot. The wire is the payload either way.

Fixed part of one small one-slot gather, **measured** (gather note, 10 KB, ~25 µs above wire). Applied once per slot exchange here; a 2.5 MB message's fixed part is not separately measured. Hop 2 is one more ~25 µs for 1,920 B.

Launches on a sampled decode round, above the base forward. **Computed** at ~7 µs.

| launch | who | count |
| --- | --- | ---: |
| pack (strided y → one buffer) | every rank | 1 |
| hop 1 stage + collect | every rank | 2 (×2 chunks on TP2) |
| ssq `(y0 + y1)` in f64 | pair r = 0 | 1 |
| hop 2 stage + collect | every rank, world 4 | 2 |
| norm accum | rank 0 | 1 |
| route accum | rank 0, every round, sampled or not | 1 |

Critical path, rank 0, one sampled decode round. **Computed.** Wire + one measured 25 µs per slot exchange + one ~7 µs kernel for pack, ssq and norm. The stage/collect launches sit inside the exchange's fixed part (the gather note's 25 µs already contains them), so they are not added again.

| | exchanges | kernels | total |
| --- | --- | --- | ---: |
| TP4 | 0.546 ms + 2·25 µs | pack + ssq + norm = 21 µs | 0.617 ms |
| TP2 | 1.092 ms + 2·25 µs (two chunks) | pack + ssq + norm = 21 µs | 1.163 ms |

R does not change this. The sample is row 0. A ~5.5-row round pays the same REAP wire as a 1-row round.

Always on while REAP is on, including rounds that are not sampled: the route kernel (~7 µs) and the redirected `y` store. The store already happens (HIP ignores `discard`). Pointing `y` at staging does not add bytes. `discard` becomes 0 in the launch argument so a graph captured with REAP on does not depend on the sample bit.

Prompt, one 2,048-row chunk, `REAP_PREFILL_ROWS` = 4. **Computed.** Pack is one copy a layer of 4 · 6 · dn floats (TP4 245,760 B, 1.1 µs at 230 GB/s) plus the ~7 µs launch. Encoder-only chunks run 20 MoE layers, a full chunk 40. Exchange once: 4 × the one-row payload.

| | wire | copies | slot fixed | total |
| --- | ---: | ---: | ---: | ---: |
| TP4, 20 layers | 4·0.546 = 2.185 ms | 20·7 µs = 0.14 ms | 3·25 µs (9.83 MB / 4.46 MB) | 2.40 ms |
| TP4, 40 layers | 2.185 ms | 0.28 ms | 75 µs | 2.54 ms |
| TP2, 40 layers | 4·1.092 = 4.369 ms | 0.28 ms | 5·25 µs | 4.77 ms |

A 2,048-row encoder chunk's GPU floor is ~298 ms and a measured 1,818-token TTFT is 1.58 s RCCL / 1.88 s RDMA (prefill note). **Measured** rounds: TP4 forward 29.3 ms at 1 row and 55.4 ms at ~5.49 rows (gather note, served build 5ecaf207, `--profile 2`). TP2, KDEC §3, gfx1151, tcp_proxy, both boxes: drafts off 30.75 ms untraced (32.38 ms traced); drafts on 55.11 ms traced.

## Overhead at the defaults

`REAP_EVERY = 8`, `REAP_PREFILL_ROWS = 4`. Expected decode cost = route kernel every round + the sampled-round cost / 8.

| round | base | REAP expected | % |
| --- | ---: | ---: | ---: |
| TP4, 1 row | 29.3 ms | (0.617 / 8 + 0.007) = 0.084 ms | 0.29% |
| TP4, ~5.5 rows | 55.4 ms | 0.084 ms | 0.15% |
| TP2, 1 row | 30.75 ms | (1.163 / 8 + 0.007) = 0.152 ms | 0.50% |
| TP2, drafts on | 55.11 ms | 0.152 ms | 0.28% |
| TP4 prompt chunk | 298 ms floor | 2.40 ms | 0.81% of the floor, 0.15% of 1.58 s |

`REAP_EVERY = 1` is 0.617 / 29.3 = 2.1% on TP4 and 1.163 / 30.75 = 3.8% on TP2, above 1% on the wire alone. `REAP_EVERY = 4` is 0.94% of the TP2 1-row round before the route kernel, inside the unmeasured fixed part of a 4.9 MB transfer. 8 keeps about 2× against that. A prompt sample of every row (2,048) is 2048/4 · 2.19 ms ≈ 1.1 s of wire on TP4, most of a short TTFT. 4 rows stays under 1% of the chunk floor.

## Export, buckets, accumulators

Device accumulators on rank 0 only: f64 sums and i64 counts, `[domains · 2, 40, 384]`. f64, not f32+Kahan. One thread per `(layer, expert)` (15,360 threads) walks the round's rows and writes every bucket that row names. That thread is the only writer of that `(layer, expert)` across buckets. Direct global adds, no atomics, so the same traffic is the same bits. Launch-bound at decode (the f64 volume is 40·6·2560 adds on the ssq kernel, ~10^6 flops). f64 matches the script's float64 and the host port bit for bit, and it keeps the kernel off the spill line. gfx1151's f64 rate does not show up at this size. Host scalars `tokens` and `norm_rows` are updated on rank 0 beside the kernel; the kernel does not touch them.

`reap_saliency[l][j] = sampled_sum(g·||f||) / max(norm_count, 1)`. Exported `reap_weighted_norm_sum = reap_saliency · token_count`. `max_l2_norm` is over the sampled rows. `token_count`, `gate_mass_sum`, `early_pos_count` are over every committed row. `sampled` carries the raw `norm_count` and the raw sampled sum, so a reload keeps accumulating exactly. With `REAP_EVERY = 1` and `REAP_PREFILL_ROWS` at least the chunk's rows, the sample is every committed row and the export matches `export_report` exactly (`reap_weighted_norm_sum` equals the raw sum because `norm_count = token_count`).

Buckets: `reap_domain` (body field on chat, completions and Anthropic messages, else header `x-reap-domain`; body wins; default `default`) must match `[a-z0-9_.-]{1,32}`. `REAP_BUCKETS` (default 16) caps distinct domains. Each domain owns two modality slots. `default` is registered at start. A new domain past the cap is HTTP 400 naming the cap. Known domains and untagged requests always work. With REAP off, a bad `reap_domain` is ignored (no 400). Rows in one batched round carry a `[R] u8` bucket map.

`GET /stats/reap` uses the cache-transfer key (`cache_routes.zig`). Absent when REAP is off or the key is unset (404). JSON key order is the script's, then `domain`, `modality`, `tokens`, `norm_rows`, `sampled`, `build`. `build` is `TF_DS_DEPLOY` when set, else `build.zig.zon`'s version. Floats are the shortest Python round-trip (`json_write.floatJson`). No domain lists buckets and their token totals. `modality=all` (the default) adds the two modality slots and takes the max of `max_l2_norm`.

`REAP_DIR` empty means no files. Otherwise, every `REAP_SNAPSHOT_MIN` (default 10) and at clean shutdown: `<dir>/<domain>.<modality>.json`, tmp in the same dir, fsync, rename, fsync the dir. On start, a file whose `model_id`, `num_layers` and `num_experts` match is loaded into the accumulators and its domain counts against the cap. Anything else is renamed `.bad` and logged, never overwritten. The file is a valid `--domain-report`.

The HTTP thread never calls the device. It sets a flag and waits up to 5 s for the engine thread to publish the host mirror (the next round boundary, or the lane host's idle tick, the same place cache transfers run). The reply is serialized from that mirror. A wait that expires is HTTP 503 with `Retry-After: 1`. Snapshot files are not written on the engine thread: that thread copies device to host into a side buffer and wakes one writer thread, which serializes and renames as before. The clean-shutdown snapshot stays synchronous in `close`.

## What the code adds

Defaults, off unless `REAP_STATS=1`: `REAP_EVERY=8`, `REAP_PREFILL_ROWS=4`, `REAP_BUCKETS=16`, `REAP_DIR` empty, `REAP_SNAPSHOT_MIN=10`. `serve_lib.sh` exports `TF_DS_REAP_STATS`, `TF_DS_REAP_EVERY`, `TF_DS_REAP_PREFILL_ROWS`, `TF_DS_REAP_BUCKETS` to every rank, and only when stats are on. Rank 0 also gets `TF_DS_REAP_DIR` (when set), `TF_DS_REAP_SNAPSHOT_MIN`, and `TF_DS_SERVED_NAME`. The plan string gains ` reap=EVERY/PREFILL/BUCKETS` only when stats are on. Off, the plan text is the stock string.

Decode `y` staging is one bank, `[40, 48, y_slots, dn]` f32 (`round.max_rows`, not the pick stride 64, and not a second bank). **Computed:** TP4 `y_slots` 7, dn 2560: 40·48·7·2560·4 = 137,625,600 B. TP2 served (the shared expert stays in the table, `y_slots` 7, dn 5120): 40·48·7·5120·4 = 275,251,200 B. The compact TP2 path (every MoE layer has a shared expert, `y_slots` 6) is not the served kit. Prompt chunks allocate no `y`. Pack is two regions because `fillSplit` interleaves two chunks. Default 4 rows, TP4: 2·40·4·6·2560·4 = 19,660,800 B (one region is the old 9,830,400 B). Prompt picks stay `[2, 40, 2048, route_slots]`.

Everything REAP allocates on the device is **computed** by `reap.deviceBytes`. The open log `reap: rank R device bytes N` is that sum of the buffers actually allocated. TP4 rank 0 is 200,056,224 B and each other rank 176,463,264 B. TP2 rank 0 is 367,173,024 B and each other rank 343,580,064 B. Rank 0's extra 23,592,960 B is the six accumulator grids at 16 domains, which is also one device-to-host pull (`reap.pullBytes`). There is no device-byte floor at open to charge this against.

The sample exchange and the stats kernels run after the forward, on the engine stream, outside the captured graph. Routing counts for a decode round wait until rank 0 knows which drafts were accepted (`lanes.verify`, after the draws). The sample counter advances once per `model.verify` on every rank, not per drafter forward. `draft.zig` is not on this path.

World-2 TCP (`tcp_coll`, and `tcp_proxy` delegating to it) has a point-to-point `exchange`: equal lengths, peer `1 - rank`, the peer's bytes land at `recv`. World-2 RDMA (`RdmaProxy`, `TF_COLL_KIND=rdma`, how a TP2 pair serves) delegates that exchange to its host link the same way. World-4 RDMA and RCCL already had `exchange`. With stats on and world ≥ 2, open refuses a collective whose vtable has no `exchange`, before any request is served. REAP off never calls it.

Host microbench (`reap host microbench`, this laptop, same harness): ReleaseFast plan (sample index, hop list, bucket map) is 5 ns/round. The call is indirect so the pure plan is not folded out of the timer. Debug is 102 ns/round. The CPU reference of route+norm, R = 8, 40 layers, 384 experts, sample dn = 64, is 0.435 ms ReleaseFast and 16.9 ms debug. That is the host port of the stats kernel, not the device. The device estimate next to it is the wire: 0.084 ms TP4 and 0.152 ms TP2 per round at `REAP_EVERY=8` (**computed** above). The host plan cost is noise beside either. One full-shape bucket (40×384, text of one domain, every cell non-zero) serializes in 8.30 ms ReleaseFast (8,303,706 ns over 8 iterations) and 109 ms debug (108,667,254 ns, one iteration). One device-to-host pull at 16 domains is 23,592,960 B (**computed**, `reap.pullBytes`; the microbench prints it beside the serialize time).

## Hardware check

Eager, `--profile 2`, `TF_DS_CLOCK_MARKS=1`, rank 0, the same draft settings as the gather note. `REAP_STATS=1` against unset.

- Replies byte-identical on vs off (same prompt, same seed, same drafts). A differing logit byte means a launch argument changed a sum.
- `phases_ms` of the forward, plus the REAP kernels' own clock if they are marked: expected added GPU time ≈ 0.084 ms a TP4 round and ≈ 0.152 ms a TP2 round (**computed** above). Under 1% is 0.29 ms of the 29.3 ms round and 0.31 ms of the 30.75 ms TP2 round. A run whose added time exceeds that by more than the noise of one 25 µs exchange has left the sample schedule (every round, or more than row 0).
- `round_launches`: base, plus one route kernel every round on rank 0, plus on every 8th round the pack, the exchange's stage/collect (2 on TP4, 4 on a two-chunk TP2 hop), the ssq on the pair's r = 0, hop 2's stage/collect on world 4, and the norm kernel on rank 0. Still the base count means the key did not take.
- Coll time: one `reap_pair` exchange of 2,457,600 B (TP4) or 4,915,200 B (TP2) and, world 4, one `reap_scalar` of 1,920 B, every 8th round. Any other byte count is the wrong pack.
- Prompt: one exchange a chunk of `min(4, rows)` times that payload, not one a layer.
