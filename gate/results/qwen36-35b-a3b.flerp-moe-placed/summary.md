# Gate of the placed loader and the block range: qwen36-35b-a3b.flerp-moe

Host control-node, written 2026-11-11T22:50:54Z. **PASSED** under `[criteria]` of `gate/qwen36-35b-a3b.flerp-moe-placed.toml` (committed before the first GPU run). Stage P1 of `docs/PHASE5-FOURNODE-DESIGN.md`.

**Gate (1) was not judged as the brief set it.** Its criterion is bit equality (sha256 of the copied logits) with today's binary. Today's binary has only the origin's loader, which reads the whole 20 GiB file through its mapping with eight workers, and the host's I/O rule of 2026-11-11 forbids that here. Stage `equal` compares with the recorded run of the binary before this stage instead: ids, and per row the log-sum-exp over the whole row and the 40 largest logits as printed.

| stage | result | when (UTC) | what |
|---|---|---|---|
| plan | PASS | 2026-11-11T22:31:59Z | 13 of 13 plans equal between the binary and gguf_census.py |
| equal | PASS | 2026-11-11T22:40:09Z | 0 failure(s); runs {'placed-plain': {'rc': 0, 'seconds': 162.1}, 'placed-traced': {'rc': 0, 'seconds': 221.8}, 'dense-placed': {'rc': 0, 'seconds': 39.8}} |
| range35 | PASS | 2026-11-11T22:41:40Z | blocks [0, 10]: pass; blocks [10, 20]: pass |
| ornith | PASS | 2026-11-11T22:50:54Z | blocks [0, 10]: 0 failure(s) |

## Stage `equal`: the 35B model, world 1, whole, through the placed loader

Every criterion judged held.

| prompt | plain ids equal the recorded run's | traced: ids, rows, log-sum-exp and top 40 of every row equal | rows | dense model through the placed loader equals its record |
|---|---|---|---|---|
| bridge | True (128 tokens) | True | 128 | True |
| code | True (128 tokens) | True | 128 | True |
| rivers | True (128 tokens) | True | 128 | True |
| records | True (128 tokens) | True | 128 | True |

`placed-plain`: load 145.06 s (fill 144.0 s with 73.4 s of pauses, 0.147 GB/s, one direct reader: True); 733 tensors, 21155768832 bytes into one allocation of 21155772928; 733 regions; RssFile 176230400 -> 193761280; not loaded: 20 tensors, 546703360 bytes.

`placed-traced`: load 203.927 s (fill 202.8 s with 73.4 s of pauses, 0.104 GB/s, one direct reader: True); 733 tensors, 21155768832 bytes into one allocation of 21155772928; 733 regions; RssFile 190246912 -> 206770176; not loaded: 20 tensors, 546703360 bytes.

`dense-placed`: load 35.7776 s (fill 35.6 s with 32.0 s of pauses, 0.036 GB/s, one direct reader: True); 320 tensors, 1263435008 bytes into one allocation of 1263439104; 320 regions; RssFile 190251008 -> 210378752; not loaded: 0 tensors, 0 bytes.

The recorded run (the origin's loader, eight buffered workers): {'load_s': 28.7627, 'device_bytes': 1872518100, 'weight_resident_bytes': 21713462624, 'weight_regions': 1}. Decode: bridge 58.5 tok/s (recorded 53.3), code 58.2 tok/s (recorded 57.5), rivers 57.6 tok/s (recorded 57.0), records 57.1 tok/s (recorded 56.2).

## Stage `range35`: block ranges of the 35B model as stages

### Blocks 0..9

Loaded blocks 0..9 of 40: 185 tensors, 5.10 GiB read into one allocation of 5472843776 bytes (the plan's: 5472843776); fill 29.9 s of which 18.5 s were pauses, 0.183 GB/s (472 direct reads by one thread: True); RssFile across the fill 190263296 -> 210391040 bytes; device free 121.0 -> 115.7 GiB of 124.0. Whole load 30.284 s. Arena and state as the executor reports them: 0.94 GiB at context 2048 (batch 2048); the origin's estimate at context 32768: 2.35 GiB. Scratch floats a token (`intermediate`): 16384.

Phase `tokens`, 10 blocks, hidden 2048:
- chunk of 1024 rows: [1.103, 1.072, 1.06] s (the first run is dumped), **106.0 ms a block**, 966 rows/s through these blocks; three runs the same bytes: True; output RMS 0.04969, non-finite 0;
- decode from an empty context, 48 steps asked for logits (the origin's graph path): first 9.7 ms, then 3.73 ms a step; eager: 3.77 ms a step; **ms a block: 0.372537674468 (graph path), 0.377 (eager)**;
- decode at context 1027: 3.87 ms a step asked for logits, 3.84 eager.

Phase `synthetic`, 10 blocks, hidden 2048:
- chunk of 1024 rows: [1.057, 1.086, 1.093] s (the first run is dumped), **108.6 ms a block**, 943 rows/s through these blocks; three runs the same bytes: True; output RMS 0.4612, non-finite 0;
- decode from an empty context, 0 steps asked for logits (the origin's graph path): first 0.0 ms, then 0.00 ms a step; eager: 3.68 ms a step; **ms a block: None (graph path), 0.368 (eager)**;
- decode at context 1027: 0.00 ms a step asked for logits, 3.69 eager.

Block check (`gate/moe_block_check.py`, numpy float64, 15.1 s, load 6.45 -> 5.91): 30 records of blocks [0, 3, 9]; the same experts in 30 (in the same order in 30; 0 excused as ties, 0 not); weights within 1.902652532748128e-07, the shared gate logit within 2.057478605266283e-07, the block output within 1.6035424245920974e-05 of its RMS; smallest float64 margin to the next expert 2.564145774764867e-05.

### Blocks 10..19

Loaded blocks 10..19 of 40: 181 tensors, 4.69 GiB read into one allocation of 5039488256 bytes (the plan's: 5039488256); fill 24.8 s of which 18.1 s were pauses, 0.204 GB/s (441 direct reads by one thread: True); RssFile across the fill 190173184 -> 210321408 bytes; device free 121.0 -> 116.1 GiB of 124.0. Whole load 25.148 s. Arena and state as the executor reports them: 0.94 GiB at context 2048 (batch 2048); the origin's estimate at context 32768: 2.35 GiB. Scratch floats a token (`intermediate`): 16384.

Phase `tokens` not run: the range does not start at block 0.

Phase `synthetic`, 10 blocks, hidden 2048:
- chunk of 1024 rows: [1.084, 1.053, 1.059] s (the first run is dumped), **105.3 ms a block**, 972 rows/s through these blocks; three runs the same bytes: True; output RMS 0.5429, non-finite 0;
- decode from an empty context, 0 steps asked for logits (the origin's graph path): first 0.0 ms, then 0.00 ms a step; eager: 3.58 ms a step; **ms a block: None (graph path), 0.358 (eager)**;
- decode at context 1027: 0.00 ms a step asked for logits, 3.66 eager.

Block check (`gate/moe_block_check.py`, numpy float64, 10.0 s, load 5.91 -> 5.7): 15 records of blocks [10, 15, 19]; the same experts in 15 (in the same order in 15; 0 excused as ties, 0 not); weights within 8.994907813031361e-08, the shared gate logit within 1.249557872728957e-07, the block output within 7.068685132036176e-06 of its RMS; smallest float64 margin to the next expert 0.00011844989726437875.

## Stage `ornith`: leading blocks of Ornith-1.5 397B as a stage

Memory rule: {'gtt_total': 133143986176, 'gtt_used': 18661376, 'mem_available': 126792548352, 'free_bytes': 126792548352, 'margin_gib': 24, 'arena_reserve_gib': 6, 'plan_gib': {1: 4.51, 2: 8.49, 3: 12.47, 4: 16.43, 5: 20.41, 6: 24.4, 7: 28.38, 8: 31.82, 9: 35.27, 10: 39.25}, 'blocks': 10}.

### Blocks 0..9

Loaded blocks 0..9 of 60: 185 tensors, 39.25 GiB read into one allocation of 42145677312 bytes (the plan's: 42145677312); fill 457.3 s of which 18.5 s were pauses, 0.092 GB/s (2650 direct reads by one thread: True); RssFile across the fill 189628416 -> 207601664 bytes; device free 121.0 -> 81.6 GiB of 124.0. Whole load 459.227 s. Arena and state as the executor reports them: 2.30 GiB at context 2048 (batch 2048); the origin's estimate at context 32768: 4.97 GiB. Scratch floats a token (`intermediate`): 40960.

Phase `tokens`, 10 blocks, hidden 4096:
- chunk of 1024 rows: [4.945, 5.123, 5.141] s (the first run is dumped), **512.3 ms a block**, 200 rows/s through these blocks; three runs the same bytes: True; output RMS 0.04551, non-finite 0;
- decode from an empty context, 48 steps asked for logits (the origin's graph path): first 19.5 ms, then 11.63 ms a step; eager: 13.77 ms a step; **ms a block: 1.16260949787 (graph path), 1.377 (eager)**;
- decode at context 1027: 11.34 ms a step asked for logits, 11.31 eager.

Phase `synthetic`, 10 blocks, hidden 4096:
- chunk of 1024 rows: [5.466, 5.088, 5.084] s (the first run is dumped), **508.4 ms a block**, 201 rows/s through these blocks; three runs the same bytes: True; output RMS 2.787, non-finite 0;
- decode from an empty context, 0 steps asked for logits (the origin's graph path): first 0.0 ms, then 0.00 ms a step; eager: 11.47 ms a step; **ms a block: None (graph path), 1.147 (eager)**;
- decode at context 1027: 0.00 ms a step asked for logits, 11.61 eager.

Block check (`gate/moe_block_check.py`, numpy float64, 60.1 s, load 5.02 -> 5.31): 30 records of blocks [0, 7, 9]; the same experts in 30 (in the same order in 30; 0 excused as ties, 0 not); weights within 2.0790300542250506e-07, the shared gate logit within 7.731045439385298e-07, the block output within 4.712881937343662e-05 of its RMS; smallest float64 margin to the next expert 6.434528825111779e-06.

## Disk reads on this host

96.2 GB by the loader (one direct reader, `--single-reader --pause-ms 100`), 894 s of reading with its pauses, between 2026-11-11T22:35:46Z and 2026-11-11T22:49:54Z:

| run | GB | seconds | ended (UTC) |
|---|---|---|---|
| placed-plain | 21.16 | 144.003524426 | 2026-11-11T22:35:46Z |
| placed-traced | 21.16 | 202.825724832 | 2026-11-11T22:39:28Z |
| dense-placed | 1.26 | 35.554564608 | 2026-11-11T22:40:08Z |
| range35-0-10 | 5.47 | 29.925110431 | 2026-11-11T22:40:46Z |
| range35-10-20 | 5.04 | 24.75760143 | 2026-11-11T22:41:15Z |
| ornith | 42.15 | 457.317707787 | 2026-11-11T22:49:54Z |
