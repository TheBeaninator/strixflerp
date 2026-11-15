# strixw panel kernels: kernel gate results

Criteria: `gate/strixw-panel-kernels.toml` (committed before the run). Raw: `raw.jsonl` (one case a line), bench log `bench.log`. 489 cases, 6 skipped (output too large), 2.232 GB read from the fleet file.

## Verdicts

| criterion | pass | detail |
|---|---|---|
| equal | PASS | 489 of 489 cases byte-equal |
| bad_id | PASS | 21 of 21 out-of-range-id cases byte-equal |
| coverage | PASS | missing: none |
| read | FAIL | 2.232 GB read |
| speed_decode_dense | FAIL | median panel % of origin time per class: attn_o 155.2, attn_qkv 116.5, gdn_in 112.7, gdn_out 152.9, lm_head 108.6, mtp_fc 146.8, ple_kv 118.1, sh_down 114.0, sh_gu 189.7 |
| speed_prefill_dense | PASS | sh_gu M2048 bf16 92.7%; sh_gu M8192 bf16 96.8%; sh_gu M2048 f32 92.8%; sh_gu M8192 f32 96.7%; sh_down M2048 bf16 93.5%; sh_down M8192 bf16 96.2%; sh_down M2048 f32 96.0%; sh_down M8192 f32 95.3%; gdn_in M2048 bf16 98.2%; gdn_in M8192 bf16 99.3%; gdn_in M2048 f32 99.2%; gdn_in M8192 f32 99.5%; gdn_out M2048 bf16 70.9%; gdn_out M8192 bf16 90.2%; gdn_out M2048 f32 67.3%; gdn_out M8192 f32 89.5%; attn_qkv M2048 bf16 93.7%; attn_qkv M8192 bf16 97.0%; attn_qkv M2048 f32 93.6%; attn_qkv M8192 f32 96.5%; attn_o M2048 bf16 72.6%; attn_o M8192 bf16 89.2%; attn_o M2048 f32 67.1%; attn_o M8192 f32 89.8%; ple_kv M2048 bf16 93.8%; ple_kv M8192 bf16 96.7%; ple_kv M2048 f32 93.2%; ple_kv M8192 f32 98.1%; lm_head M2048 bf16 96.9%; mtp_fc M2048 bf16 93.2%; mtp_fc M8192 bf16 97.8%; mtp_fc M2048 f32 93.4%; mtp_fc M8192 f32 97.0% |
| speed_prefill_expert | PASS | expert-prefill-swiglu-f16h narrow M2048 93.6% (faster); expert-prefill-combine-f16h-shared narrow M2048 93.6% (faster); expert-prefill-combine-f16h-shared wide M8192 95.2% (faster); expert-prefill-swiglu-f16h wide256 M8192 98.4% (noise) |

## Equality (cases byte-equal / cases)

| class | bits | codes | path | equal / cases | differing bytes |
|---|---|---|---|---|---|
| attn_o | Q4 | synthetic | dense-decode | 16 / 16 | 0 |
| attn_o | Q4 | synthetic | dense-prefill-wmma | 6 / 6 | 0 |
| attn_o | Q8 | real | dense-decode | 16 / 16 | 0 |
| attn_o | Q8 | real | dense-prefill-wmma | 6 / 6 | 0 |
| attn_qkv | Q4 | synthetic | dense-decode | 16 / 16 | 0 |
| attn_qkv | Q4 | synthetic | dense-prefill-wmma | 6 / 6 | 0 |
| attn_qkv | Q8 | real | dense-decode | 16 / 16 | 0 |
| attn_qkv | Q8 | real | dense-prefill-wmma | 6 / 6 | 0 |
| exp_down | Q4 | real | expert-decode-combine | 9 / 9 | 0 |
| exp_down | Q4 | real | expert-prefill-combine narrow | 4 / 4 | 0 |
| exp_down | Q4 | real | expert-prefill-combine wide | 4 / 4 | 0 |
| exp_down | Q4 | real | expert-prefill-combine wide256 | 4 / 4 | 0 |
| exp_down | Q4 | real | expert-prefill-combine-f16h narrow | 4 / 4 | 0 |
| exp_down | Q4 | real | expert-prefill-combine-f16h wide | 4 / 4 | 0 |
| exp_down | Q4 | real | expert-prefill-combine-f16h wide256 | 4 / 4 | 0 |
| exp_down | Q4 | real | expert-prefill-combine-f16h-shared narrow | 4 / 4 | 0 |
| exp_down | Q4 | real | expert-prefill-combine-f16h-shared wide | 4 / 4 | 0 |
| exp_down | Q4 | real | expert-prefill-combine-f16h-shared wide256 | 4 / 4 | 0 |
| exp_gu | Q4 | real | expert-decode-gather | 9 / 9 | 0 |
| exp_gu | Q4 | real | expert-decode-gather-swiglu | 9 / 9 | 0 |
| exp_gu | Q4 | real | expert-prefill-gather narrow | 4 / 4 | 0 |
| exp_gu | Q4 | real | expert-prefill-gather wide | 4 / 4 | 0 |
| exp_gu | Q4 | real | expert-prefill-gather wide256 | 4 / 4 | 0 |
| exp_gu | Q4 | real | expert-prefill-swiglu narrow | 4 / 4 | 0 |
| exp_gu | Q4 | real | expert-prefill-swiglu wide | 4 / 4 | 0 |
| exp_gu | Q4 | real | expert-prefill-swiglu wide256 | 4 / 4 | 0 |
| exp_gu | Q4 | real | expert-prefill-swiglu-f16h narrow | 4 / 4 | 0 |
| exp_gu | Q4 | real | expert-prefill-swiglu-f16h wide | 4 / 4 | 0 |
| exp_gu | Q4 | real | expert-prefill-swiglu-f16h wide256 | 4 / 4 | 0 |
| gdn_in | Q4 | real | dense-decode | 16 / 16 | 0 |
| gdn_in | Q4 | real | dense-prefill-wmma | 6 / 6 | 0 |
| gdn_in | Q8 | synthetic | dense-decode | 16 / 16 | 0 |
| gdn_in | Q8 | synthetic | dense-prefill-wmma | 6 / 6 | 0 |
| gdn_out | Q4 | synthetic | dense-decode | 16 / 16 | 0 |
| gdn_out | Q4 | synthetic | dense-prefill-wmma | 6 / 6 | 0 |
| gdn_out | Q8 | real | dense-decode | 16 / 16 | 0 |
| gdn_out | Q8 | real | dense-prefill-wmma | 6 / 6 | 0 |
| lm_head | Q4 | synthetic | dense-decode | 16 / 16 | 0 |
| lm_head | Q4 | synthetic | dense-prefill-wmma | 3 / 3 | 0 |
| lm_head | Q8 | real | dense-decode | 16 / 16 | 0 |
| lm_head | Q8 | real | dense-prefill-wmma | 3 / 3 | 0 |
| mtp_fc | Q4 | synthetic | dense-decode | 16 / 16 | 0 |
| mtp_fc | Q4 | synthetic | dense-prefill-wmma | 6 / 6 | 0 |
| mtp_fc | Q8 | real | dense-decode | 16 / 16 | 0 |
| mtp_fc | Q8 | real | dense-prefill-wmma | 6 / 6 | 0 |
| ple_kv | Q4 | synthetic | dense-decode | 16 / 16 | 0 |
| ple_kv | Q4 | synthetic | dense-prefill-wmma | 6 / 6 | 0 |
| ple_kv | Q8 | real | dense-decode | 16 / 16 | 0 |
| ple_kv | Q8 | real | dense-prefill-wmma | 6 / 6 | 0 |
| sh_down | Q4 | synthetic | dense-decode | 16 / 16 | 0 |
| sh_down | Q4 | synthetic | dense-prefill-wmma | 6 / 6 | 0 |
| sh_down | Q8 | real | dense-decode | 16 / 16 | 0 |
| sh_down | Q8 | real | dense-prefill-wmma | 6 / 6 | 0 |
| sh_gu | Q4 | synthetic | dense-decode | 16 / 16 | 0 |
| sh_gu | Q4 | synthetic | dense-prefill-wmma | 6 / 6 | 0 |
| sh_gu | Q8 | real | dense-decode | 16 / 16 | 0 |
| sh_gu | Q8 | real | dense-prefill-wmma | 6 / 6 | 0 |

## Speed (median ms over the reps; q1-q3 in brackets; GB/s on the weight bytes the case touches)

| class | bits | path | M | out | origin ms | panel ms | panel % | GB/s origin / panel | |
|---|---|---|---|---|---|---|---|---|---|
| sh_gu | Q8 | dense-decode | 1 | bf16 | 0.0415 [0.0412-0.0420] | 0.0440 [0.0439-0.0448] | 106.1 | 84 / 79 | slower |
| sh_gu | Q8 | dense-decode | 2 | bf16 | 0.0440 [0.0436-0.0451] | 0.0598 [0.0590-0.0605] | 135.8 | 79 / 58 | slower |
| sh_gu | Q8 | dense-decode | 3 | bf16 | 0.0468 [0.0460-0.0473] | 0.0767 [0.0764-0.0772] | 163.8 | 74 / 45 | slower |
| sh_gu | Q8 | dense-decode | 4 | bf16 | 0.0478 [0.0467-0.0485] | 0.0880 [0.0876-0.0885] | 184.1 | 73 / 40 | slower |
| sh_gu | Q8 | dense-decode | 5 | bf16 | 0.0522 [0.0515-0.0531] | 0.0989 [0.0985-0.0998] | 189.7 | 67 / 35 | slower |
| sh_gu | Q8 | dense-decode | 6 | bf16 | 0.0540 [0.0536-0.0548] | 0.1098 [0.1085-0.1105] | 203.1 | 64 / 32 | slower |
| sh_gu | Q8 | dense-decode | 7 | bf16 | 0.0552 [0.0545-0.0557] | 0.1197 [0.1186-0.1204] | 216.8 | 63 / 29 | slower |
| sh_gu | Q8 | dense-decode | 8 | bf16 | 0.0545 [0.0541-0.0552] | 0.1321 [0.1313-0.1324] | 242.3 | 64 / 26 | slower |
| sh_gu | Q8 | dense-prefill-wmma | 512 | bf16 | 0.1412 [0.1393-0.1446] | 0.1295 [0.1209-0.1332] | 91.7 | 25 / 27 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.4462 [0.4411-0.4650] | 0.4134 [0.3960-0.4331] | 92.7 | 8 / 8 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | bf16 | 1.6089 [1.5879-1.6410] | 1.5581 [1.5400-1.5923] | 96.8 | 2 / 2 | noise |
| sh_gu | Q8 | dense-decode | 1 | f32 | 0.0413 [0.0411-0.0416] | 0.0440 [0.0438-0.0444] | 106.5 | 84 / 79 | slower |
| sh_gu | Q8 | dense-decode | 2 | f32 | 0.0428 [0.0420-0.0431] | 0.0594 [0.0588-0.0602] | 138.6 | 81 / 59 | slower |
| sh_gu | Q8 | dense-decode | 3 | f32 | 0.0444 [0.0441-0.0453] | 0.0764 [0.0763-0.0766] | 172.0 | 78 / 46 | slower |
| sh_gu | Q8 | dense-decode | 4 | f32 | 0.0466 [0.0462-0.0473] | 0.0878 [0.0872-0.0882] | 188.5 | 75 / 40 | slower |
| sh_gu | Q8 | dense-decode | 5 | f32 | 0.0490 [0.0482-0.0492] | 0.0984 [0.0978-0.0990] | 200.7 | 71 / 35 | slower |
| sh_gu | Q8 | dense-decode | 6 | f32 | 0.0505 [0.0496-0.0508] | 0.1096 [0.1088-0.1106] | 216.9 | 69 / 32 | slower |
| sh_gu | Q8 | dense-decode | 7 | f32 | 0.0519 [0.0515-0.0522] | 0.1197 [0.1195-0.1202] | 230.7 | 67 / 29 | slower |
| sh_gu | Q8 | dense-decode | 8 | f32 | 0.0535 [0.0532-0.0542] | 0.1316 [0.1313-0.1325] | 245.9 | 65 / 26 | slower |
| sh_gu | Q8 | dense-prefill-wmma | 512 | f32 | 0.1482 [0.1446-0.1499] | 0.1334 [0.1314-0.1369] | 90.0 | 23 / 26 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | f32 | 0.4598 [0.4567-0.4674] | 0.4265 [0.4190-0.4301] | 92.8 | 8 / 8 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | f32 | 1.6774 [1.6649-1.6891] | 1.6213 [1.6027-1.6405] | 96.7 | 2 / 2 | faster |
| sh_gu | Q4 syn | dense-decode | 1 | bf16 | 0.0260 [0.0258-0.0263] | 0.0261 [0.0259-0.0262] | 100.2 | 71 / 71 | noise |
| sh_gu | Q4 syn | dense-decode | 2 | bf16 | 0.0285 [0.0277-0.0305] | 0.0357 [0.0353-0.0362] | 125.2 | 65 / 52 | slower |
| sh_gu | Q4 syn | dense-decode | 3 | bf16 | 0.0301 [0.0292-0.0307] | 0.0437 [0.0427-0.0447] | 145.3 | 61 / 42 | slower |
| sh_gu | Q4 syn | dense-decode | 4 | bf16 | 0.0314 [0.0304-0.0324] | 0.0504 [0.0485-0.0510] | 160.4 | 59 / 37 | slower |
| sh_gu | Q4 syn | dense-decode | 5 | bf16 | 0.0331 [0.0324-0.0340] | 0.0766 [0.0764-0.0774] | 231.6 | 56 / 24 | slower |
| sh_gu | Q4 syn | dense-decode | 6 | bf16 | 0.0345 [0.0338-0.0351] | 0.0864 [0.0855-0.0872] | 250.4 | 53 / 21 | slower |
| sh_gu | Q4 syn | dense-decode | 7 | bf16 | 0.0352 [0.0348-0.0362] | 0.0949 [0.0938-0.0964] | 269.5 | 52 / 19 | slower |
| sh_gu | Q4 syn | dense-decode | 8 | bf16 | 0.0362 [0.0355-0.0372] | 0.1044 [0.1036-0.1047] | 288.0 | 51 / 18 | slower |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.1280 [0.1265-0.1292] | 0.1219 [0.1212-0.1224] | 95.2 | 14 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.4359 [0.4256-0.4426] | 0.4202 [0.4080-0.4241] | 96.4 | 4 / 4 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 1.5278 [1.5157-1.5533] | 1.5039 [1.4922-1.5184] | 98.4 | 1 / 1 | noise |
| sh_gu | Q4 syn | dense-decode | 1 | f32 | 0.0260 [0.0257-0.0264] | 0.0260 [0.0258-0.0262] | 100.3 | 71 / 71 | noise |
| sh_gu | Q4 syn | dense-decode | 2 | f32 | 0.0284 [0.0274-0.0297] | 0.0354 [0.0351-0.0360] | 124.8 | 65 / 52 | slower |
| sh_gu | Q4 syn | dense-decode | 3 | f32 | 0.0294 [0.0288-0.0304] | 0.0438 [0.0431-0.0441] | 148.9 | 63 / 42 | slower |
| sh_gu | Q4 syn | dense-decode | 4 | f32 | 0.0297 [0.0294-0.0304] | 0.0504 [0.0492-0.0511] | 169.7 | 62 / 37 | slower |
| sh_gu | Q4 syn | dense-decode | 5 | f32 | 0.0321 [0.0311-0.0325] | 0.0765 [0.0763-0.0774] | 238.2 | 57 / 24 | slower |
| sh_gu | Q4 syn | dense-decode | 6 | f32 | 0.0336 [0.0330-0.0344] | 0.0890 [0.0874-0.0897] | 264.6 | 55 / 21 | slower |
| sh_gu | Q4 syn | dense-decode | 7 | f32 | 0.0351 [0.0347-0.0359] | 0.0967 [0.0952-0.0972] | 275.3 | 52 / 19 | slower |
| sh_gu | Q4 syn | dense-decode | 8 | f32 | 0.0355 [0.0348-0.0370] | 0.1066 [0.1049-0.1069] | 300.0 | 52 / 17 | slower |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.1267 [0.1261-0.1273] | 0.1238 [0.1225-0.1246] | 97.7 | 15 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.4191 [0.4177-0.4234] | 0.4152 [0.4127-0.4165] | 99.1 | 4 / 4 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | f32 | 1.5121 [1.4843-1.5266] | 1.4907 [1.4643-1.4996] | 98.6 | 1 / 1 | noise |
| sh_down | Q8 | dense-decode | 1 | bf16 | 0.0254 [0.0252-0.0255] | 0.0257 [0.0256-0.0259] | 101.3 | 73 / 72 | slower |
| sh_down | Q8 | dense-decode | 2 | bf16 | 0.0274 [0.0270-0.0277] | 0.0292 [0.0288-0.0294] | 106.4 | 67 / 63 | slower |
| sh_down | Q8 | dense-decode | 3 | bf16 | 0.0290 [0.0287-0.0296] | 0.0317 [0.0315-0.0322] | 109.5 | 64 / 58 | slower |
| sh_down | Q8 | dense-decode | 4 | bf16 | 0.0308 [0.0303-0.0313] | 0.0346 [0.0345-0.0349] | 112.3 | 60 / 53 | slower |
| sh_down | Q8 | dense-decode | 5 | bf16 | 0.0328 [0.0324-0.0341] | 0.0380 [0.0379-0.0385] | 116.0 | 56 / 48 | slower |
| sh_down | Q8 | dense-decode | 6 | bf16 | 0.0343 [0.0337-0.0347] | 0.0408 [0.0405-0.0412] | 118.9 | 54 / 45 | slower |
| sh_down | Q8 | dense-decode | 7 | bf16 | 0.0342 [0.0337-0.0344] | 0.0436 [0.0433-0.0443] | 127.6 | 54 / 42 | slower |
| sh_down | Q8 | dense-decode | 8 | bf16 | 0.0350 [0.0348-0.0355] | 0.0458 [0.0456-0.0461] | 130.8 | 53 / 40 | slower |
| sh_down | Q8 | dense-prefill-wmma | 512 | bf16 | 0.0773 [0.0768-0.0779] | 0.0703 [0.0700-0.0707] | 90.9 | 24 / 26 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.2490 [0.2446-0.2534] | 0.2327 [0.2278-0.2364] | 93.5 | 7 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | bf16 | 0.8764 [0.8644-0.8954] | 0.8428 [0.8285-0.8577] | 96.2 | 2 / 2 | faster |
| sh_down | Q8 | dense-decode | 1 | f32 | 0.0255 [0.0253-0.0257] | 0.0256 [0.0254-0.0258] | 100.5 | 72 / 72 | noise |
| sh_down | Q8 | dense-decode | 2 | f32 | 0.0271 [0.0267-0.0274] | 0.0291 [0.0290-0.0293] | 107.5 | 68 / 63 | slower |
| sh_down | Q8 | dense-decode | 3 | f32 | 0.0286 [0.0283-0.0289] | 0.0320 [0.0317-0.0322] | 111.9 | 65 / 58 | slower |
| sh_down | Q8 | dense-decode | 4 | f32 | 0.0298 [0.0295-0.0302] | 0.0343 [0.0341-0.0346] | 115.2 | 62 / 54 | slower |
| sh_down | Q8 | dense-decode | 5 | f32 | 0.0327 [0.0324-0.0337] | 0.0382 [0.0377-0.0388] | 117.0 | 56 / 48 | slower |
| sh_down | Q8 | dense-decode | 6 | f32 | 0.0363 [0.0359-0.0365] | 0.0418 [0.0404-0.0424] | 115.2 | 51 / 44 | slower |
| sh_down | Q8 | dense-decode | 7 | f32 | 0.0369 [0.0349-0.0378] | 0.0437 [0.0430-0.0450] | 118.4 | 50 / 42 | slower |
| sh_down | Q8 | dense-decode | 8 | f32 | 0.0344 [0.0343-0.0346] | 0.0462 [0.0458-0.0472] | 134.1 | 54 / 40 | slower |
| sh_down | Q8 | dense-prefill-wmma | 512 | f32 | 0.0854 [0.0846-0.0864] | 0.0783 [0.0776-0.0789] | 91.6 | 22 / 24 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | f32 | 0.2533 [0.2478-0.2584] | 0.2430 [0.2392-0.2458] | 96.0 | 7 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | f32 | 0.8803 [0.8619-0.8971] | 0.8387 [0.8288-0.8667] | 95.3 | 2 / 2 | noise |
| sh_down | Q4 syn | dense-decode | 1 | bf16 | 0.0167 [0.0166-0.0172] | 0.0166 [0.0165-0.0169] | 99.3 | 61 / 62 | noise |
| sh_down | Q4 syn | dense-decode | 2 | bf16 | 0.0198 [0.0196-0.0200] | 0.0186 [0.0184-0.0188] | 93.5 | 52 / 55 | faster |
| sh_down | Q4 syn | dense-decode | 3 | bf16 | 0.0221 [0.0219-0.0223] | 0.0206 [0.0203-0.0211] | 93.5 | 46 / 50 | faster |
| sh_down | Q4 syn | dense-decode | 4 | bf16 | 0.0238 [0.0233-0.0240] | 0.0220 [0.0216-0.0224] | 92.8 | 43 / 46 | faster |
| sh_down | Q4 syn | dense-decode | 5 | bf16 | 0.0271 [0.0260-0.0274] | 0.0287 [0.0272-0.0292] | 105.9 | 38 / 36 | noise |
| sh_down | Q4 syn | dense-decode | 6 | bf16 | 0.0286 [0.0284-0.0291] | 0.0315 [0.0297-0.0319] | 110.1 | 36 / 33 | slower |
| sh_down | Q4 syn | dense-decode | 7 | bf16 | 0.0294 [0.0287-0.0302] | 0.0348 [0.0315-0.0355] | 118.1 | 35 / 29 | slower |
| sh_down | Q4 syn | dense-decode | 8 | bf16 | 0.0312 [0.0311-0.0316] | 0.0361 [0.0335-0.0370] | 115.6 | 33 / 28 | slower |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.0764 [0.0762-0.0765] | 0.0681 [0.0679-0.0685] | 89.2 | 13 / 15 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.2432 [0.2315-0.2471] | 0.2216 [0.2186-0.2301] | 91.1 | 4 / 5 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 0.8633 [0.8572-0.8733] | 0.8359 [0.8266-0.8519] | 96.8 | 1 / 1 | faster |
| sh_down | Q4 syn | dense-decode | 1 | f32 | 0.0166 [0.0165-0.0167] | 0.0167 [0.0166-0.0169] | 100.5 | 62 / 61 | noise |
| sh_down | Q4 syn | dense-decode | 2 | f32 | 0.0183 [0.0181-0.0184] | 0.0186 [0.0185-0.0188] | 101.5 | 56 / 55 | slower |
| sh_down | Q4 syn | dense-decode | 3 | f32 | 0.0197 [0.0195-0.0200] | 0.0212 [0.0204-0.0220] | 107.3 | 52 / 48 | slower |
| sh_down | Q4 syn | dense-decode | 4 | f32 | 0.0206 [0.0204-0.0208] | 0.0235 [0.0212-0.0241] | 114.0 | 50 / 44 | slower |
| sh_down | Q4 syn | dense-decode | 5 | f32 | 0.0226 [0.0224-0.0231] | 0.0293 [0.0287-0.0300] | 129.3 | 45 / 35 | slower |
| sh_down | Q4 syn | dense-decode | 6 | f32 | 0.0234 [0.0233-0.0236] | 0.0325 [0.0312-0.0332] | 139.2 | 44 / 31 | slower |
| sh_down | Q4 syn | dense-decode | 7 | f32 | 0.0264 [0.0244-0.0271] | 0.0348 [0.0326-0.0355] | 131.9 | 39 / 29 | slower |
| sh_down | Q4 syn | dense-decode | 8 | f32 | 0.0282 [0.0272-0.0286] | 0.0374 [0.0352-0.0380] | 132.3 | 36 / 27 | slower |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.0817 [0.0812-0.0830] | 0.0761 [0.0730-0.0763] | 93.2 | 13 / 13 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.2604 [0.2568-0.2662] | 0.2432 [0.2388-0.2475] | 93.4 | 4 / 4 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | f32 | 0.8730 [0.8531-0.8830] | 0.8266 [0.8226-0.8457] | 94.7 | 1 / 1 | faster |
| gdn_in | Q4 | dense-decode | 1 | bf16 | 0.2355 [0.2343-0.2364] | 0.2556 [0.2542-0.2562] | 108.6 | 95 / 88 | slower |
| gdn_in | Q4 | dense-decode | 2 | bf16 | 0.2374 [0.2351-0.2391] | 0.2614 [0.2600-0.2631] | 110.1 | 94 / 86 | slower |
| gdn_in | Q4 | dense-decode | 3 | bf16 | 0.2408 [0.2395-0.2428] | 0.2658 [0.2634-0.2688] | 110.4 | 93 / 84 | slower |
| gdn_in | Q4 | dense-decode | 4 | bf16 | 0.2437 [0.2413-0.2453] | 0.2689 [0.2656-0.2728] | 110.4 | 92 / 83 | slower |
| gdn_in | Q4 | dense-decode | 5 | bf16 | 0.2535 [0.2517-0.2569] | 0.3432 [0.3408-0.3456] | 135.4 | 88 / 65 | slower |
| gdn_in | Q4 | dense-decode | 6 | bf16 | 0.2602 [0.2584-0.2614] | 0.3703 [0.3664-0.3729] | 142.3 | 86 / 61 | slower |
| gdn_in | Q4 | dense-decode | 7 | bf16 | 0.2632 [0.2622-0.2644] | 0.3925 [0.3907-0.3942] | 149.2 | 85 / 57 | slower |
| gdn_in | Q4 | dense-decode | 8 | bf16 | 0.2692 [0.2678-0.2714] | 0.4119 [0.4089-0.4155] | 153.0 | 83 / 54 | slower |
| gdn_in | Q4 | dense-prefill-wmma | 512 | bf16 | 1.2637 [1.2329-1.2808] | 1.2417 [1.1966-1.2488] | 98.3 | 18 / 18 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | bf16 | 5.0830 [4.9942-5.1326] | 4.9904 [4.9408-5.1003] | 98.2 | 4 / 4 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | bf16 | 20.9632 [20.6056-21.7887] | 20.8279 [20.4534-21.5379] | 99.3 | 1 / 1 | noise |
| gdn_in | Q4 | dense-decode | 1 | f32 | 0.2358 [0.2337-0.2371] | 0.2561 [0.2542-0.2565] | 108.6 | 95 / 88 | slower |
| gdn_in | Q4 | dense-decode | 2 | f32 | 0.2382 [0.2368-0.2400] | 0.2624 [0.2604-0.2650] | 110.2 | 94 / 85 | slower |
| gdn_in | Q4 | dense-decode | 3 | f32 | 0.2418 [0.2394-0.2429] | 0.2676 [0.2639-0.2697] | 110.7 | 93 / 84 | slower |
| gdn_in | Q4 | dense-decode | 4 | f32 | 0.2442 [0.2415-0.2463] | 0.2664 [0.2652-0.2685] | 109.1 | 92 / 84 | slower |
| gdn_in | Q4 | dense-decode | 5 | f32 | 0.2540 [0.2517-0.2560] | 0.3456 [0.3432-0.3504] | 136.1 | 88 / 65 | slower |
| gdn_in | Q4 | dense-decode | 6 | f32 | 0.2595 [0.2566-0.2620] | 0.3759 [0.3739-0.3809] | 144.9 | 86 / 60 | slower |
| gdn_in | Q4 | dense-decode | 7 | f32 | 0.2634 [0.2624-0.2646] | 0.3927 [0.3866-0.4004] | 149.1 | 85 / 57 | slower |
| gdn_in | Q4 | dense-decode | 8 | f32 | 0.2700 [0.2679-0.2717] | 0.4169 [0.4121-0.4247] | 154.4 | 83 / 54 | slower |
| gdn_in | Q4 | dense-prefill-wmma | 512 | f32 | 1.1989 [1.1818-1.2423] | 1.1578 [1.1508-1.1943] | 96.6 | 19 / 19 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | f32 | 4.8844 [4.8627-5.0820] | 4.8457 [4.8166-5.0039] | 99.2 | 5 / 5 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | f32 | 21.8458 [20.0023-21.9803] | 21.7411 [19.6932-21.8372] | 99.5 | 1 / 1 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | bf16 | 0.3914 [0.3903-0.3927] | 0.4274 [0.4258-0.4304] | 109.2 | 111 / 102 | slower |
| gdn_in | Q8 syn | dense-decode | 2 | bf16 | 0.3937 [0.3926-0.3948] | 0.4443 [0.4406-0.4508] | 112.8 | 110 / 98 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | bf16 | 0.4036 [0.4014-0.4047] | 0.4486 [0.4450-0.4504] | 111.2 | 108 / 97 | slower |
| gdn_in | Q8 syn | dense-decode | 4 | bf16 | 0.4101 [0.4088-0.4140] | 0.4540 [0.4488-0.4594] | 110.7 | 106 / 96 | slower |
| gdn_in | Q8 syn | dense-decode | 5 | bf16 | 0.4300 [0.4278-0.4314] | 0.4837 [0.4790-0.4871] | 112.5 | 101 / 90 | slower |
| gdn_in | Q8 syn | dense-decode | 6 | bf16 | 0.4394 [0.4371-0.4416] | 0.4974 [0.4915-0.5038] | 113.2 | 99 / 87 | slower |
| gdn_in | Q8 syn | dense-decode | 7 | bf16 | 0.4431 [0.4392-0.4452] | 0.5050 [0.4997-0.5168] | 114.0 | 98 / 86 | slower |
| gdn_in | Q8 syn | dense-decode | 8 | bf16 | 0.4511 [0.4466-0.4568] | 0.5161 [0.5110-0.5274] | 114.4 | 96 / 84 | slower |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | bf16 | 1.5994 [1.5686-1.6261] | 1.3495 [1.3283-1.3561] | 84.4 | 27 / 32 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | bf16 | 5.6912 [5.6596-5.7298] | 5.4570 [5.4407-5.4891] | 95.9 | 8 / 8 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | bf16 | 23.1085 [20.9259-23.2490] | 22.6903 [20.5244-22.8377] | 98.2 | 2 / 2 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | f32 | 0.3932 [0.3924-0.3950] | 0.4280 [0.4260-0.4303] | 108.8 | 111 / 102 | slower |
| gdn_in | Q8 syn | dense-decode | 2 | f32 | 0.3944 [0.3922-0.3954] | 0.4532 [0.4455-0.4570] | 114.9 | 110 / 96 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | f32 | 0.4040 [0.4027-0.4058] | 0.4529 [0.4508-0.4566] | 112.1 | 108 / 96 | slower |
| gdn_in | Q8 syn | dense-decode | 4 | f32 | 0.4104 [0.4083-0.4123] | 0.4562 [0.4517-0.4604] | 111.1 | 106 / 95 | slower |
| gdn_in | Q8 syn | dense-decode | 5 | f32 | 0.4308 [0.4282-0.4329] | 0.4852 [0.4828-0.4886] | 112.6 | 101 / 90 | slower |
| gdn_in | Q8 syn | dense-decode | 6 | f32 | 0.4397 [0.4339-0.4429] | 0.4955 [0.4945-0.4993] | 112.7 | 99 / 88 | slower |
| gdn_in | Q8 syn | dense-decode | 7 | f32 | 0.4451 [0.4431-0.4472] | 0.5022 [0.4982-0.5071] | 112.8 | 98 / 87 | slower |
| gdn_in | Q8 syn | dense-decode | 8 | f32 | 0.4512 [0.4491-0.4549] | 0.5157 [0.5106-0.5191] | 114.3 | 96 / 84 | slower |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | f32 | 1.5458 [1.5340-1.5676] | 1.3109 [1.2999-1.3236] | 84.8 | 28 / 33 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | f32 | 5.2173 [5.1666-5.6921] | 5.0269 [4.9791-5.3389] | 96.3 | 8 / 9 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | f32 | 22.7877 [20.7388-23.0778] | 22.3977 [20.2668-22.7822] | 98.3 | 2 / 2 | noise |
| gdn_out | Q8 | dense-decode | 1 | bf16 | 0.1802 [0.1795-0.1812] | 0.2342 [0.2328-0.2492] | 129.9 | 93 / 71 | slower |
| gdn_out | Q8 | dense-decode | 2 | bf16 | 0.1831 [0.1807-0.1839] | 0.2496 [0.2446-0.2559] | 136.3 | 91 / 67 | slower |
| gdn_out | Q8 | dense-decode | 3 | bf16 | 0.1865 [0.1853-0.1879] | 0.2754 [0.2679-0.2864] | 147.6 | 90 / 61 | slower |
| gdn_out | Q8 | dense-decode | 4 | bf16 | 0.1936 [0.1912-0.1962] | 0.2962 [0.2851-0.3033] | 152.9 | 86 / 56 | slower |
| gdn_out | Q8 | dense-decode | 5 | bf16 | 0.2092 [0.2025-0.2113] | 0.3476 [0.3293-0.3615] | 166.1 | 80 / 48 | slower |
| gdn_out | Q8 | dense-decode | 6 | bf16 | 0.2196 [0.2171-0.2222] | 0.3550 [0.3463-0.3778] | 161.6 | 76 / 47 | slower |
| gdn_out | Q8 | dense-decode | 7 | bf16 | 0.2250 [0.2240-0.2292] | 0.3519 [0.3423-0.3594] | 156.4 | 74 / 47 | slower |
| gdn_out | Q8 | dense-decode | 8 | bf16 | 0.2327 [0.2294-0.2365] | 0.3383 [0.3318-0.3470] | 145.4 | 72 / 49 | slower |
| gdn_out | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3981 [1.2642-1.4941] | 0.8108 [0.7875-0.8673] | 58.0 | 12 / 21 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.2904 [3.2321-3.4056] | 2.3331 [2.2720-2.4077] | 70.9 | 5 / 7 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.9525 [8.9251-9.0226] | 8.0722 [7.9850-8.1203] | 90.2 | 2 / 2 | faster |
| gdn_out | Q8 | dense-decode | 1 | f32 | 0.1797 [0.1792-0.1806] | 0.2420 [0.2355-0.2528] | 134.7 | 93 / 69 | slower |
| gdn_out | Q8 | dense-decode | 2 | f32 | 0.1824 [0.1819-0.1831] | 0.2522 [0.2428-0.2573] | 138.3 | 92 / 66 | slower |
| gdn_out | Q8 | dense-decode | 3 | f32 | 0.1864 [0.1858-0.1875] | 0.2813 [0.2744-0.2885] | 150.9 | 90 / 59 | slower |
| gdn_out | Q8 | dense-decode | 4 | f32 | 0.1908 [0.1883-0.1934] | 0.2893 [0.2850-0.2961] | 151.6 | 88 / 58 | slower |
| gdn_out | Q8 | dense-decode | 5 | f32 | 0.2084 [0.2044-0.2104] | 0.3604 [0.3531-0.3783] | 172.9 | 80 / 46 | slower |
| gdn_out | Q8 | dense-decode | 6 | f32 | 0.2159 [0.2106-0.2200] | 0.3680 [0.3597-0.3739] | 170.4 | 77 / 45 | slower |
| gdn_out | Q8 | dense-decode | 7 | f32 | 0.2228 [0.2186-0.2283] | 0.3613 [0.3492-0.3691] | 162.2 | 75 / 46 | slower |
| gdn_out | Q8 | dense-decode | 8 | f32 | 0.2246 [0.2232-0.2276] | 0.3521 [0.3410-0.3649] | 156.8 | 74 / 47 | slower |
| gdn_out | Q8 | dense-prefill-wmma | 512 | f32 | 1.2987 [1.2078-1.4021] | 0.7597 [0.7185-0.7801] | 58.5 | 13 / 22 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | f32 | 3.0916 [3.0415-3.1072] | 2.0808 [2.0409-2.1000] | 67.3 | 5 / 8 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | f32 | 8.7920 [8.4708-9.2332] | 7.8647 [7.4599-8.2516] | 89.5 | 2 / 2 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | bf16 | 0.1020 [0.1016-0.1022] | 0.1126 [0.1124-0.1128] | 110.4 | 87 / 79 | slower |
| gdn_out | Q4 syn | dense-decode | 2 | bf16 | 0.1046 [0.1042-0.1049] | 0.1290 [0.1282-0.1314] | 123.4 | 85 / 69 | slower |
| gdn_out | Q4 syn | dense-decode | 3 | bf16 | 0.1083 [0.1077-0.1095] | 0.1424 [0.1399-0.1529] | 131.6 | 82 / 62 | slower |
| gdn_out | Q4 syn | dense-decode | 4 | bf16 | 0.1123 [0.1112-0.1128] | 0.1594 [0.1428-0.1667] | 142.0 | 79 / 56 | slower |
| gdn_out | Q4 syn | dense-decode | 5 | bf16 | 0.1182 [0.1177-0.1187] | 0.2346 [0.2296-0.2408] | 198.5 | 75 / 38 | slower |
| gdn_out | Q4 syn | dense-decode | 6 | bf16 | 0.1233 [0.1215-0.1249] | 0.2563 [0.2484-0.2614] | 207.8 | 72 / 35 | slower |
| gdn_out | Q4 syn | dense-decode | 7 | bf16 | 0.1280 [0.1266-0.1290] | 0.2728 [0.2628-0.2782] | 213.1 | 69 / 32 | slower |
| gdn_out | Q4 syn | dense-decode | 8 | bf16 | 0.1324 [0.1317-0.1336] | 0.2979 [0.2960-0.2996] | 225.1 | 67 / 30 | slower |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0198 [1.0023-1.0605] | 0.6135 [0.6027-0.6554] | 60.2 | 9 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.5806 [2.5377-2.6219] | 2.0163 [1.9644-2.0525] | 78.1 | 3 / 4 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 8.0147 [7.9601-8.1637] | 7.5606 [7.4040-7.6393] | 94.3 | 1 / 1 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | f32 | 0.1024 [0.1021-0.1029] | 0.1128 [0.1123-0.1134] | 110.2 | 86 / 78 | slower |
| gdn_out | Q4 syn | dense-decode | 2 | f32 | 0.1044 [0.1030-0.1052] | 0.1296 [0.1286-0.1318] | 124.2 | 85 / 68 | slower |
| gdn_out | Q4 syn | dense-decode | 3 | f32 | 0.1084 [0.1074-0.1093] | 0.1496 [0.1448-0.1601] | 138.0 | 82 / 59 | slower |
| gdn_out | Q4 syn | dense-decode | 4 | f32 | 0.1117 [0.1110-0.1122] | 0.1662 [0.1524-0.1690] | 148.8 | 79 / 53 | slower |
| gdn_out | Q4 syn | dense-decode | 5 | f32 | 0.1179 [0.1168-0.1186] | 0.2324 [0.2310-0.2339] | 197.1 | 75 / 38 | slower |
| gdn_out | Q4 syn | dense-decode | 6 | f32 | 0.1232 [0.1222-0.1236] | 0.2638 [0.2545-0.2707] | 214.2 | 72 / 34 | slower |
| gdn_out | Q4 syn | dense-decode | 7 | f32 | 0.1276 [0.1266-0.1283] | 0.2722 [0.2657-0.2794] | 213.3 | 69 / 32 | slower |
| gdn_out | Q4 syn | dense-decode | 8 | f32 | 0.1313 [0.1302-0.1333] | 0.2966 [0.2942-0.2983] | 225.8 | 67 / 30 | slower |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0329 [0.9972-1.0593] | 0.6186 [0.6062-0.6400] | 59.9 | 9 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.4951 [2.4646-2.5579] | 1.9156 [1.8974-1.9547] | 76.8 | 4 / 5 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | f32 | 7.9798 [7.9304-8.0736] | 7.4519 [7.3547-7.6059] | 93.4 | 1 / 1 | faster |
| attn_qkv | Q8 | dense-decode | 1 | bf16 | 0.3706 [0.3671-0.3726] | 0.4017 [0.3992-0.4106] | 108.4 | 102 / 94 | slower |
| attn_qkv | Q8 | dense-decode | 2 | bf16 | 0.3691 [0.3678-0.3707] | 0.4346 [0.4263-0.4401] | 117.8 | 103 / 87 | slower |
| attn_qkv | Q8 | dense-decode | 3 | bf16 | 0.3772 [0.3767-0.3812] | 0.4418 [0.4365-0.4479] | 117.2 | 101 / 86 | slower |
| attn_qkv | Q8 | dense-decode | 4 | bf16 | 0.3841 [0.3814-0.3854] | 0.4458 [0.4401-0.4525] | 116.1 | 99 / 85 | slower |
| attn_qkv | Q8 | dense-decode | 5 | bf16 | 0.4015 [0.4001-0.4032] | 0.4380 [0.4345-0.4411] | 109.1 | 95 / 87 | slower |
| attn_qkv | Q8 | dense-decode | 6 | bf16 | 0.4083 [0.4070-0.4100] | 0.4434 [0.4410-0.4474] | 108.6 | 93 / 86 | slower |
| attn_qkv | Q8 | dense-decode | 7 | bf16 | 0.4130 [0.4106-0.4158] | 0.4479 [0.4421-0.4524] | 108.4 | 92 / 85 | slower |
| attn_qkv | Q8 | dense-decode | 8 | bf16 | 0.4226 [0.4194-0.4249] | 0.4502 [0.4454-0.4572] | 106.5 | 90 / 84 | slower |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3662 [1.3535-1.3826] | 1.1351 [1.1297-1.1438] | 83.1 | 28 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.5723 [4.5007-4.7428] | 4.2856 [4.2627-4.4589] | 93.7 | 8 / 9 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | bf16 | 19.0313 [18.4439-19.5364] | 18.4527 [17.9231-18.8113] | 97.0 | 2 / 2 | noise |
| attn_qkv | Q8 | dense-decode | 1 | f32 | 0.3690 [0.3666-0.3697] | 0.4033 [0.4002-0.4173] | 109.3 | 103 / 94 | slower |
| attn_qkv | Q8 | dense-decode | 2 | f32 | 0.3694 [0.3684-0.3717] | 0.4322 [0.4273-0.4412] | 117.0 | 103 / 88 | slower |
| attn_qkv | Q8 | dense-decode | 3 | f32 | 0.3773 [0.3760-0.3793] | 0.4482 [0.4432-0.4529] | 118.8 | 101 / 85 | slower |
| attn_qkv | Q8 | dense-decode | 4 | f32 | 0.3841 [0.3824-0.3852] | 0.4475 [0.4411-0.4507] | 116.5 | 99 / 85 | slower |
| attn_qkv | Q8 | dense-decode | 5 | f32 | 0.4000 [0.3987-0.4014] | 0.4376 [0.4351-0.4429] | 109.4 | 95 / 87 | slower |
| attn_qkv | Q8 | dense-decode | 6 | f32 | 0.4092 [0.4066-0.4111] | 0.4437 [0.4401-0.4482] | 108.4 | 93 / 86 | slower |
| attn_qkv | Q8 | dense-decode | 7 | f32 | 0.4142 [0.4116-0.4166] | 0.4479 [0.4455-0.4513] | 108.1 | 92 / 85 | slower |
| attn_qkv | Q8 | dense-decode | 8 | f32 | 0.4231 [0.4215-0.4276] | 0.4580 [0.4543-0.4637] | 108.2 | 90 / 83 | slower |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | f32 | 1.3885 [1.3688-1.4101] | 1.1396 [1.1344-1.1464] | 82.1 | 27 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | f32 | 4.5230 [4.4996-4.7997] | 4.2332 [4.2163-4.4393] | 93.6 | 8 / 9 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | f32 | 19.6100 [17.6235-19.7406] | 18.9288 [16.9997-19.1810] | 96.5 | 2 / 2 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | bf16 | 0.2144 [0.2131-0.2150] | 0.2307 [0.2296-0.2316] | 107.6 | 94 / 87 | slower |
| attn_qkv | Q4 syn | dense-decode | 2 | bf16 | 0.2151 [0.2138-0.2170] | 0.2469 [0.2444-0.2502] | 114.8 | 93 / 81 | slower |
| attn_qkv | Q4 syn | dense-decode | 3 | bf16 | 0.2194 [0.2185-0.2216] | 0.2573 [0.2532-0.2585] | 117.2 | 92 / 78 | slower |
| attn_qkv | Q4 syn | dense-decode | 4 | bf16 | 0.2238 [0.2211-0.2250] | 0.2612 [0.2559-0.2655] | 116.7 | 90 / 77 | slower |
| attn_qkv | Q4 syn | dense-decode | 5 | bf16 | 0.2327 [0.2308-0.2345] | 0.3104 [0.3070-0.3137] | 133.4 | 86 / 65 | slower |
| attn_qkv | Q4 syn | dense-decode | 6 | bf16 | 0.2379 [0.2363-0.2398] | 0.3387 [0.3358-0.3414] | 142.4 | 84 / 59 | slower |
| attn_qkv | Q4 syn | dense-decode | 7 | bf16 | 0.2420 [0.2383-0.2430] | 0.3529 [0.3509-0.3578] | 145.8 | 83 / 57 | slower |
| attn_qkv | Q4 syn | dense-decode | 8 | bf16 | 0.2483 [0.2465-0.2491] | 0.3736 [0.3698-0.3758] | 150.5 | 81 / 54 | slower |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0755 [1.0683-1.0885] | 1.0163 [1.0041-1.0278] | 94.5 | 19 / 20 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 4.4362 [4.2388-4.5816] | 4.3528 [4.1738-4.4881] | 98.1 | 5 / 5 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 18.8730 [17.3102-19.0329] | 18.5830 [16.9926-18.8275] | 98.5 | 1 / 1 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | f32 | 0.2143 [0.2126-0.2165] | 0.2321 [0.2314-0.2345] | 108.3 | 94 / 87 | slower |
| attn_qkv | Q4 syn | dense-decode | 2 | f32 | 0.2176 [0.2146-0.2190] | 0.2485 [0.2436-0.2530] | 114.2 | 92 / 81 | slower |
| attn_qkv | Q4 syn | dense-decode | 3 | f32 | 0.2203 [0.2188-0.2224] | 0.2555 [0.2529-0.2591] | 116.0 | 91 / 79 | slower |
| attn_qkv | Q4 syn | dense-decode | 4 | f32 | 0.2223 [0.2200-0.2241] | 0.2634 [0.2601-0.2685] | 118.5 | 90 / 76 | slower |
| attn_qkv | Q4 syn | dense-decode | 5 | f32 | 0.2315 [0.2290-0.2336] | 0.3034 [0.2926-0.3099] | 131.1 | 87 / 66 | slower |
| attn_qkv | Q4 syn | dense-decode | 6 | f32 | 0.2372 [0.2344-0.2389] | 0.3429 [0.3365-0.3468] | 144.6 | 85 / 59 | slower |
| attn_qkv | Q4 syn | dense-decode | 7 | f32 | 0.2420 [0.2409-0.2430] | 0.3560 [0.3533-0.3602] | 147.1 | 83 / 56 | slower |
| attn_qkv | Q4 syn | dense-decode | 8 | f32 | 0.2481 [0.2458-0.2494] | 0.3764 [0.3723-0.3834] | 151.7 | 81 / 53 | slower |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0498 [1.0408-1.0704] | 1.0100 [0.9984-1.0163] | 96.2 | 19 / 20 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 4.7573 [4.1517-4.9151] | 4.6562 [4.1120-4.8418] | 97.9 | 4 / 4 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 19.2497 [18.8660-19.6031] | 18.8077 [18.6699-19.7414] | 97.7 | 1 / 1 | noise |
| attn_o | Q8 | dense-decode | 1 | bf16 | 0.1810 [0.1795-0.1818] | 0.2480 [0.2421-0.2572] | 137.1 | 92 / 67 | slower |
| attn_o | Q8 | dense-decode | 2 | bf16 | 0.1833 [0.1827-0.1841] | 0.2520 [0.2476-0.2606] | 137.4 | 91 / 66 | slower |
| attn_o | Q8 | dense-decode | 3 | bf16 | 0.1863 [0.1844-0.1891] | 0.2789 [0.2702-0.2845] | 149.7 | 90 / 60 | slower |
| attn_o | Q8 | dense-decode | 4 | bf16 | 0.1921 [0.1900-0.1964] | 0.2953 [0.2858-0.3028] | 153.8 | 87 / 57 | slower |
| attn_o | Q8 | dense-decode | 5 | bf16 | 0.2023 [0.1992-0.2080] | 0.3479 [0.3402-0.3646] | 172.0 | 83 / 48 | slower |
| attn_o | Q8 | dense-decode | 6 | bf16 | 0.2155 [0.2092-0.2189] | 0.3465 [0.3357-0.3614] | 160.8 | 78 / 48 | slower |
| attn_o | Q8 | dense-decode | 7 | bf16 | 0.2234 [0.2170-0.2278] | 0.3484 [0.3376-0.3610] | 155.9 | 75 / 48 | slower |
| attn_o | Q8 | dense-decode | 8 | bf16 | 0.2276 [0.2252-0.2330] | 0.3345 [0.3271-0.3530] | 146.9 | 73 / 50 | slower |
| attn_o | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3806 [1.2321-1.5025] | 0.7669 [0.7352-0.7944] | 55.5 | 12 / 22 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.2324 [3.1882-3.3010] | 2.3454 [2.3013-2.4033] | 72.6 | 5 / 7 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.5115 [8.4449-8.5990] | 7.5967 [7.5060-7.7944] | 89.2 | 2 / 2 | faster |
| attn_o | Q8 | dense-decode | 1 | f32 | 0.1805 [0.1798-0.1808] | 0.2408 [0.2330-0.2508] | 133.4 | 93 / 69 | slower |
| attn_o | Q8 | dense-decode | 2 | f32 | 0.1819 [0.1798-0.1835] | 0.2502 [0.2458-0.2522] | 137.5 | 92 / 67 | slower |
| attn_o | Q8 | dense-decode | 3 | f32 | 0.1871 [0.1862-0.1880] | 0.2839 [0.2780-0.2888] | 151.8 | 89 / 59 | slower |
| attn_o | Q8 | dense-decode | 4 | f32 | 0.1902 [0.1881-0.1939] | 0.2952 [0.2867-0.3055] | 155.2 | 88 / 57 | slower |
| attn_o | Q8 | dense-decode | 5 | f32 | 0.2036 [0.1991-0.2132] | 0.3615 [0.3392-0.3757] | 177.5 | 82 / 46 | slower |
| attn_o | Q8 | dense-decode | 6 | f32 | 0.2132 [0.2077-0.2198] | 0.3736 [0.3668-0.3788] | 175.2 | 78 / 45 | slower |
| attn_o | Q8 | dense-decode | 7 | f32 | 0.2282 [0.2239-0.2322] | 0.3684 [0.3589-0.3714] | 161.4 | 73 / 45 | slower |
| attn_o | Q8 | dense-decode | 8 | f32 | 0.2285 [0.2230-0.2335] | 0.3554 [0.3379-0.3656] | 155.5 | 73 / 47 | slower |
| attn_o | Q8 | dense-prefill-wmma | 512 | f32 | 1.2860 [1.2440-1.3924] | 0.7806 [0.7745-0.8156] | 60.7 | 13 / 21 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | f32 | 3.1093 [3.0928-3.1384] | 2.0851 [2.0513-2.1189] | 67.1 | 5 / 8 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | f32 | 9.9468 [9.0418-10.0017] | 8.9360 [7.5873-9.0468] | 89.8 | 2 / 2 | noise |
| attn_o | Q4 syn | dense-decode | 1 | bf16 | 0.1013 [0.1001-0.1018] | 0.1121 [0.1118-0.1124] | 110.7 | 87 / 79 | slower |
| attn_o | Q4 syn | dense-decode | 2 | bf16 | 0.1044 [0.1036-0.1046] | 0.1296 [0.1286-0.1310] | 124.2 | 85 / 68 | slower |
| attn_o | Q4 syn | dense-decode | 3 | bf16 | 0.1086 [0.1081-0.1091] | 0.1446 [0.1380-0.1507] | 133.2 | 82 / 61 | slower |
| attn_o | Q4 syn | dense-decode | 4 | bf16 | 0.1116 [0.1111-0.1122] | 0.1556 [0.1452-0.1658] | 139.4 | 79 / 57 | slower |
| attn_o | Q4 syn | dense-decode | 5 | bf16 | 0.1188 [0.1178-0.1194] | 0.2350 [0.2329-0.2404] | 197.8 | 74 / 38 | slower |
| attn_o | Q4 syn | dense-decode | 6 | bf16 | 0.1228 [0.1222-0.1239] | 0.2528 [0.2499-0.2610] | 205.9 | 72 / 35 | slower |
| attn_o | Q4 syn | dense-decode | 7 | bf16 | 0.1277 [0.1269-0.1288] | 0.2719 [0.2692-0.2770] | 213.0 | 69 / 33 | slower |
| attn_o | Q4 syn | dense-decode | 8 | bf16 | 0.1331 [0.1324-0.1339] | 0.2986 [0.2966-0.3005] | 224.4 | 66 / 30 | slower |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0127 [0.9940-1.0744] | 0.6145 [0.6053-0.6371] | 60.7 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.5714 [2.5537-2.6048] | 1.9648 [1.9322-2.0745] | 76.4 | 3 / 4 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 8.1390 [8.0313-8.2467] | 7.5668 [7.4755-7.7333] | 93.0 | 1 / 1 | faster |
| attn_o | Q4 syn | dense-decode | 1 | f32 | 0.1033 [0.1024-0.1036] | 0.1133 [0.1127-0.1146] | 109.7 | 86 / 78 | slower |
| attn_o | Q4 syn | dense-decode | 2 | f32 | 0.1058 [0.1051-0.1076] | 0.1304 [0.1293-0.1325] | 123.2 | 84 / 68 | slower |
| attn_o | Q4 syn | dense-decode | 3 | f32 | 0.1081 [0.1064-0.1092] | 0.1479 [0.1435-0.1593] | 136.9 | 82 / 60 | slower |
| attn_o | Q4 syn | dense-decode | 4 | f32 | 0.1114 [0.1108-0.1122] | 0.1640 [0.1498-0.1703] | 147.2 | 79 / 54 | slower |
| attn_o | Q4 syn | dense-decode | 5 | f32 | 0.1176 [0.1164-0.1187] | 0.2294 [0.2280-0.2344] | 195.0 | 75 / 39 | slower |
| attn_o | Q4 syn | dense-decode | 6 | f32 | 0.1228 [0.1217-0.1238] | 0.2575 [0.2530-0.2619] | 209.6 | 72 / 34 | slower |
| attn_o | Q4 syn | dense-decode | 7 | f32 | 0.1261 [0.1251-0.1266] | 0.2752 [0.2676-0.2784] | 218.2 | 70 / 32 | slower |
| attn_o | Q4 syn | dense-decode | 8 | f32 | 0.1310 [0.1302-0.1319] | 0.2936 [0.2894-0.2968] | 224.2 | 68 / 30 | slower |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0340 [0.9965-1.0624] | 0.6320 [0.6031-0.6441] | 61.1 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.5192 [2.4769-2.5511] | 1.9795 [1.9604-2.0069] | 78.6 | 4 / 4 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | f32 | 7.9555 [7.9078-8.0316] | 7.4247 [7.3223-7.5644] | 93.3 | 1 / 1 | faster |
| ple_kv | Q8 | dense-decode | 1 | bf16 | 0.3539 [0.3527-0.3548] | 0.3826 [0.3810-0.3899] | 108.1 | 98 / 91 | slower |
| ple_kv | Q8 | dense-decode | 2 | bf16 | 0.3536 [0.3520-0.3563] | 0.4226 [0.4177-0.4279] | 119.5 | 98 / 82 | slower |
| ple_kv | Q8 | dense-decode | 3 | bf16 | 0.3624 [0.3603-0.3636] | 0.4320 [0.4247-0.4362] | 119.2 | 96 / 81 | slower |
| ple_kv | Q8 | dense-decode | 4 | bf16 | 0.3679 [0.3670-0.3688] | 0.4321 [0.4250-0.4383] | 117.4 | 95 / 81 | slower |
| ple_kv | Q8 | dense-decode | 5 | bf16 | 0.3837 [0.3821-0.3849] | 0.4216 [0.4190-0.4246] | 109.9 | 91 / 83 | slower |
| ple_kv | Q8 | dense-decode | 6 | bf16 | 0.3895 [0.3874-0.3920] | 0.4282 [0.4264-0.4317] | 109.9 | 89 / 81 | slower |
| ple_kv | Q8 | dense-decode | 7 | bf16 | 0.3934 [0.3898-0.3956] | 0.4344 [0.4288-0.4404] | 110.4 | 88 / 80 | slower |
| ple_kv | Q8 | dense-decode | 8 | bf16 | 0.4006 [0.3980-0.4024] | 0.4362 [0.4313-0.4455] | 108.9 | 87 / 80 | slower |
| ple_kv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.2594 [1.2389-1.2918] | 1.0314 [1.0209-1.0344] | 81.9 | 28 / 34 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.1271 [4.1101-4.1516] | 3.8726 [3.8547-3.8939] | 93.8 | 8 / 9 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | bf16 | 16.3383 [16.2651-16.5972] | 15.7960 [15.7524-15.9078] | 96.7 | 2 / 2 | faster |
| ple_kv | Q8 | dense-decode | 1 | f32 | 0.3542 [0.3510-0.3549] | 0.3828 [0.3803-0.3877] | 108.1 | 98 / 91 | slower |
| ple_kv | Q8 | dense-decode | 2 | f32 | 0.3544 [0.3530-0.3563] | 0.4215 [0.4149-0.4290] | 118.9 | 98 / 83 | slower |
| ple_kv | Q8 | dense-decode | 3 | f32 | 0.3602 [0.3590-0.3620] | 0.4263 [0.4224-0.4347] | 118.3 | 97 / 82 | slower |
| ple_kv | Q8 | dense-decode | 4 | f32 | 0.3654 [0.3630-0.3676] | 0.4309 [0.4266-0.4352] | 117.9 | 95 / 81 | slower |
| ple_kv | Q8 | dense-decode | 5 | f32 | 0.3797 [0.3780-0.3807] | 0.4250 [0.4186-0.4293] | 111.9 | 92 / 82 | slower |
| ple_kv | Q8 | dense-decode | 6 | f32 | 0.3881 [0.3862-0.3890] | 0.4288 [0.4247-0.4337] | 110.5 | 90 / 81 | slower |
| ple_kv | Q8 | dense-decode | 7 | f32 | 0.3941 [0.3897-0.3966] | 0.4316 [0.4292-0.4360] | 109.5 | 88 / 81 | slower |
| ple_kv | Q8 | dense-decode | 8 | f32 | 0.4015 [0.3978-0.4034] | 0.4432 [0.4391-0.4466] | 110.4 | 87 / 79 | slower |
| ple_kv | Q8 | dense-prefill-wmma | 512 | f32 | 1.2512 [1.2281-1.2692] | 1.0271 [1.0131-1.0300] | 82.1 | 28 / 34 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | f32 | 4.1219 [4.0974-4.1426] | 3.8400 [3.8218-3.8651] | 93.2 | 8 / 9 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | f32 | 16.1910 [16.1328-17.8872] | 15.8820 [15.5533-17.0252] | 98.1 | 2 / 2 | noise |
| ple_kv | Q4 syn | dense-decode | 1 | bf16 | 0.1974 [0.1966-0.1993] | 0.2121 [0.2116-0.2140] | 107.5 | 93 / 87 | slower |
| ple_kv | Q4 syn | dense-decode | 2 | bf16 | 0.2008 [0.1975-0.2018] | 0.2343 [0.2295-0.2369] | 116.7 | 92 / 79 | slower |
| ple_kv | Q4 syn | dense-decode | 3 | bf16 | 0.2038 [0.2032-0.2047] | 0.2406 [0.2359-0.2442] | 118.1 | 90 / 77 | slower |
| ple_kv | Q4 syn | dense-decode | 4 | bf16 | 0.2060 [0.2053-0.2092] | 0.2503 [0.2437-0.2551] | 121.5 | 89 / 74 | slower |
| ple_kv | Q4 syn | dense-decode | 5 | bf16 | 0.2146 [0.2132-0.2170] | 0.2941 [0.2903-0.2968] | 137.1 | 86 / 63 | slower |
| ple_kv | Q4 syn | dense-decode | 6 | bf16 | 0.2195 [0.2182-0.2210] | 0.3173 [0.3143-0.3216] | 144.6 | 84 / 58 | slower |
| ple_kv | Q4 syn | dense-decode | 7 | bf16 | 0.2240 [0.2231-0.2258] | 0.3329 [0.3309-0.3366] | 148.6 | 82 / 55 | slower |
| ple_kv | Q4 syn | dense-decode | 8 | bf16 | 0.2301 [0.2268-0.2322] | 0.3536 [0.3509-0.3590] | 153.6 | 80 / 52 | slower |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.9676 [0.9618-0.9729] | 0.9111 [0.9074-0.9150] | 94.2 | 19 / 20 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 3.9685 [3.9191-3.9838] | 3.8781 [3.8153-3.9125] | 97.7 | 5 / 5 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 16.3548 [15.7111-16.9868] | 15.8798 [15.4640-16.4762] | 97.1 | 1 / 1 | noise |
| ple_kv | Q4 syn | dense-decode | 1 | f32 | 0.1989 [0.1974-0.1999] | 0.2130 [0.2114-0.2149] | 107.1 | 93 / 87 | slower |
| ple_kv | Q4 syn | dense-decode | 2 | f32 | 0.2011 [0.1998-0.2028] | 0.2300 [0.2280-0.2338] | 114.4 | 92 / 80 | slower |
| ple_kv | Q4 syn | dense-decode | 3 | f32 | 0.2048 [0.2034-0.2064] | 0.2425 [0.2383-0.2469] | 118.4 | 90 / 76 | slower |
| ple_kv | Q4 syn | dense-decode | 4 | f32 | 0.2068 [0.2050-0.2092] | 0.2511 [0.2450-0.2557] | 121.4 | 89 / 73 | slower |
| ple_kv | Q4 syn | dense-decode | 5 | f32 | 0.2142 [0.2124-0.2155] | 0.2874 [0.2778-0.2898] | 134.2 | 86 / 64 | slower |
| ple_kv | Q4 syn | dense-decode | 6 | f32 | 0.2192 [0.2169-0.2208] | 0.3226 [0.3184-0.3268] | 147.2 | 84 / 57 | slower |
| ple_kv | Q4 syn | dense-decode | 7 | f32 | 0.2250 [0.2232-0.2270] | 0.3370 [0.3341-0.3438] | 149.8 | 82 / 55 | slower |
| ple_kv | Q4 syn | dense-decode | 8 | f32 | 0.2302 [0.2280-0.2335] | 0.3591 [0.3564-0.3624] | 156.0 | 80 / 51 | slower |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.9691 [0.9580-1.0267] | 0.9139 [0.8982-0.9671] | 94.3 | 19 / 20 | noise |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 3.8975 [3.7806-4.2763] | 3.7824 [3.6833-4.1366] | 97.0 | 5 / 5 | noise |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 17.3089 [16.9768-17.4552] | 17.1045 [16.9182-17.2717] | 98.8 | 1 / 1 | noise |
| lm_head | Q8 | dense-decode | 1 | bf16 | 3.2324 [3.2186-3.2618] | 3.4006 [3.3906-3.4366] | 105.2 | 209 / 199 | slower |
| lm_head | Q8 | dense-decode | 2 | bf16 | 3.2116 [3.2028-3.2210] | 3.4388 [3.4352-3.4512] | 107.1 | 210 / 196 | slower |
| lm_head | Q8 | dense-decode | 3 | bf16 | 3.4244 [3.4189-3.4264] | 3.4622 [3.4515-3.4712] | 101.1 | 197 / 195 | slower |
| lm_head | Q8 | dense-decode | 4 | bf16 | 3.5850 [3.5660-3.6008] | 3.6098 [3.5923-3.6188] | 100.7 | 188 / 187 | noise |
| lm_head | Q8 | dense-decode | 5 | bf16 | 3.4626 [3.4389-3.4929] | 3.7618 [3.7471-3.7753] | 108.6 | 195 / 180 | slower |
| lm_head | Q8 | dense-decode | 6 | bf16 | 3.5281 [3.5174-3.5538] | 3.9263 [3.9205-3.9373] | 111.3 | 191 / 172 | slower |
| lm_head | Q8 | dense-decode | 7 | bf16 | 3.6192 [3.6108-3.6407] | 4.2159 [4.1974-4.2314] | 116.5 | 187 / 160 | slower |
| lm_head | Q8 | dense-decode | 8 | bf16 | 3.7472 [3.7154-3.7580] | 4.5354 [4.5221-4.5575] | 121.0 | 180 / 149 | slower |
| lm_head | Q8 | dense-prefill-wmma | 512 | bf16 | 22.4862 [22.3895-23.7142] | 21.9388 [21.7528-23.1645] | 97.6 | 30 / 31 | noise |
| lm_head | Q8 | dense-prefill-wmma | 2048 | bf16 | 90.8020 [89.8595-91.2147] | 87.9864 [87.3279-88.5099] | 96.9 | 7 / 8 | faster |
| lm_head | Q8 | dense-decode | 1 | f32 | 3.2154 [3.2114-3.2315] | 3.3755 [3.3624-3.3912] | 105.0 | 210 / 200 | slower |
| lm_head | Q8 | dense-decode | 2 | f32 | 3.2245 [3.2167-3.2392] | 3.4396 [3.4238-3.4634] | 106.7 | 209 / 196 | slower |
| lm_head | Q8 | dense-decode | 3 | f32 | 3.4213 [3.4184-3.4248] | 3.4641 [3.4575-3.4729] | 101.2 | 197 / 195 | slower |
| lm_head | Q8 | dense-decode | 4 | f32 | 3.5721 [3.5668-3.5794] | 3.5640 [3.5546-3.5699] | 99.8 | 189 / 190 | noise |
| lm_head | Q8 | dense-decode | 5 | f32 | 3.4515 [3.4437-3.4650] | 3.8572 [3.8303-3.8698] | 111.8 | 196 / 175 | slower |
| lm_head | Q8 | dense-decode | 6 | f32 | 3.5298 [3.5033-3.5576] | 4.0279 [4.0163-4.0408] | 114.1 | 191 / 168 | slower |
| lm_head | Q8 | dense-decode | 7 | f32 | 3.6569 [3.6146-3.6762] | 4.2860 [4.2614-4.3025] | 117.2 | 185 / 158 | slower |
| lm_head | Q8 | dense-decode | 8 | f32 | 3.7709 [3.7381-3.8181] | 4.5909 [4.5815-4.6001] | 121.7 | 179 / 147 | slower |
| lm_head | Q8 | dense-prefill-wmma | 512 | f32 | 21.9911 [21.8617-22.4426] | 21.2983 [21.1574-21.8835] | 96.8 | 31 / 32 | noise |
| lm_head | Q4 syn | dense-decode | 1 | bf16 | 1.8250 [1.8076-1.8334] | 1.9021 [1.8891-1.9087] | 104.2 | 196 / 188 | slower |
| lm_head | Q4 syn | dense-decode | 2 | bf16 | 1.8578 [1.8479-1.8605] | 1.9398 [1.9334-1.9483] | 104.4 | 192 / 184 | slower |
| lm_head | Q4 syn | dense-decode | 3 | bf16 | 2.0682 [2.0558-2.0800] | 1.9872 [1.9757-1.9928] | 96.1 | 173 / 180 | faster |
| lm_head | Q4 syn | dense-decode | 4 | bf16 | 2.1995 [2.1907-2.2105] | 2.0827 [2.0745-2.0890] | 94.7 | 163 / 172 | faster |
| lm_head | Q4 syn | dense-decode | 5 | bf16 | 2.4038 [2.3604-2.5300] | 2.8293 [2.7598-2.9885] | 117.7 | 149 / 126 | slower |
| lm_head | Q4 syn | dense-decode | 6 | bf16 | 2.4430 [2.4253-2.4491] | 2.9861 [2.9558-3.0193] | 122.2 | 146 / 120 | slower |
| lm_head | Q4 syn | dense-decode | 7 | bf16 | 2.6365 [2.6263-2.6493] | 3.4297 [3.3987-3.4413] | 130.1 | 136 / 104 | slower |
| lm_head | Q4 syn | dense-decode | 8 | bf16 | 2.8363 [2.8179-2.8526] | 3.8054 [3.7944-3.8319] | 134.2 | 126 / 94 | slower |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | bf16 | 21.9730 [21.6035-22.4361] | 21.4573 [21.1577-22.0309] | 97.7 | 16 / 17 | noise |
| lm_head | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 88.0401 [87.1329-88.7044] | 85.8792 [85.0714-86.5134] | 97.5 | 4 / 4 | faster |
| lm_head | Q4 syn | dense-decode | 1 | f32 | 1.8123 [1.8050-1.8254] | 1.8923 [1.8880-1.8996] | 104.4 | 197 / 189 | slower |
| lm_head | Q4 syn | dense-decode | 2 | f32 | 1.8674 [1.8610-1.8725] | 1.9273 [1.9230-1.9331] | 103.2 | 191 / 186 | slower |
| lm_head | Q4 syn | dense-decode | 3 | f32 | 2.0552 [2.0509-2.0563] | 1.9657 [1.9632-1.9750] | 95.7 | 174 / 182 | faster |
| lm_head | Q4 syn | dense-decode | 4 | f32 | 2.1635 [2.1601-2.1688] | 2.0616 [2.0531-2.0654] | 95.3 | 165 / 173 | faster |
| lm_head | Q4 syn | dense-decode | 5 | f32 | 2.4219 [2.4148-2.4273] | 2.8800 [2.8679-2.8865] | 118.9 | 148 / 124 | slower |
| lm_head | Q4 syn | dense-decode | 6 | f32 | 2.5104 [2.4918-2.5156] | 3.1764 [3.1639-3.1949] | 126.5 | 142 / 113 | slower |
| lm_head | Q4 syn | dense-decode | 7 | f32 | 2.6337 [2.6246-2.6667] | 3.5156 [3.5082-3.5425] | 133.5 | 136 / 102 | slower |
| lm_head | Q4 syn | dense-decode | 8 | f32 | 2.8335 [2.8207-2.8470] | 3.9171 [3.9056-3.9546] | 138.2 | 126 / 91 | slower |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | f32 | 21.4205 [21.1176-21.6439] | 21.0420 [20.7911-21.2386] | 98.2 | 17 / 17 | noise |
| mtp_fc | Q8 | dense-decode | 1 | bf16 | 0.0810 [0.0807-0.0814] | 0.0864 [0.0858-0.0874] | 106.6 | 86 / 81 | slower |
| mtp_fc | Q8 | dense-decode | 2 | bf16 | 0.0838 [0.0835-0.0843] | 0.1010 [0.1005-0.1022] | 120.5 | 83 / 69 | slower |
| mtp_fc | Q8 | dense-decode | 3 | bf16 | 0.0864 [0.0859-0.0872] | 0.1123 [0.1118-0.1132] | 130.1 | 81 / 62 | slower |
| mtp_fc | Q8 | dense-decode | 4 | bf16 | 0.0893 [0.0888-0.0900] | 0.1238 [0.1233-0.1244] | 138.7 | 78 / 56 | slower |
| mtp_fc | Q8 | dense-decode | 5 | bf16 | 0.0929 [0.0922-0.0932] | 0.1363 [0.1358-0.1373] | 146.8 | 75 / 51 | slower |
| mtp_fc | Q8 | dense-decode | 6 | bf16 | 0.0955 [0.0952-0.0959] | 0.1478 [0.1466-0.1488] | 154.8 | 73 / 47 | slower |
| mtp_fc | Q8 | dense-decode | 7 | bf16 | 0.0980 [0.0975-0.0984] | 0.1589 [0.1568-0.1603] | 162.2 | 71 / 44 | slower |
| mtp_fc | Q8 | dense-decode | 8 | bf16 | 0.0999 [0.0994-0.1007] | 0.1676 [0.1662-0.1687] | 167.8 | 70 / 42 | slower |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | bf16 | 0.2908 [0.2874-0.2954] | 0.2272 [0.2261-0.2293] | 78.1 | 24 / 31 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.8400 [0.8268-0.8520] | 0.7826 [0.7717-0.7920] | 93.2 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | bf16 | 3.1502 [3.1314-3.1867] | 3.0807 [3.0417-3.1108] | 97.8 | 2 / 2 | faster |
| mtp_fc | Q8 | dense-decode | 1 | f32 | 0.0816 [0.0811-0.0824] | 0.0867 [0.0862-0.0873] | 106.2 | 85 / 80 | slower |
| mtp_fc | Q8 | dense-decode | 2 | f32 | 0.0834 [0.0827-0.0838] | 0.1014 [0.1005-0.1023] | 121.6 | 83 / 69 | slower |
| mtp_fc | Q8 | dense-decode | 3 | f32 | 0.0865 [0.0862-0.0869] | 0.1119 [0.1116-0.1126] | 129.4 | 81 / 62 | slower |
| mtp_fc | Q8 | dense-decode | 4 | f32 | 0.0895 [0.0890-0.0914] | 0.1239 [0.1236-0.1255] | 138.4 | 78 / 56 | slower |
| mtp_fc | Q8 | dense-decode | 5 | f32 | 0.0935 [0.0932-0.0940] | 0.1366 [0.1354-0.1372] | 146.1 | 74 / 51 | slower |
| mtp_fc | Q8 | dense-decode | 6 | f32 | 0.0951 [0.0945-0.0955] | 0.1476 [0.1470-0.1482] | 155.2 | 73 / 47 | slower |
| mtp_fc | Q8 | dense-decode | 7 | f32 | 0.0972 [0.0964-0.0974] | 0.1584 [0.1574-0.1594] | 163.0 | 72 / 44 | slower |
| mtp_fc | Q8 | dense-decode | 8 | f32 | 0.0997 [0.0990-0.1003] | 0.1675 [0.1668-0.1684] | 167.9 | 70 / 42 | slower |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | f32 | 0.2947 [0.2929-0.2976] | 0.2316 [0.2307-0.2334] | 78.6 | 24 / 30 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | f32 | 0.8313 [0.8250-0.8462] | 0.7761 [0.7721-0.7812] | 93.4 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | f32 | 3.0607 [3.0483-3.0786] | 2.9694 [2.9581-2.9962] | 97.0 | 2 / 2 | faster |
| mtp_fc | Q4 syn | dense-decode | 1 | bf16 | 0.0450 [0.0446-0.0455] | 0.0464 [0.0462-0.0466] | 103.0 | 82 / 80 | slower |
| mtp_fc | Q4 syn | dense-decode | 2 | bf16 | 0.0481 [0.0470-0.0495] | 0.0522 [0.0518-0.0532] | 108.4 | 77 / 71 | slower |
| mtp_fc | Q4 syn | dense-decode | 3 | bf16 | 0.0506 [0.0496-0.0520] | 0.0620 [0.0603-0.0629] | 122.6 | 73 / 59 | slower |
| mtp_fc | Q4 syn | dense-decode | 4 | bf16 | 0.0525 [0.0518-0.0532] | 0.0761 [0.0660-0.0764] | 145.1 | 70 / 48 | slower |
| mtp_fc | Q4 syn | dense-decode | 5 | bf16 | 0.0550 [0.0541-0.0562] | 0.0957 [0.0949-0.0965] | 174.0 | 67 / 39 | slower |
| mtp_fc | Q4 syn | dense-decode | 6 | bf16 | 0.0567 [0.0556-0.0577] | 0.1060 [0.1050-0.1067] | 186.9 | 65 / 35 | slower |
| mtp_fc | Q4 syn | dense-decode | 7 | bf16 | 0.0599 [0.0593-0.0609] | 0.1138 [0.1132-0.1151] | 190.0 | 62 / 32 | slower |
| mtp_fc | Q4 syn | dense-decode | 8 | bf16 | 0.0618 [0.0611-0.0630] | 0.1233 [0.1227-0.1238] | 199.6 | 60 / 30 | slower |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.2264 [0.2254-0.2268] | 0.2104 [0.2102-0.2118] | 92.9 | 16 / 18 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.8462 [0.8015-0.8665] | 0.8260 [0.7895-0.8502] | 97.6 | 4 / 4 | noise |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 3.0730 [3.0526-3.2398] | 3.0357 [3.0125-3.1711] | 98.8 | 1 / 1 | noise |
| mtp_fc | Q4 syn | dense-decode | 1 | f32 | 0.0447 [0.0442-0.0450] | 0.0462 [0.0459-0.0466] | 103.4 | 83 / 80 | slower |
| mtp_fc | Q4 syn | dense-decode | 2 | f32 | 0.0484 [0.0471-0.0488] | 0.0513 [0.0502-0.0517] | 106.1 | 76 / 72 | slower |
| mtp_fc | Q4 syn | dense-decode | 3 | f32 | 0.0500 [0.0492-0.0510] | 0.0624 [0.0615-0.0628] | 124.8 | 74 / 59 | slower |
| mtp_fc | Q4 syn | dense-decode | 4 | f32 | 0.0512 [0.0508-0.0520] | 0.0764 [0.0761-0.0764] | 149.0 | 72 / 48 | slower |
| mtp_fc | Q4 syn | dense-decode | 5 | f32 | 0.0557 [0.0550-0.0564] | 0.0958 [0.0927-0.0980] | 171.9 | 66 / 38 | slower |
| mtp_fc | Q4 syn | dense-decode | 6 | f32 | 0.0570 [0.0566-0.0581] | 0.1086 [0.1065-0.1097] | 190.6 | 65 / 34 | slower |
| mtp_fc | Q4 syn | dense-decode | 7 | f32 | 0.0586 [0.0577-0.0591] | 0.1160 [0.1146-0.1180] | 197.9 | 63 / 32 | slower |
| mtp_fc | Q4 syn | dense-decode | 8 | f32 | 0.0609 [0.0601-0.0620] | 0.1261 [0.1250-0.1267] | 207.0 | 61 / 29 | slower |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.2333 [0.2311-0.2354] | 0.2162 [0.2155-0.2164] | 92.7 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.7741 [0.7716-0.7831] | 0.7674 [0.7638-0.7723] | 99.2 | 5 / 5 | noise |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | f32 | 3.0373 [3.0152-3.0652] | 2.9954 [2.9795-3.0348] | 98.6 | 1 / 1 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 1 | bf16 | 0.1897 [0.1881-0.1914] | 0.2098 [0.2064-0.2125] | 110.6 | 92 / 83 | slower |
| exp_gu | Q4 | expert-decode-gather | 1 | bf16 | 0.1862 [0.1855-0.1870] | 0.2032 [0.2017-0.2050] | 109.1 | 93 / 86 | slower |
| exp_down | Q4 | expert-decode-combine | 1 | bf16 | 0.1238 [0.1233-0.1241] | 0.1214 [0.1208-0.1223] | 98.0 | 83 / 84 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 2 | bf16 | 0.3325 [0.3311-0.3346] | 0.3906 [0.3871-0.3980] | 117.5 | 99 / 85 | slower |
| exp_gu | Q4 | expert-decode-gather | 2 | bf16 | 0.3324 [0.3304-0.3339] | 0.3654 [0.3638-0.3664] | 109.9 | 100 / 91 | slower |
| exp_down | Q4 | expert-decode-combine | 2 | bf16 | 0.2228 [0.2216-0.2238] | 0.2208 [0.2200-0.2222] | 99.1 | 87 / 88 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 3 | bf16 | 0.4100 [0.4089-0.4116] | 0.4567 [0.4534-0.4631] | 111.4 | 110 / 99 | slower |
| exp_gu | Q4 | expert-decode-gather | 3 | bf16 | 0.4131 [0.4113-0.4142] | 0.4376 [0.4359-0.4395] | 105.9 | 110 / 103 | slower |
| exp_down | Q4 | expert-decode-combine | 3 | bf16 | 0.2998 [0.2976-0.3003] | 0.3016 [0.2996-0.3037] | 100.6 | 89 / 88 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 4 | bf16 | 0.4764 [0.4743-0.4776] | 0.5310 [0.5248-0.5395] | 111.5 | 124 / 111 | slower |
| exp_gu | Q4 | expert-decode-gather | 4 | bf16 | 0.4792 [0.4770-0.4811] | 0.5056 [0.5042-0.5070] | 105.5 | 124 / 117 | slower |
| exp_down | Q4 | expert-decode-combine | 4 | bf16 | 0.3706 [0.3694-0.3722] | 0.3772 [0.3733-0.3816] | 101.8 | 94 / 92 | slower |
| exp_gu | Q4 | expert-decode-gather-swiglu | 5 | bf16 | 0.5075 [0.5041-0.5094] | 0.5553 [0.5518-0.5633] | 109.4 | 120 / 110 | slower |
| exp_gu | Q4 | expert-decode-gather | 5 | bf16 | 0.5187 [0.5167-0.5203] | 0.5190 [0.5180-0.5195] | 100.0 | 117 / 117 | noise |
| exp_down | Q4 | expert-decode-combine | 5 | bf16 | 0.3790 [0.3750-0.3808] | 0.4047 [0.4016-0.4085] | 106.8 | 95 / 89 | slower |
| exp_gu | Q4 | expert-decode-gather-swiglu | 6 | bf16 | 0.5820 [0.5789-0.5850] | 0.6232 [0.6172-0.6274] | 107.1 | 129 / 120 | slower |
| exp_gu | Q4 | expert-decode-gather | 6 | bf16 | 0.5953 [0.5938-0.5971] | 0.5988 [0.5967-0.6021] | 100.6 | 126 / 125 | noise |
| exp_down | Q4 | expert-decode-combine | 6 | bf16 | 0.4283 [0.4265-0.4295] | 0.4509 [0.4481-0.4570] | 105.3 | 103 / 98 | slower |
| exp_gu | Q4 | expert-decode-gather-swiglu | 7 | bf16 | 0.6345 [0.6340-0.6374] | 0.6637 [0.6579-0.6701] | 104.6 | 129 / 123 | slower |
| exp_gu | Q4 | expert-decode-gather | 7 | bf16 | 0.6502 [0.6487-0.6519] | 0.6299 [0.6276-0.6328] | 96.9 | 126 / 130 | faster |
| exp_down | Q4 | expert-decode-combine | 7 | bf16 | 0.4576 [0.4553-0.4591] | 0.4598 [0.4574-0.4623] | 100.5 | 105 / 105 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 8 | bf16 | 0.7194 [0.7180-0.7215] | 0.7682 [0.7647-0.7723] | 106.8 | 126 / 118 | slower |
| exp_gu | Q4 | expert-decode-gather | 8 | bf16 | 0.7355 [0.7339-0.7385] | 0.7424 [0.7418-0.7447] | 101.0 | 123 / 122 | slower |
| exp_down | Q4 | expert-decode-combine | 8 | bf16 | 0.4963 [0.4947-0.4975] | 0.5005 [0.4973-0.5043] | 100.8 | 107 / 106 | noise |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 512 | bf16 | 4.4939 [4.3685-4.6181] | 4.2551 [4.1265-4.3136] | 94.7 | 177 / 187 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 512 | bf16 | 4.6760 [4.6574-4.6855] | 4.3520 [4.3361-4.3684] | 93.1 | 170 / 183 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 512 | bf16 | 4.4561 [4.4469-4.4745] | 4.4085 [4.3930-4.4325] | 98.9 | 179 / 180 | faster |
| exp_down | Q4 | expert-prefill-combine narrow | 512 | bf16 | 3.1107 [3.0990-3.1221] | 2.8269 [2.8234-2.8354] | 90.9 | 150 / 166 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 512 | bf16 | 3.0506 [3.0470-3.0614] | 2.7987 [2.7943-2.8015] | 91.7 | 153 / 167 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 512 | bf16 | 3.0729 [3.0657-3.0834] | 2.8094 [2.8041-2.8166] | 91.4 | 152 / 167 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 512 | bf16 | 5.8347 [5.7972-5.8663] | 5.7937 [5.7602-5.8109] | 99.3 | 136 / 137 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 512 | bf16 | 5.8853 [5.8766-5.9000] | 5.8551 [5.8434-5.8706] | 99.5 | 135 / 136 | faster |
| exp_gu | Q4 | expert-prefill-gather wide | 512 | bf16 | 5.8969 [5.8549-5.9456] | 5.7931 [5.7485-5.8524] | 98.2 | 135 / 137 | faster |
| exp_down | Q4 | expert-prefill-combine wide | 512 | bf16 | 3.5806 [3.5745-3.5882] | 3.2242 [3.2192-3.2280] | 90.0 | 131 / 145 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 512 | bf16 | 3.4798 [3.4672-3.4888] | 3.1223 [3.1154-3.1285] | 89.7 | 134 / 150 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 512 | bf16 | 3.4962 [3.4878-3.5022] | 3.1371 [3.1316-3.1444] | 89.7 | 134 / 149 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 512 | bf16 | 5.6172 [5.6039-5.6461] | 5.2721 [5.2384-5.3003] | 93.9 | 142 / 151 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 512 | bf16 | 5.6199 [5.5986-5.6365] | 5.2523 [5.2323-5.2776] | 93.5 | 142 / 151 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 512 | bf16 | 5.6762 [5.6487-5.7233] | 5.3319 [5.2993-5.3616] | 93.9 | 140 / 149 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 512 | bf16 | 4.3273 [4.2478-4.3489] | 2.9776 [2.9709-2.9851] | 68.8 | 108 / 157 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 512 | bf16 | 4.3696 [4.3392-4.4133] | 3.0044 [2.9608-3.0122] | 68.8 | 107 / 156 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 512 | bf16 | 4.3904 [4.3769-4.4035] | 3.0237 [3.0176-3.0257] | 68.9 | 107 / 155 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 2048 | bf16 | 7.5587 [7.4830-8.0647] | 7.0904 [7.0098-7.4421] | 93.8 | 118 / 125 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 2048 | bf16 | 8.1893 [8.1409-8.2193] | 7.6647 [7.6455-7.6964] | 93.6 | 109 / 116 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 2048 | bf16 | 7.9776 [7.9079-8.0934] | 7.9703 [7.8223-8.0623] | 99.9 | 112 / 112 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 2048 | bf16 | 5.4742 [5.4470-5.4800] | 5.0849 [5.0706-5.0986] | 92.9 | 96 / 103 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 2048 | bf16 | 5.4385 [5.4094-5.4478] | 5.0576 [5.0357-5.0710] | 93.0 | 96 / 103 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 2048 | bf16 | 5.4466 [5.4309-5.4878] | 5.0970 [5.0687-5.1158] | 93.6 | 96 / 103 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 2048 | bf16 | 9.3176 [9.2626-9.4060] | 9.3117 [9.2202-9.3962] | 99.9 | 95 / 96 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 2048 | bf16 | 9.3236 [9.2793-9.3637] | 9.2867 [9.2457-9.3542] | 99.6 | 95 / 96 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 2048 | bf16 | 9.4063 [9.2260-9.4517] | 9.3239 [9.1502-9.3650] | 99.1 | 95 / 95 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 2048 | bf16 | 5.9071 [5.8594-5.9265] | 5.4395 [5.4010-5.4826] | 92.1 | 89 / 96 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 2048 | bf16 | 5.7632 [5.7182-5.7880] | 5.2656 [5.2460-5.3150] | 91.4 | 91 / 99 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 2048 | bf16 | 5.8202 [5.7855-5.8449] | 5.3527 [5.3055-5.3725] | 92.0 | 90 / 98 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 2048 | bf16 | 8.7893 [8.6892-8.9860] | 8.4993 [8.3719-8.8322] | 96.7 | 101 / 105 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 2048 | bf16 | 8.5898 [8.5667-8.6107] | 8.2591 [8.2290-8.2953] | 96.2 | 104 / 108 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 2048 | bf16 | 8.9814 [8.7817-9.0348] | 8.6335 [8.2377-8.6735] | 96.1 | 99 / 103 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 2048 | bf16 | 6.1962 [6.1415-6.2225] | 5.1525 [5.1288-5.1671] | 83.2 | 84 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 2048 | bf16 | 6.2353 [6.1853-6.2763] | 5.2323 [5.1891-5.2862] | 83.9 | 84 / 100 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 2048 | bf16 | 6.2070 [6.1480-6.2376] | 5.2235 [5.1473-5.2505] | 84.2 | 84 / 100 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 8192 | bf16 | 23.2124 [23.0129-80.4002] | 22.6762 [22.3471-76.8473] | 97.7 | 38 / 39 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 8192 | bf16 | 81.2698 [81.1007-81.3916] | 78.0315 [77.8921-78.1744] | 96.0 | 11 / 11 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 8192 | bf16 | 22.5357 [22.1093-22.8012] | 22.2158 [21.9822-22.8614] | 98.6 | 40 / 40 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 8192 | bf16 | 14.9464 [14.9062-15.1871] | 14.5149 [14.3584-14.7346] | 97.1 | 35 / 36 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 8192 | bf16 | 14.6629 [14.6214-14.7107] | 14.1599 [14.0729-14.2203] | 96.6 | 36 / 37 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 8192 | bf16 | 14.7275 [14.1368-14.8009] | 14.1837 [13.5520-14.3184] | 96.3 | 36 / 37 | noise |
| exp_gu | Q4 | expert-prefill-swiglu wide | 8192 | bf16 | 21.1458 [21.0018-21.3981] | 21.2001 [21.0533-21.3556] | 100.3 | 42 / 42 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 8192 | bf16 | 21.5502 [21.4049-21.9192] | 21.5768 [21.4924-21.9148] | 100.1 | 41 / 41 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 8192 | bf16 | 21.2944 [21.1941-21.6321] | 21.2309 [21.1345-21.7533] | 99.7 | 42 / 42 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 8192 | bf16 | 14.1157 [14.0239-14.2431] | 13.5063 [13.2844-13.6364] | 95.7 | 37 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 8192 | bf16 | 14.0710 [14.0500-14.1091] | 13.4052 [13.3226-13.4436] | 95.3 | 37 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 8192 | bf16 | 14.0444 [14.0007-14.2128] | 13.3681 [13.3066-13.5538] | 95.2 | 37 / 39 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 8192 | bf16 | 21.4296 [21.0546-21.8967] | 20.9008 [20.5185-21.3149] | 97.5 | 42 / 43 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 8192 | bf16 | 20.8036 [20.3511-21.0872] | 20.4639 [20.0029-20.5496] | 98.4 | 43 / 44 | noise |
| exp_gu | Q4 | expert-prefill-gather wide256 | 8192 | bf16 | 20.9808 [20.6031-21.2965] | 20.0124 [19.7041-20.4318] | 95.4 | 42 / 45 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 8192 | bf16 | 15.1837 [14.9945-15.3501] | 13.6218 [13.4442-13.7602] | 89.7 | 35 / 38 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 8192 | bf16 | 15.1056 [14.8475-15.1894] | 13.5817 [13.4226-13.7436] | 89.9 | 35 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 8192 | bf16 | 15.2812 [15.1445-15.3970] | 13.8074 [13.6354-13.9028] | 90.4 | 34 / 38 | faster |
