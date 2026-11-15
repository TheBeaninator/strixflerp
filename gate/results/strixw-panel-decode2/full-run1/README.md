# strixw panel kernels: kernel gate results

Criteria: `gate/strixw-panel-kernels.toml` (committed before the run). Raw: `raw.jsonl` (one case a line), bench log `bench.log`. 489 cases, 6 skipped (output too large), 2.232 GB read from the fleet file.

## Verdicts

| criterion | pass | detail |
|---|---|---|
| equal | PASS | 489 of 489 cases byte-equal |
| bad_id | PASS | 21 of 21 out-of-range-id cases byte-equal |
| coverage | PASS | missing: none |
| read | FAIL | 2.232 GB read |
| speed_decode_dense | PASS | median panel % of origin time per class: attn_o 99.1, attn_qkv 99.2, gdn_in 98.0, gdn_out 98.5, lm_head 96.5, mtp_fc 99.0, ple_kv 99.2, sh_down 86.5, sh_gu 99.4 |
| speed_prefill_dense | PASS | sh_gu M2048 bf16 11.2%; sh_gu M8192 bf16 96.3%; sh_gu M2048 f32 90.6%; sh_gu M8192 f32 96.3%; sh_down M2048 bf16 92.5%; sh_down M8192 bf16 95.7%; sh_down M2048 f32 94.8%; sh_down M8192 f32 95.2%; gdn_in M2048 bf16 99.0%; gdn_in M8192 bf16 96.4%; gdn_in M2048 f32 99.2%; gdn_in M8192 f32 98.8%; gdn_out M2048 bf16 72.5%; gdn_out M8192 bf16 89.9%; gdn_out M2048 f32 67.2%; gdn_out M8192 f32 91.0%; attn_qkv M2048 bf16 93.9%; attn_qkv M8192 bf16 94.2%; attn_qkv M2048 f32 93.5%; attn_qkv M8192 f32 96.7%; attn_o M2048 bf16 69.2%; attn_o M8192 bf16 89.7%; attn_o M2048 f32 67.1%; attn_o M8192 f32 88.9%; ple_kv M2048 bf16 93.8%; ple_kv M8192 bf16 96.4%; ple_kv M2048 f32 93.8%; ple_kv M8192 f32 96.0%; lm_head M2048 bf16 97.2%; mtp_fc M2048 bf16 93.2%; mtp_fc M8192 bf16 98.5%; mtp_fc M2048 f32 94.6%; mtp_fc M8192 f32 97.3% |
| speed_prefill_expert | PASS | expert-prefill-swiglu-f16h narrow M2048 94.0% (faster); expert-prefill-combine-f16h-shared narrow M2048 93.4% (faster); expert-prefill-combine-f16h-shared wide M8192 95.3% (faster); expert-prefill-swiglu-f16h wide256 M8192 98.1% (noise) |

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
| sh_gu | Q8 | dense-decode | 1 | bf16 | 0.0416 [0.0411-0.0417] | 0.0415 [0.0414-0.0417] | 99.8 | 84 / 84 | noise |
| sh_gu | Q8 | dense-decode | 2 | bf16 | 0.0429 [0.0424-0.0434] | 0.0433 [0.0430-0.0438] | 100.9 | 81 / 80 | noise |
| sh_gu | Q8 | dense-decode | 3 | bf16 | 0.0455 [0.0448-0.0460] | 0.0464 [0.0460-0.0471] | 101.8 | 76 / 75 | noise |
| sh_gu | Q8 | dense-decode | 4 | bf16 | 0.0471 [0.0467-0.0478] | 0.0496 [0.0485-0.0499] | 105.3 | 74 / 70 | slower |
| sh_gu | Q8 | dense-decode | 5 | bf16 | 0.0496 [0.0490-0.0499] | 0.0536 [0.0530-0.0539] | 108.2 | 70 / 65 | slower |
| sh_gu | Q8 | dense-decode | 6 | bf16 | 0.0515 [0.0509-0.0517] | 0.0561 [0.0558-0.0569] | 109.0 | 68 / 62 | slower |
| sh_gu | Q8 | dense-decode | 7 | bf16 | 0.0525 [0.0521-0.0532] | 0.0587 [0.0584-0.0592] | 111.8 | 66 / 59 | slower |
| sh_gu | Q8 | dense-decode | 8 | bf16 | 0.0546 [0.0539-0.0547] | 0.0618 [0.0612-0.0622] | 113.3 | 64 / 56 | slower |
| sh_gu | Q8 | dense-prefill-wmma | 512 | bf16 | 0.1507 [0.1461-0.1532] | 0.1344 [0.1276-0.1407] | 89.2 | 23 / 26 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.9184 [0.4654-13.5994] | 0.4388 [0.4301-4.8926] | 11.2 | 1 / 8 | noise |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | bf16 | 1.5646 [1.5555-1.5849] | 1.5067 [1.4899-1.5213] | 96.3 | 2 / 2 | faster |
| sh_gu | Q8 | dense-decode | 1 | f32 | 0.0418 [0.0413-0.0422] | 0.0416 [0.0415-0.0423] | 99.4 | 83 / 84 | noise |
| sh_gu | Q8 | dense-decode | 2 | f32 | 0.0432 [0.0428-0.0436] | 0.0438 [0.0434-0.0444] | 101.5 | 81 / 79 | noise |
| sh_gu | Q8 | dense-decode | 3 | f32 | 0.0457 [0.0449-0.0468] | 0.0475 [0.0471-0.0477] | 103.9 | 76 / 73 | slower |
| sh_gu | Q8 | dense-decode | 4 | f32 | 0.0471 [0.0468-0.0474] | 0.0492 [0.0488-0.0499] | 104.6 | 74 / 71 | slower |
| sh_gu | Q8 | dense-decode | 5 | f32 | 0.0491 [0.0487-0.0494] | 0.0528 [0.0524-0.0532] | 107.6 | 71 / 66 | slower |
| sh_gu | Q8 | dense-decode | 6 | f32 | 0.0507 [0.0502-0.0512] | 0.0557 [0.0552-0.0562] | 109.9 | 69 / 62 | slower |
| sh_gu | Q8 | dense-decode | 7 | f32 | 0.0528 [0.0525-0.0540] | 0.0588 [0.0585-0.0594] | 111.4 | 66 / 59 | slower |
| sh_gu | Q8 | dense-decode | 8 | f32 | 0.0547 [0.0541-0.0563] | 0.0616 [0.0608-0.0621] | 112.6 | 64 / 57 | slower |
| sh_gu | Q8 | dense-prefill-wmma | 512 | f32 | 0.1480 [0.1449-0.1527] | 0.1358 [0.1330-0.1370] | 91.7 | 24 / 26 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | f32 | 0.4688 [0.4584-0.4719] | 0.4247 [0.4170-0.4344] | 90.6 | 7 / 8 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | f32 | 1.5427 [1.5301-1.5569] | 1.4863 [1.4680-1.4979] | 96.3 | 2 / 2 | faster |
| sh_gu | Q4 syn | dense-decode | 1 | bf16 | 0.0259 [0.0256-0.0262] | 0.0253 [0.0251-0.0254] | 97.8 | 71 / 73 | faster |
| sh_gu | Q4 syn | dense-decode | 2 | bf16 | 0.0292 [0.0281-0.0298] | 0.0264 [0.0262-0.0267] | 90.3 | 63 / 70 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | bf16 | 0.0298 [0.0293-0.0308] | 0.0277 [0.0274-0.0280] | 92.9 | 62 / 67 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | bf16 | 0.0312 [0.0306-0.0324] | 0.0286 [0.0284-0.0290] | 91.5 | 59 / 65 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | bf16 | 0.0324 [0.0316-0.0338] | 0.0298 [0.0296-0.0302] | 91.8 | 57 / 62 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | bf16 | 0.0348 [0.0343-0.0352] | 0.0306 [0.0306-0.0310] | 88.0 | 53 / 60 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | bf16 | 0.0348 [0.0342-0.0357] | 0.0318 [0.0313-0.0321] | 91.5 | 53 / 58 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | bf16 | 0.0363 [0.0356-0.0374] | 0.0325 [0.0324-0.0327] | 89.6 | 51 / 57 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.1286 [0.1278-0.1316] | 0.1231 [0.1214-0.1239] | 95.8 | 14 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.4594 [0.4514-0.4692] | 0.4440 [0.4279-0.4546] | 96.6 | 4 / 4 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 1.7078 [1.6791-1.7557] | 1.6848 [1.6403-1.7245] | 98.7 | 1 / 1 | noise |
| sh_gu | Q4 syn | dense-decode | 1 | f32 | 0.0267 [0.0264-0.0270] | 0.0254 [0.0252-0.0258] | 95.2 | 69 / 72 | faster |
| sh_gu | Q4 syn | dense-decode | 2 | f32 | 0.0286 [0.0281-0.0297] | 0.0266 [0.0263-0.0269] | 93.0 | 64 / 69 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | f32 | 0.0309 [0.0294-0.0318] | 0.0282 [0.0277-0.0286] | 91.2 | 60 / 65 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | f32 | 0.0316 [0.0310-0.0328] | 0.0292 [0.0287-0.0296] | 92.4 | 58 / 63 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | f32 | 0.0344 [0.0338-0.0349] | 0.0312 [0.0310-0.0315] | 90.8 | 54 / 59 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | f32 | 0.0388 [0.0372-0.0412] | 0.0338 [0.0332-0.0345] | 87.2 | 48 / 55 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | f32 | 0.0421 [0.0406-0.0441] | 0.0358 [0.0354-0.0366] | 85.1 | 44 / 51 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | f32 | 0.0450 [0.0441-0.0464] | 0.0383 [0.0377-0.0389] | 85.0 | 41 / 48 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.1474 [0.1366-0.1499] | 0.1404 [0.1313-0.1457] | 95.2 | 13 / 13 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.4251 [0.4193-0.4292] | 0.4204 [0.4179-0.4234] | 98.9 | 4 / 4 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | f32 | 1.5963 [1.5869-1.6049] | 1.5747 [1.5707-1.5866] | 98.7 | 1 / 1 | faster |
| sh_down | Q8 | dense-decode | 1 | bf16 | 0.0258 [0.0254-0.0258] | 0.0254 [0.0252-0.0255] | 98.6 | 72 / 73 | noise |
| sh_down | Q8 | dense-decode | 2 | bf16 | 0.0280 [0.0279-0.0286] | 0.0264 [0.0259-0.0268] | 94.0 | 66 / 70 | faster |
| sh_down | Q8 | dense-decode | 3 | bf16 | 0.0306 [0.0302-0.0310] | 0.0271 [0.0269-0.0272] | 88.5 | 60 / 68 | faster |
| sh_down | Q8 | dense-decode | 4 | bf16 | 0.0322 [0.0314-0.0325] | 0.0284 [0.0278-0.0287] | 88.3 | 57 / 65 | faster |
| sh_down | Q8 | dense-decode | 5 | bf16 | 0.0352 [0.0346-0.0353] | 0.0293 [0.0289-0.0296] | 83.3 | 52 / 63 | faster |
| sh_down | Q8 | dense-decode | 6 | bf16 | 0.0371 [0.0365-0.0381] | 0.0307 [0.0304-0.0309] | 82.7 | 50 / 60 | faster |
| sh_down | Q8 | dense-decode | 7 | bf16 | 0.0382 [0.0377-0.0384] | 0.0312 [0.0308-0.0316] | 81.7 | 48 / 59 | faster |
| sh_down | Q8 | dense-decode | 8 | bf16 | 0.0392 [0.0388-0.0397] | 0.0321 [0.0315-0.0325] | 82.0 | 47 / 57 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | bf16 | 0.0764 [0.0763-0.0765] | 0.0686 [0.0684-0.0688] | 89.8 | 24 / 27 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.2457 [0.2345-0.2494] | 0.2273 [0.2201-0.2304] | 92.5 | 8 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | bf16 | 0.9064 [0.8934-0.9126] | 0.8677 [0.8637-0.8749] | 95.7 | 2 / 2 | faster |
| sh_down | Q8 | dense-decode | 1 | f32 | 0.0254 [0.0249-0.0257] | 0.0253 [0.0250-0.0255] | 99.5 | 73 / 73 | noise |
| sh_down | Q8 | dense-decode | 2 | f32 | 0.0272 [0.0270-0.0274] | 0.0262 [0.0258-0.0263] | 96.5 | 68 / 70 | faster |
| sh_down | Q8 | dense-decode | 3 | f32 | 0.0286 [0.0283-0.0290] | 0.0271 [0.0268-0.0272] | 94.7 | 64 / 68 | faster |
| sh_down | Q8 | dense-decode | 4 | f32 | 0.0304 [0.0300-0.0308] | 0.0275 [0.0272-0.0280] | 90.5 | 61 / 67 | faster |
| sh_down | Q8 | dense-decode | 5 | f32 | 0.0334 [0.0328-0.0342] | 0.0291 [0.0289-0.0292] | 87.1 | 55 / 63 | faster |
| sh_down | Q8 | dense-decode | 6 | f32 | 0.0347 [0.0340-0.0362] | 0.0300 [0.0295-0.0304] | 86.5 | 53 / 61 | faster |
| sh_down | Q8 | dense-decode | 7 | f32 | 0.0372 [0.0367-0.0376] | 0.0308 [0.0304-0.0312] | 82.6 | 49 / 60 | faster |
| sh_down | Q8 | dense-decode | 8 | f32 | 0.0393 [0.0388-0.0395] | 0.0325 [0.0322-0.0326] | 82.8 | 47 / 57 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | f32 | 0.0841 [0.0828-0.0849] | 0.0766 [0.0763-0.0768] | 91.1 | 22 / 24 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | f32 | 0.2470 [0.2437-0.2525] | 0.2341 [0.2313-0.2369] | 94.8 | 7 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | f32 | 0.9235 [0.8861-0.9336] | 0.8792 [0.8536-0.8972] | 95.2 | 2 / 2 | noise |
| sh_down | Q4 syn | dense-decode | 1 | bf16 | 0.0170 [0.0168-0.0173] | 0.0173 [0.0171-0.0174] | 101.9 | 60 / 59 | noise |
| sh_down | Q4 syn | dense-decode | 2 | bf16 | 0.0208 [0.0205-0.0209] | 0.0189 [0.0186-0.0191] | 90.9 | 49 / 54 | faster |
| sh_down | Q4 syn | dense-decode | 3 | bf16 | 0.0224 [0.0223-0.0226] | 0.0193 [0.0192-0.0194] | 86.4 | 46 / 53 | faster |
| sh_down | Q4 syn | dense-decode | 4 | bf16 | 0.0234 [0.0230-0.0237] | 0.0196 [0.0194-0.0198] | 84.1 | 44 / 52 | faster |
| sh_down | Q4 syn | dense-decode | 5 | bf16 | 0.0270 [0.0266-0.0272] | 0.0209 [0.0206-0.0212] | 77.5 | 38 / 49 | faster |
| sh_down | Q4 syn | dense-decode | 6 | bf16 | 0.0294 [0.0289-0.0299] | 0.0221 [0.0217-0.0225] | 75.1 | 35 / 46 | faster |
| sh_down | Q4 syn | dense-decode | 7 | bf16 | 0.0303 [0.0297-0.0309] | 0.0226 [0.0224-0.0230] | 74.5 | 34 / 45 | faster |
| sh_down | Q4 syn | dense-decode | 8 | bf16 | 0.0316 [0.0313-0.0318] | 0.0234 [0.0231-0.0238] | 74.2 | 32 / 44 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.0764 [0.0762-0.0765] | 0.0671 [0.0670-0.0672] | 87.9 | 13 / 15 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.2395 [0.2327-0.2460] | 0.2250 [0.2189-0.2280] | 94.0 | 4 / 5 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 0.9006 [0.8862-0.9134] | 0.8672 [0.8565-0.8939] | 96.3 | 1 / 1 | noise |
| sh_down | Q4 syn | dense-decode | 1 | f32 | 0.0164 [0.0163-0.0166] | 0.0168 [0.0167-0.0170] | 102.4 | 62 / 61 | slower |
| sh_down | Q4 syn | dense-decode | 2 | f32 | 0.0185 [0.0184-0.0188] | 0.0178 [0.0175-0.0179] | 95.9 | 55 / 58 | faster |
| sh_down | Q4 syn | dense-decode | 3 | f32 | 0.0197 [0.0194-0.0199] | 0.0188 [0.0186-0.0192] | 95.5 | 52 / 54 | faster |
| sh_down | Q4 syn | dense-decode | 4 | f32 | 0.0217 [0.0214-0.0222] | 0.0193 [0.0191-0.0195] | 88.9 | 47 / 53 | faster |
| sh_down | Q4 syn | dense-decode | 5 | f32 | 0.0254 [0.0251-0.0256] | 0.0206 [0.0205-0.0208] | 81.1 | 40 / 50 | faster |
| sh_down | Q4 syn | dense-decode | 6 | f32 | 0.0285 [0.0275-0.0286] | 0.0217 [0.0214-0.0220] | 76.1 | 36 / 47 | faster |
| sh_down | Q4 syn | dense-decode | 7 | f32 | 0.0295 [0.0293-0.0298] | 0.0226 [0.0224-0.0228] | 76.6 | 35 / 45 | faster |
| sh_down | Q4 syn | dense-decode | 8 | f32 | 0.0304 [0.0301-0.0309] | 0.0230 [0.0229-0.0233] | 75.7 | 34 / 45 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.0823 [0.0819-0.0831] | 0.0762 [0.0761-0.0765] | 92.6 | 12 / 13 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.2576 [0.2530-0.2591] | 0.2348 [0.2332-0.2370] | 91.2 | 4 / 4 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | f32 | 0.8543 [0.8397-0.8662] | 0.8065 [0.7952-0.8228] | 94.4 | 1 / 1 | faster |
| gdn_in | Q4 | dense-decode | 1 | bf16 | 0.2332 [0.2309-0.2361] | 0.2335 [0.2318-0.2343] | 100.1 | 96 / 96 | noise |
| gdn_in | Q4 | dense-decode | 2 | bf16 | 0.2365 [0.2339-0.2378] | 0.2348 [0.2306-0.2358] | 99.3 | 95 / 95 | noise |
| gdn_in | Q4 | dense-decode | 3 | bf16 | 0.2386 [0.2366-0.2417] | 0.2360 [0.2348-0.2385] | 98.9 | 94 / 95 | noise |
| gdn_in | Q4 | dense-decode | 4 | bf16 | 0.2432 [0.2418-0.2454] | 0.2372 [0.2357-0.2399] | 97.5 | 92 / 94 | faster |
| gdn_in | Q4 | dense-decode | 5 | bf16 | 0.2546 [0.2528-0.2579] | 0.2429 [0.2391-0.2451] | 95.4 | 88 / 92 | faster |
| gdn_in | Q4 | dense-decode | 6 | bf16 | 0.2609 [0.2584-0.2649] | 0.2480 [0.2464-0.2547] | 95.1 | 86 / 90 | faster |
| gdn_in | Q4 | dense-decode | 7 | bf16 | 0.2653 [0.2634-0.2664] | 0.2513 [0.2482-0.2539] | 94.7 | 84 / 89 | faster |
| gdn_in | Q4 | dense-decode | 8 | bf16 | 0.2714 [0.2678-0.2731] | 0.2570 [0.2540-0.2596] | 94.7 | 83 / 87 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | bf16 | 1.2377 [1.2266-1.2570] | 1.2008 [1.1901-1.2081] | 97.0 | 18 / 19 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | bf16 | 5.0562 [4.9965-5.0663] | 5.0081 [4.9633-5.0473] | 99.0 | 4 / 4 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | bf16 | 21.3993 [19.9838-22.4944] | 20.6306 [20.1242-25.3764] | 96.4 | 1 / 1 | noise |
| gdn_in | Q4 | dense-decode | 1 | f32 | 0.2364 [0.2312-0.2377] | 0.2337 [0.2308-0.2351] | 98.8 | 95 / 96 | noise |
| gdn_in | Q4 | dense-decode | 2 | f32 | 0.2368 [0.2353-0.2390] | 0.2336 [0.2314-0.2361] | 98.6 | 95 / 96 | noise |
| gdn_in | Q4 | dense-decode | 3 | f32 | 0.2394 [0.2381-0.2418] | 0.2364 [0.2338-0.2399] | 98.7 | 94 / 95 | noise |
| gdn_in | Q4 | dense-decode | 4 | f32 | 0.2441 [0.2394-0.2467] | 0.2392 [0.2352-0.2422] | 98.0 | 92 / 94 | noise |
| gdn_in | Q4 | dense-decode | 5 | f32 | 0.2532 [0.2509-0.2558] | 0.2464 [0.2426-0.2474] | 97.3 | 89 / 91 | faster |
| gdn_in | Q4 | dense-decode | 6 | f32 | 0.2566 [0.2551-0.2588] | 0.2489 [0.2450-0.2519] | 97.0 | 87 / 90 | faster |
| gdn_in | Q4 | dense-decode | 7 | f32 | 0.2623 [0.2599-0.2641] | 0.2497 [0.2470-0.2528] | 95.2 | 85 / 90 | faster |
| gdn_in | Q4 | dense-decode | 8 | f32 | 0.2677 [0.2652-0.2700] | 0.2543 [0.2511-0.2564] | 95.0 | 84 / 88 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | f32 | 1.1795 [1.1735-1.1874] | 1.1441 [1.1407-1.1455] | 97.0 | 19 / 20 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | f32 | 4.8752 [4.8303-5.2061] | 4.8337 [4.7898-5.1255] | 99.2 | 5 / 5 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | f32 | 21.2662 [19.5595-21.5589] | 21.0113 [19.3622-21.4326] | 98.8 | 1 / 1 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | bf16 | 0.3905 [0.3888-0.3935] | 0.3950 [0.3921-0.3964] | 101.2 | 111 / 110 | noise |
| gdn_in | Q8 syn | dense-decode | 2 | bf16 | 0.3910 [0.3882-0.3921] | 0.3972 [0.3962-0.3993] | 101.6 | 111 / 110 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | bf16 | 0.4020 [0.3998-0.4034] | 0.4037 [0.4012-0.4051] | 100.4 | 108 / 108 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | bf16 | 0.4163 [0.4141-0.4185] | 0.4147 [0.4100-0.4161] | 99.6 | 105 / 105 | noise |
| gdn_in | Q8 syn | dense-decode | 5 | bf16 | 0.4342 [0.4324-0.4367] | 0.4234 [0.4196-0.4251] | 97.5 | 100 / 103 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | bf16 | 0.4423 [0.4386-0.4446] | 0.4275 [0.4237-0.4320] | 96.7 | 98 / 102 | faster |
| gdn_in | Q8 syn | dense-decode | 7 | bf16 | 0.4430 [0.4408-0.4476] | 0.4307 [0.4271-0.4333] | 97.2 | 98 / 101 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | bf16 | 0.4499 [0.4439-0.4529] | 0.4422 [0.4381-0.4454] | 98.3 | 97 / 98 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | bf16 | 1.6277 [1.5993-1.7089] | 1.4179 [1.3579-1.4713] | 87.1 | 27 / 31 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | bf16 | 5.6729 [5.6127-5.7592] | 5.4759 [5.4135-5.5008] | 96.5 | 8 / 8 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | bf16 | 23.4227 [20.9636-24.2311] | 22.8435 [20.5340-23.7777] | 97.5 | 2 / 2 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | f32 | 0.3895 [0.3867-0.3917] | 0.3944 [0.3910-0.3956] | 101.2 | 112 / 110 | noise |
| gdn_in | Q8 syn | dense-decode | 2 | f32 | 0.3896 [0.3872-0.3922] | 0.3952 [0.3936-0.3970] | 101.4 | 112 / 110 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | f32 | 0.4022 [0.3982-0.4036] | 0.4020 [0.4003-0.4031] | 99.9 | 108 / 108 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | f32 | 0.4076 [0.4060-0.4106] | 0.4052 [0.4038-0.4075] | 99.4 | 107 / 107 | noise |
| gdn_in | Q8 syn | dense-decode | 5 | f32 | 0.4274 [0.4248-0.4315] | 0.4166 [0.4152-0.4198] | 97.5 | 102 / 104 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | f32 | 0.4359 [0.4340-0.4392] | 0.4237 [0.4220-0.4276] | 97.2 | 100 / 103 | faster |
| gdn_in | Q8 syn | dense-decode | 7 | f32 | 0.4440 [0.4397-0.4470] | 0.4314 [0.4255-0.4347] | 97.2 | 98 / 101 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | f32 | 0.4519 [0.4469-0.4574] | 0.4406 [0.4373-0.4446] | 97.5 | 96 / 99 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | f32 | 1.5242 [1.5135-1.5651] | 1.3054 [1.3008-1.3134] | 85.6 | 29 / 33 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | f32 | 5.2739 [5.1392-5.7682] | 4.9673 [4.9294-5.5203] | 94.2 | 8 / 9 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | f32 | 21.5139 [20.2998-23.0516] | 20.4707 [20.0295-22.7048] | 95.2 | 2 / 2 | noise |
| gdn_out | Q8 | dense-decode | 1 | bf16 | 0.1811 [0.1787-0.1832] | 0.1832 [0.1808-0.1845] | 101.1 | 92 / 91 | noise |
| gdn_out | Q8 | dense-decode | 2 | bf16 | 0.1830 [0.1812-0.1842] | 0.1841 [0.1819-0.1856] | 100.6 | 91 / 91 | noise |
| gdn_out | Q8 | dense-decode | 3 | bf16 | 0.1850 [0.1836-0.1874] | 0.1888 [0.1877-0.1895] | 102.0 | 90 / 89 | slower |
| gdn_out | Q8 | dense-decode | 4 | bf16 | 0.1922 [0.1910-0.1947] | 0.1935 [0.1924-0.1949] | 100.7 | 87 / 86 | noise |
| gdn_out | Q8 | dense-decode | 5 | bf16 | 0.2018 [0.1984-0.2089] | 0.2054 [0.2046-0.2067] | 101.8 | 83 / 81 | noise |
| gdn_out | Q8 | dense-decode | 6 | bf16 | 0.2141 [0.2085-0.2175] | 0.2085 [0.2076-0.2100] | 97.4 | 78 / 80 | noise |
| gdn_out | Q8 | dense-decode | 7 | bf16 | 0.2220 [0.2169-0.2252] | 0.2121 [0.2107-0.2135] | 95.6 | 75 / 79 | faster |
| gdn_out | Q8 | dense-decode | 8 | bf16 | 0.2314 [0.2261-0.2348] | 0.2168 [0.2156-0.2185] | 93.7 | 72 / 77 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 512 | bf16 | 1.2804 [1.2394-1.3780] | 0.7642 [0.7300-0.7943] | 59.7 | 13 / 22 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.1007 [3.0716-3.1813] | 2.2477 [2.1865-2.3018] | 72.5 | 5 / 7 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.8970 [8.3975-9.0047] | 7.9949 [7.5670-8.0346] | 89.9 | 2 / 2 | faster |
| gdn_out | Q8 | dense-decode | 1 | f32 | 0.1794 [0.1781-0.1806] | 0.1802 [0.1797-0.1809] | 100.4 | 93 / 93 | noise |
| gdn_out | Q8 | dense-decode | 2 | f32 | 0.1817 [0.1797-0.1831] | 0.1816 [0.1813-0.1827] | 100.0 | 92 / 92 | noise |
| gdn_out | Q8 | dense-decode | 3 | f32 | 0.1861 [0.1838-0.1875] | 0.1870 [0.1859-0.1887] | 100.5 | 90 / 89 | noise |
| gdn_out | Q8 | dense-decode | 4 | f32 | 0.1915 [0.1882-0.1950] | 0.1918 [0.1898-0.1927] | 100.2 | 87 / 87 | noise |
| gdn_out | Q8 | dense-decode | 5 | f32 | 0.1989 [0.1971-0.2024] | 0.2036 [0.2029-0.2048] | 102.3 | 84 / 82 | slower |
| gdn_out | Q8 | dense-decode | 6 | f32 | 0.2114 [0.2085-0.2163] | 0.2082 [0.2078-0.2094] | 98.5 | 79 / 80 | noise |
| gdn_out | Q8 | dense-decode | 7 | f32 | 0.2234 [0.2207-0.2280] | 0.2118 [0.2102-0.2128] | 94.8 | 75 / 79 | faster |
| gdn_out | Q8 | dense-decode | 8 | f32 | 0.2309 [0.2272-0.2347] | 0.2158 [0.2151-0.2165] | 93.4 | 72 / 77 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 512 | f32 | 1.3135 [1.1816-1.3951] | 0.7752 [0.7539-0.8183] | 59.0 | 13 / 22 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | f32 | 3.0484 [3.0257-3.0788] | 2.0498 [2.0231-2.0720] | 67.2 | 5 / 8 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | f32 | 9.1577 [8.5207-9.3285] | 8.3343 [7.5111-8.4360] | 91.0 | 2 / 2 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | bf16 | 0.1012 [0.1002-0.1020] | 0.1014 [0.1008-0.1025] | 100.2 | 87 / 87 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | bf16 | 0.1043 [0.1021-0.1050] | 0.1028 [0.1025-0.1037] | 98.5 | 85 / 86 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | bf16 | 0.1079 [0.1068-0.1086] | 0.1058 [0.1053-0.1069] | 98.1 | 82 / 84 | noise |
| gdn_out | Q4 syn | dense-decode | 4 | bf16 | 0.1120 [0.1112-0.1130] | 0.1086 [0.1078-0.1095] | 97.0 | 79 / 81 | faster |
| gdn_out | Q4 syn | dense-decode | 5 | bf16 | 0.1181 [0.1172-0.1190] | 0.1168 [0.1160-0.1176] | 98.9 | 75 / 76 | noise |
| gdn_out | Q4 syn | dense-decode | 6 | bf16 | 0.1222 [0.1216-0.1233] | 0.1202 [0.1191-0.1214] | 98.3 | 72 / 74 | faster |
| gdn_out | Q4 syn | dense-decode | 7 | bf16 | 0.1279 [0.1271-0.1287] | 0.1228 [0.1220-0.1239] | 96.0 | 69 / 72 | faster |
| gdn_out | Q4 syn | dense-decode | 8 | bf16 | 0.1318 [0.1309-0.1337] | 0.1255 [0.1240-0.1267] | 95.3 | 67 / 70 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0145 [0.9920-1.0609] | 0.6241 [0.6058-0.6771] | 61.5 | 9 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.5157 [2.4754-2.5939] | 1.9610 [1.9290-2.0376] | 78.0 | 4 / 5 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 8.0548 [8.0011-8.1287] | 7.5499 [7.4403-7.6719] | 93.7 | 1 / 1 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | f32 | 0.1018 [0.1008-0.1023] | 0.1020 [0.1010-0.1028] | 100.2 | 87 / 87 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | f32 | 0.1042 [0.1033-0.1052] | 0.1036 [0.1026-0.1043] | 99.4 | 85 / 85 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | f32 | 0.1076 [0.1071-0.1085] | 0.1060 [0.1055-0.1071] | 98.5 | 82 / 84 | noise |
| gdn_out | Q4 syn | dense-decode | 4 | f32 | 0.1116 [0.1101-0.1122] | 0.1083 [0.1074-0.1091] | 97.1 | 79 / 82 | faster |
| gdn_out | Q4 syn | dense-decode | 5 | f32 | 0.1174 [0.1164-0.1184] | 0.1163 [0.1153-0.1173] | 99.1 | 75 / 76 | noise |
| gdn_out | Q4 syn | dense-decode | 6 | f32 | 0.1217 [0.1210-0.1235] | 0.1193 [0.1182-0.1201] | 98.0 | 73 / 74 | faster |
| gdn_out | Q4 syn | dense-decode | 7 | f32 | 0.1261 [0.1246-0.1277] | 0.1226 [0.1212-0.1229] | 97.2 | 70 / 72 | faster |
| gdn_out | Q4 syn | dense-decode | 8 | f32 | 0.1312 [0.1299-0.1319] | 0.1252 [0.1242-0.1263] | 95.5 | 67 / 71 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0003 [0.9861-1.0453] | 0.6197 [0.6034-0.6491] | 62.0 | 9 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.4734 [2.4274-2.4889] | 1.9130 [1.9091-1.9457] | 77.3 | 4 / 5 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | f32 | 7.9787 [7.8807-8.1010] | 7.4709 [7.4046-7.5376] | 93.6 | 1 / 1 | faster |
| attn_qkv | Q8 | dense-decode | 1 | bf16 | 0.3694 [0.3683-0.3701] | 0.3699 [0.3686-0.3712] | 100.1 | 103 / 103 | noise |
| attn_qkv | Q8 | dense-decode | 2 | bf16 | 0.3696 [0.3681-0.3722] | 0.3733 [0.3716-0.3754] | 101.0 | 103 / 102 | noise |
| attn_qkv | Q8 | dense-decode | 3 | bf16 | 0.3768 [0.3740-0.3784] | 0.3782 [0.3763-0.3796] | 100.4 | 101 / 100 | noise |
| attn_qkv | Q8 | dense-decode | 4 | bf16 | 0.3827 [0.3804-0.3834] | 0.3815 [0.3805-0.3843] | 99.7 | 99 / 99 | noise |
| attn_qkv | Q8 | dense-decode | 5 | bf16 | 0.3981 [0.3959-0.4002] | 0.3928 [0.3909-0.3967] | 98.7 | 95 / 97 | noise |
| attn_qkv | Q8 | dense-decode | 6 | bf16 | 0.4049 [0.4025-0.4066] | 0.3991 [0.3948-0.4060] | 98.6 | 94 / 95 | noise |
| attn_qkv | Q8 | dense-decode | 7 | bf16 | 0.4122 [0.4066-0.4144] | 0.4093 [0.4065-0.4115] | 99.3 | 92 / 93 | noise |
| attn_qkv | Q8 | dense-decode | 8 | bf16 | 0.4196 [0.4158-0.4219] | 0.4183 [0.4128-0.4217] | 99.7 | 90 / 91 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3648 [1.3457-1.3777] | 1.1363 [1.1299-1.1475] | 83.3 | 28 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.5376 [4.4922-4.5705] | 4.2623 [4.2271-4.2831] | 93.9 | 8 / 9 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | bf16 | 18.4995 [17.9537-18.7298] | 17.4199 [17.2291-18.2493] | 94.2 | 2 / 2 | noise |
| attn_qkv | Q8 | dense-decode | 1 | f32 | 0.3674 [0.3657-0.3690] | 0.3702 [0.3685-0.3709] | 100.8 | 103 / 103 | noise |
| attn_qkv | Q8 | dense-decode | 2 | f32 | 0.3672 [0.3648-0.3689] | 0.3709 [0.3694-0.3730] | 101.0 | 103 / 102 | slower |
| attn_qkv | Q8 | dense-decode | 3 | f32 | 0.3758 [0.3739-0.3775] | 0.3765 [0.3753-0.3794] | 100.2 | 101 / 101 | noise |
| attn_qkv | Q8 | dense-decode | 4 | f32 | 0.3852 [0.3813-0.3885] | 0.3860 [0.3823-0.3874] | 100.2 | 99 / 98 | noise |
| attn_qkv | Q8 | dense-decode | 5 | f32 | 0.4007 [0.4002-0.4032] | 0.3976 [0.3956-0.4035] | 99.2 | 95 / 95 | noise |
| attn_qkv | Q8 | dense-decode | 6 | f32 | 0.4077 [0.4045-0.4089] | 0.3985 [0.3964-0.4016] | 97.7 | 93 / 95 | faster |
| attn_qkv | Q8 | dense-decode | 7 | f32 | 0.4121 [0.4091-0.4146] | 0.4085 [0.4036-0.4099] | 99.1 | 92 / 93 | noise |
| attn_qkv | Q8 | dense-decode | 8 | f32 | 0.4180 [0.4153-0.4202] | 0.4121 [0.4098-0.4140] | 98.6 | 91 / 92 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | f32 | 1.3750 [1.3609-1.3859] | 1.1374 [1.1280-1.1496] | 82.7 | 28 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | f32 | 4.4409 [4.4044-4.4494] | 4.1538 [4.1311-4.1816] | 93.5 | 9 / 9 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | f32 | 17.6888 [17.5784-18.1244] | 17.1096 [17.0089-17.8640] | 96.7 | 2 / 2 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | bf16 | 0.2143 [0.2126-0.2156] | 0.2136 [0.2106-0.2149] | 99.7 | 94 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | bf16 | 0.2163 [0.2148-0.2178] | 0.2149 [0.2094-0.2170] | 99.3 | 93 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 3 | bf16 | 0.2182 [0.2159-0.2207] | 0.2168 [0.2140-0.2187] | 99.4 | 92 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 4 | bf16 | 0.2217 [0.2197-0.2238] | 0.2163 [0.2129-0.2179] | 97.5 | 91 / 93 | faster |
| attn_qkv | Q4 syn | dense-decode | 5 | bf16 | 0.2314 [0.2295-0.2329] | 0.2210 [0.2159-0.2231] | 95.5 | 87 / 91 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | bf16 | 0.2355 [0.2326-0.2376] | 0.2227 [0.2207-0.2232] | 94.6 | 85 / 90 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | bf16 | 0.2428 [0.2416-0.2443] | 0.2275 [0.2270-0.2290] | 93.7 | 83 / 88 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | bf16 | 0.2484 [0.2449-0.2498] | 0.2298 [0.2268-0.2317] | 92.5 | 81 / 87 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0991 [1.0886-1.1065] | 1.0423 [1.0282-1.0561] | 94.8 | 18 / 19 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 4.2129 [4.1806-4.4257] | 4.1515 [4.1084-4.3507] | 98.5 | 5 / 5 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 17.0791 [17.0066-17.2382] | 16.7686 [16.7211-17.1106] | 98.2 | 1 / 1 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | f32 | 0.2104 [0.2086-0.2160] | 0.2138 [0.2120-0.2156] | 101.6 | 95 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | f32 | 0.2172 [0.2149-0.2189] | 0.2142 [0.2120-0.2160] | 98.6 | 92 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 3 | f32 | 0.2182 [0.2151-0.2209] | 0.2161 [0.2116-0.2184] | 99.0 | 92 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 4 | f32 | 0.2195 [0.2180-0.2228] | 0.2181 [0.2147-0.2192] | 99.4 | 92 / 92 | noise |
| attn_qkv | Q4 syn | dense-decode | 5 | f32 | 0.2282 [0.2263-0.2307] | 0.2196 [0.2162-0.2206] | 96.2 | 88 / 92 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | f32 | 0.2326 [0.2288-0.2365] | 0.2223 [0.2214-0.2265] | 95.6 | 86 / 90 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | f32 | 0.2377 [0.2353-0.2403] | 0.2250 [0.2238-0.2267] | 94.7 | 85 / 89 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | f32 | 0.2447 [0.2417-0.2455] | 0.2277 [0.2257-0.2301] | 93.0 | 82 / 88 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0438 [1.0378-1.0587] | 1.0001 [0.9957-1.0077] | 95.8 | 19 / 20 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 4.1212 [4.1093-4.1588] | 4.0484 [4.0315-4.0794] | 98.2 | 5 / 5 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 18.4556 [16.9484-19.1014] | 18.0330 [16.5980-18.4683] | 97.7 | 1 / 1 | noise |
| attn_o | Q8 | dense-decode | 1 | bf16 | 0.1805 [0.1796-0.1811] | 0.1808 [0.1802-0.1812] | 100.2 | 93 / 92 | noise |
| attn_o | Q8 | dense-decode | 2 | bf16 | 0.1830 [0.1816-0.1840] | 0.1823 [0.1816-0.1828] | 99.7 | 91 / 92 | noise |
| attn_o | Q8 | dense-decode | 3 | bf16 | 0.1871 [0.1856-0.1881] | 0.1887 [0.1872-0.1894] | 100.8 | 89 / 89 | noise |
| attn_o | Q8 | dense-decode | 4 | bf16 | 0.1922 [0.1891-0.1954] | 0.1934 [0.1922-0.1946] | 100.7 | 87 / 86 | noise |
| attn_o | Q8 | dense-decode | 5 | bf16 | 0.1996 [0.1982-0.2040] | 0.2062 [0.2047-0.2072] | 103.3 | 84 / 81 | slower |
| attn_o | Q8 | dense-decode | 6 | bf16 | 0.2117 [0.2078-0.2160] | 0.2106 [0.2093-0.2116] | 99.5 | 79 / 79 | noise |
| attn_o | Q8 | dense-decode | 7 | bf16 | 0.2217 [0.2179-0.2246] | 0.2142 [0.2136-0.2158] | 96.6 | 75 / 78 | faster |
| attn_o | Q8 | dense-decode | 8 | bf16 | 0.2312 [0.2252-0.2336] | 0.2183 [0.2173-0.2194] | 94.4 | 72 / 77 | faster |
| attn_o | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3539 [1.2638-1.4073] | 0.7836 [0.7647-0.8357] | 57.9 | 12 / 21 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.0908 [3.0530-3.1343] | 2.1399 [2.1071-2.1743] | 69.2 | 5 / 8 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.4525 [8.4183-8.5729] | 7.5849 [7.5406-7.6708] | 89.7 | 2 / 2 | faster |
| attn_o | Q8 | dense-decode | 1 | f32 | 0.1813 [0.1803-0.1826] | 0.1815 [0.1811-0.1822] | 100.1 | 92 / 92 | noise |
| attn_o | Q8 | dense-decode | 2 | f32 | 0.1838 [0.1826-0.1849] | 0.1827 [0.1814-0.1834] | 99.4 | 91 / 91 | noise |
| attn_o | Q8 | dense-decode | 3 | f32 | 0.1873 [0.1860-0.1894] | 0.1880 [0.1870-0.1908] | 100.4 | 89 / 89 | noise |
| attn_o | Q8 | dense-decode | 4 | f32 | 0.1913 [0.1901-0.1948] | 0.1943 [0.1936-0.1958] | 101.6 | 87 / 86 | noise |
| attn_o | Q8 | dense-decode | 5 | f32 | 0.1988 [0.1962-0.2030] | 0.2060 [0.2050-0.2064] | 103.6 | 84 / 81 | slower |
| attn_o | Q8 | dense-decode | 6 | f32 | 0.2122 [0.2080-0.2181] | 0.2096 [0.2090-0.2102] | 98.8 | 79 / 80 | noise |
| attn_o | Q8 | dense-decode | 7 | f32 | 0.2209 [0.2171-0.2261] | 0.2132 [0.2125-0.2148] | 96.5 | 76 / 78 | faster |
| attn_o | Q8 | dense-decode | 8 | f32 | 0.2272 [0.2242-0.2303] | 0.2159 [0.2152-0.2172] | 95.0 | 74 / 77 | faster |
| attn_o | Q8 | dense-prefill-wmma | 512 | f32 | 1.3559 [1.2018-1.4690] | 0.7591 [0.7239-0.7753] | 56.0 | 12 / 22 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | f32 | 3.0594 [2.9733-3.0786] | 2.0522 [2.0176-2.0803] | 67.1 | 5 / 8 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | f32 | 8.4181 [8.3182-8.4880] | 7.4838 [7.4520-7.5869] | 88.9 | 2 / 2 | faster |
| attn_o | Q4 syn | dense-decode | 1 | bf16 | 0.1008 [0.0996-0.1017] | 0.1011 [0.1000-0.1024] | 100.3 | 88 / 88 | noise |
| attn_o | Q4 syn | dense-decode | 2 | bf16 | 0.1042 [0.1026-0.1048] | 0.1022 [0.1015-0.1034] | 98.1 | 85 / 87 | noise |
| attn_o | Q4 syn | dense-decode | 3 | bf16 | 0.1077 [0.1064-0.1086] | 0.1060 [0.1050-0.1068] | 98.5 | 82 / 83 | noise |
| attn_o | Q4 syn | dense-decode | 4 | bf16 | 0.1117 [0.1103-0.1124] | 0.1091 [0.1083-0.1098] | 97.7 | 79 / 81 | faster |
| attn_o | Q4 syn | dense-decode | 5 | bf16 | 0.1178 [0.1168-0.1186] | 0.1168 [0.1155-0.1187] | 99.1 | 75 / 76 | noise |
| attn_o | Q4 syn | dense-decode | 6 | bf16 | 0.1234 [0.1224-0.1238] | 0.1197 [0.1190-0.1214] | 97.0 | 72 / 74 | faster |
| attn_o | Q4 syn | dense-decode | 7 | bf16 | 0.1279 [0.1262-0.1284] | 0.1230 [0.1216-0.1235] | 96.2 | 69 / 72 | faster |
| attn_o | Q4 syn | dense-decode | 8 | bf16 | 0.1314 [0.1306-0.1326] | 0.1261 [0.1245-0.1275] | 96.0 | 67 / 70 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0147 [0.9919-1.0507] | 0.6301 [0.6027-0.6520] | 62.1 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.5772 [2.5245-2.6172] | 2.0236 [1.9708-2.1610] | 78.5 | 3 / 4 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 8.1420 [8.0660-8.4021] | 7.5941 [7.4920-7.7386] | 93.3 | 1 / 1 | faster |
| attn_o | Q4 syn | dense-decode | 1 | f32 | 0.1018 [0.1010-0.1027] | 0.1025 [0.1017-0.1032] | 100.6 | 87 / 86 | noise |
| attn_o | Q4 syn | dense-decode | 2 | f32 | 0.1047 [0.1039-0.1055] | 0.1042 [0.1021-0.1051] | 99.5 | 84 / 85 | noise |
| attn_o | Q4 syn | dense-decode | 3 | f32 | 0.1069 [0.1061-0.1084] | 0.1060 [0.1054-0.1072] | 99.1 | 83 / 84 | noise |
| attn_o | Q4 syn | dense-decode | 4 | f32 | 0.1115 [0.1103-0.1120] | 0.1092 [0.1082-0.1097] | 98.0 | 79 / 81 | faster |
| attn_o | Q4 syn | dense-decode | 5 | f32 | 0.1176 [0.1166-0.1184] | 0.1157 [0.1153-0.1173] | 98.4 | 75 / 76 | noise |
| attn_o | Q4 syn | dense-decode | 6 | f32 | 0.1225 [0.1209-0.1231] | 0.1175 [0.1168-0.1195] | 95.9 | 72 / 75 | faster |
| attn_o | Q4 syn | dense-decode | 7 | f32 | 0.1260 [0.1250-0.1276] | 0.1218 [0.1206-0.1227] | 96.6 | 70 / 73 | faster |
| attn_o | Q4 syn | dense-decode | 8 | f32 | 0.1310 [0.1295-0.1318] | 0.1246 [0.1242-0.1252] | 95.1 | 68 / 71 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0187 [1.0032-1.0448] | 0.6300 [0.6005-0.6442] | 61.8 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.5270 [2.4819-2.5732] | 1.9229 [1.9164-1.9559] | 76.1 | 4 / 5 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | f32 | 8.0223 [7.9223-8.1317] | 7.5476 [7.4044-7.6212] | 94.1 | 1 / 1 | faster |
| ple_kv | Q8 | dense-decode | 1 | bf16 | 0.3533 [0.3525-0.3552] | 0.3556 [0.3538-0.3562] | 100.7 | 99 / 98 | noise |
| ple_kv | Q8 | dense-decode | 2 | bf16 | 0.3520 [0.3502-0.3555] | 0.3576 [0.3556-0.3590] | 101.6 | 99 / 97 | slower |
| ple_kv | Q8 | dense-decode | 3 | bf16 | 0.3604 [0.3581-0.3617] | 0.3614 [0.3607-0.3635] | 100.3 | 97 / 96 | noise |
| ple_kv | Q8 | dense-decode | 4 | bf16 | 0.3657 [0.3619-0.3676] | 0.3670 [0.3660-0.3695] | 100.4 | 95 / 95 | noise |
| ple_kv | Q8 | dense-decode | 5 | bf16 | 0.3821 [0.3804-0.3840] | 0.3802 [0.3779-0.3840] | 99.5 | 91 / 92 | noise |
| ple_kv | Q8 | dense-decode | 6 | bf16 | 0.3886 [0.3866-0.3937] | 0.3888 [0.3868-0.3922] | 100.0 | 90 / 90 | noise |
| ple_kv | Q8 | dense-decode | 7 | bf16 | 0.3949 [0.3931-0.3981] | 0.4000 [0.3962-0.4026] | 101.3 | 88 / 87 | noise |
| ple_kv | Q8 | dense-decode | 8 | bf16 | 0.3987 [0.3956-0.4006] | 0.4038 [0.4000-0.4095] | 101.3 | 87 / 86 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.2553 [1.2338-1.2717] | 1.0203 [1.0158-1.0246] | 81.3 | 28 / 34 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.1376 [4.0890-4.2047] | 3.8800 [3.8563-3.9155] | 93.8 | 8 / 9 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | bf16 | 17.0378 [16.8797-17.6499] | 16.4194 [16.1249-16.9335] | 96.4 | 2 / 2 | noise |
| ple_kv | Q8 | dense-decode | 1 | f32 | 0.3539 [0.3503-0.3550] | 0.3553 [0.3535-0.3566] | 100.4 | 98 / 98 | noise |
| ple_kv | Q8 | dense-decode | 2 | f32 | 0.3537 [0.3505-0.3557] | 0.3595 [0.3562-0.3605] | 101.6 | 98 / 97 | slower |
| ple_kv | Q8 | dense-decode | 3 | f32 | 0.3586 [0.3561-0.3617] | 0.3636 [0.3619-0.3654] | 101.4 | 97 / 96 | slower |
| ple_kv | Q8 | dense-decode | 4 | f32 | 0.3650 [0.3626-0.3665] | 0.3671 [0.3663-0.3682] | 100.6 | 95 / 95 | noise |
| ple_kv | Q8 | dense-decode | 5 | f32 | 0.3809 [0.3781-0.3825] | 0.3778 [0.3759-0.3799] | 99.2 | 91 / 92 | noise |
| ple_kv | Q8 | dense-decode | 6 | f32 | 0.3870 [0.3842-0.3905] | 0.3862 [0.3818-0.3880] | 99.8 | 90 / 90 | noise |
| ple_kv | Q8 | dense-decode | 7 | f32 | 0.3954 [0.3907-0.3966] | 0.3922 [0.3885-0.3980] | 99.2 | 88 / 89 | noise |
| ple_kv | Q8 | dense-decode | 8 | f32 | 0.4041 [0.4002-0.4061] | 0.4000 [0.3977-0.4050] | 99.0 | 86 / 87 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 512 | f32 | 1.2606 [1.2509-1.2716] | 1.0201 [1.0172-1.0306] | 80.9 | 28 / 34 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | f32 | 4.0762 [4.0627-4.0991] | 3.8230 [3.7975-3.8324] | 93.8 | 9 / 9 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | f32 | 16.1960 [16.0605-16.7108] | 15.5451 [15.5087-16.1679] | 96.0 | 2 / 2 | noise |
| ple_kv | Q4 syn | dense-decode | 1 | bf16 | 0.1975 [0.1964-0.1988] | 0.1966 [0.1936-0.1981] | 99.5 | 93 / 94 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | bf16 | 0.1988 [0.1967-0.2009] | 0.1961 [0.1950-0.1982] | 98.7 | 93 / 94 | noise |
| ple_kv | Q4 syn | dense-decode | 3 | bf16 | 0.2022 [0.2006-0.2044] | 0.1996 [0.1966-0.2009] | 98.7 | 91 / 92 | noise |
| ple_kv | Q4 syn | dense-decode | 4 | bf16 | 0.2055 [0.2038-0.2062] | 0.1999 [0.1948-0.2017] | 97.3 | 90 / 92 | faster |
| ple_kv | Q4 syn | dense-decode | 5 | bf16 | 0.2138 [0.2113-0.2163] | 0.2047 [0.2031-0.2064] | 95.7 | 86 / 90 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | bf16 | 0.2186 [0.2175-0.2200] | 0.2059 [0.2039-0.2086] | 94.2 | 84 / 90 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | bf16 | 0.2212 [0.2204-0.2234] | 0.2088 [0.2049-0.2121] | 94.4 | 83 / 88 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | bf16 | 0.2291 [0.2253-0.2301] | 0.2124 [0.2094-0.2141] | 92.7 | 80 / 87 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.9636 [0.9558-0.9723] | 0.9012 [0.8972-0.9095] | 93.5 | 19 / 20 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 3.8524 [3.8253-3.8820] | 3.7912 [3.7654-3.8142] | 98.4 | 5 / 5 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 15.9050 [15.8239-16.2050] | 15.6002 [15.5387-15.6882] | 98.1 | 1 / 1 | faster |
| ple_kv | Q4 syn | dense-decode | 1 | f32 | 0.1984 [0.1970-0.1991] | 0.1965 [0.1942-0.1988] | 99.0 | 93 / 94 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | f32 | 0.2004 [0.1986-0.2008] | 0.1982 [0.1951-0.2001] | 98.9 | 92 / 93 | noise |
| ple_kv | Q4 syn | dense-decode | 3 | f32 | 0.2027 [0.2022-0.2046] | 0.1998 [0.1972-0.2022] | 98.6 | 91 / 92 | faster |
| ple_kv | Q4 syn | dense-decode | 4 | f32 | 0.2080 [0.2063-0.2103] | 0.2043 [0.2010-0.2064] | 98.2 | 89 / 90 | noise |
| ple_kv | Q4 syn | dense-decode | 5 | f32 | 0.2167 [0.2157-0.2185] | 0.2101 [0.2074-0.2122] | 96.9 | 85 / 88 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | f32 | 0.2180 [0.2168-0.2197] | 0.2065 [0.2042-0.2082] | 94.7 | 85 / 89 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | f32 | 0.2219 [0.2190-0.2238] | 0.2100 [0.2065-0.2116] | 94.6 | 83 / 88 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | f32 | 0.2256 [0.2252-0.2270] | 0.2118 [0.2099-0.2144] | 93.9 | 82 / 87 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.9504 [0.9441-0.9608] | 0.8988 [0.8949-0.9093] | 94.6 | 19 / 21 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 3.7480 [3.7345-3.8009] | 3.6776 [3.6570-3.7059] | 98.1 | 5 / 5 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 15.6157 [15.4671-17.1946] | 15.2190 [15.1017-16.9034] | 97.5 | 1 / 1 | noise |
| lm_head | Q8 | dense-decode | 1 | bf16 | 3.1689 [3.1669-3.1712] | 3.1848 [3.1829-3.1886] | 100.5 | 213 / 212 | slower |
| lm_head | Q8 | dense-decode | 2 | bf16 | 3.1716 [3.1672-3.1745] | 3.1953 [3.1936-3.1985] | 100.8 | 213 / 211 | slower |
| lm_head | Q8 | dense-decode | 3 | bf16 | 3.4054 [3.4010-3.4077] | 3.2119 [3.2106-3.2201] | 94.3 | 198 / 210 | faster |
| lm_head | Q8 | dense-decode | 4 | bf16 | 3.5443 [3.5394-3.5620] | 3.2469 [3.2434-3.2588] | 91.6 | 191 / 208 | faster |
| lm_head | Q8 | dense-decode | 5 | bf16 | 3.4330 [3.4178-3.4586] | 3.3120 [3.3048-3.3216] | 96.5 | 197 / 204 | faster |
| lm_head | Q8 | dense-decode | 6 | bf16 | 3.5187 [3.5002-3.5333] | 3.4520 [3.4370-3.4616] | 98.1 | 192 / 196 | faster |
| lm_head | Q8 | dense-decode | 7 | bf16 | 3.5967 [3.5650-3.6210] | 3.5199 [3.5111-3.5366] | 97.9 | 188 / 192 | faster |
| lm_head | Q8 | dense-decode | 8 | bf16 | 3.7429 [3.6958-3.7859] | 3.7322 [3.6879-3.7676] | 99.7 | 180 / 181 | noise |
| lm_head | Q8 | dense-prefill-wmma | 512 | bf16 | 22.2069 [19.6696-22.9377] | 21.5416 [19.4633-22.1770] | 97.0 | 30 / 31 | noise |
| lm_head | Q8 | dense-prefill-wmma | 2048 | bf16 | 91.0408 [89.5659-92.2199] | 88.4744 [86.4385-89.6701] | 97.2 | 7 / 8 | noise |
| lm_head | Q8 | dense-decode | 1 | f32 | 3.2795 [3.2659-3.3107] | 3.2986 [3.2822-3.3088] | 100.6 | 206 / 205 | noise |
| lm_head | Q8 | dense-decode | 2 | f32 | 3.2808 [3.2696-3.2999] | 3.3201 [3.2860-3.3264] | 101.2 | 206 / 203 | noise |
| lm_head | Q8 | dense-decode | 3 | f32 | 3.4633 [3.4521-3.4673] | 3.2760 [3.2707-3.2840] | 94.6 | 195 / 206 | faster |
| lm_head | Q8 | dense-decode | 4 | f32 | 3.5933 [3.5852-3.5980] | 3.2972 [3.2916-3.3080] | 91.8 | 188 / 205 | faster |
| lm_head | Q8 | dense-decode | 5 | f32 | 3.4645 [3.4503-3.4788] | 3.3821 [3.3772-3.3862] | 97.6 | 195 / 200 | faster |
| lm_head | Q8 | dense-decode | 6 | f32 | 3.5440 [3.5366-3.5772] | 3.4986 [3.4900-3.5146] | 98.7 | 191 / 193 | faster |
| lm_head | Q8 | dense-decode | 7 | f32 | 3.6612 [3.6466-3.6877] | 3.6672 [3.6586-3.6783] | 100.2 | 184 / 184 | noise |
| lm_head | Q8 | dense-decode | 8 | f32 | 3.7988 [3.7638-3.8570] | 3.8443 [3.8424-3.8647] | 101.2 | 178 / 176 | noise |
| lm_head | Q8 | dense-prefill-wmma | 512 | f32 | 22.7577 [22.5982-23.0788] | 22.2018 [22.0259-22.4619] | 97.6 | 30 / 30 | faster |
| lm_head | Q4 syn | dense-decode | 1 | bf16 | 1.8579 [1.8431-1.8671] | 1.8314 [1.8246-1.8429] | 98.6 | 192 / 195 | faster |
| lm_head | Q4 syn | dense-decode | 2 | bf16 | 1.8649 [1.8523-1.8703] | 1.8379 [1.8286-1.8508] | 98.5 | 192 / 195 | faster |
| lm_head | Q4 syn | dense-decode | 3 | bf16 | 2.0933 [2.0885-2.0982] | 1.9706 [1.9659-1.9767] | 94.1 | 171 / 181 | faster |
| lm_head | Q4 syn | dense-decode | 4 | bf16 | 2.2273 [2.2172-2.2308] | 2.0781 [2.0731-2.0875] | 93.3 | 161 / 172 | faster |
| lm_head | Q4 syn | dense-decode | 5 | bf16 | 2.4570 [2.4446-2.4608] | 2.3256 [2.3138-2.3298] | 94.7 | 146 / 154 | faster |
| lm_head | Q4 syn | dense-decode | 6 | bf16 | 2.5892 [2.5834-2.5930] | 2.4375 [2.4342-2.4522] | 94.1 | 138 / 147 | faster |
| lm_head | Q4 syn | dense-decode | 7 | bf16 | 2.7336 [2.7228-2.7415] | 2.5538 [2.5438-2.5731] | 93.4 | 131 / 140 | faster |
| lm_head | Q4 syn | dense-decode | 8 | bf16 | 2.8841 [2.8751-2.8991] | 2.6666 [2.6580-2.6866] | 92.5 | 124 / 134 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | bf16 | 21.9519 [21.8273-22.5243] | 21.5526 [21.3447-21.6703] | 98.2 | 16 / 17 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 88.6829 [87.2543-88.9821] | 85.9796 [84.7425-86.6064] | 97.0 | 4 / 4 | faster |
| lm_head | Q4 syn | dense-decode | 1 | f32 | 1.7998 [1.7953-1.8083] | 1.7803 [1.7766-1.7895] | 98.9 | 199 / 201 | faster |
| lm_head | Q4 syn | dense-decode | 2 | f32 | 1.8571 [1.8503-1.8607] | 1.8104 [1.8077-1.8165] | 97.5 | 193 / 198 | faster |
| lm_head | Q4 syn | dense-decode | 3 | f32 | 2.0592 [2.0467-2.0659] | 1.9226 [1.9154-1.9387] | 93.4 | 174 / 186 | faster |
| lm_head | Q4 syn | dense-decode | 4 | f32 | 2.1836 [2.1741-2.1980] | 2.0379 [2.0282-2.0505] | 93.3 | 164 / 175 | faster |
| lm_head | Q4 syn | dense-decode | 5 | f32 | 2.4333 [2.4285-2.4404] | 2.2910 [2.2854-2.2978] | 94.2 | 147 / 156 | faster |
| lm_head | Q4 syn | dense-decode | 6 | f32 | 2.5249 [2.5189-2.5324] | 2.4032 [2.3986-2.4118] | 95.2 | 142 / 149 | faster |
| lm_head | Q4 syn | dense-decode | 7 | f32 | 2.6672 [2.6575-2.6913] | 2.5401 [2.5242-2.5508] | 95.2 | 134 / 141 | faster |
| lm_head | Q4 syn | dense-decode | 8 | f32 | 2.8395 [2.8263-2.8529] | 2.6652 [2.6555-2.6709] | 93.9 | 126 / 134 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | f32 | 21.1192 [21.0584-21.4078] | 20.8017 [20.7269-20.9595] | 98.5 | 17 / 17 | faster |
| mtp_fc | Q8 | dense-decode | 1 | bf16 | 0.0811 [0.0806-0.0816] | 0.0820 [0.0816-0.0822] | 101.1 | 86 / 85 | slower |
| mtp_fc | Q8 | dense-decode | 2 | bf16 | 0.0827 [0.0817-0.0836] | 0.0833 [0.0829-0.0838] | 100.7 | 84 / 84 | noise |
| mtp_fc | Q8 | dense-decode | 3 | bf16 | 0.0858 [0.0850-0.0865] | 0.0849 [0.0842-0.0855] | 99.0 | 81 / 82 | noise |
| mtp_fc | Q8 | dense-decode | 4 | bf16 | 0.0887 [0.0885-0.0902] | 0.0879 [0.0869-0.0886] | 99.0 | 78 / 79 | noise |
| mtp_fc | Q8 | dense-decode | 5 | bf16 | 0.0928 [0.0921-0.0930] | 0.0926 [0.0921-0.0932] | 99.8 | 75 / 75 | noise |
| mtp_fc | Q8 | dense-decode | 6 | bf16 | 0.0946 [0.0937-0.0953] | 0.0950 [0.0944-0.0952] | 100.3 | 74 / 73 | noise |
| mtp_fc | Q8 | dense-decode | 7 | bf16 | 0.0968 [0.0964-0.0976] | 0.0972 [0.0965-0.0979] | 100.4 | 72 / 72 | noise |
| mtp_fc | Q8 | dense-decode | 8 | bf16 | 0.0994 [0.0987-0.0999] | 0.0995 [0.0984-0.1002] | 100.1 | 70 / 70 | noise |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | bf16 | 0.2910 [0.2874-0.2928] | 0.2276 [0.2259-0.2287] | 78.2 | 24 / 31 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.8422 [0.8299-0.8491] | 0.7847 [0.7744-0.7894] | 93.2 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | bf16 | 3.1940 [3.1213-3.2815] | 3.1447 [3.0141-3.2032] | 98.5 | 2 / 2 | noise |
| mtp_fc | Q8 | dense-decode | 1 | f32 | 0.0812 [0.0810-0.0816] | 0.0819 [0.0817-0.0823] | 100.9 | 86 / 85 | slower |
| mtp_fc | Q8 | dense-decode | 2 | f32 | 0.0830 [0.0822-0.0846] | 0.0838 [0.0833-0.0841] | 101.0 | 84 / 83 | noise |
| mtp_fc | Q8 | dense-decode | 3 | f32 | 0.0871 [0.0867-0.0872] | 0.0856 [0.0852-0.0864] | 98.3 | 80 / 81 | faster |
| mtp_fc | Q8 | dense-decode | 4 | f32 | 0.0895 [0.0884-0.0902] | 0.0886 [0.0881-0.0895] | 99.0 | 78 / 79 | noise |
| mtp_fc | Q8 | dense-decode | 5 | f32 | 0.0932 [0.0924-0.0936] | 0.0922 [0.0916-0.0930] | 98.8 | 75 / 76 | noise |
| mtp_fc | Q8 | dense-decode | 6 | f32 | 0.0946 [0.0942-0.0954] | 0.0956 [0.0945-0.0958] | 101.1 | 74 / 73 | noise |
| mtp_fc | Q8 | dense-decode | 7 | f32 | 0.0968 [0.0964-0.0970] | 0.0972 [0.0965-0.0977] | 100.5 | 72 / 72 | noise |
| mtp_fc | Q8 | dense-decode | 8 | f32 | 0.1004 [0.0995-0.1012] | 0.0999 [0.0982-0.1006] | 99.6 | 69 / 70 | noise |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | f32 | 0.2924 [0.2853-0.2948] | 0.2273 [0.2263-0.2292] | 77.7 | 24 / 31 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | f32 | 0.8242 [0.8169-0.8414] | 0.7799 [0.7732-0.7841] | 94.6 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | f32 | 3.0777 [3.0591-3.2402] | 2.9946 [2.9803-3.0246] | 97.3 | 2 / 2 | faster |
| mtp_fc | Q4 syn | dense-decode | 1 | bf16 | 0.0444 [0.0442-0.0448] | 0.0442 [0.0440-0.0446] | 99.5 | 83 / 83 | noise |
| mtp_fc | Q4 syn | dense-decode | 2 | bf16 | 0.0482 [0.0469-0.0493] | 0.0456 [0.0453-0.0459] | 94.8 | 77 / 81 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | bf16 | 0.0506 [0.0495-0.0518] | 0.0472 [0.0467-0.0476] | 93.3 | 73 / 78 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | bf16 | 0.0518 [0.0511-0.0527] | 0.0485 [0.0483-0.0488] | 93.6 | 71 / 76 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | bf16 | 0.0541 [0.0535-0.0549] | 0.0526 [0.0522-0.0530] | 97.3 | 68 / 70 | faster |
| mtp_fc | Q4 syn | dense-decode | 6 | bf16 | 0.0568 [0.0560-0.0581] | 0.0545 [0.0535-0.0550] | 96.0 | 65 / 68 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | bf16 | 0.0584 [0.0576-0.0591] | 0.0557 [0.0554-0.0564] | 95.3 | 63 / 66 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | bf16 | 0.0607 [0.0603-0.0619] | 0.0572 [0.0565-0.0577] | 94.2 | 61 / 64 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.2277 [0.2262-0.2290] | 0.2130 [0.2108-0.2134] | 93.5 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.7879 [0.7812-0.7989] | 0.7716 [0.7664-0.7800] | 97.9 | 5 / 5 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 3.1448 [3.1344-3.1624] | 3.1206 [3.0887-3.1413] | 99.2 | 1 / 1 | noise |
| mtp_fc | Q4 syn | dense-decode | 1 | f32 | 0.0444 [0.0441-0.0453] | 0.0440 [0.0436-0.0444] | 99.2 | 83 / 84 | noise |
| mtp_fc | Q4 syn | dense-decode | 2 | f32 | 0.0474 [0.0470-0.0490] | 0.0457 [0.0454-0.0459] | 96.5 | 78 / 81 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | f32 | 0.0497 [0.0488-0.0510] | 0.0471 [0.0467-0.0474] | 94.8 | 74 / 78 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | f32 | 0.0517 [0.0502-0.0523] | 0.0488 [0.0485-0.0490] | 94.4 | 71 / 76 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | f32 | 0.0541 [0.0526-0.0546] | 0.0527 [0.0519-0.0532] | 97.4 | 68 / 70 | noise |
| mtp_fc | Q4 syn | dense-decode | 6 | f32 | 0.0563 [0.0554-0.0572] | 0.0539 [0.0535-0.0548] | 95.7 | 65 / 68 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | f32 | 0.0590 [0.0577-0.0598] | 0.0556 [0.0550-0.0560] | 94.3 | 63 / 66 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | f32 | 0.0603 [0.0600-0.0616] | 0.0576 [0.0569-0.0579] | 95.4 | 61 / 64 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.2361 [0.2340-0.2387] | 0.2202 [0.2195-0.2209] | 93.3 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.7821 [0.7754-0.7889] | 0.7746 [0.7705-0.7784] | 99.0 | 5 / 5 | noise |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | f32 | 12.5816 [12.5228-12.6571] | 12.3892 [12.3595-12.4434] | 98.5 | 0 / 0 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 1 | bf16 | 0.7792 [0.7744-0.7851] | 0.7700 [0.7679-0.7783] | 98.8 | 22 / 23 | noise |
| exp_gu | Q4 | expert-decode-gather | 1 | bf16 | 0.7656 [0.7604-0.7703] | 0.7651 [0.7632-0.7692] | 99.9 | 23 / 23 | noise |
| exp_down | Q4 | expert-decode-combine | 1 | bf16 | 0.4847 [0.4825-0.4872] | 0.4770 [0.4724-0.4817] | 98.4 | 21 / 21 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 2 | bf16 | 1.3812 [1.3759-1.3949] | 1.3961 [1.3918-1.4016] | 101.1 | 24 / 24 | noise |
| exp_gu | Q4 | expert-decode-gather | 2 | bf16 | 1.3810 [1.3775-1.3865] | 1.3985 [1.3936-1.4026] | 101.3 | 24 / 24 | slower |
| exp_down | Q4 | expert-decode-combine | 2 | bf16 | 0.8799 [0.8783-0.8832] | 0.8658 [0.8650-0.8693] | 98.4 | 22 / 22 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 3 | bf16 | 1.7540 [1.7451-1.7590] | 1.7487 [1.7425-1.7525] | 99.7 | 26 / 26 | noise |
| exp_gu | Q4 | expert-decode-gather | 3 | bf16 | 1.7583 [1.7488-1.7661] | 1.7575 [1.7509-1.7653] | 100.0 | 26 / 26 | noise |
| exp_down | Q4 | expert-decode-combine | 3 | bf16 | 1.1891 [1.1846-1.1919] | 1.1781 [1.1753-1.1845] | 99.1 | 22 / 23 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 4 | bf16 | 2.0556 [2.0481-2.0617] | 2.0353 [2.0283-2.0438] | 99.0 | 29 / 29 | faster |
| exp_gu | Q4 | expert-decode-gather | 4 | bf16 | 2.0923 [2.0795-2.1033] | 2.0652 [2.0514-2.0933] | 98.7 | 28 / 29 | noise |
| exp_down | Q4 | expert-decode-combine | 4 | bf16 | 1.4245 [1.4171-1.4309] | 1.4559 [1.4528-1.4688] | 102.2 | 24 / 24 | slower |
| exp_gu | Q4 | expert-decode-gather-swiglu | 5 | bf16 | 0.5089 [0.5061-0.5107] | 0.4942 [0.4930-0.4985] | 97.1 | 120 / 123 | faster |
| exp_gu | Q4 | expert-decode-gather | 5 | bf16 | 0.5217 [0.5195-0.5243] | 0.5080 [0.5056-0.5097] | 97.4 | 117 / 120 | faster |
| exp_down | Q4 | expert-decode-combine | 5 | bf16 | 0.3736 [0.3730-0.3752] | 0.3558 [0.3537-0.3575] | 95.2 | 96 / 101 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 6 | bf16 | 0.5847 [0.5838-0.5855] | 0.5729 [0.5727-0.5733] | 98.0 | 128 / 131 | faster |
| exp_gu | Q4 | expert-decode-gather | 6 | bf16 | 0.5979 [0.5962-0.6003] | 0.5816 [0.5795-0.5848] | 97.3 | 125 / 129 | faster |
| exp_down | Q4 | expert-decode-combine | 6 | bf16 | 0.4307 [0.4300-0.4328] | 0.4143 [0.4120-0.4174] | 96.2 | 102 / 106 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 7 | bf16 | 0.6397 [0.6382-0.6417] | 0.6206 [0.6184-0.6216] | 97.0 | 128 / 132 | faster |
| exp_gu | Q4 | expert-decode-gather | 7 | bf16 | 0.6573 [0.6537-0.6586] | 0.6380 [0.6363-0.6406] | 97.1 | 124 / 128 | faster |
| exp_down | Q4 | expert-decode-combine | 7 | bf16 | 0.4558 [0.4543-0.4577] | 0.4353 [0.4311-0.4377] | 95.5 | 106 / 111 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 8 | bf16 | 0.7250 [0.7224-0.7267] | 0.7080 [0.7053-0.7098] | 97.7 | 125 / 128 | faster |
| exp_gu | Q4 | expert-decode-gather | 8 | bf16 | 0.7459 [0.7435-0.7483] | 0.7257 [0.7227-0.7311] | 97.3 | 121 / 125 | faster |
| exp_down | Q4 | expert-decode-combine | 8 | bf16 | 0.5005 [0.4990-0.5021] | 0.4735 [0.4705-0.4768] | 94.6 | 106 / 112 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 512 | bf16 | 4.4620 [4.4226-4.4924] | 4.1984 [4.1805-4.2119] | 94.1 | 178 / 189 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 512 | bf16 | 4.4844 [4.4619-4.5836] | 4.2572 [4.2091-4.3659] | 94.9 | 177 / 187 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 512 | bf16 | 4.2941 [4.2842-4.3064] | 4.2748 [4.2656-4.2837] | 99.5 | 185 / 186 | faster |
| exp_down | Q4 | expert-prefill-combine narrow | 512 | bf16 | 3.0428 [3.0241-3.0644] | 2.7916 [2.7823-2.7963] | 91.8 | 154 / 168 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 512 | bf16 | 3.0360 [3.0105-3.0840] | 2.7903 [2.7782-2.8672] | 91.9 | 154 / 168 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 512 | bf16 | 3.0282 [3.0195-3.0396] | 2.7936 [2.7873-2.8050] | 92.2 | 155 / 168 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 512 | bf16 | 5.5732 [5.4892-5.6540] | 5.5686 [5.4697-5.6180] | 99.9 | 143 / 143 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 512 | bf16 | 5.5751 [5.5051-5.6445] | 5.5334 [5.4491-5.6085] | 99.2 | 143 / 144 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 512 | bf16 | 5.5149 [5.4780-5.5880] | 5.3960 [5.3652-5.4660] | 97.8 | 144 / 147 | faster |
| exp_down | Q4 | expert-prefill-combine wide | 512 | bf16 | 3.4463 [3.4351-3.4877] | 3.0614 [3.0482-3.1175] | 88.8 | 136 / 153 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 512 | bf16 | 3.3499 [3.3334-3.3980] | 2.9865 [2.9688-3.0221] | 89.2 | 140 / 157 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 512 | bf16 | 3.3312 [3.2911-3.3406] | 2.9515 [2.9087-2.9989] | 88.6 | 140 / 159 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 512 | bf16 | 5.3289 [5.3117-5.3492] | 4.6489 [4.5074-4.6661] | 87.2 | 149 / 171 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 512 | bf16 | 5.5226 [5.4997-5.5435] | 5.1144 [5.0977-5.1396] | 92.6 | 144 / 156 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 512 | bf16 | 5.5429 [5.5312-5.5577] | 5.1243 [5.0944-5.1625] | 92.5 | 144 / 155 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 512 | bf16 | 4.3346 [4.2932-4.3700] | 2.9495 [2.9371-2.9543] | 68.0 | 108 / 159 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 512 | bf16 | 4.3841 [4.3620-4.3958] | 2.9324 [2.9249-2.9392] | 66.9 | 107 / 160 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 512 | bf16 | 4.3768 [4.3405-4.4127] | 2.9570 [2.9508-2.9662] | 67.6 | 107 / 158 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 2048 | bf16 | 7.6394 [7.3830-8.1140] | 7.0243 [6.9607-7.5879] | 92.0 | 116 / 127 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 2048 | bf16 | 8.1369 [8.1103-8.1649] | 7.6479 [7.6228-7.6606] | 94.0 | 109 / 116 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 2048 | bf16 | 8.1012 [7.0856-8.1629] | 8.1088 [6.9847-8.1687] | 100.1 | 110 / 110 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 2048 | bf16 | 5.4555 [5.4148-5.4897] | 5.0791 [4.9985-5.0934] | 93.1 | 96 / 103 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 2048 | bf16 | 5.3691 [5.3326-5.3807] | 5.0019 [4.9585-5.0147] | 93.2 | 97 / 105 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 2048 | bf16 | 5.4722 [5.4428-5.4932] | 5.1093 [5.0505-5.1259] | 93.4 | 96 / 102 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 2048 | bf16 | 9.2775 [9.0808-9.4551] | 9.2993 [9.0388-9.4539] | 100.2 | 96 / 96 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 2048 | bf16 | 9.2420 [9.1426-9.3704] | 9.2050 [9.1068-9.3703] | 99.6 | 96 / 97 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 2048 | bf16 | 9.4071 [9.2660-9.4324] | 9.2988 [9.2115-9.3501] | 98.8 | 95 / 96 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 2048 | bf16 | 5.8240 [5.7738-5.8700] | 5.3815 [5.2858-5.4099] | 92.4 | 90 / 97 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 2048 | bf16 | 5.6018 [5.5903-5.6436] | 5.1353 [5.1193-5.1618] | 91.7 | 93 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 2048 | bf16 | 5.7365 [5.7207-5.7595] | 5.2817 [5.2120-5.3062] | 92.1 | 91 / 99 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 2048 | bf16 | 8.5846 [8.5468-8.6048] | 8.2626 [8.2423-8.2998] | 96.2 | 104 / 108 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 2048 | bf16 | 8.5889 [8.5211-8.6208] | 8.2735 [8.2219-8.3127] | 96.3 | 104 / 108 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 2048 | bf16 | 8.9145 [8.5856-9.0250] | 8.5663 [7.9218-8.6502] | 96.1 | 100 / 104 | noise |
| exp_down | Q4 | expert-prefill-combine wide256 | 2048 | bf16 | 6.1282 [6.0990-6.1612] | 5.1089 [5.0780-5.1252] | 83.4 | 85 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 2048 | bf16 | 6.0873 [6.0567-6.1418] | 5.1032 [5.0513-5.1316] | 83.8 | 86 / 103 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 2048 | bf16 | 6.1522 [6.1138-6.1883] | 5.1507 [5.1160-5.1735] | 83.7 | 85 / 102 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 8192 | bf16 | 21.3311 [20.8428-22.2045] | 20.8534 [20.4468-21.2265] | 97.8 | 42 / 43 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 8192 | bf16 | 22.8528 [22.7328-22.9164] | 22.0275 [21.7480-22.1343] | 96.4 | 39 / 40 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 8192 | bf16 | 21.7570 [21.5774-22.0339] | 22.0573 [21.8192-22.2801] | 101.4 | 41 / 40 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 8192 | bf16 | 15.2023 [14.9770-15.4241] | 14.6874 [14.3909-14.8731] | 96.6 | 34 / 36 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 8192 | bf16 | 14.9462 [14.4556-15.1650] | 14.3895 [13.5108-14.5916] | 96.3 | 35 / 36 | noise |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 8192 | bf16 | 15.2933 [15.1380-15.4263] | 14.6871 [14.5939-14.8938] | 96.0 | 34 / 36 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 8192 | bf16 | 20.7095 [19.7644-31.6962] | 20.6302 [19.3252-23.2856] | 99.6 | 43 / 43 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 8192 | bf16 | 21.1228 [20.9992-21.2531] | 21.1584 [21.0611-21.3798] | 100.2 | 42 / 42 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 8192 | bf16 | 21.0485 [20.9572-21.1360] | 21.0057 [20.8189-21.0480] | 99.8 | 42 / 42 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 8192 | bf16 | 13.9502 [13.9176-14.0145] | 13.2636 [13.1952-13.3976] | 95.1 | 38 / 40 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 8192 | bf16 | 14.1350 [13.9604-14.2177] | 13.4331 [13.2959-13.5253] | 95.0 | 37 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 8192 | bf16 | 14.1843 [14.0693-14.3482] | 13.5236 [13.2736-13.6694] | 95.3 | 37 / 39 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 8192 | bf16 | 20.5862 [20.1988-21.0183] | 20.1526 [19.6268-20.6025] | 97.9 | 43 / 44 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 8192 | bf16 | 20.6810 [20.3328-20.9114] | 20.2877 [20.0096-20.4908] | 98.1 | 43 / 44 | noise |
| exp_gu | Q4 | expert-prefill-gather wide256 | 8192 | bf16 | 21.0757 [20.7452-21.2960] | 20.0130 [19.6909-20.4592] | 95.0 | 42 / 45 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 8192 | bf16 | 15.2172 [15.0515-15.3785] | 13.6625 [13.5036-13.8294] | 89.8 | 34 / 38 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 8192 | bf16 | 15.0662 [14.8873-15.2437] | 13.5675 [13.4502-13.6842] | 90.0 | 35 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 8192 | bf16 | 15.2479 [15.0763-15.4005] | 13.7833 [13.6000-13.9209] | 90.4 | 34 / 38 | faster |
