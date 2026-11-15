# strixw panel kernels: kernel gate results

Criteria: `gate/strixw-panel-kernels.toml` (committed before the run). Raw: `raw.jsonl` (one case a line), bench log `bench.log`. 489 cases, 6 skipped (output too large), 2.232 GB read from the fleet file.

## Verdicts

| criterion | pass | detail |
|---|---|---|
| equal | PASS | 489 of 489 cases byte-equal |
| bad_id | PASS | 21 of 21 out-of-range-id cases byte-equal |
| coverage | PASS | missing: none |
| read | FAIL | 2.232 GB read |
| speed_decode_dense | PASS | median panel % of origin time per class: attn_o 98.5, attn_qkv 98.7, gdn_in 98.1, gdn_out 99.2, lm_head 96.5, mtp_fc 99.0, ple_kv 99.6, sh_down 90.7, sh_gu 95.6 |
| speed_prefill_dense | PASS | sh_gu M2048 bf16 94.2%; sh_gu M8192 bf16 98.2%; sh_gu M2048 f32 92.1%; sh_gu M8192 f32 96.8%; sh_down M2048 bf16 92.6%; sh_down M8192 bf16 96.2%; sh_down M2048 f32 95.2%; sh_down M8192 f32 96.2%; gdn_in M2048 bf16 99.1%; gdn_in M8192 bf16 98.4%; gdn_in M2048 f32 99.0%; gdn_in M8192 f32 99.9%; gdn_out M2048 bf16 72.1%; gdn_out M8192 bf16 89.4%; gdn_out M2048 f32 67.1%; gdn_out M8192 f32 91.0%; attn_qkv M2048 bf16 93.6%; attn_qkv M8192 bf16 97.8%; attn_qkv M2048 f32 90.8%; attn_qkv M8192 f32 96.8%; attn_o M2048 bf16 68.8%; attn_o M8192 bf16 90.9%; attn_o M2048 f32 67.3%; attn_o M8192 f32 88.4%; ple_kv M2048 bf16 94.0%; ple_kv M8192 bf16 96.7%; ple_kv M2048 f32 93.2%; ple_kv M8192 f32 96.7%; lm_head M2048 bf16 96.2%; mtp_fc M2048 bf16 92.0%; mtp_fc M8192 bf16 97.8%; mtp_fc M2048 f32 93.6%; mtp_fc M8192 f32 97.9% |
| speed_prefill_expert | PASS | expert-prefill-swiglu-f16h narrow M2048 94.3% (noise); expert-prefill-combine-f16h-shared narrow M2048 93.5% (faster); expert-prefill-combine-f16h-shared wide M8192 95.2% (faster); expert-prefill-swiglu-f16h wide256 M8192 97.2% (noise) |

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
| sh_gu | Q8 | dense-decode | 1 | bf16 | 0.0415 [0.0413-0.0418] | 0.0417 [0.0416-0.0418] | 100.5 | 84 / 83 | noise |
| sh_gu | Q8 | dense-decode | 2 | bf16 | 0.0428 [0.0424-0.0432] | 0.0426 [0.0424-0.0428] | 99.6 | 81 / 82 | noise |
| sh_gu | Q8 | dense-decode | 3 | bf16 | 0.0449 [0.0447-0.0455] | 0.0438 [0.0434-0.0444] | 97.6 | 78 / 79 | faster |
| sh_gu | Q8 | dense-decode | 4 | bf16 | 0.0471 [0.0467-0.0476] | 0.0455 [0.0452-0.0462] | 96.6 | 74 / 77 | faster |
| sh_gu | Q8 | dense-decode | 5 | bf16 | 0.0496 [0.0493-0.0502] | 0.0472 [0.0470-0.0483] | 95.2 | 70 / 74 | faster |
| sh_gu | Q8 | dense-decode | 6 | bf16 | 0.0508 [0.0506-0.0515] | 0.0487 [0.0482-0.0494] | 96.0 | 69 / 71 | faster |
| sh_gu | Q8 | dense-decode | 7 | bf16 | 0.0521 [0.0516-0.0524] | 0.0504 [0.0501-0.0511] | 96.8 | 67 / 69 | faster |
| sh_gu | Q8 | dense-decode | 8 | bf16 | 0.0539 [0.0536-0.0545] | 0.0516 [0.0512-0.0518] | 95.6 | 65 / 68 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 512 | bf16 | 0.1456 [0.1436-0.1491] | 0.1317 [0.1271-0.1339] | 90.5 | 24 / 26 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.4548 [0.4455-0.4752] | 0.4284 [0.4245-0.4381] | 94.2 | 8 / 8 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | bf16 | 1.5576 [1.5375-1.8703] | 1.5289 [1.4738-1.8303] | 98.2 | 2 / 2 | noise |
| sh_gu | Q8 | dense-decode | 1 | f32 | 0.0415 [0.0413-0.0417] | 0.0417 [0.0416-0.0419] | 100.6 | 84 / 83 | noise |
| sh_gu | Q8 | dense-decode | 2 | f32 | 0.0425 [0.0419-0.0433] | 0.0428 [0.0424-0.0432] | 100.8 | 82 / 81 | noise |
| sh_gu | Q8 | dense-decode | 3 | f32 | 0.0453 [0.0448-0.0457] | 0.0442 [0.0439-0.0448] | 97.4 | 77 / 79 | noise |
| sh_gu | Q8 | dense-decode | 4 | f32 | 0.0466 [0.0462-0.0474] | 0.0454 [0.0450-0.0459] | 97.3 | 75 / 77 | faster |
| sh_gu | Q8 | dense-decode | 5 | f32 | 0.0494 [0.0483-0.0497] | 0.0475 [0.0467-0.0479] | 96.1 | 70 / 73 | faster |
| sh_gu | Q8 | dense-decode | 6 | f32 | 0.0505 [0.0503-0.0509] | 0.0488 [0.0482-0.0492] | 96.5 | 69 / 71 | faster |
| sh_gu | Q8 | dense-decode | 7 | f32 | 0.0519 [0.0516-0.0523] | 0.0499 [0.0498-0.0507] | 96.2 | 67 / 70 | faster |
| sh_gu | Q8 | dense-decode | 8 | f32 | 0.0534 [0.0528-0.0538] | 0.0514 [0.0512-0.0516] | 96.3 | 65 / 68 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 512 | f32 | 0.1494 [0.1468-0.1520] | 0.1361 [0.1294-0.1368] | 91.1 | 23 / 26 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | f32 | 0.4566 [0.4496-0.4638] | 0.4204 [0.4092-0.4251] | 92.1 | 8 / 8 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | f32 | 1.5207 [1.5119-1.5606] | 1.4715 [1.4543-1.5023] | 96.8 | 2 / 2 | faster |
| sh_gu | Q4 syn | dense-decode | 1 | bf16 | 0.0267 [0.0264-0.0274] | 0.0254 [0.0252-0.0259] | 94.9 | 69 / 73 | faster |
| sh_gu | Q4 syn | dense-decode | 2 | bf16 | 0.0288 [0.0280-0.0302] | 0.0269 [0.0265-0.0270] | 93.2 | 64 / 69 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | bf16 | 0.0304 [0.0296-0.0307] | 0.0283 [0.0281-0.0284] | 93.0 | 61 / 65 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | bf16 | 0.0317 [0.0315-0.0328] | 0.0291 [0.0289-0.0293] | 91.8 | 58 / 63 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | bf16 | 0.0337 [0.0332-0.0347] | 0.0305 [0.0297-0.0312] | 90.6 | 55 / 60 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | bf16 | 0.0372 [0.0360-0.0384] | 0.0331 [0.0328-0.0332] | 89.0 | 50 / 56 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | bf16 | 0.0365 [0.0360-0.0374] | 0.0337 [0.0336-0.0339] | 92.2 | 50 / 55 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | bf16 | 0.0367 [0.0361-0.0375] | 0.0333 [0.0325-0.0336] | 90.7 | 50 / 55 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.1230 [0.1219-0.1257] | 0.1200 [0.1184-0.1215] | 97.5 | 15 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.4195 [0.4113-0.4430] | 0.4129 [0.4021-0.4212] | 98.4 | 4 / 4 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 1.6581 [1.6435-1.6745] | 1.6195 [1.6064-1.6307] | 97.7 | 1 / 1 | faster |
| sh_gu | Q4 syn | dense-decode | 1 | f32 | 0.0264 [0.0259-0.0272] | 0.0256 [0.0254-0.0258] | 97.0 | 70 / 72 | faster |
| sh_gu | Q4 syn | dense-decode | 2 | f32 | 0.0284 [0.0278-0.0291] | 0.0266 [0.0264-0.0268] | 93.7 | 65 / 69 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | f32 | 0.0302 [0.0290-0.0310] | 0.0274 [0.0271-0.0278] | 90.6 | 61 / 67 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | f32 | 0.0316 [0.0307-0.0322] | 0.0289 [0.0285-0.0290] | 91.5 | 58 / 64 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | f32 | 0.0336 [0.0328-0.0350] | 0.0308 [0.0305-0.0312] | 91.8 | 55 / 60 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | f32 | 0.0343 [0.0340-0.0350] | 0.0312 [0.0307-0.0321] | 91.0 | 54 / 59 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | f32 | 0.0364 [0.0353-0.0377] | 0.0332 [0.0330-0.0337] | 91.3 | 51 / 56 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | f32 | 0.0352 [0.0342-0.0361] | 0.0320 [0.0319-0.0326] | 91.1 | 52 / 58 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.1256 [0.1240-0.1267] | 0.1219 [0.1210-0.1240] | 97.1 | 15 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.4184 [0.4133-0.4260] | 0.4091 [0.4035-0.4157] | 97.8 | 4 / 5 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | f32 | 1.5612 [1.5514-1.5765] | 1.5432 [1.5310-1.5503] | 98.8 | 1 / 1 | faster |
| sh_down | Q8 | dense-decode | 1 | bf16 | 0.0255 [0.0253-0.0256] | 0.0252 [0.0250-0.0254] | 98.9 | 72 / 73 | noise |
| sh_down | Q8 | dense-decode | 2 | bf16 | 0.0279 [0.0276-0.0281] | 0.0262 [0.0258-0.0267] | 94.0 | 66 / 70 | faster |
| sh_down | Q8 | dense-decode | 3 | bf16 | 0.0300 [0.0292-0.0303] | 0.0273 [0.0270-0.0274] | 91.0 | 62 / 68 | faster |
| sh_down | Q8 | dense-decode | 4 | bf16 | 0.0314 [0.0311-0.0317] | 0.0279 [0.0274-0.0282] | 88.9 | 59 / 66 | faster |
| sh_down | Q8 | dense-decode | 5 | bf16 | 0.0328 [0.0326-0.0334] | 0.0288 [0.0284-0.0292] | 87.7 | 56 / 64 | faster |
| sh_down | Q8 | dense-decode | 6 | bf16 | 0.0334 [0.0327-0.0336] | 0.0297 [0.0291-0.0302] | 89.1 | 55 / 62 | faster |
| sh_down | Q8 | dense-decode | 7 | bf16 | 0.0344 [0.0338-0.0348] | 0.0306 [0.0301-0.0311] | 89.0 | 54 / 60 | faster |
| sh_down | Q8 | dense-decode | 8 | bf16 | 0.0364 [0.0358-0.0370] | 0.0315 [0.0310-0.0318] | 86.7 | 51 / 58 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | bf16 | 0.0758 [0.0752-0.0764] | 0.0684 [0.0681-0.0686] | 90.1 | 24 / 27 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.2388 [0.2334-0.2418] | 0.2212 [0.2174-0.2269] | 92.6 | 8 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | bf16 | 0.8755 [0.8629-0.8919] | 0.8427 [0.8280-0.8704] | 96.2 | 2 / 2 | noise |
| sh_down | Q8 | dense-decode | 1 | f32 | 0.0253 [0.0246-0.0254] | 0.0252 [0.0250-0.0255] | 99.7 | 73 / 73 | noise |
| sh_down | Q8 | dense-decode | 2 | f32 | 0.0271 [0.0269-0.0274] | 0.0266 [0.0263-0.0267] | 97.9 | 68 / 69 | faster |
| sh_down | Q8 | dense-decode | 3 | f32 | 0.0286 [0.0283-0.0288] | 0.0270 [0.0267-0.0275] | 94.4 | 64 / 68 | faster |
| sh_down | Q8 | dense-decode | 4 | f32 | 0.0302 [0.0298-0.0307] | 0.0275 [0.0273-0.0279] | 91.0 | 61 / 67 | faster |
| sh_down | Q8 | dense-decode | 5 | f32 | 0.0334 [0.0329-0.0338] | 0.0291 [0.0286-0.0293] | 87.2 | 55 / 63 | faster |
| sh_down | Q8 | dense-decode | 6 | f32 | 0.0340 [0.0333-0.0345] | 0.0296 [0.0292-0.0300] | 86.8 | 54 / 62 | faster |
| sh_down | Q8 | dense-decode | 7 | f32 | 0.0345 [0.0343-0.0349] | 0.0306 [0.0302-0.0308] | 88.6 | 53 / 60 | faster |
| sh_down | Q8 | dense-decode | 8 | f32 | 0.0355 [0.0346-0.0358] | 0.0311 [0.0306-0.0316] | 87.6 | 52 / 59 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | f32 | 0.0835 [0.0828-0.0843] | 0.0763 [0.0754-0.0769] | 91.3 | 22 / 24 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | f32 | 0.2436 [0.2393-0.2475] | 0.2320 [0.2290-0.2370] | 95.2 | 8 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | f32 | 0.8443 [0.8352-0.8578] | 0.8118 [0.7970-0.8201] | 96.2 | 2 / 2 | faster |
| sh_down | Q4 syn | dense-decode | 1 | bf16 | 0.0166 [0.0163-0.0167] | 0.0172 [0.0170-0.0173] | 103.6 | 62 / 60 | slower |
| sh_down | Q4 syn | dense-decode | 2 | bf16 | 0.0187 [0.0184-0.0190] | 0.0176 [0.0174-0.0177] | 93.8 | 55 / 58 | faster |
| sh_down | Q4 syn | dense-decode | 3 | bf16 | 0.0198 [0.0193-0.0200] | 0.0188 [0.0186-0.0190] | 94.9 | 52 / 55 | faster |
| sh_down | Q4 syn | dense-decode | 4 | bf16 | 0.0206 [0.0204-0.0210] | 0.0192 [0.0190-0.0194] | 92.8 | 50 / 53 | faster |
| sh_down | Q4 syn | dense-decode | 5 | bf16 | 0.0225 [0.0222-0.0226] | 0.0195 [0.0194-0.0196] | 86.8 | 46 / 52 | faster |
| sh_down | Q4 syn | dense-decode | 6 | bf16 | 0.0232 [0.0230-0.0235] | 0.0202 [0.0199-0.0209] | 86.7 | 44 / 51 | faster |
| sh_down | Q4 syn | dense-decode | 7 | bf16 | 0.0241 [0.0239-0.0243] | 0.0209 [0.0208-0.0211] | 86.6 | 42 / 49 | faster |
| sh_down | Q4 syn | dense-decode | 8 | bf16 | 0.0249 [0.0247-0.0250] | 0.0213 [0.0212-0.0214] | 85.5 | 41 / 48 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.0737 [0.0733-0.0751] | 0.0668 [0.0665-0.0674] | 90.7 | 14 / 15 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.2416 [0.2310-0.2437] | 0.2262 [0.2203-0.2281] | 93.6 | 4 / 5 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 0.8705 [0.8656-0.8765] | 0.8384 [0.8338-0.8447] | 96.3 | 1 / 1 | faster |
| sh_down | Q4 syn | dense-decode | 1 | f32 | 0.0164 [0.0164-0.0167] | 0.0168 [0.0167-0.0169] | 102.2 | 62 / 61 | noise |
| sh_down | Q4 syn | dense-decode | 2 | f32 | 0.0181 [0.0180-0.0182] | 0.0176 [0.0175-0.0178] | 97.3 | 57 / 58 | faster |
| sh_down | Q4 syn | dense-decode | 3 | f32 | 0.0194 [0.0193-0.0196] | 0.0188 [0.0184-0.0190] | 97.1 | 53 / 54 | faster |
| sh_down | Q4 syn | dense-decode | 4 | f32 | 0.0202 [0.0200-0.0204] | 0.0190 [0.0188-0.0193] | 93.9 | 51 / 54 | faster |
| sh_down | Q4 syn | dense-decode | 5 | f32 | 0.0222 [0.0220-0.0224] | 0.0194 [0.0192-0.0195] | 87.7 | 46 / 53 | faster |
| sh_down | Q4 syn | dense-decode | 6 | f32 | 0.0229 [0.0226-0.0230] | 0.0208 [0.0200-0.0209] | 90.7 | 45 / 49 | faster |
| sh_down | Q4 syn | dense-decode | 7 | f32 | 0.0243 [0.0238-0.0248] | 0.0209 [0.0207-0.0212] | 86.0 | 42 / 49 | faster |
| sh_down | Q4 syn | dense-decode | 8 | f32 | 0.0244 [0.0242-0.0245] | 0.0211 [0.0209-0.0214] | 86.5 | 42 / 49 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.0837 [0.0835-0.0844] | 0.0761 [0.0756-0.0764] | 90.9 | 12 / 13 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.2640 [0.2604-0.2686] | 0.2467 [0.2407-0.2492] | 93.5 | 4 / 4 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | f32 | 0.8593 [0.8496-0.8662] | 0.8144 [0.8084-0.8269] | 94.8 | 1 / 1 | faster |
| gdn_in | Q4 | dense-decode | 1 | bf16 | 0.2308 [0.2300-0.2322] | 0.2302 [0.2292-0.2329] | 99.7 | 97 / 97 | noise |
| gdn_in | Q4 | dense-decode | 2 | bf16 | 0.2352 [0.2321-0.2368] | 0.2297 [0.2282-0.2327] | 97.7 | 95 / 98 | noise |
| gdn_in | Q4 | dense-decode | 3 | bf16 | 0.2355 [0.2324-0.2366] | 0.2333 [0.2307-0.2369] | 99.0 | 95 / 96 | noise |
| gdn_in | Q4 | dense-decode | 4 | bf16 | 0.2391 [0.2366-0.2438] | 0.2345 [0.2337-0.2369] | 98.1 | 94 / 96 | noise |
| gdn_in | Q4 | dense-decode | 5 | bf16 | 0.2505 [0.2490-0.2540] | 0.2402 [0.2393-0.2443] | 95.9 | 89 / 93 | faster |
| gdn_in | Q4 | dense-decode | 6 | bf16 | 0.2545 [0.2500-0.2577] | 0.2453 [0.2396-0.2465] | 96.4 | 88 / 91 | faster |
| gdn_in | Q4 | dense-decode | 7 | bf16 | 0.2620 [0.2589-0.2642] | 0.2507 [0.2477-0.2526] | 95.7 | 86 / 89 | faster |
| gdn_in | Q4 | dense-decode | 8 | bf16 | 0.2649 [0.2635-0.2671] | 0.2511 [0.2490-0.2546] | 94.8 | 85 / 89 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | bf16 | 1.2822 [1.2688-1.2934] | 1.2510 [1.2368-1.2636] | 97.6 | 17 / 18 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | bf16 | 5.3291 [5.2598-5.3887] | 5.2799 [5.1662-5.3891] | 99.1 | 4 / 4 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | bf16 | 21.9107 [21.6287-22.2575] | 21.5504 [21.3651-21.8075] | 98.4 | 1 / 1 | noise |
| gdn_in | Q4 | dense-decode | 1 | f32 | 0.2317 [0.2301-0.2333] | 0.2300 [0.2283-0.2314] | 99.3 | 97 / 97 | noise |
| gdn_in | Q4 | dense-decode | 2 | f32 | 0.2361 [0.2337-0.2378] | 0.2317 [0.2299-0.2339] | 98.1 | 95 / 97 | noise |
| gdn_in | Q4 | dense-decode | 3 | f32 | 0.2383 [0.2351-0.2402] | 0.2336 [0.2324-0.2353] | 98.0 | 94 / 96 | noise |
| gdn_in | Q4 | dense-decode | 4 | f32 | 0.2412 [0.2387-0.2436] | 0.2364 [0.2337-0.2377] | 98.0 | 93 / 95 | faster |
| gdn_in | Q4 | dense-decode | 5 | f32 | 0.2524 [0.2506-0.2547] | 0.2422 [0.2392-0.2444] | 96.0 | 89 / 93 | faster |
| gdn_in | Q4 | dense-decode | 6 | f32 | 0.2558 [0.2536-0.2578] | 0.2445 [0.2431-0.2498] | 95.6 | 88 / 92 | faster |
| gdn_in | Q4 | dense-decode | 7 | f32 | 0.2610 [0.2593-0.2627] | 0.2470 [0.2442-0.2503] | 94.6 | 86 / 91 | faster |
| gdn_in | Q4 | dense-decode | 8 | f32 | 0.2660 [0.2641-0.2673] | 0.2528 [0.2488-0.2551] | 95.0 | 84 / 89 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | f32 | 1.1890 [1.1792-1.2194] | 1.1530 [1.1409-1.1652] | 97.0 | 19 / 19 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | f32 | 4.8604 [4.8193-4.9133] | 4.8105 [4.7843-4.9014] | 99.0 | 5 / 5 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | f32 | 21.2197 [19.8597-21.9235] | 21.1932 [19.5638-21.7111] | 99.9 | 1 / 1 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | bf16 | 0.3869 [0.3854-0.3903] | 0.3929 [0.3907-0.3951] | 101.5 | 112 / 111 | slower |
| gdn_in | Q8 syn | dense-decode | 2 | bf16 | 0.3863 [0.3844-0.3891] | 0.3934 [0.3925-0.3952] | 101.8 | 113 / 111 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | bf16 | 0.4004 [0.3986-0.4061] | 0.4019 [0.3972-0.4049] | 100.4 | 109 / 108 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | bf16 | 0.4096 [0.4078-0.4123] | 0.4071 [0.4050-0.4093] | 99.4 | 106 / 107 | noise |
| gdn_in | Q8 syn | dense-decode | 5 | bf16 | 0.4293 [0.4270-0.4351] | 0.4177 [0.4154-0.4207] | 97.3 | 101 / 104 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | bf16 | 0.4355 [0.4340-0.4432] | 0.4236 [0.4200-0.4254] | 97.3 | 100 / 103 | faster |
| gdn_in | Q8 syn | dense-decode | 7 | bf16 | 0.4431 [0.4391-0.4489] | 0.4330 [0.4310-0.4378] | 97.7 | 98 / 100 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | bf16 | 0.4476 [0.4459-0.4527] | 0.4409 [0.4374-0.4430] | 98.5 | 97 / 99 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | bf16 | 1.6557 [1.6225-1.6933] | 1.4083 [1.3925-1.4215] | 85.1 | 26 / 31 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | bf16 | 5.4278 [5.3119-5.8737] | 5.2955 [5.0591-5.6618] | 97.6 | 8 / 8 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | bf16 | 23.0921 [22.8607-23.1980] | 22.6013 [22.0379-22.8384] | 97.9 | 2 / 2 | faster |
| gdn_in | Q8 syn | dense-decode | 1 | f32 | 0.3874 [0.3862-0.3898] | 0.3925 [0.3905-0.3938] | 101.3 | 112 / 111 | slower |
| gdn_in | Q8 syn | dense-decode | 2 | f32 | 0.3876 [0.3866-0.3900] | 0.3948 [0.3932-0.3972] | 101.8 | 112 / 110 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | f32 | 0.4002 [0.3991-0.4015] | 0.3990 [0.3976-0.4021] | 99.7 | 109 / 109 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | f32 | 0.4055 [0.4032-0.4080] | 0.4033 [0.4017-0.4052] | 99.5 | 107 / 108 | noise |
| gdn_in | Q8 syn | dense-decode | 5 | f32 | 0.4257 [0.4249-0.4286] | 0.4180 [0.4147-0.4205] | 98.2 | 102 / 104 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | f32 | 0.4370 [0.4347-0.4402] | 0.4213 [0.4179-0.4258] | 96.4 | 100 / 103 | faster |
| gdn_in | Q8 syn | dense-decode | 7 | f32 | 0.4433 [0.4393-0.4476] | 0.4337 [0.4304-0.4377] | 97.8 | 98 / 100 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | f32 | 0.4514 [0.4488-0.4543] | 0.4428 [0.4396-0.4455] | 98.1 | 96 / 98 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | f32 | 1.5587 [1.5344-1.5935] | 1.3231 [1.3087-1.3293] | 84.9 | 28 / 33 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | f32 | 5.9924 [5.3789-6.3630] | 5.8503 [5.3113-6.1063] | 97.6 | 7 / 7 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | f32 | 23.4442 [23.3100-23.7730] | 23.1632 [22.8715-23.3682] | 98.8 | 2 / 2 | noise |
| gdn_out | Q8 | dense-decode | 1 | bf16 | 0.1765 [0.1760-0.1772] | 0.1776 [0.1767-0.1798] | 100.6 | 95 / 94 | noise |
| gdn_out | Q8 | dense-decode | 2 | bf16 | 0.1781 [0.1768-0.1799] | 0.1763 [0.1754-0.1783] | 99.0 | 94 / 95 | noise |
| gdn_out | Q8 | dense-decode | 3 | bf16 | 0.1838 [0.1814-0.1852] | 0.1848 [0.1825-0.1872] | 100.5 | 91 / 90 | noise |
| gdn_out | Q8 | dense-decode | 4 | bf16 | 0.1870 [0.1858-0.1905] | 0.1888 [0.1879-0.1908] | 101.0 | 89 / 88 | noise |
| gdn_out | Q8 | dense-decode | 5 | bf16 | 0.1967 [0.1920-0.2042] | 0.2011 [0.2000-0.2034] | 102.2 | 85 / 83 | noise |
| gdn_out | Q8 | dense-decode | 6 | bf16 | 0.2128 [0.2082-0.2165] | 0.2042 [0.2031-0.2051] | 95.9 | 79 / 82 | faster |
| gdn_out | Q8 | dense-decode | 7 | bf16 | 0.2154 [0.2120-0.2172] | 0.2108 [0.2081-0.2132] | 97.8 | 78 / 79 | noise |
| gdn_out | Q8 | dense-decode | 8 | bf16 | 0.2216 [0.2160-0.2284] | 0.2131 [0.2111-0.2174] | 96.1 | 75 / 78 | noise |
| gdn_out | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3621 [1.1934-1.5963] | 0.7731 [0.7518-0.8158] | 56.8 | 12 / 22 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.1163 [3.0911-3.1896] | 2.2481 [2.1860-2.2794] | 72.1 | 5 / 7 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.4035 [8.3221-8.4347] | 7.5102 [7.4128-7.5577] | 89.4 | 2 / 2 | faster |
| gdn_out | Q8 | dense-decode | 1 | f32 | 0.1754 [0.1748-0.1764] | 0.1756 [0.1745-0.1766] | 100.1 | 95 / 95 | noise |
| gdn_out | Q8 | dense-decode | 2 | f32 | 0.1777 [0.1758-0.1788] | 0.1771 [0.1757-0.1799] | 99.7 | 94 / 94 | noise |
| gdn_out | Q8 | dense-decode | 3 | f32 | 0.1820 [0.1798-0.1829] | 0.1829 [0.1820-0.1842] | 100.5 | 92 / 91 | noise |
| gdn_out | Q8 | dense-decode | 4 | f32 | 0.1875 [0.1848-0.1903] | 0.1882 [0.1867-0.1908] | 100.4 | 89 / 89 | noise |
| gdn_out | Q8 | dense-decode | 5 | f32 | 0.1975 [0.1925-0.2026] | 0.1994 [0.1985-0.2006] | 101.0 | 85 / 84 | noise |
| gdn_out | Q8 | dense-decode | 6 | f32 | 0.2071 [0.2032-0.2105] | 0.2046 [0.2032-0.2068] | 98.8 | 81 / 82 | noise |
| gdn_out | Q8 | dense-decode | 7 | f32 | 0.2182 [0.2144-0.2202] | 0.2072 [0.2062-0.2095] | 94.9 | 77 / 81 | faster |
| gdn_out | Q8 | dense-decode | 8 | f32 | 0.2208 [0.2163-0.2254] | 0.2105 [0.2095-0.2139] | 95.3 | 76 / 79 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 512 | f32 | 1.3263 [1.2231-1.4305] | 0.7654 [0.7471-0.8130] | 57.7 | 13 / 22 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | f32 | 3.0390 [2.9743-3.1106] | 2.0387 [2.0090-2.0784] | 67.1 | 6 / 8 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | f32 | 9.0389 [8.3511-9.2508] | 8.2271 [7.4276-8.3006] | 91.0 | 2 / 2 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | bf16 | 0.0985 [0.0975-0.0994] | 0.0980 [0.0967-0.1001] | 99.5 | 90 / 90 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | bf16 | 0.1033 [0.1021-0.1066] | 0.1016 [0.0999-0.1025] | 98.3 | 86 / 87 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | bf16 | 0.1042 [0.1029-0.1050] | 0.1029 [0.1013-0.1048] | 98.7 | 85 / 86 | noise |
| gdn_out | Q4 syn | dense-decode | 4 | bf16 | 0.1081 [0.1070-0.1109] | 0.1040 [0.1034-0.1070] | 96.2 | 82 / 85 | noise |
| gdn_out | Q4 syn | dense-decode | 5 | bf16 | 0.1151 [0.1146-0.1184] | 0.1150 [0.1126-0.1174] | 99.9 | 77 / 77 | noise |
| gdn_out | Q4 syn | dense-decode | 6 | bf16 | 0.1193 [0.1185-0.1212] | 0.1153 [0.1144-0.1189] | 96.7 | 74 / 77 | noise |
| gdn_out | Q4 syn | dense-decode | 7 | bf16 | 0.1229 [0.1218-0.1246] | 0.1180 [0.1173-0.1192] | 96.0 | 72 / 75 | faster |
| gdn_out | Q4 syn | dense-decode | 8 | bf16 | 0.1284 [0.1268-0.1294] | 0.1232 [0.1206-0.1247] | 96.0 | 69 / 72 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0083 [0.9877-1.0462] | 0.6121 [0.6023-0.6259] | 60.7 | 9 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.5472 [2.5277-2.5644] | 1.9730 [1.9105-2.0392] | 77.5 | 3 / 4 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 7.9974 [7.9256-8.0652] | 7.3825 [7.3400-7.4664] | 92.3 | 1 / 1 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | f32 | 0.0977 [0.0974-0.1016] | 0.0972 [0.0966-0.1007] | 99.5 | 91 / 91 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | f32 | 0.1006 [0.0992-0.1040] | 0.1028 [0.0989-0.1036] | 102.2 | 88 / 86 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | f32 | 0.1040 [0.1034-0.1061] | 0.1033 [0.1010-0.1063] | 99.3 | 85 / 86 | noise |
| gdn_out | Q4 syn | dense-decode | 4 | f32 | 0.1111 [0.1102-0.1122] | 0.1084 [0.1077-0.1098] | 97.5 | 80 / 82 | faster |
| gdn_out | Q4 syn | dense-decode | 5 | f32 | 0.1137 [0.1125-0.1171] | 0.1128 [0.1116-0.1150] | 99.2 | 78 / 78 | noise |
| gdn_out | Q4 syn | dense-decode | 6 | f32 | 0.1181 [0.1178-0.1218] | 0.1178 [0.1145-0.1200] | 99.7 | 75 / 75 | noise |
| gdn_out | Q4 syn | dense-decode | 7 | f32 | 0.1234 [0.1218-0.1272] | 0.1204 [0.1187-0.1219] | 97.5 | 72 / 73 | noise |
| gdn_out | Q4 syn | dense-decode | 8 | f32 | 0.1275 [0.1263-0.1294] | 0.1224 [0.1205-0.1256] | 96.0 | 69 / 72 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0175 [1.0043-1.0478] | 0.6181 [0.6034-0.6237] | 60.8 | 9 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.5145 [2.4733-2.5558] | 1.9736 [1.9595-1.9882] | 78.5 | 4 / 4 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | f32 | 8.3287 [8.2476-8.3965] | 7.8372 [7.8140-7.8773] | 94.1 | 1 / 1 | faster |
| attn_qkv | Q8 | dense-decode | 1 | bf16 | 0.3663 [0.3650-0.3674] | 0.3669 [0.3651-0.3677] | 100.2 | 104 / 103 | noise |
| attn_qkv | Q8 | dense-decode | 2 | bf16 | 0.3643 [0.3618-0.3680] | 0.3692 [0.3677-0.3725] | 101.3 | 104 / 103 | noise |
| attn_qkv | Q8 | dense-decode | 3 | bf16 | 0.3722 [0.3711-0.3744] | 0.3726 [0.3707-0.3746] | 100.1 | 102 / 102 | noise |
| attn_qkv | Q8 | dense-decode | 4 | bf16 | 0.3786 [0.3765-0.3800] | 0.3791 [0.3762-0.3815] | 100.1 | 100 / 100 | noise |
| attn_qkv | Q8 | dense-decode | 5 | bf16 | 0.3957 [0.3934-0.3982] | 0.3886 [0.3850-0.3911] | 98.2 | 96 / 98 | faster |
| attn_qkv | Q8 | dense-decode | 6 | bf16 | 0.4051 [0.4004-0.4073] | 0.3946 [0.3927-0.3980] | 97.4 | 94 / 96 | faster |
| attn_qkv | Q8 | dense-decode | 7 | bf16 | 0.4082 [0.4054-0.4129] | 0.4051 [0.3982-0.4099] | 99.2 | 93 / 94 | noise |
| attn_qkv | Q8 | dense-decode | 8 | bf16 | 0.4146 [0.4132-0.4172] | 0.4136 [0.4089-0.4196] | 99.8 | 92 / 92 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3639 [1.3521-1.3841] | 1.1421 [1.1329-1.1542] | 83.7 | 28 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.6204 [4.5124-4.8154] | 4.3241 [4.2597-4.5146] | 93.6 | 8 / 9 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | bf16 | 18.5472 [17.9934-19.7441] | 18.1310 [17.4373-19.3231] | 97.8 | 2 / 2 | noise |
| attn_qkv | Q8 | dense-decode | 1 | f32 | 0.3652 [0.3642-0.3664] | 0.3661 [0.3658-0.3676] | 100.2 | 104 / 104 | noise |
| attn_qkv | Q8 | dense-decode | 2 | f32 | 0.3649 [0.3624-0.3669] | 0.3684 [0.3677-0.3697] | 100.9 | 104 / 103 | slower |
| attn_qkv | Q8 | dense-decode | 3 | f32 | 0.3727 [0.3709-0.3740] | 0.3744 [0.3712-0.3780] | 100.5 | 102 / 101 | noise |
| attn_qkv | Q8 | dense-decode | 4 | f32 | 0.3787 [0.3770-0.3799] | 0.3782 [0.3768-0.3822] | 99.9 | 100 / 100 | noise |
| attn_qkv | Q8 | dense-decode | 5 | f32 | 0.3945 [0.3931-0.3952] | 0.3864 [0.3843-0.3892] | 98.0 | 96 / 98 | faster |
| attn_qkv | Q8 | dense-decode | 6 | f32 | 0.4032 [0.4008-0.4047] | 0.3956 [0.3924-0.3996] | 98.1 | 94 / 96 | faster |
| attn_qkv | Q8 | dense-decode | 7 | f32 | 0.4127 [0.4097-0.4146] | 0.4082 [0.4038-0.4102] | 98.9 | 92 / 93 | noise |
| attn_qkv | Q8 | dense-decode | 8 | f32 | 0.4194 [0.4163-0.4226] | 0.4141 [0.4091-0.4206] | 98.7 | 90 / 92 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | f32 | 1.3589 [1.3463-1.3807] | 1.1260 [1.1176-1.1355] | 82.9 | 28 / 34 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | f32 | 4.6621 [4.4717-5.0334] | 4.2339 [4.1878-4.6676] | 90.8 | 8 / 9 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | f32 | 19.4794 [17.8681-19.5959] | 18.8502 [17.0961-18.9100] | 96.8 | 2 / 2 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | bf16 | 0.2117 [0.2104-0.2129] | 0.2121 [0.2101-0.2136] | 100.2 | 95 / 95 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | bf16 | 0.2156 [0.2140-0.2180] | 0.2112 [0.2086-0.2132] | 98.0 | 93 / 95 | faster |
| attn_qkv | Q4 syn | dense-decode | 3 | bf16 | 0.2176 [0.2159-0.2199] | 0.2138 [0.2124-0.2166] | 98.2 | 92 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 4 | bf16 | 0.2187 [0.2167-0.2210] | 0.2150 [0.2142-0.2186] | 98.3 | 92 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 5 | bf16 | 0.2302 [0.2273-0.2324] | 0.2190 [0.2175-0.2217] | 95.2 | 87 / 92 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | bf16 | 0.2345 [0.2338-0.2358] | 0.2221 [0.2213-0.2256] | 94.7 | 86 / 90 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | bf16 | 0.2376 [0.2364-0.2420] | 0.2252 [0.2206-0.2268] | 94.8 | 85 / 89 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | bf16 | 0.2448 [0.2430-0.2465] | 0.2302 [0.2277-0.2339] | 94.1 | 82 / 87 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0860 [1.0730-1.0964] | 1.0325 [1.0134-1.0505] | 95.1 | 18 / 19 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 4.4127 [4.3058-4.5104] | 4.3583 [4.2562-4.4358] | 98.8 | 5 / 5 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 18.9847 [17.3705-19.0688] | 18.6339 [17.0467-18.7856] | 98.2 | 1 / 1 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | f32 | 0.2122 [0.2113-0.2146] | 0.2105 [0.2083-0.2136] | 99.2 | 95 / 95 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | f32 | 0.2133 [0.2097-0.2147] | 0.2115 [0.2107-0.2125] | 99.2 | 94 / 95 | noise |
| attn_qkv | Q4 syn | dense-decode | 3 | f32 | 0.2175 [0.2146-0.2202] | 0.2134 [0.2100-0.2163] | 98.1 | 92 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 4 | f32 | 0.2193 [0.2175-0.2221] | 0.2165 [0.2151-0.2180] | 98.7 | 92 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 5 | f32 | 0.2292 [0.2264-0.2308] | 0.2208 [0.2180-0.2231] | 96.4 | 88 / 91 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | f32 | 0.2328 [0.2308-0.2347] | 0.2226 [0.2206-0.2266] | 95.6 | 86 / 90 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | f32 | 0.2384 [0.2359-0.2423] | 0.2266 [0.2236-0.2306] | 95.1 | 84 / 89 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | f32 | 0.2449 [0.2428-0.2486] | 0.2288 [0.2256-0.2334] | 93.4 | 82 / 88 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0586 [1.0538-1.1095] | 1.0243 [1.0069-1.0897] | 96.8 | 19 / 20 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 4.4703 [4.1487-4.8895] | 4.3796 [4.1071-4.8161] | 98.0 | 4 / 5 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 18.7063 [18.1749-18.9624] | 18.4743 [16.8962-18.6188] | 98.8 | 1 / 1 | noise |
| attn_o | Q8 | dense-decode | 1 | bf16 | 0.1758 [0.1746-0.1771] | 0.1760 [0.1752-0.1802] | 100.2 | 95 / 95 | noise |
| attn_o | Q8 | dense-decode | 2 | bf16 | 0.1776 [0.1763-0.1812] | 0.1788 [0.1758-0.1818] | 100.7 | 94 / 93 | noise |
| attn_o | Q8 | dense-decode | 3 | bf16 | 0.1821 [0.1807-0.1846] | 0.1833 [0.1822-0.1852] | 100.7 | 92 / 91 | noise |
| attn_o | Q8 | dense-decode | 4 | bf16 | 0.1858 [0.1822-0.1886] | 0.1878 [0.1872-0.1891] | 101.1 | 90 / 89 | noise |
| attn_o | Q8 | dense-decode | 5 | bf16 | 0.1936 [0.1906-0.2026] | 0.2008 [0.1998-0.2026] | 103.7 | 86 / 83 | noise |
| attn_o | Q8 | dense-decode | 6 | bf16 | 0.2074 [0.2025-0.2114] | 0.2039 [0.2028-0.2061] | 98.3 | 81 / 82 | noise |
| attn_o | Q8 | dense-decode | 7 | bf16 | 0.2166 [0.2144-0.2194] | 0.2098 [0.2076-0.2126] | 96.9 | 77 / 80 | faster |
| attn_o | Q8 | dense-decode | 8 | bf16 | 0.2246 [0.2188-0.2279] | 0.2145 [0.2128-0.2172] | 95.5 | 74 / 78 | faster |
| attn_o | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3997 [1.2983-1.4880] | 0.7877 [0.7521-0.8308] | 56.3 | 12 / 21 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.0924 [3.0530-3.1231] | 2.1264 [2.0758-2.2348] | 68.8 | 5 / 8 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.9591 [8.8773-9.0689] | 8.1450 [8.0383-8.2709] | 90.9 | 2 / 2 | faster |
| attn_o | Q8 | dense-decode | 1 | f32 | 0.1752 [0.1750-0.1760] | 0.1752 [0.1740-0.1774] | 100.0 | 95 / 95 | noise |
| attn_o | Q8 | dense-decode | 2 | f32 | 0.1779 [0.1760-0.1802] | 0.1772 [0.1764-0.1805] | 99.6 | 94 / 94 | noise |
| attn_o | Q8 | dense-decode | 3 | f32 | 0.1815 [0.1799-0.1828] | 0.1830 [0.1820-0.1835] | 100.8 | 92 / 91 | noise |
| attn_o | Q8 | dense-decode | 4 | f32 | 0.1844 [0.1819-0.1866] | 0.1883 [0.1876-0.1908] | 102.1 | 91 / 89 | slower |
| attn_o | Q8 | dense-decode | 5 | f32 | 0.1970 [0.1912-0.1994] | 0.2006 [0.1988-0.2038] | 101.8 | 85 / 83 | noise |
| attn_o | Q8 | dense-decode | 6 | f32 | 0.2092 [0.2055-0.2134] | 0.2044 [0.2029-0.2062] | 97.7 | 80 / 82 | noise |
| attn_o | Q8 | dense-decode | 7 | f32 | 0.2158 [0.2109-0.2186] | 0.2080 [0.2064-0.2109] | 96.4 | 77 / 80 | faster |
| attn_o | Q8 | dense-decode | 8 | f32 | 0.2200 [0.2166-0.2236] | 0.2113 [0.2110-0.2125] | 96.0 | 76 / 79 | faster |
| attn_o | Q8 | dense-prefill-wmma | 512 | f32 | 1.3957 [1.3202-1.4341] | 0.7614 [0.7249-0.7769] | 54.6 | 12 / 22 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | f32 | 3.0535 [2.9880-3.0965] | 2.0543 [2.0321-2.0749] | 67.3 | 5 / 8 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | f32 | 8.4346 [8.3249-8.5689] | 7.4582 [7.4060-7.6379] | 88.4 | 2 / 2 | faster |
| attn_o | Q4 syn | dense-decode | 1 | bf16 | 0.0972 [0.0967-0.1005] | 0.0978 [0.0965-0.1009] | 100.6 | 91 / 90 | noise |
| attn_o | Q4 syn | dense-decode | 2 | bf16 | 0.1005 [0.0996-0.1034] | 0.0990 [0.0984-0.1013] | 98.5 | 88 / 89 | noise |
| attn_o | Q4 syn | dense-decode | 3 | bf16 | 0.1061 [0.1044-0.1082] | 0.1025 [0.1017-0.1035] | 96.6 | 83 / 86 | faster |
| attn_o | Q4 syn | dense-decode | 4 | bf16 | 0.1093 [0.1060-0.1114] | 0.1062 [0.1043-0.1080] | 97.1 | 81 / 83 | noise |
| attn_o | Q4 syn | dense-decode | 5 | bf16 | 0.1143 [0.1129-0.1159] | 0.1138 [0.1114-0.1167] | 99.5 | 77 / 78 | noise |
| attn_o | Q4 syn | dense-decode | 6 | bf16 | 0.1189 [0.1172-0.1214] | 0.1155 [0.1145-0.1170] | 97.1 | 74 / 77 | faster |
| attn_o | Q4 syn | dense-decode | 7 | bf16 | 0.1237 [0.1223-0.1278] | 0.1192 [0.1174-0.1207] | 96.4 | 72 / 74 | faster |
| attn_o | Q4 syn | dense-decode | 8 | bf16 | 0.1290 [0.1265-0.1313] | 0.1218 [0.1206-0.1248] | 94.4 | 69 / 73 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0080 [0.9922-1.0373] | 0.6244 [0.6016-0.6624] | 62.0 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.5698 [2.5383-2.6279] | 1.9960 [1.9398-2.0420] | 77.7 | 3 / 4 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 8.3595 [8.0848-8.4282] | 7.7965 [7.7123-7.8742] | 93.3 | 1 / 1 | faster |
| attn_o | Q4 syn | dense-decode | 1 | f32 | 0.0974 [0.0962-0.0984] | 0.0979 [0.0973-0.0993] | 100.5 | 91 / 90 | noise |
| attn_o | Q4 syn | dense-decode | 2 | f32 | 0.1018 [0.0997-0.1044] | 0.0998 [0.0987-0.1009] | 98.1 | 87 / 89 | noise |
| attn_o | Q4 syn | dense-decode | 3 | f32 | 0.1040 [0.1028-0.1077] | 0.1029 [0.1014-0.1058] | 98.9 | 85 / 86 | noise |
| attn_o | Q4 syn | dense-decode | 4 | f32 | 0.1074 [0.1059-0.1085] | 0.1048 [0.1040-0.1059] | 97.6 | 82 / 84 | noise |
| attn_o | Q4 syn | dense-decode | 5 | f32 | 0.1130 [0.1123-0.1158] | 0.1121 [0.1113-0.1140] | 99.2 | 78 / 79 | noise |
| attn_o | Q4 syn | dense-decode | 6 | f32 | 0.1180 [0.1173-0.1206] | 0.1155 [0.1144-0.1166] | 97.9 | 75 / 77 | faster |
| attn_o | Q4 syn | dense-decode | 7 | f32 | 0.1223 [0.1218-0.1234] | 0.1175 [0.1168-0.1197] | 96.1 | 72 / 75 | faster |
| attn_o | Q4 syn | dense-decode | 8 | f32 | 0.1272 [0.1262-0.1294] | 0.1214 [0.1200-0.1248] | 95.5 | 70 / 73 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0126 [0.9988-1.0375] | 0.6067 [0.5967-0.6254] | 59.9 | 9 / 15 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.5234 [2.4586-2.5688] | 1.9783 [1.9507-1.9936] | 78.4 | 4 / 4 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | f32 | 8.0087 [7.9557-8.1777] | 7.5244 [7.3306-7.6948] | 94.0 | 1 / 1 | faster |
| ple_kv | Q8 | dense-decode | 1 | bf16 | 0.3511 [0.3491-0.3527] | 0.3536 [0.3517-0.3568] | 100.7 | 99 / 98 | noise |
| ple_kv | Q8 | dense-decode | 2 | bf16 | 0.3512 [0.3497-0.3527] | 0.3550 [0.3536-0.3566] | 101.1 | 99 / 98 | slower |
| ple_kv | Q8 | dense-decode | 3 | bf16 | 0.3565 [0.3547-0.3586] | 0.3589 [0.3573-0.3606] | 100.7 | 98 / 97 | noise |
| ple_kv | Q8 | dense-decode | 4 | bf16 | 0.3629 [0.3600-0.3667] | 0.3656 [0.3636-0.3686] | 100.8 | 96 / 95 | noise |
| ple_kv | Q8 | dense-decode | 5 | bf16 | 0.3781 [0.3758-0.3816] | 0.3788 [0.3754-0.3797] | 100.2 | 92 / 92 | noise |
| ple_kv | Q8 | dense-decode | 6 | bf16 | 0.3845 [0.3816-0.3887] | 0.3871 [0.3821-0.3916] | 100.7 | 91 / 90 | noise |
| ple_kv | Q8 | dense-decode | 7 | bf16 | 0.3899 [0.3874-0.3939] | 0.3953 [0.3921-0.3969] | 101.4 | 89 / 88 | noise |
| ple_kv | Q8 | dense-decode | 8 | bf16 | 0.3946 [0.3921-0.4002] | 0.4070 [0.4016-0.4120] | 103.2 | 88 / 86 | slower |
| ple_kv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.2984 [1.2545-1.3223] | 1.0516 [1.0421-1.0708] | 81.0 | 27 / 33 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.3322 [4.2381-4.3688] | 4.0734 [3.9706-4.1466] | 94.0 | 8 / 9 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | bf16 | 17.0424 [16.2775-17.6246] | 16.4730 [15.8221-16.7739] | 96.7 | 2 / 2 | noise |
| ple_kv | Q8 | dense-decode | 1 | f32 | 0.3512 [0.3483-0.3541] | 0.3522 [0.3508-0.3560] | 100.3 | 99 / 99 | noise |
| ple_kv | Q8 | dense-decode | 2 | f32 | 0.3520 [0.3484-0.3528] | 0.3564 [0.3537-0.3572] | 101.3 | 99 / 98 | slower |
| ple_kv | Q8 | dense-decode | 3 | f32 | 0.3590 [0.3572-0.3622] | 0.3629 [0.3601-0.3666] | 101.1 | 97 / 96 | noise |
| ple_kv | Q8 | dense-decode | 4 | f32 | 0.3632 [0.3617-0.3658] | 0.3677 [0.3654-0.3694] | 101.2 | 96 / 95 | noise |
| ple_kv | Q8 | dense-decode | 5 | f32 | 0.3774 [0.3764-0.3786] | 0.3756 [0.3745-0.3792] | 99.5 | 92 / 93 | noise |
| ple_kv | Q8 | dense-decode | 6 | f32 | 0.3837 [0.3808-0.3856] | 0.3844 [0.3801-0.3879] | 100.2 | 91 / 91 | noise |
| ple_kv | Q8 | dense-decode | 7 | f32 | 0.3911 [0.3887-0.3937] | 0.3913 [0.3869-0.3954] | 100.0 | 89 / 89 | noise |
| ple_kv | Q8 | dense-decode | 8 | f32 | 0.4005 [0.3991-0.4052] | 0.4003 [0.3975-0.4034] | 99.9 | 87 / 87 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 512 | f32 | 1.2527 [1.2327-1.2726] | 1.0142 [1.0064-1.0263] | 81.0 | 28 / 34 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | f32 | 4.1267 [4.0874-4.4254] | 3.8448 [3.7973-4.0319] | 93.2 | 8 / 9 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | f32 | 17.9160 [16.2501-18.0978] | 17.3271 [15.6962-17.5324] | 96.7 | 2 / 2 | noise |
| ple_kv | Q4 syn | dense-decode | 1 | bf16 | 0.1936 [0.1927-0.1961] | 0.1928 [0.1922-0.1941] | 99.6 | 95 / 96 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | bf16 | 0.1984 [0.1961-0.2009] | 0.1942 [0.1932-0.1980] | 97.9 | 93 / 95 | noise |
| ple_kv | Q4 syn | dense-decode | 3 | bf16 | 0.2004 [0.1989-0.2036] | 0.1957 [0.1930-0.1973] | 97.7 | 92 / 94 | faster |
| ple_kv | Q4 syn | dense-decode | 4 | bf16 | 0.2042 [0.2018-0.2059] | 0.1984 [0.1957-0.2011] | 97.2 | 90 / 93 | faster |
| ple_kv | Q4 syn | dense-decode | 5 | bf16 | 0.2112 [0.2104-0.2126] | 0.2017 [0.1992-0.2046] | 95.5 | 87 / 91 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | bf16 | 0.2148 [0.2122-0.2163] | 0.2044 [0.2034-0.2063] | 95.2 | 86 / 90 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | bf16 | 0.2204 [0.2171-0.2216] | 0.2062 [0.2046-0.2103] | 93.6 | 84 / 89 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | bf16 | 0.2268 [0.2250-0.2281] | 0.2139 [0.2106-0.2154] | 94.3 | 81 / 86 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0078 [0.9957-1.0218] | 0.9507 [0.9448-0.9649] | 94.3 | 18 / 19 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 4.1560 [4.1061-4.2145] | 4.1011 [4.0298-4.1291] | 98.7 | 4 / 4 | noise |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 16.9995 [15.8043-17.4862] | 16.7459 [15.4477-17.1873] | 98.5 | 1 / 1 | noise |
| ple_kv | Q4 syn | dense-decode | 1 | f32 | 0.1958 [0.1947-0.1992] | 0.1944 [0.1930-0.1959] | 99.3 | 94 / 95 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | f32 | 0.1967 [0.1944-0.2002] | 0.1940 [0.1934-0.1959] | 98.6 | 94 / 95 | noise |
| ple_kv | Q4 syn | dense-decode | 3 | f32 | 0.2025 [0.1990-0.2034] | 0.1985 [0.1973-0.1994] | 98.0 | 91 / 93 | noise |
| ple_kv | Q4 syn | dense-decode | 4 | f32 | 0.2084 [0.2049-0.2113] | 0.2060 [0.2028-0.2108] | 98.9 | 88 / 89 | noise |
| ple_kv | Q4 syn | dense-decode | 5 | f32 | 0.2125 [0.2107-0.2140] | 0.2048 [0.2031-0.2064] | 96.4 | 87 / 90 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | f32 | 0.2190 [0.2181-0.2222] | 0.2101 [0.2082-0.2128] | 96.0 | 84 / 88 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | f32 | 0.2210 [0.2192-0.2223] | 0.2105 [0.2080-0.2126] | 95.3 | 83 / 88 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | f32 | 0.2240 [0.2221-0.2252] | 0.2130 [0.2104-0.2151] | 95.1 | 82 / 87 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.9562 [0.9503-0.9681] | 0.9025 [0.8964-0.9120] | 94.4 | 19 / 20 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 3.9691 [3.7527-4.2857] | 3.8123 [3.7048-4.1653] | 96.0 | 5 / 5 | noise |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 17.2191 [15.8017-17.4391] | 17.0738 [15.3871-17.2222] | 99.2 | 1 / 1 | noise |
| lm_head | Q8 | dense-decode | 1 | bf16 | 3.2240 [3.2163-3.2504] | 3.2403 [3.2128-3.2602] | 100.5 | 210 / 208 | noise |
| lm_head | Q8 | dense-decode | 2 | bf16 | 3.2153 [3.2020-3.2369] | 3.2349 [3.2240-3.2602] | 100.6 | 210 / 209 | noise |
| lm_head | Q8 | dense-decode | 3 | bf16 | 3.4344 [3.4220-3.4866] | 3.2432 [3.2326-3.3477] | 94.4 | 197 / 208 | faster |
| lm_head | Q8 | dense-decode | 4 | bf16 | 3.5818 [3.5682-3.5960] | 3.2739 [3.2681-3.2830] | 91.4 | 189 / 206 | faster |
| lm_head | Q8 | dense-decode | 5 | bf16 | 3.4777 [3.4512-3.4844] | 3.3546 [3.3481-3.3661] | 96.5 | 194 / 201 | faster |
| lm_head | Q8 | dense-decode | 6 | bf16 | 3.5306 [3.5132-3.5579] | 3.4542 [3.4250-3.4818] | 97.8 | 191 / 196 | faster |
| lm_head | Q8 | dense-decode | 7 | bf16 | 3.6280 [3.5850-3.6561] | 3.5603 [3.5486-3.5814] | 98.1 | 186 / 190 | faster |
| lm_head | Q8 | dense-decode | 8 | bf16 | 3.7515 [3.7076-3.7799] | 3.6992 [3.6875-3.7116] | 98.6 | 180 / 183 | noise |
| lm_head | Q8 | dense-prefill-wmma | 512 | bf16 | 19.9748 [19.8827-20.2350] | 19.3825 [19.2382-19.6238] | 97.0 | 34 / 35 | faster |
| lm_head | Q8 | dense-prefill-wmma | 2048 | bf16 | 88.3304 [84.2407-89.6081] | 84.9646 [84.0781-86.5663] | 96.2 | 8 / 8 | noise |
| lm_head | Q8 | dense-decode | 1 | f32 | 3.2040 [3.1989-3.2154] | 3.2188 [3.2072-3.2268] | 100.5 | 211 / 210 | noise |
| lm_head | Q8 | dense-decode | 2 | f32 | 3.2082 [3.1887-3.2198] | 3.2364 [3.2256-3.2518] | 100.9 | 211 / 209 | slower |
| lm_head | Q8 | dense-decode | 3 | f32 | 3.4304 [3.4228-3.4420] | 3.2438 [3.2315-3.2570] | 94.6 | 197 / 208 | faster |
| lm_head | Q8 | dense-decode | 4 | f32 | 3.5577 [3.5514-3.5772] | 3.2726 [3.2655-3.2883] | 92.0 | 190 / 206 | faster |
| lm_head | Q8 | dense-decode | 5 | f32 | 3.4854 [3.4630-3.5036] | 3.3696 [3.3594-3.3819] | 96.7 | 194 / 200 | faster |
| lm_head | Q8 | dense-decode | 6 | f32 | 3.5921 [3.5758-3.6015] | 3.4918 [3.4803-3.4959] | 97.2 | 188 / 193 | faster |
| lm_head | Q8 | dense-decode | 7 | f32 | 3.6649 [3.6462-3.6870] | 3.6326 [3.6151-3.6381] | 99.1 | 184 / 186 | faster |
| lm_head | Q8 | dense-decode | 8 | f32 | 3.7870 [3.7788-3.8081] | 3.8260 [3.8144-3.8370] | 101.0 | 178 / 177 | slower |
| lm_head | Q8 | dense-prefill-wmma | 512 | f32 | 21.7798 [19.6933-22.1854] | 21.1893 [18.9130-21.6691] | 97.3 | 31 / 32 | noise |
| lm_head | Q4 syn | dense-decode | 1 | bf16 | 1.8106 [1.8059-1.8240] | 1.7926 [1.7775-1.7991] | 99.0 | 197 / 199 | faster |
| lm_head | Q4 syn | dense-decode | 2 | bf16 | 1.8569 [1.8516-1.8604] | 1.8094 [1.8044-1.8162] | 97.4 | 193 / 198 | faster |
| lm_head | Q4 syn | dense-decode | 3 | bf16 | 2.0644 [2.0551-2.0796] | 1.9464 [1.9329-1.9577] | 94.3 | 173 / 184 | faster |
| lm_head | Q4 syn | dense-decode | 4 | bf16 | 2.1953 [2.1847-2.2155] | 2.0628 [2.0539-2.0744] | 94.0 | 163 / 173 | faster |
| lm_head | Q4 syn | dense-decode | 5 | bf16 | 2.3823 [2.3299-2.4167] | 2.2493 [2.1950-2.3092] | 94.4 | 150 / 159 | faster |
| lm_head | Q4 syn | dense-decode | 6 | bf16 | 2.4524 [2.4488-2.4612] | 2.3091 [2.2987-2.3187] | 94.2 | 146 / 155 | faster |
| lm_head | Q4 syn | dense-decode | 7 | bf16 | 2.6074 [2.5801-2.6734] | 2.4432 [2.4319-2.5372] | 93.7 | 137 / 146 | faster |
| lm_head | Q4 syn | dense-decode | 8 | bf16 | 2.8784 [2.8704-2.8952] | 2.6972 [2.6824-2.7014] | 93.7 | 124 / 133 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | bf16 | 22.0550 [21.8879-22.5243] | 21.6923 [21.5093-21.9346] | 98.4 | 16 / 16 | noise |
| lm_head | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 88.1612 [87.1029-88.5678] | 85.0602 [84.7159-86.1381] | 96.5 | 4 / 4 | faster |
| lm_head | Q4 syn | dense-decode | 1 | f32 | 1.7966 [1.7899-1.8109] | 1.7747 [1.7704-1.7856] | 98.8 | 199 / 201 | faster |
| lm_head | Q4 syn | dense-decode | 2 | f32 | 1.8521 [1.8496-1.8547] | 1.8112 [1.8081-1.8140] | 97.8 | 193 / 197 | faster |
| lm_head | Q4 syn | dense-decode | 3 | f32 | 2.0490 [2.0420-2.0551] | 1.9228 [1.9117-1.9294] | 93.8 | 175 / 186 | faster |
| lm_head | Q4 syn | dense-decode | 4 | f32 | 2.1532 [2.1497-2.1585] | 2.0048 [2.0000-2.0113] | 93.1 | 166 / 178 | faster |
| lm_head | Q4 syn | dense-decode | 5 | f32 | 2.4066 [2.4011-2.4155] | 2.2630 [2.2521-2.2697] | 94.0 | 149 / 158 | faster |
| lm_head | Q4 syn | dense-decode | 6 | f32 | 2.4982 [2.4860-2.5054] | 2.3744 [2.3730-2.3883] | 95.0 | 143 / 151 | faster |
| lm_head | Q4 syn | dense-decode | 7 | f32 | 2.6332 [2.6259-2.6477] | 2.5089 [2.4999-2.5139] | 95.3 | 136 / 143 | faster |
| lm_head | Q4 syn | dense-decode | 8 | f32 | 2.8204 [2.8126-2.8364] | 2.6665 [2.6554-2.6759] | 94.5 | 127 / 134 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | f32 | 19.5096 [19.0870-21.0169] | 19.5839 [18.6707-20.5964] | 100.4 | 18 / 18 | noise |
| mtp_fc | Q8 | dense-decode | 1 | bf16 | 0.0807 [0.0804-0.0810] | 0.0817 [0.0814-0.0818] | 101.2 | 86 / 85 | slower |
| mtp_fc | Q8 | dense-decode | 2 | bf16 | 0.0834 [0.0827-0.0838] | 0.0830 [0.0825-0.0832] | 99.6 | 84 / 84 | noise |
| mtp_fc | Q8 | dense-decode | 3 | bf16 | 0.0860 [0.0854-0.0866] | 0.0850 [0.0844-0.0856] | 98.9 | 81 / 82 | noise |
| mtp_fc | Q8 | dense-decode | 4 | bf16 | 0.0890 [0.0882-0.0899] | 0.0882 [0.0872-0.0892] | 99.2 | 78 / 79 | noise |
| mtp_fc | Q8 | dense-decode | 5 | bf16 | 0.0926 [0.0919-0.0932] | 0.0920 [0.0913-0.0924] | 99.3 | 75 / 76 | noise |
| mtp_fc | Q8 | dense-decode | 6 | bf16 | 0.0950 [0.0943-0.0952] | 0.0951 [0.0945-0.0958] | 100.2 | 73 / 73 | noise |
| mtp_fc | Q8 | dense-decode | 7 | bf16 | 0.0970 [0.0960-0.0978] | 0.0970 [0.0966-0.0979] | 100.0 | 72 / 72 | noise |
| mtp_fc | Q8 | dense-decode | 8 | bf16 | 0.0996 [0.0991-0.0999] | 0.0992 [0.0983-0.0996] | 99.6 | 70 / 70 | noise |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | bf16 | 0.2900 [0.2860-0.2938] | 0.2269 [0.2262-0.2281] | 78.3 | 24 / 31 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.8413 [0.8245-0.8505] | 0.7743 [0.7687-0.7801] | 92.0 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | bf16 | 3.0546 [3.0468-3.0806] | 2.9878 [2.9560-3.0093] | 97.8 | 2 / 2 | faster |
| mtp_fc | Q8 | dense-decode | 1 | f32 | 0.0806 [0.0803-0.0810] | 0.0818 [0.0814-0.0820] | 101.4 | 86 / 85 | slower |
| mtp_fc | Q8 | dense-decode | 2 | f32 | 0.0828 [0.0826-0.0833] | 0.0830 [0.0826-0.0837] | 100.2 | 84 / 84 | noise |
| mtp_fc | Q8 | dense-decode | 3 | f32 | 0.0860 [0.0852-0.0864] | 0.0850 [0.0848-0.0856] | 98.9 | 81 / 82 | noise |
| mtp_fc | Q8 | dense-decode | 4 | f32 | 0.0882 [0.0875-0.0886] | 0.0882 [0.0876-0.0889] | 100.0 | 79 / 79 | noise |
| mtp_fc | Q8 | dense-decode | 5 | f32 | 0.0922 [0.0915-0.0925] | 0.0921 [0.0915-0.0925] | 99.8 | 75 / 76 | noise |
| mtp_fc | Q8 | dense-decode | 6 | f32 | 0.0945 [0.0940-0.0949] | 0.0940 [0.0937-0.0944] | 99.5 | 74 / 74 | noise |
| mtp_fc | Q8 | dense-decode | 7 | f32 | 0.0963 [0.0959-0.0970] | 0.0961 [0.0954-0.0970] | 99.8 | 72 / 72 | noise |
| mtp_fc | Q8 | dense-decode | 8 | f32 | 0.0991 [0.0981-0.0995] | 0.0982 [0.0978-0.0987] | 99.1 | 70 / 71 | noise |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | f32 | 0.2925 [0.2914-0.2952] | 0.2294 [0.2288-0.2305] | 78.4 | 24 / 30 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | f32 | 0.8290 [0.8233-0.8342] | 0.7757 [0.7705-0.7833] | 93.6 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | f32 | 3.0212 [3.0068-3.0322] | 2.9571 [2.9036-2.9742] | 97.9 | 2 / 2 | faster |
| mtp_fc | Q4 syn | dense-decode | 1 | bf16 | 0.0442 [0.0439-0.0444] | 0.0438 [0.0435-0.0440] | 99.2 | 83 / 84 | noise |
| mtp_fc | Q4 syn | dense-decode | 2 | bf16 | 0.0472 [0.0465-0.0479] | 0.0456 [0.0451-0.0458] | 96.5 | 78 / 81 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | bf16 | 0.0505 [0.0487-0.0518] | 0.0468 [0.0465-0.0473] | 92.6 | 73 / 79 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | bf16 | 0.0514 [0.0508-0.0526] | 0.0481 [0.0479-0.0487] | 93.5 | 72 / 77 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | bf16 | 0.0542 [0.0532-0.0554] | 0.0526 [0.0524-0.0530] | 97.0 | 68 / 70 | faster |
| mtp_fc | Q4 syn | dense-decode | 6 | bf16 | 0.0569 [0.0556-0.0575] | 0.0536 [0.0532-0.0542] | 94.3 | 65 / 69 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | bf16 | 0.0589 [0.0580-0.0596] | 0.0554 [0.0551-0.0556] | 94.1 | 63 / 67 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | bf16 | 0.0608 [0.0602-0.0624] | 0.0576 [0.0570-0.0580] | 94.7 | 61 / 64 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.2332 [0.2318-0.2342] | 0.2173 [0.2163-0.2179] | 93.2 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.7879 [0.7757-0.7966] | 0.7732 [0.7677-0.7784] | 98.1 | 5 / 5 | noise |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 3.0522 [3.0384-3.0834] | 3.0047 [2.9730-3.0311] | 98.4 | 1 / 1 | faster |
| mtp_fc | Q4 syn | dense-decode | 1 | f32 | 0.0444 [0.0438-0.0449] | 0.0440 [0.0438-0.0442] | 99.0 | 83 / 84 | noise |
| mtp_fc | Q4 syn | dense-decode | 2 | f32 | 0.0469 [0.0467-0.0476] | 0.0456 [0.0452-0.0458] | 97.2 | 79 / 81 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | f32 | 0.0494 [0.0485-0.0504] | 0.0473 [0.0469-0.0476] | 95.6 | 75 / 78 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | f32 | 0.0510 [0.0505-0.0517] | 0.0484 [0.0481-0.0486] | 94.8 | 72 / 76 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | f32 | 0.0534 [0.0529-0.0542] | 0.0526 [0.0523-0.0530] | 98.7 | 69 / 70 | noise |
| mtp_fc | Q4 syn | dense-decode | 6 | f32 | 0.0561 [0.0549-0.0576] | 0.0536 [0.0532-0.0542] | 95.4 | 66 / 69 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | f32 | 0.0578 [0.0573-0.0591] | 0.0548 [0.0546-0.0556] | 94.9 | 64 / 67 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | f32 | 0.0606 [0.0601-0.0617] | 0.0565 [0.0562-0.0572] | 93.2 | 61 / 65 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.2329 [0.2317-0.2347] | 0.2163 [0.2160-0.2172] | 92.9 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.7758 [0.7735-0.7845] | 0.7684 [0.7649-0.7724] | 99.1 | 5 / 5 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | f32 | 2.9751 [2.9539-3.0037] | 2.9337 [2.9150-2.9452] | 98.6 | 1 / 1 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 1 | bf16 | 0.1880 [0.1874-0.1886] | 0.1862 [0.1856-0.1868] | 99.0 | 93 / 94 | faster |
| exp_gu | Q4 | expert-decode-gather | 1 | bf16 | 0.1853 [0.1848-0.1862] | 0.1847 [0.1837-0.1854] | 99.7 | 94 / 94 | noise |
| exp_down | Q4 | expert-decode-combine | 1 | bf16 | 0.1239 [0.1234-0.1242] | 0.1185 [0.1181-0.1190] | 95.7 | 83 / 86 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 2 | bf16 | 0.3368 [0.3349-0.3380] | 0.3350 [0.3344-0.3356] | 99.5 | 98 / 99 | noise |
| exp_gu | Q4 | expert-decode-gather | 2 | bf16 | 0.3364 [0.3347-0.3369] | 0.3326 [0.3316-0.3340] | 98.9 | 98 / 99 | faster |
| exp_down | Q4 | expert-decode-combine | 2 | bf16 | 0.2224 [0.2210-0.2231] | 0.2135 [0.2124-0.2141] | 96.0 | 88 / 91 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 3 | bf16 | 0.4090 [0.4076-0.4112] | 0.4069 [0.4047-0.4076] | 99.5 | 111 / 111 | faster |
| exp_gu | Q4 | expert-decode-gather | 3 | bf16 | 0.4115 [0.4095-0.4126] | 0.4074 [0.4043-0.4092] | 99.0 | 110 / 111 | faster |
| exp_down | Q4 | expert-decode-combine | 3 | bf16 | 0.2988 [0.2969-0.3000] | 0.2816 [0.2809-0.2825] | 94.2 | 89 / 95 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 4 | bf16 | 0.4762 [0.4742-0.4771] | 0.4693 [0.4651-0.4704] | 98.5 | 124 / 126 | faster |
| exp_gu | Q4 | expert-decode-gather | 4 | bf16 | 0.4796 [0.4778-0.4804] | 0.4715 [0.4706-0.4740] | 98.3 | 123 / 126 | faster |
| exp_down | Q4 | expert-decode-combine | 4 | bf16 | 0.3735 [0.3712-0.3744] | 0.3613 [0.3595-0.3623] | 96.7 | 93 / 96 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 5 | bf16 | 0.5052 [0.5024-0.5073] | 0.4937 [0.4890-0.4948] | 97.7 | 121 / 123 | faster |
| exp_gu | Q4 | expert-decode-gather | 5 | bf16 | 0.5203 [0.5192-0.5214] | 0.5059 [0.5024-0.5083] | 97.2 | 117 / 120 | faster |
| exp_down | Q4 | expert-decode-combine | 5 | bf16 | 0.3777 [0.3752-0.3782] | 0.3606 [0.3593-0.3620] | 95.5 | 95 / 99 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 6 | bf16 | 0.5805 [0.5777-0.5817] | 0.5608 [0.5553-0.5616] | 96.6 | 129 / 133 | faster |
| exp_gu | Q4 | expert-decode-gather | 6 | bf16 | 0.5948 [0.5939-0.5963] | 0.5811 [0.5799-0.5829] | 97.7 | 126 / 129 | faster |
| exp_down | Q4 | expert-decode-combine | 6 | bf16 | 0.4243 [0.4236-0.4256] | 0.4141 [0.4128-0.4153] | 97.6 | 104 / 106 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 7 | bf16 | 0.6380 [0.6357-0.6397] | 0.6274 [0.6254-0.6299] | 98.3 | 128 / 130 | faster |
| exp_gu | Q4 | expert-decode-gather | 7 | bf16 | 0.6555 [0.6531-0.6578] | 0.6436 [0.6394-0.6459] | 98.2 | 125 / 127 | faster |
| exp_down | Q4 | expert-decode-combine | 7 | bf16 | 0.4592 [0.4564-0.4606] | 0.4371 [0.4339-0.4389] | 95.2 | 105 / 110 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 8 | bf16 | 0.7272 [0.7245-0.7312] | 0.7248 [0.7188-0.7263] | 99.7 | 124 / 125 | noise |
| exp_gu | Q4 | expert-decode-gather | 8 | bf16 | 0.7414 [0.7388-0.7432] | 0.7348 [0.7338-0.7362] | 99.1 | 122 / 123 | faster |
| exp_down | Q4 | expert-decode-combine | 8 | bf16 | 0.4970 [0.4957-0.5001] | 0.4791 [0.4770-0.4810] | 96.4 | 107 / 111 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 512 | bf16 | 4.3777 [4.3712-4.3856] | 4.1373 [4.1303-4.1479] | 94.5 | 182 / 192 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 512 | bf16 | 4.4070 [4.3960-4.4232] | 4.1558 [4.1513-4.1780] | 94.3 | 181 / 191 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 512 | bf16 | 4.2699 [4.2394-4.3325] | 4.2551 [4.2237-4.3596] | 99.7 | 186 / 187 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 512 | bf16 | 2.9939 [2.9845-3.0036] | 2.7528 [2.7506-2.7563] | 92.0 | 156 / 170 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 512 | bf16 | 3.0102 [3.0000-3.0280] | 2.7729 [2.7672-2.7828] | 92.1 | 155 / 169 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 512 | bf16 | 3.0410 [3.0368-3.0503] | 2.7972 [2.7930-2.8088] | 92.0 | 154 / 167 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 512 | bf16 | 5.6839 [5.6683-5.7276] | 5.6592 [5.6330-5.6950] | 99.6 | 140 / 141 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 512 | bf16 | 5.8052 [5.7397-5.8307] | 5.7609 [5.6989-5.7870] | 99.2 | 137 / 138 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 512 | bf16 | 5.8558 [5.8329-5.8760] | 5.7469 [5.7196-5.7525] | 98.1 | 136 / 138 | faster |
| exp_down | Q4 | expert-prefill-combine wide | 512 | bf16 | 3.5952 [3.5824-3.6086] | 3.2392 [3.2298-3.2474] | 90.1 | 130 / 144 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 512 | bf16 | 3.4853 [3.4742-3.4970] | 3.1350 [3.1254-3.1473] | 90.0 | 134 / 149 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 512 | bf16 | 3.4690 [3.4609-3.4791] | 3.1113 [3.1024-3.1240] | 89.7 | 135 / 150 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 512 | bf16 | 5.6290 [5.6049-5.6502] | 5.2736 [5.2525-5.3086] | 93.7 | 141 / 151 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 512 | bf16 | 5.6063 [5.5771-5.6243] | 5.2413 [5.1988-5.2520] | 93.5 | 142 / 152 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 512 | bf16 | 5.6700 [5.6417-5.6954] | 5.3110 [5.2740-5.3300] | 93.7 | 140 / 150 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 512 | bf16 | 4.3177 [4.2791-4.3274] | 2.9835 [2.9724-2.9894] | 69.1 | 108 / 157 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 512 | bf16 | 4.3863 [4.3651-4.4158] | 2.9524 [2.9408-2.9637] | 67.3 | 107 / 158 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 512 | bf16 | 4.3960 [4.3762-4.4157] | 2.9736 [2.9634-2.9839] | 67.6 | 106 / 157 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 2048 | bf16 | 7.3656 [7.3331-7.5053] | 6.9235 [6.8803-7.0582] | 94.0 | 121 / 128 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 2048 | bf16 | 7.9958 [7.4224-8.1651] | 7.5417 [6.9848-7.6361] | 94.3 | 111 / 118 | noise |
| exp_gu | Q4 | expert-prefill-gather narrow | 2048 | bf16 | 7.7451 [7.0272-7.9702] | 7.8124 [6.9593-7.9894] | 100.9 | 115 / 114 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 2048 | bf16 | 5.3598 [5.3561-5.3831] | 4.9891 [4.9796-5.0007] | 93.1 | 98 / 105 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 2048 | bf16 | 5.3361 [5.2908-5.3529] | 4.9799 [4.9278-5.0002] | 93.3 | 98 / 105 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 2048 | bf16 | 5.3715 [5.3410-5.3945] | 5.0239 [4.9720-5.0415] | 93.5 | 97 / 104 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 2048 | bf16 | 9.0931 [8.9668-9.2200] | 9.0259 [8.9461-9.2170] | 99.3 | 98 / 99 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 2048 | bf16 | 9.0301 [8.9684-9.2115] | 9.0309 [8.9411-9.1434] | 100.0 | 99 / 98 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 2048 | bf16 | 9.2226 [8.6583-9.2730] | 9.0977 [8.1598-9.1824] | 98.7 | 96 / 98 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 2048 | bf16 | 5.7367 [5.7059-5.7577] | 5.2885 [5.2569-5.3083] | 92.2 | 91 / 99 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 2048 | bf16 | 5.6107 [5.5841-5.6322] | 5.1429 [5.1249-5.1546] | 91.7 | 93 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 2048 | bf16 | 5.6935 [5.6799-5.7050] | 5.2295 [5.1912-5.2427] | 91.8 | 92 / 100 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 2048 | bf16 | 8.5986 [8.5603-8.7853] | 8.3364 [8.2602-8.5072] | 97.0 | 103 / 107 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 2048 | bf16 | 8.6038 [8.5462-8.6744] | 8.2805 [8.2497-8.3997] | 96.2 | 103 / 107 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 2048 | bf16 | 9.1476 [8.2048-9.1848] | 8.7380 [7.3033-8.8632] | 95.5 | 97 / 102 | noise |
| exp_down | Q4 | expert-prefill-combine wide256 | 2048 | bf16 | 6.1795 [6.1512-6.1952] | 5.1766 [5.1430-5.1959] | 83.8 | 85 / 101 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 2048 | bf16 | 6.1303 [6.1049-6.1743] | 5.1534 [5.0975-5.1720] | 84.1 | 85 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 2048 | bf16 | 6.1634 [6.1332-6.1984] | 5.1723 [5.1139-5.2013] | 83.9 | 85 / 101 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 8192 | bf16 | 23.0441 [22.6660-23.3285] | 22.5633 [22.1123-22.8866] | 97.9 | 39 / 40 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 8192 | bf16 | 23.1005 [22.8257-23.3184] | 22.3187 [21.9437-22.5411] | 96.6 | 39 / 40 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 8192 | bf16 | 22.3927 [22.1168-22.4807] | 22.6434 [22.4272-23.0419] | 101.1 | 40 / 39 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 8192 | bf16 | 15.2303 [15.0936-15.5679] | 14.7769 [14.5010-15.0529] | 97.0 | 34 / 35 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 8192 | bf16 | 15.0428 [14.8728-15.2378] | 14.4995 [14.3722-14.7085] | 96.4 | 35 / 36 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 8192 | bf16 | 15.0023 [14.8996-15.3427] | 14.5332 [14.3874-14.7883] | 96.9 | 35 / 36 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 8192 | bf16 | 21.7482 [21.6642-21.9511] | 21.7423 [21.6110-21.9315] | 100.0 | 41 / 41 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 8192 | bf16 | 21.6483 [21.5866-21.9094] | 21.6920 [21.5790-22.0123] | 100.2 | 41 / 41 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 8192 | bf16 | 20.8440 [20.6235-21.1360] | 20.7499 [20.6422-20.9937] | 99.5 | 43 / 43 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 8192 | bf16 | 14.0454 [13.8705-14.1197] | 13.4523 [13.1585-13.5099] | 95.8 | 37 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 8192 | bf16 | 13.8902 [13.7198-13.9585] | 13.2289 [13.0009-13.3020] | 95.2 | 38 / 40 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 8192 | bf16 | 14.1286 [13.9050-14.2319] | 13.4569 [13.1479-13.5561] | 95.2 | 37 / 39 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 8192 | bf16 | 20.6828 [20.4035-20.9766] | 20.1587 [19.9024-20.5320] | 97.5 | 43 / 44 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 8192 | bf16 | 20.7063 [20.1780-21.2614] | 20.1278 [19.6385-20.7953] | 97.2 | 43 / 44 | noise |
| exp_gu | Q4 | expert-prefill-gather wide256 | 8192 | bf16 | 20.8478 [20.4964-21.2874] | 19.8721 [19.7227-20.2979] | 95.3 | 43 / 45 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 8192 | bf16 | 15.0243 [14.8662-15.2871] | 13.5488 [13.3933-13.7150] | 90.2 | 35 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 8192 | bf16 | 15.5717 [15.4250-15.6753] | 14.1088 [14.0283-14.1521] | 90.6 | 34 / 37 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 8192 | bf16 | 15.1830 [15.0913-15.2609] | 13.7291 [13.6437-13.8052] | 90.4 | 35 / 38 | faster |
