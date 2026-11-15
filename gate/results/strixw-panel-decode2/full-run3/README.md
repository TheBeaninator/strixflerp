# strixw panel kernels: kernel gate results

Criteria: `gate/strixw-panel-kernels.toml` (committed before the run). Raw: `raw.jsonl` (one case a line), bench log `bench.log`. 489 cases, 6 skipped (output too large), 2.232 GB read from the fleet file.

## Verdicts

| criterion | pass | detail |
|---|---|---|
| equal | PASS | 489 of 489 cases byte-equal |
| bad_id | PASS | 21 of 21 out-of-range-id cases byte-equal |
| coverage | PASS | missing: none |
| read | FAIL | 2.232 GB read |
| speed_decode_dense | PASS | median panel % of origin time per class: attn_o 98.9, attn_qkv 99.2, gdn_in 97.5, gdn_out 98.5, lm_head 96.0, mtp_fc 98.0, ple_kv 99.8, sh_down 89.0, sh_gu 95.3 |
| speed_prefill_dense | PASS | sh_gu M2048 bf16 90.6%; sh_gu M8192 bf16 96.3%; sh_gu M2048 f32 91.3%; sh_gu M8192 f32 97.2%; sh_down M2048 bf16 94.2%; sh_down M8192 bf16 95.5%; sh_down M2048 f32 96.0%; sh_down M8192 f32 95.6%; gdn_in M2048 bf16 98.5%; gdn_in M8192 bf16 98.7%; gdn_in M2048 f32 97.0%; gdn_in M8192 f32 99.5%; gdn_out M2048 bf16 71.3%; gdn_out M8192 bf16 91.8%; gdn_out M2048 f32 67.5%; gdn_out M8192 f32 90.2%; attn_qkv M2048 bf16 94.3%; attn_qkv M8192 bf16 99.1%; attn_qkv M2048 f32 94.0%; attn_qkv M8192 f32 97.4%; attn_o M2048 bf16 72.7%; attn_o M8192 bf16 90.0%; attn_o M2048 f32 69.2%; attn_o M8192 f32 89.8%; ple_kv M2048 bf16 95.1%; ple_kv M8192 bf16 97.4%; ple_kv M2048 f32 92.6%; ple_kv M8192 f32 96.5%; lm_head M2048 bf16 96.8%; mtp_fc M2048 bf16 91.8%; mtp_fc M8192 bf16 97.3%; mtp_fc M2048 f32 93.5%; mtp_fc M8192 f32 99.6% |
| speed_prefill_expert | PASS | expert-prefill-swiglu-f16h narrow M2048 93.6% (faster); expert-prefill-combine-f16h-shared narrow M2048 93.7% (faster); expert-prefill-combine-f16h-shared wide M8192 94.4% (faster); expert-prefill-swiglu-f16h wide256 M8192 97.6% (noise) |

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
| sh_gu | Q8 | dense-decode | 1 | bf16 | 0.0426 [0.0422-0.0432] | 0.0427 [0.0425-0.0430] | 100.2 | 82 / 82 | noise |
| sh_gu | Q8 | dense-decode | 2 | bf16 | 0.0436 [0.0430-0.0439] | 0.0433 [0.0432-0.0437] | 99.5 | 80 / 80 | noise |
| sh_gu | Q8 | dense-decode | 3 | bf16 | 0.0454 [0.0448-0.0457] | 0.0444 [0.0443-0.0447] | 97.8 | 77 / 78 | faster |
| sh_gu | Q8 | dense-decode | 4 | bf16 | 0.0474 [0.0467-0.0478] | 0.0461 [0.0458-0.0465] | 97.3 | 73 / 75 | faster |
| sh_gu | Q8 | dense-decode | 5 | bf16 | 0.0498 [0.0493-0.0500] | 0.0473 [0.0470-0.0481] | 95.0 | 70 / 74 | faster |
| sh_gu | Q8 | dense-decode | 6 | bf16 | 0.0513 [0.0510-0.0515] | 0.0497 [0.0494-0.0503] | 96.9 | 68 / 70 | faster |
| sh_gu | Q8 | dense-decode | 7 | bf16 | 0.0536 [0.0528-0.0547] | 0.0517 [0.0507-0.0529] | 96.5 | 65 / 67 | noise |
| sh_gu | Q8 | dense-decode | 8 | bf16 | 0.0544 [0.0540-0.0551] | 0.0516 [0.0512-0.0520] | 94.8 | 64 / 68 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 512 | bf16 | 0.1443 [0.1402-0.1475] | 0.1310 [0.1231-0.1330] | 90.8 | 24 / 27 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.4646 [0.4471-0.4763] | 0.4210 [0.4132-0.4292] | 90.6 | 7 / 8 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | bf16 | 1.6536 [1.6341-1.6739] | 1.5928 [1.5766-1.6097] | 96.3 | 2 / 2 | faster |
| sh_gu | Q8 | dense-decode | 1 | f32 | 0.0415 [0.0412-0.0422] | 0.0421 [0.0419-0.0423] | 101.5 | 84 / 83 | noise |
| sh_gu | Q8 | dense-decode | 2 | f32 | 0.0431 [0.0427-0.0436] | 0.0431 [0.0428-0.0434] | 99.9 | 81 / 81 | noise |
| sh_gu | Q8 | dense-decode | 3 | f32 | 0.0452 [0.0446-0.0458] | 0.0440 [0.0438-0.0443] | 97.3 | 77 / 79 | faster |
| sh_gu | Q8 | dense-decode | 4 | f32 | 0.0477 [0.0470-0.0482] | 0.0462 [0.0453-0.0472] | 97.0 | 73 / 75 | noise |
| sh_gu | Q8 | dense-decode | 5 | f32 | 0.0502 [0.0496-0.0513] | 0.0486 [0.0478-0.0491] | 96.8 | 69 / 72 | faster |
| sh_gu | Q8 | dense-decode | 6 | f32 | 0.0510 [0.0507-0.0513] | 0.0492 [0.0485-0.0501] | 96.3 | 68 / 71 | faster |
| sh_gu | Q8 | dense-decode | 7 | f32 | 0.0542 [0.0532-0.0557] | 0.0520 [0.0510-0.0531] | 96.0 | 64 / 67 | faster |
| sh_gu | Q8 | dense-decode | 8 | f32 | 0.0557 [0.0541-0.0566] | 0.0530 [0.0523-0.0536] | 95.3 | 63 / 66 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 512 | f32 | 0.1480 [0.1441-0.1526] | 0.1345 [0.1276-0.1367] | 90.8 | 24 / 26 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | f32 | 0.4662 [0.4596-0.4773] | 0.4258 [0.4203-0.4382] | 91.3 | 7 / 8 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | f32 | 1.6504 [1.6393-1.6621] | 1.6047 [1.5909-1.6077] | 97.2 | 2 / 2 | faster |
| sh_gu | Q4 syn | dense-decode | 1 | bf16 | 0.0266 [0.0264-0.0269] | 0.0254 [0.0252-0.0258] | 95.3 | 69 / 73 | faster |
| sh_gu | Q4 syn | dense-decode | 2 | bf16 | 0.0284 [0.0279-0.0304] | 0.0265 [0.0264-0.0268] | 93.4 | 65 / 70 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | bf16 | 0.0302 [0.0292-0.0310] | 0.0280 [0.0272-0.0283] | 92.5 | 61 / 66 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | bf16 | 0.0315 [0.0304-0.0327] | 0.0284 [0.0282-0.0288] | 90.2 | 59 / 65 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | bf16 | 0.0345 [0.0336-0.0355] | 0.0315 [0.0296-0.0322] | 91.3 | 53 / 58 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | bf16 | 0.0372 [0.0368-0.0386] | 0.0332 [0.0326-0.0337] | 89.0 | 49 / 56 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | bf16 | 0.0406 [0.0401-0.0412] | 0.0351 [0.0346-0.0354] | 86.4 | 45 / 52 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | bf16 | 0.0422 [0.0415-0.0434] | 0.0360 [0.0357-0.0364] | 85.1 | 44 / 51 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.1247 [0.1242-0.1266] | 0.1210 [0.1190-0.1220] | 97.0 | 15 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.4248 [0.4135-0.4336] | 0.4103 [0.4025-0.4153] | 96.6 | 4 / 4 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 1.6218 [1.5835-1.6446] | 1.5905 [1.5579-1.6010] | 98.1 | 1 / 1 | noise |
| sh_gu | Q4 syn | dense-decode | 1 | f32 | 0.0257 [0.0254-0.0262] | 0.0254 [0.0251-0.0256] | 98.8 | 72 / 73 | noise |
| sh_gu | Q4 syn | dense-decode | 2 | f32 | 0.0280 [0.0273-0.0287] | 0.0262 [0.0261-0.0267] | 93.8 | 66 / 70 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | f32 | 0.0304 [0.0289-0.0315] | 0.0273 [0.0270-0.0279] | 89.7 | 61 / 68 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | f32 | 0.0310 [0.0299-0.0322] | 0.0286 [0.0281-0.0288] | 92.3 | 59 / 64 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | f32 | 0.0328 [0.0313-0.0334] | 0.0297 [0.0292-0.0303] | 90.7 | 56 / 62 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | f32 | 0.0329 [0.0326-0.0340] | 0.0304 [0.0302-0.0309] | 92.6 | 56 / 61 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | f32 | 0.0337 [0.0332-0.0352] | 0.0316 [0.0310-0.0320] | 93.6 | 55 / 58 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | f32 | 0.0353 [0.0348-0.0361] | 0.0325 [0.0323-0.0327] | 92.1 | 52 / 57 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.1289 [0.1275-0.1308] | 0.1247 [0.1240-0.1257] | 96.7 | 14 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.4257 [0.4211-0.4303] | 0.4172 [0.4130-0.4197] | 98.0 | 4 / 4 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | f32 | 1.5036 [1.4770-1.5129] | 1.4782 [1.4711-1.4941] | 98.3 | 1 / 1 | noise |
| sh_down | Q8 | dense-decode | 1 | bf16 | 0.0254 [0.0252-0.0257] | 0.0253 [0.0250-0.0254] | 99.5 | 73 / 73 | noise |
| sh_down | Q8 | dense-decode | 2 | bf16 | 0.0272 [0.0267-0.0277] | 0.0263 [0.0260-0.0266] | 96.8 | 68 / 70 | faster |
| sh_down | Q8 | dense-decode | 3 | bf16 | 0.0287 [0.0285-0.0292] | 0.0272 [0.0269-0.0274] | 95.0 | 64 / 68 | faster |
| sh_down | Q8 | dense-decode | 4 | bf16 | 0.0308 [0.0301-0.0314] | 0.0274 [0.0272-0.0278] | 89.0 | 60 / 67 | faster |
| sh_down | Q8 | dense-decode | 5 | bf16 | 0.0336 [0.0331-0.0340] | 0.0289 [0.0286-0.0294] | 86.0 | 55 / 64 | faster |
| sh_down | Q8 | dense-decode | 6 | bf16 | 0.0338 [0.0335-0.0343] | 0.0295 [0.0291-0.0301] | 87.3 | 55 / 63 | faster |
| sh_down | Q8 | dense-decode | 7 | bf16 | 0.0340 [0.0338-0.0348] | 0.0304 [0.0302-0.0308] | 89.5 | 54 / 61 | faster |
| sh_down | Q8 | dense-decode | 8 | bf16 | 0.0355 [0.0346-0.0360] | 0.0317 [0.0313-0.0322] | 89.4 | 52 / 58 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | bf16 | 0.0776 [0.0774-0.0780] | 0.0705 [0.0702-0.0708] | 90.8 | 24 / 26 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.2474 [0.2428-0.2505] | 0.2331 [0.2263-0.2352] | 94.2 | 7 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | bf16 | 0.8393 [0.8309-0.8436] | 0.8014 [0.7954-0.8049] | 95.5 | 2 / 2 | faster |
| sh_down | Q8 | dense-decode | 1 | f32 | 0.0254 [0.0250-0.0256] | 0.0252 [0.0250-0.0253] | 99.4 | 73 / 73 | noise |
| sh_down | Q8 | dense-decode | 2 | f32 | 0.0270 [0.0269-0.0274] | 0.0260 [0.0256-0.0265] | 96.3 | 68 / 71 | faster |
| sh_down | Q8 | dense-decode | 3 | f32 | 0.0285 [0.0280-0.0286] | 0.0269 [0.0265-0.0273] | 94.4 | 65 / 69 | faster |
| sh_down | Q8 | dense-decode | 4 | f32 | 0.0299 [0.0294-0.0302] | 0.0273 [0.0272-0.0274] | 91.3 | 62 / 68 | faster |
| sh_down | Q8 | dense-decode | 5 | f32 | 0.0322 [0.0319-0.0328] | 0.0288 [0.0286-0.0291] | 89.4 | 57 / 64 | faster |
| sh_down | Q8 | dense-decode | 6 | f32 | 0.0340 [0.0337-0.0346] | 0.0296 [0.0293-0.0300] | 87.3 | 54 / 62 | faster |
| sh_down | Q8 | dense-decode | 7 | f32 | 0.0374 [0.0368-0.0384] | 0.0311 [0.0308-0.0322] | 83.1 | 49 / 59 | faster |
| sh_down | Q8 | dense-decode | 8 | f32 | 0.0401 [0.0392-0.0405] | 0.0326 [0.0324-0.0330] | 81.4 | 46 / 56 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | f32 | 0.0857 [0.0847-0.0869] | 0.0782 [0.0774-0.0788] | 91.2 | 22 / 24 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | f32 | 0.2542 [0.2491-0.2591] | 0.2439 [0.2394-0.2458] | 96.0 | 7 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | f32 | 0.8938 [0.8820-0.9017] | 0.8545 [0.8387-0.8656] | 95.6 | 2 / 2 | faster |
| sh_down | Q4 syn | dense-decode | 1 | bf16 | 0.0169 [0.0168-0.0171] | 0.0170 [0.0166-0.0172] | 100.5 | 61 / 60 | noise |
| sh_down | Q4 syn | dense-decode | 2 | bf16 | 0.0206 [0.0203-0.0209] | 0.0180 [0.0178-0.0183] | 87.7 | 50 / 57 | faster |
| sh_down | Q4 syn | dense-decode | 3 | bf16 | 0.0227 [0.0224-0.0230] | 0.0190 [0.0187-0.0196] | 83.8 | 45 / 54 | faster |
| sh_down | Q4 syn | dense-decode | 4 | bf16 | 0.0242 [0.0237-0.0246] | 0.0194 [0.0192-0.0201] | 80.3 | 42 / 53 | faster |
| sh_down | Q4 syn | dense-decode | 5 | bf16 | 0.0274 [0.0272-0.0280] | 0.0210 [0.0209-0.0214] | 76.9 | 37 / 49 | faster |
| sh_down | Q4 syn | dense-decode | 6 | bf16 | 0.0284 [0.0282-0.0288] | 0.0217 [0.0213-0.0220] | 76.5 | 36 / 47 | faster |
| sh_down | Q4 syn | dense-decode | 7 | bf16 | 0.0312 [0.0308-0.0321] | 0.0234 [0.0233-0.0236] | 75.0 | 33 / 44 | faster |
| sh_down | Q4 syn | dense-decode | 8 | bf16 | 0.0296 [0.0293-0.0301] | 0.0228 [0.0225-0.0231] | 77.2 | 35 / 45 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.0736 [0.0734-0.0739] | 0.0673 [0.0671-0.0674] | 91.4 | 14 / 15 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.2412 [0.2386-0.2455] | 0.2257 [0.2176-0.2271] | 93.6 | 4 / 5 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 0.8966 [0.8941-0.9008] | 0.8597 [0.8564-0.8691] | 95.9 | 1 / 1 | faster |
| sh_down | Q4 syn | dense-decode | 1 | f32 | 0.0166 [0.0164-0.0168] | 0.0168 [0.0166-0.0168] | 101.2 | 62 / 61 | noise |
| sh_down | Q4 syn | dense-decode | 2 | f32 | 0.0188 [0.0185-0.0189] | 0.0178 [0.0174-0.0181] | 94.5 | 54 / 58 | faster |
| sh_down | Q4 syn | dense-decode | 3 | f32 | 0.0199 [0.0197-0.0202] | 0.0182 [0.0179-0.0183] | 91.2 | 51 / 56 | faster |
| sh_down | Q4 syn | dense-decode | 4 | f32 | 0.0206 [0.0205-0.0209] | 0.0190 [0.0188-0.0192] | 92.0 | 50 / 54 | faster |
| sh_down | Q4 syn | dense-decode | 5 | f32 | 0.0228 [0.0224-0.0238] | 0.0202 [0.0199-0.0202] | 88.3 | 45 / 51 | faster |
| sh_down | Q4 syn | dense-decode | 6 | f32 | 0.0254 [0.0249-0.0264] | 0.0210 [0.0208-0.0211] | 82.5 | 40 / 49 | faster |
| sh_down | Q4 syn | dense-decode | 7 | f32 | 0.0278 [0.0276-0.0288] | 0.0219 [0.0218-0.0222] | 78.6 | 37 / 47 | faster |
| sh_down | Q4 syn | dense-decode | 8 | f32 | 0.0293 [0.0259-0.0300] | 0.0230 [0.0226-0.0232] | 78.7 | 35 / 44 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.0811 [0.0808-0.0820] | 0.0735 [0.0731-0.0737] | 90.6 | 13 / 14 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.2574 [0.2532-0.2632] | 0.2378 [0.2338-0.2418] | 92.4 | 4 / 4 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | f32 | 0.8907 [0.8720-0.8970] | 0.8427 [0.8344-0.8540] | 94.6 | 1 / 1 | faster |
| gdn_in | Q4 | dense-decode | 1 | bf16 | 0.2357 [0.2347-0.2381] | 0.2341 [0.2325-0.2357] | 99.3 | 95 / 96 | noise |
| gdn_in | Q4 | dense-decode | 2 | bf16 | 0.2384 [0.2368-0.2428] | 0.2348 [0.2331-0.2360] | 98.5 | 94 / 95 | faster |
| gdn_in | Q4 | dense-decode | 3 | bf16 | 0.2409 [0.2380-0.2436] | 0.2342 [0.2333-0.2383] | 97.2 | 93 / 96 | noise |
| gdn_in | Q4 | dense-decode | 4 | bf16 | 0.2454 [0.2434-0.2469] | 0.2389 [0.2379-0.2405] | 97.4 | 91 / 94 | faster |
| gdn_in | Q4 | dense-decode | 5 | bf16 | 0.2546 [0.2529-0.2558] | 0.2460 [0.2416-0.2481] | 96.6 | 88 / 91 | faster |
| gdn_in | Q4 | dense-decode | 6 | bf16 | 0.2589 [0.2581-0.2608] | 0.2501 [0.2471-0.2528] | 96.6 | 87 / 90 | faster |
| gdn_in | Q4 | dense-decode | 7 | bf16 | 0.2655 [0.2638-0.2688] | 0.2527 [0.2500-0.2554] | 95.2 | 84 / 89 | faster |
| gdn_in | Q4 | dense-decode | 8 | bf16 | 0.2701 [0.2677-0.2715] | 0.2532 [0.2520-0.2576] | 93.8 | 83 / 88 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | bf16 | 1.2946 [1.2910-1.3113] | 1.2632 [1.2533-1.2733] | 97.6 | 17 / 18 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | bf16 | 5.3705 [5.3325-5.4132] | 5.2895 [5.2636-5.3586] | 98.5 | 4 / 4 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | bf16 | 21.9788 [20.0686-22.1976] | 21.6953 [19.8238-22.0033] | 98.7 | 1 / 1 | noise |
| gdn_in | Q4 | dense-decode | 1 | f32 | 0.2381 [0.2355-0.2396] | 0.2351 [0.2326-0.2372] | 98.8 | 94 / 95 | noise |
| gdn_in | Q4 | dense-decode | 2 | f32 | 0.2417 [0.2392-0.2439] | 0.2356 [0.2333-0.2385] | 97.5 | 93 / 95 | faster |
| gdn_in | Q4 | dense-decode | 3 | f32 | 0.2441 [0.2430-0.2458] | 0.2406 [0.2395-0.2416] | 98.6 | 92 / 93 | faster |
| gdn_in | Q4 | dense-decode | 4 | f32 | 0.2429 [0.2402-0.2463] | 0.2403 [0.2367-0.2445] | 98.9 | 92 / 93 | noise |
| gdn_in | Q4 | dense-decode | 5 | f32 | 0.2524 [0.2501-0.2569] | 0.2426 [0.2408-0.2454] | 96.1 | 89 / 92 | faster |
| gdn_in | Q4 | dense-decode | 6 | f32 | 0.2598 [0.2563-0.2624] | 0.2498 [0.2485-0.2522] | 96.1 | 86 / 90 | faster |
| gdn_in | Q4 | dense-decode | 7 | f32 | 0.2604 [0.2567-0.2628] | 0.2490 [0.2480-0.2529] | 95.6 | 86 / 90 | faster |
| gdn_in | Q4 | dense-decode | 8 | f32 | 0.2690 [0.2672-0.2720] | 0.2550 [0.2537-0.2585] | 94.8 | 83 / 88 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | f32 | 1.1865 [1.1805-1.1939] | 1.1573 [1.1482-1.1669] | 97.5 | 19 / 19 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | f32 | 5.1933 [4.8566-5.6518] | 5.0348 [4.7971-5.5886] | 97.0 | 4 / 4 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | f32 | 21.5332 [21.3043-21.8132] | 21.4143 [21.0664-21.6738] | 99.5 | 1 / 1 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | bf16 | 0.3888 [0.3864-0.3926] | 0.3945 [0.3905-0.3976] | 101.5 | 112 / 110 | noise |
| gdn_in | Q8 syn | dense-decode | 2 | bf16 | 0.3904 [0.3887-0.3927] | 0.3961 [0.3948-0.4001] | 101.5 | 111 / 110 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | bf16 | 0.3998 [0.3988-0.4034] | 0.4018 [0.3999-0.4033] | 100.5 | 109 / 108 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | bf16 | 0.4086 [0.4062-0.4109] | 0.4073 [0.4051-0.4088] | 99.7 | 106 / 107 | noise |
| gdn_in | Q8 syn | dense-decode | 5 | bf16 | 0.4279 [0.4270-0.4302] | 0.4158 [0.4129-0.4189] | 97.2 | 102 / 105 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | bf16 | 0.4306 [0.4285-0.4378] | 0.4214 [0.4184-0.4279] | 97.8 | 101 / 103 | faster |
| gdn_in | Q8 syn | dense-decode | 7 | bf16 | 0.4417 [0.4391-0.4458] | 0.4276 [0.4267-0.4316] | 96.8 | 98 / 102 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | bf16 | 0.4493 [0.4472-0.4550] | 0.4366 [0.4326-0.4456] | 97.2 | 97 / 100 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | bf16 | 1.5959 [1.5664-1.6269] | 1.3326 [1.3291-1.3517] | 83.5 | 27 / 33 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | bf16 | 5.5755 [5.5442-5.6532] | 5.3301 [5.2848-5.4108] | 95.6 | 8 / 8 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | bf16 | 22.0087 [20.8885-23.0539] | 22.1948 [20.4113-22.6054] | 100.8 | 2 / 2 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | f32 | 0.3888 [0.3876-0.3910] | 0.3928 [0.3907-0.3966] | 101.0 | 112 / 111 | noise |
| gdn_in | Q8 syn | dense-decode | 2 | f32 | 0.3938 [0.3912-0.3951] | 0.3988 [0.3975-0.4000] | 101.3 | 110 / 109 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | f32 | 0.4008 [0.3983-0.4024] | 0.4015 [0.3998-0.4038] | 100.2 | 109 / 108 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | f32 | 0.4087 [0.4068-0.4118] | 0.4075 [0.4043-0.4087] | 99.7 | 106 / 107 | noise |
| gdn_in | Q8 syn | dense-decode | 5 | f32 | 0.4327 [0.4296-0.4364] | 0.4209 [0.4174-0.4255] | 97.3 | 101 / 103 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | f32 | 0.4480 [0.4440-0.4556] | 0.4353 [0.4278-0.4461] | 97.2 | 97 / 100 | noise |
| gdn_in | Q8 syn | dense-decode | 7 | f32 | 0.4514 [0.4462-0.4551] | 0.4394 [0.4325-0.4437] | 97.3 | 96 / 99 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | f32 | 0.4511 [0.4467-0.4565] | 0.4398 [0.4363-0.4439] | 97.5 | 96 / 99 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | f32 | 1.5660 [1.5454-1.5905] | 1.3237 [1.3181-1.3368] | 84.5 | 28 / 33 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | f32 | 5.3710 [5.1323-6.0264] | 5.2719 [4.9330-5.8100] | 98.2 | 8 / 8 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | f32 | 22.3957 [21.5316-22.6621] | 22.1271 [20.0717-22.3318] | 98.8 | 2 / 2 | noise |
| gdn_out | Q8 | dense-decode | 1 | bf16 | 0.1830 [0.1810-0.1864] | 0.1842 [0.1831-0.1872] | 100.7 | 91 / 91 | noise |
| gdn_out | Q8 | dense-decode | 2 | bf16 | 0.1850 [0.1806-0.1872] | 0.1843 [0.1814-0.1869] | 99.6 | 90 / 91 | noise |
| gdn_out | Q8 | dense-decode | 3 | bf16 | 0.1824 [0.1788-0.1837] | 0.1838 [0.1832-0.1848] | 100.8 | 92 / 91 | noise |
| gdn_out | Q8 | dense-decode | 4 | bf16 | 0.1903 [0.1833-0.1932] | 0.1886 [0.1881-0.1898] | 99.1 | 88 / 89 | noise |
| gdn_out | Q8 | dense-decode | 5 | bf16 | 0.2062 [0.2030-0.2112] | 0.2055 [0.2038-0.2116] | 99.7 | 81 / 81 | noise |
| gdn_out | Q8 | dense-decode | 6 | bf16 | 0.2218 [0.2145-0.2246] | 0.2138 [0.2118-0.2173] | 96.4 | 75 / 78 | noise |
| gdn_out | Q8 | dense-decode | 7 | bf16 | 0.2312 [0.2241-0.2353] | 0.2181 [0.2148-0.2197] | 94.3 | 72 / 77 | faster |
| gdn_out | Q8 | dense-decode | 8 | bf16 | 0.2404 [0.2337-0.2446] | 0.2227 [0.2210-0.2239] | 92.6 | 70 / 75 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 512 | bf16 | 1.2693 [1.2094-1.3753] | 0.7889 [0.7530-0.8041] | 62.1 | 13 / 21 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.1564 [3.1096-3.2240] | 2.2491 [2.1883-2.2878] | 71.3 | 5 / 7 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.8208 [8.6975-8.9034] | 8.1002 [8.0177-8.2765] | 91.8 | 2 / 2 | faster |
| gdn_out | Q8 | dense-decode | 1 | f32 | 0.1812 [0.1778-0.1832] | 0.1812 [0.1784-0.1839] | 100.0 | 92 / 92 | noise |
| gdn_out | Q8 | dense-decode | 2 | f32 | 0.1822 [0.1808-0.1838] | 0.1810 [0.1801-0.1818] | 99.3 | 92 / 92 | noise |
| gdn_out | Q8 | dense-decode | 3 | f32 | 0.1843 [0.1806-0.1856] | 0.1852 [0.1839-0.1879] | 100.5 | 91 / 90 | noise |
| gdn_out | Q8 | dense-decode | 4 | f32 | 0.1915 [0.1896-0.1943] | 0.1934 [0.1926-0.1946] | 101.0 | 87 / 86 | noise |
| gdn_out | Q8 | dense-decode | 5 | f32 | 0.2048 [0.1989-0.2097] | 0.2065 [0.2048-0.2100] | 100.8 | 82 / 81 | noise |
| gdn_out | Q8 | dense-decode | 6 | f32 | 0.2142 [0.2062-0.2171] | 0.2097 [0.2068-0.2110] | 97.9 | 78 / 80 | noise |
| gdn_out | Q8 | dense-decode | 7 | f32 | 0.2241 [0.2192-0.2295] | 0.2129 [0.2112-0.2141] | 95.0 | 75 / 78 | faster |
| gdn_out | Q8 | dense-decode | 8 | f32 | 0.2305 [0.2260-0.2330] | 0.2178 [0.2152-0.2204] | 94.5 | 72 / 77 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 512 | f32 | 1.3201 [1.2034-1.5097] | 0.7679 [0.7235-0.7828] | 58.2 | 13 / 22 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | f32 | 3.1123 [3.0467-3.1957] | 2.0998 [2.0739-2.1480] | 67.5 | 5 / 8 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | f32 | 8.9936 [8.4103-9.2955] | 8.1082 [7.5111-8.2557] | 90.2 | 2 / 2 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | bf16 | 0.1019 [0.1016-0.1026] | 0.1022 [0.1009-0.1025] | 100.2 | 87 / 87 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | bf16 | 0.1035 [0.0992-0.1055] | 0.1013 [0.0989-0.1037] | 97.8 | 85 / 87 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | bf16 | 0.1082 [0.1074-0.1098] | 0.1066 [0.1044-0.1078] | 98.5 | 82 / 83 | noise |
| gdn_out | Q4 syn | dense-decode | 4 | bf16 | 0.1121 [0.1112-0.1132] | 0.1088 [0.1077-0.1092] | 97.0 | 79 / 81 | faster |
| gdn_out | Q4 syn | dense-decode | 5 | bf16 | 0.1153 [0.1139-0.1176] | 0.1156 [0.1127-0.1167] | 100.2 | 77 / 77 | noise |
| gdn_out | Q4 syn | dense-decode | 6 | bf16 | 0.1234 [0.1226-0.1240] | 0.1192 [0.1181-0.1212] | 96.5 | 72 / 74 | faster |
| gdn_out | Q4 syn | dense-decode | 7 | bf16 | 0.1263 [0.1254-0.1286] | 0.1213 [0.1196-0.1231] | 96.0 | 70 / 73 | faster |
| gdn_out | Q4 syn | dense-decode | 8 | bf16 | 0.1312 [0.1277-0.1346] | 0.1257 [0.1230-0.1273] | 95.8 | 67 / 70 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0055 [0.9683-1.0224] | 0.6076 [0.5845-0.6367] | 60.4 | 9 / 15 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.6137 [2.5566-2.6683] | 2.0043 [1.9448-2.0872] | 76.7 | 3 / 4 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 8.2041 [8.1016-8.2725] | 7.7260 [7.6702-7.8266] | 94.2 | 1 / 1 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | f32 | 0.1012 [0.0963-0.1022] | 0.1008 [0.1002-0.1027] | 99.7 | 87 / 88 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | f32 | 0.1039 [0.1032-0.1047] | 0.1030 [0.1017-0.1040] | 99.1 | 85 / 86 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | f32 | 0.1051 [0.1036-0.1081] | 0.1046 [0.1029-0.1074] | 99.5 | 84 / 85 | noise |
| gdn_out | Q4 syn | dense-decode | 4 | f32 | 0.1070 [0.1068-0.1080] | 0.1050 [0.1030-0.1055] | 98.1 | 83 / 84 | faster |
| gdn_out | Q4 syn | dense-decode | 5 | f32 | 0.1171 [0.1145-0.1179] | 0.1146 [0.1115-0.1160] | 97.8 | 76 / 77 | noise |
| gdn_out | Q4 syn | dense-decode | 6 | f32 | 0.1208 [0.1189-0.1224] | 0.1184 [0.1158-0.1197] | 98.0 | 73 / 75 | noise |
| gdn_out | Q4 syn | dense-decode | 7 | f32 | 0.1261 [0.1249-0.1290] | 0.1202 [0.1192-0.1215] | 95.4 | 70 / 74 | faster |
| gdn_out | Q4 syn | dense-decode | 8 | f32 | 0.1314 [0.1308-0.1336] | 0.1250 [0.1240-0.1260] | 95.2 | 67 / 71 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0118 [1.0036-1.0223] | 0.5933 [0.5868-0.6027] | 58.6 | 9 / 15 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.4983 [2.4720-2.5838] | 1.9436 [1.9371-1.9692] | 77.8 | 4 / 5 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | f32 | 7.9112 [7.7921-8.0627] | 7.3206 [7.2127-7.4861] | 92.5 | 1 / 1 | faster |
| attn_qkv | Q8 | dense-decode | 1 | bf16 | 0.3677 [0.3663-0.3777] | 0.3710 [0.3676-0.3792] | 100.9 | 103 / 102 | noise |
| attn_qkv | Q8 | dense-decode | 2 | bf16 | 0.3664 [0.3643-0.3686] | 0.3721 [0.3699-0.3733] | 101.6 | 104 / 102 | slower |
| attn_qkv | Q8 | dense-decode | 3 | bf16 | 0.3726 [0.3717-0.3757] | 0.3752 [0.3718-0.3764] | 100.7 | 102 / 101 | noise |
| attn_qkv | Q8 | dense-decode | 4 | bf16 | 0.3796 [0.3768-0.3830] | 0.3817 [0.3800-0.3844] | 100.5 | 100 / 99 | noise |
| attn_qkv | Q8 | dense-decode | 5 | bf16 | 0.3973 [0.3946-0.4002] | 0.3895 [0.3873-0.3941] | 98.0 | 96 / 97 | faster |
| attn_qkv | Q8 | dense-decode | 6 | bf16 | 0.4068 [0.4024-0.4087] | 0.3998 [0.3967-0.4033] | 98.3 | 93 / 95 | noise |
| attn_qkv | Q8 | dense-decode | 7 | bf16 | 0.4116 [0.4092-0.4168] | 0.4085 [0.4032-0.4126] | 99.2 | 92 / 93 | noise |
| attn_qkv | Q8 | dense-decode | 8 | bf16 | 0.4154 [0.4128-0.4187] | 0.4151 [0.4088-0.4166] | 99.9 | 91 / 91 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3795 [1.3500-1.3950] | 1.1512 [1.1396-1.1609] | 83.5 | 28 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.5750 [4.5579-4.6297] | 4.3160 [4.2958-4.3284] | 94.3 | 8 / 9 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | bf16 | 18.5685 [17.6629-18.9967] | 18.4001 [17.2313-18.8651] | 99.1 | 2 / 2 | noise |
| attn_qkv | Q8 | dense-decode | 1 | f32 | 0.3687 [0.3638-0.3716] | 0.3696 [0.3684-0.3720] | 100.2 | 103 / 103 | noise |
| attn_qkv | Q8 | dense-decode | 2 | f32 | 0.3663 [0.3640-0.3697] | 0.3716 [0.3708-0.3742] | 101.5 | 104 / 102 | slower |
| attn_qkv | Q8 | dense-decode | 3 | f32 | 0.3740 [0.3706-0.3764] | 0.3771 [0.3748-0.3783] | 100.8 | 101 / 101 | noise |
| attn_qkv | Q8 | dense-decode | 4 | f32 | 0.3793 [0.3756-0.3832] | 0.3820 [0.3798-0.3835] | 100.7 | 100 / 99 | noise |
| attn_qkv | Q8 | dense-decode | 5 | f32 | 0.3981 [0.3972-0.3999] | 0.3914 [0.3890-0.3941] | 98.3 | 95 / 97 | faster |
| attn_qkv | Q8 | dense-decode | 6 | f32 | 0.4038 [0.4001-0.4091] | 0.3991 [0.3934-0.4007] | 98.8 | 94 / 95 | noise |
| attn_qkv | Q8 | dense-decode | 7 | f32 | 0.4113 [0.4086-0.4134] | 0.4083 [0.4057-0.4118] | 99.3 | 92 / 93 | noise |
| attn_qkv | Q8 | dense-decode | 8 | f32 | 0.4154 [0.4131-0.4201] | 0.4138 [0.4094-0.4165] | 99.6 | 91 / 92 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | f32 | 1.3766 [1.3607-1.3895] | 1.1327 [1.1242-1.1398] | 82.3 | 28 / 34 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | f32 | 4.4401 [4.4094-4.5596] | 4.1723 [4.1520-4.2484] | 94.0 | 9 / 9 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | f32 | 18.6352 [17.6291-19.5162] | 18.1583 [17.0386-18.8968] | 97.4 | 2 / 2 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | bf16 | 0.2143 [0.2119-0.2151] | 0.2134 [0.2112-0.2148] | 99.6 | 94 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | bf16 | 0.2172 [0.2148-0.2202] | 0.2153 [0.2134-0.2184] | 99.2 | 93 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 3 | bf16 | 0.2182 [0.2154-0.2195] | 0.2147 [0.2110-0.2159] | 98.4 | 92 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 4 | bf16 | 0.2234 [0.2218-0.2248] | 0.2198 [0.2177-0.2210] | 98.4 | 90 / 91 | faster |
| attn_qkv | Q4 syn | dense-decode | 5 | bf16 | 0.2314 [0.2295-0.2329] | 0.2211 [0.2190-0.2255] | 95.5 | 87 / 91 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | bf16 | 0.2369 [0.2350-0.2390] | 0.2238 [0.2214-0.2276] | 94.5 | 85 / 90 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | bf16 | 0.2422 [0.2411-0.2442] | 0.2287 [0.2272-0.2304] | 94.4 | 83 / 88 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | bf16 | 0.2450 [0.2429-0.2460] | 0.2290 [0.2270-0.2320] | 93.5 | 82 / 88 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0900 [1.0842-1.1283] | 1.0305 [1.0229-1.0674] | 94.5 | 18 / 20 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 4.2650 [4.2298-4.6288] | 4.1787 [4.1575-4.5257] | 98.0 | 5 / 5 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 18.9808 [18.7042-19.1193] | 18.7813 [17.7597-18.9231] | 99.0 | 1 / 1 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | f32 | 0.2150 [0.2140-0.2164] | 0.2144 [0.2124-0.2157] | 99.8 | 93 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | f32 | 0.2164 [0.2129-0.2180] | 0.2152 [0.2132-0.2173] | 99.5 | 93 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 3 | f32 | 0.2176 [0.2145-0.2186] | 0.2155 [0.2129-0.2184] | 99.0 | 92 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 4 | f32 | 0.2226 [0.2211-0.2243] | 0.2198 [0.2165-0.2217] | 98.7 | 90 / 91 | noise |
| attn_qkv | Q4 syn | dense-decode | 5 | f32 | 0.2298 [0.2274-0.2332] | 0.2240 [0.2186-0.2253] | 97.5 | 87 / 90 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | f32 | 0.2345 [0.2322-0.2368] | 0.2263 [0.2230-0.2278] | 96.5 | 86 / 89 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | f32 | 0.2378 [0.2355-0.2394] | 0.2272 [0.2258-0.2292] | 95.5 | 84 / 88 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | f32 | 0.2419 [0.2383-0.2448] | 0.2294 [0.2269-0.2315] | 94.9 | 83 / 88 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0463 [1.0411-1.0567] | 1.0018 [0.9930-1.0128] | 95.8 | 19 / 20 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 4.3838 [4.1747-4.8454] | 4.2819 [4.0719-4.7687] | 97.7 | 5 / 5 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 18.8269 [18.5551-19.0081] | 18.6525 [18.5468-18.8089] | 99.1 | 1 / 1 | noise |
| attn_o | Q8 | dense-decode | 1 | bf16 | 0.1800 [0.1790-0.1810] | 0.1798 [0.1790-0.1812] | 99.9 | 93 / 93 | noise |
| attn_o | Q8 | dense-decode | 2 | bf16 | 0.1806 [0.1785-0.1825] | 0.1814 [0.1789-0.1826] | 100.4 | 93 / 92 | noise |
| attn_o | Q8 | dense-decode | 3 | bf16 | 0.1836 [0.1822-0.1869] | 0.1880 [0.1859-0.1898] | 102.4 | 91 / 89 | noise |
| attn_o | Q8 | dense-decode | 4 | bf16 | 0.1902 [0.1877-0.1938] | 0.1931 [0.1912-0.1945] | 101.5 | 88 / 87 | noise |
| attn_o | Q8 | dense-decode | 5 | bf16 | 0.2000 [0.1932-0.2032] | 0.2031 [0.2002-0.2067] | 101.6 | 84 / 82 | noise |
| attn_o | Q8 | dense-decode | 6 | bf16 | 0.2094 [0.2057-0.2115] | 0.2048 [0.2034-0.2096] | 97.8 | 80 / 82 | noise |
| attn_o | Q8 | dense-decode | 7 | bf16 | 0.2152 [0.2133-0.2187] | 0.2093 [0.2088-0.2104] | 97.3 | 78 / 80 | faster |
| attn_o | Q8 | dense-decode | 8 | bf16 | 0.2202 [0.2176-0.2232] | 0.2135 [0.2120-0.2147] | 97.0 | 76 / 78 | faster |
| attn_o | Q8 | dense-prefill-wmma | 512 | bf16 | 1.2753 [1.2051-1.3423] | 0.7668 [0.7419-0.7844] | 60.1 | 13 / 22 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.2322 [3.1781-3.3040] | 2.3498 [2.2902-2.4222] | 72.7 | 5 / 7 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.5795 [8.5193-9.2390] | 7.7222 [7.6371-8.1999] | 90.0 | 2 / 2 | faster |
| attn_o | Q8 | dense-decode | 1 | f32 | 0.1798 [0.1785-0.1807] | 0.1802 [0.1790-0.1810] | 100.2 | 93 / 93 | noise |
| attn_o | Q8 | dense-decode | 2 | f32 | 0.1802 [0.1784-0.1819] | 0.1783 [0.1769-0.1806] | 99.0 | 93 / 94 | noise |
| attn_o | Q8 | dense-decode | 3 | f32 | 0.1849 [0.1826-0.1855] | 0.1878 [0.1841-0.1884] | 101.6 | 90 / 89 | noise |
| attn_o | Q8 | dense-decode | 4 | f32 | 0.1898 [0.1870-0.1922] | 0.1934 [0.1924-0.1944] | 101.8 | 88 / 86 | slower |
| attn_o | Q8 | dense-decode | 5 | f32 | 0.1988 [0.1951-0.2042] | 0.2014 [0.2004-0.2046] | 101.3 | 84 / 83 | noise |
| attn_o | Q8 | dense-decode | 6 | f32 | 0.2091 [0.2058-0.2116] | 0.2068 [0.2057-0.2100] | 98.9 | 80 / 81 | noise |
| attn_o | Q8 | dense-decode | 7 | f32 | 0.2160 [0.2106-0.2187] | 0.2088 [0.2076-0.2106] | 96.7 | 77 / 80 | noise |
| attn_o | Q8 | dense-decode | 8 | f32 | 0.2214 [0.2179-0.2255] | 0.2124 [0.2109-0.2134] | 96.0 | 75 / 79 | faster |
| attn_o | Q8 | dense-prefill-wmma | 512 | f32 | 1.3192 [1.2578-1.3739] | 0.7359 [0.7247-0.7703] | 55.8 | 13 / 23 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | f32 | 3.1822 [3.0843-3.2247] | 2.2025 [2.1593-2.2785] | 69.2 | 5 / 8 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | f32 | 9.5580 [9.4086-9.6506] | 8.5870 [8.4329-8.6759] | 89.8 | 2 / 2 | faster |
| attn_o | Q4 syn | dense-decode | 1 | bf16 | 0.4024 [0.3997-0.4071] | 0.4081 [0.4066-0.4132] | 101.4 | 22 / 22 | noise |
| attn_o | Q4 syn | dense-decode | 2 | bf16 | 0.4181 [0.4116-0.4263] | 0.4142 [0.4093-0.4204] | 99.1 | 21 / 21 | noise |
| attn_o | Q4 syn | dense-decode | 3 | bf16 | 0.4370 [0.4325-0.4420] | 0.4329 [0.4258-0.4364] | 99.1 | 20 / 20 | noise |
| attn_o | Q4 syn | dense-decode | 4 | bf16 | 0.4492 [0.4461-0.4542] | 0.4424 [0.4315-0.4492] | 98.5 | 20 / 20 | noise |
| attn_o | Q4 syn | dense-decode | 5 | bf16 | 0.4759 [0.4724-0.4829] | 0.4716 [0.4687-0.4763] | 99.1 | 19 / 19 | noise |
| attn_o | Q4 syn | dense-decode | 6 | bf16 | 0.4962 [0.4885-0.5020] | 0.4816 [0.4766-0.4856] | 97.1 | 18 / 18 | faster |
| attn_o | Q4 syn | dense-decode | 7 | bf16 | 0.5179 [0.5123-0.5229] | 0.4995 [0.4963-0.5083] | 96.5 | 17 / 18 | faster |
| attn_o | Q4 syn | dense-decode | 8 | bf16 | 0.5354 [0.5301-0.5428] | 0.5044 [0.5026-0.5121] | 94.2 | 17 / 18 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | bf16 | 4.2318 [4.1989-4.3373] | 2.6503 [2.5588-2.7812] | 62.6 | 2 / 3 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 10.8227 [10.7664-10.9945] | 8.7876 [8.5344-9.0207] | 81.2 | 1 / 1 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 32.4558 [32.1383-32.7734] | 30.0900 [20.1155-30.2300] | 92.7 | 0 / 0 | faster |
| attn_o | Q4 syn | dense-decode | 1 | f32 | 0.1029 [0.1020-0.1041] | 0.1024 [0.1006-0.1032] | 99.5 | 86 / 86 | noise |
| attn_o | Q4 syn | dense-decode | 2 | f32 | 0.1064 [0.1051-0.1071] | 0.1042 [0.1034-0.1050] | 97.9 | 83 / 85 | faster |
| attn_o | Q4 syn | dense-decode | 3 | f32 | 0.1094 [0.1087-0.1107] | 0.1064 [0.1055-0.1084] | 97.3 | 81 / 83 | faster |
| attn_o | Q4 syn | dense-decode | 4 | f32 | 0.1103 [0.1097-0.1122] | 0.1077 [0.1059-0.1098] | 97.6 | 80 / 82 | noise |
| attn_o | Q4 syn | dense-decode | 5 | f32 | 0.1204 [0.1181-0.1224] | 0.1178 [0.1151-0.1200] | 97.8 | 74 / 75 | noise |
| attn_o | Q4 syn | dense-decode | 6 | f32 | 0.1239 [0.1210-0.1252] | 0.1212 [0.1198-0.1218] | 97.8 | 71 / 73 | noise |
| attn_o | Q4 syn | dense-decode | 7 | f32 | 0.1266 [0.1238-0.1292] | 0.1230 [0.1201-0.1252] | 97.2 | 70 / 72 | noise |
| attn_o | Q4 syn | dense-decode | 8 | f32 | 0.1289 [0.1274-0.1298] | 0.1221 [0.1208-0.1231] | 94.7 | 69 / 72 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0397 [1.0291-1.0542] | 0.6323 [0.6115-0.6603] | 60.8 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.6737 [2.6362-2.7107] | 2.1229 [2.0786-2.1766] | 79.4 | 3 / 4 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | f32 | 8.9527 [8.7361-9.2433] | 8.5122 [8.1699-8.9790] | 95.1 | 1 / 1 | noise |
| ple_kv | Q8 | dense-decode | 1 | bf16 | 0.3566 [0.3532-0.3590] | 0.3586 [0.3556-0.3604] | 100.5 | 98 / 97 | noise |
| ple_kv | Q8 | dense-decode | 2 | bf16 | 0.3565 [0.3548-0.3604] | 0.3626 [0.3602-0.3646] | 101.7 | 98 / 96 | noise |
| ple_kv | Q8 | dense-decode | 3 | bf16 | 0.3629 [0.3607-0.3653] | 0.3654 [0.3620-0.3676] | 100.7 | 96 / 95 | noise |
| ple_kv | Q8 | dense-decode | 4 | bf16 | 0.3736 [0.3658-0.3762] | 0.3752 [0.3721-0.3789] | 100.4 | 93 / 93 | noise |
| ple_kv | Q8 | dense-decode | 5 | bf16 | 0.3876 [0.3849-0.3912] | 0.3870 [0.3847-0.3940] | 99.8 | 90 / 90 | noise |
| ple_kv | Q8 | dense-decode | 6 | bf16 | 0.3934 [0.3903-0.3971] | 0.3929 [0.3859-0.3994] | 99.9 | 89 / 89 | noise |
| ple_kv | Q8 | dense-decode | 7 | bf16 | 0.3983 [0.3960-0.4008] | 0.4063 [0.3994-0.4099] | 102.0 | 87 / 86 | noise |
| ple_kv | Q8 | dense-decode | 8 | bf16 | 0.3994 [0.3984-0.4024] | 0.4084 [0.4035-0.4106] | 102.2 | 87 / 85 | slower |
| ple_kv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3506 [1.3198-1.3677] | 1.1117 [1.0968-1.1166] | 82.3 | 26 / 31 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.4729 [4.4565-4.5907] | 4.2534 [4.2010-4.3142] | 95.1 | 8 / 8 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | bf16 | 18.3378 [18.2330-18.4213] | 17.8618 [17.7038-17.9270] | 97.4 | 2 / 2 | faster |
| ple_kv | Q8 | dense-decode | 1 | f32 | 0.3538 [0.3518-0.3546] | 0.3557 [0.3550-0.3577] | 100.5 | 98 / 98 | slower |
| ple_kv | Q8 | dense-decode | 2 | f32 | 0.3524 [0.3495-0.3543] | 0.3590 [0.3561-0.3612] | 101.9 | 99 / 97 | slower |
| ple_kv | Q8 | dense-decode | 3 | f32 | 0.3578 [0.3552-0.3607] | 0.3609 [0.3589-0.3626] | 100.9 | 97 / 96 | noise |
| ple_kv | Q8 | dense-decode | 4 | f32 | 0.3661 [0.3644-0.3684] | 0.3703 [0.3687-0.3720] | 101.1 | 95 / 94 | slower |
| ple_kv | Q8 | dense-decode | 5 | f32 | 0.3802 [0.3776-0.3854] | 0.3776 [0.3747-0.3814] | 99.3 | 92 / 92 | noise |
| ple_kv | Q8 | dense-decode | 6 | f32 | 0.3872 [0.3836-0.3975] | 0.3895 [0.3837-0.3950] | 100.6 | 90 / 89 | noise |
| ple_kv | Q8 | dense-decode | 7 | f32 | 0.3950 [0.3911-0.3971] | 0.3940 [0.3923-0.3961] | 99.8 | 88 / 88 | noise |
| ple_kv | Q8 | dense-decode | 8 | f32 | 0.4018 [0.3985-0.4066] | 0.4018 [0.3995-0.4051] | 100.0 | 87 / 87 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 512 | f32 | 1.3015 [1.2783-1.3228] | 1.0779 [1.0742-1.0863] | 82.8 | 27 / 32 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | f32 | 16.8595 [16.8147-17.0048] | 15.6194 [15.5788-15.6672] | 92.6 | 2 / 2 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | f32 | 62.4929 [62.4547-62.6008] | 60.3202 [60.1512-60.4544] | 96.5 | 1 / 1 | faster |
| ple_kv | Q4 syn | dense-decode | 1 | bf16 | 0.1969 [0.1937-0.1980] | 0.1956 [0.1918-0.1991] | 99.3 | 94 / 94 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | bf16 | 0.2022 [0.2009-0.2040] | 0.1986 [0.1970-0.2003] | 98.2 | 91 / 93 | faster |
| ple_kv | Q4 syn | dense-decode | 3 | bf16 | 0.2010 [0.2000-0.2024] | 0.1997 [0.1968-0.2005] | 99.3 | 92 / 92 | noise |
| ple_kv | Q4 syn | dense-decode | 4 | bf16 | 0.2047 [0.2029-0.2066] | 0.1987 [0.1977-0.2030] | 97.1 | 90 / 93 | noise |
| ple_kv | Q4 syn | dense-decode | 5 | bf16 | 0.2163 [0.2156-0.2171] | 0.2060 [0.2046-0.2072] | 95.2 | 85 / 89 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | bf16 | 0.2182 [0.2164-0.2204] | 0.2056 [0.2030-0.2071] | 94.2 | 84 / 90 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | bf16 | 0.2247 [0.2208-0.2254] | 0.2115 [0.2085-0.2135] | 94.1 | 82 / 87 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | bf16 | 0.2309 [0.2290-0.2322] | 0.2153 [0.2127-0.2185] | 93.2 | 80 / 86 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0192 [1.0117-1.0330] | 0.9656 [0.9566-0.9688] | 94.7 | 18 / 19 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 4.2688 [4.1522-4.4044] | 4.1581 [4.1061-4.3395] | 97.4 | 4 / 4 | noise |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 17.4861 [17.3725-17.6694] | 17.0955 [16.9439-17.3319] | 97.8 | 1 / 1 | faster |
| ple_kv | Q4 syn | dense-decode | 1 | f32 | 0.1977 [0.1952-0.1995] | 0.1977 [0.1956-0.2000] | 100.0 | 93 / 93 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | f32 | 0.1993 [0.1983-0.2040] | 0.1984 [0.1956-0.1992] | 99.6 | 92 / 93 | noise |
| ple_kv | Q4 syn | dense-decode | 3 | f32 | 0.2047 [0.2009-0.2066] | 0.1996 [0.1976-0.2029] | 97.5 | 90 / 92 | noise |
| ple_kv | Q4 syn | dense-decode | 4 | f32 | 0.2041 [0.2022-0.2067] | 0.2017 [0.1982-0.2031] | 98.8 | 90 / 91 | noise |
| ple_kv | Q4 syn | dense-decode | 5 | f32 | 0.2134 [0.2118-0.2162] | 0.2067 [0.2047-0.2077] | 96.8 | 86 / 89 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | f32 | 0.2142 [0.2119-0.2184] | 0.2074 [0.2042-0.2102] | 96.8 | 86 / 89 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | f32 | 0.2225 [0.2210-0.2237] | 0.2112 [0.2092-0.2150] | 94.9 | 83 / 87 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | f32 | 0.2268 [0.2246-0.2305] | 0.2118 [0.2088-0.2166] | 93.4 | 81 / 87 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0203 [1.0072-1.0408] | 0.9737 [0.9548-0.9863] | 95.4 | 18 / 19 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 4.2834 [4.2336-4.2969] | 4.1710 [4.1593-4.1863] | 97.4 | 4 / 4 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 18.0375 [17.7382-18.1832] | 17.6520 [17.1780-17.8640] | 97.9 | 1 / 1 | noise |
| lm_head | Q8 | dense-decode | 1 | bf16 | 3.5179 [3.4816-3.5371] | 3.5382 [3.5142-3.5580] | 100.6 | 192 / 191 | noise |
| lm_head | Q8 | dense-decode | 2 | bf16 | 3.2731 [3.2552-3.2952] | 3.2758 [3.2609-3.3231] | 100.1 | 206 / 206 | noise |
| lm_head | Q8 | dense-decode | 3 | bf16 | 3.5239 [3.5034-3.5457] | 3.3191 [3.2830-3.3482] | 94.2 | 192 / 204 | faster |
| lm_head | Q8 | dense-decode | 4 | bf16 | 3.6168 [3.5687-3.6686] | 3.3388 [3.3030-3.3729] | 92.3 | 187 / 202 | faster |
| lm_head | Q8 | dense-decode | 5 | bf16 | 3.5940 [3.5748-3.6122] | 3.4504 [3.4367-3.4792] | 96.0 | 188 / 196 | faster |
| lm_head | Q8 | dense-decode | 6 | bf16 | 3.5883 [3.5494-3.6180] | 3.4569 [3.4251-3.5161] | 96.3 | 188 / 195 | faster |
| lm_head | Q8 | dense-decode | 7 | bf16 | 3.6513 [3.6300-3.6810] | 3.6708 [3.6544-3.6786] | 100.5 | 185 / 184 | noise |
| lm_head | Q8 | dense-decode | 8 | bf16 | 3.8038 [3.7692-3.8277] | 3.8510 [3.8431-3.8740] | 101.2 | 178 / 175 | slower |
| lm_head | Q8 | dense-prefill-wmma | 512 | bf16 | 23.1460 [22.9549-23.5312] | 22.4560 [22.2707-22.8392] | 97.0 | 29 / 30 | faster |
| lm_head | Q8 | dense-prefill-wmma | 2048 | bf16 | 92.4688 [91.4684-93.6703] | 89.4992 [88.9217-90.5087] | 96.8 | 7 / 8 | faster |
| lm_head | Q8 | dense-decode | 1 | f32 | 3.2264 [3.2055-3.2639] | 3.2423 [3.2326-3.2702] | 100.5 | 209 / 208 | noise |
| lm_head | Q8 | dense-decode | 2 | f32 | 3.2212 [3.2124-3.2425] | 3.2396 [3.2341-3.2510] | 100.6 | 210 / 208 | noise |
| lm_head | Q8 | dense-decode | 3 | f32 | 3.4572 [3.4485-3.4651] | 3.2702 [3.2610-3.2896] | 94.6 | 195 / 207 | faster |
| lm_head | Q8 | dense-decode | 4 | f32 | 3.5869 [3.5737-3.6004] | 3.3020 [3.2958-3.3174] | 92.1 | 188 / 205 | faster |
| lm_head | Q8 | dense-decode | 5 | f32 | 3.4726 [3.4617-3.4912] | 3.3752 [3.3665-3.3870] | 97.2 | 194 / 200 | faster |
| lm_head | Q8 | dense-decode | 6 | f32 | 3.5828 [3.5524-3.6036] | 3.5216 [3.4992-3.5449] | 98.3 | 189 / 192 | faster |
| lm_head | Q8 | dense-decode | 7 | f32 | 3.6607 [3.6378-3.6764] | 3.6497 [3.6380-3.6573] | 99.7 | 185 / 185 | noise |
| lm_head | Q8 | dense-decode | 8 | f32 | 3.7748 [3.7550-3.8340] | 3.8293 [3.8152-3.8382] | 101.4 | 179 / 176 | noise |
| lm_head | Q8 | dense-prefill-wmma | 512 | f32 | 22.0837 [22.0060-22.7014] | 21.4850 [21.3797-21.9344] | 97.3 | 31 / 31 | faster |
| lm_head | Q4 syn | dense-decode | 1 | bf16 | 8.3817 [8.3251-8.5128] | 8.4630 [8.3315-8.5700] | 101.0 | 43 / 42 | noise |
| lm_head | Q4 syn | dense-decode | 2 | bf16 | 8.5675 [8.5429-8.6239] | 8.3995 [8.3797-8.5114] | 98.0 | 42 / 43 | faster |
| lm_head | Q4 syn | dense-decode | 3 | bf16 | 9.3713 [9.3242-9.4088] | 8.8242 [8.7762-8.8781] | 94.2 | 38 / 41 | faster |
| lm_head | Q4 syn | dense-decode | 4 | bf16 | 9.7874 [9.7516-9.8563] | 9.1911 [9.1110-9.2536] | 93.9 | 37 / 39 | faster |
| lm_head | Q4 syn | dense-decode | 5 | bf16 | 10.6355 [10.4527-10.6875] | 9.9557 [9.8353-10.0846] | 93.6 | 34 / 36 | faster |
| lm_head | Q4 syn | dense-decode | 6 | bf16 | 10.7811 [10.7659-10.8857] | 10.0898 [10.0587-10.1201] | 93.6 | 33 / 35 | faster |
| lm_head | Q4 syn | dense-decode | 7 | bf16 | 2.7413 [2.7123-11.2594] | 2.6075 [2.5687-10.3844] | 95.1 | 130 / 137 | noise |
| lm_head | Q4 syn | dense-decode | 8 | bf16 | 2.8966 [2.8669-2.9270] | 2.7022 [2.6707-2.7483] | 93.3 | 123 / 132 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | bf16 | 22.8535 [22.6006-23.0373] | 22.2016 [21.8842-22.4579] | 97.2 | 16 / 16 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 89.7249 [88.9514-93.9666] | 86.7823 [84.1237-89.3048] | 96.7 | 4 / 4 | noise |
| lm_head | Q4 syn | dense-decode | 1 | f32 | 1.8697 [1.8525-1.8807] | 1.8397 [1.8256-1.8582] | 98.4 | 191 / 194 | noise |
| lm_head | Q4 syn | dense-decode | 2 | f32 | 1.8786 [1.8704-1.8956] | 1.8530 [1.8420-1.8745] | 98.6 | 190 / 193 | noise |
| lm_head | Q4 syn | dense-decode | 3 | f32 | 2.0801 [2.0669-2.1022] | 1.9622 [1.9473-1.9823] | 94.3 | 172 / 182 | faster |
| lm_head | Q4 syn | dense-decode | 4 | f32 | 2.1896 [2.1869-2.2027] | 2.0537 [2.0420-2.0594] | 93.8 | 163 / 174 | faster |
| lm_head | Q4 syn | dense-decode | 5 | f32 | 2.4356 [2.4312-2.4446] | 2.2936 [2.2875-2.3031] | 94.2 | 147 / 156 | faster |
| lm_head | Q4 syn | dense-decode | 6 | f32 | 2.5219 [2.5059-2.5346] | 2.3946 [2.3766-2.4119] | 95.0 | 142 / 149 | faster |
| lm_head | Q4 syn | dense-decode | 7 | f32 | 2.6692 [2.6593-2.6841] | 2.5423 [2.5328-2.5606] | 95.2 | 134 / 141 | faster |
| lm_head | Q4 syn | dense-decode | 8 | f32 | 2.8762 [2.8627-2.8948] | 2.7121 [2.7027-2.7194] | 94.3 | 124 / 132 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | f32 | 21.7409 [21.4044-22.4545] | 21.3372 [21.2405-21.7752] | 98.1 | 16 / 17 | noise |
| mtp_fc | Q8 | dense-decode | 1 | bf16 | 0.0781 [0.0776-0.0790] | 0.0792 [0.0787-0.0814] | 101.3 | 89 / 88 | noise |
| mtp_fc | Q8 | dense-decode | 2 | bf16 | 0.0804 [0.0798-0.0821] | 0.0810 [0.0804-0.0824] | 100.8 | 87 / 86 | noise |
| mtp_fc | Q8 | dense-decode | 3 | bf16 | 0.0842 [0.0834-0.0852] | 0.0825 [0.0816-0.0832] | 98.0 | 83 / 84 | faster |
| mtp_fc | Q8 | dense-decode | 4 | bf16 | 0.0874 [0.0862-0.0888] | 0.0843 [0.0839-0.0860] | 96.4 | 80 / 83 | faster |
| mtp_fc | Q8 | dense-decode | 5 | bf16 | 0.0915 [0.0903-0.0918] | 0.0899 [0.0894-0.0920] | 98.2 | 76 / 77 | noise |
| mtp_fc | Q8 | dense-decode | 6 | bf16 | 0.0941 [0.0926-0.0956] | 0.0924 [0.0921-0.0932] | 98.2 | 74 / 75 | noise |
| mtp_fc | Q8 | dense-decode | 7 | bf16 | 0.0947 [0.0940-0.0954] | 0.0933 [0.0928-0.0950] | 98.5 | 74 / 75 | noise |
| mtp_fc | Q8 | dense-decode | 8 | bf16 | 0.0988 [0.0975-0.1016] | 0.0972 [0.0953-0.0976] | 98.3 | 70 / 72 | noise |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | bf16 | 0.2921 [0.2876-0.2969] | 0.2276 [0.2270-0.2309] | 77.9 | 24 / 31 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.8419 [0.8251-0.8475] | 0.7727 [0.7629-0.7874] | 91.8 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | bf16 | 3.1215 [3.1017-3.2808] | 3.0374 [3.0156-3.1752] | 97.3 | 2 / 2 | noise |
| mtp_fc | Q8 | dense-decode | 1 | f32 | 0.0782 [0.0779-0.0799] | 0.0788 [0.0784-0.0798] | 100.8 | 89 / 88 | noise |
| mtp_fc | Q8 | dense-decode | 2 | f32 | 0.0796 [0.0789-0.0809] | 0.0799 [0.0794-0.0818] | 100.3 | 87 / 87 | noise |
| mtp_fc | Q8 | dense-decode | 3 | f32 | 0.0832 [0.0823-0.0840] | 0.0822 [0.0807-0.0831] | 98.9 | 84 / 85 | noise |
| mtp_fc | Q8 | dense-decode | 4 | f32 | 0.0854 [0.0848-0.0868] | 0.0858 [0.0850-0.0863] | 100.5 | 82 / 81 | noise |
| mtp_fc | Q8 | dense-decode | 5 | f32 | 0.0888 [0.0879-0.0896] | 0.0890 [0.0884-0.0904] | 100.2 | 78 / 78 | noise |
| mtp_fc | Q8 | dense-decode | 6 | f32 | 0.0907 [0.0904-0.0929] | 0.0907 [0.0902-0.0936] | 100.0 | 77 / 77 | noise |
| mtp_fc | Q8 | dense-decode | 7 | f32 | 0.0938 [0.0928-0.0951] | 0.0938 [0.0932-0.0947] | 100.0 | 74 / 74 | noise |
| mtp_fc | Q8 | dense-decode | 8 | f32 | 0.0962 [0.0956-0.0987] | 0.0948 [0.0942-0.0958] | 98.5 | 72 / 73 | noise |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | f32 | 0.2980 [0.2963-0.3010] | 0.2319 [0.2309-0.2334] | 77.8 | 23 / 30 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | f32 | 0.8288 [0.8249-0.8470] | 0.7749 [0.7673-0.7814] | 93.5 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | f32 | 3.3167 [3.0627-3.5153] | 3.3025 [2.9742-3.4158] | 99.6 | 2 / 2 | noise |
| mtp_fc | Q4 syn | dense-decode | 1 | bf16 | 0.0454 [0.0447-0.0464] | 0.0442 [0.0438-0.0448] | 97.3 | 81 / 83 | noise |
| mtp_fc | Q4 syn | dense-decode | 2 | bf16 | 0.0487 [0.0480-0.0504] | 0.0463 [0.0456-0.0466] | 95.2 | 76 / 80 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | bf16 | 0.0511 [0.0506-0.0522] | 0.0476 [0.0469-0.0486] | 93.1 | 72 / 77 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | bf16 | 0.0549 [0.0534-0.0561] | 0.0496 [0.0491-0.0502] | 90.4 | 67 / 74 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | bf16 | 0.0574 [0.0564-0.0584] | 0.0539 [0.0532-0.0543] | 93.9 | 64 / 68 | faster |
| mtp_fc | Q4 syn | dense-decode | 6 | bf16 | 0.0603 [0.0578-0.0616] | 0.0561 [0.0557-0.0567] | 93.0 | 61 / 66 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | bf16 | 0.0598 [0.0592-0.0623] | 0.0566 [0.0559-0.0576] | 94.7 | 62 / 65 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | bf16 | 0.0619 [0.0612-0.0622] | 0.0580 [0.0577-0.0588] | 93.7 | 60 / 64 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.2334 [0.2310-0.2349] | 0.2162 [0.2160-0.2172] | 92.6 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.7815 [0.7642-0.7895] | 0.7642 [0.7589-0.7687] | 97.8 | 5 / 5 | noise |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 3.0766 [3.0600-3.1972] | 3.0236 [2.9920-3.2030] | 98.3 | 1 / 1 | noise |
| mtp_fc | Q4 syn | dense-decode | 1 | f32 | 0.0457 [0.0444-0.0460] | 0.0452 [0.0444-0.0454] | 98.9 | 81 / 82 | noise |
| mtp_fc | Q4 syn | dense-decode | 2 | f32 | 0.0488 [0.0476-0.0510] | 0.0463 [0.0458-0.0467] | 94.9 | 76 / 80 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | f32 | 0.0510 [0.0492-0.0524] | 0.0477 [0.0472-0.0480] | 93.5 | 72 / 77 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | f32 | 0.0520 [0.0510-0.0527] | 0.0492 [0.0488-0.0500] | 94.5 | 71 / 75 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | f32 | 0.0548 [0.0540-0.0561] | 0.0533 [0.0522-0.0543] | 97.3 | 67 / 69 | noise |
| mtp_fc | Q4 syn | dense-decode | 6 | f32 | 0.0576 [0.0562-0.0586] | 0.0543 [0.0537-0.0559] | 94.3 | 64 / 68 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | f32 | 0.0591 [0.0584-0.0598] | 0.0561 [0.0551-0.0566] | 94.9 | 62 / 66 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | f32 | 0.0613 [0.0600-0.0630] | 0.0578 [0.0572-0.0583] | 94.2 | 60 / 64 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.2361 [0.2346-0.2375] | 0.2195 [0.2188-0.2200] | 93.0 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.7782 [0.7732-0.7828] | 0.7662 [0.7624-0.7736] | 98.5 | 5 / 5 | noise |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | f32 | 3.2176 [3.0190-3.4966] | 3.1534 [2.9842-3.3449] | 98.0 | 1 / 1 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 1 | bf16 | 0.1902 [0.1879-0.1959] | 0.1871 [0.1856-0.1883] | 98.4 | 92 / 93 | noise |
| exp_gu | Q4 | expert-decode-gather | 1 | bf16 | 0.1876 [0.1856-0.1892] | 0.1846 [0.1830-0.1889] | 98.4 | 93 / 94 | noise |
| exp_down | Q4 | expert-decode-combine | 1 | bf16 | 0.1207 [0.1198-0.1215] | 0.1158 [0.1150-0.1186] | 95.9 | 85 / 88 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 2 | bf16 | 0.3387 [0.3338-0.3456] | 0.3138 [0.3115-0.3182] | 92.7 | 98 / 105 | faster |
| exp_gu | Q4 | expert-decode-gather | 2 | bf16 | 0.3342 [0.3321-0.3396] | 0.3106 [0.3083-0.3206] | 92.9 | 99 / 106 | faster |
| exp_down | Q4 | expert-decode-combine | 2 | bf16 | 0.2202 [0.2197-0.2217] | 0.2120 [0.2104-0.2148] | 96.3 | 88 / 92 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 3 | bf16 | 0.4140 [0.4106-0.4214] | 0.4115 [0.4061-0.4185] | 99.4 | 109 / 110 | noise |
| exp_gu | Q4 | expert-decode-gather | 3 | bf16 | 0.4200 [0.4127-0.4273] | 0.4114 [0.4068-0.4179] | 98.0 | 108 / 110 | noise |
| exp_down | Q4 | expert-decode-combine | 3 | bf16 | 0.2980 [0.2964-0.3057] | 0.2892 [0.2872-0.2915] | 97.1 | 89 / 92 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 4 | bf16 | 0.4831 [0.4784-0.4948] | 0.4769 [0.4739-0.4876] | 98.7 | 123 / 124 | noise |
| exp_gu | Q4 | expert-decode-gather | 4 | bf16 | 0.4853 [0.4830-0.4927] | 0.4870 [0.4836-0.4934] | 100.3 | 122 / 122 | noise |
| exp_down | Q4 | expert-decode-combine | 4 | bf16 | 0.3761 [0.3749-0.3794] | 0.3599 [0.3563-0.3697] | 95.7 | 93 / 97 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 5 | bf16 | 0.5181 [0.5125-0.5229] | 0.4974 [0.4938-0.5024] | 96.0 | 118 / 122 | faster |
| exp_gu | Q4 | expert-decode-gather | 5 | bf16 | 0.5275 [0.5230-0.5337] | 0.5218 [0.5105-0.5262] | 98.9 | 116 / 117 | noise |
| exp_down | Q4 | expert-decode-combine | 5 | bf16 | 0.3747 [0.3705-0.3844] | 0.3754 [0.3654-0.3819] | 100.2 | 96 / 95 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 6 | bf16 | 0.5910 [0.5872-0.6034] | 0.5677 [0.5650-0.5798] | 96.1 | 127 / 132 | faster |
| exp_gu | Q4 | expert-decode-gather | 6 | bf16 | 0.6079 [0.6027-0.6175] | 0.5888 [0.5792-0.5989] | 96.9 | 123 / 127 | faster |
| exp_down | Q4 | expert-decode-combine | 6 | bf16 | 0.4237 [0.4192-0.4308] | 0.4140 [0.4129-0.4220] | 97.7 | 104 / 106 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 7 | bf16 | 0.6449 [0.6391-0.6504] | 0.6203 [0.6169-0.6317] | 96.2 | 127 / 132 | faster |
| exp_gu | Q4 | expert-decode-gather | 7 | bf16 | 0.6628 [0.6579-0.6710] | 0.6381 [0.6346-0.6550] | 96.3 | 123 / 128 | faster |
| exp_down | Q4 | expert-decode-combine | 7 | bf16 | 0.4614 [0.4562-0.4649] | 0.4373 [0.4345-0.4419] | 94.8 | 104 / 110 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 8 | bf16 | 0.7359 [0.7241-0.7462] | 0.7231 [0.7113-0.7280] | 98.3 | 123 / 125 | noise |
| exp_gu | Q4 | expert-decode-gather | 8 | bf16 | 0.7578 [0.7456-0.7625] | 0.7340 [0.7277-0.7467] | 96.9 | 119 / 123 | noise |
| exp_down | Q4 | expert-decode-combine | 8 | bf16 | 0.4989 [0.4965-0.5092] | 0.4760 [0.4741-0.4797] | 95.4 | 107 / 112 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 512 | bf16 | 4.6569 [4.4891-4.7215] | 4.3700 [4.2221-4.4053] | 93.8 | 171 / 182 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 512 | bf16 | 4.6871 [4.6556-4.7034] | 4.3711 [4.3540-4.3891] | 93.3 | 170 / 182 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 512 | bf16 | 4.4711 [4.4608-4.4879] | 4.4081 [4.4028-4.4505] | 98.6 | 178 / 180 | faster |
| exp_down | Q4 | expert-prefill-combine narrow | 512 | bf16 | 3.1044 [3.0888-3.1185] | 2.8555 [2.8277-2.8828] | 92.0 | 151 / 164 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 512 | bf16 | 3.1054 [3.0921-3.1280] | 2.8421 [2.8354-2.8591] | 91.5 | 151 / 165 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 512 | bf16 | 3.1094 [3.1028-3.1420] | 2.8537 [2.8414-2.8673] | 91.8 | 150 / 164 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 512 | bf16 | 5.8920 [5.8675-5.9050] | 5.8403 [5.8263-5.8834] | 99.1 | 135 / 136 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 512 | bf16 | 5.8783 [5.8448-5.9022] | 5.8445 [5.8183-5.8814] | 99.4 | 135 / 136 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 512 | bf16 | 5.9887 [5.9238-6.0367] | 5.8745 [5.8164-5.9084] | 98.1 | 133 / 135 | faster |
| exp_down | Q4 | expert-prefill-combine wide | 512 | bf16 | 3.6398 [3.6330-3.6469] | 3.2676 [3.2612-3.2833] | 89.8 | 129 / 143 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 512 | bf16 | 3.5230 [3.5134-3.5396] | 3.1638 [3.1501-3.1874] | 89.8 | 133 / 148 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 512 | bf16 | 3.5191 [3.5160-3.5445] | 3.1555 [3.1484-3.1598] | 89.7 | 133 / 148 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 512 | bf16 | 5.7767 [5.7476-5.8052] | 5.3340 [5.2916-5.3602] | 92.3 | 138 / 149 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 512 | bf16 | 5.7781 [5.7377-5.8134] | 5.3195 [5.2891-5.3728] | 92.1 | 138 / 150 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 512 | bf16 | 5.8523 [5.8010-5.8959] | 5.4118 [5.3773-5.4767] | 92.5 | 136 / 147 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 512 | bf16 | 4.3556 [4.3061-4.3807] | 3.0752 [3.0474-3.1033] | 70.6 | 107 / 152 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 512 | bf16 | 4.3975 [4.3685-4.4120] | 3.0141 [2.9957-3.0282] | 68.5 | 106 / 155 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 512 | bf16 | 4.3998 [4.3765-4.4408] | 3.0232 [3.0100-3.0344] | 68.7 | 106 / 155 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 2048 | bf16 | 8.4766 [7.7326-8.5468] | 7.9306 [7.2511-8.0169] | 93.6 | 105 / 112 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 2048 | bf16 | 8.2703 [8.0938-8.3480] | 7.7443 [7.6606-7.8007] | 93.6 | 108 / 115 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 2048 | bf16 | 7.9441 [7.8360-8.0667] | 7.9158 [7.8134-8.0561] | 99.6 | 112 / 112 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 2048 | bf16 | 5.4989 [5.4387-5.5176] | 5.1085 [5.0857-5.1324] | 92.9 | 95 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 2048 | bf16 | 5.4113 [5.3670-5.4275] | 5.0390 [4.9974-5.0470] | 93.1 | 97 / 104 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 2048 | bf16 | 5.4828 [5.4486-5.5192] | 5.1397 [5.0919-5.2041] | 93.7 | 95 / 102 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 2048 | bf16 | 9.1846 [9.1279-9.3129] | 9.1204 [9.0868-9.3229] | 99.3 | 97 / 98 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 2048 | bf16 | 9.2871 [9.1805-9.3980] | 9.3213 [9.1809-9.3852] | 100.4 | 96 / 95 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 2048 | bf16 | 9.3968 [9.1678-9.4478] | 9.2760 [9.1136-9.3366] | 98.7 | 95 / 96 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 2048 | bf16 | 5.8698 [5.7711-5.9127] | 5.3835 [5.2256-5.4068] | 91.7 | 89 / 97 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 2048 | bf16 | 5.6729 [5.6429-5.7037] | 5.1656 [5.1378-5.1841] | 91.1 | 92 / 101 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 2048 | bf16 | 5.7537 [5.7342-5.7817] | 5.2678 [5.2299-5.2893] | 91.5 | 91 / 99 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 2048 | bf16 | 8.8024 [8.7175-8.8847] | 8.4696 [8.3123-8.5252] | 96.2 | 101 / 105 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 2048 | bf16 | 8.7445 [8.7064-8.8836] | 8.4272 [8.3033-8.4921] | 96.4 | 102 / 106 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 2048 | bf16 | 9.0650 [8.8673-9.0941] | 8.5676 [8.4032-8.6631] | 94.5 | 98 / 104 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 2048 | bf16 | 6.2404 [6.1916-6.3013] | 5.2010 [5.1345-5.2345] | 83.3 | 84 / 101 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 2048 | bf16 | 6.2030 [6.1585-6.2509] | 5.1538 [5.1033-5.1826] | 83.1 | 84 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 2048 | bf16 | 6.2337 [6.1083-6.2706] | 5.2032 [5.1239-5.2408] | 83.5 | 84 / 101 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 8192 | bf16 | 21.4078 [20.8341-21.8993] | 21.2157 [19.9043-21.5554] | 99.1 | 42 / 42 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 8192 | bf16 | 23.1032 [22.9022-23.3634] | 22.2482 [22.0780-22.5672] | 96.3 | 39 / 40 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 8192 | bf16 | 22.0747 [21.8912-22.8193] | 22.3661 [22.1346-22.8498] | 101.3 | 40 / 40 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 8192 | bf16 | 15.6165 [15.4707-15.9500] | 15.0183 [14.8777-15.3168] | 96.2 | 34 / 35 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 8192 | bf16 | 15.2584 [15.1736-15.3105] | 14.7113 [14.5671-14.7532] | 96.4 | 34 / 36 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 8192 | bf16 | 15.3417 [15.2697-15.4361] | 14.7503 [14.7079-14.8029] | 96.2 | 34 / 36 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 8192 | bf16 | 22.2445 [22.2157-22.4428] | 22.2867 [22.2165-22.6316] | 100.2 | 40 / 40 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 8192 | bf16 | 22.2058 [22.1602-22.2968] | 22.2178 [22.2003-22.3008] | 100.0 | 40 / 40 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 8192 | bf16 | 21.5110 [21.3971-22.2074] | 21.3731 [21.2683-21.8187] | 99.4 | 41 / 42 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 8192 | bf16 | 14.1099 [14.0282-14.2634] | 13.3904 [13.3277-13.6132] | 94.9 | 37 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 8192 | bf16 | 14.3178 [14.2164-14.5277] | 13.6042 [13.5202-13.8373] | 95.0 | 37 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 8192 | bf16 | 14.3509 [14.1524-14.6969] | 13.5483 [13.4234-13.9149] | 94.4 | 37 / 39 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 8192 | bf16 | 20.6079 [20.3032-20.9037] | 20.0642 [19.6810-20.3775] | 97.4 | 43 / 44 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 8192 | bf16 | 20.6605 [20.4146-20.9912] | 20.1616 [19.7404-20.6058] | 97.6 | 43 / 44 | noise |
| exp_gu | Q4 | expert-prefill-gather wide256 | 8192 | bf16 | 20.9005 [20.6493-21.1134] | 19.8451 [19.6838-20.1371] | 95.0 | 43 / 45 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 8192 | bf16 | 15.2200 [15.0688-15.4425] | 13.6101 [13.5471-13.7729] | 89.4 | 34 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 8192 | bf16 | 15.0110 [14.9132-15.2645] | 13.6035 [13.4330-13.7799] | 90.6 | 35 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 8192 | bf16 | 15.4481 [15.3162-15.5166] | 13.8778 [13.8056-13.9454] | 89.8 | 34 / 38 | faster |
