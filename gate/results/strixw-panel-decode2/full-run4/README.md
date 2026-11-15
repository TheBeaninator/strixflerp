# strixw panel kernels: kernel gate results

Criteria: `gate/strixw-panel-kernels.toml` (committed before the run). Raw: `raw.jsonl` (one case a line), bench log `bench.log`. 489 cases, 6 skipped (output too large), 2.232 GB read from the fleet file.

## Verdicts

| criterion | pass | detail |
|---|---|---|
| equal | PASS | 489 of 489 cases byte-equal |
| bad_id | PASS | 21 of 21 out-of-range-id cases byte-equal |
| coverage | PASS | missing: none |
| read | FAIL | 2.232 GB read |
| speed_decode_dense | PASS | median panel % of origin time per class: attn_o 99.0, attn_qkv 98.6, gdn_in 98.0, gdn_out 98.8, lm_head 96.3, mtp_fc 98.4, ple_kv 99.5, sh_down 83.6, sh_gu 95.5 |
| speed_prefill_dense | PASS | sh_gu M2048 bf16 91.6%; sh_gu M8192 bf16 97.5%; sh_gu M2048 f32 93.2%; sh_gu M8192 f32 95.7%; sh_down M2048 bf16 93.3%; sh_down M8192 bf16 96.5%; sh_down M2048 f32 95.8%; sh_down M8192 f32 96.7%; gdn_in M2048 bf16 99.2%; gdn_in M8192 bf16 98.2%; gdn_in M2048 f32 99.3%; gdn_in M8192 f32 99.2%; gdn_out M2048 bf16 69.7%; gdn_out M8192 bf16 89.9%; gdn_out M2048 f32 66.8%; gdn_out M8192 f32 90.7%; attn_qkv M2048 bf16 94.7%; attn_qkv M8192 bf16 98.9%; attn_qkv M2048 f32 91.2%; attn_qkv M8192 f32 96.4%; attn_o M2048 bf16 72.2%; attn_o M8192 bf16 88.6%; attn_o M2048 f32 67.8%; attn_o M8192 f32 90.5%; ple_kv M2048 bf16 93.8%; ple_kv M8192 bf16 97.6%; ple_kv M2048 f32 90.9%; ple_kv M8192 f32 97.1%; lm_head M2048 bf16 97.8%; mtp_fc M2048 bf16 92.6%; mtp_fc M8192 bf16 97.5%; mtp_fc M2048 f32 92.8%; mtp_fc M8192 f32 97.1% |
| speed_prefill_expert | PASS | expert-prefill-swiglu-f16h narrow M2048 93.8% (faster); expert-prefill-combine-f16h-shared narrow M2048 93.5% (faster); expert-prefill-combine-f16h-shared wide M8192 95.0% (faster); expert-prefill-swiglu-f16h wide256 M8192 98.2% (noise) |

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
| sh_gu | Q8 | dense-decode | 1 | bf16 | 0.0419 [0.0416-0.0424] | 0.0422 [0.0420-0.0424] | 100.7 | 83 / 82 | noise |
| sh_gu | Q8 | dense-decode | 2 | bf16 | 0.0433 [0.0431-0.0439] | 0.0432 [0.0428-0.0438] | 99.6 | 80 / 81 | noise |
| sh_gu | Q8 | dense-decode | 3 | bf16 | 0.0462 [0.0460-0.0468] | 0.0452 [0.0446-0.0457] | 97.8 | 75 / 77 | faster |
| sh_gu | Q8 | dense-decode | 4 | bf16 | 0.0481 [0.0474-0.0489] | 0.0471 [0.0468-0.0473] | 97.8 | 72 / 74 | faster |
| sh_gu | Q8 | dense-decode | 5 | bf16 | 0.0520 [0.0515-0.0528] | 0.0498 [0.0496-0.0505] | 95.8 | 67 / 70 | faster |
| sh_gu | Q8 | dense-decode | 6 | bf16 | 0.0534 [0.0527-0.0538] | 0.0521 [0.0512-0.0526] | 97.5 | 65 / 67 | faster |
| sh_gu | Q8 | dense-decode | 7 | bf16 | 0.0564 [0.0556-0.0572] | 0.0539 [0.0534-0.0544] | 95.5 | 62 / 65 | faster |
| sh_gu | Q8 | dense-decode | 8 | bf16 | 0.0567 [0.0560-0.0578] | 0.0536 [0.0534-0.0547] | 94.6 | 61 / 65 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 512 | bf16 | 0.1455 [0.1412-0.1486] | 0.1320 [0.1274-0.1346] | 90.7 | 24 / 26 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.4628 [0.4542-0.4720] | 0.4239 [0.4193-0.4427] | 91.6 | 8 / 8 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | bf16 | 1.7332 [1.7158-1.7624] | 1.6891 [1.6579-1.7076] | 97.5 | 2 / 2 | faster |
| sh_gu | Q8 | dense-decode | 1 | f32 | 0.0424 [0.0422-0.0429] | 0.0427 [0.0423-0.0434] | 100.8 | 82 / 82 | noise |
| sh_gu | Q8 | dense-decode | 2 | f32 | 0.0440 [0.0436-0.0444] | 0.0434 [0.0430-0.0440] | 98.5 | 79 / 80 | noise |
| sh_gu | Q8 | dense-decode | 3 | f32 | 0.0455 [0.0453-0.0461] | 0.0453 [0.0448-0.0456] | 99.5 | 76 / 77 | noise |
| sh_gu | Q8 | dense-decode | 4 | f32 | 0.0475 [0.0469-0.0481] | 0.0472 [0.0464-0.0480] | 99.2 | 73 / 74 | noise |
| sh_gu | Q8 | dense-decode | 5 | f32 | 0.0518 [0.0513-0.0524] | 0.0506 [0.0501-0.0511] | 97.6 | 67 / 69 | faster |
| sh_gu | Q8 | dense-decode | 6 | f32 | 0.0539 [0.0534-0.0543] | 0.0518 [0.0515-0.0524] | 96.1 | 65 / 67 | faster |
| sh_gu | Q8 | dense-decode | 7 | f32 | 0.0555 [0.0543-0.0566] | 0.0532 [0.0526-0.0540] | 95.9 | 63 / 65 | faster |
| sh_gu | Q8 | dense-decode | 8 | f32 | 0.0586 [0.0579-0.0593] | 0.0557 [0.0553-0.0564] | 95.2 | 59 / 62 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 512 | f32 | 0.1468 [0.1409-0.1498] | 0.1349 [0.1270-0.1362] | 91.9 | 24 / 26 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | f32 | 2.0057 [0.4908-2.0307] | 1.8701 [0.4441-1.9166] | 93.2 | 2 / 2 | noise |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | f32 | 6.6859 [6.5870-13.8243] | 6.3960 [6.3622-13.0950] | 95.7 | 1 / 1 | noise |
| sh_gu | Q4 syn | dense-decode | 1 | bf16 | 0.1081 [0.1059-0.1095] | 0.1072 [0.1062-0.1081] | 99.1 | 17 / 17 | noise |
| sh_gu | Q4 syn | dense-decode | 2 | bf16 | 0.1221 [0.1199-0.1274] | 0.1122 [0.1106-0.1132] | 91.9 | 15 / 16 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | bf16 | 0.1240 [0.1214-0.1289] | 0.1152 [0.1137-0.1161] | 92.9 | 15 / 16 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | bf16 | 0.1354 [0.1322-0.1388] | 0.1185 [0.1169-0.1191] | 87.6 | 14 / 16 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | bf16 | 0.1402 [0.1377-0.1429] | 0.1248 [0.1228-0.1260] | 89.0 | 13 / 15 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | bf16 | 0.1472 [0.1431-0.1507] | 0.1295 [0.1276-0.1312] | 88.0 | 13 / 14 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | bf16 | 0.0399 [0.0382-0.0424] | 0.0354 [0.0343-0.0360] | 88.9 | 46 / 52 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | bf16 | 0.0426 [0.0412-0.0437] | 0.0365 [0.0356-0.0372] | 85.8 | 43 / 50 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.1272 [0.1256-0.1292] | 0.1250 [0.1226-0.1270] | 98.2 | 14 / 15 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.4398 [0.4222-0.4509] | 0.4215 [0.4092-0.4334] | 95.8 | 4 / 4 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 1.6991 [1.6652-1.7183] | 1.6761 [1.6435-1.6932] | 98.7 | 1 / 1 | noise |
| sh_gu | Q4 syn | dense-decode | 1 | f32 | 0.0259 [0.0257-0.0264] | 0.0252 [0.0250-0.0253] | 97.2 | 71 / 73 | faster |
| sh_gu | Q4 syn | dense-decode | 2 | f32 | 0.0284 [0.0278-0.0298] | 0.0265 [0.0264-0.0265] | 93.4 | 65 / 70 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | f32 | 0.0298 [0.0293-0.0309] | 0.0275 [0.0273-0.0281] | 92.2 | 62 / 67 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | f32 | 0.0318 [0.0313-0.0326] | 0.0292 [0.0289-0.0295] | 92.0 | 58 / 63 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | f32 | 0.0364 [0.0353-0.0374] | 0.0324 [0.0321-0.0325] | 88.8 | 51 / 57 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | f32 | 0.0375 [0.0368-0.0383] | 0.0332 [0.0331-0.0335] | 88.7 | 49 / 55 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | f32 | 0.0400 [0.0395-0.0407] | 0.0346 [0.0343-0.0349] | 86.7 | 46 / 53 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | f32 | 0.0416 [0.0405-0.0424] | 0.0355 [0.0353-0.0359] | 85.4 | 44 / 52 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.1280 [0.1262-0.1296] | 0.1227 [0.1208-0.1247] | 95.8 | 14 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.4481 [0.4454-0.4544] | 0.4426 [0.4370-0.4469] | 98.8 | 4 / 4 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | f32 | 1.6907 [1.6773-1.7242] | 1.6657 [1.6535-1.7027] | 98.5 | 1 / 1 | noise |
| sh_down | Q8 | dense-decode | 1 | bf16 | 0.0260 [0.0258-0.0264] | 0.0254 [0.0253-0.0256] | 97.5 | 71 / 73 | faster |
| sh_down | Q8 | dense-decode | 2 | bf16 | 0.0287 [0.0286-0.0291] | 0.0266 [0.0264-0.0271] | 92.6 | 64 / 69 | faster |
| sh_down | Q8 | dense-decode | 3 | bf16 | 0.0309 [0.0304-0.0312] | 0.0273 [0.0270-0.0274] | 88.2 | 60 / 68 | faster |
| sh_down | Q8 | dense-decode | 4 | bf16 | 0.0335 [0.0332-0.0338] | 0.0284 [0.0282-0.0287] | 84.7 | 55 / 65 | faster |
| sh_down | Q8 | dense-decode | 5 | bf16 | 0.0362 [0.0357-0.0371] | 0.0296 [0.0292-0.0299] | 81.8 | 51 / 62 | faster |
| sh_down | Q8 | dense-decode | 6 | bf16 | 0.0388 [0.0383-0.0393] | 0.0308 [0.0305-0.0310] | 79.5 | 48 / 60 | faster |
| sh_down | Q8 | dense-decode | 7 | bf16 | 0.0389 [0.0386-0.0392] | 0.0316 [0.0313-0.0318] | 81.4 | 47 / 58 | faster |
| sh_down | Q8 | dense-decode | 8 | bf16 | 0.0407 [0.0403-0.0423] | 0.0324 [0.0320-0.0328] | 79.7 | 45 / 57 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | bf16 | 0.0764 [0.0759-0.0770] | 0.0694 [0.0690-0.0698] | 90.8 | 24 / 27 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.2428 [0.2352-0.2484] | 0.2266 [0.2203-0.2284] | 93.3 | 8 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | bf16 | 0.9351 [0.9258-0.9496] | 0.9026 [0.8902-0.9154] | 96.5 | 2 / 2 | faster |
| sh_down | Q8 | dense-decode | 1 | f32 | 0.0255 [0.0250-0.0258] | 0.0253 [0.0251-0.0256] | 99.4 | 72 / 73 | noise |
| sh_down | Q8 | dense-decode | 2 | f32 | 0.0274 [0.0271-0.0276] | 0.0264 [0.0263-0.0266] | 96.4 | 67 / 70 | faster |
| sh_down | Q8 | dense-decode | 3 | f32 | 0.0293 [0.0287-0.0294] | 0.0270 [0.0268-0.0272] | 92.2 | 63 / 68 | faster |
| sh_down | Q8 | dense-decode | 4 | f32 | 0.0306 [0.0304-0.0309] | 0.0276 [0.0274-0.0278] | 90.3 | 60 / 67 | faster |
| sh_down | Q8 | dense-decode | 5 | f32 | 0.0338 [0.0331-0.0343] | 0.0290 [0.0286-0.0291] | 85.8 | 54 / 63 | faster |
| sh_down | Q8 | dense-decode | 6 | f32 | 0.0367 [0.0362-0.0380] | 0.0304 [0.0303-0.0308] | 83.0 | 50 / 61 | faster |
| sh_down | Q8 | dense-decode | 7 | f32 | 0.0392 [0.0378-0.0396] | 0.0319 [0.0316-0.0322] | 81.5 | 47 / 58 | faster |
| sh_down | Q8 | dense-decode | 8 | f32 | 0.0401 [0.0398-0.0407] | 0.0324 [0.0323-0.0327] | 80.8 | 46 / 57 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | f32 | 0.0847 [0.0836-0.0853] | 0.0772 [0.0766-0.0778] | 91.1 | 22 / 24 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | f32 | 0.2471 [0.2428-0.2516] | 0.2368 [0.2322-0.2397] | 95.8 | 7 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | f32 | 0.9271 [0.9106-0.9390] | 0.8962 [0.8818-0.9130] | 96.7 | 2 / 2 | noise |
| sh_down | Q4 syn | dense-decode | 1 | bf16 | 0.0168 [0.0166-0.0170] | 0.0170 [0.0168-0.0172] | 101.0 | 61 / 60 | noise |
| sh_down | Q4 syn | dense-decode | 2 | bf16 | 0.0204 [0.0202-0.0206] | 0.0177 [0.0176-0.0180] | 87.0 | 50 / 58 | faster |
| sh_down | Q4 syn | dense-decode | 3 | bf16 | 0.0222 [0.0219-0.0225] | 0.0186 [0.0183-0.0188] | 83.6 | 46 / 55 | faster |
| sh_down | Q4 syn | dense-decode | 4 | bf16 | 0.0240 [0.0239-0.0245] | 0.0196 [0.0194-0.0201] | 81.7 | 43 / 52 | faster |
| sh_down | Q4 syn | dense-decode | 5 | bf16 | 0.0286 [0.0281-0.0290] | 0.0216 [0.0212-0.0218] | 75.3 | 36 / 47 | faster |
| sh_down | Q4 syn | dense-decode | 6 | bf16 | 0.0288 [0.0285-0.0303] | 0.0222 [0.0219-0.0227] | 76.8 | 36 / 46 | faster |
| sh_down | Q4 syn | dense-decode | 7 | bf16 | 0.0305 [0.0298-0.0314] | 0.0231 [0.0228-0.0234] | 75.7 | 34 / 44 | faster |
| sh_down | Q4 syn | dense-decode | 8 | bf16 | 0.0321 [0.0316-0.0330] | 0.0242 [0.0236-0.0244] | 75.2 | 32 / 42 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.0761 [0.0742-0.0765] | 0.0678 [0.0675-0.0680] | 89.1 | 13 / 15 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.2402 [0.2303-0.2447] | 0.2262 [0.2174-0.2278] | 94.2 | 4 / 5 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 0.9042 [0.9011-0.9184] | 0.8806 [0.8697-0.8905] | 97.4 | 1 / 1 | faster |
| sh_down | Q4 syn | dense-decode | 1 | f32 | 0.0168 [0.0164-0.0169] | 0.0168 [0.0167-0.0169] | 100.0 | 61 / 61 | noise |
| sh_down | Q4 syn | dense-decode | 2 | f32 | 0.0187 [0.0186-0.0188] | 0.0177 [0.0174-0.0178] | 94.4 | 55 / 58 | faster |
| sh_down | Q4 syn | dense-decode | 3 | f32 | 0.0205 [0.0204-0.0208] | 0.0183 [0.0180-0.0185] | 89.3 | 50 / 56 | faster |
| sh_down | Q4 syn | dense-decode | 4 | f32 | 0.0219 [0.0215-0.0222] | 0.0190 [0.0188-0.0192] | 86.7 | 47 / 54 | faster |
| sh_down | Q4 syn | dense-decode | 5 | f32 | 0.0264 [0.0260-0.0268] | 0.0206 [0.0202-0.0210] | 77.9 | 39 / 50 | faster |
| sh_down | Q4 syn | dense-decode | 6 | f32 | 0.0276 [0.0271-0.0280] | 0.0217 [0.0213-0.0219] | 78.7 | 37 / 47 | faster |
| sh_down | Q4 syn | dense-decode | 7 | f32 | 0.0290 [0.0284-0.0294] | 0.0226 [0.0222-0.0228] | 77.9 | 35 / 45 | faster |
| sh_down | Q4 syn | dense-decode | 8 | f32 | 0.0312 [0.0306-0.0320] | 0.0245 [0.0239-0.0249] | 78.7 | 33 / 42 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.0830 [0.0827-0.0835] | 0.0761 [0.0753-0.0764] | 91.7 | 12 / 13 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.2554 [0.2526-0.2631] | 0.2382 [0.2341-0.2440] | 93.3 | 4 / 4 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | f32 | 0.9070 [0.8979-0.9144] | 0.8617 [0.8538-0.8685] | 95.0 | 1 / 1 | faster |
| gdn_in | Q4 | dense-decode | 1 | bf16 | 0.9807 [0.9758-0.9861] | 0.9759 [0.9697-0.9812] | 99.5 | 23 / 23 | noise |
| gdn_in | Q4 | dense-decode | 2 | bf16 | 0.9967 [0.9900-1.0045] | 0.9835 [0.9797-0.9876] | 98.7 | 22 / 23 | faster |
| gdn_in | Q4 | dense-decode | 3 | bf16 | 1.0253 [1.0123-1.0279] | 0.9979 [0.9889-1.0076] | 97.3 | 22 / 22 | faster |
| gdn_in | Q4 | dense-decode | 4 | bf16 | 1.0426 [1.0362-1.0498] | 1.0069 [0.9950-1.0169] | 96.6 | 22 / 22 | faster |
| gdn_in | Q4 | dense-decode | 5 | bf16 | 1.0807 [1.0764-1.0851] | 1.0332 [1.0226-1.0402] | 95.6 | 21 / 22 | faster |
| gdn_in | Q4 | dense-decode | 6 | bf16 | 1.0956 [1.0889-1.1020] | 1.0405 [1.0312-1.0473] | 95.0 | 20 / 22 | faster |
| gdn_in | Q4 | dense-decode | 7 | bf16 | 1.1211 [1.1160-1.1336] | 1.0591 [1.0538-1.0658] | 94.5 | 20 / 21 | faster |
| gdn_in | Q4 | dense-decode | 8 | bf16 | 1.1544 [1.1475-1.1660] | 1.0868 [1.0755-1.0972] | 94.2 | 19 / 21 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | bf16 | 5.3785 [5.3485-5.3966] | 5.2375 [5.2298-5.2522] | 97.4 | 4 / 4 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | bf16 | 20.2448 [20.1850-20.2680] | 20.0885 [20.0268-20.1516] | 99.2 | 1 / 1 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | bf16 | 23.4797 [23.1649-24.0373] | 23.0600 [22.8592-23.6856] | 98.2 | 1 / 1 | noise |
| gdn_in | Q4 | dense-decode | 1 | f32 | 0.2357 [0.2344-0.2371] | 0.2363 [0.2343-0.2393] | 100.2 | 95 / 95 | noise |
| gdn_in | Q4 | dense-decode | 2 | f32 | 0.2381 [0.2358-0.2397] | 0.2351 [0.2332-0.2377] | 98.7 | 94 / 95 | noise |
| gdn_in | Q4 | dense-decode | 3 | f32 | 0.2418 [0.2378-0.2452] | 0.2375 [0.2351-0.2398] | 98.2 | 93 / 94 | noise |
| gdn_in | Q4 | dense-decode | 4 | f32 | 0.2464 [0.2442-0.2505] | 0.2414 [0.2397-0.2458] | 98.0 | 91 / 93 | noise |
| gdn_in | Q4 | dense-decode | 5 | f32 | 0.2540 [0.2519-0.2550] | 0.2448 [0.2431-0.2458] | 96.4 | 88 / 92 | faster |
| gdn_in | Q4 | dense-decode | 6 | f32 | 0.2587 [0.2559-0.2650] | 0.2484 [0.2455-0.2510] | 96.0 | 87 / 90 | faster |
| gdn_in | Q4 | dense-decode | 7 | f32 | 0.2626 [0.2604-0.2637] | 0.2508 [0.2498-0.2550] | 95.5 | 85 / 89 | faster |
| gdn_in | Q4 | dense-decode | 8 | f32 | 0.2671 [0.2652-0.2689] | 0.2535 [0.2502-0.2555] | 94.9 | 84 / 88 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | f32 | 1.3469 [1.3365-1.3582] | 1.3334 [1.3127-1.3514] | 99.0 | 17 / 17 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | f32 | 5.5414 [5.5190-5.5732] | 5.5054 [5.4685-5.5495] | 99.3 | 4 / 4 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | f32 | 23.3041 [23.1791-23.5356] | 23.1298 [22.9121-23.2910] | 99.2 | 1 / 1 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | bf16 | 0.3939 [0.3924-0.3969] | 0.3971 [0.3958-0.3989] | 100.8 | 110 / 110 | noise |
| gdn_in | Q8 syn | dense-decode | 2 | bf16 | 0.3957 [0.3944-0.3986] | 0.4008 [0.3986-0.4024] | 101.3 | 110 / 109 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | bf16 | 0.4109 [0.4071-0.4145] | 0.4069 [0.4035-0.4109] | 99.0 | 106 / 107 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | bf16 | 0.4149 [0.4127-0.4168] | 0.4083 [0.4065-0.4118] | 98.4 | 105 / 107 | faster |
| gdn_in | Q8 syn | dense-decode | 5 | bf16 | 0.4355 [0.4311-0.4380] | 0.4204 [0.4176-0.4232] | 96.5 | 100 / 104 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | bf16 | 0.4389 [0.4359-0.4421] | 0.4228 [0.4216-0.4251] | 96.3 | 99 / 103 | faster |
| gdn_in | Q8 syn | dense-decode | 7 | bf16 | 0.4481 [0.4436-0.4515] | 0.4326 [0.4286-0.4382] | 96.5 | 97 / 101 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | bf16 | 0.4508 [0.4475-0.4556] | 0.4416 [0.4376-0.4468] | 98.0 | 97 / 99 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | bf16 | 1.6261 [1.5855-1.6433] | 1.3587 [1.3300-1.3801] | 83.6 | 27 / 32 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | bf16 | 5.8286 [5.3086-6.0697] | 5.5468 [5.0638-5.7437] | 95.2 | 7 / 8 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | bf16 | 23.9805 [23.8742-24.0823] | 23.5092 [23.4088-23.7215] | 98.0 | 2 / 2 | faster |
| gdn_in | Q8 syn | dense-decode | 1 | f32 | 0.3921 [0.3895-0.3951] | 0.3960 [0.3954-0.3994] | 101.0 | 111 / 110 | slower |
| gdn_in | Q8 syn | dense-decode | 2 | f32 | 0.3924 [0.3890-0.3954] | 0.3983 [0.3968-0.4024] | 101.5 | 111 / 109 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | f32 | 0.4026 [0.4004-0.4050] | 0.4007 [0.3998-0.4034] | 99.5 | 108 / 109 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | f32 | 0.4088 [0.4075-0.4101] | 0.4066 [0.4057-0.4088] | 99.5 | 106 / 107 | noise |
| gdn_in | Q8 syn | dense-decode | 5 | f32 | 0.4298 [0.4276-0.4325] | 0.4182 [0.4169-0.4218] | 97.3 | 101 / 104 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | f32 | 0.4416 [0.4378-0.4440] | 0.4258 [0.4209-0.4301] | 96.4 | 99 / 102 | faster |
| gdn_in | Q8 syn | dense-decode | 7 | f32 | 0.4456 [0.4439-0.4492] | 0.4351 [0.4317-0.4372] | 97.7 | 98 / 100 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | f32 | 0.4506 [0.4484-0.4558] | 0.4436 [0.4366-0.4486] | 98.5 | 97 / 98 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | f32 | 1.5815 [1.5635-1.6145] | 1.3330 [1.3183-1.3716] | 84.3 | 28 / 33 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | f32 | 6.2652 [5.6080-6.3129] | 5.9629 [5.4778-6.0935] | 95.2 | 7 / 7 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | f32 | 23.7524 [23.4634-23.9307] | 23.5248 [23.2464-23.5912] | 99.0 | 2 / 2 | noise |
| gdn_out | Q8 | dense-decode | 1 | bf16 | 0.1807 [0.1798-0.1822] | 0.1811 [0.1800-0.1820] | 100.2 | 92 / 92 | noise |
| gdn_out | Q8 | dense-decode | 2 | bf16 | 0.1822 [0.1817-0.1846] | 0.1830 [0.1818-0.1842] | 100.4 | 92 / 91 | noise |
| gdn_out | Q8 | dense-decode | 3 | bf16 | 0.1884 [0.1872-0.1904] | 0.1884 [0.1875-0.1898] | 100.0 | 89 / 89 | noise |
| gdn_out | Q8 | dense-decode | 4 | bf16 | 0.1949 [0.1918-0.1971] | 0.1926 [0.1908-0.1942] | 98.8 | 86 / 87 | noise |
| gdn_out | Q8 | dense-decode | 5 | bf16 | 0.2067 [0.2003-0.2111] | 0.2058 [0.2047-0.2065] | 99.5 | 81 / 81 | noise |
| gdn_out | Q8 | dense-decode | 6 | bf16 | 0.2181 [0.2130-0.2226] | 0.2096 [0.2087-0.2109] | 96.1 | 77 / 80 | faster |
| gdn_out | Q8 | dense-decode | 7 | bf16 | 0.2256 [0.2238-0.2298] | 0.2150 [0.2129-0.2155] | 95.3 | 74 / 78 | faster |
| gdn_out | Q8 | dense-decode | 8 | bf16 | 0.2319 [0.2285-0.2356] | 0.2185 [0.2175-0.2207] | 94.2 | 72 / 76 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3907 [1.2758-1.4957] | 0.8348 [0.7930-0.8700] | 60.0 | 12 / 20 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.0654 [3.0125-3.1233] | 2.1352 [2.0801-2.2274] | 69.7 | 5 / 8 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | bf16 | 9.2804 [8.4763-9.3937] | 8.3471 [7.6615-8.4793] | 89.9 | 2 / 2 | noise |
| gdn_out | Q8 | dense-decode | 1 | f32 | 0.1813 [0.1806-0.1821] | 0.1814 [0.1796-0.1822] | 100.0 | 92 / 92 | noise |
| gdn_out | Q8 | dense-decode | 2 | f32 | 0.1829 [0.1816-0.1835] | 0.1814 [0.1809-0.1822] | 99.2 | 91 / 92 | noise |
| gdn_out | Q8 | dense-decode | 3 | f32 | 0.1866 [0.1858-0.1874] | 0.1884 [0.1877-0.1900] | 100.9 | 90 / 89 | slower |
| gdn_out | Q8 | dense-decode | 4 | f32 | 0.1912 [0.1894-0.1941] | 0.1928 [0.1924-0.1944] | 100.9 | 87 / 87 | noise |
| gdn_out | Q8 | dense-decode | 5 | f32 | 0.2023 [0.1992-0.2094] | 0.2057 [0.2040-0.2066] | 101.7 | 83 / 81 | noise |
| gdn_out | Q8 | dense-decode | 6 | f32 | 0.2196 [0.2160-0.2229] | 0.2103 [0.2082-0.2110] | 95.8 | 76 / 79 | faster |
| gdn_out | Q8 | dense-decode | 7 | f32 | 0.2278 [0.2244-0.2302] | 0.2138 [0.2118-0.2148] | 93.8 | 73 / 78 | faster |
| gdn_out | Q8 | dense-decode | 8 | f32 | 0.2316 [0.2267-0.2362] | 0.2180 [0.2162-0.2190] | 94.1 | 72 / 77 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 512 | f32 | 1.3344 [1.2571-1.4222] | 0.7802 [0.7427-0.8029] | 58.5 | 13 / 21 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | f32 | 3.1168 [3.0464-3.1629] | 2.0821 [2.0513-2.2033] | 66.8 | 5 / 8 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | f32 | 9.8272 [9.5527-9.8789] | 8.9125 [8.6213-8.9921] | 90.7 | 2 / 2 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | bf16 | 0.1036 [0.1028-0.1041] | 0.1028 [0.1024-0.1036] | 99.2 | 85 / 86 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | bf16 | 0.1060 [0.1044-0.1073] | 0.1052 [0.1046-0.1062] | 99.2 | 84 / 84 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | bf16 | 0.1095 [0.1087-0.1117] | 0.1082 [0.1072-0.1096] | 98.8 | 81 / 82 | noise |
| gdn_out | Q4 syn | dense-decode | 4 | bf16 | 0.1128 [0.1121-0.1138] | 0.1092 [0.1080-0.1105] | 96.9 | 78 / 81 | faster |
| gdn_out | Q4 syn | dense-decode | 5 | bf16 | 0.1196 [0.1185-0.1203] | 0.1171 [0.1160-0.1182] | 98.0 | 74 / 76 | faster |
| gdn_out | Q4 syn | dense-decode | 6 | bf16 | 0.1237 [0.1230-0.1243] | 0.1201 [0.1194-0.1211] | 97.1 | 72 / 74 | faster |
| gdn_out | Q4 syn | dense-decode | 7 | bf16 | 0.1270 [0.1262-0.1293] | 0.1222 [0.1211-0.1229] | 96.2 | 70 / 72 | faster |
| gdn_out | Q4 syn | dense-decode | 8 | bf16 | 0.1325 [0.1316-0.1337] | 0.1268 [0.1255-0.1283] | 95.7 | 67 / 70 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0240 [1.0122-1.0625] | 0.6232 [0.6117-0.6608] | 60.9 | 9 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.6066 [2.5513-2.6437] | 2.0250 [1.9706-2.0913] | 77.7 | 3 / 4 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 9.0695 [8.9111-9.1538] | 8.5461 [8.4301-8.5990] | 94.2 | 1 / 1 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | f32 | 0.1024 [0.1018-0.1031] | 0.1032 [0.1023-0.1038] | 100.7 | 86 / 86 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | f32 | 0.1046 [0.1036-0.1061] | 0.1042 [0.1036-0.1049] | 99.6 | 85 / 85 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | f32 | 0.1085 [0.1076-0.1100] | 0.1070 [0.1066-0.1080] | 98.6 | 82 / 83 | noise |
| gdn_out | Q4 syn | dense-decode | 4 | f32 | 0.1116 [0.1112-0.1120] | 0.1101 [0.1091-0.1109] | 98.6 | 79 / 80 | faster |
| gdn_out | Q4 syn | dense-decode | 5 | f32 | 0.1181 [0.1178-0.1197] | 0.1170 [0.1157-0.1182] | 99.1 | 75 / 76 | noise |
| gdn_out | Q4 syn | dense-decode | 6 | f32 | 0.1228 [0.1214-0.1236] | 0.1205 [0.1186-0.1209] | 98.1 | 72 / 73 | faster |
| gdn_out | Q4 syn | dense-decode | 7 | f32 | 0.1282 [0.1266-0.1290] | 0.1237 [0.1229-0.1249] | 96.4 | 69 / 72 | faster |
| gdn_out | Q4 syn | dense-decode | 8 | f32 | 0.1318 [0.1303-0.1332] | 0.1265 [0.1250-0.1275] | 96.0 | 67 / 70 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0262 [1.0102-1.0485] | 0.6199 [0.6066-0.6333] | 60.4 | 9 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.5263 [2.4619-2.5777] | 1.9635 [1.9245-2.0531] | 77.7 | 4 / 5 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | f32 | 9.2048 [9.0910-9.3377] | 8.7408 [8.5467-8.9032] | 95.0 | 1 / 1 | faster |
| attn_qkv | Q8 | dense-decode | 1 | bf16 | 0.3756 [0.3740-0.3777] | 0.3769 [0.3754-0.3778] | 100.3 | 101 / 101 | noise |
| attn_qkv | Q8 | dense-decode | 2 | bf16 | 0.3773 [0.3743-0.3789] | 0.3796 [0.3778-0.3836] | 100.6 | 101 / 100 | noise |
| attn_qkv | Q8 | dense-decode | 3 | bf16 | 0.3824 [0.3812-0.3865] | 0.3832 [0.3823-0.3850] | 100.2 | 99 / 99 | noise |
| attn_qkv | Q8 | dense-decode | 4 | bf16 | 0.3901 [0.3886-0.3926] | 0.3887 [0.3864-0.3908] | 99.6 | 97 / 98 | noise |
| attn_qkv | Q8 | dense-decode | 5 | bf16 | 0.4098 [0.4062-0.4137] | 0.4030 [0.3979-0.4090] | 98.3 | 93 / 94 | noise |
| attn_qkv | Q8 | dense-decode | 6 | bf16 | 0.4222 [0.4152-0.4275] | 0.4123 [0.4073-0.4177] | 97.6 | 90 / 92 | noise |
| attn_qkv | Q8 | dense-decode | 7 | bf16 | 0.4223 [0.4178-0.4236] | 0.4144 [0.4101-0.4193] | 98.1 | 90 / 92 | noise |
| attn_qkv | Q8 | dense-decode | 8 | bf16 | 0.4348 [0.4256-0.4390] | 0.4283 [0.4196-0.4319] | 98.5 | 87 / 89 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3821 [1.3646-1.4231] | 1.1638 [1.1521-1.1759] | 84.2 | 27 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | bf16 | 5.3489 [5.3065-5.3947] | 5.0670 [4.9736-5.1039] | 94.7 | 7 / 7 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | bf16 | 20.4277 [20.2147-21.1578] | 20.1976 [19.7592-20.4453] | 98.9 | 2 / 2 | noise |
| attn_qkv | Q8 | dense-decode | 1 | f32 | 0.3698 [0.3684-0.3710] | 0.3707 [0.3691-0.3720] | 100.2 | 103 / 102 | noise |
| attn_qkv | Q8 | dense-decode | 2 | f32 | 0.3677 [0.3649-0.3694] | 0.3707 [0.3702-0.3729] | 100.8 | 103 / 102 | slower |
| attn_qkv | Q8 | dense-decode | 3 | f32 | 0.3780 [0.3737-0.3799] | 0.3775 [0.3751-0.3779] | 99.9 | 100 / 101 | noise |
| attn_qkv | Q8 | dense-decode | 4 | f32 | 0.3841 [0.3809-0.3858] | 0.3814 [0.3802-0.3824] | 99.3 | 99 / 100 | noise |
| attn_qkv | Q8 | dense-decode | 5 | f32 | 0.4007 [0.3977-0.4028] | 0.3950 [0.3919-0.3977] | 98.6 | 95 / 96 | noise |
| attn_qkv | Q8 | dense-decode | 6 | f32 | 0.4083 [0.4059-0.4139] | 0.4044 [0.4006-0.4066] | 99.0 | 93 / 94 | noise |
| attn_qkv | Q8 | dense-decode | 7 | f32 | 0.4173 [0.4138-0.4234] | 0.4087 [0.4054-0.4162] | 98.0 | 91 / 93 | noise |
| attn_qkv | Q8 | dense-decode | 8 | f32 | 0.4214 [0.4183-0.4237] | 0.4166 [0.4104-0.4200] | 98.9 | 90 / 91 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | f32 | 1.4057 [1.3784-1.4247] | 1.1441 [1.1322-1.1599] | 81.4 | 27 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | f32 | 5.0005 [4.6110-5.3474] | 4.5622 [4.2708-5.0493] | 91.2 | 8 / 8 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | f32 | 19.8661 [19.1782-19.9632] | 19.1528 [18.9485-19.3663] | 96.4 | 2 / 2 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | bf16 | 0.2154 [0.2132-0.2169] | 0.2162 [0.2145-0.2169] | 100.4 | 93 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | bf16 | 0.2177 [0.2160-0.2198] | 0.2156 [0.2148-0.2181] | 99.0 | 92 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 3 | bf16 | 0.2222 [0.2204-0.2246] | 0.2184 [0.2180-0.2205] | 98.3 | 90 / 92 | noise |
| attn_qkv | Q4 syn | dense-decode | 4 | bf16 | 0.2247 [0.2228-0.2265] | 0.2215 [0.2194-0.2240] | 98.6 | 89 / 91 | noise |
| attn_qkv | Q4 syn | dense-decode | 5 | bf16 | 0.2330 [0.2298-0.2346] | 0.2241 [0.2209-0.2260] | 96.2 | 86 / 90 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | bf16 | 0.2372 [0.2356-0.2386] | 0.2280 [0.2256-0.2290] | 96.1 | 85 / 88 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | bf16 | 0.2400 [0.2370-0.2435] | 0.2279 [0.2268-0.2308] | 95.0 | 84 / 88 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | bf16 | 0.2483 [0.2477-0.2504] | 0.2328 [0.2311-0.2340] | 93.8 | 81 / 86 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0762 [1.0695-1.0926] | 1.0254 [1.0074-1.0349] | 95.3 | 19 / 20 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 4.5028 [4.2260-4.8281] | 4.3202 [4.1680-4.7545] | 96.0 | 4 / 5 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 19.1581 [18.9806-19.5900] | 19.0023 [18.9431-19.2371] | 99.2 | 1 / 1 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | f32 | 0.2149 [0.2133-0.2163] | 0.2154 [0.2133-0.2162] | 100.2 | 93 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | f32 | 0.2172 [0.2160-0.2180] | 0.2142 [0.2130-0.2156] | 98.6 | 93 / 94 | faster |
| attn_qkv | Q4 syn | dense-decode | 3 | f32 | 0.2204 [0.2190-0.2214] | 0.2175 [0.2154-0.2198] | 98.7 | 91 / 92 | noise |
| attn_qkv | Q4 syn | dense-decode | 4 | f32 | 0.2223 [0.2198-0.2250] | 0.2196 [0.2175-0.2210] | 98.8 | 90 / 92 | noise |
| attn_qkv | Q4 syn | dense-decode | 5 | f32 | 0.2302 [0.2288-0.2321] | 0.2247 [0.2234-0.2262] | 97.6 | 87 / 89 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | f32 | 0.2351 [0.2330-0.2374] | 0.2268 [0.2248-0.2283] | 96.5 | 85 / 89 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | f32 | 0.2406 [0.2391-0.2425] | 0.2281 [0.2261-0.2302] | 94.8 | 83 / 88 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | f32 | 0.2452 [0.2426-0.2477] | 0.2319 [0.2296-0.2356] | 94.6 | 82 / 87 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0547 [1.0449-1.0667] | 1.0046 [0.9941-1.0102] | 95.2 | 19 / 20 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 4.4768 [4.1959-4.9134] | 4.3122 [4.1630-4.8793] | 96.3 | 4 / 5 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 18.8627 [18.7128-18.9927] | 18.6679 [18.6244-18.8303] | 99.0 | 1 / 1 | noise |
| attn_o | Q8 | dense-decode | 1 | bf16 | 0.1804 [0.1796-0.1809] | 0.1803 [0.1798-0.1814] | 99.9 | 93 / 93 | noise |
| attn_o | Q8 | dense-decode | 2 | bf16 | 0.1830 [0.1812-0.1838] | 0.1816 [0.1806-0.1820] | 99.2 | 91 / 92 | noise |
| attn_o | Q8 | dense-decode | 3 | bf16 | 0.1871 [0.1848-0.1876] | 0.1881 [0.1870-0.1892] | 100.5 | 89 / 89 | noise |
| attn_o | Q8 | dense-decode | 4 | bf16 | 0.1882 [0.1870-0.1916] | 0.1928 [0.1920-0.1932] | 102.4 | 89 / 87 | slower |
| attn_o | Q8 | dense-decode | 5 | bf16 | 0.1956 [0.1912-0.1999] | 0.2045 [0.2038-0.2055] | 104.5 | 85 / 82 | slower |
| attn_o | Q8 | dense-decode | 6 | bf16 | 0.2101 [0.2032-0.2169] | 0.2089 [0.2082-0.2099] | 99.4 | 80 / 80 | noise |
| attn_o | Q8 | dense-decode | 7 | bf16 | 0.2213 [0.2167-0.2232] | 0.2125 [0.2115-0.2148] | 96.0 | 76 / 79 | faster |
| attn_o | Q8 | dense-decode | 8 | bf16 | 0.2268 [0.2234-0.2310] | 0.2174 [0.2171-0.2185] | 95.8 | 74 / 77 | faster |
| attn_o | Q8 | dense-prefill-wmma | 512 | bf16 | 1.4031 [1.2321-1.6345] | 0.7696 [0.7511-0.8497] | 54.9 | 12 / 22 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.2172 [3.1922-3.2527] | 2.3221 [2.2699-2.3702] | 72.2 | 5 / 7 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.5717 [8.4533-9.3601] | 7.5957 [7.5368-8.4577] | 88.6 | 2 / 2 | noise |
| attn_o | Q8 | dense-decode | 1 | f32 | 0.1808 [0.1798-0.1825] | 0.1808 [0.1802-0.1817] | 100.0 | 92 / 92 | noise |
| attn_o | Q8 | dense-decode | 2 | f32 | 0.1826 [0.1823-0.1844] | 0.1823 [0.1811-0.1828] | 99.8 | 92 / 92 | noise |
| attn_o | Q8 | dense-decode | 3 | f32 | 0.1875 [0.1860-0.1892] | 0.1891 [0.1878-0.1905] | 100.8 | 89 / 88 | noise |
| attn_o | Q8 | dense-decode | 4 | f32 | 0.1922 [0.1901-0.1943] | 0.1948 [0.1921-0.1962] | 101.3 | 87 / 86 | noise |
| attn_o | Q8 | dense-decode | 5 | f32 | 0.1970 [0.1942-0.2011] | 0.2060 [0.2049-0.2068] | 104.6 | 85 / 81 | slower |
| attn_o | Q8 | dense-decode | 6 | f32 | 0.2124 [0.2072-0.2156] | 0.2102 [0.2086-0.2113] | 99.0 | 79 / 79 | noise |
| attn_o | Q8 | dense-decode | 7 | f32 | 0.2207 [0.2152-0.2242] | 0.2127 [0.2119-0.2137] | 96.4 | 76 / 79 | faster |
| attn_o | Q8 | dense-decode | 8 | f32 | 0.2263 [0.2188-0.2312] | 0.2174 [0.2160-0.2190] | 96.1 | 74 / 77 | noise |
| attn_o | Q8 | dense-prefill-wmma | 512 | f32 | 1.3259 [1.2349-1.4517] | 0.7521 [0.7403-0.7720] | 56.7 | 13 / 22 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | f32 | 3.0520 [3.0183-3.1104] | 2.0680 [2.0509-2.0981] | 67.8 | 5 / 8 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | f32 | 9.6162 [9.5226-9.6947] | 8.7054 [8.0433-8.7525] | 90.5 | 2 / 2 | faster |
| attn_o | Q4 syn | dense-decode | 1 | bf16 | 0.1020 [0.1017-0.1028] | 0.1021 [0.1008-0.1025] | 100.1 | 87 / 87 | noise |
| attn_o | Q4 syn | dense-decode | 2 | bf16 | 0.1054 [0.1040-0.1061] | 0.1036 [0.1031-0.1045] | 98.3 | 84 / 85 | noise |
| attn_o | Q4 syn | dense-decode | 3 | bf16 | 0.1095 [0.1085-0.1108] | 0.1066 [0.1058-0.1075] | 97.3 | 81 / 83 | faster |
| attn_o | Q4 syn | dense-decode | 4 | bf16 | 0.1134 [0.1122-0.1138] | 0.1090 [0.1083-0.1099] | 96.2 | 78 / 81 | faster |
| attn_o | Q4 syn | dense-decode | 5 | bf16 | 0.1196 [0.1177-0.1204] | 0.1180 [0.1170-0.1186] | 98.7 | 74 / 75 | noise |
| attn_o | Q4 syn | dense-decode | 6 | bf16 | 0.1259 [0.1246-0.1282] | 0.1233 [0.1215-0.1252] | 97.9 | 70 / 72 | noise |
| attn_o | Q4 syn | dense-decode | 7 | bf16 | 0.1281 [0.1264-0.1288] | 0.1220 [0.1212-0.1235] | 95.2 | 69 / 73 | faster |
| attn_o | Q4 syn | dense-decode | 8 | bf16 | 0.1343 [0.1335-0.1375] | 0.1314 [0.1304-0.1343] | 97.8 | 66 / 67 | noise |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0244 [1.0087-1.0511] | 0.6373 [0.6264-0.6818] | 62.2 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.7345 [2.6700-2.7727] | 2.2469 [2.2037-2.2866] | 82.2 | 3 / 4 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 8.0179 [7.9654-8.5581] | 7.4872 [7.3655-8.0999] | 93.4 | 1 / 1 | noise |
| attn_o | Q4 syn | dense-decode | 1 | f32 | 0.1031 [0.1026-0.1044] | 0.1029 [0.1025-0.1039] | 99.8 | 86 / 86 | noise |
| attn_o | Q4 syn | dense-decode | 2 | f32 | 0.1050 [0.1041-0.1061] | 0.1044 [0.1034-0.1057] | 99.5 | 84 / 85 | noise |
| attn_o | Q4 syn | dense-decode | 3 | f32 | 0.1094 [0.1084-0.1100] | 0.1074 [0.1065-0.1081] | 98.1 | 81 / 82 | faster |
| attn_o | Q4 syn | dense-decode | 4 | f32 | 0.1128 [0.1123-0.1139] | 0.1104 [0.1095-0.1106] | 97.8 | 78 / 80 | faster |
| attn_o | Q4 syn | dense-decode | 5 | f32 | 0.1196 [0.1188-0.1214] | 0.1196 [0.1178-0.1208] | 100.0 | 74 / 74 | noise |
| attn_o | Q4 syn | dense-decode | 6 | f32 | 0.1237 [0.1224-0.1247] | 0.1208 [0.1186-0.1232] | 97.7 | 72 / 73 | noise |
| attn_o | Q4 syn | dense-decode | 7 | f32 | 0.1272 [0.1257-0.1286] | 0.1229 [0.1221-0.1249] | 96.6 | 70 / 72 | faster |
| attn_o | Q4 syn | dense-decode | 8 | f32 | 0.1319 [0.1304-0.1332] | 0.1256 [0.1251-0.1270] | 95.2 | 67 / 70 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0315 [1.0075-1.0495] | 0.6181 [0.6106-0.6297] | 59.9 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.5640 [2.5088-2.6783] | 1.9381 [1.9212-2.0238] | 75.6 | 3 / 5 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | f32 | 9.1426 [9.0616-9.2111] | 8.6619 [8.4932-8.7397] | 94.7 | 1 / 1 | faster |
| ple_kv | Q8 | dense-decode | 1 | bf16 | 0.3576 [0.3557-0.3610] | 0.3579 [0.3554-0.3598] | 100.1 | 97 / 97 | noise |
| ple_kv | Q8 | dense-decode | 2 | bf16 | 0.3572 [0.3555-0.3606] | 0.3620 [0.3602-0.3646] | 101.4 | 97 / 96 | noise |
| ple_kv | Q8 | dense-decode | 3 | bf16 | 0.3632 [0.3608-0.3662] | 0.3644 [0.3636-0.3682] | 100.3 | 96 / 96 | noise |
| ple_kv | Q8 | dense-decode | 4 | bf16 | 0.3713 [0.3691-0.3732] | 0.3761 [0.3733-0.3777] | 101.3 | 94 / 93 | slower |
| ple_kv | Q8 | dense-decode | 5 | bf16 | 0.3870 [0.3857-0.3897] | 0.3893 [0.3855-0.3942] | 100.6 | 90 / 89 | noise |
| ple_kv | Q8 | dense-decode | 6 | bf16 | 0.3963 [0.3932-0.3989] | 0.3976 [0.3922-0.3997] | 100.3 | 88 / 88 | noise |
| ple_kv | Q8 | dense-decode | 7 | bf16 | 0.4015 [0.3976-0.4043] | 0.4036 [0.3996-0.4082] | 100.5 | 87 / 86 | noise |
| ple_kv | Q8 | dense-decode | 8 | bf16 | 0.4049 [0.4008-0.4087] | 0.4137 [0.4042-0.4146] | 102.2 | 86 / 84 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.2569 [1.2455-1.2886] | 1.0260 [1.0192-1.0321] | 81.6 | 28 / 34 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.3213 [4.1887-4.8005] | 4.0552 [3.9061-4.4195] | 93.8 | 8 / 9 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | bf16 | 19.4058 [18.9153-19.5303] | 18.9439 [18.5725-19.1374] | 97.6 | 2 / 2 | noise |
| ple_kv | Q8 | dense-decode | 1 | f32 | 0.3567 [0.3552-0.3613] | 0.3578 [0.3559-0.3603] | 100.3 | 98 / 97 | noise |
| ple_kv | Q8 | dense-decode | 2 | f32 | 0.3572 [0.3544-0.3616] | 0.3593 [0.3574-0.3635] | 100.6 | 97 / 97 | noise |
| ple_kv | Q8 | dense-decode | 3 | f32 | 0.3623 [0.3607-0.3650] | 0.3650 [0.3631-0.3683] | 100.8 | 96 / 95 | noise |
| ple_kv | Q8 | dense-decode | 4 | f32 | 0.3680 [0.3654-0.3708] | 0.3720 [0.3689-0.3760] | 101.1 | 95 / 94 | noise |
| ple_kv | Q8 | dense-decode | 5 | f32 | 0.3840 [0.3824-0.3853] | 0.3794 [0.3761-0.3833] | 98.8 | 91 / 92 | noise |
| ple_kv | Q8 | dense-decode | 6 | f32 | 0.3943 [0.3914-0.3963] | 0.3911 [0.3864-0.3955] | 99.2 | 88 / 89 | noise |
| ple_kv | Q8 | dense-decode | 7 | f32 | 0.3976 [0.3962-0.4002] | 0.3994 [0.3959-0.4028] | 100.5 | 88 / 87 | noise |
| ple_kv | Q8 | dense-decode | 8 | f32 | 0.4046 [0.4034-0.4098] | 0.4051 [0.4004-0.4075] | 100.1 | 86 / 86 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 512 | f32 | 1.2609 [1.2461-1.2997] | 1.0279 [1.0247-1.0473] | 81.5 | 28 / 34 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | f32 | 4.5058 [4.1350-4.9299] | 4.0960 [3.9062-4.5832] | 90.9 | 8 / 8 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | f32 | 18.4457 [18.3771-18.8964] | 17.9151 [17.7995-18.2365] | 97.1 | 2 / 2 | faster |
| ple_kv | Q4 syn | dense-decode | 1 | bf16 | 0.1986 [0.1971-0.2006] | 0.1966 [0.1937-0.1988] | 99.0 | 93 / 94 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | bf16 | 0.2008 [0.1968-0.2028] | 0.1999 [0.1962-0.2016] | 99.5 | 92 / 92 | noise |
| ple_kv | Q4 syn | dense-decode | 3 | bf16 | 0.2035 [0.2021-0.2077] | 0.2006 [0.1975-0.2031] | 98.6 | 91 / 92 | noise |
| ple_kv | Q4 syn | dense-decode | 4 | bf16 | 0.2080 [0.2049-0.2103] | 0.2044 [0.1996-0.2062] | 98.3 | 89 / 90 | noise |
| ple_kv | Q4 syn | dense-decode | 5 | bf16 | 0.2143 [0.2131-0.2196] | 0.2085 [0.2046-0.2103] | 97.3 | 86 / 88 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | bf16 | 0.2192 [0.2176-0.2210] | 0.2072 [0.2046-0.2088] | 94.5 | 84 / 89 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | bf16 | 0.2233 [0.2210-0.2262] | 0.2113 [0.2086-0.2136] | 94.6 | 83 / 87 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | bf16 | 0.2294 [0.2280-0.2307] | 0.2133 [0.2125-0.2155] | 93.0 | 80 / 86 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.9605 [0.9590-0.9678] | 0.9051 [0.8966-0.9117] | 94.2 | 19 / 20 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 4.1733 [3.8923-4.5044] | 4.0887 [3.8403-4.4005] | 98.0 | 4 / 5 | noise |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 17.9853 [17.7797-18.3859] | 17.8227 [17.5313-18.2451] | 99.1 | 1 / 1 | noise |
| ple_kv | Q4 syn | dense-decode | 1 | f32 | 0.1997 [0.1968-0.2007] | 0.1989 [0.1968-0.2006] | 99.6 | 92 / 93 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | f32 | 0.2015 [0.2006-0.2027] | 0.1990 [0.1971-0.2013] | 98.8 | 91 / 93 | noise |
| ple_kv | Q4 syn | dense-decode | 3 | f32 | 0.2038 [0.2021-0.2050] | 0.2010 [0.1992-0.2026] | 98.6 | 90 / 92 | noise |
| ple_kv | Q4 syn | dense-decode | 4 | f32 | 0.2065 [0.2050-0.2081] | 0.2048 [0.2032-0.2070] | 99.2 | 89 / 90 | noise |
| ple_kv | Q4 syn | dense-decode | 5 | f32 | 0.2159 [0.2149-0.2165] | 0.2055 [0.2028-0.2079] | 95.2 | 85 / 90 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | f32 | 0.2183 [0.2172-0.2206] | 0.2098 [0.2091-0.2114] | 96.1 | 84 / 88 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | f32 | 0.2236 [0.2220-0.2259] | 0.2101 [0.2092-0.2126] | 93.9 | 82 / 88 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | f32 | 0.2288 [0.2258-0.2308] | 0.2124 [0.2112-0.2145] | 92.8 | 81 / 87 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.9553 [0.9488-0.9632] | 0.9070 [0.9013-0.9142] | 94.9 | 19 / 20 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 4.2136 [4.0170-4.5282] | 4.1235 [3.9327-4.4582] | 97.9 | 4 / 4 | noise |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 17.8042 [17.6381-18.1977] | 17.5664 [17.3536-17.9858] | 98.7 | 1 / 1 | noise |
| lm_head | Q8 | dense-decode | 1 | bf16 | 3.2358 [3.2223-3.2675] | 3.2696 [3.2260-3.3309] | 101.0 | 209 / 207 | noise |
| lm_head | Q8 | dense-decode | 2 | bf16 | 3.2007 [3.1942-3.2354] | 3.2354 [3.2180-3.2683] | 101.1 | 211 / 209 | noise |
| lm_head | Q8 | dense-decode | 3 | bf16 | 3.4252 [3.4206-3.4494] | 3.2362 [3.2293-3.2528] | 94.5 | 197 / 209 | faster |
| lm_head | Q8 | dense-decode | 4 | bf16 | 3.5596 [3.5546-3.5844] | 3.2864 [3.2657-3.3114] | 92.3 | 190 / 206 | faster |
| lm_head | Q8 | dense-decode | 5 | bf16 | 3.4803 [3.4644-3.5224] | 3.3506 [3.3331-3.3753] | 96.3 | 194 / 202 | faster |
| lm_head | Q8 | dense-decode | 6 | bf16 | 3.5532 [3.5398-3.5641] | 3.4853 [3.4728-3.4998] | 98.1 | 190 / 194 | faster |
| lm_head | Q8 | dense-decode | 7 | bf16 | 3.6718 [3.6575-3.6857] | 3.6515 [3.6369-3.6712] | 99.5 | 184 / 185 | noise |
| lm_head | Q8 | dense-decode | 8 | bf16 | 3.7914 [3.7774-3.8167] | 3.8400 [3.8262-3.8465] | 101.3 | 178 / 176 | slower |
| lm_head | Q8 | dense-prefill-wmma | 512 | bf16 | 23.1156 [22.9859-23.4497] | 22.5387 [22.4464-22.9325] | 97.5 | 29 / 30 | faster |
| lm_head | Q8 | dense-prefill-wmma | 2048 | bf16 | 92.3627 [91.6466-92.8282] | 90.3339 [88.1692-90.8216] | 97.8 | 7 / 7 | faster |
| lm_head | Q8 | dense-decode | 1 | f32 | 3.2309 [3.2138-3.2497] | 3.2405 [3.2208-3.2537] | 100.3 | 209 / 208 | noise |
| lm_head | Q8 | dense-decode | 2 | f32 | 3.2106 [3.2049-3.2216] | 3.2276 [3.2217-3.2460] | 100.5 | 210 / 209 | slower |
| lm_head | Q8 | dense-decode | 3 | f32 | 3.4530 [3.4423-3.4800] | 3.2589 [3.2507-3.2744] | 94.4 | 196 / 207 | faster |
| lm_head | Q8 | dense-decode | 4 | f32 | 3.6017 [3.5886-3.6103] | 3.3085 [3.2861-3.3146] | 91.9 | 188 / 204 | faster |
| lm_head | Q8 | dense-decode | 5 | f32 | 3.5110 [3.5009-3.5233] | 3.4242 [3.4062-3.4283] | 97.5 | 192 / 197 | faster |
| lm_head | Q8 | dense-decode | 6 | f32 | 3.5932 [3.5640-3.6405] | 3.5118 [3.4965-3.5606] | 97.7 | 188 / 192 | faster |
| lm_head | Q8 | dense-decode | 7 | f32 | 3.6853 [3.6616-3.7191] | 3.6461 [3.6355-3.6638] | 98.9 | 183 / 185 | noise |
| lm_head | Q8 | dense-decode | 8 | f32 | 3.8346 [3.8173-3.8788] | 3.9179 [3.8902-3.9629] | 102.2 | 176 / 172 | slower |
| lm_head | Q8 | dense-prefill-wmma | 512 | f32 | 22.7229 [22.5430-22.8205] | 22.0296 [21.7389-22.2339] | 97.0 | 30 / 31 | faster |
| lm_head | Q4 syn | dense-decode | 1 | bf16 | 1.8671 [1.8422-1.8893] | 1.8380 [1.8204-1.8722] | 98.4 | 192 / 195 | noise |
| lm_head | Q4 syn | dense-decode | 2 | bf16 | 1.9055 [1.8847-1.9157] | 1.8621 [1.8526-1.8704] | 97.7 | 188 / 192 | faster |
| lm_head | Q4 syn | dense-decode | 3 | bf16 | 2.0428 [2.0397-2.0538] | 1.9145 [1.9039-1.9198] | 93.7 | 175 / 187 | faster |
| lm_head | Q4 syn | dense-decode | 4 | bf16 | 2.1261 [2.1187-2.1392] | 1.9900 [1.9832-1.9955] | 93.6 | 168 / 180 | faster |
| lm_head | Q4 syn | dense-decode | 5 | bf16 | 2.4354 [2.4213-2.4438] | 2.2847 [2.2726-2.2984] | 93.8 | 147 / 157 | faster |
| lm_head | Q4 syn | dense-decode | 6 | bf16 | 2.5512 [2.5453-2.5630] | 2.4163 [2.4085-2.4321] | 94.7 | 140 / 148 | faster |
| lm_head | Q4 syn | dense-decode | 7 | bf16 | 2.7269 [2.7163-2.7326] | 2.5663 [2.5591-2.5766] | 94.1 | 131 / 139 | faster |
| lm_head | Q4 syn | dense-decode | 8 | bf16 | 2.8958 [2.8897-2.9072] | 2.6952 [2.6852-2.6980] | 93.1 | 123 / 133 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | bf16 | 22.0772 [21.9472-22.3730] | 21.7205 [21.5240-21.9056] | 98.4 | 16 / 16 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 91.9719 [91.1948-92.0408] | 89.7666 [88.3475-90.1314] | 97.6 | 4 / 4 | faster |
| lm_head | Q4 syn | dense-decode | 1 | f32 | 1.8122 [1.8051-1.8231] | 1.8077 [1.7974-1.8153] | 99.8 | 197 / 198 | noise |
| lm_head | Q4 syn | dense-decode | 2 | f32 | 1.8668 [1.8625-1.8725] | 1.8346 [1.8279-1.8372] | 98.3 | 192 / 195 | faster |
| lm_head | Q4 syn | dense-decode | 3 | f32 | 2.0671 [2.0582-2.0816] | 1.9409 [1.9294-1.9545] | 93.9 | 173 / 184 | faster |
| lm_head | Q4 syn | dense-decode | 4 | f32 | 2.2103 [2.1895-2.2192] | 2.0446 [2.0380-2.0555] | 92.5 | 162 / 175 | faster |
| lm_head | Q4 syn | dense-decode | 5 | f32 | 2.4449 [2.4376-2.4556] | 2.3114 [2.3053-2.3171] | 94.5 | 146 / 155 | faster |
| lm_head | Q4 syn | dense-decode | 6 | f32 | 2.5994 [2.5754-2.6144] | 2.4789 [2.4456-2.5015] | 95.4 | 138 / 144 | faster |
| lm_head | Q4 syn | dense-decode | 7 | f32 | 2.7274 [2.7095-2.7464] | 2.6106 [2.6073-2.6332] | 95.7 | 131 / 137 | faster |
| lm_head | Q4 syn | dense-decode | 8 | f32 | 2.8819 [2.8670-2.8838] | 2.7151 [2.6979-2.7243] | 94.2 | 124 / 132 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | f32 | 21.8716 [21.7055-22.4601] | 21.4789 [21.3183-21.7902] | 98.2 | 16 / 17 | noise |
| mtp_fc | Q8 | dense-decode | 1 | bf16 | 0.0818 [0.0811-0.0822] | 0.0826 [0.0818-0.0829] | 101.0 | 85 / 84 | noise |
| mtp_fc | Q8 | dense-decode | 2 | bf16 | 0.0842 [0.0836-0.0845] | 0.0839 [0.0834-0.0845] | 99.7 | 83 / 83 | noise |
| mtp_fc | Q8 | dense-decode | 3 | bf16 | 0.0876 [0.0868-0.0886] | 0.0862 [0.0853-0.0867] | 98.4 | 79 / 81 | faster |
| mtp_fc | Q8 | dense-decode | 4 | bf16 | 0.0902 [0.0894-0.0908] | 0.0891 [0.0886-0.0898] | 98.7 | 77 / 78 | noise |
| mtp_fc | Q8 | dense-decode | 5 | bf16 | 0.0935 [0.0930-0.0939] | 0.0930 [0.0921-0.0938] | 99.5 | 74 / 75 | noise |
| mtp_fc | Q8 | dense-decode | 6 | bf16 | 0.0967 [0.0954-0.0970] | 0.0964 [0.0954-0.0971] | 99.7 | 72 / 72 | noise |
| mtp_fc | Q8 | dense-decode | 7 | bf16 | 0.0993 [0.0982-0.0999] | 0.0977 [0.0968-0.0988] | 98.4 | 70 / 71 | noise |
| mtp_fc | Q8 | dense-decode | 8 | bf16 | 0.1011 [0.1000-0.1023] | 0.1002 [0.0996-0.1015] | 99.2 | 69 / 69 | noise |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | bf16 | 0.2927 [0.2903-0.2961] | 0.2280 [0.2272-0.2291] | 77.9 | 24 / 31 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.8481 [0.8335-0.8641] | 0.7852 [0.7767-0.7949] | 92.6 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | bf16 | 3.0911 [3.0744-3.1854] | 3.0145 [2.9982-3.0715] | 97.5 | 2 / 2 | faster |
| mtp_fc | Q8 | dense-decode | 1 | f32 | 0.0810 [0.0803-0.0815] | 0.0822 [0.0820-0.0827] | 101.6 | 86 / 85 | slower |
| mtp_fc | Q8 | dense-decode | 2 | f32 | 0.0830 [0.0826-0.0837] | 0.0834 [0.0832-0.0837] | 100.5 | 84 / 83 | noise |
| mtp_fc | Q8 | dense-decode | 3 | f32 | 0.0863 [0.0859-0.0866] | 0.0850 [0.0844-0.0862] | 98.5 | 81 / 82 | noise |
| mtp_fc | Q8 | dense-decode | 4 | f32 | 0.0884 [0.0880-0.0896] | 0.0885 [0.0881-0.0896] | 100.1 | 79 / 79 | noise |
| mtp_fc | Q8 | dense-decode | 5 | f32 | 0.0928 [0.0921-0.0938] | 0.0928 [0.0925-0.0936] | 100.0 | 75 / 75 | noise |
| mtp_fc | Q8 | dense-decode | 6 | f32 | 0.0955 [0.0949-0.0957] | 0.0950 [0.0944-0.0964] | 99.5 | 73 / 73 | noise |
| mtp_fc | Q8 | dense-decode | 7 | f32 | 0.0980 [0.0970-0.0983] | 0.0972 [0.0966-0.0979] | 99.1 | 71 / 72 | noise |
| mtp_fc | Q8 | dense-decode | 8 | f32 | 0.0992 [0.0985-0.0997] | 0.0993 [0.0989-0.1000] | 100.2 | 70 / 70 | noise |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | f32 | 0.2934 [0.2917-0.2970] | 0.2310 [0.2295-0.2321] | 78.8 | 24 / 30 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | f32 | 0.8375 [0.8193-0.8455] | 0.7774 [0.7730-0.7882] | 92.8 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | f32 | 3.1433 [3.0608-3.3210] | 3.0521 [2.9745-3.1818] | 97.1 | 2 / 2 | noise |
| mtp_fc | Q4 syn | dense-decode | 1 | bf16 | 0.0453 [0.0448-0.0455] | 0.0440 [0.0436-0.0444] | 97.2 | 81 / 84 | faster |
| mtp_fc | Q4 syn | dense-decode | 2 | bf16 | 0.0487 [0.0478-0.0496] | 0.0459 [0.0454-0.0462] | 94.2 | 76 / 80 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | bf16 | 0.0512 [0.0504-0.0526] | 0.0474 [0.0471-0.0477] | 92.4 | 72 / 78 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | bf16 | 0.0533 [0.0522-0.0543] | 0.0490 [0.0484-0.0496] | 91.9 | 69 / 75 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | bf16 | 0.0571 [0.0563-0.0582] | 0.0539 [0.0534-0.0544] | 94.4 | 65 / 68 | faster |
| mtp_fc | Q4 syn | dense-decode | 6 | bf16 | 0.0620 [0.0609-0.0628] | 0.0565 [0.0560-0.0575] | 91.2 | 59 / 65 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | bf16 | 0.0601 [0.0594-0.0611] | 0.0563 [0.0554-0.0568] | 93.7 | 61 / 66 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | bf16 | 0.0648 [0.0636-0.0656] | 0.0592 [0.0586-0.0601] | 91.3 | 57 / 62 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.2330 [0.2311-0.2376] | 0.2162 [0.2158-0.2166] | 92.8 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.7925 [0.7792-0.8070] | 0.7692 [0.7622-0.7759] | 97.1 | 5 / 5 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 3.0690 [3.0524-3.1167] | 3.0070 [2.9913-3.0662] | 98.0 | 1 / 1 | noise |
| mtp_fc | Q4 syn | dense-decode | 1 | f32 | 0.0455 [0.0449-0.0460] | 0.0446 [0.0442-0.0448] | 98.1 | 81 / 83 | faster |
| mtp_fc | Q4 syn | dense-decode | 2 | f32 | 0.0485 [0.0476-0.0497] | 0.0462 [0.0459-0.0466] | 95.3 | 76 / 80 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | f32 | 0.0509 [0.0497-0.0523] | 0.0477 [0.0474-0.0482] | 93.8 | 72 / 77 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | f32 | 0.0534 [0.0519-0.0546] | 0.0491 [0.0487-0.0497] | 92.0 | 69 / 75 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | f32 | 0.0545 [0.0535-0.0564] | 0.0530 [0.0525-0.0535] | 97.2 | 68 / 70 | noise |
| mtp_fc | Q4 syn | dense-decode | 6 | f32 | 0.0572 [0.0562-0.0586] | 0.0541 [0.0536-0.0549] | 94.5 | 64 / 68 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | f32 | 0.0590 [0.0580-0.0598] | 0.0557 [0.0554-0.0568] | 94.4 | 63 / 66 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | f32 | 0.0611 [0.0604-0.0618] | 0.0571 [0.0566-0.0577] | 93.5 | 60 / 65 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.2348 [0.2330-0.2380] | 0.2174 [0.2168-0.2183] | 92.6 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.7777 [0.7763-0.7954] | 0.7718 [0.7668-0.7763] | 99.2 | 5 / 5 | noise |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | f32 | 3.1757 [3.0306-3.4030] | 3.1557 [2.9710-3.2687] | 99.4 | 1 / 1 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 1 | bf16 | 0.1918 [0.1908-0.1931] | 0.1899 [0.1891-0.1919] | 99.0 | 91 / 92 | noise |
| exp_gu | Q4 | expert-decode-gather | 1 | bf16 | 0.1890 [0.1876-0.1917] | 0.1886 [0.1875-0.1895] | 99.8 | 92 / 92 | noise |
| exp_down | Q4 | expert-decode-combine | 1 | bf16 | 0.1247 [0.1243-0.1254] | 0.1193 [0.1189-0.1202] | 95.7 | 82 / 86 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 2 | bf16 | 0.3362 [0.3346-0.3372] | 0.3336 [0.3316-0.3354] | 99.2 | 98 / 99 | noise |
| exp_gu | Q4 | expert-decode-gather | 2 | bf16 | 0.3392 [0.3366-0.3430] | 0.3358 [0.3310-0.3390] | 99.0 | 98 / 98 | noise |
| exp_down | Q4 | expert-decode-combine | 2 | bf16 | 0.2263 [0.2245-0.2272] | 0.2178 [0.2158-0.2188] | 96.2 | 86 / 89 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 3 | bf16 | 0.4168 [0.4151-0.4208] | 0.4112 [0.4091-0.4147] | 98.7 | 109 / 110 | faster |
| exp_gu | Q4 | expert-decode-gather | 3 | bf16 | 0.4222 [0.4186-0.4248] | 0.4114 [0.4088-0.4153] | 97.5 | 107 / 110 | faster |
| exp_down | Q4 | expert-decode-combine | 3 | bf16 | 0.3027 [0.3012-0.3044] | 0.2928 [0.2910-0.2948] | 96.7 | 88 / 91 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 4 | bf16 | 0.4848 [0.4802-0.4888] | 0.4752 [0.4707-0.4830] | 98.0 | 122 / 125 | noise |
| exp_gu | Q4 | expert-decode-gather | 4 | bf16 | 0.4873 [0.4852-0.4908] | 0.4818 [0.4768-0.4855] | 98.9 | 121 / 123 | noise |
| exp_down | Q4 | expert-decode-combine | 4 | bf16 | 0.3684 [0.3667-0.3701] | 0.3601 [0.3577-0.3618] | 97.7 | 94 / 97 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 5 | bf16 | 0.5185 [0.5159-0.5202] | 0.5065 [0.5032-0.5079] | 97.7 | 118 / 120 | faster |
| exp_gu | Q4 | expert-decode-gather | 5 | bf16 | 0.5326 [0.5283-0.5379] | 0.5214 [0.5157-0.5266] | 97.9 | 114 / 117 | faster |
| exp_down | Q4 | expert-decode-combine | 5 | bf16 | 0.3713 [0.3666-0.3746] | 0.3674 [0.3625-0.3708] | 98.9 | 97 / 98 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 6 | bf16 | 0.5924 [0.5892-0.5955] | 0.5834 [0.5775-0.5851] | 98.5 | 126 / 128 | faster |
| exp_gu | Q4 | expert-decode-gather | 6 | bf16 | 0.6090 [0.6045-0.6115] | 0.5960 [0.5885-0.6036] | 97.9 | 123 / 126 | faster |
| exp_down | Q4 | expert-decode-combine | 6 | bf16 | 0.4285 [0.4249-0.4296] | 0.4148 [0.4129-0.4168] | 96.8 | 103 / 106 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 7 | bf16 | 0.6486 [0.6452-0.6517] | 0.6273 [0.6260-0.6309] | 96.7 | 126 / 130 | faster |
| exp_gu | Q4 | expert-decode-gather | 7 | bf16 | 0.6641 [0.6600-0.6676] | 0.6448 [0.6394-0.6497] | 97.1 | 123 / 127 | faster |
| exp_down | Q4 | expert-decode-combine | 7 | bf16 | 0.4632 [0.4588-0.4662] | 0.4428 [0.4396-0.4446] | 95.6 | 104 / 109 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 8 | bf16 | 0.7364 [0.7328-0.7435] | 0.7242 [0.7214-0.7299] | 98.3 | 123 / 125 | faster |
| exp_gu | Q4 | expert-decode-gather | 8 | bf16 | 0.7535 [0.7463-0.7599] | 0.7449 [0.7337-0.7480] | 98.9 | 120 / 122 | noise |
| exp_down | Q4 | expert-decode-combine | 8 | bf16 | 0.4998 [0.4978-0.5012] | 0.4789 [0.4764-0.4814] | 95.8 | 107 / 111 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 512 | bf16 | 4.5655 [4.5022-4.6635] | 4.3020 [4.2647-4.3411] | 94.2 | 174 / 185 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 512 | bf16 | 4.6410 [4.6224-4.6604] | 4.3254 [4.3212-4.3328] | 93.2 | 171 / 184 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 512 | bf16 | 4.4261 [4.4158-4.4393] | 4.3803 [4.3732-4.3980] | 99.0 | 180 / 182 | faster |
| exp_down | Q4 | expert-prefill-combine narrow | 512 | bf16 | 3.1192 [3.1036-3.1212] | 2.8428 [2.8350-2.8543] | 91.1 | 150 / 165 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 512 | bf16 | 3.0782 [3.0752-3.0875] | 2.8229 [2.8146-2.8256] | 91.7 | 152 / 166 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 512 | bf16 | 3.0993 [3.0779-3.1056] | 2.8374 [2.8213-2.8427] | 91.5 | 151 / 165 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 512 | bf16 | 5.9081 [5.8727-5.9624] | 5.8738 [5.8412-5.9433] | 99.4 | 135 / 135 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 512 | bf16 | 5.8994 [5.8896-5.9119] | 5.8634 [5.8586-5.8735] | 99.4 | 135 / 136 | faster |
| exp_gu | Q4 | expert-prefill-gather wide | 512 | bf16 | 5.9162 [5.8657-5.9806] | 5.8161 [5.7670-5.8823] | 98.3 | 134 / 137 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 512 | bf16 | 3.6172 [3.6075-3.6236] | 3.2652 [3.2463-3.2750] | 90.3 | 129 / 143 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 512 | bf16 | 3.4912 [3.4830-3.4997] | 3.1388 [3.1333-3.1415] | 89.9 | 134 / 149 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 512 | bf16 | 3.5146 [3.5047-3.5294] | 3.1570 [3.1505-3.1600] | 89.8 | 133 / 148 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 512 | bf16 | 5.7224 [5.6887-5.7332] | 5.3415 [5.3001-5.3986] | 93.3 | 139 / 149 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 512 | bf16 | 5.6780 [5.6642-5.6898] | 5.3129 [5.2830-5.3385] | 93.6 | 140 / 150 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 512 | bf16 | 5.7727 [5.7347-5.8023] | 5.4116 [5.3741-5.4422] | 93.7 | 138 / 147 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 512 | bf16 | 4.3418 [4.3156-4.3780] | 3.0450 [3.0384-3.0556] | 70.1 | 108 / 154 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 512 | bf16 | 4.4036 [4.3824-4.4311] | 3.0093 [2.9917-3.0232] | 68.3 | 106 / 156 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 512 | bf16 | 4.4476 [4.3934-4.4738] | 3.0172 [3.0030-3.0286] | 67.8 | 105 / 155 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 2048 | bf16 | 8.3606 [7.9001-8.3934] | 7.8295 [7.2920-7.8849] | 93.7 | 106 / 114 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 2048 | bf16 | 8.2356 [8.2104-8.2735] | 7.7286 [7.6905-7.7683] | 93.8 | 108 / 115 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 2048 | bf16 | 8.0315 [7.9990-8.0708] | 8.0518 [7.9734-8.0965] | 100.2 | 111 / 110 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 2048 | bf16 | 5.4934 [5.4733-5.5061] | 5.1051 [5.0824-5.1177] | 92.9 | 95 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 2048 | bf16 | 5.3706 [5.3518-5.3843] | 5.0073 [4.9833-5.0213] | 93.2 | 97 / 104 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 2048 | bf16 | 5.5419 [5.5220-5.5670] | 5.1795 [5.1348-5.2022] | 93.5 | 94 / 101 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 2048 | bf16 | 9.2557 [9.1671-9.3555] | 9.2128 [9.1560-9.3064] | 99.5 | 96 / 97 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 2048 | bf16 | 9.2642 [9.1804-9.3615] | 9.2210 [9.1617-9.3485] | 99.5 | 96 / 96 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 2048 | bf16 | 9.2093 [9.1096-9.3456] | 9.0548 [9.0160-9.2212] | 98.3 | 97 / 98 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 2048 | bf16 | 5.8037 [5.7805-5.8189] | 5.3331 [5.3183-5.3635] | 91.9 | 90 / 98 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 2048 | bf16 | 5.6702 [5.6345-5.6959] | 5.1856 [5.1598-5.2288] | 91.5 | 92 / 101 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 2048 | bf16 | 5.7595 [5.7295-5.7865] | 5.2823 [5.2609-5.2947] | 91.7 | 91 / 99 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 2048 | bf16 | 8.7241 [8.6485-8.7822] | 8.4327 [8.3382-8.5018] | 96.7 | 102 / 105 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 2048 | bf16 | 8.8096 [8.7053-8.8649] | 8.5213 [8.4403-8.6064] | 96.7 | 101 / 104 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 2048 | bf16 | 8.9098 [8.8518-9.0655] | 8.5238 [8.4509-8.6388] | 95.7 | 100 / 104 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 2048 | bf16 | 6.1780 [6.1451-6.2007] | 5.1546 [5.1319-5.1988] | 83.4 | 85 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 2048 | bf16 | 6.1744 [6.1228-6.2073] | 5.1657 [5.1245-5.1985] | 83.7 | 85 / 101 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 2048 | bf16 | 6.2584 [6.2320-6.3157] | 5.2503 [5.2322-5.2611] | 83.9 | 84 / 100 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 8192 | bf16 | 24.1124 [21.1492-80.8226] | 23.8433 [20.2775-77.0436] | 98.9 | 37 / 37 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 8192 | bf16 | 81.4840 [81.3278-81.5493] | 78.1524 [78.0052-78.2646] | 95.9 | 11 / 11 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 8192 | bf16 | 22.8526 [22.4609-23.7229] | 23.0608 [22.1368-23.6871] | 100.9 | 39 / 39 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 8192 | bf16 | 15.6198 [15.2832-15.8911] | 15.0272 [14.6895-15.2733] | 96.2 | 34 / 35 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 8192 | bf16 | 15.0077 [14.7108-15.1645] | 14.3470 [14.0786-14.5810] | 95.6 | 35 / 37 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 8192 | bf16 | 14.3161 [13.9028-14.7871] | 14.0473 [13.3514-14.2748] | 98.1 | 37 / 37 | noise |
| exp_gu | Q4 | expert-prefill-swiglu wide | 8192 | bf16 | 21.9081 [21.5895-22.1058] | 21.9125 [21.6096-22.2130] | 100.0 | 41 / 41 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 8192 | bf16 | 22.2363 [22.1788-22.3852] | 22.2943 [22.2235-22.4058] | 100.3 | 40 / 40 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 8192 | bf16 | 21.7213 [21.6193-22.0307] | 21.5657 [21.4626-22.0539] | 99.3 | 41 / 41 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 8192 | bf16 | 14.3757 [14.3458-14.5134] | 13.7059 [13.6496-13.8455] | 95.3 | 36 / 38 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 8192 | bf16 | 14.1993 [14.0170-14.3026] | 13.5763 [13.2524-13.6120] | 95.6 | 37 / 39 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 8192 | bf16 | 14.3465 [14.1357-14.4787] | 13.6280 [13.4245-13.7547] | 95.0 | 37 / 38 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 8192 | bf16 | 21.3509 [21.0476-21.9275] | 20.9787 [20.5688-21.3668] | 98.3 | 42 / 42 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 8192 | bf16 | 20.8954 [20.5134-21.2633] | 20.5296 [20.2533-20.7020] | 98.2 | 43 / 43 | noise |
| exp_gu | Q4 | expert-prefill-gather wide256 | 8192 | bf16 | 21.3185 [20.9055-21.9389] | 20.3010 [19.7960-20.7831] | 95.2 | 42 / 44 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 8192 | bf16 | 15.2736 [15.1875-15.5279] | 13.7540 [13.5949-13.9184] | 90.0 | 34 / 38 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 8192 | bf16 | 15.0751 [14.9143-15.4491] | 13.6403 [13.4867-13.7997] | 90.5 | 35 / 38 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 8192 | bf16 | 15.4786 [15.3331-15.6341] | 13.9726 [13.9047-14.0848] | 90.3 | 34 / 38 | faster |
