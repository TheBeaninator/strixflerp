# strixw panel kernels: kernel gate results

Criteria: `gate/strixw-panel-kernels.toml` (committed before the run). Raw: `raw.jsonl` (one case a line), bench log `bench.log`. 489 cases, 6 skipped (output too large), 2.232 GB read from the fleet file.

## Verdicts

| criterion | pass | detail |
|---|---|---|
| equal | PASS | 489 of 489 cases byte-equal |
| bad_id | PASS | 21 of 21 out-of-range-id cases byte-equal |
| coverage | PASS | missing: none |
| read | FAIL | 2.232 GB read |
| speed_decode_dense | PASS | median panel % of origin time per class: attn_o 98.3, attn_qkv 99.0, gdn_in 97.4, gdn_out 98.6, lm_head 96.8, mtp_fc 98.0, ple_kv 99.4, sh_down 84.1, sh_gu 96.4 |
| speed_prefill_dense | PASS | sh_gu M2048 bf16 92.6%; sh_gu M8192 bf16 96.3%; sh_gu M2048 f32 92.2%; sh_gu M8192 f32 96.0%; sh_down M2048 bf16 91.1%; sh_down M8192 bf16 95.8%; sh_down M2048 f32 93.9%; sh_down M8192 f32 95.7%; gdn_in M2048 bf16 98.3%; gdn_in M8192 bf16 99.3%; gdn_in M2048 f32 99.3%; gdn_in M8192 f32 99.6%; gdn_out M2048 bf16 71.3%; gdn_out M8192 bf16 86.3%; gdn_out M2048 f32 65.9%; gdn_out M8192 f32 90.5%; attn_qkv M2048 bf16 95.4%; attn_qkv M8192 bf16 97.8%; attn_qkv M2048 f32 92.7%; attn_qkv M8192 f32 97.3%; attn_o M2048 bf16 70.3%; attn_o M8192 bf16 87.2%; attn_o M2048 f32 69.1%; attn_o M8192 f32 92.0%; ple_kv M2048 bf16 93.3%; ple_kv M8192 bf16 98.2%; ple_kv M2048 f32 95.6%; ple_kv M8192 f32 98.8%; lm_head M2048 bf16 97.4%; mtp_fc M2048 bf16 91.8%; mtp_fc M8192 bf16 96.5%; mtp_fc M2048 f32 93.5%; mtp_fc M8192 f32 97.8% |
| speed_prefill_expert | PASS | expert-prefill-swiglu-f16h narrow M2048 93.8% (faster); expert-prefill-combine-f16h-shared narrow M2048 93.5% (faster); expert-prefill-combine-f16h-shared wide M8192 94.8% (faster); expert-prefill-swiglu-f16h wide256 M8192 97.8% (noise) |

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
| sh_gu | Q8 | dense-decode | 1 | bf16 | 0.1751 [0.0428-0.1766] | 0.1753 [0.0434-0.1768] | 100.1 | 20 / 20 | noise |
| sh_gu | Q8 | dense-decode | 2 | bf16 | 0.1821 [0.1806-0.1831] | 0.1809 [0.1795-0.1814] | 99.3 | 19 / 19 | noise |
| sh_gu | Q8 | dense-decode | 3 | bf16 | 0.1865 [0.1850-0.1879] | 0.1838 [0.1827-0.1855] | 98.5 | 19 / 19 | noise |
| sh_gu | Q8 | dense-decode | 4 | bf16 | 0.1930 [0.1914-0.1952] | 0.1893 [0.1887-0.1911] | 98.1 | 18 / 18 | faster |
| sh_gu | Q8 | dense-decode | 5 | bf16 | 0.2020 [0.2007-0.2074] | 0.1978 [0.1966-0.1989] | 97.9 | 17 / 18 | faster |
| sh_gu | Q8 | dense-decode | 6 | bf16 | 0.2093 [0.2080-0.2122] | 0.2024 [0.2016-0.2037] | 96.7 | 17 / 17 | faster |
| sh_gu | Q8 | dense-decode | 7 | bf16 | 0.2143 [0.2138-0.2158] | 0.2070 [0.2056-0.2087] | 96.6 | 16 / 17 | faster |
| sh_gu | Q8 | dense-decode | 8 | bf16 | 0.2182 [0.2172-0.2195] | 0.2127 [0.2114-0.2134] | 97.5 | 16 / 16 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 512 | bf16 | 0.6282 [0.6086-0.6462] | 0.5576 [0.5538-0.5633] | 88.8 | 6 / 6 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | bf16 | 2.0165 [1.9832-2.0798] | 1.8665 [1.8578-1.9175] | 92.6 | 2 / 2 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | bf16 | 6.7243 [6.6394-6.8252] | 6.4788 [6.4244-6.5027] | 96.3 | 1 / 1 | faster |
| sh_gu | Q8 | dense-decode | 1 | f32 | 0.1744 [0.1732-0.1758] | 0.1752 [0.1741-0.1761] | 100.5 | 20 / 20 | noise |
| sh_gu | Q8 | dense-decode | 2 | f32 | 0.1818 [0.1787-0.1826] | 0.1809 [0.1797-0.1824] | 99.5 | 19 / 19 | noise |
| sh_gu | Q8 | dense-decode | 3 | f32 | 0.1870 [0.1842-0.1895] | 0.1839 [0.1831-0.1851] | 98.3 | 19 / 19 | noise |
| sh_gu | Q8 | dense-decode | 4 | f32 | 0.1920 [0.1913-0.1940] | 0.1890 [0.1878-0.1910] | 98.4 | 18 / 18 | faster |
| sh_gu | Q8 | dense-decode | 5 | f32 | 0.2012 [0.2000-0.2032] | 0.1984 [0.1967-0.1991] | 98.6 | 17 / 18 | faster |
| sh_gu | Q8 | dense-decode | 6 | f32 | 0.2093 [0.2075-0.2103] | 0.2000 [0.1992-0.2019] | 95.6 | 17 / 17 | faster |
| sh_gu | Q8 | dense-decode | 7 | f32 | 0.2130 [0.2122-0.2138] | 0.2052 [0.2040-0.2064] | 96.3 | 16 / 17 | faster |
| sh_gu | Q8 | dense-decode | 8 | f32 | 0.2180 [0.2172-0.2196] | 0.2119 [0.2106-0.2136] | 97.2 | 16 / 16 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 512 | f32 | 0.1520 [0.1490-0.1591] | 0.1406 [0.1378-0.1429] | 92.5 | 23 / 25 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 2048 | f32 | 0.4681 [0.4590-0.4784] | 0.4314 [0.4192-0.4348] | 92.2 | 7 / 8 | faster |
| sh_gu | Q8 | dense-prefill-wmma | 8192 | f32 | 1.5864 [1.5670-1.5944] | 1.5224 [1.5105-1.5377] | 96.0 | 2 / 2 | faster |
| sh_gu | Q4 syn | dense-decode | 1 | bf16 | 0.0265 [0.0258-0.0268] | 0.0255 [0.0253-0.0258] | 96.4 | 70 / 72 | faster |
| sh_gu | Q4 syn | dense-decode | 2 | bf16 | 0.0287 [0.0280-0.0295] | 0.0268 [0.0267-0.0272] | 93.5 | 64 / 69 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | bf16 | 0.0306 [0.0302-0.0314] | 0.0278 [0.0274-0.0281] | 90.8 | 60 / 66 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | bf16 | 0.0324 [0.0314-0.0328] | 0.0287 [0.0284-0.0291] | 88.8 | 57 / 64 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | bf16 | 0.0334 [0.0332-0.0343] | 0.0303 [0.0301-0.0305] | 90.8 | 55 / 61 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | bf16 | 0.0348 [0.0339-0.0360] | 0.0308 [0.0307-0.0312] | 88.6 | 53 / 60 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | bf16 | 0.0356 [0.0350-0.0365] | 0.0326 [0.0322-0.0328] | 91.5 | 52 / 57 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | bf16 | 0.0368 [0.0358-0.0376] | 0.0328 [0.0325-0.0331] | 89.0 | 50 / 56 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.1287 [0.1282-0.1305] | 0.1234 [0.1228-0.1250] | 95.9 | 14 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.4282 [0.4195-0.4342] | 0.4190 [0.4082-0.4244] | 97.9 | 4 / 4 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 1.5695 [1.5442-1.6169] | 1.5443 [1.5196-1.5818] | 98.4 | 1 / 1 | noise |
| sh_gu | Q4 syn | dense-decode | 1 | f32 | 0.0261 [0.0256-0.0265] | 0.0254 [0.0251-0.0255] | 97.2 | 71 / 73 | faster |
| sh_gu | Q4 syn | dense-decode | 2 | f32 | 0.0285 [0.0278-0.0295] | 0.0265 [0.0264-0.0268] | 93.0 | 65 / 70 | faster |
| sh_gu | Q4 syn | dense-decode | 3 | f32 | 0.0306 [0.0292-0.0324] | 0.0276 [0.0274-0.0281] | 90.2 | 60 / 67 | faster |
| sh_gu | Q4 syn | dense-decode | 4 | f32 | 0.0309 [0.0300-0.0322] | 0.0287 [0.0284-0.0290] | 92.9 | 60 / 64 | faster |
| sh_gu | Q4 syn | dense-decode | 5 | f32 | 0.0327 [0.0320-0.0348] | 0.0302 [0.0300-0.0307] | 92.5 | 56 / 61 | faster |
| sh_gu | Q4 syn | dense-decode | 6 | f32 | 0.0346 [0.0340-0.0352] | 0.0318 [0.0310-0.0324] | 91.9 | 53 / 58 | faster |
| sh_gu | Q4 syn | dense-decode | 7 | f32 | 0.0344 [0.0336-0.0359] | 0.0319 [0.0318-0.0321] | 92.6 | 54 / 58 | faster |
| sh_gu | Q4 syn | dense-decode | 8 | f32 | 0.0367 [0.0360-0.0372] | 0.0339 [0.0333-0.0346] | 92.5 | 50 / 54 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.1282 [0.1272-0.1310] | 0.1249 [0.1239-0.1255] | 97.4 | 14 / 15 | faster |
| sh_gu | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.4252 [0.4215-0.4294] | 0.4175 [0.4143-0.4217] | 98.2 | 4 / 4 | noise |
| sh_gu | Q4 syn | dense-prefill-wmma | 8192 | f32 | 1.6321 [1.5745-1.7018] | 1.5999 [1.5410-1.6886] | 98.0 | 1 / 1 | noise |
| sh_down | Q8 | dense-decode | 1 | bf16 | 0.0257 [0.0254-0.0258] | 0.0255 [0.0252-0.0258] | 99.2 | 72 / 72 | noise |
| sh_down | Q8 | dense-decode | 2 | bf16 | 0.0281 [0.0276-0.0284] | 0.0267 [0.0262-0.0269] | 95.2 | 66 / 69 | faster |
| sh_down | Q8 | dense-decode | 3 | bf16 | 0.0307 [0.0303-0.0313] | 0.0274 [0.0271-0.0275] | 89.1 | 60 / 67 | faster |
| sh_down | Q8 | dense-decode | 4 | bf16 | 0.0315 [0.0314-0.0318] | 0.0284 [0.0280-0.0289] | 90.3 | 59 / 65 | faster |
| sh_down | Q8 | dense-decode | 5 | bf16 | 0.0354 [0.0346-0.0358] | 0.0295 [0.0292-0.0297] | 83.3 | 52 / 63 | faster |
| sh_down | Q8 | dense-decode | 6 | bf16 | 0.0358 [0.0351-0.0366] | 0.0299 [0.0296-0.0305] | 83.5 | 52 / 62 | faster |
| sh_down | Q8 | dense-decode | 7 | bf16 | 0.0379 [0.0374-0.0384] | 0.0311 [0.0309-0.0315] | 82.0 | 49 / 59 | faster |
| sh_down | Q8 | dense-decode | 8 | bf16 | 0.0406 [0.0401-0.0409] | 0.0325 [0.0322-0.0328] | 79.9 | 45 / 57 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | bf16 | 0.0766 [0.0763-0.0772] | 0.0700 [0.0696-0.0704] | 91.3 | 24 / 26 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.2480 [0.2393-0.2500] | 0.2259 [0.2222-0.2322] | 91.1 | 7 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | bf16 | 0.9111 [0.9048-0.9197] | 0.8730 [0.8667-0.8826] | 95.8 | 2 / 2 | faster |
| sh_down | Q8 | dense-decode | 1 | f32 | 0.0253 [0.0250-0.0258] | 0.0254 [0.0253-0.0256] | 100.3 | 73 / 73 | noise |
| sh_down | Q8 | dense-decode | 2 | f32 | 0.0273 [0.0268-0.0279] | 0.0262 [0.0260-0.0264] | 96.0 | 67 / 70 | faster |
| sh_down | Q8 | dense-decode | 3 | f32 | 0.0287 [0.0284-0.0290] | 0.0270 [0.0268-0.0274] | 94.1 | 64 / 68 | faster |
| sh_down | Q8 | dense-decode | 4 | f32 | 0.0299 [0.0296-0.0305] | 0.0278 [0.0278-0.0280] | 93.0 | 62 / 66 | faster |
| sh_down | Q8 | dense-decode | 5 | f32 | 0.0344 [0.0337-0.0349] | 0.0288 [0.0286-0.0294] | 83.6 | 54 / 64 | faster |
| sh_down | Q8 | dense-decode | 6 | f32 | 0.0362 [0.0358-0.0372] | 0.0304 [0.0302-0.0307] | 84.1 | 51 / 61 | faster |
| sh_down | Q8 | dense-decode | 7 | f32 | 0.0368 [0.0364-0.0374] | 0.0310 [0.0306-0.0314] | 84.1 | 50 / 60 | faster |
| sh_down | Q8 | dense-decode | 8 | f32 | 0.0391 [0.0385-0.0392] | 0.0323 [0.0321-0.0326] | 82.7 | 47 / 57 | faster |
| sh_down | Q8 | dense-prefill-wmma | 512 | f32 | 0.0853 [0.0842-0.0866] | 0.0770 [0.0767-0.0779] | 90.3 | 22 / 24 | faster |
| sh_down | Q8 | dense-prefill-wmma | 2048 | f32 | 0.2528 [0.2486-0.2553] | 0.2373 [0.2347-0.2428] | 93.9 | 7 / 8 | faster |
| sh_down | Q8 | dense-prefill-wmma | 8192 | f32 | 0.8946 [0.8874-0.9030] | 0.8557 [0.8530-0.8663] | 95.7 | 2 / 2 | faster |
| sh_down | Q4 syn | dense-decode | 1 | bf16 | 0.0170 [0.0169-0.0172] | 0.0171 [0.0168-0.0174] | 100.2 | 60 / 60 | noise |
| sh_down | Q4 syn | dense-decode | 2 | bf16 | 0.0205 [0.0203-0.0210] | 0.0180 [0.0179-0.0182] | 87.7 | 50 / 57 | faster |
| sh_down | Q4 syn | dense-decode | 3 | bf16 | 0.0228 [0.0222-0.0229] | 0.0185 [0.0184-0.0188] | 80.9 | 45 / 55 | faster |
| sh_down | Q4 syn | dense-decode | 4 | bf16 | 0.0244 [0.0242-0.0250] | 0.0198 [0.0197-0.0201] | 81.3 | 42 / 52 | faster |
| sh_down | Q4 syn | dense-decode | 5 | bf16 | 0.0274 [0.0273-0.0281] | 0.0213 [0.0211-0.0216] | 77.7 | 37 / 48 | faster |
| sh_down | Q4 syn | dense-decode | 6 | bf16 | 0.0289 [0.0285-0.0292] | 0.0222 [0.0216-0.0225] | 76.7 | 35 / 46 | faster |
| sh_down | Q4 syn | dense-decode | 7 | bf16 | 0.0297 [0.0291-0.0304] | 0.0228 [0.0223-0.0231] | 77.0 | 34 / 45 | faster |
| sh_down | Q4 syn | dense-decode | 8 | bf16 | 0.0314 [0.0308-0.0319] | 0.0234 [0.0232-0.0238] | 74.7 | 33 / 44 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.0764 [0.0762-0.0766] | 0.0680 [0.0677-0.0684] | 89.1 | 13 / 15 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.2433 [0.2334-0.2469] | 0.2236 [0.2212-0.2293] | 91.9 | 4 / 5 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 0.9016 [0.8920-0.9140] | 0.8710 [0.8636-0.8812] | 96.6 | 1 / 1 | faster |
| sh_down | Q4 syn | dense-decode | 1 | f32 | 0.0167 [0.0166-0.0169] | 0.0168 [0.0166-0.0170] | 100.7 | 61 / 61 | noise |
| sh_down | Q4 syn | dense-decode | 2 | f32 | 0.0189 [0.0187-0.0191] | 0.0177 [0.0175-0.0179] | 93.7 | 54 / 58 | faster |
| sh_down | Q4 syn | dense-decode | 3 | f32 | 0.0208 [0.0205-0.0211] | 0.0185 [0.0182-0.0186] | 88.9 | 49 / 55 | faster |
| sh_down | Q4 syn | dense-decode | 4 | f32 | 0.0223 [0.0221-0.0225] | 0.0191 [0.0189-0.0194] | 85.7 | 46 / 54 | faster |
| sh_down | Q4 syn | dense-decode | 5 | f32 | 0.0264 [0.0254-0.0267] | 0.0208 [0.0204-0.0210] | 78.6 | 39 / 49 | faster |
| sh_down | Q4 syn | dense-decode | 6 | f32 | 0.0284 [0.0281-0.0287] | 0.0222 [0.0219-0.0226] | 78.2 | 36 / 46 | faster |
| sh_down | Q4 syn | dense-decode | 7 | f32 | 0.0292 [0.0289-0.0295] | 0.0227 [0.0223-0.0230] | 77.6 | 35 / 45 | faster |
| sh_down | Q4 syn | dense-decode | 8 | f32 | 0.0311 [0.0301-0.0315] | 0.0243 [0.0238-0.0248] | 78.2 | 33 / 42 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.0833 [0.0824-0.0850] | 0.0770 [0.0764-0.0779] | 92.4 | 12 / 13 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.2580 [0.2550-0.2632] | 0.2368 [0.2346-0.2428] | 91.8 | 4 / 4 | faster |
| sh_down | Q4 syn | dense-prefill-wmma | 8192 | f32 | 0.9121 [0.9014-0.9342] | 0.8738 [0.8617-0.8933] | 95.8 | 1 / 1 | faster |
| gdn_in | Q4 | dense-decode | 1 | bf16 | 0.2404 [0.2386-0.2414] | 0.2373 [0.2358-0.2395] | 98.7 | 93 / 94 | noise |
| gdn_in | Q4 | dense-decode | 2 | bf16 | 0.2405 [0.2378-0.2426] | 0.2363 [0.2347-0.2376] | 98.3 | 93 / 95 | faster |
| gdn_in | Q4 | dense-decode | 3 | bf16 | 0.2451 [0.2431-0.2469] | 0.2387 [0.2369-0.2422] | 97.4 | 91 / 94 | faster |
| gdn_in | Q4 | dense-decode | 4 | bf16 | 0.2478 [0.2452-0.2513] | 0.2440 [0.2417-0.2454] | 98.5 | 90 / 92 | noise |
| gdn_in | Q4 | dense-decode | 5 | bf16 | 0.2600 [0.2562-0.2621] | 0.2487 [0.2475-0.2497] | 95.7 | 86 / 90 | faster |
| gdn_in | Q4 | dense-decode | 6 | bf16 | 0.2631 [0.2618-0.2648] | 0.2511 [0.2477-0.2535] | 95.4 | 85 / 89 | faster |
| gdn_in | Q4 | dense-decode | 7 | bf16 | 0.2683 [0.2660-0.2693] | 0.2569 [0.2542-0.2580] | 95.8 | 84 / 87 | faster |
| gdn_in | Q4 | dense-decode | 8 | bf16 | 0.2738 [0.2727-0.2773] | 0.2619 [0.2582-0.2652] | 95.7 | 82 / 86 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | bf16 | 1.3507 [1.3273-1.3633] | 1.3173 [1.3006-1.3295] | 97.5 | 17 / 17 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | bf16 | 5.6058 [5.5611-5.6520] | 5.5095 [5.4753-5.5800] | 98.3 | 4 / 4 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | bf16 | 22.1548 [21.1335-22.2850] | 22.0071 [20.1395-22.0587] | 99.3 | 1 / 1 | noise |
| gdn_in | Q4 | dense-decode | 1 | f32 | 0.2379 [0.2366-0.2394] | 0.2363 [0.2349-0.2374] | 99.3 | 94 / 95 | noise |
| gdn_in | Q4 | dense-decode | 2 | f32 | 0.2415 [0.2404-0.2443] | 0.2383 [0.2364-0.2392] | 98.7 | 93 / 94 | faster |
| gdn_in | Q4 | dense-decode | 3 | f32 | 0.2469 [0.2447-0.2487] | 0.2419 [0.2406-0.2446] | 98.0 | 91 / 93 | faster |
| gdn_in | Q4 | dense-decode | 4 | f32 | 0.2480 [0.2456-0.2494] | 0.2424 [0.2417-0.2435] | 97.8 | 90 / 92 | faster |
| gdn_in | Q4 | dense-decode | 5 | f32 | 0.2570 [0.2549-0.2595] | 0.2484 [0.2462-0.2502] | 96.7 | 87 / 90 | faster |
| gdn_in | Q4 | dense-decode | 6 | f32 | 0.2625 [0.2611-0.2636] | 0.2505 [0.2468-0.2530] | 95.4 | 85 / 89 | faster |
| gdn_in | Q4 | dense-decode | 7 | f32 | 0.2684 [0.2667-0.2700] | 0.2541 [0.2534-0.2568] | 94.7 | 84 / 88 | faster |
| gdn_in | Q4 | dense-decode | 8 | f32 | 0.2724 [0.2709-0.2739] | 0.2580 [0.2556-0.2603] | 94.7 | 82 / 87 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 512 | f32 | 1.1869 [1.1778-1.1988] | 1.1555 [1.1452-1.1719] | 97.4 | 19 / 19 | faster |
| gdn_in | Q4 | dense-prefill-wmma | 2048 | f32 | 5.8705 [5.5490-5.9036] | 5.8316 [5.2923-5.8743] | 99.3 | 4 / 4 | noise |
| gdn_in | Q4 | dense-prefill-wmma | 8192 | f32 | 21.9841 [21.8346-22.0199] | 21.9007 [21.7915-21.9931] | 99.6 | 1 / 1 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | bf16 | 0.3997 [0.3976-0.4014] | 0.4039 [0.4016-0.4059] | 101.1 | 109 / 108 | slower |
| gdn_in | Q8 syn | dense-decode | 2 | bf16 | 0.3982 [0.3972-0.4000] | 0.4029 [0.4011-0.4064] | 101.2 | 109 / 108 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | bf16 | 0.4082 [0.4065-0.4104] | 0.4077 [0.4063-0.4088] | 99.9 | 107 / 107 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | bf16 | 0.4166 [0.4143-0.4177] | 0.4133 [0.4118-0.4149] | 99.2 | 104 / 105 | noise |
| gdn_in | Q8 syn | dense-decode | 5 | bf16 | 0.4350 [0.4328-0.4381] | 0.4215 [0.4196-0.4258] | 96.9 | 100 / 103 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | bf16 | 0.4442 [0.4421-0.4466] | 0.4266 [0.4255-0.4299] | 96.0 | 98 / 102 | faster |
| gdn_in | Q8 syn | dense-decode | 7 | bf16 | 0.4562 [0.4526-0.4597] | 0.4445 [0.4399-0.4483] | 97.4 | 95 / 98 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | bf16 | 0.4614 [0.4597-0.4643] | 0.4481 [0.4474-0.4547] | 97.1 | 94 / 97 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | bf16 | 1.6289 [1.5862-1.6671] | 1.3262 [1.3039-1.3488] | 81.4 | 27 / 33 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | bf16 | 5.7984 [5.3139-6.1189] | 5.3763 [5.1439-5.8828] | 92.7 | 8 / 8 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | bf16 | 24.0232 [23.8776-24.3642] | 23.5857 [23.4268-24.0277] | 98.2 | 2 / 2 | noise |
| gdn_in | Q8 syn | dense-decode | 1 | f32 | 0.3974 [0.3964-0.3999] | 0.4010 [0.3993-0.4023] | 100.9 | 109 / 108 | noise |
| gdn_in | Q8 syn | dense-decode | 2 | f32 | 0.3970 [0.3955-0.3990] | 0.4030 [0.3994-0.4047] | 101.5 | 110 / 108 | slower |
| gdn_in | Q8 syn | dense-decode | 3 | f32 | 0.4079 [0.4071-0.4088] | 0.4071 [0.4051-0.4087] | 99.8 | 107 / 107 | noise |
| gdn_in | Q8 syn | dense-decode | 4 | f32 | 0.4128 [0.4103-0.4147] | 0.4082 [0.4074-0.4114] | 98.9 | 105 / 107 | noise |
| gdn_in | Q8 syn | dense-decode | 5 | f32 | 0.4323 [0.4310-0.4352] | 0.4210 [0.4180-0.4244] | 97.4 | 101 / 103 | faster |
| gdn_in | Q8 syn | dense-decode | 6 | f32 | 0.4427 [0.4393-0.4461] | 0.4290 [0.4268-0.4320] | 96.9 | 98 / 101 | faster |
| gdn_in | Q8 syn | dense-decode | 7 | f32 | 0.4557 [0.4507-0.4578] | 0.4416 [0.4371-0.4470] | 96.9 | 95 / 99 | faster |
| gdn_in | Q8 syn | dense-decode | 8 | f32 | 0.4689 [0.4666-0.4757] | 0.4531 [0.4487-0.4576] | 96.6 | 93 / 96 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 512 | f32 | 1.6038 [1.5913-1.6494] | 1.3637 [1.3480-1.3967] | 85.0 | 27 / 32 | faster |
| gdn_in | Q8 syn | dense-prefill-wmma | 2048 | f32 | 6.3291 [5.5221-6.3983] | 6.0626 [5.3757-6.1295] | 95.8 | 7 / 7 | noise |
| gdn_in | Q8 syn | dense-prefill-wmma | 8192 | f32 | 23.4755 [23.4060-23.9604] | 23.1830 [23.0596-23.5989] | 98.8 | 2 / 2 | noise |
| gdn_out | Q8 | dense-decode | 1 | bf16 | 0.1839 [0.1831-0.1846] | 0.1836 [0.1831-0.1840] | 99.8 | 91 / 91 | noise |
| gdn_out | Q8 | dense-decode | 2 | bf16 | 0.1846 [0.1827-0.1854] | 0.1844 [0.1832-0.1852] | 99.9 | 91 / 91 | noise |
| gdn_out | Q8 | dense-decode | 3 | bf16 | 0.1873 [0.1858-0.1896] | 0.1900 [0.1892-0.1912] | 101.4 | 89 / 88 | noise |
| gdn_out | Q8 | dense-decode | 4 | bf16 | 0.1928 [0.1907-0.1944] | 0.1942 [0.1935-0.1949] | 100.8 | 87 / 86 | noise |
| gdn_out | Q8 | dense-decode | 5 | bf16 | 0.2065 [0.2010-0.2118] | 0.2078 [0.2065-0.2089] | 100.6 | 81 / 80 | noise |
| gdn_out | Q8 | dense-decode | 6 | bf16 | 0.2164 [0.2128-0.2213] | 0.2108 [0.2095-0.2126] | 97.4 | 77 / 79 | faster |
| gdn_out | Q8 | dense-decode | 7 | bf16 | 0.2270 [0.2229-0.2330] | 0.2154 [0.2138-0.2162] | 94.9 | 74 / 78 | faster |
| gdn_out | Q8 | dense-decode | 8 | bf16 | 0.2334 [0.2298-0.2350] | 0.2202 [0.2179-0.2209] | 94.3 | 72 / 76 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3995 [1.1861-1.4700] | 0.7921 [0.7593-0.8334] | 56.6 | 12 / 21 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | bf16 | 3.0578 [3.0332-3.1628] | 2.1798 [2.1070-2.2506] | 71.3 | 5 / 8 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | bf16 | 8.9339 [8.3918-9.2638] | 7.7069 [7.5063-8.3261] | 86.3 | 2 / 2 | faster |
| gdn_out | Q8 | dense-decode | 1 | f32 | 0.1812 [0.1808-0.1818] | 0.1816 [0.1805-0.1821] | 100.2 | 92 / 92 | noise |
| gdn_out | Q8 | dense-decode | 2 | f32 | 0.1823 [0.1814-0.1832] | 0.1823 [0.1815-0.1831] | 100.0 | 92 / 92 | noise |
| gdn_out | Q8 | dense-decode | 3 | f32 | 0.1868 [0.1858-0.1873] | 0.1888 [0.1877-0.1905] | 101.1 | 89 / 88 | slower |
| gdn_out | Q8 | dense-decode | 4 | f32 | 0.1924 [0.1898-0.1942] | 0.1945 [0.1937-0.1953] | 101.1 | 87 / 86 | noise |
| gdn_out | Q8 | dense-decode | 5 | f32 | 0.2057 [0.2024-0.2081] | 0.2072 [0.2058-0.2085] | 100.7 | 81 / 81 | noise |
| gdn_out | Q8 | dense-decode | 6 | f32 | 0.2163 [0.2104-0.2221] | 0.2104 [0.2095-0.2113] | 97.3 | 77 / 79 | noise |
| gdn_out | Q8 | dense-decode | 7 | f32 | 0.2198 [0.2172-0.2288] | 0.2153 [0.2132-0.2171] | 97.9 | 76 / 78 | faster |
| gdn_out | Q8 | dense-decode | 8 | f32 | 0.2315 [0.2253-0.2358] | 0.2194 [0.2176-0.2208] | 94.8 | 72 / 76 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 512 | f32 | 1.3330 [1.2317-1.4656] | 0.7780 [0.7447-0.8293] | 58.4 | 13 / 21 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 2048 | f32 | 3.1375 [3.0416-3.2225] | 2.0689 [2.0387-2.1825] | 65.9 | 5 / 8 | faster |
| gdn_out | Q8 | dense-prefill-wmma | 8192 | f32 | 9.6662 [9.4089-9.8451] | 8.7443 [8.5870-8.8986] | 90.5 | 2 / 2 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | bf16 | 0.1029 [0.1023-0.1041] | 0.1025 [0.1018-0.1041] | 99.7 | 86 / 86 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | bf16 | 0.1062 [0.1052-0.1068] | 0.1050 [0.1040-0.1053] | 98.9 | 83 / 84 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | bf16 | 0.1104 [0.1098-0.1119] | 0.1080 [0.1069-0.1090] | 97.9 | 80 / 82 | faster |
| gdn_out | Q4 syn | dense-decode | 4 | bf16 | 0.1134 [0.1126-0.1143] | 0.1105 [0.1094-0.1118] | 97.5 | 78 / 80 | faster |
| gdn_out | Q4 syn | dense-decode | 5 | bf16 | 0.1232 [0.1216-0.1242] | 0.1206 [0.1191-0.1221] | 97.9 | 72 / 73 | noise |
| gdn_out | Q4 syn | dense-decode | 6 | bf16 | 0.1250 [0.1240-0.1271] | 0.1208 [0.1198-0.1234] | 96.7 | 71 / 73 | faster |
| gdn_out | Q4 syn | dense-decode | 7 | bf16 | 0.1290 [0.1278-0.1307] | 0.1238 [0.1222-0.1252] | 96.0 | 69 / 71 | faster |
| gdn_out | Q4 syn | dense-decode | 8 | bf16 | 0.1350 [0.1329-0.1360] | 0.1285 [0.1272-0.1297] | 95.1 | 66 / 69 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0411 [1.0225-1.0591] | 0.6296 [0.6109-0.6523] | 60.5 | 8 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.5699 [2.5420-2.6179] | 1.9754 [1.9435-2.0795] | 76.9 | 3 / 4 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 8.9477 [8.8113-9.0030] | 8.4650 [8.3484-8.5198] | 94.6 | 1 / 1 | faster |
| gdn_out | Q4 syn | dense-decode | 1 | f32 | 0.1021 [0.1016-0.1028] | 0.1029 [0.1024-0.1035] | 100.8 | 87 / 86 | noise |
| gdn_out | Q4 syn | dense-decode | 2 | f32 | 0.1058 [0.1048-0.1070] | 0.1046 [0.1040-0.1054] | 98.9 | 84 / 85 | noise |
| gdn_out | Q4 syn | dense-decode | 3 | f32 | 0.1085 [0.1072-0.1096] | 0.1070 [0.1061-0.1074] | 98.6 | 82 / 83 | noise |
| gdn_out | Q4 syn | dense-decode | 4 | f32 | 0.1116 [0.1105-0.1124] | 0.1097 [0.1088-0.1105] | 98.2 | 79 / 81 | noise |
| gdn_out | Q4 syn | dense-decode | 5 | f32 | 0.1187 [0.1179-0.1197] | 0.1174 [0.1169-0.1182] | 98.9 | 75 / 75 | noise |
| gdn_out | Q4 syn | dense-decode | 6 | f32 | 0.1239 [0.1232-0.1247] | 0.1213 [0.1207-0.1231] | 97.9 | 71 / 73 | faster |
| gdn_out | Q4 syn | dense-decode | 7 | f32 | 0.1298 [0.1276-0.1310] | 0.1234 [0.1220-0.1252] | 95.1 | 68 / 72 | faster |
| gdn_out | Q4 syn | dense-decode | 8 | f32 | 0.1342 [0.1319-0.1354] | 0.1272 [0.1261-0.1287] | 94.8 | 66 / 70 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0194 [1.0113-1.0671] | 0.6150 [0.6013-0.6418] | 60.3 | 9 / 14 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.5564 [2.5031-2.6276] | 1.9583 [1.9296-2.0071] | 76.6 | 3 / 5 | faster |
| gdn_out | Q4 syn | dense-prefill-wmma | 8192 | f32 | 9.0125 [8.8297-9.1074] | 8.5560 [8.4234-8.7179] | 94.9 | 1 / 1 | faster |
| attn_qkv | Q8 | dense-decode | 1 | bf16 | 0.3717 [0.3706-0.3745] | 0.3730 [0.3712-0.3748] | 100.3 | 102 / 102 | noise |
| attn_qkv | Q8 | dense-decode | 2 | bf16 | 0.3761 [0.3736-0.3777] | 0.3794 [0.3769-0.3813] | 100.9 | 101 / 100 | noise |
| attn_qkv | Q8 | dense-decode | 3 | bf16 | 0.3807 [0.3792-0.3822] | 0.3805 [0.3776-0.3810] | 99.9 | 100 / 100 | noise |
| attn_qkv | Q8 | dense-decode | 4 | bf16 | 0.3875 [0.3861-0.3912] | 0.3872 [0.3859-0.3902] | 99.9 | 98 / 98 | noise |
| attn_qkv | Q8 | dense-decode | 5 | bf16 | 0.4070 [0.4052-0.4110] | 0.3988 [0.3972-0.4032] | 98.0 | 93 / 95 | faster |
| attn_qkv | Q8 | dense-decode | 6 | bf16 | 0.4168 [0.4152-0.4188] | 0.4060 [0.4034-0.4107] | 97.4 | 91 / 93 | faster |
| attn_qkv | Q8 | dense-decode | 7 | bf16 | 0.4208 [0.4179-0.4231] | 0.4165 [0.4117-0.4213] | 99.0 | 90 / 91 | noise |
| attn_qkv | Q8 | dense-decode | 8 | bf16 | 0.4260 [0.4240-0.4285] | 0.4241 [0.4154-0.4258] | 99.5 | 89 / 89 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3687 [1.3385-1.4007] | 1.1531 [1.1451-1.1666] | 84.2 | 28 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | bf16 | 5.3358 [4.9495-5.4646] | 5.0927 [4.7935-5.1572] | 95.4 | 7 / 7 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | bf16 | 20.6112 [20.4366-21.2648] | 20.1555 [19.9369-20.4688] | 97.8 | 2 / 2 | noise |
| attn_qkv | Q8 | dense-decode | 1 | f32 | 0.3746 [0.3728-0.3769] | 0.3748 [0.3731-0.3769] | 100.0 | 101 / 101 | noise |
| attn_qkv | Q8 | dense-decode | 2 | f32 | 0.3709 [0.3692-0.3731] | 0.3747 [0.3730-0.3765] | 101.0 | 102 / 101 | noise |
| attn_qkv | Q8 | dense-decode | 3 | f32 | 0.3797 [0.3778-0.3813] | 0.3801 [0.3794-0.3815] | 100.1 | 100 / 100 | noise |
| attn_qkv | Q8 | dense-decode | 4 | f32 | 0.3848 [0.3837-0.3864] | 0.3842 [0.3822-0.3866] | 99.9 | 99 / 99 | noise |
| attn_qkv | Q8 | dense-decode | 5 | f32 | 0.4052 [0.4020-0.4079] | 0.4006 [0.3951-0.4037] | 98.9 | 94 / 95 | noise |
| attn_qkv | Q8 | dense-decode | 6 | f32 | 0.4114 [0.4089-0.4140] | 0.4024 [0.3998-0.4047] | 97.8 | 92 / 94 | faster |
| attn_qkv | Q8 | dense-decode | 7 | f32 | 0.4165 [0.4137-0.4208] | 0.4123 [0.4069-0.4150] | 99.0 | 91 / 92 | noise |
| attn_qkv | Q8 | dense-decode | 8 | f32 | 0.4238 [0.4189-0.4268] | 0.4198 [0.4157-0.4242] | 99.1 | 90 / 90 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 512 | f32 | 1.4184 [1.3913-1.4737] | 1.1640 [1.1496-1.2452] | 82.1 | 27 / 33 | faster |
| attn_qkv | Q8 | dense-prefill-wmma | 2048 | f32 | 5.4875 [4.8276-5.6129] | 5.0876 [4.5948-5.2809] | 92.7 | 7 / 7 | noise |
| attn_qkv | Q8 | dense-prefill-wmma | 8192 | f32 | 20.9453 [20.6093-21.2173] | 20.3743 [20.0487-20.5312] | 97.3 | 2 / 2 | faster |
| attn_qkv | Q4 syn | dense-decode | 1 | bf16 | 0.2154 [0.2150-0.2172] | 0.2144 [0.2116-0.2160] | 99.5 | 93 / 94 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | bf16 | 0.2179 [0.2169-0.2210] | 0.2167 [0.2146-0.2192] | 99.5 | 92 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 3 | bf16 | 0.2226 [0.2210-0.2239] | 0.2186 [0.2173-0.2206] | 98.2 | 90 / 92 | faster |
| attn_qkv | Q4 syn | dense-decode | 4 | bf16 | 0.2246 [0.2228-0.2259] | 0.2207 [0.2197-0.2228] | 98.2 | 89 / 91 | noise |
| attn_qkv | Q4 syn | dense-decode | 5 | bf16 | 0.2344 [0.2318-0.2351] | 0.2240 [0.2236-0.2274] | 95.6 | 86 / 90 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | bf16 | 0.2389 [0.2362-0.2403] | 0.2282 [0.2252-0.2308] | 95.5 | 84 / 88 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | bf16 | 0.2426 [0.2416-0.2435] | 0.2301 [0.2270-0.2318] | 94.8 | 83 / 87 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | bf16 | 0.2490 [0.2478-0.2505] | 0.2320 [0.2313-0.2346] | 93.2 | 81 / 87 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0738 [1.0573-1.0825] | 1.0214 [1.0126-1.0323] | 95.1 | 19 / 20 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 4.5290 [4.2161-4.8016] | 4.4162 [4.1474-4.6954] | 97.5 | 4 / 5 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 19.1745 [19.0782-19.3415] | 18.9926 [18.9267-19.1489] | 99.0 | 1 / 1 | noise |
| attn_qkv | Q4 syn | dense-decode | 1 | f32 | 0.2160 [0.2151-0.2165] | 0.2158 [0.2141-0.2171] | 99.9 | 93 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 2 | f32 | 0.2180 [0.2168-0.2193] | 0.2159 [0.2149-0.2175] | 99.1 | 92 / 93 | noise |
| attn_qkv | Q4 syn | dense-decode | 3 | f32 | 0.2205 [0.2193-0.2227] | 0.2191 [0.2182-0.2199] | 99.4 | 91 / 92 | noise |
| attn_qkv | Q4 syn | dense-decode | 4 | f32 | 0.2238 [0.2216-0.2254] | 0.2200 [0.2160-0.2214] | 98.3 | 90 / 91 | faster |
| attn_qkv | Q4 syn | dense-decode | 5 | f32 | 0.2312 [0.2285-0.2334] | 0.2235 [0.2204-0.2255] | 96.6 | 87 / 90 | faster |
| attn_qkv | Q4 syn | dense-decode | 6 | f32 | 0.2372 [0.2364-0.2388] | 0.2283 [0.2270-0.2301] | 96.3 | 85 / 88 | faster |
| attn_qkv | Q4 syn | dense-decode | 7 | f32 | 0.2438 [0.2420-0.2448] | 0.2317 [0.2305-0.2357] | 95.0 | 82 / 87 | faster |
| attn_qkv | Q4 syn | dense-decode | 8 | f32 | 0.2476 [0.2454-0.2502] | 0.2333 [0.2324-0.2359] | 94.2 | 81 / 86 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0687 [1.0632-1.0918] | 1.0177 [0.9974-1.0541] | 95.2 | 19 / 20 | faster |
| attn_qkv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 4.6819 [4.3433-4.9297] | 4.6763 [4.2611-4.9061] | 99.9 | 4 / 4 | noise |
| attn_qkv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 18.8800 [18.7245-19.2059] | 18.7359 [18.5436-18.9698] | 99.2 | 1 / 1 | noise |
| attn_o | Q8 | dense-decode | 1 | bf16 | 0.1813 [0.1804-0.1816] | 0.1812 [0.1808-0.1822] | 100.0 | 92 / 92 | noise |
| attn_o | Q8 | dense-decode | 2 | bf16 | 0.1840 [0.1824-0.1852] | 0.1834 [0.1822-0.1844] | 99.7 | 91 / 91 | noise |
| attn_o | Q8 | dense-decode | 3 | bf16 | 0.1867 [0.1839-0.1900] | 0.1887 [0.1872-0.1899] | 101.1 | 90 / 89 | noise |
| attn_o | Q8 | dense-decode | 4 | bf16 | 0.1915 [0.1869-0.1935] | 0.1938 [0.1931-0.1946] | 101.2 | 87 / 86 | noise |
| attn_o | Q8 | dense-decode | 5 | bf16 | 0.2046 [0.2018-0.2101] | 0.2060 [0.2046-0.2080] | 100.7 | 82 / 81 | noise |
| attn_o | Q8 | dense-decode | 6 | bf16 | 0.2166 [0.2128-0.8637] | 0.2123 [0.2101-0.3155] | 98.0 | 77 / 79 | noise |
| attn_o | Q8 | dense-decode | 7 | bf16 | 0.9185 [0.8993-0.9309] | 0.8758 [0.8690-0.8824] | 95.3 | 18 / 19 | faster |
| attn_o | Q8 | dense-decode | 8 | bf16 | 0.9452 [0.9254-0.9629] | 0.8969 [0.8938-0.9067] | 94.9 | 18 / 19 | faster |
| attn_o | Q8 | dense-prefill-wmma | 512 | bf16 | 5.7658 [5.4295-6.1226] | 3.2570 [3.1612-3.4149] | 56.5 | 3 / 5 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | bf16 | 13.2112 [12.9591-13.4279] | 9.2855 [9.1288-9.7155] | 70.3 | 1 / 2 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | bf16 | 34.8944 [34.6677-35.1011] | 30.4108 [30.1265-30.6608] | 87.2 | 0 / 1 | faster |
| attn_o | Q8 | dense-decode | 1 | f32 | 0.7460 [0.7428-0.7535] | 0.7454 [0.7417-0.7526] | 99.9 | 22 / 22 | noise |
| attn_o | Q8 | dense-decode | 2 | f32 | 0.7716 [0.7642-0.7889] | 0.7696 [0.7647-0.7777] | 99.7 | 22 / 22 | noise |
| attn_o | Q8 | dense-decode | 3 | f32 | 0.7801 [0.7724-0.7862] | 0.7887 [0.7824-0.8037] | 101.1 | 21 / 21 | noise |
| attn_o | Q8 | dense-decode | 4 | f32 | 0.7917 [0.7873-0.8027] | 0.7977 [0.7950-0.8073] | 100.8 | 21 / 21 | noise |
| attn_o | Q8 | dense-decode | 5 | f32 | 0.8416 [0.8273-0.8568] | 0.8438 [0.8397-0.8484] | 100.3 | 20 / 20 | noise |
| attn_o | Q8 | dense-decode | 6 | f32 | 0.2177 [0.2127-0.2207] | 0.2106 [0.2085-0.2127] | 96.7 | 77 / 79 | noise |
| attn_o | Q8 | dense-decode | 7 | f32 | 0.2234 [0.2214-0.2298] | 0.2155 [0.2141-0.2164] | 96.5 | 75 / 78 | faster |
| attn_o | Q8 | dense-decode | 8 | f32 | 0.2274 [0.2254-0.2314] | 0.2190 [0.2183-0.2200] | 96.3 | 74 / 76 | faster |
| attn_o | Q8 | dense-prefill-wmma | 512 | f32 | 1.3362 [1.2190-1.3707] | 0.7845 [0.7317-0.8165] | 58.7 | 13 / 21 | faster |
| attn_o | Q8 | dense-prefill-wmma | 2048 | f32 | 3.1916 [3.1292-3.2315] | 2.2041 [2.1805-2.2377] | 69.1 | 5 / 8 | faster |
| attn_o | Q8 | dense-prefill-wmma | 8192 | f32 | 9.3674 [9.1865-9.4575] | 8.6178 [8.3864-8.7282] | 92.0 | 2 / 2 | faster |
| attn_o | Q4 syn | dense-decode | 1 | bf16 | 0.1024 [0.1023-0.1030] | 0.1026 [0.1018-0.1030] | 100.2 | 86 / 86 | noise |
| attn_o | Q4 syn | dense-decode | 2 | bf16 | 0.1058 [0.1049-0.1067] | 0.1046 [0.1036-0.1053] | 98.9 | 84 / 85 | noise |
| attn_o | Q4 syn | dense-decode | 3 | bf16 | 0.1095 [0.1087-0.1106] | 0.1076 [0.1067-0.1082] | 98.3 | 81 / 82 | faster |
| attn_o | Q4 syn | dense-decode | 4 | bf16 | 0.1129 [0.1124-0.1138] | 0.1101 [0.1091-0.1107] | 97.5 | 78 / 80 | faster |
| attn_o | Q4 syn | dense-decode | 5 | bf16 | 0.1195 [0.1186-0.1203] | 0.1173 [0.1166-0.1184] | 98.2 | 74 / 75 | faster |
| attn_o | Q4 syn | dense-decode | 6 | bf16 | 0.1239 [0.1232-0.1249] | 0.1202 [0.1198-0.1220] | 97.1 | 71 / 74 | faster |
| attn_o | Q4 syn | dense-decode | 7 | bf16 | 0.1276 [0.1270-0.1288] | 0.1238 [0.1224-0.1254] | 97.1 | 69 / 71 | faster |
| attn_o | Q4 syn | dense-decode | 8 | bf16 | 0.1330 [0.1323-0.1339] | 0.1273 [0.1260-0.1288] | 95.7 | 67 / 70 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | bf16 | 1.0248 [0.9997-1.0597] | 0.6185 [0.6113-0.6329] | 60.4 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 2.6608 [2.6057-2.6879] | 2.1163 [2.0621-2.1911] | 79.5 | 3 / 4 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 8.7415 [8.5624-8.9266] | 8.1962 [7.9772-8.4759] | 93.8 | 1 / 1 | faster |
| attn_o | Q4 syn | dense-decode | 1 | f32 | 0.1028 [0.1013-0.1034] | 0.1028 [0.1022-0.1035] | 100.0 | 86 / 86 | noise |
| attn_o | Q4 syn | dense-decode | 2 | f32 | 0.1049 [0.1038-0.1063] | 0.1046 [0.1032-0.1051] | 99.7 | 84 / 85 | noise |
| attn_o | Q4 syn | dense-decode | 3 | f32 | 0.1084 [0.1078-0.1094] | 0.1076 [0.1067-0.1083] | 99.2 | 82 / 82 | noise |
| attn_o | Q4 syn | dense-decode | 4 | f32 | 0.1124 [0.1114-0.1136] | 0.1098 [0.1090-0.1104] | 97.8 | 79 / 81 | faster |
| attn_o | Q4 syn | dense-decode | 5 | f32 | 0.1194 [0.1174-0.1206] | 0.1172 [0.1166-0.1183] | 98.2 | 74 / 76 | noise |
| attn_o | Q4 syn | dense-decode | 6 | f32 | 0.1235 [0.1227-0.1239] | 0.1205 [0.1197-0.1222] | 97.6 | 72 / 73 | faster |
| attn_o | Q4 syn | dense-decode | 7 | f32 | 0.1275 [0.1260-0.1295] | 0.1220 [0.1215-0.1249] | 95.7 | 69 / 72 | faster |
| attn_o | Q4 syn | dense-decode | 8 | f32 | 0.1328 [0.1311-0.1333] | 0.1259 [0.1246-0.1281] | 94.8 | 67 / 70 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.0294 [1.0135-1.0720] | 0.6282 [0.6114-0.6533] | 61.0 | 9 / 14 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 2048 | f32 | 2.6222 [2.5813-2.6772] | 2.0987 [2.0450-2.1131] | 80.0 | 3 / 4 | faster |
| attn_o | Q4 syn | dense-prefill-wmma | 8192 | f32 | 32.4215 [8.8921-32.7373] | 29.6008 [8.2933-29.7471] | 91.3 | 0 / 0 | noise |
| ple_kv | Q8 | dense-decode | 1 | bf16 | 1.4747 [1.4713-1.4864] | 1.4782 [1.4705-1.4871] | 100.2 | 24 / 24 | noise |
| ple_kv | Q8 | dense-decode | 2 | bf16 | 1.4980 [1.4800-1.5049] | 1.5043 [1.4995-1.5109] | 100.4 | 23 / 23 | noise |
| ple_kv | Q8 | dense-decode | 3 | bf16 | 1.5233 [1.5063-1.5369] | 1.5178 [1.5019-1.5281] | 99.6 | 23 / 23 | noise |
| ple_kv | Q8 | dense-decode | 4 | bf16 | 1.5323 [1.5248-1.5402] | 1.5235 [1.5161-1.5330] | 99.4 | 23 / 23 | noise |
| ple_kv | Q8 | dense-decode | 5 | bf16 | 1.5893 [1.5813-1.5953] | 1.5723 [1.5683-1.5859] | 98.9 | 22 / 22 | noise |
| ple_kv | Q8 | dense-decode | 6 | bf16 | 1.6110 [1.5991-1.6215] | 1.6227 [1.6111-1.6293] | 100.7 | 22 / 21 | noise |
| ple_kv | Q8 | dense-decode | 7 | bf16 | 1.6313 [1.6266-1.6397] | 1.6529 [1.6366-1.6699] | 101.3 | 21 / 21 | noise |
| ple_kv | Q8 | dense-decode | 8 | bf16 | 1.6539 [1.6362-1.6702] | 1.6851 [1.6673-1.7093] | 101.9 | 21 / 21 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 512 | bf16 | 1.3025 [1.2899-1.3446] | 1.0704 [1.0652-1.0788] | 82.2 | 27 / 33 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | bf16 | 4.4364 [4.3685-4.5753] | 4.1406 [4.0875-4.2925] | 93.3 | 8 / 8 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | bf16 | 18.9675 [18.8431-19.3012] | 18.6256 [18.4967-18.7165] | 98.2 | 2 / 2 | faster |
| ple_kv | Q8 | dense-decode | 1 | f32 | 0.3597 [0.3586-0.3638] | 0.3617 [0.3604-0.3655] | 100.6 | 97 / 96 | noise |
| ple_kv | Q8 | dense-decode | 2 | f32 | 0.3599 [0.3572-0.3619] | 0.3636 [0.3622-0.3657] | 101.0 | 97 / 96 | slower |
| ple_kv | Q8 | dense-decode | 3 | f32 | 0.3656 [0.3634-0.3672] | 0.3673 [0.3644-0.3684] | 100.5 | 95 / 95 | noise |
| ple_kv | Q8 | dense-decode | 4 | f32 | 0.3692 [0.3682-0.3711] | 0.3730 [0.3712-0.3754] | 101.0 | 94 / 93 | slower |
| ple_kv | Q8 | dense-decode | 5 | f32 | 0.3849 [0.3819-0.3879] | 0.3844 [0.3818-0.3867] | 99.9 | 90 / 91 | noise |
| ple_kv | Q8 | dense-decode | 6 | f32 | 0.3916 [0.3901-0.3947] | 0.3915 [0.3868-0.3945] | 100.0 | 89 / 89 | noise |
| ple_kv | Q8 | dense-decode | 7 | f32 | 0.4044 [0.4003-0.4061] | 0.3997 [0.3980-0.4022] | 98.8 | 86 / 87 | noise |
| ple_kv | Q8 | dense-decode | 8 | f32 | 0.4089 [0.4048-0.4121] | 0.4106 [0.4026-0.4145] | 100.4 | 85 / 85 | noise |
| ple_kv | Q8 | dense-prefill-wmma | 512 | f32 | 1.3436 [1.3267-1.3556] | 1.1171 [1.1122-1.1235] | 83.1 | 26 / 31 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 2048 | f32 | 4.5915 [4.5120-4.7085] | 4.3905 [4.2593-4.4315] | 95.6 | 8 / 8 | faster |
| ple_kv | Q8 | dense-prefill-wmma | 8192 | f32 | 18.6248 [18.2272-19.4721] | 18.3976 [17.7922-19.1575] | 98.8 | 2 / 2 | noise |
| ple_kv | Q4 syn | dense-decode | 1 | bf16 | 0.2008 [0.1998-0.2013] | 0.2000 [0.1981-0.2012] | 99.6 | 92 / 92 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | bf16 | 0.2033 [0.2022-0.2057] | 0.2004 [0.1984-0.2014] | 98.5 | 91 / 92 | faster |
| ple_kv | Q4 syn | dense-decode | 3 | bf16 | 0.2073 [0.2060-0.2084] | 0.2048 [0.2015-0.2057] | 98.8 | 89 / 90 | faster |
| ple_kv | Q4 syn | dense-decode | 4 | bf16 | 0.2090 [0.2076-0.2112] | 0.2050 [0.2030-0.2066] | 98.1 | 88 / 90 | faster |
| ple_kv | Q4 syn | dense-decode | 5 | bf16 | 0.9134 [0.9086-0.9211] | 0.8677 [0.8635-0.8715] | 95.0 | 20 / 21 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | bf16 | 0.9211 [0.9147-0.9342] | 0.8688 [0.8659-0.8718] | 94.3 | 20 / 21 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | bf16 | 0.9395 [0.9333-0.9468] | 0.8795 [0.8758-0.8935] | 93.6 | 20 / 21 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | bf16 | 0.9732 [0.9627-0.9795] | 0.8987 [0.8937-0.9080] | 92.3 | 19 / 21 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | bf16 | 4.3428 [4.3103-4.3665] | 4.1300 [4.1194-4.1413] | 95.1 | 4 / 4 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 15.9800 [15.9599-16.0556] | 15.7095 [15.6824-15.7348] | 98.3 | 1 / 1 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 19.5392 [18.8507-62.3260] | 19.4311 [18.5050-61.7344] | 99.5 | 1 / 1 | noise |
| ple_kv | Q4 syn | dense-decode | 1 | f32 | 0.2014 [0.2005-0.2028] | 0.2008 [0.1980-0.2015] | 99.7 | 92 / 92 | noise |
| ple_kv | Q4 syn | dense-decode | 2 | f32 | 0.2051 [0.2031-0.2088] | 0.2034 [0.2018-0.2063] | 99.1 | 90 / 91 | noise |
| ple_kv | Q4 syn | dense-decode | 3 | f32 | 0.2076 [0.2056-0.2099] | 0.2046 [0.2023-0.2057] | 98.5 | 89 / 90 | noise |
| ple_kv | Q4 syn | dense-decode | 4 | f32 | 0.2106 [0.2080-0.2126] | 0.2061 [0.2032-0.2093] | 97.8 | 88 / 89 | noise |
| ple_kv | Q4 syn | dense-decode | 5 | f32 | 0.2178 [0.2173-0.2194] | 0.2104 [0.2090-0.2123] | 96.6 | 85 / 88 | faster |
| ple_kv | Q4 syn | dense-decode | 6 | f32 | 0.2207 [0.2198-0.2228] | 0.2115 [0.2087-0.2133] | 95.8 | 84 / 87 | faster |
| ple_kv | Q4 syn | dense-decode | 7 | f32 | 0.2241 [0.2224-0.2268] | 0.2154 [0.2134-0.2180] | 96.1 | 82 / 86 | faster |
| ple_kv | Q4 syn | dense-decode | 8 | f32 | 0.2297 [0.2271-0.2312] | 0.2168 [0.2147-0.2198] | 94.4 | 80 / 85 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 512 | f32 | 1.1072 [1.1036-1.1278] | 1.0662 [1.0593-1.0916] | 96.3 | 17 / 17 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 2048 | f32 | 4.5701 [4.5254-4.6139] | 4.4983 [4.4708-4.5194] | 98.4 | 4 / 4 | faster |
| ple_kv | Q4 syn | dense-prefill-wmma | 8192 | f32 | 18.9799 [18.8616-19.0940] | 18.5701 [18.5349-18.7306] | 97.8 | 1 / 1 | faster |
| lm_head | Q8 | dense-decode | 1 | bf16 | 3.2368 [3.2216-3.2434] | 3.2489 [3.2407-3.2558] | 100.4 | 209 / 208 | noise |
| lm_head | Q8 | dense-decode | 2 | bf16 | 3.2766 [3.2350-3.3160] | 3.3064 [3.2551-3.3210] | 100.9 | 206 / 204 | noise |
| lm_head | Q8 | dense-decode | 3 | bf16 | 15.7511 [3.5156-15.8703] | 14.8163 [3.3260-14.9955] | 94.1 | 43 / 46 | noise |
| lm_head | Q8 | dense-decode | 4 | bf16 | 16.3975 [16.2586-16.5441] | 15.1793 [14.9855-15.3606] | 92.6 | 41 / 44 | faster |
| lm_head | Q8 | dense-decode | 5 | bf16 | 15.7428 [15.6649-15.8879] | 15.3510 [15.2330-15.4091] | 97.5 | 43 / 44 | faster |
| lm_head | Q8 | dense-decode | 6 | bf16 | 16.0879 [16.0072-16.1221] | 15.5804 [15.5283-15.7051] | 96.8 | 42 / 43 | faster |
| lm_head | Q8 | dense-decode | 7 | bf16 | 16.5703 [16.4431-16.7798] | 16.1392 [16.0195-16.2453] | 97.4 | 41 / 42 | faster |
| lm_head | Q8 | dense-decode | 8 | bf16 | 16.7279 [3.8687-17.0727] | 16.5104 [3.9660-16.6614] | 98.7 | 40 / 41 | noise |
| lm_head | Q8 | dense-prefill-wmma | 512 | bf16 | 23.5210 [22.9915-23.7380] | 22.9294 [22.5561-23.4332] | 97.5 | 29 / 29 | noise |
| lm_head | Q8 | dense-prefill-wmma | 2048 | bf16 | 91.0053 [90.2812-92.2853] | 88.6458 [87.0996-90.3063] | 97.4 | 7 / 8 | noise |
| lm_head | Q8 | dense-decode | 1 | f32 | 3.2541 [3.2380-3.2615] | 3.2655 [3.2477-3.2798] | 100.3 | 208 / 207 | noise |
| lm_head | Q8 | dense-decode | 2 | f32 | 3.2454 [3.2347-3.2588] | 3.2643 [3.2537-3.2709] | 100.6 | 208 / 207 | noise |
| lm_head | Q8 | dense-decode | 3 | f32 | 3.5012 [3.4893-3.5122] | 3.3058 [3.2898-3.3220] | 94.4 | 193 / 204 | faster |
| lm_head | Q8 | dense-decode | 4 | f32 | 3.6420 [3.6190-3.6524] | 3.3650 [3.3290-3.4025] | 92.4 | 185 / 201 | faster |
| lm_head | Q8 | dense-decode | 5 | f32 | 3.5093 [3.4851-3.5269] | 3.4158 [3.4014-3.4297] | 97.3 | 192 / 198 | faster |
| lm_head | Q8 | dense-decode | 6 | f32 | 3.5674 [3.5522-3.5803] | 3.4860 [3.4786-3.5020] | 97.7 | 189 / 194 | faster |
| lm_head | Q8 | dense-decode | 7 | f32 | 3.6924 [3.6562-3.7229] | 3.6608 [3.6474-3.6700] | 99.2 | 183 / 184 | noise |
| lm_head | Q8 | dense-decode | 8 | f32 | 3.8166 [3.7918-3.8475] | 3.8828 [3.8647-3.8904] | 101.7 | 177 / 174 | slower |
| lm_head | Q8 | dense-prefill-wmma | 512 | f32 | 22.9739 [22.7900-23.1851] | 22.4165 [22.1753-22.5681] | 97.6 | 29 / 30 | faster |
| lm_head | Q4 syn | dense-decode | 1 | bf16 | 1.8544 [1.8343-1.8975] | 1.8331 [1.8147-1.8595] | 98.8 | 193 / 195 | noise |
| lm_head | Q4 syn | dense-decode | 2 | bf16 | 1.8675 [1.8635-1.8813] | 1.8441 [1.8375-1.8522] | 98.8 | 191 / 194 | faster |
| lm_head | Q4 syn | dense-decode | 3 | bf16 | 2.0644 [2.0555-2.0785] | 1.9361 [1.9225-1.9421] | 93.8 | 173 / 185 | faster |
| lm_head | Q4 syn | dense-decode | 4 | bf16 | 2.2075 [2.2008-2.2153] | 2.0642 [2.0567-2.0723] | 93.5 | 162 / 173 | faster |
| lm_head | Q4 syn | dense-decode | 5 | bf16 | 2.4530 [2.4454-2.4581] | 2.3122 [2.3057-2.3264] | 94.3 | 146 / 155 | faster |
| lm_head | Q4 syn | dense-decode | 6 | bf16 | 2.5848 [2.5816-2.5883] | 2.4392 [2.4339-2.4479] | 94.4 | 138 / 147 | faster |
| lm_head | Q4 syn | dense-decode | 7 | bf16 | 2.7337 [2.7167-2.7428] | 2.5497 [2.5431-2.5711] | 93.3 | 131 / 140 | faster |
| lm_head | Q4 syn | dense-decode | 8 | bf16 | 2.8916 [2.8873-2.9005] | 2.6877 [2.6831-2.7037] | 93.0 | 124 / 133 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | bf16 | 22.1703 [21.7898-22.6311] | 21.7981 [21.5605-22.1597] | 98.3 | 16 / 16 | noise |
| lm_head | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 89.4805 [88.5907-90.1213] | 86.6493 [86.3297-87.4946] | 96.8 | 4 / 4 | faster |
| lm_head | Q4 syn | dense-decode | 1 | f32 | 1.8187 [1.8154-1.8229] | 1.8013 [1.7964-1.8049] | 99.0 | 197 / 199 | faster |
| lm_head | Q4 syn | dense-decode | 2 | f32 | 1.8694 [1.8642-1.8752] | 1.8286 [1.8209-1.8323] | 97.8 | 191 / 196 | faster |
| lm_head | Q4 syn | dense-decode | 3 | f32 | 2.0763 [2.0709-2.0846] | 1.9458 [1.9380-1.9499] | 93.7 | 172 / 184 | faster |
| lm_head | Q4 syn | dense-decode | 4 | f32 | 2.1955 [2.1897-2.2073] | 2.0495 [2.0448-2.0571] | 93.3 | 163 / 174 | faster |
| lm_head | Q4 syn | dense-decode | 5 | f32 | 2.4316 [2.4213-2.4386] | 2.2883 [2.2730-2.3001] | 94.1 | 147 / 156 | faster |
| lm_head | Q4 syn | dense-decode | 6 | f32 | 2.5208 [2.5162-2.5303] | 2.4002 [2.3929-2.4084] | 95.2 | 142 / 149 | faster |
| lm_head | Q4 syn | dense-decode | 7 | f32 | 2.6652 [2.6589-2.6789] | 2.5386 [2.5332-2.5439] | 95.2 | 134 / 141 | faster |
| lm_head | Q4 syn | dense-decode | 8 | f32 | 2.8689 [2.8522-2.8826] | 2.7062 [2.6842-2.7137] | 94.3 | 125 / 132 | faster |
| lm_head | Q4 syn | dense-prefill-wmma | 512 | f32 | 21.7247 [21.5529-21.8777] | 21.3658 [21.0661-21.4328] | 98.3 | 16 / 17 | faster |
| mtp_fc | Q8 | dense-decode | 1 | bf16 | 0.0827 [0.0821-0.0832] | 0.0835 [0.0830-0.0838] | 101.0 | 84 / 83 | noise |
| mtp_fc | Q8 | dense-decode | 2 | bf16 | 0.0851 [0.0843-0.0853] | 0.0840 [0.0838-0.0843] | 98.7 | 82 / 83 | noise |
| mtp_fc | Q8 | dense-decode | 3 | bf16 | 0.0884 [0.0876-0.0889] | 0.0864 [0.0857-0.0873] | 97.7 | 79 / 81 | faster |
| mtp_fc | Q8 | dense-decode | 4 | bf16 | 0.0921 [0.0913-0.0926] | 0.0901 [0.0888-0.0904] | 97.8 | 76 / 77 | faster |
| mtp_fc | Q8 | dense-decode | 5 | bf16 | 0.0957 [0.0941-0.0965] | 0.0933 [0.0925-0.0942] | 97.5 | 73 / 75 | noise |
| mtp_fc | Q8 | dense-decode | 6 | bf16 | 0.0976 [0.0967-0.0981] | 0.0964 [0.0959-0.0970] | 98.8 | 71 / 72 | noise |
| mtp_fc | Q8 | dense-decode | 7 | bf16 | 0.0986 [0.0980-0.0989] | 0.0978 [0.0970-0.0988] | 99.2 | 71 / 71 | noise |
| mtp_fc | Q8 | dense-decode | 8 | bf16 | 0.1024 [0.1019-0.1029] | 0.1011 [0.0996-0.1018] | 98.7 | 68 / 69 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | bf16 | 0.2965 [0.2944-0.2979] | 0.2288 [0.2272-0.2316] | 77.2 | 23 / 30 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | bf16 | 0.8475 [0.8296-0.8606] | 0.7783 [0.7716-0.7901] | 91.8 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | bf16 | 3.3998 [3.1821-3.5600] | 3.2794 [3.1104-3.4918] | 96.5 | 2 / 2 | noise |
| mtp_fc | Q8 | dense-decode | 1 | f32 | 0.0820 [0.0818-0.0824] | 0.0829 [0.0824-0.0836] | 101.1 | 85 / 84 | slower |
| mtp_fc | Q8 | dense-decode | 2 | f32 | 0.0837 [0.0834-0.0844] | 0.0841 [0.0838-0.0844] | 100.4 | 83 / 83 | noise |
| mtp_fc | Q8 | dense-decode | 3 | f32 | 0.0867 [0.0864-0.0876] | 0.0861 [0.0856-0.0871] | 99.3 | 80 / 81 | noise |
| mtp_fc | Q8 | dense-decode | 4 | f32 | 0.0902 [0.0895-0.0912] | 0.0892 [0.0887-0.0900] | 99.0 | 77 / 78 | noise |
| mtp_fc | Q8 | dense-decode | 5 | f32 | 0.0935 [0.0932-0.0942] | 0.0937 [0.0919-0.0943] | 100.2 | 74 / 74 | noise |
| mtp_fc | Q8 | dense-decode | 6 | f32 | 0.0964 [0.0957-0.0971] | 0.0963 [0.0955-0.0972] | 99.9 | 72 / 72 | noise |
| mtp_fc | Q8 | dense-decode | 7 | f32 | 0.0985 [0.0981-0.0991] | 0.0977 [0.0970-0.0980] | 99.2 | 71 / 71 | faster |
| mtp_fc | Q8 | dense-decode | 8 | f32 | 0.1004 [0.0996-0.1011] | 0.0998 [0.0993-0.1011] | 99.4 | 69 / 70 | noise |
| mtp_fc | Q8 | dense-prefill-wmma | 512 | f32 | 0.2971 [0.2956-0.2978] | 0.2332 [0.2316-0.2338] | 78.5 | 23 / 30 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 2048 | f32 | 0.8366 [0.8274-0.8455] | 0.7823 [0.7702-0.7902] | 93.5 | 8 / 9 | faster |
| mtp_fc | Q8 | dense-prefill-wmma | 8192 | f32 | 3.3563 [3.2194-3.5009] | 3.2832 [3.1201-3.3696] | 97.8 | 2 / 2 | noise |
| mtp_fc | Q4 syn | dense-decode | 1 | bf16 | 0.0450 [0.0446-0.0452] | 0.0446 [0.0442-0.0448] | 99.0 | 82 / 83 | noise |
| mtp_fc | Q4 syn | dense-decode | 2 | bf16 | 0.0493 [0.0484-0.0502] | 0.0461 [0.0457-0.0466] | 93.5 | 75 / 80 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | bf16 | 0.0512 [0.0505-0.0527] | 0.0475 [0.0471-0.0480] | 92.7 | 72 / 78 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | bf16 | 0.0526 [0.0521-0.0547] | 0.0494 [0.0492-0.0499] | 94.1 | 70 / 75 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | bf16 | 0.0567 [0.0558-0.0576] | 0.0536 [0.0528-0.0541] | 94.5 | 65 / 69 | faster |
| mtp_fc | Q4 syn | dense-decode | 6 | bf16 | 0.0584 [0.0576-0.0590] | 0.0549 [0.0545-0.0556] | 94.0 | 63 / 67 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | bf16 | 0.0606 [0.0590-0.0617] | 0.0560 [0.0558-0.0569] | 92.5 | 61 / 66 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | bf16 | 0.0626 [0.0618-0.0633] | 0.0584 [0.0574-0.0587] | 93.3 | 59 / 63 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | bf16 | 0.2332 [0.2320-0.2369] | 0.2168 [0.2158-0.2175] | 93.0 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | bf16 | 0.7888 [0.7811-0.7972] | 0.7693 [0.7651-0.7752] | 97.5 | 5 / 5 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | bf16 | 3.1455 [3.0481-3.4984] | 3.1335 [3.0072-3.3869] | 99.6 | 1 / 1 | noise |
| mtp_fc | Q4 syn | dense-decode | 1 | f32 | 0.0462 [0.0456-0.0471] | 0.0453 [0.0448-0.0456] | 98.0 | 80 / 81 | faster |
| mtp_fc | Q4 syn | dense-decode | 2 | f32 | 0.0482 [0.0473-0.0489] | 0.0460 [0.0458-0.0463] | 95.3 | 76 / 80 | faster |
| mtp_fc | Q4 syn | dense-decode | 3 | f32 | 0.0496 [0.0486-0.0504] | 0.0472 [0.0469-0.0475] | 95.2 | 74 / 78 | faster |
| mtp_fc | Q4 syn | dense-decode | 4 | f32 | 0.0514 [0.0507-0.0527] | 0.0488 [0.0484-0.0490] | 94.9 | 72 / 76 | faster |
| mtp_fc | Q4 syn | dense-decode | 5 | f32 | 0.0540 [0.0533-0.0546] | 0.0529 [0.0525-0.0534] | 98.0 | 68 / 70 | noise |
| mtp_fc | Q4 syn | dense-decode | 6 | f32 | 0.0561 [0.0556-0.0574] | 0.0541 [0.0538-0.0546] | 96.5 | 66 / 68 | faster |
| mtp_fc | Q4 syn | dense-decode | 7 | f32 | 0.0588 [0.0575-0.0592] | 0.0560 [0.0554-0.0566] | 95.2 | 63 / 66 | faster |
| mtp_fc | Q4 syn | dense-decode | 8 | f32 | 0.0606 [0.0599-0.0616] | 0.0565 [0.0564-0.0575] | 93.3 | 61 / 65 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 512 | f32 | 0.2356 [0.2340-0.2384] | 0.2190 [0.2186-0.2198] | 93.0 | 16 / 17 | faster |
| mtp_fc | Q4 syn | dense-prefill-wmma | 2048 | f32 | 0.7819 [0.7770-0.7958] | 0.7714 [0.7694-0.7834] | 98.7 | 5 / 5 | noise |
| mtp_fc | Q4 syn | dense-prefill-wmma | 8192 | f32 | 3.2361 [3.0462-3.4762] | 3.1821 [3.0300-3.3488] | 98.3 | 1 / 1 | noise |
| exp_gu | Q4 | expert-decode-gather-swiglu | 1 | bf16 | 0.1949 [0.1926-0.1966] | 0.1936 [0.1920-0.1951] | 99.3 | 89 / 90 | noise |
| exp_gu | Q4 | expert-decode-gather | 1 | bf16 | 0.1879 [0.1863-0.1892] | 0.1860 [0.1855-0.1871] | 99.0 | 93 / 94 | noise |
| exp_down | Q4 | expert-decode-combine | 1 | bf16 | 0.1245 [0.1240-0.1251] | 0.1195 [0.1190-0.1201] | 96.0 | 82 / 86 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 2 | bf16 | 0.3380 [0.3359-0.3394] | 0.3270 [0.3249-0.3290] | 96.8 | 98 / 101 | faster |
| exp_gu | Q4 | expert-decode-gather | 2 | bf16 | 0.3389 [0.3374-0.3404] | 0.3287 [0.3264-0.3307] | 97.0 | 98 / 101 | faster |
| exp_down | Q4 | expert-decode-combine | 2 | bf16 | 0.2243 [0.2235-0.2255] | 0.2144 [0.2136-0.2157] | 95.6 | 87 / 91 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 3 | bf16 | 0.4206 [0.4159-0.4256] | 0.4189 [0.4158-0.4249] | 99.6 | 108 / 108 | noise |
| exp_gu | Q4 | expert-decode-gather | 3 | bf16 | 0.4173 [0.4161-0.4202] | 0.4132 [0.4116-0.4153] | 99.0 | 108 / 110 | faster |
| exp_down | Q4 | expert-decode-combine | 3 | bf16 | 0.3028 [0.3021-0.3043] | 0.2909 [0.2890-0.2924] | 96.1 | 88 / 92 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 4 | bf16 | 0.4914 [0.4895-0.4948] | 0.4847 [0.4818-0.4881] | 98.6 | 120 / 122 | faster |
| exp_gu | Q4 | expert-decode-gather | 4 | bf16 | 0.4929 [0.4907-0.4973] | 0.4884 [0.4857-0.4921] | 99.1 | 120 / 121 | noise |
| exp_down | Q4 | expert-decode-combine | 4 | bf16 | 0.3792 [0.3778-0.3806] | 0.3704 [0.3674-0.3748] | 97.7 | 92 / 94 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 5 | bf16 | 0.5179 [0.5148-0.5217] | 0.5073 [0.5054-0.5095] | 98.0 | 118 / 120 | faster |
| exp_gu | Q4 | expert-decode-gather | 5 | bf16 | 0.5298 [0.5251-0.5319] | 0.5167 [0.5159-0.5184] | 97.5 | 115 / 118 | faster |
| exp_down | Q4 | expert-decode-combine | 5 | bf16 | 0.3808 [0.3785-0.3825] | 0.3673 [0.3655-0.3694] | 96.5 | 94 / 98 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 6 | bf16 | 0.5904 [0.5858-0.5947] | 0.5915 [0.5818-0.5954] | 100.2 | 127 / 127 | noise |
| exp_gu | Q4 | expert-decode-gather | 6 | bf16 | 0.6052 [0.6037-0.6084] | 0.5943 [0.5924-0.5969] | 98.2 | 124 / 126 | faster |
| exp_down | Q4 | expert-decode-combine | 6 | bf16 | 0.4348 [0.4340-0.4371] | 0.4141 [0.4124-0.4160] | 95.2 | 101 / 106 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 7 | bf16 | 0.6441 [0.6421-0.6492] | 0.6315 [0.6265-0.6341] | 98.1 | 127 / 130 | faster |
| exp_gu | Q4 | expert-decode-gather | 7 | bf16 | 0.6599 [0.6591-0.6633] | 0.6434 [0.6398-0.6460] | 97.5 | 124 / 127 | faster |
| exp_down | Q4 | expert-decode-combine | 7 | bf16 | 0.4602 [0.4580-0.4617] | 0.4386 [0.4370-0.4426] | 95.3 | 105 / 110 | faster |
| exp_gu | Q4 | expert-decode-gather-swiglu | 8 | bf16 | 0.7336 [0.7314-0.7353] | 0.7272 [0.7232-0.7281] | 99.1 | 123 / 124 | faster |
| exp_gu | Q4 | expert-decode-gather | 8 | bf16 | 0.7456 [0.7438-0.7481] | 0.7386 [0.7359-0.7403] | 99.1 | 121 / 123 | faster |
| exp_down | Q4 | expert-decode-combine | 8 | bf16 | 0.5011 [0.4995-0.5024] | 0.4793 [0.4774-0.4813] | 95.6 | 106 / 111 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 512 | bf16 | 4.6717 [4.5293-4.7131] | 4.3495 [4.3032-4.3996] | 93.1 | 170 / 183 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 512 | bf16 | 4.7123 [4.7067-4.7356] | 4.4149 [4.3962-4.4599] | 93.7 | 169 / 180 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 512 | bf16 | 4.4211 [4.4167-4.4467] | 4.3945 [4.3799-4.4086] | 99.4 | 180 / 181 | faster |
| exp_down | Q4 | expert-prefill-combine narrow | 512 | bf16 | 3.1010 [3.0884-3.1125] | 2.8313 [2.8164-2.8428] | 91.3 | 151 / 165 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 512 | bf16 | 3.0791 [3.0696-3.0911] | 2.8286 [2.8202-2.8838] | 91.9 | 152 / 165 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 512 | bf16 | 3.1095 [3.0968-3.1243] | 2.8554 [2.8432-2.8713] | 91.8 | 150 / 164 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 512 | bf16 | 5.8790 [5.8644-5.8906] | 5.8316 [5.8182-5.8445] | 99.2 | 135 / 136 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 512 | bf16 | 5.9073 [5.8974-5.9126] | 5.8573 [5.8444-5.8778] | 99.2 | 135 / 136 | faster |
| exp_gu | Q4 | expert-prefill-gather wide | 512 | bf16 | 5.8726 [5.8607-5.8836] | 5.7640 [5.7520-5.7732] | 98.2 | 135 / 138 | faster |
| exp_down | Q4 | expert-prefill-combine wide | 512 | bf16 | 3.6259 [3.6124-3.6378] | 3.2744 [3.2660-3.2929] | 90.3 | 129 / 143 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 512 | bf16 | 3.4820 [3.4707-3.4895] | 3.1286 [3.1235-3.1356] | 89.8 | 134 / 150 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 512 | bf16 | 3.5072 [3.4991-3.5156] | 3.1525 [3.1462-3.1558] | 89.9 | 133 / 148 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 512 | bf16 | 5.7234 [5.6978-5.7511] | 5.2841 [5.2526-5.3355] | 92.3 | 139 / 151 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 512 | bf16 | 5.7129 [5.6975-5.7279] | 5.2779 [5.2484-5.3063] | 92.4 | 139 / 151 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 512 | bf16 | 5.8231 [5.8042-5.8458] | 5.4046 [5.3678-5.4329] | 92.8 | 137 / 147 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 512 | bf16 | 4.3303 [4.3060-4.3549] | 2.9928 [2.9803-3.0115] | 69.1 | 108 / 156 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 512 | bf16 | 4.3887 [4.3583-4.4056] | 2.9717 [2.9603-2.9757] | 67.7 | 107 / 157 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 512 | bf16 | 4.4100 [4.3887-4.4262] | 2.9718 [2.9620-2.9742] | 67.4 | 106 / 157 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 2048 | bf16 | 8.4278 [7.9719-8.4686] | 7.9023 [7.4789-7.9429] | 93.8 | 106 / 113 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 2048 | bf16 | 8.4919 [8.4278-8.5437] | 7.9670 [7.9372-8.0257] | 93.8 | 105 / 112 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 2048 | bf16 | 7.8035 [7.7831-7.9242] | 7.8236 [7.8155-7.9002] | 100.3 | 114 / 114 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 2048 | bf16 | 5.5060 [5.4752-5.5251] | 5.1245 [5.0921-5.1398] | 93.1 | 95 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 2048 | bf16 | 5.4747 [5.4227-5.4913] | 5.0896 [5.0751-5.1257] | 93.0 | 96 / 103 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 2048 | bf16 | 5.5029 [5.4926-5.5321] | 5.1452 [5.1319-5.1734] | 93.5 | 95 / 102 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide | 2048 | bf16 | 9.3214 [9.2604-9.3442] | 9.2923 [9.2090-9.3218] | 99.7 | 95 / 96 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 2048 | bf16 | 9.2393 [9.1839-9.3392] | 9.2105 [9.1779-9.2760] | 99.7 | 96 / 97 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 2048 | bf16 | 9.3055 [9.2373-9.4159] | 9.2067 [9.1552-9.3127] | 98.9 | 96 / 97 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 2048 | bf16 | 5.7972 [5.7803-5.8102] | 5.3441 [5.3098-5.3809] | 92.2 | 90 / 98 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 2048 | bf16 | 5.6456 [5.6254-5.6557] | 5.1597 [5.1455-5.1656] | 91.4 | 93 / 101 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 2048 | bf16 | 5.6913 [5.6696-5.7129] | 5.2152 [5.2088-5.2279] | 91.6 | 92 / 100 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 2048 | bf16 | 8.7417 [8.6746-8.8630] | 8.3976 [8.2486-8.4922] | 96.1 | 102 / 106 | faster |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 2048 | bf16 | 8.7472 [8.7097-8.8060] | 8.3516 [8.2639-8.4583] | 95.5 | 102 / 107 | faster |
| exp_gu | Q4 | expert-prefill-gather wide256 | 2048 | bf16 | 8.8962 [8.8463-8.9214] | 8.3815 [8.3635-8.4139] | 94.2 | 100 / 106 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 2048 | bf16 | 6.1635 [6.1224-6.2119] | 5.1190 [5.0669-5.1637] | 83.0 | 85 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 2048 | bf16 | 6.1902 [6.1537-6.2277] | 5.1264 [5.1081-5.1478] | 82.8 | 85 / 102 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 2048 | bf16 | 6.2032 [6.1788-6.2305] | 5.1313 [5.1137-5.1448] | 82.7 | 84 / 102 | faster |
| exp_gu | Q4 | expert-prefill-swiglu narrow | 8192 | bf16 | 23.5557 [23.1748-80.4293] | 22.6766 [22.2355-76.5831] | 96.3 | 38 / 39 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h narrow | 8192 | bf16 | 81.4979 [81.3968-81.5918] | 78.1800 [77.9371-78.2997] | 95.9 | 11 / 11 | faster |
| exp_gu | Q4 | expert-prefill-gather narrow | 8192 | bf16 | 23.7726 [23.6800-24.0664] | 23.8768 [23.7572-24.0845] | 100.4 | 37 / 37 | noise |
| exp_down | Q4 | expert-prefill-combine narrow | 8192 | bf16 | 15.9027 [15.6729-16.1411] | 15.3222 [15.1113-15.5877] | 96.3 | 33 / 34 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h narrow | 8192 | bf16 | 15.3907 [15.3365-15.7782] | 14.9523 [14.8969-15.1581] | 97.2 | 34 / 35 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared narrow | 8192 | bf16 | 15.2341 [14.1479-15.3934] | 14.6358 [13.6929-14.8379] | 96.1 | 34 / 36 | noise |
| exp_gu | Q4 | expert-prefill-swiglu wide | 8192 | bf16 | 22.1165 [21.8742-22.1847] | 22.1728 [21.8099-22.2216] | 100.2 | 40 / 40 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide | 8192 | bf16 | 22.2623 [22.1727-22.4041] | 22.2507 [22.0821-22.3328] | 100.0 | 40 / 40 | noise |
| exp_gu | Q4 | expert-prefill-gather wide | 8192 | bf16 | 21.5611 [21.4713-22.1773] | 21.4770 [21.3898-21.8179] | 99.6 | 41 / 42 | noise |
| exp_down | Q4 | expert-prefill-combine wide | 8192 | bf16 | 14.2440 [14.0903-14.4613] | 13.6308 [13.4865-13.8666] | 95.7 | 37 / 38 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide | 8192 | bf16 | 13.9338 [13.8765-14.1090] | 13.2134 [13.1600-13.3702] | 94.8 | 38 / 40 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide | 8192 | bf16 | 13.8999 [13.8556-13.9509] | 13.1746 [13.0987-13.2329] | 94.8 | 38 / 40 | faster |
| exp_gu | Q4 | expert-prefill-swiglu wide256 | 8192 | bf16 | 20.6617 [20.1054-21.7150] | 19.9556 [19.6419-21.1364] | 96.6 | 43 / 45 | noise |
| exp_gu | Q4 | expert-prefill-swiglu-f16h wide256 | 8192 | bf16 | 20.7353 [20.5008-21.2153] | 20.2874 [19.8192-20.6960] | 97.8 | 43 / 44 | noise |
| exp_gu | Q4 | expert-prefill-gather wide256 | 8192 | bf16 | 21.5443 [21.0620-21.8478] | 20.4788 [19.9321-20.9293] | 95.0 | 41 / 44 | faster |
| exp_down | Q4 | expert-prefill-combine wide256 | 8192 | bf16 | 15.3558 [15.2715-15.4715] | 13.7146 [13.6638-13.7761] | 89.3 | 34 / 38 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h wide256 | 8192 | bf16 | 15.4303 [15.3609-15.5759] | 13.9643 [13.8313-14.0624] | 90.5 | 34 / 38 | faster |
| exp_down | Q4 | expert-prefill-combine-f16h-shared wide256 | 8192 | bf16 | 15.6372 [15.5355-15.7868] | 14.0515 [13.9709-14.2942] | 89.9 | 34 / 37 | faster |
