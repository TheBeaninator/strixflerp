# SHAPE-TRANSFERS R1 fold: arm a in kernel.flerp.moe_kquant (Q4_K, and Q6_K by the same recipe)

Criteria `gate/xfer-r1-fold.toml` (committed before the GPU runs). Host control-node.

## Re-measure of the erratic R1 rows (xfer-r1-splitk, 3 repeats)

Load: all runs at or under 6 (1-min load 5.26..5.48).

| case | tokens | ref GB/s (median, min..max) | arm a GB/s | a / ref | a bit-equal |
|---|---|---|---|---|---|
| ornith-down | 4 | 116.2 (112.2..119.8) | 156.1 (153.4..157.4) | 1.343 | True |
| ornith-gateup | 4 | 152.3 (144.5..154.3) | 165.6 (159.6..167.9) | 1.087 | True |
| q35-gateup | 2 | 158.9 (146.8..159.1) | 186.0 (174.8..187.3) | 1.171 | True |
| q35-gateup | 4 | 167.7 (153.1..171.0) | 193.4 (182.4..197.5) | 1.153 | True |
| q35-gateup | 8 | 163.8 (153.1..167.7) | 187.6 (177.4..190.2) | 1.145 | True |

## Kernel (kernel) - byte equality final; its GB/s are NOT usable (single 200-launch loops, clock states: see the toml's [kernel_rounds])

96 (case, tokens) results, 170131968 outputs; differing bytes in 0 of them -> **PASS** (byte equality). Runs off-load or failed: 0 

| case | type | tokens | outputs differing | ref GB/s | new GB/s | new / ref |
|---|---|---|---|---|---|---|
| q35-gateup | q4k | 1 | 0 of 4096 | 56.5 | 3.6 | 0.064 |
| q35-gateup | q4k | 2 | 0 of 8192 | 150.2 | 175.3 | 1.168 |
| q35-gateup | q4k | 4 | 0 of 16384 | 152.7 | 179.6 | 1.176 |
| q35-gateup | q4k | 8 | 0 of 32768 | 149.6 | 173.0 | 1.156 |
| q35-down-q6 | q6k | 1 | 0 of 16384 | 140.1 | 198.8 | 1.419 |
| q35-down-q6 | q6k | 2 | 0 of 32768 | 134.8 | 192.9 | 1.431 |
| q35-down-q6 | q6k | 4 | 0 of 65536 | 136.3 | 200.6 | 1.472 |
| q35-down-q6 | q6k | 8 | 0 of 131072 | 132.2 | 192.4 | 1.456 |
| q35-down-q4 | q4k | 1 | 0 of 16384 | 104.7 | 42.8 | 0.409 |
| q35-down-q4 | q4k | 2 | 0 of 32768 | 12.4 | 153.2 | 12.394 |
| q35-down-q4 | q4k | 4 | 0 of 65536 | 33.5 | 150.9 | 4.499 |
| q35-down-q4 | q4k | 8 | 0 of 131072 | 100.3 | 144.9 | 1.445 |
| q35-sh-gateup | q4k | 1 | 0 of 512 | 161.9 | 206.3 | 1.274 |
| q35-sh-gateup | q4k | 2 | 0 of 1024 | 118.1 | 141.8 | 1.201 |
| q35-sh-gateup | q4k | 4 | 0 of 2048 | 51.7 | 73.6 | 1.422 |
| q35-sh-gateup | q4k | 8 | 0 of 4096 | 29.1 | 38.9 | 1.335 |
| q35-sh-down-q6 | q6k | 1 | 0 of 2048 | 142.9 | 188.7 | 1.321 |
| q35-sh-down-q6 | q6k | 2 | 0 of 4096 | 92.5 | 128.5 | 1.390 |
| q35-sh-down-q6 | q6k | 4 | 0 of 8192 | 50.1 | 68.5 | 1.368 |
| q35-sh-down-q6 | q6k | 8 | 0 of 16384 | 24.7 | 37.2 | 1.506 |
| q35-sh-down-q4 | q4k | 1 | 0 of 2048 | 16.7 | 133.4 | 7.971 |
| q35-sh-down-q4 | q4k | 2 | 0 of 4096 | 11.2 | 92.5 | 8.272 |
| q35-sh-down-q4 | q4k | 4 | 0 of 8192 | 36.2 | 50.4 | 1.391 |
| q35-sh-down-q4 | q4k | 8 | 0 of 16384 | 10.3 | 31.1 | 3.023 |
| ornith-gateup | q4k | 1 | 0 of 10240 | 13.1 | 24.2 | 1.844 |
| ornith-gateup | q4k | 2 | 0 of 20480 | 164.5 | 181.8 | 1.105 |
| ornith-gateup | q4k | 4 | 0 of 40960 | 142.5 | 63.6 | 0.446 |
| ornith-gateup | q4k | 8 | 0 of 81920 | 112.0 | 120.4 | 1.075 |
| ornith-down-q6 | q6k | 1 | 0 of 40960 | 186.2 | 215.2 | 1.156 |
| ornith-down-q6 | q6k | 2 | 0 of 81920 | 173.6 | 195.8 | 1.127 |
| ornith-down-q6 | q6k | 4 | 0 of 163840 | 152.4 | 77.1 | 0.506 |
| ornith-down-q6 | q6k | 8 | 0 of 327680 | 67.6 | 132.2 | 1.955 |
| ornith-down-q4 | q4k | 1 | 0 of 40960 | 125.4 | 174.1 | 1.388 |
| ornith-down-q4 | q4k | 2 | 0 of 81920 | 137.0 | 172.5 | 1.259 |
| ornith-down-q4 | q4k | 4 | 0 of 163840 | 113.0 | 145.7 | 1.289 |
| ornith-down-q4 | q4k | 8 | 0 of 327680 | 47.6 | 107.8 | 2.265 |
| ornith-sh-gateup | q4k | 1 | 0 of 1024 | 208.1 | 291.1 | 1.398 |
| ornith-sh-gateup | q4k | 2 | 0 of 2048 | 99.4 | 141.0 | 1.418 |
| ornith-sh-gateup | q4k | 4 | 0 of 4096 | 56.1 | 78.2 | 1.395 |
| ornith-sh-gateup | q4k | 8 | 0 of 8192 | 32.5 | 46.9 | 1.442 |
| ornith-sh-down-q6 | q6k | 1 | 0 of 4096 | 43.7 | 84.4 | 1.930 |
| ornith-sh-down-q6 | q6k | 2 | 0 of 8192 | 42.8 | 28.4 | 0.664 |
| ornith-sh-down-q6 | q6k | 4 | 0 of 16384 | 6.8 | 4.9 | 0.716 |
| ornith-sh-down-q6 | q6k | 8 | 0 of 32768 | 31.1 | 40.2 | 1.291 |
| ornith-sh-down-q4 | q4k | 1 | 0 of 4096 | 165.0 | 233.1 | 1.413 |
| ornith-sh-down-q4 | q4k | 2 | 0 of 8192 | 85.2 | 124.1 | 1.457 |
| ornith-sh-down-q4 | q4k | 4 | 0 of 16384 | 44.7 | 64.8 | 1.449 |
| ornith-sh-down-q4 | q4k | 8 | 0 of 32768 | 22.7 | 32.3 | 1.426 |
Byte equality was checked at every tokens value of the run ([1, 2, 3, 4, 8, 21, 128, 1024]); rows without timing are not listed.

## Kernel (kernel-rounds) - 7 alternating rounds a case (the Ornith cases here ran off-load and were rerun below)

48 (case, tokens) results, 2142720 outputs; differing bytes in 0 of them -> **PASS** (byte equality). Runs off-load or failed: 5 [{'case': 'ornith-down-q6', 'rc': 0, 'load_start': 5.75, 'load_end': 6.55}, {'case': 'ornith-down-q4', 'rc': 0, 'load_start': 6.55, 'load_end': 6.77}, {'case': 'ornith-sh-gateup', 'rc': 0, 'load_start': 6.77, 'load_end': 6.77}, {'case': 'ornith-sh-down-q6', 'rc': 0, 'load_start': 6.77, 'load_end': 6.77}, {'case': 'ornith-sh-down-q4', 'rc': 0, 'load_start': 6.77, 'load_end': 6.77}]

| case | type | tokens | outputs differing | ref GB/s | new GB/s | new / ref |
|---|---|---|---|---|---|---|
| q35-gateup | q4k | 1 | 0 of 4096 | 130.9 (102.4..133.4) | 164.2 (163.6..165.5) | 1.254 |
| q35-gateup | q4k | 2 | 0 of 8192 | 147.0 (95.0..147.8) | 170.0 (168.4..171.9) | 1.156 |
| q35-gateup | q4k | 4 | 0 of 16384 | 155.7 (127.1..160.5) | 177.3 (147.0..184.1) | 1.139 |
| q35-gateup | q4k | 8 | 0 of 32768 | 155.2 (152.3..156.0) | 177.9 (173.1..179.0) | 1.146 |
| q35-down-q6 | q6k | 1 | 0 of 16384 | 138.9 (135.0..142.9) | 200.1 (195.9..200.7) | 1.441 |
| q35-down-q6 | q6k | 2 | 0 of 32768 | 137.9 (11.1..141.9) | 198.0 (195.2..202.8) | 1.436 |
| q35-down-q6 | q6k | 4 | 0 of 65536 | 141.8 (121.5..144.5) | 200.8 (190.1..205.7) | 1.416 |
| q35-down-q6 | q6k | 8 | 0 of 131072 | 132.6 (87.3..141.3) | 193.1 (152.6..197.1) | 1.456 |
| q35-down-q4 | q4k | 1 | 0 of 16384 | 101.1 (84.8..104.5) | 144.2 (135.0..153.7) | 1.427 |
| q35-down-q4 | q4k | 2 | 0 of 32768 | 101.3 (59.4..108.0) | 158.3 (153.8..163.2) | 1.562 |
| q35-down-q4 | q4k | 4 | 0 of 65536 | 102.8 (101.2..104.6) | 151.2 (120.1..153.5) | 1.470 |
| q35-down-q4 | q4k | 8 | 0 of 131072 | 82.6 (23.0..108.5) | 129.1 (28.4..147.9) | 1.562 |
| q35-sh-gateup | q4k | 1 | 0 of 512 | 144.8 (142.5..159.2) | 187.8 (182.2..201.1) | 1.297 |
| q35-sh-gateup | q4k | 2 | 0 of 1024 | 102.9 (102.2..105.9) | 134.3 (133.6..137.3) | 1.306 |
| q35-sh-gateup | q4k | 4 | 0 of 2048 | 53.9 (52.3..57.1) | 76.0 (74.3..84.3) | 1.409 |
| q35-sh-gateup | q4k | 8 | 0 of 4096 | 32.3 (18.7..33.3) | 41.0 (18.2..46.9) | 1.267 |
| q35-sh-down-q6 | q6k | 1 | 0 of 2048 | 120.7 (115.1..124.1) | 161.2 (156.6..165.4) | 1.335 |
| q35-sh-down-q6 | q6k | 2 | 0 of 4096 | 83.4 (80.6..88.2) | 117.2 (115.3..122.8) | 1.406 |
| q35-sh-down-q6 | q6k | 4 | 0 of 8192 | 50.0 (48.3..51.2) | 71.3 (70.0..73.5) | 1.425 |
| q35-sh-down-q6 | q6k | 8 | 0 of 16384 | 27.8 (25.0..28.7) | 39.7 (36.5..41.2) | 1.430 |
| q35-sh-down-q4 | q4k | 1 | 0 of 2048 | 84.0 (83.5..84.2) | 116.5 (116.3..116.8) | 1.388 |
| q35-sh-down-q4 | q4k | 2 | 0 of 4096 | 55.6 (54.9..57.9) | 82.6 (81.5..86.4) | 1.487 |
| q35-sh-down-q4 | q4k | 4 | 0 of 8192 | 34.5 (32.6..36.1) | 53.1 (50.7..54.3) | 1.539 |
| q35-sh-down-q4 | q4k | 8 | 0 of 16384 | 19.6 (7.2..20.2) | 30.0 (28.8..31.0) | 1.527 |
| ornith-gateup | q4k | 1 | 0 of 10240 | 175.7 (156.1..180.0) | 196.3 (194.8..197.5) | 1.118 |
| ornith-gateup | q4k | 2 | 0 of 20480 | 148.7 (140.4..166.3) | 168.5 (142.6..185.4) | 1.133 |
| ornith-gateup | q4k | 4 | 0 of 40960 | 145.2 (80.9..146.3) | 160.6 (159.0..161.1) | 1.106 |
| ornith-gateup | q4k | 8 | 0 of 81920 | 111.1 (110.6..111.5) | 121.6 (121.5..122.6) | 1.094 |
| ornith-down-q6 | q6k | 1 | 0 of 40960 | 171.2 (21.6..176.2) | 195.8 (24.9..196.9) | 1.143 |
| ornith-down-q6 | q6k | 2 | 0 of 81920 | 176.5 (8.6..181.7) | 203.4 (151.9..205.9) | 1.152 |
| ornith-down-q6 | q6k | 4 | 0 of 163840 | 153.3 (139.3..157.1) | 176.6 (167.8..178.1) | 1.152 |
| ornith-down-q6 | q6k | 8 | 0 of 327680 | 113.8 (80.4..115.6) | 120.9 (99.8..132.5) | 1.063 |
| ornith-down-q4 | q4k | 1 | 0 of 40960 | 134.9 (110.7..143.1) | 167.9 (22.4..184.3) | 1.244 |
| ornith-down-q4 | q4k | 2 | 0 of 81920 | 127.3 (79.7..129.2) | 161.8 (124.6..164.9) | 1.272 |
| ornith-down-q4 | q4k | 4 | 0 of 163840 | 110.7 (110.0..111.1) | 142.2 (141.4..143.4) | 1.284 |
| ornith-down-q4 | q4k | 8 | 0 of 327680 | 83.9 (83.6..84.4) | 107.1 (106.5..107.9) | 1.277 |
| ornith-sh-gateup | q4k | 1 | 0 of 1024 | 209.0 (197.5..259.3) | 273.3 (262.9..317.4) | 1.308 |
| ornith-sh-gateup | q4k | 2 | 0 of 2048 | 107.2 (95.6..119.4) | 159.0 (132.5..167.2) | 1.483 |
| ornith-sh-gateup | q4k | 4 | 0 of 4096 | 54.0 (53.2..57.4) | 77.9 (75.6..78.3) | 1.443 |
| ornith-sh-gateup | q4k | 8 | 0 of 8192 | 30.3 (20.1..33.3) | 40.7 (39.5..42.0) | 1.341 |
| ornith-sh-down-q6 | q6k | 1 | 0 of 4096 | 213.3 (33.0..268.0) | 288.8 (56.4..336.4) | 1.354 |
| ornith-sh-down-q6 | q6k | 2 | 0 of 8192 | 117.7 (114.6..133.4) | 150.3 (144.5..176.2) | 1.277 |
| ornith-sh-down-q6 | q6k | 4 | 0 of 16384 | 64.8 (57.8..74.0) | 90.3 (73.9..92.8) | 1.393 |
| ornith-sh-down-q6 | q6k | 8 | 0 of 32768 | 29.1 (28.3..29.5) | 36.9 (35.4..37.9) | 1.269 |
| ornith-sh-down-q4 | q4k | 1 | 0 of 4096 | 132.5 (10.7..140.9) | 187.0 (127.5..212.7) | 1.411 |
| ornith-sh-down-q4 | q4k | 2 | 0 of 8192 | 88.2 (81.4..91.7) | 128.8 (125.2..138.1) | 1.461 |
| ornith-sh-down-q4 | q4k | 4 | 0 of 16384 | 43.4 (42.4..45.7) | 64.7 (63.6..66.9) | 1.491 |
| ornith-sh-down-q4 | q4k | 8 | 0 of 32768 | 21.9 (21.6..22.0) | 32.0 (31.2..32.1) | 1.459 |
Byte equality was checked at every tokens value of the run ([1, 2, 4, 8]); rows without timing are not listed.

## Kernel (kernel-rounds-rerun) - the off-load Ornith cases again, 7 rounds

20 (case, tokens) results, 1367040 outputs; differing bytes in 0 of them -> **PASS** (byte equality). Runs off-load or failed: 0 

| case | type | tokens | outputs differing | ref GB/s | new GB/s | new / ref |
|---|---|---|---|---|---|---|
| ornith-down-q6 | q6k | 1 | 0 of 40960 | 183.4 (78.5..184.6) | 211.1 (206.7..213.6) | 1.151 |
| ornith-down-q6 | q6k | 2 | 0 of 81920 | 173.3 (104.8..174.6) | 198.6 (59.1..200.3) | 1.146 |
| ornith-down-q6 | q6k | 4 | 0 of 163840 | 150.7 (148.9..151.1) | 170.7 (166.2..172.5) | 1.133 |
| ornith-down-q6 | q6k | 8 | 0 of 327680 | 110.9 (66.5..112.5) | 129.7 (122.2..130.8) | 1.169 |
| ornith-down-q4 | q4k | 1 | 0 of 40960 | 138.8 (137.1..148.2) | 180.1 (173.9..184.8) | 1.297 |
| ornith-down-q4 | q4k | 2 | 0 of 81920 | 129.5 (127.5..131.5) | 163.6 (118.2..164.9) | 1.263 |
| ornith-down-q4 | q4k | 4 | 0 of 163840 | 112.0 (110.0..113.3) | 143.1 (141.4..145.3) | 1.277 |
| ornith-down-q4 | q4k | 8 | 0 of 327680 | 84.8 (82.9..85.0) | 107.5 (57.6..108.0) | 1.267 |
| ornith-sh-gateup | q4k | 1 | 0 of 1024 | 207.5 (19.2..223.4) | 276.4 (52.4..333.7) | 1.332 |
| ornith-sh-gateup | q4k | 2 | 0 of 2048 | 115.8 (104.8..122.1) | 164.9 (158.9..171.1) | 1.424 |
| ornith-sh-gateup | q4k | 4 | 0 of 4096 | 56.5 (55.8..63.4) | 80.9 (80.4..86.5) | 1.432 |
| ornith-sh-gateup | q4k | 8 | 0 of 8192 | 30.0 (19.1..31.3) | 40.9 (22.7..42.4) | 1.362 |
| ornith-sh-down-q6 | q6k | 1 | 0 of 4096 | 216.5 (208.6..236.3) | 293.8 (281.7..321.0) | 1.357 |
| ornith-sh-down-q6 | q6k | 2 | 0 of 8192 | 138.3 (132.8..145.4) | 176.7 (171.6..184.8) | 1.278 |
| ornith-sh-down-q6 | q6k | 4 | 0 of 16384 | 63.2 (61.1..78.1) | 83.5 (79.7..97.5) | 1.321 |
| ornith-sh-down-q6 | q6k | 8 | 0 of 32768 | 31.7 (31.1..32.1) | 39.9 (39.2..40.6) | 1.258 |
| ornith-sh-down-q4 | q4k | 1 | 0 of 4096 | 113.5 (53.6..129.7) | 162.9 (136.2..180.8) | 1.435 |
| ornith-sh-down-q4 | q4k | 2 | 0 of 8192 | 78.7 (26.7..85.4) | 124.3 (28.0..134.7) | 1.579 |
| ornith-sh-down-q4 | q4k | 4 | 0 of 16384 | 41.6 (34.1..46.3) | 62.8 (61.7..71.9) | 1.511 |
| ornith-sh-down-q4 | q4k | 8 | 0 of 32768 | 22.4 (19.5..23.0) | 32.6 (25.8..33.0) | 1.452 |
Byte equality was checked at every tokens value of the run ([1, 2, 4, 8]); rows without timing are not listed.

## Whole model: 35B, 4 gate prompts, --staged loopback

- run 0 `before` (sha256 01d7391466ec): exit 0, 202.4 s, load 4.67 -> 13.87 (speed INVALID: load above 6); P1/P2 hash failures: 0 
- run 1 `after` (sha256 34dbe92e9e5c): exit 0, 189.1 s, load 5.75 -> 6.28 (speed INVALID: load above 6); P1/P2 hash failures: 0 
- run 2 `before` (sha256 01d7391466ec): exit 0, 195.9 s, load 5.56 -> 4.01; P1/P2 hash failures: 0 
- run 3 `after` (sha256 34dbe92e9e5c): exit 0, 190.0 s, load 4.01 -> 13.50 (speed INVALID: load above 6); P1/P2 hash failures: 0 
- run 4 `after` (sha256 34dbe92e9e5c): exit 0, 186.0 s, load 4.72 -> 5.02; P1/P2 hash failures: 0 
- run 5 `before` (sha256 01d7391466ec): exit 0, 195.9 s, load 5.02 -> 4.20; P1/P2 hash failures: 0 
- run 6 `after` (sha256 34dbe92e9e5c): exit 0, 181.0 s, load 4.20 -> 6.50 (speed INVALID: load above 6); P1/P2 hash failures: 0 
- run 7 `before` (sha256 01d7391466ec): exit 0, 192.4 s, load 4.49 -> 6.09 (speed INVALID: load above 6); P1/P2 hash failures: 0 
- run 8 `after` (sha256 34dbe92e9e5c): exit 0, 183.3 s, load 5.83 -> 6.21 (speed INVALID: load above 6); P1/P2 hash failures: 0 
- run 9 `after` (sha256 34dbe92e9e5c): exit 0, 182.6 s, load 4.44 -> 5.67; P1/P2 hash failures: 0 

Decode tok/s, world 1, graph on (median of the valid world-1 runs, min..max):

| prompt | before | after | after / before |
|---|---|---|---|
| bridge | 55.99 (47.31..57.04) | 59.19 (54.77..61.46) | 1.057 |
| code | 57.69 (56.59..58.50) | 61.40 (57.35..63.35) | 1.064 |
| rivers | 57.40 (56.52..58.74) | 61.98 (61.31..63.20) | 1.080 |
| records | 56.86 (54.82..57.17) | 61.48 (59.66..62.27) | 1.081 |
| all | 56.95 | 61.38 | 1.078 |

