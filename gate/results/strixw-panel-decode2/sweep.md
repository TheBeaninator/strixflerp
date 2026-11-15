# decode2 sweep: panel % of origin time (median of the reps), per wave map

## dense-r16t1

load: load_before 5.38 5.68 6.69 3/1201 585220; load_after 5.24 5.63 6.64 5/1256 602061; 256 / 256 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| attn_o | Q4 | dense-decode | bf16 | 99 100 98 97 99 99 98 97 | 98.7 |
| attn_o | Q4 | dense-decode | f32 | 100 99 99 97 98 98 98 95 | 98.4 |
| attn_o | Q8 | dense-decode | bf16 | 100 99 100 99 99 95 95 94 | 99.4 |
| attn_o | Q8 | dense-decode | f32 | 100 100 101 99 98 97 94 94 | 99.3 |
| attn_qkv | Q4 | dense-decode | bf16 | 99 98 98 97 96 94 94 93 | 97.1 |
| attn_qkv | Q4 | dense-decode | f32 | 99 99 99 97 97 96 95 95 | 97.3 |
| attn_qkv | Q8 | dense-decode | bf16 | 99 100 100 99 97 98 98 99 | 99.1 |
| attn_qkv | Q8 | dense-decode | f32 | 99 102 99 99 97 97 99 99 | 98.8 |
| gdn_in | Q4 | dense-decode | bf16 | 98 98 98 97 95 94 95 93 | 96.9 |
| gdn_in | Q4 | dense-decode | f32 | 99 98 98 97 96 96 96 95 | 97.0 |
| gdn_in | Q8 | dense-decode | bf16 | 100 100 98 98 99 99 101 103 | 100.3 |
| gdn_in | Q8 | dense-decode | f32 | 101 100 100 99 100 100 102 103* | 100.3 |
| gdn_out | Q4 | dense-decode | bf16 | 100 99 98 97 99 98 98 95 | 98.2 |
| gdn_out | Q4 | dense-decode | f32 | 100 99 99 98 100 98 99 96 | 99.2 |
| gdn_out | Q8 | dense-decode | bf16 | 100 100 100 100 100 96 95 94 | 100.0 |
| gdn_out | Q8 | dense-decode | f32 | 100 100 101 99 98 96 94 95 | 99.0 |
| mtp_fc | Q4 | dense-decode | bf16 | 98 93 92 93 92 94 92 91 | 92.6 |
| mtp_fc | Q4 | dense-decode | f32 | 97 95 95 94 95 94 94 93 | 94.7 |
| mtp_fc | Q8 | dense-decode | bf16 | 100 100 98 98 101 101 101 102 | 101.0 |
| mtp_fc | Q8 | dense-decode | f32 | 100 101 100 98 100 100 101 100 | 100.2 |
| ple_kv | Q4 | dense-decode | bf16 | 98 97 98 97 96 95 95 94 | 97.1 |
| ple_kv | Q4 | dense-decode | f32 | 99 98 98 97 96 96 95 95 | 97.1 |
| ple_kv | Q8 | dense-decode | bf16 | 100 100 100 99 98 98 99 101 | 99.8 |
| ple_kv | Q8 | dense-decode | f32 | 99 100 99 100 98 97 98 99 | 98.9 |
| sh_down | Q4 | dense-decode | bf16 | 104* 91 88 86 86 85 83 82 | 86.4 |
| sh_down | Q4 | dense-decode | f32 | 104* 96 89 87 88 85 85 83 | 87.8 |
| sh_down | Q8 | dense-decode | bf16 | 97 94 90 89 90 88 88 88 | 89.5 |
| sh_down | Q8 | dense-decode | f32 | 98 94 91 89 90 89 88 89 | 90.4 |
| sh_gu | Q4 | dense-decode | bf16 | 92 90 88 90 86 88 83 86 | 87.8 |
| sh_gu | Q4 | dense-decode | f32 | 94 89 90 91 87 87 84 88 | 89.0 |
| sh_gu | Q8 | dense-decode | bf16 | 101 98 98 97 96 96 95 94 | 97.3 |
| sh_gu | Q8 | dense-decode | f32 | 100 99 97 99 96 97 96 94 | 96.9 |

## dense-r16t2

load: load_before 4.68 5.47 6.57 4/1223 608213; load_after 4.22 5.34 6.51 3/1211 612997; 256 / 256 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| attn_o | Q4 | dense-decode | bf16 | 102 100 105* 107* 101 99 99 97 | 100.6 |
| attn_o | Q4 | dense-decode | f32 | 102 101 105* 107* 101 101 98 98 | 100.8 |
| attn_o | Q8 | dense-decode | bf16 | 101 100 101 101 102 96 96 96 | 100.5 |
| attn_o | Q8 | dense-decode | f32 | 100 100 100 102 102 97 97 97 | 99.9 |
| attn_qkv | Q4 | dense-decode | bf16 | 100 99 100 100 97 96 97 97 | 99.3 |
| attn_qkv | Q4 | dense-decode | f32 | 100 100 100 100 97 97 98 98 | 99.6 |
| attn_qkv | Q8 | dense-decode | bf16 | 100 102 100 100 99 98 99 100 | 100.2 |
| attn_qkv | Q8 | dense-decode | f32 | 100 101 100 101 99 97 99 99 | 100.3 |
| gdn_in | Q4 | dense-decode | bf16 | 100 99 99 99 96 96 96 96 | 99.1 |
| gdn_in | Q4 | dense-decode | f32 | 100 99 100 98 98 97 100 99 | 99.3 |
| gdn_in | Q8 | dense-decode | bf16 | 101 101 100 99 97 97 97 97 | 98.9 |
| gdn_in | Q8 | dense-decode | f32 | 99 102 100 100 97 97 97 97 | 99.4 |
| gdn_out | Q4 | dense-decode | bf16 | 102 100 106* 106* 100 100 98 96 | 100.4 |
| gdn_out | Q4 | dense-decode | f32 | 102 102 106* 107* 101 99 98 98 | 101.9 |
| gdn_out | Q8 | dense-decode | bf16 | 100 99 101 101 101 94 93 94 | 100.0 |
| gdn_out | Q8 | dense-decode | f32 | 101 100 101 100 102 96 93 93 | 100.2 |
| mtp_fc | Q4 | dense-decode | bf16 | 101 95 92 91 91 89 87 87 | 91.1 |
| mtp_fc | Q4 | dense-decode | f32 | 100 95 93 92 90 89 89 88 | 91.9 |
| mtp_fc | Q8 | dense-decode | bf16 | 101 101 99 100 100 100 99 99 | 99.7 |
| mtp_fc | Q8 | dense-decode | f32 | 101 99 100 98 99 101 101 99 | 99.6 |
| ple_kv | Q4 | dense-decode | bf16 | 101 100 100 100 97 97 97 96 | 99.6 |
| ple_kv | Q4 | dense-decode | f32 | 101 100 100 100 98 98 98 98 | 99.7 |
| ple_kv | Q8 | dense-decode | bf16 | 100 101 101 101 100 100 102 101 | 100.7 |
| ple_kv | Q8 | dense-decode | f32 | 100 101 101 101 100 99 100 100 | 100.1 |
| sh_down | Q4 | dense-decode | bf16 | 100 95 91 86 84 82 83 81 | 86.2 |
| sh_down | Q4 | dense-decode | f32 | 100 95 90 91 86 83 77 82 | 89.9 |
| sh_down | Q8 | dense-decode | bf16 | 101 94 89 88 84 88 85 83 | 88.4 |
| sh_down | Q8 | dense-decode | f32 | 100 93 89 88 85 90 91 91 | 90.8 |
| sh_gu | Q4 | dense-decode | bf16 | 92 88 89 86 90 90 90 95 | 90.4 |
| sh_gu | Q4 | dense-decode | f32 | 93 90 84 88 90 88 89 87 | 88.8 |
| sh_gu | Q8 | dense-decode | bf16 | 100 100 102 105* 107* 105* 107* 111* | 104.9 |
| sh_gu | Q8 | dense-decode | f32 | 99 99 103 106* 106* 109* 110* 113* | 106.4 |

## dense-r16t4

load: load_before 3.72 5.15 6.42 1/1208 640808; load_after 3.73 5.08 6.38 5/1233 646348; 256 / 256 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| attn_o | Q4 | dense-decode | bf16 | 114* 123* 119* 117* 116* 111* 108* 107* | 115.8 |
| attn_o | Q4 | dense-decode | f32 | 110* 121* 122* 116* 115* 112* 109* 105* | 114.5 |
| attn_o | Q8 | dense-decode | bf16 | 108* 116* 118* 117* 112* 106* 102 102 | 112.0 |
| attn_o | Q8 | dense-decode | f32 | 109* 117* 121* 117* 113* 109* 104* 97 | 112.6 |
| attn_qkv | Q4 | dense-decode | bf16 | 102 104* 104* 103* 99 100 100 100 | 102.4 |
| attn_qkv | Q4 | dense-decode | f32 | 102 104* 104* 102 100 99 100 99 | 102.3 |
| attn_qkv | Q8 | dense-decode | bf16 | 103 106* 104* 104* 99 99 99 99 | 102.7 |
| attn_qkv | Q8 | dense-decode | f32 | 103* 107* 104* 104* 99 98 98 98 | 103.4 |
| gdn_in | Q4 | dense-decode | bf16 | 102 103 103 103 100 100 101 101 | 102.5 |
| gdn_in | Q4 | dense-decode | f32 | 103 103 103 103 100 102 102 102 | 102.5 |
| gdn_in | Q8 | dense-decode | bf16 | 103 104* 102 102 100 101 101 102 | 101.9 |
| gdn_in | Q8 | dense-decode | f32 | 102 103* 102 102 100 101 100 101 | 102.0 |
| gdn_out | Q4 | dense-decode | bf16 | 113* 123* 120* 117* 115* 110* 106* 105* | 114.8 |
| gdn_out | Q4 | dense-decode | f32 | 113* 119* 117* 118* 111* 110* 107* 105* | 112.5 |
| gdn_out | Q8 | dense-decode | bf16 | 108* 115* 120* 118* 113* 109* 105* 99 | 113.4 |
| gdn_out | Q8 | dense-decode | f32 | 108* 116* 121* 118* 113* 106* 103 99 | 113.4 |
| mtp_fc | Q4 | dense-decode | bf16 | 100 97 95 96 96 101 102 101 | 99.7 |
| mtp_fc | Q4 | dense-decode | f32 | 100 98 98 99 102 103 105* 102 | 101.6 |
| mtp_fc | Q8 | dense-decode | bf16 | 101 103* 106* 109* 111* 115* 118* 119* | 111.5 |
| mtp_fc | Q8 | dense-decode | f32 | 101 104* 105* 108* 112* 114* 118* 118* | 111.7 |
| ple_kv | Q4 | dense-decode | bf16 | 102 104* 103 102 100 100 100 101 | 101.9 |
| ple_kv | Q4 | dense-decode | f32 | 102 104* 102 102 100 101 101 101 | 102.0 |
| ple_kv | Q8 | dense-decode | bf16 | 103 105* 104* 103 100 100 100 101 | 102.6 |
| ple_kv | Q8 | dense-decode | f32 | 103 106* 103* 103 100 100 99 99 | 102.8 |
| sh_down | Q4 | dense-decode | bf16 | 101 96 94 91 89 90 92 92 | 92.3 |
| sh_down | Q4 | dense-decode | f32 | 102 95 94 93 91 92 92 94 | 94.1 |
| sh_down | Q8 | dense-decode | bf16 | 99 95 98 98 95 97 98 98 | 97.7 |
| sh_down | Q8 | dense-decode | f32 | 98 97 97 97 95 97 98 100 | 97.3 |
| sh_gu | Q4 | dense-decode | bf16 | 97 101 102 111* 113* 118* 123* 126* | 113.3 |
| sh_gu | Q4 | dense-decode | f32 | 97 99 104* 104 109* 120* 122* 125* | 108.5 |
| sh_gu | Q8 | dense-decode | bf16 | 100 111* 121* 125* 131* 150* 157* 160* | 131.4 |
| sh_gu | Q8 | dense-decode | f32 | 101 112* 121* 125* 131* 151* 156* 162* | 131.1 |

## dense-r8t1

load: load_before 2.96 4.81 6.26 2/1203 652905; load_after 6.27 5.42 6.43 2/1217 667189; 256 / 256 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| attn_o | Q4 | dense-decode | bf16 | 104* 105* 103 103 101 99 97 98 | 103.3 |
| attn_o | Q4 | dense-decode | f32 | 104* 105* 104 102 101 99 99 96 | 102.0 |
| attn_o | Q8 | dense-decode | bf16 | 107* 107* 105* 103* 103 99 96 98 | 103.2 |
| attn_o | Q8 | dense-decode | f32 | 106* 106* 105* 105* 104 98 97 97 | 104.6 |
| attn_qkv | Q4 | dense-decode | bf16 | 103* 104* 103 103 101 102 100 99 | 102.6 |
| attn_qkv | Q4 | dense-decode | f32 | 105* 103 104* 103* 102 101 102 100 | 102.9 |
| attn_qkv | Q8 | dense-decode | bf16 | 106* 107* 106* 105* 103 103* 104* 104* | 104.8 |
| attn_qkv | Q8 | dense-decode | f32 | 105* 106* 106* 105* 104* 103 103* 104* | 104.9 |
| gdn_in | Q4 | dense-decode | bf16 | 106* 105* 104* 104* 100 102 102 101 | 103.8 |
| gdn_in | Q4 | dense-decode | f32 | 105* 103* 105* 103* 104* 103* 103 102 | 103.4 |
| gdn_in | Q8 | dense-decode | bf16 | 107* 106* 106* 105* 104* 104* 105* 107* | 105.7 |
| gdn_in | Q8 | dense-decode | f32 | 106* 107* 106* 106* 104* 104* 106* 105* | 105.7 |
| gdn_out | Q4 | dense-decode | bf16 | 104* 104* 103 101 101 99 97 96 | 101.3 |
| gdn_out | Q4 | dense-decode | f32 | 103* 103* 102 102 101 100 98 97 | 102.1 |
| gdn_out | Q8 | dense-decode | bf16 | 105* 107* 105* 103 104 99 99 96 | 103.8 |
| gdn_out | Q8 | dense-decode | f32 | 108* 107* 105* 104* 104 100 98 98 | 103.9 |
| mtp_fc | Q4 | dense-decode | bf16 | 103 99 98 95 98 95 94 92 | 97.6 |
| mtp_fc | Q4 | dense-decode | f32 | 102 100 97 98 98 97 94 94 | 98.3 |
| mtp_fc | Q8 | dense-decode | bf16 | 108* 104* 104* 103 102 101 101 101 | 103.1 |
| mtp_fc | Q8 | dense-decode | f32 | 106* 105 103 103 102 101 101 101 | 102.8 |
| ple_kv | Q4 | dense-decode | bf16 | 103* 102 102 101 101 101 102 100 | 101.7 |
| ple_kv | Q4 | dense-decode | f32 | 105* 104 103 102 101 101 101 100 | 101.6 |
| ple_kv | Q8 | dense-decode | bf16 | 106* 106* 106* 106* 104* 103* 104* 104* | 105.7 |
| ple_kv | Q8 | dense-decode | f32 | 106* 105* 106* 105* 104* 103 103* 103 | 105.0 |
| sh_down | Q4 | dense-decode | bf16 | 103 96 95 91 94 93 92 89 | 93.8 |
| sh_down | Q4 | dense-decode | f32 | 105* 97 94 96 93 92 89 88 | 94.4 |
| sh_down | Q8 | dense-decode | bf16 | 101 99 97 97 94 93 92 91 | 97.0 |
| sh_down | Q8 | dense-decode | f32 | 103 99 98 98 98 96 96 96 | 97.7 |
| sh_gu | Q4 | dense-decode | bf16 | 100 95 96 95 95 94 95 94 | 94.7 |
| sh_gu | Q4 | dense-decode | f32 | 101 96 92 94 93 95 95 93 | 94.5 |
| sh_gu | Q8 | dense-decode | bf16 | 106* 104* 101 101 100 101 101 100 | 101.4 |
| sh_gu | Q8 | dense-decode | f32 | 106* 104* 103 103 101 100 101 100 | 102.8 |

## dense-r8t2

load: load_before 5.64 5.33 6.38 2/1210 677861; load_after 5.29 5.26 6.34 2/1208 693533; 256 / 256 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| attn_o | Q4 | dense-decode | bf16 | 106* 107* 108* 108* 101 100 99 98 | 105.9 |
| attn_o | Q4 | dense-decode | f32 | 106* 107* 110* 107* 102 101 99 98 | 105.6 |
| attn_o | Q8 | dense-decode | bf16 | 107* 107* 106* 104* 103* 100 99 98 | 104.3 |
| attn_o | Q8 | dense-decode | f32 | 106* 107* 107* 105* 106* 102 101 99 | 105.7 |
| attn_qkv | Q4 | dense-decode | bf16 | 105* 104* 103* 103* 102 102 102 102 | 103.1 |
| attn_qkv | Q4 | dense-decode | f32 | 105* 104* 103* 103* 102 102 102 104* | 103.4 |
| attn_qkv | Q8 | dense-decode | bf16 | 105* 107* 106* 106* 102 103 103 103 | 105.1 |
| attn_qkv | Q8 | dense-decode | f32 | 106* 108* 107* 105* 102 102 102 102 | 105.2 |
| gdn_in | Q4 | dense-decode | bf16 | 106* 105* 104* 105* 102 103* 104* 105* | 104.7 |
| gdn_in | Q4 | dense-decode | f32 | 106* 105* 104* 104* 104* 105* 105* 105* | 104.7 |
| gdn_in | Q8 | dense-decode | bf16 | 106* 107* 105* 105* 102 102 102 103* | 104.8 |
| gdn_in | Q8 | dense-decode | f32 | 106* 107* 105* 105* 103 102 102 102 | 104.9 |
| gdn_out | Q4 | dense-decode | bf16 | 105* 106* 108* 108* 100 101 99 97 | 105.4 |
| gdn_out | Q4 | dense-decode | f32 | 106* 106* 109* 109* 101 101 99 98 | 105.6 |
| gdn_out | Q8 | dense-decode | bf16 | 107* 106* 105* 105* 103 99 98 97 | 105.0 |
| gdn_out | Q8 | dense-decode | f32 | 106* 108* 106* 104* 104* 101 99 100 | 104.3 |
| mtp_fc | Q4 | dense-decode | bf16 | 104* 98 98 95 93 93 92 90 | 95.4 |
| mtp_fc | Q4 | dense-decode | f32 | 104* 99 97 96 94 93 92 91 | 95.7 |
| mtp_fc | Q8 | dense-decode | bf16 | 106* 105* 103 102 101 100 99 100 | 102.3 |
| mtp_fc | Q8 | dense-decode | f32 | 106* 104* 102 102 101 101 101 101 | 101.8 |
| ple_kv | Q4 | dense-decode | bf16 | 105* 105* 104* 104* 102 101 102 103 | 103.9 |
| ple_kv | Q4 | dense-decode | f32 | 105* 103* 103 103* 101 102 103 103* | 103.1 |
| ple_kv | Q8 | dense-decode | bf16 | 106* 107* 106* 105* 103* 103 104* 105* | 105.5 |
| ple_kv | Q8 | dense-decode | f32 | 106* 107* 107* 104* 103* 103* 103 104* | 104.5 |
| sh_down | Q4 | dense-decode | bf16 | 106* 98 98 94 88 89 86 91 | 94.1 |
| sh_down | Q4 | dense-decode | f32 | 107* 99 98 97 92 91 91 90 | 97.0 |
| sh_down | Q8 | dense-decode | bf16 | 103* 99 98 98 94 93 92 93 | 97.8 |
| sh_down | Q8 | dense-decode | f32 | 102 100 98 96 94 95 94 93 | 96.5 |
| sh_gu | Q4 | dense-decode | bf16 | 99 95 97 96 98 98 100 101 | 98.1 |
| sh_gu | Q4 | dense-decode | f32 | 102 94 95 95 99 98 100 100 | 98.5 |
| sh_gu | Q8 | dense-decode | bf16 | 104* 103* 106* 109* 110* 113* 114* 116* | 110.2 |
| sh_gu | Q8 | dense-decode | f32 | 105* 105* 106* 108* 110* 113* 116* 116* | 109.9 |

## exp-r16t2

load: load_before 4.52 5.08 6.25 3/1205 708448; load_after 4.32 5.03 6.23 2/1225 709505; 24 / 24 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| exp_down | Q4 | expert-decode-combine | bf16 | 96 95 97 97 97 99 96 96 | 96.5 |
| exp_gu | Q4 | expert-decode-gather | bf16 | 101 101 99 99 95 96 95 95 | 98.5 |
| exp_gu | Q4 | expert-decode-gather-swiglu | bf16 | 100 100 99 98 96 99 95 97 | 98.8 |

## plekv-16t1

load: load_before 9.73 11.64 11.97 2/1159 2954854; load_after 9.73 11.64 11.97 2/1184 2955707; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| ple_kv | Q4 | dense-decode | bf16 | 99 98 97 98 95 95 94 94 | 97.1 |
| ple_kv | Q4 | dense-decode | f32 | 99 99 98 99 97 95 95 94 | 98.5 |
| ple_kv | Q8 | dense-decode | bf16 | 100 99 99 99 97 98 99 100 | 99.2 |
| ple_kv | Q8 | dense-decode | f32 | 100 101 99 100 97 98 97 99 | 99.3 |

## plekv-16t2

load: load_before 5.81 8.64 10.57 2/1164 3557608; load_after 5.75 8.58 10.54 3/1171 3559728; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| ple_kv | Q4 | dense-decode | bf16 | 102 100 101 101 97 97 98 97 | 100.1 |
| ple_kv | Q4 | dense-decode | f32 | 101 99 99 100 98 98 98 98 | 98.9 |
| ple_kv | Q8 | dense-decode | bf16 | 100 101 100 100 100 99 100 100 | 100.2 |
| ple_kv | Q8 | dense-decode | f32 | 102 101 101 102 99 99 100 100 | 101.0 |

## shdown-16t1-a

load: load_before 4.63 9.78 8.35 3/1177 1490917; load_after 4.63 9.78 8.35 2/1178 1495332; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| sh_down | Q4 | dense-decode | bf16 | 102 92 87 87 88 85 83 89 | 87.5 |
| sh_down | Q4 | dense-decode | f32 | 103 94 90 91 94 91 93 93 | 93.4 |
| sh_down | Q8 | dense-decode | bf16 | 98 94 90 90 90 90 89 90 | 90.0 |
| sh_down | Q8 | dense-decode | f32 | 99 93 92 89 90 89 92 92 | 92.0 |

## shdown-16t1-b

load: load_before 5.24 8.66 8.08 2/1172 1586534; load_after 5.70 8.70 8.09 1/1171 1587254; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| sh_down | Q4 | dense-decode | bf16 | 104* 95 94 93 95 94 92 92 | 93.6 |
| sh_down | Q4 | dense-decode | f32 | 103 94 93 93 94 93 92 92 | 93.4 |
| sh_down | Q8 | dense-decode | bf16 | 98 95 94 94 96 96 96 96 | 95.7 |
| sh_down | Q8 | dense-decode | f32 | 98 97 94 92 95 95 94 95 | 95.2 |

## shdown-16t2-a

load: load_before 3.77 9.26 8.21 3/1171 1521520; load_after 3.77 9.26 8.21 2/1174 1524846; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| sh_down | Q4 | dense-decode | bf16 | 104* 91 88 85 76 77 76 74 | 85.0 |
| sh_down | Q4 | dense-decode | f32 | 102 91 89 86 81 76 79 85 | 85.8 |
| sh_down | Q8 | dense-decode | bf16 | 99 94 90 87 83 83 81 83 | 87.2 |
| sh_down | Q8 | dense-decode | f32 | 99 93 89 90 85 85 83 87 | 89.1 |

## shdown-16t2-b

load: load_before 5.48 8.19 7.98 3/1223 1647580; load_after 5.48 8.19 7.98 7/1199 1652854; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| sh_down | Q4 | dense-decode | bf16 | 104* 90 86 87 80 78 73 72 | 86.2 |
| sh_down | Q4 | dense-decode | f32 | 105* 93 89 86 77 77 83 77 | 85.9 |
| sh_down | Q8 | dense-decode | bf16 | 98 92 89 88 84 86 87 87 | 88.2 |
| sh_down | Q8 | dense-decode | f32 | 99 92 88 89 87 87 86 89 | 88.8 |

## shdown-16t4-a

load: load_before 5.36 9.31 8.25 3/1165 1539700; load_after 5.17 9.21 8.22 4/1177 1541521; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| sh_down | Q4 | dense-decode | bf16 | 103 90 83 82 78 78 77 75 | 82.1 |
| sh_down | Q4 | dense-decode | f32 | 101 88 84 83 77 77 80 79 | 82.7 |
| sh_down | Q8 | dense-decode | bf16 | 97 91 91 92 88 86 87 87 | 91.0 |
| sh_down | Q8 | dense-decode | f32 | 98 93 91 92 88 89 93 95 | 92.6 |

## shdown-16t4-b

load: load_before 4.32 7.76 7.84 1/1138 1669712; load_after 4.32 7.76 7.84 2/1147 1670113; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| sh_down | Q4 | dense-decode | bf16 | 101 95 93 91 90 91 91 92 | 91.5 |
| sh_down | Q4 | dense-decode | f32 | 100 96 93 93 92 92 92 95 | 93.3 |
| sh_down | Q8 | dense-decode | bf16 | 98 95 96 99 96 96 97 99 | 97.0 |
| sh_down | Q8 | dense-decode | f32 | 98 96 97 98 96 97 98 100 | 97.5 |

## shdown-8t8-a

load: load_before 4.35 8.76 8.10 2/1173 1555520; load_after 4.35 8.76 8.10 3/1184 1556936; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| sh_down | Q4 | dense-decode | bf16 | 99 84 86 90 98 105 108 113 | 99.3 |
| sh_down | Q4 | dense-decode | f32 | 102 93 91 91 112* 119* 121* 117* | 112.0 |
| sh_down | Q8 | dense-decode | bf16 | 100 102 104* 106* 109* 113* 117* 120* | 109.5 |
| sh_down | Q8 | dense-decode | f32 | 101 105* 107* 110* 107* 111* 110* 116* | 110.0 |

## shdown-8t8-b

load: load_before 3.32 7.25 7.67 2/1152 1684198; load_after 3.32 7.25 7.67 4/1161 1685139; 32 / 32 byte-equal

| class | bits | path | out | M=1..8 panel % | median |
|---|---|---|---|---|---|
| sh_down | Q4 | dense-decode | bf16 | 102 95 97 100 118 133* 136* 137* | 118.2 |
| sh_down | Q4 | dense-decode | f32 | 101 97 91 93 105 112* 134* 138* | 104.9 |
| sh_down | Q8 | dense-decode | bf16 | 102 106* 110* 114* 120* 119* 126* 132* | 119.0 |
| sh_down | Q8 | dense-decode | f32 | 102 106* 110* 112* 114* 124* 127* 130* | 114.0 |

`*`: slower outside noise by more than 3% (panel q1 > origin q3 and > 103%).
