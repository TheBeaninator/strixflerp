### prefill, rank 0: 8 chunks, 903.91 ms GPU a chunk, 7.23 s over the prompt

| phase | ms a chunk | ms over the prompt | share |
|---|---:|---:|---:|
| embed | 0.148 | 1.2 | 0.0% |
| engram | 93.171 | 745.4 | 10.3% |
| mix_attn | 22.700 | 181.6 | 2.5% |
| attn | 159.465 | 1275.7 | 17.6% |
| gather_a | 112.195 | 897.6 | 12.4% |
| mix_ffn | 21.360 | 170.9 | 2.4% |
| moe | 308.995 | 2472.0 | 34.2% |
| gather_m | 185.689 | 1485.5 | 20.5% |
| head | 0.185 | 1.5 | 0.0% |

### prefill, rank 1: 8 chunks, 903.97 ms GPU a chunk, 7.23 s over the prompt

| phase | ms a chunk | ms over the prompt | share |
|---|---:|---:|---:|
| embed | 0.152 | 1.2 | 0.0% |
| engram | 93.126 | 745.0 | 10.3% |
| mix_attn | 23.389 | 187.1 | 2.6% |
| attn | 155.244 | 1242.0 | 17.2% |
| gather_a | 115.907 | 927.3 | 12.8% |
| mix_ffn | 21.639 | 173.1 | 2.4% |
| moe | 376.753 | 3014.0 | 41.7% |
| gather_m | 117.565 | 940.5 | 13.0% |
| head | 0.190 | 1.5 | 0.0% |

### decode, rank 0: 71.09 ms GPU a timed round (1 rounds)

| phase | ms a round | share |
|---|---:|---:|
| embed | 0.022 | 0.0% |
| engram | 2.225 | 3.1% |
| mix_attn | 1.345 | 1.9% |
| attn | 3.191 | 4.5% |
| gather_a | 12.601 | 17.7% |
| mix_ffn | 1.302 | 1.8% |
| moe | 32.630 | 45.9% |
| gather_m | 2.503 | 3.5% |
| post | 0.019 | 0.0% |
| head | 1.769 | 2.5% |
| a_in | 2.033 | 2.9% |
| a_q | 3.820 | 5.4% |
| a_idx | 1.227 | 1.7% |
| a_core | 1.300 | 1.8% |
| a_woa | 3.176 | 4.5% |
| a_x | 0.106 | 0.1% |
| m_rt | 1.823 | 2.6% |

request p8k: 8163 prompt tokens, prefill 7.28 s
request p8k: 2 tokens in 7.35 s, 1 rounds, 1/5 drafts accepted
