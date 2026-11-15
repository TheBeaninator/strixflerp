# qwen38-flash-next strixite self-replay (FLERP-SELFID bab6510b88)

Origin: `strix_server.v070tg` (20937dc + d19945e), fleet argv, on control-node's GPU via tools/gpu-queue.sh; two runs A and B, each a fresh process with an empty prompt-cache dir; recorded seed (FreeFitty sends `seed`) kept; max_tokens capped at 256; load 50.0 s / 110.0 s.
Token-identical = same choices[].message (content, reasoning, tool_calls), finish_reason and usage.completion_tokens; byte-identical = whole body equal after dropping id, created, timings, system_fingerprint and each tool call's random `id` (call_<random>, drawn per reply). In the two-in-flight table the alone run came first, so the in-flight request hits the prompt cache and `usage.prompt_tokens_details.cached_tokens` differs: byte-identity there is expected to fail, and token identity is the measure.

## One conversation at a time: run A vs run B

| class | requests | token-identical | byte-identical | first differing id |
|---|---|---|---|---|
| long_prompt | 6 | 6 | 6 | - |
| cache_hit | 2 | 2 | 2 | - |

## Two conversations in flight at once vs the same request alone (same run)

| class (run) | requests | token-identical | byte-identical | first differing id |
|---|---|---|---|---|
| long_prompt (run A) | 4 | 0 | 0 | strixite-00037 |
| cache_hit (run A) | 2 | 2 | 0 | strixite-00067 |
| long_prompt (run B) | 4 | 2 | 0 | strixite-00037 |
| cache_hit (run B) | 2 | 2 | 0 | strixite-00067 |

## Per request

| id | labels | prompt_tokens (recorded) | A secs | B secs | A==B token | A==B bytes |
|---|---|---|---|---|---|---|
| strixite-00037 | long_prompt,cache_hit,tools | 69618 | 51.92 | 57.75 | True | True |
| strixite-00056 | long_prompt,cache_hit,tools | 105131 | 76.89 | 79.08 | True | True |
| strixite-00067 | cache_hit,tools | 18846 | 16.79 | 19.41 | True | True |
| strixite-00084 | long_prompt,cache_hit,tools | 69806 | 7.78 | 8.0 | True | True |
| strixite-00096 | long_prompt,cache_hit,tools | 70092 | 5.38 | 5.16 | True | True |
| strixite-00101 | cache_hit,tools | 20740 | 20.38 | 16.23 | True | True |
| strixite-00326 | long_prompt,cache_hit,tools | 66326 | 53.69 | 81.88 | True | True |
| strixite-00485 | long_prompt,cache_hit,tools | 67369 | 5.97 | 9.94 | True | True |

Pair (StrixFold) rows: empty. StrixFold against itself needs the pair's GPUs (F+D), which serve FreeFitty, and no pair window was agreed with its owner. The pair half of corpus-v1 exists (151 chats) for when one is.
