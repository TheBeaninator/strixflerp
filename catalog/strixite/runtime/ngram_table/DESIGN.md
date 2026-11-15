# runtime.strixite.ngram_table: design record

## What it is for
The source the forward reads PLE n-gram rows through: `NgramTableRows` (the converted table file, rows on demand
from SSD) or the interim `NgramRowsFromShards` (`src/runtime/ngram_table.hpp:3-8`).

## Choices (origin's code)
- Rows deduplicated, fetched with up to 64 concurrent preads, kept in an LRU cache (n-gram ids are skewed; frequent
  n-grams repeat across a conversation and its prefills) (`ngram_table.hpp:3-7`).
- 64 reads gather ~1.25x faster than 32 cold (16,384 random rows 168 -> 134 ms; 65,536 rows 630 -> 494 ms); 128 and
  256 gain nothing more (`src/runtime/ngram_table.cpp:20-22`).
- The engine prefetches rows while drafting and for prefill lookahead, decodes to BF16 into pinned buffers, and
  uploads them for `ple_gate` / `ple_conv` (`executor.strixite.qwen4exp` `qwen4exp.cpp:1214-1248`, `:1250-1276`).

## Coupled to
`format.strixite.ngram_table`, `runtime.strixite.ple_hash`, `runtime.strixite.host_buffer`, `kernel.strixite.ple`,
`executor.strixite.qwen4exp`.

## Prefill remarks
An 8,192-token chunk asks for up to 131,072 row ids before dedup. Inferred from the measured rate: a cold first chunk
can wait ~0.5-1 s on the SSD; later chunks overlap it with GPU work. The origin lists the first chunk's wait as open
work (`strixite docs/roadmap.md:23-25`).
