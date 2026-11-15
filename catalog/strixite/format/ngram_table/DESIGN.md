# format.strixite.ngram_table: design record

## What it is for
The PLE n-gram embedding table (51 B parameters) as its own flat file that stays on SSD; rows are read on demand.

## Choices (origin's code and docs)
- A 512-byte hashed header, then every row back to back from offset 4096: row r at `4096 + r * row_bytes`, one
  positioned read per row (`src/formats/ngram_table.hpp:3-7`).
- Filesystem compression must be off for the file: "a random 320-byte read from a zstd btrfs extent decompresses up
  to 128 KiB" (`ngram_table.hpp:7-9`).
- Row formats BF16, FP8E4M3 (one table scale), Q8Row (BF16 absmax scale per row + int8); rows decode to BF16 on the
  host (`ngram_table.hpp:19-28`). The served table is 8-bit rows, one BF16 scale per row of 160 (8.1 bits per
  weight, 48.3 GiB) (`strixite docs/weights.md:30`).
- The header carries the source checkpoint's fingerprint; the engine refuses a table from another checkpoint
  (`executor.strixite.qwen4exp` `qwen4exp.cpp:167-172`).

## What it buys / costs
Buys 48 GiB not in GTT. Costs SSD latency on the first use of rows: a cold gather of 65,536 random rows takes
494 ms with 64 concurrent reads (`runtime.strixite.ngram_table` `ngram_table.cpp:20-21`).

## Coupled to
`runtime.strixite.ngram_table` (reader), `runtime.strixite.ple_hash` (row ids), `tool.strixite.convert_ngram_table`,
`kernel.strixite.ple`.

## Prefill remarks
A prefill chunk of 8192 tokens asks for up to 8192 x 16 row ids (deduplicated). Later chunks overlap the reads with
GPU work; the first chunk of a prompt waits on them (`strixite docs/roadmap.md:23-25`, "the first chunk of a prompt
waiting on n-gram table reads from the SSD").
